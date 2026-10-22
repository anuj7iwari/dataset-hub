# ========================================
# Media Ingestion and Processing Engine
# ========================================

import os
import io
import base64
from pathlib import Path
from typing import Union, Optional, Tuple, Any, List, Dict

import numpy as np
import torch
from PIL import Image

try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import scipy.io.wavfile as wavfile
    import scipy.signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ========================================
# Vision Processing Functions
# ========================================

def _validate_image_size(size: Any, name: str = "target_size") -> Tuple[int, int]:
    """Validate image dimensions consistently before invoking a backend."""
    import numbers
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"{name} must be a (height, width) pair of positive integers, got {size!r}")
    if any(not isinstance(v, numbers.Integral) or isinstance(v, bool) or int(v) <= 0 for v in size):
        raise ValueError(f"{name} must be a (height, width) pair of positive integers, got {size!r}")
    return int(size[0]), int(size[1])


def load_image(source: Union[str, Path, bytes, Image.Image, Dict[str, Any]]) -> Image.Image:
    """Decodes an image from file path, raw bytes, base64 string, or dictionary."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, dict):
        if "bytes" in source and source["bytes"]:
            return load_image(source["bytes"])
        if "path" in source and source["path"]:
            return load_image(source["path"])
        if "image" in source:
            return load_image(source["image"])

    if isinstance(source, bytes):
        try:
            with Image.open(io.BytesIO(source)) as decoded:
                return decoded.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unable to decode image bytes: {exc}") from exc

    if isinstance(source, (str, Path)):
        source_str = str(source).strip()
        # Handle base64 encoded strings
        if source_str.startswith("data:image") and "base64," in source_str:
            base64_data = source_str.split("base64,", 1)[1]
            try:
                image_bytes = base64.b64decode(base64_data, validate=True)
                with Image.open(io.BytesIO(image_bytes)) as decoded:
                    return decoded.convert("RGB")
            except Exception as exc:
                raise ValueError(f"Invalid base64 image data URI: {exc}") from exc

        if not os.path.exists(source_str):
            raise FileNotFoundError(f"Image file not found: {source_str}")
        try:
            with Image.open(source_str) as decoded:
                return decoded.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unable to decode image file '{source_str}': {exc}") from exc

    raise ValueError(f"Cannot decode image from object of type: {type(source)}")


def _custom_image_output_to_tensor(value: Any, pil_img: Image.Image) -> torch.Tensor:
    """Validate and normalize custom image-transform output into a tensor."""
    if isinstance(value, np.ndarray):
        arr = value
        if arr.size == 0:
            raise ValueError("Custom image transform returned an empty array.")
        if arr.ndim not in (2, 3):
            raise ValueError(f"Custom image transform returned ndarray with unsupported ndim {arr.ndim}; expected HWC or HW.")
        if arr.dtype.kind in "iu":
            if arr.min() < 0:
                raise ValueError("Custom integer image array must not contain negative pixel values.")
            if arr.max() <= 255:
                arr = arr.astype(np.float32) / 255.0
            elif arr.dtype.kind == "u":
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            else:
                raise ValueError("Custom signed-integer image array exceeds 8-bit pixel range [0, 255].")
        elif arr.dtype.kind == "b":
            arr = arr.astype(np.float32)
        elif arr.dtype.kind == "f":
            arr = arr.astype(np.float32, copy=False)
            if not np.isfinite(arr).all():
                raise ValueError("Custom image transform returned NaN or infinite values.")
            if arr.min() >= 0 and arr.max() > 1:
                if arr.max() <= 255:
                    arr = arr / 255.0
                else:
                    raise ValueError("Custom floating image array must be in [0, 1], [0, 255], or a normalized range with negative values.")
        else:
            raise ValueError(f"Unsupported custom image array dtype {arr.dtype}.")
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    elif isinstance(value, torch.Tensor):
        t = value
        if t.numel() == 0:
            raise ValueError("Custom image transform returned an empty tensor (shape has zero elements).")
        if t.ndim not in (2, 3):
            raise ValueError(f"Custom image transform returned tensor with unsupported ndim {t.ndim}; expected CHW, HWC, or HW.")
        if t.ndim == 2:
            t = t.unsqueeze(0)
        else:
            # Prefer HWC when its spatial dimensions match the source image; otherwise
            # infer from a conventional 1/3/4-channel leading or trailing dimension.
            is_source_hwc = t.shape[0] == pil_img.height and t.shape[1] == pil_img.width and t.shape[-1] in (1, 3, 4)
            if is_source_hwc or (t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4)):
                t = t.permute(2, 0, 1)
            elif t.shape[0] not in (1, 3, 4):
                raise ValueError(f"Custom image transform tensor shape {tuple(t.shape)} is neither CHW nor HWC.")
        if t.dtype == torch.bool:
            t = t.to(torch.float32)
        elif not t.is_floating_point():
            # Convert before reductions so this also works on torch versions where
            # min/max kernels are not implemented for wider unsigned integer dtypes.
            wide = t.to(torch.float64)
            if float(wide.min().item()) < 0:
                raise ValueError("Custom integer image tensor must not contain negative pixel values.")
            tmax = float(wide.max().item())
            if tmax <= 255:
                t = wide.to(torch.float32) / 255.0
            else:
                unsigned_types = tuple(
                    getattr(torch, name) for name in ("uint16", "uint32", "uint64")
                    if hasattr(torch, name)
                )
                if t.dtype in unsigned_types:
                    t = wide.to(torch.float32) / float(torch.iinfo(t.dtype).max)
                else:
                    raise ValueError("Custom signed-integer image tensor exceeds 8-bit pixel range [0, 255].")
        else:
            if not bool(torch.isfinite(t).all()):
                raise ValueError("Custom image transform returned NaN or infinite values.")
            t = t.to(torch.float32)
            if bool(torch.all(t >= 0)) and float(t.max().item()) > 1:
                if float(t.max().item()) <= 255:
                    t = t / 255.0
                else:
                    raise ValueError("Custom floating image tensor with non-negative values must be in [0, 1] or [0, 255].")
        return t.contiguous()
    else:
        raise TypeError(f"Custom image transform returned unsupported type {type(value).__name__}; expected torch.Tensor or numpy.ndarray.")

    return tensor.contiguous()


def image_to_tensor(
    image: Union[str, Path, bytes, Image.Image, Dict[str, Any]],
    processor_or_transform: Optional[Any] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Converts an image into a PyTorch FloatTensor [3, H, W] or processor output."""
    target_size = _validate_image_size(target_size)
    pil_img = load_image(image)

    # 1. Hugging Face Image Processor or Feature Extractor
    if processor_or_transform is not None:
        img_proc = getattr(processor_or_transform, "image_processor", None) or getattr(
            processor_or_transform, "feature_extractor", None
        )
        target_proc = img_proc if img_proc is not None else processor_or_transform

        is_hf_processor = hasattr(target_proc, "__call__") and (
            img_proc is not None
            or hasattr(target_proc, "preprocess")
            or "image" in type(target_proc).__name__.lower()
            or "processor" in type(target_proc).__name__.lower()
        )

        if is_hf_processor:
            try:
                proc_out = target_proc(images=pil_img, return_tensors="pt")
                if isinstance(proc_out, dict):
                    if "pixel_values" in proc_out and isinstance(proc_out["pixel_values"], torch.Tensor):
                        return proc_out["pixel_values"].squeeze(0)
                    for v in proc_out.values():
                        if isinstance(v, torch.Tensor):
                            return v.squeeze(0)
                elif isinstance(proc_out, torch.Tensor):
                    return proc_out.squeeze(0)
                raise ValueError(f"Processor did not yield a valid tensor. Output: {type(proc_out)}")
            except Exception as e:
                raise RuntimeError(f"Hugging Face ImageProcessor failed on image: {str(e)}") from e

        # User-supplied callable transform or processor
        try:
            res = processor_or_transform(pil_img)
            return _custom_image_output_to_tensor(res, pil_img)
        except (TypeError, ValueError) as e:
            # Surface actionable contract errors instead of obscuring them in a wrapper.
            raise
        except Exception as e:
            raise RuntimeError(f"Image transform or processor failed on image: {str(e)}") from e

    # 2. Standard Torchvision Transform fallback (only when no processor is specified)
    if TORCHVISION_AVAILABLE:
        transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),  # Converts to [3, H, W] in range [0.0, 1.0]
        ])
        return transform(pil_img)

    # 3. Pure NumPy to Torch fallback (Zero external dependencies)
    # PIL expects (width, height), so reverse (H, W) target_size
    pil_target_size = (target_size[1], target_size[0])
    resized = pil_img.resize(pil_target_size, Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)    # [3, H, W]
    return tensor


def image_to_numpy(
    image: Union[str, Path, bytes, Image.Image, Dict[str, Any]],
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Returns image as a NumPy array with shape (H, W, 3) in uint8 where target_size is (height, width)."""
    pil_img = load_image(image)
    if target_size is not None:
        target_size = _validate_image_size(target_size)
        pil_target_size = (target_size[1], target_size[0])
        pil_img = pil_img.resize(pil_target_size, Image.Resampling.BILINEAR)
    return np.array(pil_img)


# ========================================
# Audio Processing Functions
# ========================================

def load_audio(
    source: Union[str, Path, bytes, Dict[str, Any]],
    target_sr: Optional[int] = 16000,
) -> Tuple[torch.Tensor, int]:
    """Decodes audio into a normalized PyTorch waveform FloatTensor [1, T]."""
    if target_sr is not None:
        if not isinstance(target_sr, int) or isinstance(target_sr, bool) or target_sr <= 0:
            raise ValueError(f"target_sr must be a positive integer (or None to skip resampling), got {target_sr!r}")

    raw_tensor = None
    native_sr = 16000

    # 1. Handle Hugging Face Audio dict format: {'array': np.array, 'sampling_rate': int}
    if isinstance(source, dict):
        if "array" in source and "sampling_rate" in source:
            raw_sr = source["sampling_rate"]
            import numbers
            if not isinstance(raw_sr, numbers.Integral) or isinstance(raw_sr, bool) or int(raw_sr) <= 0:
                raise ValueError(f"Audio source sampling_rate must be a positive integer, got {raw_sr!r}")
            arr = np.array(source["array"])
            if arr.ndim not in (1, 2):
                raise ValueError(f"Audio source array must have 1 or 2 dimensions, got shape {arr.shape}")
            if arr.size == 0:
                raise ValueError("Audio source array must contain at least one sample")
            arr = _normalize_audio_data(arr)
            raw_tensor = torch.from_numpy(arr)
            native_sr = int(raw_sr)
        elif "bytes" in source:
            source = source["bytes"]
        elif "path" in source:
            source = source["path"]

    # 2. Decode from bytes or file path
    if raw_tensor is None:
        if isinstance(source, bytes):
            wav_io = io.BytesIO(source)
            if SCIPY_AVAILABLE:
                native_sr, data = wavfile.read(wav_io)
                arr = _normalize_audio_data(data)
                raw_tensor = torch.from_numpy(arr)
            else:
                import wave
                with wave.open(wav_io, "rb") as wf:
                    native_sr, raw_tensor = _read_wave_stream(wf)

        else:
            file_path = str(source).strip()
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Try SciPy for WAV
            if SCIPY_AVAILABLE and file_path.lower().endswith(".wav"):
                native_sr, data = wavfile.read(file_path)
                arr = _normalize_audio_data(data)
                raw_tensor = torch.from_numpy(arr)

            # Try torchaudio if installed
            if raw_tensor is None:
                try:
                    import torchaudio
                    waveform, native_sr = torchaudio.load(file_path)
                    raw_tensor = waveform
                except ImportError:
                    pass

            # Wave fallback
            if raw_tensor is None:
                import wave
                with wave.open(file_path, "rb") as wf:
                    native_sr, raw_tensor = _read_wave_stream(wf)

    # 3. Resample if target_sr is requested and differs from native_sr
    if not isinstance(native_sr, int) or isinstance(native_sr, bool) or native_sr <= 0:
        raise ValueError(f"Audio source sampling_rate must be a positive integer, got {native_sr!r}")
    if not isinstance(raw_tensor, torch.Tensor):
        raise ValueError(f"Audio backend returned {type(raw_tensor).__name__}, expected torch.Tensor")
    if raw_tensor.ndim == 1:
        raw_tensor = raw_tensor.unsqueeze(0)
    if raw_tensor.ndim != 2 or raw_tensor.numel() == 0:
        raise ValueError(f"Decoded audio must be a non-empty [channels, time] tensor, got shape {tuple(raw_tensor.shape)}")
    if not bool(torch.isfinite(raw_tensor).all()):
        raise ValueError("Decoded audio contains NaN or infinite values")
    raw_tensor = raw_tensor.to(torch.float32).clamp(-1.0, 1.0)

    if target_sr is not None and target_sr != native_sr:
        raw_tensor = _resample_waveform(raw_tensor, orig_sr=native_sr, target_sr=target_sr)
        final_sr = target_sr
    else:
        final_sr = native_sr

    return raw_tensor, final_sr


def _read_wave_stream(wf) -> Tuple[int, torch.Tensor]:
    """Read uncompressed PCM WAV safely, including signed 24-bit samples."""
    sr = int(wf.getframerate())
    n_channels = int(wf.getnchannels())
    n_frames = int(wf.getnframes())
    sampwidth = int(wf.getsampwidth())
    if sr <= 0:
        raise ValueError(f"WAV sampling rate must be positive, got {sr}")
    if n_channels <= 0:
        raise ValueError(f"WAV channel count must be positive, got {n_channels}")
    if n_frames <= 0:
        raise ValueError("WAV file contains no audio samples")

    raw_bytes = wf.readframes(n_frames)
    if sampwidth == 1:
        # Unsigned 8-bit PCM is centered around 128.
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0
        max_val = 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype="<i2")
        max_val = 32768.0
    elif sampwidth == 3:
        # WAV stores 24-bit PCM little-endian, signed two's-complement. Expand and
        # sign-extend into int32 instead of incorrectly interpreting each 3-byte
        # sample as int16 (which silently corrupts the waveform).
        b = np.frombuffer(raw_bytes, dtype=np.uint8)
        if b.size % 3:
            raise ValueError("Malformed 24-bit PCM WAV payload: byte count is not divisible by 3")
        triples = b.reshape(-1, 3).astype(np.int32)
        data = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        data = (data ^ 0x800000) - 0x800000
        max_val = 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype="<i4")
        max_val = 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM WAV sample width {sampwidth} bytes; expected 1, 2, 3, or 4.")

    if data.size != n_frames * n_channels:
        raise ValueError(
            f"Malformed WAV payload: expected {n_frames * n_channels} samples, decoded {data.size}."
        )
    if n_channels > 1:
        data = data.reshape(-1, n_channels).T
    else:
        data = data[np.newaxis, :]

    arr = np.clip(data.astype(np.float64) / max_val, -1.0, 1.0).astype(np.float32)
    return sr, torch.from_numpy(arr)


def _normalize_audio_data(data: np.ndarray) -> np.ndarray:
    """Normalize raw numeric audio arrays to finite float32 [-1, 1], [channels, time]."""
    data = np.asarray(data)
    if data.ndim not in (1, 2):
        raise ValueError(f"Audio array must have 1 or 2 dimensions, got shape {data.shape}")
    if data.size == 0:
        raise ValueError("Audio array must contain at least one sample")
    if data.dtype.kind not in "iuf":
        raise ValueError(f"Audio array must use an integer or floating-point dtype, got {data.dtype}")
    if data.dtype.kind == "f" and not np.isfinite(data).all():
        raise ValueError("Audio array contains NaN or infinite values")

    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        # If shape is [time, channels], transpose to [channels, time]
        if data.shape[0] > data.shape[1] and data.shape[1] <= 8:
            data = data.T

    if data.dtype == np.uint8:
        # Unsigned 8-bit PCM: 128 represents silence/center
        norm = (data.astype(np.float32) - 128.0) / 128.0
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    if np.issubdtype(data.dtype, np.integer):
        float_data = data.astype(np.float32)
        neg_denom = float(-np.iinfo(data.dtype).min)
        pos_denom = float(np.iinfo(data.dtype).max)
        norm = np.where(float_data < 0, float_data / neg_denom, float_data / pos_denom)
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    return np.clip(data.astype(np.float32), -1.0, 1.0)


def _resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resamples waveform tensor [channels, time] to target_sr."""
    if not isinstance(orig_sr, int) or isinstance(orig_sr, bool) or orig_sr <= 0:
        raise ValueError(f"orig_sr must be a positive integer, got {orig_sr!r}")
    if not isinstance(target_sr, int) or isinstance(target_sr, bool) or target_sr <= 0:
        raise ValueError(f"target_sr must be a positive integer, got {target_sr!r}")
    if orig_sr == target_sr:
        return waveform

    # 1. Try torchaudio Resample
    try:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(waveform)
    except ImportError:
        pass
    except Exception as e:
        raise ValueError(f"Failed to resample audio from {orig_sr} Hz to {target_sr} Hz: {str(e)}") from e

    # 2. Try scipy.signal.resample_poly
    if SCIPY_AVAILABLE:
        try:
            from math import gcd
            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            arr = waveform.numpy()
            resampled_channels = [
                sp_signal.resample_poly(arr[c], up, down).astype(np.float32)
                for c in range(arr.shape[0])
            ]
            return torch.from_numpy(np.stack(resampled_channels, axis=0))
        except Exception as e:
            raise ValueError(f"Failed to resample audio from {orig_sr} Hz to {target_sr} Hz: {str(e)}") from e

    # If neither backend is available, inform caller rather than corrupting silently
    raise RuntimeError(
        f"Audio resampling from {orig_sr}Hz to {target_sr}Hz requires `scipy` or `torchaudio`. "
        "Please install scipy: `pip install scipy`"
    )


def audio_to_tensor(
    source: Union[str, Path, bytes, Dict[str, Any]],
    processor: Optional[Any] = None,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Loads audio and processes it with an audio processor or returns normalized waveform."""
    waveform, sr = load_audio(source, target_sr=target_sr)
    if processor is not None and hasattr(processor, "__call__"):
        try:
            proc_out = processor(waveform.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt")
            if isinstance(proc_out, dict):
                if "input_features" in proc_out:
                    return proc_out["input_features"].squeeze(0)
                if "input_values" in proc_out:
                    return proc_out["input_values"].squeeze(0)
            elif isinstance(proc_out, torch.Tensor):
                return proc_out.squeeze(0)
            raise ValueError(f"Processor output could not be mapped to tensor: {type(proc_out)}")
        except Exception as e:
            raise RuntimeError(f"Audio processor failed to preprocess audio: {str(e)}") from e

    return waveform


# ========================================
# Video Processing Functions
# ========================================

def load_video(
    source: Union[str, Path],
    max_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Extracts evenly spaced frames from a video file into a FloatTensor [T, 3, H, W]."""
    video_path = str(source).strip()
    import numbers
    if not isinstance(max_frames, numbers.Integral) or isinstance(max_frames, bool) or max_frames < 1:
        raise ValueError(f"max_frames must be a positive integer, got {max_frames!r}")
    max_frames = int(max_frames)
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2 or any(
        not isinstance(x, numbers.Integral) or isinstance(x, bool) or int(x) < 1 for x in target_size
    ):
        raise ValueError(f"target_size must be a (height, width) pair of positive ints, got {target_size!r}")
    target_size = (int(target_size[0]), int(target_size[1]))
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for video decoding. Run: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Unable to read video stream from: {video_path}")

        # Determine frame indices to sample uniformly
        if total_frames <= max_frames:
            frame_indices = set(range(total_frames))
        else:
            step = total_frames / max_frames
            frame_indices = set(int(i * step) for i in range(max_frames))

        extracted_frames: List[np.ndarray] = []
        current_idx = 0

        while cap.isOpened() and len(extracted_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in frame_indices:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target_size:
                    cv2_size = (target_size[1], target_size[0])
                    rgb_frame = cv2.resize(rgb_frame, cv2_size, interpolation=cv2.INTER_LINEAR)
                extracted_frames.append(rgb_frame)
            current_idx += 1

        if not extracted_frames:
            raise ValueError(f"No frames could be extracted from video: {video_path}")

        # If extracted frames are fewer than max_frames, repeat the last frame to guarantee shape [max_frames, 3, H, W]
        while len(extracted_frames) < max_frames:
            extracted_frames.append(extracted_frames[-1].copy())

        video_arr = np.stack(extracted_frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(video_arr).permute(0, 3, 1, 2)  # [T, 3, H, W]
        return video_tensor

    finally:
        cap.release()
