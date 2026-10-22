# 8. Media utilities

Media helpers are also exported at the package root for direct use. These are useful when you need to inspect a single media object outside the dataset pipeline.

## 8.1 Supported inputs

| Helper | Inputs accepted by the public type contract | Result |
|---|---|---|
| `load_image(source)` | path, `Path`, bytes, PIL image, dict containing `bytes`, `path`, or `image`, supported base64 image data URI | RGB PIL image |
| `image_to_tensor(image, processor_or_transform=None, target_size=(224, 224))` | same image inputs | float tensor, usually `[3, H, W]`, or processor output tensor shape |
| `image_to_numpy(image, target_size=None)` | same image inputs | uint8 RGB NumPy array `[H, W, 3]` |
| `load_audio(source, target_sr=16000)` | path, `Path`, bytes, audio dict with array/sample rate or bytes/path | `(waveform, sample_rate)`; waveform `[channels, time]` float tensor |
| `audio_to_tensor(source, processor=None, target_sr=16000)` | same audio inputs | waveform or processor-generated feature tensor |
| `load_video(source, max_frames=16, target_size=(224, 224))` | local path/`Path` | float tensor `[T, 3, H, W]` |

## 8.2 Images

### `load_image`

Decodes an image and returns an RGB PIL image. Direct PIL inputs are converted to RGB; bytes are decoded from memory; a dictionary may wrap a byte payload under `bytes`, a path under `path`, or another image value under `image`. A string beginning with a valid `data:image...;base64,` data URI is decoded from base64. Local paths must exist and the content must be decodable.

```python
from data_hub import load_image

img = load_image("./assets/example.jpg")
print(img.mode, img.size)  # RGB, (width, height)
```

Pillow's `Image.size` is `(width, height)`, unlike the public resize arguments, which use `(height, width)`.

### `image_to_tensor`

```python
from data_hub import image_to_tensor

pixels = image_to_tensor("./assets/example.jpg", target_size=(224, 320))
print(pixels.shape)  # torch.Size([3, 224, 320]) for the standard path
```

Default processing resizes and converts to a channel-first float tensor. If torchvision is installed, the implementation uses its resize/tensor transform; otherwise a NumPy/PyTorch fallback resizes through Pillow. The target-size convention is `(height, width)` in both paths.

With a processor object, the helper first attempts the image-processor style call `processor(images=pil_img, return_tensors="pt")` and uses `pixel_values` when available. For generic callables it passes the PIL image directly. Custom transforms may return a PyTorch tensor or NumPy array; shape, dtype and numeric ranges are validated/normalized. A model-normalized float transform can legitimately produce negative values. The library cannot infer whether those normalization statistics are correct for your model.

### `image_to_numpy`

```python
array = image_to_numpy("./assets/example.jpg", target_size=(224, 320))
print(array.shape, array.dtype)  # (224, 320, 3), uint8
```

The array uses HWC order and RGB channels. Unlike the standard tensor fallback, it is not scaled to 0–1; it retains uint8 pixel values from 0–255.

## 8.3 Audio

### `load_audio`

```python
from data_hub import load_audio

waveform, sr = load_audio("./audio/example.wav", target_sr=16_000)
print(waveform.shape, sr)  # [channels, time], target sample rate when resampling succeeded
```

The implementation normalizes audio to float32, clamps it into `[-1, 1]`, and validates sample rate, dimensions, and finite values. One-dimensional audio becomes one channel. A Hugging Face-style payload such as `{"array": array, "sampling_rate": 44100}` is supported as an in-memory audio source. A dict containing `bytes` or `path` may wrap a file payload.

Backend coverage depends on installed libraries:

- SciPy supports WAV file reading and the library's resampling path where available;
- `torchaudio`, if installed and importable, can be used for supported formats;
- the built-in `wave` fallback handles uncompressed PCM WAV streams;
- compressed formats such as MP3/AAC are not guaranteed to decode without a suitable installed backend.

The helper returns two values: tensor then sample rate. If `target_sr=None`, it leaves the source rate unchanged. If a different target rate is requested but the required resampling backend is unavailable, the call may raise an actionable error instead of silently pretending the rate changed.

### `audio_to_tensor`

Without a processor, this returns the waveform tensor (typically `[channels, time]`). With a callable processor, it passes the one-dimensional audio array, sample rate, and `return_tensors="pt"`. It maps `input_features` or `input_values` from a dictionary result, or accepts a tensor result. Processor output is squeezed along the leading batch dimension.

```python
features = audio_to_tensor(
    "./audio/example.wav",
    processor=my_audio_processor,
    target_sr=16_000,
)
```

The exact output semantics depend on the processor. Validate shape/dtype before passing the result to a model.

## 8.4 Video

```python
from data_hub import load_video

frames = load_video(
    "./video/clip.mp4",
    max_frames=12,
    target_size=(224, 224),
)
print(frames.shape)  # [12, 3, 224, 224]
```

`load_video()` uses OpenCV. It validates positive `max_frames` and dimensions, confirms the path exists, and releases the capture handle in a `finally` block. Frames are selected approximately evenly across the clip. Frames are converted BGR → RGB, resized, scaled to float 0–1, and returned TCHW. When fewer frames than requested can be extracted from a short video, the last successfully extracted frame is repeated to reach the fixed frame count.

OpenCV reports metadata such as frame count from the container; corrupted files or unsupported codecs can still yield decode errors. Run a smoke test on the target codec with the same build/container you intend to deploy.

## 8.5 Shape summary

| Representation | Standard shape | Order | Value convention |
|---|---|---|---|
| PIL image | `img.size == (W, H)` | Pillow image | PIL mode RGB |
| Image NumPy | `[H, W, 3]` | HWC | `uint8`, normally 0–255 |
| Image tensor | `[3, H, W]` | CHW | float; standard fallback 0–1 |
| Audio waveform | `[C, T]` | channels, time | float32, clamped to -1–1 |
| Single-video tensor | `[T, 3, H, W]` | TCHW | float32, 0–1 |
| Video batch | `[B, T, 3, H, W]` | BTCHW | float32, 0–1 |

## 8.6 Media failure handling

A helper normally raises on invalid/unsupported media. The dataset/PyTorch layer may wrap that error with the record ID or path and apply its configured policy. Don't catch every media exception and replace it with a plausible-looking zero tensor; that can introduce silent data corruption. If skipping is appropriate, record the error and compare the skipped count against the original inventory.
