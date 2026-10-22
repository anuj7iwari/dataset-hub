# ========================================
# Universal Dataset Container and Pipelines
# ========================================

import os
import json
import random
import copy
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Iterator, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from data_hub.schemas import (
    ModalityType,
    SourceType,
    ReturnType,
    ErrorPolicy,
    DatasetItem,
    DatasetConfig,
    DatasetSummary,
)
from data_hub.media import (
    load_image,
    image_to_tensor,
    image_to_numpy,
    load_audio,
    audio_to_tensor,
    load_video,
)
from data_hub.collators import UniversalBatchCollator


# ========================================
# PyTorch Adapter Engine
# ========================================

class PyTorchUniversalAdapter(Dataset):
    """PyTorch Dataset adapter for UniversalDataset items."""

    def __init__(
        self,
        items: List[DatasetItem],
        modality: ModalityType = ModalityType.TEXT,
        tokenizer_or_processor: Optional[Any] = None,
        max_length: int = 128,
        image_size: Tuple[int, int] = (224, 224),
        audio_sr: int = 16000,
        max_video_frames: int = 16,
        error_policy: ErrorPolicy = ErrorPolicy.STRICT,
        pre_skipped: Optional[List[Dict[str, Any]]] = None,
    ):
        self.items = list(items) if isinstance(items, (list, tuple)) else items
        self.modality = modality
        self.tokenizer_or_processor = tokenizer_or_processor
        self.max_length = max_length
        self.image_size = image_size
        self.audio_sr = audio_sr
        self.max_video_frames = max_video_frames
        self.error_policy = error_policy
        self.skipped_samples: List[Dict[str, Any]] = list(pre_skipped) if pre_skipped else []

        # When report_and_skip is set, filter invalid media samples ahead of training
        if self.error_policy == ErrorPolicy.REPORT_AND_SKIP and isinstance(self.items, list):
            valid_items = []
            for it in self.items:
                is_valid = True
                err_str = None
                if self.modality in (ModalityType.IMAGE, ModalityType.MULTIMODAL):
                    src = it.raw_data.get("image") if it.raw_data and "image" in it.raw_data else it.media_path
                    if src is not None:
                        try:
                            load_image(src)
                        except Exception as e:
                            is_valid = False
                            err_str = str(e)
                    else:
                        is_valid = False
                        err_str = "Missing image source"
                elif self.modality == ModalityType.AUDIO:
                    src = it.raw_data.get("audio") if it.raw_data and "audio" in it.raw_data else it.media_path
                    if src is not None:
                        try:
                            load_audio(src, target_sr=self.audio_sr)
                        except Exception as e:
                            is_valid = False
                            err_str = str(e)
                    else:
                        is_valid = False
                        err_str = "Missing audio source"

                if is_valid:
                    valid_items.append(it)
                else:
                    self.skipped_samples.append({
                        "id": it.id,
                        "media_path": it.media_path,
                        "error": err_str,
                    })
            self.items = valid_items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        res: Dict[str, Any] = {"id": item.id}

        # ========================================
        # 1. Text Modality
        # ========================================
        if self.modality == ModalityType.TEXT:
            text = item.text or ""
            if self.tokenizer_or_processor is not None:
                encoded = self.tokenizer_or_processor(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,  # Dynamic padding via UniversalBatchCollator
                    return_tensors="pt",
                )
                res.update({k: v.squeeze(0) for k, v in encoded.items()})
                if "input_ids" in res and "labels" not in res:
                    res["labels"] = res["input_ids"].clone()
            else:
                res["text"] = text

            if item.label is not None:
                res["label"] = item.label
            return res

        # ========================================
        # 2. Image / Vision Modality
        # ========================================
        elif self.modality == ModalityType.IMAGE:
            image_source = (
                item.raw_data.get("image")
                if item.raw_data and "image" in item.raw_data
                else item.media_path
            )
            if image_source is not None:
                try:
                    res["pixel_values"] = image_to_tensor(
                        image_source,
                        processor_or_transform=self.tokenizer_or_processor,
                        target_size=self.image_size,
                    )
                except Exception as e:
                    if self.error_policy == ErrorPolicy.STRICT:
                        raise RuntimeError(
                            f"Failed to load image for sample {item.id} (path={item.media_path}): {str(e)}"
                        ) from e
                    res["_invalid"] = True
                    res["error"] = str(e)
            elif self.error_policy == ErrorPolicy.STRICT:
                raise ValueError(f"Image sample {item.id} has no valid media_path or image bytes.")
            else:
                res["_invalid"] = True
                res["error"] = "Missing image source"

            if item.label is not None:
                res["label"] = item.label
            if item.media_path:
                res["media_path"] = item.media_path
            return res

        # ========================================
        # 3. Audio Modality
        # ========================================
        elif self.modality == ModalityType.AUDIO:
            audio_source = (
                item.raw_data.get("audio")
                if item.raw_data and "audio" in item.raw_data
                else item.media_path
            )
            if audio_source is not None:
                try:
                    tensor_out = audio_to_tensor(
                        audio_source,
                        processor=self.tokenizer_or_processor,
                        target_sr=self.audio_sr,
                    )
                    # When a processor is used, map output tensor to input_features
                    if self.tokenizer_or_processor is not None:
                        res["input_features"] = tensor_out
                    else:
                        res["waveform"] = tensor_out
                except Exception as e:
                    if self.error_policy == ErrorPolicy.STRICT:
                        raise RuntimeError(
                            f"Failed to load audio for sample {item.id} (path={item.media_path}): {str(e)}"
                        ) from e
                    res["_invalid"] = True
                    res["error"] = str(e)
            elif self.error_policy == ErrorPolicy.STRICT:
                raise ValueError(f"Audio sample {item.id} has no valid media_path or audio data.")
            else:
                res["_invalid"] = True
                res["error"] = "Missing audio source"

            if item.text:
                res["text"] = item.text
            if item.label is not None:
                res["label"] = item.label
            return res

        # ========================================
        # 4. Video Modality
        # ========================================
        elif self.modality == ModalityType.VIDEO:
            if item.media_path:
                try:
                    res["video_frames"] = load_video(
                        item.media_path,
                        max_frames=self.max_video_frames,
                        target_size=self.image_size,
                    )
                except Exception as e:
                    if self.error_policy == ErrorPolicy.STRICT:
                        raise RuntimeError(
                            f"Failed to load video for sample {item.id} (path={item.media_path}): {str(e)}"
                        ) from e
                    res["_invalid"] = True
                    res["error"] = str(e)
            elif self.error_policy == ErrorPolicy.STRICT:
                raise ValueError(f"Video sample {item.id} has no valid media_path.")
            else:
                res["_invalid"] = True
                res["error"] = "Missing video source"

            if item.label is not None:
                res["label"] = item.label
            return res

        # ========================================
        # 5. Multimodal Modality (Vision + Language)
        # ========================================
        elif self.modality == ModalityType.MULTIMODAL:
            # Disambiguate visual processor from pure text tokenizer
            proc = self.tokenizer_or_processor
            visual_proc = None
            text_tok = None
            if proc is not None:
                if hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor"):
                    visual_proc = getattr(proc, "image_processor", None) or getattr(proc, "feature_extractor", None)
                    text_tok = getattr(proc, "tokenizer", None) or (proc if hasattr(proc, "__call__") else None)
                elif hasattr(proc, "tokenizer"):
                    visual_proc = proc
                    text_tok = getattr(proc, "tokenizer")
                elif hasattr(proc, "preprocess"):
                    visual_proc = proc
                elif (
                    hasattr(proc, "pad_token_id")
                    or hasattr(proc, "vocab_size")
                    or hasattr(proc, "encode")
                    or "token" in type(proc).__name__.lower()
                ):
                    # Pure text tokenizer (e.g., PreTrainedTokenizer, MockTokenizer)
                    text_tok = proc
                    visual_proc = None
                else:
                    visual_proc = proc
                    text_tok = proc

            # Process Visual Component
            image_source = (
                item.raw_data.get("image")
                if item.raw_data and "image" in item.raw_data
                else item.media_path
            )
            if image_source is not None:
                try:
                    res["pixel_values"] = image_to_tensor(
                        image_source,
                        processor_or_transform=visual_proc,
                        target_size=self.image_size,
                    )
                except Exception as e:
                    if self.error_policy == ErrorPolicy.STRICT:
                        raise RuntimeError(f"Multimodal image loading failed for sample {item.id}: {str(e)}") from e
                    res["_invalid"] = True
                    res["image_error"] = str(e)
            elif self.error_policy == ErrorPolicy.STRICT:
                raise ValueError(f"Multimodal sample {item.id} has no image data or media_path.")
            else:
                res["_invalid"] = True
                res["image_error"] = "Missing image data"

            # Process Text Component
            text = item.text or ""
            if text_tok is not None:
                try:
                    encoded = text_tok(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding=False,
                        return_tensors="pt",
                    )
                    res.update({k: v.squeeze(0) for k, v in encoded.items()})
                except Exception as e:
                    if self.error_policy == ErrorPolicy.STRICT:
                        raise RuntimeError(f"Multimodal text processor failed on sample {item.id}: {str(e)}") from e
                    res["_invalid"] = True
                    res["text_error"] = str(e)
            else:
                if self.error_policy != ErrorPolicy.PERMISSIVE or not res.get("_invalid"):
                    res["text"] = text

            if item.label is not None:
                res["label"] = item.label
            return res

        # ========================================
        # 6. Tabular / JSON Fallback
        # ========================================
        if item.raw_data:
            res.update(item.raw_data)
        if item.text:
            res["text"] = item.text
        if item.label is not None:
            res["label"] = item.label
        return res


class PyTorchIterableAdapter(IterableDataset):
    """Streaming PyTorch IterableDataset adapter with multi-worker sharding."""

    def __init__(
        self,
        dataset_sequence: Any,
        modality: ModalityType = ModalityType.TEXT,
        tokenizer_or_processor: Optional[Any] = None,
        max_length: int = 128,
        image_size: Tuple[int, int] = (224, 224),
        audio_sr: int = 16000,
        max_video_frames: int = 16,
        error_policy: ErrorPolicy = ErrorPolicy.STRICT,
    ):
        self.dataset_sequence = dataset_sequence
        self.modality = modality
        self.tokenizer_or_processor = tokenizer_or_processor
        self.max_length = max_length
        self.image_size = image_size
        self.audio_sr = audio_sr
        self.max_video_frames = max_video_frames
        self.error_policy = error_policy

    def _transform_item(self, item: DatasetItem) -> Dict[str, Any]:
        # Lazy streams may infer modality from their first validated row. Read the
        # updated sequence config here instead of freezing TEXT at DataLoader creation.
        live_cfg = getattr(self.dataset_sequence, "cfg", None)
        live_modality = getattr(live_cfg, "modality", self.modality)
        adapter = PyTorchUniversalAdapter(
            items=[item],
            modality=live_modality,
            tokenizer_or_processor=self.tokenizer_or_processor,
            max_length=self.max_length,
            image_size=self.image_size,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            error_policy=self.error_policy,
        )
        return adapter[0]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        idx = 0
        for item in self.dataset_sequence:
            if idx % num_workers == worker_id:
                sample = self._transform_item(item)
                yield sample
            idx += 1


# ========================================
# Dataset Splits Container
# ========================================

class DatasetSplits:
    """Holds train, validation, and test subsets of a UniversalDataset."""

    def __init__(
        self,
        train: "UniversalDataset",
        val: Optional["UniversalDataset"] = None,
        test: Optional["UniversalDataset"] = None,
    ):
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self) -> str:
        v_len = len(self.val) if self.val else 0
        t_len = len(self.test) if self.test else 0
        return f"<DatasetSplits train={len(self.train)} val={v_len} test={t_len}>"

    def __getitem__(self, key: str) -> "UniversalDataset":
        key_lower = key.lower()
        if key_lower in ("train", "training"):
            return self.train
        elif key_lower in ("val", "valid", "validation"):
            if self.val is None:
                raise KeyError("Validation split is not present in this dataset.")
            return self.val
        elif key_lower in ("test", "testing"):
            if self.test is None:
                raise KeyError("Test split is not present in this dataset.")
            return self.test
        raise KeyError(f"Unknown split name '{key}'. Expected 'train', 'val', or 'test'.")


# ========================================
# Universal Dataset Container
# ========================================

class UniversalDataset:
    """Central dataset container for tabular, text, vision, audio, and video datasets."""

    def __init__(
        self,
        items: Union[List[DatasetItem], Any],
        modality: ModalityType = ModalityType.TEXT,
        source_type: SourceType = SourceType.LOCAL_FILE,
        return_type: ReturnType = ReturnType.DATASET,
        error_policy: ErrorPolicy = ErrorPolicy.STRICT,
        name: str = "custom_dataset",
        columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        audio_sr: int = 16000,
        max_video_frames: int = 16,
        config: Optional[DatasetConfig] = None,
    ):
        # Each eager dataset owns its mutable items. Derived datasets must not share
        # mutable Pydantic models or nested raw_data with the parent dataset.
        self.items = copy.deepcopy(list(items)) if isinstance(items, (list, tuple)) else items
        # Every dataset owns its configuration and metadata. A streaming sequence is
        # allowed to share this config internally, but never with a parent/sibling dataset.
        self.config = config.model_copy(deep=True) if config is not None else None
        self._modality = ModalityType.from_str(modality)
        self.source_type = SourceType.from_str(source_type)
        self.return_type = ReturnType.from_str(return_type)
        self.error_policy = ErrorPolicy.from_str(error_policy)
        self.name = name
        self.audio_sr = audio_sr
        self.max_video_frames = max_video_frames
        if self.config is not None:
            self.config.modality = self._modality
            self.config.source_type = self.source_type
            self.config.return_type = self.return_type
            self.config.error_policy = self.error_policy
            self.config.target_sr = audio_sr
            self.config.max_video_frames = max_video_frames
            if hasattr(self.items, "cfg"):
                # Keep lazy modality inference synchronized with this dataset's config.
                self.items.cfg = self.config
                stream_source = getattr(self.items, "stream_source", None)
                if stream_source is not None and hasattr(stream_source, "cfg"):
                    stream_source.cfg = self.config

        # Handle column detection
        sample_items = []
        if isinstance(self.items, (list, tuple)):
            sample_items = self.items[:50]
        self.columns = list(columns) if columns is not None else []
        if not self.columns:
            cols = []
            if any(getattr(it, "text", None) is not None for it in sample_items):
                cols.append("text")
            if any(getattr(it, "label", None) is not None for it in sample_items):
                cols.append("label")
            if any(getattr(it, "media_path", None) is not None for it in sample_items):
                cols.append("media_path")
            if sample_items and getattr(sample_items[0], "raw_data", None):
                for k in sample_items[0].raw_data.keys():
                    if k not in cols:
                        cols.append(k)
            self.columns = cols if cols else ["text"]

        # A lazy parser sequence maintains live parsing/skipping counters in its state.
        # Reuse that state only inside its owning wrapper; all other metadata is copied.
        if metadata is not None and getattr(self.items, "state", None) is metadata:
            self.metadata = metadata
        else:
            self.metadata = copy.deepcopy(metadata) if metadata is not None else {}
        if not self.is_streaming and hasattr(items, "__len__"):
            try:
                if len(items) == 0:
                    if "warnings" not in self.metadata:
                        self.metadata["warnings"] = ["Dataset is empty (0 samples)."]
                    if "warning" not in self.metadata:
                        self.metadata["warning"] = "Dataset is empty (0 samples)."
            except TypeError:
                pass

    @property
    def modality(self) -> ModalityType:
        if self.config and self.config.modality:
            return self.config.modality
        return self._modality

    @modality.setter
    def modality(self, val: Any) -> None:
        self._modality = ModalityType.from_str(val) if val is not None else ModalityType.TEXT
        if self.config:
            self.config.modality = self._modality

    @property
    def is_streaming(self) -> bool:
        from data_hub.parsers import LazyDatasetSequence
        if isinstance(self.items, (list, tuple)):
            return False
        return (
            isinstance(self.items, LazyDatasetSequence)
            or (hasattr(self.items, "__iter__") and not hasattr(self.items, "__getitem__"))
        )

    @property
    def num_samples(self) -> Optional[int]:
        if self.is_streaming:
            try:
                return len(self.items)
            except TypeError:
                return None
        if hasattr(self.items, "__len__"):
            return len(self.items)
        return None

    def __len__(self) -> int:
        if self.is_streaming:
            return len(self.items)
        if hasattr(self.items, "__len__"):
            return len(self.items)
        raise TypeError("Streaming dataset with unknown length does not support len().")

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if self.is_streaming:
            raise TypeError("Streaming datasets do not support random access or indexing.")

        if isinstance(idx, slice):
            return UniversalDataset(
                items=self.items[idx],
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_slice",
                columns=self.columns,
                metadata=self.metadata,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=self.config,
            )

        item = self.items[idx]
        if self.return_type == ReturnType.PYTORCH:
            adapter = PyTorchUniversalAdapter(
                items=[item],
                modality=self.modality,
                error_policy=self.error_policy,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
            )
            return adapter[0]

        elif self.return_type == ReturnType.NUMPY:
            if self.modality == ModalityType.IMAGE and (item.media_path or (item.raw_data and "image" in item.raw_data)):
                src = item.raw_data.get("image") if item.raw_data and "image" in item.raw_data else item.media_path
                return {"image": image_to_numpy(src), "label": item.label}
            return item.raw_data if item.raw_data else item.model_dump()

        elif self.return_type == ReturnType.RAW:
            return item.raw_data if item.raw_data else item.model_dump()

        return item

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)

    # ========================================
    # Inspection & Exploration
    # ========================================

    def preview(self, n: int = 5, detailed: bool = False) -> List[Dict[str, Any]]:
        """Returns JSON-serializable preview objects with sample snippets and labels."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"preview(n=...) requires a non-negative integer, got {n!r}")
        if self.is_streaming and hasattr(self.items, "peek"):
            sample_items = self.items.peek(n)
        elif isinstance(self.items, (list, tuple)):
            sample_items = self.items[:n]
        else:
            sample_items = []

        preview_samples = []
        for it in sample_items:
            rec: Dict[str, Any] = {
                "id": it.id,
                "modality": self.modality.value,
                "label": it.label,
            }
            if it.text:
                rec["text_snippet"] = (it.text[:140] + "...") if len(it.text) > 140 else it.text
                rec["char_length"] = len(it.text)
            if it.media_path:
                rec["media_path"] = it.media_path
                media_candidate = Path(it.media_path)
                if not media_candidate.is_absolute() and self.is_streaming:
                    base_dir = getattr(self.items, "base_dir", None)
                    if base_dir is not None:
                        media_candidate = Path(base_dir) / media_candidate
                rec["file_exists"] = media_candidate.is_file()
            if detailed and it.metadata:
                rec["metadata"] = it.metadata
            if detailed and it.raw_data:
                rec["raw_keys"] = list(it.raw_data.keys())
            preview_samples.append(rec)
        return preview_samples

    def get_summary(self) -> DatasetSummary:
        """Compiles complete dataset statistics, schema, and multilingual token estimates."""
        sample_preview = self.preview(n=5)
        total = self.num_samples
        is_streaming = self.is_streaming

        # Token estimation (Multilingual & Code aware: ~3.2 chars/token average)
        estimated_tokens = None
        method = None
        if self.modality in (ModalityType.TEXT, ModalityType.MULTIMODAL) and total is not None and total > 0:
            if is_streaming:
                from data_hub.parsers import LazyDatasetSequence
                if isinstance(self.items, LazyDatasetSequence):
                    peeked = self.items.peek(min(100, total))
                    sample_chars = sum(len(it.text or "") for it in peeked)
                    avg_chars = sample_chars / max(1, len(peeked))
                else:
                    sample_chars = sum(len(it.get("text_snippet", "") or "") for it in sample_preview)
                    avg_chars = sample_chars / max(1, len(sample_preview))
            else:
                sample_chars = sum(len(it.text or "") for it in self.items[:100])
                avg_chars = sample_chars / min(100, total)
            estimated_tokens = int((avg_chars / 3.2) * total)
            method = "multilingual_char_heuristic (~3.2 chars/token)"

        # Class distribution if discrete labels exist
        class_dist: Dict[str, int] = {}
        if not is_streaming and isinstance(self.items, (list, tuple)):
            for it in self.items[:5000]:
                if it.label is not None:
                    lbl = str(it.label)
                    class_dist[lbl] = class_dist.get(lbl, 0) + 1

        # Memory footprint estimation
        memory_mb = None
        if total is not None and not is_streaming and self.items:
            import sys
            sample_size = sys.getsizeof(self.items[0])
            memory_mb = round((sample_size * total * 2) / (1024 * 1024), 2)

        return DatasetSummary(
            name=self.name,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            total_samples=total,
            is_streaming=is_streaming,
            columns=self.columns,
            splits={"total": total} if total is not None else {},
            sample_preview=sample_preview,
            estimated_tokens=estimated_tokens,
            token_estimation_method=method,
            class_distribution=class_dist if class_dist else None,
            memory_mb=memory_mb,
            skipped_samples=self.metadata.get("skipped_samples", 0),
            parsing_errors=self.metadata.get("parsing_errors", []),
            warnings=self.metadata.get("warnings", []),
            warning=self.metadata.get("warning"),
            included_files=self.metadata.get("included_files"),
            excluded_files=self.metadata.get("excluded_files"),
            metadata=self.metadata,
        )

    summary = get_summary

    # ========================================
    # Sorting, Filtering & Manipulation
    # ========================================

    def sort(
        self,
        by: Optional[str] = None,
        descending: bool = False,
        key: Optional[Callable[[DatasetItem], Any]] = None,
    ) -> "UniversalDataset":
        """Sorts dataset items by length, label, column, or custom key."""
        if self.is_streaming:
            raise TypeError("Streaming datasets cannot be sorted.")

        if key is not None:
            sort_key = lambda it: key(it.model_copy(deep=True))
        elif by in ("length", "len", "char_length"):
            sort_key = lambda it: len(it.text or "")
        elif by in ("label", "target", "class"):
            sort_key = lambda it: it.label
        elif by == "id":
            # Stable string ordering is the documented behavior for mixed int/string IDs.
            sort_key = lambda it: str(it.id) if it.id is not None else None
        elif by and any(it.raw_data and by in it.raw_data for it in self.items):
            sort_key = lambda it: (it.raw_data or {}).get(by)
        else:
            sort_key = lambda it: it.text or ""

        def normalized_key(value):
            # Comparable, deterministic ordering across nullable and mixed columns.
            if value is None:
                return (1, 0, "")
            if isinstance(value, bool):
                return (0, 0, int(value))
            if isinstance(value, (int, float, np.number)):
                number = float(value)
                if np.isnan(number):
                    return (1, 0, "")
                return (0, 0, number)
            if isinstance(value, str):
                return (0, 1, value.casefold())
            if hasattr(value, "isoformat"):
                try:
                    return (0, 2, value.isoformat())
                except Exception:
                    pass
            return (0, 3, type(value).__name__, repr(value))

        keyed = [(sort_key(it), index, it) for index, it in enumerate(self.items)]
        present = [(v, i, it) for v, i, it in keyed if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v))]
        missing = [(v, i, it) for v, i, it in keyed if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))]
        present.sort(key=lambda row: (normalized_key(row[0]), row[1]), reverse=descending)
        missing.sort(key=lambda row: row[1])
        sorted_items = [row[2] for row in present + missing]
        return UniversalDataset(
            items=sorted_items,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_sorted",
            columns=self.columns,
            metadata=self.metadata,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )

    def take(self, n: int) -> "UniversalDataset":
        """Returns the first n samples as a materialized, non-streaming UniversalDataset."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"take() requires a non-negative integer count, got {n!r}")
        if self.is_streaming:
            consumed = []
            if n > 0:
                it = iter(self)
                for _ in range(n):
                    try:
                        consumed.append(next(it))
                    except StopIteration:
                        break
            meta = dict(self.metadata)
            meta["streaming"] = False
            cfg = self.config.model_copy(deep=True) if self.config else None
            if cfg:
                cfg.streaming = False
                cfg.max_samples = n
            return UniversalDataset(
                items=consumed,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_take_{n}",
                columns=self.columns,
                metadata=meta,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=cfg,
            )
        else:
            meta = dict(self.metadata)
            cfg = self.config.model_copy(deep=True) if self.config else None
            if cfg:
                cfg.max_samples = n
            return UniversalDataset(
                items=list(self.items[:n]),
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_take_{n}",
                columns=self.columns,
                metadata=meta,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=cfg,
            )

    def filter(self, condition: Callable[[DatasetItem], bool]) -> "UniversalDataset":
        """Filters dataset items matching the given predicate function."""
        if self.is_streaming:
            def _gen():
                for it in self:
                    callback_item = it.model_copy(deep=True)
                    if condition(callback_item):
                        yield callback_item
            from data_hub.parsers import LazyDatasetSequence
            cfg = self.config.model_copy(deep=True) if self.config else DatasetConfig(path_or_identifier="stream", streaming=True)
            cfg.streaming = True
            new_seq = LazyDatasetSequence(_gen(), cfg)
            meta = copy.deepcopy(self.metadata)
            meta["streaming"] = True
            new_seq.state = meta
            return UniversalDataset(
                items=new_seq,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_filtered",
                columns=self.columns,
                metadata=meta,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=cfg,
            )
        else:
            filtered = [copy.deepcopy(it) for it in self.items if condition(it.model_copy(deep=True))]
            return UniversalDataset(
                items=filtered,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_filtered",
                columns=self.columns,
                metadata=self.metadata,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=self.config,
            )

    def encode_labels(self, mapping: Optional[Dict[str, int]] = None) -> "UniversalDataset":
        """Encodes string/discrete labels to continuous integers."""
        if self.is_streaming:
            raise TypeError("encode_labels does not support streaming datasets without materialization.")
        if mapping is None:
            unique_labels = sorted(list(set(str(it.label) for it in self.items if it.label is not None)))
            mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
        new_items = []
        for it in self.items:
            cloned = it.model_copy(deep=True)
            if cloned.label is not None:
                cloned.label = mapping.get(str(cloned.label), cloned.label)
            new_items.append(cloned)
        new_meta = dict(self.metadata)
        new_meta["label_mapping"] = mapping
        return UniversalDataset(
            items=new_items,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_encoded",
            columns=self.columns,
            metadata=new_meta,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )

    def _map_item(self, it: DatasetItem, fn: Callable[[DatasetItem], Any]) -> DatasetItem:
        """Applies a map callback to a single item, validating its return type and isolating raw_data."""
        # The callback gets an isolated object so in-place edits inside fn don't mutate
        # the source dataset even if fn returns that same object.
        callback_item = it.model_copy(deep=True)
        res = fn(callback_item)
        if isinstance(res, DatasetItem):
            return res.model_copy(deep=True)
        if isinstance(res, dict):
            cloned = it.model_copy(deep=True)
            cloned.raw_data = copy.deepcopy(it.raw_data) if it.raw_data else {}
            if "text" in res:
                cloned.text = res["text"]
            if "label" in res:
                cloned.label = res["label"]
            if "media_path" in res:
                cloned.media_path = res["media_path"]
            cloned.raw_data.update(copy.deepcopy(res))
            return cloned
        raise TypeError(
            f"map() callback must return a DatasetItem or dict, got {type(res).__name__} "
            f"for sample id={it.id!r}."
        )

    def map(self, fn: Callable[[DatasetItem], Union[DatasetItem, Dict[str, Any]]]) -> "UniversalDataset":
        """Applies a transformation function to each dataset item."""
        if self.is_streaming:
            def _gen():
                for it in self:
                    yield self._map_item(it, fn)
            from data_hub.parsers import LazyDatasetSequence
            cfg = self.config.model_copy(deep=True) if self.config else DatasetConfig(path_or_identifier="stream", streaming=True)
            cfg.streaming = True
            new_seq = LazyDatasetSequence(_gen(), cfg)
            meta = copy.deepcopy(self.metadata)
            meta["streaming"] = True
            new_seq.state = meta
            return UniversalDataset(
                items=new_seq,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_mapped",
                columns=self.columns,
                metadata=meta,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=cfg,
            )

        mapped_items: List[DatasetItem] = [self._map_item(it, fn) for it in self.items]

        return UniversalDataset(
            items=mapped_items,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_mapped",
            columns=self.columns,
            metadata=self.metadata,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )

    def clean(
        self,
        deduplicate: bool = True,
        strip_whitespace: bool = True,
        drop_na: bool = True,
        drop_corrupted_media: bool = False,
    ) -> "UniversalDataset":
        """Cleans dataset by stripping whitespace, deduplicating, and removing missing/corrupt items."""
        def _clean_one(it: DatasetItem, seen_keys: set) -> Optional[DatasetItem]:
            if drop_na:
                if self.modality == ModalityType.TEXT and (not it.text or not it.text.strip()):
                    return None
                if self.modality in (ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.VIDEO):
                    has_media = bool(it.media_path or (it.raw_data and ("image" in it.raw_data or "audio" in it.raw_data)))
                    if not has_media:
                        return None

            if drop_corrupted_media and it.media_path:
                if not os.path.isfile(it.media_path) or os.path.getsize(it.media_path) == 0:
                    return None
                ext = Path(it.media_path).suffix.lower()
                from data_hub.media import IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, load_image, load_audio, load_video
                try:
                    if ext in IMAGE_EXTENSIONS or self.modality == ModalityType.IMAGE:
                        load_image(it.media_path)
                    elif ext in AUDIO_EXTENSIONS or self.modality == ModalityType.AUDIO:
                        load_audio(it.media_path, target_sr=self.audio_sr)
                    elif ext in VIDEO_EXTENSIONS or self.modality == ModalityType.VIDEO:
                        load_video(it.media_path, max_frames=self.max_video_frames)
                except Exception:
                    return None

            cur_text = it.text.strip() if it.text and strip_whitespace else it.text

            # Never collapse contradictory supervised examples just because their text
            # matches. For media without a path, use a content fingerprint when possible;
            # if neither payload nor a stable ID exists, skip deduplication for that row.
            label_key = json.dumps(it.label, sort_keys=True, ensure_ascii=False, default=str)

            def _payload_identity(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    return "bytes:" + hashlib.sha256(value).hexdigest()
                if isinstance(value, bytearray):
                    return "bytes:" + hashlib.sha256(bytes(value)).hexdigest()
                if isinstance(value, np.ndarray):
                    arr = np.ascontiguousarray(value)
                    return "ndarray:" + hashlib.sha256(
                        str(arr.dtype).encode() + repr(arr.shape).encode() + arr.tobytes()
                    ).hexdigest()
                if isinstance(value, torch.Tensor):
                    return _payload_identity(value.detach().cpu().contiguous().numpy())
                if isinstance(value, dict):
                    pieces = [(str(k), _payload_identity(v)) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
                    return "dict:" + hashlib.sha256(json.dumps(pieces, ensure_ascii=False).encode()).hexdigest()
                if isinstance(value, (list, tuple)):
                    pieces = [_payload_identity(v) for v in value]
                    return "sequence:" + hashlib.sha256(json.dumps(pieces, ensure_ascii=False).encode()).hexdigest()
                if hasattr(value, "tobytes") and hasattr(value, "size") and hasattr(value, "mode"):
                    try:
                        return "image:" + hashlib.sha256(
                            str((value.mode, value.size)).encode() + value.tobytes()
                        ).hexdigest()
                    except Exception:
                        pass
                try:
                    return "json:" + json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
                except Exception:
                    return "repr:" + repr(value)

            dedup_key = None
            if self.modality == ModalityType.TEXT:
                if cur_text:
                    dedup_key = ("text", cur_text, label_key)
            else:
                if it.media_path:
                    dedup_key = ("path", os.path.normcase(os.path.normpath(str(it.media_path))), label_key)
                else:
                    raw = it.raw_data or {}
                    payload = None
                    for media_key in ("image", "audio", "video", "media", "pixel_values", "waveform"):
                        if media_key in raw and raw[media_key] is not None:
                            payload = raw[media_key]
                            break
                    if payload is not None:
                        dedup_key = ("embedded", _payload_identity(payload), label_key)
                    elif it.id is not None:
                        dedup_key = ("id", type(it.id).__name__, str(it.id), label_key)

            if deduplicate and dedup_key is not None:
                if dedup_key in seen_keys:
                    return None
                seen_keys.add(dedup_key)

            cloned = it.model_copy(deep=True)
            cloned.text = cur_text
            return cloned

        if self.is_streaming:
            def _gen():
                seen = set()
                for it in self:
                    out = _clean_one(it, seen)
                    if out is not None:
                        yield out
            from data_hub.parsers import LazyDatasetSequence
            cfg = self.config.model_copy(deep=True) if self.config else DatasetConfig(path_or_identifier="stream", streaming=True)
            cfg.streaming = True
            new_seq = LazyDatasetSequence(_gen(), cfg)
            meta = copy.deepcopy(self.metadata)
            meta["streaming"] = True
            meta["cleaned"] = True
            new_seq.state = meta
            return UniversalDataset(
                items=new_seq,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_cleaned",
                columns=self.columns,
                metadata=meta,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=cfg,
            )

        cleaned_items: List[DatasetItem] = []
        seen_keys = set()
        for it in self.items:
            out = _clean_one(it, seen_keys)
            if out is not None:
                cleaned_items.append(out)

        return UniversalDataset(
            items=cleaned_items,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_cleaned",
            columns=self.columns,
            metadata={**self.metadata, "cleaned": True},
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )

    def _format_one(
        self,
        it: DatasetItem,
        template: str,
        column_mapping: Optional[Dict[str, str]],
        safe: bool,
    ) -> DatasetItem:
        ctx: Dict[str, Any] = {}
        if it.raw_data:
            ctx.update(it.raw_data)
        if it.text:
            ctx["text"] = it.text
        if it.label is not None:
            ctx["label"] = str(it.label)

        if column_mapping:
            for target_k, source_k in column_mapping.items():
                ctx[target_k] = ctx.get(source_k, "")

        new_text = it.text
        try:
            class SafeDict(dict):
                def __missing__(self, key):
                    return f"{{{key}}}" if safe else ""

            new_text = template.format_map(SafeDict(ctx))
        except Exception:
            new_text = it.text

        cloned = it.model_copy(deep=True)
        cloned.raw_data = copy.deepcopy(it.raw_data) if it.raw_data else {}
        cloned.text = new_text
        cloned.raw_data["text"] = new_text
        return cloned

    def format_prompt(
        self,
        template: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        safe: bool = True,
    ) -> "UniversalDataset":
        """Formats text samples using instruction templates."""
        if not template:
            return self

        if self.is_streaming:
            def _gen():
                for it in self:
                    yield self._format_one(it, template, column_mapping, safe)
            from data_hub.parsers import LazyDatasetSequence
            cfg = self.config.model_copy(deep=True) if self.config else DatasetConfig(path_or_identifier="stream", streaming=True)
            cfg.streaming = True
            new_seq = LazyDatasetSequence(_gen(), cfg)
            meta = copy.deepcopy(self.metadata)
            meta["streaming"] = True
            new_seq.state = meta
            return UniversalDataset(
                items=new_seq,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_formatted",
                columns=self.columns,
                metadata=meta,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=cfg,
            )

        formatted_items: List[DatasetItem] = [
            self._format_one(it, template, column_mapping, safe) for it in self.items
        ]

        return UniversalDataset(
            items=formatted_items,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_formatted",
            columns=self.columns,
            metadata=self.metadata,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        seed: int = 42,
    ) -> "UniversalDataset":
        """Randomly samples N elements or a fraction from the dataset."""
        if self.is_streaming:
            raise NotImplementedError(
                "Random sampling on a streaming dataset is not supported as it requires "
                "materializing the entire stream. Use .take(n) to draw the first N items instead."
            )
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
            raise ValueError(f"sample(n=...) requires a non-negative integer, got {n!r}")
        if frac is not None and (not isinstance(frac, (int, float)) or isinstance(frac, bool) or not np.isfinite(frac) or frac < 0 or frac > 1):
            raise ValueError(f"sample(frac=...) requires a finite value between 0 and 1, got {frac!r}")
        if n is not None and frac is not None:
            raise ValueError("Specify either n or frac for sample(), not both.")
        if not self.items:
            return self

        rng = random.Random(seed)
        total = len(self.items)

        if frac is not None:
            target_count = int(total * frac)
        elif n is not None:
            target_count = min(n, total)
        else:
            return self

        sampled = rng.sample(self.items, target_count)
        return UniversalDataset(
            items=sampled,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_sample_{target_count}",
            columns=self.columns,
            metadata=self.metadata,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> DatasetSplits:
        """Splits dataset into train, validation, and test subsets."""
        if self.is_streaming:
            raise NotImplementedError(
                "Splitting a streaming or one-shot iterable dataset is not supported because "
                "it would exhaust the underlying stream. Materialize the dataset into memory first "
                "or specify pre-defined train/val/test splits."
            )

        # Validate ratios so overcommitted or negative splits fail loudly instead of
        # silently reassigning leftover records.
        for name, r in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
            if not isinstance(r, (int, float)) or isinstance(r, bool) or not np.isfinite(r) or r < 0 or r > 1:
                raise ValueError(f"{name}_ratio must be a finite number between 0 and 1, got {r!r}")
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Split ratios sum to {total_ratio}; they must sum to 1.0 so every sample is assigned "
                "to exactly one of train, validation, or test. Adjust the ratios explicitly rather than "
                "silently dropping or reallocating records."
            )

        items = list(self.items)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)

        total = len(items)
        if test_ratio == 0:
            val_count = int(total * val_ratio)
            train_items = items[:total - val_count]
            val_items = items[total - val_count:]
            test_items = []
        else:
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            train_items = items[:train_end]
            val_items = items[train_end:val_end]
            test_items = items[val_end:]

        train_ds = UniversalDataset(
            items=train_items,
            modality=self.modality,
            source_type=self.source_type,
            return_type=self.return_type,
            error_policy=self.error_policy,
            name=f"{self.name}_train",
            columns=self.columns,
            metadata=self.metadata,
            audio_sr=self.audio_sr,
            max_video_frames=self.max_video_frames,
            config=self.config,
        )
        val_ds = (
            UniversalDataset(
                items=val_items,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_val",
                columns=self.columns,
                metadata=self.metadata,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=self.config,
            )
            if val_items
            else None
        )
        test_ds = (
            UniversalDataset(
                items=test_items,
                modality=self.modality,
                source_type=self.source_type,
                return_type=self.return_type,
                error_policy=self.error_policy,
                name=f"{self.name}_test",
                columns=self.columns,
                metadata=self.metadata,
                audio_sr=self.audio_sr,
                max_video_frames=self.max_video_frames,
                config=self.config,
            )
            if test_items
            else None
        )

        return DatasetSplits(train=train_ds, val=val_ds, test=test_ds)

    # ========================================
    # Framework Integration
    # ========================================

    def _infer_streaming_modality_for_training(self, sample_size: int = 10) -> None:
        """Peek a bounded prefix before adapter construction when modality was implicit."""
        if not self.is_streaming or not self.config or self.config.kwargs.get("_explicit_modality"):
            return
        if self.modality != ModalityType.TEXT or not hasattr(self.items, "peek"):
            return
        sample_items = self.items.peek(sample_size)
        if not sample_items:
            return
        has_audio = has_image = has_video = has_text = False
        for item in sample_items:
            raw = {str(k).lower(): v for k, v in (item.raw_data or {}).items()}
            ext = Path(item.media_path).suffix.lower() if item.media_path else ""
            has_audio |= bool(ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"} or raw.get("audio") is not None or raw.get("audio_path") is not None)
            has_video |= bool(ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"} or raw.get("video") is not None or raw.get("video_path") is not None)
            has_image |= bool(ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} or raw.get("image") is not None or raw.get("image_path") is not None)
            has_text |= bool(item.text)
        if has_audio:
            self.modality = ModalityType.AUDIO
        elif has_video:
            self.modality = ModalityType.MULTIMODAL if has_image and has_text else ModalityType.VIDEO
        elif has_image:
            self.modality = ModalityType.MULTIMODAL if has_text else ModalityType.IMAGE

    def to_pytorch_dataset(
        self,
        tokenizer_or_processor: Optional[Any] = None,
        max_length: int = 128,
        image_size: Tuple[int, int] = (224, 224),
        audio_sr: Optional[int] = None,
        max_video_frames: Optional[int] = None,
    ) -> Union[PyTorchUniversalAdapter, PyTorchIterableAdapter]:
        """Constructs an active PyTorch Dataset or IterableDataset instance."""
        if self.is_streaming:
            self._infer_streaming_modality_for_training()
            return PyTorchIterableAdapter(
                dataset_sequence=self.items,
                modality=self.modality,
                tokenizer_or_processor=tokenizer_or_processor,
                max_length=max_length,
                image_size=image_size,
                audio_sr=self.audio_sr if audio_sr is None else audio_sr,
                max_video_frames=self.max_video_frames if max_video_frames is None else max_video_frames,
                error_policy=self.error_policy,
            )
        return PyTorchUniversalAdapter(
            items=self.items,
            modality=self.modality,
            tokenizer_or_processor=tokenizer_or_processor,
            max_length=max_length,
            image_size=image_size,
            audio_sr=self.audio_sr if audio_sr is None else audio_sr,
            max_video_frames=self.max_video_frames if max_video_frames is None else max_video_frames,
            error_policy=self.error_policy,
            pre_skipped=[
                e for e in self.metadata.get("parsing_errors", [])
                if isinstance(e, dict) and "media_path" in e
            ],
        )

    def to_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        tokenizer_or_processor: Optional[Any] = None,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
        max_length: int = 128,
        image_size: Tuple[int, int] = (224, 224),
        audio_sr: Optional[int] = None,
        max_video_frames: Optional[int] = None,
        **kwargs,
    ) -> DataLoader:
        """Builds a training-ready PyTorch DataLoader directly from this dataset."""
        effective_shuffle = False if self.is_streaming else shuffle
        adapter = self.to_pytorch_dataset(
            tokenizer_or_processor=tokenizer_or_processor,
            max_length=max_length,
            image_size=image_size,
            audio_sr=self.audio_sr if audio_sr is None else audio_sr,
            max_video_frames=self.max_video_frames if max_video_frames is None else max_video_frames,
        )

        effective_collate = collate_fn or UniversalBatchCollator(
            modality=self.modality,
            tokenizer_or_processor=tokenizer_or_processor,
        )

        return DataLoader(
            adapter,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            num_workers=num_workers,
            collate_fn=effective_collate,
            **kwargs,
        )

    def to_texts(self, column: Optional[str] = None) -> List[str]:
        """Extracts text strings ready for tokenizers."""
        texts = []
        for it in self.items:
            if column and it.raw_data and column in it.raw_data:
                texts.append(str(it.raw_data[column]))
            elif it.text is not None:
                texts.append(it.text)
            elif it.raw_data:
                for v in it.raw_data.values():
                    if isinstance(v, str):
                        texts.append(v)
                        break
        return texts

    def to_numpy(self, columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Converts dataset columns to a dictionary of NumPy ndarrays."""
        if not self.items:
            return {}

        target_cols = columns or self.columns
        result: Dict[str, List[Any]] = {col: [] for col in target_cols}

        for it in self.items:
            for col in target_cols:
                val = None
                if col == "text":
                    val = it.text
                elif col == "label":
                    val = it.label
                elif col == "media_path":
                    val = it.media_path
                elif it.raw_data and col in it.raw_data:
                    val = it.raw_data[col]
                result[col].append(val)

        converted: Dict[str, np.ndarray] = {}
        for col, vals in result.items():
            try:
                converted[col] = np.asarray(vals)
            except (TypeError, ValueError):
                # Keep heterogeneous/ragged cells as a 1-D object array rather than
                # raising an opaque NumPy shape error or flattening the nested values.
                arr = np.empty(len(vals), dtype=object)
                arr[:] = vals
                converted[col] = arr
        return converted

    def _canonical_record(self, it: DatasetItem) -> Dict[str, Any]:
        """Builds a record where canonical fields (text/label/media_path) take precedence over raw_data."""
        row = dict(it.raw_data) if it.raw_data else {}
        if it.text is not None:
            row["text"] = it.text
        if it.label is not None:
            row["label"] = it.label
        if it.media_path is not None:
            row["media_path"] = it.media_path
        return row

    def to_pandas(self):
        """Converts dataset directly into a pandas DataFrame."""
        try:
            import pandas as pd
            records = [self._canonical_record(it) for it in self.items]
            return pd.DataFrame(records)
        except ImportError:
            raise ImportError("pandas is required for to_pandas(). Run: pip install pandas")

    def to_hf_dataset(self):
        """Converts dataset into a Hugging Face Dataset."""
        try:
            import datasets
            df = self.to_pandas()
            return datasets.Dataset.from_pandas(df)
        except ImportError:
            raise ImportError("Hugging Face datasets is required. Run: pip install datasets")

    def export(self, output_path: str, format: str = "jsonl") -> str:
        """Safely serializes and saves processed dataset to disk in JSONL, JSON, CSV, or Parquet."""
        out_p = Path(output_path)
        fmt = str(format).strip().lower()
        supported_formats = {"jsonl", "json", "csv", "parquet"}
        if fmt not in supported_formats:
            raise ValueError(f"Unsupported export format {format!r}. Expected one of: {sorted(supported_formats)}")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        def _json_serializable(val: Any) -> Any:
            if isinstance(val, (np.ndarray, torch.Tensor)):
                return val.tolist()
            if hasattr(val, "isoformat"):
                return val.isoformat()
            if hasattr(val, "save"):  # PIL Image
                return f"<PIL.Image size={val.size} mode={val.mode}>"
            return str(val)

        if fmt == "jsonl":
            with open(out_p, "w", encoding="utf-8") as f:
                for it in self.items:
                    payload = self._canonical_record(it)
                    f.write(json.dumps(payload, default=_json_serializable, ensure_ascii=False) + "\n")

        elif fmt == "json":
            with open(out_p, "w", encoding="utf-8") as f:
                data = [self._canonical_record(it) for it in self.items]
                json.dump(data, f, default=_json_serializable, ensure_ascii=False, indent=2)

        elif fmt in ("csv", "parquet"):
            df = self.to_pandas()
            if fmt == "csv":
                df.to_csv(out_p, index=False)
            else:
                try:
                    df.to_parquet(out_p, index=False)
                except (ImportError, ModuleNotFoundError) as e:
                    raise ImportError("Parquet export requires pyarrow or fastparquet. Install one with `pip install pyarrow`.") from e

        return str(out_p)
