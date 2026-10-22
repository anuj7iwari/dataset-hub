

from typing import List, Dict, Any, Optional, Union
import torch

from data_hub.schemas import ModalityType

# ========================================
# Universal Batch Collator
# ========================================
class UniversalBatchCollator:


    def __init__(
        self,
        modality: ModalityType = ModalityType.TEXT,
        pad_token_id: int = 0,
        tokenizer_or_processor: Optional[Any] = None,
    ):
        self.modality = modality
        self.pad_token_id = pad_token_id
        self.tokenizer_or_processor = tokenizer_or_processor

        if tokenizer_or_processor is not None and hasattr(tokenizer_or_processor, "pad_token_id"):
            if tokenizer_or_processor.pad_token_id is not None:
                self.pad_token_id = tokenizer_or_processor.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates a list of dataset items into a training-ready PyTorch batch dictionary."""
        if not batch:
            return {}

        # Handle permissive invalid samples
        invalid_items = [it for it in batch if it.get("_invalid")]
        valid_items = [it for it in batch if not it.get("_invalid")]
        if invalid_items:
            if not valid_items:
                raise RuntimeError("All samples in batch are invalid.")
            batch = valid_items

        collated: Dict[str, Any] = {}
        if invalid_items:
            collated["invalid_samples"] = invalid_items

        # ========================================
        # 1. NLP Dynamic Token Padding
        # ========================================
        if any("input_ids" in item and item["input_ids"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "input_ids" not in item or not isinstance(item["input_ids"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'input_ids' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('input_ids')).__name__}."
                    )
            input_ids_list = [item["input_ids"] for item in batch]
            if any(t.ndim not in (1, 2) for t in input_ids_list):
                bad = next((i, t) for i, t in enumerate(input_ids_list) if t.ndim not in (1, 2))
                raise ValueError(f"input_ids at sample index {bad[0]} must be a 1D or 2D token tensor, got shape {tuple(bad[1].shape)}.")
            collated["input_ids"] = self._pad_tensor_sequence(input_ids_list, pad_val=self.pad_token_id)

        if any("attention_mask" in item and item["attention_mask"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "attention_mask" not in item or not isinstance(item["attention_mask"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'attention_mask' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('attention_mask')).__name__}."
                    )
            mask_list = [item["attention_mask"] for item in batch]
            if "input_ids" in batch[0]:
                for i, (ids, mask) in enumerate(zip(input_ids_list, mask_list)):
                    if ids.shape != mask.shape:
                        raise ValueError(
                            f"attention_mask shape {tuple(mask.shape)} does not match input_ids shape {tuple(ids.shape)} "
                            f"at sample index {i} (id={batch[i].get('id', i)})."
                        )
            collated["attention_mask"] = self._pad_tensor_sequence(mask_list, pad_val=0)

        if any("labels" in item and item["labels"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "labels" not in item or not isinstance(item["labels"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'labels' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('labels')).__name__}."
                    )
            labels_list = [item["labels"] for item in batch]
            ranks = {t.ndim for t in labels_list}
            if len(ranks) > 1:
                raise ValueError(f"Mixed scalar and sequence labels are not supported in one batch: ranks={sorted(ranks)}.")
            if all(t.ndim == 0 for t in labels_list):
                # Scalar classification labels form [B]; they are not token-aligned.
                collated["labels"] = torch.stack(labels_list, dim=0)
            else:
                if "input_ids" in batch[0] and all(t.ndim == 1 for t in labels_list) and all(t.ndim == 1 for t in input_ids_list):
                    aligned = []
                    for i, (ids, labels) in enumerate(zip(input_ids_list, labels_list)):
                        if labels.shape[0] > ids.shape[0]:
                            raise ValueError(
                                f"Token label length {labels.shape[0]} exceeds input_ids length {ids.shape[0]} "
                                f"at sample index {i} (id={batch[i].get('id', i)})."
                            )
                        if labels.shape[0] < ids.shape[0]:
                            labels = torch.cat([
                                labels,
                                torch.full((ids.shape[0] - labels.shape[0],), -100, dtype=labels.dtype, device=labels.device),
                            ])
                        aligned.append(labels)
                    labels_list = aligned
                collated["labels"] = self._pad_tensor_sequence(labels_list, pad_val=-100)

        # ========================================
        # 2. Vision Pixel Values Stacking
        # ========================================
        if any("pixel_values" in item and item["pixel_values"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "pixel_values" not in item or not isinstance(item["pixel_values"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'pixel_values' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('pixel_values')).__name__}."
                    )
            pv_list = [item["pixel_values"] for item in batch]
            for i, pv in enumerate(pv_list):
                if pv.ndim != 3 or pv.shape[0] != 3:
                    raise ValueError(
                        f"pixel_values at sample index {i} must have CHW rank 3 with exactly 3 RGB channels; "
                        f"got shape {tuple(pv.shape)}."
                    )
            expected_shape = pv_list[0].shape
            for i, pv in enumerate(pv_list):
                if pv.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent image tensor shape in batch at sample index {i} (id={batch[i].get('id')}): "
                        f"expected {expected_shape}, but got {pv.shape}. Ensure images are resized uniformly."
                    )
            collated["pixel_values"] = torch.stack(pv_list, dim=0)

        # ========================================
        # 3. Audio Features / Waveform Stacking
        # ========================================
        if any("input_features" in item and item["input_features"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "input_features" not in item or not isinstance(item["input_features"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'input_features' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('input_features')).__name__}."
                    )
            feat_list = [item["input_features"] for item in batch]
            expected_shape = feat_list[0].shape
            for i, feat in enumerate(feat_list):
                if feat.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent audio feature shape at sample index {i}: "
                        f"expected {expected_shape}, but got {feat.shape}."
                    )
            collated["input_features"] = torch.stack(feat_list, dim=0)

        elif any("waveform" in item and item["waveform"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "waveform" not in item or not isinstance(item["waveform"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'waveform' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('waveform')).__name__}."
                    )
            waveform_list = [item["waveform"] for item in batch]
            collated["waveform"] = self._pad_tensor_sequence(waveform_list, pad_val=0.0)

        # ========================================
        # 4. Video Frame Stacking
        # ========================================
        if any("video_frames" in item and item["video_frames"] is not None for item in batch):
            for i, item in enumerate(batch):
                if "video_frames" not in item or not isinstance(item["video_frames"], torch.Tensor):
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing or invalid 'video_frames' tensor in batch at sample index {i} (id={sample_id}). "
                        f"Expected torch.Tensor, but got {type(item.get('video_frames')).__name__}."
                    )
            frames_list = [item["video_frames"] for item in batch]
            for i, vf in enumerate(frames_list):
                if vf.ndim != 4 or vf.shape[1] != 3:
                    raise ValueError(
                        f"video_frames at sample index {i} must have TCHW rank 4 with exactly 3 RGB channels "
                        f"at dimension 1; got shape {tuple(vf.shape)}."
                    )
            expected_shape = frames_list[0].shape
            for i, vf in enumerate(frames_list):
                if vf.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent video frame tensor shape at sample index {i}: "
                        f"expected {expected_shape}, but got {vf.shape}. Ensure max_frames and target_size match."
                    )
            collated["video_frames"] = torch.stack(frames_list, dim=0)

        # ========================================
        # 5. Label Collation
        # ========================================
        if any("label" in item and item["label"] is not None for item in batch):
            raw_labels = []
            for i, item in enumerate(batch):
                l = item.get("label")
                if l is None:
                    sample_id = item.get("id", i)
                    raise ValueError(
                        f"Missing label in batch at sample index {i} (id={sample_id}). "
                        f"Batch contains heterogeneous labels where some samples lack a label."
                    )
                raw_labels.append(l)

            if all(isinstance(l, int) for l in raw_labels):
                collated["label"] = torch.tensor(raw_labels, dtype=torch.long)
            elif all(isinstance(l, (int, float)) for l in raw_labels):
                collated["label"] = torch.tensor(raw_labels, dtype=torch.float32)
            else:
                collated["label"] = raw_labels

        # ========================================
        # 6. Non-Tensor Metadata
        # ========================================
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())

        for key in all_keys:
            if key not in collated and key not in ("_invalid", "error"):
                values = [item.get(key) for item in batch]
                if all(isinstance(v, torch.Tensor) for v in values):
                    if all(v.shape == values[0].shape for v in values):
                        try:
                            collated[key] = torch.stack(values, dim=0)
                            continue
                        except Exception:
                            pass
                collated[key] = values

        if invalid_items:
            collated["invalid_samples"] = [
                {"id": it.get("id"), "error": it.get("error")}
                for it in invalid_items
            ]

        return collated

    def _pad_tensor_sequence(
        self,
        tensors: List[torch.Tensor],
        pad_val: Union[int, float] = 0,
    ) -> torch.Tensor:
        """Pads 1D or 2D tensors to the longest length in the current batch while preserving channels."""
        if not tensors:
            return torch.empty(0)

        # Negative ignore-index values are not representable in unsigned dtypes;
        # promote those sequences before constructing padding tensors.
        if isinstance(pad_val, (int, float)) and pad_val < 0:
            unsigned_or_bool = {torch.uint8, torch.bool}
            tensors = [t.to(torch.long) if t.dtype in unsigned_or_bool else t for t in tensors]

        first_shape = tensors[0].shape
        first_prefix = first_shape[:-1]
        first_ndim = tensors[0].ndim
        for idx, t in enumerate(tensors):
            if t.ndim != first_ndim:
                raise ValueError(
                    f"Incompatible tensor ranks in sequence batch: sample 0 has rank {first_ndim} (shape {tuple(first_shape)}), "
                    f"but sample {idx} has rank {t.ndim} (shape {tuple(t.shape)})."
                )
            if t.shape[:-1] != first_prefix:
                raise ValueError(
                    f"Incompatible leading dimensions in sequence batch: sample 0 has prefix shape {tuple(first_prefix)} "
                    f"(full shape {tuple(first_shape)}), but sample {idx} has prefix shape {tuple(t.shape[:-1])} "
                    f"(full shape {tuple(t.shape)})."
                )

        max_len = max(t.size(-1) for t in tensors)

        padded_list = []
        for t in tensors:
            diff = max_len - t.size(-1)
            if diff > 0:
                if t.ndim == 1:
                    pad_tensor = torch.full((diff,), pad_val, dtype=t.dtype, device=t.device)
                else:
                    pad_shape = list(t.shape[:-1]) + [diff]
                    pad_tensor = torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)
                padded_t = torch.cat([t, pad_tensor], dim=-1)
            else:
                padded_t = t
            padded_list.append(padded_t)

        return torch.stack(padded_list, dim=0)
