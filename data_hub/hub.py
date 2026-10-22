# ========================================
# Dataset Hub Core Engine
# ========================================
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from data_hub.schemas import (
    ModalityType,
    SourceType,
    ReturnType,
    ErrorPolicy,
    DatasetItem,
    DatasetConfig,
    DatasetSummary,
)
from data_hub.dataset import UniversalDataset
from data_hub.parsers import (
    parse_local_file,
    parse_local_directory,
    parse_huggingface,
    parse_kaggle,
    parse_memory,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    TEXT_EXTENSIONS,
)


def _resolve_source_and_path(raw_target: str, source: Optional[Union[str, SourceType]] = None) -> Tuple[SourceType, str]:
    raw_target = str(raw_target).strip()
    if not raw_target:
        raise ValueError("Dataset target must not be empty or whitespace-only.")
    # 1. Explicit URI Schemes
    if raw_target.startswith("hf://"):
        return SourceType.HUGGINGFACE, raw_target[5:]
    if raw_target.startswith("kaggle://"):
        return SourceType.KAGGLE, raw_target[9:]
    # 2. Explicit Source Parameter
    if source:
        detected_source = SourceType.from_str(source)
        clean_path = raw_target
        if detected_source == SourceType.LOCAL_FILE and not os.path.isfile(clean_path):
            raise FileNotFoundError(f"Local file not found: '{clean_path}'")
        if detected_source == SourceType.LOCAL_DIR and not os.path.isdir(clean_path):
            raise FileNotFoundError(f"Local directory not found: '{clean_path}'")
        return detected_source, clean_path
    # 3. Existing Local Path
    if os.path.isfile(raw_target):
        return SourceType.LOCAL_FILE, raw_target
    if os.path.isdir(raw_target):
        return SourceType.LOCAL_DIR, raw_target
    # 4. Known File Extension Check (local file typo / missing file)
    if Path(raw_target).suffix.lower() in (IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | TEXT_EXTENSIONS):
        raise FileNotFoundError(f"Local dataset file not found: '{raw_target}'")
    # 5. Local directory prefix check (./, ../, C:\, /)
    if raw_target.startswith(("./", "../", "/", "\\")) or (len(raw_target) > 2 and raw_target[1:3] in (":\\", ":/")):
        raise FileNotFoundError(f"Local dataset directory not found: '{raw_target}'")
    # 6. Default to Hugging Face Hub (both repo 'owner/dataset' and bare 'dataset' IDs)
    return SourceType.HUGGINGFACE, raw_target


# DataLoader-only keys
DATALOADER_ONLY_KEYS = {
    "num_workers", "drop_last", "pin_memory", "persistent_workers",
    "prefetch_factor", "timeout", "worker_init_fn", "generator",
    "sampler", "batch_sampler", "multiprocessing_context", "collate_fn",
}


def _strip_uri_scheme(target: str, source_type: SourceType) -> str:
    t = str(target).strip()
    if source_type == SourceType.HUGGINGFACE and t.startswith("hf://"):
        return t[5:]
    if source_type == SourceType.KAGGLE and t.startswith("kaggle://"):
        return t[9:]
    return t


# ========================================
# DatasetHub Class
# ========================================
class DatasetHub:

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load(
        self,
        target: Optional[Union[str, DatasetConfig, List[Dict[str, Any]], Dict[str, Any], Any]] = None,
        src: Optional[Union[str, SourceType]] = None,
        type: Optional[Union[str, ModalityType]] = None,
        returntype: Optional[Union[str, ReturnType]] = None,
        # Keyword aliases:
        source: Optional[Union[str, SourceType]] = None,
        modality: Optional[Union[str, ModalityType]] = None,
        return_type: Optional[Union[str, ReturnType]] = None,
        error_policy: Optional[Union[str, ErrorPolicy]] = None,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        media_column: Optional[str] = None,
        max_samples: Optional[int] = None,
        prompt_template: Optional[str] = None,
        tokenizer_or_processor: Optional[Any] = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        max_length: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        audio_sr: Optional[int] = None,
        max_video_frames: Optional[int] = None,
        streaming: Optional[bool] = None,
        treat_path_as_text: Optional[bool] = None,
        label_dtype: Optional[str] = None,
        **kwargs,
    ) -> Union[UniversalDataset, Any]:
        """Universal dataset loading entry point."""
        if src is not None and source is not None:
            if SourceType.from_str(src) != SourceType.from_str(source):
                raise ValueError(f"Conflicting alias arguments: src='{src}' and source='{source}'")
        if type is not None and modality is not None:
            if ModalityType.from_str(type) != ModalityType.from_str(modality):
                raise ValueError(f"Conflicting alias arguments: type='{type}' and modality='{modality}'")
        if returntype is not None and return_type is not None:
            if ReturnType.from_str(returntype) != ReturnType.from_str(return_type):
                raise ValueError(f"Conflicting alias arguments: returntype='{returntype}' and return_type='{return_type}'")

        effective_source = src or source
        effective_modality = type or modality
        effective_return_type = returntype or return_type
        effective_error_policy = error_policy
        effective_streaming = streaming if streaming is not None else kwargs.pop("streaming", None)

        if target is None and "path" in kwargs:
            target = kwargs.pop("path")

        if target is None:
            raise ValueError("You must supply a 'target' dataset path, identifier, or record list to datahub.load().")
        if isinstance(target, (str, Path)) and not str(target).strip():
            raise ValueError("Dataset target must not be empty or whitespace-only.")

        # ========================================
        # 0. Configuration Object Dispatch
        # ========================================
        if isinstance(target, DatasetConfig):
            cfg = target.model_copy(deep=True)
            if "error_policy" in target.model_fields_set or effective_error_policy is not None:
                cfg.kwargs["_explicit_error_policy"] = True
            if "modality" in target.model_fields_set or effective_modality is not None:
                cfg.kwargs["_explicit_modality"] = True
            if effective_source is not None:
                cfg.source_type = SourceType.from_str(effective_source)
            if effective_modality is not None:
                cfg.modality = ModalityType.from_str(effective_modality)
                cfg.kwargs["_explicit_modality"] = True
            if effective_return_type is not None:
                cfg.return_type = ReturnType.from_str(effective_return_type)
            if effective_error_policy is not None:
                cfg.error_policy = ErrorPolicy.from_str(effective_error_policy)
                cfg.kwargs["_explicit_error_policy"] = True
            if split is not None:
                cfg.split = split
            if subset is not None:
                cfg.subset = subset
            if text_column is not None:
                cfg.text_column = text_column
            if label_column is not None:
                cfg.label_column = label_column
            if media_column is not None:
                cfg.media_column = media_column
            if max_samples is not None:
                cfg.max_samples = max_samples
            if prompt_template is not None:
                cfg.prompt_template = prompt_template
            if effective_streaming is not None:
                cfg.streaming = effective_streaming
            if batch_size is not None:
                cfg.batch_size = batch_size
            if shuffle is not None:
                cfg.shuffle = shuffle
            if max_length is not None:
                cfg.max_length = max_length
            if image_size is not None:
                cfg.image_size = image_size
            if audio_sr is not None:
                cfg.target_sr = audio_sr
            if max_video_frames is not None:
                cfg.max_video_frames = max_video_frames
            if treat_path_as_text is not None:
                cfg.treat_path_as_text = treat_path_as_text
            if label_dtype is not None:
                cfg.label_dtype = label_dtype
            cfg.kwargs.update(kwargs)

            # Infer source_type if None using shared unified resolver; otherwise
            # still normalize any explicit URI scheme so a typed config carrying an
            # ``hf://`` prefix reaches the connector with the bare repository id.
            if cfg.source_type is None:
                cfg.source_type, cfg.path_or_identifier = _resolve_source_and_path(
                    cfg.path_or_identifier, cfg.source_type
                )
            else:
                cfg.path_or_identifier = _strip_uri_scheme(cfg.path_or_identifier, cfg.source_type)

        # ========================================
        # 1. In-Memory Data Structures
        # ========================================
        elif (
            isinstance(target, (list, tuple, dict))
            or hasattr(target, "to_dict")
            or (hasattr(target, "__iter__") and not isinstance(target, (str, Path)))
            or (isinstance(target, str) and effective_source is not None and SourceType.from_str(effective_source) == SourceType.MEMORY)
        ):
            is_explicit_modality = effective_modality is not None
            detected_mod = None
            if is_explicit_modality:
                detected_mod = ModalityType.from_str(effective_modality)
            else:
                # Inspect representative rows (not only row 0) and case-insensitively
                sample_rows = []
                if isinstance(target, dict):
                    sample_rows = [target]
                elif isinstance(target, (list, tuple)):
                    sample_rows = [r for r in target[:10] if isinstance(r, dict)]
                elif hasattr(target, "to_dict") and hasattr(target, "columns"):
                    sample_rows = target.head(10).to_dict(orient="records")
                has_audio = False
                has_image = False
                has_video = False
                has_caption = False

                for row in sample_rows:
                    lower_keys = {str(k).lower(): v for k, v in row.items()}
                    if any(k in lower_keys for k in ("audio", "audio_path")):
                        has_audio = True
                    if any(k in lower_keys for k in ("image", "image_path")):
                        has_image = True
                    if any(k in lower_keys for k in ("video", "video_path")):
                        has_video = True
                    if any(k in lower_keys for k in ("caption", "description", "text", "transcription")):
                        has_caption = True

                if has_audio:
                    detected_mod = ModalityType.AUDIO
                elif has_image:
                    if has_caption:
                        detected_mod = ModalityType.MULTIMODAL
                    else:
                        detected_mod = ModalityType.IMAGE
                elif has_video:
                    detected_mod = ModalityType.VIDEO
                else:
                    detected_mod = ModalityType.TEXT

            extra_kw = dict(kwargs)
            if is_explicit_modality:
                extra_kw["_explicit_modality"] = True
            if effective_error_policy is not None:
                extra_kw["_explicit_error_policy"] = True

            cfg = DatasetConfig(
                source_type=SourceType.MEMORY,
                path_or_identifier="in_memory",
                modality=detected_mod,
                return_type=ReturnType.from_str(effective_return_type or ReturnType.DATASET),
                error_policy=ErrorPolicy.from_str(effective_error_policy or ErrorPolicy.STRICT),
                text_column=text_column,
                label_column=label_column,
                media_column=media_column,
                max_samples=max_samples,
                prompt_template=prompt_template,
                streaming=bool(effective_streaming),
                batch_size=batch_size if batch_size is not None else 8,
                shuffle=shuffle if shuffle is not None else True,
                max_length=max_length if max_length is not None else 128,
                image_size=image_size if image_size is not None else (224, 224),
                target_sr=audio_sr if audio_sr is not None else 16000,
                max_video_frames=max_video_frames if max_video_frames is not None else 16,
                treat_path_as_text=bool(treat_path_as_text) if treat_path_as_text is not None else False,
                label_dtype=label_dtype or "auto",
                kwargs=extra_kw,
            )
            items, columns, meta = parse_memory(target, cfg)
            ds = UniversalDataset(
                items=items,
                modality=cfg.modality,
                source_type=SourceType.MEMORY,
                return_type=cfg.return_type,
                error_policy=cfg.error_policy,
                name="in_memory_dataset",
                columns=columns,
                metadata=meta,
                audio_sr=cfg.target_sr,
                max_video_frames=cfg.max_video_frames,
                config=cfg,
            )
            return self._finalize_dataset(ds, cfg, tokenizer_or_processor=tokenizer_or_processor)

        # ========================================
        # 2. Parse Target String to Configuration
        # ========================================
        else:
            cfg = self._parse_target_to_config(
                target_str=str(target),
                source=effective_source,
                modality=effective_modality,
                return_type=effective_return_type or ReturnType.DATASET,
                error_policy=effective_error_policy or ErrorPolicy.STRICT,
                split=split or "train",
                subset=subset,
                text_column=text_column,
                label_column=label_column,
                media_column=media_column,
                max_samples=max_samples,
                prompt_template=prompt_template,
                batch_size=batch_size if batch_size is not None else 8,
                shuffle=shuffle if shuffle is not None else True,
                max_length=max_length if max_length is not None else 128,
                image_size=image_size if image_size is not None else (224, 224),
                audio_sr=audio_sr if audio_sr is not None else 16000,
                max_video_frames=max_video_frames if max_video_frames is not None else 16,
                streaming=bool(effective_streaming),
                treat_path_as_text=bool(treat_path_as_text) if treat_path_as_text is not None else False,
                label_dtype=label_dtype or "auto",
                kwargs=kwargs,
            )
            if effective_error_policy is not None:
                cfg.kwargs["_explicit_error_policy"] = True

        # ========================================
        # 3. Source Ingestion
        # ========================================
        if cfg.source_type == SourceType.LOCAL_FILE:
            items, columns, meta = parse_local_file(cfg.path_or_identifier, cfg)
            name = Path(cfg.path_or_identifier).stem
        elif cfg.source_type == SourceType.LOCAL_DIR:
            items, columns, meta = parse_local_directory(cfg.path_or_identifier, cfg)
            name = Path(cfg.path_or_identifier).name
        elif cfg.source_type == SourceType.HUGGINGFACE:
            items, columns, meta = parse_huggingface(cfg.path_or_identifier, cfg)
            name = cfg.path_or_identifier.replace("/", "_")
        elif cfg.source_type == SourceType.KAGGLE:
            items, columns, meta = parse_kaggle(cfg.path_or_identifier, cfg, self.cache_dir)
            name = cfg.path_or_identifier.replace("/", "_")
        elif cfg.source_type == SourceType.MEMORY:
            items, columns, meta = parse_memory(cfg.path_or_identifier, cfg)
            name = "in_memory_dataset"
        else:
            raise ValueError(f"Unsupported source type: {cfg.source_type}")

        ds = UniversalDataset(
            items=items,
            modality=cfg.modality,
            source_type=cfg.source_type,
            return_type=cfg.return_type,
            error_policy=cfg.error_policy,
            name=name,
            columns=columns,
            metadata=meta,
            audio_sr=cfg.target_sr,
            max_video_frames=cfg.max_video_frames,
            config=cfg,
        )

        return self._finalize_dataset(ds, cfg, tokenizer_or_processor=tokenizer_or_processor)

    def _finalize_dataset(
        self,
        ds: UniversalDataset,
        cfg: DatasetConfig,
        tokenizer_or_processor: Optional[Any] = None,
    ) -> Any:
        """Applies prompt templates and returns target representation matching ReturnType contract."""
        if cfg.prompt_template:
            ds = ds.format_prompt(template=cfg.prompt_template)

        # Enforce exact ReturnType contract
        if cfg.return_type == ReturnType.DATALOADER:
            # Forward DataLoader-only options (num_workers, drop_last, pin_memory, ...)
            # that were accepted via **kwargs but would otherwise be silently dropped.
            dataloader_kwargs = {
                k: cfg.kwargs[k] for k in DATALOADER_ONLY_KEYS if k in cfg.kwargs
            }
            return ds.to_dataloader(
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                tokenizer_or_processor=tokenizer_or_processor,
                max_length=cfg.max_length,
                image_size=cfg.image_size,
                audio_sr=cfg.target_sr,
                max_video_frames=cfg.max_video_frames,
                **dataloader_kwargs,
            )
        elif cfg.return_type == ReturnType.PYTORCH:
            return ds.to_pytorch_dataset(
                tokenizer_or_processor=tokenizer_or_processor,
                max_length=cfg.max_length,
                image_size=cfg.image_size,
                audio_sr=cfg.target_sr,
                max_video_frames=cfg.max_video_frames,
            )
        elif cfg.return_type == ReturnType.NUMPY:
            return ds.to_numpy()
        elif cfg.return_type == ReturnType.RAW:
            return [it.raw_data if it.raw_data else it.model_dump() for it in ds]

        return ds

    def _parse_target_to_config(
        self,
        target_str: str,
        source: Optional[Union[str, SourceType]],
        modality: Optional[Union[str, ModalityType]],
        return_type: Optional[Union[str, ReturnType]],
        error_policy: Optional[Union[str, ErrorPolicy]],
        split: str,
        subset: Optional[str],
        text_column: Optional[str],
        label_column: Optional[str],
        media_column: Optional[str],
        max_samples: Optional[int],
        prompt_template: Optional[str],
        batch_size: int,
        shuffle: bool,
        max_length: int,
        image_size: Tuple[int, int],
        audio_sr: int,
        max_video_frames: int = 16,
        streaming: bool = False,
        treat_path_as_text: bool = False,
        label_dtype: str = "auto",
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> DatasetConfig:
        """Disambiguates source type and modality from path or URI string."""
        kwargs = kwargs or {}
        raw_target = target_str.strip()
        detected_source = None
        clean_path = raw_target

        if modality is not None:
            kwargs["_explicit_modality"] = True

        # 1. Resolve source and clean path using shared unified resolver
        detected_source, clean_path = _resolve_source_and_path(raw_target, source)

        # 2. Modality Inference
        detected_modality = None
        if modality:
            detected_modality = ModalityType.from_str(modality)
        elif detected_source == SourceType.LOCAL_FILE:
            ext = Path(clean_path).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                detected_modality = ModalityType.IMAGE
            elif ext in AUDIO_EXTENSIONS:
                detected_modality = ModalityType.AUDIO
            elif ext in VIDEO_EXTENSIONS:
                detected_modality = ModalityType.VIDEO
            else:
                detected_modality = ModalityType.TEXT
        else:
            detected_modality = ModalityType.TEXT

        return DatasetConfig(
            source_type=detected_source,
            path_or_identifier=clean_path,
            modality=detected_modality,
            return_type=ReturnType.from_str(return_type),
            error_policy=ErrorPolicy.from_str(error_policy),
            split=split,
            subset=subset,
            text_column=text_column,
            label_column=label_column,
            media_column=media_column,
            max_samples=max_samples,
            prompt_template=prompt_template,
            streaming=streaming,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
            image_size=image_size,
            target_sr=audio_sr,
            max_video_frames=max_video_frames,
            treat_path_as_text=treat_path_as_text,
            label_dtype=label_dtype,
            kwargs=kwargs,
        )


# ========================================
# Global Entry Point
# ========================================
_DEFAULT_HUB = DatasetHub()


def load(
    target: Optional[Union[str, DatasetConfig, List[Dict[str, Any]], Dict[str, Any], Any]] = None,
    src: Optional[Union[str, SourceType]] = None,
    type: Optional[Union[str, ModalityType]] = None,
    returntype: Optional[Union[str, ReturnType]] = None,
    # Aliases
    source: Optional[Union[str, SourceType]] = None,
    modality: Optional[Union[str, ModalityType]] = None,
    return_type: Optional[Union[str, ReturnType]] = None,
    error_policy: Optional[Union[str, ErrorPolicy]] = None,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    media_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    prompt_template: Optional[str] = None,
    tokenizer_or_processor: Optional[Any] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    max_length: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    audio_sr: Optional[int] = None,
    max_video_frames: Optional[int] = None,
    streaming: Optional[bool] = None,
    treat_path_as_text: Optional[bool] = None,
    label_dtype: Optional[str] = None,
    **kwargs,
) -> Union[UniversalDataset, Any]:
    """Universal dataset loading entry point."""
    return _DEFAULT_HUB.load(
        target=target,
        src=src,
        type=type,
        returntype=returntype,
        source=source,
        modality=modality,
        return_type=return_type,
        error_policy=error_policy,
        split=split,
        subset=subset,
        text_column=text_column,
        label_column=label_column,
        media_column=media_column,
        max_samples=max_samples,
        prompt_template=prompt_template,
        tokenizer_or_processor=tokenizer_or_processor,
        batch_size=batch_size,
        shuffle=shuffle,
        max_length=max_length,
        image_size=image_size,
        audio_sr=audio_sr,
        max_video_frames=max_video_frames,
        streaming=streaming,
        treat_path_as_text=treat_path_as_text,
        label_dtype=label_dtype,
        **kwargs,
    )
