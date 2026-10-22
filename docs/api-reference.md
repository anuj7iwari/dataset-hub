# 9. Public API reference

This reference lists names exported from `data_hub.__init__`. Internal helper names in `hub.py` and `parsers.py` are implementation details unless explicitly exported/imported from their submodule.

## 9.1 Package exports

```python
import data_hub as dh
print(dh.__version__)  # 0.1.2
```

Exported names:

- Loading: `load`, `DatasetHub`
- Dataset containers/adapters: `UniversalDataset`, `DatasetSplits`, `PyTorchUniversalAdapter`
- Collation: `UniversalBatchCollator`
- Enums/models: `ModalityType`, `SourceType`, `ReturnType`, `ErrorPolicy`, `DatasetItem`, `DatasetConfig`, `DatasetSummary`
- Media: `load_image`, `image_to_tensor`, `image_to_numpy`, `load_audio`, `audio_to_tensor`, `load_video`
- Extension sets: `IMAGE_EXTENSIONS`, `AUDIO_EXTENSIONS`, `VIDEO_EXTENSIONS`, `TEXT_EXTENSIONS`

## 9.2 `load()` and `DatasetHub`

`load(...)` is a convenience wrapper around a module-level default `DatasetHub`. If you need to choose a cache folder, instantiate the hub:

```python
from data_hub import DatasetHub

hub = DatasetHub(cache_dir="./cache/datasets")
ds = hub.load("hf://owner/dataset", split="train")
```

`DatasetHub(cache_dir=None)` creates the chosen cache directory (default: `./data_cache` under the current working directory). Its `load()` method accepts the same options as the top-level function.

See [Configuration reference](configuration.md) for the complete argument table and precedence rules.

## 9.3 Enum models

### `ModalityType`

Values: `TEXT`, `IMAGE`, `AUDIO`, `VIDEO`, `MULTIMODAL`, `JSON`, `TABULAR`. `ModalityType.from_str(value, default=None)` canonicalizes supported values and aliases; an unknown non-empty value raises `ValueError`.

### `SourceType`

Values: `HUGGINGFACE`, `KAGGLE`, `LOCAL_FILE`, `LOCAL_DIR`, `MEMORY`. `SourceType.from_str(value, default=None)` canonicalizes source aliases; an unknown non-empty value raises `ValueError`.

### `ReturnType`

Values: `DATASET`, `PYTORCH`, `NUMPY`, `RAW`, `DATALOADER`. `ReturnType.from_str(value, default=None)` canonicalizes return aliases; an unknown non-empty value raises `ValueError`.

### `ErrorPolicy`

Values: `STRICT`, `REPORT_AND_SKIP`, `PERMISSIVE`. `ErrorPolicy.from_str(value)` accepts `strict`/`fail`, `report_and_skip`/`skip`, and `permissive`/`warn`; invalid values raise `ValueError`.

### `DatasetItem`

Fields:

```python
id: int | str | None = None
text: str | None = None
media_path: str | None = None
label: str | int | float | list[str] | list[int] | None = None
metadata: dict = {}
raw_data: dict | None = None
features: dict | None = None
```

The exact Pydantic field annotations are as defined by the source; `metadata` uses a factory so records do not share the same mutable default dictionary. `raw_data` holds source fields. `features` is an optional application-level slot.

### `DatasetConfig`

A Pydantic model for source and processing configuration. See [Configuration reference](configuration.md) for full defaults and validators. Construct with `path_or_identifier` and optionally explicit `source_type`, `modality`, and desired output settings.

### `DatasetSummary`

A Pydantic summary model whose fields include `name`, `modality`, `source_type`, `return_type`, `error_policy`, `total_samples`, `is_streaming`, `columns`, `splits`, `sample_preview`, `estimated_tokens`, `token_estimation_method`, `class_distribution`, `memory_mb`, `skipped_samples`, `parsing_errors`, `warnings`, `warning`, `included_files`, `excluded_files`, and `metadata`.

## 9.4 `UniversalDataset`

The common normalized dataset container. Its constructor is public but ordinary users should normally create it through `load()`.

### Inspection API

| Method/property | Contract |
|---|---|
| `modality` | Current modality enum; setter normalizes supplied values. |
| `is_streaming` | Whether `items` are represented by a lazy sequence rather than an eager collection. |
| `num_samples` | Sample count where available; can be `None` for unknown streaming cardinality. |
| `len(ds)` | Number of rows; for unknown/infinite streams, use `num_samples`/summary and consult the source's contract. |
| `ds[index]` | Record access for eager datasets; integer and slice access are supported by the implementation. |
| `iter(ds)` | Iterates `DatasetItem` instances. |
| `preview(n=5, detailed=False)` | Preview list of dictionaries. |
| `get_summary()` / `summary()` | `DatasetSummary` diagnostic model. |

### Transformation API

| Method | Returns | Notes |
|---|---|---|
| `take(n)` | `UniversalDataset` | First `n` rows; materializes a stream prefix. `n ≥ 0`. |
| `filter(condition)` | `UniversalDataset` | Predicate receives isolated item; lazy for streams. |
| `map(fn)` | `UniversalDataset` | Callback must return `DatasetItem` or dict updates. |
| `sort(by=None, descending=False, key=None)` | `UniversalDataset` | Eager-only. Sort by text length, label, ID, raw-data column, or custom key. |
| `clean(deduplicate=True, strip_whitespace=True, drop_na=True, drop_corrupted_media=False)` | `UniversalDataset` | Modality-aware filtering/cleanup. Media decode only when requested and possible. |
| `encode_labels(mapping=None)` | `UniversalDataset` | Eager-only; stores mapping in metadata. |
| `format_prompt(template=None, column_mapping=None, safe=True)` | `UniversalDataset` | Formats canonical text from row values. |
| `sample(n=None, frac=None, seed=42)` | `UniversalDataset` | Eager-only random sample; choose either `n` or `frac`. |
| `split(train_ratio=.8, val_ratio=.1, test_ratio=.1, shuffle=True, seed=42)` | `DatasetSplits` | Eager-only; ratios must sum to 1.0. |

### Conversion/output API

| Method | Returns | Notes |
|---|---|---|
| `to_pytorch_dataset(tokenizer_or_processor=None, max_length=128, image_size=(224,224), audio_sr=None, max_video_frames=None)` | PyTorch map/iterable dataset | Decodes/processes at sample access or iteration. |
| `to_dataloader(batch_size=8, shuffle=True, tokenizer_or_processor=None, collate_fn=None, num_workers=0, max_length=128, image_size=(224,224), audio_sr=None, max_video_frames=None, **kwargs)` | PyTorch DataLoader | Defaults to `UniversalBatchCollator`; forwards extra DataLoader options. |
| `to_texts(column=None)` | `list[str]` | Column, canonical text, or first string-valued raw field. |
| `to_numpy(columns=None)` | `dict[str, np.ndarray]` | Defaults to dataset columns; ragged values may become object arrays. |
| `to_pandas()` | Pandas DataFrame | Requires pandas. |
| `to_hf_dataset()` | Hugging Face Dataset | Requires `datasets` and its compatible dependencies. |
| `export(output_path, format="jsonl")` | output path string | Formats: JSONL, JSON, CSV, Parquet. |

## 9.5 `DatasetSplits`

Contains `train`, optional `val`, and optional `test` datasets. Supports string-key access, for example `splits["train"]`. A split with no rows is represented by `None` in the current implementation. Check this before using it:

```python
splits = ds.split(0.8, 0.1, 0.1)
if splits.val is not None:
    print(len(splits.val))
```

## 9.6 `UniversalBatchCollator`

Constructor:

```python
UniversalBatchCollator(
    modality=ModalityType.TEXT,
    pad_token_id=0,
    tokenizer_or_processor=None,
)
```

Callable contract: accepts `list[dict[str, Any]]` and returns one batch dictionary. It validates tensor requirements and shapes. The tokenizer's `pad_token_id`, when available and non-`None`, overrides the constructor's fallback padding ID.

## 9.7 PyTorch adapters

`PyTorchUniversalAdapter` is exported from the package root. `PyTorchIterableAdapter` is an implementation class in `data_hub.dataset` but is **not** included in the package root's `__all__` in version 0.1.2. Most users should call `to_pytorch_dataset()` or `to_dataloader()` instead.

- `PyTorchUniversalAdapter` wraps a finite list of `DatasetItem` records and implements map-style `__len__`/`__getitem__`.
- `PyTorchIterableAdapter` wraps a lazy source and implements `__iter__`; it shards rows across standard PyTorch DataLoader workers by a modulo index pattern.

The per-item dictionary schema depends on the modality and whether a tokenizer/processor was supplied. See [PyTorch training](pytorch.md).

## 9.8 Media helper API

| Function | Purpose |
|---|---|
| `load_image(source)` | Decode and return RGB PIL image. |
| `image_to_tensor(image, processor_or_transform=None, target_size=(224, 224))` | Return image tensor or processor result, normally CHW. |
| `image_to_numpy(image, target_size=None)` | Return RGB HWC uint8 array. |
| `load_audio(source, target_sr=16000)` | Return `(waveform_tensor, sample_rate)`. |
| `audio_to_tensor(source, processor=None, target_sr=16000)` | Return waveform or processor features. |
| `load_video(source, max_frames=16, target_size=(224, 224))` | Decode evenly selected frames to TCHW float tensor. |

All public helper names are imported at `data_hub` package root. Use the shape, dtype, decoder, and optional-dependency notes in [Media utilities](media-utilities.md).

## 9.9 Extension constants

- `IMAGE_EXTENSIONS`: recognized local image suffixes.
- `AUDIO_EXTENSIONS`: recognized local audio suffixes.
- `VIDEO_EXTENSIONS`: recognized local video suffixes.
- `TEXT_EXTENSIONS`: recognized local structured/text suffixes.

These sets are useful for path validation and profiling. An accepted suffix is not a guarantee of a successful decode.
