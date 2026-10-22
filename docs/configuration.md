# 4. Configuration reference

## 4.1 `DatasetConfig` at a glance

Use `DatasetConfig` when a load recipe needs to be named, reused, reviewed, or overridden. It is a Pydantic v2 model; invalid values are rejected during construction/validation.

```python
from data_hub import DatasetConfig, load

config = DatasetConfig(
    path_or_identifier="./data/train.csv",
    source_type="local_file",
    modality="text",
    return_type="dataset",
    error_policy="strict",
    split="train",
    text_column="sentence",
    label_column="sentiment",
    max_samples=50_000,
    batch_size=32,
    max_length=256,
)

ds = load(config)
```

### Field reference

| Field | Type | Default | Validation / effect |
|---|---|---|---|
| `source_type` | `SourceType \| None` | `None` | When omitted, source is inferred from the target string/path by the same resolver as `load()`; an unresolved name falls back to Hugging Face. |
| `path_or_identifier` | non-empty `str` or path-like input | required | Stripped; empty/whitespace-only values are rejected. `Path`-like objects are accepted via `__fspath__`. |
| `modality` | `ModalityType` | `text` | Normalized modality hint; source loaders may infer/promote modality only if it was not explicitly forced. |
| `return_type` | `ReturnType` | `dataset` | Final return object: universal dataset, PyTorch adapter, NumPy dictionary, raw rows, or DataLoader. |
| `error_policy` | `ErrorPolicy` | `strict` | Failure/skip/permissive behavior, subject to source and processing stage. |
| `split` | `str \| None` | `train` | Requested split for hosted datasets and a hint when selecting a manifest in a directory. |
| `subset` | `str \| None` | `None` | Optional hosted-dataset configuration/subset name when supported by the source adapter. |
| `text_column` | `str \| None` | `None` | Explicit source column that should become canonical `text`. |
| `label_column` | `str \| None` | `None` | Explicit source column that should become canonical `label`. |
| `media_column` | `str \| None` | `None` | Explicit column containing a media path/URL or supported nested reference. |
| `max_samples` | positive `int \| None` | `None` | Optional cap. When set, must be an integer ≥1; strict validation rejects booleans. |
| `prompt_template` | `str \| None` | `None` | Optional `str.format_map` template applied to text records during finalization. |
| `seed` | `int` | `42` | Stored on the config model, but the current `load()` flow does not automatically apply it to transformations. Set `seed` explicitly on `.sample()` and `.split()` when using those methods. |
| `streaming` | `bool` | `False` | Requests lazy/streaming behavior for supported parsers/connectors. Exact support is source-specific. |
| `batch_size` | positive `int` | `8` | Default batch size when return type is DataLoader. |
| `shuffle` | `bool` | `True` | DataLoader shuffle setting; streaming DataLoaders turn shuffling off because they are iterable-backed. |
| `max_length` | positive `int` | `128` | Maximum text token length passed to a compatible tokenizer. Actual truncation behavior belongs to that tokenizer. |
| `target_sr` | positive `int` | `16000` | Target sample rate in Hz for audio processing. `load(..., audio_sr=...)` maps to this field. |
| `image_size` | 2-int tuple | `(224, 224)` | Interpreted as `(height, width)`, with positive integer dimensions. |
| `max_video_frames` | positive `int` | `16` | Number of frames produced per decoded video sample by the media helper. |
| `label_dtype` | `str` | `auto` | One of `auto`, `str`, `int`, `float`; affects label coercion in local structured-file parsers, with source/mode-specific behavior described below. |
| `treat_path_as_text` | `bool` | `False` | Source parsing hint for workflows where a path should be treated as content rather than as a dataset source; exact behavior depends on the parser. |
| `kwargs` | `dict[str, Any]` | `{}` | Extra source/parser/DataLoader options. Avoid using this as a substitute for documented top-level options when one exists. |

## 4.2 `load()` options

The top-level functional API and `DatasetHub.load()` expose the same user-facing options. The leading arguments are:

```python
load(
    target=None,
    src=None,
    type=None,
    returntype=None,
    *,
    source=None,
    modality=None,
    return_type=None,
    error_policy=None,
    split=None,
    subset=None,
    text_column=None,
    label_column=None,
    media_column=None,
    max_samples=None,
    prompt_template=None,
    tokenizer_or_processor=None,
    batch_size=None,
    shuffle=None,
    max_length=None,
    image_size=None,
    audio_sr=None,
    max_video_frames=None,
    streaming=None,
    treat_path_as_text=None,
    label_dtype=None,
    **kwargs,
)
```

The actual Python signatures accept these as regular keyword-capable parameters (not all are syntactically keyword-only). Prefer readable keyword calls rather than relying on positional order.

### Common parameters

| Parameter | What it controls |
|---|---|
| `target` | Required source: path, directory, URI/repository identifier, config object, record(s), DataFrame-like input, or iterable. |
| `src` / `source` | Source kind. `src` is the legacy short name; `source` is the clearer alias. If both are supplied, they must normalize to the same `SourceType`. |
| `type` / `modality` | Data modality. If both are supplied, they must normalize to the same `ModalityType`. Prefer `modality` in new code to avoid confusion with Python's built-in `type`. |
| `returntype` / `return_type` | Output representation. If both are supplied, they must normalize to the same `ReturnType`. Prefer `return_type` in new code. |
| `error_policy` | One of `strict`, `report_and_skip`, or `permissive`. |
| `split`, `subset` | Hosted dataset split/configuration hints; `split` may also guide local manifest selection. |
| `text_column`, `label_column`, `media_column` | Explicit field mapping from source columns into canonical record fields. |
| `max_samples` | Upper limit for supported parsers. Whether it applies before or after filtering is source-dependent. |
| `prompt_template` | Template string applied before the final return object is constructed. |
| `tokenizer_or_processor` | Caller-supplied object used for text tokenization or image/audio processing. |
| `batch_size`, `shuffle` | DataLoader settings when returning a loader. |
| `max_length` | Tokenizer maximum sequence length. |
| `image_size` | Target `(height, width)` for image and video frame transforms. |
| `audio_sr` | Target audio sample rate in Hz. |
| `max_video_frames` | Number of sampled video frames. |
| `streaming` | Request lazy ingestion where supported. |
| `treat_path_as_text` | Special parser behavior hint; use only for the source patterns for which it is relevant. |
| `label_dtype` | Label type casting strategy (`auto`, `str`, `int`, `float`). |
| `**kwargs` | Extra arguments for source adapters or PyTorch DataLoader construction. DataLoader-specific keys are kept away from external dataset connectors. |

## 4.3 Source inference rules

For a string/path target, source resolution follows this order:

1. `hf://...` → Hugging Face identifier with the prefix removed.
2. `kaggle://...` → Kaggle handle with the prefix removed.
3. Explicit `src`/`source` → use that source and validate an explicitly requested local file/directory exists.
4. Existing file path → `local_file`.
5. Existing directory path → `local_dir`.
6. Recognized dataset extension on a missing local path → raise `FileNotFoundError` rather than silently interpreting it as a hosted repo.
7. Path-like prefixes (`./`, `../`, `/`, `\\`, Windows drive paths) that do not exist → raise `FileNotFoundError`.
8. Otherwise → treat the string as a Hugging Face dataset identifier.

This fallback means a misspelled bare repository ID may be sent to the Hugging Face connector. Use an explicit source when a typo could otherwise look like a valid remote identifier.

## 4.4 Explicit modality and auto detection

For a local media file, modality is inferred from extension when no explicit modality is passed. Other local structured files default to text-like ingestion. For memory inputs, the loader inspects representative dictionary rows (up to ten in supported eager forms) for keys such as `audio`, `audio_path`, `image`, `image_path`, `video`, `video_path`, captions, descriptions, text, and transcription.

For directories, the parser profiles the tree, looks for manifests, and uses recognized media extensions to choose a path. Modality detection has limits:

- a mixed folder may require an explicit `modality`;
- a generator cannot be fully inspected without consuming it;
- ambiguous column names should be set explicitly through `text_column`, `label_column`, and `media_column`;
- content inside custom nested schemas may need explicit transformation before or after loading.

## 4.5 Configuration precedence and alias conflicts

When a `DatasetConfig` is supplied, `load(config, ...)` first deep-copies it. Any corresponding keyword argument that is explicitly non-`None` overrides that config value. The caller's config object is not mutated.

When a legacy alias and its newer alias are both supplied, conflicting values raise `ValueError`:

```python
# Valid because both aliases mean the same source.
ds = dh.load("./data.csv", src="file", source="local_file")

# Invalid: source arguments conflict.
ds = dh.load("./data.csv", src="local_file", source="huggingface")
```

For a `DatasetConfig`, setting the `source_type` explicitly controls source selection. When `source_type=None`, the same resolver is used to infer the source from `path_or_identifier`.

## 4.6 Image size ordering

The public API describes image size as `(height, width)`. For example, `(160, 256)` requests a 160-pixel high by 256-pixel wide image/frame output. Keep this convention consistent across `DatasetConfig`, `load()`, `image_to_tensor()`, and `load_video()`.

## 4.7 `label_dtype` behavior

- `auto`: local structured-file parsers infer a consistent label dtype across the parsed file (integer first, then float, otherwise string). Supported streaming parsers pre-scan labels for some structured formats so streaming and eager local-file behavior can agree. For in-memory records, `auto` preserves the source label value rather than coercing a string such as `"1"` to integer.
- `str`: cast non-null labels to strings.
- `int`: cast non-null labels to integers; invalid values raise an actionable conversion error.
- `float`: cast non-null labels to floating-point values.

For classification, verify the resulting type and class mapping before constructing a loss function. Automatically numeric labels are not guaranteed to be contiguous or start at zero; use `encode_labels()` or an explicit mapping if your model requires this.
