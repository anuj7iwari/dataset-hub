# Universal Dataset Hub — Complete Documentation

**Version 0.1.2 · Detailed user and API manual**

This document is the combined manual for the repaired source tree. The same material is also organized into individual pages in [`docs/`](docs/index.md). It describes observed version 0.1.2 behavior; hosted services and optional media codecs must be validated in the target environment.

## Contents

- [1. Overview and architecture](#1-overview-and-architecture)
- [2. Getting started](#2-getting-started)
- [3. Concepts and data model](#3-concepts-and-data-model)
- [4. Configuration reference](#4-configuration-reference)
- [5. Sources and file formats](#5-sources-and-file-formats)
- [6. Dataset operations](#6-dataset-operations)
- [7. PyTorch training integration](#7-pytorch-training-integration)
- [8. Media utilities](#8-media-utilities)
- [9. Public API reference](#9-public-api-reference)
- [10. Troubleshooting and limitations](#10-troubleshooting-and-limitations)
- [Runnable examples](#11-runnable-examples)

---

## 1. Overview and architecture

### 1.1 What the library is

Universal Dataset Hub is an ingestion and adapter library. It aims to reduce one-off source-specific parsing code by converting several kinds of data sources into a common `UniversalDataset` abstraction. The core package is named `data_hub`; the distribution name used by `pip` is `universal-dataset-hub`.

It supports workflows involving:

- text and text-like documents;
- structured JSON/YAML records and tabular files;
- images, audio files, and videos;
- rows combining text, labels, and a media reference;
- records already present in Python memory;
- optional hosted sources such as Hugging Face Hub and Kaggle;
- downstream NumPy, Pandas, Hugging Face Dataset, and PyTorch use.

This is not a model-training framework, model registry, annotation platform, or complete media-management system. It prepares records and connects them to common Python/ML interfaces.

### 1.2 Architectural layers

The implementation is organized into six modules.

| Module | Responsibility | Important public objects |
|---|---|---|
| `schemas.py` | Canonical enum values and validation models | `ModalityType`, `SourceType`, `ReturnType`, `ErrorPolicy`, `DatasetItem`, `DatasetConfig`, `DatasetSummary` |
| `hub.py` | Source resolution, configuration, dispatch, return-type selection | `DatasetHub`, `load()` |
| `parsers.py` | Source-specific decoding and row normalization | `row_to_dataset_item()`, local parsers, memory and hosted adapters |
| `dataset.py` | Dataset container, transformations, summaries, exports, PyTorch adapters | `UniversalDataset`, `DatasetSplits`, `PyTorchUniversalAdapter`, `PyTorchIterableAdapter` |
| `media.py` | Standalone image/audio/video decoding and conversion | `load_image()`, `image_to_tensor()`, `load_audio()`, `audio_to_tensor()`, `load_video()` |
| `collators.py` | Batch-level validation, padding, and tensor stacking | `UniversalBatchCollator` |

### 1.3 Load lifecycle

A normal `load()` flow consists of these stages:

1. **Target classification.** A target may be a local path, repository identifier, `DatasetConfig`, list/tuple/dictionary of records, DataFrame-like object, or iterable.
2. **Source resolution.** URI prefixes such as `hf://` and `kaggle://` take precedence. Existing local paths are classified as files/directories. Explicit `src`/`source` arguments can force a source. Otherwise a non-path-like identifier is treated as a Hugging Face dataset identifier.
3. **Configuration validation.** Modality, return type, error policy, image size, label dtype, and positive-integer settings are normalized and validated through the schema models.
4. **Ingestion.** The hub dispatches to the local-file, local-directory, in-memory, Hugging Face, or Kaggle parser.
5. **Record normalization.** Source rows are converted to `DatasetItem` records. Canonical fields include `id`, `text`, `label`, and `media_path`; source columns are retained in `raw_data`; additional information is stored in `metadata`.
6. **Dataset construction.** The records, inferred columns, source metadata, warning/error records, modality, and effective configuration are assembled into `UniversalDataset`.
7. **Finalization.** An optional prompt template is applied. The hub then returns a `UniversalDataset`, PyTorch dataset adapter, NumPy column dictionary, raw records, or DataLoader according to `return_type`.

### 1.4 Typical call paths

#### Eager records

`load(list/dict/DataFrame) → parse_memory → DatasetItem list → UniversalDataset → optional transformations/output adapter`

#### Local structured file

`load(path) → source/path resolution → parse_local_file → DatasetItem list or lazy sequence → UniversalDataset → output adapter`

#### Media directory

`load(directory) → directory profile → manifest detection or modality scan → normalized items → UniversalDataset`

#### Model training

`UniversalDataset.to_dataloader() → PyTorchUniversalAdapter or PyTorchIterableAdapter → per-item media/tokenizer processing → UniversalBatchCollator → batch dictionary`

For `return_type="dataloader"`, steps are initiated by `load()` during finalization.

### 1.5 Eager versus streaming

**Eager** mode materializes data into an indexable collection. It is appropriate when the dataset fits in memory and you need slicing, reproducible random sampling, sorting, or train/validation/test splits.

**Streaming** mode retains a lazy sequence for supported file formats and iterable-backed sources. Rows may only be validated when consumed. An initial `len()` can represent an upper bound or be unknown, depending on the source; after iteration, metadata may be updated with the actual yielded/skipped counts.

Streaming tradeoffs:

- the source may be one-shot;
- random access and random splitting are not supported;
- error discovery is deferred until iteration;
- `take(0)` should not consume rows;
- generic generators cannot be exhaustively inspected to infer modality before reading them;
- a bounded peek used for training inference can consume/cache a prefix to keep those samples available, but it does not make arbitrary generators rewindable forever.

See [Concepts and data model](docs/concepts.md) and [Sources and formats](docs/sources-and-formats.md).

### 1.6 Error boundaries

The library has multiple validation layers. A parser can reject malformed source rows; media utilities can reject unsupported/corrupt content; a PyTorch adapter can wrap media errors with the sample ID/path; and the collator can reject inconsistent batch shapes or missing fields. Read [Troubleshooting](docs/troubleshooting.md) for interpreting these distinct failure stages.

---

## 2. Getting started

### 2.1 Requirements

- Python **3.10 or newer**.
- Core runtime dependencies: NumPy, PyTorch, Pydantic v2, and Pillow.
- Optional adapters require additional packages; see [Sources and formats](docs/sources-and-formats.md).

### 2.2 Install from a source checkout

From the directory containing `pyproject.toml`:

```bash
python -m pip install .
```

For editable development:

```bash
python -m pip install -e .
```

For core tests and common local parsing helpers:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Optional extras:

```bash
python -m pip install -e ".[audio]"    # SciPy-based audio support
python -m pip install -e ".[video]"    # OpenCV video decoding
python -m pip install -e ".[hf]"       # Hugging Face datasets
python -m pip install -e ".[kaggle]"   # KaggleHub
python -m pip install -e ".[parquet]" # pyarrow / Parquet
python -m pip install -e ".[yaml]"    # PyYAML
python -m pip install -e ".[all]"     # all listed optional adapters
```

Depending on the source and media file, other libraries on the runtime may be used when present (for example, a tokenizer/processor supplied by the caller). The extras shown above are the declared package extras and do not guarantee support for every media codec or hosted dataset.

### 2.3 Basic in-memory text dataset

```python
import data_hub as dh

records = [
    {"text": "A beautifully written review.", "label": 1},
    {"text": "This did not meet expectations.", "label": 0},
]

ds = dh.load(records, modality="text")

print(ds)
print(ds.num_samples)
print(ds.preview(n=2))
summary = ds.get_summary()
print(summary.model_dump())
```

The returned `ds` is a `UniversalDataset` by default. Each row is normalized to `DatasetItem`, while its original dictionary is preserved under `raw_data`.

### 2.4 Load a local CSV

Suppose `train.csv` contains:

```csv
text,label
A positive example,1
A negative example,0
```

Use:

```python
import data_hub as dh

ds = dh.load(
    "./train.csv",
    text_column="text",
    label_column="label",
    label_dtype="auto",
    error_policy="strict",
)
```

`label_dtype="auto"` attempts safe column-wide inference for string-typed CSV/TSV labels; all non-empty values must be parseable to cast them consistently. Choose `label_dtype="str"` to keep source labels as strings or `"int"`/`"float"` to require explicit numeric conversion.

### 2.5 Load an image directory

```python
images = dh.load("./images", modality="image")
print(images.preview(3, detailed=True))
```

Directory scans recognize image extensions and may assign the parent directory's name as the label. The scanner can optionally validate image decoding depending on the error policy. A directory containing a recognized manifest such as `train.csv` may be parsed as a table instead of a pure image folder when the modality is not explicitly fixed to image.

### 2.6 Get a PyTorch DataLoader

```python
loader = dh.load(
    "./train.csv",
    return_type="dataloader",
    batch_size=16,
    shuffle=True,
    tokenizer_or_processor=my_tokenizer,
    max_length=256,
)

for batch in loader:
    # Usually contains input_ids, attention_mask, labels when a tokenizer is used.
    print(batch.keys())
    break
```

`my_tokenizer` is intentionally a placeholder. Pass a callable tokenizer/processor compatible with the selected modality; the library does not download or construct one automatically.

### 2.7 Load options through a configuration object

```python
from data_hub import DatasetConfig, load

cfg = DatasetConfig(
    path_or_identifier="./train.csv",
    source_type="local_file",
    modality="text",
    label_column="label",
    text_column="text",
    max_samples=10_000,
    error_policy="report_and_skip",
)
ds = load(cfg)
```

An explicitly supplied keyword argument to `load(cfg, ...)` overrides the corresponding value on the copied configuration. The original configuration object is not mutated. See [Configuration reference](docs/configuration.md).

### 2.8 Your first data audit

A useful basic inspection sequence:

```python
print(ds.preview(5, detailed=True))
summary = ds.get_summary()
print("samples:", summary.total_samples)
print("columns:", summary.columns)
print("modality:", summary.modality)
print("skipped:", summary.skipped_samples)
print("warnings:", summary.warnings)
print("parse errors:", summary.parsing_errors[:5])
```

Inspect these values before starting long model runs. A successful `load()` does not prove that every lazy row has been read, that every optional media object has been decoded, or that the schema matches your model's objective.

---

## 3. Concepts and data model

### 3.1 Modality

`ModalityType` is the canonical description of the kind of examples a dataset contains.

| Canonical value | Intended use | Common aliases accepted by `from_str()` |
|---|---|---|
| `text` | Text/NLP records and text-only tokenization | `txt`, `nlp`, `language` |
| `image` | Image samples | `img`, `vision`, `visual`, `picture` |
| `audio` | Audio/speech samples | `sound`, `speech`, `voice` |
| `video` | Video files represented as sampled frames | `vid` |
| `multimodal` | Combined input such as image + caption/text | `vl`, `vlm`, `vision-language` |
| `json` | Structured JSON-oriented workflow marker | `structured` |
| `tabular` | Table-like records | `csv`, `table`, `dataframe` |

The parser may promote an initially text modality when records clearly contain media references. If a dataset has heterogeneous rows or you know the true training task, set modality explicitly rather than depending on inference.

`json` and `tabular` are enum values. In the current implementation, `UniversalDataset`'s core transformation methods operate on normalized records rather than implementing a separate specialized tensor adapter for every enum value. For a concrete model pipeline, test the chosen return type/modality combination with representative data.

### 3.2 Source type

`SourceType` describes where input came from, not what modality it contains.

| Canonical value | Meaning |
|---|---|
| `local_file` | A structured/text/media file on disk |
| `local_dir` | A directory scanned for manifests or files |
| `memory` | Existing Python objects or records |
| `huggingface` | A Hugging Face Hub dataset identifier |
| `kaggle` | A Kaggle dataset handle or a cached/downloaded dataset |

Accepted aliases include `hf`/`hf_hub`, `file`/`local`/`path`, `dir`/`directory`/`folder`, and `in_memory`/`list`/`dataframe`/`df`.

For string targets, `hf://owner/name` and `kaggle://owner/name` explicitly identify hosted sources. Existing local paths are detected before the fallback to Hugging Face IDs. Explicit source selection is preferred when a string could plausibly mean more than one thing.

### 3.3 Return type

`ReturnType` controls the object returned from `load()` after ingestion/finalization.

| Value | Returned representation | Notes |
|---|---|---|
| `dataset` | `UniversalDataset` | Default; supports inspection and transformations |
| `pytorch` | PyTorch `Dataset` or `IterableDataset` adapter | Per-sample token/media processing occurs at access/iteration time |
| `numpy` | `dict[str, numpy.ndarray]` | One array per selected/inferred dataset column; heterogeneous/ragged values can use object arrays |
| `raw` | `list` of raw dictionaries or model dumps | If `raw_data` is truthy, that source dictionary is returned; otherwise the normalized model dump is used |
| `dataloader` | `torch.utils.data.DataLoader` | Uses a modality-aware default collator unless one is supplied through loader kwargs |

Aliases include `universal`, `pt`, `torch`, `tensor`, `tensors`, `np`, `ndarray`, `dict`, `pil`, and `loader`. The aliases are string conveniences; they do not change the output contract.

### 3.4 The normalized record: `DatasetItem`

Every standard record has this model shape:

```python
DatasetItem(
    id=None,
    text=None,
    media_path=None,
    label=None,
    metadata={},
    raw_data=None,
    features=None,
)
```

| Field | Meaning | Example |
|---|---|---|
| `id` | Source ID or generated row index | `"sample-17"` or `17` |
| `text` | Canonical text/caption/transcript representation | `"A dog on grass"` |
| `media_path` | Canonical string reference to a local media file or URL-like value | `"images/dog.jpg"` |
| `label` | Class/target value | `0`, `"cat"`, `[1, 0]` |
| `metadata` | Hub/parser-produced supplemental metadata, e.g. preserved `messages` | `{"filename": "dog.jpg"}` |
| `raw_data` | Original source row, deep-copied during normalization | `{"caption": "A dog", "category": "animal"}` |
| `features` | Optional feature payload slot | Application-defined dictionary |

The `DatasetItem` Pydantic model allows arbitrary types, but canonical `text`, `media_path`, and common label handling remain intentionally simple. Do not assume `features` is computed automatically.

#### Canonical fields versus raw data

`raw_data` is important when a downstream export needs source-specific columns. The library uses canonical fields such as `text`, `label`, and `media_path` as authoritative values when it builds a canonical export record. Source-specific fields not promoted to canonical fields remain in `raw_data`.

A `media_path` generally represents a reference to media. An embedded source object—such as Hugging Face audio stored as `{"array": ..., "sampling_rate": ...}`—may remain inside `raw_data` and need not populate `media_path`.

### 3.5 Dataset container: `UniversalDataset`

`UniversalDataset` wraps normalized items together with:

- `modality`, `source_type`, and `return_type`;
- `error_policy`;
- a human-readable dataset `name`;
- known `columns`;
- parser/source `metadata`;
- audio sample-rate and maximum video-frame settings;
- the effective `DatasetConfig`, where available.

It supports ordinary iteration and index/slice access in eager mode, plus inspection, transformation, conversion, export, and model-framework adapters. Streaming mode intentionally has fewer random-access guarantees.

### 3.6 Summary metadata

`DatasetSummary` is a Pydantic model for high-level reporting. It includes name, modality/source/return type, error policy, sample count, streaming status, columns, split counts, preview records, token estimate and its method, class distribution, memory estimate, skipped count, parsing error records, warnings, included/excluded file lists, and metadata.

Not every field is available for every source or mode. For example, `total_samples=None` represents an unknown count and should not be treated as zero. Token counts and memory are estimates, not a resource guarantee.

### 3.7 Error policy

`ErrorPolicy` has three canonical values:

- `strict`: fail promptly on errors that the current ingestion/processing stage detects. This is the safest default for training pipelines.
- `report_and_skip`: omit invalid records where the parser/adaptor supports skipping and preserve diagnostics in metadata. For streams, the error can only be observed when the row is consumed.
- `permissive`: prioritize best-effort ingestion and preserve warning/error information where possible. This is not a promise that every corrupt value can be repaired or that downstream training will succeed.

Aliases: `fail` → strict, `skip` → report-and-skip, `warn` → permissive.

Error policy behavior depends on the specific source and stage. Review `summary.skipped_samples`, `summary.parsing_errors`, and `summary.warnings` rather than relying solely on the absence of an exception.

### 3.8 Other important terms

- **Manifest:** A structured file in a directory (commonly CSV or JSONL) that lists examples and labels/media references.
- **Collator:** A batch-level function that combines per-sample dictionaries into padded/stacked tensors.
- **Processor/tokenizer:** A user-supplied callable, often from a model library, that turns text or media into model-specific tensors.
- **Streaming sequence:** A lazy, often one-pass iterator over normalized items.
- **Eager collection:** An in-memory collection that can be indexed/sliced and randomly sampled/split.

---

## 4. Configuration reference

### 4.1 `DatasetConfig` at a glance

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

#### Field reference

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

### 4.2 `load()` options

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

#### Common parameters

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

### 4.3 Source inference rules

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

### 4.4 Explicit modality and auto detection

For a local media file, modality is inferred from extension when no explicit modality is passed. Other local structured files default to text-like ingestion. For memory inputs, the loader inspects representative dictionary rows (up to ten in supported eager forms) for keys such as `audio`, `audio_path`, `image`, `image_path`, `video`, `video_path`, captions, descriptions, text, and transcription.

For directories, the parser profiles the tree, looks for manifests, and uses recognized media extensions to choose a path. Modality detection has limits:

- a mixed folder may require an explicit `modality`;
- a generator cannot be fully inspected without consuming it;
- ambiguous column names should be set explicitly through `text_column`, `label_column`, and `media_column`;
- content inside custom nested schemas may need explicit transformation before or after loading.

### 4.5 Configuration precedence and alias conflicts

When a `DatasetConfig` is supplied, `load(config, ...)` first deep-copies it. Any corresponding keyword argument that is explicitly non-`None` overrides that config value. The caller's config object is not mutated.

When a legacy alias and its newer alias are both supplied, conflicting values raise `ValueError`:

```python
## Valid because both aliases mean the same source.
ds = dh.load("./data.csv", src="file", source="local_file")

## Invalid: source arguments conflict.
ds = dh.load("./data.csv", src="local_file", source="huggingface")
```

For a `DatasetConfig`, setting the `source_type` explicitly controls source selection. When `source_type=None`, the same resolver is used to infer the source from `path_or_identifier`.

### 4.6 Image size ordering

The public API describes image size as `(height, width)`. For example, `(160, 256)` requests a 160-pixel high by 256-pixel wide image/frame output. Keep this convention consistent across `DatasetConfig`, `load()`, `image_to_tensor()`, and `load_video()`.

### 4.7 `label_dtype` behavior

- `auto`: local structured-file parsers infer a consistent label dtype across the parsed file (integer first, then float, otherwise string). Supported streaming parsers pre-scan labels for some structured formats so streaming and eager local-file behavior can agree. For in-memory records, `auto` preserves the source label value rather than coercing a string such as `"1"` to integer.
- `str`: cast non-null labels to strings.
- `int`: cast non-null labels to integers; invalid values raise an actionable conversion error.
- `float`: cast non-null labels to floating-point values.

For classification, verify the resulting type and class mapping before constructing a loss function. Automatically numeric labels are not guaranteed to be contiguous or start at zero; use `encode_labels()` or an explicit mapping if your model requires this.

---

## 5. Sources and file formats

### 5.1 Source matrix

| Source | Example target | Typical output before finalization | Requirements / caveats |
|---|---|---|---|
| In-memory records | `[{"text": "hi", "label": 1}]` | Eager `DatasetItem` collection by default; stream behavior may be requested for iterable sources | Lists, tuples, dictionaries, DataFrame-like inputs, and iterables are supported by memory normalization. Generator modality inference is necessarily limited. |
| Local structured file | `./data/train.csv` | Parsed records or lazy stream, depending on format/options | Supported extensions are listed below. |
| Local directory | `./data/images/` | Manifest parsing, media-folder scan, or text-document bundle | Directory profiling can select a manifest and can warn about mixed modalities. |
| Hugging Face | `hf://owner/dataset` or `owner/dataset` | Dataset split converted/streamed into normalized rows | Requires `datasets`; remote access, dataset config and split availability depend on the repository. |
| Kaggle | `kaggle://owner/dataset` | Downloaded/cached files then parser routing | Requires `kagglehub` plus any required credentials and network/service access. |

### 5.2 Local file support

#### Structured/text extensions

The parser recognizes:

- `.json`
- `.jsonl`
- `.ndjson`
- `.csv`
- `.tsv`
- `.parquet`
- `.txt`
- `.md`
- `.yaml`
- `.yml`

Parquet requires `pyarrow` in this distribution setup. YAML requires PyYAML. Use UTF-8-compatible inputs where practical.

#### Media extensions

Image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`.

Audio extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`.

Video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

Extension recognition does not mean that every codec, image encoding, or container is supported by every installed backend. A file can have an accepted extension and still fail to decode.

### 5.3 CSV and TSV

CSV/TSV files are commonly used for text classification, captioning, labels, and manifest rows.

Example:

```csv
id,text,label,image_path
001,"A red car on the road",car,images/001.jpg
002,"A bus at a stop",bus,images/002.jpg
```

```python
ds = dh.load(
    "./manifest.csv",
    modality="multimodal",
    text_column="text",
    label_column="label",
    media_column="image_path",
    error_policy="strict",
)
```

Use `text_column`, `label_column`, and `media_column` to disambiguate columns rather than trusting automatic aliases. Relative media references in a manifest need to be resolvable in the context of the source layout; inspect sample `media_path` values and call `preview(detailed=True)` if files are not found.

For streaming CSV, quoted fields may contain newlines, so the parser counts logical records rather than assuming one newline equals one sample. If a CSV quoting error makes resynchronization impossible, the parser records the error and stops at that boundary rather than pretending later rows were read safely. With `streaming=True`, `.txt`/`.md` sources yield one non-empty, stripped line per item; in eager mode they use the same line-oriented record convention.

### 5.4 JSON and JSONL

JSON supports a top-level array, object, or scalar at parse time. A top-level array is a sequence of records; a top-level mapping/scalar is treated as one record. JSONL/NDJSON treats each non-empty line as an independent JSON value, normally a mapping record.

Example JSONL:

```jsonl
{"id": "a", "text": "good", "label": 1}
{"id": "b", "text": "bad", "label": 0}
```

For streaming JSON arrays, the parser reads elements incrementally rather than loading the full array as a single Python list. This still requires buffering the current element. A single very large element can therefore consume substantial memory.

Malformed JSONL lines are handled by the active error policy. Under skip-oriented handling, a malformed line may be omitted and recorded. Review `parsing_errors` rather than assuming a successful stream means all lines were valid.

### 5.5 YAML

YAML support is optional and uses PyYAML. A top-level sequence can be constructed element-by-element in streaming mode; a mapping/scalar is constructed as one document/record. As with JSON, a large individual object must exist in memory while it is parsed. Never load YAML from untrusted sources without considering PyYAML's security and resource implications and the behavior of the installed parser/version.

### 5.6 Plain text and Markdown

`.txt` and `.md` files are read as text-like records. When a directory is interpreted as a text collection, compatible text/Markdown files are recursively collected; empty content is not made into a regular text sample. The filename and character-count details may be recorded in metadata.

### 5.7 Parquet

Parquet is intended for tabular rows. Install the declared extra:

```bash
python -m pip install 'universal-dataset-hub[parquet]'
```

In development from a checkout:

```bash
python -m pip install -e '.[parquet]'
```

Parquet reading/export should be tested against the specific pandas/pyarrow versions and schema. Nested/extension columns may require manual normalization.

### 5.8 Directory profiling and manifests

Directory input is more than a recursive `glob`: the parser first profiles files and checks recognized manifest names such as:

- `train.csv`, `data.csv`, `metadata.csv`, `labels.csv`, `dataset.csv`
- `train.jsonl`, `data.jsonl`

When an eligible manifest is present, the parser selects a manifest based on the configured `split` name where possible. If multiple manifests exist, it may choose a matching one or fall back to the first and emits a warning. If the directory contains both a manifest and media/text references, the manifest path can represent a multimodal dataset.

If no manifest is selected, a directory with only images/audio/video can be scanned as a media folder. A parent directory name may be assigned as the label (e.g. `cats/001.jpg` → label `cats` when the media is under a class subdirectory). Mixed-modality folders are not automatically fused into paired examples just because files share a directory; an explicit manifest is the safer pairing strategy.

Directory profiling scans are bounded by `max_scan` (default 500 entries in the profile helper). For very large directory trees, inferred counts can be lower bounds; inspect profile metadata and do not treat a bounded profile as a complete inventory.

### 5.9 In-memory records

Supported patterns include a single mapping, a list/tuple of mappings, Pandas-like objects whose `to_dict(orient="records")` and `columns` interface is compatible, and generic iterables/generators. For other DataFrame APIs, convert the source to a list of dictionaries first.

```python
records = [
    {"sentence": "I love the product", "sentiment": "positive"},
    {"sentence": "The product is broken", "sentiment": "negative"},
]

ds = dh.load(
    records,
    text_column="sentence",
    label_column="sentiment",
    label_dtype="str",
)
```

The row normalizer recognizes common text fields (`text`, `caption`, `description`, `transcription`, `transcript`, `sentence`, `content`, and some instruction/question formats), label aliases (`label`, `category`, `target`, `class`, `sentiment`), and media references (`media_path`, `image_path`, `audio_path`, `video_path`, `file_path`, `filepath`, `url`, `media`, and selected nested path/url fields). If your schema differs, map it explicitly or transform it with `map()`.

Chat-like rows with `messages` or `conversations` preserve the structures in `metadata` while also creating a readable text representation. This readable text representation is convenient for inspection, but it is not a substitute for the preserved structured messages when applying chat-specific training logic.

### 5.10 Hugging Face Hub

Typical call:

```python
ds = dh.load(
    "hf://owner/dataset",
    split="train",
    subset=None,
    modality="text",
    streaming=True,
)
```

Install the connector extra first. The adapter maps source rows into `DatasetItem` records and attempts to honor `split`, `subset`, `max_samples`, and streaming settings supported by the upstream `datasets` API. Dataset scripts, gated repositories, private datasets, custom features, and audio/image columns may have special requirements. Hosted behavior is integration-dependent and should be tested with the target dataset.

A bare string like `owner/dataset` falls back to Hugging Face when it is not an existing local path and does not have a recognized local-file extension. If you intended a local path, use an explicit path or source so a typo cannot be misrouted.

### 5.11 Kaggle

Example:

```python
ds = dh.load("kaggle://owner/dataset", source="kaggle", modality="text")
```

Install `kagglehub` and configure the credentials required for the target account/dataset. The hub passes a cache directory to the Kaggle adapter; a `DatasetHub(cache_dir=...)` instance allows you to choose a cache location. Some Kaggle downloads contain multiple manifests, labels, and assets. Inspect the downloaded directory and parser metadata to verify which manifest was selected.

### 5.12 Remote URLs versus media URLs

The row normalizer can retain URL-like values as a media reference. This does not mean every URL is automatically downloaded, authenticated, or made available to a decoder. A media helper may accept a path or a supported direct byte/dict payload, but a remote URL may require additional download logic before use depending on the backend. Test a representative URL in your environment and handle authentication/timeouts separately.

---

## 6. Dataset operations

The main object is `UniversalDataset`, normally returned from `data_hub.load()` with its default `return_type="dataset"`. Methods generally return a new `UniversalDataset` instead of altering the original collection. Prefer assigning each result to a new variable to keep transformation order visible.

### 6.1 Inspection and indexing

#### `preview(n=5, detailed=False)`

Returns a list of JSON-friendly preview dictionaries. It includes `id`, canonical `modality`, and `label`; text samples include a truncated snippet and character length; media references include the path and a file-existence check when applicable. With `detailed=True`, available `metadata` and the list of `raw_data` keys are included.

```python
for row in ds.preview(5, detailed=True):
    print(row)
```

`n` must be a non-negative integer. A streaming preview uses the lazy sequence's bounded `peek()` method; peeking is designed to preserve the consumed prefix for later iteration.

#### `get_summary()` / `summary()`

Returns a `DatasetSummary` model. Useful fields include:

- `total_samples` and `is_streaming`;
- `columns`, `modality`, and `source_type`;
- `estimated_tokens` and `token_estimation_method` for text/multimodal data;
- `class_distribution` where available;
- `skipped_samples`, `parsing_errors`, and `warnings`;
- `included_files` and `excluded_files` for directory scans.

Token count uses a rough character-based heuristic (about 3.2 characters/token) on a bounded sample; it is not a tokenizer-exact count. Memory use is also an estimate rather than an allocator-accurate measurement. Treat both as diagnostics, not capacity guarantees.

#### `len()`, `num_samples`, indexing, iteration

```python
print(len(ds))
print(ds.num_samples)
first = ds[0]
for item in ds:
    print(item.id, item.text, item.label)
```

For eager collections, indexing and slicing operate over in-memory items. `num_samples` can be `None` when a streaming source has unknown cardinality. Do not assume an unbounded or one-shot stream can be measured without consuming it.

### 6.2 `take(n)`

Returns the first `n` samples as a materialized, non-streaming dataset. In streaming mode it consumes at most `n` rows and returns those rows in memory. `take(0)` is explicitly designed to consume no source rows.

```python
sample = stream.take(100)
print(sample.num_samples)
```

`n` must be a non-negative integer and booleans are rejected despite being Python integers.

### 6.3 `filter(condition)`

Keeps rows for which a predicate returns truthy. The predicate receives a deep-copied `DatasetItem`, so mutating the object inside the predicate should not mutate the source dataset.

```python
long_reviews = ds.filter(
    lambda item: item.text is not None and len(item.text) >= 100
)
```

In eager mode the filter is applied immediately. In streaming mode it returns another lazy dataset and the predicate is evaluated while the output is consumed. Ensure the predicate is deterministic and does not depend on side effects if repeatability matters.

### 6.4 `map(fn)`

Applies a transformation to each item. A callback must return either:

- a `DatasetItem`, or
- a dictionary of updates.

Anything else raises `TypeError` rather than silently doing nothing.

```python
def normalize_review(item):
    return {
        "text": (item.text or "").strip().lower(),
        "label": item.label,
        "normalized": True,
    }

normalized = ds.map(normalize_review)
```

Dictionary updates for canonical fields `text`, `label`, and `media_path` update the corresponding `DatasetItem` fields. The returned keys are also merged into a deep-copied `raw_data` dictionary. Returning a `DatasetItem` gives you more control but means you must deliberately preserve any fields that should remain.

The callback receives an isolated deep copy. A callback should be a pure transformation where possible; it may run lazily for streaming datasets.

### 6.5 `sort(by=None, descending=False, key=None)`

Sorts an eager collection and returns a new dataset. Streaming datasets cannot be sorted.

Supported `by` values:

- `"length"`, `"len"`, `"char_length"`: length of the canonical text;
- `"label"`, `"target"`, `"class"`: label;
- `"id"`: stable string-based ordering, so mixed integer/string IDs remain comparable;
- any key found in source `raw_data` columns;
- omitted/unrecognized `by`: canonical text string fallback.

Alternatively, pass `key=lambda item: ...` for a custom key. The custom key receives a deep-copied item. Missing values are grouped after present values in ascending sort; descending behavior reverses the present-value ordering while missing rows retain original relative order.

```python
by_length = ds.sort(by="length", descending=True)
by_score = ds.sort(key=lambda item: float((item.raw_data or {}).get("score", 0)))
```

### 6.6 `clean(...)`

Signature:

```python
clean(
    deduplicate=True,
    strip_whitespace=True,
    drop_na=True,
    drop_corrupted_media=False,
)
```

Available actions:

- trim leading/trailing whitespace from canonical text;
- drop empty/missing text for `text` modality;
- drop records that lack expected media for image/audio/video modality;
- deduplicate based on modality-relevant identity, text, labels, media references or embedded payload identity when available;
- when `drop_corrupted_media=True`, check media file existence/size and attempt decoding with the matching media helper.

```python
cleaned = ds.clean(
    deduplicate=True,
    strip_whitespace=True,
    drop_na=True,
    drop_corrupted_media=True,
)
```

Important limits:

- A media corruption check requires a decodable, local media reference in the path it examines; do not assume it validates every arbitrary remote URL or every custom payload format.
- Deduplication semantics are modality- and field-dependent. Inspect before/after counts and test on a representative dataset if duplicate identity is important.
- For streaming data, cleaning is lazy and the set of seen identities grows as rows are consumed.

### 6.7 `encode_labels(mapping=None)`

Creates integer labels and records the mapping in metadata under `label_mapping`. If no mapping is supplied, unique non-null labels are converted to strings, sorted, then assigned integer indices starting at zero.

```python
encoded = ds.encode_labels()
print(encoded.metadata.get("label_mapping"))
```

With a mapping:

```python
encoded = ds.encode_labels({"negative": 0, "neutral": 1, "positive": 2})
```

This method does not support streaming datasets; materialize first. Values missing from an explicitly provided mapping are left unchanged by the current implementation, so validate coverage if the target loss requires integers.

### 6.8 `format_prompt(template=None, column_mapping=None, safe=True)`

Formats canonical `text` from a Python format string using source fields plus canonical `text` and `label`.

```python
prompted = ds.format_prompt(
    template="Classify this review.\nReview: {text}\nSentiment: {label}",
)
```

`column_mapping` maps a new placeholder name to the source key name:

```python
prompted = ds.format_prompt(
    template="Question: {question}\nAnswer: {answer}",
    column_mapping={"question": "query", "answer": "target_answer"},
)
```

When `safe=True`, missing placeholders are left as `{placeholder}`. With `safe=False`, missing placeholders become empty strings. A formatting exception is caught and the original text is used. The formatter records the formatted value in canonical `text` and `raw_data["text"]` on the returned copy.

Prompt templates are string interpolation, not a tokenizer chat-template engine. If preserving message roles, loss masks, or model-specific chat formatting is important, use the preserved structured message data and the model's intended template logic.

### 6.9 `sample(n=None, frac=None, seed=42)`

Randomly selects a subset from an eager collection. `n` selects a count; `frac` selects a fraction. `seed` makes the sampled order reproducible with Python's random generator. Streaming datasets return an error for operations needing full random access or have to be materialized first.

```python
subset = ds.sample(n=500, seed=123)
quarter = ds.sample(frac=0.25, seed=123)
```

Do not supply both `n` and `frac` unless the specific implementation behavior has been confirmed for the version you use. Ensure `n`/`frac` matches the dataset size and desired semantics.

### 6.10 `split(...)`

```python
splits = ds.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    seed=42,
)
train = splits.train
validation = splits.val
test = splits.test
```

`DatasetSplits` exposes `train`, `val`, and `test`; it also supports string indexing such as `splits["train"]` where a split exists. `val` or `test` may be `None` if a split is empty.

Rules:

- ratios must be finite values in `[0, 1]`;
- ratios must sum to exactly 1 within a small floating-point tolerance;
- streaming datasets must be materialized before random splitting;
- if `shuffle=True`, a seeded Python random generator shuffles the item order;
- integer truncation can make small splits empty. Use counts appropriate to the dataset size.

For example, a three-record dataset cannot meaningfully guarantee train/validation/test examples for every choice of common ratios. Check `len(splits.train)`, `len(splits.val) if splits.val else 0`, and `len(splits.test) if splits.test else 0`.

### 6.11 Extract strings and columns

#### `to_texts(column=None)`

Returns a list of strings. If `column` is supplied and exists in `raw_data`, that column is converted to string. Otherwise canonical text is preferred; if it is missing, the method uses the first string-valued field found in `raw_data`.

```python
texts = ds.to_texts()
labels_as_text = ds.to_texts(column="label_name")
```

The fallback is convenient but not schema-aware; for training, use explicit columns and validate the produced list.

#### `to_numpy(columns=None)`

Returns a dictionary mapping column names to NumPy arrays. With no explicit columns, it uses `ds.columns`. Canonical `text`, `label`, and `media_path` fields take precedence, then other columns are read from `raw_data`.

```python
arrays = ds.to_numpy(columns=["text", "label"])
```

Ragged/heterogeneous values are retained in a one-dimensional object array if NumPy cannot build a regular array. Object arrays are not automatically ready for GPU training and may need explicit encoding/padding. Conversion is intended for materialized datasets: a stream of unknown length may need to be consumed/materialized first, and attempting to infer its length can raise an error.

### 6.12 Pandas and Hugging Face conversion

#### `to_pandas()`

Builds a Pandas DataFrame from canonical records and raw columns. Requires `pandas` and materializes all yielded rows, so streaming input can be consumed and a large source can exhaust memory.

#### `to_hf_dataset()`

Builds a Hugging Face `datasets.Dataset` from the Pandas conversion. Requires both `pandas` and `datasets` compatible with your environment. It materializes the data and is therefore not a bounded-memory streaming conversion.

```python
df = ds.to_pandas()
hf_ds = ds.to_hf_dataset()
```

Pandas/Hugging Face conversion may not preserve arbitrary custom Python objects in a directly trainable representation; normalize object/media columns first and verify resulting column types.

### 6.13 `export(output_path, format="jsonl")`

Supported formats: `jsonl`, `json`, `csv`, `parquet`. The output directory is created if needed. Invalid format names raise `ValueError`.

```python
ds.export("./out/processed.jsonl", format="jsonl")
ds.export("./out/processed.csv", format="csv")
```

Canonical `text`, `label`, and `media_path` fields take precedence over corresponding source fields. Tensor/NumPy data are converted to lists for JSON serialization; datetime-like objects use `isoformat`; PIL images are written as diagnostic string descriptors rather than their original pixel data. Therefore `export()` is record serialization—not a guarantee that embedded image/audio/video payloads are copied into portable assets. For reusable media datasets, export assets separately and store stable paths/IDs in the manifest.

Use Parquet only with the optional engine installed and test nested-column types. Treat an export as incomplete until you check that the target path exists, has nonzero size where expected, and can be re-read.

---

## 7. PyTorch training integration

### 7.1 Options

There are three related layers:

1. `UniversalDataset.to_pytorch_dataset()` creates a map-style `torch.utils.data.Dataset` or streaming `IterableDataset` adapter.
2. `UniversalDataset.to_dataloader()` builds a standard PyTorch `DataLoader` and chooses `UniversalBatchCollator` by default.
3. `load(..., return_type="pytorch" | "dataloader")` requests those representations during `load()` finalization.

The library does not create a model, optimizer, loss function, or training loop. It creates input dictionaries and batches; your model must agree with those keys and shapes.

### 7.2 Text examples

Without a tokenizer:

```python
pt_ds = ds.to_pytorch_dataset()
example = pt_ds[0]
## Contains id, text, and label where present.
```

With a compatible tokenizer:

```python
pt_ds = ds.to_pytorch_dataset(
    tokenizer_or_processor=tokenizer,
    max_length=256,
)
example = pt_ds[0]
## Common keys: input_ids, attention_mask, labels, id, and label (if source label exists).
```

The adapter invokes a text tokenizer with truncation enabled, `max_length`, no per-example padding, and `return_tensors="pt"`. If `input_ids` exists but `labels` does not, it copies `input_ids` to `labels`, supporting causal language-model-style labels. This is a default convenience, not a universal objective: token classification, seq2seq, instruction-tuning masking, and packed training often require explicit label preparation.

Dynamic padding is applied later by `UniversalBatchCollator`. A tokenizer must support the callable signature expected by the adapter. Some tokenizers/processors require special kwargs or need a custom wrapper.

### 7.3 Image examples

Image-only samples use keys such as:

- `id`
- `pixel_values`: RGB float tensor shaped `[3, H, W]`
- `label`, if present
- `media_path`, if present

The built-in helper resizes to `image_size=(height, width)` and produces a channel-first tensor. Integer image data are scaled to the 0–1 range by the standard fallback; a custom processor may return normalized values outside that range, which are preserved by the custom output normalizer. Use the normalization expected by the model you intend to train.

```python
loader = dh.load(
    "./cats_and_dogs",
    modality="image",
    return_type="dataloader",
    batch_size=32,
    image_size=(224, 224),
)
```

A local image directory can assign a class name from the parent directory. Verify `label` values and mapping rather than assuming this is true for every layout.

### 7.4 Audio examples

Without a specialized processor, the adapter emits `waveform`. The media helper returns normalized audio shaped `[channels, time]` and resamples to the requested sample rate when it can access a supported resampling backend. The standard WAV path is intended for uncompressed PCM WAV; SciPy and/or torchaudio improve format coverage. Optional processor output is stored under `input_features`.

```python
audio_loader = dh.load(
    "./audio_manifest.csv",
    modality="audio",
    return_type="dataloader",
    audio_sr=16_000,
    batch_size=8,
)
```

Waveforms may have variable time lengths, so the collator pads them to a batch maximum. Processor-generated feature tensors are expected to have compatible shapes for stacking; if your processor returns varying feature sequence lengths, use an appropriate custom processor/collator or pre-pad consistently.

### 7.5 Video examples

The video helper samples up to a configured frame count, converts frames from BGR to RGB via OpenCV, resizes to the target `(height, width)`, normalizes pixel values to float32 0–1, and pads a short extraction by repeating the final frame. A successful decode returns `[T, 3, H, W]`; the batch collator stacks samples into `[B, T, 3, H, W]`.

```python
video_loader = dh.load(
    "./clips",
    modality="video",
    return_type="dataloader",
    max_video_frames=16,
    image_size=(224, 224),
    batch_size=4,
)
```

OpenCV must be installed and the target video codec must be readable in that runtime. Verify `max_video_frames` and video paths before training.

### 7.6 Multimodal vision-language examples

For `modality="multimodal"`, the current training adapter implements a vision-plus-language path. Expected keys can include:

- `pixel_values` for the image;
- `input_ids`, `attention_mask`, or processor-specific token outputs for text when a text tokenizer is available;
- `text` when a tokenizer is not selected;
- `label` when present.

The adapter attempts to separate visual processor and text tokenizer when the supplied processor exposes common properties such as `image_processor`, `feature_extractor`, or `tokenizer`. Model-specific processor objects differ; validate one item before making a full DataLoader.

```python
sample = {
    "image_path": "images/001.jpg",
    "caption": "A small dog standing on grass",
    "label": "dog",
}
ds = dh.load([sample], modality="multimodal", media_column="image_path")
```

This is a record-shape example, not a guarantee that an arbitrary vision-language model processor works unmodified. In particular, specialized models may require multiple images, image crops, chat templates, special tokens, or processor-specific nested data.

### 7.7 Collator behavior

`UniversalBatchCollator` validates and combines per-example dictionaries. Its main responsibilities are:

- dynamically pad `input_ids` to the longest sequence in the batch, using the configured/tokenizer `pad_token_id`;
- pad attention masks with zero;
- align/pad token-sequence labels with `-100` where appropriate (the conventional ignore index for many PyTorch losses);
- require compatible ranks and matching attention-mask/token shapes;
- stack image tensors into `[B, 3, H, W]` after validating consistent CHW RGB shapes;
- pad waveform sequences where supported;
- stack compatible `input_features` tensors;
- stack video frames into `[B, T, 3, H, W]` with uniform frame tensor shapes;
- collate homogeneous numeric labels into tensors and retain non-numeric labels as lists;
- surface invalid-sample metadata in supported permissive paths.

If any item provides a key that requires tensor collation, the collator validates the corresponding field across the batch. Missing `input_ids`, inconsistent image sizes, mixed scalar/sequence token labels, or a missing label among otherwise labelled rows should be fixed upstream or handled with a custom collator—not silently ignored.

### 7.8 Build a DataLoader manually

```python
from torch.utils.data import DataLoader
from data_hub import UniversalBatchCollator

pt_ds = ds.to_pytorch_dataset(
    tokenizer_or_processor=tokenizer,
    max_length=128,
)
collator = UniversalBatchCollator(
    modality=ds.modality,
    tokenizer_or_processor=tokenizer,
)
loader = DataLoader(
    pt_ds,
    batch_size=16,
    shuffle=not ds.is_streaming,
    num_workers=0,
    collate_fn=collator,
)
```

For most code, `ds.to_dataloader(...)` is simpler and less error-prone.

### 7.9 `num_workers` and streaming

`to_dataloader()` forwards standard PyTorch DataLoader options via keyword arguments, including `num_workers`, `pin_memory`, and `drop_last`, subject to normal PyTorch constraints. For streaming datasets, `shuffle` is disabled at the DataLoader layer; use a source that already shuffles or implement an appropriate buffering strategy if stochastic order is required.

The iterable adapter shards items across workers by record index modulo worker count. Each worker must still have access to a safe source iterator; test multi-worker behavior against the concrete source because some external iterable backends have their own sharding behavior.

### 7.10 Error policy during training

- `strict`: surface media/tokenization failures with sample context.
- `report_and_skip`: for supported finite media datasets, pre-check invalid items and skip them; adapter failures can mark items invalid for the collator when that path supports it.
- `permissive`: return invalid marker/error metadata when a transformation fails and a non-strict fallback is supported; the collator may drop invalid items if valid peers remain, and raises if the whole batch is invalid.

Never treat a permissive batch as proof that every input was valid. Monitor invalid sample metadata and compare expected batch counts. A report-and-skip prevalidation can incur extra media-decoding cost because it touches samples before iteration.

### 7.11 Training readiness checklist

Before a long run:

1. Inspect a few raw records and several adapted samples.
2. Confirm tensor names match the model's forward signature.
3. Confirm shapes, dtypes, image normalization and audio sample rate.
4. Verify labels are numeric and mapped to the expected class indices.
5. Check how `labels` are constructed; default copying of input IDs is not right for every objective.
6. Run a complete pass or enough steps to trigger lazy errors.
7. Log parsing errors, skipped media, invalid batches, and source sizes.

---

## 8. Media utilities

Media helpers are also exported at the package root for direct use. These are useful when you need to inspect a single media object outside the dataset pipeline.

### 8.1 Supported inputs

| Helper | Inputs accepted by the public type contract | Result |
|---|---|---|
| `load_image(source)` | path, `Path`, bytes, PIL image, dict containing `bytes`, `path`, or `image`, supported base64 image data URI | RGB PIL image |
| `image_to_tensor(image, processor_or_transform=None, target_size=(224, 224))` | same image inputs | float tensor, usually `[3, H, W]`, or processor output tensor shape |
| `image_to_numpy(image, target_size=None)` | same image inputs | uint8 RGB NumPy array `[H, W, 3]` |
| `load_audio(source, target_sr=16000)` | path, `Path`, bytes, audio dict with array/sample rate or bytes/path | `(waveform, sample_rate)`; waveform `[channels, time]` float tensor |
| `audio_to_tensor(source, processor=None, target_sr=16000)` | same audio inputs | waveform or processor-generated feature tensor |
| `load_video(source, max_frames=16, target_size=(224, 224))` | local path/`Path` | float tensor `[T, 3, H, W]` |

### 8.2 Images

#### `load_image`

Decodes an image and returns an RGB PIL image. Direct PIL inputs are converted to RGB; bytes are decoded from memory; a dictionary may wrap a byte payload under `bytes`, a path under `path`, or another image value under `image`. A string beginning with a valid `data:image...;base64,` data URI is decoded from base64. Local paths must exist and the content must be decodable.

```python
from data_hub import load_image

img = load_image("./assets/example.jpg")
print(img.mode, img.size)  # RGB, (width, height)
```

Pillow's `Image.size` is `(width, height)`, unlike the public resize arguments, which use `(height, width)`.

#### `image_to_tensor`

```python
from data_hub import image_to_tensor

pixels = image_to_tensor("./assets/example.jpg", target_size=(224, 320))
print(pixels.shape)  # torch.Size([3, 224, 320]) for the standard path
```

Default processing resizes and converts to a channel-first float tensor. If torchvision is installed, the implementation uses its resize/tensor transform; otherwise a NumPy/PyTorch fallback resizes through Pillow. The target-size convention is `(height, width)` in both paths.

With a processor object, the helper first attempts the image-processor style call `processor(images=pil_img, return_tensors="pt")` and uses `pixel_values` when available. For generic callables it passes the PIL image directly. Custom transforms may return a PyTorch tensor or NumPy array; shape, dtype and numeric ranges are validated/normalized. A model-normalized float transform can legitimately produce negative values. The library cannot infer whether those normalization statistics are correct for your model.

#### `image_to_numpy`

```python
array = image_to_numpy("./assets/example.jpg", target_size=(224, 320))
print(array.shape, array.dtype)  # (224, 320, 3), uint8
```

The array uses HWC order and RGB channels. Unlike the standard tensor fallback, it is not scaled to 0–1; it retains uint8 pixel values from 0–255.

### 8.3 Audio

#### `load_audio`

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

#### `audio_to_tensor`

Without a processor, this returns the waveform tensor (typically `[channels, time]`). With a callable processor, it passes the one-dimensional audio array, sample rate, and `return_tensors="pt"`. It maps `input_features` or `input_values` from a dictionary result, or accepts a tensor result. Processor output is squeezed along the leading batch dimension.

```python
features = audio_to_tensor(
    "./audio/example.wav",
    processor=my_audio_processor,
    target_sr=16_000,
)
```

The exact output semantics depend on the processor. Validate shape/dtype before passing the result to a model.

### 8.4 Video

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

### 8.5 Shape summary

| Representation | Standard shape | Order | Value convention |
|---|---|---|---|
| PIL image | `img.size == (W, H)` | Pillow image | PIL mode RGB |
| Image NumPy | `[H, W, 3]` | HWC | `uint8`, normally 0–255 |
| Image tensor | `[3, H, W]` | CHW | float; standard fallback 0–1 |
| Audio waveform | `[C, T]` | channels, time | float32, clamped to -1–1 |
| Single-video tensor | `[T, 3, H, W]` | TCHW | float32, 0–1 |
| Video batch | `[B, T, 3, H, W]` | BTCHW | float32, 0–1 |

### 8.6 Media failure handling

A helper normally raises on invalid/unsupported media. The dataset/PyTorch layer may wrap that error with the record ID or path and apply its configured policy. Don't catch every media exception and replace it with a plausible-looking zero tensor; that can introduce silent data corruption. If skipping is appropriate, record the error and compare the skipped count against the original inventory.

---

## 9. Public API reference

This reference lists names exported from `data_hub.__init__`. Internal helper names in `hub.py` and `parsers.py` are implementation details unless explicitly exported/imported from their submodule.

### 9.1 Package exports

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

### 9.2 `load()` and `DatasetHub`

`load(...)` is a convenience wrapper around a module-level default `DatasetHub`. If you need to choose a cache folder, instantiate the hub:

```python
from data_hub import DatasetHub

hub = DatasetHub(cache_dir="./cache/datasets")
ds = hub.load("hf://owner/dataset", split="train")
```

`DatasetHub(cache_dir=None)` creates the chosen cache directory (default: `./data_cache` under the current working directory). Its `load()` method accepts the same options as the top-level function.

See [Configuration reference](docs/configuration.md) for the complete argument table and precedence rules.

### 9.3 Enum models

#### `ModalityType`

Values: `TEXT`, `IMAGE`, `AUDIO`, `VIDEO`, `MULTIMODAL`, `JSON`, `TABULAR`. `ModalityType.from_str(value, default=None)` canonicalizes supported values and aliases; an unknown non-empty value raises `ValueError`.

#### `SourceType`

Values: `HUGGINGFACE`, `KAGGLE`, `LOCAL_FILE`, `LOCAL_DIR`, `MEMORY`. `SourceType.from_str(value, default=None)` canonicalizes source aliases; an unknown non-empty value raises `ValueError`.

#### `ReturnType`

Values: `DATASET`, `PYTORCH`, `NUMPY`, `RAW`, `DATALOADER`. `ReturnType.from_str(value, default=None)` canonicalizes return aliases; an unknown non-empty value raises `ValueError`.

#### `ErrorPolicy`

Values: `STRICT`, `REPORT_AND_SKIP`, `PERMISSIVE`. `ErrorPolicy.from_str(value)` accepts `strict`/`fail`, `report_and_skip`/`skip`, and `permissive`/`warn`; invalid values raise `ValueError`.

#### `DatasetItem`

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

#### `DatasetConfig`

A Pydantic model for source and processing configuration. See [Configuration reference](docs/configuration.md) for full defaults and validators. Construct with `path_or_identifier` and optionally explicit `source_type`, `modality`, and desired output settings.

#### `DatasetSummary`

A Pydantic summary model whose fields include `name`, `modality`, `source_type`, `return_type`, `error_policy`, `total_samples`, `is_streaming`, `columns`, `splits`, `sample_preview`, `estimated_tokens`, `token_estimation_method`, `class_distribution`, `memory_mb`, `skipped_samples`, `parsing_errors`, `warnings`, `warning`, `included_files`, `excluded_files`, and `metadata`.

### 9.4 `UniversalDataset`

The common normalized dataset container. Its constructor is public but ordinary users should normally create it through `load()`.

#### Inspection API

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

#### Transformation API

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

#### Conversion/output API

| Method | Returns | Notes |
|---|---|---|
| `to_pytorch_dataset(tokenizer_or_processor=None, max_length=128, image_size=(224,224), audio_sr=None, max_video_frames=None)` | PyTorch map/iterable dataset | Decodes/processes at sample access or iteration. |
| `to_dataloader(batch_size=8, shuffle=True, tokenizer_or_processor=None, collate_fn=None, num_workers=0, max_length=128, image_size=(224,224), audio_sr=None, max_video_frames=None, **kwargs)` | PyTorch DataLoader | Defaults to `UniversalBatchCollator`; forwards extra DataLoader options. |
| `to_texts(column=None)` | `list[str]` | Column, canonical text, or first string-valued raw field. |
| `to_numpy(columns=None)` | `dict[str, np.ndarray]` | Defaults to dataset columns; ragged values may become object arrays. |
| `to_pandas()` | Pandas DataFrame | Requires pandas. |
| `to_hf_dataset()` | Hugging Face Dataset | Requires `datasets` and its compatible dependencies. |
| `export(output_path, format="jsonl")` | output path string | Formats: JSONL, JSON, CSV, Parquet. |

### 9.5 `DatasetSplits`

Contains `train`, optional `val`, and optional `test` datasets. Supports string-key access, for example `splits["train"]`. A split with no rows is represented by `None` in the current implementation. Check this before using it:

```python
splits = ds.split(0.8, 0.1, 0.1)
if splits.val is not None:
    print(len(splits.val))
```

### 9.6 `UniversalBatchCollator`

Constructor:

```python
UniversalBatchCollator(
    modality=ModalityType.TEXT,
    pad_token_id=0,
    tokenizer_or_processor=None,
)
```

Callable contract: accepts `list[dict[str, Any]]` and returns one batch dictionary. It validates tensor requirements and shapes. The tokenizer's `pad_token_id`, when available and non-`None`, overrides the constructor's fallback padding ID.

### 9.7 PyTorch adapters

`PyTorchUniversalAdapter` is exported from the package root. `PyTorchIterableAdapter` is an implementation class in `data_hub.dataset` but is **not** included in the package root's `__all__` in version 0.1.2. Most users should call `to_pytorch_dataset()` or `to_dataloader()` instead.

- `PyTorchUniversalAdapter` wraps a finite list of `DatasetItem` records and implements map-style `__len__`/`__getitem__`.
- `PyTorchIterableAdapter` wraps a lazy source and implements `__iter__`; it shards rows across standard PyTorch DataLoader workers by a modulo index pattern.

The per-item dictionary schema depends on the modality and whether a tokenizer/processor was supplied. See [PyTorch training](docs/pytorch.md).

### 9.8 Media helper API

| Function | Purpose |
|---|---|
| `load_image(source)` | Decode and return RGB PIL image. |
| `image_to_tensor(image, processor_or_transform=None, target_size=(224, 224))` | Return image tensor or processor result, normally CHW. |
| `image_to_numpy(image, target_size=None)` | Return RGB HWC uint8 array. |
| `load_audio(source, target_sr=16000)` | Return `(waveform_tensor, sample_rate)`. |
| `audio_to_tensor(source, processor=None, target_sr=16000)` | Return waveform or processor features. |
| `load_video(source, max_frames=16, target_size=(224, 224))` | Decode evenly selected frames to TCHW float tensor. |

All public helper names are imported at `data_hub` package root. Use the shape, dtype, decoder, and optional-dependency notes in [Media utilities](docs/media-utilities.md).

### 9.9 Extension constants

- `IMAGE_EXTENSIONS`: recognized local image suffixes.
- `AUDIO_EXTENSIONS`: recognized local audio suffixes.
- `VIDEO_EXTENSIONS`: recognized local video suffixes.
- `TEXT_EXTENSIONS`: recognized local structured/text suffixes.

These sets are useful for path validation and profiling. An accepted suffix is not a guarantee of a successful decode.

---

## 10. Troubleshooting, diagnostics, and limitations

### 10.1 Local path is being treated as a hosted dataset

**Symptoms:** a string target triggers a Hugging Face connector error even though you intended a local dataset.

**Why:** an unrecognized bare string that is not an existing local path falls back to Hugging Face by design. A missing path with a recognized extension or path prefix raises `FileNotFoundError` instead.

**Fix:** verify the working directory and use an explicit path or source:

```python
ds = dh.load("./data/train.csv", source="local_file")
```

### 10.2 `FileNotFoundError` for referenced media

**Symptoms:** parsing succeeds but sample preview says a file is missing, or image/audio/video decoding raises a path error.

**Checks:**

1. Inspect `ds.preview(5, detailed=True)`.
2. Print `item.media_path` and confirm whether it is absolute or relative.
3. Verify the manifest's reference base directory and working directory.
4. If possible, normalize paths in a `map()` callback before requesting strict media decoding.
5. Ensure paths use a supported extension and actually reference file content—not only a browser URL that requires a download/authentication step.

### 10.3 Unexpected text or labels

**Symptoms:** a field is not selected, labels stay strings, or canonical `text` is missing.

**Fix:** explicitly pass `text_column`, `label_column`, and `media_column`. Common aliases are convenient heuristics, not a schema language. Use `label_dtype="str"`, `"int"`, or `"float"` to control label conversion. Inspect both `item.label` and `item.raw_data` to understand whether the value was absent or simply not promoted.

### 10.4 Hosted connector import errors

**Hugging Face:** install `universal-dataset-hub[hf]` (or `datasets`). Check repository visibility, split/config name, credentials and network access.

**Kaggle:** install `universal-dataset-hub[kaggle]`, configure required credentials, and ensure the dataset handle is valid.

**Parquet:** install `universal-dataset-hub[parquet]` / `pyarrow`.

**YAML:** install `universal-dataset-hub[yaml]` / PyYAML.

A test blocked by a missing optional package is not a test pass for live source behavior.

### 10.5 Audio decode/resample errors

Check that `target_sr` is a positive integer or `None` for `load_audio()`, and `audio_sr` is positive when using the hub/training adapter. WAV fallback is designed for PCM WAV; unsupported compressed formats require a compatible decoder. Resampling requires a backend able to perform it. Confirm the returned sample rate instead of assuming the requested rate was applied.

### 10.6 OpenCV cannot read video

`load_video()` requires OpenCV and a readable video stream. Verify:

- the file exists and is not empty;
- OpenCV can open that codec/container in the target runtime;
- `max_frames` is a positive integer;
- `target_size` is a two-item positive integer `(height, width)` pair.

Test the same file using the same environment/container that will run training; local desktop codecs and server codecs may differ.

### 10.7 Tensor shape mismatch in a batch

The default collator rejects inconsistent tensor shapes rather than silently stacking incompatible examples.

- Images must be `[3, H, W]` with consistent `H/W` across a batch.
- Videos must be `[T, 3, H, W]` and match in `T/H/W` across a batch.
- Attention masks must align with `input_ids` per sample.
- Token labels must be compatible with token input lengths; token labels can be padded with `-100`.
- Audio waveform sequences can be padded to the batch maximum; processor features typically need compatible shapes.

Use uniform `image_size`, `max_video_frames`, and processor settings. If a model needs a special layout, supply a custom processor/collator and test its output directly.

### 10.8 Token labels are not right for the loss function

The text adapter copies `input_ids` to `labels` if the tokenizer returns input IDs without a labels field. This is a generic language-modeling default; it is not automatically correct for classification, token classification, seq2seq, or every instruction-tuning setup. Create labels explicitly before training if needed.

### 10.9 Missing/unknown stream length

A lazy stream can have unknown or pre-consumption length. `num_samples=None` means unknown, not zero. Row parsing/media errors may only be detected during iteration. Consume the stream in a controlled pass to collect final counts. Do not iterate a one-shot generator for diagnostics and then expect to start it from row zero again.

### 10.10 Split fails for streaming or invalid ratios

`split()` is not supported on streaming datasets. Call `take(n)` to materialize a bounded prefix or deliberately materialize the whole dataset before splitting. Ratios must each be finite and between 0 and 1 and sum to 1.0; correct your ratios rather than expecting the library to discard/reallocate leftover rows.

Small datasets can create empty validation or test splits due to integer rounding. Inspect lengths after splitting.

### 10.11 Exported media are descriptors, not assets

`export()` serializes record fields. PIL images are represented as diagnostic text descriptors; arrays/tensors serialize to lists for JSON formats. It does not package raw image/audio/video binaries into a portable media directory. Export media assets separately and maintain correct paths/IDs if the resulting dataset must be portable.

### 10.12 Diagnostic checklist

When a dataset load or training step behaves unexpectedly, capture:

```python
print("version:", dh.__version__)
print("type:", type(ds).__name__)
print("modality:", ds.modality)
print("source:", ds.source_type)
print("streaming:", ds.is_streaming)
print("num_samples:", ds.num_samples)
print("preview:", ds.preview(5, detailed=True))
summary = ds.get_summary()
print("summary:", summary.model_dump())
```

For tensor problems, test one item before batching:

```python
pt_ds = ds.to_pytorch_dataset(tokenizer_or_processor=processor)
example = pt_ds[0] if not ds.is_streaming else next(iter(pt_ds))
for key, value in example.items():
    print(key, getattr(value, "shape", None), getattr(value, "dtype", type(value).__name__))
```

For lazy media streams, catch failures at iteration and record the item ID/path. Avoid changing a failing file into fake valid data just to keep a training loop running.

### 10.13 Known limitations in version 0.1.2

- Package status is Alpha; the test suite is finite and does not prove absence of defects.
- Remote Hugging Face/Kaggle integration was not considered live-verified merely because local unit tests pass.
- Optional dependency presence, codec availability, hosted credentials, custom source schema and network access affect behavior.
- Some source-specific or model-specific schemas need explicit column mapping/transforms.
- Streaming inference and random access have inherent tradeoffs for one-shot sources.
- DataFrame/Hugging Face conversion and export may need normalization for nested or extension-backed columns.
- The source bundle did not include an authoritative upstream license; determine the correct license before publishing the library.

---

## 11. Runnable examples

The repository contains executable scripts under [`docs/examples/`](docs/examples/README.md):

- [`01_quickstart.py`](docs/examples/01_quickstart.py) — in-memory loading, inspection, mapping, label encoding, splitting, and JSONL export.
- [`02_csv_manifest.py`](docs/examples/02_csv_manifest.py) — CSV parsing with explicit source-column mapping.
- [`03_media_helpers.py`](docs/examples/03_media_helpers.py) — RGB image loading, tensor conversion, and NumPy conversion.

Run from the repository root:

```bash
python -m pip install -e ".[test]"
python docs/examples/01_quickstart.py
python docs/examples/02_csv_manifest.py
python docs/examples/03_media_helpers.py
```

The scripts use temporary data and do not require remote credentials. The first two exercise the core dataset API; the media example generates a small image locally.
