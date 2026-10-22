# 1. Overview and architecture

## 1.1 What the library is

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

## 1.2 Architectural layers

The implementation is organized into six modules.

| Module | Responsibility | Important public objects |
|---|---|---|
| `schemas.py` | Canonical enum values and validation models | `ModalityType`, `SourceType`, `ReturnType`, `ErrorPolicy`, `DatasetItem`, `DatasetConfig`, `DatasetSummary` |
| `hub.py` | Source resolution, configuration, dispatch, return-type selection | `DatasetHub`, `load()` |
| `parsers.py` | Source-specific decoding and row normalization | `row_to_dataset_item()`, local parsers, memory and hosted adapters |
| `dataset.py` | Dataset container, transformations, summaries, exports, PyTorch adapters | `UniversalDataset`, `DatasetSplits`, `PyTorchUniversalAdapter`, `PyTorchIterableAdapter` |
| `media.py` | Standalone image/audio/video decoding and conversion | `load_image()`, `image_to_tensor()`, `load_audio()`, `audio_to_tensor()`, `load_video()` |
| `collators.py` | Batch-level validation, padding, and tensor stacking | `UniversalBatchCollator` |

## 1.3 Load lifecycle

A normal `load()` flow consists of these stages:

1. **Target classification.** A target may be a local path, repository identifier, `DatasetConfig`, list/tuple/dictionary of records, DataFrame-like object, or iterable.
2. **Source resolution.** URI prefixes such as `hf://` and `kaggle://` take precedence. Existing local paths are classified as files/directories. Explicit `src`/`source` arguments can force a source. Otherwise a non-path-like identifier is treated as a Hugging Face dataset identifier.
3. **Configuration validation.** Modality, return type, error policy, image size, label dtype, and positive-integer settings are normalized and validated through the schema models.
4. **Ingestion.** The hub dispatches to the local-file, local-directory, in-memory, Hugging Face, or Kaggle parser.
5. **Record normalization.** Source rows are converted to `DatasetItem` records. Canonical fields include `id`, `text`, `label`, and `media_path`; source columns are retained in `raw_data`; additional information is stored in `metadata`.
6. **Dataset construction.** The records, inferred columns, source metadata, warning/error records, modality, and effective configuration are assembled into `UniversalDataset`.
7. **Finalization.** An optional prompt template is applied. The hub then returns a `UniversalDataset`, PyTorch dataset adapter, NumPy column dictionary, raw records, or DataLoader according to `return_type`.

## 1.4 Typical call paths

### Eager records

`load(list/dict/DataFrame) → parse_memory → DatasetItem list → UniversalDataset → optional transformations/output adapter`

### Local structured file

`load(path) → source/path resolution → parse_local_file → DatasetItem list or lazy sequence → UniversalDataset → output adapter`

### Media directory

`load(directory) → directory profile → manifest detection or modality scan → normalized items → UniversalDataset`

### Model training

`UniversalDataset.to_dataloader() → PyTorchUniversalAdapter or PyTorchIterableAdapter → per-item media/tokenizer processing → UniversalBatchCollator → batch dictionary`

For `return_type="dataloader"`, steps are initiated by `load()` during finalization.

## 1.5 Eager versus streaming

**Eager** mode materializes data into an indexable collection. It is appropriate when the dataset fits in memory and you need slicing, reproducible random sampling, sorting, or train/validation/test splits.

**Streaming** mode retains a lazy sequence for supported file formats and iterable-backed sources. Rows may only be validated when consumed. An initial `len()` can represent an upper bound or be unknown, depending on the source; after iteration, metadata may be updated with the actual yielded/skipped counts.

Streaming tradeoffs:

- the source may be one-shot;
- random access and random splitting are not supported;
- error discovery is deferred until iteration;
- `take(0)` should not consume rows;
- generic generators cannot be exhaustively inspected to infer modality before reading them;
- a bounded peek used for training inference can consume/cache a prefix to keep those samples available, but it does not make arbitrary generators rewindable forever.

See [Concepts and data model](concepts.md) and [Sources and formats](sources-and-formats.md).

## 1.6 Error boundaries

The library has multiple validation layers. A parser can reject malformed source rows; media utilities can reject unsupported/corrupt content; a PyTorch adapter can wrap media errors with the sample ID/path; and the collator can reject inconsistent batch shapes or missing fields. Read [Troubleshooting](troubleshooting.md) for interpreting these distinct failure stages.
