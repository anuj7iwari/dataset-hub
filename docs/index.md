# Universal Dataset Hub — Documentation

**Documentation for Universal Dataset Hub 0.1.2**

Universal Dataset Hub (`data_hub`) is a Python ingestion, normalization, transformation, and PyTorch-adaptation layer for datasets represented as local files, local directory trees, in-memory Python objects, Hugging Face datasets, or Kaggle datasets. It standardizes records into `DatasetItem` objects and exposes one `UniversalDataset` API across text, image, audio, video, tabular, structured, and multimodal workflows.

> **Version and verification note:** These pages describe the repaired 0.1.2 source in this repository. The package is under active development (Alpha). Optional hosted integrations must be tested with their dependencies, credentials, and network access in the target environment. The documentation does not imply that every external integration has been live-verified.

## Documentation map

1. [Overview and architecture](00-overview.md) — what the library does and how a load request moves through the system.
2. [Getting started](getting-started.md) — installation and your first dataset.
3. [Concepts and data model](concepts.md) — modalities, sources, normalized items, metadata, and streaming.
4. [Configuration reference](configuration.md) — `DatasetConfig`, defaults, accepted values, aliases, and precedence.
5. [Sources and file formats](sources-and-formats.md) — local files, directory layouts, memory records, Hugging Face, and Kaggle.
6. [Dataset operations](dataset-operations.md) — inspect, filter, map, clean, format, sample, split, convert, and export.
7. [PyTorch training](pytorch.md) — datasets, iterable datasets, DataLoaders, collators, tensors, and error policies.
8. [Media utilities](media-utilities.md) — image, audio, and video decoding/preprocessing.
9. [Public API reference](api-reference.md) — exported names and method contracts.
10. [Troubleshooting and limits](troubleshooting.md) — common errors, diagnostics, and current constraints.
11. [Runnable examples](examples/) — small examples that can be adapted for real projects.

## Quick start

```python
import data_hub as dh

records = [
    {"text": "The service was excellent", "label": 1},
    {"text": "The package arrived damaged", "label": 0},
]

dataset = dh.load(records, modality="text")
print(dataset)
print(dataset.preview(2))
print(dataset.get_summary())

splits = dataset.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
print(len(splits.train))
print(len(splits.val) if splits.val else 0)
print(len(splits.test) if splits.test else 0)
```

For tiny datasets, the default split ratios may produce empty validation/test splits because counts are rounded down. Choose ratios appropriate for your dataset size and inspect each resulting split. The example above illustrates loading and inspection; it does not imply a useful three-way split for only two records.

## Design goals

- **One normalized record shape:** text, labels, media references, metadata, original source fields, and features have predictable homes.
- **One ingestion entry point:** `data_hub.load(...)` handles source selection and can return a normalized dataset or a downstream representation.
- **Explicit failure semantics:** callers choose strict failure, skip-with-error-recording, or permissive best-effort handling.
- **Useful eager and lazy modes:** finite data can be materialized for random access; compatible sources can be consumed as streams.
- **Training integration:** PyTorch adapters and a batch collator handle common text, vision, audio, video, and multimodal tensor shapes.
- **Inspectable transformation steps:** preview and summary metadata help debug schemas, skipped samples, and parse warnings.

## Important caveats

- A normalized item is not the same thing as a fully decoded media tensor. Ingestion may keep a `media_path` or preserve an embedded object in `raw_data`; decoding often happens later in a media helper or PyTorch adapter.
- Automatic source/modality inference is convenience behavior, not a replacement for an explicit schema. Specify `source`, `modality`, and column names when the source is ambiguous.
- A stream may be one-shot. Operations that require random access—such as random splitting—need materialized data.
- Pandas, Hugging Face `datasets`, KaggleHub, Parquet support, YAML support, SciPy audio decoding/resampling, and OpenCV video decoding are optional features with separate dependencies.
- The repository did not provide an authoritative upstream license when this build was repaired. Check licensing before redistribution.
