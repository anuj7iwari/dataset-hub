# Universal Dataset Hub — repaired build 0.1.2

A Python library for normalizing local files, local directories, in-memory records, and optional hosted datasets into a unified interface for text, image, audio, video, tabular, and multimodal AI workflows.

This release fixes the 13 grouped defect areas documented in the Version 7 audit and adds regression tests. It is substantially tested, but finite tests do not prove the library bug-free. Live hosted-service integrations are not considered verified unless tested with their optional packages, credentials, and network access.

## Install

Requires Python 3.10 or later. From this directory:

```bash
python -m pip install .
```

Optional integrations:

```bash
python -m pip install '.[audio]'    # SciPy-based audio decoding/resampling
python -m pip install '.[video]'    # OpenCV video decoding
python -m pip install '.[hf]'       # Hugging Face datasets connector
python -m pip install '.[kaggle]'   # Kaggle download connector
python -m pip install '.[parquet]'  # Parquet support via pyarrow
python -m pip install '.[yaml]'     # YAML files (PyYAML)
python -m pip install '.[all]'      # all optional adapters above
python -m pip install '.[test]'     # test dependencies
```

## Detailed documentation

The full documentation is available in [`DOCUMENTATION.md`](DOCUMENTATION.md) and organized into individual guides under [`docs/`](docs/index.md). It covers architecture, schema/configuration, all supported source types, transformation APIs, PyTorch integration, media tensor shapes, errors and troubleshooting.

## Basic usage

```python
import data_hub as dh

train = dh.load([
    {"text": "This example is positive", "label": 1},
    {"text": "This example is negative", "label": 0},
])
print(train.preview(2))

local = dh.load("./data/train.csv", error_policy="strict")
for item in local:
    print(item.text, item.label)

stream = dh.load("./data/train.jsonl", streaming=True, error_policy="report_and_skip")
for item in stream:
    print(item.text, item.label)
```

## Error policies

- `strict` (default): raise a contextual error on malformed records, missing referenced local media, or unreadable supported media when validation is performed.
- `report_and_skip`: omit invalid rows and record `skipped_samples` and `parsing_errors` in dataset metadata.
- `permissive`: preserve rows where feasible and attach warning information rather than claiming invalid media was successfully validated.

For lazy streams, row-level validation takes place during iteration. Consume the stream to discover all per-row errors. An unterminated quoted CSV record cannot be safely resynchronized in general; `report_and_skip` records the parse failure and ends iteration at that boundary after preserving earlier valid rows.

## Streaming contract and limits

- JSON files whose top-level value is an array are decoded incrementally by element, retaining the input buffer and the current element rather than materializing the full array. JSON files with a top-level object/scalar are treated as a single record and must be parsed as that record.
- YAML files whose top-level value is a sequence are constructed element-by-element. A top-level YAML mapping/scalar is one record and is constructed as a whole.
- CSV/TSV streaming counts logical records, including quoted newlines. The length before consuming a source with a known size is an upper bound; after a complete pass, `num_samples` reflects valid yielded samples after parse/media skips.
- A generator-backed in-memory source is not consumed during `load()`. Inferred modality is therefore deferred: the first consumed row may promote the dataset's modality (for example, to `multimodal`). For training, specify `modality="multimodal"`, `"image"`, `"audio"`, etc. explicitly when the source schema is known. The PyTorch iterable adapter reads the live inferred modality after finalizing each item.
- `take(0)` consumes no source rows. Streaming generators are one-shot if their source is one-shot; do not expect a second full pass.
- Image batches use CHW RGB tensors and video batches use TCHW RGB tensors. The collator validates channel/rank contracts, per-sample attention-mask alignment, and token-label alignment.
- Dataset-derived eager collections own deep copies of their mutable `DatasetItem` instances and nested raw data. Filter predicates receive isolated copies.
- `split()` requires finite ratios in [0, 1] that sum to 1.0 within a small tolerance. Streaming datasets must be materialized before random splitting.

## Tests

Run the standard unit tests:

```bash
python -m unittest discover -s tests -v
pytest -q
```

Run the timed repair-specific audit suites:

```bash
python tests/audit/run_regression.py
python tests/audit/run_adversarial.py
python tests/audit/run_hardening.py
```

The audit console outputs and pre-repair report are stored under [`audit_results/`](audit_results/). External integrations are not considered verified when dependencies, credentials, or service access are unavailable; blocked is not a pass.

## Known limitations

- Live Hugging Face, Kaggle, and successful Parquet integrations require optional dependencies and the relevant credentials/network/service access. Their live behavior is unverified in an offline environment.
- Video codec support depends on OpenCV and codecs installed in the target runtime.
- Large single JSON elements and top-level YAML mappings necessarily occupy memory for the duration of that element/document, even when array/sequence streaming is enabled.
- Modality inference for an arbitrary one-shot generator cannot inspect future rows without consuming them. Specify modality explicitly if rows are heterogeneous or the first row may not represent the full schema.
- The upstream license was not supplied with the source files; determine the correct license before public redistribution.

## Revision 10 recovery build (0.1.2)

See [`REPAIR_REPORT.md`](REPAIR_REPORT.md) for the recovered behaviours, the exact repairs, and current test results.

Run the local tests from the source checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Optional source integrations can be installed with `python -m pip install -e ".[test,hf,kaggle,parquet]"`. Hugging Face/Kaggle access and credentials are needed to fully validate live remote integrations.
