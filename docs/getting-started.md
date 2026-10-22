# 2. Getting started

## 2.1 Requirements

- Python **3.10 or newer**.
- Core runtime dependencies: NumPy, PyTorch, Pydantic v2, and Pillow.
- Optional adapters require additional packages; see [Sources and formats](sources-and-formats.md).

## 2.2 Install from a source checkout

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

## 2.3 Basic in-memory text dataset

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

## 2.4 Load a local CSV

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

## 2.5 Load an image directory

```python
images = dh.load("./images", modality="image")
print(images.preview(3, detailed=True))
```

Directory scans recognize image extensions and may assign the parent directory's name as the label. The scanner can optionally validate image decoding depending on the error policy. A directory containing a recognized manifest such as `train.csv` may be parsed as a table instead of a pure image folder when the modality is not explicitly fixed to image.

## 2.6 Get a PyTorch DataLoader

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

## 2.7 Load options through a configuration object

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

An explicitly supplied keyword argument to `load(cfg, ...)` overrides the corresponding value on the copied configuration. The original configuration object is not mutated. See [Configuration reference](configuration.md).

## 2.8 Your first data audit

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
