# py-dataset-hub

`py-dataset-hub` (imported in Python as `data_hub`) is a Python library for loading, normalizing, and preprocessing datasets across multiple modalities into PyTorch Datasets, DataLoaders, and NumPy arrays.


Installation
------------

Install from PyPI:

```bash
pip install py-dataset-hub
```

Optional dependencies for specific modalities:

```bash
pip install "py-dataset-hub[audio]"    # Audio decoding and resampling (scipy)
pip install "py-dataset-hub[video]"    # Video frame extraction (opencv-python)
pip install "py-dataset-hub[hf]"       # Hugging Face datasets integration
pip install "py-dataset-hub[kaggle]"   # Kaggle dataset downloads
pip install "py-dataset-hub[parquet]"  # Parquet support (pyarrow)
pip install "py-dataset-hub[yaml]"     # YAML support (PyYAML)
pip install "py-dataset-hub[all]"      # All optional dependencies
```

To verify installation:

```bash
python -c "import data_hub as dh; print(dh.__version__)"
```

To install from source:

```bash
git clone https://github.com/anuj7iwari/dataset-hub.git
cd dataset-hub
pip install .
```


Features
--------

- Multi-modal: Unified `load()` interface for text, image, audio, video, tabular (CSV, TSV, Parquet), JSON, and YAML data.
- Framework integration: Direct export to PyTorch `Dataset` and `DataLoader` with automatic batch collation, or conversion to NumPy arrays.
- Streaming: Memory-efficient streaming for large files (JSONL, CSV, generators) without full materialization.
- Pipelines: Built-in `map`, `filter`, `batch`, `shuffle`, and `split` operations.
- Remote sources: Integration with Hugging Face Hub and Kaggle datasets.


Usage
-----

Import the package as `data_hub`:

```python
import data_hub as dh

# Load data from memory, files, or directories
ds = dh.load([
    {"text": "Sample text one", "label": 1},
    {"text": "Sample text two", "label": 0},
])

# Preview samples
print(ds.preview(2))

# Convert to a PyTorch DataLoader
loader = ds.to_dataloader(batch_size=2, shuffle=True)

# Convert to NumPy arrays
arrays = ds.to_numpy()
```


Documentation
-------------

Detailed guides and API references are available in `DOCUMENTATION.md` and the `docs/` directory:

- [API Reference](docs/api-reference.md)
- [Sources and Formats](docs/sources-and-formats.md)
- [PyTorch Integration](docs/pytorch.md)
- [Media Utilities](docs/media-utilities.md)
- [Examples](docs/examples/README.md)


License
-------

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
