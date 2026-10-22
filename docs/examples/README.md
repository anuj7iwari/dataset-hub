# Universal Dataset Hub - Example Index

This directory contains standalone, self-contained, and fully executable examples showcasing how to mount any kind of data using **Universal Dataset Hub** (`data_hub`) and convert it into every supported return format.

All example files adhere strictly to the standardized structure:
1. **Imports**
2. **Library Initialization** (Instance-based `DatasetHub` and Functional `dh.load`)
3. **Configurations** (Every configuration option offered by the library for that modality)
4. **Returns** (Every return format offered by the library: `UniversalDataset`, `NumPy ndarrays`, `PyTorch Tensors / Adapter`, `PyTorch DataLoader`, `Raw Dictionaries`, and fluent exports)

---

## Modality Examples

| File | Modality / Format | Key Configurations Demonstrated | Returns Demonstrated |
|---|---|---|---|
| [`text_example.py`](text_example.py) | **Text / NLP** | In-memory, CSV, `.txt`/`.md` files, `streaming=True`, `prompt_template`, `max_length`, `label_dtype`, `DatasetConfig`, `ErrorPolicy` | `dataset`, `numpy`, `pytorch` (with/without tokenizer), `dataloader`, `raw`, `.to_texts()`, `.to_pandas()`, `.export("jsonl")` |
| [`img_example.py`](img_example.py) | **Image / Vision** | Single image, local folder scans with subfolder classes, CSV manifests, `image_size=(H, W)`, `error_policy="report_and_skip"`, in-memory PIL images | `dataset`, `numpy` (RGB uint8), `pytorch` (`pixel_values` `[3, H, W]`), `dataloader` (`[B, 3, H, W]`), `raw`, `load_image`, `image_to_tensor` |
| [`audio_example.py`](audio_example.py) | **Audio / Speech** | Single WAV, directory scans with subfolder classes, speech manifest with transcripts, `audio_sr=16000`, in-memory audio dicts, `ErrorPolicy` | `dataset`, `numpy`, `pytorch` (`waveform` `[C, T]`), `dataloader` (`[B, C, max_T]`), `raw`, `load_audio`, `audio_to_tensor` |
| [`video_example.py`](video_example.py) | **Video** | Single video, directory scans, manifests, `max_video_frames=8`, frame resize `image_size=(H, W)`, error skipping | `dataset`, `numpy`, `pytorch` (`video_frames` `[T, 3, H, W]`), `dataloader` (`[B, T, 3, H, W]`), `raw`, `load_video` |
| [`csv_example.py`](csv_example.py) | **CSV / Tabular / Manifest** | Text classification CSV, tabular feature CSV, multimodal manifest CSV, `label_dtype="auto"\|"str"\|"int"`, `streaming=True`, column mapping | `dataset`, `numpy`, `pytorch`, `dataloader`, `raw`, `.to_pandas()`, `.to_hf_dataset()`, `.export("csv")` |
| [`json_example.py`](json_example.py) | **JSON / JSONL / Chat** | Top-level JSON array, line-delimited JSONL, multi-turn chat `messages`, `streaming=True`, prompt templates, skipping invalid lines | `dataset`, `numpy`, `pytorch`, `dataloader`, `raw`, `.to_pandas()`, `.export("jsonl")`, `.export("json")` |

---

## Starter / Quickstart Patterns

- [`01_quickstart.py`](01_quickstart.py): Basic text loading, inspection, mapping, splitting, and export.
- [`02_csv_manifest.py`](02_csv_manifest.py): CSV manifest loading with explicit column mapping.
- [`03_media_helpers.py`](03_media_helpers.py): Standalone image media helper functions.

---

## Running the Examples

All examples generate their own mock data in isolated temporary directories and clean up automatically. You can execute any script directly from the repository root:

```bash
python docs/examples/text_example.py
python docs/examples/img_example.py
python docs/examples/audio_example.py
python docs/examples/video_example.py
python docs/examples/csv_example.py
python docs/examples/json_example.py
```
