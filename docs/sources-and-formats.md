# 5. Sources and file formats

## 5.1 Source matrix

| Source | Example target | Typical output before finalization | Requirements / caveats |
|---|---|---|---|
| In-memory records | `[{"text": "hi", "label": 1}]` | Eager `DatasetItem` collection by default; stream behavior may be requested for iterable sources | Lists, tuples, dictionaries, DataFrame-like inputs, and iterables are supported by memory normalization. Generator modality inference is necessarily limited. |
| Local structured file | `./data/train.csv` | Parsed records or lazy stream, depending on format/options | Supported extensions are listed below. |
| Local directory | `./data/images/` | Manifest parsing, media-folder scan, or text-document bundle | Directory profiling can select a manifest and can warn about mixed modalities. |
| Hugging Face | `hf://owner/dataset` or `owner/dataset` | Dataset split converted/streamed into normalized rows | Requires `datasets`; remote access, dataset config and split availability depend on the repository. |
| Kaggle | `kaggle://owner/dataset` | Downloaded/cached files then parser routing | Requires `kagglehub` plus any required credentials and network/service access. |

## 5.2 Local file support

### Structured/text extensions

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

### Media extensions

Image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`.

Audio extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`.

Video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

Extension recognition does not mean that every codec, image encoding, or container is supported by every installed backend. A file can have an accepted extension and still fail to decode.

## 5.3 CSV and TSV

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

## 5.4 JSON and JSONL

JSON supports a top-level array, object, or scalar at parse time. A top-level array is a sequence of records; a top-level mapping/scalar is treated as one record. JSONL/NDJSON treats each non-empty line as an independent JSON value, normally a mapping record.

Example JSONL:

```jsonl
{"id": "a", "text": "good", "label": 1}
{"id": "b", "text": "bad", "label": 0}
```

For streaming JSON arrays, the parser reads elements incrementally rather than loading the full array as a single Python list. This still requires buffering the current element. A single very large element can therefore consume substantial memory.

Malformed JSONL lines are handled by the active error policy. Under skip-oriented handling, a malformed line may be omitted and recorded. Review `parsing_errors` rather than assuming a successful stream means all lines were valid.

## 5.5 YAML

YAML support is optional and uses PyYAML. A top-level sequence can be constructed element-by-element in streaming mode; a mapping/scalar is constructed as one document/record. As with JSON, a large individual object must exist in memory while it is parsed. Never load YAML from untrusted sources without considering PyYAML's security and resource implications and the behavior of the installed parser/version.

## 5.6 Plain text and Markdown

`.txt` and `.md` files are read as text-like records. When a directory is interpreted as a text collection, compatible text/Markdown files are recursively collected; empty content is not made into a regular text sample. The filename and character-count details may be recorded in metadata.

## 5.7 Parquet

Parquet is intended for tabular rows. Install the declared extra:

```bash
python -m pip install 'universal-dataset-hub[parquet]'
```

In development from a checkout:

```bash
python -m pip install -e '.[parquet]'
```

Parquet reading/export should be tested against the specific pandas/pyarrow versions and schema. Nested/extension columns may require manual normalization.

## 5.8 Directory profiling and manifests

Directory input is more than a recursive `glob`: the parser first profiles files and checks recognized manifest names such as:

- `train.csv`, `data.csv`, `metadata.csv`, `labels.csv`, `dataset.csv`
- `train.jsonl`, `data.jsonl`

When an eligible manifest is present, the parser selects a manifest based on the configured `split` name where possible. If multiple manifests exist, it may choose a matching one or fall back to the first and emits a warning. If the directory contains both a manifest and media/text references, the manifest path can represent a multimodal dataset.

If no manifest is selected, a directory with only images/audio/video can be scanned as a media folder. A parent directory name may be assigned as the label (e.g. `cats/001.jpg` → label `cats` when the media is under a class subdirectory). Mixed-modality folders are not automatically fused into paired examples just because files share a directory; an explicit manifest is the safer pairing strategy.

Directory profiling scans are bounded by `max_scan` (default 500 entries in the profile helper). For very large directory trees, inferred counts can be lower bounds; inspect profile metadata and do not treat a bounded profile as a complete inventory.

## 5.9 In-memory records

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

## 5.10 Hugging Face Hub

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

## 5.11 Kaggle

Example:

```python
ds = dh.load("kaggle://owner/dataset", source="kaggle", modality="text")
```

Install `kagglehub` and configure the credentials required for the target account/dataset. The hub passes a cache directory to the Kaggle adapter; a `DatasetHub(cache_dir=...)` instance allows you to choose a cache location. Some Kaggle downloads contain multiple manifests, labels, and assets. Inspect the downloaded directory and parser metadata to verify which manifest was selected.

## 5.12 Remote URLs versus media URLs

The row normalizer can retain URL-like values as a media reference. This does not mean every URL is automatically downloaded, authenticated, or made available to a decoder. A media helper may accept a path or a supported direct byte/dict payload, but a remote URL may require additional download logic before use depending on the backend. Test a representative URL in your environment and handle authentication/timeouts separately.
