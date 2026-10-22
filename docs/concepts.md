# 3. Concepts and data model

## 3.1 Modality

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

## 3.2 Source type

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

## 3.3 Return type

`ReturnType` controls the object returned from `load()` after ingestion/finalization.

| Value | Returned representation | Notes |
|---|---|---|
| `dataset` | `UniversalDataset` | Default; supports inspection and transformations |
| `pytorch` | PyTorch `Dataset` or `IterableDataset` adapter | Per-sample token/media processing occurs at access/iteration time |
| `numpy` | `dict[str, numpy.ndarray]` | One array per selected/inferred dataset column; heterogeneous/ragged values can use object arrays |
| `raw` | `list` of raw dictionaries or model dumps | If `raw_data` is truthy, that source dictionary is returned; otherwise the normalized model dump is used |
| `dataloader` | `torch.utils.data.DataLoader` | Uses a modality-aware default collator unless one is supplied through loader kwargs |

Aliases include `universal`, `pt`, `torch`, `tensor`, `tensors`, `np`, `ndarray`, `dict`, `pil`, and `loader`. The aliases are string conveniences; they do not change the output contract.

## 3.4 The normalized record: `DatasetItem`

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

### Canonical fields versus raw data

`raw_data` is important when a downstream export needs source-specific columns. The library uses canonical fields such as `text`, `label`, and `media_path` as authoritative values when it builds a canonical export record. Source-specific fields not promoted to canonical fields remain in `raw_data`.

A `media_path` generally represents a reference to media. An embedded source object—such as Hugging Face audio stored as `{"array": ..., "sampling_rate": ...}`—may remain inside `raw_data` and need not populate `media_path`.

## 3.5 Dataset container: `UniversalDataset`

`UniversalDataset` wraps normalized items together with:

- `modality`, `source_type`, and `return_type`;
- `error_policy`;
- a human-readable dataset `name`;
- known `columns`;
- parser/source `metadata`;
- audio sample-rate and maximum video-frame settings;
- the effective `DatasetConfig`, where available.

It supports ordinary iteration and index/slice access in eager mode, plus inspection, transformation, conversion, export, and model-framework adapters. Streaming mode intentionally has fewer random-access guarantees.

## 3.6 Summary metadata

`DatasetSummary` is a Pydantic model for high-level reporting. It includes name, modality/source/return type, error policy, sample count, streaming status, columns, split counts, preview records, token estimate and its method, class distribution, memory estimate, skipped count, parsing error records, warnings, included/excluded file lists, and metadata.

Not every field is available for every source or mode. For example, `total_samples=None` represents an unknown count and should not be treated as zero. Token counts and memory are estimates, not a resource guarantee.

## 3.7 Error policy

`ErrorPolicy` has three canonical values:

- `strict`: fail promptly on errors that the current ingestion/processing stage detects. This is the safest default for training pipelines.
- `report_and_skip`: omit invalid records where the parser/adaptor supports skipping and preserve diagnostics in metadata. For streams, the error can only be observed when the row is consumed.
- `permissive`: prioritize best-effort ingestion and preserve warning/error information where possible. This is not a promise that every corrupt value can be repaired or that downstream training will succeed.

Aliases: `fail` → strict, `skip` → report-and-skip, `warn` → permissive.

Error policy behavior depends on the specific source and stage. Review `summary.skipped_samples`, `summary.parsing_errors`, and `summary.warnings` rather than relying solely on the absence of an exception.

## 3.8 Other important terms

- **Manifest:** A structured file in a directory (commonly CSV or JSONL) that lists examples and labels/media references.
- **Collator:** A batch-level function that combines per-sample dictionaries into padded/stacked tensors.
- **Processor/tokenizer:** A user-supplied callable, often from a model library, that turns text or media into model-specific tensors.
- **Streaming sequence:** A lazy, often one-pass iterator over normalized items.
- **Eager collection:** An in-memory collection that can be indexed/sliced and randomly sampled/split.
