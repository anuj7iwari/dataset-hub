# 6. Dataset operations

The main object is `UniversalDataset`, normally returned from `data_hub.load()` with its default `return_type="dataset"`. Methods generally return a new `UniversalDataset` instead of altering the original collection. Prefer assigning each result to a new variable to keep transformation order visible.

## 6.1 Inspection and indexing

### `preview(n=5, detailed=False)`

Returns a list of JSON-friendly preview dictionaries. It includes `id`, canonical `modality`, and `label`; text samples include a truncated snippet and character length; media references include the path and a file-existence check when applicable. With `detailed=True`, available `metadata` and the list of `raw_data` keys are included.

```python
for row in ds.preview(5, detailed=True):
    print(row)
```

`n` must be a non-negative integer. A streaming preview uses the lazy sequence's bounded `peek()` method; peeking is designed to preserve the consumed prefix for later iteration.

### `get_summary()` / `summary()`

Returns a `DatasetSummary` model. Useful fields include:

- `total_samples` and `is_streaming`;
- `columns`, `modality`, and `source_type`;
- `estimated_tokens` and `token_estimation_method` for text/multimodal data;
- `class_distribution` where available;
- `skipped_samples`, `parsing_errors`, and `warnings`;
- `included_files` and `excluded_files` for directory scans.

Token count uses a rough character-based heuristic (about 3.2 characters/token) on a bounded sample; it is not a tokenizer-exact count. Memory use is also an estimate rather than an allocator-accurate measurement. Treat both as diagnostics, not capacity guarantees.

### `len()`, `num_samples`, indexing, iteration

```python
print(len(ds))
print(ds.num_samples)
first = ds[0]
for item in ds:
    print(item.id, item.text, item.label)
```

For eager collections, indexing and slicing operate over in-memory items. `num_samples` can be `None` when a streaming source has unknown cardinality. Do not assume an unbounded or one-shot stream can be measured without consuming it.

## 6.2 `take(n)`

Returns the first `n` samples as a materialized, non-streaming dataset. In streaming mode it consumes at most `n` rows and returns those rows in memory. `take(0)` is explicitly designed to consume no source rows.

```python
sample = stream.take(100)
print(sample.num_samples)
```

`n` must be a non-negative integer and booleans are rejected despite being Python integers.

## 6.3 `filter(condition)`

Keeps rows for which a predicate returns truthy. The predicate receives a deep-copied `DatasetItem`, so mutating the object inside the predicate should not mutate the source dataset.

```python
long_reviews = ds.filter(
    lambda item: item.text is not None and len(item.text) >= 100
)
```

In eager mode the filter is applied immediately. In streaming mode it returns another lazy dataset and the predicate is evaluated while the output is consumed. Ensure the predicate is deterministic and does not depend on side effects if repeatability matters.

## 6.4 `map(fn)`

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

## 6.5 `sort(by=None, descending=False, key=None)`

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

## 6.6 `clean(...)`

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

## 6.7 `encode_labels(mapping=None)`

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

## 6.8 `format_prompt(template=None, column_mapping=None, safe=True)`

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

## 6.9 `sample(n=None, frac=None, seed=42)`

Randomly selects a subset from an eager collection. `n` selects a count; `frac` selects a fraction. `seed` makes the sampled order reproducible with Python's random generator. Streaming datasets return an error for operations needing full random access or have to be materialized first.

```python
subset = ds.sample(n=500, seed=123)
quarter = ds.sample(frac=0.25, seed=123)
```

Do not supply both `n` and `frac` unless the specific implementation behavior has been confirmed for the version you use. Ensure `n`/`frac` matches the dataset size and desired semantics.

## 6.10 `split(...)`

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

## 6.11 Extract strings and columns

### `to_texts(column=None)`

Returns a list of strings. If `column` is supplied and exists in `raw_data`, that column is converted to string. Otherwise canonical text is preferred; if it is missing, the method uses the first string-valued field found in `raw_data`.

```python
texts = ds.to_texts()
labels_as_text = ds.to_texts(column="label_name")
```

The fallback is convenient but not schema-aware; for training, use explicit columns and validate the produced list.

### `to_numpy(columns=None)`

Returns a dictionary mapping column names to NumPy arrays. With no explicit columns, it uses `ds.columns`. Canonical `text`, `label`, and `media_path` fields take precedence, then other columns are read from `raw_data`.

```python
arrays = ds.to_numpy(columns=["text", "label"])
```

Ragged/heterogeneous values are retained in a one-dimensional object array if NumPy cannot build a regular array. Object arrays are not automatically ready for GPU training and may need explicit encoding/padding. Conversion is intended for materialized datasets: a stream of unknown length may need to be consumed/materialized first, and attempting to infer its length can raise an error.

## 6.12 Pandas and Hugging Face conversion

### `to_pandas()`

Builds a Pandas DataFrame from canonical records and raw columns. Requires `pandas` and materializes all yielded rows, so streaming input can be consumed and a large source can exhaust memory.

### `to_hf_dataset()`

Builds a Hugging Face `datasets.Dataset` from the Pandas conversion. Requires both `pandas` and `datasets` compatible with your environment. It materializes the data and is therefore not a bounded-memory streaming conversion.

```python
df = ds.to_pandas()
hf_ds = ds.to_hf_dataset()
```

Pandas/Hugging Face conversion may not preserve arbitrary custom Python objects in a directly trainable representation; normalize object/media columns first and verify resulting column types.

## 6.13 `export(output_path, format="jsonl")`

Supported formats: `jsonl`, `json`, `csv`, `parquet`. The output directory is created if needed. Invalid format names raise `ValueError`.

```python
ds.export("./out/processed.jsonl", format="jsonl")
ds.export("./out/processed.csv", format="csv")
```

Canonical `text`, `label`, and `media_path` fields take precedence over corresponding source fields. Tensor/NumPy data are converted to lists for JSON serialization; datetime-like objects use `isoformat`; PIL images are written as diagnostic string descriptors rather than their original pixel data. Therefore `export()` is record serialization—not a guarantee that embedded image/audio/video payloads are copied into portable assets. For reusable media datasets, export assets separately and store stable paths/IDs in the manifest.

Use Parquet only with the optional engine installed and test nested-column types. Treat an export as incomplete until you check that the target path exists, has nonzero size where expected, and can be re-read.
