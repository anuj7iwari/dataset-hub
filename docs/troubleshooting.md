# 10. Troubleshooting, diagnostics, and limitations

## 10.1 Local path is being treated as a hosted dataset

**Symptoms:** a string target triggers a Hugging Face connector error even though you intended a local dataset.

**Why:** an unrecognized bare string that is not an existing local path falls back to Hugging Face by design. A missing path with a recognized extension or path prefix raises `FileNotFoundError` instead.

**Fix:** verify the working directory and use an explicit path or source:

```python
ds = dh.load("./data/train.csv", source="local_file")
```

## 10.2 `FileNotFoundError` for referenced media

**Symptoms:** parsing succeeds but sample preview says a file is missing, or image/audio/video decoding raises a path error.

**Checks:**

1. Inspect `ds.preview(5, detailed=True)`.
2. Print `item.media_path` and confirm whether it is absolute or relative.
3. Verify the manifest's reference base directory and working directory.
4. If possible, normalize paths in a `map()` callback before requesting strict media decoding.
5. Ensure paths use a supported extension and actually reference file content—not only a browser URL that requires a download/authentication step.

## 10.3 Unexpected text or labels

**Symptoms:** a field is not selected, labels stay strings, or canonical `text` is missing.

**Fix:** explicitly pass `text_column`, `label_column`, and `media_column`. Common aliases are convenient heuristics, not a schema language. Use `label_dtype="str"`, `"int"`, or `"float"` to control label conversion. Inspect both `item.label` and `item.raw_data` to understand whether the value was absent or simply not promoted.

## 10.4 Hosted connector import errors

**Hugging Face:** install `universal-dataset-hub[hf]` (or `datasets`). Check repository visibility, split/config name, credentials and network access.

**Kaggle:** install `universal-dataset-hub[kaggle]`, configure required credentials, and ensure the dataset handle is valid.

**Parquet:** install `universal-dataset-hub[parquet]` / `pyarrow`.

**YAML:** install `universal-dataset-hub[yaml]` / PyYAML.

A test blocked by a missing optional package is not a test pass for live source behavior.

## 10.5 Audio decode/resample errors

Check that `target_sr` is a positive integer or `None` for `load_audio()`, and `audio_sr` is positive when using the hub/training adapter. WAV fallback is designed for PCM WAV; unsupported compressed formats require a compatible decoder. Resampling requires a backend able to perform it. Confirm the returned sample rate instead of assuming the requested rate was applied.

## 10.6 OpenCV cannot read video

`load_video()` requires OpenCV and a readable video stream. Verify:

- the file exists and is not empty;
- OpenCV can open that codec/container in the target runtime;
- `max_frames` is a positive integer;
- `target_size` is a two-item positive integer `(height, width)` pair.

Test the same file using the same environment/container that will run training; local desktop codecs and server codecs may differ.

## 10.7 Tensor shape mismatch in a batch

The default collator rejects inconsistent tensor shapes rather than silently stacking incompatible examples.

- Images must be `[3, H, W]` with consistent `H/W` across a batch.
- Videos must be `[T, 3, H, W]` and match in `T/H/W` across a batch.
- Attention masks must align with `input_ids` per sample.
- Token labels must be compatible with token input lengths; token labels can be padded with `-100`.
- Audio waveform sequences can be padded to the batch maximum; processor features typically need compatible shapes.

Use uniform `image_size`, `max_video_frames`, and processor settings. If a model needs a special layout, supply a custom processor/collator and test its output directly.

## 10.8 Token labels are not right for the loss function

The text adapter copies `input_ids` to `labels` if the tokenizer returns input IDs without a labels field. This is a generic language-modeling default; it is not automatically correct for classification, token classification, seq2seq, or every instruction-tuning setup. Create labels explicitly before training if needed.

## 10.9 Missing/unknown stream length

A lazy stream can have unknown or pre-consumption length. `num_samples=None` means unknown, not zero. Row parsing/media errors may only be detected during iteration. Consume the stream in a controlled pass to collect final counts. Do not iterate a one-shot generator for diagnostics and then expect to start it from row zero again.

## 10.10 Split fails for streaming or invalid ratios

`split()` is not supported on streaming datasets. Call `take(n)` to materialize a bounded prefix or deliberately materialize the whole dataset before splitting. Ratios must each be finite and between 0 and 1 and sum to 1.0; correct your ratios rather than expecting the library to discard/reallocate leftover rows.

Small datasets can create empty validation or test splits due to integer rounding. Inspect lengths after splitting.

## 10.11 Exported media are descriptors, not assets

`export()` serializes record fields. PIL images are represented as diagnostic text descriptors; arrays/tensors serialize to lists for JSON formats. It does not package raw image/audio/video binaries into a portable media directory. Export media assets separately and maintain correct paths/IDs if the resulting dataset must be portable.

## 10.12 Diagnostic checklist

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

## 10.13 Known limitations in version 0.1.2

- Package status is Alpha; the test suite is finite and does not prove absence of defects.
- Remote Hugging Face/Kaggle integration was not considered live-verified merely because local unit tests pass.
- Optional dependency presence, codec availability, hosted credentials, custom source schema and network access affect behavior.
- Some source-specific or model-specific schemas need explicit column mapping/transforms.
- Streaming inference and random access have inherent tradeoffs for one-shot sources.
- DataFrame/Hugging Face conversion and export may need normalization for nested or extension-backed columns.
- The source bundle did not include an authoritative upstream license; determine the correct license before publishing the library.
