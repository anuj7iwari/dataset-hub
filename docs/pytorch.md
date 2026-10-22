# 7. PyTorch training integration

## 7.1 Options

There are three related layers:

1. `UniversalDataset.to_pytorch_dataset()` creates a map-style `torch.utils.data.Dataset` or streaming `IterableDataset` adapter.
2. `UniversalDataset.to_dataloader()` builds a standard PyTorch `DataLoader` and chooses `UniversalBatchCollator` by default.
3. `load(..., return_type="pytorch" | "dataloader")` requests those representations during `load()` finalization.

The library does not create a model, optimizer, loss function, or training loop. It creates input dictionaries and batches; your model must agree with those keys and shapes.

## 7.2 Text examples

Without a tokenizer:

```python
pt_ds = ds.to_pytorch_dataset()
example = pt_ds[0]
# Contains id, text, and label where present.
```

With a compatible tokenizer:

```python
pt_ds = ds.to_pytorch_dataset(
    tokenizer_or_processor=tokenizer,
    max_length=256,
)
example = pt_ds[0]
# Common keys: input_ids, attention_mask, labels, id, and label (if source label exists).
```

The adapter invokes a text tokenizer with truncation enabled, `max_length`, no per-example padding, and `return_tensors="pt"`. If `input_ids` exists but `labels` does not, it copies `input_ids` to `labels`, supporting causal language-model-style labels. This is a default convenience, not a universal objective: token classification, seq2seq, instruction-tuning masking, and packed training often require explicit label preparation.

Dynamic padding is applied later by `UniversalBatchCollator`. A tokenizer must support the callable signature expected by the adapter. Some tokenizers/processors require special kwargs or need a custom wrapper.

## 7.3 Image examples

Image-only samples use keys such as:

- `id`
- `pixel_values`: RGB float tensor shaped `[3, H, W]`
- `label`, if present
- `media_path`, if present

The built-in helper resizes to `image_size=(height, width)` and produces a channel-first tensor. Integer image data are scaled to the 0–1 range by the standard fallback; a custom processor may return normalized values outside that range, which are preserved by the custom output normalizer. Use the normalization expected by the model you intend to train.

```python
loader = dh.load(
    "./cats_and_dogs",
    modality="image",
    return_type="dataloader",
    batch_size=32,
    image_size=(224, 224),
)
```

A local image directory can assign a class name from the parent directory. Verify `label` values and mapping rather than assuming this is true for every layout.

## 7.4 Audio examples

Without a specialized processor, the adapter emits `waveform`. The media helper returns normalized audio shaped `[channels, time]` and resamples to the requested sample rate when it can access a supported resampling backend. The standard WAV path is intended for uncompressed PCM WAV; SciPy and/or torchaudio improve format coverage. Optional processor output is stored under `input_features`.

```python
audio_loader = dh.load(
    "./audio_manifest.csv",
    modality="audio",
    return_type="dataloader",
    audio_sr=16_000,
    batch_size=8,
)
```

Waveforms may have variable time lengths, so the collator pads them to a batch maximum. Processor-generated feature tensors are expected to have compatible shapes for stacking; if your processor returns varying feature sequence lengths, use an appropriate custom processor/collator or pre-pad consistently.

## 7.5 Video examples

The video helper samples up to a configured frame count, converts frames from BGR to RGB via OpenCV, resizes to the target `(height, width)`, normalizes pixel values to float32 0–1, and pads a short extraction by repeating the final frame. A successful decode returns `[T, 3, H, W]`; the batch collator stacks samples into `[B, T, 3, H, W]`.

```python
video_loader = dh.load(
    "./clips",
    modality="video",
    return_type="dataloader",
    max_video_frames=16,
    image_size=(224, 224),
    batch_size=4,
)
```

OpenCV must be installed and the target video codec must be readable in that runtime. Verify `max_video_frames` and video paths before training.

## 7.6 Multimodal vision-language examples

For `modality="multimodal"`, the current training adapter implements a vision-plus-language path. Expected keys can include:

- `pixel_values` for the image;
- `input_ids`, `attention_mask`, or processor-specific token outputs for text when a text tokenizer is available;
- `text` when a tokenizer is not selected;
- `label` when present.

The adapter attempts to separate visual processor and text tokenizer when the supplied processor exposes common properties such as `image_processor`, `feature_extractor`, or `tokenizer`. Model-specific processor objects differ; validate one item before making a full DataLoader.

```python
sample = {
    "image_path": "images/001.jpg",
    "caption": "A small dog standing on grass",
    "label": "dog",
}
ds = dh.load([sample], modality="multimodal", media_column="image_path")
```

This is a record-shape example, not a guarantee that an arbitrary vision-language model processor works unmodified. In particular, specialized models may require multiple images, image crops, chat templates, special tokens, or processor-specific nested data.

## 7.7 Collator behavior

`UniversalBatchCollator` validates and combines per-example dictionaries. Its main responsibilities are:

- dynamically pad `input_ids` to the longest sequence in the batch, using the configured/tokenizer `pad_token_id`;
- pad attention masks with zero;
- align/pad token-sequence labels with `-100` where appropriate (the conventional ignore index for many PyTorch losses);
- require compatible ranks and matching attention-mask/token shapes;
- stack image tensors into `[B, 3, H, W]` after validating consistent CHW RGB shapes;
- pad waveform sequences where supported;
- stack compatible `input_features` tensors;
- stack video frames into `[B, T, 3, H, W]` with uniform frame tensor shapes;
- collate homogeneous numeric labels into tensors and retain non-numeric labels as lists;
- surface invalid-sample metadata in supported permissive paths.

If any item provides a key that requires tensor collation, the collator validates the corresponding field across the batch. Missing `input_ids`, inconsistent image sizes, mixed scalar/sequence token labels, or a missing label among otherwise labelled rows should be fixed upstream or handled with a custom collator—not silently ignored.

## 7.8 Build a DataLoader manually

```python
from torch.utils.data import DataLoader
from data_hub import UniversalBatchCollator

pt_ds = ds.to_pytorch_dataset(
    tokenizer_or_processor=tokenizer,
    max_length=128,
)
collator = UniversalBatchCollator(
    modality=ds.modality,
    tokenizer_or_processor=tokenizer,
)
loader = DataLoader(
    pt_ds,
    batch_size=16,
    shuffle=not ds.is_streaming,
    num_workers=0,
    collate_fn=collator,
)
```

For most code, `ds.to_dataloader(...)` is simpler and less error-prone.

## 7.9 `num_workers` and streaming

`to_dataloader()` forwards standard PyTorch DataLoader options via keyword arguments, including `num_workers`, `pin_memory`, and `drop_last`, subject to normal PyTorch constraints. For streaming datasets, `shuffle` is disabled at the DataLoader layer; use a source that already shuffles or implement an appropriate buffering strategy if stochastic order is required.

The iterable adapter shards items across workers by record index modulo worker count. Each worker must still have access to a safe source iterator; test multi-worker behavior against the concrete source because some external iterable backends have their own sharding behavior.

## 7.10 Error policy during training

- `strict`: surface media/tokenization failures with sample context.
- `report_and_skip`: for supported finite media datasets, pre-check invalid items and skip them; adapter failures can mark items invalid for the collator when that path supports it.
- `permissive`: return invalid marker/error metadata when a transformation fails and a non-strict fallback is supported; the collator may drop invalid items if valid peers remain, and raises if the whole batch is invalid.

Never treat a permissive batch as proof that every input was valid. Monitor invalid sample metadata and compare expected batch counts. A report-and-skip prevalidation can incur extra media-decoding cost because it touches samples before iteration.

## 7.11 Training readiness checklist

Before a long run:

1. Inspect a few raw records and several adapted samples.
2. Confirm tensor names match the model's forward signature.
3. Confirm shapes, dtypes, image normalization and audio sample rate.
4. Verify labels are numeric and mapped to the expected class indices.
5. Check how `labels` are constructed; default copying of input IDs is not right for every objective.
6. Run a complete pass or enough steps to trigger lazy errors.
7. Log parsing errors, skipped media, invalid batches, and source sizes.
