# ========================================
# Quickstart: In-Memory Rows
# ========================================
from pathlib import Path
import tempfile

import data_hub as dh

records = [
    {"id": "r1", "text": "  The support team was helpful.  ", "label": "positive"},
    {"id": "r2", "text": "The package arrived late.", "label": "negative"},
    {"id": "r3", "text": "Great quality and thoughtful packaging.", "label": "positive"},
    {"id": "r4", "text": "Not worth the price.", "label": "negative"},
]

ds = dh.load(records, modality="text", label_dtype="str")
print("sample count:", ds.num_samples)
print("preview:", ds.preview(2, detailed=True))
print("summary:", ds.get_summary().model_dump())

# map() returns a new dataset; the source remains unchanged.
clean_text = ds.map(lambda item: {"text": (item.text or "").strip()})
encoded = clean_text.encode_labels({"negative": 0, "positive": 1})
print("labels:", [item.label for item in encoded])

splits = encoded.split(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
print("split counts:", {
    "train": len(splits.train),
    "val": len(splits.val) if splits.val is not None else 0,
    "test": len(splits.test) if splits.test is not None else 0,
})

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "processed.jsonl"
    clean_text.export(str(output), format="jsonl")
    print("exported:", output.exists(), "bytes:", output.stat().st_size)
