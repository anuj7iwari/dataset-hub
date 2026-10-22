# ========================================
# CSV Manifest: Column Mapping
# ========================================
from pathlib import Path
import tempfile

import data_hub as dh

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    path = root / "train.csv"
    path.write_text(
        'id,sentence,sentiment\n'
        '1,"The camera is sharp",positive\n'
        '2,"The battery life is poor",negative\n',
        encoding="utf-8",
    )

    ds = dh.load(
        str(path),
        source="local_file",
        modality="text",
        text_column="sentence",
        label_column="sentiment",
        label_dtype="str",
        error_policy="strict",
    )

    for item in ds:
        print(item.id, item.text, item.label, item.raw_data)
