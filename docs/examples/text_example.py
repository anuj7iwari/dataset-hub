from pathlib import Path
import numpy as np
import torch

import data_hub as dh
from data_hub import (
    DatasetHub,
    DatasetConfig,
    ModalityType,
    SourceType,
    ReturnType,
    ErrorPolicy,
    load,
)

# ========================================
# Library Initialization
# ========================================
hub = DatasetHub(cache_dir="./data_cache")

# Path to local sample data
DATA_DIR = Path(__file__).parent / "sampledata" / "text"

# ========================================
# Config 1 : [ In-memory text mounting ]
# ========================================
records = [
    {"sentence": "The customer service was rapid and friendly.", "sentiment": "positive"},
    {"sentence": "Terrible delay in shipment and damaged box.", "sentiment": "negative"},
]
ds_mem = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    label_dtype="str",
)

# ========================================
# Config 2 : [ Local text file mounting ]
# ========================================
ds_file = dh.load(
    str(DATA_DIR / "sample.txt"),
    modality="text",
    treat_path_as_text=False,
)

# ========================================
# Config 3 : [ Declarative DatasetConfig ]
# ========================================
config = DatasetConfig(
    path_or_identifier=str(DATA_DIR / "sample.txt"),
    source_type=SourceType.LOCAL_FILE,
    modality=ModalityType.TEXT,
    return_type=ReturnType.DATASET,
    error_policy=ErrorPolicy.STRICT,
    max_samples=50,
)
ds_cfg = hub.load(config)

# ========================================
# Config 4 : [ Streaming text ingestion ]
# ========================================
ds_stream = dh.load(
    str(DATA_DIR / "sample.txt"),
    modality="text",
    streaming=True,
    max_samples=10,
)

# ========================================
# Config 5 : [ Prompt template formatting ]
# ========================================
ds_prompt = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    prompt_template="Review: {text}\nSentiment: {label}",
)

# ========================================
# Config 6 : [ Error policy configuration ]
# ========================================
ds_policy = dh.load(
    str(DATA_DIR / "sample.txt"),
    modality="text",
    error_policy=ErrorPolicy.REPORT_AND_SKIP,
)

# ========================================
# Config 7 : [ Remote Hugging Face mounting ]
# ========================================
# ds_hf = dh.load("hf://rotten_tomatoes", split="train", modality="text")

# ========================================
# Return Type 1 : [ Returns universal dataset ]
# ========================================
ds_return = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    return_type=ReturnType.DATASET,
)

# ========================================
# Return Type 2 : [ Returns numpy ndarrays ]
# ========================================
np_arrays = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    return_type=ReturnType.NUMPY,
)

# ========================================
# Return Type 3 : [ Returns pytorch tensors ]
# ========================================
pt_dataset = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    return_type=ReturnType.PYTORCH,
)

# ========================================
# Return Type 4 : [ Returns dataloader ]
# ========================================
loader = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    return_type=ReturnType.DATALOADER,
    batch_size=2,
    shuffle=False,
)

# ========================================
# Return Type 5 : [ Returns raw dictionaries ]
# ========================================
raw_items = dh.load(
    records,
    modality="text",
    text_column="sentence",
    label_column="sentiment",
    return_type=ReturnType.RAW,
)
