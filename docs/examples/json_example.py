from pathlib import Path
import json
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
DATA_DIR = Path(__file__).parent / "sampledata" / "json"

# ========================================
# Config 1 : [ Standard JSON array mounting ]
# ========================================
ds_array = dh.load(
    str(DATA_DIR / "instructions.json"),
    modality=ModalityType.TEXT,
    text_column="instruction",
    label_column="domain",
)

# ========================================
# Config 2 : [ Line-delimited JSONL mounting ]
# ========================================
ds_jsonl = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality=ModalityType.TEXT,
    text_column="prompt",
    label_column="completion",
)

# ========================================
# Config 3 : [ Structured chat conversation mounting ]
# ========================================
ds_chat = dh.load(
    str(DATA_DIR / "chat_corpus.jsonl"),
    modality=ModalityType.TEXT,
)

# ========================================
# Config 4 : [ Streaming JSONL ingestion ]
# ========================================
ds_stream = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    text_column="prompt",
    label_column="completion",
    streaming=True,
    max_samples=2,
)

# ========================================
# Config 5 : [ Prompt template formatting ]
# ========================================
ds_prompt = dh.load(
    str(DATA_DIR / "instructions.json"),
    modality="text",
    text_column="instruction",
    prompt_template="### Instruction:\n{instruction}\n\n### Response:\n{response}",
)

# ========================================
# Config 6 : [ Declarative DatasetConfig ]
# ========================================
config = DatasetConfig(
    path_or_identifier=str(DATA_DIR / "instructions.json"),
    source_type=SourceType.LOCAL_FILE,
    modality=ModalityType.TEXT,
    return_type=ReturnType.DATASET,
    text_column="instruction",
    label_column="domain",
    error_policy=ErrorPolicy.STRICT,
)
ds_cfg = hub.load(config)

# ========================================
# Config 7 : [ Error policy configuration ]
# ========================================
ds_policy = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    error_policy=ErrorPolicy.REPORT_AND_SKIP,
)

# ========================================
# Return Type 1 : [ Returns universal dataset ]
# ========================================
ds_return = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    text_column="prompt",
    label_column="completion",
    return_type=ReturnType.DATASET,
)

# ========================================
# Return Type 2 : [ Returns numpy ndarrays ]
# ========================================
np_arrays = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    text_column="prompt",
    label_column="completion",
    return_type=ReturnType.NUMPY,
)

# ========================================
# Return Type 3 : [ Returns pytorch tensors ]
# ========================================
pt_dataset = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    text_column="prompt",
    label_column="completion",
    return_type=ReturnType.PYTORCH,
)

# ========================================
# Return Type 4 : [ Returns dataloader ]
# ========================================
loader = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    text_column="prompt",
    label_column="completion",
    return_type=ReturnType.DATALOADER,
    batch_size=2,
    shuffle=False,
)

# ========================================
# Return Type 5 : [ Returns raw dictionaries ]
# ========================================
raw_items = dh.load(
    str(DATA_DIR / "dialogues.jsonl"),
    modality="text",
    text_column="prompt",
    label_column="completion",
    return_type=ReturnType.RAW,
)
