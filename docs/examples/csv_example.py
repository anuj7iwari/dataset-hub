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
DATA_DIR = Path(__file__).parent / "sampledata" / "csv"

# ========================================
# Config 1 : [ CSV text classification mounting ]
# ========================================
ds_text = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality=ModalityType.TEXT,
    text_column="feedback_text",
    label_column="satisfaction_score",
)

# ========================================
# Config 2 : [ CSV tabular data mounting ]
# ========================================
ds_tabular = dh.load(
    str(DATA_DIR / "tabular_data.csv"),
    modality=ModalityType.TABULAR,
    label_column="price",
)

# ========================================
# Config 3 : [ CSV multimodal manifest mounting ]
# ========================================
ds_multi = dh.load(
    str(DATA_DIR / "multimodal_manifest.csv"),
    modality=ModalityType.MULTIMODAL,
    media_column="image_path",
    text_column="caption",
    label_column="tag",
)

# ========================================
# Config 4 : [ Label dtype casting (auto, str, int, float) ]
# ========================================
ds_int = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    label_dtype="int",
)

# ========================================
# Config 5 : [ Streaming CSV ingestion ]
# ========================================
ds_stream = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    streaming=True,
    max_samples=3,
)

# ========================================
# Config 6 : [ Row cap and prompt templating ]
# ========================================
ds_prompt = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    prompt_template="[Customer Feedback]: {text}\n[Rating]: {label}/5",
    max_samples=2,
)

# ========================================
# Config 7 : [ Declarative DatasetConfig ]
# ========================================
config = DatasetConfig(
    path_or_identifier=str(DATA_DIR / "text_classification.csv"),
    source_type=SourceType.LOCAL_FILE,
    modality=ModalityType.TEXT,
    return_type=ReturnType.DATASET,
    text_column="feedback_text",
    label_column="satisfaction_score",
    label_dtype="int",
    error_policy=ErrorPolicy.STRICT,
)
ds_cfg = hub.load(config)

# ========================================
# Return Type 1 : [ Returns universal dataset ]
# ========================================
ds_return = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    label_dtype="int",
    return_type=ReturnType.DATASET,
)

# ========================================
# Return Type 2 : [ Returns numpy ndarrays ]
# ========================================
np_arrays = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    label_dtype="int",
    return_type=ReturnType.NUMPY,
)

# ========================================
# Return Type 3 : [ Returns pytorch tensors ]
# ========================================
pt_dataset = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    label_dtype="int",
    return_type=ReturnType.PYTORCH,
)

# ========================================
# Return Type 4 : [ Returns dataloader ]
# ========================================
loader = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    label_dtype="int",
    return_type=ReturnType.DATALOADER,
    batch_size=2,
    shuffle=False,
)

# ========================================
# Return Type 5 : [ Returns raw dictionaries ]
# ========================================
raw_items = dh.load(
    str(DATA_DIR / "text_classification.csv"),
    modality="text",
    text_column="feedback_text",
    label_column="satisfaction_score",
    return_type=ReturnType.RAW,
)
