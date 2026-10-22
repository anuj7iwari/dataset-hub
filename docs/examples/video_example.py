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
DATA_DIR = Path(__file__).parent / "sampledata" / "video"

# ========================================
# Config 1 : [ Single video mounting ]
# ========================================
ds_single = dh.load(
    str(DATA_DIR / "single_video.mp4"),
    modality=ModalityType.VIDEO,
)

# ========================================
# Config 2 : [ Video directory scan with subfolders ]
# ========================================
ds_dir = dh.load(
    str(DATA_DIR),
    source=SourceType.LOCAL_DIR,
    modality=ModalityType.VIDEO,
)

# ========================================
# Config 3 : [ Video manifest mounting ]
# ========================================
ds_manifest = dh.load(
    str(DATA_DIR / "video_manifest.csv"),
    modality=ModalityType.VIDEO,
    media_column="video_path",
    label_column="activity",
)

# ========================================
# Config 4 : [ Frame extraction count max_video_frames ]
# ========================================
ds_frames = dh.load(
    str(DATA_DIR),
    modality="video",
    max_video_frames=8,
)

# ========================================
# Config 5 : [ Frame resolution image_size ]
# ========================================
ds_resized = dh.load(
    str(DATA_DIR),
    modality="video",
    max_video_frames=8,
    image_size=(64, 64),  # (height, width)
)

# ========================================
# Config 6 : [ Declarative DatasetConfig ]
# ========================================
config = DatasetConfig(
    path_or_identifier=str(DATA_DIR),
    source_type=SourceType.LOCAL_DIR,
    modality=ModalityType.VIDEO,
    return_type=ReturnType.DATASET,
    max_video_frames=8,
    image_size=(64, 64),
    error_policy=ErrorPolicy.STRICT,
)
ds_cfg = hub.load(config)

# ========================================
# Config 7 : [ Error policy configuration ]
# ========================================
ds_policy = dh.load(
    str(DATA_DIR),
    modality="video",
    error_policy=ErrorPolicy.REPORT_AND_SKIP,
)

# ========================================
# Return Type 1 : [ Returns universal dataset ]
# ========================================
ds_return = dh.load(
    str(DATA_DIR),
    modality="video",
    max_video_frames=6,
    return_type=ReturnType.DATASET,
)

# ========================================
# Return Type 2 : [ Returns numpy ndarrays ]
# ========================================
np_arrays = dh.load(
    str(DATA_DIR),
    modality="video",
    return_type=ReturnType.NUMPY,
)

# ========================================
# Return Type 3 : [ Returns pytorch tensors ]
# ========================================
pt_dataset = dh.load(
    str(DATA_DIR),
    modality="video",
    max_video_frames=6,
    image_size=(48, 48),
    return_type=ReturnType.PYTORCH,
)

# ========================================
# Return Type 4 : [ Returns dataloader ]
# ========================================
loader = dh.load(
    str(DATA_DIR),
    modality="video",
    max_video_frames=6,
    image_size=(48, 48),
    return_type=ReturnType.DATALOADER,
    batch_size=2,
    shuffle=True,
)

# ========================================
# Return Type 5 : [ Returns raw dictionaries ]
# ========================================
raw_items = dh.load(
    str(DATA_DIR),
    modality="video",
    return_type=ReturnType.RAW,
)
