from pathlib import Path
import numpy as np
import torch
from PIL import Image

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
DATA_DIR = Path(__file__).parent / "sampledata" / "img"

# ========================================
# Config 1 : [ Single image mounting ]
# ========================================
ds_single = dh.load(
    str(DATA_DIR / "single_image.png"),
    modality=ModalityType.IMAGE,
)

# ========================================
# Config 2 : [ Image directory scan with subfolders ]
# ========================================
ds_dir = dh.load(
    str(DATA_DIR),
    source=SourceType.LOCAL_DIR,
    modality=ModalityType.IMAGE,
)

# ========================================
# Config 3 : [ Image manifest mounting ]
# ========================================
ds_manifest = dh.load(
    str(DATA_DIR / "image_manifest.csv"),
    modality=ModalityType.IMAGE,
    media_column="image_file",
    label_column="category",
)

# ========================================
# Config 4 : [ Target resolution image_size ]
# ========================================
ds_resized = dh.load(
    str(DATA_DIR),
    modality="image",
    image_size=(128, 128),  # (height, width)
)

# ========================================
# Config 5 : [ In-memory PIL image mounting ]
# ========================================
pil_image = Image.new("RGB", (64, 64), color="blue")
ds_mem = dh.load(
    [{"id": 1, "image": pil_image, "label": "blue_box"}],
    modality=ModalityType.IMAGE,
)

# ========================================
# Config 6 : [ Declarative DatasetConfig ]
# ========================================
config = DatasetConfig(
    path_or_identifier=str(DATA_DIR),
    source_type=SourceType.LOCAL_DIR,
    modality=ModalityType.IMAGE,
    return_type=ReturnType.DATASET,
    image_size=(224, 224),
    error_policy=ErrorPolicy.STRICT,
)
ds_cfg = hub.load(config)

# ========================================
# Config 7 : [ Error policy configuration ]
# ========================================
ds_policy = dh.load(
    str(DATA_DIR),
    modality="image",
    error_policy=ErrorPolicy.REPORT_AND_SKIP,
)

# ========================================
# Return Type 1 : [ Returns universal dataset ]
# ========================================
ds_return = dh.load(
    str(DATA_DIR),
    modality="image",
    return_type=ReturnType.DATASET,
)

# ========================================
# Return Type 2 : [ Returns numpy ndarrays ]
# ========================================
np_arrays = dh.load(
    str(DATA_DIR),
    modality="image",
    return_type=ReturnType.NUMPY,
)

# ========================================
# Return Type 3 : [ Returns pytorch tensors ]
# ========================================
pt_dataset = dh.load(
    str(DATA_DIR),
    modality="image",
    image_size=(64, 64),
    return_type=ReturnType.PYTORCH,
)

# ========================================
# Return Type 4 : [ Returns dataloader ]
# ========================================
loader = dh.load(
    str(DATA_DIR),
    modality="image",
    image_size=(64, 64),
    return_type=ReturnType.DATALOADER,
    batch_size=2,
    shuffle=True,
)

# ========================================
# Return Type 5 : [ Returns raw dictionaries ]
# ========================================
raw_items = dh.load(
    str(DATA_DIR),
    modality="image",
    return_type=ReturnType.RAW,
)
