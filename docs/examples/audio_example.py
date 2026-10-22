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
DATA_DIR = Path(__file__).parent / "sampledata" / "audio"

# ========================================
# Config 1 : [ Single audio file mounting ]
# ========================================
ds_single = dh.load(
    str(DATA_DIR / "single_audio.wav"),
    modality=ModalityType.AUDIO,
)

# ========================================
# Config 2 : [ Audio directory scan with subfolders ]
# ========================================
ds_dir = dh.load(
    str(DATA_DIR),
    source=SourceType.LOCAL_DIR,
    modality=ModalityType.AUDIO,
)

# ========================================
# Config 3 : [ Audio manifest mounting ]
# ========================================
ds_manifest = dh.load(
    str(DATA_DIR / "audio_manifest.csv"),
    modality=ModalityType.AUDIO,
    media_column="audio_file",
    text_column="transcript",
    label_column="speaker",
)

# ========================================
# Config 4 : [ Target resampling rate audio_sr ]
# ========================================
ds_resampled = dh.load(
    str(DATA_DIR),
    modality="audio",
    audio_sr=16000,
)

# ========================================
# Config 5 : [ In-memory audio array mounting ]
# ========================================
audio_samples = np.zeros(8000, dtype=np.float32)
ds_mem = dh.load(
    [{"id": 1, "audio": {"array": audio_samples, "sampling_rate": 16000}, "label": "silence"}],
    modality=ModalityType.AUDIO,
)

# ========================================
# Config 6 : [ Declarative DatasetConfig ]
# ========================================
config = DatasetConfig(
    path_or_identifier=str(DATA_DIR),
    source_type=SourceType.LOCAL_DIR,
    modality=ModalityType.AUDIO,
    return_type=ReturnType.DATASET,
    target_sr=16000,
    error_policy=ErrorPolicy.STRICT,
)
ds_cfg = hub.load(config)

# ========================================
# Config 7 : [ Error policy configuration ]
# ========================================
ds_policy = dh.load(
    str(DATA_DIR),
    modality="audio",
    error_policy=ErrorPolicy.REPORT_AND_SKIP,
)

# ========================================
# Return Type 1 : [ Returns universal dataset ]
# ========================================
ds_return = dh.load(
    str(DATA_DIR),
    modality="audio",
    audio_sr=16000,
    return_type=ReturnType.DATASET,
)

# ========================================
# Return Type 2 : [ Returns numpy ndarrays ]
# ========================================
np_arrays = dh.load(
    str(DATA_DIR),
    modality="audio",
    return_type=ReturnType.NUMPY,
)

# ========================================
# Return Type 3 : [ Returns pytorch tensors ]
# ========================================
pt_dataset = dh.load(
    str(DATA_DIR),
    modality="audio",
    audio_sr=16000,
    return_type=ReturnType.PYTORCH,
)

# ========================================
# Return Type 4 : [ Returns dataloader ]
# ========================================
loader = dh.load(
    str(DATA_DIR),
    modality="audio",
    audio_sr=16000,
    return_type=ReturnType.DATALOADER,
    batch_size=2,
    shuffle=True,
)

# ========================================
# Return Type 5 : [ Returns raw dictionaries ]
# ========================================
raw_items = dh.load(
    str(DATA_DIR),
    modality="audio",
    return_type=ReturnType.RAW,
)
