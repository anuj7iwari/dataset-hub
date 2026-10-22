# ========================================
# Schemas and Data Models
# ========================================
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Accepted values for DatasetConfig.label_dtype: "auto", "str", "int", "float"
LABEL_DTYPES = ("auto", "str", "int", "float")


# ========================================
# ModalityType Enumeration
# ========================================
class ModalityType(str, Enum):
    """Supported data modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    JSON = "json"
    TABULAR = "tabular"

    @classmethod
    def from_str(cls, val: Union[str, "ModalityType", None], default: Optional["ModalityType"] = None) -> "ModalityType":
        """Normalizes user-supplied string or alias into a canonical ModalityType with strict validation."""
        if isinstance(val, cls):
            return val
        if val is None or val == "":
            if default is not None:
                return default
            return cls.TEXT
        clean = str(val).strip().lower()
        mapping = {
            "text": cls.TEXT,
            "txt": cls.TEXT,
            "nlp": cls.TEXT,
            "language": cls.TEXT,
            "image": cls.IMAGE,
            "img": cls.IMAGE,
            "vision": cls.IMAGE,
            "visual": cls.IMAGE,
            "picture": cls.IMAGE,
            "audio": cls.AUDIO,
            "sound": cls.AUDIO,
            "speech": cls.AUDIO,
            "voice": cls.AUDIO,
            "video": cls.VIDEO,
            "vid": cls.VIDEO,
            "multimodal": cls.MULTIMODAL,
            "vl": cls.MULTIMODAL,
            "vlm": cls.MULTIMODAL,
            "vision-language": cls.MULTIMODAL,
            "json": cls.JSON,
            "structured": cls.JSON,
            "tabular": cls.TABULAR,
            "csv": cls.TABULAR,
            "table": cls.TABULAR,
            "dataframe": cls.TABULAR,
        }
        if clean in mapping:
            return mapping[clean]
        valid_opts = sorted(list(set(mapping.keys())))
        raise ValueError(f"Unknown modality '{val}'. Expected one of: {valid_opts}")


# ========================================
# SourceType Enumeration
# ========================================
class SourceType(str, Enum):
    """Origin source of the dataset."""
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    LOCAL_FILE = "local_file"
    LOCAL_DIR = "local_dir"
    MEMORY = "memory"

    @classmethod
    def from_str(cls, val: Union[str, "SourceType", None], default: Optional["SourceType"] = None) -> "SourceType":
        """Normalizes user-supplied string or alias into a canonical SourceType with strict validation."""
        if isinstance(val, cls):
            return val
        if val is None or val == "":
            if default is not None:
                return default
            return cls.LOCAL_FILE
        clean = str(val).strip().lower()
        mapping = {
            "hf": cls.HUGGINGFACE,
            "huggingface": cls.HUGGINGFACE,
            "hf_hub": cls.HUGGINGFACE,
            "kaggle": cls.KAGGLE,
            "kaggle_hub": cls.KAGGLE,
            "local_file": cls.LOCAL_FILE,
            "file": cls.LOCAL_FILE,
            "local": cls.LOCAL_FILE,
            "path": cls.LOCAL_FILE,
            "local_dir": cls.LOCAL_DIR,
            "dir": cls.LOCAL_DIR,
            "directory": cls.LOCAL_DIR,
            "folder": cls.LOCAL_DIR,
            "memory": cls.MEMORY,
            "in_memory": cls.MEMORY,
            "list": cls.MEMORY,
            "dataframe": cls.MEMORY,
            "df": cls.MEMORY,
        }
        if clean in mapping:
            return mapping[clean]
        valid_opts = sorted(list(set(mapping.keys())))
        raise ValueError(f"Unknown source type '{val}'. Expected one of: {valid_opts}")


# ========================================
# ReturnType Enumeration
# ========================================
class ReturnType(str, Enum):
    """Specifies the target data representation yielded by the dataset."""
    DATASET = "dataset"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    RAW = "raw"
    DATALOADER = "dataloader"

    @classmethod
    def from_str(cls, val: Union[str, "ReturnType", None], default: Optional["ReturnType"] = None) -> "ReturnType":
        """Normalizes user-supplied return type string into a canonical ReturnType with strict validation."""
        if isinstance(val, cls):
            return val
        if val is None or val == "":
            if default is not None:
                return default
            return cls.DATASET
        clean = str(val).strip().lower()
        mapping = {
            "dataset": cls.DATASET,
            "universal": cls.DATASET,
            "pytorch": cls.PYTORCH,
            "pt": cls.PYTORCH,
            "torch": cls.PYTORCH,
            "tensor": cls.PYTORCH,
            "tensors": cls.PYTORCH,
            "numpy": cls.NUMPY,
            "np": cls.NUMPY,
            "ndarray": cls.NUMPY,
            "raw": cls.RAW,
            "dict": cls.RAW,
            "pil": cls.RAW,
            "dataloader": cls.DATALOADER,
            "loader": cls.DATALOADER,
        }
        if clean in mapping:
            return mapping[clean]
        valid_opts = sorted(list(set(mapping.keys())))
        raise ValueError(f"Unknown return type '{val}'. Expected one of: {valid_opts}")


# ========================================
# ErrorPolicy Enumeration
# ========================================
class ErrorPolicy(str, Enum):
    """Policy for handling malformed, corrupted, or unreadable samples during ingestion."""
    STRICT = "strict"                      # Fail immediately with sample/file context (default for AI training)
    REPORT_AND_SKIP = "report_and_skip"    # Skip invalid rows but record counts, IDs/paths, and error reasons
    PERMISSIVE = "permissive"              # Best-effort fallback with clear warning logs

    @classmethod
    def from_str(cls, val: Union[str, "ErrorPolicy", None]) -> "ErrorPolicy":
        if isinstance(val, cls):
            return val
        if not val:
            return cls.STRICT
        clean = str(val).strip().lower()
        mapping = {
            "strict": cls.STRICT,
            "fail": cls.STRICT,
            "report_and_skip": cls.REPORT_AND_SKIP,
            "skip": cls.REPORT_AND_SKIP,
            "permissive": cls.PERMISSIVE,
            "warn": cls.PERMISSIVE,
        }
        if clean in mapping:
            return mapping[clean]
        raise ValueError(f"Unknown error policy '{val}'. Expected 'strict', 'report_and_skip', or 'permissive'.")


# ========================================
# DatasetItem Model
# ========================================
class DatasetItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[Union[int, str]] = None
    text: Optional[str] = None
    media_path: Optional[str] = None
    label: Optional[Union[str, int, float, List[str], List[int]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None


# ========================================
# DatasetConfig Model
# ========================================
class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_type: Optional[SourceType] = None
    path_or_identifier: str
    modality: ModalityType = ModalityType.TEXT
    return_type: ReturnType = ReturnType.DATASET
    error_policy: ErrorPolicy = ErrorPolicy.STRICT
    split: Optional[str] = "train"
    subset: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    media_column: Optional[str] = None
    max_samples: Optional[int] = Field(default=None, ge=1, strict=True)
    prompt_template: Optional[str] = None
    seed: int = 42
    streaming: bool = False
    batch_size: int = Field(default=8, ge=1, strict=True)
    shuffle: bool = True
    max_length: int = Field(default=128, ge=1, strict=True)
    target_sr: int = Field(default=16000, ge=1, strict=True)
    image_size: Tuple[int, int] = (224, 224)
    max_video_frames: int = Field(default=16, ge=1, strict=True)
    label_dtype: str = "auto"
    treat_path_as_text: bool = False
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("path_or_identifier", mode="before")
    @classmethod
    def _check_path_or_identifier(cls, v: Any) -> str:
        # PathLike inputs are convenient and were historically accepted by callers.
        if not isinstance(v, str) and hasattr(v, "__fspath__"):
            v = v.__fspath__()
        if not isinstance(v, str):
            raise ValueError("path_or_identifier must be a non-empty string or path-like value")
        clean = v.strip()
        if not clean:
            raise ValueError("path_or_identifier must not be empty or whitespace-only")
        return clean

    @field_validator("label_dtype")
    @classmethod
    def _check_label_dtype(cls, v: str) -> str:
        clean = str(v).strip().lower()
        if clean not in LABEL_DTYPES:
            raise ValueError(f"label_dtype must be one of {list(LABEL_DTYPES)}, got '{v}'")
        return clean

    @field_validator("image_size", mode="before")
    @classmethod
    def _check_image_size(cls, v: Any) -> Tuple[int, int]:
        # Validate before Pydantic can coerce bools/floats/strings into integers.
        import numbers
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError(f"image_size must be a (height, width) pair of positive ints, got {v!r}")
        if any(not isinstance(x, numbers.Integral) or isinstance(x, bool) or int(x) < 1 for x in v):
            raise ValueError(f"image_size must be a (height, width) pair of positive ints, got {v!r}")
        return (int(v[0]), int(v[1]))


# ========================================
# DatasetSummary Model
# ========================================
class DatasetSummary(BaseModel):
    name: str
    modality: ModalityType
    source_type: SourceType
    return_type: ReturnType = ReturnType.DATASET
    error_policy: ErrorPolicy = ErrorPolicy.STRICT
    #: ``None`` means the length is unknown (e.g. an unbounded stream) -- never 0.
    total_samples: Optional[int]
    is_streaming: bool = False
    columns: List[str] = Field(default_factory=list)
    splits: Dict[str, int] = Field(default_factory=dict)
    sample_preview: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_tokens: Optional[int] = None
    token_estimation_method: Optional[str] = None
    class_distribution: Optional[Dict[str, int]] = None
    memory_mb: Optional[float] = None
    skipped_samples: int = 0
    parsing_errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    warning: Optional[str] = None
    included_files: Optional[List[str]] = None
    excluded_files: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
