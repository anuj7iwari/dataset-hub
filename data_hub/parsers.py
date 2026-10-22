# ========================================
# Source Parsers Engine
# ========================================

import os
import json
import csv
import itertools
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from data_hub.schemas import (
    ModalityType,
    SourceType,
    ErrorPolicy,
    DatasetItem,
    DatasetConfig,
)
from data_hub.media import (
    load_image, load_audio, load_video,
    IMAGE_EXTENSIONS as _MEDIA_IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS as _MEDIA_AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS as _MEDIA_VIDEO_EXTENSIONS,
)

# Keep parser, media, and public Hub extension registries consistent.
IMAGE_EXTENSIONS = set(_MEDIA_IMAGE_EXTENSIONS)
AUDIO_EXTENSIONS = set(_MEDIA_AUDIO_EXTENSIONS)
VIDEO_EXTENSIONS = set(_MEDIA_VIDEO_EXTENSIONS)
TEXT_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".parquet", ".txt", ".md", ".yaml", ".yml"}
MANIFEST_NAMES = {"train.csv", "data.csv", "metadata.csv", "labels.csv", "dataset.csv", "train.jsonl", "data.jsonl"}

#: DataLoader-only kwargs that must never be forwarded to an external connector.
_DATALOADER_ONLY_KEYS = {
    "num_workers", "drop_last", "pin_memory", "persistent_workers",
    "prefetch_factor", "timeout", "worker_init_fn", "generator",
    "sampler", "batch_sampler", "multiprocessing_context", "collate_fn",
}


def profile_directory(dir_path: Path, max_scan: int = 500) -> Dict[str, Any]:
    """Profile a directory without silently omitting unseen modality types."""
    import numbers
    if not isinstance(max_scan, numbers.Integral) or isinstance(max_scan, bool) or max_scan <= 0:
        raise ValueError(f"max_scan must be a positive integer, got {max_scan!r}")
    max_scan = int(max_scan)
    counts = {
        "image": 0,
        "audio": 0,
        "video": 0,
        "text": 0,
        "tabular": 0,
        "manifests": [],
        "included_files": [],
        "excluded_files": [],
        "files_by_modality": {"image": [], "audio": [], "video": [], "text": [], "tabular": []},
        "scan_truncated": False,
        "counts_are_lower_bounds": False,
        "profile_complete": True,
    }
    scanned = 0
    scan_truncated = False

    all_files_iter = dir_path.glob("**/*")
    for p in all_files_iter:
        if not p.is_file():
            continue
        scanned += 1
        name_lower = p.name.lower()
        ext = p.suffix.lower()

        if name_lower in MANIFEST_NAMES or (ext in (".csv", ".tsv", ".parquet", ".jsonl", ".ndjson", ".json", ".yaml", ".yml") and p.parent == dir_path):
            counts["manifests"].append(str(p))
            counts["tabular"] += 1
            counts["files_by_modality"]["tabular"].append(str(p))
        elif ext in IMAGE_EXTENSIONS:
            counts["image"] += 1
            counts["files_by_modality"]["image"].append(str(p))
        elif ext in AUDIO_EXTENSIONS:
            counts["audio"] += 1
            counts["files_by_modality"]["audio"].append(str(p))
        elif ext in VIDEO_EXTENSIONS:
            counts["video"] += 1
            counts["files_by_modality"]["video"].append(str(p))
        elif ext in (".txt", ".md", ".yaml", ".yml"):
            counts["text"] += 1
            counts["files_by_modality"]["text"].append(str(p))

        if scanned >= max_scan:
            break

    # Distinguish exact-cap exhaustion from true truncation by checking for remaining files.
    remaining_file = None
    if scanned >= max_scan:
        for rem in all_files_iter:
            if rem.is_file():
                remaining_file = rem
                scan_truncated = True
                break

    counts["scan_truncated"] = scan_truncated
    counts["profile_complete"] = not scan_truncated
    counts["counts_are_lower_bounds"] = scan_truncated

    # If the primary pass is truncated, search by extension for modalities absent from
    # that window. This keeps the presence check independent of thousands of unrelated
    # files. We cap the extra matching files considered per modality; profile_complete
    # and counts_are_lower_bounds are explicit so callers cannot mistake partial totals.
    if scan_truncated:
        missing_modalities = {k for k in ("image", "audio", "video", "text", "tabular") if counts[k] == 0}

        def _case_insensitive_pattern(ext: str) -> str:
            return "**/*." + "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in ext.lstrip("."))

        modality_extensions = {
            "image": IMAGE_EXTENSIONS,
            "audio": AUDIO_EXTENSIONS,
            "video": VIDEO_EXTENSIONS,
            "text": {".txt", ".md", ".yaml", ".yml"},
            "tabular": {".csv", ".tsv", ".parquet", ".jsonl", ".ndjson", ".json", ".yaml", ".yml"},
        }
        targeted_budget = 150
        for modality in list(missing_modalities):
            extra_count = 0
            seen_paths = set()
            for ext in sorted(modality_extensions[modality]):
                for p in dir_path.glob(_case_insensitive_pattern(ext)):
                    if extra_count >= targeted_budget:
                        break
                    actual_ext = p.suffix.lower()
                    # Filter before Path.is_file() to preserve the intent of the work cap
                    # even when a custom filesystem iterator yields broad candidates.
                    if actual_ext not in modality_extensions[modality] or str(p) in seen_paths:
                        continue
                    name_lower = p.name.lower()
                    if modality == "tabular":
                        is_manifest = name_lower in MANIFEST_NAMES or (
                            actual_ext in (".csv", ".tsv", ".parquet", ".jsonl", ".ndjson", ".json", ".yaml", ".yml")
                            and p.parent == dir_path
                        )
                        if not is_manifest:
                            continue
                    elif modality == "text" and actual_ext in (".yaml", ".yml") and p.parent == dir_path:
                        # A top-level YAML file is classified as a structured manifest.
                        continue
                    if not p.is_file():
                        continue
                    seen_paths.add(str(p))
                    if modality == "tabular":
                        counts["manifests"].append(str(p))
                        counts["tabular"] += 1
                        counts["files_by_modality"]["tabular"].append(str(p))
                    elif modality == "image":
                        counts["image"] += 1
                        counts["files_by_modality"]["image"].append(str(p))
                    elif modality == "audio":
                        counts["audio"] += 1
                        counts["files_by_modality"]["audio"].append(str(p))
                    elif modality == "video":
                        counts["video"] += 1
                        counts["files_by_modality"]["video"].append(str(p))
                    elif modality == "text":
                        counts["text"] += 1
                        counts["files_by_modality"]["text"].append(str(p))
                    extra_count += 1
                if extra_count >= targeted_budget:
                    break

        counts["profile_complete"] = False
        counts["counts_are_lower_bounds"] = True

    return counts


def row_to_dataset_item(
    row: Dict[str, Any],
    idx: int,
    cfg: DatasetConfig,
) -> DatasetItem:
    """Normalizes an arbitrary data dictionary into a standardized DatasetItem."""
    text = None
    label = None
    media_path = None
    metadata: Dict[str, Any] = {}

    MEDIA_COL_NAMES = {
        "image_path", "image", "audio_path", "audio", "video_path", "video",
        "file_path", "filepath", "path", "filename", "file_name", "url"
    }
    LABEL_COL_NAMES = {"label", "category", "target", "class", "sentiment"}

    # 1. Text extraction and Structured Chat Preservation
    if cfg.text_column and cfg.text_column in row:
        # Null means a missing value; do not turn it into literal training text "None".
        text = None if row[cfg.text_column] is None else str(row[cfg.text_column])
    elif "text" in row and row["text"] is not None:
        text = str(row["text"])
    elif "caption" in row and row["caption"] is not None:
        text = str(row["caption"])
    elif "description" in row and row["description"] is not None:
        text = str(row["description"])
    elif "transcription" in row and row["transcription"] is not None:
        text = str(row["transcription"])
    elif "transcript" in row and row["transcript"] is not None:
        text = str(row["transcript"])
    elif "sentence" in row and row["sentence"] is not None:
        text = str(row["sentence"])
    elif "content" in row and row["content"] is not None:
        text = str(row["content"])
    elif "prompt" in row and "response" in row:
        text = f"{row['prompt']}\n{row['response']}"
    elif "instruction" in row and "output" in row:
        inp = row.get("input", "")
        if inp:
            text = f"Instruction: {row['instruction']}\nInput: {inp}\nResponse: {row['output']}"
        else:
            text = f"Instruction: {row['instruction']}\nResponse: {row['output']}"
    elif "question" in row and "answer" in row:
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
    elif "messages" in row and isinstance(row["messages"], list):
        # Preserve original structured chat turns for loss masking / chat templates
        metadata["messages"] = copy.deepcopy(row["messages"])
        turns = []
        for turn_idx, m in enumerate(row["messages"]):
            if not isinstance(m, dict):
                raise ValueError(
                    f"Invalid chat row id={row.get('id', idx)!r}: messages[{turn_idx}] "
                    f"must be a mapping, got {type(m).__name__}."
                )
            role = m.get("role", "user")
            content = m.get("content", "")
            turns.append(f"{str(role).capitalize()}: {content}")
        text = "\n".join(turns)
    elif "conversations" in row and isinstance(row["conversations"], list):
        metadata["conversations"] = copy.deepcopy(row["conversations"])
        turns = []
        for turn_idx, m in enumerate(row["conversations"]):
            if not isinstance(m, dict):
                raise ValueError(
                    f"Invalid conversation row id={row.get('id', idx)!r}: conversations[{turn_idx}] "
                    f"must be a mapping, got {type(m).__name__}."
                )
            sender = m.get("from", "user")
            value = m.get("value", "")
            turns.append(f"{str(sender).capitalize()}: {value}")
        text = "\n".join(turns)
    elif cfg.modality in (ModalityType.TEXT, ModalityType.MULTIMODAL):
        for k, v in row.items():
            if k.lower() in MEDIA_COL_NAMES or k.lower() in LABEL_COL_NAMES or k.lower() == "id":
                continue
            if isinstance(v, str) and len(v.strip()) > 0:
                text = v
                break

    # 2. Label extraction
    if cfg.label_column and cfg.label_column in row:
        label = row[cfg.label_column]
    elif "label" in row:
        label = row["label"]
    elif "category" in row:
        label = row["category"]
    elif "target" in row:
        label = row["target"]
    elif "class" in row:
        label = row["class"]
    elif "sentiment" in row:
        label = row["sentiment"]
    elif not cfg.label_column:
        lower_row = {str(k).lower(): v for k, v in row.items()}
        for alias in ("label", "category", "target", "class", "sentiment"):
            if alias in lower_row and lower_row[alias] is not None:
                label = lower_row[alias]
                break

    # 3. Media path extraction
    media_extensions = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    lower_row = {str(k).lower(): v for k, v in row.items()}

    def _lookup(*names):
        for n in names:
            if n in lower_row and lower_row[n] is not None:
                return lower_row[n]
        return None

    def _media_reference(value):
        """Return a path/URL from a direct or nested media field, if present."""
        if isinstance(value, (str, Path)):
            clean_value = str(value).strip()
            return clean_value or None
        if isinstance(value, dict):
            for field_name in ("path", "file_path", "filepath", "filename", "file_name", "url", "src", "uri"):
                nested = value.get(field_name)
                if isinstance(nested, (str, Path)) and str(nested).strip():
                    return str(nested).strip()
        return None

    if cfg.media_column and cfg.media_column in row and row[cfg.media_column] is not None:
        media_value = row[cfg.media_column]
        media_path = _media_reference(media_value)
        # Structured embedded media (e.g. HF audio {array, sampling_rate}) is
        # a payload, not a filesystem path. Leave media_path unset in that case.
        if media_path is None and not isinstance(media_value, (dict, list, tuple, bytes, bytearray)):
            media_path = str(media_value)
    else:
        direct = _lookup("media_path", "image_path", "audio_path", "video_path", "file_path", "filepath", "url", "media")
        if direct is not None:
            media_path = _media_reference(direct)
        if media_path is None:
            for alias in ("filename", "file_name"):
                val = _lookup(alias)
                if val is not None:
                    cand = str(val).strip()
                    if Path(cand).suffix.lower() in media_extensions or cfg.modality in (ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.VIDEO, ModalityType.MULTIMODAL):
                        media_path = cand
                    break
        if media_path is None:
            val = _lookup("path")
            if val is not None:
                cand = str(val).strip()
                if Path(cand).suffix.lower() in media_extensions:
                    media_path = cand
        if media_path is None:
            for alias in ("image", "audio", "video"):
                val = _lookup(alias)
                candidate = _media_reference(val)
                if candidate is not None:
                    media_path = candidate
                    break

    return DatasetItem(
        id=row.get("id", idx),
        text=text,
        label=label,
        media_path=media_path,
        metadata=copy.deepcopy(metadata),
        raw_data=copy.deepcopy(row),
    )


def _apply_label_dtype(label: Any, label_dtype: str) -> Any:
    """Cast a single label value according to label_dtype."""
    if label is None:
        return None
    if label_dtype == "str":
        return str(label)
    if label_dtype == "int":
        return int(label)
    if label_dtype == "float":
        return float(label)
    if label_dtype == "auto":
        if isinstance(label, bool):
            return str(label)
        if isinstance(label, (int, float)):
            return label
        s = str(label).strip()
        try:
            return int(s)
        except (ValueError, TypeError):
            pass
        try:
            return float(s)
        except (ValueError, TypeError):
            pass
        return str(label)
    return label


def _promote_modality_from_items(items: List[DatasetItem], cfg: DatasetConfig) -> None:
    """Promote a TEXT modality to AUDIO/VIDEO/IMAGE/MULTIMODAL based on loaded items."""
    if cfg.kwargs.get("_explicit_modality", False):
        return
    if cfg.modality != ModalityType.TEXT:
        return
    has_media = any(it.media_path for it in items)
    has_text = any(it.text for it in items)
    if not has_media:
        return
    first_ext = None
    for it in items:
        if it.media_path:
            first_ext = Path(it.media_path).suffix.lower()
            break
    if first_ext in AUDIO_EXTENSIONS:
        cfg.modality = ModalityType.AUDIO
    elif first_ext in VIDEO_EXTENSIONS:
        cfg.modality = ModalityType.VIDEO
    elif has_text:
        cfg.modality = ModalityType.MULTIMODAL
    else:
        cfg.modality = ModalityType.IMAGE


class LocalFileStream:
    """Re-iterable file stream for local structured files."""
    def __init__(self, path: Path, ext: str, cfg: DatasetConfig, state: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.ext = ext
        self.cfg = cfg
        self.state = state if state is not None else {}
        self._recorded_lines = set()
        self.inferred_label_dtype = self.cfg.label_dtype
        if self.cfg.label_dtype == "auto" and self.ext in (".csv", ".tsv", ".jsonl", ".ndjson"):
            self.inferred_label_dtype = self._infer_label_dtype()

    def _infer_label_dtype(self) -> str:
        """Infer column-wide label dtype matching eager mode."""
        labels = []
        try:
            if self.ext in (".csv", ".tsv"):
                delimiter = "\t" if self.ext == ".tsv" else ","
                with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for r in reader:
                        item = row_to_dataset_item(r, 0, self.cfg)
                        if item.label is not None:
                            labels.append(item.label)
            elif self.ext in (".jsonl", ".ndjson"):
                with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                r = json.loads(line)
                                if isinstance(r, dict):
                                    item = row_to_dataset_item(r, 0, self.cfg)
                                    if item.label is not None:
                                        labels.append(item.label)
                            except Exception:
                                pass
        except Exception:
            return "str"

        if not labels:
            return "str"

        can_int = True
        for lbl in labels:
            if isinstance(lbl, bool):
                can_int = False
                break
            try:
                s = str(lbl).strip()
                int(s)
            except (ValueError, TypeError):
                can_int = False
                break
        if can_int:
            return "int"

        can_float = True
        for lbl in labels:
            if isinstance(lbl, bool):
                can_float = False
                break
            try:
                s = str(lbl).strip()
                float(s)
            except (ValueError, TypeError):
                can_float = False
                break
        if can_float:
            return "float"

        return "str"

    def _record_error(self, line: Any, error: str) -> None:
        if line in self._recorded_lines:
            return
        self._recorded_lines.add(line)
        self.state.setdefault("parsing_errors", []).append({"line": line, "error": error})
        self.state["skipped_samples"] = len(self.state["parsing_errors"])

    def __iter__(self):
        if self.ext in (".jsonl", ".ndjson"):
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        yield row_to_dataset_item(row if isinstance(row, dict) else {"text": str(row)}, idx, self.cfg)
                    except json.JSONDecodeError as err:
                        if self.cfg.error_policy == ErrorPolicy.STRICT:
                            raise ValueError(f"Malformed JSON on line {idx + 1} of {self.path}: {line}") from err
                        self._record_error(idx + 1, f"Malformed JSON: {err}")
        elif self.ext in (".csv", ".tsv"):
            delimiter = "\t" if self.ext == ".tsv" else ","
            with open(self.path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter, strict=True)
                idx = 0
                while True:
                    try:
                        row = next(reader)
                    except StopIteration:
                        break
                    except csv.Error as err:
                        line_num = max(1, getattr(reader, "line_num", idx + 2))
                        message = f"Malformed CSV record near line {line_num} of {self.path}: {err}"
                        if self.cfg.error_policy == ErrorPolicy.STRICT:
                            raise ValueError(message) from err
                        if self.cfg.error_policy == ErrorPolicy.REPORT_AND_SKIP:
                            self._record_error(line_num, message)
                        else:
                            self.state.setdefault("warnings", []).append(message)
                        # An unterminated quoted field makes record-boundary recovery ambiguous.
                        break
                    yield row_to_dataset_item(row, idx, self.cfg)
                    idx += 1
        elif self.ext == ".json":
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                # Find the first non-whitespace character without reading the document.
                first = ""
                while True:
                    ch = f.read(1)
                    if not ch:
                        break
                    if not ch.isspace():
                        first = ch
                        break
                if first == "[":
                    # Incrementally decode top-level JSON arrays, retaining only a small
                    # input buffer and the currently decoded value.
                    decoder = json.JSONDecoder()
                    buffer = "["
                    pos = 1
                    eof = False
                    idx = 0
                    first_value = True
                    while True:
                        # Keep enough data available to recognize delimiters and parse the
                        # next item. raw_decode raises until a complete JSON value exists.
                        while True:
                            while pos < len(buffer) and buffer[pos].isspace():
                                pos += 1
                            if pos < len(buffer):
                                break
                            if eof:
                                raise ValueError(f"Unexpected end of JSON array in {self.path}")
                            chunk = f.read(4096)
                            if chunk:
                                buffer, pos = buffer[pos:] + chunk, 0
                            else:
                                eof = True
                        if buffer[pos] == "]":
                            pos += 1
                            trailing = buffer[pos:] + f.read()
                            if trailing.strip():
                                raise ValueError(f"Unexpected content after JSON array in {self.path}")
                            break
                        if not first_value:
                            if buffer[pos] != ",":
                                raise ValueError(f"Expected ',' or ']' in JSON array {self.path}")
                            pos += 1
                            while True:
                                while pos < len(buffer) and buffer[pos].isspace():
                                    pos += 1
                                if pos < len(buffer):
                                    break
                                if eof:
                                    raise ValueError(f"Unexpected end after ',' in JSON array {self.path}")
                                chunk = f.read(4096)
                                if chunk:
                                    buffer, pos = buffer[pos:] + chunk, 0
                                else:
                                    eof = True
                        while True:
                            try:
                                value, end_pos = decoder.raw_decode(buffer, pos)
                                break
                            except json.JSONDecodeError:
                                if eof:
                                    raise ValueError(f"Malformed JSON array in {self.path}")
                                chunk = f.read(4096)
                                if not chunk:
                                    eof = True
                                else:
                                    # Retain the current value prefix, but not any already
                                    # consumed prefix; this bounds memory to the largest row.
                                    if pos:
                                        buffer, pos = buffer[pos:] + chunk, 0
                                    else:
                                        buffer += chunk
                        pos = end_pos
                        if isinstance(value, dict):
                            yield row_to_dataset_item(value, idx, self.cfg)
                        else:
                            yield DatasetItem(id=idx, text=str(value), raw_data={"value": copy.deepcopy(value)})
                        idx += 1
                        first_value = False
                        if pos >= 4096:
                            buffer, pos = buffer[pos:], 0
                else:
                    # Non-array JSON is treated as one record. It cannot be item-streamed
                    # generically without a schema; retain compatibility as a single row.
                    rest = first + f.read()
                    try:
                        raw_data = json.loads(rest)
                    except json.JSONDecodeError as err:
                        if self.cfg.error_policy == ErrorPolicy.STRICT:
                            raise ValueError(f"Malformed JSON in file {self.path}: {err}") from err
                        self._record_error(0, f"Malformed JSON: {err}")
                        return
                    if isinstance(raw_data, dict):
                        yield row_to_dataset_item(raw_data, 0, self.cfg)
                    else:
                        yield DatasetItem(id=0, text=str(raw_data), raw_data={"value": copy.deepcopy(raw_data)})
        elif self.ext in (".txt", ".md"):
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                idx = 0
                for line in f:
                    line = line.strip()
                    if line:
                        yield DatasetItem(id=idx, text=line, raw_data={"text": line})
                        idx += 1
        elif self.ext in (".yaml", ".yml"):
            import yaml
            # Parse a top-level sequence one element at a time. A scalar/mapping root
            # is naturally one record and is constructed only once.
            with open(self.path, "r", encoding="utf-8") as f:
                loader = yaml.SafeLoader(f)
                idx = 0
                try:
                    event = loader.get_event()
                    if not isinstance(event, yaml.events.StreamStartEvent):
                        raise ValueError(f"Invalid YAML stream in {self.path}")
                    while True:
                        event = loader.get_event()
                        if isinstance(event, yaml.events.StreamEndEvent):
                            break
                        if not isinstance(event, yaml.events.DocumentStartEvent):
                            continue
                        root_event = loader.peek_event()
                        if isinstance(root_event, yaml.events.SequenceStartEvent):
                            loader.get_event()  # consume the root sequence start
                            while not isinstance(loader.peek_event(), yaml.events.SequenceEndEvent):
                                node = loader.compose_node(None, None)
                                row = loader.construct_document(node)
                                if isinstance(row, dict):
                                    yield row_to_dataset_item(row, idx, self.cfg)
                                else:
                                    yield DatasetItem(id=idx, text=str(row), raw_data={"text": str(row)})
                                idx += 1
                            loader.get_event()  # consume sequence end
                        else:
                            node = loader.compose_node(None, None)
                            row = loader.construct_document(node)
                            if isinstance(row, dict):
                                yield row_to_dataset_item(row, idx, self.cfg)
                            else:
                                yield DatasetItem(id=idx, text=str(row), raw_data={"text": str(row)})
                            idx += 1
                        # Consume document end event and continue to next YAML document.
                        if isinstance(loader.peek_event(), yaml.events.DocumentEndEvent):
                            loader.get_event()
                finally:
                    loader.dispose()

    def __len__(self) -> int:
        if self.ext in (".csv", ".tsv"):
            delimiter = "\t" if self.ext == ".tsv" else ","
            count = 0
            with open(self.path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter, strict=True)
                try:
                    next(reader, None)  # header
                    while True:
                        try:
                            next(reader)
                            count += 1
                        except StopIteration:
                            return min(count, self.cfg.max_samples) if self.cfg.max_samples is not None else count
                        except csv.Error:
                            if self.cfg.error_policy == ErrorPolicy.STRICT:
                                raise ValueError(f"Malformed CSV while counting records in {self.path}: {reader.line_num}")
                            return min(count, self.cfg.max_samples) if self.cfg.max_samples is not None else count
                except csv.Error as err:
                    if self.cfg.error_policy == ErrorPolicy.STRICT:
                        raise ValueError(f"Malformed CSV while counting records in {self.path}: {err}") from err
                    return min(count, self.cfg.max_samples) if self.cfg.max_samples is not None else count
        elif self.ext in (".txt", ".md", ".jsonl", ".ndjson"):
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                count = sum(1 for line in f if line.strip())
            return min(count, self.cfg.max_samples) if self.cfg.max_samples is not None else count
        elif self.ext == ".json":
            # Count records with the incremental iterator, never by materializing the JSON array.
            return sum(1 for _ in self)
        raise TypeError(f"Streaming length not supported for extension {self.ext}")

# ========================================
# 1. Local File Parser
# ========================================
def parse_local_file(
    file_path: Union[str, Path],
    cfg: DatasetConfig,
) -> Tuple[List[DatasetItem], List[str], Dict[str, Any]]:
    """Parses a local structured file or single media file into DatasetItems."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local file does not exist: {path}")

    ext = path.suffix.lower()
    items: List[DatasetItem] = []
    columns: List[str] = []
    parsing_errors: List[Dict[str, Any]] = []

    # ========================================
    # A. Single Media File Dispatch
    # ========================================
    is_explicit_modality = cfg.kwargs.get("_explicit_modality", False)

    if ext in IMAGE_EXTENSIONS:
        if is_explicit_modality and cfg.modality != ModalityType.IMAGE:
            if cfg.modality == ModalityType.TEXT and cfg.treat_path_as_text:
                pass
            else:
                raise ValueError(
                    f"File '{path.name}' is an image ({ext}), but requested modality is '{cfg.modality.value}'. "
                    "Pass treat_path_as_text=True to treat image paths as plain text."
                )
        elif not is_explicit_modality:
            cfg.modality = ModalityType.IMAGE

        if cfg.error_policy != ErrorPolicy.PERMISSIVE:
            try:
                load_image(str(path))
            except Exception as e:
                message = f"Corrupted or unreadable image file '{path}': {str(e)}"
                if cfg.error_policy == ErrorPolicy.STRICT:
                    raise ValueError(message) from e
                return [], ["media_path"], {
                    "file_path": str(path), "media_type": "image",
                    "skipped_samples": 1, "parsing_errors": [{"id": 0, "media_path": str(path), "error": message}],
                }

        item = DatasetItem(
            id=0,
            text=str(path) if cfg.modality == ModalityType.TEXT else None,
            media_path=str(path),
            metadata={"filename": path.name, "format": ext, "size_bytes": path.stat().st_size},
            raw_data={"media_path": str(path), "text": str(path)},
        )
        media_type = "text" if cfg.modality == ModalityType.TEXT else "image"
        cols = ["text", "media_path"] if cfg.modality == ModalityType.TEXT else ["media_path"]
        return [item], cols, {"file_path": str(path), "media_type": media_type}

    elif ext in AUDIO_EXTENSIONS:
        if is_explicit_modality and cfg.modality != ModalityType.AUDIO:
            if cfg.modality == ModalityType.TEXT and cfg.treat_path_as_text:
                pass
            else:
                raise ValueError(
                    f"File '{path.name}' is an audio file ({ext}), but requested modality is '{cfg.modality.value}'. "
                    "Pass treat_path_as_text=True to treat audio paths as plain text."
                )
        elif not is_explicit_modality:
            cfg.modality = ModalityType.AUDIO

        if cfg.error_policy != ErrorPolicy.PERMISSIVE:
            try:
                load_audio(str(path), target_sr=cfg.target_sr)
            except Exception as e:
                message = f"Corrupted or unreadable audio file '{path}': {str(e)}"
                if cfg.error_policy == ErrorPolicy.STRICT:
                    raise ValueError(message) from e
                return [], ["media_path"], {
                    "file_path": str(path), "media_type": "audio",
                    "skipped_samples": 1, "parsing_errors": [{"id": 0, "media_path": str(path), "error": message}],
                }

        item = DatasetItem(
            id=0,
            text=str(path) if cfg.modality == ModalityType.TEXT else None,
            media_path=str(path),
            metadata={"filename": path.name, "format": ext, "size_bytes": path.stat().st_size},
            raw_data={"media_path": str(path), "text": str(path)},
        )
        media_type = "text" if cfg.modality == ModalityType.TEXT else "audio"
        cols = ["text", "media_path"] if cfg.modality == ModalityType.TEXT else ["media_path"]
        return [item], cols, {"file_path": str(path), "media_type": media_type}

    elif ext in VIDEO_EXTENSIONS:
        if is_explicit_modality and cfg.modality != ModalityType.VIDEO:
            if cfg.modality == ModalityType.TEXT and cfg.treat_path_as_text:
                pass
            else:
                raise ValueError(
                    f"File '{path.name}' is a video file ({ext}), but requested modality is '{cfg.modality.value}'. "
                    "Pass treat_path_as_text=True to treat video paths as plain text."
                )
        elif not is_explicit_modality:
            cfg.modality = ModalityType.VIDEO

        if cfg.error_policy != ErrorPolicy.PERMISSIVE:
            try:
                load_video(str(path), max_frames=cfg.max_video_frames)
            except Exception as e:
                message = f"Corrupted or unreadable video file '{path}': {str(e)}"
                if cfg.error_policy == ErrorPolicy.STRICT:
                    raise ValueError(message) from e
                return [], ["media_path"], {
                    "file_path": str(path), "media_type": "video",
                    "skipped_samples": 1, "parsing_errors": [{"id": 0, "media_path": str(path), "error": message}],
                }

        item = DatasetItem(
            id=0,
            text=str(path) if cfg.modality == ModalityType.TEXT else None,
            media_path=str(path),
            metadata={"filename": path.name, "format": ext, "size_bytes": path.stat().st_size},
            raw_data={"media_path": str(path), "text": str(path)},
        )
        media_type = "text" if cfg.modality == ModalityType.TEXT else "video"
        cols = ["text", "media_path"] if cfg.modality == ModalityType.TEXT else ["media_path"]
        return [item], cols, {"file_path": str(path), "media_type": media_type}

    # ========================================
    # B. Streaming for Structured & Text Files
    # ========================================
    if cfg.streaming and ext in (".csv", ".tsv", ".jsonl", ".ndjson", ".txt", ".md", ".yaml", ".yml", ".json"):
        state: Dict[str, Any] = {
            "file_path": str(path),
            "filename": path.name,
            "format": ext,
            "size_bytes": path.stat().st_size,
            "streaming": True,
            "parsing_errors": [],
            "skipped_samples": 0,
        }
        stream = LocalFileStream(path, ext, cfg, state)
        seq = LazyDatasetSequence(stream, cfg, base_dir=path.parent, state=state)
        if ext not in (".yaml", ".yml"):
            first_peek = seq.peek(1)
            cols = list(first_peek[0].raw_data.keys()) if first_peek and first_peek[0].raw_data else ["text"]
            # Streaming must apply the same modality promotion as the eager parser.
            if first_peek:
                _promote_modality_from_items(first_peek, cfg)
        else:
            cols = []
        return seq, cols, state

    # ========================================
    # C. Eager Structured & Text Files
    # ========================================
    warnings: List[str] = []

    if ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if not columns and isinstance(row, dict):
                        columns = list(row.keys())
                    items.append(row_to_dataset_item(row if isinstance(row, dict) else {"text": str(row)}, idx, cfg))
                    if cfg.max_samples and len(items) >= cfg.max_samples:
                        break
                except json.JSONDecodeError as err:
                    if cfg.error_policy == ErrorPolicy.STRICT:
                        raise ValueError(f"Malformed JSON on line {idx + 1} of {path}: {line}") from err
                    parsing_errors.append({"line": idx + 1, "error": str(err), "sample_snippet": line[:80]})

    elif ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON in file {path}: {str(err)}") from err

        data_list = raw_data if isinstance(raw_data, list) else [raw_data]
        for idx, row in enumerate(data_list):
            if isinstance(row, dict):
                if not columns:
                    columns = list(row.keys())
                items.append(row_to_dataset_item(row, idx, cfg))
            else:
                items.append(DatasetItem(id=idx, text=str(row), raw_data={"value": row}))
            if cfg.max_samples and len(items) >= cfg.max_samples:
                break

    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=delimiter, strict=True)
                columns = reader.fieldnames or []
                for idx, row in enumerate(reader):
                    items.append(row_to_dataset_item(row, idx, cfg))
                    if cfg.max_samples and len(items) >= cfg.max_samples:
                        break
        except csv.Error as err:
            if cfg.error_policy == ErrorPolicy.STRICT:
                raise ValueError(f"Malformed CSV in file {path}: {str(err)}") from err
            parsing_errors.append({"line": getattr(reader, "line_num", 0), "error": str(err)})

    elif ext == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            columns = list(df.columns)
            if cfg.max_samples:
                df = df.head(cfg.max_samples)
            for idx, row in df.iterrows():
                items.append(row_to_dataset_item(row.to_dict(), idx, cfg))
        except ImportError:
            raise ImportError("pandas and pyarrow are required to read .parquet files. Run: pip install pandas pyarrow")

    elif ext in (".txt", ".md") or ext == "":
        try:
            with open(path, "r", encoding="utf-8", errors="strict" if ext == "" else "replace") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            columns = ["text"]
            for idx, line in enumerate(lines):
                items.append(DatasetItem(id=idx, text=line, raw_data={"text": line}))
                if cfg.max_samples and len(items) >= cfg.max_samples:
                    break
        except UnicodeDecodeError:
            raise ValueError(
                f"Unsupported binary file format without extension '{path.name}'. Supported formats: "
                f"{IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | TEXT_EXTENSIONS}"
            )

    elif ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            data_list = data if isinstance(data, list) else [data]
            for idx, row in enumerate(data_list):
                if isinstance(row, dict):
                    if not columns:
                        columns = list(row.keys())
                    items.append(row_to_dataset_item(row, idx, cfg))
                else:
                    if not columns:
                        columns = ["text"]
                    items.append(DatasetItem(id=idx, text=str(row), raw_data={"text": str(row)}))
                if cfg.max_samples and len(items) >= cfg.max_samples:
                    break
        except ImportError:
            raise ImportError("PyYAML is required to read .yaml files. Run: pip install pyyaml")

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: "
            f"{IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | TEXT_EXTENSIONS}"
        )

    # 1. Resolve media paths and validate existence
    base_dir = path.parent
    valid_items: List[DatasetItem] = []
    for it in items:
        if it.media_path:
            p_cand = Path(it.media_path)
            if not p_cand.is_absolute():
                resolved = (base_dir / p_cand).resolve()
            else:
                resolved = p_cand.resolve()
            if resolved.is_file():
                try:
                    ext = resolved.suffix.lower()
                    if cfg.error_policy != ErrorPolicy.PERMISSIVE:
                        if ext in IMAGE_EXTENSIONS:
                            load_image(resolved)
                        elif ext in AUDIO_EXTENSIONS:
                            load_audio(resolved, target_sr=cfg.target_sr)
                        elif ext in VIDEO_EXTENSIONS:
                            load_video(resolved, max_frames=cfg.max_video_frames)
                    it.media_path = str(resolved)
                    valid_items.append(it)
                except Exception as exc:
                    if cfg.error_policy == ErrorPolicy.STRICT:
                        raise ValueError(
                            f"Corrupted or unreadable media file '{resolved}' referenced in '{path.name}': {exc}"
                        ) from exc
                    parsing_errors.append({"line": it.id, "error": f"Unreadable media file '{resolved}': {exc}"})
            else:
                if cfg.error_policy == ErrorPolicy.STRICT:
                    raise FileNotFoundError(
                        f"Media file '{it.media_path}' referenced in '{path.name}' does not exist (resolved path: '{resolved}')."
                    )
                else:
                    parsing_errors.append({"line": it.id, "error": f"Media file '{it.media_path}' not found."})
        else:
            valid_items.append(it)
    items = valid_items

    # 2. Label dtype inference and casting
    label_info = {"inferred_dtype": "str"}
    if cfg.label_dtype == "str":
        for it in items:
            if it.label is not None:
                it.label = str(it.label)
        label_info = {"inferred_dtype": "str"}
    elif cfg.label_dtype == "int":
        for it in items:
            if it.label is not None:
                try:
                    it.label = int(it.label)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to cast label '{it.label}' to int: {e}") from e
        label_info = {"inferred_dtype": "int"}
    elif cfg.label_dtype == "float":
        for it in items:
            if it.label is not None:
                try:
                    it.label = float(it.label)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to cast label '{it.label}' to float: {e}") from e
        label_info = {"inferred_dtype": "float"}
    elif cfg.label_dtype == "auto":
        non_none = [it for it in items if it.label is not None]
        if non_none:
            can_int = True
            for it in non_none:
                if isinstance(it.label, bool):
                    can_int = False
                    break
                try:
                    s = str(it.label).strip()
                    int(s)
                except (ValueError, TypeError):
                    can_int = False
                    break
            if can_int:
                for it in non_none:
                    it.label = int(it.label)
                label_info = {"inferred_dtype": "int"}
            else:
                can_float = True
                for it in non_none:
                    if isinstance(it.label, bool):
                        can_float = False
                        break
                    try:
                        s = str(it.label).strip()
                        float(s)
                    except (ValueError, TypeError):
                        can_float = False
                        break
                if can_float:
                    for it in non_none:
                        it.label = float(it.label)
                    label_info = {"inferred_dtype": "float"}
                else:
                    for it in non_none:
                        it.label = str(it.label)
                    label_info = {"inferred_dtype": "str"}

    # 3. Detect modality promotion if table contains media
    if not is_explicit_modality and cfg.modality == ModalityType.TEXT:
        has_media = any(it.media_path for it in items)
        has_text = any(it.text for it in items)
        if has_media and has_text:
            first_ext = None
            for it in items:
                if it.media_path:
                    first_ext = Path(it.media_path).suffix.lower()
                    break
            if first_ext in AUDIO_EXTENSIONS:
                cfg.modality = ModalityType.AUDIO
            elif first_ext in VIDEO_EXTENSIONS:
                cfg.modality = ModalityType.VIDEO
            else:
                cfg.modality = ModalityType.MULTIMODAL
        elif has_media and not has_text:
            first_ext = None
            for it in items:
                if it.media_path:
                    first_ext = Path(it.media_path).suffix.lower()
                    break
            if first_ext in AUDIO_EXTENSIONS:
                cfg.modality = ModalityType.AUDIO
            elif first_ext in VIDEO_EXTENSIONS:
                cfg.modality = ModalityType.VIDEO
            else:
                cfg.modality = ModalityType.IMAGE

    if len(items) == 0:
        warnings.append(f"Loaded dataset from '{path.name}' is empty (0 samples).")

    metadata = {
        "file_path": str(path),
        "filename": path.name,
        "format": ext,
        "size_bytes": path.stat().st_size,
        "parsing_errors": parsing_errors,
        "skipped_samples": len(parsing_errors),
        "label_inference": label_info,
        "warnings": warnings,
        "warning": warnings[0] if warnings else None,
    }
    return items, columns, metadata


# ========================================
# 2. Local Directory Parser
# ========================================

def _scan_media_directory(
    root: Path,
    ext_set: set,
    modality: ModalityType,
    cfg: DatasetConfig,
) -> Tuple[List[DatasetItem], List[Dict[str, Any]]]:
    """Collect media files, eagerly validating them when an error policy was explicitly requested."""
    items: List[DatasetItem] = []
    skipped: List[Dict[str, Any]] = []
    enforce = cfg.error_policy != ErrorPolicy.PERMISSIVE
    idx = 0
    for p in root.glob("**/*"):
        if not (p.is_file() and p.suffix.lower() in ext_set):
            continue
        path_str = str(p.resolve())
        label = p.parent.name if p.parent != root else None
        if enforce:
            try:
                if modality == ModalityType.IMAGE:
                    load_image(path_str)
                elif modality == ModalityType.AUDIO:
                    load_audio(path_str, target_sr=cfg.target_sr)
                elif modality == ModalityType.VIDEO:
                    load_video(path_str, max_frames=cfg.max_video_frames)
            except Exception as e:
                if cfg.error_policy == ErrorPolicy.STRICT:
                    raise ValueError(f"Failed to load {modality.value} '{path_str}': {str(e)}") from e
                skipped.append({"id": idx, "media_path": path_str, "error": str(e)})
                idx += 1
                continue
        items.append(
            DatasetItem(
                id=idx,
                media_path=path_str,
                label=label,
                metadata={"filename": p.name, "parent_dir": p.parent.name},
                raw_data={"media_path": path_str, "label": label},
            )
        )
        idx += 1
        if cfg.max_samples and len(items) >= cfg.max_samples:
            break
    return items, skipped


def parse_local_directory(
    dir_path: Union[str, Path],
    cfg: DatasetConfig,
) -> Tuple[List[DatasetItem], List[str], Dict[str, Any]]:
    """Parses a directory with profiling and manifest detection."""
    root = Path(dir_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {root}")

    # ========================================
    # Profile directory if modality was not explicitly forced
    # ========================================
    profile = profile_directory(root)

    # 1. Check for Manifest Table (e.g., Kaggle competitions or table-referenced datasets)
    if profile["manifests"] and cfg.modality not in (ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.VIDEO):
        target_split = (cfg.split or "train").lower()
        manifest_path = None
        for m in profile["manifests"]:
            m_stem = Path(m).stem.lower()
            if m_stem == target_split or target_split in m_stem:
                manifest_path = m
                break
        if manifest_path is None:
            manifest_path = profile["manifests"][0]

        items, cols, meta = parse_local_file(manifest_path, cfg)
        if len(profile["manifests"]) > 1:
            multi_warn = (
                f"Multiple manifest files found in directory: {profile['manifests']}. "
                f"Selected '{manifest_path}'. Specify 'split' to disambiguate."
            )
            meta.setdefault("warnings", []).append(multi_warn)
            if not meta.get("warning"):
                meta["warning"] = multi_warn

        has_media = any(it.media_path for it in items)
        has_text = any(it.text for it in items)
        if has_media and has_text and not cfg.kwargs.get("_explicit_modality"):
            cfg.modality = ModalityType.MULTIMODAL
            meta["media_type"] = "multimodal"
        meta["manifest_used"] = manifest_path
        meta["directory"] = str(root)
        return items, cols, meta

    detected_modalities = [
        mod for mod, cnt in [
            ("image", profile["image"]),
            ("audio", profile["audio"]),
            ("video", profile["video"]),
            ("text", profile["text"]),
        ] if cnt > 0
    ]

    # 2. Automatic Modality Inference if left as default TEXT
    if cfg.modality == ModalityType.TEXT and not cfg.kwargs.get("_explicit_modality"):
        if profile["image"] > 0 and profile["audio"] == 0 and profile["video"] == 0:
            cfg.modality = ModalityType.IMAGE
        elif profile["audio"] > 0 and profile["image"] == 0 and profile["video"] == 0:
            cfg.modality = ModalityType.AUDIO
        elif profile["video"] > 0 and profile["image"] == 0 and profile["audio"] == 0:
            cfg.modality = ModalityType.VIDEO

    # Build warning and file lists for mixed modalities
    warnings: List[str] = []
    included_files: List[str] = []
    excluded_files: List[str] = []

    if len(detected_modalities) > 1:
        current_mod_key = cfg.modality.value
        for mod_key, files in profile.get("files_by_modality", {}).items():
            if mod_key == current_mod_key:
                included_files.extend(files)
            else:
                excluded_files.extend(files)
        if excluded_files:
            msg = (
                f"Directory contains mixed modalities: {detected_modalities}. "
                f"Auto-selected modality: '{cfg.modality.value}'. "
                f"Excluded {len(excluded_files)} files of other modalities: {[Path(f).name for f in excluded_files[:10]]}."
            )
            warnings.append(msg)

    items: List[DatasetItem] = []
    columns: List[str] = []
    idx = 0

    # Vision Scan
    if cfg.modality == ModalityType.IMAGE:
        columns = ["media_path", "label", "filename"]
        items, skipped = _scan_media_directory(root, IMAGE_EXTENSIONS, ModalityType.IMAGE, cfg)
        if not items and not skipped:
            raise ValueError(f"No image files found in directory: {root}")
        return items, columns, {
            "directory": str(root),
            "media_type": "image",
            "profile": profile,
            "warnings": warnings,
            "warning": warnings[0] if warnings else None,
            "included_files": included_files,
            "excluded_files": excluded_files,
            "skipped_samples": len(skipped),
            "parsing_errors": skipped,
        }

    # Audio Scan
    elif cfg.modality == ModalityType.AUDIO:
        columns = ["media_path", "label", "filename"]
        items, skipped = _scan_media_directory(root, AUDIO_EXTENSIONS, ModalityType.AUDIO, cfg)
        if not items and not skipped:
            raise ValueError(f"No audio files found in directory: {root}")
        return items, columns, {
            "directory": str(root),
            "media_type": "audio",
            "profile": profile,
            "warnings": warnings,
            "warning": warnings[0] if warnings else None,
            "included_files": included_files,
            "excluded_files": excluded_files,
            "skipped_samples": len(skipped),
            "parsing_errors": skipped,
        }

    # Video Scan
    elif cfg.modality == ModalityType.VIDEO:
        columns = ["media_path", "label", "filename"]
        items, skipped = _scan_media_directory(root, VIDEO_EXTENSIONS, ModalityType.VIDEO, cfg)
        if not items and not skipped:
            raise ValueError(f"No video files found in directory: {root}")
        return items, columns, {
            "directory": str(root),
            "media_type": "video",
            "profile": profile,
            "warnings": warnings,
            "warning": warnings[0] if warnings else None,
            "included_files": included_files,
            "excluded_files": excluded_files,
            "skipped_samples": len(skipped),
            "parsing_errors": skipped,
        }

    # Text Document Collection
    columns = ["text", "filename"]
    for p in root.glob("**/*"):
        if p.is_file() and p.suffix.lower() in (".txt", ".md"):
            try:
                content = p.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    items.append(
                        DatasetItem(
                            id=idx,
                            text=content,
                            metadata={"filename": p.name, "chars": len(content)},
                            raw_data={"text": content, "filename": p.name},
                        )
                    )
                    idx += 1
                    if cfg.max_samples and len(items) >= cfg.max_samples:
                        break
            except Exception as e:
                if cfg.error_policy == ErrorPolicy.STRICT:
                    raise RuntimeError(f"Failed to read file {p}: {str(e)}") from e

    if not items:
        raise ValueError(
            f"No compatible dataset files found in directory '{root}'. "
            f"Directory profile detected: {profile}. Please specify an explicit modality or ensure files exist."
        )

    return items, columns, {
        "directory": str(root),
        "media_type": "text",
        "profile": profile,
        "warnings": warnings,
        "warning": warnings[0] if warnings else None,
        "included_files": included_files,
        "excluded_files": excluded_files,
    }


# ========================================
# 3. Hugging Face Dataset Parser
# ========================================

class LazyDatasetSequence:
    """Lazy iterator that processes dataset items on-the-fly without eager materialization."""
    def __init__(
        self,
        stream_source: Any,
        cfg: DatasetConfig,
        base_dir: Optional[Path] = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        self.stream_source = stream_source
        self.cfg = cfg
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.state = state if state is not None else {}
        self._recorded_errors = set()
        self._is_one_shot = (iter(stream_source) is stream_source)
        self._source_iter = None
        self._buffer: List[DatasetItem] = []
        self._consumed = False
        self._epoch = 0
        self._idx = 0
        self.inferred_label_dtype = getattr(stream_source, "inferred_label_dtype", cfg.label_dtype)

    def _record_error(self, key: Any, error: str) -> None:
        if key in self._recorded_errors:
            return
        self._recorded_errors.add(key)
        self.state.setdefault("parsing_errors", []).append({"line": key, "error": error})
        self.state["skipped_samples"] = len(self.state["parsing_errors"])

    def _finalize_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        record_key = item.metadata.get("_data_hub_source_index", item.id)
        # 1. Resolve relative media paths and enforce the error policy on missing media.
        if item.media_path:
            p_cand = Path(item.media_path)
            if not p_cand.is_absolute() and self.base_dir is not None:
                resolved = (self.base_dir / p_cand).resolve()
            else:
                resolved = p_cand.resolve()
            if resolved.is_file():
                item.media_path = str(resolved)
                ext = resolved.suffix.lower()
                if self.cfg.error_policy != ErrorPolicy.PERMISSIVE:
                    try:
                        if ext in IMAGE_EXTENSIONS:
                            load_image(resolved)
                        elif ext in AUDIO_EXTENSIONS:
                            load_audio(resolved, target_sr=self.cfg.target_sr)
                        elif ext in VIDEO_EXTENSIONS:
                            load_video(resolved, max_frames=self.cfg.max_video_frames)
                    except Exception as exc:
                        message = f"Unreadable media file '{resolved}': {exc}"
                        if self.cfg.error_policy == ErrorPolicy.STRICT:
                            raise ValueError(f"Sample {item.id!r} references {message}") from exc
                        if self.cfg.error_policy == ErrorPolicy.REPORT_AND_SKIP:
                            self._record_error(record_key, message)
                            return None
                        self.state.setdefault("warnings", []).append(message)
                        item.metadata = copy.deepcopy(item.metadata)
                        item.metadata["media_warning"] = message
                        item.media_path = None
            else:
                message = f"Media file '{item.media_path}' referenced in the streaming source does not exist (resolved path: '{resolved}')."
                if self.cfg.error_policy == ErrorPolicy.STRICT:
                    raise FileNotFoundError(message)
                if self.cfg.error_policy == ErrorPolicy.REPORT_AND_SKIP:
                    self._record_error(record_key, message)
                    return None
                self.state.setdefault("warnings", []).append(message)
                item.metadata = copy.deepcopy(item.metadata)
                item.metadata["media_warning"] = message
                item.media_path = None
        # Embedded media payloads (PIL images, bytes, HF Audio dictionaries, etc.)
        # must receive the same error-policy treatment as path-based media.
        if not item.media_path and item.raw_data:
            for key, kind in (("image", "image"), ("audio", "audio"), ("video", "video")):
                payload = item.raw_data.get(key)
                if payload is None:
                    continue
                try:
                    if kind == "image":
                        load_image(payload)
                    elif kind == "audio":
                        load_audio(payload, target_sr=self.cfg.target_sr)
                    else:
                        raise ValueError("Embedded video payloads are unsupported; provide a video media_path.")
                except Exception as exc:
                    message = f"Invalid embedded {kind} payload for sample {item.id!r}: {exc}"
                    if self.cfg.error_policy == ErrorPolicy.STRICT:
                        raise ValueError(message) from exc
                    if self.cfg.error_policy == ErrorPolicy.REPORT_AND_SKIP:
                        self._record_error(record_key, message)
                        return None
                    self.state.setdefault("warnings", []).append(message)
                    item.metadata = copy.deepcopy(item.metadata)
                    item.metadata["media_warning"] = message
                break

        # 2. Label dtype casting (auto uses per-value inference or column inferred dtype).
        if item.label is not None:
            # In-memory labels are already typed values. "auto" must not reinterpret
            # strings such as "001" as integers; keep eager and streaming memory loads equivalent.
            if not (self.cfg.source_type == SourceType.MEMORY and self.cfg.label_dtype == "auto"):
                dtype_to_use = self.inferred_label_dtype if self.inferred_label_dtype != "auto" else self.cfg.label_dtype
                item.label = _apply_label_dtype(item.label, dtype_to_use)
        # 3. Dynamic modality promotion for streaming sources. Embedded payloads
        # count too; modality must not depend on whether a file path exists.
        if self.cfg.modality == ModalityType.TEXT and not self.cfg.kwargs.get("_explicit_modality"):
            raw = item.raw_data or {}
            media_path = item.media_path
            ext = Path(media_path).suffix.lower() if media_path else ""
            has_audio = bool(ext in AUDIO_EXTENSIONS or raw.get("audio") is not None or raw.get("audio_path") is not None)
            has_video = bool(ext in VIDEO_EXTENSIONS or raw.get("video") is not None or raw.get("video_path") is not None)
            has_image = bool(ext in IMAGE_EXTENSIONS or raw.get("image") is not None or raw.get("image_path") is not None)
            if has_audio:
                self.cfg.modality = ModalityType.AUDIO
            elif has_video:
                self.cfg.modality = ModalityType.MULTIMODAL if item.text and has_image else ModalityType.VIDEO
            elif has_image:
                self.cfg.modality = ModalityType.MULTIMODAL if item.text else ModalityType.IMAGE
        item.metadata.pop("_data_hub_source_index", None)
        return item

    def _to_item(self, row: Any, idx: int) -> DatasetItem:
        if isinstance(row, DatasetItem):
            item = row.model_copy(deep=True)
            row_dict = copy.deepcopy(item.raw_data or {})
        else:
            row_dict = dict(row) if isinstance(row, dict) else {"text": str(row)}
            item = row_to_dataset_item(row_dict, idx, self.cfg)
        if "image" in row_dict:
            val = row_dict["image"]
            if hasattr(val, "convert") or isinstance(val, dict):
                if item.raw_data is None:
                    item.raw_data = {}
                item.raw_data["image"] = val
        if "audio" in row_dict:
            val = row_dict["audio"]
            if isinstance(val, dict):
                if item.raw_data is None:
                    item.raw_data = {}
                item.raw_data["audio"] = copy.deepcopy(val)
        # Internal stable row identity keeps duplicate user IDs from collapsing skip counts.
        item.metadata = copy.deepcopy(item.metadata)
        item.metadata["_data_hub_source_index"] = idx
        return item

    def peek(self, n: int) -> List[DatasetItem]:
        if self._source_iter is None:
            self._source_iter = iter(self.stream_source)
        while len(self._buffer) < n:
            if self.cfg.max_samples and len(self._buffer) >= self.cfg.max_samples:
                break
            try:
                row = next(self._source_iter)
            except StopIteration:
                break
            item = self._to_item(row, self._idx)
            self._idx += 1
            self._buffer.append(item)
        return list(self._buffer[:n])

    def __iter__(self):
        if self._is_one_shot:
            if self._consumed:
                raise RuntimeError("One-shot streaming dataset has already been iterated.")
            self._consumed = True
            source_iter = self._source_iter if self._source_iter is not None else iter(self.stream_source)
            buffered = True
            idx = self._idx
        else:
            self._epoch += 1
            if self._epoch == 1 and self._buffer:
                source_iter = self._source_iter if self._source_iter is not None else iter(self.stream_source)
                buffered = True
                idx = self._idx
            else:
                self._buffer = []
                self._source_iter = None
                source_iter = iter(self.stream_source)
                buffered = False
                idx = 0
                self._idx = 0

        self.state["yielded_samples"] = 0
        self.state["iteration_complete"] = False
        count = 0
        completed = False
        try:
            # Rows loaded by peek() are already included in _idx. They remain raw until
            # now so schema probing never performs strict media validation prematurely.
            while buffered and self._buffer:
                if self.cfg.max_samples is not None and count >= self.cfg.max_samples:
                    completed = True
                    break
                raw_item = self._buffer.pop(0)
                item = self._finalize_item(raw_item)
                if item is None:
                    continue
                count += 1
                self.state["yielded_samples"] = count
                yield item
            buffered = False

            while not completed:
                if self.cfg.max_samples is not None and count >= self.cfg.max_samples:
                    completed = True
                    break
                try:
                    row = next(source_iter)
                except StopIteration:
                    completed = True
                    break
                raw_item = self._to_item(row, idx)
                idx += 1
                self._idx = idx
                item = self._finalize_item(raw_item)
                if item is None:
                    continue
                count += 1
                self.state["yielded_samples"] = count
                yield item
        finally:
            self.state["yielded_samples"] = count
            self.state["iteration_complete"] = bool(completed)

    def __len__(self) -> int:
        # Once consumed, report valid yielded samples after parse/media skips rather
        # than the number of raw source rows. Before consumption, a known source length
        # is only an upper bound because lazy validation may reject samples.
        if self.state.get("iteration_complete"):
            return int(self.state.get("yielded_samples", 0))
        if hasattr(self.stream_source, "__len__"):
            src_len = len(self.stream_source)
            if self.cfg.max_samples is not None:
                return min(src_len, self.cfg.max_samples)
            return src_len
        if self._is_one_shot and self._consumed and "yielded_samples" in self.state:
            return int(self.state["yielded_samples"])
        raise TypeError("Streaming dataset with unknown length does not support len() before iteration completes.")


def parse_huggingface(
    repo_id: str,
    cfg: DatasetConfig,
) -> Tuple[Union[List[DatasetItem], LazyDatasetSequence], List[str], Dict[str, Any]]:
    """Downloads or streams datasets directly from Hugging Face Hub."""
    try:
        import datasets
    except ImportError:
        raise ImportError("Hugging Face `datasets` is required. Run: pip install datasets")

    split = cfg.split or "train"
    # Only forward user-supplied connector kwargs; never leak internal flags
    # (keys starting with '_') or DataLoader-only options to the external API.
    load_kwargs = {
        k: v for k, v in cfg.kwargs.items()
        if not k.startswith("_") and k not in _DATALOADER_ONLY_KEYS
    }
    if cfg.streaming:
        load_kwargs["streaming"] = True

    try:
        if cfg.subset:
            hf_ds = datasets.load_dataset(repo_id, cfg.subset, split=split, **load_kwargs)
        else:
            hf_ds = datasets.load_dataset(repo_id, split=split, **load_kwargs)
    except Exception as e:
        err_msg = str(e)
        # Distinguish missing split from network or auth errors
        if "BadSplitError" in type(e).__name__ or "Unknown split" in err_msg or "Available splits" in err_msg:
            available = None
            try:
                hf_dict = datasets.load_dataset(repo_id, **load_kwargs)
                if hasattr(hf_dict, "keys"):
                    available = list(hf_dict.keys())
            except Exception:
                pass

            if available:
                raise ValueError(
                    f"Requested split '{split}' not found in Hugging Face dataset '{repo_id}'. "
                    f"Available splits are: {available}."
                ) from e
            else:
                raise ValueError(
                    f"Requested split '{split}' not found in Hugging Face dataset '{repo_id}': {err_msg}"
                ) from e
        raise RuntimeError(f"Failed to load Hugging Face dataset '{repo_id}': {err_msg}") from e

    if isinstance(hf_ds, dict):
        if split in hf_ds:
            hf_ds = hf_ds[split]
        elif "train" in hf_ds:
            hf_ds = hf_ds["train"]
        elif hf_ds:
            hf_ds = next(iter(hf_ds.values()))

    columns = list(hf_ds.column_names) if hasattr(hf_ds, "column_names") else []

    if cfg.streaming or load_kwargs.get("streaming"):
        lazy_items = LazyDatasetSequence(hf_ds, cfg)
        if hasattr(hf_ds, "features") and hf_ds.features and not cfg.kwargs.get("_explicit_modality"):
            feat_keys = {str(k).lower() for k in hf_ds.features.keys()}
            if any(k in feat_keys for k in ("audio", "audio_path")):
                cfg.modality = ModalityType.AUDIO
            elif any(k in feat_keys for k in ("image", "image_path")):
                if any(k in feat_keys for k in ("caption", "text", "description", "transcription")):
                    cfg.modality = ModalityType.MULTIMODAL
                else:
                    cfg.modality = ModalityType.IMAGE
            elif any(k in feat_keys for k in ("video", "video_path")):
                cfg.modality = ModalityType.VIDEO
        metadata = {
            "repo_id": repo_id,
            "split": split,
            "subset": cfg.subset,
            "streaming": True,
            "hf_features": str(getattr(hf_ds, "features", {})),
        }
        return lazy_items, columns, metadata

    if cfg.max_samples and hasattr(hf_ds, "select") and hasattr(hf_ds, "__len__"):
        hf_ds = hf_ds.select(range(min(cfg.max_samples, len(hf_ds))))

    items: List[DatasetItem] = []

    for idx, row in enumerate(hf_ds):
        row_dict = dict(row)
        item = row_to_dataset_item(row_dict, idx, cfg)

        # Preserve PIL Image objects
        if "image" in row_dict:
            val = row_dict["image"]
            if hasattr(val, "convert") or isinstance(val, dict):
                if item.raw_data is None:
                    item.raw_data = {}
                item.raw_data["image"] = val

        # Preserve Audio waveforms / dicts
        if "audio" in row_dict:
            val = row_dict["audio"]
            if isinstance(val, dict):
                if item.raw_data is None:
                    item.raw_data = {}
                item.raw_data["audio"] = val

        items.append(item)
        if cfg.max_samples and len(items) >= cfg.max_samples:
            break

    if items:
        _promote_modality_from_items(items, cfg)

    metadata = {
        "repo_id": repo_id,
        "split": split,
        "subset": cfg.subset,
        "hf_features": str(getattr(hf_ds, "features", {})),
    }
    return items, columns, metadata


# ========================================
# 4. Kaggle Dataset Parser
# ========================================

def parse_kaggle(
    handle: str,
    cfg: DatasetConfig,
    cache_dir: str,
) -> Tuple[List[DatasetItem], List[str], Dict[str, Any]]:
    """Downloads or accesses locally cached Kaggle datasets."""
    # 1. Check local cache
    cached_path = os.path.join(cache_dir, "kaggle", handle.replace("/", "_"))
    if os.path.exists(cached_path) and os.listdir(cached_path):
        return parse_local_directory(cached_path, cfg)

    # 2. Download via kagglehub
    try:
        import kagglehub
        download_path = kagglehub.dataset_download(handle)
        if os.path.isdir(download_path):
            return parse_local_directory(download_path, cfg)
        elif os.path.isfile(download_path):
            return parse_local_file(download_path, cfg)
    except ImportError:
        pass

    raise RuntimeError(
        f"Kaggle dataset '{handle}' requested, but kagglehub is not installed or authenticated. "
        "Install with `pip install kagglehub`, or download the dataset folder and provide its local path."
    )


# ========================================
# 5. In-Memory Parser
# ========================================

def parse_memory(
    records: Any,
    cfg: DatasetConfig,
) -> Tuple[Union[List[DatasetItem], LazyDatasetSequence], List[str], Dict[str, Any]]:
    """Normalizes in-memory Python structures into standard DatasetItems."""
    # 0. Lazy Streaming Iterator / Generator
    if cfg.streaming and hasattr(records, "__iter__") and not isinstance(records, (list, tuple, dict, str)) and not hasattr(records, "to_dict"):
        state = {"source": "in_memory", "streaming": True, "parsing_errors": [], "skipped_samples": 0}
        seq = LazyDatasetSequence(records, cfg, state=state)
        cols = ["text"]
        return seq, cols, state

    # 1. Pandas DataFrame
    if hasattr(records, "to_dict") and hasattr(records, "columns"):
        columns = list(records.columns)
        records_list = records.to_dict(orient="records")
    # 2. Single Dictionary
    elif isinstance(records, dict):
        records_list = [records]
        columns = list(records.keys())
    # 3. List or Tuple of Records
    elif isinstance(records, (list, tuple)):
        records_list = list(records)
        columns = list(records_list[0].keys()) if records_list and isinstance(records_list[0], dict) else ["text"]
    # 4. String target (single scalar text record, never split into characters)
    elif isinstance(records, str):
        records_list = [{"text": records}]
        columns = ["text"]
    # 5. Iterator / Generator
    elif hasattr(records, "__iter__"):
        records_list = []
        columns = []
        for idx, row in enumerate(records):
            records_list.append(row)
            if not columns and isinstance(row, dict):
                columns = list(row.keys())
            if cfg.max_samples and len(records_list) >= cfg.max_samples:
                break
        if not columns:
            columns = ["text"]
    else:
        raise ValueError(
            f"Unsupported in-memory data type: {type(records)}. "
            "Supported formats: dict, list of dicts, tuple, pandas.DataFrame, or iterable."
        )

    items: List[DatasetItem] = []
    parsing_errors: List[Dict[str, Any]] = []
    media_extensions = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    for idx, row in enumerate(records_list):
        if isinstance(row, dict):
            item = row_to_dataset_item(row, idx, cfg)
        else:
            item = DatasetItem(id=idx, text=str(row), raw_data={"text": str(row)})

        media_payload = None
        media_kind = None
        if item.media_path:
            candidate = str(item.media_path).strip()
            is_remote = candidate.startswith(("http://", "https://", "data:"))
            media_path = Path(candidate)
            if not is_remote:
                error = None
                if not media_path.is_file():
                    error = FileNotFoundError(f"Media file '{candidate}' referenced in in-memory dataset does not exist.")
                elif cfg.error_policy != ErrorPolicy.PERMISSIVE:
                    ext = media_path.suffix.lower()
                    try:
                        if ext in IMAGE_EXTENSIONS:
                            load_image(media_path)
                        elif ext in AUDIO_EXTENSIONS:
                            load_audio(media_path, target_sr=cfg.target_sr)
                        elif ext in VIDEO_EXTENSIONS:
                            load_video(media_path, max_frames=cfg.max_video_frames)
                    except Exception as exc:
                        error = ValueError(f"Invalid media file '{candidate}' in in-memory dataset: {exc}")
                if error is not None:
                    if cfg.error_policy == ErrorPolicy.STRICT:
                        raise error
                    if cfg.error_policy == ErrorPolicy.REPORT_AND_SKIP:
                        parsing_errors.append({"id": item.id, "media_path": candidate, "error": str(error)})
                        continue
                    item.metadata = {**item.metadata, "media_warning": str(error)}
                else:
                    item.media_path = str(media_path.resolve())
        elif item.raw_data:
            # Validate in-memory PIL/bytes/structured media payloads that have no local path.
            for key, kind in (("image", "image"), ("audio", "audio"), ("video", "video")):
                if key in item.raw_data and item.raw_data[key] is not None:
                    media_payload, media_kind = item.raw_data[key], kind
                    break
            if media_payload is not None and cfg.error_policy != ErrorPolicy.PERMISSIVE:
                try:
                    if media_kind == "image":
                        load_image(media_payload)
                    elif media_kind == "audio":
                        load_audio(media_payload, target_sr=cfg.target_sr)
                    else:
                        # In-memory video objects aren't supported by the decoder; require a path.
                        raise ValueError("In-memory video payloads must provide a local media_path.")
                except Exception as exc:
                    if cfg.error_policy == ErrorPolicy.STRICT:
                        raise ValueError(f"Invalid in-memory {media_kind} media in sample {item.id!r}: {exc}") from exc
                    parsing_errors.append({"id": item.id, "media_path": None, "error": str(exc)})
                    if cfg.error_policy == ErrorPolicy.REPORT_AND_SKIP:
                        continue


        items.append(item)
        if cfg.max_samples and len(items) >= cfg.max_samples:
            break

    if items:
        _promote_modality_from_items(items, cfg)

    return items, columns, {
        "source": "in_memory",
        "record_count": len(items),
        "skipped_samples": len(parsing_errors),
        "parsing_errors": parsing_errors,
        "warnings": [f"Skipped {len(parsing_errors)} invalid media sample(s)."] if parsing_errors else [],
    }
