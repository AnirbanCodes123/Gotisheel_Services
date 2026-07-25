"""Resolve YOLO weight paths across data/models, backend, and sibling bypass apps."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..core.config import BACKEND_DIR, ROOT_DIR, get_config

# Ultralytics may auto-download these if missing; custom project weights must exist on disk.
_AUTODOWNLOAD_PREFIXES = ("yolo", "yolov8", "yolov9", "yolov10", "yolo11", "rtdetr")


def _candidate_dirs() -> list[Path]:
    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    parent = ROOT_DIR.parent
    return [
        models_dir,
        ROOT_DIR,
        ROOT_DIR / "models",
        BACKEND_DIR,
        BACKEND_DIR / "models",
        Path.cwd(),
        Path.cwd() / "data" / "models",
        Path.cwd() / "models",
        ROOT_DIR / "backend" / "vendor" / "frisking_bypass" / "yolov8_model" / "security",
        BACKEND_DIR / "vendor" / "frisking_bypass" / "yolov8_model" / "security",
        parent / "ppe_bypass",
        parent / "ppe_bypass" / "models",
        parent / "sling_bypass",
        parent / "sling_bypass" / "models",
        parent / "sling_bypass_latest",
        parent / "sling_bypass_latest" / "models",
        parent / "crowd_loitering_bypass",
        parent / "crowd_loitering_bypass" / "models",
        parent / "frisking_bypass",
        parent / "frisking_bypass" / "models",
        parent / "frisking_bypass" / "yolov8_model" / "security",
    ]


def can_autodownload(model_name: str) -> bool:
    stem = Path(model_name).name.lower()
    return any(stem.startswith(p) for p in _AUTODOWNLOAD_PREFIXES)


def resolve_model_path(model_name: str | None, extra_dirs: Optional[Iterable[str | Path]] = None) -> Optional[str]:
    """Return an existing filesystem path for a model file, or None."""
    if not model_name:
        return None
    raw = str(model_name).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return str(path.resolve())

    name = path.name
    dirs = list(_candidate_dirs())
    if extra_dirs:
        dirs = [Path(d) for d in extra_dirs] + dirs

    seen: set[str] = set()
    for folder in dirs:
        key = str(folder)
        if key in seen:
            continue
        seen.add(key)
        candidate = folder / name
        if candidate.is_file():
            return str(candidate.resolve())
        if raw != name:
            nested = folder / raw
            if nested.is_file():
                return str(nested.resolve())
    return None
