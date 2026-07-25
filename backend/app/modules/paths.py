"""Resolve YOLO weight paths across data/models and sibling bypass apps."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..core.config import ROOT_DIR, get_config


def _candidate_dirs() -> list[Path]:
    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    parent = ROOT_DIR.parent
    home_ds = Path("/Users/anirban/Desktop/DS")
    return [
        models_dir,
        ROOT_DIR,
        ROOT_DIR / "models",
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
        parent / "gotisheel_ai_2" / "data" / "models",
        home_ds / "SLING" / "SLING_YOLO" / "models",
        home_ds / "SLING" / "SLING_YOLO",
        home_ds / "frisking",
    ]


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

    for folder in dirs:
        candidate = folder / name
        if candidate.is_file():
            return str(candidate.resolve())
        # also try nested relative path as given
        if raw != name:
            nested = folder / raw
            if nested.is_file():
                return str(nested.resolve())
    return None
