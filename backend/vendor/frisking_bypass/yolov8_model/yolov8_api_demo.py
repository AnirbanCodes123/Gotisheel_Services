"""Minimal yolov8_model package for frisking security personnel detection.

Mirrors the import path expected by frisking_rtsp_VA_*.py:
  from yolov8_model.yolov8_api_demo import yolov8_detect_security
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

_SECURITY_MODEL = None
_SECURITY_LOCK = threading.Lock()
_SECURITY_PATH: Optional[str] = None
_LOAD_ERROR: Optional[str] = None

YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "960"))
_PKG_DIR = Path(__file__).resolve().parent


def _candidate_security_paths() -> list[Path]:
    env = os.getenv("GOTISHEEL_SECURITY_MODEL") or os.getenv("FRISKING_SECURITY_MODEL") or ""
    # Walk up to find Gotisheel_Services root (…/backend/vendor/frisking_bypass/yolov8_model)
    gotisheel_root = _PKG_DIR.parents[3] if len(_PKG_DIR.parents) > 3 else _PKG_DIR
    roots = [
        Path(env) if env else None,
        _PKG_DIR / "security" / "security_7.pt",
        _PKG_DIR / "security" / "security_6.pt",
        _PKG_DIR / "security" / "security_1.pt",
        gotisheel_root / "data" / "models" / "security_7.pt",
        gotisheel_root / "data" / "models" / "security_6.pt",
        gotisheel_root / "data" / "models" / "security_1.pt",
        Path.cwd() / "data" / "models" / "security_7.pt",
        Path.cwd() / "security_7.pt",
    ]
    return [p for p in roots if p is not None]


def resolve_security_model_path(explicit: str | None = None) -> Optional[str]:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path.resolve())
        name = path.name
        for cand in _candidate_security_paths():
            if cand.name == name and cand.is_file():
                return str(cand.resolve())
    for cand in _candidate_security_paths():
        if cand.is_file():
            return str(cand.resolve())
    return None


def ensure_security_model(model_path: str | None = None, device: str | None = None):
    """Lazy-load the security YOLO model once. Raises on failure."""
    global _SECURITY_MODEL, _SECURITY_PATH, _LOAD_ERROR
    with _SECURITY_LOCK:
        if _SECURITY_MODEL is not None:
            return _SECURITY_MODEL
        path = resolve_security_model_path(model_path)
        if not path:
            _LOAD_ERROR = (
                "security model file not found — put security_7.pt in "
                "Gotisheel_Services/data/models/ or vendor/.../yolov8_model/security/"
            )
            raise FileNotFoundError(_LOAD_ERROR)
        from ultralytics import YOLO

        model = YOLO(path)
        if device:
            try:
                model.to(device)
            except Exception:
                pass
        _SECURITY_MODEL = model
        _SECURITY_PATH = path
        _LOAD_ERROR = None
        print(f"[security] loaded model={path} device={device or 'default'}")
        return _SECURITY_MODEL


def security_status() -> dict[str, Any]:
    return {
        "loaded": _SECURITY_MODEL is not None,
        "path": _SECURITY_PATH,
        "error": _LOAD_ERROR,
    }


def _run_detection(frame: np.ndarray, conf_threshold: float) -> Tuple[List[int], List[list], List[float]]:
    model = ensure_security_model()
    with _SECURITY_LOCK, torch.inference_mode():
        results = model(frame, batch=1, save=False, verbose=False, imgsz=YOLO_IMG_SIZE)

    selected_class_ids: List[int] = []
    selected_boxes: List[list] = []
    selected_confidences: List[float] = []
    for r in results:
        if r.boxes is None:
            continue
        conf_data = r.boxes.conf.tolist()
        bbox_data = r.boxes.xyxy.tolist()
        class_data = r.boxes.cls.tolist()
        for conf, bbox, class_val in zip(conf_data, bbox_data, class_data):
            class_id = int(class_val)
            if class_id == 0 and conf > conf_threshold:
                selected_class_ids.append(class_id)
                selected_boxes.append(bbox)
                selected_confidences.append(conf)
    return selected_class_ids, selected_boxes, selected_confidences


def yolov8_detect_security(frame, conf_threshold=0.4):
    """Detect security personnel (class 0). Compatible with frisking_bypass."""
    try:
        return _run_detection(frame, float(conf_threshold))
    except Exception as exc:
        print(f"[security] detect error: {exc}")
        return [], [], []
