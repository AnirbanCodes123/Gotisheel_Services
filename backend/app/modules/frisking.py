"""Frisking missed — wraps frisking_bypass FriskingDetector (entrance/out)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import CameraContext, DetectionEvent
from .paths import resolve_model_path

FRISKING_BYPASS = Path(__file__).resolve().parents[4] / "frisking_bypass"


def _ensure_frisking_path() -> None:
    path = str(FRISKING_BYPASS)
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve_mode(ctx: CameraContext) -> str:
    extra = ctx.state.get("extra") or {}
    mode = str(extra.get("frisking_mode") or "").strip().lower()
    if mode in ("entrance", "in", "entry"):
        return "entrance"
    if mode in ("out", "exit", "outgoing"):
        return "out"
    name = (ctx.name or ctx.key or "").lower()
    if "out" in name or "exit" in name:
        return "out"
    return "entrance"


class FriskingModule:
    id = "frisking"
    labels = ["frisking_missed"]

    def __init__(self) -> None:
        self.pose_model_path: Optional[str] = None
        self.person_model_path: Optional[str] = None
        self.security_model_path: Optional[str] = None
        self.device = "cpu"
        self.config: dict[str, Any] = {}
        self._detector_cls_entrance = None
        self._detector_cls_out = None
        self.security_loaded = False

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        self.device = device
        self.config = config or {}
        self.pose_model_path = model_path
        person = config.get("person_model") or ""
        resolved_person = resolve_model_path(person) if person else None
        self.person_model_path = resolved_person or (person if str(person).endswith(".pt") else None)

        # Resolve + preload security model BEFORE importing VA modules so
        # `from yolov8_model.yolov8_api_demo import yolov8_detect_security` succeeds.
        _ensure_frisking_path()
        security_name = config.get("security_model") or "security_7.pt"
        security_path = resolve_model_path(security_name) or resolve_model_path("security_6.pt")
        if not security_path:
            # fall back to package-local resolver
            try:
                from yolov8_model.yolov8_api_demo import resolve_security_model_path

                security_path = resolve_security_model_path(security_name)
            except Exception:
                security_path = None
        self.security_model_path = security_path
        if security_path:
            os.environ["GOTISHEEL_SECURITY_MODEL"] = security_path
            os.environ["FRISKING_SECURITY_MODEL"] = security_path

        try:
            from yolov8_model.yolov8_api_demo import (
                ensure_security_model,
                security_status,
                yolov8_detect_security,
            )

            ensure_security_model(security_path, device=device)
            self.security_loaded = True
            print(f"[frisking] security model ready: {security_status()}")
        except Exception as exc:
            self.security_loaded = False
            print(f"[frisking] WARNING security model not loaded: {exc}")
            print("[frisking] place security_7.pt (or security_6.pt) in data/models/")

        try:
            import frisking_rtsp_VA_entrance as entrance_logic
            import frisking_rtsp_VA_out as out_logic

            # If detectors were previously imported with a stub, patch the live function.
            if self.security_loaded:
                entrance_logic.yolov8_detect_security = yolov8_detect_security
                out_logic.yolov8_detect_security = yolov8_detect_security

            self._detector_cls_entrance = entrance_logic.FriskingDetector
            self._detector_cls_out = out_logic.FriskingDetector
            print(
                f"[frisking] detectors ready pose={self.pose_model_path} "
                f"person={self.person_model_path} security={self.security_model_path} "
                f"security_loaded={self.security_loaded} device={device}"
            )
        except Exception as exc:
            print(f"[frisking] failed to import FriskingDetector: {exc}")
            raise

    def unload(self) -> None:
        self._detector_cls_entrance = None
        self._detector_cls_out = None

    def _get_detector(self, ctx: CameraContext):
        mode = _resolve_mode(ctx)
        state = ctx.state.setdefault("frisking", {})
        detector = state.get("detector")
        if detector is not None and state.get("mode") == mode:
            return detector
        cls = self._detector_cls_out if mode == "out" else self._detector_cls_entrance
        if cls is None:
            return None
        conf = float(self.config.get("confidence", 0.5))
        detector = cls(
            model_path=self.pose_model_path,
            person_model_path=self.person_model_path,
            confidence=conf,
        )
        state["detector"] = detector
        state["mode"] = mode
        print(
            f"[frisking] created {mode} detector for camera={ctx.name} "
            f"security_loaded={self.security_loaded}"
        )
        return detector

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self._detector_cls_entrance is None and self._detector_cls_out is None:
            return []
        detector = self._get_detector(ctx)
        if detector is None:
            return []

        try:
            _processed, missed_events, _prev = detector.process_single_frame(frame)
        except Exception as exc:
            print(f"[frisking] process error camera={ctx.name}: {exc}")
            return []

        events: list[DetectionEvent] = []
        for event in missed_events or []:
            bbox = event.get("bbox_xyxy") or event.get("bbox")
            if not bbox:
                continue
            if "bbox_xyxy" in event:
                box = [float(v) for v in event["bbox_xyxy"]]
            elif hasattr(detector, "bbox_xywh_to_xyxy"):
                box = [float(v) for v in detector.bbox_xywh_to_xyxy(bbox)]
            else:
                box = [float(v) for v in bbox]

            event_frame = event.get("event_frame")
            events.append(
                DetectionEvent(
                    label="frisking_missed",
                    module_id=self.id,
                    boxes=[box],
                    scores=[1.0],
                    detail={
                        "person_id": event.get("person_id"),
                        "mode": (ctx.state.get("frisking") or {}).get("mode", "entrance"),
                        "security_loaded": self.security_loaded,
                    },
                    frame=event_frame.copy() if event_frame is not None else frame.copy(),
                )
            )
        return events
