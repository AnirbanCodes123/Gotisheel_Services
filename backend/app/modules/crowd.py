"""Crowd counting module (ported from crowd_loitering_bypass/app.py)."""

from __future__ import annotations

import time
from datetime import datetime, time as dt_time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent


def _parse_hhmm(value: str) -> dt_time:
    hour, minute = value.split(":")
    return dt_time(int(hour), int(minute))


class CrowdModule:
    id = "crowd"
    labels = ["crowd"]

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.config: dict[str, Any] = {}

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        from ultralytics import YOLO

        self.device = device
        self.config = config or {}
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            self.device = "cpu"

    def unload(self) -> None:
        self.model = None

    def _quiet_hours(self) -> bool:
        windows = self.config.get("disabled_windows") or []
        now = datetime.now().time()
        for start_s, end_s in windows:
            start, end = _parse_hhmm(start_s), _parse_hhmm(end_s)
            if start <= now < end:
                return True
        return False

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self.model is None:
            return []
        # Still run inference for live person_count; suppress events in quiet hours.
        threshold = int(self.config.get("person_threshold", 15))
        conf = float(self.config.get("confidence", 0.4))
        cooldown = float(self.config.get("cooldown_seconds", 120))
        results = self.model.predict(frame, conf=conf, classes=[0], verbose=False, device=self.device)
        boxes: list[list[float]] = []
        scores: list[float] = []
        for result in results:
            if result.boxes is None:
                continue
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append([float(v) for v in box])
                scores.append(float(score))
        count = len(boxes)
        ctx.state["person_count"] = count
        if count < threshold or self._quiet_hours():
            return []
        state = ctx.state.setdefault("crowd", {"last_upload_at": 0.0})
        now = time.time()
        if now - float(state["last_upload_at"]) < cooldown:
            return []
        # Advance cooldown when event is emitted (upload success is handled by EventService).
        state["last_upload_at"] = now
        return [
            DetectionEvent(
                label="crowd",
                module_id=self.id,
                boxes=boxes,
                scores=scores,
                detail={"person_count": count, "threshold": threshold},
                frame=frame.copy(),
            )
        ]
