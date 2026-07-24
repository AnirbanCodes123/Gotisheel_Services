"""Crowd counting module."""

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
        self._last_event: dict[str, float] = {}

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
        if self.model is None or self._quiet_hours():
            return []
        threshold = int(self.config.get("person_threshold", 15))
        conf = float(self.config.get("confidence", 0.4))
        cooldown = float(self.config.get("cooldown_seconds", 120))
        results = self.model.predict(frame, conf=conf, classes=[0], verbose=False, device=self.device)
        boxes = []
        scores = []
        for result in results:
            if result.boxes is None:
                continue
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append([float(v) for v in box])
                scores.append(float(score))
        count = len(boxes)
        ctx.state["person_count"] = count
        if count < threshold:
            return []
        key = f"{ctx.key}:crowd"
        now = time.time()
        if now - self._last_event.get(key, 0) < cooldown:
            return []
        self._last_event[key] = now
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
