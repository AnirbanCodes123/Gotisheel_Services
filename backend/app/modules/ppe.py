"""PPE module — nohairnet / nomask."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent

VIOLATION_CLASSES = {2: "nohairnet", 3: "nomask"}


class PPEModule:
    id = "ppe"
    labels = ["nohairnet", "nomask"]

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

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self.model is None:
            return []
        conf_map = {
            "nohairnet": float(self.config.get("nohairnet_confidence", 0.45)),
            "nomask": float(self.config.get("nomask_confidence", 0.45)),
        }
        cooldown = float(self.config.get("cooldown_seconds", 45))
        results = self.model.predict(frame, conf=0.25, verbose=False, device=self.device)
        events: list[DetectionEvent] = []
        now = time.time()

        for result in results:
            if result.boxes is None:
                continue
            for box, cls_id, score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                label = VIOLATION_CLASSES.get(int(cls_id))
                if label is None:
                    continue
                if float(score) < conf_map.get(label, 0.45):
                    continue
                key = f"{ctx.key}:{label}"
                if now - self._last_event.get(key, 0) < cooldown:
                    continue
                self._last_event[key] = now
                events.append(
                    DetectionEvent(
                        label=label,
                        module_id=self.id,
                        boxes=[[float(v) for v in box]],
                        scores=[float(score)],
                        detail={"confidence": float(score)},
                        frame=frame.copy(),
                    )
                )
        return events
