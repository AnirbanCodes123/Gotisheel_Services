"""Frisking missed — simplified pose/person presence event for entrance lanes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent


class FriskingModule:
    id = "frisking"
    labels = ["frisking_missed"]

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.config: dict[str, Any] = {}
        self._last_event: dict[str, float] = {}

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        from ultralytics import YOLO

        self.device = device
        self.config = config or {}
        # Prefer person model path if provided separately
        person_model = config.get("person_model") or model_path
        self.model = YOLO(person_model if str(person_model).endswith(".pt") else model_path)
        try:
            self.model.to(device)
        except Exception:
            self.device = "cpu"

    def unload(self) -> None:
        self.model = None

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        """
        Emits frisking_missed when a person is tracked crossing without an attendant
        signal. Simplified portable version: person detection + cooldown.
        Full VA logic from frisking_bypass can replace this process() later.
        """
        if self.model is None:
            return []
        conf = float(self.config.get("confidence", 0.5))
        cooldown = float(self.config.get("cooldown_seconds", 60))
        results = self.model.predict(frame, conf=conf, classes=[0], verbose=False, device=self.device)
        boxes, scores = [], []
        for result in results:
            if result.boxes is None:
                continue
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append([float(v) for v in box])
                scores.append(float(score))
        if not boxes:
            return []
        # Only fire when camera extra says zone mode expects events, or always with cooldown
        key = f"{ctx.key}:frisking"
        now = time.time()
        if now - self._last_event.get(key, 0) < cooldown:
            return []
        if not ctx.state.get("frisking_armed") and not (ctx.state.get("extra") or {}).get("frisking_armed"):
            # Avoid constant alerts unless camera is explicitly armed for frisking VA.
            return []
        self._last_event[key] = now
        return [
            DetectionEvent(
                label="frisking_missed",
                module_id=self.id,
                boxes=boxes[:1],
                scores=scores[:1],
                detail={"persons": len(boxes)},
                frame=frame.copy(),
            )
        ]
