"""Sling / psychrometer detection with confirmation tracker."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent


class SlingModule:
    id = "sling"
    labels = ["sling_psychrometer"]

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

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self.model is None:
            return []
        conf = float(self.config.get("confidence", 0.45))
        need = int(self.config.get("confirmation_count", 3))
        cooldown = float(self.config.get("cooldown_seconds", 3600))
        tracker = ctx.state.setdefault(
            "sling_confirm",
            {"count": 0, "last_hit": 0.0, "uploaded_at": 0.0, "last_box": None, "last_score": 0.0},
        )
        now = time.time()
        results = self.model.predict(frame, conf=conf, verbose=False, device=self.device)
        best_box, best_score = None, 0.0
        for result in results:
            if result.boxes is None:
                continue
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                if float(score) > best_score:
                    best_score = float(score)
                    best_box = [float(v) for v in box]

        if best_box is None:
            # decay confirmation slowly
            if now - tracker["last_hit"] > 5:
                tracker["count"] = 0
            return []

        if now - tracker["last_hit"] > 8:
            tracker["count"] = 0
        tracker["count"] += 1
        tracker["last_hit"] = now
        tracker["last_box"] = best_box
        tracker["last_score"] = best_score
        ctx.state["sling_pending"] = tracker["count"]

        if tracker["count"] < need:
            return []
        if now - tracker["uploaded_at"] < cooldown:
            tracker["count"] = 0
            return []

        tracker["uploaded_at"] = now
        tracker["count"] = 0
        return [
            DetectionEvent(
                label="sling_psychrometer",
                module_id=self.id,
                boxes=[best_box],
                scores=[best_score],
                detail={"confirmation": need, "confidence": best_score},
                frame=frame.copy(),
            )
        ]
