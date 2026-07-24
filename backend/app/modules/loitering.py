"""Loitering — person presence in frame for N seconds."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


class LoiteringModule:
    id = "loitering"
    labels = ["loitering"]

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
        threshold = float(self.config.get("threshold_seconds", 1800))
        conf = float(self.config.get("confidence", 0.35))
        miss_gap = float(self.config.get("miss_gap_seconds", 180))
        cooldown = float(self.config.get("cooldown_seconds", 1800))
        slots = ctx.state.setdefault("loiter_slots", {})
        now = time.time()

        results = self.model.track(
            frame,
            conf=conf,
            classes=[0],
            persist=True,
            verbose=False,
            device=self.device,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)
            scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [1.0] * len(boxes)
            for box, tid, score in zip(boxes, ids, scores):
                detections.append((tid, [float(v) for v in box], float(score)))

        used = set()
        events: list[DetectionEvent] = []
        for tid, box, score in detections:
            matched = None
            if tid is not None and f"y{tid}" in slots:
                matched = f"y{tid}"
            else:
                best_iou, best_key = 0.0, None
                for key, slot in slots.items():
                    iou = _iou(slot["bbox"], box)
                    if iou > best_iou and iou >= 0.2:
                        best_iou, best_key = iou, key
                matched = best_key

            if matched is None:
                matched = f"p{len(slots)+1}_{int(now)}"
                slots[matched] = {
                    "bbox": box,
                    "started": now,
                    "last": now,
                    "uploaded_at": 0.0,
                    "yolo": tid,
                }
            else:
                slot = slots[matched]
                if now - slot["last"] > miss_gap:
                    slot["started"] = now
                slot["last"] = now
                slot["bbox"] = box
                if tid is not None:
                    slots[f"y{tid}"] = slot
            used.add(matched)
            present = now - slots[matched]["started"]
            if present >= threshold and now - slots[matched]["uploaded_at"] >= cooldown:
                slots[matched]["uploaded_at"] = now
                events.append(
                    DetectionEvent(
                        label="loitering",
                        module_id=self.id,
                        boxes=[box],
                        scores=[score],
                        detail={"present_seconds": round(present, 1), "track": matched},
                        frame=frame.copy(),
                    )
                )

        # prune
        for key in list(slots.keys()):
            if key.startswith("y"):
                continue
            if now - slots[key]["last"] > miss_gap * 1.5:
                slots.pop(key, None)
        ctx.state["active_tracks"] = len(used)
        return events
