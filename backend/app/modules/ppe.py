"""PPE module — nohairnet / nomask (ported from ppe_bypass/ppe2.py)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent

VIOLATION_CLASSES = {2: "nohairnet", 3: "nomask"}


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


def _area(box) -> float:
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def _center_distance(a, b) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _area_ratio(a, b) -> float:
    return max(_area(a), _area(b)) / max(1.0, min(_area(a), _area(b)))


class PPEModule:
    id = "ppe"
    labels = ["nohairnet", "nomask"]

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

    def _boxes_same_person(self, box_a, box_b) -> bool:
        iou_thresh = float(self.config.get("iou_dedup_threshold", 0.25))
        center_px = float(self.config.get("center_dedup_distance_px", 280))
        area_ratio = float(self.config.get("area_ratio_dedup_threshold", 4.0))
        if _iou(box_a, box_b) >= iou_thresh:
            return True
        return _center_distance(box_a, box_b) <= center_px and _area_ratio(box_a, box_b) <= area_ratio

    def _is_same_person_event(self, label, track_id, box, event) -> bool:
        if event["label"] != label:
            return False
        if track_id is not None and event.get("track_id") is not None and track_id == event["track_id"]:
            return True
        return self._boxes_same_person(box, event["box"])

    def _is_duplicate(self, state, label, track_id, box, pending, now, cooldown) -> bool:
        for event in pending:
            if event["label"] != label:
                continue
            if track_id is not None and event.get("track_id") is not None and track_id == event["track_id"]:
                return True
            if self._boxes_same_person(box, event["box"]):
                return True
        for event in state["uploaded"]:
            if now - event["event_time"] >= cooldown:
                continue
            if self._is_same_person_event(label, track_id, box, event):
                event["last_seen"] = now
                event["box"] = box
                if track_id is not None:
                    event["track_id"] = track_id
                return True
        return False

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self.model is None:
            return []

        conf_map = {
            "nohairnet": float(self.config.get("nohairnet_confidence", 0.78)),
            "nomask": float(self.config.get("nomask_confidence", 0.78)),
        }
        cooldown = float(self.config.get("cooldown_seconds", 45))
        min_conf = min(conf_map.values())
        tracker = self.config.get("tracker", "bytetrack.yaml")
        state = ctx.state.setdefault("ppe", {"uploaded": []})
        now = time.time()

        results = self.model.track(
            frame,
            conf=min_conf,
            persist=True,
            tracker=tracker,
            verbose=False,
            device=self.device,
        )

        pending: list[dict[str, Any]] = []
        events: list[DetectionEvent] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            ids = (
                result.boxes.id.int().cpu().tolist()
                if getattr(result.boxes, "id", None) is not None
                else [None] * len(boxes)
            )
            for box, cls_id, score, track_id in zip(boxes, classes, scores, ids):
                label = VIOLATION_CLASSES.get(int(cls_id))
                if label is None:
                    continue
                if float(score) < conf_map.get(label, 0.78):
                    continue
                box_list = [float(v) for v in box]
                if self._is_duplicate(state, label, track_id, box_list, pending, now, cooldown):
                    continue
                pending.append(
                    {
                        "label": label,
                        "track_id": track_id,
                        "box": box_list,
                        "score": float(score),
                    }
                )
                state["uploaded"].append(
                    {
                        "label": label,
                        "track_id": track_id,
                        "box": box_list,
                        "event_time": now,
                        "last_seen": now,
                    }
                )
                events.append(
                    DetectionEvent(
                        label=label,
                        module_id=self.id,
                        boxes=[box_list],
                        scores=[float(score)],
                        detail={"confidence": float(score), "track_id": track_id},
                        frame=frame.copy(),
                    )
                )

        # prune history (2x cooldown like ppe2)
        cutoff = now - cooldown * 2
        state["uploaded"] = [e for e in state["uploaded"] if e["event_time"] >= cutoff]
        return events
