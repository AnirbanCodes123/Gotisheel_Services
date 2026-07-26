"""Sling / no-sling motion detection (ported from sling_bypass/sling_yolo2.py)."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from .base import CameraContext, DetectionEvent
from .overlay import append_overlays


class ConfirmationTracker:
    """Count sling hits within a sliding window; confirm after N spaced detections."""

    def __init__(self, required_count: int, window_seconds: float, min_interval: float) -> None:
        self.required_count = required_count
        self.window_seconds = window_seconds
        self.min_interval = min_interval
        self.window_id: Optional[str] = None
        self.detections: deque = deque()
        self.last_counted_at = 0.0

    def reset(self) -> None:
        self.window_id = None
        self.detections.clear()
        self.last_counted_at = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.detections and self.detections[0]["time"] < cutoff:
            self.detections.popleft()

    def record(self, window_id: str, frame, box, score, now: float):
        if window_id != self.window_id:
            self.reset()
            self.window_id = window_id
        self._prune(now)
        if self.last_counted_at and (now - self.last_counted_at) < self.min_interval:
            return "skipped", None
        self.last_counted_at = now
        detection = {"time": now, "frame": frame.copy(), "box": box, "score": score}
        self.detections.append(detection)
        if len(self.detections) >= self.required_count:
            latest = self.detections[-1]
            self.reset()
            return "confirmed", latest
        return "counted", None

    @property
    def pending_count(self) -> int:
        return len(self.detections)


class SlingModule:
    id = "sling"
    labels = ["sling_psychrometer", "no_sling_motion"]

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
        print(f"[sling] model ready path={model_path} device={self.device}")

    def unload(self) -> None:
        self.model = None

    def _detection_window(self, now: Optional[datetime] = None):
        """
        HH:40 – next-hour HH:20 → sling_motion / sling_psychrometer
        HH:20 – HH:40           → no_sling_motion_detect / no_sling_motion
        """
        now = now or datetime.now()
        start_min = int(self.config.get("window_start_minute", 40))
        end_min = int(self.config.get("window_end_minute", 20))
        sling_label = self.config.get("sling_label", "sling_psychrometer")
        no_sling_label = self.config.get("no_sling_label", "no_sling_motion")
        minute = now.minute

        if minute >= start_min:
            window_start = now.replace(minute=0, second=0, microsecond=0)
            return "sling_motion", f"sling_motion:{window_start:%Y%m%d%H}", sling_label
        if minute < end_min:
            previous_hour = now - timedelta(hours=1)
            window_start = previous_hour.replace(minute=0, second=0, microsecond=0)
            return "sling_motion", f"sling_motion:{window_start:%Y%m%d%H}", sling_label
        window_start = now.replace(minute=0, second=0, microsecond=0)
        return "no_sling_motion_detect", f"no_sling_motion_detect:{window_start:%Y%m%d%H}", no_sling_label

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self.model is None:
            return []

        conf = float(self.config.get("confidence", 0.55))
        need = int(self.config.get("confirmation_count", 2))
        confirm_window = float(self.config.get("confirmation_window_seconds", 120))
        min_interval = float(self.config.get("confirmation_min_interval_seconds", 1.0))
        events_per_window = int(self.config.get("events_per_window", 1))

        state = ctx.state.setdefault(
            "sling",
            {
                "tracker": ConfirmationTracker(need, confirm_window, min_interval),
                "uploaded_window_id": None,
                "window_event_count": {},
            },
        )
        tracker: ConfirmationTracker = state["tracker"]
        # refresh tracker params if config changed
        tracker.required_count = need
        tracker.window_seconds = confirm_window
        tracker.min_interval = min_interval

        window_mode, window_id, event_label = self._detection_window()
        ctx.state["sling_window_mode"] = window_mode
        ctx.state["sling_window_id"] = window_id
        active = window_mode in ("sling_motion", "no_sling_motion_detect")

        results = self.model.predict(frame, conf=conf, verbose=False, device=self.device)
        boxes, scores = [], []
        for result in results:
            if result.boxes is None:
                continue
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append([float(v) for v in box])
                scores.append(float(score))

        if boxes:
            append_overlays(
                ctx.state,
                [
                    {"label": "sling", "box": b, "score": s, "color": (0, 220, 0)}
                    for b, s in zip(boxes, scores)
                ],
            )

        if not boxes or not active:
            ctx.state["sling_pending"] = tracker.pending_count
            return []

        used = int(state["window_event_count"].get(window_id, 0))
        if used >= events_per_window or state.get("uploaded_window_id") == window_id:
            ctx.state["sling_pending"] = 0
            return []

        best_idx = scores.index(max(scores))
        best_box = boxes[best_idx]
        best_score = scores[best_idx]
        now = time.time()
        action, confirmed = tracker.record(window_id, frame, best_box, best_score, now)
        ctx.state["sling_pending"] = tracker.pending_count

        if action != "confirmed" or not confirmed:
            return []

        state["window_event_count"][window_id] = used + 1
        state["uploaded_window_id"] = window_id
        return [
            DetectionEvent(
                label=event_label,
                module_id=self.id,
                boxes=[confirmed["box"]],
                scores=[float(confirmed["score"])],
                detail={
                    "confirmation": need,
                    "confidence": float(confirmed["score"]),
                    "window_mode": window_mode,
                    "window_id": window_id,
                },
                frame=confirmed["frame"],
            )
        ]
