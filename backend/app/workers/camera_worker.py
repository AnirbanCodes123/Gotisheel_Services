"""Per-camera capture + detect worker (runs inside a shard process/thread group)."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..core.config import get_config
from ..engine.ffmpeg_cuda import create_capture
from ..engine.infer_scheduler import InferJob, SCHEDULER
from ..modules.base import CameraContext, DetectionEvent
from ..modules.overlay import draw_overlays


class CameraWorker:
    def __init__(
        self,
        camera_row: dict[str, Any],
        on_events: Optional[Callable[[dict[str, Any], list[DetectionEvent]], None]] = None,
    ):
        self.camera = camera_row
        self.key = camera_row["name"]
        self.on_events = on_events
        self.capture = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.latest_jpeg: Optional[bytes] = None
        self.online = False
        self.capture_fps = 0.0
        self.detect_fps = 0.0
        self.last_error: Optional[str] = None
        self.ctx = CameraContext(
            key=self.key,
            name=camera_row["name"],
            camera_id=camera_row.get("camera_id") or camera_row["name"],
            device=camera_row.get("device") or "",
            modules=list(camera_row.get("modules") or []),
        )
        self.ctx.state["extra"] = dict(camera_row.get("extra") or {})
        self.ctx.state["frisking_armed"] = bool((camera_row.get("extra") or {}).get("frisking_armed"))
        self.ctx.state["overlays"] = []
        self._detect_counter = 0
        self._detect_t0 = time.time()
        self._lock = threading.Lock()
        self._overlay_lock = threading.Lock()
        self._latest_overlays: list[dict[str, Any]] = []
        self._overlays_expire_at = 0.0

    def start(self) -> None:
        if self._running:
            return
        config = get_config()
        device = self.camera.get("device") or config.get("hardware", {}).get("device", "cpu")
        self.capture = create_capture(self.camera["rtsp_url"], device=device, camera_key=self.key)
        self.capture.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.key}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self.capture:
            self.capture.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.key,
            "camera_id": self.ctx.camera_id,
            "online": self.online,
            "capture_fps": round(self.capture_fps, 2),
            "detect_fps": round(self.detect_fps, 2),
            "modules": self.ctx.modules,
            "device": self.ctx.device or get_config().get("hardware", {}).get("device"),
            "error": self.last_error,
            "backend": getattr(self.capture, "backend_name", None),
            "state": {
                "person_count": self.ctx.state.get("person_count"),
                "active_tracks": self.ctx.state.get("active_tracks"),
                "sling_pending": self.ctx.state.get("sling_pending"),
                "overlay_count": len(self._latest_overlays),
            },
        }

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

    def _encode(self, frame: np.ndarray) -> None:
        quality = int(get_config().get("detect", {}).get("jpeg_quality", 80))
        # Draw last inference overlays onto live preview (ppe2-style)
        with self._overlay_lock:
            overlays = list(self._latest_overlays) if time.time() <= self._overlays_expire_at else []
        annotated = draw_overlays(frame, overlays)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self._lock:
                self.latest_jpeg = buf.tobytes()

    def _on_infer_done(self, camera_key: str, events: list) -> None:
        self._detect_counter += 1
        elapsed = time.time() - self._detect_t0
        if elapsed >= 1.0:
            self.detect_fps = self._detect_counter / elapsed
            self._detect_counter = 0
            self._detect_t0 = time.time()
        # Capture overlays produced during this inference tick
        overlays = list(self.ctx.state.get("overlays") or [])
        with self._overlay_lock:
            self._latest_overlays = overlays
            # Keep boxes visible briefly between detect ticks
            self._overlays_expire_at = time.time() + 1.5
        if events and self.on_events:
            self.on_events(self.camera, events)

    def _loop(self) -> None:
        config = get_config()
        detect_fps = float(self.camera.get("detect_fps") or 0) or float(config.get("detect", {}).get("default_fps", 4))
        min_interval = 1.0 / max(detect_fps, 0.1)
        last_submit = 0.0
        role = self.camera.get("stream_role") or "both"

        while self._running:
            frame = self.capture.get_frame() if self.capture else None
            if frame is None:
                self.online = False
                self.last_error = getattr(self.capture, "error", None) or "waiting for frames"
                time.sleep(0.05)
                continue

            self.online = True
            self.last_error = getattr(self.capture, "error", None)
            self.capture_fps = getattr(self.capture, "fps", 0.0) or self.capture_fps
            if role in ("live", "both"):
                self._encode(frame)

            now = time.time()
            if role in ("detect", "both") and self.ctx.modules and now - last_submit >= min_interval:
                last_submit = now
                SCHEDULER.submit(
                    InferJob(
                        camera_key=self.key,
                        frame=frame,
                        modules=list(self.ctx.modules),
                        callback=self._on_infer_done,
                    )
                )
            time.sleep(0.01)
