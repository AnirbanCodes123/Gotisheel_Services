"""Per-camera capture + detect worker — inline inference like ppe_bypass/ppe2.py."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..core.config import get_config
from ..engine.ffmpeg_cuda import create_capture
from ..modules import REGISTRY
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
        self.display_fps = 0.0
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
        self._display_counter = 0
        self._display_t0 = time.time()
        self._loop_counter = 0
        self._loop_t0 = time.time()
        self._lock = threading.Lock()
        self._emit_pool_lock = threading.Lock()

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
            "capture_fps": round(float(self.capture_fps), 2),
            "detect_fps": round(float(self.detect_fps), 2),
            "display_fps": round(float(self.display_fps), 2),
            "modules": self.ctx.modules,
            "device": self.ctx.device or get_config().get("hardware", {}).get("device"),
            "error": self.last_error,
            "backend": getattr(self.capture, "backend_name", None),
            "state": {
                "person_count": self.ctx.state.get("person_count"),
                "active_tracks": self.ctx.state.get("active_tracks"),
                "sling_pending": self.ctx.state.get("sling_pending"),
                "overlay_count": len(self.ctx.state.get("overlays") or []),
            },
        }

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

    def _set_jpeg(self, frame: np.ndarray) -> None:
        quality = int(get_config().get("detect", {}).get("jpeg_quality", 80))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self._lock:
                self.latest_jpeg = buf.tobytes()

    def _emit_events_async(self, events: list[DetectionEvent]) -> None:
        if not events or not self.on_events:
            return

        def _run():
            try:
                self.on_events(self.camera, events)
            except Exception as exc:
                print(f"[{self.key}] event emit error: {exc}")

        threading.Thread(target=_run, name=f"evt-{self.key}", daemon=True).start()

    def _loop(self) -> None:
        """Inline detect+draw like ppe2.process_single_frame — same frame, immediate overlay."""
        config = get_config()
        # ppe2 runs every captured frame; default higher FPS for snappy boxes
        detect_fps = float(self.camera.get("detect_fps") or 0) or float(
            config.get("detect", {}).get("default_fps", 12)
        )
        min_interval = 1.0 / max(detect_fps, 0.1)
        last_detect = 0.0
        role = self.camera.get("stream_role") or "both"
        last_overlays: list[dict[str, Any]] = []

        while self._running:
            frame = self.capture.get_frame() if self.capture else None
            if frame is None:
                self.online = False
                self.last_error = getattr(self.capture, "error", None) or "waiting for frames"
                time.sleep(0.02)
                continue

            self.online = True
            self.last_error = getattr(self.capture, "error", None)
            now = time.time()
            display = frame

            # Realtime loop FPS (worker consume rate) — update every 0.5s
            self._loop_counter += 1
            loop_elapsed = now - self._loop_t0
            if loop_elapsed >= 0.5:
                self.capture_fps = self._loop_counter / loop_elapsed
                self._loop_counter = 0
                self._loop_t0 = now
            # Prefer capture-backend fps when available, else loop fps
            backend_fps = float(getattr(self.capture, "fps", 0.0) or 0.0)
            if backend_fps > 0:
                self.capture_fps = backend_fps

            do_detect = (
                role in ("detect", "both")
                and self.ctx.modules
                and (now - last_detect) >= min_interval
            )
            if do_detect:
                last_detect = now
                try:
                    REGISTRY.ensure_loaded()
                    events = REGISTRY.process_frame(frame, self.ctx, list(self.ctx.modules))
                    last_overlays = list(self.ctx.state.get("overlays") or [])
                    # Draw on THIS frame (ppe2 style) before publishing JPEG
                    display = draw_overlays(frame, last_overlays)
                    self._detect_counter += 1
                    elapsed = now - self._detect_t0
                    if elapsed >= 0.5:
                        self.detect_fps = self._detect_counter / elapsed
                        self._detect_counter = 0
                        self._detect_t0 = now
                    if events:
                        self._emit_events_async(events)
                except Exception as exc:
                    self.last_error = f"detect: {exc}"
                    print(f"[{self.key}] detect error: {exc}")
            elif role in ("live", "both") and last_overlays:
                # Between detect ticks keep last boxes visible (short persistence)
                display = draw_overlays(frame, last_overlays)

            if role in ("live", "both"):
                self._set_jpeg(display)
                self._display_counter += 1
                disp_elapsed = now - self._display_t0
                if disp_elapsed >= 0.5:
                    self.display_fps = self._display_counter / disp_elapsed
                    self._display_counter = 0
                    self._display_t0 = now
            time.sleep(0.001)
