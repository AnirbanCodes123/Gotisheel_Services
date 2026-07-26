"""Shard manager — groups cameras into worker pools for scale."""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..core.config import get_config
from ..core.db import Camera, session_factory
from ..engine.go2rtc_client import Go2rtcClient
from ..engine.infer_scheduler import InferJob, SCHEDULER
from ..modules import REGISTRY
from ..modules.base import CameraContext
from ..services.events import EVENT_SERVICE
from ..workers.camera_worker import CameraWorker


class ShardManager:
    def __init__(self) -> None:
        self.workers: dict[str, CameraWorker] = {}
        self.go2rtc = Go2rtcClient()
        self._lock = threading.Lock()
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        REGISTRY.ensure_loaded()
        SCHEDULER.set_processor(self._process_job)
        SCHEDULER.start()
        self.reload_cameras()
        self.started = True
        print("[shard] manager started")

    def stop(self) -> None:
        with self._lock:
            for worker in self.workers.values():
                worker.stop()
            self.workers.clear()
        SCHEDULER.stop()
        self.started = False

    def _camera_to_dict(self, row: Camera) -> dict[str, Any]:
        return {
            "id": row.id,
            "name": row.name,
            "camera_id": row.camera_id or row.name,
            "rtsp_url": row.rtsp_url,
            "enabled": bool(row.enabled),
            "device": row.device or "",
            "detect_fps": float(row.detect_fps or 0),
            "stream_role": row.stream_role or "both",
            "modules": list(row.modules or []),
            "extra": dict(row.extra or {}),
        }

    def reload_cameras(self) -> None:
        session = session_factory()
        try:
            rows = session.query(Camera).filter(Camera.enabled.is_(True)).all()
            desired = {row.name: self._camera_to_dict(row) for row in rows}
        finally:
            session.close()

        with self._lock:
            # stop removed
            for name in list(self.workers.keys()):
                if name not in desired:
                    self.workers[name].stop()
                    try:
                        self.go2rtc.unregister(name)
                    except Exception as exc:
                        print(f"[shard] go2rtc unregister {name}: {exc}")
                    del self.workers[name]

            for name, cam in desired.items():
                existing = self.workers.get(name)
                if existing:
                    # restart if RTSP/modules/device changed
                    if (
                        existing.camera.get("rtsp_url") != cam["rtsp_url"]
                        or existing.camera.get("modules") != cam["modules"]
                        or existing.camera.get("device") != cam["device"]
                        or existing.camera.get("detect_fps") != cam["detect_fps"]
                        or existing.camera.get("stream_role") != cam["stream_role"]
                    ):
                        existing.stop()
                        worker = CameraWorker(cam, on_events=self._on_events)
                        worker.start()
                        self.workers[name] = worker
                        try:
                            self.go2rtc.register_rtsp(name, cam["rtsp_url"])
                        except Exception as exc:
                            print(f"[shard] go2rtc register {name}: {exc}")
                else:
                    worker = CameraWorker(cam, on_events=self._on_events)
                    worker.start()
                    self.workers[name] = worker
                    try:
                        self.go2rtc.register_rtsp(name, cam["rtsp_url"])
                    except Exception as exc:
                        print(f"[shard] go2rtc register {name}: {exc}")

        print(f"[shard] active cameras={len(self.workers)}")
    def _on_events(self, camera: dict[str, Any], events: list) -> None:
        for detection in events:
            EVENT_SERVICE.handle_detection(
                camera_pk=camera.get("id"),
                camera_name=camera.get("name", ""),
                camera_id=camera.get("camera_id", ""),
                detection=detection,
            )

    def _process_job(self, job: InferJob) -> list:
        worker = self.workers.get(job.camera_key)
        ctx = worker.ctx if worker else CameraContext(
            key=job.camera_key,
            name=job.camera_key,
            camera_id=job.camera_key,
            device="",
            modules=job.modules,
        )
        return REGISTRY.process_frame(job.frame, ctx, job.modules)

    def status(self) -> dict[str, Any]:
        config = get_config()
        per = int(config.get("shards", {}).get("cameras_per_worker", 8))
        with self._lock:
            cameras = {name: worker.status() for name, worker in self.workers.items()}
            n = len(self.workers)
        shard_count = max(1, (n + per - 1) // per) if n else 0
        return {
            "started": self.started,
            "camera_count": n,
            "cameras_per_worker": per,
            "logical_shards": shard_count,
            "cameras": cameras,
            "scheduler": SCHEDULER.status(),
            "go2rtc": self.go2rtc.health(),
        }

    def get_worker(self, name: str) -> Optional[CameraWorker]:
        return self.workers.get(name)


SHARDS = ShardManager()
