"""Event persistence + multipart webhook POST (upload_api_parameters compatible)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

from ..core.config import get_config
from ..core.db import Event, session_factory
from ..modules.base import DetectionEvent


def _compress_jpeg(frame: np.ndarray, max_kb: int) -> Optional[bytes]:
    quality = 92
    while quality >= 20:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        data = buf.tobytes()
        if len(data) <= max_kb * 1024:
            return data
        quality -= 8
    return data if ok else None


def _thumbnail(frame: np.ndarray, size: tuple[int, int], max_kb: int) -> Optional[bytes]:
    thumb = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
    return _compress_jpeg(thumb, max_kb)


class EventService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.recent: list[dict[str, Any]] = []

    def handle_detection(
        self,
        camera_pk: Optional[int],
        camera_name: str,
        camera_id: str,
        detection: DetectionEvent,
    ) -> dict[str, Any]:
        config = get_config()
        webhook = config.get("webhook", {})
        events_dir = Path(config["paths"]["events_dir"])
        events_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"{camera_name}_{detection.label}_{stamp}"
        image_path = ""
        thumb_path = ""
        image_bytes = None
        thumb_bytes = None

        if detection.frame is not None:
            annotated = detection.frame.copy()
            for box in detection.boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    detection.label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            image_bytes = _compress_jpeg(annotated, int(webhook.get("event_image_max_kb", 900)))
            size = webhook.get("thumbnail_size", [175, 175])
            thumb_bytes = _thumbnail(annotated, (int(size[0]), int(size[1])), int(webhook.get("thumbnail_max_kb", 25)))
            if image_bytes:
                image_path = str(events_dir / f"{base}.jpg")
                Path(image_path).write_bytes(image_bytes)
            if thumb_bytes:
                thumb_path = str(events_dir / f"{base}_thumb.jpg")
                Path(thumb_path).write_bytes(thumb_bytes)

        uploaded = False
        if webhook.get("enabled") and image_bytes is not None:
            uploaded = self._post_webhook(
                webhook,
                camera_id=camera_id or camera_name,
                label=detection.label,
                boxes=detection.boxes,
                image_bytes=image_bytes,
                thumb_bytes=thumb_bytes,
            )

        session = session_factory()
        try:
            row = Event(
                camera_pk=camera_pk,
                camera_name=camera_name,
                camera_id=camera_id,
                label=detection.label,
                module_id=detection.module_id,
                detail=detection.detail,
                bbox=detection.boxes,
                image_path=image_path,
                thumbnail_path=thumb_path,
                uploaded=uploaded,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            payload = {
                "id": row.id,
                "camera_name": camera_name,
                "camera_id": camera_id,
                "label": detection.label,
                "module_id": detection.module_id,
                "detail": detection.detail,
                "uploaded": uploaded,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "thumbnail_path": thumb_path,
            }
        finally:
            session.close()

        with self._lock:
            self.recent.insert(0, payload)
            self.recent = self.recent[:200]
        return payload

    def _post_webhook(
        self,
        webhook: dict[str, Any],
        camera_id: str,
        label: str,
        boxes: list,
        image_bytes: bytes,
        thumb_bytes: Optional[bytes],
    ) -> bool:
        url = f"{str(webhook.get('server_url', '')).rstrip('/')}{webhook.get('upload_endpoint', '')}"
        timeout = float(webhook.get("upload_timeout_seconds", 15))
        event_json = json.dumps(
            {
                "camera_id": camera_id,
                "label": label,
                "bbox": boxes,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        files = {
            "image": ("event.jpg", image_bytes, "image/jpeg"),
        }
        if thumb_bytes:
            files["thumbnail"] = ("thumb.jpg", thumb_bytes, "image/jpeg")
        data = {
            "event": event_json,
            "label": label,
            "bbox": json.dumps(boxes),
            "camera_id": camera_id,
        }
        try:
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                print(f"[webhook] OK {label} camera={camera_id}")
                return True
            print(f"[webhook] FAIL HTTP {response.status_code}: {response.text[:200]}")
            return False
        except requests.RequestException as exc:
            print(f"[webhook] ERROR {exc}")
            return False

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        session = session_factory()
        try:
            rows = session.query(Event).order_by(Event.id.desc()).limit(limit).all()
            return [
                {
                    "id": row.id,
                    "camera_name": row.camera_name,
                    "camera_id": row.camera_id,
                    "label": row.label,
                    "module_id": row.module_id,
                    "detail": row.detail,
                    "uploaded": row.uploaded,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "thumbnail_path": row.thumbnail_path,
                }
                for row in rows
            ]
        finally:
            session.close()


EVENT_SERVICE = EventService()
