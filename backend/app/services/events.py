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


def _union_bbox(boxes: list) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    xs1 = [float(b[0]) for b in boxes]
    ys1 = [float(b[1]) for b in boxes]
    xs2 = [float(b[2]) for b in boxes]
    ys2 = [float(b[3]) for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _build_event_payload(camera_id: str, label: str, boxes: list, scores: list | None = None) -> dict[str, Any]:
    """Frigate-compatible event body used by ppe_bypass upload_api."""
    ts = time.time()
    event_id = f"{label}-{camera_id}-{int(ts * 1000)}"
    norm_boxes = [[float(v) for v in box] for box in boxes]
    primary = _union_bbox(norm_boxes)
    width = max(primary[2] - primary[0], 1.0)
    height = max(primary[3] - primary[1], 1.0)
    top_score = float(max(scores)) if scores else 1.0
    state = {
        "id": event_id,
        "camera": camera_id,
        "frame_time": ts,
        "snapshot": {
            "frame_time": ts,
            "box": primary,
            "area": width * height,
            "region": [0, 0, 1280, 720],
            "score": top_score,
            "attributes": [],
        },
        "label": label,
        "sub_label": None,
        "top_score": top_score,
        "false_positive": False,
        "start_time": ts,
        "end_time": ts + 10,
        "score": top_score,
        "box": primary,
        "boxes": norm_boxes,
        "area": width * height,
        "ratio": round(width / height, 2),
        "region": [0, 0, 1280, 720],
        "stationary": False,
        "motionless_count": 0,
        "position_changes": 1,
        "current_zones": [],
        "entered_zones": [],
        "has_clip": False,
        "has_snapshot": True,
        "attributes": {},
        "current_attributes": [],
    }
    return {"before": {**state}, "after": {**state}, "type": "end"}


class EventService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.recent: list[dict[str, Any]] = []
        self._temp_id = -1

    def handle_detection(
        self,
        camera_pk: Optional[int],
        camera_name: str,
        camera_id: str,
        detection: DetectionEvent,
    ) -> dict[str, Any]:
        """Publish to UI immediately, then persist + webhook in background work."""
        config = get_config()
        webhook = config.get("webhook", {})
        events_dir = Path(config["paths"]["events_dir"])
        events_dir.mkdir(parents=True, exist_ok=True)
        upload_camera_id = (camera_id or "").strip() or camera_name

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"{camera_name}_{detection.label}_{stamp}_{int(time.time() * 1000) % 100000}"
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
            thumb_bytes = _thumbnail(
                annotated, (int(size[0]), int(size[1])), int(webhook.get("thumbnail_max_kb", 25))
            )
            if image_bytes:
                image_path = str(events_dir / f"{base}.jpg")
                Path(image_path).write_bytes(image_bytes)
            if thumb_bytes:
                thumb_path = str(events_dir / f"{base}_thumb.jpg")
                Path(thumb_path).write_bytes(thumb_bytes)

        # Optimistic UI payload first (instant Events page)
        with self._lock:
            self._temp_id -= 1
            temp_id = self._temp_id
        payload = {
            "id": temp_id,
            "camera_name": camera_name,
            "camera_id": upload_camera_id,
            "label": detection.label,
            "module_id": detection.module_id,
            "detail": detection.detail,
            "uploaded": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "thumbnail_path": thumb_path,
            "pending": True,
        }
        with self._lock:
            self.recent.insert(0, payload)
            self.recent = self.recent[:200]

        uploaded = False
        if webhook.get("enabled") and image_bytes is not None:
            uploaded = self._post_webhook(
                webhook,
                camera_id=upload_camera_id,
                label=detection.label,
                boxes=detection.boxes,
                scores=detection.scores,
                image_bytes=image_bytes,
                thumb_bytes=thumb_bytes,
            )

        session = session_factory()
        try:
            row = Event(
                camera_pk=camera_pk,
                camera_name=camera_name,
                camera_id=upload_camera_id,
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
            final = {
                "id": row.id,
                "camera_name": camera_name,
                "camera_id": upload_camera_id,
                "label": detection.label,
                "module_id": detection.module_id,
                "detail": detection.detail,
                "uploaded": uploaded,
                "created_at": row.created_at.isoformat() if row.created_at else payload["created_at"],
                "thumbnail_path": thumb_path,
                "pending": False,
            }
        finally:
            session.close()

        with self._lock:
            # Replace optimistic temp entry with persisted row
            self.recent = [final if e.get("id") == temp_id else e for e in self.recent]
            # Dedup if both exist
            seen = set()
            cleaned = []
            for e in self.recent:
                key = e.get("id")
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(e)
            self.recent = cleaned[:200]
        return final

    def _post_webhook(
        self,
        webhook: dict[str, Any],
        camera_id: str,
        label: str,
        boxes: list,
        scores: list,
        image_bytes: bytes,
        thumb_bytes: Optional[bytes],
    ) -> bool:
        url = f"{str(webhook.get('server_url', '')).rstrip('/')}{webhook.get('upload_endpoint', '')}"
        timeout = float(webhook.get("upload_timeout_seconds", 15))
        event_payload = _build_event_payload(camera_id, label, boxes, scores)
        primary = _union_bbox(boxes)
        files = {
            "image": ("event.jpg", image_bytes, "image/jpeg"),
        }
        if thumb_bytes:
            files["thumbnail"] = ("thumbnail.jpg", thumb_bytes, "image/jpeg")
        data = {
            "event": json.dumps(event_payload),
            "label": label,
            "bbox": json.dumps([float(v) for v in primary]),
        }
        try:
            print(f"[webhook] REQUEST {label} camera={camera_id} url={url}")
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    result = {}
                if result.get("skipped"):
                    print(f"[webhook] SKIPPED {label} camera={camera_id}: {result.get('message', '')}")
                else:
                    print(f"[webhook] OK {label} camera={camera_id}")
                return True
            print(f"[webhook] FAIL HTTP {response.status_code}: {response.text[:200]}")
            return False
        except requests.RequestException as exc:
            print(f"[webhook] ERROR {exc}")
            return False

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        # Prefer in-memory list for instant Events page (includes optimistic entries)
        with self._lock:
            if self.recent:
                return list(self.recent[:limit])
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
                    "pending": False,
                }
                for row in rows
            ]
        finally:
            session.close()


EVENT_SERVICE = EventService()
