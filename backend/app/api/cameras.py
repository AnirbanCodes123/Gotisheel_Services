from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.config import ROOT_DIR, get_config
from ..core.db import Camera, get_session
from ..engine.shard_manager import SHARDS

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


class CameraIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    camera_id: str = ""
    rtsp_url: str = Field(..., min_length=5)
    enabled: bool = True
    device: str = ""  # cuda:0 | cpu | ""
    detect_fps: float = 0.0
    stream_role: str = "both"
    modules: List[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    camera_id: Optional[str] = None
    rtsp_url: Optional[str] = None
    enabled: Optional[bool] = None
    device: Optional[str] = None
    detect_fps: Optional[float] = None
    stream_role: Optional[str] = None
    modules: Optional[List[str]] = None
    extra: Optional[dict[str, Any]] = None


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return cleaned or "camera"


def _serialize(row: Camera) -> dict[str, Any]:
    worker = SHARDS.get_worker(row.name)
    live = worker.status() if worker else {}
    return {
        "id": row.id,
        "name": row.name,
        "camera_id": row.camera_id,
        "rtsp_url": row.rtsp_url,
        "enabled": row.enabled,
        "device": row.device,
        "detect_fps": row.detect_fps,
        "stream_role": row.stream_role,
        "modules": row.modules or [],
        "extra": row.extra or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "runtime": live,
        "webrtc_url": SHARDS.go2rtc.stream_webrtc_url(row.name),
        "mse_url": SHARDS.go2rtc.stream_mse_url(row.name),
    }


@router.get("")
def list_cameras(db: Session = Depends(get_session)):
    rows = db.query(Camera).order_by(Camera.id.asc()).all()
    return [_serialize(row) for row in rows]


@router.post("")
def create_camera(payload: CameraIn, db: Session = Depends(get_session)):
    name = _sanitize_name(payload.name)
    if db.query(Camera).filter(Camera.name == name).first():
        raise HTTPException(400, f"Camera name already exists: {name}")
    row = Camera(
        name=name,
        camera_id=payload.camera_id or name,
        rtsp_url=payload.rtsp_url.strip(),
        enabled=payload.enabled,
        device=payload.device,
        detect_fps=payload.detect_fps,
        stream_role=payload.stream_role,
        modules=payload.modules,
        extra=payload.extra,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    SHARDS.reload_cameras()
    return _serialize(row)


@router.get("/{camera_id}")
def get_camera(camera_id: int, db: Session = Depends(get_session)):
    row = db.query(Camera).filter(Camera.id == camera_id).first()
    if not row:
        raise HTTPException(404, "Camera not found")
    return _serialize(row)


@router.patch("/{camera_id}")
def update_camera(camera_id: int, payload: CameraUpdate, db: Session = Depends(get_session)):
    row = db.query(Camera).filter(Camera.id == camera_id).first()
    if not row:
        raise HTTPException(404, "Camera not found")
    data = payload.model_dump(exclude_unset=True)
    if "name" in data and data["name"]:
        data["name"] = _sanitize_name(data["name"])
    for key, value in data.items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    SHARDS.reload_cameras()
    return _serialize(row)


@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_session)):
    row = db.query(Camera).filter(Camera.id == camera_id).first()
    if not row:
        raise HTTPException(404, "Camera not found")
    db.delete(row)
    db.commit()
    SHARDS.reload_cameras()
    return {"ok": True}


@router.post("/reload")
def reload_runtime():
    SHARDS.reload_cameras()
    return SHARDS.status()


@router.post("/import")
def import_mapping(path: str, default_modules: List[str] | None = None, db: Session = Depends(get_session)):
    """Import alternating camera_id / rtsp lines from existing mapping files."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT_DIR.parent / path
        if not file_path.exists():
            file_path = ROOT_DIR / path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {path}")

    modules = default_modules or []
    created = []
    current_id = None
    text = file_path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("rtsp://") or lower.startswith("rtsp:"):
            rtsp = line.split(":", 1)[1].strip() if lower.startswith("rtsp:") and not lower.startswith("rtsp://") else line
            if not current_id:
                continue
            name = _sanitize_name(current_id.split("_", 1)[0])
            original = name
            suffix = 2
            while db.query(Camera).filter(Camera.name == name).first():
                name = f"{original}-{suffix}"
                suffix += 1
            row = Camera(
                name=name,
                camera_id=current_id,
                rtsp_url=rtsp,
                enabled=True,
                modules=modules,
            )
            db.add(row)
            created.append(name)
            current_id = None
        else:
            current_id = line.rstrip(":").strip()
    db.commit()
    SHARDS.reload_cameras()
    return {"imported": len(created), "cameras": created, "source": str(file_path)}
