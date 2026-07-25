"""MJPEG + stream helper endpoints (WebRTC via go2rtc URLs)."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from ..core.db import Camera, session_factory
from ..engine.shard_manager import SHARDS

router = APIRouter(tags=["streams"])


def _stream_payload(camera_name: str, online: bool, runtime: dict | None = None) -> dict:
    health = SHARDS.go2rtc.health()
    payload = {
        "camera": camera_name,
        "online": online,
        "webrtc_url": SHARDS.go2rtc.stream_webrtc_url(camera_name),
        "go2rtc_webrtc_url": SHARDS.go2rtc.stream_go2rtc_webrtc_url(camera_name),
        "mse_url": SHARDS.go2rtc.stream_mse_url(camera_name),
        "mjpeg_url": f"/api/streams/{camera_name}/mjpeg",
        "snapshot_url": f"/api/streams/{camera_name}/snapshot.jpg",
        "go2rtc_api_url": SHARDS.go2rtc.api_url,
        "go2rtc_ok": bool(health.get("ok")),
        "go2rtc": health,
    }
    if runtime is not None:
        payload["runtime"] = runtime
    return payload


@router.get("/api/streams/{camera_name}/info")
def stream_info(camera_name: str):
    worker = SHARDS.get_worker(camera_name)
    if worker is None:
        return _stream_payload(camera_name, online=False)
    status = worker.status()
    return _stream_payload(camera_name, online=bool(status.get("online")), runtime=status)


@router.post("/api/streams/{camera_name}/ensure-webrtc")
def ensure_webrtc(camera_name: str):
    """Register camera RTSP with go2rtc so the in-app player can connect."""
    session = session_factory()
    try:
        row = session.query(Camera).filter(Camera.name == camera_name).one_or_none()
        if row is None:
            raise HTTPException(404, "Camera not found")
        result = SHARDS.go2rtc.ensure_stream(row.name, row.rtsp_url)
    finally:
        session.close()
    health = SHARDS.go2rtc.health()
    return {
        "camera": camera_name,
        "register": result,
        "go2rtc_ok": bool(health.get("ok")),
        "webrtc_url": SHARDS.go2rtc.stream_webrtc_url(camera_name),
        "go2rtc_webrtc_url": SHARDS.go2rtc.stream_go2rtc_webrtc_url(camera_name),
    }


@router.get("/api/streams/{camera_name}/snapshot.jpg")
def snapshot(camera_name: str):
    worker = SHARDS.get_worker(camera_name)
    if worker is None:
        raise HTTPException(404, "Camera worker not running")
    frame = worker.get_jpeg()
    if frame is None:
        raise HTTPException(503, "No frame yet")
    return Response(content=frame, media_type="image/jpeg")


@router.get("/api/streams/{camera_name}/mjpeg")
def mjpeg(camera_name: str):
    worker = SHARDS.get_worker(camera_name)
    if worker is None:
        raise HTTPException(404, "Camera worker not running")

    def generate():
        while True:
            frame = worker.get_jpeg()
            if frame is None:
                time.sleep(0.2)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.08)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
