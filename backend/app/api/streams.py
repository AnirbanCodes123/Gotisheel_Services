"""MJPEG + stream helper endpoints (WebRTC via go2rtc URLs)."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from ..engine.shard_manager import SHARDS

router = APIRouter(tags=["streams"])


@router.get("/api/streams/{camera_name}/info")
def stream_info(camera_name: str):
    worker = SHARDS.get_worker(camera_name)
    if worker is None:
        # still return go2rtc URLs for cameras not yet online
        return {
            "camera": camera_name,
            "online": False,
            "webrtc_url": SHARDS.go2rtc.stream_webrtc_url(camera_name),
            "mse_url": SHARDS.go2rtc.stream_mse_url(camera_name),
            "mjpeg_url": f"/api/streams/{camera_name}/mjpeg",
            "snapshot_url": f"/api/streams/{camera_name}/snapshot.jpg",
        }
    status = worker.status()
    return {
        "camera": camera_name,
        "online": status.get("online"),
        "webrtc_url": SHARDS.go2rtc.stream_webrtc_url(camera_name),
        "mse_url": SHARDS.go2rtc.stream_mse_url(camera_name),
        "mjpeg_url": f"/api/streams/{camera_name}/mjpeg",
        "snapshot_url": f"/api/streams/{camera_name}/snapshot.jpg",
        "runtime": status,
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
