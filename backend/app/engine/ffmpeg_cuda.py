"""ffmpeg CUDA / CPU RTSP frame capture."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np

from ..core.config import get_config


def normalize_rtsp_url(rtsp_url: str) -> str:
    parsed = urlsplit(rtsp_url)
    if parsed.scheme.lower() != "rtsp" or "@" not in parsed.netloc:
        return rtsp_url
    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    if ":" in userinfo:
        username, password = userinfo.split(":", 1)
        safe = f"{quote(unquote(username), safe='')}:{quote(unquote(password), safe='')}"
    else:
        safe = quote(unquote(userinfo), safe="")
    return urlunsplit((parsed.scheme, f"{safe}@{hostinfo}", parsed.path, parsed.query, parsed.fragment))


class FFmpegCudaCapture:
    """Low-latency RTSP reader via ffmpeg (NVDEC when hwaccel=cuda)."""

    def __init__(self, rtsp_url: str, hwaccel: str | None = None, camera_key: str = ""):
        config = get_config()
        ff = config.get("ffmpeg", {})
        hardware = config.get("hardware", {})
        self.rtsp_url = normalize_rtsp_url(rtsp_url)
        self.camera_key = camera_key
        self.hwaccel = (hwaccel or hardware.get("ffmpeg_hwaccel", "cuda")).lower()
        self.ffmpeg_path = ff.get("path", "ffmpeg")
        self.transport = ff.get("transport", "tcp")
        self.hw_device = str(ff.get("hw_device", "0"))
        self.connect_timeout_us = str(ff.get("connect_timeout_us", "5000000"))
        self.width = int(ff.get("output_width", 1280))
        self.height = int(ff.get("output_height", 720))
        self.frame_bytes = self.width * self.height * 3
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._error: Optional[str] = None
        self._fps = 0.0
        self.backend_name = "ffmpeg-cuda" if self.hwaccel == "cuda" else "ffmpeg"

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"ffcap-{self.camera_key}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_proc()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def fps(self) -> float:
        return self._fps

    def _stop_proc(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            self.transport,
            "-timeout",
            self.connect_timeout_us,
        ]
        if self.hwaccel == "cuda" and shutil.which(self.ffmpeg_path):
            cmd += [
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                self.hw_device,
                "-hwaccel_output_format",
                "nv12",
            ]
        cmd += [
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            self.rtsp_url,
            "-an",
            "-vf",
            f"scale={self.width}:{self.height},format=bgr24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        return cmd

    def _run(self) -> None:
        while self._running:
            try:
                self._error = None
                self._proc = subprocess.Popen(
                    self._build_cmd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=self.frame_bytes * 2,
                )
                assert self._proc.stdout is not None
                frames = 0
                t0 = time.time()
                while self._running:
                    raw = self._proc.stdout.read(self.frame_bytes)
                    if not raw or len(raw) < self.frame_bytes:
                        self._error = "ffmpeg pipe ended"
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                    with self._lock:
                        self._latest = frame
                    frames += 1
                    elapsed = time.time() - t0
                    if elapsed >= 1.0:
                        self._fps = frames / elapsed
                        frames = 0
                        t0 = time.time()
            except Exception as exc:
                self._error = str(exc)
            finally:
                self._stop_proc()
            if self._running:
                time.sleep(2.0)


class OpenCVCapture:
    """Fallback OpenCV FFMPEG capture."""

    def __init__(self, rtsp_url: str, camera_key: str = ""):
        self.rtsp_url = rtsp_url
        self.camera_key = camera_key
        self.backend_name = "opencv"
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._error: Optional[str] = None
        self._fps = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"ocv-{self.camera_key}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def fps(self) -> float:
        return self._fps

    def _run(self) -> None:
        while self._running:
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._cap.isOpened():
                self._error = "Could not open RTSP"
                time.sleep(3)
                continue
            frames = 0
            t0 = time.time()
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    self._error = "RTSP read failed"
                    break
                with self._lock:
                    self._latest = frame
                self._error = None
                frames += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self._fps = frames / elapsed
                    frames = 0
                    t0 = time.time()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if self._running:
                time.sleep(2.0)


def create_capture(rtsp_url: str, device: str = "", camera_key: str = ""):
    config = get_config()
    hwaccel = config.get("hardware", {}).get("ffmpeg_hwaccel", "cuda")
    if device.startswith("cpu"):
        hwaccel = "none"
    elif device.startswith("cuda"):
        hwaccel = "cuda"
    try:
        cap = FFmpegCudaCapture(rtsp_url, hwaccel=hwaccel, camera_key=camera_key)
        if shutil.which(cap.ffmpeg_path):
            return cap
    except Exception:
        pass
    return OpenCVCapture(rtsp_url, camera_key=camera_key)
