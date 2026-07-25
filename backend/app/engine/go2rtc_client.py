"""go2rtc client for WebRTC / MSE restream registration."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests
import yaml

from ..core.config import ROOT_DIR, get_config


class Go2rtcClient:
    def __init__(self) -> None:
        config = get_config()
        self.enabled = bool(config.get("go2rtc", {}).get("enabled", True))
        self.api_url = str(config.get("go2rtc", {}).get("api_url", "http://127.0.0.1:1984")).rstrip("/")
        self.config_path = ROOT_DIR / config.get("go2rtc", {}).get("config_path", "go2rtc/go2rtc.yaml")

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "ok": False, "detail": "disabled"}
        try:
            response = requests.get(f"{self.api_url}/api", timeout=2)
            return {"enabled": True, "ok": response.ok, "status_code": response.status_code}
        except requests.RequestException as exc:
            return {"enabled": True, "ok": False, "detail": str(exc)}

    def player_url(self, stream_name: str) -> str:
        """In-app player (reliable). Opens on Gotisheel, not blank go2rtc stream.html."""
        return f"/player.html?src={quote(stream_name, safe='')}"

    def stream_webrtc_url(self, stream_name: str) -> str:
        # Prefer Gotisheel player; keep direct go2rtc link as secondary.
        return self.player_url(stream_name)

    def stream_go2rtc_webrtc_url(self, stream_name: str) -> str:
        return f"{self.api_url}/webrtc.html?src={quote(stream_name, safe='')}&media=video"

    def stream_mse_url(self, stream_name: str) -> str:
        return f"{self.api_url}/api/stream.mp4?src={quote(stream_name, safe='')}"

    def has_stream(self, stream_name: str) -> bool:
        if not self.enabled:
            return False
        try:
            response = requests.get(f"{self.api_url}/api/streams", timeout=2)
            if not response.ok:
                return False
            data = response.json()
            return stream_name in (data or {})
        except requests.RequestException:
            return False

    def register_rtsp(self, stream_name: str, rtsp_url: str) -> dict[str, Any]:
        """Register/update a stream. Tries API put; also writes yaml for persistence."""
        if not self.enabled:
            return {"ok": False, "detail": "go2rtc disabled"}

        api_ok = False
        api_detail = ""
        try:
            # go2rtc: PUT /api/streams?name=x&src=rtsp://...
            response = requests.put(
                f"{self.api_url}/api/streams",
                params={"name": stream_name, "src": rtsp_url},
                timeout=5,
            )
            api_ok = response.ok
            api_detail = response.text[:200]
            # Some builds accept POST with JSON body
            if not api_ok:
                response = requests.put(
                    f"{self.api_url}/api/streams",
                    params={"src": f"{stream_name}:{rtsp_url}"},
                    timeout=5,
                )
                api_ok = response.ok
                api_detail = response.text[:200] or api_detail
        except requests.RequestException as exc:
            api_detail = str(exc)

        self._upsert_yaml(stream_name, rtsp_url)
        return {
            "ok": api_ok,
            "detail": api_detail,
            "webrtc_url": self.stream_webrtc_url(stream_name),
            "go2rtc_webrtc_url": self.stream_go2rtc_webrtc_url(stream_name),
        }

    def ensure_stream(self, stream_name: str, rtsp_url: str) -> dict[str, Any]:
        if self.has_stream(stream_name):
            return {"ok": True, "detail": "already registered", "webrtc_url": self.stream_webrtc_url(stream_name)}
        return self.register_rtsp(stream_name, rtsp_url)

    def unregister(self, stream_name: str) -> None:
        try:
            requests.delete(f"{self.api_url}/api/streams", params={"src": stream_name}, timeout=2)
        except requests.RequestException:
            pass
        self._remove_yaml(stream_name)

    def _load_yaml(self) -> dict[str, Any]:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            return {"streams": {}}
        with open(self.config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        data.setdefault("streams", {})
        return data

    def _save_yaml(self, data: dict[str, Any]) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False)

    def _upsert_yaml(self, stream_name: str, rtsp_url: str) -> None:
        data = self._load_yaml()
        data.setdefault("streams", {})[stream_name] = rtsp_url
        data.setdefault("api", {}).setdefault("listen", ":1984")
        webrtc = data.setdefault("webrtc", {})
        webrtc.setdefault("listen", ":8555")
        # Help browsers on the same host connect ICE candidates.
        candidates = webrtc.setdefault("candidates", [])
        for candidate in ("127.0.0.1", "stun:stun.l.google.com:19302"):
            if candidate not in candidates:
                candidates.append(candidate)
        self._save_yaml(data)

    def _remove_yaml(self, stream_name: str) -> None:
        data = self._load_yaml()
        data.get("streams", {}).pop(stream_name, None)
        self._save_yaml(data)
