"""Gotisheel AI 2.0 configuration loader."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]  # gotisheel_ai_2/
BACKEND_DIR = Path(__file__).resolve().parents[2]  # backend/
DEFAULT_CONFIG_PATH = BACKEND_DIR / "config" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if os.getenv("GOTISHEEL_HOST"):
        result.setdefault("server", {})["host"] = os.getenv("GOTISHEEL_HOST")
    if os.getenv("GOTISHEEL_PORT"):
        result.setdefault("server", {})["port"] = int(os.getenv("GOTISHEEL_PORT"))
    if os.getenv("GOTISHEEL_DEVICE"):
        result.setdefault("hardware", {})["device"] = os.getenv("GOTISHEEL_DEVICE")
    if os.getenv("GOTISHEEL_FFMPEG_HWACCEL"):
        result.setdefault("hardware", {})["ffmpeg_hwaccel"] = os.getenv("GOTISHEEL_FFMPEG_HWACCEL")
    if os.getenv("GOTISHEEL_DB_URL"):
        result.setdefault("db", {})["url"] = os.getenv("GOTISHEEL_DB_URL")
    if os.getenv("GOTISHEEL_WEBHOOK_URL"):
        result.setdefault("webhook", {})["server_url"] = os.getenv("GOTISHEEL_WEBHOOK_URL")
    if os.getenv("GOTISHEEL_GO2RTC_URL"):
        result.setdefault("go2rtc", {})["api_url"] = os.getenv("GOTISHEEL_GO2RTC_URL")
    return result


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    config_path = Path(os.getenv("GOTISHEEL_CONFIG", str(DEFAULT_CONFIG_PATH)))
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    local_path = config_path.with_name("local.yaml")
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})

    config = _env_overrides(config)

    # Resolve relative paths against project root
    data_dir = ROOT_DIR / config.get("paths", {}).get("data_dir", "data")
    models_dir = ROOT_DIR / config.get("paths", {}).get("models_dir", "data/models")
    events_dir = ROOT_DIR / config.get("paths", {}).get("events_dir", "data/events")
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("paths", {})
    config["paths"]["data_dir"] = str(data_dir)
    config["paths"]["models_dir"] = str(models_dir)
    config["paths"]["events_dir"] = str(events_dir)
    config["_root_dir"] = str(ROOT_DIR)

    db_url = config.get("db", {}).get("url", "sqlite:///./data/gotisheel.db")
    if db_url.startswith("sqlite:///./"):
        rel = db_url.replace("sqlite:///./", "", 1)
        config.setdefault("db", {})["url"] = f"sqlite:///{ROOT_DIR / rel}"

    return config


def reload_config() -> dict[str, Any]:
    get_config.cache_clear()
    return get_config()
