"""Module registry — load configured detection plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import get_config
from .crowd import CrowdModule
from .frisking import FriskingModule
from .loitering import LoiteringModule
from .ppe import PPEModule
from .sling import SlingModule

MODULE_TYPES = {
    "ppe": PPEModule,
    "crowd": CrowdModule,
    "loitering": LoiteringModule,
    "frisking": FriskingModule,
    "sling": SlingModule,
}


class ModuleRegistry:
    def __init__(self) -> None:
        self.modules: dict[str, Any] = {}
        self.loaded = False

    def available(self) -> list[dict[str, Any]]:
        config = get_config().get("modules", {})
        rows = []
        for module_id, cls in MODULE_TYPES.items():
            mod_cfg = config.get(module_id, {})
            rows.append(
                {
                    "id": module_id,
                    "enabled": bool(mod_cfg.get("enabled", True)),
                    "labels": list(getattr(cls, "labels", mod_cfg.get("labels", []))),
                    "model": mod_cfg.get("model"),
                    "config": mod_cfg,
                }
            )
        return rows

    def ensure_loaded(self, device_override: str | None = None) -> None:
        if self.loaded:
            return
        config = get_config()
        device = device_override or config.get("hardware", {}).get("device", "cpu")
        models_dir = Path(config["paths"]["models_dir"])
        for module_id, cls in MODULE_TYPES.items():
            mod_cfg = dict(config.get("modules", {}).get(module_id, {}))
            if not mod_cfg.get("enabled", True):
                continue
            instance = cls()
            model_name = mod_cfg.get("model") or ""
            model_path = models_dir / model_name if model_name else None
            # Fall back to bare name (Ultralytics may download) or sibling project models
            path_str = str(model_path) if model_path and model_path.exists() else model_name
            if not path_str:
                print(f"[modules] skip {module_id}: no model configured")
                continue
            try:
                instance.load(path_str, device, mod_cfg)
                self.modules[module_id] = instance
                print(f"[modules] loaded {module_id} model={path_str} device={device}")
            except Exception as exc:
                print(f"[modules] failed to load {module_id}: {exc}")
        self.loaded = True

    def get(self, module_id: str):
        return self.modules.get(module_id)

    def process_frame(self, frame, ctx, module_ids: list[str]):
        events = []
        for module_id in module_ids:
            module = self.modules.get(module_id)
            if module is None:
                continue
            try:
                events.extend(module.process(frame, ctx))
            except Exception as exc:
                print(f"[modules] {module_id} process error: {exc}")
        return events


REGISTRY = ModuleRegistry()
