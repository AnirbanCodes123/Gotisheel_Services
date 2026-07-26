"""Module registry — load configured detection plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import get_config
from .crowd import CrowdModule
from .frisking import FriskingModule
from .loitering import LoiteringModule
from .paths import can_autodownload, resolve_model_path
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
        self.load_errors: dict[str, str] = {}

    def available(self) -> list[dict[str, Any]]:
        config = get_config().get("modules", {})
        rows = []
        for module_id, cls in MODULE_TYPES.items():
            mod_cfg = config.get(module_id, {})
            labels = list(getattr(cls, "labels", mod_cfg.get("labels", [])))
            # Camera UI options: expand PPE into nomask/nohairnet (ppe2 labels)
            if module_id == "ppe":
                options = [{"id": label, "label": label, "module_id": "ppe"} for label in labels]
            else:
                options = [{"id": module_id, "label": module_id, "module_id": module_id}]
            rows.append(
                {
                    "id": module_id,
                    "enabled": bool(mod_cfg.get("enabled", True)),
                    "loaded": module_id in self.modules,
                    "labels": labels,
                    "options": options,
                    "model": mod_cfg.get("model"),
                    "error": self.load_errors.get(module_id),
                    "config": mod_cfg,
                }
            )
        return rows

    def ensure_loaded(self, device_override: str | None = None) -> None:
        if self.loaded:
            return
        config = get_config()
        device = device_override or config.get("hardware", {}).get("device", "cpu")
        print(f"[modules] loading detection plugins device={device}")
        for module_id, cls in MODULE_TYPES.items():
            mod_cfg = dict(config.get("modules", {}).get(module_id, {}))
            if not mod_cfg.get("enabled", True):
                print(f"[modules] skip {module_id}: disabled in config")
                continue
            instance = cls()
            model_name = mod_cfg.get("model") or ""
            path_str = resolve_model_path(model_name)
            if not path_str and model_name and can_autodownload(model_name):
                # Official Ultralytics names may download; custom weights must exist on disk
                path_str = model_name
            if not path_str:
                msg = f"model file not found: {model_name or '(empty)'} (put it in data/models/)"
                self.load_errors[module_id] = msg
                print(f"[modules] skip {module_id}: {msg}")
                continue
            if Path(path_str).is_file():
                print(f"[modules] resolved {module_id} -> {path_str}")
            # Resolve optional secondary models onto config for modules that need them
            if module_id == "frisking" and mod_cfg.get("person_model"):
                person_path = resolve_model_path(mod_cfg.get("person_model"))
                if person_path:
                    mod_cfg["person_model"] = person_path
                elif not can_autodownload(str(mod_cfg.get("person_model"))):
                    print(f"[modules] warn frisking person_model missing: {mod_cfg.get('person_model')}")
            if module_id == "frisking" and mod_cfg.get("security_model"):
                security_path = resolve_model_path(mod_cfg.get("security_model"))
                if security_path:
                    mod_cfg["security_model"] = security_path
                    print(f"[modules] frisking security_model={security_path}")
                else:
                    print(f"[modules] warn frisking security_model missing: {mod_cfg.get('security_model')}")
            try:
                instance.load(path_str, device, mod_cfg)
                self.modules[module_id] = instance
                print(f"[modules] loaded {module_id} model={path_str} device={device}")
            except Exception as exc:
                self.load_errors[module_id] = str(exc)
                print(f"[modules] FAILED to load {module_id}: {exc}")
        self.loaded = True
        loaded_ids = ", ".join(self.modules.keys()) or "(none)"
        print(f"[modules] ready: {loaded_ids}")
        if self.load_errors:
            for mid, err in self.load_errors.items():
                print(f"[modules] missing/failed {mid}: {err}")

    def get(self, module_id: str):
        return self.modules.get(module_id)

    def process_frame(self, frame, ctx, module_ids: list[str]):
        # Fresh overlays each inference tick (modules append live boxes)
        ctx.state["overlays"] = []
        events = []
        # Normalize UI labels → engine module ids (nomask/nohairnet → ppe)
        normalized: list[str] = []
        ppe_labels: list[str] = []
        for mid in module_ids or []:
            key = str(mid).strip().lower()
            if key in ("nomask", "nohairnet"):
                if "ppe" not in normalized:
                    normalized.append("ppe")
                ppe_labels.append(key)
            elif key and key not in normalized:
                normalized.append(key)
        if ppe_labels:
            extra = dict(ctx.state.get("extra") or {})
            # Prefer explicit camera extra; else use checkbox selection
            if not extra.get("ppe_labels"):
                extra["ppe_labels"] = ppe_labels
                ctx.state["extra"] = extra
        for module_id in normalized:
            module = self.modules.get(module_id)
            if module is None:
                continue
            try:
                events.extend(module.process(frame, ctx))
            except Exception as exc:
                print(f"[modules] {module_id} process error: {exc}")
        return events


REGISTRY = ModuleRegistry()
