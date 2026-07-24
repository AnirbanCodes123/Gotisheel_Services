from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from ..core.config import get_config, reload_config
from ..engine.shard_manager import SHARDS
from ..modules import REGISTRY
from ..services.events import EVENT_SERVICE
from ..services.metrics import collect_metrics
from ..services.model_registry import list_models, save_uploaded_model

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    return {"ok": True, "name": "Gotisheel AI 2.0"}


@router.get("/config")
def read_config():
    return get_config()


@router.post("/config/reload")
def config_reload():
    return reload_config()


@router.get("/modules")
def modules():
    return REGISTRY.available()


@router.get("/models")
def models():
    return list_models()


@router.post("/models/upload")
async def models_upload(
    file: UploadFile = File(...),
    module_id: str | None = Form(None),
    name: str | None = Form(None),
):
    return await save_uploaded_model(file, module_id=module_id, name=name)


@router.get("/events")
def events(limit: int = 50):
    return EVENT_SERVICE.list_recent(limit=limit)


@router.get("/system")
def system():
    return collect_metrics(
        {
            "shards": SHARDS.status(),
            "brand": "Gotisheel AI 2.0",
            "device": get_config().get("hardware", {}).get("device"),
        }
    )


@router.get("/runtime")
def runtime():
    return SHARDS.status()


@router.get("/event-media")
def event_media(path: str):
    from pathlib import Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    file_path = Path(path).resolve()
    events_root = Path(get_config()["paths"]["events_dir"]).resolve()
    if events_root not in file_path.parents and file_path != events_root:
        raise HTTPException(403, "Invalid media path")
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)
