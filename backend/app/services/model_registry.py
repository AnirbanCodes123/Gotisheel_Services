"""Model file registry helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile

from ..core.config import get_config
from ..core.db import ModelAsset, session_factory


def list_models() -> list[dict[str, Any]]:
    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    disk = []
    for path in sorted(models_dir.glob("*.pt")):
        disk.append(
            {
                "filename": path.name,
                "path": str(path),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        )
    session = session_factory()
    try:
        rows = session.query(ModelAsset).order_by(ModelAsset.id.desc()).all()
        registered = [
            {
                "id": row.id,
                "name": row.name,
                "filename": row.filename,
                "module_id": row.module_id,
                "description": row.description,
            }
            for row in rows
        ]
    finally:
        session.close()
    return {"disk": disk, "registered": registered}


async def save_uploaded_model(file: UploadFile, module_id: Optional[str] = None, name: Optional[str] = None) -> dict[str, Any]:
    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "model.pt").name
    if not filename.endswith(".pt"):
        filename = f"{filename}.pt"
    dest = models_dir / filename
    with open(dest, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    display = name or filename
    session = session_factory()
    try:
        existing = session.query(ModelAsset).filter(ModelAsset.filename == filename).first()
        if existing:
            existing.name = display
            existing.module_id = module_id
            row = existing
        else:
            row = ModelAsset(name=display, filename=filename, module_id=module_id)
            session.add(row)
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
            "name": row.name,
            "filename": row.filename,
            "module_id": row.module_id,
            "path": str(dest),
        }
    finally:
        session.close()
