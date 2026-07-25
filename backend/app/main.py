"""Gotisheel AI 2.0 — FastAPI application entrypoint."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api import cameras_router, streams_router, system_router
from app.core.config import get_config
from app.core.db import init_db
from app.engine.shard_manager import SHARDS


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        SHARDS.start()
    except Exception as exc:
        print(f"[boot] shard start warning: {exc}")
    yield
    SHARDS.stop()


app = FastAPI(title="Gotisheel AI 2.0", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras_router)
app.include_router(streams_router)
app.include_router(system_router)

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
STATIC_UI = ROOT_DIR / "frontend" / "ui"
ui_root = FRONTEND_DIST if (FRONTEND_DIST / "index.html").exists() else STATIC_UI

if ui_root.exists():
    app.mount("/static", StaticFiles(directory=str(ui_root)), name="static")

    @app.get("/")
    def index():
        return FileResponse(ui_root / "index.html")

    @app.get("/styles.css")
    def styles():
        return FileResponse(ui_root / "styles.css", media_type="text/css")

    @app.get("/app.js")
    def app_js():
        return FileResponse(
            ui_root / "app.js",
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/player.html")
    def player():
        return FileResponse(
            ui_root / "player.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )


def main():
    import uvicorn

    config = get_config()
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = int(config.get("server", {}).get("port", 9100))
    uvicorn.run("app.main:app", host=host, port=port, reload=False, app_dir=str(BACKEND_DIR))


if __name__ == "__main__":
    main()
