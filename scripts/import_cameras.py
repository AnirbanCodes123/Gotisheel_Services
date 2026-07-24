#!/usr/bin/env python3
"""Import RTSP mapping files from sibling bypass apps into Gotisheel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.db import Camera, init_db, session_factory  # noqa: E402
from app.engine.shard_manager import SHARDS  # noqa: E402


DEFAULTS = {
    "ppe": (ROOT.parent / "ppe_bypass" / "ppe_camera_id_rtsp.txt", ["ppe"]),
    "crowd": (ROOT.parent / "crowd_loitering_bypass" / "crowd_loitering_id_rtsp.txt", ["crowd", "loitering"]),
    "sling": (ROOT.parent / "sling_bypass" / "sling_camera_id_rtsp.txt", ["sling"]),
}


def sanitize(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-").lower()
    return out or "camera"


def infer_modules(camera_id: str, fallback: list[str]) -> list[str]:
    name = camera_id.lower()
    if name.startswith("crowd"):
        return ["crowd"]
    if name.startswith("cafe"):
        return ["loitering"]
    return list(fallback)


def import_file(path: Path, modules: list[str]) -> list[str]:
    created = []
    session = session_factory()
    current_id = None
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("rtsp://") or lower.startswith("rtsp:"):
                rtsp = (
                    line.split(":", 1)[1].strip()
                    if lower.startswith("rtsp:") and not lower.startswith("rtsp://")
                    else line
                )
                if not current_id:
                    continue
                mods = infer_modules(current_id, modules)
                name = sanitize(current_id.split("_", 1)[0])
                base = name
                n = 2
                while session.query(Camera).filter(Camera.name == name).first():
                    name = f"{base}-{n}"
                    n += 1
                session.add(
                    Camera(
                        name=name,
                        camera_id=current_id,
                        rtsp_url=rtsp,
                        enabled=True,
                        modules=mods,
                    )
                )
                created.append(name)
                current_id = None
            else:
                current_id = line.rstrip(":").strip()
        session.commit()
    finally:
        session.close()
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Import cameras into Gotisheel AI 2.0")
    parser.add_argument("--preset", choices=list(DEFAULTS.keys()) + ["all"], default="all")
    parser.add_argument("--path", help="Custom mapping file")
    parser.add_argument("--modules", nargs="*", default=[])
    args = parser.parse_args()

    init_db()
    imported = []
    if args.path:
        imported += import_file(Path(args.path), args.modules)
    else:
        presets = DEFAULTS if args.preset == "all" else {args.preset: DEFAULTS[args.preset]}
        for key, (path, modules) in presets.items():
            if not path.exists():
                print(f"[skip] {key}: missing {path}")
                continue
            batch = import_file(path, modules)
            print(f"[ok] {key}: imported {len(batch)}")
            imported += batch

    try:
        SHARDS.reload_cameras()
    except Exception as exc:
        print(f"[warn] reload skipped (server may not be running): {exc}")

    print(f"Done. Total imported this run: {len(imported)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
