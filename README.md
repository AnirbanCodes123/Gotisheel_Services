# Gotisheel AI 2.0

Unified Frigate/OpenNVR-style video AI platform for PPE, frisking, crowd, loitering, and sling detection.

## Features

- Add RTSP cameras from the UI (name, RTSP, modules, GPU/CPU)
- Pluggable YOLO modules: `ppe`, `frisking`, `crowd`, `loitering`, `sling`
- ffmpeg CUDA decode (CPU fallback)
- Shared inference scheduler + camera sharding (scale toward 200 cams)
- Live MJPEG + go2rtc WebRTC URLs
- Event store + multipart POST webhook (same contract as existing `upload_api_parameters.json`)
- CPU / GPU / RAM system panel
- Model `.pt` upload registry

## Quick start

```bash
cd gotisheel_ai_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy your .pt weights into data/models/
# e.g. DS_PPE_6.pt, yolo11s.pt, sling_4_11m_1500.pt

chmod +x scripts/start_all.sh
./scripts/start_all.sh
```

Open **http://localhost:9100**

Optional WebRTC: install [go2rtc](https://github.com/AlexxIT/go2rtc/releases) so `go2rtc` is on `PATH`.

### Import existing camera lists

```bash
python3 scripts/import_cameras.py --preset all
# or
python3 scripts/import_cameras.py --path ../ppe_bypass/ppe_camera_id_rtsp.txt --modules ppe
```

## Configuration

Edit [`backend/config/default.yaml`](backend/config/default.yaml) or create `backend/config/local.yaml`.

Env overrides:

| Env | Meaning |
|-----|---------|
| `GOTISHEEL_PORT` | API port (default 9100) |
| `GOTISHEEL_DEVICE` | `cuda:0` or `cpu` |
| `GOTISHEEL_FFMPEG_HWACCEL` | `cuda` or `none` |
| `GOTISHEEL_WEBHOOK_URL` | Event POST base URL |
| `GOTISHEEL_DB_URL` | SQLAlchemy URL |
| `GOTISHEEL_GO2RTC_URL` | go2rtc API |

## API (selected)

- `GET/POST /api/cameras` — list / create
- `PATCH/DELETE /api/cameras/{id}`
- `POST /api/cameras/import?path=...`
- `GET /api/streams/{name}/mjpeg` — live MJPEG
- `GET /api/streams/{name}/info` — WebRTC/MSE URLs
- `GET /api/modules` `GET/POST /api/models`
- `GET /api/events` `GET /api/system` `GET /api/runtime`

## Scaling notes

- `shards.cameras_per_worker` groups cameras logically
- Detect FPS is independent of live preview FPS
- Inference is serialized through one scheduler queue per process (protects GPU)
- Browse streams on demand (WebRTC/MJPEG); do not open 200 browser sockets at once
- Hardware (GPU count / NVDEC / NIC) still bounds absolute camera count

## Layout

```
gotisheel_ai_2/
  backend/app/     FastAPI + engine + modules
  frontend/ui/     Professional dashboard (served by API)
  go2rtc/          WebRTC sidecar config
  scripts/         start + import helpers
  data/models/     .pt weights
  data/events/     saved event images
```

Existing bypass apps under `ppe_bypass/`, `frisking_bypass/`, etc. are **not deleted**; their logic is ported as plugins here.
