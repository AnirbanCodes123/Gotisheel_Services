#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/models data/events

# Start go2rtc if binary exists
if command -v go2rtc >/dev/null 2>&1; then
  echo "[start] launching go2rtc on :1984"
  go2rtc -config "$ROOT/go2rtc/go2rtc.yaml" >/tmp/gotisheel-go2rtc.log 2>&1 &
  echo $! > /tmp/gotisheel-go2rtc.pid
else
  echo "[start] go2rtc not found in PATH — WebRTC optional. Install from https://github.com/AlexxIT/go2rtc/releases"
fi

export PYTHONPATH="$ROOT/backend:${PYTHONPATH:-}"
cd "$ROOT/backend"
echo "[start] Gotisheel AI 2.0 on http://0.0.0.0:9100"
exec python3 -m app.main
