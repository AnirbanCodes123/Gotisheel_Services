#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/models data/events bin go2rtc

GO2RTC_BIN=""
GO2RTC_LOCAL="$ROOT/bin/go2rtc"
GO2RTC_VERSION="${GOTISHEEL_GO2RTC_VERSION:-v1.9.14}"

resolve_go2rtc_asset() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    linux)
      case "$arch" in
        x86_64|amd64) echo "go2rtc_linux_amd64" ;;
        aarch64|arm64) echo "go2rtc_linux_arm64" ;;
        armv7l|armhf) echo "go2rtc_linux_arm" ;;
        i386|i686) echo "go2rtc_linux_i386" ;;
        *) return 1 ;;
      esac
      ;;
    darwin)
      case "$arch" in
        x86_64) echo "go2rtc_mac_amd64.zip" ;;
        arm64) echo "go2rtc_mac_arm64.zip" ;;
        *) return 1 ;;
      esac
      ;;
    *) return 1 ;;
  esac
}

ensure_go2rtc() {
  if [[ -n "${GOTISHEEL_SKIP_GO2RTC:-}" ]]; then
    echo "[start] GOTISHEEL_SKIP_GO2RTC set — skipping go2rtc"
    return 1
  fi

  if [[ -x "$GO2RTC_LOCAL" ]]; then
    GO2RTC_BIN="$GO2RTC_LOCAL"
    return 0
  fi

  if command -v go2rtc >/dev/null 2>&1; then
    GO2RTC_BIN="$(command -v go2rtc)"
    return 0
  fi

  local asset url tmp
  asset="$(resolve_go2rtc_asset)" || {
    echo "[start] unsupported OS/arch for auto-download of go2rtc"
    return 1
  }
  url="https://github.com/AlexxIT/go2rtc/releases/download/${GO2RTC_VERSION}/${asset}"
  echo "[start] go2rtc not found — downloading ${asset} (${GO2RTC_VERSION})"
  tmp="$(mktemp -d)"
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "[start] need curl or wget to download go2rtc"
    rm -rf "$tmp"
    return 1
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$tmp/$asset"
  else
    wget -qO "$tmp/$asset" "$url"
  fi

  if [[ "$asset" == *.zip ]]; then
    if command -v unzip >/dev/null 2>&1; then
      unzip -qo "$tmp/$asset" -d "$tmp"
    else
      python3 - "$tmp/$asset" "$tmp" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
    fi
    # mac zip contains go2rtc binary
    if [[ -f "$tmp/go2rtc" ]]; then
      mv "$tmp/go2rtc" "$GO2RTC_LOCAL"
    else
      echo "[start] zip did not contain go2rtc binary"
      rm -rf "$tmp"
      return 1
    fi
  else
    mv "$tmp/$asset" "$GO2RTC_LOCAL"
  fi
  chmod +x "$GO2RTC_LOCAL"
  rm -rf "$tmp"
  GO2RTC_BIN="$GO2RTC_LOCAL"
  echo "[start] installed go2rtc -> $GO2RTC_LOCAL"
  return 0
}

start_go2rtc() {
  if ! ensure_go2rtc; then
    echo "[start] WebRTC disabled (go2rtc unavailable). MJPEG live view still works."
    echo "[start] Manual install: https://github.com/AlexxIT/go2rtc/releases"
    return 0
  fi

  # Stop previous managed instance if still running
  if [[ -f /tmp/gotisheel-go2rtc.pid ]]; then
    old_pid="$(cat /tmp/gotisheel-go2rtc.pid 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[start] stopping previous go2rtc pid=$old_pid"
      kill "$old_pid" 2>/dev/null || true
      sleep 0.5
    fi
  fi

  echo "[start] launching go2rtc ($GO2RTC_BIN) on :1984"
  "$GO2RTC_BIN" -config "$ROOT/go2rtc/go2rtc.yaml" >/tmp/gotisheel-go2rtc.log 2>&1 &
  echo $! > /tmp/gotisheel-go2rtc.pid
  sleep 0.4
  if kill -0 "$(cat /tmp/gotisheel-go2rtc.pid)" 2>/dev/null; then
    echo "[start] go2rtc ready — UI http://127.0.0.1:1984/  log /tmp/gotisheel-go2rtc.log"
  else
    echo "[start] go2rtc failed to start — see /tmp/gotisheel-go2rtc.log"
  fi
}

start_go2rtc

export PYTHONPATH="$ROOT/backend:${PYTHONPATH:-}"
cd "$ROOT/backend"
echo "[start] Gotisheel AI 2.0 on http://0.0.0.0:9100"
exec python3 -m app.main
