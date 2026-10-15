#!/usr/bin/env bash
# Starts (or restarts) the dev Electron build of tulpa with an isolated data
# directory and debug ports, detached from the calling shell. See
# docs/desktop-dev-testing.md.
#
#   scripts/dev/start-dev-app.sh [data-dir]
#
# data-dir defaults to ~/Library/Application Support/tulpa-dev (not $TMPDIR:
# macOS purges files there after a few days unused). Pages are reachable over CDP on
# :9230 (renderer) and the main process on :9231 (Node inspector). Output
# goes to <data-dir>/app.log.
set -euo pipefail

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${1:-$HOME/Library/Application Support/tulpa-dev}"
cdp_port="${TULPA_CDP_PORT:-9230}"
inspect_port="${TULPA_INSPECT_PORT:-9231}"
# Electron 42+ downloads its binary on first use rather than at npm install.
(cd "$pet_root" && node -e "require('electron')" >/dev/null)
electron_app="$pet_root/node_modules/electron/dist/Electron.app"
mkdir -p "$data_dir"

# One dev instance at a time: quit the previous one and its local service.
pkill -f "MacOS/Electron .*$pet_root" 2>/dev/null || true
for _ in $(seq 1 15); do
  pgrep -f "MacOS/Electron .*$pet_root|desktop_runtime api" >/dev/null || break
  sleep 1
done

: > "$data_dir/app.log"
# `open -n -g`: a new instance, launched in the background so the person's
# current app keeps focus; the app is its own process, not a child of this shell.
open -n -g -a "$electron_app" --stdout "$data_dir/app.log" --stderr "$data_dir/app.log" \
  --env FISH_MEMORY_USER_DATA_DIR="$data_dir" \
  --args "--inspect=$inspect_port" "$pet_root" "--remote-debugging-port=$cdp_port"

for _ in $(seq 1 30); do
  curl -s --max-time 1 "http://127.0.0.1:$cdp_port/json/list" | grep -q index.html && break
  sleep 1
done
echo "tulpa 开发版已启动：数据目录 $data_dir，页面调试 :$cdp_port，主进程调试 :$inspect_port，日志 $data_dir/app.log"
