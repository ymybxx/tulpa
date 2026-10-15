#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "桌宠输入桥接目前只支持 macOS Apple Silicon。" >&2
  exit 1
fi

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
node_binary="$(command -v node)"
node_headers="${FISH_MEMORY_NODE_HEADERS:-}"
if [[ -z "$node_headers" ]]; then
  for candidate in "${node_binary%/bin/node}/include/node" \
    "/usr/local/include/node" "/opt/homebrew/include/node" \
    "$HOME/Library/Caches/node-gyp/$(node -p 'process.versions.node')/include/node"; do
    if [[ -f "$candidate/node_api.h" ]]; then
      node_headers="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$node_headers/node_api.h" ]]; then
  echo "缺少 Node-API 头文件；安装 Node 开发头文件或设置 FISH_MEMORY_NODE_HEADERS。" >&2
  exit 1
fi

output="$pet_root/runtime/mac-arm64/pet-application-activation.node"
mkdir -p "$(dirname "$output")"
# Only stable Node-API is used; no Electron/V8 ABI-specific dependency.
xcrun clang++ -std=c++17 -fobjc-arc -bundle -undefined dynamic_lookup \
  -DNAPI_VERSION=8 -arch arm64 -mmacosx-version-min=12.3 \
  -I "$node_headers" -framework AppKit \
  "$pet_root/native/pet-input/application-activation.mm" \
  "$pet_root/native/pet-input/mouse-monitor.mm" -o "$output"
codesign --force --sign - "$output"
echo "桌宠输入桥接已生成：$output"
