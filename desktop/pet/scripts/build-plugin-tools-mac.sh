#!/usr/bin/env bash
set -euo pipefail

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$pet_root/../.." && pwd)"
export UV_CACHE_DIR="$project_root/.cache/uv"
tools_root="$pet_root/runtime/mac-arm64/fish-plugin-tools"
live_plugin_root="${FISH_LIVE_PLUGIN_DIR:-$project_root/plugins/live}"
plugin_python="$(uv python find 3.11.15)"
node "$pet_root/scripts/assert-packaging-idle.mjs"
mkdir -p "$tools_root/plugins"

# A standard interpreter, independent of the frozen API and native audio runtime.
plugin_base="$("$plugin_python" -c 'import sys; print(sys.base_prefix)')"
if [[ ! -x "$tools_root/python/bin/python3" ]]; then
  cp -R "$plugin_base" "$tools_root/python"
fi
tool_uv="$(uv run --no-project --isolated --python "$plugin_python" --with uv==0.11.21 python -c 'import uv; print(uv.find_uv_bin())')"
cp "$tool_uv" "$tools_root/uv"

cd "$project_root"
rm -f "$tools_root"/fish_plugin_sdk-*.whl
uv build --package fish-plugin-sdk --wheel --out-dir "$tools_root"
# The live plugin is an independent, optional project; a clean host checkout
# must still build its SDK/runtime without that project's source code.
rm -f "$tools_root"/plugins/fish.live-*.fish-plugin
if [[ -f "$live_plugin_root/fish-plugin.json" ]]; then
  sdk_wheels=("$tools_root"/fish_plugin_sdk-*.whl)
  # Notarized releases need the native extensions inside bundled wheels signed.
  sign_args=()
  if [[ "${FISH_MEMORY_RELEASE_SIGN:-}" == "1" ]]; then
    sign_args=(--codesign-identity "$FISH_MEMORY_CODESIGN_IDENTITY")
  fi
  PYTHONPATH="$live_plugin_root/src" \
  uv run --no-project --isolated --python "$plugin_python" \
    --with "$project_root/sdk/python[tools]" --with 'httpx>=0.28,<1' --with 'Brotli>=1.1,<2' \
    python -m fish_plugin_sdk.cli build "$live_plugin_root" --sdk-wheel "${sdk_wheels[0]}" ${sign_args[@]+"${sign_args[@]}"}
  plugin_version="$("$plugin_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$live_plugin_root/fish-plugin.json")"
  cp "$live_plugin_root/dist/fish.live-$plugin_version-darwin-arm64-py311.fish-plugin" "$tools_root/plugins/"
fi
"$plugin_python" "$pet_root/scripts/verify-macos-compatibility.py" "$tools_root" --target 12.3
"$tools_root/python/bin/python3" -c 'import ssl, venv, sys; assert sys.version_info[:3] == (3, 11, 15); print("Plugin CPython ready:", sys.version.split()[0])'
echo "插件 SDK 与运行时已准备：$tools_root"
