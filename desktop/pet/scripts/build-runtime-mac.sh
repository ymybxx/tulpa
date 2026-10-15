#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Fish Memory 首版运行时只能在 macOS Apple Silicon 上构建。" >&2
  exit 1
fi

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$pet_root/../.." && pwd)"
export UV_CACHE_DIR="$project_root/.cache/uv"
runtime_parent="$pet_root/runtime/mac-arm64"
work_root="$project_root/.cache/pyinstaller-mac-arm64"
download_proxy="${FISH_MEMORY_DOWNLOAD_PROXY:-auto}"

node "$pet_root/scripts/assert-packaging-idle.mjs"

if [[ "$download_proxy" == "auto" ]]; then
  if nc -z 127.0.0.1 7890 >/dev/null 2>&1; then
    download_proxy="http://127.0.0.1:7890"
  else
    download_proxy="none"
  fi
fi
if [[ "$download_proxy" == "none" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
else
  export HTTP_PROXY="$download_proxy"
  export HTTPS_PROXY="$download_proxy"
  export ALL_PROXY="$download_proxy"
fi

export MACOSX_DEPLOYMENT_TARGET="12.3"
unset FISH_MEMORY_AEC_ABSEIL_LIB

# Keep compatibility overrides inside the desktop build. The server/developer
# lock remains unchanged, and no packages are installed into the project venv.
build_python="$(uv python find 3.11)"
ort_compat_root="$(bash "$pet_root/scripts/build-onnxruntime-compat-mac.sh" "$build_python")"
aec_wheel="$(bash "$pet_root/scripts/build-aec-compat-mac.sh" "$build_python")"
sherpa_wheel_dir="$(bash "$pet_root/scripts/build-sherpa-compat-mac.sh" "$build_python" "$ort_compat_root/sdk")"
shopt -s nullglob
ort_wheels=("$ort_compat_root"/dist/onnxruntime-1.28.0-*.whl)
sherpa_wheels=("$sherpa_wheel_dir"/sherpa_onnx*.whl)
if [[ ! -f "$aec_wheel" || "${#ort_wheels[@]}" -ne 1 || "${#sherpa_wheels[@]}" -ne 2 ]]; then
  echo "macOS 12.3 原生兼容依赖未完整生成。" >&2
  exit 1
fi
compatibility_args=(
  --with-requirements "$pet_root/packaging/runtime-macos-12.3.txt"
  --with "${ort_wheels[0]}"
  --with "$aec_wheel"
)
for sherpa_wheel in "${sherpa_wheels[@]}"; do
  compatibility_args+=(--with "$sherpa_wheel")
done

# The user may have launched the app while native dependencies were building.
node "$pet_root/scripts/assert-packaging-idle.mjs"
rm -rf "$runtime_parent/fish-memory-runtime" "$work_root"
mkdir -p "$runtime_parent" "$work_root"

cd "$project_root"
PYINSTALLER_CONFIG_DIR="$work_root/config" \
UV_CACHE_DIR="$project_root/.cache/uv" \
uv run --locked --isolated --python "$build_python" \
  --no-dev --group packaging --extra voice \
  "${compatibility_args[@]}" pyinstaller \
  --noconfirm \
  --clean \
  --distpath "$runtime_parent" \
  --workpath "$work_root/build" \
  "$pet_root/packaging/fish-memory-runtime.spec"

"$runtime_parent/fish-memory-runtime/fish-memory-runtime" doctor >/dev/null
"$build_python" "$pet_root/scripts/verify-macos-compatibility.py" \
  "$runtime_parent/fish-memory-runtime" --target 12.3
echo "Python 运行时已生成：$runtime_parent/fish-memory-runtime"
