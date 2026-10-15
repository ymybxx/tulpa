#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "ONNX Runtime compatibility builds require macOS arm64." >&2
  exit 1
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../.." && pwd)"
build_python="${1:?Pass the desktop Python 3.11 interpreter}"
"$build_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Desktop runtime requires Python 3.11"'
cache_root="$project_root/.cache/macos-12.3-wheels/onnxruntime-1.28.0"
source_dir="$cache_root/source"
source_commit=45de2a8b06d62989b3ab55ba7dc58a27ca83f9fc
helper="$script_dir/onnxruntime-compat-artifacts.py"

if "$build_python" "$helper" verify --root "$cache_root" --script "$0" 2>/dev/null; then
  printf '%s\n' "$cache_root"
  exit 0
fi
mkdir -p "$cache_root"
if [[ ! -d "$source_dir/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-checkout --branch v1.28.0 --filter=blob:none \
    https://github.com/microsoft/onnxruntime.git "$source_dir" >&2
  git -C "$source_dir" fetch --depth 1 origin "$source_commit" >&2
  GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" checkout --detach "$source_commit" >&2
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$source_commit" ]]; then
  echo "Cached ONNX Runtime source is not the installed runtime's exact commit." >&2
  exit 1
fi
git -C "$source_dir" diff --quiet HEAD --
git -C "$source_dir" submodule update --init --depth 1 cmake/external/onnx >&2
if [[ ! -x "$cache_root/venv/bin/python" ]]; then
  UV_CACHE_DIR="$project_root/.cache/uv" uv venv --python "$build_python" "$cache_root/venv" >&2
fi
UV_CACHE_DIR="$project_root/.cache/uv" uv pip install --python "$cache_root/venv/bin/python" \
  setuptools==78.1.1 wheel==0.45.1 numpy==2.4.2 packaging==26.0 sympy==1.14.0 flatbuffers==25.12.19 >&2

# Match the released CPU kernels and optimization defaults, not the 14.0-only
# distribution's optional CoreML/Azure providers. Fetch pinned upstream deps;
# Homebrew's newer Protobuf headers cannot be mixed with the pinned protoc.
MACOSX_DEPLOYMENT_TARGET=12.3 _PYTHON_HOST_PLATFORM=macosx-12.0-arm64 \
  "$cache_root/venv/bin/python" "$source_dir/tools/ci_build/build.py" \
    --build_dir "$cache_root/build" --config Release \
    --build_shared_lib --build_wheel --osx_arch arm64 --apple_deploy_target 12.3 \
    --skip_tests --skip_submodule_sync --compile_no_warning_as_error \
    --use_binskim_compliant_compile_flags --parallel 8 \
    --cmake_extra_defines CMAKE_OSX_DEPLOYMENT_TARGET=12.3 CMAKE_OSX_ARCHITECTURES=arm64 \
      onnxruntime_BUILD_UNIT_TESTS=OFF FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
      CMAKE_IGNORE_PREFIX_PATH=/opt/homebrew \
    --update --build >&2
"$build_python" "$helper" publish --root "$cache_root" --script "$0"
printf '%s\n' "$cache_root"
