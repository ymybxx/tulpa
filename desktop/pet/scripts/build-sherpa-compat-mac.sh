#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "sherpa compatibility wheels require macOS arm64." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../.." && pwd)"
build_python="${1:?Pass the Python 3.11 interpreter used for the desktop runtime}"
ort_root="${2:?Pass the verified ONNX Runtime 1.28.0 SDK directory}"
if [[ ! -f "$ort_root/lib/libonnxruntime.1.28.0.dylib" || ! -f "$ort_root/include/onnxruntime_cxx_api.h" ]]; then
  echo "The same-version ONNX Runtime 1.28.0 SDK is required." >&2
  exit 1
fi
"$build_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Desktop runtime requires Python 3.11"'
cache_root="$project_root/.cache/macos-12.3-wheels/sherpa-onnx-1.13.4"
helper="$script_dir/sherpa-compat-wheels.py"
dist="$cache_root/dist"

if "$build_python" "$helper" verify --dist "$dist" --script "$0" --ort-root "$ort_root" 2>/dev/null; then
  printf '%s\n' "$dist"
  exit 0
fi

mkdir -p "$cache_root/sources"
download() {
  local url="$1" expected="$2" target="$3"
  if [[ -f "$target" && "$(shasum -a 256 "$target" | awk '{print $1}')" == "$expected" ]]; then
    return
  fi
  curl --fail --location --retry 2 --connect-timeout 20 "$url" -o "$target.part" >&2
  if [[ "$(shasum -a 256 "$target.part" | awk '{print $1}')" != "$expected" ]]; then
    echo "Checksum mismatch: $target.part" >&2
    exit 1
  fi
  mv "$target.part" "$target"
}

download \
  https://files.pythonhosted.org/packages/ca/f8/735244770b4bc63f85fabdad0e46d6ec1f4cc24e64f6e082c2e0fea92b8c/sherpa_onnx-1.13.4.tar.gz \
  29547692418513ad88034c2b5f98985e33042b2351e4ab375469f19a8de18c5f \
  "$cache_root/sources/sherpa_onnx-1.13.4.tar.gz"

build_root="$(mktemp -d "$cache_root/build.XXXXXX")"
tar -xzf "$cache_root/sources/sherpa_onnx-1.13.4.tar.gz" -C "$build_root"
UV_CACHE_DIR="$project_root/.cache/uv" uv venv --python "$build_python" "$build_root/venv" >&2
UV_CACHE_DIR="$project_root/.cache/uv" uv pip install --python "$build_root/venv/bin/python" setuptools==84.0.0 wheel==0.45.1 >&2

export MACOSX_DEPLOYMENT_TARGET=12.3
export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ort_root/include"
export SHERPA_ONNXRUNTIME_LIB_DIR="$ort_root/lib"
export SHERPA_ONNX_MAKE_ARGS=-j4
# Do not let pybind11 choose a newer Homebrew Python; PYTHON_EXECUTABLE alone
# is ignored by modern CMake's FindPython. Keep all CPU inference/TTS features;
# upstream's macro omits optional CoreML linkage absent from the CPU-only SDK.
export SHERPA_ONNX_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=12.3 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_CXX_FLAGS=-DSHERPA_ONNX_DISABLE_COREML -DPython_EXECUTABLE=$build_root/venv/bin/python -DPython_FIND_VIRTUALENV=ONLY -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=ON"
unset SHERPA_ONNX_SPLIT_PYTHON_PACKAGE SHERPA_ONNX_IS_FOR_PYPI
(
  cd "$build_root/sherpa_onnx-1.13.4"
  "$build_root/venv/bin/python" setup.py bdist_wheel >&2
)
wheels=("$build_root/sherpa_onnx-1.13.4/dist/"*.whl)
if [[ ${#wheels[@]} != 1 || ! -f "${wheels[0]}" ]]; then
  echo "Expected one source-built sherpa wheel." >&2
  exit 1
fi
"$build_python" "$helper" split --wheel "${wheels[0]}" --dist "$build_root/dist" --script "$0" \
  --ort-root "$ort_root"
if [[ -d "$dist" ]]; then
  mv "$dist" "$build_root/previous-dist"
fi
mv "$build_root/dist" "$dist"
printf '%s\n' "$dist"
