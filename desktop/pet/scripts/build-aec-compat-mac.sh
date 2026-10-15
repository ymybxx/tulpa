#!/usr/bin/env bash
# Build the unchanged AEC 1.0.1 sources with their bundled static Abseil.
# stdout is only the verified wheel path; diagnostics belong on stderr.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "AEC compatibility wheels require macOS Apple Silicon." >&2
  exit 1
fi

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "$script_path")/../../.." && pwd)"
python_executable="${1:-$project_root/.venv/bin/python}"
uv_executable="$(command -v uv)"
source_sha="f03a3ce3f45bf15a04a8504d0f692a651180ca2d3171bd9f3de0516e5f2b8360"
cache_root="$project_root/.cache/macos-12.3-wheels/aec-audio-processing-1.0.1"
python_abi="$("$python_executable" -c 'import sys; print(sys.implementation.cache_tag)')"
cache_dir="$cache_root/$python_abi"
mkdir -p "$cache_dir" "$cache_root/downloads"

verify_cached() {
  "$python_executable" - "$cache_dir" "$source_sha" "$script_path" "$project_root" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

cache, source_sha, script, project = sys.argv[1:]
cache = Path(cache)
manifest = cache / "wheel.json"
if not manifest.is_file():
    raise SystemExit(1)
try:
    info = json.loads(manifest.read_text())
    assert info["sourceSha256"] == source_sha
    assert info["buildScriptSha256"] == hashlib.sha256(Path(script).read_bytes()).hexdigest()
    wheel = cache / info["wheel"]
    assert wheel.parent == cache and wheel.suffix == ".whl"
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == info["wheelSha256"]
    with tempfile.TemporaryDirectory(prefix="verify-", dir=cache) as temporary:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(temporary)
        binaries = [p for p in Path(temporary).rglob("*") if p.suffix in {".so", ".dylib"}]
        assert len(binaries) == 2, binaries
        for binary in binaries:
            assert subprocess.check_output(["/usr/bin/lipo", "-archs", str(binary)], text=True).strip() == "arm64"
            build = subprocess.check_output(["/usr/bin/otool", "-l", str(binary)], text=True)
            minimum = re.search(r"\bminos\s+([\d.]+)", build)
            assert minimum and tuple(map(int, minimum[1].split("."))) <= (12, 3), build
            linked = subprocess.check_output(["/usr/bin/otool", "-L", str(binary)], text=True)
            for line in linked.splitlines()[1:]:
                dependency = line.strip().split(" (", 1)[0]
                assert dependency.startswith(("/usr/lib/", "/System/Library/", "@rpath/")), dependency
                if dependency.startswith("@rpath/"):
                    assert dependency == "@rpath/libwebrtc-audio-processing-2.dylib", dependency
        # Import the relocated wheel, never a build-tree or installed AEC copy.
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((temporary, str(Path(project) / "src"))),
            "PYTHONNOUSERSITE": "1",
        }
        smoke_test = """
from pathlib import Path
import aec_audio_processing
from fish_memory.companion.audio_processing import WebRtcAec
assert Path(aec_audio_processing.__file__).is_relative_to(Path.cwd())
processor = WebRtcAec()
silence = bytes(320)
for _ in range(100):
    processor.process_reverse_stream(silence)
    assert len(processor.process_stream(silence)) == 320
"""
        subprocess.run(
            [sys.executable, "-c", smoke_test], cwd=temporary,
            env=environment, check=True, stdout=sys.stderr,
        )
    print(wheel.resolve())
except Exception as error:
    print(f"AEC cached wheel needs rebuilding: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

if cached_wheel="$(verify_cached)"; then
  printf '%s\n' "$cached_wheel"
  exit 0
fi

archive="$cache_root/downloads/aec_audio_processing-1.0.1.tar.gz"
if [[ ! -f "$archive" ]] || [[ "$(shasum -a 256 "$archive" | cut -d ' ' -f 1)" != "$source_sha" ]]; then
  curl --fail --location --retry 2 --output "$archive" \
    "https://files.pythonhosted.org/packages/9e/f1/d4c91eeaed6a2d91b448eb3eb17c150a505973aa7c5495d8772325918e38/aec_audio_processing-1.0.1.tar.gz" >&2
fi
[[ "$(shasum -a 256 "$archive" | cut -d ' ' -f 1)" == "$source_sha" ]] || { echo "AEC source SHA256 mismatch" >&2; exit 1; }
work_dir="$(mktemp -d "$cache_dir/build.XXXXXX")"
tar -xzf "$archive" -C "$work_dir"
source_dir="$work_dir/aec_audio_processing-1.0.1"
webrtc_dir="$source_dir/webrtc-audio-processing"

# A separate environment prevents changing the application/developer venv.
export UV_CACHE_DIR="$project_root/.cache/uv"
"$uv_executable" venv --python "$python_executable" "$work_dir/venv" >&2
"$uv_executable" pip install --python "$work_dir/venv/bin/python" \
  setuptools==84.0.0 wheel==0.48.0 build==1.5.0 \
  meson==1.12.0 ninja==1.13.0 swig==4.4.1 >&2
export PATH="$work_dir/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export MACOSX_DEPLOYMENT_TARGET=12.3
export CC="$(xcrun --find clang)" CXX="$(xcrun --find clang++)"
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CFLAGS="-mmacosx-version-min=12.3" CXXFLAGS="-mmacosx-version-min=12.3" LDFLAGS="-mmacosx-version-min=12.3"
mkdir -p "$work_dir/pkgconfig-empty"
export PKG_CONFIG_LIBDIR="$work_dir/pkgconfig-empty" PKG_CONFIG_PATH="" CMAKE_PREFIX_PATH="" CPPFLAGS=""
unset FISH_MEMORY_AEC_ABSEIL_LIB DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH

meson setup --wrap-mode=forcefallback --buildtype=release --default-library=shared \
  --prefix="$webrtc_dir/install" "$webrtc_dir/build" "$webrtc_dir" >&2
ninja -C "$webrtc_dir/build" >&2
ninja -C "$webrtc_dir/build" install >&2
# Upstream installs an absolute dylib ID. Make the wheel independently relocatable.
install_name_tool -id '@rpath/libwebrtc-audio-processing-2.dylib' \
  "$webrtc_dir/install/lib/libwebrtc-audio-processing-2.dylib" >&2
codesign --force --sign - "$webrtc_dir/install/lib/libwebrtc-audio-processing-2.dylib" >&2
cp "$webrtc_dir/subprojects/abseil-cpp-20240722.0/LICENSE" "$source_dir/src/files/ABSEIL-LICENSE.txt"
"$work_dir/venv/bin/python" -m build --wheel --no-isolation --outdir "$cache_dir" "$source_dir" >&2
"$python_executable" - "$cache_dir" "$source_sha" "$script_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
cache, source_sha, script = sys.argv[1:]
cache = Path(cache)
wheels = list(cache.glob("aec_audio_processing-1.0.1-*.whl"))
assert len(wheels) == 1, wheels
wheel = wheels[0]
(cache / "wheel.json").write_text(json.dumps({"wheel": wheel.name, "wheelSha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "sourceSha256": source_sha, "buildScriptSha256": hashlib.sha256(Path(script).read_bytes()).hexdigest()}, indent=2) + "\n")
PY
verify_cached
