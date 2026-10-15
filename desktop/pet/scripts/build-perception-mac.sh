#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Fish Memory 感知 Helper 首版只能在 macOS Apple Silicon 上构建。" >&2
  exit 1
fi

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_root="$pet_root/native/perception"
output_root="$pet_root/runtime/mac-arm64"
bundle_root="$output_root/FishMemoryPerceptionHelper.app"
bundle_contents="$bundle_root/Contents"
bundle_executable_root="$bundle_contents/MacOS"
module_cache="$package_root/.build/module-cache"

if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

rm -rf "$bundle_root" "$output_root/fish-memory-perception-helper"
mkdir -p "$bundle_executable_root" "$module_cache"
export MACOSX_DEPLOYMENT_TARGET="12.3"
export CLANG_MODULE_CACHE_PATH="$module_cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$module_cache"
xcrun swift build --disable-sandbox --package-path "$package_root" -c release --arch arm64
binary_root="$(xcrun swift build --disable-sandbox --package-path "$package_root" -c release --arch arm64 --show-bin-path)"
cp "$binary_root/fish-memory-perception-helper" \
  "$bundle_executable_root/fish-memory-perception-helper"
cp "$package_root/Resources/Info.plist" "$bundle_contents/Info.plist"
chmod 755 "$bundle_executable_root/fish-memory-perception-helper"
codesign --force --sign - "$bundle_root"
codesign --verify --strict "$bundle_root"
echo "感知 Helper 已生成：$bundle_root"
