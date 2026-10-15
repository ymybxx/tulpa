#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Fish Memory LiveKit 音频 Helper 首版只能在 macOS Apple Silicon 上构建。" >&2
  exit 1
fi

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_root="$pet_root/native/audio"
output_root="$pet_root/runtime/mac-arm64"
bundle_root="$output_root/FishMemoryAudioHelper.app"
bundle_contents="$bundle_root/Contents"
bundle_executable_root="$bundle_contents/MacOS"
bundle_frameworks_root="$bundle_contents/Frameworks"
bundle_resources_root="$bundle_contents/Resources"
bundle_licenses_root="$bundle_resources_root/Licenses"
module_cache="$package_root/.build/module-cache"
executable_name="fish-memory-livekit-audio-helper"

if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

rm -rf "$bundle_root"
mkdir -p \
  "$bundle_executable_root" \
  "$bundle_frameworks_root" \
  "$bundle_resources_root" \
  "$bundle_licenses_root" \
  "$module_cache"

export MACOSX_DEPLOYMENT_TARGET="12.3"
export CLANG_MODULE_CACHE_PATH="$module_cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$module_cache"

xcrun swift build \
  --disable-sandbox \
  --package-path "$package_root" \
  -c release \
  --arch arm64

binary_root="$(xcrun swift build \
  --disable-sandbox \
  --package-path "$package_root" \
  -c release \
  --arch arm64 \
  --show-bin-path)"

helper_executable="$bundle_executable_root/$executable_name"
cp "$binary_root/$executable_name" "$helper_executable"
chmod 755 "$helper_executable"
cp "$package_root/Resources/Info.plist" "$bundle_contents/Info.plist"
cp \
  "$package_root/Resources/THIRD_PARTY_NOTICES.md" \
  "$bundle_resources_root/THIRD_PARTY_NOTICES.md"

framework_names=(LiveKitWebRTC RustLiveKitUniFFI)
for framework_name in "${framework_names[@]}"; do
  framework_source="$binary_root/$framework_name.framework"
  framework_destination="$bundle_frameworks_root/$framework_name.framework"
  if [[ ! -d "$framework_source" ]]; then
    echo "$framework_name.framework 未出现在 SwiftPM release 产物中。" >&2
    exit 1
  fi
  ditto "$framework_source" "$framework_destination"

  framework_binary="$framework_destination/Versions/A/$framework_name"
  if [[ ! -f "$framework_binary" ]]; then
    echo "$framework_name.framework 缺少 macOS 二进制。" >&2
    exit 1
  fi
  if lipo -archs "$framework_binary" | grep -qw x86_64; then
    lipo -thin arm64 "$framework_binary" -output "$framework_binary.arm64"
    mv "$framework_binary.arm64" "$framework_binary"
  fi
  if [[ "$(lipo -archs "$framework_binary")" != "arm64" ]]; then
    echo "$framework_name.framework 不是纯 arm64 产物。" >&2
    exit 1
  fi
  if ! find "$framework_destination" -name PrivacyInfo.xcprivacy -print -quit | grep -q .; then
    echo "$framework_name.framework 缺少 PrivacyInfo.xcprivacy。" >&2
    exit 1
  fi
done

if [[ "$(otool -l "$helper_executable")" != *'@executable_path/../Frameworks'* ]]; then
  install_name_tool -add_rpath '@executable_path/../Frameworks' "$helper_executable"
fi

# These bundles contain dependency privacy manifests. Their generated
# Bundle.module accessors are not used by the SDK at runtime, so keep them in
# the conventional signed macOS app Resources directory rather than placing
# unsealed content at the application-bundle root.
resource_bundle_count=0
for resource_bundle in "$binary_root"/*.bundle; do
  if [[ ! -d "$resource_bundle" ]]; then
    continue
  fi
  ditto \
    "$resource_bundle" \
    "$bundle_resources_root/$(basename "$resource_bundle")"
  resource_bundle_count=$((resource_bundle_count + 1))
done
if [[ "$resource_bundle_count" -eq 0 ]]; then
  echo "SwiftPM resource bundles 未生成，无法保留 LiveKit PrivacyInfo.xcprivacy。" >&2
  exit 1
fi
if [[ ! -f "$bundle_resources_root/LiveKit_LiveKit.bundle/PrivacyInfo.xcprivacy" ]]; then
  echo "LiveKit PrivacyInfo.xcprivacy 未打包。" >&2
  exit 1
fi
if [[ ! -f "$bundle_resources_root/SwiftProtobuf_SwiftProtobuf.bundle/PrivacyInfo.xcprivacy" ]]; then
  echo "SwiftProtobuf PrivacyInfo.xcprivacy 未打包。" >&2
  exit 1
fi

dependency_checkouts="$package_root/.build/checkouts"
cp "$dependency_checkouts/client-sdk-swift/LICENSE" \
  "$bundle_licenses_root/LiveKit-Swift-LICENSE"
cp "$dependency_checkouts/client-sdk-swift/NOTICE" \
  "$bundle_licenses_root/LiveKit-Swift-NOTICE"
cp "$dependency_checkouts/livekit-uniffi-xcframework/LICENSE" \
  "$bundle_licenses_root/LiveKit-UniFFI-LICENSE"
cp "$dependency_checkouts/webrtc-xcframework/LICENSE" \
  "$bundle_licenses_root/LiveKit-WebRTC-XCFramework-MIT-LICENSE"
cp \
  "$package_root/.build/artifacts/webrtc-xcframework/LiveKitWebRTC/LiveKitWebRTC.xcframework/LICENSE" \
  "$bundle_licenses_root/LiveKit-WebRTC-BSD-3-Clause-LICENSE"
cp "$dependency_checkouts/swift-protobuf/LICENSE.txt" \
  "$bundle_licenses_root/SwiftProtobuf-LICENSE"

plutil -lint "$bundle_contents/Info.plist" >/dev/null
if [[ "$(lipo -archs "$helper_executable")" != "arm64" ]]; then
  echo "音频 Helper 不是纯 arm64 产物。" >&2
  exit 1
fi
helper_dependencies="$(otool -L "$helper_executable")"
if [[ "$helper_dependencies" != *'@rpath/LiveKitWebRTC.framework/LiveKitWebRTC'* ]]; then
  echo "音频 Helper 缺少 LiveKitWebRTC 动态依赖。" >&2
  exit 1
fi
if [[ "$helper_dependencies" != *'@rpath/RustLiveKitUniFFI.framework/RustLiveKitUniFFI'* ]]; then
  echo "音频 Helper 缺少 RustLiveKitUniFFI 动态依赖。" >&2
  exit 1
fi

# Several upstream license and privacy files are distributed read-only. Make
# the assembled bundle owner-writable before removing quarantine attributes;
# otherwise a normal non-root desktop build fails before code signing.
chmod -R u+w "$bundle_root"
xattr -cr "$bundle_root"
for framework_name in "${framework_names[@]}"; do
  codesign --force --sign - "$bundle_frameworks_root/$framework_name.framework"
done
codesign --force --sign - "$helper_executable"
codesign --force --sign - "$bundle_root"
codesign --verify --deep --strict "$bundle_root"

echo "LiveKit 音频 Helper 已生成：$bundle_root"
