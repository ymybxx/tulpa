#!/usr/bin/env bash
set -euo pipefail

pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$pet_root/../.." && pwd)"
export UV_CACHE_DIR="$project_root/.cache/uv"
application="$pet_root/release/mac-arm64/tulpa.app"
default_identity="Fish Memory Local Development"
signing_identity="${FISH_MEMORY_CODESIGN_IDENTITY:-}"
signing_kind="ad-hoc"

if [[ ! -d "$application" ]]; then
  application="$pet_root/release/mac/tulpa.app"
fi
if [[ ! -d "$application" ]]; then
  echo "没有找到 tulpa.app，无法执行签名。" >&2
  exit 1
fi

build_python="$(uv python find 3.11)"
"$build_python" "$pet_root/scripts/verify-macos-compatibility.py" \
  "$application" --target 12.3

xattr -cr "$application"
if [[ -z "$signing_identity" ]]; then
  identity_listing="$(security find-identity -v -p codesigning)"
  identity_hashes=()
  while IFS= read -r identity_line; do
    if [[ "$identity_line" == *"\"$default_identity\""* ]]; then
      identity_hash="$(awk '{print $2}' <<<"$identity_line")"
      if [[ "$identity_hash" =~ ^[[:xdigit:]]{40}$ ]]; then
        identity_hashes+=("$identity_hash")
      fi
    fi
  done <<<"$identity_listing"

  if [[ "${#identity_hashes[@]}" -eq 1 ]]; then
    signing_identity="${identity_hashes[0]}"
  elif [[ "${#identity_hashes[@]}" -gt 1 ]]; then
    echo "找到多个名为“${default_identity}”的有效代码签名身份，无法安全选择。" >&2
    echo "请删除重复身份，或用 FISH_MEMORY_CODESIGN_IDENTITY 指定证书 SHA-1。" >&2
    exit 1
  fi
fi

# Release builds (FISH_MEMORY_RELEASE_SIGN=1, Developer ID identity) need the
# hardened runtime, a secure timestamp and explicit entitlements for notarization.
release_sign="${FISH_MEMORY_RELEASE_SIGN:-}"
entitlements="$pet_root/packaging/entitlements.mac.plist"

sign_code_preserving_metadata() {
  local identity="$1"
  local target="$2"
  local metadata="entitlements,flags,runtime"
  local output

  if [[ "$release_sign" == "1" ]]; then
    if output="$(
      codesign --force --sign "$identity" --options runtime --timestamp \
        --entitlements "$entitlements" "$target" 2>&1
    )"; then
      return 0
    fi
    echo "代码签名失败：${target}" >&2
    printf '%s\n' "$output" >&2
    return 1
  fi

  # Preserve security-relevant metadata from the existing signature, but do not
  # preserve its requirements: an ad-hoc requirement is tied to one cdhash.
  # Structured bundles must derive their identifier from Info.plist. Preserving
  # the stock Electron executable identifier would make tulpa sign as
  # `com.github.Electron`, so TCC could no longer match rebuilds reliably.
  if codesign -d "$target" >/dev/null 2>&1; then
    if output="$(
      codesign --force --sign "$identity" \
        --preserve-metadata="$metadata" \
        "$target" 2>&1
    )"; then
      return 0
    fi
  elif output="$(codesign --force --sign "$identity" "$target" 2>&1)"; then
    return 0
  fi
  echo "代码签名失败：${target}" >&2
  if [[ -n "$output" ]]; then
    printf '%s\n' "$output" >&2
  fi
  return 1
}

sign_application_inside_out() {
  local identity="$1"
  local candidate
  local description
  local signed_bundles=0
  local signed_macho_files=0

  # Sign every real Mach-O leaf first. This includes the executables and dynamic
  # libraries in both native Helpers, the Electron frameworks, and the bundled
  # Python runtime living outside the standard nested-code directories.
  while IFS= read -r -d '' candidate; do
    description="$(file -b "$candidate")"
    if [[ "$description" != *"Mach-O"* ]]; then
      continue
    fi
    sign_code_preserving_metadata "$identity" "$candidate"
    signed_macho_files=$((signed_macho_files + 1))
  done < <(
    find "$application/Contents" -type f \
      \( -perm -111 -o -name '*.dylib' -o -name '*.so' \) -print0
  )

  # `find -depth` gives us children before parents. Re-sign structured code
  # containers only after all code and containers inside them are complete.
  while IFS= read -r -d '' candidate; do
    sign_code_preserving_metadata "$identity" "$candidate"
    signed_bundles=$((signed_bundles + 1))
  done < <(
    find "$application/Contents" -depth -type d \
      \( -name '*.framework' -o -name '*.xpc' -o -name '*.appex' -o -name '*.app' \) \
      -print0
  )

  sign_code_preserving_metadata "$identity" "$application"
  echo "已由内向外签名 ${signed_macho_files} 个 Mach-O 文件和 ${signed_bundles} 个嵌套代码包。"
}

if [[ "$release_sign" == "1" && ( -z "$signing_identity" || "$signing_identity" == "-" ) ]]; then
  echo "正式签名需要用 FISH_MEMORY_CODESIGN_IDENTITY 指定 Developer ID Application 证书。" >&2
  exit 1
fi
if [[ -n "$signing_identity" && "$signing_identity" != "-" ]]; then
  sign_application_inside_out "$signing_identity"
  signing_kind="稳定本地证书 $signing_identity"
else
  codesign --force --deep --sign - "$application"
  if [[ "$signing_identity" == "-" ]]; then
    echo "警告：FISH_MEMORY_CODESIGN_IDENTITY=- 明确要求使用 ad-hoc 签名；每次重建的代码身份都会变化，macOS 可能再次要求屏幕录制授权。" >&2
  else
    echo "警告：没有找到代码签名身份，将使用 ad-hoc 签名；每次重建的代码身份都会变化，macOS 可能再次要求屏幕录制授权。" >&2
    echo "请在钥匙串中创建名为“${default_identity}”的 Code Signing 自签名证书，或设置 FISH_MEMORY_CODESIGN_IDENTITY。" >&2
  fi
fi
codesign --verify --deep --strict "$application"

if [[ "$signing_kind" != "ad-hoc" ]]; then
  expected_authority=""
  verify_stable_signature() {
    local label="$1"
    local target="$2"
    local details
    local authority
    local requirements
    local bundle_identifier
    local signed_identifier

    if [[ ! -e "$target" ]]; then
      echo "稳定签名核验失败：缺少${label}：${target}" >&2
      exit 1
    fi
    if ! codesign --verify --deep --strict "$target"; then
      echo "稳定签名核验失败：${label} 未通过严格验证。" >&2
      exit 1
    fi
    details="$(codesign -dvv "$target" 2>&1)"
    if grep -Fq "Signature=adhoc" <<<"$details"; then
      echo "稳定签名核验失败：${label} 仍是 ad-hoc 签名。" >&2
      exit 1
    fi
    authority="$(awk -F= '/^Authority=/{print substr($0, index($0, "=") + 1); exit}' <<<"$details")"
    if [[ -z "$authority" ]]; then
      echo "稳定签名核验失败：无法读取${label}的 Authority。" >&2
      exit 1
    fi
    if [[ -z "$expected_authority" ]]; then
      expected_authority="$authority"
    elif [[ "$authority" != "$expected_authority" ]]; then
      echo "稳定签名核验失败：${label} 的 Authority 为“${authority}”，预期为“${expected_authority}”。" >&2
      exit 1
    fi
    requirements="$(codesign -d --requirements - "$target" 2>&1)"
    if ! grep -Fq "designated =>" <<<"$requirements"; then
      echo "稳定签名核验失败：无法读取${label}的 designated requirement。" >&2
      exit 1
    fi
    if grep -Eq '(^|[[:space:]])cdhash([[:space:]]|$)' <<<"$requirements"; then
      echo "稳定签名核验失败：${label} 的 designated requirement 仍绑定 cdhash。" >&2
      exit 1
    fi
    if [[ -f "$target/Contents/Info.plist" ]]; then
      bundle_identifier="$(plutil -extract CFBundleIdentifier raw -o - "$target/Contents/Info.plist" 2>/dev/null || true)"
      signed_identifier="$(awk -F= '/^Identifier=/{print substr($0, index($0, "=") + 1); exit}' <<<"$details")"
      if [[ -n "$bundle_identifier" && "$signed_identifier" != "$bundle_identifier" ]]; then
        echo "稳定签名核验失败：${label} 的签名 Identifier 为“${signed_identifier}”，Bundle Identifier 为“${bundle_identifier}”。" >&2
        exit 1
      fi
    fi
  }

  verify_stable_signature "tulpa.app" "$application"
  verified_nested_bundles=0
  while IFS= read -r -d '' nested_bundle; do
    verify_stable_signature "嵌套代码包" "$nested_bundle"
    verified_nested_bundles=$((verified_nested_bundles + 1))
  done < <(
    find "$application/Contents" -depth -type d \
      \( -name '*.framework' -o -name '*.xpc' -o -name '*.appex' -o -name '*.app' \) \
      -print0
  )
  echo "已核验主程序与 ${verified_nested_bundles} 个嵌套代码包的稳定签名，Authority：${expected_authority}"
fi

echo "已完成${signing_kind}签名：${application}"
