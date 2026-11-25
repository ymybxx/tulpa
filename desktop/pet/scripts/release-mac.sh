#!/usr/bin/env bash
set -euo pipefail

# Build a Developer ID signed, notarized and stapled tulpa DMG for public release.
# Needs a "Developer ID Application" certificate in the keychain and notary
# credentials stored with: xcrun notarytool store-credentials tulpa-notary
pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notary_profile="${FISH_MEMORY_NOTARY_PROFILE:-tulpa-notary}"
application="$pet_root/release/mac-arm64/tulpa.app"
version="$(node -p "require('$pet_root/package.json').version")"
image="$pet_root/release/tulpa-${version}-arm64.dmg"

identity="${FISH_MEMORY_CODESIGN_IDENTITY:-}"
if [[ -z "$identity" ]]; then
  identities="$(security find-identity -v -p codesigning | grep '"Developer ID Application:' || true)"
  if [[ "$(grep -c . <<<"$identities")" -ne 1 ]]; then
    echo "需要钥匙串里恰好有一个 Developer ID Application 证书，或用 FISH_MEMORY_CODESIGN_IDENTITY 指定。" >&2
    exit 1
  fi
  identity="$(awk '{print $2}' <<<"$identities")"
fi
if ! notary_check="$(xcrun notarytool history --keychain-profile "$notary_profile" 2>&1)"; then
  printf '%s\n' "$notary_check" >&2
  if grep -q "agreement" <<<"$notary_check"; then
    echo "Apple 开发者协议需要重新同意：登录 developer.apple.com/account 接受新协议后再试。" >&2
  else
    echo "公证凭据“${notary_profile}”不可用。如果还没存，先运行：xcrun notarytool store-credentials ${notary_profile}" >&2
  fi
  exit 1
fi

cd "$pet_root"
FISH_MEMORY_CODESIGN_IDENTITY="$identity" FISH_MEMORY_RELEASE_SIGN=1 npm run dist:mac:local

notarize() {
  local target="$1"
  local log
  log="$(xcrun notarytool submit "$target" --keychain-profile "$notary_profile" --wait 2>&1)"
  printf '%s\n' "$log"
  if ! grep -q "status: Accepted" <<<"$log"; then
    local submission
    submission="$(awk '/^  id:/{print $2; exit}' <<<"$log")"
    if [[ -n "$submission" ]]; then
      xcrun notarytool log "$submission" --keychain-profile "$notary_profile" >&2 || true
    fi
    echo "公证未通过：$target" >&2
    exit 1
  fi
}

archive="$(mktemp -d)/tulpa.zip"
ditto -c -k --keepParent "$application" "$archive"
notarize "$archive"
xcrun stapler staple "$application"

bash scripts/make-dmg-mac.sh
codesign --force --sign "$identity" --timestamp "$image"
notarize "$image"
xcrun stapler staple "$image"

spctl --assess --type execute --verbose "$application"
spctl --assess --type open --context context:primary-signature --verbose "$image"
echo "正式发布包已就绪：$image"
