#!/usr/bin/env bash
set -euo pipefail

# Wrap the signed app in an LZMA-compressed disk image (macOS 10.15+) with the
# usual drag-to-Applications layout. Run after adhoc-sign-mac.sh.
pet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
application="$pet_root/release/mac-arm64/tulpa.app"
version="$(node -p "require('$pet_root/package.json').version")"
image="$pet_root/release/tulpa-${version}-arm64.dmg"

if [[ ! -d "$application" ]]; then
  echo "没有找到 tulpa.app，请先运行 npm run dist:mac:local。" >&2
  exit 1
fi
codesign --verify --deep --strict "$application"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
ditto "$application" "$staging/tulpa.app"
ln -s /Applications "$staging/Applications"

rm -f "$image"
hdiutil create -quiet -volname "tulpa" -srcfolder "$staging" \
  -fs APFS -format ULMO "$image"
hdiutil verify -quiet "$image"
echo "安装镜像已生成：$image（$(du -h "$image" | cut -f1)，安装后 $(du -sh "$application" | cut -f1)）"
