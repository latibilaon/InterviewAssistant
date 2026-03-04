#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d dist/InterviewAssistant.app ]]; then
  echo "[error] dist/InterviewAssistant.app not found. Run scripts/build_macos_app.sh first."
  exit 1
fi

mkdir -p release/dmg_root
rm -rf release/dmg_root/InterviewAssistant.app release/dmg_root/Applications
cp -R dist/InterviewAssistant.app release/dmg_root/
ln -s /Applications release/dmg_root/Applications

DMG_PATH="release/InterviewAssistant-macos.dmg"
rm -f "$DMG_PATH"
hdiutil create -volname "InterviewAssistant" -srcfolder release/dmg_root -ov -format UDZO "$DMG_PATH"

echo "[OK] DMG created: $DMG_PATH"
