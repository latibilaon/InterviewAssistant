#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# 兼容不同 artifact 解压目录结构，自动搜索 .app 产物
APP_BUNDLE="$(find "$ROOT_DIR" -maxdepth 6 -type d -name 'InterviewAssistant.app' 2>/dev/null | head -n 1 || true)"
if [[ -z "${APP_BUNDLE}" || ! -d "${APP_BUNDLE}" ]]; then
  echo "[error] InterviewAssistant.app not found after artifact download."
  echo "[debug] current tree (top 3 levels):"
  find "$ROOT_DIR" -maxdepth 3 -print
  exit 1
fi
echo "[build] using app bundle: ${APP_BUNDLE}"

mkdir -p release/dmg_root
rm -rf release/dmg_root/InterviewAssistant.app release/dmg_root/Applications
cp -R "${APP_BUNDLE}" release/dmg_root/InterviewAssistant.app
ln -s /Applications release/dmg_root/Applications

DMG_PATH="release/InterviewAssistant-macos.dmg"
rm -f "$DMG_PATH"
hdiutil create -volname "InterviewAssistant" -srcfolder release/dmg_root -ov -format UDZO "$DMG_PATH"

echo "[OK] DMG created: $DMG_PATH"
