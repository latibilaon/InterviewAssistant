#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pyinstaller
python -m pip install -r requirements.txt

rm -rf build dist release
pyinstaller --noconfirm --clean --windowed --name InterviewAssistant app/main.py

mkdir -p release/dmg_root
cp -R dist/InterviewAssistant.app release/dmg_root/
ln -s /Applications release/dmg_root/Applications

DMG_PATH="release/InterviewAssistant-macos.dmg"
rm -f "$DMG_PATH"
hdiutil create -volname "InterviewAssistant" -srcfolder release/dmg_root -ov -format UDZO "$DMG_PATH"

echo "[OK] DMG created: $DMG_PATH"
