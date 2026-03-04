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

mkdir -p release
LOG_PATH="release/pyinstaller-macos.log"

rm -rf build dist
PYI_ARGS=(
  --noconfirm
  --clean
  --windowed
  --name InterviewAssistant
  app/main.py
)

echo "[build] Running: pyinstaller ${PYI_ARGS[*]}"
pyinstaller "${PYI_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

if [[ ! -d dist/InterviewAssistant.app ]]; then
  echo "[error] dist/InterviewAssistant.app not found. See $LOG_PATH"
  exit 1
fi

echo "[OK] macOS app bundle built: dist/InterviewAssistant.app"
