#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

bash "$ROOT_DIR/scripts/build_macos_app.sh"
bash "$ROOT_DIR/scripts/package_macos_dmg.sh"
