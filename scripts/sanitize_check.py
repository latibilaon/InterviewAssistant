from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "openrouter_key": re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    "github_pat": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "mac_abs_path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
}
IGNORE = {".git", ".venv", "dist", "build"}


def main() -> int:
    issues = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or any(part in IGNORE for part in p.parts):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".faiss", ".bin"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, pat in PATTERNS.items():
            if pat.search(text):
                issues.append((name, str(p)))
    if issues:
        print("[FAIL] Sensitive patterns found:")
        for kind, file in issues:
            print(f"  - {kind}: {file}")
        return 1
    print("[OK] No obvious sensitive patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
