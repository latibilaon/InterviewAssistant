#!/usr/bin/env python3
"""
一键构建分层资料库 RAG 索引。

用法示例：
  python build_rag_layered.py
  python build_rag_layered.py --force
  python build_rag_layered.py --include-raw --max-files 24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build layered RAG index")
    parser.add_argument("--force", action="store_true", help="force rebuild index")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="include 原始材料 in selection",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=12,
        help="max files selected by priority policy",
    )
    parser.add_argument("--docs-dir", type=str, default="", help="override docs dir")
    parser.add_argument("--index-dir", type=str, default="", help="override index dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 先设置环境，再导入 config/rag_search，确保覆盖生效
    os.environ["RAG_ENABLED"] = "true"
    os.environ["RAG_NON_BLOCKING_STARTUP"] = "false"  # 脚本场景下需要前台完成构建
    os.environ["RAG_PRIORITY_PICK_ENABLED"] = "true"
    os.environ["RAG_PRIORITY_INCLUDE_RAW"] = "true" if args.include_raw else "false"
    os.environ["RAG_PRIORITY_MAX_FILES"] = str(max(1, args.max_files))
    if args.docs_dir:
        os.environ["DOCS_DIR"] = args.docs_dir
    if args.index_dir:
        os.environ["INDEX_DIR"] = args.index_dir

    from config import DOCS_DIR, INDEX_DIR
    from rag_search import init_rag

    docs_dir = Path(DOCS_DIR)
    index_dir = Path(INDEX_DIR)
    print(f"[RAG] docs_dir={docs_dir}")
    print(f"[RAG] index_dir={index_dir}")
    print(
        f"[RAG] force={args.force} include_raw={args.include_raw} max_files={max(1, args.max_files)}"
    )

    ok = init_rag(
        force_rebuild=args.force,
        progress_cb=lambda msg: print(f"[RAG] {msg}"),
    )
    if ok:
        print("[RAG] done")
        return 0

    print("[RAG] failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
