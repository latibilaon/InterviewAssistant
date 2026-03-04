"""
rag_search.py — 本地 RAG 检索模块（支持缓存复用、分批建库、可取消）
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore[no-redef]

from config import (
    DOCS_DIR,
    EMBED_MODEL_NAME,
    INDEX_DIR,
    RAG_BUILD_BATCH_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_ENABLED,
    RAG_FORCE_REBUILD,
    RAG_NON_BLOCKING_STARTUP,
    RAG_PRIORITY_INCLUDE_RAW,
    RAG_PRIORITY_MAX_FILES,
    RAG_PRIORITY_PICK_ENABLED,
    RAG_TOP_K,
)

logger = logging.getLogger("RAG")

_MANIFEST_FILE = "manifest.json"
_READ_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")

# 避免 tokenizer 线程与 BLAS 线程争抢，降低卡顿概率
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "1"))


ProgressCallback = Callable[[str], None]


_embeddings: Optional[HuggingFaceEmbeddings] = None


def _index_files_exist(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def _collect_md_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.md"))
    if not files:
        return files

    if not RAG_PRIORITY_PICK_ENABLED:
        return files

    selected: list[tuple[int, Path]] = []
    skipped = 0

    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        score = 100

        if not RAG_PRIORITY_INCLUDE_RAW and "/原始材料/" in f"/{rel}":
            skipped += 1
            continue

        # 分层库优先级：A层核心 > B层题库 > C层索引 > 其他
        if rel.startswith("01_A层-核心画像/") and "/原始材料/" not in f"/{rel}":
            score = 0
        elif rel.startswith("02_B层-高频问答/") and "/原始材料/" not in f"/{rel}":
            score = 10
        elif rel == "03_C层-行业知识/C1_行业与公司研究索引.md":
            score = 20
        elif rel.startswith("90_上下文快照/") and not rel.endswith("/README.md"):
            score = 25
        elif rel.startswith("03_C层-行业知识/") and "/原始材料/" not in f"/{rel}":
            score = 30
        elif rel == "00_使用说明.md":
            score = 40
        elif rel.startswith("99_迁移清单/"):
            score = 80
        elif "/原始材料/" in f"/{rel}":
            score = 120

        selected.append((score, p))

    selected.sort(key=lambda x: (x[0], str(x[1])))
    max_files = max(1, RAG_PRIORITY_MAX_FILES)
    picked = [p for _, p in selected[:max_files]]

    logger.info(
        "RAG 选材完成：候选 %d，跳过(原始材料) %d，最终纳入 %d（max=%d）",
        len(files),
        skipped,
        len(picked),
        max_files,
    )
    for i, p in enumerate(picked[:10], start=1):
        logger.info("RAG 选材[%02d] %s", i, p.relative_to(root))

    return picked


def _read_text_file(path: Path) -> str:
    for enc in _READ_ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # 最后兜底：替换非法字符，确保不会因为单个文件崩溃
    return path.read_text(encoding="utf-8", errors="replace")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(docs_dir: Path) -> dict:
    files = _collect_md_files(docs_dir)
    file_items = []
    h = hashlib.sha256()

    for p in files:
        rel = str(p.relative_to(docs_dir))
        digest = _hash_file(p)
        h.update(rel.encode("utf-8", errors="ignore"))
        h.update(digest.encode("ascii"))
        file_items.append({"path": rel, "sha256": digest, "size": p.stat().st_size})

    return {
        "version": 1,
        "docs_dir": str(docs_dir),
        "docs_hash": h.hexdigest(),
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "file_count": len(files),
        "files": file_items,
    }


def _load_manifest(index_dir: Path) -> Optional[dict]:
    p = index_dir / _MANIFEST_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("manifest 读取失败，视为需要重建: %s", e)
        return None


def _write_manifest_atomic(index_dir: Path, manifest: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_dir / f"{_MANIFEST_FILE}.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, index_dir / _MANIFEST_FILE)


def _need_rebuild(index_dir: Path, docs_dir: Path) -> tuple[bool, dict]:
    current = _build_manifest(docs_dir)
    cached = _load_manifest(index_dir)
    if cached is None:
        return True, current

    keys = ("docs_hash", "embed_model", "chunk_size", "chunk_overlap")
    changed = any(cached.get(k) != current.get(k) for k in keys)
    return changed, current


def load_documents(
    docs_dir: Path = DOCS_DIR,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[Document]:
    md_files = _collect_md_files(docs_dir)
    if not md_files:
        raise FileNotFoundError(f"未在 {docs_dir} 中找到任何 .md 文件")

    docs: list[Document] = []
    failed: list[str] = []

    logger.info("发现 %d 个 Markdown 文件，开始加载", len(md_files))
    if progress_cb:
        preview = ", ".join(str(p.relative_to(docs_dir)) for p in md_files[:5])
        progress_cb(f"RAG 选材: 共 {len(md_files)} 个文件，前5个: {preview}")
    for path in md_files:
        try:
            text = _read_text_file(path)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path.relative_to(docs_dir)),
                        "file_name": path.name,
                    },
                )
            )
        except Exception as e:
            failed.append(path.name)
            logger.warning("跳过文件 %s（加载失败: %s）", path.name, e)

    logger.info("文档加载完成：成功 %d，失败 %d", len(docs), len(failed))
    if progress_cb:
        progress_cb(f"RAG 文档加载完成: 成功 {len(docs)}，失败 {len(failed)}")
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("文档分块完成：%d 文件 -> %d chunks", len(documents), len(chunks))
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("加载 Embedding 模型：%s", EMBED_MODEL_NAME)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding 模型加载完成")
    return _embeddings


def _save_local_atomic(vectorstore: FAISS, index_dir: Path, manifest: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = index_dir.parent / f"{index_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    vectorstore.save_local(str(tmp_dir))
    (tmp_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name in ("index.faiss", "index.pkl", _MANIFEST_FILE):
        src = tmp_dir / name
        if src.exists():
            os.replace(src, index_dir / name)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def _backup_corrupted_index(index_dir: Path, reason: str) -> None:
    if not index_dir.exists():
        return
    ts = int(time.time())
    dst = index_dir.parent / f"{index_dir.name}_corrupt_{ts}"
    try:
        shutil.move(str(index_dir), str(dst))
        logger.warning("索引损坏已备份到 %s（原因: %s）", dst, reason)
    except Exception as e:
        logger.warning("索引备份失败（%s），将直接覆盖重建", e)


def build_vectorstore_incremental(
    chunks: list[Document],
    index_dir: Path,
    manifest: dict,
    batch_size: int = RAG_BUILD_BATCH_SIZE,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> FAISS:
    if not chunks:
        raise ValueError("没有可用于建库的 chunks")

    bs = max(1, batch_size)
    total = len(chunks)
    built = 0
    t0 = time.perf_counter()

    embeddings = get_embeddings()
    vectorstore: Optional[FAISS] = None
    if progress_cb:
        progress_cb(f"RAG 建库进度: 0/{total} (0.0%)")

    for start in range(0, total, bs):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("RAG 建库被取消")

        end = min(start + bs, total)
        batch = chunks[start:end]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        built = end
        pct = built / total * 100
        elapsed = time.perf_counter() - t0
        eta = (elapsed / built * (total - built)) if built > 0 else 0.0
        msg = (
            f"RAG 建库进度: {built}/{total} ({pct:.1f}%), "
            f"耗时 {elapsed:.1f}s, 预计剩余 {max(0.0, eta):.1f}s"
        )
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    assert vectorstore is not None
    _save_local_atomic(vectorstore, index_dir, manifest)
    return vectorstore


def load_vectorstore(index_dir: Path = INDEX_DIR) -> Optional[FAISS]:
    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    if not (index_file.exists() and pkl_file.exists()):
        return None

    try:
        logger.info("从磁盘加载 FAISS 索引：%s", index_dir)
        vs = FAISS.load_local(
            str(index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS 索引加载完成")
        return vs
    except Exception as e:
        _backup_corrupted_index(index_dir, str(e))
        return None


class RAGEngine:
    def __init__(self, docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR):
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self._vectorstore: Optional[FAISS] = None
        self._enabled: bool = RAG_ENABLED
        self._cancel_event = threading.Event()
        self._build_thread: Optional[threading.Thread] = None
        self._build_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self, reason: str) -> None:
        self._enabled = False
        self._vectorstore = None
        logger.warning("RAG 已禁用：%s", reason)

    def cancel_build(self) -> None:
        self._cancel_event.set()

    def _try_load_cache_with_manifest_backfill(self, manifest: dict) -> bool:
        if not _index_files_exist(self.index_dir):
            return False

        loaded = load_vectorstore(self.index_dir)
        if loaded is None:
            return False

        self._vectorstore = loaded
        # 兼容老索引：缺少 manifest 时补写当前 manifest，避免后续每次都触发重建
        if _load_manifest(self.index_dir) is None:
            try:
                _write_manifest_atomic(self.index_dir, manifest)
                logger.info("检测到旧索引缺少 manifest，已自动补写")
            except Exception as e:
                logger.warning("manifest 自动补写失败: %s", e)
        logger.info("RAG 索引命中缓存，跳过重建")
        return True

    def _is_building(self) -> bool:
        t = self._build_thread
        return t is not None and t.is_alive()

    def init(
        self,
        force_rebuild: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> bool:
        if not self._enabled:
            logger.info("RAG_ENABLED=false，跳过 RAG 初始化")
            return False

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_event.clear()

        try:
            if progress_cb:
                progress_cb("RAG 阶段: 检查缓存与变更")
            force = force_rebuild or RAG_FORCE_REBUILD
            need_rebuild, manifest = _need_rebuild(self.index_dir, self.docs_dir)
            if force:
                need_rebuild = True
                logger.info("RAG_FORCE_REBUILD=true，强制重建索引")

            if not need_rebuild and self._try_load_cache_with_manifest_backfill(manifest):
                if progress_cb:
                    progress_cb("RAG 阶段: 命中本地缓存，初始化完成")
                return True

            # manifest 丢失但存在旧索引时，优先直接加载以保证秒开，再在后续按需重建
            if not force and _load_manifest(self.index_dir) is None:
                if self._try_load_cache_with_manifest_backfill(manifest):
                    if progress_cb:
                        progress_cb("RAG 阶段: 旧索引恢复成功，初始化完成")
                    return True

            if progress_cb:
                progress_cb("RAG 阶段: 读取文档")
            docs = load_documents(self.docs_dir, progress_cb=progress_cb)
            if progress_cb:
                progress_cb(f"RAG 阶段: 文档读取完成，共 {len(docs)} 个")
                progress_cb("RAG 阶段: 文档分块")
            chunks = split_documents(docs)
            if progress_cb:
                progress_cb(f"RAG 开始建库：{len(chunks)} chunks")
            self._vectorstore = build_vectorstore_incremental(
                chunks,
                index_dir=self.index_dir,
                manifest=manifest,
                cancel_event=self._cancel_event,
                progress_cb=progress_cb,
            )
            if progress_cb:
                progress_cb("RAG 阶段: 保存索引到本地缓存")
            logger.info("RAG 索引构建完成并已缓存")
            if progress_cb:
                progress_cb("RAG 阶段: 初始化完成")
            return True

        except Exception as e:
            self.disable(f"初始化失败: {type(e).__name__}: {e}")
            return False

    def init_fast_then_background(
        self,
        force_rebuild: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> bool:
        """
        秒开策略：
        1) 先尝试缓存加载（命中则立即可检索）
        2) 若需重建，则后台线程建库，不阻塞主启动流程
        """
        if not self._enabled:
            logger.info("RAG_ENABLED=false，跳过 RAG 初始化")
            return False

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_event.clear()

        try:
            if progress_cb:
                progress_cb("RAG 阶段: 快速模式检查缓存")
            force = force_rebuild or RAG_FORCE_REBUILD
            need_rebuild, manifest = _need_rebuild(self.index_dir, self.docs_dir)
            if force:
                need_rebuild = True
                logger.info("RAG_FORCE_REBUILD=true，强制重建索引")

            if not need_rebuild and self._try_load_cache_with_manifest_backfill(manifest):
                if progress_cb:
                    progress_cb("RAG 阶段: 命中本地缓存，初始化完成")
                return True

            if not force and _load_manifest(self.index_dir) is None:
                if self._try_load_cache_with_manifest_backfill(manifest):
                    if progress_cb:
                        progress_cb("RAG 阶段: 旧索引恢复成功，初始化完成")
                    return True

            with self._build_lock:
                if self._is_building():
                    logger.info("RAG 后台建库已在进行中，跳过重复触发")
                    return self._vectorstore is not None

                def _bg_task() -> None:
                    try:
                        ok = self.init(force_rebuild=force, progress_cb=progress_cb)
                        if ok:
                            logger.info("RAG 后台建库完成，检索已可用")
                        else:
                            logger.warning("RAG 后台建库未完成，将继续纯 LLM 模式")
                    except Exception as e:
                        logger.warning("RAG 后台建库异常: %s", e)

                self._build_thread = threading.Thread(
                    target=_bg_task,
                    name="rag-build-bg",
                    daemon=True,
                )
                self._build_thread.start()
                logger.info("RAG 启动时不阻塞：已进入后台建库")
                if progress_cb:
                    progress_cb("RAG 阶段: 已切到后台建库（前台可继续使用）")
            return self._vectorstore is not None
        except Exception as e:
            logger.warning("RAG 快速初始化失败，回退纯 LLM: %s", e)
            return False

    def _ensure_initialized(self) -> bool:
        if not self._enabled or self._vectorstore is None:
            return False
        return True

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        if not self._ensure_initialized():
            return []

        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        output = []
        for doc, score in results:
            output.append(
                {
                    "content": doc.page_content.strip(),
                    "source": doc.metadata.get("source", "未知来源"),
                    "score": round(score, 4),
                }
            )
        return output

    async def async_search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        if not self._ensure_initialized():
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(query, top_k))


rag_engine = RAGEngine()


def init_rag(force_rebuild: bool = False, progress_cb: Optional[ProgressCallback] = None) -> bool:
    return rag_engine.init(force_rebuild=force_rebuild, progress_cb=progress_cb)


def init_rag_fast(force_rebuild: bool = False, progress_cb: Optional[ProgressCallback] = None) -> bool:
    if RAG_NON_BLOCKING_STARTUP:
        return rag_engine.init_fast_then_background(force_rebuild=force_rebuild, progress_cb=progress_cb)
    return rag_engine.init(force_rebuild=force_rebuild, progress_cb=progress_cb)


def cancel_rag_build() -> None:
    rag_engine.cancel_build()


def rag_search(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    return rag_engine.search(query, top_k)


async def async_rag_search(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    return await rag_engine.async_search(query, top_k)
