"""
config.py — 项目全局配置中枢
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载与 config.py 同目录的 .env 文件
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)


def _get_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_int_or_none(key: str) -> int | None:
    val = os.getenv(key, "").strip()
    return int(val) if val.isdigit() else None


def _get_path(key: str, default: str) -> Path:
    raw = os.getenv(key, default).strip()
    return Path(raw).expanduser().resolve()


def _get_secret(key: str) -> str:
    raw = os.getenv(key, "").strip()
    if raw in ("", "sk-...", "sk-or-v1-...", "your_api_key_here"):
        return ""
    if raw.endswith("..."):
        return ""
    return raw


def _get_csv_list(key: str) -> list[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


# ─── API 密钥 ──────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _get_secret("OPENAI_API_KEY")
OPENROUTER_API_KEY: str = _get_secret("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip() or "https://openrouter.ai/api/v1"
OPENROUTER_CUSTOM_MODELS: list[str] = _get_csv_list("OPENROUTER_CUSTOM_MODELS")

# ─── STT 配置 ──────────────────────────────────────────────────────────────
# "api"   → OpenAI Whisper API（需联网）
# "local" → faster-whisper 本地模型（无需网络）
STT_MODE: str = os.getenv("STT_MODE", "local").strip().lower()
STT_API_TIMEOUT_SECONDS: float = _get_float("STT_API_TIMEOUT_SECONDS", 15.0)
STT_API_MAX_RETRIES: int = _get_int("STT_API_MAX_RETRIES", 1)
STT_STITCH_ENABLED: bool = _get_bool("STT_STITCH_ENABLED", True)
STT_STITCH_FLUSH_SECONDS: float = _get_float("STT_STITCH_FLUSH_SECONDS", 1.2)
STT_STITCH_MAX_CHARS: int = _get_int("STT_STITCH_MAX_CHARS", 220)
STT_STITCH_MAX_HOLD_SECONDS: float = _get_float("STT_STITCH_MAX_HOLD_SECONDS", 2.5)
STT_AUDIO_QUEUE_MAX: int = _get_int("STT_AUDIO_QUEUE_MAX", 64)
STT_RAW_QUEUE_MAX: int = _get_int("STT_RAW_QUEUE_MAX", 1200)
STT_SEGMENT_MODE: str = os.getenv("STT_SEGMENT_MODE", "auto").strip().lower()  # auto | vad | fixed
STT_FIXED_SEGMENT_SECONDS: float = _get_float("STT_FIXED_SEGMENT_SECONDS", 6.0)
STT_FIXED_OVERLAP_SECONDS: float = _get_float("STT_FIXED_OVERLAP_SECONDS", 1.5)
STT_FIXED_MIN_RMS: float = _get_float("STT_FIXED_MIN_RMS", 0.003)

# faster-whisper 模型规格：tiny / base / small / medium / large-v3
WHISPER_LOCAL_MODEL: str = os.getenv("WHISPER_LOCAL_MODEL", "base")
WHISPER_LANGUAGE: str | None = os.getenv("WHISPER_LANGUAGE") or None
STT_DEFAULT_LANGUAGE: str = os.getenv("STT_DEFAULT_LANGUAGE", "zh").strip().lower() or "zh"
STT_ALLOWED_LANGUAGES: str = os.getenv("STT_ALLOWED_LANGUAGES", "zh,en").strip().lower()
STT_LANGUAGE_PROFILE: str = os.getenv("STT_LANGUAGE_PROFILE", "auto").strip().lower()  # zh | en | auto
WHISPER_BEAM_SIZE: int = _get_int("WHISPER_BEAM_SIZE", 1)
WHISPER_BEST_OF: int = _get_int("WHISPER_BEST_OF", 1)
WHISPER_CONDITION_ON_PREV: bool = _get_bool("WHISPER_CONDITION_ON_PREV", False)
WHISPER_INTERNAL_VAD: bool = _get_bool("WHISPER_INTERNAL_VAD", False)

# ─── 音频设备配置 ───────────────────────────────────────────────────────────
AUDIO_DEVICE_INDEX: int | None = _get_int_or_none("AUDIO_DEVICE_INDEX")
AUDIO_DEVICE_NAME: str = os.getenv("AUDIO_DEVICE_NAME", "").strip()

# ─── VAD 参数 ───────────────────────────────────────────────────────────────
VAD_SPEECH_THRESHOLD: float = _get_float("VAD_SPEECH_THRESHOLD", 0.5)
VAD_SILENCE_DURATION: float = _get_float("VAD_SILENCE_DURATION", 1.5)
VAD_PRE_ROLL: float = _get_float("VAD_PRE_ROLL", 0.4)
VAD_MAX_SPEECH_DURATION: float = _get_float("VAD_MAX_SPEECH_DURATION", 15.0)
VAD_MIN_SPEECH_DURATION: float = _get_float("VAD_MIN_SPEECH_DURATION", 0.3)
VAD_FORCE_SPLIT_OVERLAP: float = _get_float("VAD_FORCE_SPLIT_OVERLAP", 0.8)

# ─── OpenRouter / LLM 配置 ────────────────────────────────────────────────
# 仅需 OPENROUTER_API_KEY + OPENROUTER_MODEL 即可跑通纯 LLM 模式
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_REASONING_EFFORT: str = os.getenv("OPENROUTER_REASONING_EFFORT", "minimal").strip().lower()
LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.4)
LLM_MAX_TOKENS: int = _get_int("LLM_MAX_TOKENS", 500)
LLM_STREAMING: bool = _get_bool("LLM_STREAMING", True)
LLM_TIMEOUT_SECONDS: float = _get_float("LLM_TIMEOUT_SECONDS", 25.0)
LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 2)
LLM_RETRY_BACKOFF_SECONDS: float = _get_float("LLM_RETRY_BACKOFF_SECONDS", 0.8)
LLM_CONTEXT_CHAR_BUDGET: int = _get_int("LLM_CONTEXT_CHAR_BUDGET", 4200)
LLM_RETRY_SHRINK_RATIO: float = _get_float("LLM_RETRY_SHRINK_RATIO", 0.65)
LLM_MIN_ANSWER_CHARS: int = _get_int("LLM_MIN_ANSWER_CHARS", 220)

# ─── 调度引擎参数 ───────────────────────────────────────────────────────────
DEBOUNCE_SECONDS: float = _get_float("DEBOUNCE_SECONDS", 1.5)
AUTO_ANSWER_ENABLED: bool = _get_bool("AUTO_ANSWER_ENABLED", False)
TRANSCRIPT_MAX_ITEMS: int = _get_int("TRANSCRIPT_MAX_ITEMS", 300)

# ─── 联网搜索参数 ───────────────────────────────────────────────────────────
WEB_SEARCH_ENABLED: bool = _get_bool("WEB_SEARCH_ENABLED", False)
WEB_SEARCH_MAX_RESULTS: int = _get_int("WEB_SEARCH_MAX_RESULTS", 3)
WEB_SEARCH_TIMEOUT: float = _get_float("WEB_SEARCH_TIMEOUT", 3.0)
OPENROUTER_WEB_PLUGIN_ENABLED: bool = _get_bool("OPENROUTER_WEB_PLUGIN_ENABLED", True)

# ─── RAG 配置 ───────────────────────────────────────────────────────────────
RAG_ENABLED: bool = _get_bool("RAG_ENABLED", False)
RAG_FORCE_REBUILD: bool = _get_bool("RAG_FORCE_REBUILD", False)
RAG_NON_BLOCKING_STARTUP: bool = _get_bool("RAG_NON_BLOCKING_STARTUP", True)
RAG_PRIORITY_PICK_ENABLED: bool = _get_bool("RAG_PRIORITY_PICK_ENABLED", True)
RAG_PRIORITY_INCLUDE_RAW: bool = _get_bool("RAG_PRIORITY_INCLUDE_RAW", False)
RAG_PRIORITY_MAX_FILES: int = _get_int("RAG_PRIORITY_MAX_FILES", 12)
RAG_TOP_K: int = _get_int("RAG_TOP_K", 4)
RAG_CHUNK_SIZE: int = _get_int("RAG_CHUNK_SIZE", 512)
RAG_CHUNK_OVERLAP: int = _get_int("RAG_CHUNK_OVERLAP", 64)
RAG_BUILD_BATCH_SIZE: int = _get_int("RAG_BUILD_BATCH_SIZE", 256)
RAG_BUILD_WARN_SECONDS: float = _get_float("RAG_BUILD_WARN_SECONDS", 45.0)
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# 路径可配置，默认指向仓库内目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR: Path = _get_path("DOCS_DIR", str(_PROJECT_ROOT / "面试准备资料_md"))
INDEX_DIR: Path = _get_path("INDEX_DIR", str(Path(__file__).resolve().parent / "faiss_index"))
