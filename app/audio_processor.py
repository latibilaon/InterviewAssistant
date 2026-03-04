"""
audio_processor.py — 音频捕获与语音识别模块
=============================================
数据流（全异步，UI 线程永不阻塞）：

  sounddevice 回调 (C 内部线程)
      ↓ float32 PCM 块（非阻塞 put_nowait）
  _raw_queue (threading.Queue, 无锁环形缓冲)
      ↓
  VAD 线程 (Silero / Energy 状态机)
      ↓ 完整语音段 (call_soon_threadsafe → asyncio)
  _speech_queue (asyncio.Queue)
      ↓
  异步 STT 任务 (faster-whisper 线程池 / OpenAI API)
      ↓
  text_queue (asyncio.Queue) ──→ M3 调度引擎

【macOS 扬声器捕获】
  视频面试时面试官的声音来自扬声器，需要虚拟音频设备转发：
  brew install blackhole-2ch
  → 在"音频 MIDI 设置"中创建多输出设备（扬声器 + BlackHole 2ch）
  → 系统输出切换到多输出设备（你仍可听到声音）
  → python audio_processor.py --list-devices  找到 BlackHole 编号
  → 在 .env 中设置 AUDIO_DEVICE_INDEX=<编号>
"""

import asyncio
import importlib.util
import io
import logging
import math
import re
import queue
import threading
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto
from typing import Optional

import numpy as np
try:
    import sounddevice as sd
except Exception:  # pragma: no cover - 依赖缺失时降级
    sd = None  # type: ignore[assignment]

torch = None  # 延迟导入，避免无关路径触发 OpenMP/torchaudio 问题

from config import (
    AUDIO_DEVICE_INDEX,
    AUDIO_DEVICE_NAME,
    OPENAI_API_KEY,
    STT_MODE,
    STT_SEGMENT_MODE,
    STT_FIXED_SEGMENT_SECONDS,
    STT_FIXED_OVERLAP_SECONDS,
    STT_FIXED_MIN_RMS,
    STT_DEFAULT_LANGUAGE,
    STT_ALLOWED_LANGUAGES,
    STT_LANGUAGE_PROFILE,
    STT_AUDIO_QUEUE_MAX,
    STT_API_MAX_RETRIES,
    STT_API_TIMEOUT_SECONDS,
    STT_RAW_QUEUE_MAX,
    STT_STITCH_ENABLED,
    STT_STITCH_FLUSH_SECONDS,
    STT_STITCH_MAX_HOLD_SECONDS,
    STT_STITCH_MAX_CHARS,
    VAD_PRE_ROLL,
    VAD_MAX_SPEECH_DURATION,
    VAD_FORCE_SPLIT_OVERLAP,
    VAD_MIN_SPEECH_DURATION,
    VAD_SILENCE_DURATION,
    VAD_SPEECH_THRESHOLD,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_CONDITION_ON_PREV,
    WHISPER_INTERNAL_VAD,
    WHISPER_LANGUAGE,
    WHISPER_LOCAL_MODEL,
)

# ─────────────────────────────────────────────
#  音频常量（Silero VAD 强制要求 16kHz）
# ─────────────────────────────────────────────

SAMPLE_RATE = 16000          # Hz，Silero VAD 推荐
CHUNK_SAMPLES = 512          # 32ms/块 @ 16kHz（Silero VAD 官方推荐值）
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE   # 0.032 秒

MIN_SPEECH_DURATION = VAD_MIN_SPEECH_DURATION    # 低于此时长的段落视为噪声，丢弃（秒）
MAX_SPEECH_DURATION = VAD_MAX_SPEECH_DURATION  # 超过此时长强制截断（秒）

logger = logging.getLogger("AudioProcessor")
_BASE_ALLOWED_LANGS = {x.strip() for x in STT_ALLOWED_LANGUAGES.split(",") if x.strip()}
if not _BASE_ALLOWED_LANGS:
    _BASE_ALLOWED_LANGS = {"zh", "en"}
_BASE_DEFAULT_LANG = STT_DEFAULT_LANGUAGE if STT_DEFAULT_LANGUAGE in _BASE_ALLOWED_LANGS else "zh"
_RUNTIME_LANGUAGE_PROFILE = STT_LANGUAGE_PROFILE if STT_LANGUAGE_PROFILE in {"zh", "en", "auto"} else "zh"


def set_runtime_language_profile(profile: str) -> None:
    """运行时切换转录语言策略：zh | en | auto。"""
    global _RUNTIME_LANGUAGE_PROFILE
    p = (profile or "").strip().lower()
    if p not in {"zh", "en", "auto"}:
        logger.warning("未知语言策略 %r，忽略", profile)
        return
    _RUNTIME_LANGUAGE_PROFILE = p
    logger.info("STT 语言策略已切换为: %s", p)


def _current_lang_policy() -> tuple[Optional[str], set[str], str]:
    # profile 优先于 WHISPER_LANGUAGE 配置
    if _RUNTIME_LANGUAGE_PROFILE == "zh":
        return "zh", {"zh"}, "zh"
    if _RUNTIME_LANGUAGE_PROFILE == "en":
        return "en", {"en"}, "en"
    # auto: 允许中英，默认中文；若显式设置 WHISPER_LANGUAGE 则强制该语言
    explicit = (WHISPER_LANGUAGE or "").strip().lower()
    if explicit in _BASE_ALLOWED_LANGS:
        return explicit, {explicit}, explicit
    return None, set(_BASE_ALLOWED_LANGS), _BASE_DEFAULT_LANG


# ─────────────────────────────────────────────
#  VAD 状态机基类
# ─────────────────────────────────────────────

class _VADState(Enum):
    WAITING  = auto()   # 静音等待中
    SPEAKING = auto()   # 语音录制中


class _BaseVAD(ABC):
    """
    VAD 状态机基类。
    子类只需实现 _get_prob(chunk) 返回 0~1 的语音概率。
    状态机逻辑、预滚缓冲、超时截断由基类统一处理。
    """

    def __init__(
        self,
        speech_threshold: float = VAD_SPEECH_THRESHOLD,
        silence_duration: float = VAD_SILENCE_DURATION,
        pre_roll: float = VAD_PRE_ROLL,
    ):
        self.speech_threshold = speech_threshold
        # 滞后阈值：进入说话比退出说话的阈值高，防止频繁状态切换
        self.silence_threshold = max(0.0, speech_threshold - 0.15)
        self._silence_frames = max(1, int(silence_duration / CHUNK_DURATION))
        self._pre_roll_frames = max(1, int(pre_roll / CHUNK_DURATION))
        self._split_overlap_frames = max(1, int(VAD_FORCE_SPLIT_OVERLAP / CHUNK_DURATION))
        self._max_speech_frames = int(MAX_SPEECH_DURATION / CHUNK_DURATION)

        self._state = _VADState.WAITING
        self._speech_frames: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self._pre_roll_frames)
        self._silence_counter = 0

    @abstractmethod
    def _get_prob(self, chunk: np.ndarray) -> float:
        """返回该音频块的语音概率（0~1）"""
        ...

    def _on_segment_end(self) -> None:
        """语音段结束时的钩子（子类可覆写，如重置模型状态）"""
        pass

    def process(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        处理一个 PCM 块，返回完整语音段（仅在段落结束时）。

        Args:
            chunk: shape=[CHUNK_SAMPLES], dtype=float32, range=[-1, 1]

        Returns:
            np.ndarray: 完整语音段的 PCM 数据（触发结束时）
            None: 语音未结束或段落过短被丢弃
        """
        prob = self._get_prob(chunk)

        if self._state == _VADState.WAITING:
            self._pre_roll.append(chunk)
            if prob >= self.speech_threshold:
                # 语音开始：带入预滚缓冲，避免截掉开头音节
                self._state = _VADState.SPEAKING
                self._speech_frames = list(self._pre_roll)
                self._silence_counter = 0
                logger.debug(f"语音开始（概率 {prob:.2f}）")

        elif self._state == _VADState.SPEAKING:
            self._speech_frames.append(chunk)

            if prob < self.silence_threshold:
                self._silence_counter += 1
                if self._silence_counter >= self._silence_frames:
                    return self._finalize()
            else:
                self._silence_counter = 0

            # 超时强制截断
            if len(self._speech_frames) >= self._max_speech_frames:
                logger.warning(f"语音段超过 {MAX_SPEECH_DURATION}s，强制截断")
                return self._finalize(forced_split=True)

        return None

    def _finalize(self, forced_split: bool = False) -> Optional[np.ndarray]:
        """结束当前语音段，过短则丢弃"""
        audio = np.concatenate(self._speech_frames)
        duration = len(audio) / SAMPLE_RATE
        tail_for_next = []
        if forced_split and self._speech_frames:
            tail_for_next = self._speech_frames[-self._split_overlap_frames:]

        # 重置状态机
        self._state = _VADState.WAITING
        self._speech_frames = []
        self._pre_roll.clear()
        if tail_for_next:
            self._pre_roll.extend(tail_for_next)
        self._silence_counter = 0
        self._on_segment_end()

        if duration < MIN_SPEECH_DURATION:
            logger.debug(f"段落过短（{duration:.2f}s < {MIN_SPEECH_DURATION}s），丢弃")
            return None

        logger.info(f"语音段完成，时长 {duration:.2f}s")
        return audio


# ─────────────────────────────────────────────
#  Silero VAD（主选）
# ─────────────────────────────────────────────

class SileroVAD(_BaseVAD):
    """
    Silero VAD：Google Research 开源的高精度本地 VAD 模型。
    模型大小 ~1.8MB，首次运行自动下载到 ~/.cache/torch/hub/。
    对噪声、口音、中英混合的鲁棒性远优于能量阈值方案。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global torch
        if torch is None:
            try:
                import torch as _torch
                torch = _torch
            except Exception as e:
                raise RuntimeError(f"torch 未安装或加载失败，无法加载 Silero VAD: {e}") from e
        logger.info("正在加载 Silero VAD 模型（首次需联网下载 ~1.8MB）...")
        torch.set_num_threads(1)   # 单线程推理，降低调度延迟
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        self._model.eval()
        logger.info("Silero VAD 加载完成")

    def _get_prob(self, chunk: np.ndarray) -> float:
        tensor = torch.from_numpy(chunk).unsqueeze(0)   # [1, N]
        with torch.no_grad():
            return float(self._model(tensor, SAMPLE_RATE).item())

    def _on_segment_end(self) -> None:
        # 重置 Silero 内部 LSTM 状态，避免跨段干扰
        self._model.reset_states()


# ─────────────────────────────────────────────
#  Energy VAD（备用，Silero 加载失败时自动降级）
# ─────────────────────────────────────────────

class EnergyVAD(_BaseVAD):
    """
    基于 RMS 能量的简易 VAD。
    精度低于 Silero（对背景噪声敏感），作为 Silero 失败时的降级方案。
    """

    def __init__(self, energy_threshold: float = 0.015, **kwargs):
        super().__init__(**kwargs)
        self._threshold = energy_threshold
        logger.warning(
            "使用 EnergyVAD（备用）：精度较低，建议检查 Silero VAD 加载错误"
        )

    def _get_prob(self, chunk: np.ndarray) -> float:
        # RMS 能量归一化为 0~1 概率（简单线性映射）
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        return min(1.0, rms / self._threshold)


def create_vad() -> _BaseVAD:
    """工厂函数：优先使用 Silero，失败时降级到 Energy VAD"""
    if importlib.util.find_spec("torchaudio") is None:
        logger.warning("torchaudio 未安装，直接使用 EnergyVAD（可用但精度较低）")
        return EnergyVAD()
    try:
        return SileroVAD()
    except Exception as e:
        logger.warning(f"Silero VAD 加载失败，降级到 EnergyVAD: {e}")
        return EnergyVAD()


def _resolve_device_index(
    preferred_index: Optional[int],
    preferred_name: str = "",
) -> Optional[int]:
    """
    优先规则：
      1) 显式 AUDIO_DEVICE_INDEX
      2) AUDIO_DEVICE_NAME 子串匹配（仅输入设备）
      3) sounddevice 默认输入设备
    """
    if sd is None:
        return preferred_index
    if preferred_index is not None:
        return preferred_index

    name = preferred_name.strip().lower()
    if name:
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) < 1:
                continue
            if name in str(dev.get("name", "")).lower():
                return i
        logger.warning("未找到匹配 AUDIO_DEVICE_NAME=%r 的输入设备，将尝试默认输入", preferred_name)

    default_input = sd.default.device[0]
    if isinstance(default_input, int) and default_input >= 0:
        return default_input
    return None


# ─────────────────────────────────────────────
#  音频格式转换工具
# ─────────────────────────────────────────────

def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 numpy 数组 → 16-bit WAV 字节流（供 OpenAI API 使用）"""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)            # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ─────────────────────────────────────────────
#  STT 引擎（双模式）
# ─────────────────────────────────────────────

class STTEngine:
    """
    双模式语音识别引擎，通过 config.py 中的 STT_MODE 切换。

    模式 A (local):  faster-whisper（CPU int8 量化，无需联网）
    模式 B (api):    OpenAI Whisper API（更快，需要联网 + API Key）
                     网络失败时自动降级到本地模式
    """

    def __init__(self, mode: str = STT_MODE):
        if mode not in ("local", "api"):
            raise ValueError(f"STT_MODE 须为 'local' 或 'api'，当前：{mode!r}")
        if mode == "api" and not OPENAI_API_KEY:
            logger.warning("STT_MODE=api 但未设置 OPENAI_API_KEY，自动降级到 local")
            mode = "local"

        # 启动阶段就校验 STT 后端可用性，避免运行中每段语音都报同样的异常
        if mode == "local":
            try:
                import faster_whisper  # noqa: F401
            except Exception as e:
                if OPENAI_API_KEY:
                    logger.warning("本地 STT 不可用（%s），自动切换到 API 模式", e)
                    mode = "api"
                else:
                    raise RuntimeError(
                        "本地 STT 不可用（缺少 faster-whisper），且未配置 OPENAI_API_KEY；"
                        "请安装依赖或改用 --mock 模式。"
                    ) from e

        self.mode = mode
        self._local_model = None       # 懒加载，首次调用时初始化
        self._openai_client = None
        logger.info(f"STT 引擎已初始化，模式: {mode.upper()}")

    # ── 本地模式 ──────────────────────────────────────────────────────────

    def _ensure_local_model(self) -> None:
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
            except Exception as e:
                raise RuntimeError(
                    "faster-whisper 未安装或加载失败，请安装依赖后重试"
                ) from e
            logger.info(
                f"加载 faster-whisper 模型（{WHISPER_LOCAL_MODEL}），"
                "首次需要下载，请稍候..."
            )
            self._local_model = WhisperModel(
                WHISPER_LOCAL_MODEL,
                device="cpu",
                compute_type="int8",  # int8 量化：速度提升 2~3x，精度损失可忽略
            )
            logger.info("faster-whisper 加载完成")

    def _transcribe_local_sync(self, audio: np.ndarray) -> str:
        """本地 faster-whisper 转录（同步，供 run_in_executor 调用）"""
        self._ensure_local_model()
        forced_lang, allowed_langs, default_lang = _current_lang_policy()
        def _run_once(lang: Optional[str], use_vad: bool) -> tuple[str, str, float]:
            segments, info = self._local_model.transcribe(
                audio,
                beam_size=max(1, WHISPER_BEAM_SIZE),
                best_of=max(1, WHISPER_BEST_OF),
                condition_on_previous_text=WHISPER_CONDITION_ON_PREV,
                language=lang,
                vad_filter=use_vad,
                vad_parameters={"min_silence_duration_ms": 320},
            )
            text = "".join(seg.text for seg in segments).strip()
            return text, str(getattr(info, "language", "")), float(getattr(info, "language_probability", 0.0) or 0.0)

        # 强制语言：最快且最稳（面试场景推荐）
        if forced_lang:
            text, detected_lang, prob = _run_once(forced_lang, WHISPER_INTERNAL_VAD)
            logger.debug("[本地] 强制语言=%s 检测=%s(%.2f) 文本=%r", forced_lang, detected_lang, prob, text[:60])
            return text

        # 自动双语：先跑 auto；只有失败时才单次回退，避免多轮重试拖慢
        text, detected_lang, prob = _run_once(None, WHISPER_INTERNAL_VAD)
        logger.debug("[本地] 自动语言=%s(%.2f)，文本=%r", detected_lang, prob, text[:60])
        if text and (detected_lang in allowed_langs or prob >= 0.78):
            return text

        t2, lang2, prob2 = _run_once(default_lang, True)
        if t2 and len(t2) >= max(2, len(text)):
            logger.info("STT 语言回退命中: %s->%s(%.2f)", default_lang, lang2 or default_lang, prob2)
            return t2
        return text

    # ── API 模式 ───────────────────────────────────────────────────────────

    async def _transcribe_api(self, audio: np.ndarray) -> str:
        """OpenAI Whisper API 异步转录"""
        import openai
        forced_lang, _, _ = _current_lang_policy()
        if self._openai_client is None:
            self._openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        wav_bytes = _numpy_to_wav_bytes(audio)
        last_err: Optional[Exception] = None
        for attempt in range(1, STT_API_MAX_RETRIES + 2):
            try:
                async with asyncio.timeout(STT_API_TIMEOUT_SECONDS):
                    response = await self._openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("segment.wav", wav_bytes, "audio/wav"),
                        response_format="text",
                        language=forced_lang,
                    )
                text = str(response).strip()
                logger.debug(f"[API] 返回文本={text[:60]!r}")
                return text
            except Exception as e:
                last_err = e
                if attempt <= STT_API_MAX_RETRIES:
                    await asyncio.sleep(0.5 * attempt)
        raise RuntimeError(f"Whisper API 调用失败: {last_err}")

    # ── 公开接口 ───────────────────────────────────────────────────────────

    async def transcribe(self, audio: np.ndarray) -> str:
        """
        异步转录：自动根据模式选择本地或 API，API 失败时降级到本地。

        本地推理通过 run_in_executor 在线程池执行，不阻塞事件循环。
        """
        loop = asyncio.get_event_loop()

        if self.mode == "local":
            return await loop.run_in_executor(
                None, lambda: self._transcribe_local_sync(audio)
            )

        # API 模式：失败时自动降级
        try:
            return await self._transcribe_api(audio)
        except Exception as e:
            logger.warning(f"API 转录失败（{e}），自动降级到本地模式")
            return await loop.run_in_executor(
                None, lambda: self._transcribe_local_sync(audio)
            )


# ─────────────────────────────────────────────
#  主处理器（整合 sounddevice + VAD + STT）
# ─────────────────────────────────────────────

class AudioProcessor:
    """
    音频处理器，负责整合音频捕获、VAD 分段、STT 转录三个阶段。

    并发模型：
      - sounddevice InputStream：由 PortAudio 的 C 层回调线程驱动（不可阻塞）
      - VAD 线程：独立 daemon 线程，消费 _raw_queue
      - STT 协程：asyncio 事件循环内，消费 _speech_queue
      - 线程→asyncio 桥接：call_soon_threadsafe（唯一安全方式）
    """

    def __init__(
        self,
        device_index: Optional[int] = AUDIO_DEVICE_INDEX,
        stt_mode: str = STT_MODE,
    ):
        self.device_index = _resolve_device_index(device_index, AUDIO_DEVICE_NAME)
        self._stt = STTEngine(mode=stt_mode)
        # fixed 分段时不依赖 VAD，避免无谓模型加载与依赖问题
        self._vad: Optional[_BaseVAD] = None
        self._segment_mode = STT_SEGMENT_MODE
        self._active_segment_mode = "vad"

        # 原始 PCM 块队列（线程安全，有界防止内存溢出）
        self._raw_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(200, STT_RAW_QUEUE_MAX))

        # 完整语音段队列（asyncio 安全）
        self._speech_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=max(16, STT_AUDIO_QUEUE_MAX))

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._stream_channels: int = 1
        self._last_audio_ts: float = 0.0
        self._last_audio_rms: float = 0.0
        self._last_audio_log_ts: float = 0.0

    def _decide_segment_mode(self, dev_name: str) -> str:
        mode = (self._segment_mode or "auto").lower()
        if mode in ("vad", "fixed"):
            return mode
        n = (dev_name or "").lower()
        # 回环设备优先固定分段，保证连续语音全量
        for kw in ("blackhole", "background", "lark", "wemeet", "boom", "virtual"):
            if kw in n:
                return "fixed"
        return "vad"

    # ── sounddevice 回调（C 线程，严禁阻塞）──────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            logger.warning(f"音频流状态: {status}")
        try:
            # 多声道输入时做平均混音，避免仅取单通道导致“有声却检测不到”
            mono = indata.mean(axis=1).astype(np.float32, copy=False)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            now = time.monotonic()
            self._last_audio_ts = now
            self._last_audio_rms = rms
            if now - self._last_audio_log_ts >= 3.0:
                db = 20.0 * math.log10(max(rms, 1e-8))
                logger.info("输入电平: rms=%.5f (%.1f dBFS)", rms, db)
                self._last_audio_log_ts = now
            self._raw_queue.put_nowait(mono.copy())
        except queue.Full:
            pass  # 队列满时静默丢弃，绝对不能阻塞此回调

    # ── VAD 处理线程 ──────────────────────────────────────────────────────

    def _vad_worker(self) -> None:
        """独立线程：从 _raw_queue 取音频块 → 通过 VAD → 段落完成送入 asyncio 队列"""
        logger.info("VAD 线程启动")
        if self._vad is None:
            self._vad = create_vad()
        while not self._stop_event.is_set():
            try:
                chunk = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            segment = self._vad.process(chunk)
            if segment is not None:
                self._bridge_to_asyncio(segment)

        logger.info("VAD 线程退出")

    def _fixed_chunk_worker(self) -> None:
        """固定时窗切段：连续语音场景下更稳定，尽量避免遗漏。"""
        logger.info("固定分段线程启动")
        seg_frames = max(1, int(STT_FIXED_SEGMENT_SECONDS / CHUNK_DURATION))
        overlap_frames = max(1, int(STT_FIXED_OVERLAP_SECONDS / CHUNK_DURATION))
        keep_frames = min(seg_frames - 1, overlap_frames)
        buf: list[np.ndarray] = []

        while not self._stop_event.is_set():
            try:
                chunk = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            buf.append(chunk)
            if len(buf) >= seg_frames:
                segment = np.concatenate(buf[:seg_frames])
                seg_rms = float(np.sqrt(np.mean(np.square(segment)))) if segment.size else 0.0
                if seg_rms >= STT_FIXED_MIN_RMS:
                    self._bridge_to_asyncio(segment)
                else:
                    logger.debug("固定分段跳过静音块: rms=%.5f", seg_rms)
                buf = buf[seg_frames - keep_frames :]

        # 停止前把剩余音频也送出，避免尾部丢失
        if len(buf) >= max(1, int(1.0 / CHUNK_DURATION)):
            tail = np.concatenate(buf)
            tail_rms = float(np.sqrt(np.mean(np.square(tail)))) if tail.size else 0.0
            if tail_rms >= STT_FIXED_MIN_RMS:
                self._bridge_to_asyncio(tail)
        logger.info("固定分段线程退出")

    def _bridge_to_asyncio(self, segment: np.ndarray) -> None:
        """
        将语音段从工作线程安全地投递到 asyncio 队列。
        call_soon_threadsafe 是从线程操作 asyncio 的唯一安全方式。
        """
        if self._loop is None:
            return

        def _safe_put() -> None:
            try:
                self._speech_queue.put_nowait(segment)
            except asyncio.QueueFull:
                # 高负载下优先保留最新段，淘汰最旧段，减少“当前内容”漏识别
                try:
                    _ = self._speech_queue.get_nowait()
                    self._speech_queue.put_nowait(segment)
                    logger.warning("语音队列已满，已丢弃最旧段并写入新段")
                except Exception:
                    logger.warning("语音队列已满，丢弃此段（STT 处理跟不上）")

        self._loop.call_soon_threadsafe(_safe_put)

    # ── 异步 STT 协程 ─────────────────────────────────────────────────────

    async def _stt_worker(self, text_queue: asyncio.Queue) -> None:
        """持续从 _speech_queue 取语音段 → STT → 结果放入 text_queue"""
        logger.info("STT 协程启动")
        last_silence_warn = 0.0
        auto_tuned_vad = False
        stitched = ""
        stitched_at = 0.0
        stitched_first_at = 0.0
        while not self._stop_event.is_set():
            try:
                audio = await asyncio.wait_for(
                    self._speech_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                now = time.monotonic()
                if STT_STITCH_ENABLED and stitched and now - stitched_at >= STT_STITCH_FLUSH_SECONDS:
                    await text_queue.put(stitched)
                    logger.info("[STT 拼接输出] %s", stitched[:100])
                    stitched = ""
                    stitched_at = 0.0
                    stitched_first_at = 0.0
                if self._last_audio_ts > 0 and now - self._last_audio_ts >= 8.0 and now - last_silence_warn >= 8.0:
                    logger.warning(
                        "连续 %.1fs 未检测到有效输入音频（当前 rms=%.5f）。"
                        "请检查系统输出是否路由到 BlackHole/多输出设备。",
                        now - self._last_audio_ts,
                        self._last_audio_rms,
                    )
                    last_silence_warn = now
                # 有持续输入但不出段时，自动降低一次阈值，减少“监听中无响应”
                if (
                    self._active_segment_mode == "vad"
                    and self._vad is not None
                    and
                    not auto_tuned_vad
                    and self._last_audio_rms > 0.002
                    and self._last_audio_ts > 0
                    and now - self._last_audio_ts < 2.0
                    and getattr(self._vad, "speech_threshold", 0.5) > 0.35
                ):
                    old = float(getattr(self._vad, "speech_threshold", 0.5))
                    setattr(self._vad, "speech_threshold", 0.35)
                    setattr(self._vad, "silence_threshold", max(0.0, 0.35 - 0.15))
                    auto_tuned_vad = True
                    logger.warning("检测到有输入但未触发 VAD，阈值自动从 %.2f 下调到 0.35", old)
                continue

            t0 = time.perf_counter()
            try:
                text = await self._stt.transcribe(audio)
            except Exception as e:
                logger.error(f"STT 转录异常: {e}")
                continue

            elapsed = time.perf_counter() - t0
            text = _to_simplified_zh(text).strip()
            if text:
                logger.info(f"[STT {elapsed:.2f}s] {text[:80]}")
                if not STT_STITCH_ENABLED:
                    await text_queue.put(text)
                    continue
                if not stitched:
                    stitched = text
                    stitched_at = time.monotonic()
                    stitched_first_at = stitched_at
                    continue
                stitched = _merge_with_overlap(stitched, text)
                stitched_at = time.monotonic()
                # 句末或过长则立即输出，避免无限累积
                hold = stitched_at - (stitched_first_at or stitched_at)
                if (
                    _looks_sentence_finished(stitched)
                    or len(stitched) >= STT_STITCH_MAX_CHARS
                    or hold >= STT_STITCH_MAX_HOLD_SECONDS
                ):
                    await text_queue.put(stitched)
                    logger.info("[STT 拼接输出] %s", stitched[:100])
                    stitched = ""
                    stitched_at = 0.0
                    stitched_first_at = 0.0
            else:
                logger.debug(f"STT 返回空文本（{elapsed:.2f}s），跳过")

        if STT_STITCH_ENABLED and stitched:
            await text_queue.put(stitched)
            logger.info("[STT 拼接输出] %s", stitched[:100])
        logger.info("STT 协程退出")

    # ── 公开接口 ──────────────────────────────────────────────────────────

    async def run(self, text_queue: asyncio.Queue) -> None:
        """
        启动完整音频处理管道（阻塞运行，直到调用 stop()）。

        Args:
            text_queue: 外部 asyncio.Queue，识别到的文本字符串将被 put 进此队列
        """
        if sd is None:
            raise RuntimeError("sounddevice 未安装，无法启用音频采集")

        self._loop = asyncio.get_event_loop()
        self._stop_event.clear()

        # 启动 VAD 守护线程
        seg_mode = "vad"
        self._vad_thread = threading.Thread(
            target=self._vad_worker,
            daemon=True,
            name="VAD-Worker",
        )

        # 打印设备信息
        try:
            if self.device_index is None:
                raise RuntimeError("未检测到可用输入设备，请设置 AUDIO_DEVICE_INDEX 或 AUDIO_DEVICE_NAME")
            dev_info = sd.query_devices(self.device_index)
        except Exception as e:
            raise RuntimeError(
                f"音频设备初始化失败（设备编号/权限/采样率异常）: {e}"
            ) from e
        logger.info(
            f"音频设备: [{self.device_index if self.device_index is not None else '默认'}] "
            f"{dev_info['name']} | 采样率: {SAMPLE_RATE}Hz"
        )
        self._stream_channels = 2 if int(dev_info.get("max_input_channels", 1) or 1) >= 2 else 1
        logger.info("音频通道模式: 输入 %d 通道 -> 混音单声道", self._stream_channels)
        seg_mode = self._decide_segment_mode(str(dev_info.get("name", "")))
        self._active_segment_mode = seg_mode
        logger.info("音频切段模式: %s", seg_mode)
        if seg_mode == "fixed":
            self._vad_thread = threading.Thread(
                target=self._fixed_chunk_worker,
                daemon=True,
                name="FixedSeg-Worker",
            )
        else:
            self._vad = create_vad()
        self._vad_thread.start()

        # 启动 sounddevice 输入流
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=self._stream_channels,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                device=self.device_index,
                callback=self._audio_callback,
                latency="low",
            )

            with self._stream:
                logger.info("开始监听音频...（Ctrl+C 退出）")
                await self._stt_worker(text_queue)
        except Exception as e:
            raise RuntimeError(
                f"音频流启动失败（请检查麦克风权限/设备采样率）: {e}"
            ) from e

    def stop(self) -> None:
        """停止音频处理管道"""
        logger.info("正在停止音频处理器...")
        self._stop_event.set()
        if self._stream and self._stream.active:
            self._stream.stop()


# ─────────────────────────────────────────────
#  公开接口（供 M4 主程序调用）
# ─────────────────────────────────────────────

async def start_listening(
    text_queue: asyncio.Queue,
    device_index: Optional[int] = None,
    stt_mode: Optional[str] = None,
) -> None:
    """
    顶层异步接口：启动音频监听，将 STT 结果逐条 put 进 text_queue。

    M4 主程序调用示例：
        text_q = asyncio.Queue()
        asyncio.create_task(start_listening(text_q))
        # 之后通过 await text_q.get() 获取识别文本

    Args:
        text_queue:   接收 STT 结果的 asyncio.Queue（字符串）
        device_index: 覆盖配置文件中的音频设备编号（None = 使用配置）
        stt_mode:     覆盖配置文件中的 STT 模式（None = 使用配置）
    """
    processor = AudioProcessor(
        device_index=device_index if device_index is not None else AUDIO_DEVICE_INDEX,
        stt_mode=stt_mode if stt_mode is not None else STT_MODE,
    )
    await processor.run(text_queue)


def list_audio_devices() -> None:
    """
    列出系统所有音频输入设备，用于确认 BlackHole 等虚拟设备的编号。

    输出示例：
        [ 0] Built-in Microphone        (默认)
        [ 2] BlackHole 2ch              ← 捕获扬声器输出
        [ 3] MacBook Pro Speakers
    """
    if sd is None:
        print("sounddevice 未安装，无法列出音频设备")
        return

    print("\n" + "=" * 64)
    print("  可用音频输入设备")
    print("=" * 64)
    default_input = sd.default.device[0]
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        tag = " ← 默认" if i == default_input else ""
        print(f"  [{i:2d}] {dev['name']}{tag}")
        print(
            f"       采样率: {int(dev['default_samplerate'])}Hz  "
            f"输入通道: {dev['max_input_channels']}"
        )
    print("=" * 64)
    print("\n提示：捕获扬声器输出请使用 BlackHole 2ch（brew install blackhole-2ch）")
    print("      设备确认后在 .env 中设置 AUDIO_DEVICE_INDEX=<编号>\n")


# ─────────────────────────────────────────────
#  CLI 测试入口
# ─────────────────────────────────────────────

async def _test_main(device: Optional[int] = None) -> None:
    """实时音频转录测试"""
    text_queue: asyncio.Queue[str] = asyncio.Queue()

    async def _printer() -> None:
        print("\n[监听中] 请说话，识别结果将实时显示（Ctrl+C 退出）\n")
        while True:
            text = await text_queue.get()
            print(f"\n{'─' * 50}")
            print(f"  识别：{text}")
            print(f"{'─' * 50}\n")

    await asyncio.gather(
        start_listening(text_queue, device_index=device),
        _printer(),
    )


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="音频捕获与 STT 模块测试")
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="列出所有音频输入设备并退出",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="指定音频输入设备编号（覆盖 .env 配置）",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["local", "api"],
        default=None,
        help="指定 STT 模式（覆盖 .env 配置）",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # 覆盖模式
    if args.mode:
        import os
        os.environ["STT_MODE"] = args.mode

    try:
        asyncio.run(_test_main(device=args.device))
    except KeyboardInterrupt:
        print("\n已停止监听")
_S2T_CONVERTER = None
_SENTENCE_ENDERS = ("。", "！", "？", ".", "!", "?", "…")


def _to_simplified_zh(text: str) -> str:
    """将中文文本尽量转为简体（若 opencc 不可用则原样返回）。"""
    if not text:
        return text
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    global _S2T_CONVERTER
    if _S2T_CONVERTER is None:
        try:
            from opencc import OpenCC
            _S2T_CONVERTER = OpenCC("t2s")
        except Exception:
            _S2T_CONVERTER = False
    if _S2T_CONVERTER:
        try:
            return _S2T_CONVERTER.convert(text)
        except Exception:
            return text
    return text


def _looks_sentence_finished(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return t.endswith(_SENTENCE_ENDERS)


def _merge_with_overlap(prev: str, nxt: str, max_overlap: int = 50) -> str:
    """拼接相邻 STT 片段并去掉重复重叠，减少截断导致的断裂。"""
    a = prev.strip()
    b = nxt.strip()
    if not a:
        return b
    if not b:
        return a
    m = min(max_overlap, len(a), len(b))
    overlap = 0
    for k in range(m, 0, -1):
        if a[-k:] == b[:k]:
            overlap = k
            break
    sep = "" if re.search(r"[\u4e00-\u9fff]$", a) and re.match(r"^[\u4e00-\u9fff]", b) else " "
    return a + sep + b[overlap:]
