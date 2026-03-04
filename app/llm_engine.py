"""
llm_engine.py — LLM 调度引擎（OpenRouter + 可降级检索）
"""

import asyncio
import hashlib
import logging
import re
import time
from typing import Callable, Optional

import openai

from config import (
    DEBOUNCE_SECONDS,
    LLM_CONTEXT_CHAR_BUDGET,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_RETRY_BACKOFF_SECONDS,
    LLM_RETRY_SHRINK_RATIO,
    LLM_MIN_ANSWER_CHARS,
    LLM_STREAMING,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_CUSTOM_MODELS,
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
    OPENROUTER_WEB_PLUGIN_ENABLED,
    RAG_TOP_K,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_MAX_RESULTS,
)
from rag_search import async_rag_search

logger = logging.getLogger("LLMEngine")

CLEAR_SIGNAL = "<<<CLEAR>>>"
DONE_SIGNAL = "<<<DONE>>>"
ERROR_SIGNAL = "<<<ERROR>>>"
CANCELED_SIGNAL = "<<<CANCELED>>>"

UICallback = Callable[[str], None]

DEFAULT_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-pro",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
]

_runtime_model: str = OPENROUTER_MODEL
_runtime_base_url: str = OPENROUTER_BASE_URL
_runtime_api_key: str = OPENROUTER_API_KEY
_runtime_custom_models: list[str] = list(OPENROUTER_CUSTOM_MODELS)
_transcript_context: str = ""
_extra_prompt: str = ""

SYSTEM_PROMPT = """\
你是「高级面试实战教练 + 行业专家」。
目标：帮助候选人在短时间给出“高密度、可验证、可追问展开”的专业回答，让面试官感受到结构化思维、业务判断力和执行深度。

【总原则】
1. 先给结论，再给依据；观点必须可落地、可追问。
2. 尽量引用候选人资料中的真实项目、职责、指标；无法确认时，明确为“通用实践”。
3. 对技术/产品/商业问题，必须体现取舍：为什么选A，不选B，代价是什么。
4. 回答要兼顾“当下可说”与“追问可扩展”。
5. 禁止空泛词：如“赋能、闭环、全栈领先”等，除非给出证据。

【推荐输出结构（默认）】
• 核心判断：一句话直接回答问题本质
• 关键依据：2~3条（数据、事实、项目经历、方法论）
• 方案取舍：给出备选方案对比与选择理由
• 风险与边界：1条主要风险 + 应对策略
• 追问延展：给出下一层可展开的方向（便于继续作答）

【质量标准】
1. 专业：术语准确，逻辑严密，不自相矛盾。
2. 深度：包含因果链和决策依据，而非表面描述。
3. 准确：不编造履历，不虚构数据；不确定就明确“待验证”。
4. 实战：回答应可直接口述，控制在面试场景可用长度。
5. 当题目涉及事实、行业、公司、政策、数据时：优先使用检索信息与候选人资料，附上来源线索，不要泛泛而谈。
6. 禁止空泛开场（例如“好的，这是一个典型场景”）。直接进入分析结论。

【语言要求】
1. 中文问题 -> 简体中文回答；英文问题 -> 英文回答。
2. 严禁输出繁体中文。
3. 默认使用短句，避免长段堆叠。

【输出硬约束】
1. 必须包含以下5段小标题：核心判断、关键依据、方案取舍、风险与边界、可落地动作。
2. 总长度不少于 350 个中文字符（或等价英文信息量）。
3. 不要只输出一两句；即使问题不完整，也要先合理补全问题意图后再给完整答案。
"""

_client: Optional[openai.AsyncOpenAI] = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        if not _runtime_api_key:
            raise EnvironmentError("OPENROUTER_API_KEY 未设置")
        _client = openai.AsyncOpenAI(
            api_key=_runtime_api_key,
            base_url=_runtime_base_url,
            default_headers={
                "HTTP-Referer": "https://interview-assistant.local",
                "X-Title": "AI Interview Assistant",
            },
        )
        logger.info("OpenRouter 客户端初始化完成，默认模型: %s", _runtime_model)
    return _client


async def close_client() -> None:
    global _client
    if _client is None:
        return
    try:
        await _client.close()
    except Exception:
        pass
    finally:
        _client = None


async def probe_connection(api_key: str, base_url: str, model: str) -> tuple[bool, str]:
    """用于设置页快速连通性验证（API Key + Base URL + 模型）。"""
    api_key = (api_key or "").strip()
    base_url = (base_url or "").strip()
    model = (model or "").strip()
    if not api_key:
        return False, "API Key 不能为空"
    if not base_url:
        return False, "Base URL 不能为空"
    if not model:
        return False, "测试模型不能为空"
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        async with asyncio.timeout(16.0):
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with: pong"}],
                max_tokens=12,
                temperature=0,
                stream=False,
            )
        text = ""
        if resp.choices and getattr(resp.choices[0], "message", None):
            text = str(getattr(resp.choices[0].message, "content", "") or "").strip()
        if text:
            return True, f"测试成功：{text[:60]}"
        return True, "测试成功：模型已返回响应"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        try:
            await client.close()
        except Exception:
            pass


def set_runtime_model(model: str) -> None:
    global _runtime_model
    model = (model or "").strip()
    if not model:
        return
    _runtime_model = model
    logger.info("运行时模型已切换为: %s", _runtime_model)


def get_runtime_model() -> str:
    return _runtime_model


def get_available_models() -> list[str]:
    seen = set()
    out: list[str] = []
    for name in [*DEFAULT_MODELS, *_runtime_custom_models]:
        n = (name or "").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    if _runtime_model and _runtime_model not in seen:
        out.insert(0, _runtime_model)
    return out


def set_runtime_custom_models(models: list[str]) -> None:
    global _runtime_custom_models
    cleaned = []
    seen = set()
    for m in models:
        n = (m or "").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        cleaned.append(n)
    _runtime_custom_models = cleaned
    logger.info("运行时自定义模型已更新，数量=%d", len(_runtime_custom_models))


def set_runtime_connection(*, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
    global _runtime_api_key, _runtime_base_url, _client
    changed = False
    if api_key is not None:
        v = api_key.strip()
        if v and v != _runtime_api_key:
            _runtime_api_key = v
            changed = True
    if base_url is not None:
        v = base_url.strip()
        if v and v != _runtime_base_url:
            _runtime_base_url = v
            changed = True
    if changed:
        old = _client
        _client = None
        if old is not None:
            try:
                asyncio.create_task(old.close())
            except Exception:
                pass
        logger.info("运行时连接配置已更新")


def set_transcript_context(text: str) -> None:
    global _transcript_context
    _transcript_context = (text or "").strip()
    logger.info("已更新转录上下文，长度=%d", len(_transcript_context))


def set_extra_prompt(text: str) -> None:
    global _extra_prompt
    _extra_prompt = (text or "").strip()
    logger.info("已更新自定义 Prompt，长度=%d", len(_extra_prompt))


def get_extra_prompt() -> str:
    return _extra_prompt


def _question_id(text: str) -> str:
    return hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()[:10]


def _fmt_rag(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["【个人资料（本地简历 / 项目文档）】"]
    for i, r in enumerate(results, 1):
        snippet = r.get("content", "")[:450].replace("\n", " ").strip()
        lines.append(f"{i}. [{r.get('source', '未知来源')}] {snippet}")
    return "\n".join(lines)


def _fmt_web(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["【联网搜索结果（最新信息）】"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "（无标题）")
        body = r.get("body", "")[:260].replace("\n", " ").strip()
        lines.append(f"{i}. {title} — {body}")
    return "\n".join(lines)


def _budget_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 12)] + "\n...[截断]"


def _build_messages(question: str, rag_ctx: str, web_ctx: str, budget: int) -> list[dict]:
    # 优先保留问题本身，其次 RAG，上下文超预算时裁剪
    question = question.strip()
    q_budget = min(max(600, budget // 3), budget)
    q_part = _budget_text(question, q_budget)

    remain = max(0, budget - len(q_part) - 120)
    transcript_budget = int(remain * 0.35) if _transcript_context else 0
    rag_budget = int((remain - transcript_budget) * 0.7)
    web_budget = max(0, remain - transcript_budget - rag_budget)

    parts: list[str] = []
    rag_part = _budget_text(rag_ctx, rag_budget)
    web_part = _budget_text(web_ctx, web_budget)

    if rag_part:
        parts.append(rag_part)
    if web_part:
        parts.append(web_part)
    if _transcript_context:
        parts.append("【面试实录上下文】\n" + _budget_text(_transcript_context, transcript_budget))
    if _extra_prompt:
        parts.append("【回答策略补充（用户自定义 Prompt）】\n" + _budget_text(_extra_prompt, max(300, int(remain * 0.25))))

    parts.append(f"【面试官问题】\n{q_part}")
    parts.append(
        "请按“核心判断/关键依据/方案取舍/风险与边界/可落地动作”完整输出。"
        "如果面试官问题是陈述句或不完整，请先提炼为可回答的问题，再给完整回答。"
        "回答目标长度：中文约 450-900 字，英文约 250-450 words。"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _model_chain() -> list[str]:
    """严格使用当前选择模型，不自动降级。"""
    return [_runtime_model]


def _plain_len(text: str) -> int:
    return len(re.sub(r"\s+", "", (text or "").strip()))


def _needs_continuation(finish_reason: str) -> bool:
    return (finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}


async def _call_llm_once(messages: list[dict], callback: UICallback, model: str) -> tuple[str, str]:
    client = _get_client()
    model_name = (model or "").lower()
    use_stream = LLM_STREAMING
    timeout_seconds = max(LLM_TIMEOUT_SECONDS, 70.0 if "pro" in model_name else LLM_TIMEOUT_SECONDS)
    req = {
        "model": model,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "max_completion_tokens": LLM_MAX_TOKENS,
    }
    # OpenRouter 扩展字段必须放在 extra_body，确保被网关正确接收
    extra_body: dict = {}
    if OPENROUTER_WEB_PLUGIN_ENABLED and WEB_SEARCH_ENABLED:
        extra_body["plugins"] = [{"id": "web", "max_results": WEB_SEARCH_MAX_RESULTS}]
    # Gemini 系列显式压低 reasoning 强度，降低隐藏推理 token 挤占可见输出
    if "gemini" in model_name and OPENROUTER_REASONING_EFFORT in {"minimal", "low", "medium", "high"}:
        extra_body["reasoning"] = {"effort": OPENROUTER_REASONING_EFFORT}
    # 兼容 Gemini/OpenRouter 对输出 token 参数的不同映射
    extra_body["max_output_tokens"] = LLM_MAX_TOKENS
    if extra_body:
        req["extra_body"] = extra_body

    async with asyncio.timeout(timeout_seconds):
        if use_stream:
            full = []
            final_reason = "unknown"
            stream = await client.chat.completions.create(
                stream=True,
                **req,
            )
            async for chunk in stream:
                if chunk.choices and getattr(chunk.choices[0], "finish_reason", None):
                    final_reason = str(chunk.choices[0].finish_reason)
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    callback(delta)
                    full.append(delta)
            text = "".join(full)
            logger.info("LLM 返回统计 model=%s chars=%d finish=%s", model, _plain_len(text), final_reason)
            return text, final_reason
        else:
            resp = await client.chat.completions.create(
                stream=False,
                **req,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            finish_reason = str(resp.choices[0].finish_reason) if resp.choices else "unknown"
            if content:
                text = str(content)
                callback(text)
                logger.info("LLM 返回统计 model=%s chars=%d finish=%s", model, _plain_len(text), finish_reason)
                return text, finish_reason
            logger.info("LLM 返回统计 model=%s chars=0 finish=%s", model, finish_reason)
            return "", finish_reason


async def _stream_llm_with_retry(
    question: str,
    rag_ctx: str,
    web_ctx: str,
    callback: UICallback,
) -> None:
    last_err: Optional[Exception] = None

    models = _model_chain()
    max_attempt = LLM_MAX_RETRIES + 1
    for m_idx, model in enumerate(models, start=1):
        for attempt in range(1, max_attempt + 1):
            factor = LLM_RETRY_SHRINK_RATIO ** (attempt - 1)
            budget = max(900, int(LLM_CONTEXT_CHAR_BUDGET * factor))
            messages = _build_messages(question, rag_ctx, web_ctx, budget=budget)
            try:
                if m_idx > 1 and attempt == 1:
                    logger.warning("主模型失败，自动降级到: %s", model)
                callback(f"{CLEAR_SIGNAL}:{_question_id(question)}")
                answer, finish_reason = await _call_llm_once(messages, callback, model=model)
                cont_round = 0
                # 仅在 API 明确“长度截断”时续传同一回答，避免半句中断
                while _needs_continuation(finish_reason) and cont_round < 2:
                    cont_round += 1
                    logger.warning("检测到输出截断（finish=%s），执行续传 round=%d", finish_reason, cont_round)
                    cont_messages = list(messages)
                    cont_messages.append({"role": "assistant", "content": answer})
                    cont_messages.append({
                        "role": "user",
                        "content": "继续上一条回答，从中断处接着写，不要重复前文。"
                    })
                    callback("\n")
                    more, finish_reason = await _call_llm_once(cont_messages, callback, model=model)
                    if not more.strip():
                        break
                    answer = (answer.rstrip() + "\n" + more.lstrip()).strip()
                if _plain_len(answer) < LLM_MIN_ANSWER_CHARS:
                    logger.warning(
                        "回答偏短（chars=%d, finish=%s, model=%s）。请提高 LLM_MAX_TOKENS 或调整提示词约束。",
                        _plain_len(answer), finish_reason, model
                    )
                callback(DONE_SIGNAL)
                return
            except asyncio.CancelledError:
                logger.info("LLM 输出被取消（新问题抢占）")
                callback(CANCELED_SIGNAL)
                raise
            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM 调用失败（model=%s attempt=%d/%d）: %s %r",
                    model, attempt, max_attempt, type(e).__name__, e
                )
                if attempt < max_attempt:
                    await asyncio.sleep(LLM_RETRY_BACKOFF_SECONDS * attempt)

    msg = f"LLM 调用失败：{type(last_err).__name__ if last_err else 'UnknownError'}"
    callback(f"{ERROR_SIGNAL}:{msg}")


async def _handle_question(question: str, callback: UICallback) -> None:
    logger.info("开始处理问题: %s", question[:80])
    t0 = time.perf_counter()

    try:
        rag_results = await async_rag_search(question, top_k=RAG_TOP_K)
    except Exception as e:
        logger.warning("RAG 检索失败，已降级为空: %s", e)
        rag_results = []
    web_results: list[dict] = []

    if WEB_SEARCH_ENABLED and OPENROUTER_WEB_PLUGIN_ENABLED:
        logger.info("联网策略：使用模型内置 web 插件")

    await _stream_llm_with_retry(
        question=question,
        rag_ctx=_fmt_rag(rag_results),
        web_ctx=_fmt_web(web_results),
        callback=callback,
    )

    logger.info("问题处理完成，耗时 %.2fs", time.perf_counter() - t0)


async def handle_single_query(question: str, callback: UICallback) -> None:
    """公开接口：UI 可在非监听状态下手动触发一次 AI 回答。"""
    await _handle_question(question, callback)


async def process_queries(text_queue: asyncio.Queue, ui_signal_callback: UICallback) -> None:
    logger.info("LLM 调度引擎启动")
    pending_auto: Optional[str] = None
    pending_manual: Optional[str] = None
    current_task: Optional[asyncio.Task] = None
    current_q_norm: str = ""
    last_asked_norm: str = ""
    last_asked_at: float = 0.0

    def _norm(text: str) -> str:
        return re.sub(r"[\W_]+", "", (text or "").lower())

    def _is_similar(a: str, b: str) -> bool:
        if not a or not b:
            return False
        # 仅完全重复视为同题，避免误杀“同主题不同问题”
        return a == b

    def _decode(item) -> tuple[str, int]:
        if isinstance(item, tuple) and len(item) >= 2:
            text = str(item[0]).strip()
            try:
                priority = int(item[1])
            except Exception:
                priority = 0
            return text, priority
        return str(item).strip(), 0

    async def _next_question() -> str:
        nonlocal pending_auto, pending_manual
        if pending_manual:
            q = pending_manual
            pending_manual = None
            return q
        if pending_auto:
            q = pending_auto
            pending_auto = None
            return q

        first_item = await text_queue.get()
        first_text, first_prio = _decode(first_item)
        if first_prio >= 10:
            return first_text

        accumulated: list[str] = [first_text]
        while True:
            deadline = asyncio.get_event_loop().time() + DEBOUNCE_SECONDS
            remaining = deadline - asyncio.get_event_loop().time()
            try:
                more_item = await asyncio.wait_for(text_queue.get(), timeout=remaining)
                more_text, more_prio = _decode(more_item)
                if more_prio >= 10:
                    pending_manual = more_text
                    break
                accumulated.append(more_text)
            except asyncio.TimeoutError:
                break

        return " ".join(accumulated).strip()

    while True:
        question = (await _next_question()).strip()
        if not question:
            continue
        if len(question) < 8:
            logger.debug("跳过过短片段: %r", question)
            continue
        qn = _norm(question)
        now = time.monotonic()
        if _is_similar(qn, last_asked_norm) and now - last_asked_at < 30.0:
            logger.info("跳过重复问题（30s 内）: %s", question[:60])
            continue

        current_task = asyncio.create_task(_handle_question(question, ui_signal_callback))
        current_q_norm = qn
        last_asked_norm = qn
        last_asked_at = now

        while True:
            try:
                incoming = await asyncio.wait_for(text_queue.get(), timeout=0.2)
                text, prio = _decode(incoming)
                if not text:
                    continue
                tn = _norm(text)
                if prio >= 10:
                    if _is_similar(tn, current_q_norm):
                        logger.info("忽略重复手动提问（与当前问题相同）")
                        continue
                    # 手动提问优先级最高：立即抢占当前自动回答
                    pending_manual = text
                    if current_task and not current_task.done():
                        logger.info("收到手动提问，抢占当前回答")
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
                    break
                # 自动转录只保留最新一条，避免队列堆积
                if _is_similar(tn, current_q_norm):
                    continue
                pending_auto = text
            except asyncio.TimeoutError:
                if current_task.done():
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.warning("处理问题失败，已跳过: %s", e)
                    break
