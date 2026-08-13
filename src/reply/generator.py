#!/usr/bin/env python3
"""L4 Reply Generator - 回复内容生成."""

import logging
import json
import os
import re
import threading
import time
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.base import ChatMessage, SenderType
from src.reply.evidence_utils import strip_assistant_history_lines
from src.reply.few_shot import PersonaFewShotRetriever
from src.reply.session_memory import SessionMemory, _extract_query_key
from src.tools import get_registry

_logger = logging.getLogger("src.reply.generator")

# Badcase JudgeWorker（可选，默认不启用）
_judge_worker = None
_judge_worker_lock = threading.Lock()

def _get_judge_worker():
    global _judge_worker
    if _judge_worker is None:
        with _judge_worker_lock:
            if _judge_worker is None:
                try:
                    from src.badcase.judge_worker import JudgeWorker
                    _judge_worker = JudgeWorker()
                    _logger.info("JudgeWorker initialized")
                except Exception as e:
                    _logger.warning("JudgeWorker init failed: %s", e)
    return _judge_worker


# ReAct 工具调用上限
MAX_TOOL_CALLS = 10
FINAL_RESPONSE_RESERVE_SECONDS = 3.0


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.environ.get(name, str(default)))))
    except ValueError:
        return default


def _collect_fact_issues(claims: Any) -> Tuple[List[str], List[str]]:
    """从抽取的 claims 汇总阻塞问题与独立复核候选。

    这是通用证据策略（只依据 LLM 返回的 verdict 标签，不做关键词/语义路由）：

    - ``contradicted`` 与 ``unknown`` 都是阻塞问题：``unknown`` 即“证据不足”，
      与矛盾一样不能作为事实进入最终回复；
    - ``nonfactual``（明显荒诞夸张、纯情绪或疑问式调侃）永不阻塞；
    - 所有非空 ``entailed`` 命题都进入独立复核候选，由复核器在既有 deadline 内逐条重判，
      不按方向性关键词过滤。

    返回 ``(issues, verify_claims)``。``issues`` 为阻塞问题文本列表；
    ``verify_claims`` 为需要交给独立复核器的 claim 文本列表。
    """
    issues: List[str] = []
    verify_claims: List[str] = []
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, dict):
            continue
        verdict = claim.get("verdict")
        claim_value = claim.get("claim")
        reason_value = claim.get("reason", "")
        if (
            verdict not in ("entailed", "contradicted", "unknown", "nonfactual")
            or not isinstance(claim_value, str)
            or not isinstance(reason_value, str)
        ):
            continue
        claim_text = claim_value.strip()
        reason = reason_value.strip()
        if not claim_text:
            continue
        if verdict in ("contradicted", "unknown"):
            label = "事实矛盾" if verdict == "contradicted" else "事实无依据"
            issues.append(f"{label}：{claim_text}；{reason}" if reason else f"{label}：{claim_text}")
        elif verdict == "entailed" and claim_text:
            verify_claims.append(claim_text)
    return issues, verify_claims

# Skill 路由优先级（数字越小优先级越高）
_SKILL_PRIORITY = [
    "3d_print_automation",
    "tuya_smart_home",
    "handling_vent",
    "handling_praise",
    "value_investing",
    "answering_questions",
    "receiving_share",
    "group_banter",
    "casual_chat",
]


class ReplyGenerator:
    def __init__(self, llm_client=None, memory_engine=None,
                 tool_registry=None,
                 judge_worker=None,
                 enable_time_awareness: bool = True,
                 enable_reply_restraint: bool = True,
                 enable_unread_dedup: bool = True,
                 enable_timestamps: bool = True):
        self.llm_client = llm_client
        self.memory_engine = memory_engine
        self.tool_registry = tool_registry or get_registry()
        self.judge_worker = judge_worker
        self.enable_time_awareness = enable_time_awareness
        self.enable_reply_restraint = enable_reply_restraint
        self.enable_unread_dedup = enable_unread_dedup
        self.enable_timestamps = enable_timestamps
        project_root = Path(__file__).parent.parent.parent
        few_shot_path = Path(os.environ.get(
            "PERSONA_FEW_SHOT_PATH",
            str(project_root / "data" / "few_shot" / "persona_examples.jsonl"),
        ))
        if not few_shot_path.is_absolute():
            few_shot_path = project_root / few_shot_path
        self.enable_persona_few_shots = os.environ.get("ENABLE_PERSONA_FEW_SHOTS", "0").lower() in ("1", "true", "yes", "on")
        self.persona_few_shot_count = _env_int("PERSONA_FEW_SHOT_COUNT", 8, 0, 12)
        self.persona_few_shot_max_chars = _env_int("PERSONA_FEW_SHOT_MAX_CHARS", 2500, 500, 5000)
        self.persona_few_shot_retriever = PersonaFewShotRetriever(few_shot_path)
        self.persona_few_shot_allow_unreviewed = os.environ.get("PERSONA_FEW_SHOT_ALLOW_UNREVIEWED", "0").lower() in ("1", "true", "yes", "on")
        self.persona_few_shot_ready = self.persona_few_shot_retriever.is_approved() or self.persona_few_shot_allow_unreviewed
        self.persona_wiki_dir = project_root / "data" / "memory" / "wiki" / "users"
        if self.enable_persona_few_shots and not self.persona_few_shot_ready:
            _logger.warning("[PersonaFewShot] 已启用但 report.json 未审核通过，跳过注入")
        _logger.info(f"[ReplyGenerator] init: llm_client={type(llm_client).__name__ if llm_client else None}")
        # 最后一次调用的 prompt/response（供 debug 使用）
        self.last_system_prompt: str = ""
        self.last_tools_context: str = ""
        self.last_user_prompt: str = ""
        self.last_raw_response: str = ""
        self.last_generation_failed: bool = False
        self.last_thinking: str = ""
        # 多轮调用完整链路（供 debug 使用）
        self.last_llm_calls: List[Dict] = []
        self.last_tool_calls: List[Dict] = []
        self.last_generation_trace: List[Dict] = []
        # Skill 加载状态（供 debug 使用）
        self.last_loaded_skills: List[str] = []
        self.last_skill_injected_content: str = ""
        self.last_few_shot_ids: List[str] = []
        self.last_few_shot_content: str = ""
        # 当前使用的模型名（供 debug 使用）
        self.last_active_llm: str = ""
        # 传给 Judge 的完整 LLM 上下文
        self.last_llm_messages: List[Dict] = []
        # 短期记忆（跨 tick 缓存工具结果）
        self.session_memory = SessionMemory()
        # 动态注册统一记忆搜索工具（如果 memory_engine 可用）
        # 同时返回两路结果：wiki 摘要 + 历史聊天原文（如果索引就绪）
        _history_available: Optional[Callable[[], bool]] = None
        _search_history: Optional[Callable[..., str]] = None
        try:
            from src.memory.history_search import is_available as _history_available
            from src.memory.history_search import search_history as _search_history
        except Exception as e:  # pragma: no cover - 防御性
            _logger.warning("[HistorySearch] 模块加载失败，仅启用 wiki 记忆搜索: %s", e)

        if self.memory_engine is not None:
            def _search_memory_adapter(query: str = "", top_k: int = 5) -> str:
                """适配器：同时搜索 wiki 摘要和历史聊天原文，合并返回。"""
                parts: List[str] = []

                # 1. wiki 长期记忆
                wiki_result = self.memory_engine.search_keyword(query, max_chars=3000)
                if wiki_result and "未在本地记忆中找到" not in wiki_result:
                    parts.append(f"【人物/群聊摘要】\n{wiki_result}")

                # 2. 历史聊天原文（可选，依赖索引）
                if _history_available is not None and _search_history is not None and _history_available():
                    try:
                        history_result = _search_history(query, top_k=top_k)
                        if history_result and "未找到" not in history_result:
                            parts.append(f"【历史聊天原文】\n{history_result}")
                    except Exception as e:
                        _logger.warning("[search_memory] 历史原文检索失败: %s", e)

                if not parts:
                    return "未找到相关记忆。"
                return "\n\n".join(parts)

            self.tool_registry.register(
                name="search_memory",
                description=(
                    "搜索本地记忆。同时返回两路结果："
                    "(1) 人物/群聊摘要（wiki）：用于查身边的人是谁、什么关系、近况/在哪/做什么/在什么公司；"
                    "(2) 历史聊天原文：用于回忆过去某段对话具体说了什么。"
                    "只要消息涉及示例用户甲现实生活中认识的人、所在的群、去过的地方、做过的事、工作/投资/家庭相关的人或公司，"
                    "都应该先调用 search_memory 查本地记忆，确认身份和背景后再回答或决定是否需要其他工具。"
                    "也用于：对话中提到'上次''之前说过''你记得吗''叫什么来着'等需要回忆旧事的语境。"
                    "不要用于：天气、股票、时间、新闻、网页搜索、纯闲聊、陈述句、陌生人的八卦。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "搜索关键词。用简短关键词组合，优先人名/昵称/群名/地点/事件标签，"
                                "不要用完整问句。底层是关键字匹配，关键词越具体结果越准。"
                                "示例：'示例用户丁'、'示例别名丁'、'示例用户丙'、'示例交流群'、'示例社区'、"
                                "'3D打印材料'、'示例用户丁 上海'。"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "历史原文返回片段数量，默认 5，范围 1-20。wiki 摘要不受此参数影响。",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                func=_search_memory_adapter,
            )
            if _history_available is not None and _history_available():
                _logger.info("[MemoryTool] search_memory 已注册（wiki + 历史原文）")
            else:
                _logger.info("[MemoryTool] search_memory 已注册（仅 wiki，历史索引未就绪）")

        # 读取 Self-Refine prompt 文件
        _prompt_dir = Path(__file__).parent.parent.parent / "prompts"
        self._feedback_prompt = (_prompt_dir / "feedback.md").read_text(encoding="utf-8")
        self._fact_check_prompt = (_prompt_dir / "fact_check.md").read_text(encoding="utf-8")
        self._fact_verify_prompt = (_prompt_dir / "fact_verify.md").read_text(encoding="utf-8")
        self._iterate_prompt = (_prompt_dir / "iterate.md").read_text(encoding="utf-8")

        # Self-Refine 观测字段
        self.last_self_refine_applied: bool = False
        self.last_feedback_decision: str = ""
        self.last_feedback_issues: List[str] = []
        self.last_iterate_count: int = 0
        self.last_feedback_raw: str = ""
        self.last_iterate_raw: str = ""
        self.last_iterate_skipped_no_budget: bool = False
        self._fact_check_client: Optional[Any] = None
        self.enable_fact_check = os.environ.get("ENABLE_FACT_CHECK", "1").lower() in ("1", "true", "yes", "on")

        # 开关（环境变量，默认启用）
        self.enable_react_tools = os.environ.get("ENABLE_REACT_TOOLS", "1").lower() in ("1", "true", "yes", "on")
        self.enable_self_refine = os.environ.get("ENABLE_SELF_REFINE", "1").lower() in ("1", "true", "yes", "on")
        if self.enable_self_refine:
            self.enable_react_tools = True

    def text_for_logging(self, text: str) -> str:
        """日志中只保留 few-shot ID，避免复制历史聊天正文。"""
        if not self.last_few_shot_content or self.last_few_shot_content not in text:
            return text
        marker = f"【persona few-shot 正文已省略；ids={','.join(self.last_few_shot_ids)}】"
        return text.replace(self.last_few_shot_content, marker)

    def messages_for_logging(self, messages: List[Dict]) -> List[Dict]:
        redacted = []
        for message in messages:
            item = dict(message)
            if isinstance(item.get("content"), str):
                item["content"] = self.text_for_logging(item["content"])
            redacted.append(item)
        return redacted

    def _submit_to_judge(self, tick_id: int, replies: List[str], unreplied: List[ChatMessage], all_messages: List[ChatMessage], is_group: bool):
        """把当前 tick 的数据提交给 JudgeWorker 异步判定"""
        import json
        try:
            worker = self.judge_worker
            if worker is None:
                return
            tick_data = {
                "tick_id": tick_id,
                "chat_name": unreplied[-1].chat_name if unreplied else "",
                "session_input_messages": [
                    {
                        "sender": m.sender,
                        "sender_type": m.sender_type.value,
                        "text": m.text or "",
                        "chat_name": m.chat_name,
                    }
                    for m in all_messages
                ],
                "bot_reply_text": " | ".join(replies) if replies else "",
                "reply_text": " | ".join(replies) if replies else "",
                "tool_calls": self.last_tool_calls,
                "memory_injected": self.text_for_logging(self.last_user_prompt),
                "full_user_prompt": self.text_for_logging(self.last_user_prompt),
                "reply_raw_response": self.last_raw_response,
                "reply_generation_trace": self.last_generation_trace,
                "full_system_prompt": self.last_system_prompt,
                "full_tools_context": self.last_tools_context,
                "full_llm_messages": self.messages_for_logging(self.last_llm_messages),
                "created_at": __import__('datetime').datetime.now().isoformat(),
                "tool_results_json": json.dumps(
                    [{"tool": t.get("tool_name", ""), "args": t.get("arguments", ""), "result": str(t.get("result_preview", ""))}
                     for t in (self.last_tool_calls or [])], ensure_ascii=False
                ),
            }
            worker.submit(tick_data)
        except Exception as e:
            _logger.debug("JudgeWorker submit failed: %s", e)

    def _self_refine(
        self,
        messages: List[Dict],
        deadline: float,
    ) -> Tuple[str, Optional[List[str]], List[Dict], str]:
        """Feedback 阶段。返回 decision, issues, updated_messages, raw_text。"""
        feedback_messages = messages + [
            {"role": "user", "content": self._feedback_prompt}
        ]
        source_text = next(
            (str(m.get("content", "")) for m in messages
             if m.get("role") == "user" and "<request>" in str(m.get("content", ""))),
            "",
        )
        evidence_parts = []
        for tag in ("session", "context", "history", "unread"):
            match = re.search(rf"<{tag}>.*?</{tag}>", source_text, re.DOTALL)
            if match:
                evidence_parts.append(match.group(0))
        tool_results = [
            str(m.get("content", "")) for m in messages if m.get("role") == "tool"
        ]
        if tool_results:
            evidence_parts.append("<tool_results>\n" + "\n".join(tool_results) + "\n</tool_results>")
        if not evidence_parts and source_text:
            evidence_parts.append(source_text[-6000:])
        # 证据审计前，按结构角色剔除 <history> 中 Bot 自己的历史回复，
        # 防止 Bot 用自己以前说过的话自证。tool/memory 证据保留。
        evidence_text = "\n\n".join(strip_assistant_history_lines(p) for p in evidence_parts)
        final_reply = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "assistant"),
            "",
        )
        audit_messages = [
            {"role": "system", "content": self._feedback_prompt},
            {"role": "user", "content": evidence_text},
            {"role": "assistant", "content": final_reply},
            {"role": "user", "content": "现在执行回复质量审计，只输出规定的 JSON。"},
        ]
        fact_issues, fact_raw = self._fact_check(evidence_text, final_reply, deadline)
        if fact_issues:
            text = json.dumps(
                {"decision": "fail", "issues": fact_issues, "fact_check": self._extract_json(fact_raw) or {}},
                ensure_ascii=False,
            )
            self.last_feedback_raw = text
            feedback_messages.append({"role": "assistant", "content": text})
            return "fail", fact_issues, feedback_messages, text
        timeout = max(1.0, deadline - time.time())
        try:
            raw = self.llm_client.chat(
                messages=audit_messages,
                max_tokens=300,
                temperature=0.3,
                timeout=timeout,
                response_format={"type": "json_object"},
                raise_on_error=True,
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        except Exception as e:
            _logger.warning("[SelfRefine] Feedback 调用失败: %s", e)
            text = ""
        self.last_feedback_raw = text or ""

        # 空回复 → 调用失败，返回 error 而非 pass（避免管理端显示"Self-Refine: pass"误导）
        if not text or not text.strip():
            return "error", None, feedback_messages, ""

        # 把 Feedback 阶段的 assistant 回复也追加到消息流，便于后续按顺序展示
        feedback_messages.append({"role": "assistant", "content": text})

        data = self._extract_json(text)
        if not isinstance(data, dict):
            return "error", None, feedback_messages, text

        decision = data.get("decision")
        issues = data.get("issues")
        if (
            decision not in ("pass", "fail")
            or not isinstance(issues, list)
            or not all(isinstance(issue, str) and issue.strip() for issue in issues)
            or (decision == "pass" and issues)
            or (decision == "fail" and not issues)
        ):
            return "error", None, feedback_messages, text
        if decision == "fail":
            return "fail", [issue.strip() for issue in issues], feedback_messages, text
        return "pass", None, feedback_messages, text

    def _fact_check(self, evidence: str, reply: str, deadline: float) -> Tuple[List[str], str]:
        """用独立窄任务检查事实问题；unknown 与 contradicted 同样拦截，nonfactual 不拦截。"""
        if not self.enable_fact_check or not evidence or not reply or deadline - time.time() < 2.0:
            return [], ""
        if self._fact_check_client is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
            if not api_key or not base_url:
                return [], ""
            try:
                from src.utils.qwen_client import QwenClient
                self._fact_check_client = QwenClient(
                    model=os.environ.get("FACT_CHECK_MODEL", "qwen3.6-flash"),
                    api_key=api_key,
                    base_url=base_url,
                )
            except Exception as e:
                _logger.warning("[FactCheck] 客户端初始化失败: %s", e)
                return [], ""
        messages = [
            {"role": "system", "content": self._fact_check_prompt},
            {"role": "user", "content": f"<evidence>\n{evidence}\n</evidence>\n\n<reply>\n{reply}\n</reply>"},
        ]
        try:
            raw = self._fact_check_client.chat(
                messages=messages,
                max_tokens=300,
                temperature=0,
                timeout=min(8.0, max(1.0, deadline - time.time())),
                response_format={"type": "json_object"},
                raise_on_error=True,
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        except Exception as e:
            _logger.warning("[FactCheck] 调用失败: %s", e)
            return [], ""
        data = self._extract_json(text) or {}
        claims = data.get("claims", []) if isinstance(data, dict) else []
        # 通用证据策略：contradicted 与 unknown 均阻塞；nonfactual 不阻塞；
        # 所有 entailed 命题交给独立复核器在既有 deadline 内逐条重判（不做方向性过滤）。
        issues, verify_claims = _collect_fact_issues(claims)
        verify_raw = ""
        if verify_claims and deadline - time.time() >= 2.0:
            verify_messages = [
                {"role": "system", "content": self._fact_verify_prompt},
                {"role": "user", "content": (
                    f"<evidence>\n{evidence}\n</evidence>\n\n"
                    f"<claims>\n{json.dumps(verify_claims, ensure_ascii=False)}\n</claims>"
                )},
            ]
            try:
                verify_response = self._fact_check_client.chat(
                    messages=verify_messages,
                    max_tokens=300,
                    temperature=0,
                    timeout=min(8.0, max(1.0, deadline - time.time())),
                    response_format={"type": "json_object"},
                    raise_on_error=True,
                    max_retries=0,
                )
                verify_raw = verify_response if isinstance(verify_response, str) else getattr(
                    verify_response, "content", str(verify_response)
                )
                verify_data = self._extract_json(verify_raw) or {}
                verified = verify_data.get("claims", []) if isinstance(verify_data, dict) else []
                for claim in verified if isinstance(verified, list) else []:
                    if not isinstance(claim, dict):
                        continue
                    verdict = claim.get("verdict")
                    if verdict not in ("contradicted", "unknown"):
                        continue
                    claim_value = claim.get("claim")
                    reason_value = claim.get("reason", "")
                    if not isinstance(claim_value, str) or not claim_value.strip() or not isinstance(reason_value, str):
                        continue
                    claim_text = claim_value.strip()
                    reason = reason_value.strip()
                    issues.append(
                        f"事实矛盾：{claim_text}；{reason}"
                        if verdict == "contradicted"
                        else f"事实无依据：{claim_text}；{reason}"
                    )
            except Exception as e:
                _logger.warning("[FactCheck] 独立复核失败: %s", e)
        if issues:
            _logger.warning("[FactCheck] 命中明确矛盾: %s", issues)
        combined_raw = json.dumps(
            {"extract": self._extract_json(text) or {}, "verify": self._extract_json(verify_raw) or {}},
            ensure_ascii=False,
        )
        return issues, combined_raw

    def _iterate(
        self,
        messages: List[Dict],
        issues: List[str],
        deadline: float,
    ) -> Tuple[List[str], List[Dict], str]:
        """Iterate 阶段。返回 replies, updated_messages, raw_text。"""
        revision_prompt = self._iterate_prompt.replace(
            "{{issues_json}}",
            xml_escape(json.dumps(issues, ensure_ascii=False)),
        )
        iterate_messages = messages + [
            {"role": "user", "content": revision_prompt}
        ]
        timeout = max(1.0, deadline - time.time())
        try:
            raw = self.llm_client.chat(
                messages=iterate_messages,
                max_tokens=1000,
                temperature=0.7,
                timeout=timeout,
                raise_on_error=True,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        except Exception as e:
            _logger.warning("[SelfRefine] Iterate 调用失败: %s", e)
            text = ""
        self.last_iterate_raw = text or ""

        # 把 Iterate 阶段的 assistant 回复也追加到消息流
        iterate_messages.append({"role": "assistant", "content": text or ""})
        replies = self._parse_replies(text)
        return replies or [], iterate_messages, text or ""

    def _react_generate(
        self,
        messages: List[Dict],
        tools: List[Dict],
        deadline: float,
        chat_name: str = "",
    ) -> Tuple[List[str], List[Dict]]:
        """ReAct 生成阶段。返回 replies, final_messages。"""
        llm_calls: List[Dict] = []
        tool_calls: List[Dict] = []
        trace: List[Dict] = []
        max_retries = 2
        tool_round_count = 0
        tool_call_count = 0

        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                while True:
                    # 剩余时间不足，直接结束
                    remaining = deadline - time.time()
                    if remaining < 1.0:
                        _logger.warning("[ReactGenerate] 剩余时间不足，结束生成")
                        self.last_generation_failed = True
                        self.last_llm_calls = llm_calls
                        self.last_tool_calls = tool_calls
                        self.last_generation_trace.extend(trace)
                        return [], messages

                    elapsed = time.time() - start_time

                    # 为最终回复预留时间；工具上限按实际调用数计算。
                    force_no_tools = (
                        remaining <= FINAL_RESPONSE_RESERVE_SECONDS
                        or tool_call_count >= MAX_TOOL_CALLS
                    )

                    actual_tools = None if force_no_tools else (tools if tools else None)
                    llm_timeout = max(1.0, deadline - time.time())
                    _logger.info("[LLM] attempt=%d round=%d force_no_tools=%s tools=%s timeout=%s msg_count=%d",
                                 attempt + 1, tool_round_count, force_no_tools, bool(actual_tools), llm_timeout, len(messages))
                    t_llm_start = time.time()
                    # force_no_tools 阶段强制 JSON 输出模式，避免 LLM 输出非法 JSON
                    response_format = {"type": "json_object"} if force_no_tools else None
                    raw = self.llm_client.chat(
                        messages=messages, tools=actual_tools,
                        max_tokens=10000, timeout=llm_timeout,
                        response_format=response_format,
                    )
                    self.last_thinking = getattr(self.llm_client, "last_thinking", "") or ""
                    self.last_llm_messages = [dict(m) for m in messages]
                    t_llm_ms = (time.time() - t_llm_start) * 1000
                    raw_content = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
                    raw_tool_calls = getattr(raw, "tool_calls", None)
                    _logger.info("[LLM] attempt=%d round=%d 完成 耗时=%.0fms type=%s",
                                 attempt + 1, tool_round_count, t_llm_ms,
                                 "tool_calls" if raw_tool_calls else "text")

                    # 记录 LLM 调用（summary）
                    llm_calls.append({
                        "attempt": attempt + 1,
                        "elapsed": round(elapsed, 2),
                        "messages_count": len(messages),
                        "has_tools": bool(actual_tools),
                        "has_tool_calls": bool(raw_tool_calls),
                        "response_preview": raw_content[:500] if raw_content else "",
                    })

                    # 记录完整 trace（请求）
                    trace.append({
                        "round": len(trace) // 3 + 1,
                        "type": "llm_request",
                        "timestamp": time.time(),
                        "attempt": attempt + 1,
                        "messages": self._truncate_messages(messages),
                        "tools": actual_tools,
                        "force_no_tools": force_no_tools,
                    })

                    # 记录完整 trace（响应）
                    trace.append({
                        "round": len(trace) // 3 + 1,
                        "type": "llm_response",
                        "timestamp": time.time(),
                        "content": raw_content[:2000] if raw_content else "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in (raw_tool_calls or [])
                        ],
                    })

                    # 工具调用处理
                    if raw_tool_calls and not force_no_tools:
                        assistant_msg = {
                            "role": "assistant",
                            "content": raw.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in raw_tool_calls
                            ],
                        }
                        # DeepSeek thinking mode 需要回传 reasoning_content，否则 round 1 会空返
                        if hasattr(raw, "reasoning_content") and raw.reasoning_content:
                            assistant_msg["reasoning_content"] = raw.reasoning_content
                        elif self.last_thinking:
                            assistant_msg["reasoning_content"] = self.last_thinking
                        messages.append(assistant_msg)

                        for tc in raw_tool_calls:
                            tool_name = tc.function.name
                            tool_args = tc.function.arguments
                            _logger.info("[Tool] 执行开始: %s(%s)", tool_name, tool_args[:100] if isinstance(tool_args, str) else str(tool_args)[:100])
                            t_tool_start = time.time()
                            if tool_call_count >= MAX_TOOL_CALLS:
                                result = f"工具调用已达上限 {MAX_TOOL_CALLS} 次，未执行"
                            elif deadline - time.time() <= FINAL_RESPONSE_RESERVE_SECONDS:
                                result = "剩余时间不足，未执行工具，请直接生成最终回复"
                            elif self.tool_registry.has(tool_name):
                                tool_call_count += 1
                                result = self.tool_registry.get(tool_name).execute(tool_args)
                            else:
                                result = f"工具 {tool_name} 不存在"
                            t_tool_ms = (time.time() - t_tool_start) * 1000
                            _logger.info("[Tool] 执行完成: %s 耗时=%.0fms result_len=%d", tool_name, t_tool_ms, len(str(result)) if result else 0)

                            # 保存到 session memory（跨 tick 缓存）
                            try:
                                query_key = _extract_query_key(tool_name, tool_args)
                                self.session_memory.add_tool_result(chat_name, tool_name, query_key, str(result)[:1000])
                            except Exception as e:
                                _logger.warning("add tool result failed: %s", e)

                            # 记录工具调用（summary）
                            tool_calls.append({
                                "attempt": attempt + 1,
                                "tool_call_id": tc.id,
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result_preview": str(result) if result else "",
                            })

                            # 记录完整 trace（工具执行）
                            trace.append({
                                "round": len(trace) // 3 + 1,
                                "type": "tool_execution",
                                "timestamp": time.time(),
                                "tool_call_id": tc.id,
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result": str(result)[:2000] if result else "",
                            })

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            })

                        # 完成一轮 tool 调用
                        tool_round_count += 1
                        continue

                    # 文本回复（或强制无 tools 后的回复）
                    text = raw_content
                    self.last_raw_response = text
                    # 把 assistant 的回复追加到消息流，便于按顺序展示
                    assistant_msg = {"role": "assistant", "content": text or ""}
                    if self.last_thinking:
                        assistant_msg["reasoning_content"] = self.last_thinking
                    messages.append(assistant_msg)
                    replies = self._parse_replies(text)
                    if replies:
                        _logger.info("[Generate] deepseek 直接生成 replies=%d 条", len(replies))
                        self.last_llm_calls = llm_calls
                        self.last_tool_calls = tool_calls
                        self.last_generation_trace.extend(trace)
                        return replies, messages

                    # 只有结构合法的 {"replies": []} 才表示明确不回复。
                    parsed = self._extract_json(text) if text else None
                    if isinstance(parsed, dict) and parsed.get("replies") == []:
                        _logger.info("[Generate] LLM 输出空 replies → 正确决策不回复")
                        self.last_llm_calls = llm_calls
                        self.last_tool_calls = tool_calls
                        self.last_generation_trace.extend(trace)
                        return [], messages

                    # 空回复处理（LLM 返回空字符串或无效内容）
                    if force_no_tools:
                        # 禁用 tools 后返回空，可能是 LLM 还在尝试调用工具
                        # 继续外层 retry，给 LLM 一次基于已有信息直接回复的机会
                        _logger.info("[Generate] force_no_tools 空回复，继续 retry")
                        self.last_raw_response = f"[空回复且已禁用tools，attempt={attempt+1}]"
                        break  # 跳出 while，进入下一次 retry

                    self.last_raw_response = f"[空回复，attempt={attempt+1}]"
                    break  # 跳出 while，进入下一次 retry

            except Exception as e:
                self.last_raw_response = f"[ERROR attempt={attempt+1}: {type(e).__name__}: {e}]"
                llm_calls.append({
                    "attempt": attempt + 1,
                    "error": f"{type(e).__name__}: {e}",
                })
                trace.append({
                    "type": "error",
                    "timestamp": time.time(),
                    "attempt": attempt + 1,
                    "error": f"{type(e).__name__}: {e}",
                })
                if attempt < max_retries:
                    time.sleep(1)

        _logger.info("[Generate] generate 最终返回空 ( exhausted retries )")
        self.last_generation_failed = True
        self.last_llm_calls = llm_calls
        self.last_tool_calls = tool_calls
        self.last_generation_trace.extend(trace)
        return [], messages

    def generate(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                 is_group: bool = False, tick_id: int = 0,
                 enable_time_awareness: Optional[bool] = None,
                 enable_reply_restraint: Optional[bool] = None,
                 enable_unread_dedup: Optional[bool] = None,
                 enable_timestamps: Optional[bool] = None) -> List[str]:
        """
        生成回复内容，返回多条回复列表（最多3条）。
        支持 ReAct 多轮工具调用，总超时预算 20s；启用 Self-Refine 时生成后追加 Feedback + Iterate。
        """
        t_generate_start = time.time()
        self.last_generation_failed = False
        if not unreplied:
            self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
            return []

        if self.llm_client is None:
            self.last_generation_failed = True
            self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
            return []

        # 重置 debug 状态
        self.last_system_prompt = ""
        self.last_tools_context = ""
        self.last_user_prompt = ""
        self.last_raw_response = ""
        self.last_llm_calls = []
        self.last_tool_calls = []
        self.last_generation_trace = []
        self.last_loaded_skills = []
        self.last_skill_injected_content = ""
        self.last_few_shot_ids = []
        self.last_few_shot_content = ""
        self.last_llm_messages = []
        self.last_self_refine_applied = False
        self.last_feedback_decision = ""
        self.last_feedback_issues = []
        self.last_iterate_count = 0
        self.last_feedback_raw = ""
        self.last_iterate_raw = ""
        self.last_iterate_skipped_no_budget = False

        chat_name = unreplied[-1].chat_name if unreplied else ""

        # 模型辅助路由：提前到 system prompt 之前，让 prompt 也能感知已加载 skill
        last_msg = unreplied[-1]
        route_text = last_msg.text or last_msg.image_description or ""
        # 取最近消息作为路由上下文（含历史消息 + 同批次其它未读），帮助路由理解对话语境
        n_unreplied = len(unreplied)
        prior = list(all_messages[:-n_unreplied]) + list(unreplied[:-1])
        context_msgs = [(m.sender, m.text) for m in prior if m.text][-3:]

        deadline = time.time() + 20.0
        t_route_start = time.time()
        matched_skills = self._route_skills(
            route_text,
            context_messages=context_msgs,
            is_group=is_group,
            deadline=deadline,
        )
        _logger.info(f"[Perf][Generate] skill routing={(time.time()-t_route_start)*1000:.0f}ms")

        # FORCE_SKILL 环境变量：跳过路由，强制注入指定 skill（用于隔离测试 skill 内容）
        force_skill = os.environ.get("FORCE_SKILL", "").strip()
        if force_skill:
            manifest_names = {s["name"] for s in self._load_skill_manifest()}
            if force_skill not in manifest_names:
                _logger.warning("[SkillRouter] FORCE_SKILL=%s 不在可用 skill 列表中，忽略", force_skill)
            else:
                matched_skills = [force_skill]
                _logger.warning("[SkillRouter] FORCE_SKILL=%s，跳过路由（注意：这是调试覆写，勿留生产环境）", force_skill)

        self.last_loaded_skills = matched_skills

        # 模型选择：固定使用主 LLM（带 tools），让 LLM 自己决定是否需要工具。
        _logger.info(f"[ModelSelect] matched_skills={matched_skills} → active_llm=deepseek（tools 可用）")
        self.last_active_llm = "deepseek"

        system_prompt = self._system_prompt(
            enable_reply_restraint=enable_reply_restraint,
            unreplied=unreplied,
            all_messages=all_messages,
            is_group=is_group,
        )

        tools_context = self._build_tools_context(chat_name)

        user_prompt = self._build_user_prompt(
            unreplied, all_messages, is_group,
            enable_time_awareness=enable_time_awareness,
            enable_unread_dedup=enable_unread_dedup,
            enable_timestamps=enable_timestamps,
            tools_context=tools_context,
        )

        # Skill 兜底：明确场景未命中时，群聊默认 group_banter，私聊默认 casual_chat
        if not matched_skills:
            matched_skills = ["group_banter"] if is_group else ["casual_chat"]
            _logger.info(f"[SkillRouter] 未命中明确 skill，兜底: {matched_skills}")

        # Skill 注入：主模型只接收路由选中的 skill 全文。
        skill_parts = []
        if matched_skills:
            for skill_name in matched_skills:
                content = self._load_skill_content(skill_name)
                if content:
                    skill_parts.append(
                        f'<active_skill name="{xml_escape(skill_name, quote=True)}">\n'
                        f"{content}\n"
                        "</active_skill>"
                    )
            if skill_parts:
                user_prompt = self._append_request_sections(user_prompt, skill_parts)
        self.last_skill_injected_content = "\n\n".join(skill_parts) if skill_parts else ""

        self.persona_few_shot_ready = (
            self.persona_few_shot_allow_unreviewed
            or self.persona_few_shot_retriever.is_approved()
        )
        if self.enable_persona_few_shots and self.persona_few_shot_ready and self.persona_few_shot_count:
            few_shot_rows = self.persona_few_shot_retriever.retrieve(
                query="\n".join([text for _, text in context_msgs] + [route_text]),
                chat_name=chat_name,
                is_group=is_group,
                limit=self.persona_few_shot_count,
            )
            few_shot_content, few_shot_ids = self.persona_few_shot_retriever.render(
                few_shot_rows,
                max_chars=self.persona_few_shot_max_chars,
            )
            if few_shot_content:
                user_prompt = self._append_request_sections(user_prompt, [few_shot_content])
                self.last_few_shot_content = few_shot_content
                self.last_few_shot_ids = few_shot_ids
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "persona_few_shot",
                    "timestamp": time.time(),
                    "ids": few_shot_ids,
                })
                _logger.info("[PersonaFewShot] chat=%s ids=%s", chat_name, few_shot_ids)

        self.last_system_prompt = system_prompt
        self.last_tools_context = tools_context

        self.last_user_prompt = user_prompt

        # 构建 messages：system（人设）+ user（上下文含缓存）
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_prompt})

        # 工具开关：关闭 ReAct 时不向 LLM 暴露 tools
        tools = self.tool_registry.to_openai_schemas() if self.enable_react_tools else []

        # SkillRouter + ReAct + Self-Refine 共用 20s 总预算
        t_react_start = time.time()
        replies, final_messages = self._react_generate(messages, tools, deadline, chat_name)
        _logger.info(f"[Perf][Generate] react_generate={(time.time()-t_react_start)*1000:.0f}ms replies={len(replies)}")

        # Self-Refine：Feedback + Iterate
        if self.enable_self_refine and replies and deadline - time.time() >= 1.0:
            decision, raw_issues, feedback_messages, _ = self._self_refine(final_messages, deadline)
            issues = raw_issues or []
            self.last_self_refine_applied = True
            self.last_feedback_decision = decision
            self.last_feedback_issues = issues or []
            # Feedback 阶段本身已经产生了一条 assistant 消息，统一用 feedback_messages 作为最终消息流
            final_messages = feedback_messages

            if decision == "fail" and issues:
                current_replies = replies
                current_messages = feedback_messages

                # 单次 Iterate（无循环：issues 不刷新时循环无意义）
                if deadline - time.time() >= 1.0:
                    new_replies, new_messages, _ = self._iterate(current_messages, issues, deadline)
                    if new_replies:
                        current_replies = new_replies
                        current_messages = new_messages
                    else:
                        _logger.warning("[SelfRefine] Iterate 未生成有效回复，保留原回复")
                    self.last_iterate_count = 1
                else:
                    self.last_iterate_skipped_no_budget = True
                    _logger.warning("[SelfRefine] Iterate 因预算不足跳过 (deadline=%.1fs)",
                                    deadline - time.time())

                replies = current_replies
                final_messages = current_messages

        # 更新最终观测字段
        self.last_llm_messages = [dict(m) for m in final_messages]
        # last_raw_response 保留原始 generation 的 raw_response，self-refine 的原文单独存
        # 在 last_feedback_raw / last_iterate_raw 中

        t_total_ms = (time.time() - t_generate_start) * 1000
        _logger.info("[Perf][Generate] total=%.0fms replies=%d self_refine=%s decision=%s iterate=%d",
                     t_total_ms, len(replies), self.last_self_refine_applied,
                     self.last_feedback_decision, self.last_iterate_count)
        self._submit_to_judge(tick_id, replies, unreplied, all_messages, is_group)
        return replies

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """截断 OpenAI message 数组，防止 debug JSON 过大。

        注意：此函数处理的是 List[Dict] 结构（含 content + tool_calls），
        与通用 text_utils._truncate_text（处理 str）不同，因此保留独立实现。
        阈值设高些，markdown 会单独保存完整版。
        """
        truncated = []
        for m in messages:
            cm = dict(m)
            if "content" in cm and isinstance(cm["content"], str) and len(cm["content"]) > 10000:
                cm["content"] = cm["content"][:10000] + "\n\n... [truncated, see markdown for full content]"
            if "tool_calls" in cm:
                cm["tool_calls"] = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name")} for tc in cm["tool_calls"]]
            truncated.append(cm)
        return truncated

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从 LLM 回复中提取 JSON 对象。委托给 shared utility。"""
        from src.utils.json_extractor import extract_json
        return extract_json(text)

    def _parse_replies(self, text: str) -> List[str]:
        """解析 LLM 回复：{"replies": ["msg1", "msg2"]}。prompt 已要求此格式。"""
        if not text or not text.strip():
            return []
        data = self._extract_json(text)
        if data is not None:
            replies = data.get("replies")
            if not isinstance(replies, list) or not all(isinstance(r, str) for r in replies):
                return []
            return [r.strip() for r in replies if r.strip() not in ("收到", "好的", "嗯", "OK", "1")][:3]
        # JSON 解析失败。如果文本含 { 或 "replies"，说明 LLM 尝试输出 JSON 但格式无效，
        # 不做文本切分回退（防止不完整 JSON 串泄漏给用户），由上层按"不回复"处理。
        stripped = text.strip()
        if '{' in stripped or '"replies"' in stripped:
            return []
        # fallback: 按段落拆分（仅对纯文本生效，不含 JSON 特征）
        text = text.strip()
        for sep in ("\n\n", "\n"):
            parts = [p.strip().replace("\n", " ") for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                return [p for p in parts if p not in ("收到", "好的", "嗯", "OK", "1")][:3]
        return [text.replace("\n", " ")] if text not in ("收到", "好的", "嗯", "OK", "1") else []

    def _load_skill_manifest(self) -> List[Dict[str, str]]:
        """扫描 skills/ 目录，返回技能清单（name + trigger 描述），不含正文。"""
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            return []
        manifest = []
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            md_file = skill_dir / "SKILL.md"
            if md_file.exists():
                text = md_file.read_text(encoding="utf-8")
                summary_match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
                if summary_match:
                    description = re.sub(r"\s+", " ", summary_match.group(1)).strip()
                else:
                    description = text.strip().split("\n")[0].replace("#", "").strip()
                manifest.append({
                    "name": skill_dir.name,
                    "description": description,
                })
        return manifest

    def _load_skill_content(self, skill_name: str) -> str:
        """加载指定 skill 的完整 SKILL.md 内容。"""
        md_file = Path(__file__).parent.parent.parent / "skills" / skill_name / "SKILL.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8").strip()
        return ""

    def _route_skills(self, user_text: str, context_messages: Optional[List[Tuple[str, str]]] = None,
                      is_group: bool = False, deadline: Optional[float] = None) -> List[str]:
        """模型辅助路由：根据用户消息判断需要加载哪个 skill。

        返回列表是为了兼容旧逻辑，但内部强制单选：
        - 提示词要求 LLM 只输出一个最匹配的 skill
        - 如果 LLM 返回多个，按 _SKILL_PRIORITY 取优先级最高者
        """
        if not user_text or not self.llm_client:
            return []

        manifest = self._load_skill_manifest()
        if not manifest:
            return []

        # 构建轻量、结构化路由 prompt；具体触发语义来自各 skill 的 summary。
        skill_list = "\n".join(
            f'<skill name="{xml_escape(s["name"], quote=True)}">'
            f'{xml_escape(s["description"])}'
            f'</skill>'
            for s in manifest
        )

        chat_type = "群聊" if is_group else "私聊"

        # 上下文块
        context_block = ""
        if context_messages:
            lines = "\n".join(
                f'<message sender="{xml_escape(str(s), quote=True)}">{xml_escape(str(t))}</message>'
                for s, t in context_messages
            )
            context_block = f"<recent_messages>\n{lines}\n</recent_messages>\n"

        priority_hint = " > ".join(_SKILL_PRIORITY)

        router_prompt = (
            '<skill_routing_request>\n'
            '  <task>从 available_skills 中选择最匹配当前消息的一个 skill。</task>\n'
            f'  <conversation type="{xml_escape(chat_type, quote=True)}">\n'
            f'{context_block}'
            f'    <current_message>{xml_escape(user_text)}</current_message>\n'
            '  </conversation>\n'
            '  <available_skills>\n'
            f'{skill_list}\n'
            '  </available_skills>\n'
            f'  <tie_break_order>{xml_escape(priority_hint)}</tie_break_order>\n'
            '  <selection_rules>\n'
            '    <rule>只选择语义最匹配的一个 skill；不要仅凭单个关键词判断。</rule>\n'
            '    <rule>结合聊天类型、当前消息和最近对话判断完整意图。</rule>\n'
            '    <rule>没有匹配项时返回空数组。</rule>\n'
            '  </selection_rules>\n'
            '  <output_schema>{"skills":["skill_name"]}</output_schema>\n'
            '</skill_routing_request>'
        )

        try:
            # 记录路由请求到 trace
            if hasattr(self, 'last_generation_trace') and isinstance(self.last_generation_trace, list):
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "skill_router_request",
                    "timestamp": time.time(),
                    "messages": [{"role": "user", "content": router_prompt[:500] + "..." if len(router_prompt) > 500 else router_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 256,
                })
            raw = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是 SkillRouter。无论用户消息多复杂，你只输出 JSON，不要输出任何分析、解释或自然语言。"},
                    {"role": "user", "content": router_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                timeout=max(1.0, deadline - time.time()) if deadline is not None else 20.0,
                response_format={"type": "json_object"},
            )
            raw_str = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            # 记录路由响应到 trace
            raw_preview = (raw_str[:200]) if raw_str else ""
            if hasattr(self, 'last_generation_trace') and isinstance(self.last_generation_trace, list):
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "skill_router_response",
                    "timestamp": time.time(),
                    "content": raw_preview,
                })
            # 尝试解析 JSON
            if raw_str:
                data = self._extract_json(raw_str)
                if isinstance(data, dict):
                    matched = data.get("skills", [])
                    if not isinstance(matched, list) or not all(isinstance(name, str) for name in matched):
                        return []
                    # 过滤有效 skill
                    valid = {s["name"] for s in manifest}
                    result = [name for name in matched if name in valid]
                    # 强制单选：如果返回多个，按优先级取最高者
                    if len(result) > 1:
                        priority_map = {name: idx for idx, name in enumerate(_SKILL_PRIORITY)}
                        chosen = min(result, key=lambda x: priority_map.get(x, 9999))
                        _logger.info(
                            f"[SkillRouter] 用户消息: {user_text[:30]}... -> 多技能 {result}，按优先级选定: {chosen}"
                        )
                        result = [chosen]
                    _logger.info(f"[SkillRouter] 用户消息: {user_text[:30]}... -> 匹配技能: {result}")
                    return result
                else:
                    _logger.warning(f"[SkillRouter] 用户消息: {user_text[:30]}... -> 未找到 JSON，原始响应: {raw_str[:100]}")
        except Exception as e:
            _logger.warning(f"[SkillRouter] 路由异常: {type(e).__name__}: {e}")
            if hasattr(self, 'last_generation_trace') and isinstance(self.last_generation_trace, list):
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "skill_router_error",
                    "timestamp": time.time(),
                    "error": f"{type(e).__name__}: {e}",
                })
        return []

    def _system_prompt(self, enable_reply_restraint: Optional[bool] = None,
                       unreplied: Optional[List[ChatMessage]] = None,
                       all_messages: Optional[List[ChatMessage]] = None,
                       is_group: bool = False) -> str:
        """核心 system prompt：读取稳定的人设、策略和输出 schema。"""
        if enable_reply_restraint is None:
            enable_reply_restraint = self.enable_reply_restraint

        project_root = Path(__file__).parent.parent.parent
        configured_path = os.environ.get("PERSONA_PATH", "data/persona.md")
        prompt_path = Path(configured_path)
        if not prompt_path.is_absolute():
            prompt_path = project_root / prompt_path
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            # fallback 到默认 prompt
            prompt = "<persona>你是示例用户甲本人。用户不是在跟AI聊天，是在微信上给示例用户甲发消息。</persona>"

        return prompt

    def _build_tools_context(self, chat_name: str) -> str:
        """构建工具上下文：已缓存数据 + 工具结果提示。"""
        cache_lines = self.session_memory.get_cache_lines(chat_name, include_expired=True)
        if not cache_lines:
            return ""
        lines_local = ["已缓存数据（来自之前查询，无需重复调用）"]
        lines_local.extend(cache_lines)
        lines_local.append("")
        return "\n".join(lines_local)

    @staticmethod
    def _format_time_tag(ts: Optional[int], now_ts: float) -> str:
        """根据时间戳生成绝对时间标签 YYYY-MM-DD HH:MM。"""
        if not ts:
            return ""
        tm = time.localtime(int(ts))
        return f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d} {tm.tm_hour:02d}:{tm.tm_min:02d}"

    @staticmethod
    def _format_message_line(m: ChatMessage, enable_timestamps: bool = True) -> str:
        """将单条消息渲染为 prompt 中的一行文本，含时间戳。"""
        sender_name = "我" if m.sender_type == SenderType.SELF else m.sender
        msg_type = m.message_type or "text"
        now_ts = time.time()

        # 时间标签（优先 create_time int，fallback timestamp str）
        time_tag = ""
        if enable_timestamps:
            ts = getattr(m, 'create_time', None)
            if not ts:
                ts_str = getattr(m, 'timestamp', None)
                if ts_str:
                    try:
                        from datetime import datetime
                        ts = int(datetime.strptime(ts_str.replace('  ', ' '), "%Y-%m-%d %H:%M:%S").timestamp())
                    except Exception as e:
                        _logger.warning("[Generator] 时间戳解析失败: %s (raw=%r)", e, ts_str)
            time_tag = ReplyGenerator._format_time_tag(ts, now_ts) if ts else ""

        def _line(body: str) -> str:
            if time_tag:
                return f"{sender_name}（{time_tag}）：{body}"
            return f"{sender_name}：{body}"

        if msg_type == "image":
            desc = m.image_description or "图片"
            text_part = m.image_text or m.text or ""
            if text_part:
                return _line(f"[图片] {desc}（图上文字：{text_part}）")
            return _line(f"[图片] {desc}")

        elif msg_type == "sticker":
            desc = m.image_description or "表情包"
            text_part = m.image_text or m.text or ""
            if text_part:
                return _line(f"[表情包] {desc}（配字：{text_part}）")
            return _line(f"[表情包] {desc}")

        elif msg_type == "mixed":
            desc = m.image_description or ""
            text_part = m.text or ""
            if desc:
                return _line(f"[图片+文字] {text_part} | 图片描述：{desc}")
            return _line(f"[图片+文字] {text_part}")

        elif msg_type == "link_card":
            desc = m.image_description or "链接卡片"
            return _line(f"[链接卡片] {desc}")

        elif msg_type == "video":
            desc = m.image_description or "视频"
            text_part = m.image_text or m.text or ""
            if text_part:
                return _line(f"[视频] {desc}（视频文字：{text_part}）")
            return _line(f"[视频] {desc}")

        else:
            body = m.text or ""
            if m.quoted_text:
                body = f"「引用：{m.quoted_text}」{body}"
            return _line(body)

    @staticmethod
    def _append_request_sections(prompt: str, sections: List[str]) -> str:
        """把可信指令/示例插在最终 reply_guard 前，否则追加到 request 末尾。"""
        closing = "</request>"
        if not sections or not prompt.rstrip().endswith(closing):
            return prompt
        body = prompt.rstrip()[:-len(closing)].rstrip()
        guard_start = body.rfind('<reply_guard priority="final" enforcement="hard">')
        insertion = "\n\n".join(sections)
        if guard_start >= 0:
            prefix = body[:guard_start].rstrip()
            guard = body[guard_start:]
            return prefix + "\n\n" + insertion + "\n\n" + guard + "\n" + closing
        return body + "\n\n" + insertion + "\n" + closing

    @staticmethod
    def _message_element(m: ChatMessage, index: int, enable_timestamps: bool,
                         handled: Optional[bool] = None) -> str:
        role = "self" if m.sender_type == SenderType.SELF else "other"
        attrs = [
            f'index="{index}"',
            f'role="{role}"',
            f'type="{xml_escape(m.message_type or "text", quote=True)}"',
        ]
        if handled is not None:
            attrs.append(f'handled="{str(handled).lower()}"')
        content = xml_escape(ReplyGenerator._format_message_line(m, enable_timestamps))
        return f"<message {' '.join(attrs)}>{content}</message>"

    def _build_user_prompt(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                           is_group: bool = False,
                           enable_time_awareness: Optional[bool] = None,
                           enable_unread_dedup: Optional[bool] = None,
                           enable_timestamps: Optional[bool] = None,
                           tools_context: str = "") -> str:
        """构建边界明确的动态请求：会话、记忆、缓存、历史和未读。"""
        if enable_time_awareness is None:
            enable_time_awareness = self.enable_time_awareness
        if enable_unread_dedup is None:
            enable_unread_dedup = self.enable_unread_dedup
        if enable_timestamps is None:
            enable_timestamps = self.enable_timestamps

        from datetime import datetime
        chat_name = unreplied[-1].chat_name if unreplied else ""
        is_at = any(getattr(m, "is_at_me", False) for m in unreplied)
        now_dt = datetime.now()
        session_fields = [
            f"<chat_name>{xml_escape(chat_name)}</chat_name>",
            f"<chat_type>{'group' if is_group else 'private'}</chat_type>",
            f"<mentioned>{str(is_at).lower()}</mentioned>",
        ]
        if enable_time_awareness:
            session_fields.append(f"<current_time>{now_dt.strftime('%Y-%m-%d %H:%M')}</current_time>")
            session_fields.append("<timestamp_semantics>消息时间是绝对时间，不代表消息刚刚发出。</timestamp_semantics>")

        sections = ["<session>\n" + "\n".join(session_fields) + "\n</session>"]
        context_fields: List[str] = []

        t_mem_ms = {}
        if self.memory_engine is not None and unreplied:
            if not is_group:
                last_sender = unreplied[-1].sender
                clean_sender = last_sender.split(" @")[0] if last_sender and " @" in last_sender else last_sender
                if clean_sender and clean_sender != "我":
                    t_m2 = time.time()
                    memory_text = self.memory_engine.get_user_memory(clean_sender, max_chars=6000)
                    t_mem_ms["other"] = (time.time() - t_m2) * 1000
                    if memory_text:
                        context_fields.append(
                            f'<participant_memory subject="{xml_escape(clean_sender, quote=True)}">'
                            f"{xml_escape(memory_text)}</participant_memory>"
                        )
            if is_group and chat_name:
                t_m3 = time.time()
                group_text = self.memory_engine.get_group_memory(chat_name, max_chars=6000)
                t_mem_ms["group"] = (time.time() - t_m3) * 1000
                if group_text:
                    context_fields.append(f"<group_memory>{xml_escape(group_text)}</group_memory>")
            _logger.info("[Perf][Memory] %s", " ".join(f"{k}={v:.0f}ms" for k, v in t_mem_ms.items()))

        if tools_context:
            context_fields.append(f"<tool_cache>{xml_escape(tools_context.strip())}</tool_cache>")
        if context_fields:
            sections.append("<context>\n" + "\n".join(context_fields) + "\n</context>")

        # all_messages 的尾部是本批未读；从 history 中排除，避免同一输入出现两次。
        history_messages = list(all_messages[:-len(unreplied)]) if unreplied else list(all_messages)
        if history_messages:
            def _msg_ts(m: ChatMessage) -> float:
                if m.sender_type == SenderType.SELF and m.reply_time:
                    return m.reply_time
                if m.create_time:
                    return float(m.create_time)
                return time.time()

            cutoff_ts = time.time() - 1800
            recent_50 = history_messages[-50:]
            recent_30min = [m for m in history_messages if _msg_ts(m) >= cutoff_ts]
            union_ids = {id(m) for m in recent_50} | {id(m) for m in recent_30min}
            for m in [m for m in history_messages if m.sender_type == SenderType.SELF][-5:]:
                union_ids.add(id(m))
            candidate = [m for m in history_messages if id(m) in union_ids]
            recent = candidate[-80:]
            history_items = [
                self._message_element(m, i, enable_timestamps)
                for i, m in enumerate(recent, 1)
                if m.sender_type != SenderType.SELF
            ]
            if history_items:
                sections.append("<history>\n" + "\n".join(history_items) + "\n</history>")

            consumed_self_items = [
                self._message_element(m, i, enable_timestamps)
                for i, m in enumerate(recent, 1)
                if m.sender_type == SenderType.SELF
            ]
        else:
            consumed_self_items = []

        unread_items = []
        for i, m in enumerate(unreplied, 1):
            ts = getattr(m, "create_time", None)
            already_handled = bool(
                enable_unread_dedup and ts and any(
                    hm.sender_type == SenderType.SELF
                    and hm.reply_time
                    and hm.reply_time > ts
                    for hm in history_messages
                )
            )
            unread_items.append(self._message_element(m, i, enable_timestamps, already_handled))
        sections.append("<unread>\n" + "\n".join(unread_items) + "\n</unread>")

        if consumed_self_items:
            sections.append(
                '<consumed_self_replies reuse="forbidden">\n'
                + "\n".join(consumed_self_items)
                + "\n</consumed_self_replies>"
            )

        guard_rules = [
            "<rule>unread 用错误姓名或错误身份称呼你时，必须按 identity_provocation 处理，并在回复中简短明确纠正一次；不得只反击而漏掉纠正。</rule>",
            "<rule>本轮主要语义是要求停止交流时，replies 必须为空。</rule>",
            "<rule>不得使用来源中没有的设备状态、职业、经历、关系或行为，不得攻击家人或性别。</rule>",
            "<rule>幽默新增内容只能是对 unread 措辞或表达行为的非事实评价、反转或文字游戏；若引入来源中没有的具体人物、物品、交易、事件或行为，必须丢弃。</rule>",
        ]
        if consumed_self_items:
            guard_rules.extend([
                "<rule>consumed_self_replies 是你已经发过的话，只用于排除素材，不用于模仿或延续。</rule>",
                "<rule>候选回复只要依赖其中已用过的事实、数字、人物标签、虚构关系、处境、反差或包袱，就必须丢弃。</rule>",
                "<rule>不得围绕已用素材新增动作、状态或场景；已用素材不能成为回复的主语、宾语、修饰对象、背景或隐含前提。</rule>",
                "<rule>即使 unread 再次引用这些内容，也不得 callback；宁可字面回应、换全新角度或不回复。</rule>",
                "<rule>若 unread 除追问、复述或抱怨已用素材外没有独立的新内容，replies 必须为空。</rule>",
            ])
        sections.append(
            '<reply_guard priority="final" enforcement="hard">\n'
            + "\n".join(guard_rules)
            + "\n</reply_guard>"
        )

        return "<request>\n" + "\n\n".join(sections) + "\n</request>"
