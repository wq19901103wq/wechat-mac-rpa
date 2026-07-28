#!/usr/bin/env python3
"""L4 Reply Generator - 回复内容生成."""

import logging
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.base import MEDIA_MESSAGE_TYPES, ChatMessage, SenderType
from src.reply.few_shot import PersonaFewShotRetriever, _query_response_mode, resolve_relationship
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


# 简单规则：哪些消息明显不该提供 search_memory，避免 LLM 过度调用
_NON_MEMORY_QUERY_PATTERNS = [
    re.compile(r"天气|气温|温度|下雨|下雪"),
    re.compile(r"股票|股价|股市|大盘|涨了吗|跌了吗|涨跌"),
    re.compile(r"几点|现在时间|今天星期|几号"),
    re.compile(r"搜一下|搜索|新闻|百度|谷歌"),
    re.compile(r"讲个笑话|说个笑话"),
]
_QUESTION_MARKS = set("?？")
_QUESTION_WORDS = {"吗", "呢", "谁", "什么", "哪里", "哪儿", "怎么", "多少", "哪些",
                   "几点", "干嘛", "干什么", "做什么", "住哪", "在哪", "是谁", "有谁"}

# ReAct 工具调用上限
MAX_TOOL_CALLS = 10


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.environ.get(name, str(default)))))
    except ValueError:
        return default

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


def _should_offer_search_memory(text: str) -> bool:
    """基于简单规则判断是否为明显非记忆查询。

    仅用于在把 tools 传给 LLM 前临时隐藏 search_memory，减少过度调用。
    真正需要调用时（人物身份/关系/背景/属性/群成员），仍保留该工具。
    """
    if not text:
        return False
    lower = text.lower()
    for pat in _NON_MEMORY_QUERY_PATTERNS:
        if pat.search(lower):
            return False
    has_question = any(c in text for c in _QUESTION_MARKS)
    has_question_word = any(w in lower for w in _QUESTION_WORDS)
    # 短陈述句（无疑问词/问号）→ 不主动提供记忆搜索
    if len(text) <= 18 and not has_question and not has_question_word:
        return False
    return True


class ReplyGenerator:
    def __init__(self, llm_client=None, memory_engine=None,
                 tool_registry=None,
                 judge_worker=None,
                 enable_time_awareness: bool = True,
                 enable_reply_restraint: bool = True,
                 enable_unread_dedup: bool = True,
                 enable_timestamps: bool = True,
                 enable_mode_detection: Optional[bool] = None):
        self.llm_client = llm_client
        self.memory_engine = memory_engine
        self.tool_registry = tool_registry or get_registry()
        self.judge_worker = judge_worker
        self.enable_time_awareness = enable_time_awareness
        self.enable_reply_restraint = enable_reply_restraint
        self.enable_unread_dedup = enable_unread_dedup
        self.enable_timestamps = enable_timestamps
        if enable_mode_detection is None:
            enable_mode_detection = os.environ.get("WECHAT_MODE_DETECTION", "0").lower() in ("1", "true", "yes", "on")
        self.enable_mode_detection = enable_mode_detection
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
                    "只要消息涉及王芊现实生活中认识的人、所在的群、去过的地方、做过的事、工作/投资/家庭相关的人或公司，"
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
                                "示例：'王海'、'小海哥'、'王艺涵'、'ai开发小分队'、'外滩玺'、"
                                "'3D打印材料'、'王海 上海'。"
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
             if m.get("role") == "user" and "<history>" in str(m.get("content", ""))),
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
        final_reply = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "assistant"),
            "",
        )
        audit_messages = [
            {"role": "system", "content": self._feedback_prompt},
            {"role": "user", "content": "\n\n".join(evidence_parts)},
            {"role": "assistant", "content": final_reply},
            {"role": "user", "content": "现在执行证据审计，只输出规定的 JSON。"},
        ]
        fact_issues, fact_raw = self._fact_check("\n\n".join(evidence_parts), final_reply, deadline)
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

        data = self._extract_json(text) or {}
        if not isinstance(data, dict):
            return "pass", None, feedback_messages, text

        decision = data.get("decision", "pass")
        issues = data.get("issues", [])
        if decision == "fail" and isinstance(issues, list) and issues:
            return "fail", issues, feedback_messages, text
        return "pass", None, feedback_messages, text

    def _fact_check(self, evidence: str, reply: str, deadline: float) -> Tuple[List[str], str]:
        """用独立窄任务检查明确的事实矛盾；unknown 不拦截，避免误杀幽默。"""
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
        issues = []
        for claim in claims if isinstance(claims, list) else []:
            if not isinstance(claim, dict) or claim.get("verdict") != "contradicted":
                continue
            claim_text = str(claim.get("claim", "")).strip()
            reason = str(claim.get("reason", "")).strip()
            issues.append(f"事实矛盾：{claim_text}；{reason}" if reason else f"事实矛盾：{claim_text}")
        directional = re.compile(r"高|低|贵|便宜|涨|跌|买|卖|成功|失败|是|否|\d")
        verify_claims = [
            str(claim.get("claim", "")).strip()
            for claim in claims if isinstance(claim, dict)
            and claim.get("verdict") == "entailed"
            and directional.search(str(claim.get("claim", "")))
        ] if isinstance(claims, list) else []
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
                    verdict = claim.get("verdict") if isinstance(claim, dict) else ""
                    if verdict not in ("contradicted", "unknown"):
                        continue
                    claim_text = str(claim.get("claim", "")).strip()
                    reason = str(claim.get("reason", "")).strip()
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

    def _local_rule_check(self, replies: List[str], skills: List[str]) -> List[str]:
        """Feedback 超时/error 时的本地兜底规则检查。"""
        issues = []
        combined = "\n".join(replies)
        if "value_investing" in skills:
            has_target = "目标" in combined or "看到" in combined or "区间" in combined or "点位" in combined
            if not has_target:
                issues.append("投资分析缺少目标价区间")
            if "止损" not in combined:
                issues.append("投资分析缺少止损价")
            if "风险" not in combined:
                issues.append("投资分析缺少风险提示")
            has_disclaimer = "声明" in combined or "不构成" in combined or "Disclaimer" in combined or "投资建议" in combined
            if not has_disclaimer:
                issues.append("投资分析缺少免责声明")
        # 纯附和检测：回复只是情绪复读词，没有任何信息增量
        ECHO_ONLY = {"确实", "哈哈", "牛逼", "扎心", "对的", "是啊", "好的", "嗯", "OK", "收到", "1", "哦", "行", "666", "太强了", "太牛了", "牛啊", "厉害了", "真的假的"}
        if len(replies) == 1 and replies[0].strip() in ECHO_ONLY:
            issues.append("回复是纯附和词（无信息增量），请换个角度给新信息、追问、质疑或调侃")
        return issues

    def _iterate(
        self,
        messages: List[Dict],
        issues: List[str],
        deadline: float,
    ) -> Tuple[List[str], List[Dict], str]:
        """Iterate 阶段。返回 replies, updated_messages, raw_text。"""
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        iterate_messages = messages + [
            {"role": "user", "content": f"{self._iterate_prompt}\n\n反馈问题：\n{issues_text}"}
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
        overall_start_time = time.time()

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
                    total_elapsed = time.time() - overall_start_time

                    # 时间或工具轮次达到上限后强制模型输出 JSON
                    force_no_tools = (total_elapsed > (deadline - overall_start_time) or
                                      tool_round_count >= MAX_TOOL_CALLS)

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
                            if self.tool_registry.has(tool_name):
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

                    # LLM 明确输出了 {"replies": []} → 正确决策（不想回复），不 retry
                    if text and '"replies"' in text:
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

        matched_skills = self._route_skills(route_text, context_messages=context_msgs, is_group=is_group)

        # 队列模式检测：连续 3+ 条相同文本（非 bot 自身消息）且未命中具体 skill → 用 group_banter
        if not matched_skills:
            recent_texts = [
                m.text for m in all_messages[-5:]
                if m.text and m.sender_type != SenderType.SELF
            ]
            if len(recent_texts) >= 3 and len(set(recent_texts[-3:])) == 1:
                matched_skills.append("group_banter")
                _logger.info("[SkillRouter] 检测到队列模式（重复消息 %s），使用 group_banter", recent_texts[-1][:30])

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

        # Skill 注入：根据路由结果注入 Bot 的 skill
        skill_parts = []
        if matched_skills:
            for skill_name in matched_skills:
                content = self._load_skill_content(skill_name)
                if content:
                    skill_parts.append(f"【{skill_name} 技能指南】\n{content}")
            if skill_parts:
                user_prompt += "\n\n" + "\n\n".join(skill_parts)
        self.last_skill_injected_content = "\n\n".join(skill_parts) if skill_parts else ""

        self.persona_few_shot_ready = (
            self.persona_few_shot_allow_unreviewed
            or self.persona_few_shot_retriever.is_approved()
        )
        if self.enable_persona_few_shots and self.persona_few_shot_ready and self.persona_few_shot_count:
            relationship = None if is_group else resolve_relationship(
                chat_name,
                self.persona_wiki_dir,
                os.environ.get("PERSONA_NAME", "本人"),
            )
            few_shot_rows = self.persona_few_shot_retriever.retrieve(
                query="\n".join([text for _, text in context_msgs] + [route_text]),
                chat_name=chat_name,
                is_group=is_group,
                limit=self.persona_few_shot_count,
                relationship=relationship,
            )
            few_shot_content, few_shot_ids = self.persona_few_shot_retriever.render(
                few_shot_rows,
                max_chars=self.persona_few_shot_max_chars,
            )
            if few_shot_content:
                user_prompt += "\n\n" + few_shot_content
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

        # 追加生成阶段格式指令
        reply_format_path = Path(__file__).parent.parent.parent / "prompts" / "reply_format.txt"
        if reply_format_path.exists():
            user_prompt += "\n\n" + reply_format_path.read_text(encoding="utf-8")

        self.last_user_prompt = user_prompt

        # 构建 messages：system（人设）+ user（上下文含缓存）
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_prompt})

        # 工具开关：关闭 ReAct 时不向 LLM 暴露 tools
        tools = self.tool_registry.to_openai_schemas() if self.enable_react_tools else []

        # 生成阶段：ReAct 循环，总预算 30s
        deadline = time.time() + 30.0
        replies, final_messages = self._react_generate(messages, tools, deadline, chat_name)

        # Self-Refine：Feedback + Iterate
        if self.enable_self_refine and replies and deadline - time.time() >= 1.0:
            # 身份泄漏关键词预筛（零 LLM 成本，命中的直接走 iterate 不调 feedback）
            IDENTITY_BREACH_PATTERNS = ["我是AI", "我是机器人", "作为AI", "语言模型", "芊小微"]
            pre_issues = []
            for r in replies:
                for pat in IDENTITY_BREACH_PATTERNS:
                    if pat in r:
                        pre_issues.append(f"回复含身份泄漏关键词'{pat}'")
                        break
            # 本地硬性规则预筛：不依赖 LLM feedback，直接抓关键要素缺失
            local_issues = self._local_rule_check(replies, self.last_loaded_skills)
            if local_issues:
                _logger.warning("[SelfRefine] 本地规则预筛命中: %s", local_issues)
                pre_issues.extend(local_issues)
            if pre_issues:
                decision = "fail"
                issues = pre_issues
                feedback_messages = final_messages
            else:
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

            elif decision == "error":
                fallback_issues = self._local_rule_check(replies, self.last_loaded_skills)
                if fallback_issues:
                    _logger.warning("[SelfRefine] Feedback 调用失败，本地规则检查命中: %s", fallback_issues)
                    self.last_feedback_issues = fallback_issues
                    if deadline - time.time() >= 1.0:
                        new_replies, new_messages, _ = self._iterate(feedback_messages, fallback_issues, deadline)
                        if new_replies:
                            replies = new_replies
                            final_messages = new_messages
                        self.last_iterate_count = 1
                    else:
                        _logger.warning("[SelfRefine] 本地规则命中但预算不足，返回空回复")
                        replies = []

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
            replies = data.get("replies", [])
            return [str(r).strip() for r in replies if str(r).strip() not in ("收到", "好的", "嗯", "OK", "1")][:3]
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
                # 提取第一行标题作为描述
                first_line = text.strip().split("\n")[0].replace("#", "").strip()
                manifest.append({
                    "name": skill_dir.name,
                    "description": first_line,
                })
        return manifest

    def _load_skill_content(self, skill_name: str) -> str:
        """加载指定 skill 的完整 SKILL.md 内容。"""
        md_file = Path(__file__).parent.parent.parent / "skills" / skill_name / "SKILL.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8").strip()
        return ""

    def _route_skills(self, user_text: str, context_messages: Optional[List[Tuple[str, str]]] = None,
                      is_group: bool = False) -> List[str]:
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

        # 构建轻量路由 prompt
        skill_list = "\n".join(
            f"{i+1}. {s['name']}：{s['description']}"
            for i, s in enumerate(manifest)
        )

        chat_type = "群聊" if is_group else "私聊"

        # 上下文块
        context_block = ""
        if context_messages:
            lines = "\n".join(f"  \"{s}: {t}\"" for s, t in context_messages)
            context_block = f"最近对话：\n{lines}\n\n"

        priority_hint = " > ".join(_SKILL_PRIORITY)

        router_prompt = (
            "你是 SkillRouter，只负责判断用户消息需要哪一个技能。\n\n"
            f"可用技能：\n{skill_list}\n\n"
            f"当前是：{chat_type}\n"
            f"{context_block}"
            f"用户消息：\"{user_text}\"\n\n"
            "路由规则（按优先级从高到低，只选一个）：\n"
            f"{priority_hint}\n"
            "- 3d_print_automation：明确提到 3MF/3D 打印/打印机状态\n"
            "- tuya_smart_home：明确提到智能家居设备控制/场景\n"
            "- handling_vent：对方在发泄负面情绪（烦、累、气死、无语、崩溃）\n"
            "- handling_praise：对方在夸奖/称赞/表达羡慕\n"
            "- value_investing：用户明确要求分析具体股票/标的，或提到具体股票代码\n"
            "- answering_questions：对方在寻求信息/知识/事实（带问号或疑问词）\n"
            "- receiving_share：对方在发长消息/链接/图片/推荐/分享经历\n"
            "- group_banter：群聊中 @你、cue 你、排队祝贺/调侃、炫耀接梗；群里随口聊行情/板块但没有要求分析也走这里\n"
            "- casual_chat：私聊中日常问候、闲聊、无明确主题\n\n"
            "注意：\n"
            "- 只输出一个最匹配的 skill，不要返回多个\n"
            "- 群里有人炫耀/排队/接梗，优先 group_banter，不要走 answering_questions\n"
            "- 群里有人吐槽但被 cue 到且能接梗，优先 group_banter；私聊真诉苦才走 handling_vent\n"
            "- 股票问题走 value_investing，不要走 answering_questions\n"
            "- 智能家居/3D 打印等工具意图走对应工具 skill\n"
            "- 群里随口提到'中概''白酒''大盘'等行情/板块但没人要求分析，不要走 value_investing，走 group_banter\n\n"
            "请输出 JSON，只包含一个技能 name，不要其他内容：\n"
            '{"skills": ["skill_name"]}\n'
            "如果不需要任何技能，输出：{\"skills\": []}"
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
                if data is not None:
                    matched = data.get("skills", [])
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

    def _load_skill_one_liners(self) -> str:
        """加载所有 skill 的一句话摘要（始终放在 system prompt 中）。
        优先从 SKILL.md 的'一句话摘要'段落提取，没有则 fallback 到'触发条件'。"""
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            return ""
        parts = []
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            md_file = skill_dir / "SKILL.md"
            if md_file.exists():
                text = md_file.read_text(encoding="utf-8").strip()
                if text.strip():
                    name = skill_dir.name
                    lines = text.split("\n")
                    summary = ""
                    in_section = None
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("## 一句话摘要"):
                            in_section = "summary"
                            continue
                        if stripped.startswith("## 触发条件"):
                            in_section = "trigger"
                            continue
                        if in_section:
                            if stripped.startswith("##"):
                                if summary:
                                    break
                                in_section = None
                                continue
                            if not stripped:
                                continue
                            summary = stripped
                            if in_section == "summary":
                                break
                    if summary:
                        # 截断到 80 字以内，保证可读性
                        if len(summary) > 80:
                            summary = summary[:77] + "..."
                        parts.append(f"- {name}：{summary}")
        if parts:
            return "\n可用技能（系统会根据对话内容自动下发详细框架）：\n" + "\n".join(parts) + "\n"
        return ""

    def _system_prompt(self, enable_reply_restraint: Optional[bool] = None,
                       unreplied: Optional[List[ChatMessage]] = None,
                       all_messages: Optional[List[ChatMessage]] = None,
                       is_group: bool = False) -> str:
        """核心 system prompt：读 prompts/persona.md（XML 风格），注入工具描述 + skill 摘要。"""
        if enable_reply_restraint is None:
            enable_reply_restraint = self.enable_reply_restraint

        # 支持通过环境变量/参数切换 system prompt 版本
        if self.enable_mode_detection:
            prompt_file = "persona_mode_detection.md"
        else:
            prompt_file = "persona.md"
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_file
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            # fallback 到默认 prompt
            prompt = "<persona>你是王芊本人。用户不是在跟AI聊天，是在微信上给王芊发消息。</persona>"

        # 注入工具描述
        tools_desc = "\n".join(
            f"- {t.name}：{t.description}"
            for t in self.tool_registry._tools.values()
        )
        prompt = prompt.replace("{tools_description}", tools_desc)

        # 注入 skill 摘要
        skill_hint = self._load_skill_one_liners()
        prompt = prompt.replace("{skills_description}", skill_hint.strip() if skill_hint else "（无）")

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
    def _build_mode_hints(
        unreplied: List[ChatMessage],
        all_messages: List[ChatMessage],
        is_group: bool,
        hour: int,
    ) -> str:
        """基于当前消息和历史上下文，生成帮助 LLM 过五关的语境速查提示。"""
        hints: List[str] = []

        # 1. 读场子：从当前未读消息提取意图和情绪
        last_texts = " ".join(
            (m.text or m.image_text or "") for m in unreplied
        )

        celebration_signals = [
            "升职", "晋升", "涨薪", "加薪", "获奖", "通过了", "恭喜", "🎉",
            "喜提", "买房", "买车", "结婚", "订婚", "生娃", "上岸", "拿到",
            "offer", "签了", "中奖", "发财", "暴富", "脱单",
        ]
        complaint_signals = [
            "烦", "累", "倒霉", "亏了", "无语", "难受", "裂开", "崩溃",
            "什么鬼", "服了", "妈的", "烦死", "不想干了",
        ]
        question_signals = [
            "怎么办", "怎么", "吗", "呢", "请问", "谁知道", "推荐",
            "求", "有没有", "能不能", "为什么", "是啥", "是什么",
        ]
        confirmation_signals = [
            "确认", "收到", "麻烦", "核对", "登记", "预约", "扣款",
            "到账", "办理", "合同", "发票",
        ]

        if _query_response_mode(last_texts) == "sincere":
            hints.append("【场子】对方是真的受挫/难过，进入真诚模式：具体回应这件事和对方的感受，零调侃、零口号，不急着给建议")
        elif any(s in last_texts for s in celebration_signals):
            hints.append("【场子】对方刚分享好消息/喜事，优先考虑祝贺模式")
        elif any(s in last_texts for s in complaint_signals):
            hints.append("【场子】对方情绪负面，可能在吐槽/受挫，优先考虑安慰模式")
        elif any(s in last_texts for s in question_signals):
            hints.append("【场子】对方在提问/求助，优先考虑解答模式")
        elif any(s in last_texts for s in confirmation_signals):
            hints.append("【场子】对方在确认事务，优先考虑事务模式")
        elif unreplied and all(
            (m.message_type or "text") in MEDIA_MESSAGE_TYPES
            for m in unreplied
        ):
            hints.append("【场子】对方发了图片/表情包，先看图内容再决定接不接梗")

        # 1.5 记忆查询检测：对方在问 Bot 身边的人/旧事
        memory_query_signals = [
            "认识", "记得", "初中", "高中", "大学", "同学", "同事",
            "朋友", "亲戚", "表哥", "表姐", "表弟", "表妹", "堂哥",
            "堂姐", "堂弟", "堂妹", "舅舅", "姨妈", "姑姑", "叔叔",
            "伯伯", "爷爷", "奶奶", "外公", "外婆", "爸爸", "妈妈",
            "老公", "老婆", "媳妇", "男朋友", "女朋友", "兄弟", "姐妹",
            "谁", "哪人", "哪里的", "干什么的", "做什么工作",
        ]
        # 检测是否提到具体人名（2-4个字的常见中文名模式）
        import re
        # 简单人名检测：连续2-4个中文字符，且不在常见虚词列表中
        common_words = {"什么", "怎么", "为什么", "是不是", "有没有", "可以", "这个", "那个", "他们", "我们", "你们", "自己", "别人", "人家", "大家", "一起", "现在", "今天", "明天", "昨天", "上次", "下次", "一直", "总是", "真的", "假的", "当然", "可能", "应该", "大概", "好像", "似乎", "确实", "其实", "本来", "原来", "刚刚", "马上", "立刻", "终于", "已经", "曾经", "正在", "将要", "打算", "准备", "开始", "结束", "完成", "成功", "失败", "不错", "很好", "太差", "一般", "还行", "可以", "不行", "不能", "不会", "不要", "不用", "没空", "有事", "没事", "好的", "收到", "了解", "知道", "明白", "清楚", "确定", "确认", "同意", "反对", "支持", "拒绝", "接受", "放弃", "坚持", "努力", "加油", "注意", "小心", "当心", "放心", "担心", "害怕", "紧张", "开心", "高兴", "难过", "伤心", "生气", "愤怒", "惊讶", "意外", "失望", "满意", "舒服", "难受", "累", "困", "饿", "渴", "冷", "热", "疼", "痛", "痒", "酸", "麻", "胀", "晕", "傻", "笨", "聪明", "漂亮", "帅", "丑", "胖", "瘦", "高", "矮", "大", "小", "长", "短", "宽", "窄", "厚", "薄", "深", "浅", "远", "近", "快", "慢", "早", "晚", "新", "旧", "老", "年轻", "贵", "便宜", "多", "少", "满", "空", "干", "湿", "脏", "干净", "整齐", "乱", "安静", "吵", "亮", "暗", "黑", "白", "红", "黄", "蓝", "绿", "紫", "橙", "灰", "粉", "棕", "金", "银", "透明", "彩色", "单色", "漂亮", "美丽", "好看", "可爱", "帅", "酷", "棒", "厉害", "牛", "强", "弱", "好", "坏", "对", "错", "真", "假", "正", "反", "上", "下", "左", "右", "前", "后", "里", "外", "内", "中", "间", "边", "角", "头", "尾", "根", "底", "顶", "面", "背", "侧", "正", "反", "东", "西", "南", "北", "方向", "位置", "地方", "地区", "城市", "国家", "世界", "地球", "宇宙", "天空", "太阳", "月亮", "星星", "云", "风", "雨", "雪", "雷", "电", "雾", "霜", "露", "冰", "水", "火", "土", "木", "金", "石头", "沙子", "泥土", "灰尘", "空气", "氧气", "氢气", "氮气", "二氧化碳", "一氧化碳", "甲烷", "乙醇", "酒精", "汽油", "柴油", "煤油", "天然气", "石油", "煤炭", "矿石", "金属", "钢铁", "铜", "铁", "铝", "锌", "铅", "锡", "镍", "铬", "钛", "钨", "银", "金", "铂", "钻石", "宝石", "玉石", "珍珠", "玛瑙", "水晶", "玻璃", "塑料", "橡胶", "皮革", "木头", "竹子", "纸张", "布料", "丝绸", "棉花", "羊毛", "麻", "化纤", "尼龙", "涤纶", "腈纶", "维纶", "丙纶", "氯纶", "氨纶", "碳纤维", "玻璃纤维", "陶瓷", "瓷器", "陶器", "砖瓦", "水泥", "混凝土", "沥青", "油漆", "涂料", "染料", "颜料", "墨水", "油墨", "胶水", "胶带", "胶布", "橡皮", "橡皮筋", "绳子", "线", "铁丝", "钢丝", "铜丝", "铝丝", "钉子", "螺丝", "螺母", "螺栓", "垫圈", "弹簧", "轴承", "齿轮", "皮带", "链条", "轮子", "轮胎", "轮毂", "刹车", "油门", "离合器", "变速器", "发动机", "电动机", "发电机", "变压器", "电容器", "电阻器", "电感器", "二极管", "三极管", "集成电路", "芯片", "CPU", "GPU", "内存", "硬盘", "固态硬盘", "U盘", "光盘", "软盘", "磁带", "数据线", "充电线", "电源线", "网线", "光纤", "电缆", "电线", "电池", "电瓶", "充电宝", "充电器", "插头", "插座", "开关", "灯泡", "灯管", "灯带", "LED", "LCD", "OLED", "屏幕", "显示器", "电视", "投影仪", "摄像机", "相机", "手机", "电话", "电脑", "笔记本", "平板", "手表", "手环", "耳机", "音箱", "麦克风", "键盘", "鼠标", "触控板", "遥控器", "路由器", "交换机", "防火墙", "服务器", "主机", "客户端", "浏览器", "软件", "硬件", "系统", "程序", "代码", "脚本", "算法", "模型", "数据", "数据库", "表", "字段", "记录", "文件", "文件夹", "目录", "路径", "链接", "URL", "域名", "IP", "端口", "协议", "接口", "API", "SDK", "框架", "库", "包", "模块", "组件", "插件", "扩展", "主题", "模板", "样式", "布局", "设计", "架构", "结构", "模式", "规范", "标准", "文档", "说明", "注释", "日志", "报告", "图表", "图像", "照片", "视频", "音频", "音乐", "歌曲", "歌词", "乐器", "钢琴", "吉他", "小提琴", "大提琴", "二胡", "笛子", "箫", "唢呐", "琵琶", "古筝", "古琴", "鼓", "锣", "钹", "钟", "铃", "喇叭", "号角", "口琴", "手风琴", "电子琴", "合成器", "音箱", "功放", "调音台", "麦克风", "话筒", "耳机", "耳塞", "耳麦", "音响", "家庭影院", "KTV", "录音棚", "直播间", "舞台", "剧院", "电影院", "音乐厅", "美术馆", "博物馆", "图书馆", "书店", "学校", "大学", "中学", "小学", "幼儿园", "培训机构", "公司", "企业", "工厂", "车间", "仓库", "办公室", "会议室", "前台", "后台", "领导", "老板", "经理", "主管", "总监", "副总裁", "总裁", "CEO", "CTO", "CFO", "COO", "CMO", "CHO", "CIO", "CDO", "CSO", "CCO", "创始人", "合伙人", "股东", "董事", "监事", "员工", "同事", "下属", "上级", "客户", "用户", "顾客", "乘客", "游客", "观众", "读者", "听众", "粉丝", "关注者", "订阅者", "会员", "VIP", "嘉宾", "评委", "选手", "运动员", "教练", "裁判", "裁判", "解说", "记者", "编辑", "作者", "作家", "诗人", "画家", "音乐家", "演员", "导演", "制片人", "编剧", "摄影师", "设计师", "工程师", "科学家", "教授", "老师", "学生", "同学", "校友", "室友", "同桌", "班长", "团支书", "学习委员", "文艺委员", "体育委员", "生活委员", "劳动委员", "纪律委员", "组织委员", "宣传委员", "心理委员", "安全委员", "课代表", "组长", "队长", "社长", "会长", "主席", "部长", "干事", "志愿者", "义工", "慈善家", "捐赠者", "受助者", "患者", "医生", "护士", "药师", "技师", "护工", "保洁", "保安", "司机", "厨师", "服务员", "接待员", "销售员", "导购", "收银员", "会计", "出纳", "审计", "税务", "律师", "法官", "检察官", "警察", "军人", "消防员", "救护员", "救生员", "导游", "翻译", "秘书", "助理", "顾问", "咨询师", "分析师", "研究员", "调查员", "检测员", "质检员", "操作员", "管理员", "调度员", "快递员", "外卖员", "配送员", "搬运工", "装修工", "电工", "水工", "木工", "瓦工", "油漆工", "焊工", "车工", "铣工", "钳工", "铸工", "锻工", "模具工", "数控工", "编程员", "操作手", "驾驶员", "飞行员", "船长", "船员", "列车员", "乘务员", "空乘", "地勤", "塔台", "调度", "指挥", "控制", "监控", "巡检", "维护", "维修", "保养", "安装", "调试", "测试", "检验", "校准", "标定", "认证", "许可", "注册", "备案", "审批", "审核", "检查", "稽查", "督查", "巡视", "巡查", "巡逻", "守望", "守卫", "保卫", "守护", "监护", "看管", "看守", "监禁", "拘留", "逮捕", "关押", "服刑", "改造", "教育", "感化", "挽救", "救助", "救援", "急救", "抢救", "治疗", "康复", "护理", "照料", "照顾", "看护", "抚养", "养育", "培养", "教育", "教导", "指导", "辅导", "培训", "训练", "锻炼", "练习", "实习", "实践", "实验", "试验", "尝试", "探索", "研究", "钻研", "攻克", "突破", "创新", "发明", "发现", "发掘", "开发", "开拓", "拓展", "扩展", "扩张", "推广", "宣传", "传播", "流传", "传承", "继承", "发扬", "光大", "振兴", "复兴", "崛起", "腾飞", "飞跃", "跨越", "超越", "突破", "打破", "粉碎", "摧毁", "毁灭", "消灭", "消除", "清除", "清理", "整理", "整顿", "整治", "治理", "管理", "管制", "控制", "限制", "约束", "束缚", "捆绑", "束缚", "压迫", "压制", "镇压", "制止", "阻止", "阻挡", "阻拦", "拦截", "截获", "缴获", "没收", "征收", "征税", "收费", "罚款", "处罚", "惩罚", "制裁", "报复", "复仇", "反击", "反攻", "反抗", "抵抗", "抵御", "防御", "防守", "保卫", "保护", "维护", "维持", "保持", "保留", "保存", "储存", "存储", "收藏", "珍藏", "保管", "看守", "看护", "照料", "照顾", "伺候", "侍奉", "服侍", "服务", "效劳", "效力", "效忠", "忠诚", "忠心", "虔诚", "信仰", "信念", "信心", "信任", "信赖", "依靠", "依赖", "依附", "附着", "粘贴", "贴合", "紧靠", "靠近", "接近", "接触", "碰触", "碰撞", "撞击", "冲击", "打击", "攻击", "进攻", "侵犯", "侵略", "侵占", "占领", "占据", "占有", "拥有", "具有", "具备", "配备", "配置", "安置", "安排", "布置", "部署", "分布", "散布", "散播", "传播", "传染", "感染", "影响", "感动", "打动", "触动", "震撼", "震惊", "惊讶", "惊喜", "欢喜", "喜欢", "喜爱", "爱好", "热衷", "痴迷", "迷恋", "迷恋", "贪恋", "贪图", "图谋", "谋划", "策划", "筹划", "筹备", "准备", "预备", "预先", "提前", "趁早", "及时", "准时", "按时", "按期", "定期", "长期", "短期", "暂时", "临时", "永久", "永远", "永恒", "瞬间", "刹那", "片刻", "一会儿", "不久", "很快", "缓慢", "慢慢", "渐渐", "逐渐", "逐步", "逐步", "陆续", "连续", "持续", "不断", "不停", "不止", "不息", "不休", "不眠", "不休", "不懈", "不怠", "勤奋", "勤劳", "勤勉", "刻苦", "努力", "奋斗", "拼搏", "拼搏", "拼命", "拼死", "誓死", "决死", "决意", "决心", "决计", "决定", "决心", "决意", "决计", "决死", "拼死", "拼命", "拼搏", "奋斗", "努力", "刻苦", "勤奋", "勤劳", "勤勉", "不懈", "不怠", "不休", "不息", "不止", "不停", "不断", "持续", "连续", "陆续", "逐步", "逐渐", "渐渐", "慢慢", "缓慢", "很快", "不久", "一会儿", "片刻", "刹那", "瞬间", "永恒", "永远", "永久", "临时", "暂时", "短期", "长期", "定期", "按期", "按时", "准时", "及时", "趁早", "提前", "预先", "预备", "准备", "筹备", "筹划", "策划", "谋划", "图谋", "贪图", "贪恋", "迷恋", "痴迷", "热衷", "爱好", "喜爱", "喜欢", "欢喜", "惊喜", "惊讶", "震惊", "震撼", "触动", "打动", "感动", "影响", "感染", "传染", "传播", "散播", "散布", "分布", "部署", "布置", "安排", "安置", "配置", "配备", "具备", "具有", "拥有", "占有", "占据", "占领", "侵占", "侵略", "侵犯", "进攻", "攻击", "打击", "冲击", "撞击", "碰撞", "碰触", "接触", "接近", "靠近", "紧靠", "贴合", "粘贴", "附着", "依附", "依赖", "依靠", "信赖", "信任", "信心", "信念", "信仰", "虔诚", "忠心", "忠诚", "效忠", "效力", "效劳", "服务", "服侍", "侍奉", "伺候", "照顾", "照料", "看护", "看守", "保管", "珍藏", "收藏", "存储", "储存", "保存", "保留", "保持", "维持", "维护", "保护", "保卫", "防守", "防御", "抵御", "抵抗", "反抗", "反攻", "反击", "复仇", "报复", "制裁", "惩罚", "处罚", "罚款", "收费", "征税", "征收", "没收", "缴获", "截获", "拦截", "阻拦", "阻挡", "阻止", "制止", "镇压", "压制", "压迫", "束缚", "捆绑", "约束", "限制", "控制", "管制", "管理", "治理", "整治", "整顿", "整理", "清理", "清除", "消除", "消灭", "毁灭", "摧毁", "粉碎", "打破", "突破", "超越", "跨越", "飞跃", "腾飞", "崛起", "复兴", "振兴", "光大", "发扬", "继承", "传承", "流传", "传播", "宣传", "推广", "扩张", "扩展", "拓展", "开拓", "开发", "发掘", "发现", "发明", "创新", "突破", "攻克", "钻研", "研究", "探索", "尝试", "试验", "实验", "实践", "实习", "练习", "锻炼", "训练", "培训", "辅导", "指导", "教导", "教育", "培养", "养育", "抚养", "看护", "照料", "护理", "康复", "治疗", "抢救", "急救", "救援", "救助", "挽救", "感化", "教育", "改造", "服刑", "关押", "逮捕", "拘留", "监禁", "看守", "看管", "监护", "守护", "保卫", "守卫", "守望", "巡逻", "巡查", "巡视", "督查", "稽查", "检查", "审核", "审批", "备案", "注册", "许可", "认证", "标定", "校准", "检验", "测试", "调试", "安装", "保养", "维修", "维护", "巡检", "监控", "控制", "指挥", "调度", "塔台", "地勤", "空乘", "乘务员", "列车员", "船员", "船长", "飞行员", "驾驶员", "操作手", "编程员", "数控工", "模具工", "锻工", "铸工", "钳工", "铣工", "车工", "焊工", "油漆工", "瓦工", "木工", "水工", "电工", "装修工", "搬运工", "配送员", "外卖员", "快递员", "调度员", "管理员", "操作员", "检测员", "质检员", "调查员", "研究员", "分析师", "咨询师", "顾问", "助理", "秘书", "翻译", "导游", "救生员", "救护员", "消防员", "军人", "警察", "检察官", "法官", "律师", "税务", "审计", "出纳", "会计", "收银员", "导购", "销售员", "接待员", "服务员", "厨师", "司机", "保安", "保洁", "护工", "技师", "药师", "护士", "医生", "患者", "受助者", "捐赠者", "慈善家", "义工", "志愿者", "干事", "部长", "主席", "会长", "社长", "队长", "组长", "课代表", "安全委员", "心理委员", "宣传委员", "组织委员", "纪律委员", "劳动委员", "生活委员", "体育委员", "文艺委员", "学习委员", "团支书", "班长", "同桌", "室友", "校友", "同学", "学生", "老师", "教授", "科学家", "工程师", "设计师", "摄影师", "编剧", "制片人", "导演", "演员", "音乐家", "画家", "诗人", "作家", "作者", "编辑", "记者", "解说", "裁判", "教练", "运动员", "选手", "评委", "嘉宾", "VIP", "会员", "订阅者", "关注者", "粉丝", "听众", "读者", "观众", "游客", "乘客", "顾客", "用户", "客户", "下属", "上级", "同事", "员工", "监事", "董事", "股东", "合伙人", "创始人", "CCO", "CSO", "CDO", "CIO", "CHO", "CMO", "COO", "CFO", "CTO", "CEO", "总裁", "副总裁", "总监", "主管", "经理", "老板", "领导", "后台", "前台", "会议室", "办公室", "仓库", "车间", "工厂", "企业", "公司", "培训机构", "幼儿园", "小学", "中学", "大学", "学校", "书店", "图书馆", "博物馆", "美术馆", "音乐厅", "电影院", "剧院", "舞台", "直播间", "录音棚", "KTV", "家庭影院", "音响", "耳麦", "耳塞", "耳机", "话筒", "麦克风", "调音台", "功放", "音箱", "合成器", "电子琴", "手风琴", "口琴", "号角", "喇叭", "铃", "钟", "钹", "锣", "鼓", "古琴", "古筝", "琵琶", "唢呐", "箫", "笛子", "二胡", "大提琴", "小提琴", "吉他", "钢琴", "乐器", "歌词", "歌曲", "音乐", "音频", "视频", "照片", "图像", "图表", "报告", "日志", "注释", "说明", "文档", "标准", "规范", "模式", "结构", "架构", "设计", "布局", "样式", "模板", "主题", "扩展", "插件", "组件", "模块", "包", "库", "框架", "SDK", "API", "接口", "协议", "端口", "IP", "域名", "URL", "链接", "路径", "目录", "文件夹", "文件", "记录", "字段", "表", "数据库", "数据", "模型", "算法", "脚本", "代码", "程序", "系统", "硬件", "软件", "浏览器", "客户端", "主机", "服务器", "防火墙", "交换机", "路由器", "遥控器", "触控板", "鼠标", "键盘", "麦克风", "音箱", "耳机", "手表", "手环", "平板", "笔记本", "电脑", "电话", "手机", "相机", "摄像机", "投影仪", "电视", "显示器", "屏幕", "OLED", "LCD", "LED", "灯带", "灯管", "灯泡", "开关", "插座", "插头", "充电器", "充电宝", "电瓶", "电池", "电线", "电缆", "光纤", "网线", "电源线", "充电线", "数据线", "磁带", "软盘", "光盘", "U盘", "固态硬盘", "硬盘", "内存", "GPU", "CPU", "芯片", "集成电路", "三极管", "二极管", "电感器", "电阻器", "电容器", "变压器", "发电机", "电动机", "发动机", "变速器", "离合器", "油门", "刹车", "轮毂", "轮胎", "轮子", "链条", "皮带", "齿轮", "轴承", "弹簧", "垫圈", "螺栓", "螺母", "螺丝", "钉子", "铝丝", "铜丝", "钢丝", "铁丝", "线", "绳子", "橡皮筋", "橡皮", "胶布", "胶带", "胶水", "油墨", "墨水", "颜料", "染料", "涂料", "油漆", "沥青", "混凝土", "水泥", "砖瓦", "陶器", "瓷器", "陶瓷", "氨纶", "氯纶", "丙纶", "维纶", "腈纶", "涤纶", "尼龙", "化纤", "麻", "羊毛", "棉花", "丝绸", "布料", "纸张", "竹子", "木头", "皮革", "橡胶", "塑料", "玻璃", "水晶", "玛瑙", "珍珠", "玉石", "宝石", "钻石", "铂", "金", "银", "钨", "钛", "铬", "镍", "锡", "铅", "锌", "铝", "铁", "铜", "钢铁", "金属", "矿石", "煤炭", "石油", "天然气", "煤油", "柴油", "汽油", "乙醇", "酒精", "甲烷", "一氧化碳", "二氧化碳", "氮气", "氢气", "氧气", "空气", "灰尘", "泥土", "沙子", "石头", "木", "土", "火", "水", "冰", "露", "霜", "雾", "电", "雷", "雪", "雨", "风", "云", "星星", "月亮", "太阳", "天空", "宇宙", "地球", "世界", "国家", "城市", "地区", "地方", "位置", "方向", "北", "南", "西", "东", "反", "正", "侧", "背", "面", "顶", "底", "根", "尾", "头", "角", "边", "间", "中", "内", "外", "里", "后", "前", "右", "左", "下", "上", "错", "对", "坏", "好", "弱", "强", "棒", "酷", "帅", "可爱", "好看", "美丽", "漂亮", "彩色", "单色", "透明", "银", "金", "棕", "粉", "灰", "橙", "紫", "绿", "蓝", "黄", "红", "白", "黑", "暗", "亮", "吵", "安静", "乱", "整齐", "干净", "脏", "湿", "干", "空", "满", "少", "多", "便宜", "贵", "年轻", "老", "旧", "新", "晚", "早", "快", "慢", "近", "远", "浅", "深", "薄", "厚", "窄", "宽", "短", "长", "小", "大", "矮", "高", "瘦", "胖", "丑", "漂亮"}
        # 提取可能的人名
        potential_names = []
        for match in re.finditer(r'[\u4e00-\u9fff]{2,4}', last_texts):
            name = match.group()
            if name not in common_words and len(name) >= 2:
                potential_names.append(name)

        if potential_names and any(s in last_texts for s in memory_query_signals):
            names_str = "、".join(potential_names[:3])
            hints.append(f"【场子】对方在问身边的人/旧事（提到：{names_str}），如果不确定，必须调用 search_memory 查记忆，不要猜测不要编造")

        # 2. 读气氛：看最近 10 轮非我发的消息
        recent_others = [
            m for m in all_messages[-10:]
            if m.sender_type != SenderType.SELF
        ]
        recent_texts = [
            m.text for m in recent_others
            if m.text and len(m.text.strip()) > 1
        ]

        if recent_texts:
            avg_len = sum(len(t) for t in recent_texts) / len(recent_texts)
            if avg_len < 8:
                hints.append("【气氛】近期大家回复很短，适合短句/接梗")
            elif avg_len > 40:
                hints.append("【气氛】近期在正经长聊，回复可以适当展开")

            banter_signals = ["哈哈", "笑死", "牛逼", "废物", "妈的", "绝了", "艹"]
            banter_count = sum(
                1 for t in recent_texts
                if any(s in t for s in banter_signals)
            )
            if banter_count / len(recent_texts) >= 0.3:
                hints.append("【气氛】近期基调轻松调侃，大家都在互损接梗")

            late_night_signals = ["睡不着", "emo", "烦", "想", "回忆", "曾经"]
            if 22 <= hour <= 24 or 0 <= hour <= 3:
                if any(s in " ".join(recent_texts) for s in late_night_signals):
                    hints.append("【气氛】深夜感性聊天，避免理性分析和说教")

        # 3. 读场域：时间、群/私
        if is_group:
            hints.append("【场域】这是群聊，注意分寸，不要乱接话")
        else:
            hints.append("【场域】这是私聊，可以更个人化")

        if 22 <= hour or hour <= 6:
            hints.append("【时间】深夜时段，避免事务性回复和长文分析")
        elif 9 <= hour <= 18:
            hints.append("【时间】工作时间，回复可以简洁直接")

        return "\n".join(f"- {h}" for h in hints)

    def _build_user_prompt(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                           is_group: bool = False,
                           enable_time_awareness: Optional[bool] = None,
                           enable_unread_dedup: Optional[bool] = None,
                           enable_timestamps: Optional[bool] = None,
                           tools_context: str = "") -> str:
        """构建 XML 结构化 user prompt：会话信息 + 记忆 + 缓存 + 历史 + 未读。"""
        if enable_time_awareness is None:
            enable_time_awareness = self.enable_time_awareness
        if enable_unread_dedup is None:
            enable_unread_dedup = self.enable_unread_dedup
        if enable_timestamps is None:
            enable_timestamps = self.enable_timestamps

        from datetime import datetime
        chat_name = unreplied[-1].chat_name if unreplied else ""
        parts: List[str] = []

        # 会话信息
        is_at = any(getattr(m, "is_at_me", False) for m in unreplied)
        chat_type = "群聊" if is_group else "私聊"
        now_dt = datetime.now()
        now = now_dt.strftime("%Y年%m月%d日 %H:%M")
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][now_dt.weekday()]
        hour = now_dt.hour
        time_period = "凌晨" if hour < 6 else ("早上" if hour < 9 else ("上午" if hour < 12 else ("下午" if hour < 18 else ("晚上" if hour < 22 else "深夜"))))

        session_lines = []
        if enable_time_awareness:
            session_lines.append(f"当前时间：{now} {weekday} {time_period}")
        session_lines.append(f"聊天：{chat_name}")
        session_lines.append(f"类型：{chat_type}")
        session_lines.append(f"被@：{'是' if is_at else '否'}")
        if enable_time_awareness:
            session_lines.append("注意：每条消息后的时间戳是消息发出的绝对时间（YYYY-MM-DD HH:MM），请根据时间戳推断语境，不要假设消息是刚刚发的。")
        parts.append("<session>\n" + "\n".join(session_lines) + "\n</session>")

        # 语境与记忆上下文
        context_parts: List[str] = []

        # 语境速查（仅开启模式检测时注入）
        if self.enable_mode_detection and unreplied:
            mode_hints = self._build_mode_hints(unreplied, all_messages, is_group, hour)
            if mode_hints:
                context_parts.append("<mode_hints>\n" + mode_hints + "\n</mode_hints>")

        # wiki 记忆
        t_mem_ms = {}
        if self.memory_engine is not None and unreplied:
            # Bot 自己的身份信息已固化到 system prompt，不再每次注入 user prompt

            # 对方信息：仅私聊加载
            if not is_group:
                last_sender = unreplied[-1].sender
                clean_sender = last_sender.split(" @")[0] if last_sender and " @" in last_sender else last_sender
                if clean_sender and clean_sender != "我":
                    t_m2 = time.time()
                    memory_text = self.memory_engine.get_user_memory(clean_sender, max_chars=6000)
                    t_mem_ms["other"] = (time.time() - t_m2) * 1000
                    if memory_text:
                        context_parts.append("<other_info>\n" + memory_text + "\n</other_info>")
            if is_group and chat_name:
                t_m3 = time.time()
                group_text = self.memory_engine.get_group_memory(chat_name, max_chars=6000)
                t_mem_ms["group"] = (time.time() - t_m3) * 1000
                if group_text:
                    context_parts.append("<group_info>\n" + group_text + "\n</group_info>")

            mem_summary = " ".join(f"{k}={v:.0f}ms" for k, v in t_mem_ms.items())
            _logger.info(f"[Perf][Memory] {mem_summary}")

        # 已缓存数据（session_memory 中的工具结果）
        if tools_context:
            context_parts.append("<cached_data>\n" + tools_context.strip() + "\n</cached_data>")

        if context_parts:
            parts.append("<context>\n" + "\n".join(context_parts) + "\n</context>")

        # 历史消息
        if all_messages:
            history_lines = []

            def _msg_ts(m: ChatMessage) -> float:
                if m.sender_type == SenderType.SELF and m.reply_time:
                    return m.reply_time
                if m.create_time:
                    return float(m.create_time)
                return time.time()

            now_ts = time.time()
            cutoff_ts = now_ts - 1800  # 30分钟

            recent_50 = list(all_messages[-50:]) if len(all_messages) > 50 else list(all_messages)
            recent_30min = [m for m in all_messages if _msg_ts(m) >= cutoff_ts]

            union_ids = {id(m) for m in recent_50} | {id(m) for m in recent_30min}

            # 兜底：强制保留最近 5 条 bot 自己发的消息
            self_msgs = [m for m in all_messages if m.sender_type == SenderType.SELF]
            for m in self_msgs[-5:]:
                union_ids.add(id(m))
            candidate = [m for m in all_messages if id(m) in union_ids]

            max_history = 80
            recent = list(candidate[-max_history:]) if len(candidate) > max_history else list(candidate)

            for m in recent:
                history_lines.append(self._format_message_line(m, enable_timestamps))

            if len(all_messages) > len(recent):
                history_lines.append(f"（共 {len(all_messages)} 条历史，显示 {len(recent)} 条：最近50条 + 30分钟内）")
            parts.append("<history>\n" + "\n".join(history_lines) + "\n</history>")

        # 未读消息
        unread_lines = []
        skipped_hint = []
        for i, m in enumerate(unreplied, 1):
            ts = getattr(m, 'create_time', None)
            already_handled = False
            if ts and all_messages:
                for hm in all_messages:
                    if hm.sender_type == SenderType.SELF and hm.reply_time and hm.reply_time > ts:
                        already_handled = True
                        break
            tag = " ⚠️(历史中已有回复，可跳过)" if (enable_unread_dedup and already_handled) else ""
            unread_lines.append(f"{i}. {self._format_message_line(m, enable_timestamps)}{tag}")
            if enable_unread_dedup and already_handled:
                skipped_hint.append(str(i))

        if enable_unread_dedup and skipped_hint:
            unread_lines.append(f"提示：第{','.join(skipped_hint)}条未读消息在历史中已有回复，可能不需要再次回复。仅回复真正未处理的新消息。")
        unread_lines.append("回复重点：仅回复真正需要回应的未读消息。纯表情/OK/好的等确认性消息可以不回复。")
        parts.append("<unread>\n" + "\n".join(unread_lines) + "\n</unread>")

        return "\n\n".join(parts)
