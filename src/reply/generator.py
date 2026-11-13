#!/usr/bin/env python3
"""L4 Reply Generator - 回复内容生成."""

import logging
import json
import os
import re
import threading
import time
from datetime import datetime
from html import escape as xml_escape, unescape as xml_unescape
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Tuple

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
GENERATION_RETRY_RESERVE_SECONDS = 5.0

_EXPLICIT_REQUEST_RE = re.compile(
    r"[?？]|发我|发一下|给我|(?:^|[，,。；;！!\s])(?:为什么|怎么|咋|是不是|能不能|可不可以|可以吗|请|帮我|帮忙|麻烦|告诉我|解释|看一下|查一下|记得|啥|什么|谁|哪|几|多少)"
)
_INLINE_QUESTION_RE = re.compile(
    r"(?:有|剩|要|是|叫|在|去|来|给|发|用|准备|打算|需要|想|知道|清楚|记得|属于|找)"
    r"(?:了|着|过|还|大概|具体|到底|现在)*"
    r"(?:什么|啥|谁|哪(?:个|里|儿|边|些)?|多少|几(?:个|位|人|条|件|次|点|号|天|月|年|岁|层|套|份|张|杯|本|台|家|组|块|钱)?)"
)
_INLINE_QUESTION_EXCLUSION_RE = re.compile(
    r"(?:没|没有|没啥|没什么|不管|无论|甭管|管他).{0,4}(?:什么|啥|谁|哪|多少|几)|多少有点"
)
_PRESENCE_CALL_RE = re.compile(r"(?:^|[，,。；;！!\s])(?:人呢|在吗)(?:$|[，,。；;！!？?\s])")
_GROUP_AT_MENTION_RE = re.compile(r"@(?:\[[^\]\r\n]{1,20}\]\s*)?[^\s@，,。；;！!?？/]{1,24}")
_GROUP_SELF_NAMES = "|".join(re.escape(name.strip()) for name in
    os.getenv("WECHAT_SELF_NAMES", "[联系人]").split(",") if name.strip()) or re.escape("[联系人]")
_GROUP_SELF_MENTION_RE = re.compile(
    rf"@(?:{_GROUP_SELF_NAMES})(?=$|[\s\u2005，,。；;！!?？/])"
)
_GROUP_DIRECT_ACTIVITY_INVITE_RE = re.compile(
    rf"@(?:{_GROUP_SELF_NAMES})(?=$|[\s\u2005，,。；;！!?？/])"
    r"[\s\u2005，,。；;！!?？/]*(?:来|去|玩|打)(?:一?下)?[^，,。；;！!?？\r\n]{0,16}$"
)
_GROUP_SECOND_PERSON_RE = re.compile(r"(?:^|[，,。；;！!?？\s])你(?:们)?(?:[，,。；;！!?？\s]|$)|你(?:到|在|去|来|下|上|走|弄|搞|做|吃|喝|收|发|给|能|会|要|想|有|是)")
_GROUP_OPEN_ADDRESS_RE = re.compile(r"(?:大家|各位|全员|所有人|你们|谁|有没有人|群里)")
_GROUP_STATUS_QUESTION_RE = re.compile(
    r"(?:你|人).{0,8}(?:到哪|在哪|到了|到没|出发|走了|下去|上来|回来|弄好|搞定|做完|完成|收到|吃了|喝了|来不来|去不去|能来|还来|有空)"
)
_GROUP_TIME_PLACE_QUESTION_RE = re.compile(
    r"(?:几点|什么时候|啥时候|多会儿|多久|哪儿|哪里|在哪|到哪|"
    r"去哪(?:儿|里)?(?:接|找|等|碰头|集合)|什么地方|位置|地址|时间)"
)
_GROUP_QUESTION_SIGNAL_RE = re.compile(r"[?？]|(?:吗|嘛|么|呢|没)(?:$|[，,。；;！!?？\s])|哪|几|什么时候|啥时候|多久")
_GROUP_OPERATION_RE = re.compile(r"(?:接龙|报数|报名|投票|打卡|签到|按这个格式|复制格式|下一位)")
_GROUP_COORDINATION_RE = re.compile(
    r"(?:来不来|去不去|能不能来|要不要|帮我|帮忙|麻烦|发我|发一下|给我|记得|别忘了)"
)
_GROUP_WEAK_COORDINATION_RE = re.compile(r"(?:一起|参加|报名|安排|负责)")
_GROUP_COORDINATION_SIGNAL_RE = re.compile(
    r"[?？]|(?:一起|参加|报名|安排|负责)(?:吗|么|不|吧|呗|一下|下)|"
    r"(?:请|需要|谁|大家|你们|@).{0,12}(?:一起|参加|报名|安排|负责)"
)
_GROUP_PUBLIC_QUESTION_EXCLUSION_RE = re.compile(
    r"(?:订单|收款|付款|下单|红包|团购|快递|发货|退款|转账|报销|"
    r"地址|位置|定位|几点|什么时候|啥时候|多久|行程|到没到|在哪|到哪|"
    r"接龙|报名|报备|排班|打卡|签到|投票|谁负责|负责(?:吗|不|一下)|"
    r"发(?:给|我)|给我|帮我|帮忙|麻烦|请(?:问|帮|发))"
)
_GROUP_PUBLIC_QUESTION_GENERIC_RE = re.compile(
    r"^(?:哈+|哈哈+|哦+|啊+|嗯+|额+|咋了|怎么了|咋样|怎么样|"
    r"有人吗|有人在吗|在吗|人呢|有事吗)[。！!？?～~\s]*$"
)
_GROUP_PUBLIC_QUESTION_FILLER_RE = re.compile(
    r"(?:这(?:个|种)?|那个|请问|想问(?:下)?|有人知道(?:吗)?|"
    r"大家知道(?:吗)?|有没有人知道(?:吗)?|到底|具体|是不是|能不能|"
    r"可不可以|怎么|咋|为什么|什么|啥|谁|哪(?:个|里|儿|边)?|"
    r"多少|几(?:个|位|条|件|次|点|号|天|月|年)?|吗|嘛|么|呢)"
)
_GROUP_DECISION_AFTER_SELF_QUESTION_RE = re.compile(
    r"(?:好的?[，,\s]*)?(?:那|就)[^。！？!?\r\n]{0,20}(?:吧|呗)(?:$|[。！!])"
)
_GROUP_SHARED_STANCE_RE = re.compile(
    r"^(?:还有我(?:一个)?|我也是|俺也一样|算我一个|同上|\+1|＋1)[啊呀呢吧呗！!。\s]*$"
)
_GROUP_SHARED_DEFERRAL_RE = re.compile(
    r"^(?:都行|我都行|你们定|你们决定|随便|听你们的|我没意见)[啊呀呢吧呗！!。\s]*$"
)


def _looks_like_explicit_request(text: str) -> bool:
    if _EXPLICIT_REQUEST_RE.search(text):
        return True
    return bool(
        _INLINE_QUESTION_RE.search(text)
        and not _INLINE_QUESTION_EXCLUSION_RE.search(text)
    )


def _looks_like_group_coordination_request(text: str) -> bool:
    if _GROUP_COORDINATION_RE.search(text):
        return True
    if _GROUP_DIRECT_ACTIVITY_INVITE_RE.search(text):
        return True
    return bool(
        _GROUP_WEAK_COORDINATION_RE.search(text)
        and _GROUP_COORDINATION_SIGNAL_RE.search(text)
    )


def _looks_like_group_status_or_detail_question(text: str) -> bool:
    return bool(
        _GROUP_QUESTION_SIGNAL_RE.search(text)
        and (
            _GROUP_STATUS_QUESTION_RE.search(text)
            or _GROUP_TIME_PLACE_QUESTION_RE.search(text)
        )
    )


def _looks_like_group_question(text: str) -> bool:
    return bool(
        _looks_like_explicit_request(text)
        or _PRESENCE_CALL_RE.search(text)
        or re.search(r"(?:吗|嘛|么|呢)[?？]?$", text.strip())
        or _looks_like_group_status_or_detail_question(text)
    )


def _latest_explicit_request_index(
    messages: List[ChatMessage], *, include_presence_call: bool = False
) -> Optional[int]:
    """Return the 1-based index of the latest explicit request-like unread message."""
    for index in range(len(messages), 0, -1):
        message = messages[index - 1]
        text = str(getattr(message, "text", "") or "").strip()
        if text and (
            _looks_like_explicit_request(text)
            or (include_presence_call and _PRESENCE_CALL_RE.search(text))
        ):
            return index
    return None


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
                 enable_timestamps: bool = True,
                 enable_active_thread_continuation: bool = False,
                 enable_response_action_grounding: bool = False,
                 enable_latest_request_first: bool = False,
                 enable_unresolved_fact_choice_guard: bool = False,
                 enable_no_restatement_action: bool = False,
                 enable_plain_response_before_styling: bool = False,
                 enable_current_stance_continuity: bool = False,
                 enable_group_relevant_self_evidence: bool = False,
                 enable_group_self_context_freshness_boundary: bool = False,
                 enable_group_adjacent_self_response: bool = False,
                 enable_group_speaker_target_attribution: bool = False,
                 enable_group_explicit_third_party_guard: bool = False,
                 enable_group_unified_history: bool = False,
                 enable_group_unified_relevant_self: bool = False,
                 enable_group_target_action_contract: bool = False,
                 enable_group_public_question_specific_followup: bool = False,
                 enable_group_instant_self_correction: bool = False,
                 enable_group_visible_self_stance_continuity: bool = False,
                 enable_group_pending_self_action: bool = False,
                 enable_group_pattern_participation: bool = False,
                 enable_group_current_event_self_role_continuity: bool = False,
                 enable_group_explicit_self_stance_action: bool = False,
                 enable_group_answer_authority_rewrite: bool = False,
                 enable_group_feature_plan_before_generation: bool = False,
                 enable_precise_situation_matching: bool = True,
                 enable_information_gap_response_move: bool = True,
                 prefer_latest_turn_form: bool = True,
                 group_prefer_latest_turn_form: bool = False,
                 group_prefer_participant_structure: bool = False,
                 prefer_same_chat_few_shots: bool = False,
                 generation_timeout_seconds: float = 20.0,
                 group_llm_client=None,
                 group_feature_plan_llm_client=None,
                 group_generation_timeout_seconds: float | None = None,
                 persona_few_shot_path: str | Path | None = None,
                 group_persona_few_shot_path: str | Path | None = None,
                 enable_persona_few_shots: bool | None = None,
                 persona_few_shot_count: int | None = None,
                 persona_few_shot_max_chars: int | None = None,
                 enable_react_tools: bool | None = None,
                 enable_self_refine: bool | None = None,
                 disabled_skills: Collection[str] | None = None,
                 skill_content_overrides: Mapping[str, str] | None = None):
        self.llm_client = llm_client
        self.group_llm_client = group_llm_client or llm_client
        # The optional pre-generation analysis must not inherit the temporary
        # group-model switch in generate(); by default it remains on the
        # construction-time client.
        self.group_feature_plan_llm_client = group_feature_plan_llm_client or llm_client
        self.memory_engine = memory_engine
        self.tool_registry = tool_registry or get_registry()
        self.judge_worker = judge_worker
        self.enable_time_awareness = enable_time_awareness
        self.enable_reply_restraint = enable_reply_restraint
        self.enable_unread_dedup = enable_unread_dedup
        self.enable_timestamps = enable_timestamps
        self.enable_active_thread_continuation = enable_active_thread_continuation
        self.enable_response_action_grounding = enable_response_action_grounding
        self.enable_latest_request_first = enable_latest_request_first
        self.enable_unresolved_fact_choice_guard = enable_unresolved_fact_choice_guard
        self.enable_no_restatement_action = enable_no_restatement_action
        self.enable_plain_response_before_styling = enable_plain_response_before_styling
        self.enable_current_stance_continuity = enable_current_stance_continuity
        self.enable_group_relevant_self_evidence = enable_group_relevant_self_evidence
        self.enable_group_self_context_freshness_boundary = enable_group_self_context_freshness_boundary
        self.enable_group_adjacent_self_response = enable_group_adjacent_self_response
        self.enable_group_speaker_target_attribution = enable_group_speaker_target_attribution
        self.enable_group_explicit_third_party_guard = enable_group_explicit_third_party_guard
        self.enable_group_unified_history = enable_group_unified_history
        self.enable_group_unified_relevant_self = enable_group_unified_relevant_self
        self.enable_group_target_action_contract = enable_group_target_action_contract
        self.enable_group_public_question_specific_followup = enable_group_public_question_specific_followup
        self.enable_group_instant_self_correction = enable_group_instant_self_correction
        self.enable_group_visible_self_stance_continuity = enable_group_visible_self_stance_continuity
        self.enable_group_pending_self_action = enable_group_pending_self_action
        self.enable_group_pattern_participation = enable_group_pattern_participation
        self.enable_group_current_event_self_role_continuity = enable_group_current_event_self_role_continuity
        self.enable_group_explicit_self_stance_action = enable_group_explicit_self_stance_action
        self.enable_group_answer_authority_rewrite = enable_group_answer_authority_rewrite
        self.enable_group_feature_plan_before_generation = enable_group_feature_plan_before_generation
        self.enable_precise_situation_matching = enable_precise_situation_matching
        self.enable_information_gap_response_move = enable_information_gap_response_move
        self.prefer_latest_turn_form = prefer_latest_turn_form
        self.group_prefer_latest_turn_form = group_prefer_latest_turn_form
        self.group_prefer_participant_structure = group_prefer_participant_structure
        self.prefer_same_chat_few_shots = prefer_same_chat_few_shots
        self.generation_timeout_seconds = max(1.0, float(generation_timeout_seconds))
        self.group_generation_timeout_seconds = max(
            1.0,
            float(group_generation_timeout_seconds or generation_timeout_seconds),
        )
        self.default_disabled_skills = frozenset(
            str(name).strip()
            for name in (disabled_skills or ())
            if str(name).strip()
        )
        self.skill_content_overrides = dict(skill_content_overrides or {})
        project_root = Path(__file__).parent.parent.parent
        few_shot_path = Path(persona_few_shot_path or os.environ.get(
            "PERSONA_FEW_SHOT_PATH", str(project_root / "data" / "few_shot" / "persona_examples.jsonl")
        ))
        if not few_shot_path.is_absolute():
            few_shot_path = project_root / few_shot_path
        group_few_shot_path = Path(
            group_persona_few_shot_path
            or os.environ.get("GROUP_PERSONA_FEW_SHOT_PATH", str(few_shot_path))
        )
        if not group_few_shot_path.is_absolute():
            group_few_shot_path = project_root / group_few_shot_path
        few_shot_render_path_value = os.environ.get("PERSONA_FEW_SHOT_RENDER_PATH", "").strip()
        few_shot_render_path = Path(few_shot_render_path_value) if few_shot_render_path_value else None
        if few_shot_render_path is not None and not few_shot_render_path.is_absolute():
            few_shot_render_path = project_root / few_shot_render_path
        group_few_shot_render_path_value = os.environ.get(
            "GROUP_PERSONA_FEW_SHOT_RENDER_PATH",
            few_shot_render_path_value,
        ).strip()
        group_few_shot_render_path = (
            Path(group_few_shot_render_path_value) if group_few_shot_render_path_value else None
        )
        if group_few_shot_render_path is not None and not group_few_shot_render_path.is_absolute():
            group_few_shot_render_path = project_root / group_few_shot_render_path
        self.enable_persona_few_shots = (
            enable_persona_few_shots
            if enable_persona_few_shots is not None
            else os.environ.get("ENABLE_PERSONA_FEW_SHOTS", "0").lower() in ("1", "true", "yes", "on")
        )
        self.persona_few_shot_count = (
            int(persona_few_shot_count)
            if persona_few_shot_count is not None
            else _env_int("PERSONA_FEW_SHOT_COUNT", 8, 0, 12)
        )
        self.persona_few_shot_max_chars = (
            int(persona_few_shot_max_chars)
            if persona_few_shot_max_chars is not None
            else _env_int("PERSONA_FEW_SHOT_MAX_CHARS", 2500, 500, 5000)
        )
        self.persona_few_shot_retriever = PersonaFewShotRetriever(
            few_shot_path,
            render_path=few_shot_render_path,
            enable_precise_situation=enable_precise_situation_matching,
            enable_information_gap_response_move=enable_information_gap_response_move,
            prefer_latest_turn_form=prefer_latest_turn_form,
            prefer_same_chat_after_situation=prefer_same_chat_few_shots,
        )
        self.group_persona_few_shot_retriever = PersonaFewShotRetriever(
            group_few_shot_path,
            render_path=group_few_shot_render_path,
            enable_precise_situation=enable_precise_situation_matching,
            enable_information_gap_response_move=enable_information_gap_response_move,
            prefer_latest_turn_form=group_prefer_latest_turn_form,
            prefer_group_participant_structure=group_prefer_participant_structure,
            prefer_same_chat_after_situation=prefer_same_chat_few_shots,
        )
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
        # 串行化并发 generate：防止私聊/群聊流水线互相覆盖 llm_client/超时与 last_* 状态
        self._generate_lock = threading.Lock()
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
        self.last_fact_check: Dict[str, Any] = {}
        self.last_latest_request_grounding: Dict[str, Any] = {}
        self.last_current_stance_grounding: Dict[str, Any] = {}
        self.last_group_answer_authority_review: Dict[str, Any] = {}
        self._fact_check_client: Optional[Any] = None
        self.enable_fact_check = os.environ.get("ENABLE_FACT_CHECK", "1").lower() in ("1", "true", "yes", "on")

        # 开关（环境变量，默认启用）
        self.enable_react_tools = (
            os.environ.get("ENABLE_REACT_TOOLS", "1").lower() in ("1", "true", "yes", "on")
            if enable_react_tools is None
            else bool(enable_react_tools)
        )
        self.enable_self_refine = (
            os.environ.get("ENABLE_SELF_REFINE", "1").lower() in ("1", "true", "yes", "on")
            if enable_self_refine is None
            else bool(enable_self_refine)
        )
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
        self.last_fact_check = {
            "enabled": bool(self.enable_fact_check),
            "attempted": False,
            "status": "not_attempted",
            "skipped_reason": "",
            "model": os.environ.get("FACT_CHECK_MODEL", "qwen3.6-flash"),
            "reply_before": reply,
            "extract_request_messages": [],
            "extract_raw_response": "",
            "extract_parsed": {},
            "verify_request_messages": [],
            "verify_raw_response": "",
            "verify_parsed": {},
            "issues": [],
            "caused_feedback_fail": False,
            "rewrite_attempted": False,
            "rewrite_applied": False,
            "reply_after": reply,
            "error": "",
        }
        if not self.enable_fact_check:
            self.last_fact_check["skipped_reason"] = "disabled"
            return [], ""
        if not evidence:
            self.last_fact_check["skipped_reason"] = "no_evidence"
            return [], ""
        if not reply:
            self.last_fact_check["skipped_reason"] = "empty_reply"
            return [], ""
        if deadline - time.time() < 2.0:
            self.last_fact_check["skipped_reason"] = "insufficient_time_budget"
            return [], ""
        if self._fact_check_client is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
            if not api_key or not base_url:
                self.last_fact_check["skipped_reason"] = "missing_dashscope_config"
                return [], ""
            try:
                from src.utils.qwen_client import QwenClient
                self._fact_check_client = QwenClient(
                    model=os.environ.get("FACT_CHECK_MODEL", "qwen3.6-flash"),
                    api_key=api_key,
                    base_url=base_url,
                )
                archive = getattr(self, "_production_replay_archive", None)
                if archive is not None:
                    self._fact_check_client = archive.wrap_llm_client(
                        self._fact_check_client,
                        "fact_check",
                    )
            except Exception as e:
                _logger.warning("[FactCheck] 客户端初始化失败: %s", e)
                self.last_fact_check["status"] = "client_error"
                self.last_fact_check["error"] = f"{type(e).__name__}: {e}"
                return [], ""
        messages = [
            {"role": "system", "content": self._fact_check_prompt},
            {"role": "user", "content": f"<evidence>\n{evidence}\n</evidence>\n\n<reply>\n{reply}\n</reply>"},
        ]
        self.last_fact_check["attempted"] = True
        self.last_fact_check["status"] = "extract_requested"
        self.last_fact_check["extract_request_messages"] = [dict(message) for message in messages]
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
            self.last_fact_check["status"] = "extract_error"
            self.last_fact_check["error"] = f"{type(e).__name__}: {e}"
            return [], ""
        data = self._extract_json(text) or {}
        self.last_fact_check["extract_raw_response"] = text or ""
        self.last_fact_check["extract_parsed"] = data
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
            self.last_fact_check["verify_request_messages"] = [dict(message) for message in verify_messages]
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
                self.last_fact_check["verify_raw_response"] = verify_raw or ""
                self.last_fact_check["verify_parsed"] = verify_data
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
                self.last_fact_check["error"] = f"verify {type(e).__name__}: {e}"
        if issues:
            _logger.warning("[FactCheck] 命中明确矛盾: %s", issues)
        combined_raw = json.dumps(
            {"extract": self._extract_json(text) or {}, "verify": self._extract_json(verify_raw) or {}},
            ensure_ascii=False,
        )
        self.last_fact_check["status"] = "issues_found" if issues else "passed"
        self.last_fact_check["issues"] = list(issues)
        self.last_fact_check["caused_feedback_fail"] = bool(issues)
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
        base_messages = list(messages)

        for attempt in range(max_retries + 1):
            start_time = time.time()
            if attempt > 0:
                # 重试时重置消息流：上一轮残留的 assistant/tool 往返不带入下一轮，
                # 避免上下文只增不减、带着错误示范，以及对部分 API 构成非法消息序列
                messages = list(base_messages)
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
                    if attempt < max_retries:
                        llm_timeout = max(
                            1.0,
                            llm_timeout - GENERATION_RETRY_RESERVE_SECONDS,
                        )
                    _logger.info("[LLM] attempt=%d round=%d force_no_tools=%s tools=%s timeout=%s msg_count=%d",
                                 attempt + 1, tool_round_count, force_no_tools, bool(actual_tools), llm_timeout, len(messages))
                    t_llm_start = time.time()
                    # force_no_tools 阶段强制 JSON 输出模式，避免 LLM 输出非法 JSON
                    response_format = {"type": "json_object"} if not actual_tools else None
                    raw = self.llm_client.chat(
                        messages=messages, tools=actual_tools,
                        max_tokens=10000, timeout=llm_timeout,
                        response_format=response_format,
                        max_retries=0,
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

                    parsed = self._extract_json(text) if text else None
                    raw_replies = parsed.get("replies") if isinstance(parsed, dict) else None
                    if isinstance(raw_replies, list) and any(
                        isinstance(reply, str) and reply.strip() for reply in raw_replies
                    ):
                        messages.append({
                            "role": "user",
                            "content": (
                                "刚才的 replies 只有会被过滤的占位回复。"
                                "请改为与当前上下文有关、包含实际含义的非空回复，并继续严格输出 JSON。"
                            ),
                        })
                        _logger.info("[Generate] replies 被过滤，携带原因继续 retry")
                        break

                    # 只有结构合法的 {"replies": []} 才表示明确不回复。
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
                    if tool_call_count:
                        _logger.warning(
                            "[ReactGenerate] 重试发生在已执行 %d 次工具之后，有副作用的工具可能被重复执行",
                            tool_call_count)
                    # 睡眠不超过剩余 deadline，超时边缘不再白等
                    time.sleep(min(1.0, max(0.0, deadline - time.time())))

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
                 enable_timestamps: Optional[bool] = None,
                 evaluation_time: Optional[datetime] = None,
                 few_shot_exclude_ids: set[str] | None = None) -> List[str]:
        """Select the adopted private/group model, then run one generation.

        持锁防止并发 generate 互相覆盖模型客户端/超时与 last_* 观测状态。"""
        with self._generate_lock:
            return self._generate_swapped(
                unreplied, all_messages, is_group=is_group, tick_id=tick_id,
                enable_time_awareness=enable_time_awareness,
                enable_reply_restraint=enable_reply_restraint,
                enable_unread_dedup=enable_unread_dedup,
                enable_timestamps=enable_timestamps,
                evaluation_time=evaluation_time,
                few_shot_exclude_ids=few_shot_exclude_ids,
            )

    def _generate_swapped(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                 is_group: bool = False, tick_id: int = 0,
                 enable_time_awareness: Optional[bool] = None,
                 enable_reply_restraint: Optional[bool] = None,
                 enable_unread_dedup: Optional[bool] = None,
                 enable_timestamps: Optional[bool] = None,
                 evaluation_time: Optional[datetime] = None,
                 few_shot_exclude_ids: set[str] | None = None) -> List[str]:
        previous_client = self.llm_client
        previous_timeout = self.generation_timeout_seconds
        self.llm_client = self.group_llm_client if is_group else previous_client
        self.generation_timeout_seconds = (
            self.group_generation_timeout_seconds if is_group else previous_timeout
        )
        try:
            return self._generate(
                unreplied,
                all_messages,
                is_group=is_group,
                tick_id=tick_id,
                enable_time_awareness=enable_time_awareness,
                enable_reply_restraint=enable_reply_restraint,
                enable_unread_dedup=enable_unread_dedup,
                enable_timestamps=enable_timestamps,
                evaluation_time=evaluation_time,
                few_shot_exclude_ids=few_shot_exclude_ids,
            )
        finally:
            self.llm_client = previous_client
            self.generation_timeout_seconds = previous_timeout

    def _generate(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                 is_group: bool = False, tick_id: int = 0,
                 enable_time_awareness: Optional[bool] = None,
                 enable_reply_restraint: Optional[bool] = None,
                 enable_unread_dedup: Optional[bool] = None,
                 enable_timestamps: Optional[bool] = None,
                 evaluation_time: Optional[datetime] = None,
                 few_shot_exclude_ids: set[str] | None = None) -> List[str]:
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
        self.last_fact_check = {
            "enabled": bool(self.enable_fact_check),
            "attempted": False,
            "status": "not_reached" if self.enable_self_refine else "self_refine_disabled",
            "skipped_reason": (
                "self_refine_not_reached" if self.enable_self_refine else "self_refine_disabled"
            ),
            "issues": [],
            "caused_feedback_fail": False,
            "rewrite_attempted": False,
            "rewrite_applied": False,
        }
        self.last_latest_request_grounding = {
            "attempted": False,
            "status": "candidate_disabled",
            "reply_before": [],
            "reply_after": [],
            "elapsed_ms": 0,
        }
        self.last_current_stance_grounding = {
            "attempted": False,
            "status": "candidate_disabled",
            "current_stance": "none",
            "reply_before": [],
            "reply_after": [],
            "elapsed_ms": 0,
        }
        self.last_group_answer_authority_review = {
            "attempted": False,
            "status": "candidate_disabled",
            "reply_before": [],
            "reply_after": [],
            "elapsed_ms": 0,
        }

        chat_name = unreplied[-1].chat_name if unreplied else ""

        # 模型辅助路由：提前到 system prompt 之前，让 prompt 也能感知已加载 skill
        last_msg = unreplied[-1]
        route_text = last_msg.text or last_msg.image_description or ""
        # 取最近消息作为路由上下文（含历史消息 + 同批次其它未读），帮助路由理解对话语境
        n_unreplied = len(unreplied)
        prior = list(all_messages[:-n_unreplied]) + list(unreplied[:-1])
        context_msgs = [(m.sender, m.text) for m in prior if m.text][-3:]

        overall_deadline = time.time() + self.generation_timeout_seconds
        current_stance_review_reserve = (
            min(15.0, self.generation_timeout_seconds * 0.2)
            if (
                self.enable_group_self_context_freshness_boundary
                or self.enable_group_answer_authority_rewrite
            ) and is_group
            else 0.0
        )
        deadline = overall_deadline - current_stance_review_reserve
        t_route_start = time.time()
        matched_skills = self._route_skills(
            route_text,
            context_messages=context_msgs,
            is_group=is_group,
            deadline=deadline,
        )
        _logger.info(f"[Perf][Generate] skill routing={(time.time()-t_route_start)*1000:.0f}ms")

        # DISABLED_SKILLS 支持以逗号分隔的 skill 名，仅用于可逆的逐项 A/B。
        # 它优先于路由和 FORCE_SKILL，避免实验时意外注入被关闭的内容。
        disabled_skills = self.default_disabled_skills | {
            name.strip()
            for name in os.environ.get("DISABLED_SKILLS", "").split(",")
            if name.strip()
        }

        # FORCE_SKILL 环境变量：跳过路由，强制注入指定 skill（用于隔离测试 skill 内容）
        force_skill = os.environ.get("FORCE_SKILL", "").strip()
        if force_skill:
            manifest_names = {s["name"] for s in self._load_skill_manifest()}
            if force_skill not in manifest_names:
                _logger.warning("[SkillRouter] FORCE_SKILL=%s 不在可用 skill 列表中，忽略", force_skill)
            else:
                matched_skills = [force_skill]
                _logger.warning("[SkillRouter] FORCE_SKILL=%s，跳过路由（注意：这是调试覆写，勿留生产环境）", force_skill)

        had_selected_skill = bool(matched_skills)
        if disabled_skills:
            routed_skills = list(matched_skills)
            matched_skills = [name for name in matched_skills if name not in disabled_skills]
            if routed_skills != matched_skills:
                _logger.warning(
                    "[SkillRouter] DISABLED_SKILLS=%s，未注入: %s",
                    sorted(disabled_skills),
                    sorted(set(routed_skills) - set(matched_skills)),
                )

        # 模型选择：固定使用主 LLM（带 tools），让 LLM 自己决定是否需要工具。
        self.last_active_llm = str(
            getattr(self.llm_client, "model", None)
            or getattr(self.llm_client, "resolved_model", None)
            or type(self.llm_client).__name__
        )
        _logger.info("[ModelSelect] matched_skills=%s active_llm=%s", matched_skills, self.last_active_llm)

        system_prompt = self._system_prompt(
            enable_reply_restraint=enable_reply_restraint,
            unreplied=unreplied,
            all_messages=all_messages,
            is_group=is_group,
        )
        if self.enable_group_self_context_freshness_boundary and is_group:
            system_prompt = self._with_group_self_context_freshness_boundary(system_prompt)
            if self.enable_group_adjacent_self_response:
                system_prompt = self._with_group_adjacent_self_response(system_prompt)
        if self.enable_group_unified_relevant_self and is_group:
            system_prompt = self._with_group_unified_relevant_self_schema(system_prompt)
        if self.enable_latest_request_first:
            system_prompt = self._with_latest_request_evidence_schema(system_prompt)
        if self.enable_current_stance_continuity:
            system_prompt = self._with_current_stance_schema(system_prompt)

        tools_context = self._build_tools_context(chat_name)

        user_prompt = self._build_user_prompt(
            unreplied, all_messages, is_group,
            enable_time_awareness=enable_time_awareness,
            enable_unread_dedup=enable_unread_dedup,
            enable_timestamps=enable_timestamps,
            tools_context=tools_context,
            current_time=evaluation_time,
        )

        # Skill 兜底：明确场景未命中时，群聊默认 group_banter，私聊默认 casual_chat
        if not matched_skills and not had_selected_skill:
            fallback_skill = "group_banter" if is_group else "casual_chat"
            if fallback_skill in disabled_skills:
                _logger.info("[SkillRouter] 兜底 skill=%s 已禁用，不注入 skill", fallback_skill)
            else:
                matched_skills = [fallback_skill]
                _logger.info(f"[SkillRouter] 未命中明确 skill，兜底: {matched_skills}")

        # Skill 注入：主模型只接收路由选中的 skill 全文。
        skill_parts = []
        injected_skills = []
        if matched_skills:
            for skill_name in matched_skills:
                content = self._load_skill_content(skill_name)
                if content:
                    injected_skills.append(skill_name)
                    skill_parts.append(
                        f'<active_skill name="{xml_escape(skill_name, quote=True)}">\n'
                        f"{content}\n"
                        "</active_skill>"
                    )
            if skill_parts:
                user_prompt = self._append_request_sections(user_prompt, skill_parts)
        self.last_loaded_skills = injected_skills
        self.last_skill_injected_content = "\n\n".join(skill_parts) if skill_parts else ""

        active_few_shot_retriever = (
            self.group_persona_few_shot_retriever if is_group else self.persona_few_shot_retriever
        )
        self.persona_few_shot_ready = (
            self.persona_few_shot_allow_unreviewed
            or active_few_shot_retriever.is_approved()
        )
        if self.enable_persona_few_shots and self.persona_few_shot_ready and self.persona_few_shot_count:
            few_shot_rows = active_few_shot_retriever.retrieve(
                query="\n".join([text for _, text in context_msgs] + [route_text]),
                chat_name=chat_name,
                is_group=is_group,
                limit=self.persona_few_shot_count,
                relationship="group" if is_group else "acquaintance",
                interaction_context=" ".join(matched_skills),
                exclude_ids=few_shot_exclude_ids,
                use_situation=is_group or self.enable_precise_situation_matching,
                current_context_messages=[
                    {"sender": sender, "text": text}
                    for sender, text in [*context_msgs, (last_msg.sender, route_text)]
                    if text
                ],
            )
            few_shot_content, few_shot_ids = active_few_shot_retriever.render_selected(
                few_shot_rows,
                max_chars=self.persona_few_shot_max_chars,
            )
            if few_shot_content:
                user_prompt = self._append_request_sections(user_prompt, [few_shot_content])
                if self.enable_latest_request_first:
                    user_prompt = self._append_request_sections(user_prompt, [
                        '<style_example_usage priority="final" enforcement="hard">\n'
                        '<rule>style_examples 只能学接话动作、语气、节奏和长度；其中的人物、事实、肯定或否定状态、理由、数字、计划和后续动作一律不能带入当前回复。</rule>\n'
                        '<rule>先只根据当前 unread、history、可信人设、可靠记忆或工具决定“回什么”，再用 style_examples 决定“怎么说”。删掉 style_examples 后无法得出的内容必须删除。</rule>\n'
                        '</style_example_usage>'
                    ])
                self.last_few_shot_content = few_shot_content
                self.last_few_shot_ids = few_shot_ids
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "persona_few_shot",
                    "timestamp": time.time(),
                    "ids": few_shot_ids,
                })
                _logger.info("[PersonaFewShot] chat=%s ids=%s", chat_name, few_shot_ids)

        if self.enable_latest_request_first:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<latest_request_contract priority="final" enforcement="hard">\n'
                '<rule>先实际处理当前对话里最近尚未解决、且确实指向你的诉求；结合说话人、称呼、@ 对象和上下文判断，不要机械追最后一句。回答、拒绝、澄清或问一个必要问题后立即结束，不得转去旧话题、调侃、建议或总结。</rule>\n'
                '<rule>response_target_candidate 只是帮助定位的候选，必须核对指向；确认指向你后，最终回复必须真正处理它，不能只接它旁边更容易发挥的话。</rule>\n'
                '<rule>active_skill 和 style_examples 只能帮助决定表达方式，不能提供当前事实。删掉它们后无法从当前证据得出的事实、判断、承诺或后续动作必须删除。</rule>\n'
                '<rule>硬伤只指回复与当前上下文或可信 Wiki 存在可以直接指出的矛盾。没有可见依据、措辞不自然或信息不够丰富，本身都不算硬伤。</rule>\n'
                '</latest_request_contract>'
            ])
            self.last_latest_request_grounding = {
                "attempted": False,
                "status": "one_pass_contract",
                "reply_before": [],
                "reply_after": [],
                "elapsed_ms": 0,
            }

        if (
            self.enable_unresolved_fact_choice_guard
            and self._has_numeric_fact_choice(self._latest_request_target_text(user_prompt))
        ):
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<unresolved_fact_choice_guard priority="final" enforcement="hard">\n'
                '<rule>最新问题把两个或更多数字、比例或金额列成候选答案时，候选答案本身不证明其中任何一个正确。只有当前 unread、history、可信 Wiki 或工具结果直接支持某一项时才能选定；没有直接证据时，回答能确认的部分，并把具体选项留作待核对，不得猜一个、拼接新数字或从 style_examples 带入结论。</rule>\n'
                '</unresolved_fact_choice_guard>'
            ])

        if self.enable_no_restatement_action:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<no_restatement_action priority="final" enforcement="hard">\n'
                '<rule>对方已经完整说清的事实、安排或结论，不得换词复述、概括或包装成建议。先确定当前回复唯一需要完成的互动动作，再直接完成它。</rule>\n'
                '<rule>互动动作只能来自当前现场：追问确实缺失的来源或条件、表达本人已有立场、直接接受或拒绝，或者短促共鸣。不要机械追问；当前现场不缺信息时不得为了套用规则而提问。</rule>\n'
                '<rule>不得凭空增加方案、动机、事实、后续动作，也不得跨回已经结束的旧话题。style_examples 只提供表达方式，不能提供当前事实。</rule>\n'
                '</no_restatement_action>'
            ])

        if self.enable_plain_response_before_styling:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<plain_response_before_styling priority="final" enforcement="hard">\n'
                '<rule>先确定本轮唯一要完成的互动动作，并先写一个短、字面、正常的回应草稿。</rule>\n'
                '<rule>只有当前连续聊天或选中的真人 style_examples 已经出现同类玩笑节奏时，才可以把草稿加工成玩笑；仅仅话题或用词相似不算现场依据。</rule>\n'
                '<rule>生成后检查：如果普通回应被升级成对仗、成语总结、比喻、口号、通用网络模板或完整包袱，而当前现场没有同类节奏，就丢弃加工版本并输出字面草稿。</rule>\n'
                '<rule>不得借加工改变事实、人物身份、关系站位、本人立场或最新诉求；style_examples 只提供表达方式，不能提供当前事实。</rule>\n'
                '</plain_response_before_styling>'
            ])

        if self.enable_current_stance_continuity:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<current_stance_continuity priority="final" enforcement="hard">\n'
                '<rule>生成 replies 前必须先填写 decision.current_stance。只有同时满足两项时才可引用本人原话：本人刚明确说出自己的安排、行动限制或时间压力；unread 正在直接要求、建议或提出方案改变这项安排。先按消息顺序判断本人后说的话是否更新了人物、行动或先后关系；若已更新，较早安排失效，不得引用。只能逐字引用最新且仍同时成立的 1 至 3 条原话，多条之间用“ || ”分隔；不得把不同阶段或互相冲突的安排拼在一起；否则必须填 none。这个内部字段不得出现在 replies 中。</rule>\n'
                '<rule>对方只是调侃、预测、假设、重复本人态度、询问看法，或谈论不会改变本人当前安排的事情时，必须填 none；不得把一般偏好、判断或旧话题扩大成当前安排冲突。</rule>\n'
                '<rule>current_stance 不是 none 时，回复必须直接拒绝或纠正对方提出的改变，同时仍处理最新诉求；不得接受、弱化或忽略原安排，也不得把“要赶时间”改成让对方慢慢来。</rule>\n'
                '<rule>只能保持原话已经表达的内容，不得加强成“定死了”“绝对不行”或补充新的原因、事实和安排。</rule>\n'
                '<rule>写出每一条 reply 后，逐条与 current_stance 中所有用“ || ”分隔的原话核对。若 reply 接受了对方提出的改变、与任一原话矛盾，或增加了原话没有的事实，必须丢弃并重写。</rule>\n'
                '<rule>current_stance 为 none 时正常接话，本功能不得再改变回复方向。</rule>\n'
                '<rule>仍须处理确实指向本人的最新诉求；保持立场不等于重复原话，也不得跨回已经结束的旧话题。</rule>\n'
                '</current_stance_continuity>'
            ])

        if self.enable_group_speaker_target_attribution and is_group:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<group_speaker_target_attribution priority="final" enforcement="hard">\n'
                '<rule>写回复前先在心里逐条标清：每句是谁说的；最后一句是在对谁说；其中的“我、你、他、这边、那边”分别指谁；每个判断、状态和立场属于谁。不要输出这份核对。</rule>\n'
                '<rule>只有最后一句明确对本人说，或它确实延续了本人正在参与的同一轮问答时，才以本人身份回应。若明确在问、评价或催促群里其他人，不得冒充对方回答，也不得把对方的状态或立场说成本人的。</rule>\n'
                '<rule>回复必须站在本人的位置接对人。不得因为 style_examples 里话题或用词相似，就照搬示例中的人物关系、身份、事实或立场。</rule>\n'
                '<rule>生成后再核对一次：回复中的人称、回答对象和立场归属是否都能由当前群聊直接支持；不能支持就改成不冒充他人的自然接话，或按群聊参与规则不回复。</rule>\n'
                '</group_speaker_target_attribution>'
            ])

        if self.enable_group_unified_relevant_self and is_group:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<group_unified_relevant_self priority="final" enforcement="hard">\n'
                '<rule>先保持 history 的原始消息顺序，判断本批 unread 是否在具体回应其中某条 role=self 消息。生成 replies 前必须填写 decision.directly_answered_self_quote：逐字选择被回应且时间最新的 1 条 role=self 原话，或填写 none。</rule>\n'
                '<rule>只有能指出 unread 与该 role=self 原话之间的具体接话关系时才选择；仅话题、用词、人物相同或时间接近都必须填 none。后面的本人消息若已更新或否定前面的信息，以后话为准。</rule>\n'
                '<rule>选中原话时，只允许保持该句已有的事实、状态、计划、限制或判断；不得照抄旧句、复用旧梗、使用其他本人历史或添加输入外事实。填 none 时，role=self 历史只用于理解时间线，不得成为回复素材。</rule>\n'
                '</group_unified_relevant_self>'
            ])

        if (
            self.enable_group_explicit_self_stance_action
            and is_group
            and '<group_explicit_self_stance_action ' in user_prompt
        ):
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<group_pending_self_judgment_priority priority="final" enforcement="hard">\n'
                '<rule>先检查 local_timeline：如果本人正在形成一个尚未完成的具体问题或判断，后续消息又提供了会直接改变该判断的新事实，当前接话应优先完成或更新这个判断。</rule>\n'
                '<rule>末尾只有惊叹、表情、重复，或同一现场中更显眼但不改变该判断的旁支事实时，不能覆盖这个待完成判断；只有末尾向本人提出了独立的新问题、请求、选择或动作，才改为优先处理它。</rule>\n'
                '<rule>若不存在上述待完成判断，本段不得改变原接话方向。不得强迫正面作答、照抄本人旧话，或补充 local_timeline 中不可见的理由和事实。</rule>\n'
                '</group_pending_self_judgment_priority>'
            ])

        if enable_reply_restraint is False:
            user_prompt = self._append_request_tail_sections(user_prompt, [
                '<offline_force_reply_contract priority="final" enforcement="hard">\n'
                '<rule>本轮是离线强制回复实验。只要输入可读，replies 必须包含至少一条非空回复；任何通常要求静默、留空或不参与的规则在本轮均不适用。</rule>\n'
                '<rule>若消息没有明确指向你，也要给出不虚构事实、不冒充被询问者的最短自然接话。</rule>\n'
                '</offline_force_reply_contract>'
            ])

        if is_group and self.enable_group_feature_plan_before_generation:
            user_prompt = self._run_group_feature_plan_before_generation(
                user_prompt, overall_deadline
            )

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
                if self.last_fact_check.get("caused_feedback_fail"):
                    self.last_fact_check["rewrite_attempted"] = deadline - time.time() >= 1.0

                # 单次 Iterate（无循环：issues 不刷新时循环无意义）
                if deadline - time.time() >= 1.0:
                    new_replies, new_messages, _ = self._iterate(current_messages, issues, deadline)
                    if new_replies:
                        current_replies = new_replies
                        current_messages = new_messages
                        if self.last_fact_check.get("caused_feedback_fail"):
                            self.last_fact_check["rewrite_applied"] = True
                    else:
                        _logger.warning("[SelfRefine] Iterate 未生成有效回复，保留原回复")
                    self.last_iterate_count = 1
                else:
                    self.last_iterate_skipped_no_budget = True
                    _logger.warning("[SelfRefine] Iterate 因预算不足跳过 (deadline=%.1fs)",
                                    deadline - time.time())

                replies = current_replies
                final_messages = current_messages
                if self.last_fact_check.get("caused_feedback_fail"):
                    self.last_fact_check["reply_after"] = "\n".join(replies)

        combined_review = self._run_combined_reply_review(
            user_prompt,
            replies,
            final_messages,
            deadline,
            overall_deadline,
            is_group=is_group,
        )
        if combined_review is not None:
            replies, final_messages = combined_review
            # 确定性兜底：LLM rewrite 之后仍缺直接证据时收敛回复，禁止编造本人行为/承诺。
            # 与 latest-request-first 同一开关控制，默认关闭不影响生产。
            replies = self._apply_latest_request_evidence_guard(
                user_prompt,
                replies,
                final_messages,
            )
        else:
            replies, final_messages = self._check_latest_request_reply(
                user_prompt,
                replies,
                final_messages,
                deadline,
            )
            # 确定性兜底：LLM rewrite 之后仍缺直接证据时收敛回复，禁止编造本人行为/承诺。
            # 与 latest-request-first 同一开关控制，默认关闭不影响生产。
            replies = self._apply_latest_request_evidence_guard(
                user_prompt,
                replies,
                final_messages,
            )
            replies, final_messages = self._check_current_stance_reply(
                replies,
                final_messages,
                overall_deadline,
                is_group=is_group,
            )
            replies, final_messages = self._check_group_answer_authority_reply(
                replies,
                final_messages,
                overall_deadline,
                is_group=is_group,
            )
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

    @staticmethod
    def _validated_prompt_quotes(value: Any, source_prompt: str, *, field: str) -> tuple[Optional[List[str]], str]:
        """Accept only short, literal evidence quotes from the allowed chat source."""
        if not isinstance(value, list) or not value:
            return None, f"{field}_missing"
        if len(value) > 5:
            return None, f"{field}_too_many"
        quotes: List[str] = []
        for item in value:
            if not isinstance(item, str):
                return None, f"{field}_not_string"
            quote = item.strip()
            if not quote or len(quote) > 1000:
                return None, f"{field}_invalid_length"
            if quote not in source_prompt:
                return None, f"{field}_not_in_source_prompt"
            quotes.append(quote)
        return quotes, ""

    @staticmethod
    def _current_chat_evidence_source(user_prompt: str) -> str:
        """Return only message-bearing current-chat sections from a generation prompt."""
        sections: List[str] = []
        for tag in ("history", "unread", "consumed_self_replies", "local_timeline"):
            sections.extend(
                match.group(0)
                for match in re.finditer(
                    rf"<{tag}\b[^>]*>.*?</{tag}>", user_prompt, flags=re.DOTALL
                )
            )
        return "\n".join(sections)

    @staticmethod
    def _validated_string_list(value: Any, *, field: str) -> tuple[Optional[List[str]], str]:
        if not isinstance(value, list) or len(value) > 8:
            return None, f"{field}_invalid"
        items: List[str] = []
        for item in value:
            if not isinstance(item, str) or len(item.strip()) > 300:
                return None, f"{field}_invalid_item"
            if item.strip():
                items.append(item.strip())
        return items, ""

    def _validate_group_generation_features(
        self, data: Any, source_prompt: str
    ) -> tuple[Optional[Dict[str, Any]], str]:
        if not isinstance(data, dict):
            return None, "features_not_object"
        targets = {"self", "specific_other", "group", "uncertain"}
        actions = {"answer", "acknowledge", "clarify", "decline", "banter", "wait", "other"}
        stages = {"active", "resolved", "topic_shift", "uncertain"}
        target = data.get("reply_target")
        action = data.get("interaction_action")
        stage = data.get("event_stage")
        if target not in targets:
            return None, "invalid_reply_target"
        if action not in actions:
            return None, "invalid_interaction_action"
        if stage not in stages:
            return None, "invalid_event_stage"
        if not isinstance(data.get("self_context_relevance"), bool):
            return None, "invalid_self_context_relevance"
        if not isinstance(data.get("pattern_participation"), bool):
            return None, "invalid_pattern_participation"
        evidence, reason = self._validated_prompt_quotes(
            data.get("evidence"), source_prompt, field="evidence"
        )
        if evidence is None:
            return None, reason
        prohibited, reason = self._validated_string_list(
            data.get("prohibited_assumptions"), field="prohibited_assumptions"
        )
        if prohibited is None:
            return None, reason
        return {
            "reply_target": target,
            "interaction_action": action,
            "event_stage": stage,
            "self_context_relevance": data["self_context_relevance"],
            "pattern_participation": data["pattern_participation"],
            "evidence": evidence,
            "prohibited_assumptions": prohibited,
        }, ""

    def _validate_group_generation_plan(
        self, data: Any, source_prompt: str, features: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], str]:
        if not isinstance(data, dict):
            return None, "plan_not_object"
        if data.get("reply_target") != features["reply_target"]:
            return None, "plan_reply_target_mismatch"
        if data.get("reply_action") != features["interaction_action"]:
            return None, "plan_reply_action_mismatch"
        allowed_facts, reason = self._validated_prompt_quotes(
            data.get("allowed_facts"), source_prompt, field="allowed_facts"
        )
        if allowed_facts is None:
            return None, reason
        forbidden, reason = self._validated_string_list(
            data.get("forbidden_assumptions"), field="forbidden_assumptions"
        )
        if forbidden is None:
            return None, reason
        tone = data.get("tone")
        plan = data.get("plan")
        if not isinstance(tone, str) or not tone.strip() or len(tone.strip()) > 80:
            return None, "invalid_plan_tone"
        if not isinstance(plan, str) or not plan.strip() or len(plan.strip()) > 300:
            return None, "invalid_plan"
        return {
            "reply_target": features["reply_target"],
            "reply_action": features["interaction_action"],
            "allowed_facts": allowed_facts,
            "forbidden_assumptions": forbidden,
            "tone": tone.strip(),
            "plan": plan.strip(),
        }, ""

    @staticmethod
    def _render_group_generation_plan(features: Dict[str, Any], plan: Dict[str, Any]) -> str:
        def _items(tag: str, values: List[str]) -> str:
            return "".join(
                f"<{tag}>{xml_escape(value, quote=True)}</{tag}>" for value in values
            )

        return (
            '<group_generation_plan priority="final" source="validated_auxiliary">\n'
            f"<reply_target>{xml_escape(features['reply_target'], quote=True)}</reply_target>\n"
            f"<reply_action>{xml_escape(plan['reply_action'], quote=True)}</reply_action>\n"
            f"<event_stage>{xml_escape(features['event_stage'], quote=True)}</event_stage>\n"
            f"<self_context_relevance>{str(features['self_context_relevance']).lower()}</self_context_relevance>\n"
            f"<pattern_participation>{str(features['pattern_participation']).lower()}</pattern_participation>\n"
            f"<evidence>{_items('item', features['evidence'])}</evidence>\n"
            f"<allowed_facts>{_items('item', plan['allowed_facts'])}</allowed_facts>\n"
            f"<forbidden_assumptions>{_items('item', plan['forbidden_assumptions'])}</forbidden_assumptions>\n"
            f"<tone>{xml_escape(plan['tone'], quote=True)}</tone>\n"
            f"<plan>{xml_escape(plan['plan'], quote=True)}</plan>\n"
            "<rule>该计划只帮助锁定回复对象、互动动作和事实边界，不能强迫正面作答。"
            "调侃、澄清、拒绝、等待或自然不参与仍可在当前上下文中选择；任何当前上下文无法支持的事实、结论或承诺都不得写入回复。</rule>\n"
            "</group_generation_plan>"
        )

    def _run_group_feature_plan_before_generation(
        self, user_prompt: str, overall_deadline: float
    ) -> str:
        """Build a fail-closed, auditable group reply plan from the final prompt."""
        final_reserve = min(120.0, max(8.0, self.generation_timeout_seconds * 0.5))
        auxiliary_deadline = overall_deadline - final_reserve
        chat_evidence_source = self._current_chat_evidence_source(user_prompt)

        def _trace(
            trace_type: str, status: str, prompt: str, raw_response: str,
            parsed: Optional[Dict[str, Any]], started_at: float, fallback_reason: str = "",
        ) -> None:
            self.last_generation_trace.append({
                "round": 0,
                "type": trace_type,
                "status": status,
                "timestamp": started_at,
                "source_prompt": "final_user_prompt",
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed": parsed or {},
                "elapsed_ms": round((time.time() - started_at) * 1000),
                "fallback_reason": fallback_reason,
            })

        def _call(prompt: str, *, max_tokens: int) -> tuple[str, str]:
            available = auxiliary_deadline - time.time()
            if available < 1.0:
                return "", "auxiliary_budget_exhausted"
            # Keep enough time for the second auxiliary call and final generation,
            # but do not impose the former 30-second cap on a Turbo group request.
            timeout = min(90.0, max(1.0, available / 2))
            try:
                raw = self.group_feature_plan_llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    max_tokens=max_tokens,
                    temperature=0,
                    timeout=timeout,
                    max_retries=0,
                    raise_on_error=True,
                )
            except Exception as exc:
                _logger.warning("[GroupFeaturePlan] auxiliary call failed: %s", exc)
                return "", f"auxiliary_call_failed:{type(exc).__name__}"
            return (raw if isinstance(raw, str) else getattr(raw, "content", str(raw))) or "", ""

        feature_prompt = (
            "分析群聊回复的事实边界。仅以 current_chat_evidence 为事实来源；不生成回复、不写推理。"
            "只输出 JSON。evidence 的每项必须是其中逐字出现的非空片段；"
            "示例、技能、规则和人设都不是当前事实。不能由证据支持的内容写入 prohibited_assumptions："
            '{"reply_target":"self|specific_other|group|uncertain","interaction_action":"answer|acknowledge|clarify|decline|banter|wait|other",'
            '"event_stage":"active|resolved|topic_shift|uncertain","self_context_relevance":true,"pattern_participation":false,'
            '"evidence":["逐字证据"],"prohibited_assumptions":["不可假设的事实"]}\n'
            "<current_chat_evidence>\n"
            + chat_evidence_source
            + "\n</current_chat_evidence>"
        )
        feature_started = time.time()
        feature_raw, feature_error = _call(feature_prompt, max_tokens=500)
        feature_data = self._extract_json(feature_raw) if feature_raw else None
        features, validation_error = self._validate_group_generation_features(
            feature_data, chat_evidence_source
        )
        feature_error = feature_error or validation_error
        if features is None:
            _trace(
                "group_feature_extraction", "fallback", feature_prompt, feature_raw,
                None, feature_started, feature_error or "invalid_features",
            )
            return user_prompt
        _trace("group_feature_extraction", "accepted", feature_prompt, feature_raw, features, feature_started)

        plan_prompt = (
            "你是群聊回复前的短计划器。只能使用同一 current_chat_evidence 中逐字可见的当前聊天事实，"
            "不得生成最终回复或补充事实。给定的 feature_contract 已经通过验证，reply_target 与 reply_action 必须逐字保持一致。"
            "allowed_facts 必须是 current_chat_evidence 中逐字出现的非空原文片段；"
            "style_examples、技能说明或规则中的人物事实不能放入 allowed_facts。只输出 JSON："
            '{"reply_target":"必须等于 feature_contract.reply_target","reply_action":"必须等于 feature_contract.interaction_action",'
            '"allowed_facts":["逐字事实"],"forbidden_assumptions":["不可假设的事实"],'
            '"tone":"简短语气要求","plan":"不超过一句的接话计划"}\n'
            "<feature_contract>\n"
            + json.dumps(features, ensure_ascii=False, separators=(",", ":"))
            + "\n</feature_contract>\n<current_chat_evidence>\n"
            + chat_evidence_source
            + "\n</current_chat_evidence>"
        )
        plan_started = time.time()
        plan_raw, plan_error = _call(plan_prompt, max_tokens=500)
        plan_data = self._extract_json(plan_raw) if plan_raw else None
        plan, validation_error = self._validate_group_generation_plan(
            plan_data, chat_evidence_source, features
        )
        plan_error = plan_error or validation_error
        if plan is None:
            _trace(
                "group_generation_plan", "fallback", plan_prompt, plan_raw,
                None, plan_started, plan_error or "invalid_plan",
            )
            return user_prompt
        _trace("group_generation_plan", "accepted", plan_prompt, plan_raw, plan, plan_started)
        return self._append_request_tail_sections(
            user_prompt, [self._render_group_generation_plan(features, plan)]
        )

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

    @staticmethod
    def _is_short_necessary_question(replies: List[str]) -> bool:
        if len(replies) != 1:
            return False
        reply = replies[0].strip()
        if not reply or len(reply) > 24:
            return False
        if re.search(r"(?:回头|稍后|到时候|之后再|拍(?:下|一下|几)|发(?:你|给你)|我(?:去|会))", reply):
            return False
        return bool(
            re.search(r"[?？]$", reply)
            or re.search(r"(?:几点|哪(?:个|里|儿|边)?|谁|什么|啥|怎么|多久|多长|多少|几(?:点|个|天|次)?)", reply)
        )

    @staticmethod
    def _latest_request_target_text(user_prompt: str) -> str:
        match = re.search(
            r"<response_target_candidate\b[^>]*>(.*?)</response_target_candidate>",
            user_prompt,
            flags=re.DOTALL,
        )
        if match:
            target = re.sub(r"\s+", "", xml_unescape(match.group(1))).strip()
            labelled = re.search(r"目标原文[:：](.*?)(?:。|$)", target)
            return labelled.group(1).strip() if labelled else target
        return ""

    @staticmethod
    def _has_numeric_fact_choice(target: str) -> bool:
        """只识别把两个数字事实列为候选的提问，不拦截日常偏好选择。"""
        normalized = re.sub(r"\s+", "", target)
        if not normalized or not re.search(r"(?:还是|或者|或是)", normalized):
            return False
        if re.search(r"(?:想|喜欢|爱|要|选|挑|喝|吃|买|穿|玩|去).{0,12}(?:还是|或者|或是)", normalized):
            return False
        numeric_options = re.findall(
            r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|％|元|块|万|亿|岁|年|月|天|小时|分钟|层|楼|个|条|套|平|平方米)?",
            normalized,
        )
        return len(numeric_options) >= 2

    @staticmethod
    def _target_requires_evidence_guard(target: str) -> bool:
        """只对真正需要当前事实或本人动作的问题/请求收紧回复。"""
        normalized = re.sub(r"\s+", "", target)
        if not normalized:
            return False
        if re.search(r"(?:还请你吃饭|滚床单)", normalized):
            return False
        if re.search(
            r"(?:不会|不能|没法|不可能).{0,8}(?:给我|帮我|发我)",
            normalized,
        ):
            return False
        if re.search(
            r"(?:给我|帮我|麻烦|记得|发(?:我|一下|下)|拍(?:一下|下)|"
            r"查(?:一下|下)|看(?:一下|下))",
            normalized,
        ):
            return True
        if re.search(
            r"(?:你(?:要不要|要|想不想|愿不愿意).{0,10}(?:跟我)?一起|"
            r"要不要一起)",
            normalized,
        ):
            return True
        if re.search(
            r"(?:一起|跟我).{0,8}(?:吃饭|见面|碰面|约)",
            normalized,
        ):
            return True
        if re.search(
            r"你(?:们|家|叔叔|阿姨|爸|妈|哥|姐|弟|妹)?.{0,14}"
            r"(?:是不是|有没有|在不在|在哪|多少|几个|多久|什么时候|哪天|几点|"
            r"准备|打算|想不想|会不会|要不要)",
            normalized,
        ):
            return True
        if re.search(
            r"(?:是不是|有没有|是否|在不在|有.{0,20}吗$)",
            normalized,
        ):
            return True
        return bool(
            re.search(r"(?:多少|几(?:个|位|人|条|件|次|点|号|天|月|年|岁|层|套|份|张))", normalized)
            and re.search(r"(?:[?？]|啊$|吗$|呢$)", normalized)
        )

    @staticmethod
    def _is_clean_uncertainty_reply(replies: List[str]) -> bool:
        if len(replies) != 1:
            return False
        reply = replies[0].strip()
        if not re.search(r"(?:不知道|不清楚|不确定|说不准|没想好)", reply):
            return False
        return not re.search(
            r"(?:没(?:敢|细)?看|没(?:细)?问|没(?:细|具体)?说|问(?:下|一下|问)|回头|稍后|到时候|之后再|整理|拍(?:下|一下)?|发(?:你|给你)|确认|通知|处理|查(?:下|一下)?)",
            reply,
        )

    @staticmethod
    def _missing_evidence_fallback(target: str) -> List[str]:
        if re.search(r"(?:一起|吃饭|见面|碰面|约(?:一下)?)", target):
            if re.search(r"(?:今天|明天|早上|中午|下午|晚上|周末|星期|周[一二三四五六日天]|\d{1,2}[点:：])", target):
                return ["在哪边"]
            return ["几点"]
        if re.search(r"(?:发|拍)", target):
            return ["这会儿发不了"]
        if re.search(r"(?:给|帮|查|看一下|看下|记得|麻烦)", target):
            return ["这会儿不行"]
        if re.search(r"你.{0,6}(?:准备|打算|想不想|会不会|要不要|动(?:了|吗)?)", target):
            return ["还没定"]
        if re.search(r"你.{0,8}(?:多少|几个|多久|什么时候|哪天|几点)", target):
            return ["我也说不准"]
        return ["这我真不清楚"]

    def _apply_latest_request_evidence_guard(
        self,
        user_prompt: str,
        replies: List[str],
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """候选版确定性收敛：缺少直接证据时不允许编造本人行为或承诺。"""
        if not self.enable_latest_request_first or not replies:
            return replies

        decision: Dict[str, Any] = {}
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            parsed = self._extract_json(str(message.get("content") or ""))
            candidate = parsed.get("decision") if isinstance(parsed, dict) else None
            if isinstance(candidate, dict) and "evidence" in candidate:
                decision = candidate
                break

        evidence = str(decision.get("evidence") or "").strip()
        if not evidence:
            return replies
        normalized_evidence = re.sub(r"\s+", "", xml_unescape(evidence))
        target = self._latest_request_target_text(user_prompt)
        effective_missing = evidence.lower() == "missing"
        if evidence.lower().startswith("unread:") and target:
            cited = normalized_evidence.split(":", 1)[-1]
            if cited and (cited in target or target in cited):
                effective_missing = True

        state = self.last_latest_request_grounding
        state.update({
            "attempted": True,
            "status": "evidence_guard_passed",
            "evidence": evidence,
            "effective_missing": effective_missing,
            "reply_before": list(replies),
            "reply_after": list(replies),
        })
        if not effective_missing:
            return replies
        if not self._target_requires_evidence_guard(target):
            state["status"] = "missing_not_grounded_request"
            return replies
        if self._is_short_necessary_question(replies):
            state["status"] = "missing_kept_necessary_question"
            return replies
        if self._is_clean_uncertainty_reply(replies):
            state["status"] = "missing_kept_uncertainty"
            return replies

        guarded = self._missing_evidence_fallback(target)
        state["status"] = "missing_reply_constrained"
        state["reply_after"] = guarded
        return guarded

    def _active_reply_review_keys(
        self,
        user_prompt: str,
        replies: List[str],
        messages: List[Dict[str, Any]],
        *,
        is_group: bool,
    ) -> List[str]:
        """按 _generate 中的固定顺序返回本轮会真正发起 LLM 调用的后置审查键。

        条件与各检查的 gate 保持一致（不含 deadline 预算判断，预算由调用方统一检查）。
        """
        active: List[str] = []
        if self._latest_request_target_text(user_prompt):
            active.append("latest_request")
        if (self.enable_current_stance_continuity
                or self.enable_group_self_context_freshness_boundary
                or self.enable_group_adjacent_self_response) and is_group and replies:
            active.append("stance")
        if (self.enable_group_answer_authority_rewrite and is_group and replies
                and any("<group_target_action_contract" in str(message.get("content", ""))
                        for message in messages)):
            active.append("authority")
        return active

    def _run_combined_reply_review(
        self,
        user_prompt: str,
        replies: List[str],
        messages: List[Dict[str, Any]],
        deadline: float,
        overall_deadline: float,
        *,
        is_group: bool,
    ) -> Tuple[List[str], List[Dict[str, Any]]] | None:
        """>=2 个后置审查同时激活时合并为一次 LLM 调用；否则返回 None 走原顺序调用。

        仅支持可表达为「完整消息流 + 追加 user 审查指令」的检查；stance 的
        freshness_boundary 形态（独立 system + 聚焦上下文）不支持组合，返回 None 降级。
        应用顺序、rewrite 链式传递与观测 state 语义与顺序调用一致；未激活审查的
        state 仍由原方法写入（gate 直接返回，无额外 LLM 调用）。
        """
        budget_deadline = min(deadline, overall_deadline)
        if budget_deadline - time.time() < 1.0:
            return None
        active = self._active_reply_review_keys(user_prompt, replies, messages, is_group=is_group)
        if len(active) <= 1:
            return None
        if "stance" in active and self.enable_group_self_context_freshness_boundary:
            return None

        started_at = time.time()
        states: Dict[str, Dict[str, Any]] = {}
        for key in active:
            if key == "latest_request":
                state = {"attempted": False, "status": "not_needed", "reply_before": list(replies),
                         "reply_after": list(replies), "elapsed_ms": 0}
                self.last_latest_request_grounding = state
            elif key == "stance":
                generated_stance = self._current_stance_from_messages(messages)
                state = {"attempted": False, "status": "not_needed",
                         "generated_current_stance": generated_stance, "current_stance": "none",
                         "current_assignment": "none", "current_attitude": "none",
                         "reply_before": list(replies), "reply_after": list(replies),
                         "raw_response": "", "elapsed_ms": 0}
                self.last_current_stance_grounding = state
            else:
                state = {"attempted": False, "status": "not_needed", "reply_before": list(replies),
                         "reply_after": list(replies), "raw_response": "", "elapsed_ms": 0}
                self.last_group_answer_authority_review = state
            states[key] = state

        sections: List[str] = []
        if "latest_request" in active:
            sections.append(self._latest_request_review_prompt(replies, combined=True))
        if "stance" in active:
            sections.append(self._stance_review_prompt(
                replies, states["stance"]["generated_current_stance"], combined=True))
        if "authority" in active:
            sections.append(self._answer_authority_review_prompt(replies, combined=True))
        results_guide = {
            "latest_request": {"decision": "pass|rewrite", "reason": "一句话", "replies": ["修正回复"]},
            "stance": {"decision": "pass|rewrite", "current_assignment": "证明最新分工的原话；没有则填 none",
                       "current_attitude": "证明本人当前态度的本人原话；没有则填 none",
                       "reason": "两项成立的直接证据；没有则留空", "replies": ["修正回复"]},
            "authority": {"decision": "pass|rewrite", "reason": "无依据的具体结论；没有则留空",
                          "replies": ["修正回复"]},
        }
        review_prompt = (
            f"<combined_reply_review>\n"
            f"以下有 {len(sections)} 个相互独立的回复审查指令。每个审查只按自己标签内的标准判断，"
            "互不替代、互不让步；全部审查针对同一份 current_replies。\n"
            "逐条完成全部审查后只输出一个 JSON：{\"results\":{...}}，每个键的结构如下："
            f"{json.dumps({key: results_guide[key] for key in active}, ensure_ascii=False)}\n"
            "只输出实际执行的审查键。\n"
            + "\n".join(sections)
            + "\n</combined_reply_review>"
        )
        review_messages = [dict(message) for message in messages]
        review_messages.append({"role": "user", "content": review_prompt})
        for key in active:
            states[key]["attempted"] = True
            states[key]["status"] = "requested"
        try:
            raw = self.llm_client.chat(
                messages=review_messages,
                tools=None,
                max_tokens=400,
                timeout=max(1.0, budget_deadline - time.time()),
                response_format={"type": "json_object"},
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            review_messages.append({"role": "assistant", "content": text or ""})
            data = self._extract_json(text or "")
            results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(results, dict):
                results = {}
            for key in active:
                state = states[key]
                if key in ("stance", "authority"):
                    state["raw_response"] = text or ""
                sub = results.get(key)
                if not isinstance(sub, dict):
                    state["status"] = "invalid_response"
                    continue
                decision = sub.get("decision")
                if key in ("stance", "authority") and isinstance(decision, str):
                    decision = decision.strip().lower()
                checked_replies = self._parse_replies(json.dumps(sub, ensure_ascii=False))
                if key == "stance":
                    state["current_assignment"] = str(sub.get("current_assignment") or "none").strip()
                    state["current_attitude"] = str(sub.get("current_attitude") or "none").strip()
                    state["answered_self_quote"] = str(sub.get("answered_self_quote") or "none").strip()
                if decision == "rewrite" and checked_replies:
                    replies = checked_replies
                    state["status"] = "rewritten"
                elif decision == "pass":
                    state["status"] = "passed"
                else:
                    state["status"] = "invalid_response"
                state["reason"] = sub.get("reason", "")
        except Exception as exc:
            for key in active:
                states[key]["status"] = "error"
                states[key]["error"] = f"{type(exc).__name__}: {exc}"
            _logger.warning("[CombinedReplyReview] failed: %s", exc)

        for key in active:
            states[key]["reply_after"] = list(replies)
            states[key]["elapsed_ms"] = round((time.time() - started_at) * 1000)
        # 未激活审查的观测 state 仍由原方法写入（gate 直接返回，无额外 LLM 调用）
        if "latest_request" not in active:
            _, review_messages = self._check_latest_request_reply(
                user_prompt, replies, review_messages, deadline)
        if "stance" not in active:
            _, review_messages = self._check_current_stance_reply(
                replies, review_messages, overall_deadline, is_group=is_group)
        if "authority" not in active:
            _, review_messages = self._check_group_answer_authority_reply(
                replies, review_messages, overall_deadline, is_group=is_group)
        return replies, review_messages

    def _latest_request_review_prompt(self, replies: List[str], *, combined: bool = False) -> str:
        if combined:
            output_clause = (
                "将本检查结论写入 results.latest_request 字段："
                "{\"decision\":\"pass|rewrite\",\"reason\":\"一句话\","
                "\"replies\":[\"修正回复\"]}。pass 时 replies 必须原样返回。\n"
            )
        else:
            output_clause = (
                "输出 JSON：{\"decision\":\"pass|rewrite\",\"reason\":\"一句话\","
                "\"replies\":[\"修正回复\"]}。pass 时 replies 必须原样返回。\n"
            )
        return (
            "<latest_request_reply_check>\n"
            "只检查当前回复是否完成了当前对话中最近仍未解决、且确实指向本人的诉求。\n"
            "只在以下任一情况 rewrite：\n"
            "1. 回复转去回答旧话题、调侃、建议或总结，实际漏掉该诉求；\n"
            "2. 回复与当前上下文或可信 Wiki 存在可以直接指出的矛盾。\n"
            "仅仅缺少可见依据、措辞不自然、信息不够丰富，都不属于硬伤，不能因此 rewrite。"
            "不要因为措辞、幽默程度或信息不够丰富而 rewrite。问题明确问别人时 pass。"
            "rewrite 时先直接回应最新诉求，随后立即结束，不得转回旧话题。\n"
            + output_clause +
            f"<current_replies>{xml_escape(' | '.join(replies))}</current_replies>\n"
            "</latest_request_reply_check>"
        )

    def _check_latest_request_reply(
        self,
        user_prompt: str,
        replies: List[str],
        messages: List[Dict[str, Any]],
        deadline: float,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """候选版窄检查：只修正漏回最新诉求或无证据代答。"""
        started_at = time.time()
        state = {
            "attempted": False,
            "status": "not_needed",
            "reply_before": list(replies),
            "reply_after": list(replies),
            "elapsed_ms": 0,
        }
        self.last_latest_request_grounding = state
        if not self._latest_request_target_text(user_prompt):
            state["status"] = "no_request_candidate"
            return replies, messages
        if deadline - time.time() < 1.0:
            state["status"] = "insufficient_time_budget"
            return replies, messages

        review_prompt = self._latest_request_review_prompt(replies)
        review_messages = [dict(message) for message in messages]
        review_messages.append({"role": "user", "content": review_prompt})
        state["attempted"] = True
        state["status"] = "requested"
        try:
            raw = self.llm_client.chat(
                messages=review_messages,
                tools=None,
                max_tokens=400,
                timeout=max(1.0, deadline - time.time()),
                response_format={"type": "json_object"},
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            data = self._extract_json(text or "")
            decision = data.get("decision") if isinstance(data, dict) else None
            checked_replies = self._parse_replies(text or "")
            review_messages.append({"role": "assistant", "content": text or ""})
            if decision == "rewrite" and checked_replies:
                replies = checked_replies
                state["status"] = "rewritten"
            elif decision == "pass":
                state["status"] = "passed"
            else:
                state["status"] = "invalid_response"
            state["reason"] = data.get("reason", "") if isinstance(data, dict) else ""
        except Exception as exc:
            state["status"] = "error"
            state["error"] = f"{type(exc).__name__}: {exc}"
            _logger.warning("[LatestRequestCheck] failed: %s", exc)

        state["reply_after"] = list(replies)
        state["elapsed_ms"] = round((time.time() - started_at) * 1000)
        return replies, review_messages

    def _check_group_answer_authority_reply(
        self,
        replies: List[str],
        messages: List[Dict[str, Any]],
        deadline: float,
        *,
        is_group: bool,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Remove only unsupported concrete conclusions from a triggered group reply."""
        started_at = time.time()
        state = {
            "attempted": False,
            "status": "not_needed",
            "reply_before": list(replies),
            "reply_after": list(replies),
            "raw_response": "",
            "elapsed_ms": 0,
        }
        self.last_group_answer_authority_review = state
        if not self.enable_group_answer_authority_rewrite:
            state["status"] = "candidate_disabled"
            return replies, messages
        if not is_group:
            state["status"] = "private_chat"
            return replies, messages
        if not replies:
            state["status"] = "empty_reply"
            return replies, messages
        has_contract = any(
            "<group_target_action_contract" in str(message.get("content", ""))
            for message in messages
        )
        if not has_contract:
            state["status"] = "contract_absent"
            return replies, messages
        if deadline - time.time() < 1.0:
            state["status"] = "insufficient_time_budget"
            return replies, messages

        review_prompt = self._answer_authority_review_prompt(replies)
        review_messages = [dict(message) for message in messages]
        review_messages.append({"role": "user", "content": review_prompt})
        state["attempted"] = True
        state["status"] = "requested"
        try:
            raw = self.llm_client.chat(
                messages=review_messages,
                tools=None,
                max_tokens=300,
                timeout=max(1.0, deadline - time.time()),
                response_format={"type": "json_object"},
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            state["raw_response"] = text or ""
            data = self._extract_json(text or "")
            decision = data.get("decision") if isinstance(data, dict) else None
            if isinstance(decision, str):
                decision = decision.strip().lower()
            checked_replies = self._parse_replies(text or "")
            review_messages.append({"role": "assistant", "content": text or ""})
            if decision == "rewrite" and checked_replies:
                replies = checked_replies
                state["status"] = "rewritten"
            elif decision == "pass":
                state["status"] = "passed"
            else:
                state["status"] = "invalid_response"
            state["reason"] = data.get("reason", "") if isinstance(data, dict) else ""
        except Exception as exc:
            state["status"] = "error"
            state["error"] = f"{type(exc).__name__}: {exc}"
            _logger.warning("[GroupAnswerAuthorityCheck] failed: %s", exc)

        state["reply_after"] = list(replies)
        state["elapsed_ms"] = round((time.time() - started_at) * 1000)
        return replies, review_messages

    def _answer_authority_review_prompt(self, replies: List[str], *, combined: bool = False) -> str:
        if combined:
            output_clause = (
                "将本检查结论写入 results.authority 字段："
                '{"decision":"pass|rewrite","reason":"无依据的具体结论；没有则留空",'
                '"replies":["修正回复"]}。\n'
            )
        else:
            output_clause = (
                "pass 时 replies 必须逐字原样返回。输出 JSON："
                '{"decision":"pass|rewrite","reason":"无依据的具体结论；没有则留空",'
                '"replies":["修正回复"]}。\n'
            )
        return (
            "<group_answer_authority_check>\n"
            "只检查 current_replies 是否在没有独立依据时，替本人、他人、群体或组织确认、否认或承诺了"
            "当前的具体状态、计划、规则、建议或结论。不要评价措辞、长度、幽默和是否正面回答。\n"
            "独立依据只包括前文较早的本人直接发言、可信 memory 或 tool 结果。"
            "目标消息中的问题、假设、要求或建议不是答案证据。"
            "本人较早表达的泛化态度，也不能证明当前这件事的具体状态、计划或结论。\n"
            "如果回复有直接依据，或只是反问、澄清、接梗、调侃、说明不知道、自然回避，decision 必须为 pass。"
            "不要因为回复没有正面作答而 rewrite。\n"
            "只有能逐句指出无依据的具体确认、否认或承诺时才允许 rewrite。rewrite 只能删除该无依据结论，"
            "同时保持原回复的互动动作、话题、语气和关系位置；不得改成通用拒绝，不得编造新事实。\n"
            + output_clause +
            f"<current_replies>{xml_escape(' | '.join(replies))}</current_replies>\n"
            "</group_answer_authority_check>"
        )

    @classmethod
    def _current_stance_from_messages(cls, messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            parsed = cls._extract_json(str(message.get("content") or ""))
            decision = parsed.get("decision") if isinstance(parsed, dict) else None
            if not isinstance(decision, dict) or "current_stance" not in decision:
                continue
            return str(decision.get("current_stance") or "none").strip()
        return "none"

    @staticmethod
    def _current_stance_review_context(messages: List[Dict[str, Any]]) -> str:
        """Keep only the evidence needed by the group self-context review."""
        source = ""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            if "<unread>" in content:
                source = content
                break

        def _section(tag: str) -> str:
            match = re.search(
                rf"<{tag}\b[^>]*>.*?</{tag}>",
                source,
                flags=re.DOTALL,
            )
            return match.group(0) if match else f"<{tag}></{tag}>"

        def _speaker(item: str) -> str:
            text = re.sub(r"<[^>]+>", "", item).strip()
            match = re.match(r"([^（\n]{1,80})（\d{4}-\d{2}-\d{2}", text)
            if match:
                return match.group(1).strip()
            match = re.match(r"([^：:\n]{1,80})[：:]", text)
            return match.group(1).strip() if match else ""

        ordered_messages: List[Tuple[int, int, str, str]] = []
        sequence = 0
        for tag in ("history", "consumed_self_replies"):
            for item in re.findall(
                r"<message\b[^>]*>.*?</message>",
                _section(tag),
                flags=re.DOTALL,
            ):
                index_match = re.search(r'\bindex="(\d+)"', item)
                index = int(index_match.group(1)) if index_match else 10**9
                ordered_messages.append((index, sequence, tag, item))
                sequence += 1
        ordered_messages.sort(key=lambda value: (value[0], value[1]))
        unread_messages = re.findall(
            r"<message\b[^>]*>.*?</message>",
            _section("unread"),
            flags=re.DOTALL,
        )
        unread_speaker = _speaker(unread_messages[0]) if unread_messages else ""
        previous_speaker_segment: List[str] = []
        message_before_previous_segment = ""
        if unread_speaker:
            found = False
            segment_start = -1
            for position in range(len(ordered_messages) - 1, -1, -1):
                item = ordered_messages[position][3]
                if _speaker(item) == unread_speaker:
                    previous_speaker_segment.append(item)
                    found = True
                    segment_start = position
                elif found:
                    break
            previous_speaker_segment.reverse()
            if segment_start > 0:
                message_before_previous_segment = ordered_messages[segment_start - 1][3]

        recent_sequences = {sequence for _, sequence, _, _ in ordered_messages[-12:]}
        older_self_messages = [
            item
            for _, sequence, tag, item in ordered_messages
            if tag == "consumed_self_replies" and sequence not in recent_sequences
        ]
        reply_chain = (
            '<reply_chain order="old_to_new">\n'
            + _section("current_stance_evidence")
            + (
                "\n<message_before_same_speaker_segment>\n"
                + message_before_previous_segment
                + "\n</message_before_same_speaker_segment>"
                if message_before_previous_segment
                else "\n<message_before_same_speaker_segment></message_before_same_speaker_segment>"
            )
            + (
                f'\n<same_speaker_previous_segment speaker="{xml_escape(unread_speaker)}">\n'
                + "\n".join(previous_speaker_segment)
                + "\n</same_speaker_previous_segment>"
                if previous_speaker_segment
                else "\n<same_speaker_previous_segment></same_speaker_previous_segment>"
            )
            + "\n"
            + _section("unread")
            + "\n</reply_chain>"
        )
        return (
            "<current_stance_review_context>\n"
            + reply_chain
            + (
                "\n<older_self_material reuse=\"forbidden\">\n"
                + "\n".join(older_self_messages)
                + "\n</older_self_material>"
                if older_self_messages
                else ""
            )
            + "\n</current_stance_review_context>"
        )

    def _stance_review_prompt(self, replies: List[str], generated_stance: str, *, combined: bool = False) -> str:
        if combined:
            output_clause = (
                "将本检查结论写入 results.stance 字段："
                "{\"decision\":\"pass|rewrite\",\"current_assignment\":\"证明最新分工的原话；没有则填 none\","
                "\"current_attitude\":\"证明本人当前态度的本人原话；没有则填 none\","
                "\"reason\":\"两项成立的直接证据；没有则留空\","
                "\"replies\":[\"修正回复\"]}。\n"
            )
        else:
            output_clause = (
                "输出 JSON：{\"decision\":\"pass|rewrite\",\"current_assignment\":\"证明最新分工的原话；没有则填 none\","
                "\"current_attitude\":\"证明本人当前态度的本人原话；没有则填 none\","
                "\"reason\":\"两项成立的直接证据；没有则留空\","
                "\"replies\":[\"修正回复\"]}。\n"
            )
        return (
            "<current_stance_reply_check>\n"
            "必须根据前面的完整群聊独立判断，不得相信第一次生成填写的 decision.current_stance。\n"
            "先分别抽取两个互不替代的状态：\n"
            "1. current_assignment：同一处境中，最新明确的谁先到、谁执行什么、人物之间的先后关系。"
            "证据可以来自本人或其他参与者；后续明确消息改变较早分工后，只保留新分工。没有则填 none。\n"
            "2. current_attitude：同一处境中，本人最近明确表达的催促、反对、同意等待或其他当前态度。"
            "只能以本人原话为证据；后续本人消息改变较早态度后，只保留新态度。没有则填 none。\n"
            "只有当前回复与 current_assignment 或 current_attitude 至少一项存在可以逐句指出的直接矛盾时，"
            "才允许 rewrite；rewrite 只能修正冲突，并且必须同时保留另一项仍成立的状态。\n"
            "例如，已明确本人先到并先点时，不得回复让其他人先点；本人刚在催对方快点时，不得回复让对方慢慢来。"
            "不要求机械复述“快点”或任何完整旧句。\n"
            "若两项都为 none，或回复没有直接写反任何一项，decision 必须为 pass。\n"
            "措辞、长度、幽默、简略、不够自然或没有重复全部原话，都不能作为 rewrite 理由。\n"
            "不得补充上下文没有的新事实。pass 时 replies 必须逐字原样返回。\n"
            + output_clause +
            f"<untrusted_first_generation_current_stance>{xml_escape(generated_stance)}</untrusted_first_generation_current_stance>\n"
            f"<current_replies>{xml_escape(' | '.join(replies))}</current_replies>\n"
            "</current_stance_reply_check>"
        )

    def _check_current_stance_reply(
        self,
        replies: List[str],
        messages: List[Dict[str, Any]],
        deadline: float,
        *,
        is_group: bool,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """只在群聊当前分工或本人当前态度被写反时修正回复。"""
        started_at = time.time()
        generated_stance = self._current_stance_from_messages(messages)
        state = {
            "attempted": False,
            "status": "not_needed",
            "generated_current_stance": generated_stance,
            "current_stance": "none",
            "current_assignment": "none",
            "current_attitude": "none",
            "reply_before": list(replies),
            "reply_after": list(replies),
            "raw_response": "",
            "elapsed_ms": 0,
        }
        self.last_current_stance_grounding = state
        if not (
            self.enable_current_stance_continuity
            or self.enable_group_self_context_freshness_boundary
            or self.enable_group_adjacent_self_response
        ):
            state["status"] = "candidate_disabled"
            return replies, messages
        if not is_group:
            state["status"] = "private_chat"
            return replies, messages
        if not replies:
            state["status"] = "empty_reply"
            return replies, messages
        if deadline - time.time() < 1.0:
            state["status"] = "insufficient_time_budget"
            return replies, messages

        if self.enable_group_self_context_freshness_boundary:
            focused_context = self._current_stance_review_context(messages)
            adjacent_response_rule = (
                "紧邻本人消息后的短感叹、赞同、邀请、追问、调侃或顺势继续，不因形式简短而排除："
                "只有 message_before_same_speaker_segment 是本人消息，并且 same_speaker_previous_segment 或 unread "
                "能指出与该句的具体接话关系时，才选择这 1 条本人原话。仅仅时间相邻、同一话题或普通感叹仍填 none。"
                if self.enable_group_adjacent_self_response
                else ""
            )
            no_response_rule = (
                "仅话题相同、用词相同、时间接近、图片、表情，或找不到具体回应句的反应，"
                if self.enable_group_adjacent_self_response
                else "仅话题相同、用词相同、时间接近、短感叹、图片、表情或没有明确指向的反应，"
            )
            review_prompt = (
                "<current_stance_reply_check>\n"
                "必须只根据 current_stance_review_context 独立判断，不得相信第一次生成填写的 "
                "decision.directly_answered_self_quote。\n"
                "分成两个阶段，顺序不可交换。\n"
                "阶段1只判断接话关系：此时必须完全忽略 current_replies，不得根据候选回复是否冲突来反推接话关系。"
                "判断本批 unread 是否明确否定、纠正、回答、挑战、辩解或反驳本人之前说过的某一句话。"
                "不要求 unread 复述本人原话；若 unread 是紧接该原话的辩解或反驳，并且能指出 unread 中具体哪一句"
                "只有结合该本人原话才成立，也算正在回应本人。"
                "还允许向前追溯同一发言人的紧邻一段消息：只有前一段能指出具体哪一句在回应本人原话，"
                "并且 unread 能指出具体哪一句在延续或修正前一段，才算间接回应本人；两处证据缺一不可。"
                "只要上述接话关系成立，answered_self_quote 就逐字填写被回应的本人原话；"
                "有多条时只选时间最新的1条。\n"
                + adjacent_response_rule + "\n"
                + no_response_rule
                + "answered_self_quote 必须填 none。\n"
                "阶段2才读取 current_replies 并检查候选回复：如果 answered_self_quote 不是 none，当前回复不得把该原话表达的"
                "事实、判断或态度写反，也不得把对方的身份或立场认领为本人；"
                "如果 answered_self_quote 是 none，当前回复不得重新使用本人已结束的旧话题、旧人物或旧梗。\n"
                "只有能逐句指出上述一种违规时才允许 rewrite；rewrite 只修正这项违规，"
                "不得照抄本人旧话、补充上下文没有的新事实，或为了更自然、更幽默而改写。\n"
                "没有上述违规时 decision 必须为 pass；pass 时 replies 必须逐字原样返回。\n"
                "输出 JSON：{\"decision\":\"pass|rewrite\","
                "\"answered_self_quote\":\"被回应的本人原话；没有则填 none\","
                "\"reason\":\"当前回复与该原话的直接冲突，或重新使用的旧素材；没有则留空\","
                "\"replies\":[\"修正回复\"]}。\n"
                f"<current_replies>{xml_escape(' | '.join(replies))}</current_replies>\n"
                "</current_stance_reply_check>"
            )
        else:
            review_prompt = self._stance_review_prompt(replies, generated_stance)
        if self.enable_group_self_context_freshness_boundary:
            review_messages = [
                {
                    "role": "system",
                    "content": (
                        "你只负责核对群聊回复有没有写反本人当前立场，或带回已经结束的本人旧素材。"
                        "只能使用用户消息中提供的聚焦证据，严格输出 JSON。"
                    ),
                },
                {"role": "user", "content": focused_context + "\n" + review_prompt},
            ]
        else:
            review_messages = [dict(message) for message in messages]
            review_messages.append({"role": "user", "content": review_prompt})
        state["attempted"] = True
        state["status"] = "requested"
        try:
            raw = self.llm_client.chat(
                messages=review_messages,
                tools=None,
                max_tokens=400,
                timeout=max(1.0, deadline - time.time()),
                response_format={"type": "json_object"},
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            state["raw_response"] = text or ""
            data = self._extract_json(text or "")
            decision = data.get("decision") if isinstance(data, dict) else None
            if isinstance(decision, str):
                decision = decision.strip().lower()
            checked_assignment = data.get("current_assignment") if isinstance(data, dict) else None
            checked_attitude = data.get("current_attitude") if isinstance(data, dict) else None
            answered_self_quote = data.get("answered_self_quote") if isinstance(data, dict) else None
            state["current_assignment"] = str(checked_assignment or "none").strip()
            state["current_attitude"] = str(checked_attitude or "none").strip()
            state["answered_self_quote"] = str(answered_self_quote or "none").strip()
            checked_replies = self._parse_replies(text or "")
            review_messages.append({"role": "assistant", "content": text or ""})
            if decision == "rewrite" and checked_replies:
                replies = checked_replies
                state["status"] = "rewritten"
            elif decision == "pass":
                state["status"] = "passed"
            else:
                state["status"] = "invalid_response"
            state["reason"] = data.get("reason", "") if isinstance(data, dict) else ""
        except Exception as exc:
            state["status"] = "error"
            state["error"] = f"{type(exc).__name__}: {exc}"
            _logger.warning("[CurrentStanceCheck] failed: %s", exc)

        state["reply_after"] = list(replies)
        state["elapsed_ms"] = round((time.time() - started_at) * 1000)
        return replies, review_messages

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
        if skill_name in self.skill_content_overrides:
            return self.skill_content_overrides[skill_name].strip()
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
                timeout=min(5.0, max(1.0, deadline - time.time())) if deadline is not None else 5.0,
                response_format={"type": "json_object"},
                max_retries=0,
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
            prompt = "<instructions><persona>你是示例用户甲本人。用户不是在跟AI聊天，是在微信上给示例用户甲发消息。</persona></instructions>"

        return prompt

    @staticmethod
    def _with_group_self_context_freshness_boundary(prompt: str) -> str:
        """Allow exactly one directly answered self quote without reopening old material."""
        replacements = (
            (
                "<ineligible>role=self 历史已经用作笑点的任何素材；即使 unread 再次引用、推进或反驳也不重新开放。</ineligible>",
                "<ineligible>role=self 历史已经用作笑点的任何素材。decision.directly_answered_self_quote 为 none 时，本规则完整适用，不开放任何本人历史素材。该字段逐字选中 current_stance_evidence 中一条本人原话时，仅允许把该原句表达的当前事实、状态、计划、限制或判断作为背景；该原句中的具体笑点、比喻、数字梗、人物标签、虚构关系，以及其他 role=self 历史仍不可使用。</ineligible>",
            ),
            (
                "<rule>不得围绕 ineligible 素材新增动作、状态或场景；它不能成为回复的主语、宾语、修饰对象、背景或隐含前提。</rule>",
                "<rule>不得围绕 ineligible 素材新增动作、状态或场景；它不能成为回复的主语、宾语、修饰对象、背景或隐含前提。directly_answered_self_quote 选中原句时，也只能保持该原句表达的当前信息连续，不得照抄原句、复用旧笑点、带回原句之外的旧人物或旧事件，或凭空添加新事实。</rule>",
            ),
        )
        for old, new in replacements:
            if old not in prompt:
                raise ValueError("group self-context freshness boundary did not match persona prompt")
            prompt = prompt.replace(old, new, 1)
        schema_replacements = (
            (
                '"decision":{"conflict_type"',
                '"decision":{"directly_answered_self_quote":"none","conflict_type"',
            ),
            (
                '\\"decision\\":{\\"conflict_type\\"',
                '\\"decision\\":{\\"directly_answered_self_quote\\":\\"none\\",\\"conflict_type\\"',
            ),
        )
        for old, new in schema_replacements:
            if old in prompt:
                prompt = prompt.replace(old, new, 1)
                break
        else:
            raise ValueError("group self-context freshness boundary did not match output schema")
        quote_field = (
            '<field name="directly_answered_self_quote">只能填写 none，或从 current_stance_evidence '
            '逐字复制 1 条本人原话。本批 unread 中任一条明确否定、纠正、回答或挑战某条本人原话时，必须选择被回应的原话；'
            '若多条本人原话都可能相关，只选 unread 实际回应且时间最新的 1 条；'
            '仅话题相同、用词相同、时间接近、短感叹、图片、表情或没有明确指向的反应都必须填 none。不得概括、改写、拼接多条或引用他人消息。</field>'
        )
        marker = '<decision_fields>'
        if marker not in prompt:
            raise ValueError("group self-context freshness boundary did not match decision fields")
        prompt = prompt.replace(marker, marker + "\n      " + quote_field, 1)
        return prompt

    @staticmethod
    def _with_group_unified_relevant_self_schema(prompt: str) -> str:
        """Select one relevant self message directly from chronological group history."""
        schema_replacements = (
            (
                '"decision":{"conflict_type"',
                '"decision":{"directly_answered_self_quote":"none","conflict_type"',
            ),
            (
                '\\"decision\\":{\\"conflict_type\\"',
                '\\"decision\\":{\\"directly_answered_self_quote\\":\\"none\\",\\"conflict_type\\"',
            ),
        )
        for old, new in schema_replacements:
            if old in prompt:
                prompt = prompt.replace(old, new, 1)
                break
        else:
            raise ValueError("group unified relevant-self schema did not match output schema")
        quote_field = (
            '<field name="directly_answered_self_quote">只能填写 none，或从 history 中 role=self '
            '消息逐字复制 1 条本人原话。只有本批 unread 具体回应这条原话时才选择；若后面的本人消息已经更新或否定它，'
            '选择仍成立且时间最新的 1 条。仅话题、用词、人物相同或时间接近都必须填 none；'
            '不得概括、改写、拼接多条或引用他人消息。</field>'
        )
        marker = '<decision_fields>'
        if marker not in prompt:
            raise ValueError("group unified relevant-self schema did not match decision fields")
        return prompt.replace(marker, marker + "\n      " + quote_field, 1)

    @staticmethod
    def _with_group_adjacent_self_response(prompt: str) -> str:
        """Allow one adjacent self quote when the next group turn actually responds to it."""
        old = (
            "仅话题相同、用词相同、时间接近、短感叹、图片、表情或没有明确指向的反应都必须填 none。"
        )
        new = (
            "短感叹、赞同、邀请、追问、调侃或顺势继续不自动填 none：仅当它在消息顺序上紧跟本人原话，"
            "且能指出它与该原话的具体接话关系时，才逐字选择这 1 条本人原话。"
            "仅话题相同、用词相同、时间接近、图片、表情或没有具体接话关系的反应仍必须填 none。"
        )
        if old not in prompt:
            raise ValueError("group adjacent self response did not match freshness prompt")
        return prompt.replace(old, new, 1)

    @staticmethod
    def _with_latest_request_evidence_schema(prompt: str) -> str:
        """Make the candidate's one-pass decision expose the evidence it actually used."""
        evidence_rule = (
            '只能填写 missing，或填写一个直接支持答案的来源标签和原句，格式为 '
            'unread: 原句、history: 原句、persona: 原句、memory: 原句、tool: 原句。'
            '不得填写 supported、unsupported 或 not_applicable；'
            '不能直接支持答案的相关话题也必须填 missing。'
        )
        replacements = (
            ('"evidence":"not_applicable"', '"evidence":"missing"'),
            ('\\"evidence\\":\\"not_applicable\\"', '\\"evidence\\":\\"missing\\"'),
            (
                '<field name="evidence">只能是 supported、unsupported、not_applicable。</field>',
                f'<field name="evidence">{evidence_rule}</field>',
            ),
            (
                '<field name=\\"evidence\\">只能是 supported、unsupported、not_applicable。</field>',
                f'<field name=\\"evidence\\">{evidence_rule}</field>',
            ),
        )
        for old, new in replacements:
            prompt = prompt.replace(old, new, 1)
        return prompt

    @staticmethod
    def _with_current_stance_schema(prompt: str) -> str:
        """Require the candidate to identify current self stance before replying."""
        replacements = (
            (
                '"decision":{"conflict_type"',
                '"decision":{"current_stance":"none","conflict_type"',
            ),
            (
                '\\"decision\\":{\\"conflict_type\\"',
                '\\"decision\\":{\\"current_stance\\":\\"none\\",\\"conflict_type\\"',
            ),
        )
        for old, new in replacements:
            if old in prompt:
                prompt = prompt.replace(old, new, 1)
                break
        stance_field = (
            '<field name="current_stance">只能填写 none，或从 current_stance_evidence '
            '逐字引用最新且仍同时成立的 1 至 3 条本人原话；多条之间用“ || ”分隔。先按消息顺序判断：本人后说的话若更新了人物、行动或先后关系，'
            '较早安排失效，不得引用；不得把不同阶段或互相冲突的安排拼在一起。只有本人刚明确说出自己的安排、行动限制或时间压力，'
            '且 unread 正在直接要求、建议或提出方案改变这项安排时，才可引用；否则必须填写 none。'
            '调侃、预测、假设、重复本人态度、询问看法和一般偏好都不触发；不得概括、猜测或引用他人消息。</field>'
        )
        marker = '<decision_fields>'
        if marker in prompt:
            prompt = prompt.replace(marker, marker + "\n      " + stance_field, 1)
        response_target_field = (
            '<field name="response_target">填写本轮回复要完成的唯一作用，例如承认并纠正、针对本轮措辞幽默反击且不认领指控、简短纠正身份，或停止交流。</field>'
        )
        stance_response_target_field = (
            '<field name="response_target">current_stance 不是 none 时，先写“拒绝改变安排：”并说明回复如何直接拒绝或纠正对方提出的改变；'
            '必须同时满足 current_stance 中所有用“ || ”分隔的原话，不得只延续其中一条；'
            '不得顺从对方、另起话题，也不得加强成原话没有的绝对限制或补充新事实。生成每条 reply 后必须逐条核对：若接受了对方提出的改变、'
            '与任一原话矛盾或增加无依据事实，必须丢弃并按全部 current_stance 重写。'
            'current_stance 为 none 时，填写本轮回复要完成的唯一作用。</field>'
        )
        prompt = prompt.replace(response_target_field, stance_response_target_field, 1)
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
    def _append_request_tail_sections(prompt: str, sections: List[str]) -> str:
        """把最终响应合同追加为 request 的最后子元素。"""
        closing = "</request>"
        if not sections or not prompt.rstrip().endswith(closing):
            return prompt
        body = prompt.rstrip()[:-len(closing)].rstrip()
        return body + "\n\n" + "\n\n".join(sections) + "\n" + closing

    @staticmethod
    def _has_explicit_third_party_introduction(messages: List[ChatMessage]) -> bool:
        """Detect an explicit @-named introduction, not an ordinary mention."""
        mention = re.compile(
            r"@(?:\[[^\]\r\n]{1,20}\]\s*)?"
            r"(?:[A-Za-z0-9_.\-·]{1,32}(?=\s|是|叫|担任|负责|$)|[\u4e00-\u9fff]{1,12}(?=\s|$))"
        )
        role_intro = re.compile(
            r"(?:是[^。！？!?\r\n]{0,32}(?:负责(?:人|的)?|大佬|老板|领导|同事|朋友|老师|医生|律师|客户|甲方|产品|运营|开发|设计|财务|法务|人事|HR)"
            r"|叫(?:做)?[^。！？!?\r\n]{1,20}"
            r"|担任[^。！？!?\r\n]{1,24}"
            r"|负责[^。！？!?\r\n]{1,24})"
        )
        addressed_intro = re.compile(
            r"^\s*(?:[A-Za-z0-9_.\-·]{1,32}|[\u4e00-\u9fff]{1,12})"
            r"(?:是[^。！？!?\r\n]{0,32}(?:负责(?:人|的)?|大佬|老板|领导|同事|朋友|老师|医生|律师|客户|甲方|产品|运营|开发|设计|财务|法务|人事|HR)"
            r"|叫(?:做)?[^。！？!?\r\n]{1,20}"
            r"|担任[^。！？!?\r\n]{1,24}"
            r"|负责[^。！？!?\r\n]{1,24})"
        )
        for message in messages:
            text = str(getattr(message, "text", "") or "").strip()
            matched = mention.search(text)
            remainder = text[matched.end():] if matched else ""
            if matched and (role_intro.search(remainder) or addressed_intro.search(remainder)):
                return True
        return False

    @staticmethod
    def _group_target_action_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, str]]:
        """Return a conservative, visible-input-only target/action contract."""
        target: Optional[ChatMessage] = None
        for message in reversed(unreplied):
            text = str(getattr(message, "text", "") or "").strip()
            if not text:
                continue
            if (
                getattr(message, "is_at_me", False)
                or _looks_like_group_question(text)
                or _GROUP_OPERATION_RE.search(text)
                or _looks_like_group_coordination_request(text)
                or _looks_like_group_status_or_detail_question(text)
            ):
                target = message
                break
        if target is None:
            return None

        text = str(target.text or "").strip()
        has_mention = bool(_GROUP_AT_MENTION_RE.search(text))
        explicitly_mentions_self = bool(_GROUP_SELF_MENTION_RE.search(text))
        addressed_to = "uncertain"
        if getattr(target, "is_at_me", False) or explicitly_mentions_self:
            addressed_to = "self"
        elif has_mention:
            addressed_to = "named_other"
        elif _GROUP_OPEN_ADDRESS_RE.search(text):
            addressed_to = "group"
        elif _GROUP_SECOND_PERSON_RE.search(text):
            target_index = next(
                (index for index in range(len(all_messages) - 1, -1, -1)
                 if all_messages[index] is target),
                len(all_messages) - 1,
            )
            recent = all_messages[max(0, target_index - 5):target_index]
            last_self_index = next(
                (index for index in range(len(recent) - 1, -1, -1)
                 if recent[index].sender_type == SenderType.SELF),
                None,
            )
            if last_self_index is not None:
                intervening_senders = {
                    message.sender
                    for message in recent[last_self_index + 1:]
                    if message.sender_type == SenderType.OTHER
                }
                if not intervening_senders or intervening_senders == {target.sender}:
                    addressed_to = "self"
        elif _looks_like_explicit_request(text) or _GROUP_OPERATION_RE.search(text):
            addressed_to = "group"

        is_question = _looks_like_group_question(text)
        if addressed_to == "named_other":
            required_action = "react_only"
        elif is_question and _GROUP_STATUS_QUESTION_RE.search(text):
            required_action = "report_own_status"
        elif is_question and _GROUP_TIME_PLACE_QUESTION_RE.search(text):
            required_action = "give_concrete_time_or_place"
        elif _GROUP_OPERATION_RE.search(text):
            required_action = "perform_group_operation"
        elif _looks_like_group_coordination_request(text):
            required_action = "coordinate_participation"
        elif is_question:
            required_action = "answer"
        elif getattr(target, "is_at_me", False):
            required_action = "react_only"
        else:
            required_action = "acknowledge"

        return {
            "speaker": str(target.sender or "unknown"),
            "addressed_to": addressed_to,
            "required_action": required_action,
            "answer_authority": (
                "named_person_only"
                if addressed_to == "named_other"
                else "independent_support_required"
            ),
            "action_evidence": text,
            "must_not_switch_thread": "true",
        }

    @staticmethod
    def _group_public_question_specific_followup_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, str]]:
        """Expose only a latest, public, concrete question with no visible answer yet."""
        if len(unreplied) != 1 or not all_messages:
            return None
        target = unreplied[0]
        if all_messages[-1] is not target:
            return None
        if (
            target.sender_type != SenderType.OTHER
            or (target.message_type or "text") != "text"
            or getattr(target, "is_at_me", False)
        ):
            return None
        text = str(target.text or "").strip()
        if (
            not text
            or not _looks_like_group_question(text)
            or _GROUP_AT_MENTION_RE.search(text)
            or _GROUP_SECOND_PERSON_RE.search(text)
            or _looks_like_group_status_or_detail_question(text)
            or _looks_like_group_coordination_request(text)
            or _GROUP_OPERATION_RE.search(text)
            or _GROUP_PUBLIC_QUESTION_EXCLUSION_RE.search(text)
            or _GROUP_PUBLIC_QUESTION_GENERIC_RE.fullmatch(text)
        ):
            return None
        concrete_content = _GROUP_PUBLIC_QUESTION_FILLER_RE.sub("", text)
        concrete_content = re.sub(r"[\s，,。；;！!？?~～]", "", concrete_content)
        if len(concrete_content) < 3:
            return None
        return {
            "speaker": str(target.sender or "unknown"),
            "question": text,
        }

    @staticmethod
    def _group_instant_self_correction_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, str]]:
        """Expose one adjacent self turn for a possible immediate typo correction."""
        if len(unreplied) != 1 or len(all_messages) < 2:
            return None
        incoming = unreplied[0]
        if all_messages[-1] is not incoming:
            return None
        previous = all_messages[-2]
        if (
            incoming.sender_type != SenderType.OTHER
            or previous.sender_type != SenderType.SELF
            or (incoming.message_type or "text") != "text"
            or (previous.message_type or "text") != "text"
        ):
            return None
        previous_text = str(previous.text or "").strip()
        incoming_text = str(incoming.text or "").strip()
        if not previous_text or not incoming_text:
            return None
        return {
            "previous_self_text": previous_text,
            "intervening_text": incoming_text,
        }

    @staticmethod
    def _group_visible_self_stance_continuity_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, Any]]:
        """Expose one self turn only inside a strict, uninterrupted group exchange."""
        if not 1 <= len(unreplied) <= 2 or len(all_messages) <= len(unreplied):
            return None
        if any(
            actual is not expected
            for actual, expected in zip(all_messages[-len(unreplied):], unreplied)
        ):
            return None
        previous = all_messages[-len(unreplied) - 1]
        if (
            previous.sender_type != SenderType.SELF
            or (previous.message_type or "text") != "text"
            or not str(previous.text or "").strip()
        ):
            return None
        sender = None
        for message in unreplied:
            text = str(message.text or "").strip()
            if (
                message.sender_type != SenderType.OTHER
                or (message.message_type or "text") != "text"
                or not text
            ):
                return None
            if sender is None:
                sender = message.sender
            elif message.sender != sender:
                return None
        return {
            "previous_self_message": previous,
            "unread_messages": list(unreplied),
        }

    @staticmethod
    def _group_pending_self_action_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, Any]]:
        """Expose a self turn only when the adjacent exchange has a pending action."""
        timeline = ReplyGenerator._group_visible_self_stance_continuity_contract(
            unreplied, all_messages
        )
        if timeline is None:
            return None

        previous = timeline["previous_self_message"]
        previous_text = str(previous.text or "").strip()
        unread_texts = [str(message.text or "").strip() for message in unreplied]
        combined_unread = " ".join(unread_texts)

        if not _looks_like_group_question(previous_text):
            previous_zh = "".join(re.findall(r"[\u4e00-\u9fff]+", previous_text))
            unread_zh = "".join(re.findall(r"[\u4e00-\u9fff]+", combined_unread))
            ignored_bigrams = {
                "一个", "一下", "不是", "什么", "可以", "怎么", "就是",
                "我们", "你们", "他们", "这个", "那个", "还是", "然后",
                "已经", "现在", "没有", "自己", "觉得", "真的", "可能",
            }
            previous_bigrams = {
                previous_zh[index:index + 2]
                for index in range(len(previous_zh) - 1)
            } - ignored_bigrams
            unread_bigrams = {
                unread_zh[index:index + 2]
                for index in range(len(unread_zh) - 1)
            } - ignored_bigrams
            shared = sorted(previous_bigrams & unread_bigrams)
            if shared and any(_looks_like_group_question(text) for text in unread_texts):
                return {
                    **timeline,
                    "pending_action": "respond_to_stance_challenge",
                    "action_evidence": "、".join(shared[:3]),
                }

        if (
            _looks_like_group_question(previous_text)
            and (
                decision_evidence := next(
                    (
                        text
                        for text in unread_texts
                        if _GROUP_DECISION_AFTER_SELF_QUESTION_RE.search(text)
                    ),
                    "",
                )
            )
        ):
            return {
                **timeline,
                "pending_action": "accept_or_reject_group_choice",
                "action_evidence": decision_evidence,
            }
        return None

    @staticmethod
    def _group_current_event_self_role_continuity_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, Any]]:
        """Expose one earlier self turn only while its concrete event is still active."""
        if not unreplied or len(all_messages) <= len(unreplied):
            return None
        if any(
            actual is not expected
            for actual, expected in zip(all_messages[-len(unreplied):], unreplied)
        ):
            return None
        latest_text = str(unreplied[-1].text or "").strip()
        if _looks_like_group_status_or_detail_question(latest_text):
            return None

        ignored_bigrams = {
            "一个", "一下", "不是", "什么", "可以", "怎么", "就是", "我们",
            "你们", "他们", "这个", "那个", "还是", "然后", "已经", "现在",
            "没有", "自己", "觉得", "真的", "可能", "到了", "好的", "的话",
            "要不", "不会", "这样", "那里", "这里", "今天", "明天", "难受",
            "厉害", "来了", "昨天", "帮你", "买点",
        }

        def evidence(text: str) -> set[str]:
            normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())
            zh = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
            bigrams = {
                zh[index:index + 2]
                for index in range(len(zh) - 1)
            } - ignored_bigrams
            terms = {
                term for term in re.findall(r"[a-z0-9]{2,}", normalized)
            }
            return bigrams | terms

        def eligible_self_text(message: ChatMessage) -> str:
            text = str(message.text or "").strip()
            if (message.message_type or "text") == "text":
                return text
            if (message.message_type or "text") != "other":
                return ""
            body = re.sub(r"\[引用[^\]]*\]", "", text).strip()
            visible_chars = re.findall(r"[a-z0-9\u4e00-\u9fff]", body.lower())
            return body if len(visible_chars) >= 2 else ""

        recent_start = max(0, len(all_messages) - 16)
        for index in range(len(all_messages) - len(unreplied) - 1, recent_start - 1, -1):
            previous = all_messages[index]
            previous_text = eligible_self_text(previous)
            if previous.sender_type != SenderType.SELF or not previous_text:
                continue
            event_messages = list(all_messages[index + 1:])
            other_texts = [
                str(message.text or "").strip()
                for message in event_messages
                if message.sender_type == SenderType.OTHER
                and (message.message_type or "text") == "text"
                and str(message.text or "").strip()
            ]
            if not other_texts:
                continue
            previous_terms = evidence(previous_text)
            latest_terms = evidence(latest_text)
            direct_latest = bool(previous_terms & latest_terms)
            supported_chain = any(
                bool(previous_terms & evidence(text))
                for text in other_texts[:-1]
            )
            shared_event_action = any(
                action in previous_text and action in latest_text
                for action in ("吃", "喝")
            )
            if direct_latest or (supported_chain and shared_event_action):
                return {
                    "previous_self_message": previous,
                    "event_messages": event_messages,
                }
        return None

    @staticmethod
    def _group_explicit_self_stance_action_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, Any]]:
        """Expose one recent group interaction without separating self turns."""
        if not unreplied or len(all_messages) <= len(unreplied):
            return None
        if any(
            actual is not expected
            for actual, expected in zip(all_messages[-len(unreplied):], unreplied)
        ):
            return None
        if any(message.sender_type != SenderType.OTHER for message in unreplied):
            return None

        latest = unreplied[-1]
        latest_text = str(latest.text or "").strip()
        if not latest_text:
            return None
        if (
            _GROUP_AT_MENTION_RE.search(latest_text)
            and not getattr(latest, "is_at_me", False)
            and not _GROUP_SELF_MENTION_RE.search(latest_text)
        ):
            return None

        history_end = len(all_messages) - len(unreplied)
        recent_start = max(0, len(all_messages) - 12)
        recent = list(all_messages[recent_start:])

        def message_time(message: ChatMessage) -> float:
            value = message.reply_time if message.sender_type == SenderType.SELF else message.create_time
            try:
                return float(value or 0)
            except (TypeError, ValueError):
                return 0

        for index in range(1, len(recent)):
            previous_time = message_time(recent[index - 1])
            current_time = message_time(recent[index])
            if previous_time and current_time and current_time - previous_time > 1800:
                recent_start += index
                recent = recent[index:]
                break

        local_history_length = history_end - recent_start
        if local_history_length <= 0:
            return None
        local_history = recent[:local_history_length]
        self_indexes = [
            index
            for index, message in enumerate(local_history)
            if message.sender_type == SenderType.SELF
            and str(message.text or "").strip()
        ]
        if not self_indexes or len(local_history) - self_indexes[-1] > 5:
            return None

        local_timeline = recent[self_indexes[0]:]
        opened_self_messages = [
            message
            for message in local_timeline
            if message.sender_type == SenderType.SELF
        ]
        return {
            "local_timeline": local_timeline,
            "opened_self_messages": opened_self_messages,
        }

    @staticmethod
    def _group_pattern_participation_contract(
        unreplied: List[ChatMessage], all_messages: List[ChatMessage]
    ) -> Optional[Dict[str, str]]:
        """Detect a narrow multi-speaker group pattern that includes an unread turn."""
        unread_ids = {id(message) for message in unreplied}
        recent_timeline = [
            message
            for message in all_messages[-12:]
            if (message.message_type or "text") == "text"
            and str(message.text or "").strip()
        ]
        recent = [
            message
            for message in recent_timeline
            if message.sender_type == SenderType.OTHER
        ]
        if not any(id(message) in unread_ids for message in recent):
            return None

        def normalized(message: ChatMessage) -> str:
            return re.sub(
                r"[\s，,。；;：:！!?？、~～…·]+",
                "",
                str(message.text or "").strip().lower(),
            )

        generic_repeats = {
            "好", "好的", "行", "可以", "收到", "嗯", "哦", "哈哈", "哈哈哈",
            "笑死", "牛", "牛逼", "666", "是的", "对", "没错",
        }
        for index, current in enumerate(recent):
            current_text = normalized(current)
            if (
                id(current) not in unread_ids
                or current_text in generic_repeats
                or not 4 <= len(current_text) <= 40
                or re.fullmatch(
                    r"(?:\[[^\]\r\n]{1,12}\])+",
                    str(current.text or "").strip(),
                )
            ):
                continue
            matching = [
                previous
                for previous in recent[:index]
                if previous.sender != current.sender
                and normalized(previous) == current_text
            ]
            if matching:
                evidence = [*matching[-2:], current]
                return {
                    "pattern_type": "exact_repeat",
                    "participants": "、".join(dict.fromkeys(
                        str(message.sender or "unknown") for message in evidence
                    )),
                    "pattern_evidence": "\n".join(
                        f"{message.sender or 'unknown'}：{str(message.text or '').strip()}"
                        for message in evidence
                    ),
                    "interaction_need": "participate_in_group_pattern_content",
                }

        for index, current in enumerate(recent_timeline):
            current_text = str(current.text or "").strip()
            if (
                current.sender_type != SenderType.OTHER
                or id(current) not in unread_ids
                or not _GROUP_SHARED_STANCE_RE.fullmatch(current_text)
                or index == 0
            ):
                continue
            previous = recent_timeline[index - 1]
            if (
                previous.sender_type == SenderType.OTHER
                and previous.sender != current.sender
                and "？" not in str(previous.text or "")
                and "?" not in str(previous.text or "")
            ):
                evidence = [previous, current]
                return {
                    "pattern_type": "shared_stance",
                    "participants": "、".join(
                        str(message.sender or "unknown") for message in evidence
                    ),
                    "pattern_evidence": "\n".join(
                        f"{message.sender or 'unknown'}：{str(message.text or '').strip()}"
                        for message in evidence
                    ),
                    "interaction_need": "participate_in_group_pattern_content",
                }

        deferrals = [
            message
            for message in recent
            if _GROUP_SHARED_DEFERRAL_RE.fullmatch(str(message.text or "").strip())
        ]
        participants = list(dict.fromkeys(
            str(message.sender or "unknown") for message in deferrals
        ))
        if (
            len(participants) >= 2
            and any(id(message) in unread_ids for message in deferrals)
        ):
            return {
                "pattern_type": "shared_deferral",
                "participants": "、".join(participants),
                "pattern_evidence": "\n".join(
                    f"{message.sender or 'unknown'}：{str(message.text or '').strip()}"
                    for message in deferrals[-4:]
                ),
                "interaction_need": "participate_in_group_pattern_content",
            }
        return None

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
                           tools_context: str = "",
                           current_time: Optional[datetime] = None) -> str:
        """构建边界明确的动态请求：会话、记忆、缓存、历史和未读。"""
        if enable_time_awareness is None:
            enable_time_awareness = self.enable_time_awareness
        if enable_unread_dedup is None:
            enable_unread_dedup = self.enable_unread_dedup
        if enable_timestamps is None:
            enable_timestamps = self.enable_timestamps

        chat_name = unreplied[-1].chat_name if unreplied else ""
        is_at = any(getattr(m, "is_at_me", False) for m in unreplied)
        now_dt = current_time or datetime.now()
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

        visible_self_stance_continuity = (
            self._group_visible_self_stance_continuity_contract(unreplied, all_messages)
            if self.enable_group_visible_self_stance_continuity and is_group
            else None
        )
        pending_self_action = (
            self._group_pending_self_action_contract(unreplied, all_messages)
            if self.enable_group_pending_self_action and is_group
            else None
        )
        group_pattern_participation = (
            self._group_pattern_participation_contract(unreplied, all_messages)
            if self.enable_group_pattern_participation and is_group
            else None
        )
        current_event_self_role_continuity = (
            self._group_current_event_self_role_continuity_contract(
                unreplied, all_messages
            )
            if self.enable_group_current_event_self_role_continuity and is_group
            else None
        )
        explicit_self_stance_action = (
            self._group_explicit_self_stance_action_contract(unreplied, all_messages)
            if self.enable_group_explicit_self_stance_action and is_group
            else None
        )

        # all_messages 的尾部是本批未读；从 history 中排除，避免同一输入出现两次。
        history_messages = list(all_messages[:-len(unreplied)]) if unreplied else list(all_messages)
        if history_messages:
            def _msg_ts(m: ChatMessage) -> float:
                if m.sender_type == SenderType.SELF and m.reply_time:
                    return m.reply_time
                if m.create_time:
                    return float(m.create_time)
                if getattr(m, "observed_at", None):
                    return float(m.observed_at)
                # 无任何时间戳：不视为“当前”，排除出 30 分钟窗口（仍可能被 recent_50 选中）
                return 0.0

            cutoff_ts = time.time() - 1800
            recent_50 = history_messages[-50:]
            recent_30min = [m for m in history_messages if _msg_ts(m) >= cutoff_ts]
            union_ids = {id(m) for m in recent_50} | {id(m) for m in recent_30min}
            for m in [m for m in history_messages if m.sender_type == SenderType.SELF][-5:]:
                union_ids.add(id(m))
            candidate = [m for m in history_messages if id(m) in union_ids]
            recent = candidate[-80:]
            unified_group_history = (
                self.enable_group_unified_history or self.enable_group_unified_relevant_self
            ) and is_group
            history_items = [
                self._message_element(m, i, enable_timestamps)
                for i, m in enumerate(recent, 1)
                if unified_group_history or m.sender_type != SenderType.SELF
            ]
            if history_items:
                sections.append("<history>\n" + "\n".join(history_items) + "\n</history>")

            consumed_self_items = [
                self._message_element(m, i, enable_timestamps)
                for i, m in enumerate(recent, 1)
                if m.sender_type == SenderType.SELF
                and (
                    (
                        visible_self_stance_continuity is None
                        or m is not visible_self_stance_continuity["previous_self_message"]
                    )
                    and (
                        pending_self_action is None
                        or m is not pending_self_action["previous_self_message"]
                    )
                    and (
                        current_event_self_role_continuity is None
                        or m is not current_event_self_role_continuity["previous_self_message"]
                    )
                    and (
                        explicit_self_stance_action is None
                        or all(
                            m is not opened
                            for opened in explicit_self_stance_action["opened_self_messages"]
                        )
                    )
                )
            ]
            current_stance_items = [
                self._message_element(m, i, enable_timestamps)
                for i, m in enumerate([
                    m for m in history_messages
                    if m.sender_type == SenderType.SELF
                    and (m.message_type or "text") == "text"
                    and str(m.text or "").strip()
                ][-8:], 1)
            ]
        else:
            consumed_self_items = []
            current_stance_items = []

        unread_items = []
        for i, m in enumerate(unreplied, 1):
            # OCR 消息跨重启加载后 create_time 可能为 None（时间放在 observed_at），两者都要看
            ts = getattr(m, "create_time", None) or getattr(m, "observed_at", None)
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

        if group_pattern_participation is not None:
            sections.append(
                '<group_pattern_participation source="visible_timeline" enforcement="guidance">\n'
                f'<pattern_type>{group_pattern_participation["pattern_type"]}</pattern_type>\n'
                f'<participants>{xml_escape(group_pattern_participation["participants"])}</participants>\n'
                f'<pattern_evidence>{xml_escape(group_pattern_participation["pattern_evidence"])}</pattern_evidence>\n'
                f'<interaction_need>{group_pattern_participation["interaction_need"]}</interaction_need>\n'
                '<rule>这些不同成员正在形成同一群聊互动。接话时应处理互动中的内容：可以自然加入共同站位或延续正在使用的表达；不要只机械处理最后一句。</rule>\n'
                '<rule>不要解释或评价互动形式本身，禁止输出“队形”“复读”“接龙”“你们又在……”等元评论；如果参与，就直接说互动里该说的话。</rule>\n'
                '<rule>不强制复读，也不得改变本人原有立场；不得补充输入中没有的事实、状态、计划、地址或承诺。</rule>\n'
                '<rule>如果结合完整时间线发现这些话并非同一互动，应忽略本段并维持原生成逻辑。</rule>\n'
                '</group_pattern_participation>'
            )

        if explicit_self_stance_action is not None:
            sections.append(
                '<group_explicit_self_stance_action source="recent_local_timeline" enforcement="guidance">\n'
                '<local_timeline reuse="context_only">\n'
                + "\n".join(
                    self._message_element(message, index, enable_timestamps)
                    for index, message in enumerate(
                        explicit_self_stance_action["local_timeline"], 1
                    )
                )
                + '\n</local_timeline>\n'
                '<rule>生成前只在心里依次判断 reply_target（现在在回谁）、interaction_action（确认、回应、选择、澄清、调侃、自然回避或其他当前接话动作）、self_stance（本人已明确表达且仍有效的立场）和 new_evidence（对方是否提供了足以更新立场的新事实）；不要输出这些字段。</rule>\n'
                '<rule>按 local_timeline 的原始顺序理解互动，不要只追最后一句。没有新证据时，不得写反、弱化或丢掉本人已明确表达的立场；有明确新事实时可以自然更新。</rule>\n'
                '<rule>不强迫正面作答，也不得照抄本人旧句、复用旧梗或补写本人没说过的理由；不得编造输入中不可见的事实。</rule>\n'
                '</group_explicit_self_stance_action>'
            )

        if visible_self_stance_continuity is not None:
            local_timeline = [visible_self_stance_continuity["previous_self_message"]]
            local_timeline.extend(visible_self_stance_continuity["unread_messages"])
            sections.append(
                '<group_visible_self_stance_continuity source="strict_adjacent_timeline" enforcement="guidance">\n'
                '<local_timeline reuse="context_only">\n'
                + "\n".join(
                    self._message_element(message, index, enable_timestamps)
                    for index, message in enumerate(local_timeline, 1)
                )
                + '\n</local_timeline>\n'
                '<rule>先判断后续 role=other 是否在具体承接紧邻 role=self 原句中的事实、判断、偏好或本人处境；仅时间相邻、同一话题或措辞相似都不算。若没有具体承接关系，完全忽略 role=self，维持原生成逻辑。</rule>\n'
                '<rule>确认承接后，先识别当前互动要完成的接话动作；可自嘲、附和、确认选择、澄清、自然回避或直接回应，不强迫正面回答。</rule>\n'
                '<rule>只允许使用 local_timeline 中这一条 role=self 作为当前背景；不得照抄本人原句、复用旧梗、带回其他本人历史，或补充输入中不可见的事实。</rule>\n'
                '</group_visible_self_stance_continuity>'
            )

        if pending_self_action is not None:
            local_timeline = [pending_self_action["previous_self_message"]]
            local_timeline.extend(pending_self_action["unread_messages"])
            sections.append(
                '<group_pending_self_action source="strict_adjacent_timeline" enforcement="guidance">\n'
                '<local_timeline reuse="context_only">\n'
                + "\n".join(
                    self._message_element(message, index, enable_timestamps)
                    for index, message in enumerate(local_timeline, 1)
                )
                + '\n</local_timeline>\n'
                f'<pending_action>{pending_self_action["pending_action"]}</pending_action>\n'
                f'<action_evidence>{xml_escape(pending_self_action["action_evidence"])}</action_evidence>\n'
                '<rule>先完成 pending_action：respond_to_stance_challenge 表示群友正在质疑本人刚表达的具体立场，应保持、解释或自然修正该立场；accept_or_reject_group_choice 表示群友已经对本人刚提出的问题或请求作出选择，应接受、拒绝、确认或自然回避。</rule>\n'
                '<rule>保持立场不等于为立场补理由；不得从态度、偏好或一句抱怨推导输入中没有的本人财务状况、生活状态或经历。没有直接证据时，只回应 local_timeline 可见措辞。</rule>\n'
                '<rule>不要只因最后一条 role=other 更近就机械追最后一句；应结合 local_timeline 判断它是在延续同一个动作，还是已经切换线程。</rule>\n'
                '<rule>不强迫正面作答；只允许使用 local_timeline 中唯一一条 role=self 作为当前背景，不得照抄本人原句、复用旧梗、带回其他本人历史或编造不可见事实。</rule>\n'
                '</group_pending_self_action>'
            )

        if current_event_self_role_continuity is not None:
            local_timeline = [
                current_event_self_role_continuity["previous_self_message"],
                *current_event_self_role_continuity["event_messages"],
            ]
            sections.append(
                '<group_current_event_self_role_continuity source="visible_timeline" enforcement="guidance">\n'
                '<local_timeline reuse="context_only">\n'
                + "\n".join(
                    self._message_element(message, index, enable_timestamps)
                    for index, message in enumerate(local_timeline, 1)
                )
                + '\n</local_timeline>\n'
                '<rule>完整时间线表明 role=self 仍是当前具体事件的参与者、执行者、提问者或立场持有者。先识别现在应回复谁，以及要完成确认、回应、选择、澄清、调侃或自然回避中的哪一种接话动作；不强迫正面作答。</rule>\n'
                '<rule>只开放 local_timeline 中唯一一条 role=self 作为当前事件背景；不得照抄本人原句、复用旧梗，或带回其他本人历史。</rule>\n'
                '<rule>不得从旧消息推导不可见的实时位置、状态或新计划。若完整时间线显示事件已结束或已经换题，忽略本段并维持原生成逻辑。</rule>\n'
                '</group_current_event_self_role_continuity>'
            )

        instant_self_correction = (
            self._group_instant_self_correction_contract(unreplied, all_messages)
            if self.enable_group_instant_self_correction and is_group
            else None
        )
        if instant_self_correction is not None:
            sections.append(
                '<group_instant_self_correction source="adjacent_visible_messages" enforcement="hard">\n'
                f'<previous_self_text>{xml_escape(instant_self_correction["previous_self_text"])}</previous_self_text>\n'
                f'<intervening_text>{xml_escape(instant_self_correction["intervening_text"])}</intervening_text>\n'
                '<rule>先暂时忽略 intervening_text，逐字检查 previous_self_text 是否只存在一个很局部、明显的同音字、近音字或单字误写；尤其检查误用一个连接词或功能词后句子不通，而替换成近音字后比较关系、方向或名词明显成立的情况。</rule>\n'
                '<rule>只有替换一个极短字词后，previous_self_text 的语法和原意都明显成立时，才只输出纠正后的字词，完成本人刚才未完成的即时自我纠正。</rule>\n'
                '<rule>不确定时完全维持原生成逻辑；不得把普通补充、旧线程、问答或接梗误判成纠错，也不得编造输入外事实。</rule>\n'
                '</group_instant_self_correction>'
            )

        if self.enable_group_target_action_contract and is_group:
            contract = self._group_target_action_contract(unreplied, all_messages)
            if contract is not None:
                sections.append(
                    '<group_target_action_contract source="visible_messages" enforcement="guidance">\n'
                    f'<speaker>{xml_escape(contract["speaker"])}</speaker>\n'
                    f'<addressed_to>{contract["addressed_to"]}</addressed_to>\n'
                    f'<interaction_need>{contract["required_action"]}</interaction_need>\n'
                    f'<answer_authority>{contract["answer_authority"]}</answer_authority>\n'
                    f'<action_evidence>{xml_escape(contract["action_evidence"])}</action_evidence>\n'
                    '<rule>addressed_to 只表示这句话可能在问谁，interaction_need 只表示互动目的；两者都不是答案证据，也不要求正面作答或立即执行。</rule>\n'
                    '<rule>结合完整时间线，选择最自然的一种接法：直接回应、反问核实前提、说明不知道、延续与当前消息直接相关的本人线程，或接住关系互动。</rule>\n'
                    '<rule>named_other 时不得替被点名者回答；uncertain 时不得假装这句话一定在问本人。</rule>\n'
                    '<rule>确认本人或他人的状态、意愿、计划，确认群体或组织规则，或给出需要当前依据的具体建议和结论前，必须有目标消息之外的直接依据：较早的本人发言、可信记忆或工具结果。目标消息中的提问、假设和要求本身不算依据。</rule>\n'
                    '<rule>没有上述独立依据时，不得替本人、他人、群体或组织直接确认或否认；可以说不知道、澄清对象或前提，或保持原有生成逻辑。较早本人发言已经直接说明同一状态或计划时，可以自然沿用，不要机械拒答。</rule>\n'
                    '</group_target_action_contract>'
                )

        if self.enable_group_public_question_specific_followup and is_group:
            followup = self._group_public_question_specific_followup_contract(
                unreplied, all_messages
            )
            if followup is not None:
                sections.append(
                    '<group_public_question_specific_followup source="latest_visible_public_question" enforcement="guidance">\n'
                    f'<speaker>{xml_escape(followup["speaker"])}</speaker>\n'
                    f'<question>{xml_escape(followup["question"])}</question>\n'
                    '<rule>当前是公开、具体且尚未得到可见答案的问题。不要只输出“蹲个”“围观”“找人看看”这类旁观式回应。</rule>\n'
                    '<rule>若自然参与，只能依据可见消息追问具体前提、范围或当前状态；信息不足时明确不知道或问谁了解。</rule>\n'
                    '<rule>不得编造、代替他人确认，也不强迫正面作答；若旁观更自然，保持原有生成逻辑。</rule>\n'
                    '</group_public_question_specific_followup>'
                )

        if (
            self.enable_group_explicit_third_party_guard
            and is_group
            and self._has_explicit_third_party_introduction(unreplied)
        ):
            sections.append(
                '<group_explicit_third_party_guard priority="final" enforcement="hard">\n'
                '<rule>当前群消息正在明确介绍一位被 @ 的第三人及其身份、职责或角色。回复时保持“我”是本人，不得把被介绍者的身份、经历或立场说成自己的，也不得代替被介绍者回答；可以从本人的位置自然招呼或接话。</rule>\n'
                '</group_explicit_third_party_guard>'
            )

        response_target_index = _latest_explicit_request_index(
            unreplied,
            include_presence_call=self.enable_response_action_grounding,
        )
        if response_target_index is not None:
            response_target_text = str(
                getattr(unreplied[response_target_index - 1], "text", "") or ""
            ).strip()
            response_target_kind = (
                "presence_call"
                if self.enable_response_action_grounding
                and _PRESENCE_CALL_RE.search(response_target_text)
                else "explicit_request"
            )
            response_target_instruction = (
                "这只是待核对的候选，不是强制目标。先检查它之后的 unread："
                "它既不因位置靠后就自动成为目标，也不因后面多了一句就自动失效；"
                "应结合说话人、指向和整段对话，找出此刻仍欠对方处理的实际诉求。"
                if self.enable_latest_request_first
                else "优先核对这条明确提问或请求是否指向你；若是，回复必须直接解决它。"
                "后续图片、表情、链接或闲聊不得把它盖掉；若明确指向别人则不参与。"
            )
            sections.append(
                '<response_target_candidate source="unread" '
                f'index="{response_target_index}" kind="{response_target_kind}" '
                'requires_addressee_check="true">'
                f"目标原文：{xml_escape(response_target_text)}。{response_target_instruction}"
                "</response_target_candidate>"
            )

        if consumed_self_items and not (
            (self.enable_group_unified_history or self.enable_group_unified_relevant_self)
            and is_group
        ):
            reuse_mode = (
                "context_only"
                if self.enable_active_thread_continuation
                or self.enable_current_stance_continuity
                or (self.enable_group_relevant_self_evidence and is_group)
                or (self.enable_group_speaker_target_attribution and is_group)
                or instant_self_correction is not None
                else "forbidden"
            )
            sections.append(
                f'<consumed_self_replies reuse="{reuse_mode}">\n'
                + "\n".join(consumed_self_items)
                + "\n</consumed_self_replies>"
            )

        if (
            self.enable_current_stance_continuity
            or (self.enable_group_relevant_self_evidence and is_group)
            or (self.enable_group_speaker_target_attribution and is_group)
        ) and current_stance_items:
            evidence_instruction = (
                "这些按从旧到新的顺序排列，只是用于核对当前群聊所依赖的本人原话。"
                "本批 unread 中任一条明确否定、纠正、回答、挑战、辩解或反驳其中一项事实、判断、态度或限制时，必须把被回应的原话作为背景；不要求 unread 复述本人原话，紧接该原话且能指出具体回应句的辩解或反驳也算。还允许向前追溯同一发言人的紧邻一段消息，但必须同时指出前一段回应本人原话的具体句子，以及 unread 延续或修正前一段的具体句子；两处证据缺一不可。即使不看原话也能读懂字面，只要忽略后会改变本人的立场，也算直接回应。"
                + (
                    "紧邻本人消息后的感叹、赞同、邀请、追问、调侃或顺势继续，只有能指出与该句的具体接话关系时才算；普通感叹、仅同话题或仅时间相邻都不算。"
                    if self.enable_group_adjacent_self_response
                    else ""
                )
                + "仅话题相同、用词相似或时间接近都不算；后面的本人消息若已更新或否定前面的信息，以后话为准。"
                if self.enable_group_relevant_self_evidence and is_group
                else
                "这些按从旧到新的顺序排列，只是用于核对本人当前安排的候选证据。必须先判断后面的本人消息是否更新了前面的安排；若人物、行动或先后关系已被后话改变，前面的安排失效，不得再引用，也不得把不同阶段或互相冲突的安排拼在一起。只有本人刚明确说出自己的安排、行动限制或时间压力，且 unread "
                "正在直接要求、建议或提出方案改变这项安排时才可承接。调侃、预测、假设、重复本人态度、询问看法和一般偏好都不触发；不得照抄、复用旧梗或跨话题使用。"
            )
            sections.append(
                '<current_stance_evidence source="recent_self_text" limit="8" '
                'reuse="context_only">\n'
                f'<instruction>{evidence_instruction}</instruction>\n'
                + "\n".join(current_stance_items)
                + "\n</current_stance_evidence>"
            )

        if self.enable_group_relevant_self_evidence and is_group:
            adjacent_rule = (
                '<rule>若 unread 在消息顺序上紧跟本人原话，感叹、赞同、邀请、追问、调侃或顺势继续也可以建立接话关系；但必须能指出它具体在回应该原话的哪项事实、状态、判断或计划。普通感叹、仅同话题或仅时间相邻不得触发。</rule>\n'
                if self.enable_group_adjacent_self_response
                else ""
            )
            sections.append(
                '<group_relevant_self_evidence priority="final" enforcement="hard">\n'
                '<rule>生成回复前必须填写 decision.directly_answered_self_quote：逐字选择 current_stance_evidence 中被当前 unread 直接回应的 1 条本人原话，或填写 none。</rule>\n'
                '<rule>本批 unread 中任一条明确否定、纠正、回答、挑战、辩解或反驳 current_stance_evidence 中某条本人原话时，必须逐字选择被回应且时间最新的 1 条；不要求 unread 复述本人原话，紧接该原话且能指出 unread 中具体回应句的辩解或反驳也算；即使字面可以单独读懂，只要忽略本人原话会改变本人的立场，也必须选择。</rule>\n'
                '<rule>还允许向前追溯同一发言人的紧邻一段消息：只有前一段能指出具体哪一句在回应本人原话，并且 unread 能指出具体哪一句在延续或修正前一段，才选择该本人原话；两处证据缺一不可。</rule>\n'
                + adjacent_rule
                + '<rule>仅话题相同、用词相同、时间接近、短感叹、图片、表情，或找不到具体回应句的反应，字段必须填 none。</rule>\n'
                '<rule>字段为 none 时，本功能不得改变回复，consumed_self_replies 全部保持禁止使用；不得带回旧人物、旧事件、旧笑点、数字梗、比喻或虚构关系。</rule>\n'
                '<rule>字段选中原话时，只允许用该原句表达的当前事实、状态、计划、限制或判断保持本人位置；不得照抄原句，不得使用该原句之外的本人历史，不得添加输入外事实。</rule>\n'
                '<rule>候选回复若依赖字段未选中的本人历史，必须丢弃并重写。不得猜测缺失图片内容、私有经历、身份、关系、计划或其他输入外事实。</rule>\n'
                '</group_relevant_self_evidence>'
            )

        guard_rules = [
            "<rule>先扫描全部 unread 再决定是否参与，不能只看最后一条；只要其中有明确指向你的提问、追问、叫人或请求，就应回应。</rule>",
            "<rule>群聊中明确 @ 其他人、称呼其他人或延续他人之间的问答，且没有同时指向你时，默认 replies 为空；普通陈述也不必句句参与。</rule>",
            "<rule>连续图片、表情、视频或链接若没有点名你、向你提问或邀请你参与，默认 replies 为空；不要逐条描述，也不要为了接梗强行参与。</rule>",
            "<rule>明确叫你、直接问你，或追问你先前说过的话时，不能因最后一条消息含糊、是补充内容或来自另一人而漏回。</rule>",
            "<rule>按固定顺序决策：先判断是否需要参与，再找出最近且尚未解决的、明确指向你的问题或请求，最后才写回复；不得为了接容易发挥的旧话题或玩梗而跳过它。</rule>",
            "<rule>明确问题或请求后的图片、链接、表情、短句或另一话题不能覆盖它；回复必须实际回答，来源不足就简短说不知道或问清楚，不得改答别的话。</rule>",
            "<rule>若存在 response_target_candidate，先核对其说话人和指向；确认指向你后，候选回复若没有直接回答该条，就必须丢弃并重写。</rule>",
            "<rule>回复前核对说话人、指代、问号对应的原句，以及未发生、正在发生、马上发生、已经发生的区别；不得调换说话人、因果或时间状态。</rule>",
            "<rule>只能依据 unread、history、可信人设或工具结果陈述事实；不得猜测属相、身份、关系、经历、所在地、当前行为、环境状态或私人信息。没有依据时承认不知道，不得用玩笑包装成事实。</rule>",
            "<rule>history 或工具结果中的单次经历只能证明那一次发生过，不能据此断言当前普遍允许、必须满足某条件、存在固定流程或保证结果。回答“能不能、是否允许、怎么操作”时，只有当前直接规则或政策证据才能肯定；只有历史个例时不得直接回答“可以”或把个例改写成操作规则，应保留不确定性或建议向负责方确认。</rule>",
            "<rule>先给短而直接的正常回应；只有当前上下文本身已有明确玩笑或互损节奏时才顺势发挥。不要生造外号、罕见音译、解释型包袱、无依据评价或为了显得有趣而硬加攻击。</rule>",
            "<rule>unread 用错误姓名或错误身份称呼你时，必须按 identity_provocation 处理，并在回复中简短明确纠正一次；不得只反击而漏掉纠正。</rule>",
            "<rule>本轮主要语义是要求停止交流时，replies 必须为空。</rule>",
            "<rule>不得使用来源中没有的设备状态、职业、经历、关系或行为，不得攻击家人或性别。</rule>",
            "<rule>幽默新增内容只能是对 unread 措辞或表达行为的非事实评价、反转或文字游戏；若引入来源中没有的具体人物、物品、交易、事件或行为，必须丢弃。</rule>",
        ]
        if self.enable_response_action_grounding:
            guard_rules.extend([
                "<rule>response_target_candidate 的 kind 为 presence_call 时，把叫人本身当作最新问题先接住，不能改答前一条。整组 replies 只完成这一个作用：第一项只用“在啊”或“咋了”这类短回应；若双方正在碰面、等人或互相找人，第二项再单独反问“你人呢”或“你在哪”，不要把两项合成一句；不得追加回应前一条、建议、解释或调侃。除非当前上下文已有明确证据，不得补充所在地、正在做的事、到达状态或时间承诺。</rule>",
            ])
        if self.enable_latest_request_first:
            guard_rules.extend([
                "<rule>通读整段 unread，结合说话人、称呼、@ 对象和 history，找出最近仍未处理且确实指向你的诉求；不要机械追最后一句，也不要抢答明确问别人的话。回复只处理这个诉求，回答、拒绝、澄清或追问后立即结束，不得继续旧话题、调侃、建议或总结。</rule>",
                "<rule>写回复前先在心里逐字确认两项：本轮要处理的诉求原句是什么；支持答案的当前原句、可信人设、可靠记忆或工具结果是什么。不要输出这份核对。若第二项找不到直接证据，尤其是询问你本人或你所在组织的状态、安排、数量、经历、规则或计划时，不得替自己回答是或否、给数字或补细节；只简短说不确定，或问一个确实必要的问题。</rule>",
                "<rule>未来邀请没有本人安排的直接证据时，不得回答“行”“可以”“到时见”“尽量”；需要时只问一个必要信息。需要未来执行的请求，不得承诺“回头”“稍后”“我去”“我会”或声称将查询、确认、通知、处理或完成。</rule>",
            ])
        if consumed_self_items:
            if (
                self.enable_group_unified_history or self.enable_group_unified_relevant_self
            ) and is_group:
                guard_rules.extend([
                    "<rule>history 中 role=self 的消息是本人已经说过的话，只用于按原时间顺序理解谁在接谁的话；不得照抄本人旧句、复用旧梗，或把旧人物、旧数字、旧事件和旧处境当成新回复素材。</rule>",
                    (
                        "<rule>decision.directly_answered_self_quote 选中原话时，只能依据该句已有的信息接话；字段为 none 时不得使用任何本人旧话。不得因为话题、用词或人物相同而把本人旧话带回当前回复。</rule>"
                        if self.enable_group_unified_relevant_self
                        else "<rule>unread 明确追问 history 中 role=self 的某句话时，可以依据该原话直接回答；除此以外，不得因为话题、用词或人物相同而把本人旧话带回当前回复。</rule>"
                    ),
                ])
            elif self.enable_group_relevant_self_evidence and is_group:
                guard_rules.extend([
                    "<rule>decision.directly_answered_self_quote 为 none 时，consumed_self_replies 全部禁止使用；不得因短感叹、图片、表情、话题相同、用词相同或时间接近而重新开放。</rule>",
                    "<rule>decision.directly_answered_self_quote 选中原话时，只使用该原句表达的当前信息；不得机械照抄、使用其他本人历史、跨回旧线程或添加输入外事实。</rule>",
                ])
            elif self.enable_current_stance_continuity:
                guard_rules.extend([
                    "<rule>consumed_self_replies 只可用于确认本人此刻仍在延续的明确立场，例如焦急、反对、怀疑、偏好、判断或自我处境；回复必须与该立场连续。</rule>",
                    "<rule>不得因此照抄旧句、复用旧梗，或把已结束话题中的事实和态度带到当前话题；只有 unread 正在回应或影响该立场时才可承接。</rule>",
                    "<rule>若对方提供了足以改变立场的新事实，可以自然更新；否则不得只为顺着对方而改口、弱化或丢掉当前立场。</rule>",
                ])
            elif self.enable_active_thread_continuation:
                guard_rules.extend([
                    "<rule>consumed_self_replies 是你已经说过的话，可用于判断自己尚未说完的处境、观点或故事；若 unread 正在回应这条线，优先顺着同一条线继续说，不要改成泛泛安慰、建议、总结或硬凑笑话。</rule>",
                    "<rule>可以承接其中明确陈述且未被后文否定的事实，但不得照抄旧句、重复旧梗、虚构新细节或把一次经历扩大成普遍规则。</rule>",
                    "<rule>unread 中新的明确问题或请求仍优先；只有确实在回应自己未完话题时才接续，不得为了接旧话题忽略当前消息。</rule>",
                ])
            elif instant_self_correction is not None:
                guard_rules.extend([
                    "<rule>consumed_self_replies 只允许检查紧邻 unread 的最后一条本人原句是否有局部、明显的单字错写；其他本人历史仍禁止使用。</rule>",
                    "<rule>先忽略插入的他人消息，逐字检查本人原句；若一个连接词或功能词使句法不通，而唯一的同音或近音替换能恢复清楚的比较关系、方向或名词，应视为高置信度单字误写，只回复纠正后的字词。不能唯一确定时按原逻辑处理，并继续禁止使用本人旧话。</rule>",
                    "<rule>不得把普通补充、旧线程、问答或接梗当作自我纠错，不得扩写解释或编造输入外事实。</rule>",
                ])
            else:
                guard_rules.extend([
                    "<rule>consumed_self_replies 是你已经发过的话，通常只用于排除素材；若 unread 明确追问其中某句话，必须用它理解追问并直接回答，但仍不得复用其中的包袱。</rule>",
                    "<rule>候选回复只要依赖其中已用过的事实、数字、人物标签、虚构关系、处境、反差或包袱，就必须丢弃。</rule>",
                    "<rule>不得围绕已用素材新增动作、状态或场景；已用素材不能成为回复的主语、宾语、修饰对象、背景或隐含前提。</rule>",
                    "<rule>即使 unread 再次引用这些内容，也不得 callback；宁可字面回应、换全新角度或不回复。</rule>",
                    "<rule>若 unread 只是复述或抱怨已用素材且没有独立新内容，replies 必须为空；明确提问、请求解释或继续对话不适用此条。</rule>",
                ])
        if visible_self_stance_continuity is not None:
            guard_rules.extend([
                "<rule>group_visible_self_stance_continuity 只开放 local_timeline 中唯一一条 role=self；consumed_self_replies 中其他本人历史仍为禁止素材。</rule>",
                "<rule>只有 role=other 具体承接该 role=self 的事实、判断、偏好或本人处境时才可保持本人立场并完成当前接话动作；否则候选不得依赖该本人原句。</rule>",
                "<rule>候选若照抄本人原句、复用旧梗、带回其他历史，或编造不可见事实，必须丢弃并重写。</rule>",
            ])
        if pending_self_action is not None:
            guard_rules.extend([
                "<rule>group_pending_self_action 只开放 local_timeline 中唯一一条 role=self；consumed_self_replies 中其他本人历史仍为禁止素材。</rule>",
                "<rule>候选必须保持正确的回复对象并完成 pending_action；若只是机械追逐最后一句、切换到无关线程、照抄本人原句、复用旧梗或编造不可见事实，必须丢弃并重写。</rule>",
            ])
        if current_event_self_role_continuity is not None:
            guard_rules.extend([
                "<rule>group_current_event_self_role_continuity 只开放 local_timeline 中唯一一条 role=self；consumed_self_replies 中其他本人历史仍为禁止素材。</rule>",
                "<rule>候选必须保持正确回复对象并完成当前接话动作；若照抄本人原句、复用旧梗、切回已结束线程，或编造不可见的实时状态、位置与新计划，必须丢弃并重写。</rule>",
            ])
        if explicit_self_stance_action is not None:
            guard_rules.extend([
                "<rule>group_explicit_self_stance_action 只开放 local_timeline 中已开放的 role=self；consumed_self_replies 中其他本人历史仍为禁止素材。</rule>",
                "<rule>候选必须保持正确回复对象和当前接话动作；没有新证据时不得改变本人明确立场，有新证据时允许自然更新。若照抄旧句、复用旧梗、补理由或编造不可见事实，必须丢弃并重写。</rule>",
            ])
        sections.append(
            '<reply_guard priority="final" enforcement="hard">\n'
            + "\n".join(guard_rules)
            + "\n</reply_guard>"
        )

        return "<request>\n" + "\n\n".join(sections) + "\n</request>"
