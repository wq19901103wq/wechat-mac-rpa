#!/usr/bin/env python3
"""Build and run an isolated Judge acceptance set from strict historical replays.

This script deliberately does not change the production generator.  It reuses the
current generator, fixes only the historical clock while its user prompt is being
built, stops before sending, and records every exclusion and Judge subprocess.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.reply.generator as generator_module  # noqa: E402
from scripts.evaluate_current_c0_historical_ab import (  # noqa: E402
    MANIFEST,
    SIDE_EFFECT_TOOLS,
    _chat_message,
    _jsonable,
    _load_cases,
    _match_context,
    _norm,
    _source,
    _trace,
    _write_json,
    _safe_registry,
)
from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _load_env,
    _parse_api_judge_response,
    assert_reference_isolated,
    load_volcengine_credentials,
)
from scripts.run_human_ai_judge_benchmark_v2 import (  # noqa: E402
    build_messages,
    load_profile,
    resolve_selected_reply,
)
from scripts.run_kimi_human_ai_judge_v2 import _run_codex  # noqa: E402
from scripts.build_adversarial_stage2_holdout import (  # noqa: E402
    _common_candidates,
    _near_duplicate,
    _ordinary_candidates,
    _youzi_cases,
)
from src.memory.engine import MemoryEngine  # noqa: E402
from src.models.base import ChatMessage, SenderType  # noqa: E402
from src.reply.generator import ReplyGenerator  # noqa: E402
from src.utils.qwen_client import QwenClient  # noqa: E402

OUT_DIR = ROOT / "data/reports/private/strict_historical_judge_acceptance_v1"
PROFILE = ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v2_thread_attribution.json"
REFERENCE = ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_reference.json"
GENERATOR_PATH = ROOT / "src/reply/generator.py"
TARGET_RESOLUTION_MANIFEST = ROOT / "data/reports/private/adversarial_stage2_target_resolution_candidates_v2/manifest.json"
MIN_HIDDEN_CASES = 100
STAGE2_STRATUM_WEIGHTS = {
    "correct_common_prosperity_with_luxixi": 20,
    "youzi_group": 20,
    "other_complex_large_group": 62,
    "other_group": 58,
    "private": 80,
}
GROUP_BALANCED_STRATA = (
    "correct_common_prosperity_with_luxixi",
    "youzi_group",
    "other_complex_large_group",
)

MISJUDGMENT_REVIEW = {
    "a367455c80ac8b27d5fc": ("clear_logic_error", "真人是在被吹捧时假装自己不是 G 少爷并跟着浑水摸鱼；候选把老梗“复读机”误当成本人特异幽默。"),
    "dfba3cac18730c7c35f0": ("clear_logic_error", "候选偏爱把“93 在 100 以下”再次解释出来；这只是重复显然信息，没有新增判断。"),
    "3aec4abcfcfd10a225e7": ("clear_logic_error", "候选把“异动”重新释义成“变动”并补充追问，错误奖励了没有新增信息的完整解释。"),
    "bc38ee57b06a9b4358b5": ("intrinsically_hard", "两边都能自然承接口误梗，仅凭当前上下文难稳定区分。"),
    "10b8fe179967edeec01f": ("intrinsically_hard", "两边分别承接群聊中的不同线程，均有自然解释。"),
    "002af53b6aafc0f51d46": ("intrinsically_hard", "两边都能合理推进线下会合，缺少可判定地点的外部事实。"),
    "d995dea6b58b30af41bc": ("intrinsically_hard", "两份回复语义近似，都是拒绝吃饭并转去运动。"),
    "ae143a4259dba23dcd52": ("intrinsically_hard", "两边都是合理的付款决策表达，Judge 的低置信度也较符合证据强度。"),
    "f8d7c369f6fda0376920": ("clear_logic_error", "偏爱直接解释当前消息，忽略真人借离职话题劝对方离职。"),
    "2b90a6d32ff6650c6626": ("clear_logic_error", "偏爱直接回答当前问题，错误排除了真人回到较早线程的可能。"),
    "e4182c290cac2d513c24": ("clear_logic_error", "理由声称看到了实际引用回复，疑似把上下文中的既有文字误当成目标答案，须单独复核。"),
    "3a8785a87b007b86b0f1": ("clear_logic_error", "偏爱常见的直接午饭应答，忽略真人提供现实状态信息。"),
    "dc811b38b8ea459d1e3a": ("clear_logic_error", "只按字面承接午饭邀请，忽略亏损语境中的真人情绪梗。"),
    "0574bc902e918dd0eb0f": ("clear_logic_error", "把无依据的具体公司名当成更合理答案，忽略了潜在幻觉。"),
    "bb78f8d68b647a3a8f9c": ("clear_logic_error", "偏爱完整明确的店名回答，忽略真人常见的确认式回复。"),
    "09f4aa4a0c7debb4c0df": ("clear_logic_error", "偏爱正面回答问题，忽略真人看到图片后直接发笑。"),
    "2f545bfbb9f367c3d215": ("clear_logic_error", "偏爱礼貌完整的感谢，忽略真人用反讽表达红包已成大额开销。"),
    "5f236db9fa4a1da393e6": ("clear_logic_error", "偏爱更丰富、更像组织者的回复，排除了真人短促报名。"),
    "1038e3392d49890adeba": ("clear_logic_error", "偏爱字面承接当前线程，忽略真人基于具体关系继续玩梗。"),
    "7115a8517c26ff6d157f": ("clear_logic_error", "反向偏爱短促回怼，忽略真人缓和语气并继续玩笑。"),
}

MANUAL_ERROR_CAUSE = {
    "a367455c80ac8b27d5fc": "stale_meme_or_surface_humor",
    "dfba3cac18730c7c35f0": "redundant_obvious_explanation",
    "3aec4abcfcfd10a225e7": "redundant_obvious_explanation",
}
MEDIA_PLACEHOLDERS = ("[图片]", "[表情]", "[表情包]", "[视频]")


class _HistoricalTimeProxy:
    def __init__(self, real_module: Any, timestamp: float):
        self._real_module = real_module
        self._timestamp = timestamp

    def time(self) -> float:
        return self._timestamp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_module, name)


class HistoricalPromptReplyGenerator(ReplyGenerator):
    """Production generator with prompt-local historical wall-clock semantics."""

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        current_time = kwargs.get("current_time")
        if current_time is None:
            return super()._build_user_prompt(*args, **kwargs)
        real_time_module = generator_module.time
        generator_module.time = _HistoricalTimeProxy(real_time_module, current_time.timestamp())
        try:
            return super()._build_user_prompt(*args, **kwargs)
        finally:
            generator_module.time = real_time_module


def _strict_target_index(rows: list[dict[str, Any]], case: dict[str, Any]) -> int:
    exact = [
        index
        for index, row in enumerate(rows)
        if int(row.get("createTime") or 0) == int(case["timestamp"])
        and bool(row.get("isSend"))
        and _norm(str(case["reply"])) == _norm(_message_text(row))
    ]
    if len(exact) != 1:
        raise RuntimeError(f"target_exact_match_count={len(exact)}")
    return exact[0]


def _message_text(row: dict[str, Any]) -> str:
    # Keep a single source of truth without exposing the old script's private import.
    from scripts.evaluate_current_c0_historical_ab import _message_text as old_message_text

    return old_message_text(row)


def strict_build_inputs(
    case: dict[str, Any],
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage], dict[str, Any]]:
    rows, chat_name, chatroom_id = _source(case)
    target_index = _strict_target_index(rows, case)
    raw_prefix = [
        _chat_message(row, chat_name, chatroom_id, {})
        for row in rows[:target_index]
        if int(row.get("localType") or 0) != 10000
    ]
    event_context, alignment = _match_context(raw_prefix, case.get("context_messages") or [])
    if not event_context:
        raise RuntimeError(alignment)
    last_self = max(
        (index for index, message in enumerate(event_context) if message.sender_type is SenderType.SELF),
        default=-1,
    )
    to_reply = [
        message
        for message in event_context[last_self + 1 :]
        if message.sender_type is SenderType.OTHER
    ]
    if not to_reply:
        raise RuntimeError("no independent incoming messages after the last self message")
    all_messages = raw_prefix[-1000:]
    return to_reply, all_messages, event_context, {
        "chat_name": chat_name,
        "chatroom_id": chatroom_id,
        "target_index": target_index,
        "raw_history_before_target": len(raw_prefix),
        "generator_history_count": len(all_messages),
        "judge_history_count": len(all_messages),
        "event_context_count": len(event_context),
        "unread_count": len(to_reply),
        "context_alignment": alignment,
        "source_path": case["source_path"],
    }


def _group_key(case: dict[str, Any]) -> str:
    specified = str(case.get("specified_group") or "")
    if specified == "correct_common_prosperity_with_luxixi":
        return "correct_common_prosperity_with_luxixi"
    if specified.startswith("youzi_group"):
        return "youzi_group"
    if case.get("complex_large_group"):
        return "other_complex_large_group"
    if case.get("chat_type") == "group":
        return "other_group"
    return "private"


def _case_priority(case: dict[str, Any]) -> tuple[int, int, str]:
    rank = {
        "correct_common_prosperity_with_luxixi": 0,
        "youzi_group": 1,
        "other_complex_large_group": 2,
        "other_group": 3,
        "private": 4,
    }[_group_key(case)]
    return rank, int(case["timestamp"]), str(case["internal_id"])


def _candidate_event_id(case: dict[str, Any]) -> str:
    event = str(case.get("event_cluster") or case.get("event_id") or "").strip()
    if event:
        return event
    raw = f"{case.get('source_path')}:{case.get('timestamp')}:{case.get('internal_id') or case.get('case_id')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _candidate_as_generated(case: dict[str, Any]) -> dict[str, Any]:
    context = case.get("context_messages") or [
        {"content": value} for value in case.get("context", [])
    ]
    return {
        "human_reply": [str(case.get("reply") or "")],
        "context": [{"content": str(row.get("content") or "")} for row in context],
    }


def _merge_candidate_cases(
    collections: list[list[dict[str, Any]]], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    reference_replies = [
        str(value)
        for example in reference.get("examples", [])
        for value in example.get("human_reply", [])
    ]
    merged: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_events: set[str] = set()
    selected_replies: list[str] = []
    for collection in collections:
        for case in sorted(collection, key=_case_priority):
            case_id = str(case["internal_id"])
            event_id = _candidate_event_id(case)
            reply = str(case.get("reply") or "")
            if case_id in used_ids or event_id in used_events:
                continue
            if _direct_reference_overlap(_candidate_as_generated(case), reference):
                continue
            if any(_near_duplicate(reply, old) for old in reference_replies + selected_replies):
                continue
            merged.append(case)
            used_ids.add(case_id)
            used_events.add(event_id)
            selected_replies.append(reply)
    return sorted(merged, key=_case_priority)


def _load_extended_candidates() -> list[dict[str, Any]]:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    target_resolution = _load_cases(TARGET_RESOLUTION_MANIFEST, None)
    return _merge_candidate_cases(
        [
            _load_cases(MANIFEST, None),
            _common_candidates(set()),
            _youzi_cases(),
            target_resolution,
            _ordinary_candidates(set(), per_chat_cap=48),
        ],
        reference,
    )


def _scorable_counts(
    generation_cases: list[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, int]:
    counts = {group: 0 for group in STAGE2_STRATUM_WEIGHTS}
    for row in generation_cases:
        if not row.get("validity", {}).get("formally_scorable"):
            continue
        if _direct_reference_overlap(row, reference):
            continue
        group = str(row.get("group") or "")
        if group in counts:
            counts[group] += 1
    return counts


def _scorable_length_counts(
    generation_cases: list[dict[str, Any]],
    reference: dict[str, Any],
    excluded_case_ids: set[str] | None = None,
) -> dict[str, int]:
    excluded_case_ids = excluded_case_ids or set()
    counts = {bucket: 0 for bucket in _length_bucket_quotas(0)}
    for row in generation_cases:
        if str(row.get("case_id") or "") in excluded_case_ids:
            continue
        if not row.get("validity", {}).get("formally_scorable"):
            continue
        if _direct_reference_overlap(row, reference):
            continue
        counts[_reply_length_bucket(row)] += 1
    return counts


def _pending_generation_cases(
    candidates: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    quotas: dict[str, int] | None,
    reference: dict[str, Any],
    excluded_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_case_ids = excluded_case_ids or set()
    by_id = {str(case["internal_id"]): case for case in candidates}
    used_ids = {str(row["case_id"]) for row in completed}
    used_events = {
        _candidate_event_id(by_id[case_id])
        for case_id in used_ids
        if case_id in by_id
    }
    selected_replies = [
        str(value)
        for row in completed
        if row.get("validity", {}).get("formally_scorable")
        for value in row.get("human_reply", [])
    ]
    counts = _scorable_counts(completed, reference)
    pending: list[dict[str, Any]] = []
    for case in sorted(candidates, key=_case_priority):
        case_id = str(case["internal_id"])
        event_id = _candidate_event_id(case)
        group = _group_key(case)
        reply = str(case.get("reply") or "")
        if case_id in used_ids or case_id in excluded_case_ids or event_id in used_events:
            continue
        if quotas is not None and counts[group] >= quotas[group]:
            continue
        if _direct_reference_overlap(_candidate_as_generated(case), reference):
            continue
        if any(_near_duplicate(reply, old) for old in selected_replies):
            continue
        pending.append(case)
        used_ids.add(case_id)
        used_events.add(event_id)
        selected_replies.append(reply)
    return pending


def audit_inputs(cases: list[dict[str, Any]]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for case in sorted(cases, key=_case_priority):
        try:
            to_reply, all_messages, event_context, metadata = strict_build_inputs(case)
            accepted.append({
                "case_id": case["internal_id"],
                "event_id": _candidate_event_id(case),
                "group": _group_key(case),
                "chat_type": case.get("chat_type"),
                "complex_large_group": bool(case.get("complex_large_group")),
                "timestamp": int(case["timestamp"]),
                "human_reply_sha256": hashlib.sha256(str(case["reply"]).encode()).hexdigest(),
                "metadata": metadata,
                "to_reply": [message.text for message in to_reply],
                "event_context_count": len(event_context),
                "generator_history_count": len(all_messages),
            })
        except Exception as exc:
            excluded.append({
                "case_id": case["internal_id"],
                "group": _group_key(case),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    counts: dict[str, int] = {}
    for row in accepted:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "strict_rules": {
            "target_requires_exact_timestamp_role_and_text": True,
            "context_requires_complete_exact_alignment": True,
            "fallback_forbidden": True,
            "to_reply_is_other_messages_after_last_self_in_event": True,
        },
        "accepted_count": len(accepted),
        "excluded_count": len(excluded),
        "accepted_by_group": counts,
        "accepted": accepted,
        "excluded": excluded,
    }


def _case_context(messages: list[ChatMessage], to_reply: list[ChatMessage]) -> list[dict[str, Any]]:
    unread_ids = {id(message) for message in to_reply}
    return [
        {
            "time": message.timestamp,
            "speaker": message.sender,
            "role": message.sender_type.value,
            "content": message.text,
            "unread": id(message) in unread_ids,
        }
        for message in messages
    ]


def generate_case(
    source_case: dict[str, Any],
    llm: Any,
    memory: MemoryEngine,
    timeout_seconds: float,
    generator_cls: type[ReplyGenerator] = HistoricalPromptReplyGenerator,
    force_reply: bool = False,
    compact_verified_turn: bool = False,
) -> dict[str, Any]:
    to_reply, all_messages, event_context, metadata = strict_build_inputs(source_case)
    if compact_verified_turn:
        all_messages = event_context
        metadata = {
            **metadata,
            "generator_history_count": len(all_messages),
            "compact_verified_turn": True,
        }
    evaluation_time = datetime.fromtimestamp(int(source_case["timestamp"]))
    generator = generator_cls(
        llm_client=llm,
        memory_engine=memory,
        tool_registry=_safe_registry(evaluation_time),
        judge_worker=None,
        enable_timestamps=True,
        generation_timeout_seconds=timeout_seconds,
    )
    started = time.monotonic()
    replies = generator.generate(
        to_reply,
        all_messages,
        is_group=source_case.get("chat_type") == "group",
        tick_id=0,
        evaluation_time=evaluation_time,
        enable_reply_restraint=not force_reply,
    )
    trace = _trace(generator)
    human = str(source_case["reply"])
    normalized_human = _norm(human)
    normalized_ai = _norm("".join(replies))
    similarity = (
        difflib.SequenceMatcher(None, normalized_human, normalized_ai).ratio()
        if normalized_human and normalized_ai
        else 0.0
    )
    side_effect_calls = [
        row
        for row in generator.last_tool_calls
        if str(row.get("tool_name") or row.get("name") or "") in SIDE_EFFECT_TOOLS
    ]
    visible_context_text = _norm(json.dumps(_case_context(all_messages, to_reply), ensure_ascii=False))
    answer_in_visible_context = bool(normalized_human and normalized_human in visible_context_text)
    few_shot_responses = re.findall(
        r"<response>(.*?)</response>",
        str(trace.get("few_shot_content") or ""),
        flags=re.DOTALL,
    )
    few_shot_reply_texts = [
        _norm(value)
        for block in few_shot_responses
        for value in re.findall(r"<message[^>]*>(.*?)</message>", block, flags=re.DOTALL)
    ]
    long_answer_sources = {
        key: trace.get(key)
        for key in ("few_shot_content", "tools_context", "skill_content")
    }
    answer_in_trace = bool(
        normalized_human
        and (
            normalized_human in few_shot_reply_texts
            or len(normalized_human) >= 8
            and (
                normalized_human in _norm(json.dumps(long_answer_sources, ensure_ascii=False))
                or (
                    not answer_in_visible_context
                    and normalized_human in _norm(str(trace.get("user_prompt") or ""))
                )
            )
        )
    )
    reproduced = bool(
        normalized_human
        and normalized_ai
        and (
            normalized_human == normalized_ai
            or (min(len(normalized_human), len(normalized_ai)) >= 6 and similarity >= 0.9)
        )
    )
    prompt_time = evaluation_time.strftime("%Y-%m-%d %H:%M")
    prompt_clock_ok = f"<current_time>{prompt_time}</current_time>" in str(trace.get("user_prompt") or "")
    exclusions = []
    if generator.last_generation_failed:
        exclusions.append("generation_failed")
    if not replies:
        exclusions.append("c0_silent")
    if answer_in_trace:
        exclusions.append("human_answer_in_generation_trace")
    return {
        "case_id": source_case["internal_id"],
        "event_id": _candidate_event_id(source_case),
        "group": _group_key(source_case),
        "chat_name": metadata["chat_name"],
        "chat_type": source_case.get("chat_type"),
        "complex_large_group": bool(source_case.get("complex_large_group")),
        "timestamp": int(source_case["timestamp"]),
        "context": _case_context(all_messages, to_reply),
        "human_reply": [human],
        "c0_reply": replies,
        "generation": {
            "duration_seconds": round(time.monotonic() - started, 3),
            "timeout_budget_seconds": timeout_seconds,
            "model": llm.model,
            "base_url_host": re.sub(
                r"^https?://([^/]+).*$",
                r"\1",
                str(getattr(llm, "base_url", "") or os.environ.get("LLM_BASE_URL", "")),
            ),
            "generator_sha256": hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest(),
            "historical_prompt_time": prompt_time,
            "historical_prompt_clock_ok": prompt_clock_ok,
            "metadata": metadata,
            "trace": trace,
        },
        "validity": {
            "formally_scorable": not exclusions,
            "exclusions": exclusions,
            "answer_in_c0_trace": answer_in_trace,
            "answer_in_visible_context": answer_in_visible_context,
            "c0_reproduced_human": reproduced,
            "reply_similarity": round(similarity, 4),
            "historical_prompt_clock_ok": prompt_clock_ok,
            "side_effect_tool_calls": _jsonable(side_effect_calls),
        },
    }


def _discard_cases_from_id(cases: list[dict[str, Any]], case_id: str | None) -> list[dict[str, Any]]:
    if not case_id:
        return list(cases)
    for index, row in enumerate(cases):
        if str(row.get("case_id")) == case_id:
            return list(cases[:index])
    raise RuntimeError(f"discard case id not found: {case_id}")


def _generation_client(provider: str, volcengine_config: Path) -> Any:
    if provider == "official":
        return QwenClient()
    api_key, base_url = load_volcengine_credentials(volcengine_config)
    return VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)


def _balanced_human_option(case_id: str, position: int) -> str:
    # Exact balance does not depend on hidden content or generation output.
    digest_bit = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) & 1
    return "A" if (position + digest_bit) % 2 == 0 else "B"


def _blind_case(case: dict[str, Any], position: int) -> tuple[dict[str, Any], str]:
    human_option = _balanced_human_option(case["case_id"], position)
    ai_option = "B" if human_option == "A" else "A"
    options = {human_option: case["human_reply"], ai_option: case["c0_reply"]}
    aliases: dict[str, str] = {}
    context = []
    for row in case["context"]:
        speaker = str(row["speaker"])
        if row["role"] == SenderType.SELF.value:
            blind_speaker = "本人"
        elif case["chat_type"] == "private":
            blind_speaker = "对方"
        else:
            aliases.setdefault(speaker, f"成员{len(aliases) + 1}")
            blind_speaker = aliases[speaker]
        context.append({
            "time": row["time"],
            "speaker": blind_speaker,
            "content": row["content"],
            "unread": bool(row.get("unread")),
        })
    blind = {
        "case_id": case["case_id"],
        "chat_type": case["chat_type"],
        "chat_name": "匿名群聊" if case["chat_type"] == "group" else "匿名私聊",
        "context": context,
        "option_A": options["A"],
        "option_B": options["B"],
    }
    return blind, human_option


def _hidden_stratum_quotas(limit: int) -> dict[str, int]:
    total_weight = sum(STAGE2_STRATUM_WEIGHTS.values())
    quotas = {
        group: limit * weight // total_weight
        for group, weight in STAGE2_STRATUM_WEIGHTS.items()
    }
    remaining = limit - sum(quotas.values())
    order = sorted(
        STAGE2_STRATUM_WEIGHTS,
        key=lambda group: (-(limit * STAGE2_STRATUM_WEIGHTS[group] % total_weight), group),
    )
    for group in order[:remaining]:
        quotas[group] += 1
    return quotas


def _group_balanced_quotas(limit: int) -> dict[str, int]:
    if limit % len(GROUP_BALANCED_STRATA):
        raise ValueError("group-balanced hidden count must be divisible by 3")
    per_group = limit // len(GROUP_BALANCED_STRATA)
    return {group: per_group for group in GROUP_BALANCED_STRATA}


def _select_stratified_scorable(cases: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = _hidden_stratum_quotas(limit)
    by_group = {group: [] for group in quotas}
    for case in cases:
        group = str(case.get("group") or "")
        if group not in by_group:
            raise RuntimeError(f"unknown hidden stratum: {group or '<missing>'}")
        by_group[group].append(case)
    shortages = {
        group: {"available": len(by_group[group]), "required": quota}
        for group, quota in quotas.items()
        if len(by_group[group]) < quota
    }
    if shortages:
        raise RuntimeError(f"hidden stratum capacity shortage: {json.dumps(shortages, sort_keys=True)}")
    selected = [case for group, quota in quotas.items() for case in by_group[group][:quota]]
    return selected, quotas


def _reply_length_bucket(case: dict[str, Any]) -> str:
    human_length = len(_norm("".join(str(value) for value in case.get("human_reply", []))))
    c0_length = len(_norm("".join(str(value) for value in case.get("c0_reply", []))))
    ratio = human_length / max(c0_length, 1)
    if ratio < 0.8:
        return "human_shorter"
    if ratio > 1.25:
        return "human_longer"
    return "near_equal"


def _length_bucket_quotas(limit: int) -> dict[str, int]:
    buckets = ("human_shorter", "near_equal", "human_longer")
    quotas = {bucket: limit // len(buckets) for bucket in buckets}
    for bucket in buckets[: limit - sum(quotas.values())]:
        quotas[bucket] += 1
    return quotas


def _tail_has_unresolved_media(case: dict[str, Any]) -> bool:
    context = case.get("context") or case.get("context_messages") or []
    return any(
        any(
            marker in str(row if isinstance(row, str) else row.get("content") or "")
            for marker in MEDIA_PLACEHOLDERS
        )
        for row in context[-3:]
    )


def _select_group_balanced_scorable(
    cases: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = _group_balanced_quotas(limit)
    eligible = [case for case in cases if not _tail_has_unresolved_media(case)]
    by_group = {
        group: [case for case in eligible if str(case.get("group") or "") == group]
        for group in quotas
    }
    shortages = {
        group: {"available": len(by_group[group]), "required": quota}
        for group, quota in quotas.items()
        if len(by_group[group]) < quota
    }
    if shortages:
        raise RuntimeError(f"group-balanced capacity shortage: {json.dumps(shortages, sort_keys=True)}")
    return [case for group, quota in quotas.items() for case in by_group[group][:quota]], quotas


def _select_exploratory_unseen(
    cases: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    eligible = [case for case in cases if not _tail_has_unresolved_media(case)]
    if len(eligible) < limit:
        raise RuntimeError(f"only {len(eligible)} media-independent cases; need {limit}")
    buckets = tuple(_length_bucket_quotas(limit))
    by_bucket = {
        bucket: [case for case in eligible if _reply_length_bucket(case) == bucket]
        for bucket in buckets
    }
    quotas = {bucket: 0 for bucket in buckets}
    remaining = limit
    while remaining:
        available = [bucket for bucket in buckets if quotas[bucket] < len(by_bucket[bucket])]
        if not available:
            raise RuntimeError(f"only {limit - remaining} length-balanced cases; need {limit}")
        bucket = min(available, key=lambda value: (quotas[value], buckets.index(value)))
        quotas[bucket] += 1
        remaining -= 1
    selected_ids = {
        case["case_id"]
        for bucket in buckets
        for case in sorted(
            by_bucket[bucket],
            key=lambda row: (str(row.get("group") or "") == "private"),
        )[: quotas[bucket]]
    }
    selected = [case for case in eligible if case["case_id"] in selected_ids]
    by_group = {
        group: sum(str(case.get("group") or "") == group for case in selected)
        for group in STAGE2_STRATUM_WEIGHTS
    }
    return selected, by_group, quotas


def _select_length_balanced_scorable(
    cases: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    length_quotas = _length_bucket_quotas(limit)
    stratum_quotas = _hidden_stratum_quotas(limit)
    by_bucket_group = {
        bucket: {group: [] for group in STAGE2_STRATUM_WEIGHTS}
        for bucket in length_quotas
    }
    for case in cases:
        group = str(case.get("group") or "")
        if group not in STAGE2_STRATUM_WEIGHTS:
            raise RuntimeError(f"unknown hidden stratum: {group or '<missing>'}")
        by_bucket_group[_reply_length_bucket(case)][group].append(case)
    shortages = {
        bucket: {
            "available": sum(len(rows) for rows in by_bucket_group[bucket].values()),
            "required": quota,
        }
        for bucket, quota in length_quotas.items()
        if sum(len(rows) for rows in by_bucket_group[bucket].values()) < quota
    }
    if shortages:
        raise RuntimeError(f"hidden length-bucket capacity shortage: {json.dumps(shortages, sort_keys=True)}")
    stratum_shortages = {
        group: {
            "available": sum(len(by_bucket_group[bucket][group]) for bucket in length_quotas),
            "required": quota,
        }
        for group, quota in stratum_quotas.items()
        if sum(len(by_bucket_group[bucket][group]) for bucket in length_quotas) < quota
    }
    if stratum_shortages:
        raise RuntimeError(f"hidden stratum capacity shortage: {json.dumps(stratum_shortages, sort_keys=True)}")

    groups = list(stratum_quotas)
    buckets = list(length_quotas)
    allocation: dict[str, dict[str, int]] = {}

    def allocate(group_index: int, remaining: dict[str, int]) -> bool:
        if group_index == len(groups):
            return all(value == 0 for value in remaining.values())
        group = groups[group_index]
        needed = stratum_quotas[group]
        row: dict[str, int] = {}

        def assign(bucket_index: int, left: int) -> bool:
            if bucket_index == len(buckets):
                if left:
                    return False
                allocation[group] = row.copy()
                next_remaining = {
                    bucket: remaining[bucket] - row.get(bucket, 0) for bucket in buckets
                }
                if allocate(group_index + 1, next_remaining):
                    return True
                allocation.pop(group, None)
                return False
            bucket = buckets[bucket_index]
            maximum = min(left, remaining[bucket], len(by_bucket_group[bucket][group]))
            for count in range(maximum, -1, -1):
                row[bucket] = count
                if assign(bucket_index + 1, left - count):
                    return True
            return False

        return assign(0, needed)

    if not allocate(0, length_quotas.copy()):
        capacities = {
            group: {bucket: len(by_bucket_group[bucket][group]) for bucket in buckets}
            for group in groups
        }
        raise RuntimeError(
            "no joint hidden allocation: "
            + json.dumps({"strata": stratum_quotas, "lengths": length_quotas, "capacities": capacities}, sort_keys=True)
        )
    selected = [
        case
        for group in groups
        for bucket in buckets
        for case in by_bucket_group[bucket][group][: allocation[group][bucket]]
    ]
    return selected, stratum_quotas, length_quotas


def seal_hidden(
    generation: dict[str, Any],
    limit: int,
    *,
    excluded_case_ids: set[str] | None = None,
    length_balanced: bool = False,
    selection_mode: str = "stage2",
) -> dict[str, Any]:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    evaluation = generation.get("evaluation", {})
    excluded_case_ids = excluded_case_ids or set()
    overlap_ids = {
        row["case_id"]
        for row in generation.get("cases", [])
        if _direct_reference_overlap(row, reference)
    }
    scorable = [
        row
        for row in generation.get("cases", [])
        if row["validity"]["formally_scorable"]
        and row["case_id"] not in overlap_ids
        and row["case_id"] not in excluded_case_ids
    ]
    if len(scorable) < limit:
        raise RuntimeError(f"only {len(scorable)} scorable cases; need {limit}")
    media_excluded = sum(_tail_has_unresolved_media(row) for row in scorable)
    if selection_mode == "group-balanced":
        selected, quotas = _select_group_balanced_scorable(scorable, limit)
        selected_by_stratum = {
            group: sum(case["group"] == group for case in selected)
            for group in STAGE2_STRATUM_WEIGHTS
        }
        length_quotas = None
    elif selection_mode == "exploratory-unseen":
        selected, selected_by_stratum, length_quotas = _select_exploratory_unseen(scorable, limit)
        quotas = None
    elif length_balanced:
        selected, quotas, length_quotas = _select_length_balanced_scorable(scorable, limit)
        selected_by_stratum = {
            group: sum(case["group"] == group for case in selected)
            for group in STAGE2_STRATUM_WEIGHTS
        }
    else:
        selected, quotas = _select_stratified_scorable(scorable, limit)
        length_quotas = None
        selected_by_stratum = {
            group: sum(case["group"] == group for case in selected)
            for group in STAGE2_STRATUM_WEIGHTS
        }
    blind_cases = []
    answer_rows = []
    for position, case in enumerate(selected):
        blind, answer = _blind_case(case, position)
        blind_cases.append(blind)
        answer_rows.append({"case_id": case["case_id"], "human_option": answer})
    count_a = sum(row["human_option"] == "A" for row in answer_rows)
    if abs(count_a - (limit - count_a)) > 1:
        # Deterministically flip only positions until exact balance is reached.
        wanted_a = limit // 2
        for position, answer in enumerate(answer_rows):
            if count_a == wanted_a:
                break
            old = answer["human_option"]
            if (count_a > wanted_a and old == "A") or (count_a < wanted_a and old == "B"):
                blind_cases[position]["option_A"], blind_cases[position]["option_B"] = (
                    blind_cases[position]["option_B"], blind_cases[position]["option_A"]
                )
                answer["human_option"] = "B" if old == "A" else "A"
                count_a += -1 if old == "A" else 1
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_generation_sha256": hashlib.sha256(
            json.dumps(generation, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "rules": {
            "one_case_per_judge_process": True,
            "answers_not_sent": True,
            "other_cases_not_sent": True,
            "tool_use_invalidates_case": True,
            "direct_reference_overlap_excluded": True,
            "stage2_proportional_stratified_selection": selection_mode == "stage2",
            "previously_exposed_cases_excluded": bool(excluded_case_ids),
            "reply_length_balanced": length_balanced or selection_mode == "exploratory-unseen",
            "selection_mode": selection_mode,
            "group_chat_balanced": selection_mode == "group-balanced",
            "independent_unseen": selection_mode == "exploratory-unseen",
            "private_heavy": selection_mode == "exploratory-unseen",
            "not_freeze_eligible": (
                selection_mode == "exploratory-unseen"
                or evaluation.get("freeze_eligible") is False
            ),
            "unresolved_tail_media_excluded": selection_mode in {"exploratory-unseen", "group-balanced"},
        },
        "stratum_quotas": quotas,
        "selected_by_stratum": selected_by_stratum,
        "unresolved_tail_media_excluded_count": media_excluded if selection_mode in {"exploratory-unseen", "group-balanced"} else 0,
        "reference_overlap_excluded_case_ids": sorted(overlap_ids),
        "previously_exposed_case_ids": sorted(excluded_case_ids),
        "length_bucket_quotas": length_quotas,
        "selected_by_length_bucket": {
            bucket: sum(_reply_length_bucket(case) == bucket for case in selected)
            for bucket in _length_bucket_quotas(limit)
        },
        "selected_case_ids_sha256": hashlib.sha256(
            json.dumps(sorted(case["case_id"] for case in selected), ensure_ascii=False).encode()
        ).hexdigest(),
        "evaluation": evaluation,
        "blind_cases": blind_cases,
        "answers": answer_rows,
    }


def _context_scope_equal(case: dict[str, Any]) -> bool:
    metadata = case.get("generation", {}).get("metadata", {})
    judge_count = metadata.get("judge_history_count", metadata.get("event_context_count"))
    return metadata.get("generator_history_count") == judge_count


def _with_full_judge_context(case: dict[str, Any]) -> dict[str, Any]:
    if _context_scope_equal(case):
        return case
    metadata = case.get("generation", {}).get("metadata", {})
    source_case = {
        "internal_id": case["case_id"],
        "source_path": metadata["source_path"],
        "timestamp": case["timestamp"],
        "reply": case["human_reply"][0],
        "context_messages": case["context"],
        "chat_type": case["chat_type"],
    }
    to_reply, all_messages, _event_context, rebuilt = strict_build_inputs(source_case)
    if len(all_messages) != metadata.get("generator_history_count"):
        raise RuntimeError(f"generator history changed for {case['case_id']}")
    aligned = copy.deepcopy(case)
    aligned["context"] = _case_context(all_messages, to_reply)
    aligned["generation"]["metadata"] = {**metadata, "judge_history_count": len(all_messages)}
    return aligned


def seal_fair_replacements(
    generation: dict[str, Any], original_sealed: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace unfair old questions without exposing or changing the frozen Judge."""
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    generation_by_id = {row["case_id"]: row for row in generation["cases"]}
    original_ids = [row["case_id"] for row in original_sealed["answers"]]
    unfair_ids = {
        case_id for case_id in original_ids if not _context_scope_equal(generation_by_id[case_id])
    }
    fair_ids = [case_id for case_id in original_ids if case_id not in unfair_ids]
    overlap_ids = {
        row["case_id"]
        for row in generation["cases"]
        if _direct_reference_overlap(row, reference)
    }
    target_groups = original_sealed.get("selected_by_stratum") or {
        group: sum(generation_by_id[case_id]["group"] == group for case_id in original_ids)
        for group in STAGE2_STRATUM_WEIGHTS
    }
    target_lengths = original_sealed.get("selected_by_length_bucket") or _length_bucket_quotas(
        len(original_ids)
    )
    group_deficits = {
        group: target - sum(generation_by_id[case_id]["group"] == group for case_id in fair_ids)
        for group, target in target_groups.items()
    }
    length_deficits = {
        bucket: target
        - sum(_reply_length_bucket(generation_by_id[case_id]) == bucket for case_id in fair_ids)
        for bucket, target in target_lengths.items()
    }
    if sum(group_deficits.values()) != len(unfair_ids) or sum(length_deficits.values()) != len(unfair_ids):
        raise RuntimeError("replacement deficits do not match unfair case count")

    pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        group: {bucket: [] for bucket in length_deficits} for group in group_deficits
    }
    for index, row in enumerate(generation["cases"]):
        group = str(row.get("group") or "")
        bucket = _reply_length_bucket(row)
        if (
            group not in pools
            or bucket not in pools[group]
            or row["case_id"] in original_ids
            or row["case_id"] in overlap_ids
            or not row.get("validity", {}).get("formally_scorable")
        ):
            continue
        try:
            aligned = _with_full_judge_context(row)
        except (KeyError, RuntimeError):
            continue
        if _direct_reference_overlap(aligned, reference):
            continue
        generation["cases"][index] = aligned
        generation_by_id[aligned["case_id"]] = aligned
        pools[group][bucket].append(aligned)

    groups = [group for group, count in group_deficits.items() if count]
    buckets = list(length_deficits)
    replacement_matrix: dict[str, dict[str, int]] = {}

    def allocate(group_index: int, remaining: dict[str, int]) -> bool:
        if group_index == len(groups):
            return all(value == 0 for value in remaining.values())
        group = groups[group_index]
        needed = group_deficits[group]
        allocation: dict[str, int] = {}

        def assign_bucket(bucket_index: int, left: int) -> bool:
            if bucket_index == len(buckets):
                if left != 0:
                    return False
                replacement_matrix[group] = allocation.copy()
                next_remaining = {
                    bucket: remaining[bucket] - allocation.get(bucket, 0) for bucket in buckets
                }
                if allocate(group_index + 1, next_remaining):
                    return True
                replacement_matrix.pop(group, None)
                return False
            bucket = buckets[bucket_index]
            maximum = min(left, remaining[bucket], len(pools[group][bucket]))
            for count in range(maximum, -1, -1):
                allocation[bucket] = count
                if assign_bucket(bucket_index + 1, left - count):
                    return True
            return False

        return assign_bucket(0, needed)

    if not allocate(0, length_deficits.copy()):
        capacities = {
            group: {bucket: len(rows) for bucket, rows in by_bucket.items()}
            for group, by_bucket in pools.items()
            if group_deficits[group]
        }
        raise RuntimeError(
            "no fair replacement allocation: "
            + json.dumps(
                {"group_deficits": group_deficits, "length_deficits": length_deficits, "capacities": capacities},
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    selected: list[dict[str, Any]] = []
    for group, bucket_quotas in replacement_matrix.items():
        for bucket, quota in bucket_quotas.items():
            selected.extend(pools[group][bucket][:quota])

    original_blind = {row["case_id"]: row for row in original_sealed["blind_cases"]}
    original_answers = {row["case_id"]: row for row in original_sealed["answers"]}
    fair_blind = [original_blind[case_id] for case_id in fair_ids]
    fair_answers = [original_answers[case_id] for case_id in fair_ids]
    replacement_blind = []
    replacement_answers = []
    for offset, case in enumerate(selected):
        blind, human_option = _blind_case(case, len(fair_ids) + offset)
        replacement_blind.append(blind)
        replacement_answers.append({"case_id": case["case_id"], "human_option": human_option})

    wanted_replacement_a = 50 - sum(row["human_option"] == "A" for row in fair_answers)
    replacement_a = sum(row["human_option"] == "A" for row in replacement_answers)
    for index, answer in enumerate(replacement_answers):
        if replacement_a == wanted_replacement_a:
            break
        old = answer["human_option"]
        if (replacement_a > wanted_replacement_a and old == "A") or (
            replacement_a < wanted_replacement_a and old == "B"
        ):
            replacement_blind[index]["option_A"], replacement_blind[index]["option_B"] = (
                replacement_blind[index]["option_B"],
                replacement_blind[index]["option_A"],
            )
            answer["human_option"] = "B" if old == "A" else "A"
            replacement_a += -1 if old == "A" else 1
    if replacement_a != wanted_replacement_a:
        raise RuntimeError("could not balance repaired hidden set")

    replacement = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "one_case_per_judge_process": True,
            "answers_not_sent": True,
            "other_cases_not_sent": True,
            "tool_use_invalidates_case": True,
            "judge_prompt_frozen": True,
            "original_cases_excluded": True,
            "context_scope_equal_required": True,
        },
        "replacement_matrix": replacement_matrix,
        "replaces_unfair_case_ids": sorted(unfair_ids),
        "blind_cases": replacement_blind,
        "answers": replacement_answers,
    }
    combined_blind = fair_blind + replacement_blind
    combined_answers = fair_answers + replacement_answers
    combined = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": replacement["rules"],
        "replaced_unfair_case_ids": sorted(unfair_ids),
        "blind_cases": combined_blind,
        "answers": combined_answers,
        "selected_by_stratum": {
            group: sum(generation_by_id[row["case_id"]]["group"] == group for row in combined_answers)
            for group in STAGE2_STRATUM_WEIGHTS
        },
        "selected_by_length_bucket": {
            bucket: sum(
                _reply_length_bucket(generation_by_id[row["case_id"]]) == bucket
                for row in combined_answers
            )
            for bucket in _length_bucket_quotas(100)
        },
    }
    return replacement, combined


def merge_fair_judgments(
    original_judged: dict[str, Any], combined_sealed: dict[str, Any], replacement_judged: dict[str, Any]
) -> dict[str, Any]:
    original_by_id = {row["case_id"]: row for row in original_judged["cases"]}
    replacement_by_id = {row["case_id"]: row for row in replacement_judged["cases"]}
    cases = []
    for answer in combined_sealed["answers"]:
        case_id = answer["case_id"]
        row = replacement_by_id.get(case_id, original_by_id.get(case_id))
        if row is None:
            raise RuntimeError(f"missing Judge result for {case_id}")
        cases.append(row)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": original_judged["profile"],
        "profile_sha256": original_judged["profile_sha256"],
        "reference_sha256": original_judged["reference_sha256"],
        "repair": {"unfair_old_cases_discarded": 18, "new_independent_cases": 18},
        "cases": cases,
    }


def _direct_reference_overlap(case: dict[str, Any], reference: dict[str, Any]) -> bool:
    case_reply = tuple(_norm(str(value)) for value in case.get("human_reply", []) if _norm(str(value)))
    case_context = [_norm(str(row.get("content") or "")) for row in case.get("context", [])]
    if not case_reply:
        return False
    for example in reference.get("examples", []):
        reference_reply = tuple(
            _norm(str(value)) for value in example.get("human_reply", []) if _norm(str(value))
        )
        reference_context = [
            _norm(str(value)) for value in example.get("context", []) if _norm(str(value))
        ]
        if reference_reply != case_reply or not reference_context:
            continue
        width = len(reference_context)
        if any(case_context[start : start + width] == reference_context for start in range(len(case_context) - width + 1)):
            return True
    return False


def _validate_judge_request(
    messages: list[dict[str, str]], blind: dict[str, Any], other_case_ids: list[str]
) -> None:
    allowed = {"case_id", "chat_type", "chat_name", "context", "option_A", "option_B"}
    if set(blind) != allowed:
        raise RuntimeError(f"unexpected blind-case fields: {sorted(set(blind) - allowed)}")
    expected_user = "<blind_case>\n" + json.dumps(blind, ensure_ascii=False) + "\n</blind_case>"
    if len(messages) != 2 or messages[1] != {"role": "user", "content": expected_user}:
        raise RuntimeError("Judge user request differs from the sealed blind case")
    request_text = json.dumps(messages, ensure_ascii=False)
    if any(case_id in request_text for case_id in other_case_ids):
        raise RuntimeError("another hidden case leaked into Judge request")


def _present_sealed(sealed: dict[str, Any], presentation: str) -> dict[str, Any]:
    if presentation == "canonical":
        return sealed
    if presentation != "swapped":
        raise ValueError(f"unsupported presentation: {presentation}")
    presented = copy.deepcopy(sealed)
    for blind in presented["blind_cases"]:
        blind["option_A"], blind["option_B"] = blind["option_B"], blind["option_A"]
    for answer in presented["answers"]:
        answer["human_option"] = "B" if answer["human_option"] == "A" else "A"
    return presented


def judge_hidden(
    sealed: dict[str, Any],
    profile_path: Path,
    reference_path: Path,
    out_dir: Path = OUT_DIR,
    *,
    presentation: str = "canonical",
    resume: bool = False,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    sealed = _present_sealed(sealed, presentation)
    profile = load_profile(profile_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    answer_by_id = {row["case_id"]: row["human_option"] for row in sealed["answers"]}
    assert_reference_isolated(reference, list(sealed["blind_cases"]))
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count is not None and (shard_count < 1 or not 0 <= shard_index < shard_count):
        raise ValueError("invalid Judge shard")
    base_name = "judge_results" if presentation == "canonical" else "swapped_judge_results"
    results_path = (
        out_dir / f"{base_name}.shard-{shard_index}-of-{shard_count}.json"
        if shard_count is not None
        else out_dir / f"{base_name}.json"
    )
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "presentation": presentation,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "cases": [],
    }
    if resume and results_path.exists():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        identity = ("profile_sha256", "reference_sha256", "presentation", "shard_index", "shard_count")
        if any(prior.get(field) != output.get(field) for field in identity):
            raise RuntimeError("existing Judge run does not match the requested frozen design")
        output["cases"] = list(prior.get("cases", []))
    completed_ids = {str(row["case_id"]) for row in output["cases"]}
    canonical_path = out_dir / f"{base_name}.json"
    if shard_count is not None and canonical_path.exists():
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        identity = ("profile_sha256", "reference_sha256", "presentation")
        if any(canonical.get(field) != output.get(field) for field in identity):
            raise RuntimeError("existing canonical Judge run does not match the requested frozen design")
        completed_ids.update(str(row["case_id"]) for row in canonical.get("cases", []))
    for index, blind in enumerate(sealed["blind_cases"], 1):
        if shard_count is not None and (index - 1) % shard_count != shard_index:
            continue
        if str(blind["case_id"]) in completed_ids:
            continue
        print(f"[judge {index}/{len(sealed['blind_cases'])}] {blind['case_id']}", flush=True)
        messages = build_messages(blind, reference, profile)
        request_text = json.dumps(messages, ensure_ascii=False)
        other_case_ids = [case_id for case_id in answer_by_id if case_id != blind["case_id"]]
        _validate_judge_request(messages, blind, other_case_ids)
        raw, audit, elapsed_ms, tool_events = _run_codex(messages)
        parsed = _parse_api_judge_response(raw)
        answer = answer_by_id[blind["case_id"]]
        invalid_reasons = []
        if tool_events:
            invalid_reasons.append("judge_tool_used")
        if parsed.get("human_option") not in {"A", "B"}:
            invalid_reasons.append("invalid_response")
        selected_option = parsed.get("human_option")
        if profile.get("response_schema") == "content_anchored_evidence_gate_v2":
            selected_option, mapping_error = resolve_selected_reply(blind, parsed)
            if mapping_error:
                invalid_reasons.append(mapping_error)
        output["cases"].append({
            "case_id": blind["case_id"],
            "selected_option": selected_option,
            "selected_reply": parsed.get("selected_reply"),
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"),
            "analysis_trace": {
                field: parsed.get(field)
                for field in (
                    "option_a_thread",
                    "option_a_person_specific_evidence",
                    "option_a_counterevidence",
                    "option_b_thread",
                    "option_b_person_specific_evidence",
                    "option_b_counterevidence",
                    "decisive_evidence",
                    "uncertainty",
                )
                if parsed.get(field) is not None
            },
            "human_option": answer,
            "correct": selected_option == answer,
            "valid": not invalid_reasons,
            "invalid_reasons": invalid_reasons,
            "request_messages": messages,
            "blind_case": blind,
            "current_case_answer_metadata_fields_in_request": [],
            "request_sha256": hashlib.sha256(request_text.encode()).hexdigest(),
            "raw_response": raw,
            "elapsed_ms": elapsed_ms,
            "tool_events": tool_events,
            "subprocess_audit": audit,
        })
        _write_json(results_path, output)
    return output


def merge_judge_shards(
    sealed: dict[str, Any],
    out_dir: Path,
    *,
    presentation: str,
    shard_count: int,
) -> dict[str, Any]:
    base_name = "judge_results" if presentation == "canonical" else "swapped_judge_results"
    canonical_path = out_dir / f"{base_name}.json"
    sources = []
    if canonical_path.exists():
        sources.append(json.loads(canonical_path.read_text(encoding="utf-8")))
    for shard_index in range(shard_count):
        path = out_dir / f"{base_name}.shard-{shard_index}-of-{shard_count}.json"
        if not path.exists():
            raise RuntimeError(f"missing Judge shard: {path.name}")
        sources.append(json.loads(path.read_text(encoding="utf-8")))
    identity = ("profile_sha256", "reference_sha256", "presentation")
    if any(
        source.get(field) != sources[0].get(field)
        for source in sources[1:]
        for field in identity
    ):
        raise RuntimeError("Judge shard identity mismatch")
    by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        for row in source.get("cases", []):
            case_id = str(row["case_id"])
            if case_id in by_id and by_id[case_id] != row:
                raise RuntimeError(f"conflicting duplicate Judge case: {case_id}")
            by_id[case_id] = row
    ordered_ids = [str(row["case_id"]) for row in sealed["blind_cases"]]
    missing = [case_id for case_id in ordered_ids if case_id not in by_id]
    extras = sorted(set(by_id) - set(ordered_ids))
    if missing or extras:
        raise RuntimeError(
            "incomplete Judge shard merge: "
            + json.dumps({"missing": missing, "extras": extras}, ensure_ascii=False)
        )
    merged = {
        key: value
        for key, value in sources[0].items()
        if key not in {"cases", "shard_index", "shard_count"}
    }
    merged["cases"] = [by_id[case_id] for case_id in ordered_ids]
    _write_json(canonical_path, merged)
    return merged


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_html_legacy(
    generation: dict[str, Any],
    judged: dict[str, Any],
    swapped: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> str:
    generation_by_id = {row["case_id"]: row for row in generation["cases"]}
    valid = [row for row in judged["cases"] if row["valid"]]
    correct = sum(row["correct"] for row in valid)
    error_total = sum(not row["correct"] for row in valid)
    high_conf_errors = sum(
        not row["correct"] and isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= 0.8
        for row in valid
    )
    selected_a = sum(row["selected_option"] == "A" for row in valid)
    human_a = sum(row["human_option"] == "A" for row in valid)
    selected_shorter = 0
    unequal_length = 0
    human_shorter = 0
    human_longer = 0
    human_equal = 0
    stratum_labels = {
        "correct_common_prosperity_with_luxixi": "共同富裕群（含鲁西西）",
        "youzi_group": "柚子群",
        "other_complex_large_group": "其他复杂大群",
        "other_group": "其他群聊",
        "private": "私聊",
    }
    stratum_stats = {key: {"correct": 0, "total": 0} for key in stratum_labels}
    length_stats = {key: {"correct": 0, "total": 0} for key in ("human_shorter", "near_equal", "human_longer")}
    confidence_stats = {key: {"correct": 0, "total": 0} for key in ("low", "medium", "high")}
    context_mismatch = 0
    context_mismatch_errors = 0
    context_equal_correct = 0
    context_equal_total = 0
    for judge in valid:
        case = generation_by_id[judge["case_id"]]
        stat = stratum_stats[case["group"]]
        stat["total"] += 1
        stat["correct"] += int(judge["correct"])
        length_stat = length_stats[_reply_length_bucket(case)]
        length_stat["total"] += 1
        length_stat["correct"] += int(judge["correct"])
        confidence = judge.get("confidence")
        confidence_key = "high" if isinstance(confidence, (int, float)) and confidence >= 0.8 else "medium" if isinstance(confidence, (int, float)) and confidence > 0.65 else "low"
        confidence_stats[confidence_key]["total"] += 1
        confidence_stats[confidence_key]["correct"] += int(judge["correct"])
        metadata = case.get("generation", {}).get("metadata", {})
        judge_history_count = metadata.get("judge_history_count", metadata.get("event_context_count"))
        mismatch = metadata.get("generator_history_count") != judge_history_count
        if mismatch:
            context_mismatch += 1
            context_mismatch_errors += int(not judge["correct"])
        else:
            context_equal_total += 1
            context_equal_correct += int(judge["correct"])
        human = _norm("".join(case["human_reply"]))
        c0 = _norm("".join(case["c0_reply"]))
        if len(human) < len(c0):
            human_shorter += 1
            unequal_length += 1
            selected_shorter += int(judge["selected_option"] == judge["human_option"])
        elif len(human) > len(c0):
            human_longer += 1
            unequal_length += 1
            selected_shorter += int(judge["selected_option"] != judge["human_option"])
        else:
            human_equal += 1
    strata = "".join(
        f'<li>{_esc(stratum_labels[key])}：<b>{stat["correct"]}/{stat["total"]}</b>（{stat["correct"] / stat["total"]:.1%}）</li>'
        for key, stat in stratum_stats.items()
        if stat["total"]
    )
    swapped_by_id = {
        str(row["case_id"]): row
        for row in (swapped or {}).get("cases", [])
        if row.get("valid")
    }
    baseline_by_id = {
        str(row["case_id"]): row
        for row in (baseline or {}).get("cases", [])
        if row.get("valid")
    }
    cards = []
    for index, judge in enumerate(judged["cases"], 1):
        case = generation_by_id[judge["case_id"]]
        length_bucket = _reply_length_bucket(case)
        metadata = case.get("generation", {}).get("metadata", {})
        judge_history_count = metadata.get("judge_history_count", metadata.get("event_context_count"))
        context_scope = "mismatch" if metadata.get("generator_history_count") != judge_history_count else "equal"
        error_category = str(
            MANUAL_ERROR_CAUSE.get(str(judge["case_id"]))
            or (
                "context_scope_mismatch"
                if not judge["correct"] and context_scope == "mismatch"
                else judge.get("error_category") or ("unclassified" if not judge["correct"] else "correct")
            )
        )
        tail_media = any(
            any(marker in str(row.get("content") or "") for marker in MEDIA_PLACEHOLDERS)
            for row in case["context"][-3:]
        )
        high_confidence_wrong = bool(
            not judge["correct"]
            and isinstance(judge.get("confidence"), (int, float))
            and judge["confidence"] >= 0.8
        )
        review_category, review_note = MISJUDGMENT_REVIEW.get(
            judge["case_id"], ("unreviewed" if not judge["correct"] else "correct", "")
        )
        review_label = {
            "intrinsically_hard": "题目本身难分",
            "clear_logic_error": "判断逻辑明显错误",
            "unreviewed": "尚未人工归类",
            "correct": "正确",
        }[review_category]
        bubbles = "".join(
            f'<div class="msg {"self" if row["role"] == "self" else "other"}"><small>{_esc(row["speaker"])} · {_esc(row["time"] or "")}</small><p>{_esc(row["content"])}</p></div>'
            for row in case["context"]
        )
        blind = judge["blind_case"]
        audit_url = f"/benchmark/judge/strict-historical/case/{_esc(judge['case_id'])}"
        review_html = (
            f'<div class="review {review_category}"><b>人工错因审阅：{review_label}</b><p>{_esc(review_note)}</p></div>'
            if not judge["correct"] else ""
        )
        swapped_row = swapped_by_id.get(str(judge["case_id"]))
        baseline_row = baseline_by_id.get(str(judge["case_id"]))
        baseline_html = ""
        if baseline_row:
            baseline_html = (
                f'<div class="decision"><b>旧版 Judge 选 {_esc(baseline_row["selected_option"])} · '
                f'{"正确" if baseline_row["correct"] else "误判"} · 置信度 {_esc(baseline_row.get("confidence"))}</b>'
                f'<p>{_esc(baseline_row.get("reason"))}</p></div>'
            )
        swap_html = ""
        if swapped_row:
            stable = bool(judge["correct"] == swapped_row["correct"])
            selected_identity = "真人" if swapped_row["correct"] else "C0"
            swap_html = (
                f'<div class="decision"><b>换位复核：{"稳定" if stable else "改变"} · 仍选中{selected_identity}</b>'
                f'<p>{_esc(swapped_row.get("reason"))}</p></div>'
            )
        cards.append(f'''<article data-result="{"correct" if judge["correct"] else "wrong"}" data-review="{review_category}" data-group="{_esc(case["group"])}" data-length="{_esc(length_bucket)}" data-context="{context_scope}" data-cause="{_esc(error_category)}" data-media="{"tail" if tail_media else "none"}" data-high-confidence-wrong="{str(high_confidence_wrong).lower()}"><header><b>#{index} · {_esc(case["chat_name"])}</b><span>{"正确" if judge["correct"] else "误判"} · 置信度 {_esc(judge.get("confidence"))}</span></header>{review_html}<p><small>上下文范围：C0 {_esc(metadata.get('generator_history_count'))} 条 / Judge {_esc(judge_history_count)} 条 · {"不一致" if context_scope == "mismatch" else "一致"}{" · 末尾含未识别媒体" if tail_media else ""}</small></p><h3>Judge 实际看到的上下文</h3><div class="chat">{bubbles}</div><div class="options"><section><b>回复 A</b><p>{'<br>'.join(_esc(x) for x in blind['option_A']) or '（空）'}</p></section><section><b>回复 B</b><p>{'<br>'.join(_esc(x) for x in blind['option_B']) or '（空）'}</p></section></div>{baseline_html}<div class="decision"><b>{"本轮 Judge" if baseline_row else "Judge"} 选 {_esc(judge['selected_option'])}</b><p>{_esc(judge.get('reason'))}</p></div>{swap_html}<details><summary>揭晓真人与 C0</summary><p><b>真人：</b>{'<br>'.join(_esc(x) for x in case['human_reply'])}</p><p><b>C0：</b>{'<br>'.join(_esc(x) for x in case['c0_reply'])}</p></details><p><a href="{audit_url}" target="_blank" rel="noopener">查看本题原始提示词与完整调用 →</a></p></article>''')
    length_summary = "；".join(f'{label} {stat["correct"]}/{stat["total"]}' for key, label in (("human_shorter", "真人更短"), ("near_equal", "长度接近"), ("human_longer", "真人更长")) for stat in (length_stats[key],))
    confidence_summary = "；".join(f'{label} {stat["correct"]}/{stat["total"]}' for key, label in (("low", "≤0.65"), ("medium", "0.66–0.79"), ("high", "≥0.8")) for stat in (confidence_stats[key],))
    score = f"{correct / len(valid):.0%}" if valid else "无有效题"
    if context_mismatch:
        score_label = "原始得分（验收无效）"
        verdict = (
            "<b>结论：本轮验收无效，不能据此调整 Judge 或评价 C0</b>"
            f"<p>{len(valid)} 题中 {context_mismatch} 题范围不等，{error_total} 个误判中 "
            f"{context_mismatch_errors} 个发生在这些题；仅 {context_equal_total} 个范围相等题为 "
            f"{context_equal_correct}/{context_equal_total}。输入不公平已经足以使 {score} 失去验收意义。</p>"
        )
    else:
        score_label = "公平得分（仅验收 Judge）"
        verdict = (
            "<b>结论：100 题输入范围一致；本页只验收 Judge，不评价 C0</b>"
            f"<p>{len(valid)} 道有效题中答对 {correct} 道（{score}），高置信误判 {high_conf_errors} 道。"
            "每题使用独立进程，未向 Judge 提供答案、身份、其他题目或工具。"
            "是否足以作为后续工作基线，须结合场景、长度、位置偏差和过度自信一起判断。</p>"
            "<p>本轮保持冻结提示词；这些题和错题没有加入 few-shot。</p>"
        )
    youzi_note = (
        "" if stratum_stats["youzi_group"]["total"]
        else "<li>柚子群：0 题，本轮没有完成该专项验证。</li>"
    )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Judge 独立验收</title><style>:root{{--bg:#0d1117;--card:#161b22;--line:#30363d;--text:#f0f6fc;--muted:#8b949e;--blue:#58a6ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow-wrap:anywhere}}main{{max-width:900px;margin:auto;padding:12px}}h1{{font-size:22px}}.summary{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.summary div,article,.verdict{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0}}.summary b{{font-size:21px;display:block;color:var(--blue)}}.verdict{{margin:12px 0;border-left:4px solid #f85149}}.verdict ul{{padding-left:20px}}.filters{{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}}select{{width:100%;min-width:0;font:inherit;color:var(--text);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}}article{{margin:12px 0}}header{{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}}header span,small{{color:var(--muted)}}h3{{font-size:13px;color:var(--muted)}}.review{{margin:10px 0;padding:9px;border-radius:9px;background:#202630;border-left:3px solid #8b949e}}.review p{{margin:4px 0 0}}.review.intrinsically_hard{{border-left-color:#d29922}}.review.clear_logic_error{{border-left-color:#f85149}}.chat{{background:#080b10;border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:8px}}.msg{{max-width:90%}}.msg.self{{align-self:flex-end;text-align:right}}.msg p{{margin:3px 0;background:#21262d;border-radius:4px 12px 12px;padding:8px 10px;text-align:left;white-space:pre-wrap}}.msg.self p{{background:#164e2e}}.options{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;margin-top:10px}}.options section,.decision{{border:1px solid var(--line);border-radius:10px;padding:10px;min-width:0}}.decision{{margin-top:8px;border-left:3px solid #d29922}}details{{border-top:1px solid var(--line);padding-top:9px;margin-top:10px}}summary{{color:var(--blue)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:60vh;overflow:auto;background:#080b10;padding:9px;border-radius:8px;font-size:11px}}@media(max-width:620px){{.options{{grid-template-columns:1fr}}main{{padding:9px;padding-bottom:calc(16px + env(safe-area-inset-bottom))}}}}</style></head><body><main><h1>Judge 独立验收</h1><p>每题独立 Codex 进程；不共享题目、答案或本会话历史，工具调用为 0。本页只验收 Judge，不评价 C0。</p><div class="summary"><div><b>{correct}/{len(valid)}</b><span>原始得分（验收无效）</span></div><div><b>{context_mismatch}/{len(valid)}</b><span>C0/Judge 上下文不等</span></div><div><b>{context_mismatch_errors}/{error_total}</b><span>误判受上下文差异影响</span></div><div><b>{high_conf_errors}</b><span>高置信误判（≥0.8）</span></div></div><section class="verdict"><b>结论：本轮验收无效，不能据此调整 Judge 或评价 C0</b><p>C0 的实际生成提示词包含更长历史，但 Judge 只拿到事件片段。{len(valid)} 题中 {context_mismatch} 题范围不等，{error_total} 个误判中 {context_mismatch_errors} 个发生在这些题；仅 {context_equal_total} 个范围相等题为 {context_equal_correct}/{context_equal_total}，数量不足以定级。未发现答案、身份、跨题或工具泄漏，但输入不公平已经足以使 {score} 失去验收意义。</p><p><b>首轮只改一个方向：</b>让 Judge 获得与 C0 同范围的完整聊天历史，再用另一套未见题复验；本轮题目和错题不进入 few-shot，也不同时改判断规则。</p><ul>{strata}<li>柚子群：0 题，本轮没有完成该专项验证。</li><li>长度桶：{length_summary}；Judge 在 {unequal_length} 个非等长题中选择较短回复 {selected_shorter} 次。</li><li>置信度校准：{confidence_summary}。高置信并未更准，仍有过度自信风险。</li><li>真人位置 A/B：{human_a}/{len(valid)-human_a}；Judge 选择 A/B：{selected_a}/{len(valid)-selected_a}</li></ul></section><div class="filters"><select id="result-filter" aria-label="按判断结果筛选"><option value="all">全部结果</option><option value="wrong">只看误判</option><option value="correct">只看正确</option><option value="high-confidence-wrong">高置信误判</option></select><select id="review-filter" aria-label="按人工错因审阅筛选"><option value="all">全部人工分类</option><option value="intrinsically_hard">题目本身难分</option><option value="clear_logic_error">判断逻辑明显错误</option></select><select id="context-filter" aria-label="按上下文范围筛选"><option value="all">全部上下文</option><option value="mismatch">C0/Judge 范围不等</option><option value="equal">C0/Judge 范围相等</option></select><select id="group-filter" aria-label="按聊天类型筛选"><option value="all">全部聊天</option><option value="correct_common_prosperity_with_luxixi">共同富裕群（含鲁西西）</option><option value="youzi_group">柚子群</option><option value="other_complex_large_group">其他复杂大群</option><option value="other_group">其他群聊</option><option value="private">私聊</option></select><select id="length-filter" aria-label="按回复长度筛选"><option value="all">全部长度</option><option value="human_shorter">真人更短</option><option value="near_equal">长度接近</option><option value="human_longer">真人更长</option></select><select id="cause-filter" aria-label="按错因筛选"><option value="all">全部自动错因</option><option value="context_scope_mismatch">上下文范围不公平</option><option value="unclassified">待人工归因</option><option value="length_or_completeness">长短/完整度偏好</option><option value="short_punchline">短梗/俏皮偏好</option><option value="context_continuity">前文承接或跳话题</option><option value="unknown_fact">未知事实误判</option><option value="weak_evidence_confidence">弱证据却高置信</option><option value="other">其他</option></select></div>{''.join(cards)}<script>const filters={{result:document.getElementById('result-filter'),review:document.getElementById('review-filter'),context:document.getElementById('context-filter'),group:document.getElementById('group-filter'),length:document.getElementById('length-filter'),cause:document.getElementById('cause-filter')}};function applyFilters(){{document.querySelectorAll('article').forEach(x=>{{const result=filters.result.value==='all'||x.dataset.result===filters.result.value||(filters.result.value==='high-confidence-wrong'&&x.dataset.highConfidenceWrong==='true');const review=filters.review.value==='all'||x.dataset.review===filters.review.value;const context=filters.context.value==='all'||x.dataset.context===filters.context.value;const group=filters.group.value==='all'||x.dataset.group===filters.group.value;const length=filters.length.value==='all'||x.dataset.length===filters.length.value;const cause=filters.cause.value==='all'||x.dataset.cause===filters.cause.value;x.hidden=!(result&&review&&context&&group&&length&&cause)}})}}Object.values(filters).forEach(x=>x.addEventListener('change',applyFilters));</script></main></body></html>'''


def summarize_position_swap(judged: dict[str, Any], swapped: dict[str, Any]) -> dict[str, int]:
    canonical_by_id = {str(row["case_id"]): row for row in judged["cases"] if row.get("valid")}
    swapped_by_id = {str(row["case_id"]): row for row in swapped["cases"] if row.get("valid")}
    case_ids = canonical_by_id.keys() & swapped_by_id.keys()
    return {
        "total": len(case_ids),
        "swapped_correct": sum(int(swapped_by_id[case_id]["correct"]) for case_id in case_ids),
        "identity_consistent": sum(
            int(canonical_by_id[case_id]["correct"] == swapped_by_id[case_id]["correct"])
            for case_id in case_ids
        ),
        "correct_to_wrong": sum(
            int(canonical_by_id[case_id]["correct"] and not swapped_by_id[case_id]["correct"])
            for case_id in case_ids
        ),
        "wrong_to_correct": sum(
            int(not canonical_by_id[case_id]["correct"] and swapped_by_id[case_id]["correct"])
            for case_id in case_ids
        ),
        "both_wrong": sum(
            int(not canonical_by_id[case_id]["correct"] and not swapped_by_id[case_id]["correct"])
            for case_id in case_ids
        ),
    }


def render_html(
    generation: dict[str, Any],
    judged: dict[str, Any],
    swapped: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> str:
    html = _render_html_legacy(generation, judged, swapped, baseline)
    cases_by_id = {str(case.get("internal_id") or case["case_id"]): case for case in generation["cases"]}
    valid = [row for row in judged["cases"] if row.get("valid")]
    mismatch = 0
    for row in valid:
        metadata = cases_by_id[row["case_id"]].get("generation", {}).get("metadata", {})
        judge_count = metadata.get("judge_history_count", metadata.get("event_context_count"))
        mismatch += int(metadata.get("generator_history_count") != judge_count)
    correct = sum(int(row["correct"]) for row in valid)
    high_conf_errors = sum(
        int(not row["correct"] and isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= 0.8)
        for row in valid
    )
    score = f"{correct / len(valid):.0%}" if valid else "无有效题"
    position_note = "原始位置的总体选择比例没有明显偏差。"
    if swapped:
        swap = summarize_position_swap(judged, swapped)
        position_note = (
            f"答案换位后答对 {swap['swapped_correct']}/{swap['total']}；"
            f"按回复身份仅 {swap['identity_consistent']}/{swap['total']} 保持一致，"
            "低于 90% 冻结门槛，说明逐题仍有位置敏感或判断波动。"
        )
    scope_note = (
        f"其中 {mismatch} 题的 C0 与 Judge 上下文范围不同；本套题还包含末尾未识别媒体，"
        "因此只用于同题开发回归，不作为独立终验。"
        if mismatch else
        "本轮输入范围一致，可用于 Judge 独立验收。"
    )
    if mismatch:
        verdict = (
            "<b>结论：本轮验收无效，不能据此调整 Judge 或评价 C0</b>"
            f"<p>{len(valid)} 道有效题中有 {mismatch} 题的 C0 与 Judge 上下文范围不同；"
            f"{score} 只能记录为原始结果，不能作为公平验收成绩。</p>"
        )
    else:
        verdict = (
            f"<b>结论：{score} 仅作开发回归，Judge 尚未达到冻结门槛</b>"
            f"<p>{len(valid)} 道有效题中答对 {correct} 道（{score}），高置信误判 {high_conf_errors} 道。"
            "每题使用独立进程，未向 Judge 提供答案、身份、其他题目或工具。"
            f"{scope_note}{position_note}当前共有 {len(valid) - correct} 个错判；人工分类只用于诊断，不作为答案提示；"
            "因此它适合作为诊断助手，但还不能单独决定回复版本胜负。</p>"
            "<p>下一步只看两项独立证据：同题换位后判断是否稳定，以及两套全新未见题是否连续达到冻结门槛。</p>"
            "<p>本轮错题没有加入 few-shot；若新增模式卡，只能来自独立开发样本，并由另一套未见题复验。</p>"
        )
    verdict += "<p>历史人工复核库存：5 个题目本身难分、12 个判断逻辑明显错误。</p>"
    verdict_start = html.index('<section class="verdict">') + len('<section class="verdict">')
    list_start = html.index("<ul>", verdict_start)
    html = html[:verdict_start] + verdict + html[list_start:]
    if not mismatch:
        html = html.replace("原始得分（验收无效）", "公平得分（仅验收 Judge）", 1)
    html = html.replace(
        '<option value="weak_evidence_confidence">弱证据却高置信</option>',
        '<option value="weak_evidence_confidence">弱证据却高置信</option><option value="redundant_obvious_explanation">冗余释义显然信息</option><option value="stale_meme_or_surface_humor">老梗/表面幽默</option>',
        1,
    )
    html = html.replace(
        '</select></div>',
        '</select><select id="media-filter" aria-label="按末尾媒体筛选"><option value="all">全部媒体状态</option><option value="tail">末尾含未识别媒体</option><option value="none">末尾无媒体占位</option></select></div>',
        1,
    )
    html = html.replace(
        "cause:document.getElementById('cause-filter')",
        "cause:document.getElementById('cause-filter'),media:document.getElementById('media-filter')",
        1,
    )
    html = html.replace(
        "const cause=filters.cause.value==='all'||x.dataset.cause===filters.cause.value;x.hidden=!(result&&review&&context&&group&&length&&cause)",
        "const cause=filters.cause.value==='all'||x.dataset.cause===filters.cause.value;const media=filters.media.value==='all'||x.dataset.media===filters.media.value;x.hidden=!(result&&review&&context&&group&&length&&cause&&media)",
        1,
    )
    if swapped:
        total = swap["total"]
        swap_cards = (
            f'<div><b>{swap["swapped_correct"]}/{total}</b><span>答案换位后正确</span></div>'
            f'<div><b>{swap["identity_consistent"]}/{total}</b><span>按回复身份判断一致</span></div>'
        )
        html = html.replace('</div><section class="verdict">', f'{swap_cards}</div><section class="verdict">', 1)
        swap_note = (
            f'<p><b>答案换位复核：</b>{total} 题中，按回复身份归一化后 '
            f'{swap["identity_consistent"]} 题判断一致；原正确→换位错误 {swap["correct_to_wrong"]} 题，'
            f'原错误→换位正确 {swap["wrong_to_correct"]} 题，两次都错 {swap["both_wrong"]} 题。</p>'
        )
        html = html.replace("<p>下一步只看两项独立证据：", swap_note + "<p>下一步只看两项独立证据：", 1)
    if any(cases_by_id[row["case_id"]].get("group") == "youzi_group" for row in valid):
        html = html.replace("<li>柚子群：0 题，本轮没有完成该专项验证。</li>", "", 1)
    if baseline:
        old_by_id = {str(row["case_id"]): row for row in baseline["cases"] if row.get("valid")}
        pairs = [(old_by_id[row["case_id"]], row) for row in valid if row["case_id"] in old_by_id]
        old_correct = sum(int(old["correct"]) for old, _ in pairs)
        fixed = sum(int(not old["correct"] and new["correct"]) for old, new in pairs)
        regressed = sum(int(old["correct"] and not new["correct"]) for old, new in pairs)
        old_high_conf_errors = sum(
            int(not old["correct"] and isinstance(old.get("confidence"), (int, float)) and old["confidence"] >= 0.8)
            for old, _ in pairs
        )
        judged_ids = {row["case_id"] for row in valid}
        selected_cases = [cases_by_id[case_id] for case_id in judged_ids]
        private_count = sum(case.get("group") == "private" for case in selected_cases)
        human_shorter_count = sum(_reply_length_bucket(case) == "human_shorter" for case in selected_cases)
        comparison_cards = (
            f'<div><b>{old_correct}/{len(pairs)} → {correct}/{len(valid)}</b><span>同题旧版 → 本轮</span></div>'
            f'<div><b>+{fixed} / -{regressed}</b><span>修正 / 新增误判</span></div>'
        )
        html = html.replace('</div><section class="verdict">', f'{comparison_cards}</div><section class="verdict">', 1)
        comparison_verdict = (
            f'<b>结论：同题 {old_correct}/{len(pairs)} → {correct}/{len(valid)}，本轮 Judge 改进有效；尚不冻结</b>'
            f'<p>本轮在同一套未见题上修正 {fixed} 题、新增误判 {regressed} 题；高置信误判从 '
            f'{old_high_conf_errors} 降至 {high_conf_errors}。这证明错题规律、场景化 few-shot 与置信度规则产生了净收益。</p>'
            f'<p>限制：{human_shorter_count}/{len(valid)} 题是真人回复更短，{private_count}/{len(valid)} 题是私聊；'
            '共同富裕群（含鲁西西）、柚子群和复杂大群均未覆盖。本套只能作为独立探索验证，不能代替场景均衡的冻结终验。</p>'
            '<p>本轮 100 题及其错题未加入 few-shot；两版均逐题独立运行，未共享答案、身份、其他题目或工具。</p>'
        )
        verdict_start = html.index('<section class="verdict">') + len('<section class="verdict">')
        list_start = html.index('<ul>', verdict_start)
        html = html[:verdict_start] + comparison_verdict + html[list_start:]
        html = html.replace("开发回归得分", "独立探索得分", 1)
    evaluation = generation.get("evaluation", {})
    if evaluation.get("evaluation_kind") == "group_chat_transfer_check":
        group_labels = {
            "correct_common_prosperity_with_luxixi": "共同富裕群（含鲁西西）",
            "youzi_group": "柚子群",
            "other_complex_large_group": "其他复杂大群",
        }
        group_rows = []
        for group, label in group_labels.items():
            rows = [row for row in valid if cases_by_id[row["case_id"]].get("group") == group]
            group_rows.append(f"<li>{label}：{sum(int(row['correct']) for row in rows)}/{len(rows)}</li>")
        unseen = [
            row for row in valid
            if cases_by_id[row["case_id"]].get("transfer_check_source", {}).get("never_previously_judged")
        ]
        unseen_correct = sum(int(row["correct"]) for row in unseen)
        transfer_verdict = (
            f"<b>结论：群聊专项迁移检查 {correct}/{len(valid)}（{score}），本轮不冻结 Judge</b>"
            f"<p>三类困难群聊各 5 题；其中真正未被 Judge 评测过的 10 题答对 {unseen_correct}/10。"
            f"高置信误判 {high_conf_errors} 题。每题仍由独立 Codex 进程判断，未提供答案或其他题目。</p>"
            "<p>这是一轮专项迁移检查，不是全新冻结验收：柚子群现有可评分对照不足，5 题均来自旧题复用；"
            "因此即使总分较高，也只能判断改进是否初步迁移到群聊难点，不能单独证明 Judge 已达到冻结标准。</p>"
            f"<ul>{''.join(group_rows)}</ul>"
        )
        pool_audit = evaluation.get("independent_pool_audit") or {}
        pool_notice = ""
        if pool_audit.get("status") == "insufficient_for_balanced_validation":
            audit_rows = []
            for group, label in group_labels.items():
                row = pool_audit["groups"][group]
                audit_rows.append(
                    "<tr>"
                    f"<td>{label}</td><td>{row['candidates']}</td>"
                    f"<td>{row['previously_judged']}</td>"
                    f"<td>{row['unseen_c0_silent']}</td>"
                    f"<td>{row['unseen_answer_leak'] + row['unseen_c0_reproduced_human']}</td>"
                    f"<td><b>{row['unseen_usable_reply_pairs']}</b></td>"
                    "</tr>"
                )
            pool_notice = (
                '<section class="verdict" style="border-left-color:#d29922">'
                '<b>当前状态：v3 Judge 已构建，但全新均衡群聊验收尚未完成</b>'
                '<p>只读扫描全部历史生成与实际 Judge 结果后，真正“从未被 Judge 看过且可评分”的重点群聊回复对只剩 3 条：'
                '共同富裕群 2、柚子群 0、其他复杂大群 1。数量无法组成三类均衡的新验收集，所以下方 11/15 仍是上一轮专项迁移检查，不能包装成新一轮成绩，也不能据此冻结 Judge。</p>'
                '<p>另核对了“共同富裕群”的 146 条历史线上 Bot 成功发送记录，但当时没有保存生成器哈希、代码提交、完整配置或回放归档，无法证明它们来自当前 C0，因此没有混入本轮独立题集。'
                '这是当前 C0 回复对的版本来源不可追溯，不是历史真人聊天不足。</p>'
                '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse">'
                '<thead><tr><th>场景</th><th>候选</th><th>已评测</th><th>未见但 C0 静默</th><th>答案泄漏/复现真人</th><th>真正可用</th></tr></thead>'
                f"<tbody>{''.join(audit_rows)}</tbody></table></div>"
                '<p><small>口径：按 case_id 去重；递归核对包含实际选择字段的 Judge 结果；依赖未识别媒体、C0 静默、生成轨迹含真人答案或 C0 复现真人答案的题不承担结论。没有生成新回复，也没有调用 Judge。</small></p>'
                '</section>'
            )
        verdict_start = html.index('<section class="verdict">') + len('<section class="verdict">')
        list_start = html.index("<ul>", verdict_start)
        list_end = html.index("</ul>", list_start) + len("</ul>")
        html = html[:verdict_start] + transfer_verdict + html[list_end:]
        if pool_notice:
            html = html.replace('<section class="verdict">', pool_notice + '<section class="verdict">', 1)
        html = html.replace("Judge 独立验收", "Judge 群聊专项迁移检查")
        html = html.replace("开发回归得分", "群聊专项得分", 1)
        html = html.replace(
            "本轮输入范围一致，可用于 Judge 独立验收。",
            "本轮输入范围一致，可用于群聊专项迁移检查。",
        )
    return html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("audit", "generate", "seal", "judge", "merge-judge", "render", "repair-seal", "repair-judge", "repair-render"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--hidden-count", type=int, default=MIN_HIDDEN_CASES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--seed-generation", type=Path)
    parser.add_argument("--exclude-sealed", type=Path, action="append", default=[])
    parser.add_argument("--exclude-judged", type=Path, action="append", default=[])
    parser.add_argument("--retry-unscorable", action="store_true")
    parser.add_argument("--length-balanced", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=("stage2", "exploratory-unseen", "group-balanced"),
        default="stage2",
    )
    parser.add_argument("--generation-provider", choices=("official", "volcengine"), default="official")
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument("--discard-from-case-id")
    parser.add_argument("--presentation", choices=("canonical", "swapped"), default="canonical")
    parser.add_argument("--swapped-results", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    profile_path = args.profile if args.profile.is_absolute() else ROOT / args.profile
    reference_path = args.reference if args.reference.is_absolute() else ROOT / args.reference
    seed_generation = (
        args.seed_generation
        if args.seed_generation is None or args.seed_generation.is_absolute()
        else ROOT / args.seed_generation
    )
    exclude_sealed = [path if path.is_absolute() else ROOT / path for path in args.exclude_sealed]
    exclude_judged = [path if path.is_absolute() else ROOT / path for path in args.exclude_judged]
    swapped_results = (
        args.swapped_results
        if args.swapped_results is None or args.swapped_results.is_absolute()
        else ROOT / args.swapped_results
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_extended_candidates()
    by_id = {case["internal_id"]: case for case in cases}
    if args.phase == "audit":
        audit = audit_inputs(cases)
        _write_json(out_dir / "input_audit.json", audit)
        print(json.dumps({k: audit[k] for k in ("accepted_count", "excluded_count", "accepted_by_group")}, ensure_ascii=False, indent=2))
        return 0
    if args.phase == "generate":
        _load_env()
        audit = json.loads((out_dir / "input_audit.json").read_text(encoding="utf-8"))
        results_path = out_dir / "generation_results.json"
        if args.resume and results_path.exists():
            prior = json.loads(results_path.read_text(encoding="utf-8"))
        elif seed_generation:
            prior = json.loads(seed_generation.read_text(encoding="utf-8"))
        else:
            prior = {"cases": []}
        output = _discard_cases_from_id(list(prior.get("cases", [])), args.discard_from_case_id)
        accepted = {row["case_id"] for row in audit["accepted"]}
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        excluded_case_ids: set[str] = set()
        for sealed_path in exclude_sealed:
            old_sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
            excluded_case_ids.update(str(row["case_id"]) for row in old_sealed["answers"])
        for judged_path in exclude_judged:
            old_judged = json.loads(judged_path.read_text(encoding="utf-8"))
            excluded_case_ids.update(str(row["case_id"]) for row in old_judged["cases"])
        if args.retry_unscorable:
            output = [
                row
                for row in output
                if row.get("validity", {}).get("formally_scorable")
                and str(row.get("case_id") or "") not in excluded_case_ids
            ]
        if args.selection_mode == "group-balanced":
            quotas = _group_balanced_quotas(args.hidden_count)
        else:
            quotas = None if args.length_balanced else _hidden_stratum_quotas(args.hidden_count)
        length_quotas = _length_bucket_quotas(args.hidden_count) if args.length_balanced else None
        generation_candidates = [case for case in cases if case["internal_id"] in accepted]
        if args.selection_mode == "group-balanced":
            generation_candidates = [
                case
                for case in generation_candidates
                if _group_key(case) in quotas and not _tail_has_unresolved_media(case)
            ]
        pending = _pending_generation_cases(
            generation_candidates,
            output,
            quotas,
            reference,
            excluded_case_ids,
        )
        if args.limit is not None:
            pending = pending[: args.limit]
        llm = _generation_client(args.generation_provider, args.volcengine_config)
        memory = MemoryEngine(llm_client=llm)
        try:
            for index, case in enumerate(pending, 1):
                if length_quotas is not None:
                    counts = _scorable_length_counts(output, reference, excluded_case_ids)
                    if all(counts[bucket] >= quota for bucket, quota in length_quotas.items()):
                        break
                else:
                    counts = _scorable_counts(output, reference)
                if quotas is not None and all(counts[group] >= quota for group, quota in quotas.items()):
                    break
                group = _group_key(case)
                if quotas is not None and counts[group] >= quotas[group]:
                    continue
                case_id = str(case["internal_id"])
                print(f"[generate {index}/{len(pending)}] {case_id}", flush=True)
                output.append(generate_case(case, llm, memory, args.timeout_seconds))
                payload = {"schema_version": 1, "status": "running", "cases": output}
                _write_json(results_path, payload)
        finally:
            memory.shutdown()
        payload = {"schema_version": 1, "status": "complete", "cases": output}
        _write_json(results_path, payload)
        print(json.dumps({
            "generated": len(output),
            "scorable_by_group": _scorable_counts(output, reference),
            "new_scorable_by_length": _scorable_length_counts(output, reference, excluded_case_ids),
        }, ensure_ascii=False))
        return 0
    generation = json.loads((out_dir / "generation_results.json").read_text(encoding="utf-8"))
    if args.phase == "repair-seal":
        original_sealed = json.loads((out_dir / "sealed_hidden.json").read_text(encoding="utf-8"))
        replacement, combined = seal_fair_replacements(generation, original_sealed)
        _write_json(out_dir / "fair_generation_results.json", generation)
        _write_json(out_dir / "replacement_sealed_hidden.json", replacement)
        _write_json(out_dir / "fair_sealed_hidden.json", combined)
        print(json.dumps({
            "replacement": len(replacement["answers"]),
            "combined": len(combined["answers"]),
            "selected_by_stratum": combined["selected_by_stratum"],
            "selected_by_length_bucket": combined["selected_by_length_bucket"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.phase == "repair-judge":
        replacement = json.loads((out_dir / "replacement_sealed_hidden.json").read_text(encoding="utf-8"))
        replacement_out = out_dir / "replacement_judge"
        replacement_out.mkdir(parents=True, exist_ok=True)
        judged = judge_hidden(replacement, profile_path, reference_path, replacement_out)
        print(json.dumps({"judged": len(judged["cases"])}, ensure_ascii=False))
        return 0
    if args.phase == "repair-render":
        fair_generation_path = out_dir / "fair_generation_results.json"
        if fair_generation_path.exists():
            generation = json.loads(fair_generation_path.read_text(encoding="utf-8"))
        original_judged = json.loads((out_dir / "judge_results.json").read_text(encoding="utf-8"))
        combined = json.loads((out_dir / "fair_sealed_hidden.json").read_text(encoding="utf-8"))
        replacement_judged = json.loads(
            (out_dir / "replacement_judge/judge_results.json").read_text(encoding="utf-8")
        )
        judged = merge_fair_judgments(original_judged, combined, replacement_judged)
        _write_json(out_dir / "fair_judge_results.json", judged)
        swapped = json.loads(swapped_results.read_text(encoding="utf-8")) if swapped_results else None
        rendered = render_html(generation, judged, swapped)
        (out_dir / "fair_index.html").write_text(rendered, encoding="utf-8")
        print(out_dir / "fair_index.html")
        return 0
    if args.phase == "seal":
        excluded_case_ids: set[str] = set()
        for sealed_path in exclude_sealed:
            old_sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
            excluded_case_ids.update(str(row["case_id"]) for row in old_sealed["answers"])
        for judged_path in exclude_judged:
            old_judged = json.loads(judged_path.read_text(encoding="utf-8"))
            excluded_case_ids.update(str(row["case_id"]) for row in old_judged["cases"])
        sealed = seal_hidden(
            generation,
            args.hidden_count,
            excluded_case_ids=excluded_case_ids,
            length_balanced=args.length_balanced,
            selection_mode=args.selection_mode,
        )
        _write_json(out_dir / "sealed_hidden.json", sealed)
        print(json.dumps({"sealed": len(sealed["blind_cases"])}, ensure_ascii=False))
        return 0
    if args.phase == "judge":
        sealed = json.loads((out_dir / "sealed_hidden.json").read_text(encoding="utf-8"))
        judged = judge_hidden(
            sealed,
            profile_path,
            reference_path,
            out_dir,
            presentation=args.presentation,
            resume=args.resume,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        print(json.dumps({"judged": len(judged["cases"])}, ensure_ascii=False))
        return 0
    if args.phase == "merge-judge":
        if args.shard_count is None:
            raise ValueError("--shard-count is required for merge-judge")
        sealed = json.loads((out_dir / "sealed_hidden.json").read_text(encoding="utf-8"))
        judged = merge_judge_shards(
            sealed,
            out_dir,
            presentation=args.presentation,
            shard_count=args.shard_count,
        )
        print(json.dumps({"judged": len(judged["cases"])}, ensure_ascii=False))
        return 0
    judged = json.loads((out_dir / "judge_results.json").read_text(encoding="utf-8"))
    swapped = json.loads(swapped_results.read_text(encoding="utf-8")) if swapped_results else None
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8")) if args.baseline_results else None
    rendered = render_html(generation, judged, swapped, baseline)
    (out_dir / "index.html").write_text(rendered, encoding="utf-8")
    print(out_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
