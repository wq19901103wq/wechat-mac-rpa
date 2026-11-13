"""Offline, privacy-preserving review baseline for production replies.

The private packet contains the text needed for human review.  The public
report contains aggregates only.  No model or network service is called.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Iterable


SCHEMA_VERSION = 2
GRADES = ("good", "acceptable", "bad", "critical")

# These expressions create review hints, never automatic safety verdicts.
PRIVATE_FACT_HINTS = {
    "relationship_fact": re.compile(r"结婚|领证|老婆|老公|对象|分手|离婚"),
    "financial_fact": re.compile(r"余额|月供|持仓|资产|存款|欠款|负债|\d+[万亿]"),
}
ROUGH_LANGUAGE_HINT = re.compile(
    r"你全家|他全家|她全家|裸照|裸体|阉割|强奸|弄死|杀了|妈的|傻逼|滚蛋|蠢货|废物"
)
CORRECTION_HINT = re.compile(r"不是|不对|我说的是|你记错|谁说|别瞎说|哪有|搞错|说错")
DISLIKE_HINT = re.compile(r"别说了|闭嘴|烦死|恶心|尴尬|无语|别回复|别回")
POSITIVE_HINT = re.compile(r"哈{2,}|笑死|哈哈|牛|可以|确实|没错|对的|厉害|绝了|爱了")
INFORMATION_VALUE_HINT = re.compile(r"因为|所以|建议|可以先|试试|需要|记得|如果|其实|意思是|时间|地址|\d")
EMOTIONAL_VALUE_HINT = re.compile(r"哈哈|笑死|别急|没事|恭喜|牛|绝了|离谱|心疼|抱抱|加油|确实|服了")
EMOJI_HINT = re.compile(r"[\U0001F300-\U0001FAFF]|[😂🤣😭😅🥲🙂🙃😄😊😡🤔👍🙏❤️❤]")
COLLOQUIAL_HINT = re.compile(r"哈哈|笑死|卧槽|我靠|牛|绝了|离谱|确实|行吧|呗|嘛|啊|呀|呢")


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    timestamp: float | None
    content_type: str = "text"


@dataclass(frozen=True)
class ReplyTick:
    db_id: int
    tick_id: int
    created_at: str
    chat_key: str
    is_group: bool
    replies: tuple[str, ...]
    input_messages: tuple[Message, ...]
    target_messages: tuple[Message, ...]
    human_is_badcase: int | None
    human_badcase_type: str | None
    human_notes: str | None
    judge_score: float | None
    judge_is_badcase: int | None
    judge_reason: str | None


def _json_list(raw: str | None) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _parse_replies(raw: str | None) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in _json_list(raw) if str(item).strip())


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _parse_messages(raw: str | None) -> tuple[Message, ...]:
    messages: list[Message] = []
    for item in _json_list(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("image_description") or "").strip()
        if not text:
            continue
        sender_type = str(item.get("sender_type") or "other").lower()
        role = "bot" if sender_type == "self" else "human"
        content_type = str(item.get("message_type") or item.get("type") or "text")
        timestamp = _number(item.get("reply_time")) or _number(item.get("create_time"))
        if timestamp is None:
            timestamp = _number(item.get("timestamp"))
        messages.append(Message(role=role, text=text, timestamp=timestamp, content_type=content_type))
    return tuple(messages)


def _created_timestamp(created_at: str) -> float | None:
    try:
        return datetime.fromisoformat(created_at).timestamp()
    except (TypeError, ValueError):
        return None


def load_sent_ticks(db_path: Path) -> tuple[list[ReplyTick], int]:
    """Load every successfully sent tick from SQLite in read-only mode."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        total_ticks = connection.execute("SELECT COUNT(*) FROM tick_log").fetchone()[0]
        rows = connection.execute(
            """
            SELECT id, tick_id, created_at, chat_name, is_group,
                   replies_sent_json, session_input_messages_json,
                   session_output_unreplied_json,
                   human_is_badcase, human_badcase_type, human_notes,
                   judge_score, judge_is_badcase, judge_reason
            FROM tick_log
            WHERE send_success = 1
            ORDER BY created_at, id
            """
        ).fetchall()
    finally:
        connection.close()
    ticks = []
    for row in rows:
        ticks.append(
            ReplyTick(
                db_id=row["id"],
                tick_id=row["tick_id"],
                created_at=row["created_at"] or "",
                chat_key=row["chat_name"] or "",
                is_group=bool(row["is_group"]),
                replies=_parse_replies(row["replies_sent_json"]),
                input_messages=_parse_messages(row["session_input_messages_json"]),
                target_messages=_parse_messages(row["session_output_unreplied_json"]),
                human_is_badcase=row["human_is_badcase"],
                human_badcase_type=row["human_badcase_type"],
                human_notes=row["human_notes"],
                judge_score=row["judge_score"],
                judge_is_badcase=row["judge_is_badcase"],
                judge_reason=row["judge_reason"],
            )
        )
    return ticks, total_ticks


def _normalize(text: str) -> str:
    return "".join(re.findall(r"[\w\u4e00-\u9fff]", text.lower()))


def _fingerprint(ticks: list[ReplyTick]) -> str:
    payload = [[t.db_id, t.tick_id, t.created_at, t.is_group, t.replies] for t in ticks]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _unique_messages(ticks: Iterable[ReplyTick]) -> dict[str, list[Message]]:
    by_chat: dict[str, list[Message]] = defaultdict(list)
    seen: dict[str, set[tuple[str, str, float | None, str]]] = defaultdict(set)
    for tick in ticks:
        for message in tick.input_messages:
            key = (message.role, message.text, message.timestamp, message.content_type)
            if key not in seen[tick.chat_key]:
                seen[tick.chat_key].add(key)
                by_chat[tick.chat_key].append(message)
    for messages in by_chat.values():
        messages.sort(key=lambda m: (m.timestamp is None, m.timestamp or 0))
    return by_chat


def _text_style_profile(texts: list[str]) -> dict[str, Any]:
    lengths = [len(text) for text in texts]
    count = len(texts)
    seen: set[str] = set()
    repeated = 0
    eligible = 0
    for text in texts:
        normalized = _normalize(text)
        if len(normalized) < 8:
            continue
        eligible += 1
        if normalized in seen:
            repeated += 1
        seen.add(normalized)
    return {
        "messages": count,
        "median_chars": median(lengths) if lengths else None,
        "p90_chars": round(_percentile(lengths, 0.9), 2) if lengths else None,
        "short_message_rate": round(sum(len(t) <= 8 for t in texts) / count, 4) if count else None,
        "question_rate": round(sum(bool(re.search(r"[？?]", t)) for t in texts) / count, 4) if count else None,
        "emoji_rate": round(sum(bool(EMOJI_HINT.search(t)) for t in texts) / count, 4) if count else None,
        "colloquial_rate": round(sum(bool(COLLOQUIAL_HINT.search(t)) for t in texts) / count, 4) if count else None,
        "rough_language_rate": round(sum(bool(ROUGH_LANGUAGE_HINT.search(t)) for t in texts) / count, 4) if count else None,
        "exact_repetition_rate": round(repeated / eligible, 4) if eligible else None,
    }


def _style_profile(messages: Iterable[Message]) -> dict[str, Any]:
    texts = [
        message.text for message in messages
        if message.role == "human" and message.content_type == "text"
    ]
    profile = _text_style_profile(texts)
    profile["human_messages"] = profile.pop("messages")
    profile.pop("exact_repetition_rate")
    profile["content_repetition_rate"] = None
    profile["content_repetition_status"] = "unavailable_due_to_repeated_snapshots"
    return profile


def _bot_style_profile(ticks: Iterable[ReplyTick]) -> dict[str, Any]:
    profile = _text_style_profile([reply for tick in ticks for reply in tick.replies])
    profile["bot_messages"] = profile.pop("messages")
    return profile


def build_human_reference(ticks: list[ReplyTick]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build style references from historical human messages, not Bot replies."""
    timelines = _unique_messages(ticks)
    group_messages: list[Message] = []
    private_messages: list[Message] = []
    chat_is_group = {tick.chat_key: tick.is_group for tick in ticks}
    chat_profiles = {}
    for chat, messages in timelines.items():
        chat_profiles[chat] = _style_profile(messages)
        (group_messages if chat_is_group.get(chat) else private_messages).extend(messages)
    public = {
        "method": "deduplicated historical human messages, split by group/private chat",
        "group": _style_profile(group_messages),
        "private": _style_profile(private_messages),
        "interpretation": (
            "Profanity, teasing and aggression can be normal style features when they fit the relationship "
            "and context; lexical matches are not defects by themselves. Stale jokes and cross-turn "
            "repetition remain quality problems. Human repetition rate is unavailable because the source "
            "contains repeated snapshots of the same messages."
        ),
    }
    return public, chat_profiles


def _sent_timestamp(tick: ReplyTick, timeline: list[Message]) -> float | None:
    created = _created_timestamp(tick.created_at)
    candidates = []
    reply_set = set(tick.replies)
    for message in timeline:
        if message.role == "bot" and message.text in reply_set and message.timestamp is not None:
            distance = abs(message.timestamp - created) if created is not None else 0
            if created is None or distance <= 3600:
                candidates.append((distance, message.timestamp))
    return min(candidates)[1] if candidates else created


def _private_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "text": message.text,
        "timestamp": message.timestamp,
        "content_type": message.content_type,
    }


def _reaction_window(tick: ReplyTick, timeline: list[Message], sent_at: float | None) -> list[Message]:
    if sent_at is None:
        return []
    later_ticks = [
        _created_timestamp(other.created_at)
        for other in _CURRENT_TICKS
        if other.chat_key == tick.chat_key and other.db_id != tick.db_id
    ]
    next_bot = min((value for value in later_ticks if value and value > sent_at), default=sent_at + 86400)
    end = min(next_bot, sent_at + 86400)
    return [
        message for message in timeline
        if message.role == "human" and message.timestamp is not None and sent_at < message.timestamp <= end
    ][:6]


def _reaction_evidence(messages: list[Message]) -> dict[str, Any]:
    joined = "\n".join(message.text for message in messages)
    if CORRECTION_HINT.search(joined):
        kind = "correction"
    elif DISLIKE_HINT.search(joined):
        kind = "dislike"
    elif POSITIVE_HINT.search(joined):
        kind = "positive"
    elif messages:
        kind = "continued"
    else:
        kind = "none"
    return {
        "type": kind,
        "evidence_message_indexes": list(range(len(messages))) if kind != "none" else [],
        "interpretation": "Historical reaction evidence, not a human quality label.",
    }


def _provisional_grade(
    tick: ReplyTick,
    reaction: dict[str, Any],
    style: dict[str, Any],
    repeated: bool,
) -> dict[str, Any]:
    combined = "\n".join(tick.replies)
    tags: list[str] = []
    reasons: list[str] = []
    reaction_type = reaction["type"]
    target_text = "\n".join(message.text for message in tick.target_messages)
    private_fact = any(pattern.search(combined) for pattern in PRIVATE_FACT_HINTS.values())

    if repeated:
        tags.append("cross_turn_repetition_candidate")
        reasons.append("与该聊天中较早的 Bot 回复高度重复")
    if len(tick.replies) > 1:
        tags.append("multi_message_reply")
    if private_fact:
        tags.append("private_fact_verification_needed")
    if ROUGH_LANGUAGE_HINT.search(combined):
        tags.append("rough_language_context_review")
    p90 = style.get("p90_chars")
    if p90 and len(combined) > max(80, p90 * 2.5):
        tags.append("human_style_length_outlier")
        reasons.append("长度明显偏离该聊天历史真人表达")
    if reaction_type in {"correction", "dislike"}:
        tags.append(f"historical_{reaction_type}_reaction")
        reasons.append("随后出现纠正或明确负面反应")

    # A critical provisional label requires contextual evidence, never a keyword alone.
    if reaction_type == "correction" and private_fact:
        grade, confidence = "critical", 0.82
        reasons.append("私人事实候选随后被真人纠正，需优先人工确认")
    elif reaction_type in {"correction", "dislike"} or repeated or "human_style_length_outlier" in tags:
        grade, confidence = "bad", 0.72
    else:
        normalized_length = len(_normalize(combined))
        target_is_question = bool(re.search(r"[？?]", target_text))
        answers_question = target_is_question and normalized_length >= 6
        information_value = bool(INFORMATION_VALUE_HINT.search(combined)) and normalized_length >= 8
        emotional_value = bool(EMOTIONAL_VALUE_HINT.search(combined))
        relationship_banter = bool(ROUGH_LANGUAGE_HINT.search(combined)) and bool(style.get("rough_language_rate"))
        advances_topic = bool(re.search(r"[？?]", combined)) and normalized_length >= 6
        has_value = answers_question or information_value or emotional_value or relationship_banter or advances_topic
        if reaction_type == "positive" or has_value:
            grade, confidence = "good", 0.62 if reaction_type != "positive" else 0.75
            reasons.append("表达贴近历史真人风格，并提供信息、情绪或推进价值")
        else:
            grade, confidence = "acceptable", 0.55
            reasons.append("未见明确错误，但可自动确认的互动价值有限")

    return {
        "grade": grade,
        "diagnostic_tags": sorted(set(tags)),
        "reason": "；".join(reasons),
        "confidence": confidence,
        "status": "ai_provisional_not_human_ground_truth",
    }


def _repeat_flags(ticks: list[ReplyTick]) -> dict[int, bool]:
    prior: dict[str, list[str]] = defaultdict(list)
    flags = {}
    for tick in ticks:
        repeated = False
        for reply in tick.replies:
            normalized = _normalize(reply)
            if len(normalized) >= 8 and any(
                len(old) >= 8 and SequenceMatcher(None, normalized, old).ratio() >= 0.82
                for old in prior[tick.chat_key]
            ):
                repeated = True
            prior[tick.chat_key].append(normalized)
        flags[tick.db_id] = repeated
    return flags


def _low_information_candidate(tick: ReplyTick) -> bool:
    combined = "\n".join(tick.replies)
    normalized_length = len(_normalize(combined))
    has_signal = bool(
        INFORMATION_VALUE_HINT.search(combined)
        or EMOTIONAL_VALUE_HINT.search(combined)
        or re.search(r"[？?]", combined)
    )
    return normalized_length <= 8 and not has_signal


def _rapid_same_chat_sequences(ticks: list[ReplyTick]) -> int:
    previous: dict[str, float] = {}
    count = 0
    for tick in ticks:
        timestamp = _created_timestamp(tick.created_at)
        if timestamp is None:
            continue
        prior = previous.get(tick.chat_key)
        if prior is not None and 0 < timestamp - prior <= 120:
            count += 1
        previous[tick.chat_key] = timestamp
    return count


# Kept module-local only while a packet is being built; it avoids placing chat
# identity in serialized case records.
_CURRENT_TICKS: list[ReplyTick] = []


def build_review_packet(ticks: list[ReplyTick], dataset_fingerprint: str) -> dict[str, Any]:
    """Create a local-only packet containing all context needed for review."""
    global _CURRENT_TICKS
    _CURRENT_TICKS = ticks
    timelines = _unique_messages(ticks)
    _, chat_profiles = build_human_reference(ticks)
    repeats = _repeat_flags(ticks)
    cases = []
    try:
        for tick in ticks:
            timeline = timelines.get(tick.chat_key, [])
            sent_at = _sent_timestamp(tick, timeline)
            reactions = _reaction_window(tick, timeline, sent_at)
            reaction = _reaction_evidence(reactions)
            profile = chat_profiles.get(tick.chat_key, _style_profile([]))
            provisional = _provisional_grade(tick, reaction, profile, repeats[tick.db_id])
            case_id = hashlib.sha256(
                f"{dataset_fingerprint}:{tick.db_id}".encode()
            ).hexdigest()[:20]
            pre_context = list(tick.input_messages[-12:])
            cases.append(
                {
                    "case_id": case_id,
                    "source_db_id": tick.db_id,
                    "created_at": tick.created_at,
                    "is_group": tick.is_group,
                    "pre_context": [_private_message(message) for message in pre_context],
                    "target_messages": [_private_message(message) for message in tick.target_messages],
                    "bot_replies": list(tick.replies),
                    "post_reaction_window": [_private_message(message) for message in reactions],
                    "historical_reaction_evidence": reaction,
                    "human_style_reference": profile,
                    "ai_provisional": provisional,
                    "human_ground_truth": (
                        {
                            "is_badcase": bool(tick.human_is_badcase),
                            "type": tick.human_badcase_type,
                            "notes": tick.human_notes,
                        }
                        if tick.human_is_badcase is not None else None
                    ),
                    "judge_fact_safety": (
                        {
                            "score": tick.judge_score,
                            "is_badcase": tick.judge_is_badcase,
                            "reason": tick.judge_reason,
                            "usage": "fact_and_safety_hint_only",
                        }
                        if tick.judge_score is not None or tick.judge_is_badcase is not None else None
                    ),
                }
            )
    finally:
        _CURRENT_TICKS = []
    return {
        "schema_version": SCHEMA_VERSION,
        "privacy": "LOCAL PRIVATE REVIEW DATA: contains chat text; never publish or commit",
        "dataset_fingerprint": dataset_fingerprint,
        "rubric": {
            "good": "Fits the real relationship/context and clearly adds information, emotion, or momentum.",
            "acceptable": "Fits and is harmless, but adds limited value.",
            "bad": "Should not reply, misses context, repeats, feels mechanical, or adds low/misaligned value.",
            "critical": "Context-supported serious identity, factual, object, or boundary failure; keywords alone never qualify.",
        },
        "cases": cases,
    }


def build_baseline(
    ticks: list[ReplyTick], total_ticks: int, review_packet: dict[str, Any]
) -> dict[str, Any]:
    """Build the publishable aggregate report without chat text or identity."""
    messages = [reply for tick in ticks for reply in tick.replies]
    group_ticks = [tick for tick in ticks if tick.is_group]
    private_ticks = [tick for tick in ticks if not tick.is_group]
    cases = review_packet["cases"]
    grade_counts = Counter(case["ai_provisional"]["grade"] for case in cases)
    tag_counts = Counter(
        tag for case in cases for tag in case["ai_provisional"]["diagnostic_tags"]
    )
    reaction_counts = Counter(
        case["historical_reaction_evidence"]["type"] for case in cases
    )
    human_labeled = sum(case["human_ground_truth"] is not None for case in cases)
    public_human_reference, _ = build_human_reference(ticks)
    bot_style = {
        "group": _bot_style_profile(group_ticks),
        "private": _bot_style_profile(private_ticks),
    }
    chat_counts = Counter(tick.chat_key for tick in ticks)
    date_values = [tick.created_at for tick in ticks if tick.created_at]
    problem_share = (
        (grade_counts["bad"] + grade_counts["critical"]) / len(cases) if cases else None
    )
    group_human_messages = public_human_reference["group"]["human_messages"] or 0
    private_human_messages = public_human_reference["private"]["human_messages"] or 0
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": review_packet["dataset_fingerprint"],
        "scope": {
            "source": "tick_log WHERE send_success = 1",
            "database_ticks": total_ticks,
            "sent_ticks": len(ticks),
            "sent_messages": len(messages),
            "archived_review_cases": len(cases),
            "coverage": round(len(cases) / len(ticks), 4) if ticks else 0.0,
            "date_start": min(date_values, default=None),
            "date_end": max(date_values, default=None),
        },
        "volume": {
            "group_ticks": len(group_ticks),
            "group_messages": sum(len(tick.replies) for tick in group_ticks),
            "private_ticks": len(private_ticks),
            "private_messages": sum(len(tick.replies) for tick in private_ticks),
            "multi_message_ticks": sum(len(tick.replies) > 1 for tick in ticks),
            "dominant_chat_ticks": max(chat_counts.values(), default=0),
            "dominant_chat_share": round(max(chat_counts.values(), default=0) / len(ticks), 4) if ticks else None,
        },
        "human_chat_reference": public_human_reference,
        "bot_style_observed": bot_style,
        "ai_provisional_baseline": {
            "status": "not_human_ground_truth",
            "grade_counts": {grade: grade_counts[grade] for grade in GRADES},
            "bad_or_critical_share": round(problem_share, 4) if problem_share is not None else None,
            "diagnostic_tag_counts": dict(sorted(tag_counts.items())),
            "interpretation": "A reproducible first-pass triage baseline; it cannot replace human review.",
        },
        "historical_reaction_evidence": dict(sorted(reaction_counts.items())),
        "human_ground_truth": {
            "labeled_cases": human_labeled,
            "total_cases": len(cases),
            "coverage": round(human_labeled / len(cases), 4) if cases else None,
            "status": "pending" if human_labeled < len(cases) else "complete",
        },
        "judge_role": {
            "sent_cases_with_judge_hint": sum(case["judge_fact_safety"] is not None for case in cases),
            "usage": "fact_and_safety_diagnostic_only",
            "excluded_from_experience_grade": True,
        },
        "conversation_problem_signals": {
            "multi_message_reply_cases": sum(len(tick.replies) > 1 for tick in ticks),
            "low_information_reply_candidates": sum(_low_information_candidate(tick) for tick in ticks),
            "cross_turn_repetition_candidates": tag_counts["cross_turn_repetition_candidate"],
            "human_style_length_outliers": tag_counts["human_style_length_outlier"],
            "rough_language_context_review_candidates": tag_counts["rough_language_context_review"],
            "rapid_same_chat_reply_sequences": _rapid_same_chat_sequences(ticks),
            "group_bot_messages_per_100_human_messages": (
                round(100 * sum(len(tick.replies) for tick in group_ticks) / group_human_messages, 2)
                if group_human_messages else None
            ),
            "private_bot_messages_per_100_human_messages": (
                round(100 * sum(len(tick.replies) for tick in private_ticks) / private_human_messages, 2)
                if private_human_messages else None
            ),
            "interpretation": "Conversation-level screening signals; human confirmation is required.",
        },
        "baseline_status": "fully_archived_ai_provisional",
        "release_decision": "human_ground_truth_pending",
        "limitations": [
            "AI provisional grades are triage judgments, not historical human ground truth.",
            "Profanity, teasing and aggression are contextual style features, not automatic defects; stale jokes and repetition remain defects.",
            "Historical replies span multiple prompt/model versions and are not a replay of the current model.",
            "Sent-only history cannot measure missed replies or reply-decision recall.",
            "No chat identity, message text, prompt text, or human notes are present in this public report.",
        ],
    }


def generate_evaluation(db_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ticks, total_ticks = load_sent_ticks(db_path)
    fingerprint = _fingerprint(ticks)
    packet = build_review_packet(ticks, fingerprint)
    report = build_baseline(ticks, total_ticks, packet)
    generated_at = datetime.now().astimezone().isoformat()
    packet["generated_at"] = generated_at
    report["generated_at"] = generated_at
    return report, packet


def generate_baseline(db_path: Path) -> dict[str, Any]:
    """Compatibility entry point returning only the privacy-safe report."""
    report, _ = generate_evaluation(db_path)
    return report
