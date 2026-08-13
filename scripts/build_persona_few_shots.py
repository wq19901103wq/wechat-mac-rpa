#!/usr/bin/env python3
"""从 WeFlow JSON 导出中本地抽取、脱敏并分层选择本人回复样本。"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SENSITIVE_PATTERNS = [
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?i)https?://|www\."),
    re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?i)wxid_[a-z0-9_]+"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
]
PLACEHOLDER_PATTERNS = [
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
    (re.compile(r"(?i)https?://\S+|www\.\S+"), "[链接]"),
    (re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"), "[邮箱]"),
    (re.compile(r"(?i)wxid_[a-z0-9_]+"), "[微信账号]"),
]
MAX_TURN_GAP_SECONDS = 600


@dataclass
class Candidate:
    relation: str
    chat_id: str
    context: list[str]
    replies: list[str]
    score: float
    timestamp: int
    intent: str = "comment"
    reply_shape: str = "single"
    context_dependency: str = "low"
    style_tags: tuple[str, ...] = ()
    topic: str = "daily_chat"
    humor_type: str = "none"
    response_mode: str = "neutral"
    priority: bool = False
    source_provenance: str = "unverified"
    semantic_profile: dict[str, str] | None = None


def _stable_id(value: str, prefix: str = "chat") -> str:
    normalized = re.sub(r"^(私聊_|群聊_|曾经的好友_)", "", value).strip()
    return f"{prefix}_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:10]}"


def _is_text(message: dict) -> bool:
    return message.get("localType") == 1 and bool(str(message.get("content") or "").strip())


def _safe_text(text: str) -> bool:
    text = text.strip()
    if not 1 <= len(text) <= 160 or text.startswith("<"):
        return False
    return not any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _is_low_signal_reply(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return bool(
        re.fullmatch(r"[?？!！.,，。~～]+", compact)
        or re.fullmatch(r"@[^\s]+", compact)
        or re.fullmatch(r"\[[^\]]+\]", compact)
    )


def _timestamp_seconds(value: object) -> float:
    try:
        timestamp = float(value or 0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp


def _is_coherent_turn(incoming: list[dict], outgoing: list[dict]) -> bool:
    first_incoming_time = _timestamp_seconds(incoming[0].get("createTime"))
    incoming_time = _timestamp_seconds(incoming[-1].get("createTime"))
    outgoing_time = _timestamp_seconds(outgoing[0].get("createTime"))
    if not incoming_time or not outgoing_time:
        return True
    gap = outgoing_time - incoming_time
    full_span = outgoing_time - first_incoming_time if first_incoming_time else gap
    return 0 <= gap <= MAX_TURN_GAP_SECONDS and full_span <= MAX_TURN_GAP_SECONDS * 3


def _human_provenance(message: dict, human_before: float | None) -> str | None:
    if message.get("authorship") == "human":
        return "explicit_human_marker"
    timestamp = _timestamp_seconds(message.get("createTime"))
    if human_before is not None and timestamp and timestamp < human_before:
        return "before_automation_cutoff"
    return None


def _topic(context: list[str], replies: list[str]) -> str:
    del context, replies
    return "daily_chat"


def _humor_type(context: list[str], replies: list[str]) -> str:
    del context, replies
    return "none"


def _intent(context: list[str], replies: list[str]) -> str:
    del context, replies
    return "comment"


def _response_mode(context: list[str], replies: list[str]) -> str:
    del context, replies
    return "neutral"


def _needs_sincere_response(context: list[str]) -> bool:
    del context
    return False


def _reply_metadata(context: list[str], replies: list[str]) -> tuple[str, str, tuple[str, ...]]:
    reply_text = " ".join(replies)
    if len(replies) > 1:
        shape = "multi"
    elif len(reply_text) <= 4:
        shape = "reaction"
    else:
        shape = "single"
    del context
    return shape, "low", ()


def _sanitize(text: str, names: Iterable[str]) -> str:
    result = text.strip()
    for pattern, replacement in PLACEHOLDER_PATTERNS:
        result = pattern.sub(replacement, result)
    for name in sorted({n.strip() for n in names if 2 <= len(n.strip()) <= 30}, key=len, reverse=True):
        result = result.replace(name, "[联系人]")
    return re.sub(r"\s+", " ", result).strip()


def _wiki_relation(export_path: Path, wiki_dir: Path) -> str:
    stem = export_path.stem
    if stem.startswith("群聊_"):
        return "group"
    del wiki_dir
    return "acquaintance"


def _score(context: list[str], replies: list[str]) -> float:
    reply = "".join(replies)
    score = 10.0 - abs(len(reply) - 14) * 0.08
    score += min(len(context), 3) * 0.5
    score -= max(0, len(reply) - 80) * 0.08
    return score


def extract_candidates(
    export_dir: Path,
    wiki_dir: Path,
    human_before: float | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for path in sorted(export_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        messages = payload.get("messages", [])
        relation = _wiki_relation(path, wiki_dir)
        chat_id = _stable_id(path.stem)
        session_name = str(payload.get("session", {}).get("displayName") or "")
        all_names = {session_name, re.sub(r"^(私聊_|群聊_|曾经的好友_)", "", path.stem)}
        all_names.update(str(m.get("senderDisplayName") or "") for m in messages)
        i = 0
        while i < len(messages):
            if not _is_text(messages[i]) or messages[i].get("isSend"):
                i += 1
                continue
            incoming: list[dict] = []
            while i < len(messages) and _is_text(messages[i]) and not messages[i].get("isSend") and len(incoming) < 5:
                incoming.append(messages[i])
                i += 1
            outgoing: list[dict] = []
            while i < len(messages) and _is_text(messages[i]) and messages[i].get("isSend") and len(outgoing) < 3:
                outgoing.append(messages[i])
                i += 1
            if not incoming or not outgoing:
                continue
            provenance = {_human_provenance(message, human_before) for message in outgoing}
            if None in provenance or len(provenance) != 1:
                continue
            source_provenance = provenance.pop()
            if source_provenance is None:
                continue
            if not _is_coherent_turn(incoming, outgoing):
                continue
            if relation == "group":
                senders = {m.get("senderUsername") or m.get("senderDisplayName") for m in incoming}
                if len(senders) != 1:
                    continue
            raw_context = [str(m.get("content") or "") for m in incoming]
            raw_replies = [str(m.get("content") or "") for m in outgoing]
            if not all(_safe_text(t) for t in raw_context + raw_replies):
                continue
            context = [_sanitize(t, all_names) for t in raw_context]
            replies = [_sanitize(t, all_names) for t in raw_replies]
            if not all(context + replies):
                continue
            if _is_low_signal_reply("".join(replies)):
                continue
            fingerprint = re.sub(r"\W+", "", "|".join(context + replies)).lower()
            if len(fingerprint) < 3 or fingerprint in seen:
                continue
            seen.add(fingerprint)
            reply_shape, context_dependency, style_tags = _reply_metadata(context, replies)
            humor_type = _humor_type(context, replies)
            candidates.append(Candidate(
                relation=relation,
                chat_id=chat_id,
                context=context,
                replies=replies,
                score=_score(context, replies),
                timestamp=int(outgoing[0].get("createTime") or 0),
                intent=_intent(context, replies),
                reply_shape=reply_shape,
                context_dependency=context_dependency,
                style_tags=style_tags,
                topic=_topic(context, replies),
                humor_type=humor_type,
                response_mode=_response_mode(context, replies),
                source_provenance=source_provenance,
            ))
    return candidates


def extract_chat_backup(path: Path, human_before: float | None = None) -> list[Candidate]:
    """抽取有明确真人来源标记或早于自动化 cutoff 的本人发言。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    chat_name = str(payload.get("chat_name") or path.stem)
    chat_id = _stable_id(chat_name)
    names = {chat_name}
    names.update(str(m.get("sender") or "") for m in messages if m.get("sender_type") != "self")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for index, message in enumerate(messages):
        if message.get("sender_type") != "self" or message.get("message_type") != "text":
            continue
        provenance = _human_provenance(message, human_before)
        if provenance is None:
            continue
        reply = str(message.get("text") or "").strip()
        if not _safe_text(reply):
            continue
        incoming: list[str] = []
        cursor = index - 1
        while cursor >= 0 and len(incoming) < 5:
            previous = messages[cursor]
            if previous.get("sender_type") == "self":
                break
            text = str(previous.get("text") or "").strip()
            if previous.get("message_type") == "text" and _safe_text(text):
                incoming.append(text)
            cursor -= 1
        if not incoming:
            continue
        context = [_sanitize(text, names) for text in reversed(incoming)]
        replies = [_sanitize(reply, names)]
        if _is_low_signal_reply("".join(replies)):
            continue
        fingerprint = re.sub(r"\W+", "", "|".join(context + replies)).lower()
        if len(fingerprint) < 3 or fingerprint in seen:
            continue
        seen.add(fingerprint)
        reply_shape, context_dependency, style_tags = _reply_metadata(context, replies)
        humor_type = _humor_type(context, replies)
        candidates.append(Candidate(
            relation="group",
            chat_id=chat_id,
            context=context,
            replies=replies,
            score=_score(context, replies) + 6.0,
            timestamp=int(message.get("createTime") or 0),
            intent=_intent(context, replies),
            reply_shape=reply_shape,
            context_dependency=context_dependency,
            style_tags=style_tags,
            topic=_topic(context, replies),
            humor_type=humor_type,
            response_mode=_response_mode(context, replies),
            priority=True,
            source_provenance=provenance,
        ))
    return candidates


def load_verified_examples(path: Path) -> list[Candidate]:
    """加载人工核验的真人样本；来源标记缺失或不可信时拒绝整批数据。"""
    candidates: list[Candidate] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        provenance = row.get("source_provenance")
        if provenance not in {"explicit_human_marker", "before_automation_cutoff"}:
            raise ValueError(f"{path}:{line_number} 缺少可信真人来源标记")
        context = row.get("context")
        replies = row.get("reply")
        if (
            not isinstance(context, list)
            or not isinstance(replies, list)
            or not context
            or not replies
            or not all(isinstance(text, str) and _safe_text(text) for text in context + replies)
        ):
            raise ValueError(f"{path}:{line_number} 样本格式或内容不合法")
        profile = row.get("semantic_profile") or None
        if profile is not None and (
            not isinstance(profile, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in profile.items())
        ):
            raise ValueError(f"{path}:{line_number} semantic_profile 必须是字符串键值对象")
        reply_shape, context_dependency, style_tags = _reply_metadata(context, replies)
        candidates.append(Candidate(
            relation=str(row.get("relationship") or "group"),
            chat_id=str(row.get("chat_id") or _stable_id(str(path), "verified")),
            context=context,
            replies=replies,
            score=float(row.get("score", 100.0)),
            timestamp=int(row.get("timestamp") or 0),
            intent=str(row.get("intent") or "comment"),
            reply_shape=str(row.get("reply_shape") or reply_shape),
            context_dependency=str(row.get("context_dependency") or context_dependency),
            style_tags=tuple(row.get("style_tags") or style_tags),
            topic=str(row.get("topic") or "daily_chat"),
            humor_type=str(row.get("humor_type") or "situational"),
            response_mode=str(row.get("response_mode") or "playful"),
            priority=True,
            source_provenance=provenance,
            semantic_profile=profile,
        ))
    return candidates


def select_balanced(candidates: list[Candidate], limit: int, group_target: int = 0) -> list[Candidate]:
    by_relation: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_relation[candidate.relation].append(candidate)
    for rows in by_relation.values():
        rows.sort(key=lambda c: (-c.score, -c.timestamp, c.chat_id))
    selected: list[Candidate] = []
    per_chat: Counter[str] = Counter()
    per_intent: Counter[tuple[str, str]] = Counter()
    per_shape: Counter[tuple[str, str]] = Counter()

    def pop_diverse(
        rows: list[Candidate],
        max_per_chat: int = 3,
        prefer_priority: bool = False,
    ) -> Candidate | None:
        eligible = [
            (index, candidate) for index, candidate in enumerate(rows)
            if per_chat[candidate.chat_id] < (15 if prefer_priority and candidate.priority else max_per_chat)
        ]
        if not eligible:
            rows.clear()
            return None
        index, candidate = min(
            eligible,
            key=lambda item: (
                not item[1].priority if prefer_priority else False,
                per_intent[(item[1].relation, item[1].intent)],
                per_shape[(item[1].relation, item[1].reply_shape)],
                -item[1].score,
                -item[1].timestamp,
                item[1].chat_id,
            ),
        )
        rows.pop(index)
        return candidate

    if group_target:
        group_rows = by_relation.get("group", [])
        group_rows.sort(key=lambda c: (not c.priority, -c.score, -c.timestamp, c.chat_id))
        while group_rows and len(selected) < min(group_target, limit):
            group_candidate = pop_diverse(group_rows, prefer_priority=True)
            if group_candidate is None:
                break
            selected.append(group_candidate)
            per_chat[group_candidate.chat_id] += 1
            per_intent[(group_candidate.relation, group_candidate.intent)] += 1
            per_shape[(group_candidate.relation, group_candidate.reply_shape)] += 1
    relations = ["family", "friend", "colleague", "acquaintance", "service"]
    if not group_target:
        relations.append("group")
    while len(selected) < limit:
        progressed = False
        for relation in relations:
            rows = by_relation.get(relation, [])
            rel_candidate = pop_diverse(rows)
            if rel_candidate is None:
                continue
            selected.append(rel_candidate)
            per_chat[rel_candidate.chat_id] += 1
            per_intent[(rel_candidate.relation, rel_candidate.intent)] += 1
            per_shape[(rel_candidate.relation, rel_candidate.reply_shape)] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return selected


def select_stratified(
    candidates: list[Candidate],
    limit: int,
    min_per_chat: int = 4,
    max_per_chat: int = 20,
    humor_ratio: float = 0.4,
    sincere_ratio: float = 0.1,
) -> list[Candidate]:
    """按聊天对象、场景和话题分层，控制幽默比例与单对象样本数。"""
    if limit <= 0 or min_per_chat <= 0 or max_per_chat < min_per_chat:
        return []
    humor_ratio = min(1.0, max(0.0, humor_ratio))
    sincere_ratio = min(0.3, max(0.0, sincere_ratio))
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_chat[candidate.chat_id].append(candidate)
    eligible_chats = [
        (chat_id, rows) for chat_id, rows in by_chat.items()
        if len(rows) >= min_per_chat
    ]
    eligible_chats.sort(key=lambda item: (
        -len({(row.intent, row.topic) for row in item[1]}),
        -len(item[1]),
        -max(row.timestamp for row in item[1]),
        item[0],
    ))
    target_per_chat = min(max_per_chat, max(min_per_chat, 8))
    target_chat_count = min(len(eligible_chats), math.ceil(limit / target_per_chat))
    pools = {
        chat_id: sorted(rows, key=lambda row: (not row.priority, -row.score, -row.timestamp))
        for chat_id, rows in eligible_chats[:target_chat_count]
    }
    chat_ids = list(pools)
    selected: list[Candidate] = []
    per_chat: Counter[str] = Counter()
    per_bucket: Counter[tuple[str, str, str]] = Counter()
    per_scenario: Counter[tuple[str, str]] = Counter()
    per_topic: Counter[tuple[str, str]] = Counter()
    humor_per_chat: Counter[str] = Counter()
    humor_count = 0

    def chat_humor_ratio(chat_id: str) -> float:
        relation = pools[chat_id][0].relation if pools[chat_id] else ""
        if relation in {"friend", "group"}:
            return min(0.55, humor_ratio + 0.05)
        if relation == "family":
            return min(humor_ratio, 0.35)
        if relation == "service":
            return min(humor_ratio, 0.2)
        return humor_ratio

    def choose(chat_id: str) -> Candidate | None:
        nonlocal humor_count
        want_humor = humor_per_chat[chat_id] < round(
            (per_chat[chat_id] + 1) * chat_humor_ratio(chat_id)
        )
        rows = pools[chat_id]
        eligible = [
            (index, row) for index, row in enumerate(rows)
            if per_bucket[(chat_id, row.intent, row.topic)] < 2
        ]
        if not eligible:
            return None
        index, candidate = min(eligible, key=lambda item: (
            (item[1].humor_type != "none") != want_humor,
            per_scenario[(chat_id, item[1].intent)],
            per_topic[(chat_id, item[1].topic)],
            not item[1].priority,
            -item[1].score,
            -item[1].timestamp,
        ))
        rows.pop(index)
        return candidate

    def add(candidate: Candidate) -> None:
        nonlocal humor_count
        selected.append(candidate)
        per_chat[candidate.chat_id] += 1
        per_bucket[(candidate.chat_id, candidate.intent, candidate.topic)] += 1
        per_scenario[(candidate.chat_id, candidate.intent)] += 1
        per_topic[(candidate.chat_id, candidate.topic)] += 1
        humor_count += candidate.humor_type != "none"
        humor_per_chat[candidate.chat_id] += candidate.humor_type != "none"

    for _ in range(min_per_chat):
        for chat_id in chat_ids:
            if len(selected) >= limit:
                return selected
            fill_candidate = choose(chat_id)
            if fill_candidate is None:
                continue
            add(fill_candidate)

    while len(selected) < limit:
        progressed = False
        for chat_id in chat_ids:
            if per_chat[chat_id] >= max_per_chat:
                continue
            more_candidate = choose(chat_id)
            if more_candidate is None:
                continue
            add(more_candidate)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    target_humor_count = round(len(selected) * humor_ratio)
    relation_priority = {"group": 0, "friend": 1, "colleague": 2, "acquaintance": 3, "family": 4, "service": 5}
    rebalance_chats = sorted(
        chat_ids,
        key=lambda chat_id: (
            relation_priority.get(pools[chat_id][0].relation if pools[chat_id] else "", 9),
            humor_per_chat[chat_id],
        ),
    )
    while humor_count < target_humor_count:
        swapped = False
        for chat_id in rebalance_chats:
            incoming_rows = [
                (index, row) for index, row in enumerate(pools[chat_id])
                if row.humor_type != "none"
            ]
            outgoing_rows = [
                (index, row) for index, row in enumerate(selected)
                if row.chat_id == chat_id and row.humor_type == "none"
            ]
            if not incoming_rows or not outgoing_rows:
                continue
            incoming_rows.sort(key=lambda item: (
                per_scenario[(chat_id, item[1].intent)],
                per_topic[(chat_id, item[1].topic)],
                -item[1].score,
            ))
            outgoing_rows.sort(key=lambda item: (item[1].score, item[1].timestamp))
            replacement = None
            for incoming_index, incoming in incoming_rows:
                incoming_bucket = (chat_id, incoming.intent, incoming.topic)
                for outgoing_index, outgoing in outgoing_rows:
                    outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
                    resulting_bucket_count = per_bucket[incoming_bucket] - (incoming_bucket == outgoing_bucket)
                    if resulting_bucket_count < 2:
                        replacement = (incoming_index, incoming, outgoing_index, outgoing)
                        break
                if replacement:
                    break
            if replacement is None:
                continue
            incoming_index, incoming, outgoing_index, outgoing = replacement
            pools[chat_id].pop(incoming_index)
            selected[outgoing_index] = incoming
            outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
            incoming_bucket = (chat_id, incoming.intent, incoming.topic)
            per_bucket[outgoing_bucket] -= 1
            per_bucket[incoming_bucket] += 1
            per_scenario[(chat_id, outgoing.intent)] -= 1
            per_scenario[(chat_id, incoming.intent)] += 1
            per_topic[(chat_id, outgoing.topic)] -= 1
            per_topic[(chat_id, incoming.topic)] += 1
            humor_count += 1
            humor_per_chat[chat_id] += 1
            swapped = True
            if humor_count >= target_humor_count:
                break
        if not swapped:
            break

    sincere_count = sum(row.response_mode == "sincere" for row in selected)
    target_sincere_count = round(len(selected) * sincere_ratio)
    while sincere_count < target_sincere_count:
        swapped = False
        for chat_id in chat_ids:
            incoming_rows = [
                (index, row) for index, row in enumerate(pools[chat_id])
                if row.response_mode == "sincere"
            ]
            outgoing_rows = [
                (index, row) for index, row in enumerate(selected)
                if row.chat_id == chat_id and row.response_mode in {"neutral", "practical"}
            ]
            if not incoming_rows or not outgoing_rows:
                continue
            incoming_rows.sort(key=lambda item: (-item[1].score, -item[1].timestamp))
            outgoing_rows.sort(key=lambda item: (item[1].score, item[1].timestamp))
            replacement = None
            for incoming_index, incoming in incoming_rows:
                incoming_bucket = (chat_id, incoming.intent, incoming.topic)
                for outgoing_index, outgoing in outgoing_rows:
                    outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
                    if per_bucket[incoming_bucket] - (incoming_bucket == outgoing_bucket) < 2:
                        replacement = (incoming_index, incoming, outgoing_index, outgoing)
                        break
                if replacement:
                    break
            if replacement is None:
                continue
            incoming_index, incoming, outgoing_index, outgoing = replacement
            pools[chat_id].pop(incoming_index)
            selected[outgoing_index] = incoming
            outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
            incoming_bucket = (chat_id, incoming.intent, incoming.topic)
            per_bucket[outgoing_bucket] -= 1
            per_bucket[incoming_bucket] += 1
            sincere_count += 1
            swapped = True
            if sincere_count >= target_sincere_count:
                break
        if not swapped:
            break
    return selected


def split_temporal_holdout(
    candidates: list[Candidate],
    holdout_ratio: float = 0.2,
    min_train_per_chat: int = 8,
    min_holdout_per_chat: int = 2,
) -> tuple[list[Candidate], list[Candidate]]:
    """每个对象按时间切分，较新的回复只进入 holdout。"""
    holdout_ratio = min(0.5, max(0.05, holdout_ratio))
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_chat[candidate.chat_id].append(candidate)
    train: list[Candidate] = []
    holdout: list[Candidate] = []
    for rows in by_chat.values():
        ordered = sorted(rows, key=lambda row: (row.timestamp, row.context, row.replies))
        if len(ordered) < min_train_per_chat + min_holdout_per_chat:
            train.extend(ordered)
            continue
        holdout_count = max(min_holdout_per_chat, int(len(ordered) * holdout_ratio))
        holdout_count = min(holdout_count, len(ordered) - min_train_per_chat)
        train.extend(ordered[:-holdout_count])
        holdout.extend(ordered[-holdout_count:])
    return train, holdout


def select_holdout_cases(
    candidates: list[Candidate],
    limit: int,
    max_per_chat: int = 2,
) -> list[Candidate]:
    """从真实留出候选中覆盖不同对象、场景和话题。"""
    buckets: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate.intent, candidate.topic, candidate.response_mode)].append(candidate)
    rows_by_bucket = []
    for key, rows in sorted(buckets.items()):
        rows.sort(key=lambda row: (-row.timestamp, -row.score, row.chat_id))
        rows_by_bucket.append((key, rows))
    selected: list[Candidate] = []
    per_chat: Counter[str] = Counter()
    selected_ids: set[int] = set()
    sincerity_rows = sorted(
        (row for row in candidates if _needs_sincere_response(row.context)),
        key=lambda row: (-row.timestamp, -row.score),
    )
    for candidate in sincerity_rows:
        if len(selected) >= min(limit, 40) or per_chat[candidate.chat_id] >= max_per_chat:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        per_chat[candidate.chat_id] += 1
    while len(selected) < limit:
        progressed = False
        for _, rows in rows_by_bucket:
            while rows and (
                per_chat[rows[0].chat_id] >= max_per_chat or id(rows[0]) in selected_ids
            ):
                rows.pop(0)
            if not rows:
                continue
            candidate = rows.pop(0)
            selected.append(candidate)
            selected_ids.add(id(candidate))
            per_chat[candidate.chat_id] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def write_holdout_cases(
    selected: list[Candidate],
    output_dir: Path,
    train_candidate_count: int,
    holdout_candidate_count: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, 1):
        rows.append({
            "id": f"holdout_{index:03d}",
            "relationship": candidate.relation,
            "chat_id": candidate.chat_id,
            "context": candidate.context,
            "expected_reply": candidate.replies,
            "intent": candidate.intent,
            "topic": candidate.topic,
            "humor_type": candidate.humor_type,
            "response_mode": candidate.response_mode,
            "source_provenance": candidate.source_provenance,
            "desired_response_mode": "sincere" if _needs_sincere_response(candidate.context) else "auto",
            "timestamp": candidate.timestamp,
        })
    (output_dir / "holdout_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "train_candidate_count": train_candidate_count,
        "holdout_candidate_count": holdout_candidate_count,
        "selected_holdout_count": len(rows),
        "unique_chat_count": len({row["chat_id"] for row in rows}),
        "intent_counts": dict(sorted(Counter(row["intent"] for row in rows).items())),
        "topic_counts": dict(sorted(Counter(row["topic"] for row in rows).items())),
        "response_mode_counts": dict(sorted(Counter(row["response_mode"] for row in rows).items())),
        "desired_sincere_count": sum(row["desired_response_mode"] == "sincere" for row in rows),
    }
    (output_dir / "holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_outputs(selected: list[Candidate], output_dir: Path, candidate_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "persona_examples.jsonl"
    md_path = output_dir / "persona_examples.md"
    report_path = output_dir / "report.json"
    object_report_path = output_dir / "object_report.json"
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, 1):
        rows.append({
            "id": f"example_{index:03d}",
            "relationship": candidate.relation,
            "chat_id": candidate.chat_id,
            "context": candidate.context,
            "reply": candidate.replies,
            "intent": candidate.intent,
            "reply_shape": candidate.reply_shape,
            "context_dependency": candidate.context_dependency,
            "style_tags": list(candidate.style_tags),
            "topic": candidate.topic,
            "humor_type": candidate.humor_type,
            "response_mode": candidate.response_mode,
            "source_provenance": candidate.source_provenance,
            **({"semantic_profile": candidate.semantic_profile} if candidate.semantic_profile else {}),
        })
    jsonl_content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    jsonl_path.write_text(jsonl_content, encoding="utf-8")
    examples_sha256 = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()
    md_lines = ["# 本人真实回复 Few-shot（已脱敏）", "", "> 纯本地生成；请人工复核后再接入生产 prompt。", ""]
    for row in rows:
        md_lines.extend([
            f"## {row['id']} · {row['relationship']} · {row['topic']} / {row['intent']} · humor={row['humor_type']}", "",
            *[f"- 对方：{text}" for text in row["context"]],
            *[f"- 本人：{text}" for text in row["reply"]], "",
        ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    report = {
        "candidate_count": candidate_count,
        "selected_count": len(rows),
        "relationship_counts": dict(sorted(Counter(row["relationship"] for row in rows).items())),
        "intent_counts": dict(sorted(Counter(row["intent"] for row in rows).items())),
        "reply_shape_counts": dict(sorted(Counter(row["reply_shape"] for row in rows).items())),
        "topic_counts": dict(sorted(Counter(row["topic"] for row in rows).items())),
        "humor_type_counts": dict(sorted(Counter(row["humor_type"] for row in rows).items())),
        "response_mode_counts": dict(sorted(Counter(row["response_mode"] for row in rows).items())),
        "humor_ratio": round(sum(row["humor_type"] != "none" for row in rows) / len(rows), 4) if rows else 0,
        "unique_chat_count": len({row["chat_id"] for row in rows}),
        "samples_per_chat": {
            "min": min(Counter(row["chat_id"] for row in rows).values(), default=0),
            "max": max(Counter(row["chat_id"] for row in rows).values(), default=0),
            "average": round(len(rows) / len({row["chat_id"] for row in rows}), 2) if rows else 0,
        },
        "sensitive_pattern_scan_passed": not any(
            pattern.search(text)
            for row in rows
            for text in [*row["context"], *row["reply"]]
            for pattern in SENSITIVE_PATTERNS
        ),
        "review_status": "pending",
        "examples_sha256": examples_sha256,
        "external_model_used": False,
        "source_provenance_counts": dict(sorted(Counter(row["source_provenance"] for row in rows).items())),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_by_chat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_chat[row["chat_id"]].append(row)
    object_report = {
        "object_count": len(rows_by_chat),
        "objects": {
            chat_id: {
                "relationship": chat_rows[0]["relationship"],
                "sample_count": len(chat_rows),
                "humor_count": sum(row["humor_type"] != "none" for row in chat_rows),
                "humor_ratio": round(sum(row["humor_type"] != "none" for row in chat_rows) / len(chat_rows), 4),
                "scenario_counts": dict(sorted(Counter(row["intent"] for row in chat_rows).items())),
                "topic_counts": dict(sorted(Counter(row["topic"] for row in chat_rows).items())),
                "scenario_topic_counts": dict(sorted(Counter(
                    f"{row['intent']} × {row['topic']}" for row in chat_rows
                ).items())),
                "example_ids": [row["id"] for row in chat_rows],
            }
            for chat_id, chat_rows in sorted(rows_by_chat.items())
        },
    }
    object_report_path.write_text(
        json.dumps(object_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/exports/b"))
    parser.add_argument("--wiki-dir", type=Path, default=Path("data/memory/wiki/users"))
    parser.add_argument("--output", type=Path, default=Path("data/few_shot"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--group-target", type=int, default=0)
    parser.add_argument("--chat-backup", type=Path, action="append", default=[])
    parser.add_argument("--verified-examples", type=Path, action="append", default=[])
    parser.add_argument(
        "--human-before",
        type=float,
        help="Unix timestamp；仅该时间之前的本人发言可作为真人素材。未设置时只接受 authorship=human。",
    )
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--min-per-chat", type=int, default=4)
    parser.add_argument("--max-per-chat", type=int, default=20)
    parser.add_argument("--humor-ratio", type=float, default=0.4)
    parser.add_argument("--sincere-ratio", type=float, default=0.1)
    parser.add_argument("--temporal-holdout", action="store_true")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--holdout-limit", type=int, default=200)
    args = parser.parse_args()
    candidates = extract_candidates(args.input, args.wiki_dir, human_before=args.human_before)
    for backup in args.chat_backup:
        candidates.extend(extract_chat_backup(backup, human_before=args.human_before))
    for verified_examples in args.verified_examples:
        candidates.extend(load_verified_examples(verified_examples))
    holdout_candidates: list[Candidate] = []
    if args.temporal_holdout:
        candidates, holdout_candidates = split_temporal_holdout(
            candidates,
            holdout_ratio=args.holdout_ratio,
        )
    if args.stratified:
        selected = select_stratified(
            candidates,
            args.limit,
            min_per_chat=args.min_per_chat,
            max_per_chat=args.max_per_chat,
            humor_ratio=args.humor_ratio,
            sincere_ratio=args.sincere_ratio,
        )
    else:
        selected = select_balanced(candidates, args.limit, group_target=args.group_target)
    if len(selected) < args.limit:
        raise SystemExit(f"候选不足：需要 {args.limit}，实际 {len(selected)}")
    write_outputs(selected, args.output, len(candidates))
    if args.temporal_holdout:
        holdout_selected = select_holdout_cases(holdout_candidates, args.holdout_limit)
        write_holdout_cases(
            holdout_selected,
            args.output,
            train_candidate_count=len(candidates),
            holdout_candidate_count=len(holdout_candidates),
        )
    print(json.dumps({"candidates": len(candidates), "selected": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
