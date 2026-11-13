#!/usr/bin/env python3
"""Build one leak-free, per-case historical-human few-shot manifest for Judge."""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import (  # noqa: E402
    _event_has_tool_call,
    _json_sha256,
    _write_private,
)
from src.reply.few_shot import PersonaFewShotRetriever  # noqa: E402


MATCHER_VERSION = "judge_dynamic_few_shot_behavior_divergence_v7"
MATCH_SCORE_FIELDS = (
    "situation_match",
    "event_stage_match",
    "addressee_match",
)
OPTION_ACTIONS = {
    "answer_now", "defer", "accept", "refuse", "confirm", "joke_tease",
    "agree", "disagree", "ask", "clarify", "supplement",
    "acknowledge_close", "empathize_support", "evaluate_criticize",
    "correct", "coordinate", "other",
}
DEFAULT_RETRIEVAL_CANDIDATES_PER_CASE = 100
DEFAULT_MATCHER_BATCH_SIZE = 20
MAX_ADMITTED_EXAMPLES_PER_CASE = 10
MIN_SITUATION_MATCH = 2
MIN_INDEPENDENT_EVENTS = 2


def _normalized_messages(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    return tuple(
        re.sub(r"\s+", "", html.unescape(str(value))).strip()
        for value in values or []
        if re.sub(r"\s+", "", html.unescape(str(value))).strip()
    )


def _contexts_overlap(
    historical_context: Any,
    historical_reply: Any,
    target_contexts: list[list[str]],
) -> bool:
    historical = (
        _normalized_messages(historical_context)
        + _normalized_messages(historical_reply)
    )
    targets = [_normalized_messages(context) for context in target_contexts]
    historical_pairs = {
        historical[index:index + 2]
        for index in range(len(historical) - 1)
    }
    if any(
        historical_pairs.intersection(
            target[index:index + 2]
            for index in range(len(target) - 1)
        )
        for target in targets
    ):
        return True
    historical_long_messages = {message for message in historical if len(message) >= 12}
    return any(
        historical_long_messages.intersection(target)
        for target in targets
    )


def extract_timeline(values: list[str]) -> list[dict[str, str]]:
    """Return XML messages in their original order without A/B reply content."""
    messages: list[dict[str, str]] = []
    for value in values:
        if "<message" not in value:
            text = html.unescape(str(value)).strip()
            if text:
                messages.append({"sender": "", "role": "other", "text": text})
            continue
        try:
            root = ET.fromstring(f"<root>{value}</root>")
        except ET.ParseError as exc:
            raise ValueError(f"blind context contains malformed XML: {exc}") from exc
        for element in root.iter("message"):
            rendered = html.unescape("".join(element.itertext())).strip()
            sender = ""
            text = rendered
            match = re.match(r"^(.*?)（[^）]+）：(.*)$", rendered, re.DOTALL)
            if match:
                sender, text = match.group(1).strip(), match.group(2).strip()
            role = str(element.get("role") or "other")
            if role == "self":
                sender = "我"
            if text:
                messages.append({"sender": sender, "role": role, "text": text})
    return messages


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pool(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row_ids = [str(row.get("id") or "") for row in rows]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(rows):
        raise ValueError("historical few-shot pool contains duplicate or empty ids")
    return rows


def _historical_event_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("historical_event_id") or "")
    return explicit or str(row["id"])


def prepare_from_retrieval_audit(
    audit: dict[str, Any],
    *,
    blind: dict[str, Any],
    answers: dict[str, Any],
    replay: dict[str, Any],
    reference: dict[str, Any],
    pool_sha256: str,
    pool_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reuse a frozen retrieval audit without rerunning retrieval."""
    blind_sha = _json_sha256(blind)
    reference_sha = str(reference.get("reference_sha256") or "")
    if audit.get("kind") != "judge_dynamic_few_shot_retrieval_audit":
        raise ValueError("retrieval candidate audit has the wrong kind")
    if audit.get("blind_sha256") != blind_sha:
        raise ValueError("retrieval candidate audit does not match blind packet")
    if audit.get("reference_sha256") != reference_sha:
        raise ValueError("retrieval candidate audit does not match reference pack")
    if audit.get("pool_sha256") != pool_sha256:
        raise ValueError("retrieval candidate audit does not match few-shot pool")

    source_by_blind_id = {
        str(row["case_id"]): str(row["source_case_id"])
        for row in answers.get("cases") or []
    }
    replay_by_source_id = {str(row["case_id"]): row for row in replay.get("cases") or []}
    audit_cases = list(audit.get("cases") or [])
    audit_by_id = {str(row.get("case_id") or ""): row for row in audit_cases}
    blind_cases = list(blind.get("cases") or [])
    blind_ids = [str(row["case_id"]) for row in blind_cases]
    if len(audit_by_id) != len(audit_cases) or set(audit_by_id) != set(blind_ids):
        raise ValueError("retrieval candidate audit does not cover blind cases exactly once")
    configured_count = int(
        audit.get("configuration", {}).get("retrieval_candidates_per_case")
        or len((audit_cases[0] if audit_cases else {}).get("selected") or [])
    )
    if configured_count < 1:
        raise ValueError("retrieval candidate audit has no configured candidates")

    prepared: dict[str, dict[str, Any]] = {}
    for blind_case in blind_cases:
        case_id = str(blind_case["case_id"])
        source_case_id = source_by_blind_id[case_id]
        replay_case = replay_by_source_id[source_case_id]
        row = audit_by_id[case_id]
        relationship = str(blind_case["relationship"])
        timeline = extract_timeline(list(blind_case.get("context_original") or []))
        query_messages = timeline[-8:]
        if (
            str(row.get("source_case_id") or "") != source_case_id
            or str(row.get("source_chat_id") or "")
            != str(replay_case.get("source_chat_id") or "")
            or str(row.get("relationship") or "") != relationship
            or row.get("query_messages") != query_messages
        ):
            raise ValueError(f"retrieval candidate audit case metadata changed: {case_id}")
        selected = list(row.get("selected") or [])
        if not 1 <= len(selected) <= configured_count:
            raise ValueError(
                f"retrieval candidate audit case {case_id} does not contain "
                f"between 1 and {configured_count} candidates"
            )
        expected_pool_relationship = "group" if relationship == "group" else "acquaintance"
        if any(str(item.get("pool_relationship") or "") != expected_pool_relationship for item in selected):
            raise ValueError(f"retrieval candidate audit crossed relationship boundary: {case_id}")
        target_contexts = [
            [message["text"] for message in timeline],
            list(replay_case.get("context_original") or []),
        ]
        option_pairs = {
            _normalized_messages(blind_case.get(option) or [])
            for option in ("option_A", "option_B")
        }
        if any(
            str(item.get("chat_id") or "") == str(replay_case.get("source_chat_id") or "")
            and _contexts_overlap(
                item.get("context") or [],
                item.get("human_reply") or [],
                target_contexts,
            )
            for item in selected
        ):
            raise ValueError(f"retrieval candidate audit contains overlapping context: {case_id}")
        if any(
            _normalized_messages(item.get("human_reply") or []) in option_pairs
            for item in selected
        ):
            raise ValueError(f"retrieval candidate audit contains exact option reply: {case_id}")
        candidates = []
        for rank, item in enumerate(selected, start=1):
            item_id = str(item["id"])
            source_row = (pool_by_id or {}).get(item_id, item)
            if pool_by_id is not None and item_id not in pool_by_id:
                raise ValueError(f"retrieval candidate is missing from few-shot pool: {item_id}")
            candidates.append({
                "id": item_id,
                "relationship": relationship,
                "context": [str(value) for value in item.get("context") or []],
                "human_reply": [str(value) for value in item.get("human_reply") or []],
                "chat_id": str(item.get("chat_id") or ""),
                "historical_event_id": _historical_event_id(source_row),
                "same_source_chat": (
                    str(item.get("chat_id") or "")
                    == str(replay_case.get("source_chat_id") or "")
                ),
                "semantic_rank": rank,
            })
        case_audit = copy.deepcopy(row)
        case_audit.pop("matcher", None)
        case_audit.pop("admitted_example_ids", None)
        prepared[case_id] = {
            "relationship": relationship,
            "timeline": timeline,
            "option_A": [str(value) for value in blind_case.get("option_A") or []],
            "option_B": [str(value) for value in blind_case.get("option_B") or []],
            "candidates": candidates,
            "retrieval_audit": case_audit,
        }
    return prepared


def leakage_exclusions(
    pool_rows: list[dict[str, Any]],
    *,
    source_chat_id: str,
    target_contexts: list[list[str]],
    options: list[list[str]],
) -> tuple[set[str], Counter[str]]:
    """Exclude overlapping same-chat context and exact A/B replies."""
    option_pairs = {_normalized_messages(option) for option in options}
    excluded: set[str] = set()
    reasons: Counter[str] = Counter()
    for row in pool_rows:
        row_id = str(row["id"])
        row_context = _normalized_messages(row.get("context") or [])
        row_reply = _normalized_messages(row.get("reply") or [])
        row_reasons = []
        if (
            str(row.get("chat_id") or "") == source_chat_id
            and _contexts_overlap(row_context, row_reply, target_contexts)
        ):
            row_reasons.append("overlapping_source_context")
        if row_reply in option_pairs:
            row_reasons.append("exact_option_reply")
        if row_reasons:
            excluded.add(row_id)
            reasons.update(row_reasons)
    return excluded, reasons


def retrieve_case_candidates(
    retriever: PersonaFewShotRetriever,
    pool_rows: list[dict[str, Any]],
    blind_case: dict[str, Any],
    replay_case: dict[str, Any],
    retrieval_candidates_per_case: int = DEFAULT_RETRIEVAL_CANDIDATES_PER_CASE,
    diversity_prefix_size: int | None = None,
    minimum_candidates_per_case: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if retrieval_candidates_per_case < 1:
        raise ValueError("retrieval candidates per case must be positive")
    relationship = str(blind_case.get("relationship") or "")
    if relationship not in {"private", "group"}:
        raise ValueError(f"unsupported relationship: {relationship}")
    if replay_case.get("relationship") != relationship:
        raise ValueError("blind and replay relationships differ")

    timeline = extract_timeline(list(blind_case.get("context_original") or []))
    if not timeline:
        raise ValueError("blind case has no usable context messages")
    query_messages = timeline[-8:]
    query = "\n".join(message["text"] for message in query_messages)
    full_context = [message["text"] for message in timeline]
    options = [list(blind_case.get("option_A") or []), list(blind_case.get("option_B") or [])]
    excluded_ids, exclusion_reasons = leakage_exclusions(
        pool_rows,
        source_chat_id=str(replay_case.get("source_chat_id") or ""),
        target_contexts=[full_context, list(replay_case.get("context_original") or [])],
        options=options,
    )
    selected = retriever.retrieve(
        query,
        chat_name="",
        is_group=relationship == "group",
        limit=retrieval_candidates_per_case,
        relationship="group" if relationship == "group" else "acquaintance",
        chat_id=str(replay_case.get("source_chat_id") or ""),
        exclude_ids=excluded_ids,
        use_situation=True,
        current_context_messages=query_messages,
        diversity_prefix_size=diversity_prefix_size,
    )
    situation_candidate_count = len(selected)
    if len(selected) < retrieval_candidates_per_case:
        supplemental = retriever.retrieve(
            query,
            chat_name="",
            is_group=relationship == "group",
            limit=retrieval_candidates_per_case - len(selected),
            relationship="group" if relationship == "group" else "acquaintance",
            chat_id=str(replay_case.get("source_chat_id") or ""),
            exclude_ids=excluded_ids | {str(row["id"]) for row in selected},
            use_situation=False,
            current_context_messages=query_messages,
            diversity_prefix_size=0,
        )
        selected.extend(supplemental)
    minimum_candidates = (
        retrieval_candidates_per_case
        if minimum_candidates_per_case is None
        else minimum_candidates_per_case
    )
    if not 1 <= minimum_candidates <= retrieval_candidates_per_case:
        raise ValueError("minimum retrieval candidates must be within the requested limit")
    if len(selected) < minimum_candidates:
        raise ValueError(
            f"case {blind_case['case_id']} retrieved {len(selected)} examples; "
            f"minimum is {minimum_candidates}"
        )
    expected_pool_relationship = "group" if relationship == "group" else "acquaintance"
    if any(row.get("relationship") != expected_pool_relationship for row in selected):
        raise ValueError("retriever crossed the private/group boundary")
    if any(str(row["id"]) in excluded_ids for row in selected):
        raise ValueError("retriever returned an excluded example")

    candidates = [
        {
            "id": str(row["id"]),
            "relationship": relationship,
            "context": [str(value) for value in row.get("context") or []],
            "human_reply": [str(value) for value in row.get("reply") or []],
            "chat_id": str(row.get("chat_id") or ""),
            "historical_event_id": _historical_event_id(row),
            "same_source_chat": (
                str(row.get("chat_id") or "")
                == str(replay_case.get("source_chat_id") or "")
            ),
            "semantic_rank": rank,
        }
        for rank, row in enumerate(selected, start=1)
    ]
    audit = {
        "case_id": str(blind_case["case_id"]),
        "source_case_id": str(replay_case["case_id"]),
        "source_chat_id": str(replay_case.get("source_chat_id") or ""),
        "relationship": relationship,
        "query_message_count": len(query_messages),
        "requested_candidate_limit": retrieval_candidates_per_case,
        "retrieved_candidate_count": len(selected),
        "situation_candidate_count": situation_candidate_count,
        "supplemental_candidate_count": len(selected) - situation_candidate_count,
        "query_messages": query_messages,
        "excluded_id_count": len(excluded_ids),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "selected": [
            {
                "id": str(row["id"]),
                "pool_relationship": str(row.get("relationship") or ""),
                "chat_id": str(row.get("chat_id") or ""),
                "timestamp": row.get("timestamp"),
                "context": list(row.get("context") or []),
                "human_reply": list(row.get("reply") or []),
            }
            for row in selected
        ],
    }
    return candidates, audit


def build_option_divergence_prompt(
    relationship: str,
    timeline: list[dict[str, str]],
    option_a: list[str],
    option_b: list[str],
) -> str:
    packet = {
        "relationship": relationship,
        "current_timeline": timeline,
        "anonymous_options": {"A": option_a, "B": option_b},
    }
    actions = "、".join(sorted(OPTION_ACTIONS))
    return (
        "你只判断匿名候选 A/B 的主要接话行为，不判断哪条是真人，也不得猜测或输出正确答案。\n"
        "先看两条候选真正决定后续互动的分歧，例如立即回答/延后回答、接受/拒绝、"
        "确认/调侃。不要用话题、关键词、语气或措辞代替行为判断。候选包含多个动作时，"
        "选择对 A/B 分歧起决定作用的一个主要动作。\n"
        f"option_a_action 和 option_b_action 只能取：{actions}。"
        "has_behavior_divergence 只能为 true 或 false；仅当两侧行为不同且两侧都不是 other 时为 true。\n"
        "只输出一个 JSON 对象，不要 Markdown，格式必须是："
        '{"option_a_action":"answer_now","option_b_action":"defer",'
        '"has_behavior_divergence":true,"reason":"说明 A/B 的真正行为分歧"}。\n\n'
        f"输入：\n{json.dumps(packet, ensure_ascii=False, separators=(',', ':'))}"
    )


def parse_option_divergence_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"option divergence matcher returned invalid JSON: {exc}") from exc
    expected_fields = {
        "option_a_action", "option_b_action", "has_behavior_divergence", "reason"
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("option divergence response fields changed")
    if payload["option_a_action"] not in OPTION_ACTIONS:
        raise ValueError("option A action is unsupported")
    if payload["option_b_action"] not in OPTION_ACTIONS:
        raise ValueError("option B action is unsupported")
    if type(payload["has_behavior_divergence"]) is not bool:
        raise ValueError("has_behavior_divergence must be boolean")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("option divergence reason must be non-empty")
    eligible = (
        payload["option_a_action"] != payload["option_b_action"]
        and "other" not in {payload["option_a_action"], payload["option_b_action"]}
    )
    if payload["has_behavior_divergence"] != eligible:
        raise ValueError("has_behavior_divergence contradicts fixed option actions")
    return payload


def _run_codex_json_prompt(prompt: str, timeout_seconds: int, prefix: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        output_path = Path(temp_dir) / "last_message.json"
        command = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "read-only", "--cd", temp_dir, "--skip-git-repo-check",
            "--json", "--output-last-message", str(output_path), "-",
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        events = []
        parse_errors = []
        for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_number, "error": str(exc)})
        tool_call_count = sum(_event_has_tool_call(event) for event in events)
        if completed.returncode:
            raise RuntimeError(f"few-shot matcher failed: {completed.stderr[-2000:]}")
        if parse_errors or tool_call_count:
            raise ValueError("few-shot matcher isolation failed")
        raw = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    return {
        "raw": raw,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "returncode": completed.returncode,
        "elapsed_ms": elapsed_ms,
        "tool_call_count": tool_call_count,
        "event_parse_errors": parse_errors,
    }


def run_codex_option_divergence(
    relationship: str,
    timeline: list[dict[str, str]],
    option_a: list[str],
    option_b: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    prompt = build_option_divergence_prompt(
        relationship, timeline, option_a, option_b
    )
    run = _run_codex_json_prompt(
        prompt, timeout_seconds, "judge-option-divergence-"
    )
    return {
        **{key: value for key, value in run.items() if key != "raw"},
        "decision": parse_option_divergence_response(run["raw"]),
    }


def build_matcher_prompt(
    relationship: str,
    timeline: list[dict[str, str]],
    option_a_action: str,
    option_b_action: str,
    candidates: list[dict[str, Any]],
) -> str:
    packet = {
        "relationship": relationship,
        "current_timeline": timeline,
        "fixed_option_actions": {"A": option_a_action, "B": option_b_action},
        "historical_candidates": [
            {
                "candidate_id": row["id"],
                "relationship": row["relationship"],
                "context": row["context"],
                "human_reply": row["human_reply"],
            }
            for row in candidates
        ],
    }
    actions = "、".join(sorted(OPTION_ACTIONS))
    return (
        "你是历史真人案例的严格准入器。A/B 的主要行为标签已经由同一次独立判断固定，"
        "你不得重新判断、改写或推测 A/B，也不得判断哪条是真人。\n"
        "对每条历史案例输出三项 0、1、2 分：situation_match 判断是否处于同一种具体处境并"
        "面对同一种行为选择；event_stage_match 判断事件阶段是否一致；addressee_match 判断"
        "回复对象以及谁在对谁说是否一致。0=不一致或无证据，1=部分一致，2=明确一致。"
        "只有 situation_match=2 才可能准入；同聊天、同话题、关键词、措辞或表面语义相似"
        "都不能提高 situation_match。\n"
        f"historical_action 只能取：{actions}。它必须描述历史真人回复的主要行为。"
        "只有 historical_action 与固定 A 或 B 的行为标签完全相同，才可能成为该侧证据。"
        "is_generic_formula 只能为0或1；通用套话或不依赖具体处境的安全回答填1，只有0才可能准入。\n"
        "只输出一个 JSON 对象，不要 Markdown。格式必须是："
        '{"candidates":[{"candidate_id":"...","situation_match":0,'
        '"event_stage_match":0,"addressee_match":0,'
        '"historical_action":"answer_now","is_generic_formula":0,'
        '"reason":"说明处境、阶段、回复对象以及历史真人行为"}]}。'
        "candidate_id 必须原样返回，顺序与输入一致，不得遗漏或增加。\n\n"
        f"输入：\n{json.dumps(packet, ensure_ascii=False, separators=(',', ':'))}"
    )


def parse_matcher_response(raw: str, candidate_ids: list[str]) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"few-shot matcher returned invalid JSON: {exc}") from exc
    if set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("few-shot matcher response must contain only candidates")
    rows = payload["candidates"]
    returned_ids = [str(row.get("candidate_id") or "") for row in rows]
    if len(returned_ids) == len(candidate_ids) and set(returned_ids) != set(candidate_ids):
        unmatched_expected = set(candidate_ids) - set(returned_ids)
        unmatched_returned = set(returned_ids) - set(candidate_ids)
        repairs = {}
        for returned_id in unmatched_returned:
            matches = [
                expected_id
                for expected_id in unmatched_expected
                if min(len(returned_id), len(expected_id)) >= 12
                and 1 <= abs(len(returned_id) - len(expected_id)) <= 2
                and (
                    returned_id.startswith(expected_id)
                    or expected_id.startswith(returned_id)
                )
            ]
            if len(matches) != 1:
                repairs = {}
                break
            repairs[returned_id] = matches[0]
        if len(repairs) == len(unmatched_returned) and len(set(repairs.values())) == len(repairs):
            for row in rows:
                returned_id = str(row.get("candidate_id") or "")
                if returned_id in repairs:
                    row["candidate_id"] = repairs[returned_id]
            returned_ids = [str(row.get("candidate_id") or "") for row in rows]
    if len(returned_ids) != len(candidate_ids) or set(returned_ids) != set(candidate_ids):
        missing_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in returned_ids]
        unexpected_ids = [candidate_id for candidate_id in returned_ids if candidate_id not in candidate_ids]
        raise ValueError(
            "few-shot matcher candidate ids changed: "
            f"expected_count={len(candidate_ids)} returned_count={len(returned_ids)} "
            f"missing={missing_ids[:3]!r} unexpected={unexpected_ids[:3]!r}"
        )
    allowed_fields = {
        "candidate_id", *MATCH_SCORE_FIELDS, "historical_action",
        "is_generic_formula", "reason",
    }
    for row in rows:
        if set(row) != allowed_fields:
            raise ValueError("few-shot matcher candidate contains unsupported fields")
        if any(type(row[field]) is not int or not 0 <= row[field] <= 2 for field in MATCH_SCORE_FIELDS):
            raise ValueError("few-shot matcher scores must be integers from zero to two")
        unsupported_action = row["historical_action"] not in OPTION_ACTIONS
        if unsupported_action:
            unsupported_value = row["historical_action"]
            row["historical_action"] = "other"
            row["reason"] = (
                f"{row['reason']}；枚举外标签，严格拒绝：{unsupported_value}"
            )
        if type(row["is_generic_formula"]) is not int or row["is_generic_formula"] not in {0, 1}:
            raise ValueError("few-shot matcher is_generic_formula must be zero or one")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("few-shot matcher reason must be non-empty")
    return payload


def run_codex_matcher(
    relationship: str,
    timeline: list[dict[str, str]],
    option_a: list[str],
    option_b: list[str],
    candidates: list[dict[str, Any]],
    timeout_seconds: int,
    batch_size: int = DEFAULT_MATCHER_BATCH_SIZE,
    batch_workers: int = 1,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("matcher batch size must be positive")
    if batch_workers < 1:
        raise ValueError("matcher batch workers must be positive")
    candidate_ids = [str(row["id"]) for row in candidates]
    option_divergence = run_codex_option_divergence(
        relationship, timeline, option_a, option_b, timeout_seconds
    )
    option_decision = option_divergence["decision"]
    option_a_action = str(option_decision["option_a_action"])
    option_b_action = str(option_decision["option_b_action"])
    if not option_decision["has_behavior_divergence"]:
        rejected = [{
            "candidate_id": candidate_id,
            "situation_match": 0,
            "event_stage_match": 0,
            "addressee_match": 0,
            "historical_action": "other",
            "is_generic_formula": 1,
            "reason": "A/B 没有可用的固定行为分歧，未调用历史案例准入器",
            "supported_option": "none",
            "admitted": False,
        } for candidate_id in candidate_ids]
        return {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "transport": "codex_exec_ephemeral_behavior_then_batched_history",
            "matcher_batch_size": batch_size,
            "matcher_batch_workers": batch_workers,
            "batch_count": 0,
            "option_divergence_audit": option_divergence,
            "batch_audit": [],
            "prompt_sha256": _json_sha256([option_divergence["prompt_sha256"]]),
            "returncode": 0,
            "elapsed_ms": int(option_divergence["elapsed_ms"]),
            "tool_call_count": int(option_divergence["tool_call_count"]),
            "event_parse_errors": list(option_divergence["event_parse_errors"]),
            "decision": {
                **option_decision,
                "candidates": rejected,
            },
        }
    batch_specs = [
        (batch_index, candidates[start:start + batch_size])
        for batch_index, start in enumerate(
            range(0, len(candidates), batch_size), start=1
        )
    ]

    def run_batch(
        batch_index: int, batch_candidates: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        batch_candidate_ids = [str(row["id"]) for row in batch_candidates]
        prompt = build_matcher_prompt(
            relationship, timeline, option_a_action, option_b_action, batch_candidates
        )
        run = _run_codex_json_prompt(
            prompt, timeout_seconds, "judge-few-shot-matcher-"
        )
        batch_decision = parse_matcher_response(run["raw"], batch_candidate_ids)
        for row in batch_decision["candidates"]:
            if row["historical_action"] == option_a_action:
                supported_option = "A"
            elif row["historical_action"] == option_b_action:
                supported_option = "B"
            else:
                supported_option = "none"
            row["supported_option"] = supported_option
            row["admitted"] = (
                row["situation_match"] == MIN_SITUATION_MATCH
                and supported_option != "none"
                and row["is_generic_formula"] == 0
            )
        return batch_index, list(batch_decision["candidates"]), {
            "batch_index": batch_index,
            "candidate_count": len(batch_candidates),
            **{key: value for key, value in run.items() if key != "raw"},
        }

    if batch_workers == 1:
        batch_results = [run_batch(*spec) for spec in batch_specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(batch_workers, len(batch_specs))
        ) as executor:
            futures = [executor.submit(run_batch, *spec) for spec in batch_specs]
            batch_results = [future.result() for future in futures]
    batch_results.sort(key=lambda row: row[0])
    decisions_by_id = {
        str(row["candidate_id"]): row
        for _, decisions, _ in batch_results
        for row in decisions
    }
    batch_audit = [audit for _, _, audit in batch_results]
    if set(decisions_by_id) != set(candidate_ids):
        raise ValueError("batched few-shot matcher did not cover all candidates")
    decision = {
        **option_decision,
        "candidates": [decisions_by_id[candidate_id] for candidate_id in candidate_ids],
    }
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "codex_exec_ephemeral_behavior_then_batched_history",
        "matcher_batch_size": batch_size,
        "matcher_batch_workers": batch_workers,
        "batch_count": len(batch_audit),
        "option_divergence_audit": option_divergence,
        "batch_audit": batch_audit,
        "prompt_sha256": _json_sha256([
            option_divergence["prompt_sha256"],
            *[row["prompt_sha256"] for row in batch_audit],
        ]),
        "returncode": 0,
        "elapsed_ms": int(option_divergence["elapsed_ms"]) + sum(
            int(row["elapsed_ms"]) for row in batch_audit
        ),
        "tool_call_count": int(option_divergence["tool_call_count"]) + sum(
            int(row["tool_call_count"]) for row in batch_audit
        ),
        "event_parse_errors": list(option_divergence["event_parse_errors"]) + [
            {"batch_index": row["batch_index"], **error}
            for row in batch_audit
            for error in row["event_parse_errors"]
        ],
        "decision": decision,
    }


def admitted_examples(
    candidates: list[dict[str, Any]],
    matcher_result: dict[str, Any],
    limit: int = MAX_ADMITTED_EXAMPLES_PER_CASE,
) -> list[dict[str, Any]]:
    decisions = matcher_result["decision"]["candidates"]
    candidate_by_id = {str(row["id"]): row for row in candidates}
    admitted = [row for row in decisions if row["admitted"]]
    if len({str(row["supported_option"]) for row in admitted}) != 1:
        return []

    def row_rank(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
        candidate = candidate_by_id[str(row["candidate_id"])]
        return (
            -int(row["event_stage_match"]),
            -int(row["addressee_match"]),
            -int(bool(candidate.get("same_source_chat"))),
            int(candidate.get("semantic_rank") or 10**9),
            str(row["candidate_id"]),
        )

    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in admitted:
        candidate = candidate_by_id[str(row["candidate_id"])]
        rows_by_event.setdefault(_historical_event_id(candidate), []).append(row)
    if len(rows_by_event) < MIN_INDEPENDENT_EVENTS:
        return []
    representatives = sorted(
        (min(rows, key=row_rank) for rows in rows_by_event.values()),
        key=row_rank,
    )
    selected: list[dict[str, Any]] = []
    for representative in representatives[:limit]:
        candidate_id = str(representative["candidate_id"])
        candidate = candidate_by_id[candidate_id]
        selected.append({
            "id": candidate_id,
            "relationship": candidate["relationship"],
            "context": list(candidate["context"]),
            "human_reply": list(candidate["human_reply"]),
        })
    return selected


def build_manifest(
    blind: dict[str, Any],
    answers: dict[str, Any],
    replay: dict[str, Any],
    reference: dict[str, Any],
    pool_sha256: str,
    pool_rows_count: int,
    *,
    prepared: dict[str, dict[str, Any]],
    matcher_results: dict[str, dict[str, Any]],
    retrieval_candidates_per_case: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blind_sha = _json_sha256(blind)
    if answers.get("blind_sha256") != blind_sha:
        raise ValueError("answer mapping does not match blind packet")
    reference_sha = str(reference.get("reference_sha256") or "")
    if blind.get("reference_sha256") != reference_sha:
        raise ValueError("reference pack does not match blind packet")

    source_by_blind_id = {
        str(row["case_id"]): str(row["source_case_id"])
        for row in answers.get("cases") or []
    }
    replay_by_source_id = {str(row["case_id"]): row for row in replay.get("cases") or []}
    blind_ids = {str(row["case_id"]) for row in blind.get("cases") or []}
    if set(source_by_blind_id) != blind_ids:
        raise ValueError("answer mapping does not cover blind cases exactly once")
    if not set(source_by_blind_id.values()).issubset(replay_by_source_id):
        raise ValueError("answer mapping refers to missing replay cases")

    manifest_cases = []
    audit_cases = []
    for blind_case in blind.get("cases") or []:
        case_id = str(blind_case["case_id"])
        if case_id not in prepared:
            raise ValueError(f"missing prepared retrieval candidates for {case_id}")
        candidates = prepared[case_id]["candidates"]
        case_audit = copy.deepcopy(prepared[case_id]["retrieval_audit"])
        if case_id not in matcher_results:
            raise ValueError(f"missing few-shot matcher result for {case_id}")
        matcher_result = matcher_results[case_id]
        examples = admitted_examples(candidates, matcher_result)
        case_audit["matcher"] = matcher_result
        case_audit["admitted_example_ids"] = [row["id"] for row in examples]
        manifest_cases.append({
            "case_id": case_id,
            "relationship": str(blind_case["relationship"]),
            "examples": examples,
        })
        audit_cases.append(case_audit)

    manifest = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_manifest",
        "blind_sha256": blind_sha,
        "reference_sha256": reference_sha,
        "pool_sha256": pool_sha256,
        "cases": manifest_cases,
    }
    relationship_counts = Counter(row["relationship"] for row in manifest_cases)
    examples_per_case = Counter(len(row["examples"]) for row in manifest_cases)
    retrieved_candidates_per_case = Counter(
        len(row["candidates"]) for row in prepared.values()
    )
    selected_safety_counts: Counter[str] = Counter()
    blind_by_id = {str(row["case_id"]): row for row in blind.get("cases") or []}
    audit_by_id = {str(row["case_id"]): row for row in audit_cases}
    for case_id, source_case_id in source_by_blind_id.items():
        blind_case = blind_by_id[case_id]
        replay_case = replay_by_source_id[source_case_id]
        target_contexts = [
            [message["text"] for message in prepared[case_id]["timeline"]],
            list(replay_case.get("context_original") or []),
        ]
        option_pairs = {
            _normalized_messages(blind_case.get(option) or [])
            for option in ("option_A", "option_B")
        }
        for selected in audit_by_id[case_id]["selected"]:
            same_source_chat = (
                str(selected.get("chat_id") or "")
                == str(replay_case.get("source_chat_id") or "")
            )
            selected_safety_counts["same_source_chat_selected"] += same_source_chat
            selected_safety_counts["overlapping_context_selected"] += (
                same_source_chat
                and _contexts_overlap(
                    selected.get("context") or [],
                    selected.get("human_reply") or [],
                    target_contexts,
                )
            )
            selected_safety_counts["exact_option_reply_selected"] += (
                _normalized_messages(selected.get("human_reply") or []) in option_pairs
            )
    audit = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_retrieval_audit",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "blind_sha256": blind_sha,
        "reference_sha256": reference_sha,
        "pool_sha256": pool_sha256,
        "configuration": {
            "pool_rows": pool_rows_count,
            "query_latest_messages": 8,
            "retrieval_candidates_per_case": retrieval_candidates_per_case,
            "admitted_examples_per_case": f"0_to_{MAX_ADMITTED_EXAMPLES_PER_CASE}",
            "matcher_version": MATCHER_VERSION,
            "option_action_labels": sorted(OPTION_ACTIONS),
            "option_behavior_divergence_required": True,
            "option_behavior_divergence_calls_per_case": 1,
            "match_score_fields": list(MATCH_SCORE_FIELDS),
            "situation_match_required": MIN_SITUATION_MATCH,
            "historical_action_rule": "strictly_equals_fixed_option_a_or_b_action",
            "is_generic_formula_required": 0,
            "minimum_independent_events": MIN_INDEPENDENT_EVENTS,
            "historical_event_identity": (
                "explicit_historical_event_id_else_unique_pool_candidate_id"
            ),
            "same_historical_event_slices_count_as": 1,
            "opposite_support": "delete_all_case_dynamic_examples",
            "ranking": [
                "event_stage_match_desc",
                "addressee_match_desc",
                "same_source_chat_desc",
                "semantic_rank_asc",
                "candidate_id_asc",
            ],
            "maximum_admitted_examples_per_case": MAX_ADMITTED_EXAMPLES_PER_CASE,
            "prioritize_latest_turn": True,
            "enable_precise_situation": True,
            "prefer_response_move_before_relevance": True,
            "prefer_latest_turn_form": True,
            "enable_information_gap_response_move": True,
        },
        "summary": {
            "cases": len(manifest_cases),
            "private_cases": relationship_counts["private"],
            "group_cases": relationship_counts["group"],
            "examples": sum(len(row["examples"]) for row in manifest_cases),
            "matched_cases": sum(bool(row["examples"]) for row in manifest_cases),
            "unmatched_cases": sum(not row["examples"] for row in manifest_cases),
            "matched_private_cases": sum(
                row["relationship"] == "private" and bool(row["examples"])
                for row in manifest_cases
            ),
            "matched_group_cases": sum(
                row["relationship"] == "group" and bool(row["examples"])
                for row in manifest_cases
            ),
            "examples_per_case_distribution": {
                str(count): cases for count, cases in sorted(examples_per_case.items())
            },
            "retrieved_candidates_per_case_distribution": {
                str(count): cases
                for count, cases in sorted(retrieved_candidates_per_case.items())
            },
            "same_source_chat_selected": selected_safety_counts["same_source_chat_selected"],
            "overlapping_context_selected": selected_safety_counts[
                "overlapping_context_selected"
            ],
            "exact_option_reply_selected": selected_safety_counts[
                "exact_option_reply_selected"
            ],
        },
        "cases": audit_cases,
    }
    return manifest, audit


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Judge 每题动态历史真人 few-shot")
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--retrieval-candidate-audit", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--matcher-timeout-seconds", type=int, default=180)
    parser.add_argument("--matcher-batch-size", type=int, default=DEFAULT_MATCHER_BATCH_SIZE)
    parser.add_argument("--matcher-batch-workers", type=int, default=1)
    parser.add_argument(
        "--retrieval-candidates-per-case",
        type=int,
        default=DEFAULT_RETRIEVAL_CANDIDATES_PER_CASE,
    )
    args = parser.parse_args()

    blind = _load_json(args.blind)
    answers = _load_json(args.answers)
    replay = _load_json(args.replay)
    reference = _load_json(args.reference)
    pool_bytes = args.pool.read_bytes()
    pool_sha256 = hashlib.sha256(pool_bytes).hexdigest()
    pool_rows = _load_pool(args.pool)
    pool_by_id = {str(row["id"]): row for row in pool_rows}
    blind_sha = _json_sha256(blind)
    reference_sha = str(reference.get("reference_sha256") or "")
    source_by_blind_id = {
        str(row["case_id"]): str(row["source_case_id"])
        for row in answers.get("cases") or []
    }
    replay_by_source_id = {str(row["case_id"]): row for row in replay.get("cases") or []}
    if args.retrieval_candidates_per_case < 1:
        parser.error("--retrieval-candidates-per-case must be positive")
    if args.matcher_batch_size < 1:
        parser.error("--matcher-batch-size must be positive")
    if args.matcher_batch_workers < 1:
        parser.error("--matcher-batch-workers must be positive")
    prior_candidate_count = 0
    reuse_prior_matcher = False
    if args.retrieval_candidate_audit is not None:
        retrieval_audit = _load_json(args.retrieval_candidate_audit)
        prior_prepared = prepare_from_retrieval_audit(
            retrieval_audit,
            blind=blind,
            answers=answers,
            replay=replay,
            reference=reference,
            pool_sha256=pool_sha256,
            pool_by_id=pool_by_id,
        )
        pool_rows_count = int(retrieval_audit.get("configuration", {}).get("pool_rows") or 0)
        if pool_rows_count <= 0:
            raise ValueError("retrieval candidate audit is missing pool row count")
        prior_candidate_count = int(
            retrieval_audit.get("configuration", {}).get("retrieval_candidates_per_case")
            or len(next(iter(prior_prepared.values()))["candidates"])
        )
        if prior_candidate_count > args.retrieval_candidates_per_case:
            raise ValueError("retrieval candidate audit contains more candidates than requested")
        prior_rows_by_id = {
            str(row["case_id"]): row for row in retrieval_audit.get("cases") or []
        }
        reuse_prior_matcher = (
            str(retrieval_audit.get("configuration", {}).get("matcher_version") or "")
            == MATCHER_VERSION
            and prior_candidate_count == args.retrieval_candidates_per_case
        )
        if reuse_prior_matcher:
            prior_matcher_results: dict[str, dict[str, Any]] = {}
            for case_id, row in prior_prepared.items():
                matcher = prior_rows_by_id[case_id].get("matcher")
                expected_ids = [str(candidate["id"]) for candidate in row["candidates"]]
                decisions = list((matcher or {}).get("decision", {}).get("candidates") or [])
                if [str(decision.get("candidate_id") or "") for decision in decisions] != expected_ids:
                    raise ValueError(f"prior matcher result does not cover frozen candidates: {case_id}")
                prior_matcher_results[case_id] = matcher
        if prior_candidate_count == args.retrieval_candidates_per_case:
            prepared = prior_prepared
        else:
            retriever = PersonaFewShotRetriever(
                args.pool,
                prioritize_latest_turn=True,
                enable_precise_situation=True,
                prefer_response_move_before_relevance=True,
                prefer_latest_turn_form=True,
                enable_information_gap_response_move=True,
            )
            if not retriever.is_approved():
                raise ValueError("historical few-shot pool is not approved or its integrity check failed")
            prepared = {}
            for blind_case in blind.get("cases") or []:
                case_id = str(blind_case["case_id"])
                replay_case = replay_by_source_id[source_by_blind_id[case_id]]
                candidates, case_audit = retrieve_case_candidates(
                    retriever,
                    pool_rows,
                    blind_case,
                    replay_case,
                    args.retrieval_candidates_per_case,
                    diversity_prefix_size=prior_candidate_count,
                    minimum_candidates_per_case=prior_candidate_count,
                )
                prior_candidates = prior_prepared[case_id]["candidates"]
                if candidates[:prior_candidate_count] != prior_candidates:
                    raise ValueError(f"expanded retrieval changed frozen candidate prefix: {case_id}")
                prepared[case_id] = {
                    "relationship": str(blind_case["relationship"]),
                    "timeline": extract_timeline(list(blind_case.get("context_original") or [])),
                    "option_A": [str(value) for value in blind_case.get("option_A") or []],
                    "option_B": [str(value) for value in blind_case.get("option_B") or []],
                    "candidates": candidates,
                    "retrieval_audit": case_audit,
                }
    else:
        retriever = PersonaFewShotRetriever(
            args.pool,
            prioritize_latest_turn=True,
            enable_precise_situation=True,
            prefer_response_move_before_relevance=True,
            prefer_latest_turn_form=True,
            enable_information_gap_response_move=True,
        )
        if not retriever.is_approved():
            raise ValueError("historical few-shot pool is not approved or its integrity check failed")
        pool_rows_count = len(pool_rows)
        prepared = {}
        for blind_case in blind.get("cases") or []:
            case_id = str(blind_case["case_id"])
            replay_case = replay_by_source_id[source_by_blind_id[case_id]]
            candidates, case_audit = retrieve_case_candidates(
                retriever,
                pool_rows,
                blind_case,
                replay_case,
                args.retrieval_candidates_per_case,
                diversity_prefix_size=min(3, args.retrieval_candidates_per_case),
            )
            prepared[case_id] = {
                "relationship": str(blind_case["relationship"]),
                "timeline": extract_timeline(list(blind_case.get("context_original") or [])),
                "option_A": [str(value) for value in blind_case.get("option_A") or []],
                "option_B": [str(value) for value in blind_case.get("option_B") or []],
                "candidates": candidates,
                "retrieval_audit": case_audit,
            }
    matcher_input_sha256 = _json_sha256({
        "matcher_version": MATCHER_VERSION,
        "matcher_batch_size": args.matcher_batch_size,
        "blind_sha256": blind_sha,
        "reference_sha256": reference_sha,
        "pool_sha256": pool_sha256,
        "cases": {
            case_id: {
                "relationship": row["relationship"],
                "timeline": row["timeline"],
                "option_A": row["option_A"],
                "option_B": row["option_B"],
                "candidates": row["candidates"],
            }
            for case_id, row in prepared.items()
        },
    })
    matcher_results: dict[str, dict[str, Any]] = {}
    if args.audit.exists():
        prior_audit = _load_json(args.audit)
        if prior_audit.get("matcher_input_sha256") != matcher_input_sha256:
            raise ValueError("existing matcher audit does not match current inputs")
        matcher_results = {
            str(row["case_id"]): row["matcher"]
            for row in prior_audit.get("cases") or []
            if row.get("matcher")
        }
    elif reuse_prior_matcher and prior_candidate_count == args.retrieval_candidates_per_case:
        matcher_results = dict(prior_matcher_results)

    pending = [case_id for case_id in prepared if case_id not in matcher_results]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_codex_matcher,
                prepared[case_id]["relationship"],
                prepared[case_id]["timeline"],
                prepared[case_id]["option_A"],
                prepared[case_id]["option_B"],
                prepared[case_id]["candidates"],
                args.matcher_timeout_seconds,
                args.matcher_batch_size,
                args.matcher_batch_workers,
            ): case_id
            for case_id in pending
        }
        for future in concurrent.futures.as_completed(futures):
            case_id = futures[future]
            try:
                matcher_results[case_id] = future.result()
            except Exception as exc:
                print(f"matcher case failed; leaving it pending for resume: {case_id}: {exc}", file=sys.stderr)
                continue
            checkpoint = {
                "schema_version": 2,
                "kind": "judge_dynamic_few_shot_retrieval_audit_checkpoint",
                "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
                "matcher_input_sha256": matcher_input_sha256,
                "cases": [
                    {"case_id": row_id, "matcher": matcher_results[row_id]}
                    for row_id in prepared if row_id in matcher_results
                ],
            }
            _write_private(args.audit, checkpoint)
    if len(matcher_results) != len(prepared):
        raise RuntimeError(
            f"few-shot matcher incomplete: {len(matcher_results)}/{len(prepared)}; rerun to resume"
        )

    manifest, audit = build_manifest(
        blind,
        answers,
        replay,
        reference,
        pool_sha256,
        pool_rows_count,
        prepared=prepared,
        matcher_results=matcher_results,
        retrieval_candidates_per_case=args.retrieval_candidates_per_case,
    )
    audit["matcher_input_sha256"] = matcher_input_sha256
    _write_private(args.manifest, manifest)
    _write_private(args.audit, audit)
    print(json.dumps(audit["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
