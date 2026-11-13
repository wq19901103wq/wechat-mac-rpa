#!/usr/bin/env python3
"""Build and run the private group-chat Judge review.

This benchmark intentionally reconstructs historical unread windows.  It is a
Judge diagnostic, not a production replay or a Bot quality measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_persona_few_shots import Candidate, extract_candidates
from scripts.evaluate_human_ai_judge import (
    CUTOFF,
    PROJECT_ROOT,
    PROTECTED_C0_SHA256,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _assert_c0,
    _case_id,
    _export_payloads,
    _few_shot_identities,
    _json_sha256,
    _normalized_reply,
    _previously_exposed_case_ids,
    _run_c0,
    _write_private,
    detect_degenerate_case,
    extract_judge_chat_context,
    load_volcengine_credentials,
)
from scripts.run_human_ai_judge_benchmark_v2 import build_messages, load_profile, profile_sha256
from scripts.run_kimi_human_ai_judge_v2 import _run_codex
from scripts.evaluate_human_ai_judge import _parse_api_judge_response


PRIVATE_DIR = PROJECT_ROOT / "data/reports/private/group_chat_judge_review"
MANIFEST_PATH = PRIVATE_DIR / "manifest.json"
CHECKPOINT_PATH = PRIVATE_DIR / "generation_checkpoint.json"
BLIND_PATH = PRIVATE_DIR / "blind.json"
ANSWERS_PATH = PRIVATE_DIR / "answers.json"
PREDICTIONS_PATH = PRIVATE_DIR / "predictions.json"
AUDIT_PATH = PRIVATE_DIR / "audit.json"
RESULT_PATH = PRIVATE_DIR / "result.json"
HTML_PATH = PRIVATE_DIR / "index.html"
BENCHMARK_V2_DIR = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
BALANCED160_DIR = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v3_balanced160"
BALANCED160_BLIND_PATH = BALANCED160_DIR / "dev_blind.json"
BALANCED160_ANSWERS_PATH = BALANCED160_DIR / "dev_answers.json"
BALANCED160_BASELINE_RESULT_PATH = BALANCED160_DIR / "judge_candidate_single_rule_full_timeline_v1_result.json"
BALANCED160_BASELINE_AUDIT_PATH = BALANCED160_DIR / "judge_candidate_single_rule_full_timeline_v1_audit.json"
BALANCED160_CANDIDATE_PROFILE_PATH = BALANCED160_DIR / "judge_profile_two_rules_chat_irregularity_v2.json"
BALANCED160_CANDIDATE_RESULT_PATH = BALANCED160_DIR / "judge_candidate_two_rules_chat_irregularity_v2_result.json"
BALANCED160_CANDIDATE_AUDIT_PATH = BALANCED160_DIR / "judge_candidate_two_rules_chat_irregularity_v2_audit.json"
CHAT_IRREGULARITY_RULE = (
    "不得仅因某个选项有错字、漏字、残缺时间格式、语法不完整或口语不规范，就判它更像 AI；"
    "即时聊天中的真人原话可以包含这些表达。只有当该表达无法在当前时间线中形成任何可解释含义时，"
    "才把不通顺作为反证。"
)
REFERENCE_PATH = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_reference.json"
PROFILE_PATH = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v2_thread_attribution.json"
BOUNDARY_PROFILE_PATH = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v2_boundary_candidate.json"
BOUNDARY_PREDICTIONS_PATH = PRIVATE_DIR / "boundary_predictions.json"
BOUNDARY_AUDIT_PATH = PRIVATE_DIR / "boundary_audit.json"
BOUNDARY_RESULT_PATH = PRIVATE_DIR / "boundary_result.json"
HOLDOUT_SELECTION_PATH = PRIVATE_DIR / "holdout_selection.json"
HOLDOUT_BLIND_PATH = PRIVATE_DIR / "holdout_blind.json"
HOLDOUT_ANSWERS_PATH = PRIVATE_DIR / "holdout_answers.json"
HOLDOUT_V2_PREDICTIONS_PATH = PRIVATE_DIR / "holdout_v2_predictions.json"
HOLDOUT_V2_AUDIT_PATH = PRIVATE_DIR / "holdout_v2_audit.json"
HOLDOUT_V2_RESULT_PATH = PRIVATE_DIR / "holdout_v2_result.json"
HOLDOUT_CANDIDATE_PROFILE_PATH = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v2_group_echo_candidate.json"
HOLDOUT_CANDIDATE_PREDICTIONS_PATH = PRIVATE_DIR / "holdout_candidate_predictions.json"
HOLDOUT_CANDIDATE_AUDIT_PATH = PRIVATE_DIR / "holdout_candidate_audit.json"
HOLDOUT_CANDIDATE_RESULT_PATH = PRIVATE_DIR / "holdout_candidate_result.json"
COMMON_EXPORT_DIR = PROJECT_ROOT / "data/exports/b"
YUZU_EXPORT_DIR = PROJECT_ROOT / "data/exports/main"
WIKI_DIR = PROJECT_ROOT / "data/memory/wiki/users"
VOLCENGINE_CONFIG = Path.home() / ".cc-connect/config.toml"
COMMON_CHAT_ID = "chat_26"
COMMON_SOURCE_PATH = PROJECT_ROOT / "data/chats.bak.20260703_230700/共同富裕群.json"
CASES_DB_PATH = PROJECT_ROOT / "data/cases.db"
TARGET_GROUP_ROOM_ID = "_chats_共同富裕群_f5e4ec66"
WITHDRAWN_COMMON_CHAT_ID = "chat_de96f4621f"
YUZU_CHAT_ID = "chat_923597a52e"
COMMON_TARGET = 10
YUZU_TARGET = 20
COMMON_POOL_SIZE = 10
YUZU_POOL_SIZE = 40
SEED = "group-chat-judge-review-v1-20260818"
DISCLAIMER = "历史近似，非生产复现"
GENERATION_DISCLAIMER = "C0 逻辑近似，非生产 C0"
DEFAULT_OFFLINE_BUDGET_MULTIPLIER = 3.0
HOLDOUT_POOL_SIZE = 40
HOLDOUT_TARGET = 12
TARGET_GROUP_SELF_EVIDENCE = (
    {
        "text": "你还加老王的", "observed_at": "2026-07-03 16:05:47", "tick_id": 1783039898,
        "bot_candidate": "本轮生成 0 条回复", "send_status": "未进入发送",
    },
    {
        "text": "老王", "observed_at": "2026-07-03 16:06:07", "tick_id": 1783039899,
        "bot_candidate": "那是西西给老王修图修得好呀。", "send_status": "静默模式跳过发送",
    },
    {
        "text": "年轻时候的老王", "observed_at": "2026-07-03 16:06:25", "tick_id": 1783039900,
        "bot_candidate": "游戏数值策划，在杭州恺英网络上班呢。", "send_status": "静默模式跳过发送",
    },
    {
        "text": "老王不止30", "observed_at": "2026-07-03 16:07:08", "tick_id": 1783039902,
        "bot_candidate": "确实，我印象里老王今年38吧。", "send_status": "静默模式跳过发送",
    },
    {
        "text": "太小瞧老王的", "observed_at": "2026-07-03 16:09:00", "tick_id": 1783039908,
        "bot_candidate": "老王亏的钱够我买半套外滩玺了吧。", "send_status": "静默模式跳过发送",
    },
    {
        "text": "什么时候我也能赚这么多", "observed_at": "2026-07-03 16:17:47", "tick_id": 1783039963,
        "bot_candidate": "等降薪了直接负收入，倒贴钱上班。", "send_status": "静默模式跳过发送",
    },
    {
        "text": "你现在已经和科长混这么熟了", "observed_at": "2026-07-03 16:44:37", "tick_id": 1783040216,
        "bot_candidate": "本轮生成 0 条回复", "send_status": "未进入发送",
    },
    {
        "text": "你不是刚入职吗", "observed_at": "2026-07-03 16:44:37", "tick_id": 1783040216,
        "bot_candidate": "本轮生成 0 条回复", "send_status": "未进入发送",
    },
    {
        "text": "太可怕了", "observed_at": "2026-07-03 19:51:28", "tick_id": 1783041631,
        "bot_candidate": "合着我也就赚个1000多万的水平呗。", "send_status": "静默模式跳过发送",
    },
    {
        "text": "江波龙封装的", "observed_at": "2026-07-03 19:56:42", "tick_id": 1783041655,
        "bot_candidate": "无未回复消息，本轮未生成", "send_status": "未进入发送",
    },
    {
        "text": "买入有巨额现金流的公司", "observed_at": "2026-07-03 21:53:08", "tick_id": 1783086021,
        "bot_candidate": "无未回复消息，本轮未生成", "send_status": "未进入发送",
    },
    {
        "text": "pdd", "observed_at": "2026-07-03 21:53:08", "tick_id": 1783086021,
        "bot_candidate": "无未回复消息，本轮未生成", "send_status": "未进入发送",
    },
)
BOT_BAD_REPLY_MARKERS = (
    "138一个人急头白脸",
    "人均36带饮料",
    "低工资高消费",
    "吃席了 羡慕",
    "扶贫席还是正经席",
    "500万在北京",
    "130万欢乐豆",
    "复读机石锤",
    "复读机三号实锤",
    "罚站怕是要站到收盘",
    "罚站的人 没资格发红包",
    "月供3.6 少一分你补",
)


class _ScaledTime:
    def __init__(self, real_time: Any, multiplier: float):
        self._real_time = real_time
        self._multiplier = multiplier
        self._real_started = real_time.time()
        self._virtual_started = self._real_started

    def time(self) -> float:
        elapsed = self._real_time.time() - self._real_started
        return self._virtual_started + elapsed / self._multiplier

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_time, name)


class _ScaledTimeoutClient:
    def __init__(self, client: Any, multiplier: float):
        self._client = client
        self._multiplier = multiplier

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        if timeout is not None:
            kwargs["timeout"] = timeout * self._multiplier
        return self._client.chat(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _run_c0_logic_approx(
    candidate: Candidate,
    payload: dict[str, Any],
    client: Any,
    memory: Any,
    budget_multiplier: float,
) -> dict[str, Any]:
    import src.reply.generator as generator_module

    original_time = generator_module.time
    generator_module.time = _ScaledTime(original_time, budget_multiplier)
    try:
        return _run_c0(
            candidate,
            payload,
            _ScaledTimeoutClient(client, budget_multiplier),
            memory,
        )
    finally:
        generator_module.time = original_time


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _timestamp_seconds(value: int | float) -> float:
    return value / (1000 if value > 10_000_000_000 else 1)


def _record(candidate: Candidate, group_name: str, scored: bool) -> dict[str, Any]:
    row = asdict(candidate)
    row["style_tags"] = list(candidate.style_tags)
    row.update({
        "source_case_id": _case_id(candidate),
        "group_name": group_name,
        "scored": scored,
        "replay_type": "historical_approximation",
        "production_reproduction": False,
        "disclaimer": DISCLAIMER,
    })
    if not scored:
        row["replies"] = []
        row["source_provenance"] = "timestamp_anchor_only_not_human_ground_truth"
    return row


def _candidate(row: dict[str, Any]) -> Candidate:
    fields = set(Candidate.__dataclass_fields__)
    values = {key: value for key, value in row.items() if key in fields}
    values["style_tags"] = tuple(values.get("style_tags") or ())
    return Candidate(**values)


def _round_robin_by_month(candidates: list[Candidate], target: int) -> list[Candidate]:
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        seconds = candidate.timestamp / (1000 if candidate.timestamp > 10_000_000_000 else 1)
        month = datetime.fromtimestamp(seconds).strftime("%Y-%m")
        buckets[month].append(candidate)
    for month in buckets:
        buckets[month].sort(key=lambda row: _rank(_case_id(row)))
    months = sorted(buckets, key=lambda value: _rank(f"month:{value}"))
    selected: list[Candidate] = []
    depth = 0
    while len(selected) < target:
        before = len(selected)
        for month in months:
            if depth < len(buckets[month]):
                selected.append(buckets[month][depth])
                if len(selected) == target:
                    break
        if len(selected) == before:
            break
        depth += 1
    return selected


def _target_group_export_payload() -> dict[str, Any]:
    inventory = load_target_group_inventory()
    return {
        "session": {"displayName": inventory["chat_name"], "wxid": inventory["chat_id"]},
        "messages": [
            {
                "createTime": row.get("create_time"),
                "isSend": row.get("sender_type") == "self",
                "localType": 1 if row.get("message_type") == "text" else 3,
                "content": str(row.get("text") or row.get("image_description") or ""),
                "senderDisplayName": row.get("sender"),
                "senderUsername": row.get("sender_wxid") or "",
                "localId": row.get("local_id"),
                "serverId": row.get("server_id"),
            }
            for row in inventory["messages"]
        ],
    }


def _target_group_candidates() -> list[Candidate]:
    inventory = load_target_group_inventory()
    messages = inventory["messages"]
    evidence_texts = [row["text"] for row in TARGET_GROUP_SELF_EVIDENCE]
    observed_texts = [
        str(row.get("text") or "").strip()
        for row in messages
        if row.get("sender_type") == "self"
    ]
    if Counter(observed_texts) != Counter(evidence_texts):
        raise ValueError("target-group replies do not match verified identity evidence")

    candidates: list[Candidate] = []
    index = 0
    while index < len(messages):
        row = messages[index]
        if row.get("sender_type") != "self":
            index += 1
            continue
        replies = [str(row.get("text") or "").strip()]
        next_index = index + 1
        while next_index < len(messages) and messages[next_index].get("sender_type") == "self":
            replies.append(str(messages[next_index].get("text") or "").strip())
            next_index += 1
        context_start = max(0, index - 12)
        context = []
        for message in messages[context_start:index]:
            text = str(message.get("text") or message.get("image_description") or "").strip()
            if text:
                context.append(f"{message.get('sender') or '未知'}：{text}")
        candidates.append(Candidate(
            relation="group",
            chat_id=COMMON_CHAT_ID,
            context=context,
            replies=replies,
            score=1,
            timestamp=int(row.get("create_time") or 0),
            reply_shape="multi" if len(replies) > 1 else "single",
            source_provenance="project_logs_exclude_current_bot_path",
        ))
        index = next_index
    if len(candidates) != COMMON_TARGET:
        raise ValueError(f"target group has {len(candidates)} verified reply turns, expected {COMMON_TARGET}")
    return candidates


def build_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return load_manifest()
    excluded = _few_shot_identities(PROJECT_ROOT / "data")
    exposed = _previously_exposed_case_ids(PROJECT_ROOT / "data/reports/private")
    common_selected = _target_group_candidates()

    # These post-cutoff self turns are timestamps only. Their replies are removed
    # immediately and can never enter the answer key or accuracy calculation.
    yuzu_all = extract_candidates(YUZU_EXPORT_DIR, WIKI_DIR, human_before=9_999_999_999)
    yuzu = [
        row for row in yuzu_all
        if row.chat_id == YUZU_CHAT_ID and _timestamp_seconds(row.timestamp) >= CUTOFF
    ]
    yuzu.sort(key=lambda row: _rank(_case_id(row)))
    yuzu_selected = yuzu[:YUZU_POOL_SIZE]
    if len(yuzu_selected) < YUZU_TARGET:
        raise RuntimeError(f"柚子群只有 {len(yuzu_selected)} 个压力锚点，需要至少 {YUZU_TARGET} 个")

    manifest = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "group_chat_judge_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "disclaimer": DISCLAIMER,
        "validates": "Judge behavior only; not Bot production quality",
        "production_reproduction": False,
        "approved_difference": "Unread windows are reconstructed from historical exports after the previous self message.",
        "c0_generator_sha256": _assert_c0(),
        "targets": {"scored_common_prosperity": COMMON_TARGET, "unscored_yuzu_pressure": YUZU_TARGET},
        "candidate_pool": {
            "common_prosperity": len(common_selected),
            "yuzu_pressure": len(yuzu_selected),
            "purpose": "允许跳过生成失败、沉默或无效样本，直到达到目标数量",
        },
        "selection": {
            "few_shot_rows_excluded": len(excluded),
            "previously_exposed_case_ids_excluded": len(exposed),
            "eligible_common_prosperity": len(common_selected),
            "verified_target_group_messages": 12,
            "verified_target_group_reply_turns": len(common_selected),
            "target_group_time_boundary": "post-2026-04; Judge diagnostic only",
            "yuzu_ground_truth_available": False,
        },
        "cases": [
            *[_record(row, "共同富裕群", True) for row in common_selected],
            *[_record(row, "柚子群2", False) for row in yuzu_selected],
        ],
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    scored = [row for row in manifest["cases"] if row.get("scored")]
    if {row.get("chat_id") for row in scored} != {COMMON_CHAT_ID}:
        raise ValueError("scored cases must come only from the target chat_26 group")
    if any(row.get("chat_id") == WITHDRAWN_COMMON_CHAT_ID for row in manifest["cases"]):
        raise ValueError("withdrawn same-name group entered the manifest")
    if Counter(reply for row in scored for reply in row.get("replies", [])) != Counter(
        evidence["text"] for evidence in TARGET_GROUP_SELF_EVIDENCE
    ):
        raise ValueError("manifest answers do not match verified target-group replies")
    _write_private(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    unhashed = dict(manifest)
    expected = unhashed.pop("manifest_sha256")
    if _json_sha256(unhashed) != expected:
        raise ValueError("manifest hash mismatch")
    if manifest.get("c0_generator_sha256") != PROTECTED_C0_SHA256:
        raise ValueError("manifest is not bound to protected C0")
    scored_chat_ids = {row.get("chat_id") for row in manifest.get("cases", []) if row.get("scored")}
    if scored_chat_ids and scored_chat_ids != {COMMON_CHAT_ID}:
        raise ValueError("manifest belongs to the withdrawn same-name group")
    return manifest


def generate(
    max_calls: int | None = None,
    offline_budget_multiplier: float = DEFAULT_OFFLINE_BUDGET_MULTIPLIER,
    source_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    from src.memory import MemoryEngine

    manifest = load_manifest()
    _assert_c0()
    checkpoint = (
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if CHECKPOINT_PATH.exists() else {
            "schema_version": 1,
            "manifest_sha256": manifest["manifest_sha256"],
            "c0_generator_sha256": PROTECTED_C0_SHA256,
            "entries": {},
        }
    )
    if checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("generation checkpoint belongs to another manifest")
    api_key, base_url = load_volcengine_credentials(VOLCENGINE_CONFIG)
    client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    if offline_budget_multiplier < 1:
        raise ValueError("offline budget multiplier must be at least 1")
    scaled_client = _ScaledTimeoutClient(client, offline_budget_multiplier)
    memory = MemoryEngine(llm_client=scaled_client)
    payloads = {**_export_payloads(COMMON_EXPORT_DIR), **_export_payloads(YUZU_EXPORT_DIR)}
    payloads[COMMON_CHAT_ID] = _target_group_export_payload()
    calls = 0
    try:
        for row in manifest["cases"]:
            if source_case_ids is not None and row["source_case_id"] not in source_case_ids:
                continue
            group_target = COMMON_TARGET if row["scored"] else YUZU_TARGET
            group_valid = sum(
                entry.get("status") == "valid" and bool(entry.get("scored")) == bool(row["scored"])
                for entry in checkpoint["entries"].values()
            )
            if group_valid >= group_target:
                continue
            source_case_id = row["source_case_id"]
            if source_case_id in checkpoint["entries"]:
                continue
            if max_calls is not None and calls >= max_calls:
                break
            calls += 1
            candidate = _candidate(row)
            try:
                output = _run_c0_logic_approx(
                    candidate,
                    payloads[candidate.chat_id],
                    client,
                    memory,
                    offline_budget_multiplier,
                )
                reasons = detect_degenerate_case(candidate.replies, output) if row["scored"] else []
                status = "valid"
                if output["generation_failed"]:
                    status = "generation_failed"
                elif not output["replies"]:
                    status = "silent"
                elif reasons:
                    status = "degenerate"
                checkpoint["entries"][source_case_id] = {
                    "status": status,
                    "reasons": reasons,
                    "source_case_id": source_case_id,
                    "group_name": row["group_name"],
                    "scored": row["scored"],
                    "timestamp": row["timestamp"],
                    "human_reply_original": row["replies"] if row["scored"] else None,
                    "ai_reply_original": output["replies"],
                    "context_original": output["context_original"],
                    "c0_trace": output["trace"],
                    "input_metadata": output["input_metadata"],
                    "duration_seconds": output["duration_seconds"],
                    "disclaimer": DISCLAIMER,
                    "generation_disclaimer": GENERATION_DISCLAIMER,
                    "offline_budget_multiplier": offline_budget_multiplier,
                }
            except Exception as exc:
                checkpoint["entries"][source_case_id] = {
                    "status": "error", "source_case_id": source_case_id,
                    "group_name": row["group_name"], "scored": row["scored"],
                    "error": f"{type(exc).__name__}: {exc}", "disclaimer": DISCLAIMER,
                    "generation_disclaimer": GENERATION_DISCLAIMER,
                    "offline_budget_multiplier": offline_budget_multiplier,
                }
            _write_private(CHECKPOINT_PATH, checkpoint)
    finally:
        memory.shutdown()
    _assert_c0()
    return {
        "calls": calls,
        "outcomes": dict(Counter(row["status"] for row in checkpoint["entries"].values())),
        "generation_disclaimer": GENERATION_DISCLAIMER,
        "offline_budget_multiplier": offline_budget_multiplier,
    }


def freeze_holdout_selection() -> dict[str, Any]:
    """Freeze untouched source IDs before any of their generated answers are inspected."""
    if HOLDOUT_SELECTION_PATH.exists():
        selection = json.loads(HOLDOUT_SELECTION_PATH.read_text(encoding="utf-8"))
        unhashed = dict(selection)
        expected = unhashed.pop("selection_sha256")
        if _json_sha256(unhashed) != expected:
            raise ValueError("holdout selection hash mismatch")
        return selection
    manifest = load_manifest()
    checkpoint = (
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if CHECKPOINT_PATH.exists() else {"entries": {}}
    )
    attempted = set(checkpoint.get("entries", {}))
    eligible = [
        row["source_case_id"] for row in manifest["cases"]
        if row["scored"] and row["source_case_id"] not in attempted
    ]
    selected = sorted(eligible, key=lambda value: _rank(f"untouched-holdout:{value}"))[:HOLDOUT_POOL_SIZE]
    if len(selected) < HOLDOUT_TARGET:
        raise RuntimeError(f"未尝试的共同富裕群候选不足：{len(selected)}/{HOLDOUT_TARGET}")
    selection = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "deterministic rank over previously unattempted scored manifest rows",
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_profile_sha256": profile_sha256(load_profile(HOLDOUT_CANDIDATE_PROFILE_PATH)),
        "attempted_source_case_ids_at_freeze": sorted(attempted),
        "pool_size": len(selected),
        "target_valid_cases": HOLDOUT_TARGET,
        "source_case_ids": selected,
    }
    selection["selection_sha256"] = _json_sha256(selection)
    _write_private(HOLDOUT_SELECTION_PATH, selection)
    return selection


def generate_holdout(
    max_calls: int | None = None,
    offline_budget_multiplier: float = DEFAULT_OFFLINE_BUDGET_MULTIPLIER,
) -> dict[str, Any]:
    selection = freeze_holdout_selection()
    return generate(max_calls, offline_budget_multiplier, set(selection["source_case_ids"]))


def freeze(
    allow_partial: bool = False,
    *,
    source_case_ids: set[str] | None = None,
    blind_path: Path = BLIND_PATH,
    answers_path: Path = ANSWERS_PATH,
) -> dict[str, Any]:
    manifest = load_manifest()
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    valid = [row for row in checkpoint["entries"].values() if row["status"] == "valid"]
    scored = [
        row for row in valid
        if row["scored"] and (source_case_ids is None or row["source_case_id"] in source_case_ids)
    ]
    pressure = [row for row in valid if not row["scored"]]
    scored_target = HOLDOUT_TARGET if source_case_ids is not None else COMMON_TARGET
    if not allow_partial and (len(scored) < scored_target or (source_case_ids is None and len(pressure) < YUZU_TARGET)):
        raise RuntimeError(f"有效结果不足：计分 {len(scored)}/{scored_target}")
    if allow_partial and not scored:
        raise RuntimeError("没有可用的计分样本")
    scored = sorted(scored, key=lambda row: _rank(row["source_case_id"]))[:scored_target]
    pressure = sorted(pressure, key=lambda row: _rank(row["source_case_id"]))[:YUZU_TARGET]
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    target_replies = {_normalized_reply(row["human_reply_original"]) for row in scored}
    for example in reference.get("examples", []):
        if _normalized_reply(example.get("human_reply") or []) in target_replies:
            raise ValueError("Judge reference pack contains an exact target reply")

    blind_cases, answer_cases = [], []
    for row in scored:
        public_id = "group_case_" + _rank(row["source_case_id"])[:16]
        human_option = "A" if int(_rank(f"position:{row['source_case_id']}")[-1], 16) % 2 == 0 else "B"
        ai_option = "B" if human_option == "A" else "A"
        context = extract_judge_chat_context(str(row["c0_trace"].get("user_prompt") or ""))
        if not context:
            context = row["context_original"]
        blind_cases.append({
            "case_id": public_id,
            "relationship": "group",
            "replay_type": "historical_approximation",
            "production_reproduction": False,
            "disclaimer": DISCLAIMER,
            "context_original": context,
            f"option_{human_option}": row["human_reply_original"],
            f"option_{ai_option}": row["ai_reply_original"],
        })
        answer_cases.append({"case_id": public_id, "source_case_id": row["source_case_id"], "human_option": human_option})
    blind = {
        "schema_version": 1, "benchmark_kind": "group_chat_human_ai_blind",
        "disclaimer": DISCLAIMER, "reference_sha256": reference["reference_sha256"],
        "partial_review": allow_partial, "cases": blind_cases,
    }
    answers = {
        "schema_version": 1, "benchmark_kind": "private_answer_key", "blind_sha256": _json_sha256(blind),
        "cases": answer_cases,
    }
    _write_private(blind_path, blind)
    _write_private(answers_path, answers)
    if source_case_ids is None:
        _write_private(PRIVATE_DIR / "pressure.json", {
            "schema_version": 1, "disclaimer": DISCLAIMER, "scored": False,
            "note": "柚子群2压力场景，不计分，无真人标准答案", "cases": pressure,
        })
    return {
        "scored": len(scored), "pressure": len(pressure),
        "partial_review": allow_partial, "blind_sha256": answers["blind_sha256"],
    }


def freeze_holdout() -> dict[str, Any]:
    selection = freeze_holdout_selection()
    return freeze(
        source_case_ids=set(selection["source_case_ids"]),
        blind_path=HOLDOUT_BLIND_PATH,
        answers_path=HOLDOUT_ANSWERS_PATH,
    )


def run_judge(
    max_cases: int | None = None,
    *,
    profile_path: Path = PROFILE_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    audit_path: Path = AUDIT_PATH,
    blind_path: Path = BLIND_PATH,
) -> dict[str, Any]:
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    profile = load_profile(profile_path)
    identity = {
        "blind_sha256": _json_sha256(blind), "reference_sha256": reference["reference_sha256"],
        "profile_sha256": profile_sha256(profile), "provider": "codex-cli", "model": "codex-cli-default",
    }
    predictions = json.loads(predictions_path.read_text()) if predictions_path.exists() else {"schema_version": 1, **identity, "cases": []}
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {
        "schema_version": 1, **identity, "profile": profile,
        "isolation": {"one_fresh_process_per_case": True, "shared_context": False, "answer_key_loaded": False, "tool_use_invalidates_case": True},
        "calls": [],
    }
    for key, value in identity.items():
        if predictions.get(key) != value or audit.get(key) != value:
            raise ValueError(f"resume identity mismatch: {key}")
    completed = {row["case_id"] for row in predictions["cases"]}
    all_ids = [row["case_id"] for row in blind["cases"]]
    calls = 0
    for case in blind["cases"]:
        if case["case_id"] in completed:
            continue
        if max_cases is not None and calls >= max_cases:
            break
        messages = build_messages(case, reference, profile)
        serialized = json.dumps(messages, ensure_ascii=False)
        forbidden = ("source_case_id", "human_reply_original", "ai_reply_original", "correct")
        if any(marker in serialized for marker in forbidden):
            raise ValueError("Judge request contains answer/source identity fields")
        if any(marker in case for marker in ("human_option", "actual", "correct")):
            raise ValueError("Blind case contains answer fields")
        if any(other in serialized for other in all_ids if other != case["case_id"]):
            raise ValueError("Judge request contains another test case")
        started_at = datetime.now(timezone.utc).isoformat()
        raw, response, elapsed_ms, tool_events, error = "", {}, 0, [], None
        try:
            raw, response, elapsed_ms, tool_events = _run_codex(messages)
            parsed = _parse_api_judge_response(raw)
            if tool_events:
                raise ValueError("Judge used a tool")
        except Exception as exc:
            parsed = {"human_option": None, "confidence": None, "reason": "invalid response"}
            error = f"{type(exc).__name__}: {exc}"
        predictions["cases"].append({"case_id": case["case_id"], **parsed, "invalid": parsed.get("human_option") not in {"A", "B"}})
        audit["calls"].append({
            "case_id": case["case_id"], "recorded_at_utc": started_at, "request_messages": messages,
            "request_sha256": _json_sha256(messages), "answer_fields_in_request": [],
            "other_test_case_ids_in_request": [], "tools_configured": True, "tool_events": tool_events,
            "elapsed_ms": elapsed_ms, "raw_response": raw, "response": response, "error": error,
        })
        _write_private(predictions_path, predictions)
        _write_private(audit_path, audit)
        calls += 1
    return {"calls": calls, "completed": len(predictions["cases"]), "total": len(blind["cases"])}


def score(
    *,
    predictions_path: Path = PREDICTIONS_PATH,
    audit_path: Path = AUDIT_PATH,
    result_path: Path = RESULT_PATH,
    blind_path: Path = BLIND_PATH,
    answers_path: Path = ANSWERS_PATH,
) -> dict[str, Any]:
    blind = json.loads(blind_path.read_text())
    answers = {row["case_id"]: row for row in json.loads(answers_path.read_text())["cases"]}
    predictions = {row["case_id"]: row for row in json.loads(predictions_path.read_text())["cases"]}
    audit = json.loads(audit_path.read_text())
    if len(audit["calls"]) != len(blind["cases"]):
        raise RuntimeError("Judge run is incomplete")
    rows = []
    for case in blind["cases"]:
        case_id = case["case_id"]
        prediction = predictions[case_id]
        answer = answers[case_id]
        call = next(row for row in audit["calls"] if row["case_id"] == case_id)
        isolation_ok = not call["tool_events"] and not call["answer_fields_in_request"] and not call["other_test_case_ids_in_request"]
        rows.append({
            "case_id": case_id, "source_case_id": answer["source_case_id"], "actual": answer["human_option"],
            "predicted": prediction.get("human_option"), "correct": prediction.get("human_option") == answer["human_option"],
            "confidence": prediction.get("confidence"), "reason": prediction.get("reason"), "isolation_ok": isolation_ok,
        })
    correct = sum(row["correct"] for row in rows)
    result = {
        "schema_version": 1, "disclaimer": DISCLAIMER, "validates": "Judge behavior only; not Bot production quality",
        "answered": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None,
        "human_position": dict(Counter(answers[row["case_id"]]["human_option"] for row in blind["cases"])),
        "isolation_verified": all(row["isolation_ok"] for row in rows), "cases": rows,
    }
    _write_private(result_path, result)
    return result


def _pre(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f"<pre>{html.escape(text)}</pre>"


def _reply_view(value: Any) -> str:
    replies = value if isinstance(value, list) else [value]
    replies = [str(reply).strip() for reply in replies if reply is not None and str(reply).strip()]
    if not replies:
        return "<div class='reply-empty'>（没有回复）</div>"
    return "<div class='reply-view'>" + "".join(
        f"<div class='reply-line'>{html.escape(reply)}</div>" for reply in replies
    ) + "</div>"


def _context_view(value: Any) -> str:
    chunks = value if isinstance(value, list) else [value]
    messages: list[tuple[str, str, str]] = []
    for chunk in chunks:
        raw = str(chunk or "")
        matches = re.findall(r'<message\b([^>]*)>(.*?)</message>', raw, flags=re.DOTALL | re.IGNORECASE)
        if not matches:
            text = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
            if text:
                messages.append(("other", "聊天记录", text))
            continue
        for attrs, body in matches:
            role_match = re.search(r'\brole=["\']([^"\']+)["\']', attrs, flags=re.IGNORECASE)
            role = role_match.group(1).lower() if role_match else "other"
            text = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
            if not text:
                text = "[非文字消息]"
            speaker, content = ("我", text)
            split = re.match(r"^(.*?)：([\s\S]*)$", text)
            if split:
                speaker, content = split.group(1).strip(), split.group(2).strip() or "[非文字消息]"
            messages.append((role, speaker, content))
    if not messages:
        return "<div class='chat-empty'>（没有聊天上下文）</div>"
    return "<div class='chat-view'>" + "".join(
        f"<div class='chat-row {'self' if role == 'self' else 'other'}'>"
        f"<div class='chat-speaker'>{html.escape(speaker)}</div>"
        f"<div class='chat-bubble'>{html.escape(content)}</div></div>"
        for role, speaker, content in messages
    ) + "</div>"


def _labeled_reply(option: str, value: Any, identity: str) -> str:
    option_label = f"回复 {option}" if option else "回复"
    return (
        "<div class='reply-card'>"
        f"<div class='reply-title'><b>{html.escape(option_label)}</b><span>{html.escape(identity)}</span></div>"
        f"{_reply_view(value)}</div>"
    )


def _fixed_validation_v5_html() -> str:
    blind = json.loads((BENCHMARK_V2_DIR / "validation_blind.json").read_text(encoding="utf-8"))["cases"]
    answers = {
        row["case_id"]: row["human_option"]
        for row in json.loads((BENCHMARK_V2_DIR / "validation_answers.json").read_text(encoding="utf-8"))["cases"]
    }
    predictions: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for part in range(3):
        prediction_path = BENCHMARK_V2_DIR / f"validation_v5_codex_cli_canonical_p{part}.json"
        audit_path = BENCHMARK_V2_DIR / f"validation_v5_codex_cli_canonical_p{part}_audit.json"
        predictions.update({row["case_id"]: row for row in json.loads(prediction_path.read_text(encoding="utf-8"))["cases"]})
        audits.update({row["case_id"]: row for row in json.loads(audit_path.read_text(encoding="utf-8"))["calls"]})

    cards = []
    for number, case in enumerate(blind, 1):
        case_id = case["case_id"]
        answer = answers[case_id]
        prediction = predictions[case_id]
        audit = audits[case_id]
        correct = prediction.get("human_option") == answer
        option_a_identity = "真人原回复" if answer == "A" else "匿名对照回复"
        option_b_identity = "真人原回复" if answer == "B" else "匿名对照回复"
        cards.append(f"""
        <details class='card case validation-case {'correct' if correct else 'incorrect'}' data-correct='{'1' if correct else '0'}'>
          <summary><b>固定验证题 {number} · {html.escape(str(case.get('relationship') or ''))}</b><span>{'判断正确' if correct else '判断错误'}</span></summary>
          <h3>完整聊天上下文</h3>{_context_view(case.get('context_original', []))}
          <div class='options'>{_labeled_reply('A', case.get('option_A', []), option_a_identity)}{_labeled_reply('B', case.get('option_B', []), option_b_identity)}</div>
          <section class='judge-result'><p><b>Judge 选择：</b>{html.escape(str(prediction.get('human_option') or ''))}　<b>真人位置：</b>{html.escape(answer)}　<b>置信度：</b>{html.escape(str(prediction.get('confidence', '')))}</p><p><b>理由：</b>{html.escape(str(prediction.get('reason') or ''))}</p></section>
          <details><summary>查看 Judge 原始提示词与调用记录</summary><h3>完整请求消息</h3>{_pre(audit.get('request_messages', []))}<h3>隔离字段与调用参数</h3>{_pre({key: audit.get(key) for key in ('recorded_at_utc','request_sha256','answer_fields_in_request','other_test_case_ids_in_request','tools_configured','tool_events','elapsed_ms','error')})}<h3>模型原始输出</h3>{_pre(audit.get('raw_response', ''))}</details>
        </details>""")
    return "".join(cards)


def _balanced_group_comparison() -> tuple[str, str]:
    base = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
    blind = json.loads((base / "final_blind.json").read_text(encoding="utf-8"))
    answers = {
        row["case_id"]: row["human_option"]
        for row in json.loads((base / "final_answers.json").read_text(encoding="utf-8"))["cases"]
    }
    run_files = {
        "V2": "final_v2_codex_cli_canonical_comparison.json",
        "V3": "final_v3_codex_cli_canonical_comparison.json",
        "边界候选": "final_v2_boundary_candidate_codex_cli_canonical_comparison.json",
    }
    audit_files = {
        label: filename.replace(".json", "_audit.json") for label, filename in run_files.items()
    }
    runs = {
        label: {row["case_id"]: row for row in json.loads((base / filename).read_text(encoding="utf-8"))["cases"]}
        for label, filename in run_files.items()
    }
    audits = {
        label: {row["case_id"]: row for row in json.loads((base / filename).read_text(encoding="utf-8"))["calls"]}
        for label, filename in audit_files.items()
    }
    cases = [row for row in blind["cases"] if row.get("relationship") == "group"]
    positions = Counter(answers[row["case_id"]] for row in cases)
    scores = {
        label: sum(runs[label][row["case_id"]]["human_option"] == answers[row["case_id"]] for row in cases)
        for label in runs
    }
    summary = (
        f"旧冻结集群聊子集：{len(cases)} 题，真人位置 A={positions['A']}、B={positions['B']}。"
        + "　".join(f"{label} {scores[label]}/{len(cases)}" for label in runs)
        + "。该集合已参与迭代，只用于诊断，不是新隐藏终测。"
    )
    cards = []
    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        actual = answers[case_id]
        human = case[f"option_{actual}"]
        ai = case[f"option_{'B' if actual == 'A' else 'A'}"]
        decisions = []
        raw_calls = []
        for label in runs:
            prediction = runs[label][case_id]
            correct = prediction.get("human_option") == actual
            decisions.append(
                f"<p><b>{label}：</b>{html.escape(str(prediction.get('human_option')))} "
                f"（{'正确' if correct else '误判'}，置信度 {html.escape(str(prediction.get('confidence'))) }）<br>"
                f"{html.escape(str(prediction.get('reason') or ''))}</p>"
            )
            call = audits[label][case_id]
            raw_calls.append(
                f"<h3>{label} 请求与原始输出</h3>{_pre({'request_messages': call.get('request_messages'), 'tool_events': call.get('tool_events'), 'error': call.get('error'), 'raw_response': call.get('raw_response')})}"
            )
        cards.append(f"""
        <section class='card case'>
          <div class='head'><b>旧冻结集群聊 · 题 {index}</b><span>诊断集，非隐藏终测</span><em>真人在 {actual}</em></div>
          <h3>完整聊天上下文</h3>{_context_view(case.get('context_original', []))}
          <div class='options'>{_labeled_reply('A', case.get('option_A', []), '真人原回复' if actual == 'A' else 'C0 原回复')}{_labeled_reply('B', case.get('option_B', []), '真人原回复' if actual == 'B' else 'C0 原回复')}</div>
          {''.join(decisions)}
          <details><summary>查看原始提示词、上下文与调用记录</summary><h3>原始上下文数据</h3>{_pre(case.get('context_original', []))}{''.join(raw_calls)}</details>
        </section>""")
    return summary, "".join(cards)


def _boundary_tradeoffs() -> tuple[str, str]:
    base = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
    blind = {
        row["case_id"]: row
        for row in json.loads((base / "final_blind.json").read_text(encoding="utf-8"))["cases"]
    }
    answers = {
        row["case_id"]: row["human_option"]
        for row in json.loads((base / "final_answers.json").read_text(encoding="utf-8"))["cases"]
    }
    paths = {
        "V3": base / "final_v3_codex_cli_canonical_comparison.json",
        "边界候选": base / "final_v2_boundary_candidate_codex_cli_canonical_comparison.json",
    }
    audit_paths = {label: path.with_name(path.stem + "_audit.json") for label, path in paths.items()}
    runs = {
        label: {row["case_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))["cases"]}
        for label, path in paths.items()
    }
    audits = {
        label: {row["case_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))["calls"]}
        for label, path in audit_paths.items()
    }
    changed = [
        case_id for case_id in blind
        if runs["V3"][case_id].get("human_option") != runs["边界候选"][case_id].get("human_option")
    ]
    fixes = [case_id for case_id in changed if runs["边界候选"][case_id].get("human_option") == answers[case_id]]
    regressions = [case_id for case_id in changed if runs["V3"][case_id].get("human_option") == answers[case_id]]
    v3_score = sum(runs["V3"][case_id].get("human_option") == answers[case_id] for case_id in blind)
    boundary_score = sum(runs["边界候选"][case_id].get("human_option") == answers[case_id] for case_id in blind)
    summary = (
        f"旧 100 题整体：V3 {v3_score}/100，边界候选 {boundary_score}/100。"
        f"两版共有 {len(changed)} 题发生翻转，其中修正 {len(fixes)} 题、新增误判 {len(regressions)} 题。"
        "这说明偏差发生了迁移，不能把边界候选升级为通用 Judge。"
    )
    cards = []
    ordered = [("修正", case_id) for case_id in fixes] + [("退步", case_id) for case_id in regressions]
    for index, (change_type, case_id) in enumerate(ordered, 1):
        case = blind[case_id]
        actual = answers[case_id]
        human = case[f"option_{actual}"]
        ai = case[f"option_{'B' if actual == 'A' else 'A'}"]
        decisions = []
        raw_calls = []
        for label in ("V3", "边界候选"):
            prediction = runs[label][case_id]
            correct = prediction.get("human_option") == actual
            decisions.append(
                f"<p><b>{label}：</b>{html.escape(str(prediction.get('human_option')))} "
                f"（{'正确' if correct else '误判'}，置信度 {html.escape(str(prediction.get('confidence'))) }）<br>"
                f"{html.escape(str(prediction.get('reason') or ''))}</p>"
            )
            call = audits[label][case_id]
            raw_calls.append(
                f"<h3>{label} 请求与原始输出</h3>"
                f"{_pre({'request_messages': call.get('request_messages'), 'tool_events': call.get('tool_events'), 'error': call.get('error'), 'raw_response': call.get('raw_response')})}"
            )
        cards.append(f"""
        <section class='card case {'ok' if change_type == '修正' else 'wrong'}'>
          <div class='head'><b>翻转题 {index} · {change_type}</b><span>旧题诊断，非隐藏终测</span><em>真人在 {actual}</em></div>
          <h3>完整聊天上下文</h3>{_context_view(case.get('context_original', []))}
          <div class='options'>{_labeled_reply('A', case.get('option_A', []), '真人原回复' if actual == 'A' else 'C0 原回复')}{_labeled_reply('B', case.get('option_B', []), '真人原回复' if actual == 'B' else 'C0 原回复')}</div>
          {''.join(decisions)}
          <details><summary>查看原始提示词、上下文与调用记录</summary><h3>原始上下文数据</h3>{_pre(case.get('context_original', []))}{''.join(raw_calls)}</details>
        </section>""")
    return summary, "".join(cards)


def _render_withdrawn_html() -> Path:
    manifest = load_manifest()
    checkpoint = json.loads(CHECKPOINT_PATH.read_text()) if CHECKPOINT_PATH.exists() else {"entries": {}}
    blind = json.loads(BLIND_PATH.read_text()) if BLIND_PATH.exists() else {"cases": []}
    answers = {row["case_id"]: row for row in json.loads(ANSWERS_PATH.read_text()).get("cases", [])} if ANSWERS_PATH.exists() else {}
    predictions = {row["case_id"]: row for row in json.loads(PREDICTIONS_PATH.read_text()).get("cases", [])} if PREDICTIONS_PATH.exists() else {}
    calls = {row["case_id"]: row for row in json.loads(AUDIT_PATH.read_text()).get("calls", [])} if AUDIT_PATH.exists() else {}
    result = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else {}
    boundary_predictions = {
        row["case_id"]: row
        for row in json.loads(BOUNDARY_PREDICTIONS_PATH.read_text()).get("cases", [])
    } if BOUNDARY_PREDICTIONS_PATH.exists() else {}
    boundary_calls = {
        row["case_id"]: row
        for row in json.loads(BOUNDARY_AUDIT_PATH.read_text()).get("calls", [])
    } if BOUNDARY_AUDIT_PATH.exists() else {}
    boundary_result = json.loads(BOUNDARY_RESULT_PATH.read_text()) if BOUNDARY_RESULT_PATH.exists() else {}
    balanced_summary, balanced_cards = _balanced_group_comparison()
    tradeoff_summary, tradeoff_cards = _boundary_tradeoffs()
    source_entries = checkpoint.get("entries", {})
    generation_counts = Counter(row.get("status", "unknown") for row in source_entries.values())
    generation_cards = []
    for index, source in enumerate(source_entries.values(), 1):
        generation_cards.append(f"""
        <section class='card case'>
          <div class='head'><b>{html.escape(str(source.get('group_name', '')))} · 生成 {index}</b><span>{GENERATION_DISCLAIMER}</span><em>{html.escape(str(source.get('status', 'unknown')))}</em></div>
          <p><b>实际耗时：</b>{html.escape(str(source.get('duration_seconds', '')))} 秒　<b>预算倍率：</b>{html.escape(str(source.get('offline_budget_multiplier', '生产原预算')))}</p>
          <h3>重建上下文</h3>{_context_view(source.get('context_original', []))}
          <div class='options'>{_labeled_reply('', source.get('human_reply_original'), '真人原回复')}{_labeled_reply('', source.get('ai_reply_original', []), 'C0 逻辑近似回复')}</div>
          <details><summary>查看 C0 原始提示词、上下文与完整调用痕迹</summary><h3>原始上下文数据</h3>{_pre(source.get('context_original', []))}{_pre(source.get('c0_trace', {}))}</details>
          {f"<details><summary>查看错误</summary>{_pre(source.get('error'))}</details>" if source.get('error') else ''}
        </section>""")
    scored_cards = []
    for index, case in enumerate(blind.get("cases", []), 1):
        answer = answers.get(case["case_id"], {})
        prediction = predictions.get(case["case_id"], {})
        boundary_prediction = boundary_predictions.get(case["case_id"], {})
        call = calls.get(case["case_id"], {})
        boundary_call = boundary_calls.get(case["case_id"], {})
        source = source_entries.get(answer.get("source_case_id"), {})
        actual, predicted = answer.get("human_option"), prediction.get("human_option")
        human = case.get(f"option_{actual}", []) if actual else []
        ai = case.get(f"option_{'B' if actual == 'A' else 'A'}", []) if actual else []
        correctness = "正确" if actual and predicted == actual else "误判"
        scored_cards.append(f"""
        <section class='card case {'ok' if correctness == '正确' else 'wrong'}'>
          <div class='head'><b>共同富裕群 · 题 {index}</b><span>{DISCLAIMER}</span><em>{correctness}</em></div>
          <h3>完整重建上下文</h3>{_context_view(case.get('context_original', []))}
          <div class='options'>{_labeled_reply('A', case.get('option_A', []), '真人原回复' if actual == 'A' else 'C0 原回复')}{_labeled_reply('B', case.get('option_B', []), '真人原回复' if actual == 'B' else 'C0 原回复')}</div>
          <p><b>Judge 选择：</b>{html.escape(str(predicted or '未完成'))}　<b>正确答案：</b>{html.escape(str(actual or ''))}　<b>置信度：</b>{html.escape(str(prediction.get('confidence', '')))}<br><b>理由：</b>{html.escape(str(prediction.get('reason', '')))}</p>
          <p><b>边界候选：</b>{html.escape(str(boundary_prediction.get('human_option') or '未完成'))}　<b>置信度：</b>{html.escape(str(boundary_prediction.get('confidence', '')))}<br><b>理由：</b>{html.escape(str(boundary_prediction.get('reason', '')))}</p>
          <details><summary>查看 C0 原始提示词、上下文与调用痕迹</summary><h3>原始上下文数据</h3>{_pre(case.get('context_original', []))}<h3>System prompt</h3>{_pre(source.get('c0_trace', {}).get('system_prompt', ''))}<h3>User prompt</h3>{_pre(source.get('c0_trace', {}).get('user_prompt', ''))}<h3>Few-shot</h3>{_pre(source.get('c0_trace', {}).get('few_shot_content', ''))}<h3>工具上下文与调用</h3>{_pre({'tools_context': source.get('c0_trace', {}).get('tools_context'), 'tool_calls': source.get('c0_trace', {}).get('tool_calls'), 'llm_calls': source.get('c0_trace', {}).get('llm_calls'), 'input_metadata': source.get('input_metadata')})}</details>
          <details><summary>查看 Judge 原始提示词与完整调用记录</summary><h3>请求消息</h3>{_pre(call.get('request_messages', []))}<h3>调用参数与隔离检查</h3>{_pre({key: call.get(key) for key in ('recorded_at_utc','request_sha256','answer_fields_in_request','other_test_case_ids_in_request','tools_configured','tool_events','elapsed_ms','error')})}<h3>原始回复</h3>{_pre(call.get('raw_response', ''))}<h3>命令、stdout、stderr、事件</h3>{_pre(call.get('response', {}))}</details>
          <details><summary>查看边界候选原始提示词与完整调用记录</summary><h3>请求消息</h3>{_pre(boundary_call.get('request_messages', []))}<h3>调用参数与隔离检查</h3>{_pre({key: boundary_call.get(key) for key in ('recorded_at_utc','request_sha256','answer_fields_in_request','other_test_case_ids_in_request','tools_configured','tool_events','elapsed_ms','error')})}<h3>原始回复</h3>{_pre(boundary_call.get('raw_response', ''))}<h3>命令、stdout、stderr、事件</h3>{_pre(boundary_call.get('response', {}))}</details>
        </section>""")
    pressure_cards = []
    pressure = json.loads((PRIVATE_DIR / "pressure.json").read_text()) if (PRIVATE_DIR / "pressure.json").exists() else {"cases": []}
    for index, source in enumerate(pressure.get("cases", []), 1):
        context = extract_judge_chat_context(str(source.get("c0_trace", {}).get("user_prompt") or "")) or source.get("context_original", [])
        pressure_cards.append(f"""
        <section class='card case pressure'><div class='head'><b>柚子群2 · 压力场景 {index}</b><span>{DISCLAIMER}</span><em>不计分</em></div>
          <p class='note'>无可验证的 2026 年 4 月前本人回复，因此没有真人标准答案，也不进入 Judge 准确率。</p>
          <h3>完整重建上下文</h3>{_context_view(context)}<h3>C0 原回复</h3>{_reply_view(source.get('ai_reply_original', []))}
          <details><summary>查看 C0 原始提示词、上下文与调用痕迹</summary><h3>原始上下文数据</h3>{_pre(context)}{_pre(source.get('c0_trace', {}))}</details>
        </section>""")
    accuracy = result.get("accuracy")
    score_text = f"{result.get('correct')}/{result.get('answered')}（{accuracy:.1%}）" if isinstance(accuracy, float) else "尚未完成"
    partial_note = "临时小样本审阅，非正式结论。" if blind.get("partial_review") else ""
    boundary_accuracy = boundary_result.get("accuracy")
    boundary_score_text = f"{boundary_result.get('correct')}/{boundary_result.get('answered')}（{boundary_accuracy:.1%}）" if isinstance(boundary_accuracy, float) else "尚未完成"
    conclusion = (
        f"当前 5 题：原 Judge {score_text}，边界候选 {boundary_score_text}；同一题仍然误判，只是置信度由 0.74 降到 0.60。"
        f"旧 13 个位置均衡群聊题：V2/V3 均为 12/13，边界候选为 13/13，但这些题参与过迭代。"
        f"旧 100 题整体：V3 89/100，边界候选 88/100。结论：只有群聊诊断信号，没有稳定的通用提升。"
    )
    body = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'><title>群聊 Judge 审计</title><style>
    .judge-page{{max-width:920px;margin:auto;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55;min-width:0}}.judge-page *{{box-sizing:border-box}}.judge-page h1{{font-size:22px;margin:4px 0 14px}}.judge-page h2{{font-size:18px;margin:18px 0 10px}}.judge-page h3{{font-size:13px;color:#8b949e;margin:16px 0 6px}}.judge-page p{{overflow-wrap:anywhere}}.judge-page .card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px;margin:12px 0}}.judge-page .summary{{border-left:5px solid #d29922}}.judge-page .summary h2{{margin-top:0}}.judge-page .decision{{display:inline-flex;align-items:center;gap:6px;background:#3d1f25;color:#ffb3ba;border:1px solid #f85149;border-radius:99px;padding:5px 10px;font-weight:700}}.judge-page .scores{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:12px 0}}.judge-page .score{{background:#0d1117;border:1px solid #30363d;border-radius:9px;padding:10px}}.judge-page .score b{{display:block;color:#f0f6fc;font-size:16px}}.judge-page .score span{{font-size:12px;color:#8b949e}}.judge-page .head{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.judge-page .head span,.judge-page .head em{{font-size:12px;border:1px solid #d29922;border-radius:99px;padding:2px 7px;font-style:normal}}.judge-page .head em{{margin-left:auto;border-color:#58a6ff}}.judge-page .wrong{{border-left:4px solid #f85149}}.judge-page .ok{{border-left:4px solid #3fb950}}.judge-page .pressure{{border-left:4px solid #d29922}}.judge-page .options{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}}.judge-page .chat-view{{display:flex;flex-direction:column;gap:8px;background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:10px;min-width:0}}.judge-page .chat-row{{display:flex;flex-direction:column;align-items:flex-start;max-width:88%}}.judge-page .chat-row.self{{align-self:flex-end;align-items:flex-end}}.judge-page .chat-speaker{{font-size:11px;color:#8b949e;margin:0 6px 3px}}.judge-page .chat-bubble{{background:#21262d;border-radius:4px 12px 12px 12px;padding:7px 10px;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}}.judge-page .chat-row.self .chat-bubble{{background:#164e2e;border-radius:12px 4px 12px 12px}}.judge-page .reply-card{{background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;min-width:0}}.judge-page .reply-title{{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-bottom:1px solid #30363d}}.judge-page .reply-title span{{font-size:12px;color:#58a6ff}}.judge-page .reply-view{{padding:10px;display:flex;flex-direction:column;gap:7px}}.judge-page .reply-line{{background:#21262d;border-radius:8px;padding:8px 10px;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}}.judge-page .reply-empty,.judge-page .chat-empty{{padding:10px;color:#8b949e}}.judge-page pre{{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;background:#0d1117;border:1px solid #30363d;border-radius:7px;padding:10px;font-size:12px;line-height:1.55;max-height:34rem;overflow:auto}}.judge-page summary{{cursor:pointer;color:#58a6ff;margin:8px 0;padding:4px 0}}.judge-page details[open]{{border-top:1px solid #30363d;margin-top:12px}}.judge-page .note{{color:#d29922}}.judge-page .controls{{position:sticky;top:0;z-index:10;width:auto;background:rgba(13,17,23,.96);backdrop-filter:blur(8px);border:1px solid #30363d;border-radius:12px;padding:10px;margin:14px 0}}.judge-page .control-row{{display:flex;align-items:center;gap:8px}}.judge-page .control-row button{{appearance:none;border:1px solid #58a6ff;background:#161b22;color:#f0f6fc;border-radius:8px;padding:9px 13px;font-size:14px}}.judge-page .control-row button:disabled{{opacity:.35}}.judge-page .progress{{flex:1;text-align:center;font-size:13px;color:#8b949e}}.judge-page .filter{{display:flex;align-items:center;gap:7px;margin-top:9px;font-size:14px}}.judge-page .dataset{{display:none}}.judge-page .dataset.active{{display:block}}.judge-page .case{{display:none}}.judge-page .case.active{{display:block}}.judge-page .empty{{display:none;text-align:center;padding:30px 12px;color:#8b949e}}.judge-page .technical-audit{{margin:16px 0}}.judge-page .technical-audit>.case{{display:block}}.judge-page .technical-audit details>.case{{display:block}}.judge-page .technical-audit details>summary{{font-weight:700}}@media(max-width:700px){{.judge-page{{padding-bottom:calc(18px + env(safe-area-inset-bottom))}}.judge-page h1{{font-size:20px}}.judge-page .summary{{padding:12px}}.judge-page .summary .long-conclusion{{font-size:14px}}.judge-page .scores{{grid-template-columns:1fr 1fr}}.judge-page .score:last-child{{grid-column:1/-1}}.judge-page .options{{grid-template-columns:1fr}}.judge-page .head em{{margin-left:0}}.judge-page pre{{font-size:12px;max-height:none}}.judge-page .controls{{top:6px}}.judge-page .control-row button{{min-width:74px;padding:9px 10px}}.judge-page .progress{{font-size:12px}}.judge-page details pre{{max-height:60vh}}}}
    </style></head><body><div class='judge-page'><div class='judge-header'><h1>群聊 Judge 审计</h1><section class='card summary'><h2>当前结论</h2><p class='decision'>候选版不升级</p><p class='long-conclusion'>{html.escape(conclusion)}</p><div class='scores'><div class='score'><b>{score_text}</b><span>当前 5 题 · 原 Judge</span></div><div class='score'><b>{boundary_score_text}</b><span>当前 5 题 · 候选版</span></div><div class='score'><b>89 → 88 / 100</b><span>旧 100 题 · 整体退步</span></div></div><p class='note'>5 题的 80% 不算达标。本页只验证 Judge，不代表 Bot 生产质量。</p><details><summary>查看完整结论、协议与生成说明</summary><p><b>{GENERATION_DISCLAIMER}。</b> 历史未读窗口按上一次本人消息后重建；生成端使用火山端点，并在离线脚本中放宽调用预算。</p><p>{html.escape(partial_note)}</p><p><b>生成进度：</b>{html.escape(json.dumps(dict(generation_counts), ensure_ascii=False))}<br><b>隔离检查：</b>{'通过' if result.get('isolation_verified') and boundary_result.get('isolation_verified') else '尚未完成或未通过'}<br><b>柚子群2：</b>仅压力场景，不计分，无真人标准答案。</p><p><b>位置均衡旧群聊集：</b>{html.escape(balanced_summary)}</p><p><b>边界候选的代价：</b>{html.escape(tradeoff_summary)}</p>{_pre(manifest)}</details></section></div><div class='controls' aria-label='题目导航'><div class='control-row'><button id='prev-case' type='button'>上一题</button><span class='progress' id='case-progress'>准备中</span><button id='next-case' type='button'>下一题</button></div><label class='filter'><input id='errors-only' type='checkbox'> 只看误判</label></div><div id='case-deck'><section class='dataset' data-title='当前共同富裕群盲测'><h2>当前共同富裕群盲测</h2>{''.join(scored_cards) or '<div class="card">尚未生成盲测题。</div>'}</section><section class='dataset' data-title='候选版翻转诊断'><h2>候选版翻转诊断</h2><p class='note'>检查新提示词是否只是把偏差从一种形式搬到另一种形式。</p>{tradeoff_cards}</section><section class='dataset' data-title='旧群聊集诊断'><h2>旧群聊集诊断</h2>{balanced_cards}</section><section class='dataset' data-title='柚子群压力场景'><h2>柚子群压力场景</h2>{''.join(pressure_cards) or '<div class="card">尚未生成压力场景。</div>'}</section><div class='card empty' id='empty-state'>没有符合当前筛选条件的题目。</div></div><section class='technical-audit'><details><summary>离线对照生成审计（技术记录）</summary>{''.join(generation_cards) or '<div class="card">尚未开始生成。</div>'}</details></section></div><script>
    (() => {{
      const cards = [...document.querySelectorAll('#case-deck .case')];
      const datasets = [...document.querySelectorAll('#case-deck .dataset')];
      const errorsOnly = document.getElementById('errors-only');
      const progress = document.getElementById('case-progress');
      const empty = document.getElementById('empty-state');
      const prev = document.getElementById('prev-case');
      const next = document.getElementById('next-case');
      let index = 0;
      const visibleCards = () => cards.filter(card => !errorsOnly.checked || card.classList.contains('wrong'));
      function show(nextIndex) {{
        const visible = visibleCards();
        cards.forEach(card => card.classList.remove('active'));
        datasets.forEach(dataset => dataset.classList.remove('active'));
        if (!visible.length) {{
          empty.style.display = 'block'; progress.textContent = '0 / 0'; prev.disabled = true; next.disabled = true; return;
        }}
        empty.style.display = 'none';
        index = Math.max(0, Math.min(nextIndex, visible.length - 1));
        const card = visible[index];
        const dataset = card.closest('.dataset');
        dataset.classList.add('active'); card.classList.add('active');
        progress.textContent = `${{index + 1}} / ${{visible.length}} · ${{dataset.dataset.title}}`;
        prev.disabled = index === 0; next.disabled = index === visible.length - 1;
        window.scrollTo({{top: document.querySelector('.controls').offsetTop - 6, behavior: 'smooth'}});
      }}
      prev.addEventListener('click', () => show(index - 1));
      next.addEventListener('click', () => show(index + 1));
      errorsOnly.addEventListener('change', () => show(0));
      show(0);
    }})();
    </script></body></html>"""
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(body, encoding="utf-8")
    HTML_PATH.chmod(0o600)
    return HTML_PATH


def load_target_group_inventory() -> dict[str, Any]:
    payload = json.loads(COMMON_SOURCE_PATH.read_text(encoding="utf-8"))
    if payload.get("chat_id") != COMMON_CHAT_ID or payload.get("chat_name") != "共同富裕群":
        raise ValueError("target group source identity mismatch")
    messages = sorted(payload.get("messages", []), key=lambda row: row.get("create_time") or 0)
    members = sorted({str(row.get("sender") or "").strip() for row in messages if row.get("sender")})
    if "招财鲁" not in members:
        raise ValueError("target group does not contain 招财鲁（西西）")
    self_indexes = [index for index, row in enumerate(messages) if row.get("sender_type") == "self"]
    evidence_by_text = {row["text"]: row for row in TARGET_GROUP_SELF_EVIDENCE}
    verified_evidence = []
    for index in self_indexes:
        text = str(messages[index].get("text") or "").strip()
        evidence = evidence_by_text.get(text)
        if evidence is None:
            raise ValueError(f"missing target-group identity evidence: {text}")
        verified_evidence.append({
            **evidence,
            "same_text_decision_count": 0,
            "same_text_send_count": 0,
            "verdict": "真人原文（生产日志已排除当前 Bot）",
        })
    generated_replies = [
        row for row in messages
        if str(row.get("reply_text") or "").strip() not in {"", "(未回复)"}
    ]
    return {
        "chat_id": payload["chat_id"],
        "chat_name": payload["chat_name"],
        "messages": messages,
        "members": members,
        "self_indexes": self_indexes,
        "verified_evidence": verified_evidence,
        "generated_reply_count": len(generated_replies),
        "min_time": min((row.get("create_time") or 0 for row in messages), default=0),
        "max_time": max((row.get("create_time") or 0 for row in messages), default=0),
    }


def load_target_group_identity_audit() -> dict[str, Any]:
    """Separate provable Bot sends from human evidence and unresolved self rows."""
    if not CASES_DB_PATH.exists():
        raise FileNotFoundError(CASES_DB_PATH)
    connection = sqlite3.connect(f"file:{CASES_DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, session_id, tick_id, created_at, replies_sent_json,
                   session_input_messages_json, send_success
            FROM tick_log
            WHERE chat_name = '共同富裕群' AND send_success = 1
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    confirmed_bot: list[dict[str, Any]] = []
    all_bot_texts: set[str] = set()
    parsed_contexts: list[tuple[sqlite3.Row, list[dict[str, Any]]]] = []
    parsed_rows: list[tuple[sqlite3.Row, list[dict[str, Any]], list[str], bool]] = []
    for row in rows:
        try:
            context = json.loads(row["session_input_messages_json"] or "[]")
            replies = json.loads(row["replies_sent_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(context, list) or not isinstance(replies, list):
            continue
        belongs_to_target = any(
            isinstance(message, dict)
            and (
                message.get("chatroom_id") == TARGET_GROUP_ROOM_ID
                or str(message.get("sender") or "").strip() in {"招财鲁", "鲁西西", "西西"}
            )
            for message in context
        )
        reply_texts = [str(reply).strip() for reply in replies if str(reply).strip()]
        if not reply_texts:
            continue
        parsed_rows.append((row, context, reply_texts, belongs_to_target))

    target_sessions = {
        str(row["session_id"])
        for row, _context, _replies, direct_match in parsed_rows
        if direct_match
    }
    for row, context, reply_texts, direct_match in parsed_rows:
        if not direct_match and str(row["session_id"]) not in target_sessions:
            continue
        all_bot_texts.update(reply_texts)
        parsed_contexts.append((row, context))
        confirmed_bot.append({
            "tick_log_id": row["id"],
            "session_id": row["session_id"],
            "tick_id": row["tick_id"],
            "created_at": row["created_at"],
            "send_success": row["send_success"],
            "replies": reply_texts,
            "context": context,
        })

    representative_bot = [
        row for row in confirmed_bot
        if any(marker in reply for marker in BOT_BAD_REPLY_MARKERS for reply in row["replies"])
    ]
    verified_human_texts = {row["text"] for row in TARGET_GROUP_SELF_EVIDENCE}
    unresolved: list[dict[str, Any]] = []
    seen_unresolved: set[tuple[str, float]] = set()
    for source, context in reversed(parsed_contexts):
        for index, message in enumerate(context):
            if not isinstance(message, dict) or message.get("sender_type") != "self":
                continue
            text = str(message.get("text") or "").strip()
            created_at = float(message.get("create_time") or 0)
            key = (text, created_at)
            if not text or text in all_bot_texts or text in verified_human_texts or key in seen_unresolved:
                continue
            seen_unresolved.add(key)
            unresolved.append({
                "text": text,
                "create_time": created_at,
                "source_tick_log_id": source["id"],
                "context": context[max(0, index - 6):index + 1],
            })
            if len(unresolved) == 8:
                break
        if len(unresolved) == 8:
            break
    return {
        "confirmed_bot": confirmed_bot,
        "representative_bot": representative_bot,
        "unresolved_self": unresolved,
    }


def _source_message_view(row: dict[str, Any], *, target: bool = False) -> str:
    timestamp = datetime.fromtimestamp(row.get("create_time") or 0).strftime("%m-%d %H:%M:%S")
    sender = str(row.get("sender") or "未知")
    text = str(row.get("text") or "").strip()
    if not text:
        text = f"[{row.get('message_type') or '非文字消息'}] {row.get('image_description') or ''}".strip()
    role = "self" if row.get("sender_type") == "self" else "other"
    target_class = " target" if target else ""
    return (
        f"<div class='chat-row {role}{target_class}'>"
        f"<div class='chat-speaker'>{html.escape(sender)} · {html.escape(timestamp)}</div>"
        f"<div class='chat-bubble'>{html.escape(text)}</div></div>"
    )


def _source_context_view(messages: list[dict[str, Any]], *, target_index: int | None = None) -> str:
    if not messages:
        return "<div class='chat-empty'>（没有聊天上下文）</div>"
    return "<div class='chat-view'>" + "".join(
        _source_message_view(row, target=index == target_index)
        for index, row in enumerate(messages)
    ) + "</div>"


def _identity_audit_html(inventory: dict[str, Any], identity: dict[str, Any]) -> str:
    bot_cards = []
    for number, row in enumerate(identity["representative_bot"], 1):
        bot_cards.append(f"""
        <details class='card identity-card bot-evidence' {'open' if number == 1 else ''}>
          <summary><b>Bot 实发差回复 {number}</b><span class='identity-tag bot-tag'>已确认 Bot 实际发送</span></summary>
          <div class='reply-card'><div class='reply-title'><b>原始发送内容</b><span>不是离线生成</span></div>{_reply_view(row['replies'])}</div>
          <h3>发送前完整输入上下文</h3>{_source_context_view(row['context'])}
          <details><summary>查看发送链证据</summary>{_pre({key: row[key] for key in ('tick_log_id','session_id','tick_id','created_at','send_success')})}</details>
        </details>""")

    human_cards = []
    messages = inventory["messages"]
    human_evidence = {row["text"]: row for row in inventory["verified_evidence"]}
    for number, index in enumerate(inventory["self_indexes"], 1):
        message = messages[index]
        evidence = human_evidence[str(message.get("text") or "").strip()]
        start = max(0, index - 12)
        context = messages[start:index + 1]
        human_cards.append(f"""
        <details class='card identity-card human-evidence'>
          <summary><b>真人原文 {number}</b><span class='identity-tag human-tag'>已确认真人原文</span></summary>
          <h3>原文及前 12 条历史消息</h3>{_source_context_view(context, target_index=len(context) - 1)}
          <details><summary>查看身份依据</summary>{_pre(evidence)}</details>
        </details>""")

    unresolved_cards = []
    for number, row in enumerate(identity["unresolved_self"], 1):
        unresolved_cards.append(f"""
        <details class='card identity-card unknown-evidence'>
          <summary><b>未确认 self 消息 {number}</b><span class='identity-tag unknown-tag'>身份无法确认</span></summary>
          <p class='exclude'><b>不计入真人标准。</b>这里只知道它从你的微信号发出，现有证据没有把它可靠对应到手发或 Bot 发送链。</p>
          <h3>出现时的上下文</h3>{_source_context_view(row['context'], target_index=len(row['context']) - 1)}
          <details><summary>查看来源字段</summary>{_pre({'source_tick_log_id': row['source_tick_log_id'], 'sender_type': 'self', 'create_time': row['create_time']})}</details>
        </details>""")

    return f"""
      <section class='card identity-summary'>
        <h2>先看清：哪些是 Bot，哪些是真人</h2>
        <p><span class='identity-tag bot-tag'>已确认 Bot 实际发送</span> 有 <b>{len(identity['confirmed_bot'])}</b> 个发送轮次；依据是生产发送链 <code>send_success=1</code>，不是看语气猜的。</p>
        <p><span class='identity-tag human-tag'>已确认真人原文</span> 有 <b>{len(inventory['verified_evidence'])}</b> 条；依据是历史生产日志排除了当前 Bot 生成及发送。</p>
        <p><span class='identity-tag unknown-tag'>身份无法确认</span> 仅有 <code>sender_type=self</code> 的内容一律不冒充真人，<b>不计入真人标准</b>。</p>
      </section>
      <h2>已确认 Bot 实发的差回复</h2>
      <p>以下是发送链中能直接证明由 Bot 发出的代表性差回复。点开可看该次发送前传给系统的完整输入上下文。</p>
      {''.join(bot_cards)}
      <h2>已确认真人原文</h2>
      <p>这些原文才可作为真人对照；每条展示原文前 12 条历史消息和核验依据。</p>
      {''.join(human_cards)}
      <h2>身份无法确认的 self 消息</h2>
      <p>下面是仅凭现有字段无法判断手发还是 Bot 的样例，只用于说明数据边界，不参与计分。</p>
      {''.join(unresolved_cards) or '<div class="card">当前没有可展示的未确认样例。</div>'}
    """


def render_html() -> Path:
    blind_rows = json.loads(BALANCED160_BLIND_PATH.read_text(encoding="utf-8"))["cases"]
    answer_rows = json.loads(BALANCED160_ANSWERS_PATH.read_text(encoding="utf-8"))["cases"]
    baseline_result = json.loads(BALANCED160_BASELINE_RESULT_PATH.read_text(encoding="utf-8"))
    candidate_result = json.loads(BALANCED160_CANDIDATE_RESULT_PATH.read_text(encoding="utf-8"))
    baseline_audit = json.loads(BALANCED160_BASELINE_AUDIT_PATH.read_text(encoding="utf-8"))
    candidate_audit = json.loads(BALANCED160_CANDIDATE_AUDIT_PATH.read_text(encoding="utf-8"))
    candidate_profile = json.loads(BALANCED160_CANDIDATE_PROFILE_PATH.read_text(encoding="utf-8"))
    assert CHAT_IRREGULARITY_RULE in candidate_profile["instructions"]
    answers = {row["case_id"]: row["human_option"] for row in answer_rows}
    baseline_predictions = {row["case_id"]: row for row in baseline_result["cases"]}
    candidate_predictions = {row["case_id"]: row for row in candidate_result["cases"]}
    baseline_calls = {row["case_id"]: row for row in baseline_audit["calls"]}
    candidate_calls = {row["case_id"]: row for row in candidate_audit["calls"]}

    cards = []
    counts = Counter()
    relationship_correct = defaultdict(lambda: {"baseline": 0, "candidate": 0, "total": 0})
    for number, case in enumerate(blind_rows, 1):
        case_id = case["case_id"]
        actual = answers[case_id]
        baseline = baseline_predictions[case_id]
        candidate = candidate_predictions[case_id]
        baseline_correct = baseline.get("predicted") == actual
        candidate_correct = candidate.get("predicted") == actual
        either_wrong = not baseline_correct or not candidate_correct
        candidate_fix = not baseline_correct and candidate_correct
        candidate_new_mistake = baseline_correct and not candidate_correct
        counts["all"] += 1
        counts["either_wrong"] += int(either_wrong)
        counts["candidate_fix"] += int(candidate_fix)
        counts["candidate_new_mistake"] += int(candidate_new_mistake)
        relationship = str(case.get("relationship") or "unknown")
        relationship_correct[relationship]["total"] += 1
        relationship_correct[relationship]["baseline"] += int(baseline_correct)
        relationship_correct[relationship]["candidate"] += int(candidate_correct)
        if candidate_fix:
            change_text = "候选修正：基线判错，候选判对"
            change_class = "fix"
        elif candidate_new_mistake:
            change_text = "候选新增判错：基线判对，候选判错"
            change_class = "regression"
        elif baseline_correct:
            change_text = "两版都判对"
            change_class = "same-correct"
        else:
            change_text = "两版都判错"
            change_class = "same-wrong"
        option_a_identity = "真人原回复" if actual == "A" else "匿名对照回复"
        option_b_identity = "真人原回复" if actual == "B" else "匿名对照回复"
        cards.append(f"""
        <details class='card case balanced-case {change_class}' data-either-wrong='{int(either_wrong)}' data-candidate-fix='{int(candidate_fix)}' data-candidate-new-mistake='{int(candidate_new_mistake)}'>
          <summary><b>题目 {number} · {'私聊' if relationship == 'private' else '群聊'}</b><span>{change_text}</span></summary>
          <p><b>题目 ID：</b><code>{html.escape(case_id)}</code>　<b>已知真人答案：</b>回复 {html.escape(actual)}</p>
          <h3>完整聊天时间线</h3>{_context_view(case.get('context_original', []))}
          <div class='options'>{_labeled_reply('A', case.get('option_A', []), option_a_identity)}{_labeled_reply('B', case.get('option_B', []), option_b_identity)}</div>
          <div class='judges'>
            <section class='judge-result {'correct' if baseline_correct else 'incorrect'}'><h3>当前基线 Judge</h3><p><b>选择：</b>{html.escape(str(baseline.get('predicted') or ''))}　<b>结果：</b>{'正确' if baseline_correct else '错误'}　<b>置信度：</b>{html.escape(str(baseline.get('confidence', '')))}</p><p><b>理由：</b>{html.escape(str(baseline.get('reason') or ''))}</p></section>
            <section class='judge-result {'correct' if candidate_correct else 'incorrect'}'><h3>候选 Judge</h3><p><b>选择：</b>{html.escape(str(candidate.get('predicted') or ''))}　<b>结果：</b>{'正确' if candidate_correct else '错误'}　<b>置信度：</b>{html.escape(str(candidate.get('confidence', '')))}</p><p><b>理由：</b>{html.escape(str(candidate.get('reason') or ''))}</p></section>
          </div>
          <details><summary>当前基线 Judge · 完整原始调用记录</summary>{_pre(baseline_calls[case_id])}</details>
          <details><summary>候选 Judge · 完整原始调用记录</summary>{_pre(candidate_calls[case_id])}</details>
        </details>""")

    assert counts == {"all": 160, "either_wrong": 18, "candidate_fix": 1, "candidate_new_mistake": 7}
    assert relationship_correct == {
        "private": {"baseline": 77, "candidate": 74, "total": 80},
        "group": {"baseline": 72, "candidate": 69, "total": 80},
    }
    body = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'><title>Judge 160 题统一结果</title><style>
    :root{{color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3}}.judge-page{{max-width:920px;margin:auto;padding:12px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55;min-width:0}}.judge-page *{{box-sizing:border-box}}h1{{font-size:22px;margin:4px 0 12px}}h2{{font-size:18px;margin:18px 0 8px}}h3{{font-size:13px;color:#8b949e;margin:10px 0 6px}}p{{overflow-wrap:anywhere}}.card{{display:block;background:#161b22;border:1px solid #30363d;border-radius:12px;padding:13px;margin:12px 0}}.summary{{border-left:5px solid #f85149}}.status{{display:inline-block;background:#3d1719;color:#ff7b72;border:1px solid #f85149;border-radius:99px;padding:5px 10px;font-weight:700}}.facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}}.fact{{background:#0d1117;border:1px solid #30363d;border-radius:9px;padding:9px}}.fact b{{display:block;font-size:18px}}.fact span{{font-size:12px;color:#8b949e}}.definitions{{border-left:5px solid #58a6ff}}.controls{{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:7px;background:#0d1117e8;backdrop-filter:blur(8px);padding:9px 0}}button{{border:1px solid #30363d;border-radius:99px;background:#161b22;color:#e6edf3;padding:7px 10px;cursor:pointer}}button.active{{border-color:#58a6ff;background:#13233a;color:#79c0ff}}details.case>summary{{display:flex;align-items:center;justify-content:space-between;gap:10px;cursor:pointer;list-style:none}}details.case>summary::-webkit-details-marker{{display:none}}details.case>summary span{{font-size:12px;border:1px solid currentColor;border-radius:99px;padding:3px 7px;text-align:right}}.fix{{border-left:4px solid #3fb950}}.regression,.same-wrong{{border-left:4px solid #f85149}}.same-correct{{border-left:4px solid #30363d}}.chat-view{{display:flex;flex-direction:column;gap:8px;background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:10px;min-width:0;max-height:60vh;overflow:auto}}.chat-row{{display:flex;flex-direction:column;align-items:flex-start;max-width:92%}}.chat-row.self{{align-self:flex-end;align-items:flex-end}}.chat-speaker{{font-size:11px;color:#8b949e;margin:0 6px 3px}}.chat-bubble{{background:#21262d;border-radius:4px 12px 12px 12px;padding:7px 10px;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}}.chat-row.self .chat-bubble{{background:#164e2e;border-radius:12px 4px 12px 12px}}.options,.judges{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}}.reply-card{{background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;min-width:0}}.reply-title{{display:flex;justify-content:space-between;gap:8px;padding:8px 10px;border-bottom:1px solid #30363d}}.reply-title span{{font-size:12px;color:#58a6ff}}.reply-view{{padding:10px;display:flex;flex-direction:column;gap:7px}}.reply-line{{background:#21262d;border-radius:8px;padding:8px 10px;white-space:pre-wrap;overflow-wrap:anywhere}}.reply-empty{{padding:10px;color:#8b949e}}.judge-result{{background:#0d1117;border:1px solid #30363d;border-radius:9px;padding:10px}}.judge-result.correct{{border-color:#2f6f44}}.judge-result.incorrect{{border-color:#f85149}}details details{{margin-top:10px;border-top:1px solid #30363d;padding-top:8px}}details details>summary{{cursor:pointer;color:#58a6ff}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;background:#0d1117;border:1px solid #30363d;border-radius:7px;padding:10px;font-size:12px;line-height:1.5;max-height:60vh;overflow:auto}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.judge-page{{padding:10px 10px calc(20px + env(safe-area-inset-bottom))}}.facts{{grid-template-columns:1fr 1fr}}.options,.judges{{grid-template-columns:1fr}}.card{{padding:11px}}.chat-row{{max-width:96%}}details.case>summary{{align-items:flex-start}}}}
    </style></head><body><main class='judge-page'>
      <h1>Judge 迭代 · 私聊 80 题 + 群聊 80 题</h1>
      <section class='card summary'><span class='status'>不采用候选，保留当前 Judge 开发基线</span>
        <p><b>本轮只改了什么：</b>在当前基线规则后只增加这一条：<q>{html.escape(CHAT_IRREGULARITY_RULE)}</q>题目、答案、参考资料、模型、输出格式和其他 Judge 规则均不变。</p>
        <p><b>为什么验证：</b>当前基线仍共同判错 10 题，其中真人回复有错字、漏字、残缺时间或口语不完整。实验要验证：禁止 Judge 单凭这些聊天不规范现象判断“更像 AI”，能否减少错判。</p>
        <p><b>结果：</b>当前基线判断正确 <b>149/160 = 93.125%</b>；候选判断正确 <b>143/160 = 89.375%</b>。候选修正 <b>1</b> 题，新增判错 <b>7</b> 题，净值为 <b>1 - 7 = -6/160</b>。</p>
        <div class='facts'><div class='fact'><b>77/80 → 74/80</b><span>私聊准确率 · 96.25% → 92.50%</span></div><div class='fact'><b>72/80 → 69/80</b><span>群聊准确率 · 90.00% → 86.25%</span></div><div class='fact'><b>1</b><span>候选修正</span></div><div class='fact'><b>7</b><span>候选新增判错</span></div></div>
        <p><b>采用门槛：</b>总体至少 <code>+1/160</code>、私聊至少 <code>0/80</code>、群聊至少 <code>0/80</code>、新增“判错且置信度不低于 0.80”的题必须为 <code>0</code>。本轮依次为 <code>-6/160</code>、<code>-3/80</code>、<code>-3/80</code>、<code>3</code>，四项均未通过。</p>
        <p><b>统计结果：</b>1 条修正与 7 条新增判错的双侧精确符号检验为 <b>p=0.0703125</b>，未达到流程规定的 <code>p≤0.05</code>。</p>
        <p><b>因果边界：</b>8 道翻转题的单次判断确实发生了变化，但两版理由均没有直接引用新增规则；现有证据不能证明这 1 个修正或 7 个新增错误由该规则直接导致，也不能把随机变化强行归因于这个功能。是否采用只按四项固定门槛决定，因此仍然不采用。</p>
        <p><b>目标错题：</b><code>case_26e508563b1a2172a0dbd299</code> 和 <code>case_524892d61e766a2005514e32</code> 在候选中仍然判错，本轮规则没有修复原计划解决的两题。</p>
        <p><b>速度：</b>当前基线平均 10,795.31875 ms、p50 10,350.5 ms、p95 16,778 ms、最快 4,885 ms、最慢 18,620 ms；候选平均 10,108.05 ms、p50 8,363.5 ms、p95 16,486 ms、最快 5,228 ms、最慢 19,091 ms。两版不是同一时刻运行，速度差不作为本轮采用依据。</p>
        <p><b>最终决定：</b>候选不采用；后续 Judge 迭代继续使用 <b>149/160</b> 的当前开发基线。本轮不是独立正式验收。</p>
        <p><b>未改动：</b>C0 未运行、Bot 未启动、没有发送微信、线上 Judge 配置未修改。</p>
      </section>
      <section class='card definitions'><h2>页面里每个词是什么意思</h2>
        <p><b>准确率：</b>已知真人答案的题里，Judge 选中真人回复的比例。<b>候选修正：</b>基线选错、候选选对。<b>候选新增判错：</b>基线选对、候选选错。<b>净值：</b>候选修正数减去候选新增判错数。</p>
        <p><b>置信度：</b>Judge 自己认为判断有多确定，不是实际正确概率。<b>反证：</b>说明某个回复为什么可能不符合当前聊天或真人习惯的证据。<b>p50：</b>一半请求不超过该时间；<b>p95：</b>95% 请求不超过该时间。</p>
      </section>
      <h2>完整 160 题逐题对照</h2>
      <p>每题都保留原始交错时间线、回复 A/B、已知真人答案、两版 Judge 的选择/置信度/理由，以及两版完整原始调用记录。</p>
      <div class='controls'><button class='active' data-filter='all'>全部 160</button><button data-filter='either-wrong'>任一版本判错 18</button><button data-filter='candidate-fix'>候选修正 1</button><button data-filter='candidate-new-mistake'>候选新增判错 7</button><span id='visible-count'>160 / 160</span></div>
      <div id='balanced-cases'>{''.join(cards)}</div>
    </main><script>const rows=[...document.querySelectorAll('.balanced-case')],buttons=[...document.querySelectorAll('[data-filter]')],count=document.getElementById('visible-count');function show(kind){{let shown=0;rows.forEach(row=>{{const visible=kind==='all'||row.getAttribute('data-'+kind)==='1';row.hidden=!visible;if(visible)shown++;}});count.textContent=shown+' / '+rows.length;buttons.forEach(button=>button.classList.toggle('active',button.dataset.filter===kind));}}buttons.forEach(button=>button.addEventListener('click',()=>show(button.dataset.filter)));show('all');</script></body></html>"""
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(body, encoding="utf-8")
    HTML_PATH.chmod(0o600)
    return HTML_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "manifest", "generate", "freeze", "judge", "score",
            "boundary-judge", "boundary-score", "holdout-select", "holdout-generate",
            "holdout-freeze", "holdout-v2-judge", "holdout-v2-score",
            "holdout-candidate-judge", "holdout-candidate-score", "render", "all",
        ),
    )
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--offline-budget-multiplier",
        type=float,
        default=DEFAULT_OFFLINE_BUDGET_MULTIPLIER,
    )
    args = parser.parse_args()
    outputs = []
    phases = ("manifest", "generate", "freeze", "judge", "score", "render") if args.phase == "all" else (args.phase,)
    for phase in phases:
        value = {
            "manifest": build_manifest,
            "generate": lambda: generate(args.max_calls, args.offline_budget_multiplier),
            "freeze": lambda: freeze(args.allow_partial),
            "judge": lambda: run_judge(args.max_calls),
            "score": score,
            "boundary-judge": lambda: run_judge(
                args.max_calls,
                profile_path=BOUNDARY_PROFILE_PATH,
                predictions_path=BOUNDARY_PREDICTIONS_PATH,
                audit_path=BOUNDARY_AUDIT_PATH,
            ),
            "boundary-score": lambda: score(
                predictions_path=BOUNDARY_PREDICTIONS_PATH,
                audit_path=BOUNDARY_AUDIT_PATH,
                result_path=BOUNDARY_RESULT_PATH,
            ),
            "holdout-select": freeze_holdout_selection,
            "holdout-generate": lambda: generate_holdout(args.max_calls, args.offline_budget_multiplier),
            "holdout-freeze": freeze_holdout,
            "holdout-v2-judge": lambda: run_judge(
                args.max_calls,
                predictions_path=HOLDOUT_V2_PREDICTIONS_PATH,
                audit_path=HOLDOUT_V2_AUDIT_PATH,
                blind_path=HOLDOUT_BLIND_PATH,
            ),
            "holdout-v2-score": lambda: score(
                predictions_path=HOLDOUT_V2_PREDICTIONS_PATH,
                audit_path=HOLDOUT_V2_AUDIT_PATH,
                result_path=HOLDOUT_V2_RESULT_PATH,
                blind_path=HOLDOUT_BLIND_PATH,
                answers_path=HOLDOUT_ANSWERS_PATH,
            ),
            "holdout-candidate-judge": lambda: run_judge(
                args.max_calls,
                profile_path=HOLDOUT_CANDIDATE_PROFILE_PATH,
                predictions_path=HOLDOUT_CANDIDATE_PREDICTIONS_PATH,
                audit_path=HOLDOUT_CANDIDATE_AUDIT_PATH,
                blind_path=HOLDOUT_BLIND_PATH,
            ),
            "holdout-candidate-score": lambda: score(
                predictions_path=HOLDOUT_CANDIDATE_PREDICTIONS_PATH,
                audit_path=HOLDOUT_CANDIDATE_AUDIT_PATH,
                result_path=HOLDOUT_CANDIDATE_RESULT_PATH,
                blind_path=HOLDOUT_BLIND_PATH,
                answers_path=HOLDOUT_ANSWERS_PATH,
            ),
            "render": lambda: str(render_html()),
        }[phase]()
        outputs.append({phase: value})
    print(json.dumps(outputs, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
