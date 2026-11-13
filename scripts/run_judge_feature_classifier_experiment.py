#!/usr/bin/env python3
"""Leak-safe Judge experiment: LLM features plus a symmetric linear classifier."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import difflib
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_human_ai_judge_benchmark_v2 import (  # noqa: E402
    candidate_from_record,
    candidate_record,
)
from scripts.build_human_ai_judge_benchmark_v3 import (  # noqa: E402
    current_c0_sha256,
)
from scripts.build_judge_dynamic_few_shots import _normalized_messages  # noqa: E402
from scripts.build_persona_few_shots import Candidate, extract_candidates  # noqa: E402
from scripts.evaluate_human_ai_judge import (  # noqa: E402
    CUTOFF,
    _case_id,
    _event_has_tool_call,
    _export_payloads,
    _few_shot_identities,
    _json_sha256,
    _normalized_reply,
    _run_c0,
    _write_private,
    assert_reference_isolated,
    build_blind_packet,
    reference_leakage_details,
    rebuild_generation_inputs,
    run_codex_judge_independently,
)
from src.reply.c0_config import current_c0_baseline_id, load_current_c0  # noqa: E402
from src.reply.c0_factory import (  # noqa: E402
    create_current_c0_clients,
    create_current_c0_generator,
)


def _write_feature_classifier_result(path: Path, result: dict[str, Any]) -> None:
    _write_private(path, {
        **result,
        "experiment_type": "feature_classifier_experiment",
        "official_judge_baseline": False,
    })


PRIVATE_DIR = PROJECT_ROOT / "data/reports/private/judge_feature_classifier_v1"
DEFAULT_GUARD = PRIVATE_DIR / "holdout_guard.json"
DEFAULT_MANIFEST = PRIVATE_DIR / "manifest.json"
DEFAULT_CHECKPOINT = PRIVATE_DIR / "c0_checkpoint.json"
DEFAULT_BLIND = PRIVATE_DIR / "development_blind.json"
DEFAULT_ANSWERS = PRIVATE_DIR / "development_answers.json"
DEFAULT_REPLAY = PRIVATE_DIR / "development_replay.json"
DEFAULT_FEATURES = PRIVATE_DIR / "feature_checkpoint.json"
DEFAULT_RESULT = PRIVATE_DIR / "learning_curve_result.json"
DEFAULT_REPLY_BUBBLE_COUNT_RESULT = (
    PRIVATE_DIR / "reply_bubble_count_learning_curve_result.json"
)
DEFAULT_SHARED_MLP_RESULT = PRIVATE_DIR / "shared_mlp_learning_curve_result.json"
DEFAULT_EXPLICIT_CROSS_RESULT = PRIVATE_DIR / "explicit_cross_learning_curve_result.json"
DEFAULT_DEEP_SHARED_MLP_RESULT = PRIVATE_DIR / "deep_shared_mlp_learning_curve_result.json"
DEFAULT_BOOLEAN_CONTEXT_CROSS_RESULT = (
    PRIVATE_DIR / "boolean_context_cross_learning_curve_result.json"
)
DEFAULT_REFINED_BOOLEAN_CONTEXT_RESULT = (
    PRIVATE_DIR / "refined_boolean_context_learning_curve_result.json"
)
DEFAULT_GROUP_PATTERN_BOOLEAN_RESULT = (
    PRIVATE_DIR / "group_pattern_boolean_result.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_MANIFEST = (
    PROJECT_ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_CHECKPOINT = (
    PRIVATE_DIR / "group_pattern_holdout_c0_checkpoint.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_REPLAY = (
    PRIVATE_DIR / "group_pattern_holdout_replay.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_BLIND = (
    PRIVATE_DIR / "group_pattern_holdout_blind.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_ANSWERS = (
    PRIVATE_DIR / "group_pattern_holdout_answers.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_FEATURES = (
    PRIVATE_DIR / "group_pattern_holdout_features.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_RESULT = (
    PRIVATE_DIR / "group_pattern_holdout_240_result.json"
)
DEFAULT_EFFECTIVE_GROUP_PATTERN_HOLDOUT_MANIFEST = (
    PRIVATE_DIR / "group_pattern_holdout_effective_manifest.json"
)
DEFAULT_EFFECTIVE_GROUP_PATTERN_HOLDOUT_CHECKPOINT = (
    PRIVATE_DIR / "group_pattern_holdout_effective_c0_checkpoint.json"
)
DEFAULT_GROUP_PATTERN_HOLDOUT_CANDIDATE_MANIFESTS = (
    PROJECT_ROOT / "data/reports/private/adversarial_hidden_holdout_v1/manifest.json",
    PROJECT_ROOT / "data/reports/private/fixed_c0_holdout_final/manifest.json",
)
DEFAULT_STABLE_REFERENCE_SOURCE = (
    PROJECT_ROOT
    / "data/reports/private/human_ai_judge_benchmark_v3_current_c0_f977633e9391c0f4/judge_reference.json"
)
DEFAULT_STABLE_PROFILE = (
    PROJECT_ROOT
    / "data/reports/private/human_ai_judge_benchmark_v3_balanced160/judge_profile_single_rule_full_timeline_v1.json"
)
DEFAULT_STABLE_REFERENCE = PRIVATE_DIR / "validation_reference_stable_judge_v1.json"
DEFAULT_STABLE_BLIND = PRIVATE_DIR / "validation_blind_stable_judge_v1.json"
DEFAULT_STABLE_ANSWERS = PRIVATE_DIR / "validation_answers_stable_judge_v1.json"
DEFAULT_STABLE_PREDICTIONS = PRIVATE_DIR / "stable_judge_validation_predictions.json"
DEFAULT_STABLE_AUDIT = PRIVATE_DIR / "stable_judge_validation_audit.json"
DEFAULT_STABLE_COMPARISON = PRIVATE_DIR / "stable_judge_vs_feature_classifier_1200.json"
DEFAULT_BASE_CORRECTION_DEVELOPMENT_RESULT = (
    PRIVATE_DIR / "base_correction_development_train1200_only_result.json"
)
DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT = (
    PRIVATE_DIR / "reworked_boolean_correction_development_result.json"
)
DEFAULT_REWORKED_BOOLEAN_CORRECTION_FIXED_240_RESULT = (
    PRIVATE_DIR / "reworked_boolean_correction_fixed_240_result.json"
)
DEFAULT_BASE_CORRECTION_FIXED_240_RESULT = (
    PRIVATE_DIR / "base_correction_fixed_240_result.json"
)
DEFAULT_FIXED_240_BLIND = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_blind.json"
)
DEFAULT_FIXED_240_ANSWERS = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_answers.json"
)
DEFAULT_FIXED_240_FEATURES = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_features.json"
)
DEFAULT_FIXED_240_FEATURE_SOURCE_BLIND = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_blind.json"
)
DEFAULT_FIXED_240_FEATURE_SOURCE_ANSWERS = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_answers.json"
)
DEFAULT_FIXED_240_REPLAY = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_replay.json"
)
DEFAULT_FIXED_240_BASE_PREDICTIONS = (
    PRIVATE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_predictions.json"
)
DEFAULT_EXPORTS = PROJECT_ROOT / "data/exports/b"
DEFAULT_WIKI = PROJECT_ROOT / "data/memory/wiki/users"
DEFAULT_VOLCENGINE_CONFIG = Path.home() / ".cc-connect/config.toml"

SEED = "judge-feature-classifier-v1-20260827"
RELATIONS = ("private", "group")
TRAIN_TARGETS = {"private": 600, "group": 600}
VALIDATION_TARGETS = {"private": 80, "group": 80}
TRAIN_TOTAL = 1200
VALIDATION_TOTAL = 160
LEARNING_CURVE_TOTALS = (480, 800, 1200)
LONG_MESSAGE_MIN_CHARS = 12
C0_GENERATION_INPUT_CONTRACT = {"compact_verified_turn": True}
STABLE_VALIDATION_BATCH_ID = "judge_feature_classifier_stable_judge_validation_v1"
STABLE_PROFILE_NAME = "judge_single_rule_full_timeline_v1"
STABLE_PROFILE_MODE = "append_to_builtin"
STABLE_PROFILE_SHA256 = "b84e04ce1a1ca5cdfd104b57ffebb6d9837eaebd9ccd6db4a4dc4d4ca310a80e"
HIGH_CONFIDENCE_THRESHOLD = 0.80
BASE_CONFIDENCE_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
BASE_CORRECTION_THRESHOLD_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
SHARED_MLP_HIDDEN_UNITS = 8
SHARED_MLP_L2 = 0.001
SHARED_MLP_MAX_ITERATIONS = 1000
DEEP_SHARED_MLP_HIDDEN_UNITS = (16, 8)
DEEP_SHARED_MLP_L2 = 0.001
DEEP_SHARED_MLP_MAX_ITERATIONS = 5000
RECENT_SPEAKER_POSITIONS = 5
IDENTIFIED_GROUP_SPEAKER_POSITIONS = 3
GROUP_NAME_MIN_TRAIN_CASES = 5
GROUP_MEMBER_MIN_TRAIN_OCCURRENCES = 8
GROUP_PATTERN_RECENT_MESSAGE_LIMIT = 20
GROUP_PATTERN_HOLDOUT_MANIFEST_SHA256 = (
    "6541a50b4dda3cbcce02424c2c5365ab376f57024cd02d611669fdd4d087d918"
)
GROUP_PATTERN_HOLDOUT_TARGETS = {"private": 80, "group": 160}
GROUP_PATTERN_HOLDOUT_TOTAL = 240

OBSERVABLE_OPTION_BOOLEAN_NAMES = (
    "reply_bubble_count>=2",
    "reply_bubble_count>=3",
    "reply_char_count>=8",
    "reply_char_count>=16",
    "reply_char_count>=32",
    "reply_contains_question_mark",
    "reply_contains_at_mention",
    "reply_contains_digit",
    "reply_mentions_latest_speaker",
    "reply_contains_laughter_marker",
)
GROUP_PATTERN_OPTION_BOOLEAN_NAMES = (
    "reply_reuses_active_group_pattern",
)
GROUP_PARTICIPATION_FACT_PATTERN_FEATURE_NAME = (
    "is_group*recent_5_has_self*depends_on_unstated_fact*group_pattern_alignment>=1"
)

INTEGER_FIELDS = (
    "context_fit",
    "addressee_fit",
    "timeline_continuity",
    "stance_continuity",
    "specificity",
    "conversational_naturalness",
    "ai_template_signal",
    "unsupported_fact_signal",
    "over_explanation_signal",
    "group_pattern_alignment",
)
REPLY_QUALITY_ISSUE_BOOLEAN_FIELDS = (
    "wrong_addressee",
    "social_tone_misread",
    "unnecessary_inquiry",
    "missed_pending_question",
    "paraphrase_agreement_without_contribution",
    "self_repetition",
    "mechanical_repetition",
    "address_term_mismatch",
    "unsupported_detail",
    "forced_humor_or_defense",
    "inappropriate_participation",
)
BOOLEAN_FIELDS = (
    "depends_on_unstated_fact",
    "repeats_context_needlessly",
) + REPLY_QUALITY_ISSUE_BOOLEAN_FIELDS
ENUM_FIELDS = {
    "reply_action": (
        "answer_now", "defer", "accept", "refuse", "confirm", "joke_tease",
        "agree", "disagree", "ask", "clarify", "supplement", "acknowledge_close",
        "empathize_support", "evaluate_criticize", "correct", "coordinate", "other",
    ),
    "tone": (
        "plain", "warm", "playful", "teasing", "supportive", "serious",
        "impatient", "formal", "other",
    ),
    "length_band": ("very_short", "short", "medium", "long"),
}
OPTION_FIELDS = set(INTEGER_FIELDS) | set(BOOLEAN_FIELDS) | set(ENUM_FIELDS)


def _content_char_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_content_char_count(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _content_char_count(item)
            for key, item in value.items()
            if key in {"content", "text"}
        )
    return 0


def _safe_error_field(value: Any) -> str | None:
    text = str(value or "")
    return text if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", text) else None


def _server_error_metadata(exc: Exception) -> dict[str, str | None]:
    response = getattr(exc, "response", None)
    try:
        payload = response.json() if response is not None else {}
    except Exception:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else {}
    error = error if isinstance(error, dict) else {}
    message = str(error.get("message") or payload.get("message") or "").lower()
    if "max_tokens" in message and any(
        term in message for term in ("maximum", "less than", "at most", "exceed", "range")
    ):
        category = "max_tokens_limit"
    elif any(term in message for term in (
        "context length", "context_length", "maximum context", "prompt is too long",
        "input is too long", "input token",
    )):
        category = "context_length_limit"
    elif any(term in message for term in ("tool schema", "tool definition", "invalid tool", "tools.")):
        category = "invalid_tools"
    elif any(term in message for term in (
        "invalid message", "messages.", "message role", "message content",
    )):
        category = "invalid_message_structure"
    else:
        category = "unclassified_http_error"
    return {
        "server_error_type": _safe_error_field(error.get("type") or payload.get("type")),
        "server_error_code": _safe_error_field(error.get("code") or payload.get("code")),
        "server_error_category": category,
    }


class DiagnosticChatClient:
    """Preserve client behavior while retaining content-free failure metadata."""

    def __init__(self, client: Any):
        self._client = client
        self._diagnostic_calls: list[dict[str, Any]] = []
        self._stage_counts: Counter[str] = Counter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def reset_diagnostics(self) -> None:
        self._diagnostic_calls = []
        self._stage_counts = Counter()

    def diagnostic_calls(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._diagnostic_calls]

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        requested_raise = bool(kwargs.get("raise_on_error", False))
        strict_kwargs = dict(kwargs)
        strict_kwargs["raise_on_error"] = True
        stage = "skill_router" if strict_kwargs.get("max_tokens") == 512 else "generation"
        messages = strict_kwargs.get("messages")
        if messages is None and args:
            messages = args[0]
        messages = messages if isinstance(messages, list) else []
        system_prompt_chars = sum(
            _content_char_count(row.get("content"))
            for row in messages
            if isinstance(row, dict) and row.get("role") == "system"
        ) + _content_char_count(strict_kwargs.get("system_prompt"))
        message_chars = sum(
            _content_char_count(row.get("content"))
            for row in messages
            if isinstance(row, dict) and row.get("role") != "system"
        ) + _content_char_count(strict_kwargs.get("message"))
        self._stage_counts[stage] += 1
        diagnostic = {
            "call_index": len(self._diagnostic_calls) + 1,
            "stage": stage,
            "attempt": self._stage_counts[stage],
            "returned_empty_content": False,
            "exception_type": None,
            "http_status_code": None,
            "request_message_count": len(messages),
            "request_character_count": message_chars + system_prompt_chars,
            "system_prompt_character_count": system_prompt_chars,
            "tool_definition_count": len(strict_kwargs.get("tools") or []),
            "max_tokens": strict_kwargs.get("max_tokens"),
        }
        if hasattr(self._client, "last_status_code"):
            self._client.last_status_code = None
        started = time.monotonic()
        try:
            result = self._client.chat(*args, **strict_kwargs)
            content = result if isinstance(result, str) else getattr(result, "content", "")
            tool_calls = [] if isinstance(result, str) else getattr(result, "tool_calls", None)
            diagnostic["returned_empty_content"] = not str(content or "").strip() and not tool_calls
            return result
        except Exception as exc:
            diagnostic["returned_empty_content"] = True
            diagnostic["exception_type"] = type(exc).__name__
            response = getattr(exc, "response", None)
            diagnostic["http_status_code"] = (
                getattr(response, "status_code", None)
                or getattr(self._client, "last_status_code", None)
            )
            diagnostic.update(_server_error_metadata(exc))
            if requested_raise:
                raise
            return ""
        finally:
            diagnostic["duration_seconds"] = round(time.monotonic() - started, 3)
            self._diagnostic_calls.append(diagnostic)


def c0_failure_diagnostic(
    output: dict[str, Any], client: DiagnosticChatClient
) -> dict[str, Any]:
    calls = client.diagnostic_calls()
    return {
        "llm_call_count": len(calls),
        "llm_calls": calls,
        "budget_exhausted_before_first_llm_call": bool(
            output.get("generation_failed") and not calls
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rank(value: str) -> str:
    return _hash(f"{SEED}:{value}")


def _normalized_sequence(values: Any) -> tuple[str, ...]:
    return _normalized_messages(values)


def _message_texts(candidate: Candidate) -> tuple[str, ...]:
    rows = candidate.context_messages or []
    if rows:
        return _normalized_sequence(row.get("text", "") for row in rows)
    return _normalized_sequence(candidate.context)


def _reply_texts(candidate: Candidate) -> tuple[str, ...]:
    rows = candidate.reply_messages or []
    if rows:
        return _normalized_sequence(row.get("text", "") for row in rows)
    return _normalized_sequence(candidate.replies)


def _few_shot_reply_signature(candidate: Candidate) -> dict[str, Any]:
    rows = candidate.reply_messages or []
    replies = (
        [str(row.get("text", "")) for row in rows]
        if rows
        else [str(reply) for reply in candidate.replies]
    )
    normalized_messages = tuple(
        normalized for reply in replies
        if (normalized := _normalized_reply(reply))
    )
    return {
        "reply_hash": _hash(_normalized_reply(replies)),
        "reply_message_hashes": sorted(_hash(reply) for reply in normalized_messages),
    }


def event_signature(candidate: Candidate) -> dict[str, Any]:
    """Return hashes only; plaintext never enters the sealed holdout guard."""
    messages = _message_texts(candidate)
    replies = _reply_texts(candidate)
    pairs = [messages[index:index + 2] for index in range(len(messages) - 1)]
    long_messages = [message for message in messages if len(message) >= LONG_MESSAGE_MIN_CHARS]
    return {
        "case_hash": _hash(_case_id(candidate)),
        "chat_hash": _hash(candidate.chat_id),
        "timestamp_hash": _hash(f"{candidate.chat_id}:{candidate.timestamp}"),
        "full_context_hash": _hash("\u241e".join(messages)),
        "message_pair_hashes": sorted(_hash("\u241e".join(pair)) for pair in pairs),
        "long_message_hashes": sorted(_hash(message) for message in long_messages),
        "reply_hash": _hash("\u241e".join(replies)),
        "reply_message_hashes": sorted(_hash(reply) for reply in replies),
    }


def signature_from_replay_case(row: dict[str, Any]) -> dict[str, Any]:
    candidate = Candidate(
        relation="group" if row.get("relationship") == "group" else "acquaintance",
        chat_id=str(row.get("source_chat_id") or row.get("chat_id") or ""),
        context=list(row.get("context_original") or []),
        replies=list(row.get("human_reply_original") or []),
        score=0.0,
        timestamp=int(row.get("timestamp") or 0),
    )
    return event_signature(candidate)


def signatures_overlap(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if left["case_hash"] == right["case_hash"]:
        reasons.append("same_case")
    same_chat = left["chat_hash"] == right["chat_hash"]
    if same_chat and left["timestamp_hash"] == right["timestamp_hash"]:
        reasons.append("same_timestamp")
    if same_chat and left["full_context_hash"] == right["full_context_hash"]:
        reasons.append("same_full_context")
    if same_chat and set(left["message_pair_hashes"]) & set(right["message_pair_hashes"]):
        reasons.append("overlapping_message_pair")
    if same_chat and set(left["long_message_hashes"]) & set(right["long_message_hashes"]):
        reasons.append("overlapping_long_message")
    if left["reply_hash"] == right["reply_hash"]:
        reasons.append("same_answer")
    if set(left["reply_message_hashes"]) & set(right["reply_message_hashes"]):
        reasons.append("answer_message_leak")
    return reasons


def seal_holdout_guard(replay_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("holdout guard already exists; refusing to overwrite")
    replay = _load_json(replay_path)
    cases = list(replay.get("cases") or [])
    if len(cases) != 160:
        raise ValueError(f"fixed holdout must contain exactly 160 cases, got {len(cases)}")
    counts = Counter(str(row.get("relationship")) for row in cases)
    if counts != Counter({"private": 80, "group": 80}):
        raise ValueError(f"fixed holdout relationship counts are invalid: {dict(counts)}")
    signatures = [signature_from_replay_case(row) for row in cases]
    guard = {
        "schema_version": 1,
        "kind": "judge_feature_classifier_holdout_hash_guard",
        "privacy": "hashes only; fixed holdout plaintext is not copied",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        "case_count": len(signatures),
        "relationship_counts": dict(counts),
        "signatures": signatures,
    }
    guard["guard_sha256"] = _json_sha256(guard)
    _write_private(output_path, guard)
    return guard


def validate_guard(guard: dict[str, Any]) -> None:
    if guard.get("case_count") != 160:
        raise ValueError("holdout guard must contain exactly 160 cases")
    if Counter(guard.get("relationship_counts") or {}) != Counter({"private": 80, "group": 80}):
        raise ValueError("holdout guard must contain 80 private and 80 group cases")
    expected = str(guard.get("guard_sha256") or "")
    copy = dict(guard)
    copy.pop("guard_sha256", None)
    if expected != _json_sha256(copy):
        raise ValueError("holdout guard checksum mismatch")
    serialized = json.dumps(guard, ensure_ascii=False)
    forbidden = ("context_original", "human_reply_original", "option_A", "option_B")
    if any(field in serialized for field in forbidden):
        raise ValueError("holdout guard contains plaintext benchmark fields")


def candidate_overlaps(candidate: Candidate, signatures: Iterable[dict[str, Any]]) -> list[str]:
    own = event_signature(candidate)
    reasons: list[str] = []
    for signature in signatures:
        reasons.extend(signatures_overlap(own, signature))
    return sorted(set(reasons))


def load_few_shot_exclusion_index(path: Path) -> dict[str, dict[str, set[str]]]:
    index: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set)
        for name in (
            "timestamp",
            "full_context",
            "message_pair",
            "long_message",
            "reply",
            "reply_message",
        )
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            few_shot_id = str(row["id"])
            signature = event_signature(Candidate(
                relation=str(row.get("relationship") or ""),
                chat_id=str(row.get("chat_id") or ""),
                context=list(row.get("context") or []),
                replies=list(row.get("reply") or []),
                score=0.0,
                timestamp=int(row.get("timestamp") or 0),
            ))
            reply_signature = _few_shot_reply_signature(Candidate(
                relation=str(row.get("relationship") or ""),
                chat_id=str(row.get("chat_id") or ""),
                context=[],
                replies=list(row.get("reply") or []),
                score=0.0,
                timestamp=0,
            ))
            chat_hash = signature["chat_hash"]
            index["timestamp"][signature["timestamp_hash"]].add(few_shot_id)
            index["full_context"][f"{chat_hash}:{signature['full_context_hash']}"].add(
                few_shot_id
            )
            for value in signature["message_pair_hashes"]:
                index["message_pair"][f"{chat_hash}:{value}"].add(few_shot_id)
            for value in signature["long_message_hashes"]:
                index["long_message"][f"{chat_hash}:{value}"].add(few_shot_id)
            index["reply"][reply_signature["reply_hash"]].add(few_shot_id)
            for value in reply_signature["reply_message_hashes"]:
                index["reply_message"][value].add(few_shot_id)
    return index


def few_shot_exclude_ids(
    candidate: Candidate,
    index: dict[str, dict[str, set[str]]],
) -> set[str]:
    signature = event_signature(candidate)
    reply_signature = _few_shot_reply_signature(candidate)
    chat_hash = signature["chat_hash"]
    excluded = set(index["timestamp"].get(signature["timestamp_hash"], set()))
    excluded.update(index["full_context"].get(
        f"{chat_hash}:{signature['full_context_hash']}", set()
    ))
    for value in signature["message_pair_hashes"]:
        excluded.update(index["message_pair"].get(f"{chat_hash}:{value}", set()))
    for value in signature["long_message_hashes"]:
        excluded.update(index["long_message"].get(f"{chat_hash}:{value}", set()))
    excluded.update(index["reply"].get(reply_signature["reply_hash"], set()))
    for value in reply_signature["reply_message_hashes"]:
        excluded.update(index["reply_message"].get(value, set()))
    return excluded


def injected_material_diagnostic(
    human_reply: list[str], output: dict[str, Any]
) -> dict[str, Any]:
    trace = output.get("trace") or {}
    fields = {
        "few_shot": ("few_shot_content",),
        "system_prompt": ("system_prompt",),
        "skills": ("skill_content",),
        "tools": ("tools_context", "tool_calls"),
    }
    material_types = sorted(
        material_type
        for material_type, keys in fields.items()
        if any(
            reference_material_contains_reply(human_reply, trace.get(key))
            for key in keys
            if trace.get(key)
        )
    )
    return {
        "material_types": material_types,
        "few_shot_ids": list(output.get("few_shot_ids") or []) if "few_shot" in material_types else [],
    }


def _labeled_reference_replies(value: Any) -> list[str]:
    labeled: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if any(marker in normalized_key for marker in (
                "answer", "reply", "response", "output", "option_a", "option_b"
            )):
                labeled.extend(_all_reference_strings(item))
            else:
                labeled.extend(_labeled_reference_replies(item))
    elif isinstance(value, list):
        for item in value:
            labeled.extend(_labeled_reference_replies(item))
    elif isinstance(value, str) and "<" in value:
        try:
            root = ET.fromstring(f"<root>{value}</root>")
        except ET.ParseError:
            return []
        for element in root.iter():
            if element.tag.lower() not in {
                "answer", "reply", "response", "output", "option_a", "option_b"
            }:
                continue
            text = html.unescape("".join(element.itertext())).strip()
            if text:
                labeled.append(text)
    return labeled


def _all_reference_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _all_reference_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _all_reference_strings(item)]
    return []


def reference_material_contains_reply(
    human_reply: list[str], material: Any, *, minimum_standalone_chars: int = 8
) -> bool:
    messages = [
        normalized for raw in human_reply
        if (normalized := _normalized_reply(raw))
    ]
    if not messages or material is None:
        return False
    normalized_material = _normalized_reply(json.dumps(material, ensure_ascii=False))
    if any(
        len(message) >= minimum_standalone_chars and message in normalized_material
        for message in messages
    ):
        return True
    labeled = {
        normalized for raw in _labeled_reference_replies(material)
        if (normalized := _normalized_reply(raw))
    }
    return any(message in labeled for message in messages)


def detect_experiment_degenerate_case(
    human_reply: list[str], output: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if injected_material_diagnostic(human_reply, output)["material_types"]:
        reasons.append("target human reply appeared in C0 reference material")
    human = _normalized_reply(human_reply)
    ai = _normalized_reply(output.get("replies", []))
    if human and ai:
        similarity = difflib.SequenceMatcher(None, human, ai).ratio()
        if human == ai or (min(len(human), len(ai)) >= 6 and similarity >= 0.9):
            reasons.append(f"C0 reproduced the human reply (similarity={similarity:.3f})")
    return reasons


def eligible_candidates(
    exports: Path,
    wiki: Path,
    guard: dict[str, Any],
) -> tuple[list[Candidate], Counter[str]]:
    validate_guard(guard)
    candidates = extract_candidates(exports, wiki, human_before=CUTOFF)
    few_shot_exact = _few_shot_identities(PROJECT_ROOT / "data")
    holdout_signatures = list(guard["signatures"])
    outcomes: Counter[str] = Counter()
    eligible: list[Candidate] = []
    for row in candidates:
        identity = (row.relation, row.chat_id, tuple(row.context), tuple(row.replies))
        if identity in few_shot_exact:
            outcomes["exact_c0_few_shot"] += 1
            continue
        if not (
            2 <= len("".join(row.replies)) <= 80
            and 2 <= len("".join(row.context)) <= 240
            and row.timestamp
            and row.timestamp / (1000 if row.timestamp > 10_000_000_000 else 1) < CUTOFF
        ):
            outcomes["basic_filter"] += 1
            continue
        overlap = candidate_overlaps(row, holdout_signatures)
        if overlap:
            outcomes.update(f"holdout:{reason}" for reason in overlap)
            continue
        eligible.append(row)
    return eligible, outcomes


def round_robin(candidates: list[Candidate], relation: str) -> list[Candidate]:
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidate_relation = "group" if candidate.relation == "group" else "private"
        if candidate_relation == relation:
            by_chat[candidate.chat_id].append(candidate)
    for chat_id, rows in by_chat.items():
        rows.sort(key=lambda row: _rank(f"case:{chat_id}:{_case_id(row)}"))
    chat_ids = sorted(by_chat, key=lambda chat_id: _rank(f"chat:{relation}:{chat_id}"))
    queue: list[Candidate] = []
    depth = 0
    while True:
        additions = [by_chat[chat_id][depth] for chat_id in chat_ids if depth < len(by_chat[chat_id])]
        if not additions:
            return queue
        queue.extend(additions)
        depth += 1


def select_nonoverlapping(
    queue: list[Candidate],
    target: int,
    existing: list[dict[str, Any]],
) -> tuple[list[Candidate], list[dict[str, Any]], Counter[str]]:
    selected: list[Candidate] = []
    signatures = list(existing)
    rejected: Counter[str] = Counter()
    for candidate in queue:
        overlap = candidate_overlaps(candidate, signatures)
        if overlap:
            rejected.update(overlap)
            continue
        selected.append(candidate)
        signatures.append(event_signature(candidate))
        if len(selected) == target:
            return selected, signatures, rejected
    raise ValueError(f"only {len(selected)} non-overlapping cases available; need {target}")


def initialize_manifest(
    guard_path: Path,
    output_path: Path,
    exports: Path,
    wiki: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("experiment manifest already exists; refusing to overwrite")
    guard = _load_json(guard_path)
    candidates, exclusions = eligible_candidates(exports, wiki, guard)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    split_rejections: Counter[str] = Counter()
    queues = {relation: round_robin(candidates, relation) for relation in RELATIONS}
    signatures: list[dict[str, Any]] = []
    validation_ids: dict[str, set[str]] = {}
    for relation in RELATIONS:
        validation, signatures, rejected = select_nonoverlapping(
            queues[relation], VALIDATION_TARGETS[relation], signatures
        )
        validation_ids[relation] = {_case_id(row) for row in validation}
        splits["validation"].extend(candidate_record(row) for row in validation)
        split_rejections.update(rejected)

    for relation in RELATIONS:
        train_queue = [
            row for row in queues[relation]
            if _case_id(row) not in validation_ids[relation]
        ]
        train, signatures, train_rejected = select_nonoverlapping(
            train_queue, TRAIN_TARGETS[relation], signatures
        )
        split_rejections.update(train_rejected)
        splits["train"].extend(candidate_record(row) for row in train)
    validate_split_records(splits, guard)
    config = load_current_c0()
    manifest = {
        "schema_version": 1,
        "kind": "judge_feature_classifier_development_manifest",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "protocol": {
            "train_targets": TRAIN_TARGETS,
            "validation_targets": VALIDATION_TARGETS,
            "learning_curve_totals": LEARNING_CURVE_TOTALS,
            "same_chat_nonoverlap_allowed": True,
            "maximum_cases_per_chat": None,
            "fixed_holdout_plaintext_forbidden_before_promotion": True,
        },
        "sources": {
            "holdout_guard_sha256": guard["guard_sha256"],
            "current_c0_baseline_id": current_c0_baseline_id(),
            "c0_generator_sha256": current_c0_sha256(),
            "few_shot_path": config["few_shot_path"],
            "eligible_candidates": len(candidates),
            "eligibility_exclusions": dict(exclusions),
            "split_overlap_rejections": dict(split_rejections),
        },
        "splits": splits,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    _write_private(output_path, manifest)
    return manifest


def validate_split_records(splits: dict[str, list[dict[str, Any]]], guard: dict[str, Any]) -> None:
    expected = {"train": TRAIN_TOTAL, "validation": VALIDATION_TOTAL}
    for split, count in expected.items():
        if len(splits.get(split) or []) != count:
            raise ValueError(f"{split} must contain exactly {count} cases")
        relation_counts = Counter(str(row["relationship"]) for row in splits[split])
        targets = TRAIN_TARGETS if split == "train" else VALIDATION_TARGETS
        if relation_counts != Counter(targets):
            raise ValueError(f"{split} relationship counts are invalid: {dict(relation_counts)}")
    case_ids = [str(row["case_id"]) for rows in splits.values() for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("development splits contain duplicate case ids")
    holdout = list(guard["signatures"])
    chosen: list[dict[str, Any]] = []
    for row in splits["validation"] + splits["train"]:
        signature = event_signature(candidate_from_record(row))
        if any(signatures_overlap(signature, other) for other in holdout):
            raise ValueError("development case overlaps fixed holdout")
        if any(signatures_overlap(signature, other) for other in chosen):
            raise ValueError("training and validation contain an overlapping event")
        chosen.append(signature)


def validate_manifest(manifest: dict[str, Any], guard: dict[str, Any]) -> None:
    expected = str(manifest.get("manifest_sha256") or "")
    copy = dict(manifest)
    copy.pop("manifest_sha256", None)
    if expected != _json_sha256(copy):
        raise ValueError("manifest checksum mismatch")
    if manifest.get("sources", {}).get("holdout_guard_sha256") != guard.get("guard_sha256"):
        raise ValueError("manifest was not built against this holdout guard")
    if manifest.get("sources", {}).get("current_c0_baseline_id") != current_c0_baseline_id():
        raise ValueError("adopted C0 baseline changed after initialization")
    if manifest.get("sources", {}).get("c0_generator_sha256") != current_c0_sha256():
        raise ValueError("C0 generator changed after initialization")
    validate_split_records(manifest["splits"], guard)


def _new_c0_checkpoint(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "current_c0_baseline_id": current_c0_baseline_id(),
        "generation_input_contract": C0_GENERATION_INPUT_CONTRACT,
        "entries": {},
    }


def _load_c0_checkpoint(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _load_json(path) if path.exists() else _new_c0_checkpoint(manifest)
    if checkpoint.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("C0 checkpoint does not match manifest")
    if checkpoint.get("current_c0_baseline_id") != current_c0_baseline_id():
        raise ValueError("C0 checkpoint does not use the adopted baseline")
    if checkpoint.get("generation_input_contract") != C0_GENERATION_INPUT_CONTRACT:
        raise ValueError("C0 checkpoint does not use the adopted generation input contract")
    return checkpoint


def should_retry(entry: dict[str, Any] | None, retry_failed: bool) -> bool:
    if entry is None:
        return True
    return retry_failed and entry.get("status") != "valid"


def shard_rows(
    rows: list[tuple[str, dict[str, Any]]], shard_index: int, shard_count: int
) -> list[tuple[str, dict[str, Any]]]:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index must be within shard count")
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


def merge_c0_checkpoint_data(
    target: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    for key in (
        "schema_version",
        "manifest_sha256",
        "current_c0_baseline_id",
        "generation_input_contract",
    ):
        if target.get(key) != source.get(key):
            raise ValueError(f"C0 checkpoint {key} mismatch")
    merged = dict(target)
    entries = dict(target.get("entries") or {})
    for case_id, entry in (source.get("entries") or {}).items():
        if case_id in entries and entries[case_id] != entry:
            raise ValueError(f"C0 checkpoint entry conflict: {case_id}")
        entries[case_id] = entry
    merged["entries"] = entries
    return merged


def run_current_c0_for_judge_experiment(
    candidate: Candidate,
    payload: dict[str, Any],
    generator: Any,
    excluded_few_shot_ids: set[str],
) -> dict[str, Any]:
    return _run_c0(
        candidate,
        payload,
        force_reply=True,
        reply_generator=generator,
        few_shot_exclude_ids=excluded_few_shot_ids,
        compact_verified_turn=True,
    )


def effective_input_compatibility(
    entry: dict[str, Any],
    candidate: Candidate,
    payload: dict[str, Any],
) -> tuple[bool, dict[str, int]]:
    to_reply, all_messages, metadata = rebuild_generation_inputs(
        candidate, payload, compact_verified_turn=True
    )
    replay = entry.get("replay_case") or {}
    old_metadata = ((replay.get("c0_trace") or {}).get("input_metadata") or {})
    old_context = list(replay.get("context_original") or [])
    new_context = [message.text for message in to_reply]
    diagnostic = {
        "previous_unread_count": int(old_metadata.get("unread_count") or 0),
        "required_unread_count": int(metadata["unread_count"]),
        "previous_history_count": int(old_metadata.get("history_count") or 0),
        "required_history_count": len(all_messages),
    }
    compatible = (
        old_context == new_context
        and diagnostic["previous_history_count"] == diagnostic["required_history_count"]
    )
    return compatible, diagnostic


def upgrade_c0_checkpoint_input_contract(
    manifest_path: Path,
    guard_path: Path,
    source_path: Path,
    target_path: Path,
    exports: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    guard = _load_json(guard_path)
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, guard)
    source = _load_json(source_path)
    if source.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("source C0 checkpoint does not match manifest")
    if source.get("current_c0_baseline_id") != current_c0_baseline_id():
        raise ValueError("source C0 checkpoint does not use the adopted baseline")

    payloads = _export_payloads(exports)
    checkpoint = _new_c0_checkpoint(manifest)
    rows = [
        (split, row)
        for split in ("validation", "train")
        for row in manifest["splits"][split]
    ]
    rows = shard_rows(rows, shard_index, shard_count)
    outcomes: Counter[str] = Counter()
    for split, row in rows:
        case_id = str(row["case_id"])
        entry = (source.get("entries") or {}).get(case_id)
        if entry is None:
            outcomes["missing"] += 1
            continue
        if entry.get("status") != "valid":
            checkpoint["entries"][case_id] = entry
            outcomes[str(entry.get("status") or "unknown")] += 1
            continue
        candidate = candidate_from_record(row)
        payload = payloads.get(candidate.chat_id)
        if payload is None:
            checkpoint["entries"][case_id] = {
                "split": split,
                "relationship": row["relationship"],
                "status": "missing_export",
            }
            outcomes["missing_export"] += 1
            continue
        try:
            compatible, diagnostic = effective_input_compatibility(
                entry, candidate, payload
            )
        except Exception as exc:
            checkpoint["entries"][case_id] = {
                "split": split,
                "relationship": row["relationship"],
                "status": "input_contract_error",
                "error_type": type(exc).__name__,
            }
            outcomes["input_contract_error"] += 1
            continue
        if compatible:
            checkpoint["entries"][case_id] = entry
            outcomes["reused_valid"] += 1
        else:
            checkpoint["entries"][case_id] = {
                "split": split,
                "relationship": row["relationship"],
                "status": "input_contract_mismatch",
                "input_count_diagnostic": diagnostic,
            }
            outcomes["input_contract_mismatch"] += 1
    _write_private(target_path, checkpoint)
    return {
        "owned_cases": len(rows),
        "checkpoint_entries": len(checkpoint["entries"]),
        "outcomes": dict(sorted(outcomes.items())),
    }


def generate_c0(
    manifest_path: Path,
    guard_path: Path,
    checkpoint_path: Path,
    exports: Path,
    volcengine_config: Path,
    max_calls: int | None,
    retry_failed: bool,
    relation: str | None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    from src.memory import MemoryEngine
    from src.tools.tool_registry import ToolRegistry

    guard = _load_json(guard_path)
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, guard)
    checkpoint = _load_c0_checkpoint(checkpoint_path, manifest)
    rows = [
        (split, row)
        for split in ("validation", "train")
        for row in manifest["splits"][split]
        if relation is None or row["relationship"] == relation
    ]
    rows = shard_rows(rows, shard_index, shard_count)
    config = load_current_c0()
    private_client, group_client = create_current_c0_clients(
        config=config, volcengine_config=volcengine_config
    )
    private_client = DiagnosticChatClient(private_client)
    group_client = DiagnosticChatClient(group_client)
    memory = MemoryEngine(llm_client=private_client)
    generator = create_current_c0_generator(
        config=config,
        volcengine_config=volcengine_config,
        private_client=private_client,
        group_client=group_client,
        memory_engine=memory,
        tool_registry=ToolRegistry(),
        judge_worker=None,
        enable_timestamps=True,
    )
    few_shot_index = load_few_shot_exclusion_index(Path(config["few_shot_path"]))
    payloads = _export_payloads(exports)
    entries = checkpoint["entries"]
    calls = 0
    try:
        for split, row in rows:
            case_id = str(row["case_id"])
            if not should_retry(entries.get(case_id), retry_failed):
                continue
            if max_calls is not None and calls >= max_calls:
                return c0_status(manifest, checkpoint, calls)
            candidate = candidate_from_record(row)
            payload = payloads.get(candidate.chat_id)
            if payload is None:
                entries[case_id] = {"split": split, "relationship": row["relationship"], "status": "missing_export"}
                _write_private(checkpoint_path, checkpoint)
                continue
            calls += 1
            try:
                private_client.reset_diagnostics()
                group_client.reset_diagnostics()
                output = run_current_c0_for_judge_experiment(
                    candidate,
                    payload,
                    generator,
                    few_shot_exclude_ids(candidate, few_shot_index),
                )
                if output["generation_failed"] or not output["replies"]:
                    status = "generation_failed" if output["generation_failed"] else "silent"
                    entries[case_id] = {
                        "split": split,
                        "relationship": row["relationship"],
                        "status": status,
                        "duration_seconds": output["duration_seconds"],
                        "failure_diagnostic": c0_failure_diagnostic(
                            output,
                            group_client if row["relationship"] == "group" else private_client,
                        ),
                    }
                else:
                    reasons = detect_experiment_degenerate_case(candidate.replies, output)
                    if reasons:
                        entries[case_id] = {
                            "split": split,
                            "relationship": row["relationship"],
                            "status": "degenerate",
                            "reasons": reasons,
                            "input_leak_diagnostic": injected_material_diagnostic(
                                candidate.replies, output
                            ),
                        }
                    else:
                        entries[case_id] = {
                            "split": split,
                            "relationship": row["relationship"],
                            "status": "valid",
                            "replay_case": {
                                "case_id": case_id,
                                "source_chat_id": candidate.chat_id,
                                "relationship": row["relationship"],
                                "timestamp": candidate.timestamp,
                                "context_original": output["context_original"],
                                "human_reply_original": candidate.replies,
                                "ai_reply_original": output["replies"],
                                "c0_trace": {
                                    "few_shot_ids": output["few_shot_ids"],
                                    "duration_seconds": output["duration_seconds"],
                                    "input_metadata": output["input_metadata"],
                                    "generation_trace": output["trace"],
                                },
                            },
                        }
            except Exception as exc:
                entries[case_id] = {
                    "split": split,
                    "relationship": row["relationship"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _write_private(checkpoint_path, checkpoint)
    finally:
        memory.shutdown()
    return c0_status(manifest, checkpoint, calls)


def _load_group_pattern_holdout_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = _load_json(path)
    if digest != GROUP_PATTERN_HOLDOUT_MANIFEST_SHA256:
        expected = str(manifest.get("manifest_sha256") or "")
        canonical = dict(manifest)
        canonical.pop("manifest_sha256", None)
        if (
            manifest.get("kind") != "judge_group_pattern_holdout_effective_v1"
            or manifest.get("source_manifest_sha256")
            != GROUP_PATTERN_HOLDOUT_MANIFEST_SHA256
            or expected != _json_sha256(canonical)
        ):
            raise ValueError("fixed 240 manifest checksum mismatch")
    cases = list(manifest.get("cases") or [])
    ids = [str(row.get("internal_id") or "") for row in cases]
    relationships = Counter(
        "group" if row.get("chat_type") == "group" else "private"
        for row in cases
    )
    if len(cases) != GROUP_PATTERN_HOLDOUT_TOTAL or len(set(ids)) != len(ids) or "" in ids:
        raise ValueError("fixed 240 manifest must contain 240 unique internal ids")
    if relationships != Counter(GROUP_PATTERN_HOLDOUT_TARGETS):
        raise ValueError(
            f"fixed 240 relationship counts are invalid: {dict(relationships)}"
        )
    return manifest


def _select_effective_group_pattern_holdout_cases(
    original_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
    training_signatures: list[dict[str, Any]],
    invalid_output_ids: set[str],
    row_has_valid_input: Callable[[dict[str, Any]], bool],
    targets: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = dict(targets or GROUP_PATTERN_HOLDOUT_TARGETS)
    original_signatures = [
        event_signature(_group_pattern_holdout_candidate(row))
        for row in original_cases
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    kept = Counter()
    removed = Counter()
    for row in original_cases:
        relationship = "group" if row.get("chat_type") == "group" else "private"
        case_id = str(row.get("internal_id") or "")
        if case_id in invalid_output_ids:
            removed[f"{relationship}:invalid_output"] += 1
            continue
        if not row_has_valid_input(row):
            removed[f"{relationship}:invalid_input"] += 1
            continue
        selected.append(copy.deepcopy(row))
        selected_ids.add(case_id)
        kept[relationship] += 1

    replacement_signatures: list[dict[str, Any]] = []
    replacement_counts = Counter()
    rejected = Counter()
    original_ids = {str(row.get("internal_id") or "") for row in original_cases}
    for row in candidate_cases:
        relationship = "group" if row.get("chat_type") == "group" else "private"
        if kept[relationship] + replacement_counts[relationship] >= targets[relationship]:
            continue
        case_id = str(row.get("internal_id") or "")
        if not case_id or case_id in original_ids or case_id in selected_ids:
            rejected[f"{relationship}:duplicate_id"] += 1
            continue
        if not row_has_valid_input(row):
            rejected[f"{relationship}:invalid_input"] += 1
            continue
        signature = event_signature(_group_pattern_holdout_candidate(row))
        blocked = training_signatures + original_signatures + replacement_signatures
        if any(signatures_overlap(signature, existing) for existing in blocked):
            rejected[f"{relationship}:overlap"] += 1
            continue
        selected.append(copy.deepcopy(row))
        selected_ids.add(case_id)
        replacement_signatures.append(signature)
        replacement_counts[relationship] += 1

    actual = Counter(
        "group" if row.get("chat_type") == "group" else "private"
        for row in selected
    )
    if actual != Counter(targets):
        raise ValueError(
            "insufficient leak-safe fixed 240 replacements: "
            + json.dumps({"required": targets, "actual": dict(actual)}, sort_keys=True)
        )
    return selected, {
        "kept_original": dict(kept),
        "removed_original": dict(sorted(removed.items())),
        "replacements": dict(replacement_counts),
        "rejected_candidates": dict(sorted(rejected.items())),
    }


def _compatible_group_pattern_holdout_entry(
    row: dict[str, Any], entry: dict[str, Any]
) -> bool:
    if entry.get("status") != "valid":
        return False
    candidate = _group_pattern_holdout_candidate(row)
    replay = entry.get("replay_case") or {}
    return (
        entry.get("relationship") == candidate.relation
        and replay.get("case_id") == str(row.get("internal_id") or "")
        and replay.get("source_chat_id") == candidate.chat_id
        and replay.get("relationship") == candidate.relation
        and int(replay.get("timestamp") or 0) == candidate.timestamp
        and list(replay.get("human_reply_original") or []) == candidate.replies
    )


def prepare_effective_group_pattern_holdout(
    source_manifest_path: Path,
    candidate_manifest_paths: list[Path],
    development_manifest_path: Path,
    source_checkpoint_paths: list[Path],
    output_manifest_path: Path,
    output_checkpoint_path: Path,
) -> dict[str, Any]:
    from scripts.run_situation_c0_ab import _payload_for_case

    source_manifest = _load_group_pattern_holdout_manifest(source_manifest_path)
    source_entries: dict[str, dict[str, Any]] = {}
    for checkpoint_path in source_checkpoint_paths:
        checkpoint = _load_group_pattern_holdout_checkpoint(
            checkpoint_path, source_manifest_path
        )
        for case_id, entry in (checkpoint.get("entries") or {}).items():
            if case_id in source_entries and source_entries[case_id] != entry:
                raise ValueError(f"fixed 240 C0 checkpoint entry conflict: {case_id}")
            source_entries[case_id] = entry

    development = _load_json(development_manifest_path)
    training_signatures = [
        event_signature(candidate_from_record(row))
        for row in development["splits"]["train"]
    ]
    candidate_cases = [
        row
        for path in candidate_manifest_paths
        for row in (_load_json(path).get("cases") or [])
    ]
    source_cache: dict[Path, dict[str, Any]] = {}

    def row_has_valid_input(row: dict[str, Any]) -> bool:
        try:
            candidate = _group_pattern_holdout_candidate(row)
            payload = _payload_for_case(row, "formal240", {}, source_cache)
            if payload is None:
                return False
            rebuild_generation_inputs(
                candidate, payload, compact_verified_turn=True
            )
            return True
        except Exception:
            return False

    invalid_output_ids = {
        case_id
        for case_id, entry in source_entries.items()
        if entry.get("status") != "valid"
    }
    cases, selection = _select_effective_group_pattern_holdout_cases(
        list(source_manifest["cases"]),
        candidate_cases,
        training_signatures,
        invalid_output_ids,
        row_has_valid_input,
    )
    manifest = {
        "schema_version": 1,
        "kind": "judge_group_pattern_holdout_effective_v1",
        "status": "fixed",
        "source_manifest_sha256": str(
            source_manifest.get("source_manifest_sha256")
            or hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
        ),
        "candidate_manifest_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in candidate_manifest_paths
        },
        "development_train_manifest_sha256": str(
            development.get("manifest_sha256") or ""
        ),
        "selection_policy": (
            "keep objectively valid original cases; replace only invalid-input or "
            "non-generatable cases with input-valid candidates that do not overlap the "
            "1,200-case Judge training split, any original fixed case, or another replacement"
        ),
        "selection_audit": selection,
        "cases": cases,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    if output_manifest_path.exists():
        if _load_json(output_manifest_path) != manifest:
            raise ValueError("existing effective fixed 240 manifest differs")
    else:
        _write_private(output_manifest_path, manifest)

    target = _load_group_pattern_holdout_checkpoint(
        output_checkpoint_path, output_manifest_path
    )
    rows_by_id = {str(row["internal_id"]): row for row in cases}
    migrated = 0
    for case_id, entry in source_entries.items():
        row = rows_by_id.get(case_id)
        if row is None or not _compatible_group_pattern_holdout_entry(row, entry):
            continue
        existing = (target.get("entries") or {}).get(case_id)
        if existing is not None and existing != entry:
            raise ValueError(f"effective fixed 240 checkpoint entry conflict: {case_id}")
        if existing is None:
            target["entries"][case_id] = copy.deepcopy(entry)
            migrated += 1
    _write_private(output_checkpoint_path, target)
    return {
        "manifest": str(output_manifest_path),
        "checkpoint": str(output_checkpoint_path),
        "selection": selection,
        "migrated_valid_entries": migrated,
        "checkpoint_status": _group_pattern_holdout_status(manifest, target),
    }


def _group_pattern_holdout_candidate(row: dict[str, Any]) -> Candidate:
    reply = row.get("reply")
    replies = [str(value) for value in reply] if isinstance(reply, list) else [str(reply or "")]
    return Candidate(
        relation="group" if row.get("chat_type") == "group" else "private",
        chat_id=str(row["chat_id"]),
        context=[str(value) for value in row.get("context") or []],
        replies=replies,
        score=0.0,
        timestamp=int(row["timestamp"]),
        source_provenance="fixed_adversarial_stage2_holdout_v1",
    )


def _new_group_pattern_holdout_checkpoint(manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixed_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "current_c0_baseline_id": current_c0_baseline_id(),
        "c0_generator_sha256": current_c0_sha256(),
        "generation_input_contract": C0_GENERATION_INPUT_CONTRACT,
        "entries": {},
    }


def _load_group_pattern_holdout_checkpoint(
    path: Path, manifest_path: Path
) -> dict[str, Any]:
    checkpoint = (
        _load_json(path)
        if path.exists()
        else _new_group_pattern_holdout_checkpoint(manifest_path)
    )
    expected = _new_group_pattern_holdout_checkpoint(manifest_path)
    for key in (
        "fixed_manifest_sha256",
        "current_c0_baseline_id",
        "c0_generator_sha256",
        "generation_input_contract",
    ):
        if checkpoint.get(key) != expected[key]:
            raise ValueError(f"fixed 240 C0 checkpoint {key} mismatch")
    return checkpoint


def _group_pattern_holdout_status(
    manifest: dict[str, Any], checkpoint: dict[str, Any], calls: int = 0
) -> dict[str, Any]:
    entries = checkpoint.get("entries") or {}
    outcomes = Counter(str(row.get("status") or "unknown") for row in entries.values())
    valid = Counter(
        str(row.get("relationship"))
        for row in entries.values()
        if row.get("status") == "valid"
    )
    return {
        "api_calls_this_run": calls,
        "outcomes": dict(outcomes),
        "valid": {relation: valid[relation] for relation in RELATIONS},
        "required": dict(GROUP_PATTERN_HOLDOUT_TARGETS),
        "manifest_cases": len(manifest["cases"]),
    }


def generate_group_pattern_holdout(
    manifest_path: Path,
    checkpoint_path: Path,
    volcengine_config: Path,
    max_calls: int | None,
    retry_failed: bool,
    relation: str | None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    from scripts.run_situation_c0_ab import _payload_for_case
    from src.memory import MemoryEngine
    from src.tools.tool_registry import ToolRegistry

    manifest = _load_group_pattern_holdout_manifest(manifest_path)
    checkpoint = _load_group_pattern_holdout_checkpoint(checkpoint_path, manifest_path)
    rows = [
        (
            "group" if row.get("chat_type") == "group" else "private",
            row,
        )
        for row in manifest["cases"]
        if relation is None
        or ("group" if row.get("chat_type") == "group" else "private") == relation
    ]
    rows = shard_rows(rows, shard_index, shard_count)
    config = load_current_c0()
    private_client, group_client = create_current_c0_clients(
        config=config, volcengine_config=volcengine_config
    )
    private_client = DiagnosticChatClient(private_client)
    group_client = DiagnosticChatClient(group_client)
    memory = MemoryEngine(llm_client=private_client)
    generator = create_current_c0_generator(
        config=config,
        volcengine_config=volcengine_config,
        private_client=private_client,
        group_client=group_client,
        memory_engine=memory,
        tool_registry=ToolRegistry(),
        judge_worker=None,
        enable_timestamps=True,
    )
    few_shot_index = load_few_shot_exclusion_index(Path(config["few_shot_path"]))
    source_cache: dict[Path, dict[str, Any]] = {}
    entries = checkpoint["entries"]
    calls = 0
    try:
        for relationship, row in rows:
            case_id = str(row["internal_id"])
            if not should_retry(entries.get(case_id), retry_failed):
                continue
            if max_calls is not None and calls >= max_calls:
                return _group_pattern_holdout_status(manifest, checkpoint, calls)
            candidate = _group_pattern_holdout_candidate(row)
            payload = _payload_for_case(row, "formal240", {}, source_cache)
            if payload is None:
                entries[case_id] = {
                    "split": "holdout",
                    "relationship": relationship,
                    "status": "missing_export",
                }
                _write_private(checkpoint_path, checkpoint)
                continue
            calls += 1
            try:
                private_client.reset_diagnostics()
                group_client.reset_diagnostics()
                output = run_current_c0_for_judge_experiment(
                    candidate,
                    payload,
                    generator,
                    few_shot_exclude_ids(candidate, few_shot_index),
                )
                if output["generation_failed"] or not output["replies"]:
                    status = "generation_failed" if output["generation_failed"] else "silent"
                    entries[case_id] = {
                        "split": "holdout",
                        "relationship": relationship,
                        "status": status,
                        "duration_seconds": output["duration_seconds"],
                        "failure_diagnostic": c0_failure_diagnostic(
                            output,
                            group_client if relationship == "group" else private_client,
                        ),
                    }
                else:
                    reasons = detect_experiment_degenerate_case(candidate.replies, output)
                    if reasons:
                        entries[case_id] = {
                            "split": "holdout",
                            "relationship": relationship,
                            "status": "degenerate",
                            "reasons": reasons,
                            "input_leak_diagnostic": injected_material_diagnostic(
                                candidate.replies, output
                            ),
                        }
                    else:
                        entries[case_id] = {
                            "split": "holdout",
                            "relationship": relationship,
                            "status": "valid",
                            "replay_case": {
                                "case_id": case_id,
                                "source_chat_id": candidate.chat_id,
                                "relationship": relationship,
                                "timestamp": candidate.timestamp,
                                "context_original": output["context_original"],
                                "human_reply_original": candidate.replies,
                                "ai_reply_original": output["replies"],
                                "c0_trace": {
                                    "few_shot_ids": output["few_shot_ids"],
                                    "duration_seconds": output["duration_seconds"],
                                    "input_metadata": output["input_metadata"],
                                    "generation_trace": output["trace"],
                                },
                            },
                        }
            except Exception as exc:
                entries[case_id] = {
                    "split": "holdout",
                    "relationship": relationship,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _write_private(checkpoint_path, checkpoint)
    finally:
        memory.shutdown()
    return _group_pattern_holdout_status(manifest, checkpoint, calls)


def merge_group_pattern_holdout_checkpoints(
    manifest_path: Path, target_path: Path, source_paths: list[Path]
) -> dict[str, Any]:
    manifest = _load_group_pattern_holdout_manifest(manifest_path)
    target = _load_group_pattern_holdout_checkpoint(target_path, manifest_path)
    entries = dict(target.get("entries") or {})
    for source_path in source_paths:
        source = _load_group_pattern_holdout_checkpoint(source_path, manifest_path)
        for case_id, entry in (source.get("entries") or {}).items():
            if case_id in entries and entries[case_id] != entry:
                raise ValueError(f"fixed 240 C0 checkpoint entry conflict: {case_id}")
            entries[case_id] = entry
    target["entries"] = entries
    _write_private(target_path, target)
    return _group_pattern_holdout_status(manifest, target)


def merge_c0_checkpoints(
    manifest_path: Path,
    guard_path: Path,
    target_path: Path,
    source_paths: list[Path],
) -> dict[str, Any]:
    guard = _load_json(guard_path)
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, guard)
    checkpoint = _load_c0_checkpoint(target_path, manifest)
    before = len(checkpoint.get("entries") or {})
    for source_path in source_paths:
        source = _load_c0_checkpoint(source_path, manifest)
        checkpoint = merge_c0_checkpoint_data(checkpoint, source)
    _write_private(target_path, checkpoint)
    result = c0_status(manifest, checkpoint)
    result["merged_sources"] = len(source_paths)
    result["entries_added"] = len(checkpoint.get("entries") or {}) - before
    return result


def merge_owned_c0_shards(
    manifest_path: Path,
    guard_path: Path,
    target_path: Path,
    source_paths: list[Path],
) -> dict[str, Any]:
    if not source_paths:
        raise ValueError("at least one C0 shard is required")
    guard = _load_json(guard_path)
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, guard)
    sources = [_load_c0_checkpoint(path, manifest) for path in source_paths]
    checkpoint = _new_c0_checkpoint(manifest)
    entries = checkpoint["entries"]
    rows = [
        row
        for split in ("validation", "train")
        for row in manifest["splits"][split]
    ]
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        entry = (sources[index % len(sources)].get("entries") or {}).get(case_id)
        if entry is not None:
            entries[case_id] = entry
    _write_private(target_path, checkpoint)
    result = c0_status(manifest, checkpoint)
    result["owned_sources"] = len(sources)
    result["owned_entries"] = len(entries)
    return result


def c0_status(manifest: dict[str, Any], checkpoint: dict[str, Any], calls: int = 0) -> dict[str, Any]:
    entries = checkpoint.get("entries") or {}
    outcomes = Counter(str(row.get("status") or "unknown") for row in entries.values())
    valid = Counter((row.get("split"), row.get("relationship")) for row in entries.values() if row.get("status") == "valid")
    return {
        "api_calls_this_run": calls,
        "outcomes": dict(outcomes),
        "valid": {f"{split}:{relation}": valid[(split, relation)] for split in ("train", "validation") for relation in RELATIONS},
        "required": {
            "train:private": TRAIN_TARGETS["private"],
            "train:group": TRAIN_TARGETS["group"],
            "validation:private": VALIDATION_TARGETS["private"],
            "validation:group": VALIDATION_TARGETS["group"],
        },
        "manifest_cases": sum(len(rows) for rows in manifest["splits"].values()),
    }


def audit_c0_checkpoint_for_freeze(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    exports: Path,
) -> dict[str, Any]:
    payloads = _export_payloads(exports)
    trace_contracts: Counter[str] = Counter()
    issues: list[dict[str, str]] = []
    checked = 0
    for split in ("validation", "train"):
        for row in manifest["splits"][split]:
            case_id = str(row["case_id"])
            entry = (checkpoint.get("entries") or {}).get(case_id) or {}
            status = str(entry.get("status") or "missing")
            if status != "valid":
                issues.append({"case_id": case_id, "reason": f"status:{status}"})
                continue
            candidate = candidate_from_record(row)
            replay = entry.get("replay_case") or {}
            payload = payloads.get(candidate.chat_id)
            if payload is None:
                issues.append({"case_id": case_id, "reason": "missing_export"})
                continue
            try:
                compatible, _ = effective_input_compatibility(entry, candidate, payload)
            except Exception as exc:
                issues.append({
                    "case_id": case_id,
                    "reason": f"input_contract_error:{type(exc).__name__}",
                })
                continue
            if not compatible:
                issues.append({"case_id": case_id, "reason": "input_contract_mismatch"})
                continue
            metadata = ((replay.get("c0_trace") or {}).get("input_metadata") or {})
            if "compact_verified_turn" not in metadata:
                trace_contracts["missing"] += 1
            else:
                trace_contracts[str(bool(metadata["compact_verified_turn"])).lower()] += 1
            if (
                replay.get("source_chat_id") != candidate.chat_id
                or replay.get("relationship") != row["relationship"]
                or int(replay.get("timestamp") or 0) != candidate.timestamp
                or list(replay.get("human_reply_original") or []) != list(candidate.replies)
            ):
                issues.append({"case_id": case_id, "reason": "manifest_replay_mismatch"})
                continue
            trace = replay.get("c0_trace") or {}
            output = {
                "replies": list(replay.get("ai_reply_original") or []),
                "few_shot_ids": list(trace.get("few_shot_ids") or []),
                "trace": trace.get("generation_trace") or {},
            }
            reasons = detect_experiment_degenerate_case(candidate.replies, output)
            if reasons:
                issues.extend({"case_id": case_id, "reason": reason} for reason in reasons)
                continue
            checked += 1
    if issues:
        counts = Counter(row["reason"].split(":", 1)[0] for row in issues)
        raise ValueError(
            "C0 pre-freeze audit failed: "
            + json.dumps({"issue_counts": dict(sorted(counts.items())), "issues": issues[:20]})
        )
    expected = TRAIN_TOTAL + VALIDATION_TOTAL
    if checked != expected:
        raise ValueError(f"C0 pre-freeze audit checked {checked} cases; expected {expected}")
    return {
        "checked_cases": checked,
        "input_contract_equivalent": checked,
        "trace_compact_verified_turn": dict(sorted(trace_contracts.items())),
        "invalid_status_or_output": 0,
        "reference_material_leaks": 0,
        "human_reply_reproductions": 0,
        "manifest_replay_mismatches": 0,
    }


def freeze_pairs(
    manifest_path: Path,
    guard_path: Path,
    checkpoint_path: Path,
    exports: Path,
    replay_path: Path,
    blind_path: Path,
    answers_path: Path,
) -> dict[str, Any]:
    if any(path.exists() for path in (replay_path, blind_path, answers_path)):
        raise ValueError("development pair artifacts already exist; refusing to overwrite")
    guard = _load_json(guard_path)
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, guard)
    checkpoint = _load_c0_checkpoint(checkpoint_path, manifest)
    audit = audit_c0_checkpoint_for_freeze(manifest, checkpoint, exports)
    entries = checkpoint["entries"]
    replay_cases: list[dict[str, Any]] = []
    split_case_ids: dict[str, list[str]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        for row in manifest["splits"][split]:
            case_id = str(row["case_id"])
            entry = entries.get(case_id) or {}
            if entry.get("status") != "valid":
                raise ValueError(f"C0 case is not valid: {case_id} ({entry.get('status', 'missing')})")
            replay_cases.append(entry["replay_case"])
            split_case_ids[split].append(case_id)
    if len(replay_cases) != TRAIN_TOTAL + VALIDATION_TOTAL:
        raise ValueError("pair freeze requires exactly 1,360 valid C0 cases")
    blind, answers = build_blind_packet(replay_cases, batch_id="judge_feature_classifier_development_v1")
    source_to_split = {
        case_id: split for split, case_ids in split_case_ids.items() for case_id in case_ids
    }
    for answer in answers["cases"]:
        answer["split"] = source_to_split[str(answer["source_case_id"])]
    public_to_split = {
        str(answer["case_id"]): str(answer["split"]) for answer in answers["cases"]
    }
    for blind_case in blind["cases"]:
        blind_case["split"] = public_to_split[str(blind_case["case_id"])]
    blind["instruction"] = (
        "Extract symmetric fixed-schema features for options A and B. "
        "No answer, recommendation, free-text label, or cross-case context is permitted."
    )
    answers["blind_sha256"] = _json_sha256(blind)
    replay = {
        "schema_version": 1,
        "kind": "judge_feature_classifier_development_replay",
        "manifest_sha256": manifest["manifest_sha256"],
        "pre_freeze_audit": audit,
        "cases": replay_cases,
    }
    _write_private(replay_path, replay)
    _write_private(blind_path, blind)
    _write_private(answers_path, answers)
    return {
        "cases": len(replay_cases),
        "blind_sha256": answers["blind_sha256"],
        "pre_freeze_audit": audit,
    }


def audit_group_pattern_holdout_checkpoint_for_freeze(
    manifest: dict[str, Any], checkpoint: dict[str, Any]
) -> dict[str, Any]:
    from scripts.run_situation_c0_ab import _payload_for_case

    source_cache: dict[Path, dict[str, Any]] = {}
    trace_contracts: Counter[str] = Counter()
    issues: list[dict[str, str]] = []
    checked = 0
    for row in manifest["cases"]:
        case_id = str(row["internal_id"])
        entry = (checkpoint.get("entries") or {}).get(case_id) or {}
        status = str(entry.get("status") or "missing")
        if status != "valid":
            issues.append({"case_id": case_id, "reason": f"status:{status}"})
            continue
        candidate = _group_pattern_holdout_candidate(row)
        replay = entry.get("replay_case") or {}
        payload = _payload_for_case(row, "formal240", {}, source_cache)
        if payload is None:
            issues.append({"case_id": case_id, "reason": "missing_export"})
            continue
        try:
            compatible, _ = effective_input_compatibility(entry, candidate, payload)
        except Exception as exc:
            issues.append({
                "case_id": case_id,
                "reason": f"input_contract_error:{type(exc).__name__}",
            })
            continue
        if not compatible:
            issues.append({"case_id": case_id, "reason": "input_contract_mismatch"})
            continue
        metadata = ((replay.get("c0_trace") or {}).get("input_metadata") or {})
        if "compact_verified_turn" not in metadata:
            trace_contracts["missing"] += 1
        else:
            trace_contracts[str(bool(metadata["compact_verified_turn"])).lower()] += 1
        if (
            replay.get("source_chat_id") != candidate.chat_id
            or replay.get("relationship") != candidate.relation
            or int(replay.get("timestamp") or 0) != candidate.timestamp
            or list(replay.get("human_reply_original") or []) != candidate.replies
        ):
            issues.append({"case_id": case_id, "reason": "manifest_replay_mismatch"})
            continue
        trace = replay.get("c0_trace") or {}
        output = {
            "replies": list(replay.get("ai_reply_original") or []),
            "few_shot_ids": list(trace.get("few_shot_ids") or []),
            "trace": trace.get("generation_trace") or {},
        }
        reasons = detect_experiment_degenerate_case(candidate.replies, output)
        if reasons:
            issues.extend({"case_id": case_id, "reason": reason} for reason in reasons)
            continue
        checked += 1
    if issues:
        counts = Counter(row["reason"].split(":", 1)[0] for row in issues)
        raise ValueError(
            "fixed 240 C0 pre-freeze audit failed: "
            + json.dumps({"issue_counts": dict(sorted(counts.items())), "issues": issues[:20]})
        )
    if checked != GROUP_PATTERN_HOLDOUT_TOTAL:
        raise ValueError(
            f"fixed 240 C0 pre-freeze audit checked {checked} cases; "
            f"expected {GROUP_PATTERN_HOLDOUT_TOTAL}"
        )
    return {
        "checked_cases": checked,
        "input_contract_equivalent": checked,
        "trace_compact_verified_turn": dict(sorted(trace_contracts.items())),
        "invalid_status_or_output": 0,
        "reference_material_leaks": 0,
        "human_reply_reproductions": 0,
        "manifest_replay_mismatches": 0,
    }


def freeze_group_pattern_holdout(
    manifest_path: Path,
    checkpoint_path: Path,
    replay_path: Path,
    blind_path: Path,
    answers_path: Path,
) -> dict[str, Any]:
    if any(path.exists() for path in (replay_path, blind_path, answers_path)):
        raise ValueError("fixed 240 artifacts already exist; refusing to overwrite")
    manifest = _load_group_pattern_holdout_manifest(manifest_path)
    checkpoint = _load_group_pattern_holdout_checkpoint(checkpoint_path, manifest_path)
    audit = audit_group_pattern_holdout_checkpoint_for_freeze(manifest, checkpoint)
    entries = checkpoint["entries"]
    replay_cases = [
        entries[str(row["internal_id"])]["replay_case"]
        for row in manifest["cases"]
    ]
    blind, answers = build_blind_packet(
        replay_cases, batch_id="judge_group_pattern_holdout_240_v1"
    )
    for blind_case in blind["cases"]:
        blind_case["split"] = "holdout"
    for answer in answers["cases"]:
        answer["split"] = "holdout"
    blind["instruction"] = (
        "Extract symmetric fixed-schema features for options A and B. "
        "No answer, recommendation, free-text label, or cross-case context is permitted."
    )
    answers["blind_sha256"] = _json_sha256(blind)
    replay = {
        "schema_version": 1,
        "kind": "judge_group_pattern_holdout_240_replay",
        "fixed_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "current_c0_baseline_id": current_c0_baseline_id(),
        "c0_generator_sha256": current_c0_sha256(),
        "pre_freeze_audit": audit,
        "cases": replay_cases,
    }
    relationship_counts = Counter(row["relationship"] for row in replay_cases)
    if relationship_counts != Counter(GROUP_PATTERN_HOLDOUT_TARGETS):
        raise ValueError("fixed 240 replay relationship counts changed during freeze")
    _write_private(replay_path, replay)
    _write_private(blind_path, blind)
    _write_private(answers_path, answers)
    return {
        "cases": len(replay_cases),
        "relationships": dict(relationship_counts),
        "blind_sha256": answers["blind_sha256"],
        "pre_freeze_audit": audit,
    }


def feature_extractor_prompt(case: dict[str, Any]) -> str:
    allowed = {
        "integer_fields": {field: "integer 0..3" for field in INTEGER_FIELDS},
        "boolean_fields": {field: "boolean" for field in BOOLEAN_FIELDS},
        "enum_fields": ENUM_FIELDS,
    }
    safe_case = {
        "relationship": case["relationship"],
        "context": case["context_original"],
        "option_A": case["option_A"],
        "option_B": case["option_B"],
    }
    forbidden_fields = {"human_option", "source_case_id", "answer", "label", "case_id"}
    if set(safe_case) & forbidden_fields:
        raise ValueError("feature request contains a forbidden answer-bearing field")
    prompt = (
        "你是回复特征提取器，不是分类器。你每次只看到一道匿名题。"
        "分别、独立、使用完全相同的字段提取 A 和 B 的可观察特征。"
        "不得判断或暗示哪项是真人，不得推荐 A/B，不得输出自由文本理由。"
        "上下文没有提供的事实必须视为未知。group_pattern_alignment 仅在群聊里衡量复读、队形、接力调侃等承接，"
        "私聊固定填 0。所有评分都只用 0、1、2、3。"
        "以 issue 结尾语义的问题布尔字段（wrong_addressee 等 11 个）逐条只看该候选回复本身是否命中，与另一项无关："
        "wrong_addressee：把对别人的提问当成问自己，或把转述者当成当事人；看发送者、引用、@、话题往返及人物对应。"
        "群内公开话题自然加入不算。"
        "social_tone_misread：把被夸后的自谦当严肃论点认可，或把玩笑当名词求定义；看前几轮谁夸谁、谁在客气、"
        "打趣还是认真请教。"
        "unnecessary_inquiry：上文已指明对象仍问“是什么”，或可查的客观知识反问群友；私人情况、主观意见、"
        "会改变回答的真歧义可问，不算。"
        "missed_pending_question：接了最后一句感叹却漏掉仍待回答的单位、条件或具体办法；已被回答、取消或"
        "明确转走的问题不算。"
        "paraphrase_agreement_without_contribution：把对方内容压成一句总结再评“没错”，没有回答、确认、安慰、"
        "情绪或收束等互动作用；具体确认和合适的情绪应和不算。"
        "self_repetition：本人刚说过的话又发一次或改几个词再说；新请求下的再次确认、新事件下的笑声或强调不算。"
        "mechanical_repetition：没有队形和接梗语境，看到一句就照抄或换说法复述；已有复读队形、有趣段子、"
        "首次合适接梗不算。"
        "address_term_mismatch：拆昵称认人、沿用已纠正的错误资料、知道是谁却用错本人惯用称呼。"
        "unsupported_detail：把别人的经历或无关参考写成本人经历，或编公司规则、持仓、行动与承诺；"
        "有依据的能力推断、明确推测和语境清楚的玩笑不算。"
        "forced_humor_or_defense：追加比喻、反转或针对人的点评，或普通反馈也防御、反击；自然打趣、必要解释、"
        "短句、笑声、粗口不算。"
        "inappropriate_participation：无需回应也强接，或替被问者回答私人情况、替别人作决定；"
        "有明确问题或有用补充时应接住，不算。"
        "只输出一个 JSON 对象，顶层只能是 option_A、option_B。\n"
        "<schema>\n" + json.dumps(allowed, ensure_ascii=False, sort_keys=True) + "\n</schema>\n"
        "<case>\n" + json.dumps(safe_case, ensure_ascii=False) + "\n</case>"
    )
    return prompt


def validate_option_features(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != OPTION_FIELDS:
        raise ValueError("feature option does not match the fixed schema")
    result: dict[str, Any] = {}
    for field in INTEGER_FIELDS:
        field_value = value[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or not 0 <= field_value <= 3:
            raise ValueError(f"{field} must be an integer from 0 through 3")
        result[field] = field_value
    for field in BOOLEAN_FIELDS:
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
        result[field] = value[field]
    for field, allowed in ENUM_FIELDS.items():
        if value[field] not in allowed:
            raise ValueError(f"{field} has an unsupported enum value")
        result[field] = value[field]
    return result


def parse_feature_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("feature extractor did not return JSON")
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict) or set(payload) != {"option_A", "option_B"}:
        raise ValueError("feature response top level must contain only option_A and option_B")
    return {
        "option_A": validate_option_features(payload["option_A"]),
        "option_B": validate_option_features(payload["option_B"]),
    }


def feature_response_json_schema() -> dict[str, Any]:
    option_properties: dict[str, Any] = {
        field: {"type": "integer", "minimum": 0, "maximum": 3}
        for field in INTEGER_FIELDS
    }
    option_properties.update({field: {"type": "boolean"} for field in BOOLEAN_FIELDS})
    option_properties.update({
        field: {"type": "string", "enum": list(allowed)}
        for field, allowed in ENUM_FIELDS.items()
    })
    option_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(OPTION_FIELDS),
        "properties": option_properties,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["option_A", "option_B"],
        "properties": {"option_A": option_schema, "option_B": option_schema},
    }


def run_codex_feature_extractor(
    prompt: str,
    timeout_seconds: int,
    model: str | None,
    reasoning_effort: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="judge-feature-extractor-") as temp_dir:
        last_message = Path(temp_dir) / "last_message.json"
        response_schema = Path(temp_dir) / "feature_response_schema.json"
        response_schema.write_text(
            json.dumps(feature_response_json_schema(), ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "workspace-write", "--cd", temp_dir, "--skip-git-repo-check",
            "--json", "--output-schema", str(response_schema),
            "--output-last-message", str(last_message),
        ]
        if model:
            command.extend(["--model", model])
        if reasoning_effort:
            command.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
        command.append("-")
        total_duration = 0.0
        for attempt in range(1, 3):
            started = time.monotonic()
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            total_duration += time.monotonic() - started
            events = []
            for line in completed.stdout.splitlines():
                if line.strip():
                    events.append(json.loads(line))
            if completed.returncode:
                raise RuntimeError(f"Codex feature extraction failed: {completed.stderr[-1000:]}")
            if any(_event_has_tool_call(event) for event in events):
                raise ValueError("feature extractor attempted a tool call")
            raw = last_message.read_text(encoding="utf-8") if last_message.exists() else ""
            try:
                features = parse_feature_response(raw)
            except ValueError:
                if attempt == 2:
                    raise
                continue
            audit = {
                "duration_seconds": round(total_duration, 3),
                "prompt_sha256": _hash(prompt),
                "returncode": completed.returncode,
                "tool_call_count": sum(_event_has_tool_call(event) for event in events),
                "attempt_count": attempt,
                "schema_retry_count": attempt - 1,
            }
            return features, audit
    raise AssertionError("feature extraction retry loop did not return or raise")


def _new_feature_checkpoint(blind: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "blind_sha256": _json_sha256(blind),
        "feature_schema_sha256": _json_sha256({
            "integer": INTEGER_FIELDS, "boolean": BOOLEAN_FIELDS, "enum": ENUM_FIELDS
        }),
        "entries": {},
    }


def extract_features(
    blind_path: Path,
    checkpoint_path: Path,
    max_cases: int | None,
    retry_failed: bool,
    workers: int,
    timeout_seconds: int,
    model: str | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    checkpoint = _load_json(checkpoint_path) if checkpoint_path.exists() else _new_feature_checkpoint(blind)
    if checkpoint.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("feature checkpoint does not match blind packet")
    entries = checkpoint["entries"]
    pending = [
        case for case in blind["cases"]
        if should_retry(entries.get(str(case["case_id"])), retry_failed)
    ]
    if max_cases is not None:
        pending = pending[:max_cases]

    def run(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        case_id = str(case["case_id"])
        try:
            prompt = feature_extractor_prompt(case)
            features, audit = run_codex_feature_extractor(
                prompt, timeout_seconds, model, reasoning_effort
            )
            return case_id, {
                "status": "valid",
                "split": case["split"],
                "relationship": case["relationship"],
                "features": features,
                "audit": audit,
            }
        except Exception as exc:
            return case_id, {
                "status": "error",
                "split": case.get("split"),
                "relationship": case.get("relationship"),
                "error": f"{type(exc).__name__}: {exc}",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for case_id, entry in executor.map(run, pending):
            entries[case_id] = entry
            _write_private(checkpoint_path, checkpoint)
    outcomes = Counter(str(row.get("status") or "unknown") for row in entries.values())
    return {
        "processed_this_run": len(pending),
        "outcomes": dict(outcomes),
        "required": len(blind["cases"]),
    }


def vectorize_pair(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    left = validate_option_features(features["option_A"])
    right = validate_option_features(features["option_B"])
    values: list[float] = []
    names: list[str] = []
    for field in INTEGER_FIELDS:
        values.append(float(left[field] - right[field]))
        names.append(field)
    for field in BOOLEAN_FIELDS:
        values.append(float(int(left[field]) - int(right[field])))
        names.append(field)
    for field, allowed in ENUM_FIELDS.items():
        for choice in allowed:
            values.append(float(int(left[field] == choice) - int(right[field] == choice)))
            names.append(f"{field}={choice}")
    vector = np.asarray(values, dtype=float)
    swapped = {"option_A": right, "option_B": left}
    swapped_vector, swapped_names = vectorize_pair_without_swap_check(swapped)
    if names != swapped_names or not np.array_equal(swapped_vector, -vector):
        raise ValueError("A/B swap did not strictly negate the feature vector")
    return vector, names


def vectorize_pair_without_swap_check(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    left = validate_option_features(features["option_A"])
    right = validate_option_features(features["option_B"])
    values: list[float] = []
    names: list[str] = []
    for field in INTEGER_FIELDS:
        values.append(float(left[field] - right[field]))
        names.append(field)
    for field in BOOLEAN_FIELDS:
        values.append(float(int(left[field]) - int(right[field])))
        names.append(field)
    for field, allowed in ENUM_FIELDS.items():
        for choice in allowed:
            values.append(float(int(left[field] == choice) - int(right[field] == choice)))
            names.append(f"{field}={choice}")
    return np.asarray(values, dtype=float), names


def reply_bubble_count_difference(case: dict[str, Any]) -> float:
    left = case.get("option_A")
    right = case.get("option_B")
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("reply bubble count requires option_A and option_B lists")
    return float(len(left) - len(right))


def vectorize_option(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    option = validate_option_features(features)
    values: list[float] = []
    names: list[str] = []
    for field in INTEGER_FIELDS:
        values.append(float(option[field]))
        names.append(field)
    for field in BOOLEAN_FIELDS:
        values.append(float(int(option[field])))
        names.append(field)
    for field, allowed in ENUM_FIELDS.items():
        for choice in allowed:
            values.append(float(int(option[field] == choice)))
            names.append(f"{field}={choice}")
    return np.asarray(values, dtype=float), names


def vectorize_options(features: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    left, left_names = vectorize_option(features["option_A"])
    right, right_names = vectorize_option(features["option_B"])
    if left_names != right_names:
        raise ValueError("A/B option feature schemas differ")
    return left, right, left_names


def vectorize_boolean_option(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    option = validate_option_features(features)
    values: list[float] = []
    names: list[str] = []
    for field in INTEGER_FIELDS:
        for threshold in (1, 2, 3):
            values.append(float(option[field] >= threshold))
            names.append(f"{field}>={threshold}")
    for field in BOOLEAN_FIELDS:
        values.append(float(option[field]))
        names.append(field)
    for field, allowed in ENUM_FIELDS.items():
        for choice in allowed:
            values.append(float(option[field] == choice))
            names.append(f"{field}={choice}")
    vector = np.asarray(values, dtype=float)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError("boolean option encoding produced a non-boolean value")
    return vector, names


def vectorize_boolean_options(
    features: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    left, left_names = vectorize_boolean_option(features["option_A"])
    right, right_names = vectorize_boolean_option(features["option_B"])
    if left_names != right_names:
        raise ValueError("A/B boolean option feature schemas differ")
    return left, right, left_names


_SPEAKER_PREFIX = re.compile(
    r"^(.*?)[（(]\d{4}-\d{2}-\d{2}[^）)]*[）)][:：]",
    flags=re.DOTALL,
)


def speaker_timeline(context_original: Any) -> tuple[str, ...]:
    if not isinstance(context_original, list):
        raise ValueError("context_original must be a list of XML fragments")
    speakers: list[str] = []
    for fragment in context_original:
        if not isinstance(fragment, str):
            raise ValueError("context XML fragment must be text")
        root = ET.fromstring(fragment)
        for message in root.iter("message"):
            role = str(message.attrib.get("role") or "")
            if role == "self":
                speakers.append("__self__")
                continue
            text_value = "".join(message.itertext()).strip()
            matched = _SPEAKER_PREFIX.match(text_value)
            speaker = matched.group(1).strip() if matched else "__unknown_other__"
            speakers.append(speaker or "__unknown_other__")
    return tuple(speakers)


def message_text_timeline(context_original: Any) -> tuple[str, ...]:
    if not isinstance(context_original, list):
        raise ValueError("context_original must be a list of XML fragments")
    messages: list[str] = []
    for fragment in context_original:
        if not isinstance(fragment, str):
            raise ValueError("context XML fragment must be text")
        root = ET.fromstring(fragment)
        messages.extend("".join(message.itertext()).strip() for message in root.iter("message"))
    return tuple(messages)


def group_other_text_timeline(context_original: Any) -> tuple[str, ...]:
    if not isinstance(context_original, list):
        raise ValueError("context_original must be a list of XML fragments")
    messages: list[str] = []
    for fragment in context_original:
        if not isinstance(fragment, str):
            raise ValueError("context XML fragment must be text")
        root = ET.fromstring(fragment)
        messages.extend(
            "".join(message.itertext()).strip()
            for message in root.iter("message")
            if message.attrib.get("role") == "other"
            and message.attrib.get("type") == "text"
        )
    return tuple(messages)


def load_context_metadata(
    blind: dict[str, Any],
    answers: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if answers.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("answer key does not match blind packet")
    answer_by_id = {str(row["case_id"]): row for row in answers["cases"]}
    replay_by_source_id = {str(row["case_id"]): row for row in replay["cases"]}
    metadata: dict[str, dict[str, Any]] = {}
    for case in blind["cases"]:
        case_id = str(case["case_id"])
        answer = answer_by_id.get(case_id)
        if answer is None:
            raise ValueError(f"missing answer metadata for case {case_id}")
        source_case_id = str(answer.get("source_case_id") or "")
        source = replay_by_source_id.get(source_case_id)
        if source is None:
            raise ValueError(f"missing replay source for case {case_id}")
        relationship = str(case["relationship"])
        if relationship != str(source.get("relationship")):
            raise ValueError(f"blind and replay relationships differ for case {case_id}")
        if str(case["split"]) != str(answer["split"]):
            raise ValueError(f"blind and answer splits differ for case {case_id}")
        message_texts = message_text_timeline(case["context_original"])
        latest_message = message_texts[-1] if message_texts else ""
        recent_group_other_texts = group_other_text_timeline(
            case["context_original"]
        )[-GROUP_PATTERN_RECENT_MESSAGE_LIMIT:]
        metadata[case_id] = {
            "split": str(case["split"]),
            "relationship": relationship,
            "source_chat_id": str(source.get("source_chat_id") or "__unknown_chat__"),
            "group_name": str(
                ((source.get("c0_trace") or {}).get("input_metadata") or {}).get(
                    "chat_name"
                )
                or "__unknown_group_name__"
            ),
            "recent_speakers": tuple(reversed(speaker_timeline(case["context_original"])))[:RECENT_SPEAKER_POSITIONS],
            "latest_message_has_question_mark": any(
                marker in latest_message for marker in ("?", "？")
            ),
            "latest_message_has_at_mention": "@" in latest_message,
            "recent_group_other_texts": recent_group_other_texts,
        }
    return metadata


def build_context_boolean_schema(
    metadata: dict[str, dict[str, Any]],
    *,
    group_name_min_train_cases: int = GROUP_NAME_MIN_TRAIN_CASES,
    group_member_min_train_occurrences: int = GROUP_MEMBER_MIN_TRAIN_OCCURRENCES,
) -> dict[str, tuple[Any, ...]]:
    group_names: Counter[str] = Counter()
    group_members: Counter[tuple[str, str]] = Counter()
    for row in metadata.values():
        if row["split"] != "train" or row["relationship"] != "group":
            continue
        chat_id = str(row["source_chat_id"])
        group_names[str(row["group_name"])] += 1
        for speaker in row["recent_speakers"][:IDENTIFIED_GROUP_SPEAKER_POSITIONS]:
            if speaker != "__self__":
                group_members[(chat_id, str(speaker))] += 1
    return {
        "known_group_names": tuple(sorted(
            group_name for group_name, count in group_names.items()
            if count >= group_name_min_train_cases
        )),
        "known_group_members": tuple(sorted(
            member for member, count in group_members.items()
            if count >= group_member_min_train_occurrences
        )),
    }


def _private_category_name(prefix: str, value: str) -> str:
    return f"{prefix}={_hash(value)[:12]}"


def vectorize_context_booleans(
    metadata: dict[str, Any], schema: dict[str, tuple[Any, ...]]
) -> tuple[np.ndarray, list[str]]:
    relationship = str(metadata["relationship"])
    chat_id = str(metadata["source_chat_id"])
    group_name = str(metadata["group_name"])
    recent = tuple(str(value) for value in metadata["recent_speakers"])
    known_group_names = tuple(str(value) for value in schema["known_group_names"])
    known_group_members = tuple(
        (str(chat), str(speaker)) for chat, speaker in schema["known_group_members"]
    )
    values: list[float] = [
        float(relationship == "private"),
        float(relationship == "group"),
        float(bool(metadata.get("latest_message_has_question_mark"))),
        float(bool(metadata.get("latest_message_has_at_mention"))),
    ]
    names: list[str] = [
        "is_private",
        "is_group",
        "latest_message_has_question_mark",
        "latest_message_has_at_mention",
    ]
    for known_group_name in known_group_names:
        values.append(float(
            relationship == "group" and group_name == known_group_name
        ))
        names.append(_private_category_name("group_name", known_group_name))
    values.append(float(
        relationship == "group" and group_name not in known_group_names
    ))
    names.append("group_name=unknown")
    for position in range(RECENT_SPEAKER_POSITIONS):
        exists = position < len(recent)
        speaker = recent[position] if exists else ""
        values.extend((
            float(exists),
            float(exists and speaker == "__self__"),
            float(exists and speaker != "__self__"),
        ))
        names.extend((
            f"recent_{position + 1}_exists",
            f"recent_{position + 1}_is_self",
            f"recent_{position + 1}_is_other",
        ))
    for left_position in range(RECENT_SPEAKER_POSITIONS):
        for right_position in range(left_position + 1, RECENT_SPEAKER_POSITIONS):
            values.append(float(
                right_position < len(recent)
                and recent[left_position] == recent[right_position]
            ))
            names.append(
                f"recent_{left_position + 1}_same_speaker_as_{right_position + 1}"
            )
    values.append(float(len(recent) >= 2 and recent[0] != recent[1]))
    names.append("latest_turn_switches_speaker")
    known_member_set = set(known_group_members)
    for position in range(IDENTIFIED_GROUP_SPEAKER_POSITIONS):
        speaker = recent[position] if position < len(recent) else ""
        current_member = (chat_id, speaker)
        for known_chat_id, known_speaker in known_group_members:
            values.append(float(
                relationship == "group"
                and current_member == (known_chat_id, known_speaker)
            ))
            names.append(_private_category_name(
                f"recent_{position + 1}_group_member",
                f"{known_chat_id}\0{known_speaker}",
            ))
        values.append(float(
            relationship == "group"
            and bool(speaker)
            and speaker != "__self__"
            and current_member not in known_member_set
        ))
        names.append(f"recent_{position + 1}_group_member=unknown")
    vector = np.asarray(values, dtype=float)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError("context encoding produced a non-boolean value")
    return vector, names


_GROUP_PATTERN_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")
_GROUP_PATTERN_QUOTE = re.compile(r"\[引用.*$", flags=re.DOTALL)
_GROUP_PATTERN_PUNCTUATION = re.compile(
    r"[\s，,、…~～—\-：:。！？!?；;\"'“”‘’（）()\[\]【】]+"
)


def _group_pattern_sentences(value: str) -> tuple[str, ...]:
    matched = _SPEAKER_PREFIX.match(value)
    body = value[matched.end():] if matched else value
    body = _GROUP_PATTERN_QUOTE.sub("", body).strip()
    while body.startswith("@") and "\u2005" in body:
        body = body.split("\u2005", 1)[1].lstrip()
    body = re.sub(r"^同(?=\s)", "", body).strip()
    sentences = (
        _GROUP_PATTERN_PUNCTUATION.sub("", sentence)
        for sentence in _GROUP_PATTERN_SENTENCE_SPLIT.split(body)
    )
    return tuple(sentence for sentence in sentences if len(sentence) >= 2)


def _has_distinctive_group_pattern_match(left: str, right: str) -> bool:
    if min(len(left), len(right)) < 8:
        return False
    blocks = [
        block.size
        for block in difflib.SequenceMatcher(
            None, left, right, autojunk=False
        ).get_matching_blocks()
        if block.size >= 2
    ]
    return (
        len(blocks) >= 2
        and max(blocks) >= 3
        and sum(blocks) >= 6
        and sum(blocks) / min(len(left), len(right)) >= 0.65
    )


def reply_reuses_active_group_pattern(
    reply: Any, metadata: dict[str, Any]
) -> bool:
    if metadata.get("relationship") != "group":
        return False
    if not isinstance(reply, list) or not all(isinstance(item, str) for item in reply):
        raise ValueError("candidate reply must be a list of strings")
    candidate_sentences = tuple(
        sentence
        for bubble in reply
        for sentence in _group_pattern_sentences(bubble)
    )
    history_by_message = tuple(
        _group_pattern_sentences(str(message))
        for message in metadata.get("recent_group_other_texts") or ()
    )
    exact_counts: Counter[str] = Counter()
    for sentences in history_by_message:
        exact_counts.update(set(sentences))
    if any(exact_counts[sentence] >= 2 for sentence in candidate_sentences):
        return True
    history_sentences = tuple(
        sentence for sentences in history_by_message for sentence in sentences
    )
    return any(
        _has_distinctive_group_pattern_match(candidate, historical)
        for candidate in candidate_sentences
        for historical in history_sentences
    )


def vectorize_group_pattern_option_boolean(
    reply: Any, metadata: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(
        [reply_reuses_active_group_pattern(reply, metadata)], dtype=float
    )
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("group pattern encoding produced a non-boolean value")
    return values, list(GROUP_PATTERN_OPTION_BOOLEAN_NAMES)


def vectorize_observable_option_booleans(
    reply: Any, metadata: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(reply, list) or not all(isinstance(item, str) for item in reply):
        raise ValueError("candidate reply must be a list of strings")
    text = "".join(reply)
    compact_text = "".join(character for character in text if not character.isspace())
    recent = tuple(str(value) for value in metadata.get("recent_speakers") or ())
    latest_speaker = recent[0] if recent else ""
    mentions_latest_speaker = (
        latest_speaker not in {"", "__self__", "__unknown_other__"}
        and latest_speaker in text
    )
    contains_laughter_marker = any(
        marker in text.lower() for marker in ("哈", "笑", "hh", "lol")
    )
    values = np.asarray(
        [
            len(reply) >= 2,
            len(reply) >= 3,
            len(compact_text) >= 8,
            len(compact_text) >= 16,
            len(compact_text) >= 32,
            any(marker in text for marker in ("?", "？")),
            "@" in text,
            any(character.isdigit() for character in text),
            mentions_latest_speaker,
            contains_laughter_marker,
        ],
        dtype=float,
    )
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("observable reply encoding produced a non-boolean value")
    return values, list(OBSERVABLE_OPTION_BOOLEAN_NAMES)


def expand_boolean_option_with_context(
    option: np.ndarray,
    option_names: list[str],
    context: np.ndarray,
    context_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if len(option) != len(option_names) or len(context) != len(context_names):
        raise ValueError("boolean values and feature names are not aligned")
    if not np.isin(option, (0.0, 1.0)).all() or not np.isin(context, (0.0, 1.0)).all():
        raise ValueError("context crosses require boolean inputs")
    crossed = np.multiply.outer(context, option).reshape(-1)
    crossed_names = [
        f"{context_name}*{option_name}"
        for context_name in context_names
        for option_name in option_names
    ]
    return np.concatenate((option, crossed)), [*option_names, *crossed_names]


def refined_boolean_context_crosses(
    option_names: list[str], context_names: list[str]
) -> list[tuple[str, str]]:
    option_set = set(option_names)
    context_set = set(context_names)
    crosses: list[tuple[str, str]] = []

    def add(context_name: str, option_name: str) -> None:
        pair = (context_name, option_name)
        if (
            context_name in context_set
            and option_name in option_set
            and pair not in crosses
        ):
            crosses.append(pair)

    relation_features = (
        "reply_action=answer_now",
        "reply_action=defer",
        "reply_action=accept",
        "reply_action=refuse",
        "reply_action=confirm",
        "reply_action=joke_tease",
        "reply_action=ask",
        "reply_action=clarify",
        "reply_action=supplement",
        "reply_action=acknowledge_close",
        "reply_action=coordinate",
        "tone=plain",
        "tone=warm",
        "tone=playful",
        "tone=teasing",
        "tone=serious",
        "tone=formal",
        "conversational_naturalness>=2",
        "conversational_naturalness>=3",
        "ai_template_signal>=2",
        "ai_template_signal>=3",
        "unsupported_fact_signal>=1",
        "unsupported_fact_signal>=2",
        "over_explanation_signal>=2",
        "over_explanation_signal>=3",
        "depends_on_unstated_fact",
        "repeats_context_needlessly",
        "reply_bubble_count>=2",
        "reply_bubble_count>=3",
        "reply_char_count>=8",
        "reply_char_count>=16",
        "reply_char_count>=32",
        "reply_contains_question_mark",
        "reply_contains_at_mention",
        "reply_contains_digit",
        "reply_contains_laughter_marker",
    )
    for context_name in ("is_private", "is_group"):
        for option_name in relation_features:
            add(context_name, option_name)
    for option_name in (
        "group_pattern_alignment>=1",
        "group_pattern_alignment>=2",
        "group_pattern_alignment>=3",
    ):
        add("is_group", option_name)

    for context_name in (
        "recent_1_is_self",
        "recent_1_is_other",
        "latest_turn_switches_speaker",
    ):
        for option_name in (
            "addressee_fit>=2",
            "addressee_fit>=3",
            "timeline_continuity>=2",
            "timeline_continuity>=3",
            "stance_continuity>=2",
            "stance_continuity>=3",
            "reply_mentions_latest_speaker",
        ):
            add(context_name, option_name)
    for option_name in (
        "reply_action=answer_now",
        "reply_action=defer",
        "reply_action=confirm",
        "reply_action=ask",
        "reply_action=clarify",
        "reply_contains_question_mark",
    ):
        add("latest_message_has_question_mark", option_name)
    for option_name in (
        "addressee_fit>=2",
        "addressee_fit>=3",
        "reply_mentions_latest_speaker",
        "reply_contains_at_mention",
    ):
        add("latest_message_has_at_mention", option_name)
    for context_name in (
        "recent_1_same_speaker_as_2",
        "recent_1_same_speaker_as_3",
        "recent_2_same_speaker_as_3",
    ):
        for option_name in (
            "group_pattern_alignment>=2",
            "group_pattern_alignment>=3",
            "reply_action=joke_tease",
        ):
            add(context_name, option_name)

    for context_name in context_names:
        if context_name.startswith("group_name="):
            for option_name in (
                "group_pattern_alignment>=2",
                "tone=playful",
                "reply_bubble_count>=2",
            ):
                add(context_name, option_name)
        elif context_name.startswith("recent_1_group_member="):
            for option_name in (
                "addressee_fit>=2",
                "reply_mentions_latest_speaker",
            ):
                add(context_name, option_name)
        elif context_name.startswith("recent_2_group_member="):
            add(context_name, "timeline_continuity>=2")
        elif context_name.startswith("recent_3_group_member="):
            add(context_name, "stance_continuity>=2")
    return crosses


def expand_boolean_option_with_refined_context(
    option: np.ndarray,
    option_names: list[str],
    context: np.ndarray,
    context_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if len(option) != len(option_names) or len(context) != len(context_names):
        raise ValueError("boolean values and feature names are not aligned")
    if not np.isin(option, (0.0, 1.0)).all() or not np.isin(context, (0.0, 1.0)).all():
        raise ValueError("context crosses require boolean inputs")
    option_indexes = {name: index for index, name in enumerate(option_names)}
    context_indexes = {name: index for index, name in enumerate(context_names)}
    crosses = refined_boolean_context_crosses(option_names, context_names)
    crossed_values = np.asarray(
        [
            context[context_indexes[context_name]] * option[option_indexes[option_name]]
            for context_name, option_name in crosses
        ],
        dtype=float,
    )
    crossed_names = [
        f"{context_name}*{option_name}" for context_name, option_name in crosses
    ]
    return np.concatenate((option, crossed_values)), [*option_names, *crossed_names]


def expand_option_with_pairwise_crosses(
    values: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    if values.ndim != 1 or len(values) != len(feature_names):
        raise ValueError("option values and feature names must be aligned one-dimensional data")
    expanded_values = values.astype(float).tolist()
    expanded_names = list(feature_names)
    for left_index, left_name in enumerate(feature_names):
        for right_index in range(left_index + 1, len(feature_names)):
            expanded_values.append(float(values[left_index] * values[right_index]))
            expanded_names.append(f"{left_name}*{feature_names[right_index]}")
    return np.asarray(expanded_values, dtype=float), expanded_names


def expand_option_matrix_with_pairwise_crosses(
    values: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    if values.ndim != 2:
        raise ValueError("option matrix must be two-dimensional")
    rows = [
        expand_option_with_pairwise_crosses(row, feature_names)
        for row in values
    ]
    expanded_names = rows[0][1] if rows else []
    if any(names != expanded_names for _, names in rows):
        raise ValueError("expanded feature schema changed between rows")
    return np.stack([row for row, _ in rows]), expanded_names


def load_feature_matrix(
    blind: dict[str, Any],
    answers: dict[str, Any],
    checkpoint: dict[str, Any],
    include_reply_bubble_count: bool = False,
) -> tuple[dict[str, list[tuple[str, str, np.ndarray, int]]], list[str]]:
    if answers.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("answer key does not match blind packet")
    if checkpoint.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("feature checkpoint does not match blind packet")
    answer_by_id = {str(row["case_id"]): row for row in answers["cases"]}
    rows: dict[str, list[tuple[str, str, np.ndarray, int]]] = {"train": [], "validation": []}
    feature_names: list[str] | None = None
    for case in blind["cases"]:
        case_id = str(case["case_id"])
        entry = (checkpoint.get("entries") or {}).get(case_id) or {}
        if entry.get("status") != "valid":
            raise ValueError(f"missing valid features for case {case_id}")
        vector, names = vectorize_pair(entry["features"])
        if include_reply_bubble_count:
            vector = np.append(vector, reply_bubble_count_difference(case))
            names = [*names, "reply_bubble_count"]
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("feature vector schema changed between cases")
        answer = answer_by_id[case_id]
        split = str(answer["split"])
        label = 1 if answer["human_option"] == "A" else 0
        if split != case["split"]:
            raise ValueError("blind and answer split labels differ")
        rows[split].append((case_id, str(case["relationship"]), vector, label))
    if len(rows["train"]) != TRAIN_TOTAL or len(rows["validation"]) != VALIDATION_TOTAL:
        raise ValueError("training requires exactly 1,200 train and 160 validation feature rows")
    return rows, feature_names or []


def load_option_feature_matrix(
    blind: dict[str, Any],
    answers: dict[str, Any],
    checkpoint: dict[str, Any],
) -> tuple[dict[str, list[tuple[str, str, np.ndarray, np.ndarray, int]]], list[str]]:
    if answers.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("answer key does not match blind packet")
    if checkpoint.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("feature checkpoint does not match blind packet")
    answer_by_id = {str(row["case_id"]): row for row in answers["cases"]}
    rows: dict[str, list[tuple[str, str, np.ndarray, np.ndarray, int]]] = {
        "train": [],
        "validation": [],
    }
    feature_names: list[str] | None = None
    for case in blind["cases"]:
        case_id = str(case["case_id"])
        entry = (checkpoint.get("entries") or {}).get(case_id) or {}
        if entry.get("status") != "valid":
            raise ValueError(f"missing valid features for case {case_id}")
        left, right, names = vectorize_options(entry["features"])
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("feature vector schema changed between cases")
        answer = answer_by_id[case_id]
        split = str(answer["split"])
        label = 1 if answer["human_option"] == "A" else 0
        if split != case["split"]:
            raise ValueError("blind and answer split labels differ")
        rows[split].append(
            (case_id, str(case["relationship"]), left, right, label)
        )
    if len(rows["train"]) != TRAIN_TOTAL or len(rows["validation"]) != VALIDATION_TOTAL:
        raise ValueError("training requires exactly 1,200 train and 160 validation feature rows")
    return rows, feature_names or []


def balanced_learning_subset(
    rows: list[tuple[str, str, np.ndarray, int]], total: int
) -> list[tuple[str, str, np.ndarray, int]]:
    per_relation = total // 2
    selected: list[tuple[str, str, np.ndarray, int]] = []
    for relation in RELATIONS:
        relation_rows = [row for row in rows if row[1] == relation]
        relation_rows.sort(key=lambda row: _rank(f"learning:{row[0]}"))
        if len(relation_rows) < per_relation:
            raise ValueError(f"not enough {relation} rows for learning curve size {total}")
        selected.extend(relation_rows[:per_relation])
    return selected


def train_learning_curve(
    blind_path: Path,
    answers_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
    include_reply_bubble_count: bool = False,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, feature_names = load_feature_matrix(
        blind,
        answers,
        checkpoint,
        include_reply_bubble_count=include_reply_bubble_count,
    )
    validation = rows["validation"]
    x_validation = np.stack([row[2] for row in validation])
    y_validation = np.asarray([row[3] for row in validation], dtype=int)
    curve = []
    for total in LEARNING_CURVE_TOTALS:
        training = balanced_learning_subset(rows["train"], total)
        x_train = np.stack([row[2] for row in training])
        y_train = np.asarray([row[3] for row in training], dtype=int)
        classifier = LogisticRegression(fit_intercept=False, max_iter=2000, random_state=0)
        classifier.fit(x_train, y_train)
        probability = classifier.predict_proba(x_validation)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        swapped_probability = classifier.predict_proba(-x_validation)[:, 1]
        if not np.allclose(probability + swapped_probability, 1.0, atol=1e-12):
            raise ValueError("A/B swap probabilities are not complementary")
        strata = {}
        for relation in RELATIONS:
            indexes = [index for index, row in enumerate(validation) if row[1] == relation]
            strata[relation] = {
                "correct": int(sum(prediction[index] == y_validation[index] for index in indexes)),
                "total": len(indexes),
                "accuracy": float(accuracy_score(y_validation[indexes], prediction[indexes])),
            }
        curve.append({
            "training_total": total,
            "training_relationships": dict(Counter(row[1] for row in training)),
            "validation": {
                "correct": int((prediction == y_validation).sum()),
                "total": len(validation),
                "accuracy": float(accuracy_score(y_validation, prediction)),
                "relationships": strata,
            },
            "coefficients": {
                name: float(value) for name, value in zip(feature_names, classifier.coef_[0])
            },
            "predictions": [
                {
                    "case_id": row[0],
                    "relationship": row[1],
                    "predicted_human_option": "A" if int(prediction[index]) == 1 else "B",
                    "human_option": "A" if int(y_validation[index]) == 1 else "B",
                    "probability_A": float(probability[index]),
                }
                for index, row in enumerate(validation)
            ],
        })
    result = {
        "schema_version": 1,
        "kind": "judge_feature_classifier_learning_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "LogisticRegression(fit_intercept=False)",
        "feature_encoding": "numeric/boolean A-B; enum one_hot(A)-one_hot(B)",
        "deterministic_pair_features": (
            ["reply_bubble_count"] if include_reply_bubble_count else []
        ),
        "train_total": TRAIN_TOTAL,
        "validation_total": VALIDATION_TOTAL,
        "learning_curve": curve,
        "promotion_gate": {
            "minimum_overall_net_improvement": 1,
            "required_coverage": "same complete comparison set",
            "reported_diagnostics": [
                "private_net_change",
                "group_net_change",
                "new_error_count",
                "high_confidence_new_error_count",
            ],
            "status": "requires current Judge baseline predictions on this validation set",
        },
    }
    _write_feature_classifier_result(result_path, result)
    return result


def _shared_mlp_unpack(
    parameters: np.ndarray, input_dim: int, hidden_units: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix_end = hidden_units * input_dim
    bias_end = matrix_end + hidden_units
    weights = parameters[:matrix_end].reshape(hidden_units, input_dim)
    bias = parameters[matrix_end:bias_end]
    output_weights = parameters[bias_end:bias_end + hidden_units]
    return weights, bias, output_weights


def _shared_mlp_scores(
    values: np.ndarray,
    parameters: np.ndarray,
    input_dim: int,
    hidden_units: int,
) -> np.ndarray:
    weights, bias, output_weights = _shared_mlp_unpack(
        parameters, input_dim, hidden_units
    )
    hidden = np.tanh(values @ weights.T + bias)
    return hidden @ output_weights


def fit_shared_pair_mlp(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_units: int = SHARED_MLP_HIDDEN_UNITS,
    l2: float = SHARED_MLP_L2,
    max_iterations: int = SHARED_MLP_MAX_ITERATIONS,
) -> dict[str, Any]:
    combined = np.concatenate((left, right), axis=0)
    mean = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized_left = (left - mean) / scale
    normalized_right = (right - mean) / scale
    input_dim = left.shape[1]
    rng = np.random.default_rng(0)
    input_limit = np.sqrt(6.0 / (input_dim + hidden_units))
    output_limit = np.sqrt(6.0 / (hidden_units + 1))
    initial = np.concatenate((
        rng.uniform(-input_limit, input_limit, size=(hidden_units, input_dim)).ravel(),
        np.zeros(hidden_units, dtype=float),
        rng.uniform(-output_limit, output_limit, size=hidden_units),
    ))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias, output_weights = _shared_mlp_unpack(
            parameters, input_dim, hidden_units
        )
        left_hidden = np.tanh(normalized_left @ weights.T + bias)
        right_hidden = np.tanh(normalized_right @ weights.T + bias)
        logits = (left_hidden - right_hidden) @ output_weights
        loss = float(
            np.mean(np.logaddexp(0.0, logits) - labels * logits)
            + 0.5 * l2 * (np.square(weights).sum() + np.square(output_weights).sum())
        )
        logit_gradient = (expit(logits) - labels) / len(labels)
        output_gradient = (
            (left_hidden - right_hidden).T @ logit_gradient + l2 * output_weights
        )
        left_pre_gradient = (
            logit_gradient[:, None]
            * output_weights[None, :]
            * (1.0 - np.square(left_hidden))
        )
        right_pre_gradient = (
            -logit_gradient[:, None]
            * output_weights[None, :]
            * (1.0 - np.square(right_hidden))
        )
        weights_gradient = (
            left_pre_gradient.T @ normalized_left
            + right_pre_gradient.T @ normalized_right
            + l2 * weights
        )
        bias_gradient = (left_pre_gradient + right_pre_gradient).sum(axis=0)
        gradient = np.concatenate((
            weights_gradient.ravel(), bias_gradient, output_gradient
        ))
        return loss, gradient

    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    if not optimized.success:
        raise RuntimeError(
            f"shared MLP optimization failed: status={optimized.status} {optimized.message}"
        )
    return {
        "parameters": optimized.x,
        "mean": mean,
        "scale": scale,
        "input_dim": input_dim,
        "hidden_units": hidden_units,
        "l2": l2,
        "iterations": int(optimized.nit),
        "final_loss": float(optimized.fun),
    }


def shared_pair_mlp_probability(
    model: dict[str, Any], left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    normalized_left = (left - model["mean"]) / model["scale"]
    normalized_right = (right - model["mean"]) / model["scale"]
    left_scores = _shared_mlp_scores(
        normalized_left,
        model["parameters"],
        model["input_dim"],
        model["hidden_units"],
    )
    right_scores = _shared_mlp_scores(
        normalized_right,
        model["parameters"],
        model["input_dim"],
        model["hidden_units"],
    )
    return expit(left_scores - right_scores)


def train_shared_mlp_learning_curve(
    blind_path: Path,
    answers_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, feature_names = load_option_feature_matrix(blind, answers, checkpoint)
    validation = rows["validation"]
    validation_left = np.stack([row[2] for row in validation])
    validation_right = np.stack([row[3] for row in validation])
    validation_labels = np.asarray([row[4] for row in validation], dtype=int)
    curve = []
    for total in LEARNING_CURVE_TOTALS:
        training = balanced_learning_subset(rows["train"], total)
        training_left = np.stack([row[2] for row in training])
        training_right = np.stack([row[3] for row in training])
        training_labels = np.asarray([row[4] for row in training], dtype=int)
        model = fit_shared_pair_mlp(training_left, training_right, training_labels)
        probability = shared_pair_mlp_probability(
            model, validation_left, validation_right
        )
        prediction = (probability >= 0.5).astype(int)
        swapped_probability = shared_pair_mlp_probability(
            model, validation_right, validation_left
        )
        if not np.allclose(probability + swapped_probability, 1.0, atol=1e-12):
            raise ValueError("A/B swap probabilities are not complementary")
        strata = {}
        for relation in RELATIONS:
            indexes = [index for index, row in enumerate(validation) if row[1] == relation]
            strata[relation] = {
                "correct": int(sum(
                    prediction[index] == validation_labels[index] for index in indexes
                )),
                "total": len(indexes),
                "accuracy": float(
                    accuracy_score(validation_labels[indexes], prediction[indexes])
                ),
            }
        curve.append({
            "training_total": total,
            "training_relationships": dict(Counter(row[1] for row in training)),
            "validation": {
                "correct": int((prediction == validation_labels).sum()),
                "total": len(validation),
                "accuracy": float(accuracy_score(validation_labels, prediction)),
                "relationships": strata,
            },
            "optimization": {
                "iterations": model["iterations"],
                "final_loss": model["final_loss"],
            },
            "predictions": [
                {
                    "case_id": row[0],
                    "relationship": row[1],
                    "predicted_human_option": "A" if int(prediction[index]) == 1 else "B",
                    "human_option": "A" if int(validation_labels[index]) == 1 else "B",
                    "probability_A": float(probability[index]),
                }
                for index, row in enumerate(validation)
            ],
        })
    result = {
        "schema_version": 1,
        "kind": "judge_shared_pair_mlp_learning_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "shared MLP: 42->8 tanh->1; sigmoid(logit_A-logit_B)",
        "feature_encoding": "numeric/boolean values; enum one-hot; A/B kept separate",
        "feature_names": feature_names,
        "hidden_units": SHARED_MLP_HIDDEN_UNITS,
        "l2": SHARED_MLP_L2,
        "max_iterations": SHARED_MLP_MAX_ITERATIONS,
        "train_total": TRAIN_TOTAL,
        "validation_total": VALIDATION_TOTAL,
        "learning_curve": curve,
    }
    _write_feature_classifier_result(result_path, result)
    return result


def train_explicit_cross_learning_curve(
    blind_path: Path,
    answers_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, feature_names = load_option_feature_matrix(blind, answers, checkpoint)
    validation = rows["validation"]
    validation_left, expanded_names = expand_option_matrix_with_pairwise_crosses(
        np.stack([row[2] for row in validation]), feature_names
    )
    validation_right, right_names = expand_option_matrix_with_pairwise_crosses(
        np.stack([row[3] for row in validation]), feature_names
    )
    if expanded_names != right_names:
        raise ValueError("A/B expanded feature schemas differ")
    validation_labels = np.asarray([row[4] for row in validation], dtype=int)
    curve = []
    for total in LEARNING_CURVE_TOTALS:
        training = balanced_learning_subset(rows["train"], total)
        training_left, training_names = expand_option_matrix_with_pairwise_crosses(
            np.stack([row[2] for row in training]), feature_names
        )
        training_right, training_right_names = expand_option_matrix_with_pairwise_crosses(
            np.stack([row[3] for row in training]), feature_names
        )
        if training_names != expanded_names or training_right_names != expanded_names:
            raise ValueError("training and validation expanded feature schemas differ")
        training_labels = np.asarray([row[4] for row in training], dtype=int)
        classifier = LogisticRegression(
            fit_intercept=False, max_iter=2000, random_state=0
        )
        classifier.fit(training_left - training_right, training_labels)
        logits = (validation_left - validation_right) @ classifier.coef_[0]
        probability = expit(logits)
        prediction = (probability >= 0.5).astype(int)
        swapped_probability = expit(-logits)
        if not np.allclose(probability + swapped_probability, 1.0, atol=1e-12):
            raise ValueError("A/B swap probabilities are not complementary")
        strata = {}
        for relation in RELATIONS:
            indexes = [index for index, row in enumerate(validation) if row[1] == relation]
            strata[relation] = {
                "correct": int(sum(
                    prediction[index] == validation_labels[index] for index in indexes
                )),
                "total": len(indexes),
                "accuracy": float(
                    accuracy_score(validation_labels[indexes], prediction[indexes])
                ),
            }
        curve.append({
            "training_total": total,
            "training_relationships": dict(Counter(row[1] for row in training)),
            "validation": {
                "correct": int((prediction == validation_labels).sum()),
                "total": len(validation),
                "accuracy": float(accuracy_score(validation_labels, prediction)),
                "relationships": strata,
            },
            "predictions": [
                {
                    "case_id": row[0],
                    "relationship": row[1],
                    "predicted_human_option": "A" if int(prediction[index]) == 1 else "B",
                    "human_option": "A" if int(validation_labels[index]) == 1 else "B",
                    "probability_A": float(probability[index]),
                }
                for index, row in enumerate(validation)
            ],
        })
    result = {
        "schema_version": 1,
        "kind": "judge_explicit_pairwise_cross_learning_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": (
            "shared linear scorer over raw option features plus all i<j pairwise products; "
            "sigmoid(logit_A-logit_B)"
        ),
        "feature_encoding": (
            "42 raw numeric/boolean/one-hot features plus explicit within-option "
            "second-order crosses; A/B kept separate until logit subtraction"
        ),
        "raw_feature_count": len(feature_names),
        "cross_feature_count": len(expanded_names) - len(feature_names),
        "expanded_feature_count": len(expanded_names),
        "regularization": "LogisticRegression default L2, C=1.0",
        "train_total": TRAIN_TOTAL,
        "validation_total": VALIDATION_TOTAL,
        "learning_curve": curve,
    }
    _write_feature_classifier_result(result_path, result)
    return result


def load_boolean_context_rows(
    blind: dict[str, Any],
    answers: dict[str, Any],
    replay: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    context_schema: dict[str, tuple[Any, ...]] | None = None,
    expected_split_counts: dict[str, int] | None = None,
) -> tuple[
    dict[str, list[tuple[Any, ...]]],
    list[str],
    list[str],
    list[str],
    dict[str, tuple[Any, ...]],
]:
    if checkpoint.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("feature checkpoint does not match blind packet")
    answer_by_id = {str(row["case_id"]): row for row in answers["cases"]}
    metadata = load_context_metadata(blind, answers, replay)
    expected_split_counts = expected_split_counts or {
        "train": TRAIN_TOTAL,
        "validation": VALIDATION_TOTAL,
    }
    if context_schema is None:
        if "holdout" in expected_split_counts:
            raise ValueError("holdout rows require a frozen training context schema")
        training_metadata = {
            case_id: row
            for case_id, row in metadata.items()
            if row["split"] == "train"
        }
        context_schema = build_context_boolean_schema(training_metadata)
    rows: dict[str, list[tuple[Any, ...]]] = {
        split: [] for split in expected_split_counts
    }
    numeric_names: list[str] | None = None
    boolean_names: list[str] | None = None
    context_names: list[str] | None = None
    for case in blind["cases"]:
        case_id = str(case["case_id"])
        entry = (checkpoint.get("entries") or {}).get(case_id) or {}
        if entry.get("status") != "valid":
            raise ValueError(f"missing valid features for case {case_id}")
        numeric_left, numeric_right, current_numeric_names = vectorize_options(
            entry["features"]
        )
        boolean_left, boolean_right, current_boolean_names = vectorize_boolean_options(
            entry["features"]
        )
        context, current_context_names = vectorize_context_booleans(
            metadata[case_id], context_schema
        )
        observable_left, current_observable_names = vectorize_observable_option_booleans(
            case["option_A"], metadata[case_id]
        )
        observable_right, other_observable_names = vectorize_observable_option_booleans(
            case["option_B"], metadata[case_id]
        )
        group_pattern_left, current_group_pattern_names = (
            vectorize_group_pattern_option_boolean(
                case["option_A"], metadata[case_id]
            )
        )
        group_pattern_right, other_group_pattern_names = (
            vectorize_group_pattern_option_boolean(
                case["option_B"], metadata[case_id]
            )
        )
        if (
            current_observable_names != list(OBSERVABLE_OPTION_BOOLEAN_NAMES)
            or other_observable_names != current_observable_names
            or current_group_pattern_names
            != list(GROUP_PATTERN_OPTION_BOOLEAN_NAMES)
            or other_group_pattern_names != current_group_pattern_names
        ):
            raise ValueError("observable option feature schema changed between candidates")
        if numeric_names is None:
            numeric_names = current_numeric_names
            boolean_names = current_boolean_names
            context_names = current_context_names
        elif (
            numeric_names != current_numeric_names
            or boolean_names != current_boolean_names
            or context_names != current_context_names
        ):
            raise ValueError("boolean/context feature schema changed between cases")
        answer = answer_by_id[case_id]
        split = str(answer["split"])
        if split not in rows:
            raise ValueError(f"unexpected split in boolean/context rows: {split}")
        label = 1 if answer["human_option"] == "A" else 0
        rows[split].append((
            case_id,
            str(case["relationship"]),
            numeric_left,
            numeric_right,
            boolean_left,
            boolean_right,
            context,
            observable_left,
            observable_right,
            group_pattern_left,
            group_pattern_right,
            label,
        ))
    actual_split_counts = {split: len(split_rows) for split, split_rows in rows.items()}
    if actual_split_counts != expected_split_counts:
        raise ValueError(
            "boolean/context split counts differ: "
            f"expected {expected_split_counts}, got {actual_split_counts}"
        )
    return (
        rows,
        numeric_names or [],
        boolean_names or [],
        context_names or [],
        context_schema,
    )


def _boolean_context_difference_matrix(
    rows: list[tuple[Any, ...]],
) -> sparse.csr_matrix:
    boolean_difference = sparse.csr_matrix(np.stack([
        row[4] - row[5] for row in rows
    ]))
    context = np.stack([row[6] for row in rows])
    blocks: list[sparse.spmatrix] = [boolean_difference]
    blocks.extend(
        boolean_difference.multiply(context[:, position].reshape(-1, 1))
        for position in range(context.shape[1])
    )
    return sparse.hstack(blocks, format="csr")


def _refined_boolean_difference_matrix(
    rows: list[tuple[Any, ...]],
    option_names: list[str],
    context_names: list[str],
    *,
    include_context_crosses: bool,
    include_group_pattern: bool = False,
) -> sparse.csr_matrix:
    differences: list[np.ndarray] = []
    for row in rows:
        left = np.concatenate((row[4], row[7]))
        right = np.concatenate((row[5], row[8]))
        if include_group_pattern:
            left = np.concatenate((left, row[9]))
            right = np.concatenate((right, row[10]))
        if include_context_crosses:
            left, left_names = expand_boolean_option_with_refined_context(
                left, option_names, row[6], context_names
            )
            right, right_names = expand_boolean_option_with_refined_context(
                right, option_names, row[6], context_names
            )
            if left_names != right_names:
                raise ValueError("A/B refined feature schemas differ")
        differences.append(left - right)
    return sparse.csr_matrix(np.stack(differences))


def _pair_variant_matrix(
    rows: list[tuple[Any, ...]],
    variant: str,
    refined_option_names: list[str] | None = None,
    context_names: list[str] | None = None,
) -> sparse.csr_matrix:
    if variant == "legacy_numeric_42":
        return sparse.csr_matrix(np.stack([row[2] - row[3] for row in rows]))
    if variant == "boolean_option_only":
        return sparse.csr_matrix(np.stack([row[4] - row[5] for row in rows]))
    if variant == "boolean_context_cross":
        return _boolean_context_difference_matrix(rows)
    if variant in {
        "refined_boolean_option_only",
        "refined_boolean_context_cross",
        "refined_boolean_group_pattern",
    }:
        if refined_option_names is None or context_names is None:
            raise ValueError("refined boolean variants require option and context schemas")
        return _refined_boolean_difference_matrix(
            rows,
            refined_option_names,
            context_names,
            include_context_crosses=variant == "refined_boolean_context_cross",
            include_group_pattern=variant == "refined_boolean_group_pattern",
        )
    raise ValueError(f"unsupported pair model variant: {variant}")


def _evaluate_pair_classifier(
    training: list[tuple[Any, ...]],
    validation: list[tuple[Any, ...]],
    variant: str,
    feature_names: list[str],
    *,
    refined_option_names: list[str] | None = None,
    context_names: list[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    x_train = _pair_variant_matrix(
        training, variant, refined_option_names, context_names
    )
    x_validation = _pair_variant_matrix(
        validation, variant, refined_option_names, context_names
    )
    y_train = np.asarray([row[-1] for row in training], dtype=int)
    y_validation = np.asarray([row[-1] for row in validation], dtype=int)
    classifier = LogisticRegression(
        fit_intercept=False,
        max_iter=5000,
        random_state=0,
    )
    classifier.fit(x_train, y_train)
    logits = classifier.decision_function(x_validation)
    probability = expit(logits)
    prediction = (probability >= 0.5).astype(int)
    if not np.allclose(probability + expit(-logits), 1.0, atol=1e-12):
        raise ValueError("A/B swap probabilities are not complementary")
    relationships: dict[str, Any] = {}
    for relation in RELATIONS:
        indexes = [index for index, row in enumerate(validation) if row[1] == relation]
        correct = int(sum(prediction[index] == y_validation[index] for index in indexes))
        relationships[relation] = {
            "correct": correct,
            "total": len(indexes),
            "accuracy": correct / len(indexes),
        }
    coefficients = classifier.coef_[0]
    strongest = np.argsort(np.abs(coefficients))[-30:][::-1]
    summary = {
        "feature_count": x_train.shape[1],
        "validation": {
            "correct": int((prediction == y_validation).sum()),
            "total": len(validation),
            "accuracy": float(accuracy_score(y_validation, prediction)),
            "relationships": relationships,
        },
        "top_absolute_coefficients": [
            {"feature": feature_names[index], "coefficient": float(coefficients[index])}
            for index in strongest
            if abs(float(coefficients[index])) > 0.0
        ],
        "predictions": [
            {
                "case_id": row[0],
                "relationship": row[1],
                "predicted_human_option": "A" if int(prediction[index]) == 1 else "B",
                "human_option": "A" if int(y_validation[index]) == 1 else "B",
                "probability_A": float(probability[index]),
            }
            for index, row in enumerate(validation)
        ],
    }
    return summary, prediction, y_validation


def _prediction_comparison(
    validation: list[tuple[Any, ...]],
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    corrected: list[str] = []
    new_errors: list[str] = []
    relation_changes: dict[str, dict[str, int]] = {
        relation: {"corrected": 0, "new_errors": 0, "net": 0}
        for relation in RELATIONS
    }
    for index, row in enumerate(validation):
        baseline_correct = bool(baseline[index] == labels[index])
        candidate_correct = bool(candidate[index] == labels[index])
        relation = str(row[1])
        if not baseline_correct and candidate_correct:
            corrected.append(str(row[0]))
            relation_changes[relation]["corrected"] += 1
        elif baseline_correct and not candidate_correct:
            new_errors.append(str(row[0]))
            relation_changes[relation]["new_errors"] += 1
    for values in relation_changes.values():
        values["net"] = values["corrected"] - values["new_errors"]
    return {
        "corrected_count": len(corrected),
        "new_error_count": len(new_errors),
        "net": len(corrected) - len(new_errors),
        "relationships": relation_changes,
        "corrected_case_ids": corrected,
        "new_error_case_ids": new_errors,
    }


def _prediction_validation_summary(
    validation: list[tuple[Any, ...]],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    relationships: dict[str, Any] = {}
    for relation in RELATIONS:
        indexes = [index for index, row in enumerate(validation) if row[1] == relation]
        correct = int(sum(predictions[index] == labels[index] for index in indexes))
        relationships[relation] = {
            "correct": correct,
            "total": len(indexes),
            "accuracy": correct / len(indexes),
        }
    correct = int((predictions == labels).sum())
    return {
        "correct": correct,
        "total": len(validation),
        "accuracy": correct / len(validation),
        "relationships": relationships,
    }


def _relation_routed_predictions(
    validation: list[tuple[Any, ...]],
    private_predictions: np.ndarray,
    group_predictions: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        private_predictions[index] if row[1] == "private" else group_predictions[index]
        for index, row in enumerate(validation)
    ])


def _refined_boolean_decision_summary(
    final_variants: dict[str, Any],
    context_increment: dict[str, Any],
    routing_validation: dict[str, Any],
) -> dict[str, Any]:
    best_variant = max(
        final_variants,
        key=lambda name: int(final_variants[name]["validation"]["correct"]),
    )
    context_passed = (
        context_increment["net"] > 0
        and all(
            context_increment["relationships"][relation]["net"] >= 0
            for relation in RELATIONS
        )
    )
    return {
        "best_single_shared_variant": {
            "variant": best_variant,
            "validation": final_variants[best_variant]["validation"],
        },
        "context_cross_candidate": {
            "baseline_variant": "refined_boolean_option_only",
            "variant": "refined_boolean_context_cross",
            "comparison": context_increment,
            "decision": "adopt" if context_passed else "reject",
        },
        "best_observed_routing_candidate": {
            "private_variant": "refined_boolean_option_only",
            "group_variant": "refined_boolean_context_cross",
            "validation": routing_validation,
            "status": "requires_independent_validation",
            "adopted": False,
        },
        "promotion_gate": {
            "baseline_variant": "refined_boolean_option_only",
            "requires_positive_overall_net": True,
            "requires_no_private_or_group_regression": True,
            "passed": context_passed,
        },
        "decision": (
            "adopt_refined_boolean_context_cross"
            if context_passed
            else f"retain_{best_variant}"
        ),
    }


def train_boolean_context_cross_learning_curve(
    blind_path: Path,
    answers_path: Path,
    replay_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    replay = _load_json(replay_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, numeric_names, boolean_names, context_names, context_schema = (
        load_boolean_context_rows(blind, answers, replay, checkpoint)
    )
    _, crossed_names = expand_boolean_option_with_context(
        np.zeros(len(boolean_names)),
        boolean_names,
        np.zeros(len(context_names)),
        context_names,
    )
    variant_names = {
        "legacy_numeric_42": numeric_names,
        "boolean_option_only": boolean_names,
        "boolean_context_cross": crossed_names,
    }
    validation = rows["validation"]
    curve: list[dict[str, Any]] = []
    final_predictions: dict[str, np.ndarray] = {}
    final_labels: np.ndarray | None = None
    for total in LEARNING_CURVE_TOTALS:
        training = balanced_learning_subset(rows["train"], total)
        variant_results: dict[str, Any] = {}
        for variant, feature_names in variant_names.items():
            variant_result, predictions, labels = _evaluate_pair_classifier(
                training, validation, variant, feature_names
            )
            variant_results[variant] = variant_result
            if total == LEARNING_CURVE_TOTALS[-1]:
                final_predictions[variant] = predictions
                final_labels = labels
        curve.append({
            "training_total": total,
            "training_relationships": dict(Counter(row[1] for row in training)),
            "variants": variant_results,
        })
    if final_labels is None:
        raise ValueError("missing final learning-curve predictions")
    booleanization = _prediction_comparison(
        validation,
        final_labels,
        final_predictions["legacy_numeric_42"],
        final_predictions["boolean_option_only"],
    )
    context_increment = _prediction_comparison(
        validation,
        final_labels,
        final_predictions["boolean_option_only"],
        final_predictions["boolean_context_cross"],
    )
    candidate_vs_baseline = _prediction_comparison(
        validation,
        final_labels,
        final_predictions["legacy_numeric_42"],
        final_predictions["boolean_context_cross"],
    )
    relationship_nets = candidate_vs_baseline["relationships"]
    adopted = (
        candidate_vs_baseline["net"] > 0
        and all(relationship_nets[relation]["net"] >= 0 for relation in RELATIONS)
    )
    result = {
        "schema_version": 1,
        "kind": "judge_boolean_context_cross_learning_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": (
            "shared linear option scorer; all candidate/context inputs boolean; "
            "sigmoid(logit_A-logit_B)"
        ),
        "feature_design": {
            "integer_option_encoding": "three threshold booleans: >=1, >=2, >=3",
            "boolean_option_feature_count": len(boolean_names),
            "context_boolean_feature_count": len(context_names),
            "candidate_feature_count": len(crossed_names),
            "context_cross": "each context boolean multiplied by each option boolean",
            "recent_speaker_positions": RECENT_SPEAKER_POSITIONS,
            "identified_group_speaker_positions": IDENTIFIED_GROUP_SPEAKER_POSITIONS,
            "known_group_name_count": len(context_schema["known_group_names"]),
            "known_group_member_count": len(context_schema["known_group_members"]),
            "group_name_min_train_cases": GROUP_NAME_MIN_TRAIN_CASES,
            "group_member_min_train_occurrences": GROUP_MEMBER_MIN_TRAIN_OCCURRENCES,
            "category_names_in_result_are_hashed": True,
        },
        "large_model_features_reused": True,
        "large_model_calls": 0,
        "train_total": TRAIN_TOTAL,
        "validation_total": VALIDATION_TOTAL,
        "learning_curve": curve,
        "final_1200_training_comparisons": {
            "boolean_option_only_vs_legacy_numeric_42": booleanization,
            "context_cross_vs_boolean_option_only": context_increment,
            "candidate_vs_legacy_numeric_42": candidate_vs_baseline,
        },
        "promotion_gate": {
            "requires_positive_overall_net_vs_legacy": True,
            "requires_no_private_or_group_regression": True,
            "passed": adopted,
        },
        "decision": "adopt" if adopted else "reject",
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def train_refined_boolean_context_learning_curve(
    blind_path: Path,
    answers_path: Path,
    replay_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    replay = _load_json(replay_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, numeric_names, boolean_names, context_names, context_schema = (
        load_boolean_context_rows(blind, answers, replay, checkpoint)
    )
    refined_option_names = [*boolean_names, *OBSERVABLE_OPTION_BOOLEAN_NAMES]
    _, refined_crossed_names = expand_boolean_option_with_refined_context(
        np.zeros(len(refined_option_names)),
        refined_option_names,
        np.zeros(len(context_names)),
        context_names,
    )
    variant_names = {
        "legacy_numeric_42": numeric_names,
        "boolean_option_only": boolean_names,
        "refined_boolean_option_only": refined_option_names,
        "refined_boolean_context_cross": refined_crossed_names,
    }
    validation = rows["validation"]
    curve: list[dict[str, Any]] = []
    final_predictions: dict[str, np.ndarray] = {}
    final_labels: np.ndarray | None = None
    final_routing_validation: dict[str, Any] | None = None
    for total in LEARNING_CURVE_TOTALS:
        training = balanced_learning_subset(rows["train"], total)
        variant_results: dict[str, Any] = {}
        level_predictions: dict[str, np.ndarray] = {}
        level_labels: np.ndarray | None = None
        for variant, feature_names in variant_names.items():
            variant_result, predictions, labels = _evaluate_pair_classifier(
                training,
                validation,
                variant,
                feature_names,
                refined_option_names=refined_option_names,
                context_names=context_names,
            )
            variant_results[variant] = variant_result
            level_predictions[variant] = predictions
            level_labels = labels
            if total == LEARNING_CURVE_TOTALS[-1]:
                final_predictions[variant] = predictions
                final_labels = labels
        if level_labels is None:
            raise ValueError("missing learning-curve labels")
        routed_predictions = _relation_routed_predictions(
            validation,
            level_predictions["refined_boolean_option_only"],
            level_predictions["refined_boolean_context_cross"],
        )
        routing_validation = _prediction_validation_summary(
            validation, level_labels, routed_predictions
        )
        if total == LEARNING_CURVE_TOTALS[-1]:
            final_routing_validation = routing_validation
        curve.append({
            "training_total": total,
            "training_relationships": dict(Counter(row[1] for row in training)),
            "variants": variant_results,
            "observed_relation_routing": {
                "private_variant": "refined_boolean_option_only",
                "group_variant": "refined_boolean_context_cross",
                "validation": routing_validation,
            },
        })
    if final_labels is None or final_routing_validation is None:
        raise ValueError("missing final learning-curve predictions")

    comparisons = {
        "boolean_option_only_vs_legacy_numeric_42": _prediction_comparison(
            validation,
            final_labels,
            final_predictions["legacy_numeric_42"],
            final_predictions["boolean_option_only"],
        ),
        "observable_booleans_vs_boolean_option_only": _prediction_comparison(
            validation,
            final_labels,
            final_predictions["boolean_option_only"],
            final_predictions["refined_boolean_option_only"],
        ),
        "whitelisted_context_crosses_vs_refined_option_only": _prediction_comparison(
            validation,
            final_labels,
            final_predictions["refined_boolean_option_only"],
            final_predictions["refined_boolean_context_cross"],
        ),
        "candidate_vs_legacy_numeric_42": _prediction_comparison(
            validation,
            final_labels,
            final_predictions["legacy_numeric_42"],
            final_predictions["refined_boolean_context_cross"],
        ),
    }
    decision_summary = _refined_boolean_decision_summary(
        curve[-1]["variants"],
        comparisons["whitelisted_context_crosses_vs_refined_option_only"],
        final_routing_validation,
    )
    result = {
        "schema_version": 1,
        "kind": "judge_refined_boolean_context_learning_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": (
            "shared linear option scorer; A/B encoded separately; "
            "sigmoid(logit_A-logit_B)"
        ),
        "feature_design": {
            "legacy_numeric_feature_count": len(numeric_names),
            "llm_boolean_option_feature_count": len(boolean_names),
            "observable_option_boolean_feature_count": len(
                OBSERVABLE_OPTION_BOOLEAN_NAMES
            ),
            "refined_option_boolean_feature_count": len(refined_option_names),
            "context_boolean_feature_count": len(context_names),
            "whitelisted_context_cross_count": (
                len(refined_crossed_names) - len(refined_option_names)
            ),
            "candidate_feature_count": len(refined_crossed_names),
            "context_cross": (
                "explicit semantic whitelist; no context-option Cartesian product"
            ),
            "observable_option_boolean_features": list(
                OBSERVABLE_OPTION_BOOLEAN_NAMES
            ),
            "recent_speaker_positions": RECENT_SPEAKER_POSITIONS,
            "identified_group_speaker_positions": (
                IDENTIFIED_GROUP_SPEAKER_POSITIONS
            ),
            "known_group_name_count": len(context_schema["known_group_names"]),
            "known_group_member_count": len(
                context_schema["known_group_members"]
            ),
            "group_name_min_train_cases": GROUP_NAME_MIN_TRAIN_CASES,
            "group_member_min_train_occurrences": (
                GROUP_MEMBER_MIN_TRAIN_OCCURRENCES
            ),
            "category_names_in_result_are_hashed": True,
        },
        "large_model_features_reused": True,
        "large_model_calls": 0,
        "train_total": TRAIN_TOTAL,
        "validation_total": VALIDATION_TOTAL,
        "learning_curve": curve,
        "final_1200_training_comparisons": comparisons,
        **decision_summary,
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def _group_pattern_feature_evidence(
    rows: list[tuple[Any, ...]],
) -> dict[str, Any]:
    differing: list[dict[str, Any]] = []
    triggered_any = 0
    for row in rows:
        left = bool(row[9][0])
        right = bool(row[10][0])
        if left or right:
            triggered_any += 1
        if left == right:
            continue
        human_option = "A" if int(row[-1]) == 1 else "B"
        pattern_option = "A" if left else "B"
        differing.append({
            "case_id": row[0],
            "relationship": row[1],
            "pattern_option": pattern_option,
            "human_option": human_option,
            "pattern_side_is_human": pattern_option == human_option,
        })
    return {
        "triggered_any_count": triggered_any,
        "different_between_options_count": len(differing),
        "pattern_side_is_human_count": sum(
            bool(row["pattern_side_is_human"]) for row in differing
        ),
        "pattern_side_is_ai_count": sum(
            not bool(row["pattern_side_is_human"]) for row in differing
        ),
        "different_between_options": differing,
    }


def train_group_pattern_boolean_candidate(
    blind_path: Path,
    answers_path: Path,
    replay_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    replay = _load_json(replay_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, _, boolean_names, context_names, _ = load_boolean_context_rows(
        blind, answers, replay, checkpoint
    )
    baseline_names = [*boolean_names, *OBSERVABLE_OPTION_BOOLEAN_NAMES]
    candidate_names = [*baseline_names, *GROUP_PATTERN_OPTION_BOOLEAN_NAMES]
    training = balanced_learning_subset(rows["train"], TRAIN_TOTAL)
    validation = rows["validation"]
    baseline, baseline_predictions, labels = _evaluate_pair_classifier(
        training,
        validation,
        "refined_boolean_option_only",
        baseline_names,
        refined_option_names=baseline_names,
        context_names=context_names,
    )
    candidate, candidate_predictions, candidate_labels = _evaluate_pair_classifier(
        training,
        validation,
        "refined_boolean_group_pattern",
        candidate_names,
        refined_option_names=candidate_names,
        context_names=context_names,
    )
    if not np.array_equal(labels, candidate_labels):
        raise ValueError("baseline and group-pattern labels differ")
    comparison = _prediction_comparison(
        validation, labels, baseline_predictions, candidate_predictions
    )
    adopted = (
        comparison["net"] > 0
        and all(
            comparison["relationships"][relation]["net"] >= 0
            for relation in RELATIONS
        )
    )
    result = {
        "schema_version": 1,
        "kind": "judge_group_pattern_boolean_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": (
            "shared linear option scorer; A/B encoded separately; "
            "sigmoid(logit_A-logit_B)"
        ),
        "only_variable": "reply_reuses_active_group_pattern",
        "feature_definition": (
            "group chat only: candidate repeats a phrase seen in at least two "
            "recent group messages, or reuses a distinctive sentence skeleton"
        ),
        "baseline_feature_count": len(baseline_names),
        "candidate_feature_count": len(candidate_names),
        "large_model_features_reused": True,
        "large_model_calls": 0,
        "train_total": len(training),
        "validation_total": len(validation),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_vs_baseline": comparison,
        "feature_evidence": {
            "training": _group_pattern_feature_evidence(training),
            "validation": _group_pattern_feature_evidence(validation),
        },
        "promotion_gate": {
            "requires_positive_overall_net": True,
            "requires_no_private_or_group_regression": True,
            "passed": adopted,
        },
        "decision": "adopt" if adopted else "reject",
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def validate_group_pattern_holdout(
    development_blind_path: Path,
    development_answers_path: Path,
    development_replay_path: Path,
    development_features_path: Path,
    holdout_blind_path: Path,
    holdout_answers_path: Path,
    holdout_replay_path: Path,
    holdout_features_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    development_blind = _load_json(development_blind_path)
    development_answers = _load_json(development_answers_path)
    development_replay = _load_json(development_replay_path)
    development_features = _load_json(development_features_path)
    development_rows, development_numeric_names, development_boolean_names, development_context_names, context_schema = (
        load_boolean_context_rows(
            development_blind,
            development_answers,
            development_replay,
            development_features,
        )
    )
    training = balanced_learning_subset(development_rows["train"], TRAIN_TOTAL)

    holdout_blind = _load_json(holdout_blind_path)
    holdout_answers = _load_json(holdout_answers_path)
    holdout_replay = _load_json(holdout_replay_path)
    holdout_features = _load_json(holdout_features_path)
    holdout_rows, holdout_numeric_names, holdout_boolean_names, holdout_context_names, _ = (
        load_boolean_context_rows(
            holdout_blind,
            holdout_answers,
            holdout_replay,
            holdout_features,
            context_schema=context_schema,
            expected_split_counts={"holdout": GROUP_PATTERN_HOLDOUT_TOTAL},
        )
    )
    holdout = holdout_rows["holdout"]
    relationship_counts = Counter(row[1] for row in holdout)
    if relationship_counts != Counter(GROUP_PATTERN_HOLDOUT_TARGETS):
        raise ValueError(
            "fixed 240 relationship counts differ: "
            f"expected {GROUP_PATTERN_HOLDOUT_TARGETS}, got {dict(relationship_counts)}"
        )
    development_source_ids = {
        str(row["source_case_id"])
        for row in development_answers["cases"]
        if row["split"] == "train"
    }
    holdout_source_ids = {
        str(row["source_case_id"]) for row in holdout_answers["cases"]
    }
    source_overlap = development_source_ids & holdout_source_ids
    if source_overlap:
        raise ValueError(
            "development training and fixed 240 source cases overlap: "
            + ", ".join(sorted(source_overlap)[:20])
        )
    if (
        development_numeric_names != holdout_numeric_names
        or development_boolean_names != holdout_boolean_names
        or development_context_names != holdout_context_names
    ):
        raise ValueError("development and fixed 240 feature schemas differ")

    baseline_names = [
        *development_boolean_names,
        *OBSERVABLE_OPTION_BOOLEAN_NAMES,
    ]
    candidate_names = [*baseline_names, *GROUP_PATTERN_OPTION_BOOLEAN_NAMES]
    if len(baseline_names) != 72 or len(candidate_names) != 73:
        raise ValueError(
            "group-pattern comparison must be exactly 72 versus 73 features; "
            f"got {len(baseline_names)} versus {len(candidate_names)}"
        )
    baseline, baseline_predictions, labels = _evaluate_pair_classifier(
        training,
        holdout,
        "refined_boolean_option_only",
        baseline_names,
        refined_option_names=baseline_names,
        context_names=development_context_names,
    )
    candidate, candidate_predictions, candidate_labels = _evaluate_pair_classifier(
        training,
        holdout,
        "refined_boolean_group_pattern",
        candidate_names,
        refined_option_names=candidate_names,
        context_names=development_context_names,
    )
    if not np.array_equal(labels, candidate_labels):
        raise ValueError("baseline and group-pattern holdout labels differ")
    comparison = _prediction_comparison(
        holdout, labels, baseline_predictions, candidate_predictions
    )
    gates = {
        "overall_net_positive": comparison["net"] > 0,
        "group_net_positive": comparison["relationships"]["group"]["net"] > 0,
        "private_net_nonnegative": (
            comparison["relationships"]["private"]["net"] >= 0
        ),
    }
    adopted = all(gates.values())
    result = {
        "schema_version": 1,
        "kind": "judge_group_pattern_boolean_fixed_240_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "only_variable": "reply_reuses_active_group_pattern",
        "development_training_total": len(training),
        "fixed_holdout_total": len(holdout),
        "fixed_holdout_relationships": dict(relationship_counts),
        "holdout_used_for_training": False,
        "holdout_used_for_context_schema": False,
        "development_train_holdout_source_overlap": 0,
        "large_model_features_reused": True,
        "large_model_calls_during_comparison": 0,
        "baseline_feature_count": len(baseline_names),
        "candidate_feature_count": len(candidate_names),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_vs_baseline": comparison,
        "feature_evidence": {
            "training": _group_pattern_feature_evidence(training),
            "fixed_holdout": _group_pattern_feature_evidence(holdout),
        },
        "promotion_gate": {**gates, "passed": adopted},
        "decision": "adopt" if adopted else "reject",
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def _deep_shared_mlp_unpack(
    parameters: np.ndarray, input_dim: int, hidden_units: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_units, second_units = hidden_units
    first_matrix_end = first_units * input_dim
    first_bias_end = first_matrix_end + first_units
    second_matrix_end = first_bias_end + second_units * first_units
    second_bias_end = second_matrix_end + second_units
    first_weights = parameters[:first_matrix_end].reshape(first_units, input_dim)
    first_bias = parameters[first_matrix_end:first_bias_end]
    second_weights = parameters[first_bias_end:second_matrix_end].reshape(
        second_units, first_units
    )
    second_bias = parameters[second_matrix_end:second_bias_end]
    output_weights = parameters[second_bias_end:second_bias_end + second_units]
    return first_weights, first_bias, second_weights, second_bias, output_weights


def _deep_shared_mlp_forward(
    values: np.ndarray,
    parameters: np.ndarray,
    input_dim: int,
    hidden_units: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_weights, first_bias, second_weights, second_bias, output_weights = (
        _deep_shared_mlp_unpack(parameters, input_dim, hidden_units)
    )
    first_hidden = np.tanh(values @ first_weights.T + first_bias)
    second_hidden = np.tanh(first_hidden @ second_weights.T + second_bias)
    return first_hidden, second_hidden, second_hidden @ output_weights


def fit_deep_shared_pair_mlp(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_units: tuple[int, int] = DEEP_SHARED_MLP_HIDDEN_UNITS,
    l2: float = DEEP_SHARED_MLP_L2,
    max_iterations: int = DEEP_SHARED_MLP_MAX_ITERATIONS,
) -> dict[str, Any]:
    combined = np.concatenate((left, right), axis=0)
    mean = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized_left = (left - mean) / scale
    normalized_right = (right - mean) / scale
    input_dim = left.shape[1]
    first_units, second_units = hidden_units
    rng = np.random.default_rng(0)
    first_limit = np.sqrt(6.0 / (input_dim + first_units))
    second_limit = np.sqrt(6.0 / (first_units + second_units))
    output_limit = np.sqrt(6.0 / (second_units + 1))
    initial = np.concatenate((
        rng.uniform(-first_limit, first_limit, size=(first_units, input_dim)).ravel(),
        np.zeros(first_units, dtype=float),
        rng.uniform(-second_limit, second_limit, size=(second_units, first_units)).ravel(),
        np.zeros(second_units, dtype=float),
        rng.uniform(-output_limit, output_limit, size=second_units),
    ))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        first_weights, _, second_weights, _, output_weights = _deep_shared_mlp_unpack(
            parameters, input_dim, hidden_units
        )
        left_first, left_second, left_score = _deep_shared_mlp_forward(
            normalized_left, parameters, input_dim, hidden_units
        )
        right_first, right_second, right_score = _deep_shared_mlp_forward(
            normalized_right, parameters, input_dim, hidden_units
        )
        logits = left_score - right_score
        loss = float(
            np.mean(np.logaddexp(0.0, logits) - labels * logits)
            + 0.5 * l2 * (
                np.square(first_weights).sum()
                + np.square(second_weights).sum()
                + np.square(output_weights).sum()
            )
        )
        logit_gradient = (expit(logits) - labels) / len(labels)
        output_gradient = (
            (left_second - right_second).T @ logit_gradient
            + l2 * output_weights
        )
        left_second_pre_gradient = (
            logit_gradient[:, None]
            * output_weights[None, :]
            * (1.0 - np.square(left_second))
        )
        right_second_pre_gradient = (
            -logit_gradient[:, None]
            * output_weights[None, :]
            * (1.0 - np.square(right_second))
        )
        second_weights_gradient = (
            left_second_pre_gradient.T @ left_first
            + right_second_pre_gradient.T @ right_first
            + l2 * second_weights
        )
        second_bias_gradient = (
            left_second_pre_gradient + right_second_pre_gradient
        ).sum(axis=0)
        left_first_pre_gradient = (
            left_second_pre_gradient @ second_weights
        ) * (1.0 - np.square(left_first))
        right_first_pre_gradient = (
            right_second_pre_gradient @ second_weights
        ) * (1.0 - np.square(right_first))
        first_weights_gradient = (
            left_first_pre_gradient.T @ normalized_left
            + right_first_pre_gradient.T @ normalized_right
            + l2 * first_weights
        )
        first_bias_gradient = (
            left_first_pre_gradient + right_first_pre_gradient
        ).sum(axis=0)
        gradient = np.concatenate((
            first_weights_gradient.ravel(),
            first_bias_gradient,
            second_weights_gradient.ravel(),
            second_bias_gradient,
            output_gradient,
        ))
        return loss, gradient

    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    if not optimized.success:
        raise RuntimeError(
            f"deep shared MLP optimization failed: status={optimized.status} "
            f"{optimized.message}; iterations={optimized.nit}; "
            f"final_loss={optimized.fun:.12g}; "
            f"max_abs_gradient={np.max(np.abs(optimized.jac)):.12g}"
        )
    return {
        "parameters": optimized.x,
        "mean": mean,
        "scale": scale,
        "input_dim": input_dim,
        "hidden_units": hidden_units,
        "l2": l2,
        "iterations": int(optimized.nit),
        "final_loss": float(optimized.fun),
    }


def deep_shared_pair_mlp_probability(
    model: dict[str, Any], left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    normalized_left = (left - model["mean"]) / model["scale"]
    normalized_right = (right - model["mean"]) / model["scale"]
    _, _, left_scores = _deep_shared_mlp_forward(
        normalized_left,
        model["parameters"],
        model["input_dim"],
        model["hidden_units"],
    )
    _, _, right_scores = _deep_shared_mlp_forward(
        normalized_right,
        model["parameters"],
        model["input_dim"],
        model["hidden_units"],
    )
    return expit(left_scores - right_scores)


def train_deep_shared_mlp_learning_curve(
    blind_path: Path,
    answers_path: Path,
    feature_checkpoint_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, feature_names = load_option_feature_matrix(blind, answers, checkpoint)
    validation = rows["validation"]
    validation_left = np.stack([row[2] for row in validation])
    validation_right = np.stack([row[3] for row in validation])
    validation_labels = np.asarray([row[4] for row in validation], dtype=int)
    curve = []
    for total in LEARNING_CURVE_TOTALS:
        training = balanced_learning_subset(rows["train"], total)
        training_left = np.stack([row[2] for row in training])
        training_right = np.stack([row[3] for row in training])
        training_labels = np.asarray([row[4] for row in training], dtype=int)
        model = fit_deep_shared_pair_mlp(
            training_left, training_right, training_labels
        )
        probability = deep_shared_pair_mlp_probability(
            model, validation_left, validation_right
        )
        prediction = (probability >= 0.5).astype(int)
        swapped_probability = deep_shared_pair_mlp_probability(
            model, validation_right, validation_left
        )
        if not np.allclose(probability + swapped_probability, 1.0, atol=1e-12):
            raise ValueError("A/B swap probabilities are not complementary")
        strata = {}
        for relation in RELATIONS:
            indexes = [index for index, row in enumerate(validation) if row[1] == relation]
            strata[relation] = {
                "correct": int(sum(
                    prediction[index] == validation_labels[index] for index in indexes
                )),
                "total": len(indexes),
                "accuracy": float(
                    accuracy_score(validation_labels[indexes], prediction[indexes])
                ),
            }
        curve.append({
            "training_total": total,
            "training_relationships": dict(Counter(row[1] for row in training)),
            "validation": {
                "correct": int((prediction == validation_labels).sum()),
                "total": len(validation),
                "accuracy": float(accuracy_score(validation_labels, prediction)),
                "relationships": strata,
            },
            "optimization": {
                "iterations": model["iterations"],
                "final_loss": model["final_loss"],
            },
            "predictions": [
                {
                    "case_id": row[0],
                    "relationship": row[1],
                    "predicted_human_option": "A" if int(prediction[index]) == 1 else "B",
                    "human_option": "A" if int(validation_labels[index]) == 1 else "B",
                    "probability_A": float(probability[index]),
                }
                for index, row in enumerate(validation)
            ],
        })
    result = {
        "schema_version": 1,
        "kind": "judge_deep_shared_pair_mlp_learning_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "shared MLP: 42->16 tanh->8 tanh->1; sigmoid(logit_A-logit_B)",
        "feature_encoding": "raw numeric/boolean values and enum one-hot; A/B kept separate",
        "feature_names": feature_names,
        "hidden_units": list(DEEP_SHARED_MLP_HIDDEN_UNITS),
        "l2": DEEP_SHARED_MLP_L2,
        "max_iterations": DEEP_SHARED_MLP_MAX_ITERATIONS,
        "train_total": TRAIN_TOTAL,
        "validation_total": VALIDATION_TOTAL,
        "learning_curve": curve,
    }
    _write_feature_classifier_result(result_path, result)
    return result


def repair_reference_for_cases(
    reference: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    batch_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove complete leaked examples; refuse leaks that cannot be removed safely."""
    leaks = reference_leakage_details(reference, cases)
    indexes: set[int] = set()
    for leak in leaks:
        path = str(leak.get("reference_path") or "")
        match = re.fullmatch(r"reference\.examples\[(\d+)\](?:\..+)?", path)
        if match is None:
            raise ValueError(f"reference leak is not safely removable as an example: {path}")
        indexes.add(int(match.group(1)))
    repaired = copy.deepcopy(reference)
    examples = list(repaired.get("examples") or [])
    if any(index < 0 or index >= len(examples) for index in indexes):
        raise ValueError("reference leak points outside the examples list")
    repaired["examples"] = [
        example for index, example in enumerate(examples) if index not in indexes
    ]
    repaired.pop("reference_sha256", None)
    repaired["batch_id"] = batch_id
    remaining = reference_leakage_details(repaired, cases)
    if remaining:
        raise ValueError(
            "reference leaks remain after deleting complete examples: "
            + json.dumps(remaining, ensure_ascii=False, sort_keys=True)
        )
    assert_reference_isolated(repaired, cases)
    repaired["reference_sha256"] = _json_sha256(repaired)
    return repaired, {
        "initial_leak_count": len(leaks),
        "removed_example_indexes": sorted(indexes),
        "removed_example_count": len(indexes),
        "remaining_leak_count": 0,
    }


def _validation_projection(
    blind: dict[str, Any], answers: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if answers.get("batch_id") != blind.get("batch_id"):
        raise ValueError("development blind packet and answers use different batches")
    if answers.get("reference_sha256", "") != blind.get("reference_sha256", ""):
        raise ValueError("development blind packet and answers use different references")
    if answers.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("development answers do not match the blind packet")
    blind_cases = list(blind.get("cases") or [])
    answer_cases = list(answers.get("cases") or [])
    blind_ids = [str(row.get("case_id") or "") for row in blind_cases]
    answer_ids = [str(row.get("case_id") or "") for row in answer_cases]
    if not all(blind_ids) or len(set(blind_ids)) != len(blind_ids):
        raise ValueError("development blind packet has missing or duplicate case IDs")
    if answer_ids != blind_ids:
        raise ValueError("development answers must follow the blind case order exactly")
    validation_cases: list[dict[str, Any]] = []
    validation_answers: list[dict[str, Any]] = []
    for case, answer in zip(blind_cases, answer_cases):
        if case.get("split") != answer.get("split"):
            raise ValueError("development blind and answer split labels differ")
        if case.get("split") == "validation":
            validation_cases.append(copy.deepcopy(case))
            validation_answers.append(copy.deepcopy(answer))
    counts = Counter(str(row.get("relationship") or "") for row in validation_cases)
    if len(validation_cases) != VALIDATION_TOTAL:
        raise ValueError(f"validation projection must contain 160 cases, got {len(validation_cases)}")
    if counts != Counter(VALIDATION_TARGETS):
        raise ValueError(f"validation relationship counts are invalid: {dict(counts)}")
    return validation_cases, validation_answers


def validate_stable_validation_artifacts(
    reference: dict[str, Any],
    blind: dict[str, Any],
    answers: dict[str, Any],
) -> None:
    batch_id = str(reference.get("batch_id") or "")
    if not batch_id or blind.get("batch_id") != batch_id or answers.get("batch_id") != batch_id:
        raise ValueError("stable Judge reference, blind packet, and answers must share one batch")
    reference_sha = str(reference.get("reference_sha256") or "")
    unhashed_reference = copy.deepcopy(reference)
    unhashed_reference.pop("reference_sha256", None)
    if not reference_sha or reference_sha != _json_sha256(unhashed_reference):
        raise ValueError("stable Judge reference checksum mismatch")
    if blind.get("reference_sha256") != reference_sha or answers.get("reference_sha256") != reference_sha:
        raise ValueError("stable Judge artifacts do not share one reference checksum")
    if answers.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("stable Judge answers do not match the blind packet")
    cases = list(blind.get("cases") or [])
    answer_cases = list(answers.get("cases") or [])
    case_ids = [str(row.get("case_id") or "") for row in cases]
    answer_ids = [str(row.get("case_id") or "") for row in answer_cases]
    if len(cases) != VALIDATION_TOTAL or len(answer_cases) != VALIDATION_TOTAL:
        raise ValueError("stable Judge validation must contain exactly 160 cases and answers")
    if not all(case_ids) or len(set(case_ids)) != len(case_ids) or answer_ids != case_ids:
        raise ValueError("stable Judge cases and answers must have unique IDs in identical order")
    counts = Counter(str(row.get("relationship") or "") for row in cases)
    if counts != Counter(VALIDATION_TARGETS):
        raise ValueError("stable Judge validation must contain 80 private and 80 group cases")
    if any(row.get("split") != "validation" for row in cases + answer_cases):
        raise ValueError("stable Judge validation contains a non-validation case")
    assert_reference_isolated(reference, cases)


def validate_stable_judge_profile(profile_path: Path) -> dict[str, Any]:
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    if profile.get("name") != STABLE_PROFILE_NAME:
        raise ValueError("stable Judge profile name changed")
    if profile.get("mode") != STABLE_PROFILE_MODE:
        raise ValueError("stable Judge profile must append to the built-in profile")
    actual_sha = hashlib.sha256(profile_bytes).hexdigest()
    if actual_sha != STABLE_PROFILE_SHA256:
        raise ValueError("stable Judge profile checksum changed")
    return profile


def build_stable_judge_validation(
    development_blind_path: Path,
    development_answers_path: Path,
    reference_source_path: Path,
    reference_output_path: Path,
    blind_output_path: Path,
    answers_output_path: Path,
) -> dict[str, Any]:
    development_blind = _load_json(development_blind_path)
    development_answers = _load_json(development_answers_path)
    validation_cases, validation_answers = _validation_projection(
        development_blind, development_answers
    )
    reference, repair = repair_reference_for_cases(
        _load_json(reference_source_path),
        validation_cases,
        batch_id=STABLE_VALIDATION_BATCH_ID,
    )
    if repair["initial_leak_count"] != 1 or repair["removed_example_count"] != 1:
        raise ValueError(
            "stable Judge source reference must contain exactly 1 leak in exactly 1 removable example"
        )
    reference_sha = reference["reference_sha256"]
    blind = copy.deepcopy(development_blind)
    blind["batch_id"] = STABLE_VALIDATION_BATCH_ID
    blind["reference_sha256"] = reference_sha
    blind["cases"] = validation_cases
    answers = copy.deepcopy(development_answers)
    answers["batch_id"] = STABLE_VALIDATION_BATCH_ID
    answers["reference_sha256"] = reference_sha
    answers["cases"] = validation_answers
    answers["blind_sha256"] = _json_sha256(blind)
    validate_stable_validation_artifacts(reference, blind, answers)
    paths = (reference_output_path, blind_output_path, answers_output_path)
    payloads = (reference, blind, answers)
    if any(path.exists() for path in paths):
        if not all(path.exists() for path in paths):
            raise ValueError("stable Judge validation artifacts are only partially present")
        existing = tuple(_load_json(path) for path in paths)
        if existing != payloads:
            raise ValueError("existing stable Judge validation artifacts do not match fixed inputs")
        reused = True
    else:
        for path, payload in zip(paths, payloads):
            _write_private(path, payload)
        reused = False
    return {
        "cases": len(validation_cases),
        "relationships": dict(Counter(row["relationship"] for row in validation_cases)),
        "batch_id": STABLE_VALIDATION_BATCH_ID,
        "reference_sha256": reference_sha,
        "blind_sha256": answers["blind_sha256"],
        "reference_repair": repair,
        "reused_existing_artifacts": reused,
    }


def _validate_completed_rows(
    rows: list[dict[str, Any]],
    all_case_ids: list[str],
    *,
    label: str,
) -> list[str]:
    ids = [str(row.get("case_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains missing or duplicate case IDs")
    order = {case_id: index for index, case_id in enumerate(all_case_ids)}
    if any(case_id not in order for case_id in ids):
        raise ValueError(f"{label} contains a case outside the fixed validation set")
    if ids != sorted(ids, key=order.__getitem__):
        raise ValueError(f"{label} does not follow fixed validation order")
    return ids


def _validate_stable_judge_resume(
    predictions_path: Path,
    audit_path: Path,
    reference: dict[str, Any],
    blind: dict[str, Any],
) -> None:
    if predictions_path.exists() != audit_path.exists():
        raise ValueError("stable Judge resume requires both predictions and audit files")
    if not predictions_path.exists():
        return
    predictions = _load_json(predictions_path)
    audit = _load_json(audit_path)
    expected = {
        "batch_id": blind["batch_id"],
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": reference["reference_sha256"],
        "judge_profile_name": STABLE_PROFILE_NAME,
        "judge_profile_sha256": STABLE_PROFILE_SHA256,
    }
    if any(predictions.get(field) != value for field, value in expected.items()):
        raise ValueError("stable Judge predictions do not match the fixed validation configuration")
    if audit.get("invocation_id") != predictions.get("invocation_id"):
        raise ValueError("stable Judge prediction and audit invocation IDs differ")
    if any(audit.get(field) != value for field, value in expected.items() if field != "batch_id"):
        raise ValueError("stable Judge audit does not match the fixed validation configuration")
    all_case_ids = [str(row["case_id"]) for row in blind["cases"]]
    prediction_ids = _validate_completed_rows(
        list(predictions.get("cases") or []), all_case_ids, label="stable Judge predictions"
    )
    audit_ids = _validate_completed_rows(
        list(audit.get("calls") or []), all_case_ids, label="stable Judge audit"
    )
    if prediction_ids != audit_ids:
        raise ValueError("stable Judge predictions and audit cover different cases")
    expected_indexes = [all_case_ids.index(case_id) + 1 for case_id in audit_ids]
    if [row.get("index") for row in audit.get("calls") or []] != expected_indexes:
        raise ValueError("stable Judge audit indexes do not match fixed validation order")


def run_stable_judge_validation(
    blind_path: Path,
    answers_path: Path,
    reference_path: Path,
    predictions_path: Path,
    audit_path: Path,
    profile_path: Path,
    *,
    max_cases: int | None,
    workers: int,
) -> dict[str, Any]:
    reference = _load_json(reference_path)
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    validate_stable_validation_artifacts(reference, blind, answers)
    validate_stable_judge_profile(profile_path)
    _validate_stable_judge_resume(predictions_path, audit_path, reference, blind)
    result = run_codex_judge_independently(
        blind_path,
        predictions_path,
        audit_path,
        reference_path,
        max_cases=max_cases,
        workers=workers,
        judge_profile_path=profile_path,
    )
    _validate_stable_judge_resume(predictions_path, audit_path, reference, blind)
    return {
        **result,
        "remaining": VALIDATION_TOTAL - int(result["completed"]),
        "profile": STABLE_PROFILE_NAME,
    }


def promotion_gate_results(
    *,
    stable_covered: int,
    classifier_covered: int,
    overall_net: int,
) -> dict[str, dict[str, Any]]:
    return {
        "same_fixed_160_coverage": {
            "required": {"stable_judge": 160, "feature_classifier": 160},
            "actual": {
                "stable_judge": stable_covered,
                "feature_classifier": classifier_covered,
            },
            "passed": stable_covered == 160 and classifier_covered == 160,
        },
        "overall_net_improvement": {
            "required": 1,
            "actual": overall_net,
            "passed": overall_net >= 1,
        },
    }


def _score_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relation in ("overall", *RELATIONS):
        selected = rows if relation == "overall" else [
            row for row in rows if row["relationship"] == relation
        ]
        correct = sum(bool(row["correct"]) for row in selected)
        result[relation] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected) if selected else None,
        }
    return result


def _group_participation_fact_pattern_value(
    option: dict[str, Any],
    relationship: str,
    recent_speakers: tuple[str, ...],
) -> float:
    return float(
        relationship == "group"
        and "__self__" in recent_speakers
        and option["depends_on_unstated_fact"]
        and option["group_pattern_alignment"] >= 1
    )


def _validate_feature_checkpoint_blind_compatibility(
    blind: dict[str, Any],
    checkpoint: dict[str, Any],
    feature_source_blind: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_sha256 = _json_sha256(blind)
    checkpoint_sha256 = str(checkpoint.get("blind_sha256") or "")
    if checkpoint_sha256 == current_sha256:
        return {
            "mode": "exact_packet_hash",
            "checkpoint_blind_sha256": checkpoint_sha256,
            "current_blind_sha256": current_sha256,
        }
    if feature_source_blind is None:
        raise ValueError("feature checkpoint does not match blind packet")
    source_sha256 = _json_sha256(feature_source_blind)
    if checkpoint_sha256 != source_sha256:
        raise ValueError("feature checkpoint does not match its source blind packet")
    if feature_source_blind.get("cases") != blind.get("cases"):
        raise ValueError("feature source and current blind packet cases differ")
    return {
        "mode": "metadata_only_packet_change",
        "checkpoint_blind_sha256": checkpoint_sha256,
        "feature_source_blind_sha256": source_sha256,
        "current_blind_sha256": current_sha256,
        "cases_sha256": _json_sha256(blind.get("cases") or []),
        "cases_exactly_equal": True,
    }


def load_base_correction_feature_rows(
    blind: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    include_group_participation_cross: bool = False,
    feature_source_blind: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str, np.ndarray, str]], list[str]]:
    """Load option-difference features without reading answer labels."""
    _validate_feature_checkpoint_blind_compatibility(
        blind, checkpoint, feature_source_blind
    )
    rows: list[tuple[str, str, np.ndarray, str]] = []
    feature_names: list[str] | None = None
    for case in blind["cases"]:
        case_id = str(case["case_id"])
        entry = (checkpoint.get("entries") or {}).get(case_id) or {}
        if entry.get("status") != "valid":
            raise ValueError(f"missing valid features for case {case_id}")
        left, right, boolean_names = vectorize_boolean_options(entry["features"])
        relationship = str(case["relationship"])
        recent_speakers = tuple(
            reversed(speaker_timeline(case["context_original"]))
        )[:RECENT_SPEAKER_POSITIONS]
        metadata = {"recent_speakers": recent_speakers}
        observable_left, observable_names = vectorize_observable_option_booleans(
            case["option_A"], metadata
        )
        observable_right, other_observable_names = vectorize_observable_option_booleans(
            case["option_B"], metadata
        )
        left_values = np.concatenate((left, observable_left))
        right_values = np.concatenate((right, observable_right))
        names = [*boolean_names, *observable_names]
        if include_group_participation_cross:
            option_a = entry["features"]["option_A"]
            option_b = entry["features"]["option_B"]
            left_values = np.append(
                left_values,
                _group_participation_fact_pattern_value(
                    option_a, relationship, recent_speakers
                ),
            )
            right_values = np.append(
                right_values,
                _group_participation_fact_pattern_value(
                    option_b, relationship, recent_speakers
                ),
            )
            names.append(GROUP_PARTICIPATION_FACT_PATTERN_FEATURE_NAME)
        if observable_names != other_observable_names:
            raise ValueError("A/B observable feature schemas differ")
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("base-correction feature schema changed between cases")
        rows.append((
            case_id,
            relationship,
            left_values - right_values,
            str(case["split"]),
        ))
    expected_feature_count = 73 if include_group_participation_cross else 72
    if len(feature_names or []) != expected_feature_count:
        raise ValueError(
            "base correction feature count differs from the selected branch; "
            f"got {len(feature_names or [])}"
        )
    return rows, feature_names or []


def _labels_for_base_correction_rows(
    rows: list[tuple[str, str, np.ndarray, str]],
    blind: dict[str, Any],
    answers: dict[str, Any],
) -> np.ndarray:
    if answers.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("answer key does not match blind packet")
    answer_rows = list(answers.get("cases") or [])
    expected_ids = [str(row[0]) for row in rows]
    answer_ids = _validate_completed_rows(
        answer_rows, expected_ids, label="base-correction answers"
    )
    if answer_ids != expected_ids:
        raise ValueError("base-correction answers do not follow blind order")
    labels: list[int] = []
    for row, answer in zip(rows, answer_rows):
        option = str(answer.get("human_option") or "")
        if option not in {"A", "B"}:
            raise ValueError(f"invalid human option for case {row[0]}")
        if str(answer.get("split") or "") != row[3]:
            raise ValueError(f"blind and answer splits differ for case {row[0]}")
        labels.append(1 if option == "A" else 0)
    return np.asarray(labels, dtype=int)


def _base_prediction_difference_features(
    selected_option: str,
    confidence: float,
) -> np.ndarray:
    if selected_option not in {"A", "B"}:
        raise ValueError("Base prediction must select A or B")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Base confidence must be between 0 and 1")
    direction = 1.0 if selected_option == "A" else -1.0
    return np.asarray([
        direction,
        *(
            direction * float(confidence >= threshold)
            for threshold in BASE_CONFIDENCE_THRESHOLDS
        ),
    ])


def _load_base_prediction_features(
    blind: dict[str, Any],
    predictions: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if predictions.get("blind_sha256") != _json_sha256(blind):
        raise ValueError("Base predictions do not match blind packet")
    if (
        predictions.get("judge_profile_name") != STABLE_PROFILE_NAME
        or predictions.get("judge_profile_sha256") != STABLE_PROFILE_SHA256
        or predictions.get("presentation") != "canonical"
    ):
        raise ValueError("Base predictions use a different Judge configuration")
    expected_ids = [str(row["case_id"]) for row in blind["cases"]]
    prediction_rows = list(predictions.get("cases") or [])
    prediction_ids = _validate_completed_rows(
        prediction_rows, expected_ids, label="Base predictions"
    )
    if prediction_ids != expected_ids:
        raise ValueError("Base predictions do not follow blind order")
    choices: list[int] = []
    confidences: list[float] = []
    features: list[np.ndarray] = []
    for row in prediction_rows:
        option = str(row.get("human_option") or "")
        try:
            confidence = float(row["confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Base prediction confidence is missing or invalid") from error
        features.append(_base_prediction_difference_features(option, confidence))
        choices.append(1 if option == "A" else 0)
        confidences.append(confidence)
    names = [
        "base_selects_this_option",
        *(f"base_selects_this_option_at_confidence>={threshold:.2f}"
          for threshold in BASE_CONFIDENCE_THRESHOLDS),
    ]
    return (
        np.stack(features),
        np.asarray(choices, dtype=int),
        np.asarray(confidences, dtype=float),
        names,
    )


def _apply_base_correction(
    base_predictions: np.ndarray,
    probability_a: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(base_predictions) != len(probability_a):
        raise ValueError("Base and correction predictions are not aligned")
    classifier_predictions = (probability_a >= 0.5).astype(int)
    classifier_confidence = np.maximum(probability_a, 1.0 - probability_a)
    flipped = (
        (classifier_predictions != base_predictions)
        & (classifier_confidence >= threshold)
    )
    corrected = np.where(flipped, classifier_predictions, base_predictions)
    return corrected.astype(int), flipped


def _evaluate_base_correction(
    rows: list[tuple[str, str, np.ndarray, str]],
    labels: np.ndarray,
    base_predictions: np.ndarray,
    base_confidences: np.ndarray,
    probability_a: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    corrected, flipped = _apply_base_correction(
        base_predictions, probability_a, threshold
    )
    classifier_confidence = np.maximum(probability_a, 1.0 - probability_a)
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        base_correct = bool(base_predictions[index] == labels[index])
        corrected_correct = bool(corrected[index] == labels[index])
        cases.append({
            "case_id": row[0],
            "relationship": row[1],
            "human_option": "A" if labels[index] == 1 else "B",
            "base_prediction": "A" if base_predictions[index] == 1 else "B",
            "base_confidence": float(base_confidences[index]),
            "small_model_prediction": "A" if probability_a[index] >= 0.5 else "B",
            "small_model_probability_A": float(probability_a[index]),
            "small_model_confidence": float(classifier_confidence[index]),
            "final_prediction": "A" if corrected[index] == 1 else "B",
            "flipped_base": bool(flipped[index]),
            "base_correct": base_correct,
            "final_correct": corrected_correct,
            "corrected_base_error": not base_correct and corrected_correct,
            "new_error_from_flip": base_correct and not corrected_correct,
            "high_confidence_new_error": bool(
                base_correct
                and not corrected_correct
                and classifier_confidence[index] >= HIGH_CONFIDENCE_THRESHOLD
            ),
        })
    base_scores = _score_prediction_rows([
        {"relationship": row["relationship"], "correct": row["base_correct"]}
        for row in cases
    ])
    final_scores = _score_prediction_rows([
        {"relationship": row["relationship"], "correct": row["final_correct"]}
        for row in cases
    ])
    net = {
        relation: final_scores[relation]["correct"] - base_scores[relation]["correct"]
        for relation in ("overall", *RELATIONS)
    }
    corrected_cases = [row for row in cases if row["corrected_base_error"]]
    new_errors = [row for row in cases if row["new_error_from_flip"]]
    high_confidence_new_errors = [
        row for row in cases if row["high_confidence_new_error"]
    ]
    gates = {
        "overall_net_improvement": {
            "required": 1,
            "actual": net["overall"],
            "passed": net["overall"] >= 1,
        },
    }
    return {
        "threshold": threshold,
        "scores": {
            "large_model_base": base_scores,
            "conservative_correction": final_scores,
            "net_change": net,
        },
        "changed_cases": {
            "flipped_count": int(flipped.sum()),
            "corrected_count": len(corrected_cases),
            "new_error_count": len(new_errors),
            "high_confidence_new_error_count": len(high_confidence_new_errors),
            "corrected_case_ids": [row["case_id"] for row in corrected_cases],
            "new_error_case_ids": [row["case_id"] for row in new_errors],
        },
        "promotion_gate": gates,
        "passed": all(row["passed"] for row in gates.values()),
        "cases": cases,
    }


def _select_base_correction_threshold(
    rows: list[tuple[str, str, np.ndarray, str]],
    labels: np.ndarray,
    base_predictions: np.ndarray,
    base_confidences: np.ndarray,
    probability_a: np.ndarray,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    evaluations = [
        _evaluate_base_correction(
            rows,
            labels,
            base_predictions,
            base_confidences,
            probability_a,
            threshold,
        )
        for threshold in BASE_CORRECTION_THRESHOLD_GRID
    ]
    passing = [row for row in evaluations if row["passed"]]
    if not passing:
        return None, evaluations
    selected = max(
        passing,
        key=lambda row: (
            row["scores"]["net_change"]["overall"],
            row["threshold"],
            -row["changed_cases"]["flipped_count"],
        ),
    )
    return selected, evaluations


def _reworked_option_vector(row: tuple[Any, ...], side: str) -> np.ndarray:
    if side == "A":
        indexes = (4, 7, 9)
    elif side == "B":
        indexes = (5, 8, 10)
    else:
        raise ValueError("side must be A or B")
    vector = np.concatenate(tuple(row[index] for index in indexes))
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError("reworked option features must be boolean")
    return vector


def _reworked_shared_option_matrix(
    rows: list[tuple[Any, ...]], side: str
) -> np.ndarray:
    values = [
        np.concatenate((_reworked_option_vector(row, side), row[6]))
        for row in rows
    ]
    matrix = np.stack(values)
    if not np.isin(matrix, (0.0, 1.0)).all():
        raise ValueError("reworked shared scorer inputs must be boolean")
    return matrix


def _shared_tree_pair_probability(
    classifier: HistGradientBoostingClassifier,
    rows: list[tuple[Any, ...]],
) -> np.ndarray:
    left_logit = classifier.decision_function(
        _reworked_shared_option_matrix(rows, "A")
    )
    right_logit = classifier.decision_function(
        _reworked_shared_option_matrix(rows, "B")
    )
    probability_a = expit(left_logit - right_logit)
    swapped_probability_a = expit(right_logit - left_logit)
    if not np.allclose(probability_a + swapped_probability_a, 1.0, atol=1e-12):
        raise ValueError("shared tree A/B swap probabilities are not complementary")
    return probability_a


def train_reworked_boolean_correction_development(
    blind_path: Path,
    answers_path: Path,
    replay_path: Path,
    feature_checkpoint_path: Path,
    validation_blind_path: Path,
    base_predictions_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    replay = _load_json(replay_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, _, boolean_names, context_names, context_schema = load_boolean_context_rows(
        blind,
        answers,
        replay,
        checkpoint,
        expected_split_counts={"train": TRAIN_TOTAL, "validation": VALIDATION_TOTAL},
    )
    training_rows = rows["train"]
    validation_rows = rows["validation"]
    training_labels = np.asarray([row[-1] for row in training_rows], dtype=int)
    validation_labels = np.asarray([row[-1] for row in validation_rows], dtype=int)
    validation_blind = _load_json(validation_blind_path)
    if [row[0] for row in validation_rows] != [
        str(case["case_id"]) for case in validation_blind["cases"]
    ]:
        raise ValueError("Base validation packet differs from the development 160")
    base_payload = _load_json(base_predictions_path)
    _, base_predictions, base_confidences, _ = _load_base_prediction_features(
        validation_blind, base_payload
    )

    option_names = [
        *boolean_names,
        *OBSERVABLE_OPTION_BOOLEAN_NAMES,
        *GROUP_PATTERN_OPTION_BOOLEAN_NAMES,
    ]
    sample_linear, linear_feature_names = expand_boolean_option_with_refined_context(
        _reworked_option_vector(training_rows[0], "A"),
        option_names,
        training_rows[0][6],
        context_names,
    )
    linear_training = _refined_boolean_difference_matrix(
        training_rows,
        option_names,
        context_names,
        include_context_crosses=True,
        include_group_pattern=True,
    )
    linear_validation = _refined_boolean_difference_matrix(
        validation_rows,
        option_names,
        context_names,
        include_context_crosses=True,
        include_group_pattern=True,
    )
    if linear_training.shape[1] != len(sample_linear):
        raise ValueError("reworked linear feature schema is inconsistent")
    linear_classifier = LogisticRegression(
        fit_intercept=False,
        max_iter=5000,
        random_state=0,
    )
    linear_classifier.fit(linear_training, training_labels)
    linear_logits = linear_classifier.decision_function(linear_validation)
    linear_probability = expit(linear_logits)
    if not np.allclose(linear_probability + expit(-linear_logits), 1.0, atol=1e-12):
        raise ValueError("reworked linear A/B swap probabilities are not complementary")

    tree_training_left = _reworked_shared_option_matrix(training_rows, "A")
    tree_training_right = _reworked_shared_option_matrix(training_rows, "B")
    tree_classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=0,
    )
    tree_classifier.fit(
        np.vstack((tree_training_left, tree_training_right)),
        np.concatenate((training_labels, 1 - training_labels)),
    )
    tree_probability = _shared_tree_pair_probability(
        tree_classifier, validation_rows
    )

    candidates: list[dict[str, Any]] = []
    candidate_specs = (
        (
            "symmetric_logistic_refined_boolean_crosses",
            linear_probability,
            {
                "model": "LogisticRegression",
                "fit_intercept": False,
                "regularization": "L2",
                "C": 1.0,
                "max_iter": 5000,
                "random_state": 0,
                "input_feature_count": len(linear_feature_names),
                "feature_names": linear_feature_names,
                "coefficients": linear_classifier.coef_[0].tolist(),
            },
        ),
        (
            "shared_hist_gradient_boosting_boolean_scorer",
            tree_probability,
            {
                "model": "HistGradientBoostingClassifier",
                "learning_rate": 0.05,
                "max_iter": 150,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                "random_state": 0,
                "input_feature_count": len(option_names) + len(context_names),
                "feature_names": [
                    *(f"option:{name}" for name in option_names),
                    *(f"context:{name}" for name in context_names),
                ],
                "training_option_instances": TRAIN_TOTAL * 2,
                "pair_probability": "sigmoid(shared_logit_A-shared_logit_B)",
                "retrain_from_frozen_1200": True,
            },
        ),
    )
    for model_name, probability_a, model_spec in candidate_specs:
        selected, evaluations = _select_base_correction_threshold(
            validation_rows,
            validation_labels,
            base_predictions,
            base_confidences,
            probability_a,
        )
        candidates.append({
            "name": model_name,
            "model": model_spec,
            "selected_development_result": selected,
            "threshold_evaluations": [
                {key: value for key, value in row.items() if key != "cases"}
                for row in evaluations
            ],
        })
    passing_candidates = [
        row for row in candidates if row["selected_development_result"] is not None
    ]
    selected_candidate = None
    if passing_candidates:
        selected_candidate = max(
            passing_candidates,
            key=lambda row: (
                row["selected_development_result"]["scores"]["net_change"]["overall"],
                -row["selected_development_result"]["changed_cases"]["new_error_count"],
                row["selected_development_result"]["threshold"],
            ),
        )
    final_model = None
    selected_result = None
    if selected_candidate is not None:
        selected_result = selected_candidate["selected_development_result"]
        final_model = {
            **selected_candidate["model"],
            "name": selected_candidate["name"],
            "correction_threshold": selected_result["threshold"],
        }

    result = {
        "schema_version": 1,
        "kind": "judge_reworked_boolean_conservative_correction_development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "only_variable": "reworked boolean feature system and small-model family",
        "large_model_calls": 0,
        "large_model_base_reused": True,
        "large_model_base": {
            "prediction_artifact": str(base_predictions_path),
            "prediction_sha256": _json_sha256(base_payload),
            "judge_profile_name": STABLE_PROFILE_NAME,
            "judge_profile_sha256": STABLE_PROFILE_SHA256,
            "presentation": "canonical",
        },
        "training": {
            "training_cases": len(training_rows),
            "development_cases": len(validation_rows),
            "development_used_for_model_fit": False,
            "development_used_for_model_and_threshold_selection": True,
            "model_candidates_compared": len(candidates),
            "threshold_grid": list(BASE_CORRECTION_THRESHOLD_GRID),
        },
        "feature_system": {
            "encoding": "boolean facts; enum values are independent one-hot booleans",
            "llm_option_boolean_count": len(boolean_names),
            "observable_option_boolean_count": len(OBSERVABLE_OPTION_BOOLEAN_NAMES),
            "group_pattern_option_boolean_count": len(GROUP_PATTERN_OPTION_BOOLEAN_NAMES),
            "option_boolean_count": len(option_names),
            "context_boolean_count": len(context_names),
            "refined_linear_cross_count": len(linear_feature_names) - len(option_names),
            "linear_input_feature_count": len(linear_feature_names),
            "tree_input_feature_count": len(option_names) + len(context_names),
            "option_feature_names": option_names,
            "context_feature_names": context_names,
            "context_schema": {
                key: list(value) for key, value in context_schema.items()
            },
        },
        "correction_policy": {
            "large_model_base_used_as_small_model_feature": False,
            "large_model_base_feature_omission_reason": (
                "the frozen 1,200 training cases do not have predictions from the same "
                "official Base configuration; no training feature was fabricated"
            ),
            "large_model_base_used_as_default_prediction": True,
            "flip_only_on_small_model_disagreement_above_threshold": True,
            "promotion_rule": "overall net improvement on the 160-case development set",
        },
        "model_candidates": candidates,
        "selected_model_name": (
            selected_candidate["name"] if selected_candidate is not None else None
        ),
        "selected_development_result": selected_result,
        "final_model": final_model,
        "decision": "eligible_for_next_stage" if selected_candidate is not None else "reject",
        "entered_fixed_240": False,
        "fixed_240_answers_read": False,
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def train_base_correction_development(
    blind_path: Path,
    answers_path: Path,
    feature_checkpoint_path: Path,
    validation_blind_path: Path,
    base_predictions_path: Path,
    result_path: Path,
    *,
    include_group_participation_cross: bool = False,
) -> dict[str, Any]:
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    checkpoint = _load_json(feature_checkpoint_path)
    rows, original_names = load_base_correction_feature_rows(
        blind,
        checkpoint,
        include_group_participation_cross=include_group_participation_cross,
    )
    labels = _labels_for_base_correction_rows(rows, blind, answers)
    training_indexes = [index for index, row in enumerate(rows) if row[3] == "train"]
    validation_indexes = [
        index for index, row in enumerate(rows) if row[3] == "validation"
    ]
    if len(training_indexes) != TRAIN_TOTAL or len(validation_indexes) != VALIDATION_TOTAL:
        raise ValueError("Base correction requires 1,200 train and 160 development cases")
    validation_rows = [rows[index] for index in validation_indexes]
    validation_labels = labels[validation_indexes]
    validation_blind = _load_json(validation_blind_path)
    if [row[0] for row in validation_rows] != [
        str(case["case_id"]) for case in validation_blind["cases"]
    ]:
        raise ValueError("Base validation packet differs from the development 160")
    base_payload = _load_json(base_predictions_path)
    _, base_predictions, base_confidences, _ = (
        _load_base_prediction_features(validation_blind, base_payload)
    )
    x_original = np.stack([row[2] for row in rows])
    training_original = x_original[training_indexes]
    validation_original = x_original[validation_indexes]
    training_labels = labels[training_indexes]
    classifier = LogisticRegression(
        fit_intercept=False,
        max_iter=5000,
        random_state=0,
    )
    classifier.fit(training_original, training_labels)
    development_probability = expit(
        classifier.decision_function(validation_original)
    )
    selected, threshold_evaluations = _select_base_correction_threshold(
        validation_rows,
        validation_labels,
        base_predictions,
        base_confidences,
        development_probability,
    )
    feature_names = list(original_names)
    final_model: dict[str, Any] | None = None
    if selected is not None:
        final_model = {
            "model": "LogisticRegression",
            "fit_intercept": False,
            "regularization": "L2",
            "C": 1.0,
            "max_iter": 5000,
            "random_state": 0,
            "feature_names": feature_names,
            "coefficients": classifier.coef_[0].tolist(),
            "threshold": selected["threshold"],
        }
    result = {
        "schema_version": 1,
        "kind": "judge_large_model_base_conservative_correction_development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "only_variable": (
            GROUP_PARTICIPATION_FACT_PATTERN_FEATURE_NAME
            if include_group_participation_cross
            else "use the frozen 72-feature classifier as a conservative correction layer "
            "over the existing large-model Base"
        ),
        "large_model_calls": 0,
        "large_model_base_reused": True,
        "large_model_base": {
            "prediction_artifact": str(base_predictions_path),
            "prediction_sha256": _json_sha256(base_payload),
            "judge_profile_name": STABLE_PROFILE_NAME,
            "judge_profile_sha256": STABLE_PROFILE_SHA256,
            "presentation": "canonical",
        },
        "training": {
            "training_cases": len(training_indexes),
            "development_cases": len(validation_indexes),
            "development_used_for_model_fit": False,
            "development_used_for_threshold_selection": True,
            "model_fit_count": 1,
            "threshold_grid": list(BASE_CORRECTION_THRESHOLD_GRID),
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "small_model_feature_variant": (
            "group_participation_fact_pattern_cross"
            if include_group_participation_cross
            else "frozen_72"
        ),
        "correction_policy": {
            "large_model_base_used_as_small_model_feature": False,
            "large_model_base_used_as_default_prediction": True,
            "flip_only_on_small_model_disagreement_above_threshold": True,
        },
        "threshold_evaluations": [
            {key: value for key, value in row.items() if key != "cases"}
            for row in threshold_evaluations
        ],
        "selected_development_result": selected,
        "final_model": final_model,
        "decision": "enter_fixed_240" if selected is not None else "reject",
        "entered_fixed_240": False,
        "fixed_240_answers_read": False,
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def validate_base_correction_fixed_240(
    development_result_path: Path,
    blind_path: Path,
    answers_path: Path,
    feature_checkpoint_path: Path,
    base_predictions_path: Path,
    result_path: Path,
    feature_source_blind_path: Path | None = None,
) -> dict[str, Any]:
    development = _load_json(development_result_path)
    if development.get("decision") != "enter_fixed_240":
        raise ValueError("development gate did not allow fixed-240 validation")
    model = development.get("final_model") or {}
    threshold = float(model["threshold"])
    feature_names = list(model.get("feature_names") or [])
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    feature_variant = str(
        development.get("small_model_feature_variant") or "frozen_72"
    )
    if feature_variant not in {"frozen_72", "group_participation_fact_pattern_cross"}:
        raise ValueError("unsupported development feature variant")
    training = development.get("training") or {}
    if (
        training.get("training_cases") != TRAIN_TOTAL
        or training.get("development_cases") != VALIDATION_TOTAL
        or training.get("development_used_for_model_fit") is not False
        or training.get("development_used_for_threshold_selection") is not True
        or training.get("model_fit_count") != 1
    ):
        raise ValueError("development result does not preserve the 1200/160 data boundary")

    # Predict all fixed cases before the answer key is opened.
    blind = _load_json(blind_path)
    checkpoint = _load_json(feature_checkpoint_path)
    feature_source_blind = (
        _load_json(feature_source_blind_path)
        if feature_source_blind_path is not None
        else None
    )
    feature_checkpoint_compatibility = (
        _validate_feature_checkpoint_blind_compatibility(
            blind, checkpoint, feature_source_blind
        )
    )
    rows, original_names = load_base_correction_feature_rows(
        blind,
        checkpoint,
        include_group_participation_cross=(
            feature_variant == "group_participation_fact_pattern_cross"
        ),
        feature_source_blind=feature_source_blind,
    )
    if len(rows) != GROUP_PATTERN_HOLDOUT_TOTAL:
        raise ValueError("fixed validation must contain exactly 240 cases")
    relationship_counts = Counter(row[1] for row in rows)
    if relationship_counts != Counter(GROUP_PATTERN_HOLDOUT_TARGETS):
        raise ValueError("fixed validation must contain 80 private and 160 group cases")
    base_payload = _load_json(base_predictions_path)
    _, base_predictions, base_confidences, _ = (
        _load_base_prediction_features(blind, base_payload)
    )
    if feature_names != original_names or len(coefficients) != len(feature_names):
        raise ValueError("fixed validation feature schema differs from development")
    probability_a = expit(
        np.stack([row[2] for row in rows]) @ coefficients
    )

    answers = _load_json(answers_path)
    labels = _labels_for_base_correction_rows(rows, blind, answers)
    evaluation = _evaluate_base_correction(
        rows,
        labels,
        base_predictions,
        base_confidences,
        probability_a,
        threshold,
    )
    adopted = bool(evaluation["passed"])
    result = {
        "schema_version": 1,
        "kind": "judge_large_model_base_conservative_correction_fixed_240",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_result": str(development_result_path),
        "development_result_sha256": _json_sha256(development),
        "model_and_threshold_selected_without_fixed_240": True,
        "fixed_240_answers_read_after_predictions": True,
        "fixed_240": {
            "total": len(rows),
            "relationships": dict(relationship_counts),
            "used_for_training": False,
            "used_for_threshold_selection": False,
        },
        "feature_checkpoint_compatibility": feature_checkpoint_compatibility,
        "large_model_calls": 0,
        "large_model_base_reused": True,
        "evaluation": evaluation,
        "decision": "adopt" if adopted else "reject",
        "official_judge_baseline": False,
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def validate_reworked_boolean_correction_fixed_240(
    development_result_path: Path,
    blind_path: Path,
    answers_path: Path,
    replay_path: Path,
    feature_checkpoint_path: Path,
    base_predictions_path: Path,
    result_path: Path,
    feature_source_blind_path: Path,
    feature_source_answers_path: Path,
) -> dict[str, Any]:
    development = _load_json(development_result_path)
    if development.get("decision") != "eligible_for_next_stage":
        raise ValueError("development gate did not allow fixed-240 validation")
    model = development.get("final_model") or {}
    if (
        development.get("selected_model_name")
        != "symmetric_logistic_refined_boolean_crosses"
        or model.get("name") != "symmetric_logistic_refined_boolean_crosses"
        or model.get("model") != "LogisticRegression"
        or int(model.get("input_feature_count") or 0) != 524
    ):
        raise ValueError("fixed-240 validation requires the frozen 524-feature model")
    threshold = float(model["correction_threshold"])
    feature_names = list(model.get("feature_names") or [])
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    if threshold != 0.7 or len(feature_names) != 524 or len(coefficients) != 524:
        raise ValueError("fixed-240 model or threshold differs from development")
    training = development.get("training") or {}
    if (
        training.get("training_cases") != TRAIN_TOTAL
        or training.get("development_cases") != VALIDATION_TOTAL
        or training.get("development_used_for_model_fit") is not False
        or training.get("development_used_for_model_and_threshold_selection") is not True
    ):
        raise ValueError("development result does not preserve the 1200/160 data boundary")

    blind = _load_json(blind_path)
    checkpoint = _load_json(feature_checkpoint_path)
    feature_source_blind = _load_json(feature_source_blind_path)
    feature_checkpoint_compatibility = _validate_feature_checkpoint_blind_compatibility(
        blind, checkpoint, feature_source_blind
    )
    feature_source_answers = _load_json(feature_source_answers_path)
    replay = _load_json(replay_path)
    frozen_context_schema = {
        key: tuple(value)
        for key, value in (development.get("feature_system") or {})
        .get("context_schema", {})
        .items()
    }
    rows_by_split, _, boolean_names, context_names, _ = load_boolean_context_rows(
        feature_source_blind,
        feature_source_answers,
        replay,
        checkpoint,
        context_schema=frozen_context_schema,
        expected_split_counts={"holdout": GROUP_PATTERN_HOLDOUT_TOTAL},
    )
    rows = rows_by_split["holdout"]
    if [row[0] for row in rows] != [str(case["case_id"]) for case in blind["cases"]]:
        raise ValueError("fixed-240 feature rows differ from the Base packet")
    relationship_counts = Counter(row[1] for row in rows)
    if relationship_counts != Counter(GROUP_PATTERN_HOLDOUT_TARGETS):
        raise ValueError("fixed validation must contain 80 private and 160 group cases")

    option_names = [
        *boolean_names,
        *OBSERVABLE_OPTION_BOOLEAN_NAMES,
        *GROUP_PATTERN_OPTION_BOOLEAN_NAMES,
    ]
    _, current_feature_names = expand_boolean_option_with_refined_context(
        _reworked_option_vector(rows[0], "A"),
        option_names,
        rows[0][6],
        context_names,
    )
    if current_feature_names != feature_names:
        raise ValueError("fixed-240 feature schema differs from development")
    matrix = _refined_boolean_difference_matrix(
        rows,
        option_names,
        context_names,
        include_context_crosses=True,
        include_group_pattern=True,
    )
    probability_a = expit(np.asarray(matrix @ coefficients).reshape(-1))
    base_payload = _load_json(base_predictions_path)
    _, base_predictions, base_confidences, _ = _load_base_prediction_features(
        blind, base_payload
    )

    answers = _load_json(answers_path)
    if answers.get("cases") != feature_source_answers.get("cases"):
        raise ValueError("fixed-240 answer packets disagree")
    label_rows = [
        (row[0], row[1], np.asarray([]), str(case["split"]))
        for row, case in zip(rows, blind["cases"])
    ]
    labels = _labels_for_base_correction_rows(label_rows, blind, answers)
    evaluation = _evaluate_base_correction(
        rows,
        labels,
        base_predictions,
        base_confidences,
        probability_a,
        threshold,
    )
    result = {
        "schema_version": 1,
        "kind": "judge_reworked_boolean_conservative_correction_fixed_240",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_result": str(development_result_path),
        "development_result_sha256": _json_sha256(development),
        "model_and_threshold_frozen_from_development": True,
        "fixed_240": {
            "total": len(rows),
            "relationships": dict(relationship_counts),
            "used_for_training": False,
            "used_for_model_selection": False,
            "used_for_threshold_selection": False,
        },
        "feature_checkpoint_compatibility": feature_checkpoint_compatibility,
        "large_model_calls": 0,
        "large_model_base_reused": True,
        "selected_model_name": model["name"],
        "correction_threshold": threshold,
        "input_feature_count": len(feature_names),
        "evaluation": evaluation,
        "decision": "validated" if evaluation["passed"] else "reject",
        "official_judge_baseline": False,
        "online_components_modified": [],
    }
    _write_feature_classifier_result(result_path, result)
    return result


def compare_stable_judge_classifier(
    blind_path: Path,
    answers_path: Path,
    reference_path: Path,
    stable_predictions_path: Path,
    learning_curve_path: Path,
    profile_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    reference = _load_json(reference_path)
    blind = _load_json(blind_path)
    answers = _load_json(answers_path)
    validate_stable_validation_artifacts(reference, blind, answers)
    validate_stable_judge_profile(profile_path)
    stable = _load_json(stable_predictions_path)
    expected_stable = {
        "batch_id": blind["batch_id"],
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": reference["reference_sha256"],
        "judge_profile_name": STABLE_PROFILE_NAME,
        "judge_profile_sha256": STABLE_PROFILE_SHA256,
    }
    if any(stable.get(field) != value for field, value in expected_stable.items()):
        raise ValueError("stable Judge predictions use a different fixed configuration")
    learning_curve = _load_json(learning_curve_path)
    matching = [
        row for row in learning_curve.get("learning_curve") or []
        if row.get("training_total") == TRAIN_TOTAL
    ]
    if len(matching) != 1:
        raise ValueError("learning curve must contain exactly one 1,200-example classifier")
    classifier_rows = list(matching[0].get("predictions") or [])
    stable_rows = list(stable.get("cases") or [])
    cases = list(blind["cases"])
    answer_rows = list(answers["cases"])
    fixed_ids = [str(row["case_id"]) for row in cases]
    stable_ids = _validate_completed_rows(stable_rows, fixed_ids, label="stable Judge predictions")
    classifier_ids = _validate_completed_rows(
        classifier_rows, fixed_ids, label="feature classifier predictions"
    )
    if stable_ids != fixed_ids or classifier_ids != fixed_ids:
        raise ValueError("both candidates must cover the same fixed 160 cases in order")
    comparisons: list[dict[str, Any]] = []
    for case, answer, judge_row, classifier_row in zip(
        cases, answer_rows, stable_rows, classifier_rows
    ):
        case_id = str(case["case_id"])
        actual = str(answer["human_option"])
        if (
            str(answer["case_id"]) != case_id
            or str(judge_row["case_id"]) != case_id
            or str(classifier_row["case_id"]) != case_id
            or classifier_row.get("human_option") != actual
            or classifier_row.get("relationship") != case.get("relationship")
        ):
            raise ValueError(f"comparison inputs disagree for case {case_id}")
        judge_prediction = str(judge_row.get("human_option") or "")
        classifier_prediction = str(classifier_row.get("predicted_human_option") or "")
        if judge_prediction not in {"A", "B"} or classifier_prediction not in {"A", "B"}:
            raise ValueError(f"invalid A/B prediction for case {case_id}")
        probability_a = float(classifier_row["probability_A"])
        if not 0.0 <= probability_a <= 1.0:
            raise ValueError(f"invalid classifier probability for case {case_id}")
        confidence = probability_a if classifier_prediction == "A" else 1.0 - probability_a
        judge_correct = judge_prediction == actual
        classifier_correct = classifier_prediction == actual
        comparisons.append({
            "case_id": case_id,
            "relationship": case["relationship"],
            "human_option": actual,
            "stable_judge_prediction": judge_prediction,
            "feature_classifier_prediction": classifier_prediction,
            "feature_classifier_probability_A": probability_a,
            "feature_classifier_confidence": confidence,
            "stable_judge_correct": judge_correct,
            "feature_classifier_correct": classifier_correct,
            "corrected_by_classifier": not judge_correct and classifier_correct,
            "new_error_from_classifier": judge_correct and not classifier_correct,
            "high_confidence_new_error": (
                judge_correct and not classifier_correct
                and confidence >= HIGH_CONFIDENCE_THRESHOLD
            ),
        })
    stable_scoring_rows = [
        {"relationship": row["relationship"], "correct": row["stable_judge_correct"]}
        for row in comparisons
    ]
    classifier_scoring_rows = [
        {"relationship": row["relationship"], "correct": row["feature_classifier_correct"]}
        for row in comparisons
    ]
    stable_scores = _score_prediction_rows(stable_scoring_rows)
    classifier_scores = _score_prediction_rows(classifier_scoring_rows)
    net = {
        relation: classifier_scores[relation]["correct"] - stable_scores[relation]["correct"]
        for relation in ("overall", *RELATIONS)
    }
    corrected = [row for row in comparisons if row["corrected_by_classifier"]]
    new_errors = [row for row in comparisons if row["new_error_from_classifier"]]
    high_confidence_new_errors = [
        row for row in comparisons if row["high_confidence_new_error"]
    ]
    gates = promotion_gate_results(
        stable_covered=len(stable_rows),
        classifier_covered=len(classifier_rows),
        overall_net=net["overall"],
    )
    adopted = all(row["passed"] for row in gates.values())
    result = {
        "schema_version": 1,
        "kind": "stable_judge_vs_feature_classifier_1200",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": blind["batch_id"],
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": reference["reference_sha256"],
        "stable_judge_profile_name": STABLE_PROFILE_NAME,
        "stable_judge_profile_sha256": STABLE_PROFILE_SHA256,
        "training": {"total": 1200, "private": 600, "group": 600},
        "validation": {"total": 160, "private": 80, "group": 80},
        "scores": {
            "stable_judge": stable_scores,
            "feature_classifier": classifier_scores,
            "feature_classifier_net_change": net,
        },
        "changed_cases": {
            "corrected_count": len(corrected),
            "new_error_count": len(new_errors),
            "high_confidence_new_error_count": len(high_confidence_new_errors),
            "corrected": corrected,
            "new_errors": new_errors,
            "high_confidence_new_errors": high_confidence_new_errors,
        },
        "promotion_gate": gates,
        "decision": "adopt" if adopted else "reject",
        "entered_fixed_240": False,
        "online_components_modified": [],
        "cases": comparisons,
    }
    _write_feature_classifier_result(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal-holdout-guard")
    seal.add_argument("--replay", type=Path, required=True)
    seal.add_argument("--output", type=Path, default=DEFAULT_GUARD)

    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    initialize.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    initialize.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    initialize.add_argument("--wiki", type=Path, default=DEFAULT_WIKI)

    generate = subparsers.add_parser("generate-c0")
    generate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    generate.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    generate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    generate.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    generate.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    generate.add_argument("--max-calls", type=int)
    generate.add_argument("--retry-failed", action="store_true")
    generate.add_argument("--relationship", choices=RELATIONS)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--shard-count", type=int, default=1)

    upgrade = subparsers.add_parser("upgrade-c0-input-contract")
    upgrade.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    upgrade.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    upgrade.add_argument("--source", type=Path, required=True)
    upgrade.add_argument("--target", type=Path, required=True)
    upgrade.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    upgrade.add_argument("--shard-index", type=int, required=True)
    upgrade.add_argument("--shard-count", type=int, required=True)

    merge = subparsers.add_parser("merge-c0-checkpoints")
    merge.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    merge.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    merge.add_argument("--target", type=Path, default=DEFAULT_CHECKPOINT)
    merge.add_argument("--sources", type=Path, nargs="+", required=True)

    merge_shards = subparsers.add_parser("merge-owned-c0-shards")
    merge_shards.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    merge_shards.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    merge_shards.add_argument("--target", type=Path, required=True)
    merge_shards.add_argument("--sources", type=Path, nargs="+", required=True)

    freeze = subparsers.add_parser("freeze-pairs")
    freeze.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    freeze.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    freeze.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    freeze.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    freeze.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    freeze.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    freeze.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)

    features = subparsers.add_parser("extract-features")
    features.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    features.add_argument("--checkpoint", type=Path, default=DEFAULT_FEATURES)
    features.add_argument("--max-cases", type=int)
    features.add_argument("--retry-failed", action="store_true")
    features.add_argument("--workers", type=int, default=4)
    features.add_argument("--timeout-seconds", type=int, default=600)
    features.add_argument("--model")
    features.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"))

    train = subparsers.add_parser("train")
    train.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    train.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    train.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train.add_argument("--output", type=Path, default=DEFAULT_RESULT)

    train_reply_bubbles = subparsers.add_parser("train-reply-bubble-count")
    train_reply_bubbles.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    train_reply_bubbles.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    train_reply_bubbles.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_reply_bubbles.add_argument(
        "--output", type=Path, default=DEFAULT_REPLY_BUBBLE_COUNT_RESULT
    )

    train_shared_mlp = subparsers.add_parser("train-shared-mlp")
    train_shared_mlp.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    train_shared_mlp.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    train_shared_mlp.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_shared_mlp.add_argument("--output", type=Path, default=DEFAULT_SHARED_MLP_RESULT)

    train_explicit_cross = subparsers.add_parser("train-explicit-crosses")
    train_explicit_cross.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    train_explicit_cross.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    train_explicit_cross.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_explicit_cross.add_argument(
        "--output", type=Path, default=DEFAULT_EXPLICIT_CROSS_RESULT
    )

    train_deep_shared_mlp = subparsers.add_parser("train-deep-shared-mlp")
    train_deep_shared_mlp.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    train_deep_shared_mlp.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    train_deep_shared_mlp.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_deep_shared_mlp.add_argument(
        "--output", type=Path, default=DEFAULT_DEEP_SHARED_MLP_RESULT
    )

    train_boolean_context = subparsers.add_parser("train-boolean-context-crosses")
    train_boolean_context.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    train_boolean_context.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    train_boolean_context.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    train_boolean_context.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_boolean_context.add_argument(
        "--output", type=Path, default=DEFAULT_BOOLEAN_CONTEXT_CROSS_RESULT
    )

    train_refined_boolean_context = subparsers.add_parser(
        "train-refined-boolean-context"
    )
    train_refined_boolean_context.add_argument(
        "--blind", type=Path, default=DEFAULT_BLIND
    )
    train_refined_boolean_context.add_argument(
        "--answers", type=Path, default=DEFAULT_ANSWERS
    )
    train_refined_boolean_context.add_argument(
        "--replay", type=Path, default=DEFAULT_REPLAY
    )
    train_refined_boolean_context.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES
    )
    train_refined_boolean_context.add_argument(
        "--output", type=Path, default=DEFAULT_REFINED_BOOLEAN_CONTEXT_RESULT
    )

    train_group_pattern_boolean = subparsers.add_parser(
        "train-group-pattern-boolean"
    )
    train_group_pattern_boolean.add_argument(
        "--blind", type=Path, default=DEFAULT_BLIND
    )
    train_group_pattern_boolean.add_argument(
        "--answers", type=Path, default=DEFAULT_ANSWERS
    )
    train_group_pattern_boolean.add_argument(
        "--replay", type=Path, default=DEFAULT_REPLAY
    )
    train_group_pattern_boolean.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES
    )
    train_group_pattern_boolean.add_argument(
        "--output", type=Path, default=DEFAULT_GROUP_PATTERN_BOOLEAN_RESULT
    )

    generate_group_pattern_holdout = subparsers.add_parser(
        "generate-group-pattern-holdout"
    )
    generate_group_pattern_holdout.add_argument(
        "--manifest", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_MANIFEST
    )
    generate_group_pattern_holdout.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_CHECKPOINT
    )
    generate_group_pattern_holdout.add_argument(
        "--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG
    )
    generate_group_pattern_holdout.add_argument("--max-calls", type=int)
    generate_group_pattern_holdout.add_argument("--retry-failed", action="store_true")
    generate_group_pattern_holdout.add_argument("--relationship", choices=RELATIONS)
    generate_group_pattern_holdout.add_argument("--shard-index", type=int, default=0)
    generate_group_pattern_holdout.add_argument("--shard-count", type=int, default=1)

    merge_group_pattern_holdout = subparsers.add_parser(
        "merge-group-pattern-holdout-checkpoints"
    )
    merge_group_pattern_holdout.add_argument(
        "--manifest", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_MANIFEST
    )
    merge_group_pattern_holdout.add_argument(
        "--target", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_CHECKPOINT
    )
    merge_group_pattern_holdout.add_argument(
        "--sources", type=Path, nargs="+", required=True
    )

    prepare_effective_group_pattern_holdout_parser = subparsers.add_parser(
        "prepare-effective-group-pattern-holdout"
    )
    prepare_effective_group_pattern_holdout_parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_MANIFEST
    )
    prepare_effective_group_pattern_holdout_parser.add_argument(
        "--candidate-manifests",
        type=Path,
        nargs="+",
        default=list(DEFAULT_GROUP_PATTERN_HOLDOUT_CANDIDATE_MANIFESTS),
    )
    prepare_effective_group_pattern_holdout_parser.add_argument(
        "--development-manifest", type=Path, default=DEFAULT_MANIFEST
    )
    prepare_effective_group_pattern_holdout_parser.add_argument(
        "--source-checkpoints", type=Path, nargs="+", required=True
    )
    prepare_effective_group_pattern_holdout_parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_EFFECTIVE_GROUP_PATTERN_HOLDOUT_MANIFEST,
    )
    prepare_effective_group_pattern_holdout_parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=DEFAULT_EFFECTIVE_GROUP_PATTERN_HOLDOUT_CHECKPOINT,
    )

    freeze_group_pattern_holdout = subparsers.add_parser(
        "freeze-group-pattern-holdout"
    )
    freeze_group_pattern_holdout.add_argument(
        "--manifest", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_MANIFEST
    )
    freeze_group_pattern_holdout.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_CHECKPOINT
    )
    freeze_group_pattern_holdout.add_argument(
        "--replay", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_REPLAY
    )
    freeze_group_pattern_holdout.add_argument(
        "--blind", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_BLIND
    )
    freeze_group_pattern_holdout.add_argument(
        "--answers", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_ANSWERS
    )

    validate_group_pattern_holdout = subparsers.add_parser(
        "validate-group-pattern-holdout"
    )
    validate_group_pattern_holdout.add_argument(
        "--development-blind", type=Path, default=DEFAULT_BLIND
    )
    validate_group_pattern_holdout.add_argument(
        "--development-answers", type=Path, default=DEFAULT_ANSWERS
    )
    validate_group_pattern_holdout.add_argument(
        "--development-replay", type=Path, default=DEFAULT_REPLAY
    )
    validate_group_pattern_holdout.add_argument(
        "--development-features", type=Path, default=DEFAULT_FEATURES
    )
    validate_group_pattern_holdout.add_argument(
        "--holdout-blind", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_BLIND
    )
    validate_group_pattern_holdout.add_argument(
        "--holdout-answers", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_ANSWERS
    )
    validate_group_pattern_holdout.add_argument(
        "--holdout-replay", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_REPLAY
    )
    validate_group_pattern_holdout.add_argument(
        "--holdout-features", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_FEATURES
    )
    validate_group_pattern_holdout.add_argument(
        "--output", type=Path, default=DEFAULT_GROUP_PATTERN_HOLDOUT_RESULT
    )

    stable_build = subparsers.add_parser("build-stable-judge-validation")
    stable_build.add_argument("--development-blind", type=Path, default=DEFAULT_BLIND)
    stable_build.add_argument("--development-answers", type=Path, default=DEFAULT_ANSWERS)
    stable_build.add_argument(
        "--reference-source", type=Path, default=DEFAULT_STABLE_REFERENCE_SOURCE
    )
    stable_build.add_argument("--reference-output", type=Path, default=DEFAULT_STABLE_REFERENCE)
    stable_build.add_argument("--blind-output", type=Path, default=DEFAULT_STABLE_BLIND)
    stable_build.add_argument("--answers-output", type=Path, default=DEFAULT_STABLE_ANSWERS)

    stable_run = subparsers.add_parser("run-stable-judge-validation")
    stable_run.add_argument("--blind", type=Path, default=DEFAULT_STABLE_BLIND)
    stable_run.add_argument("--answers", type=Path, default=DEFAULT_STABLE_ANSWERS)
    stable_run.add_argument("--reference", type=Path, default=DEFAULT_STABLE_REFERENCE)
    stable_run.add_argument("--predictions", type=Path, default=DEFAULT_STABLE_PREDICTIONS)
    stable_run.add_argument("--audit", type=Path, default=DEFAULT_STABLE_AUDIT)
    stable_run.add_argument("--profile", type=Path, default=DEFAULT_STABLE_PROFILE)
    stable_run.add_argument("--max-cases", type=int)
    stable_run.add_argument("--workers", type=int, default=4)

    stable_compare = subparsers.add_parser("compare-stable-judge-classifier")
    stable_compare.add_argument("--blind", type=Path, default=DEFAULT_STABLE_BLIND)
    stable_compare.add_argument("--answers", type=Path, default=DEFAULT_STABLE_ANSWERS)
    stable_compare.add_argument("--reference", type=Path, default=DEFAULT_STABLE_REFERENCE)
    stable_compare.add_argument(
        "--stable-predictions", type=Path, default=DEFAULT_STABLE_PREDICTIONS
    )
    stable_compare.add_argument("--learning-curve", type=Path, default=DEFAULT_RESULT)
    stable_compare.add_argument("--profile", type=Path, default=DEFAULT_STABLE_PROFILE)
    stable_compare.add_argument("--output", type=Path, default=DEFAULT_STABLE_COMPARISON)

    base_correction_development = subparsers.add_parser(
        "train-base-correction-development"
    )
    base_correction_development.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    base_correction_development.add_argument(
        "--answers", type=Path, default=DEFAULT_ANSWERS
    )
    base_correction_development.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES
    )
    base_correction_development.add_argument(
        "--validation-blind", type=Path, default=DEFAULT_STABLE_BLIND
    )
    base_correction_development.add_argument(
        "--base-predictions", type=Path, default=DEFAULT_STABLE_PREDICTIONS
    )
    base_correction_development.add_argument(
        "--output", type=Path, default=DEFAULT_BASE_CORRECTION_DEVELOPMENT_RESULT
    )
    base_correction_development.add_argument(
        "--include-group-participation-cross", action="store_true"
    )

    reworked_boolean_correction = subparsers.add_parser(
        "train-reworked-boolean-correction-development"
    )
    reworked_boolean_correction.add_argument(
        "--blind", type=Path, default=DEFAULT_BLIND
    )
    reworked_boolean_correction.add_argument(
        "--answers", type=Path, default=DEFAULT_ANSWERS
    )
    reworked_boolean_correction.add_argument(
        "--replay", type=Path, default=DEFAULT_REPLAY
    )
    reworked_boolean_correction.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES
    )
    reworked_boolean_correction.add_argument(
        "--validation-blind", type=Path, default=DEFAULT_STABLE_BLIND
    )
    reworked_boolean_correction.add_argument(
        "--base-predictions", type=Path, default=DEFAULT_STABLE_PREDICTIONS
    )
    reworked_boolean_correction.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
    )

    base_correction_fixed = subparsers.add_parser(
        "validate-base-correction-fixed-240"
    )
    base_correction_fixed.add_argument(
        "--development-result",
        type=Path,
        default=DEFAULT_BASE_CORRECTION_DEVELOPMENT_RESULT,
    )
    base_correction_fixed.add_argument(
        "--blind", type=Path, default=DEFAULT_FIXED_240_BLIND
    )
    base_correction_fixed.add_argument(
        "--answers", type=Path, default=DEFAULT_FIXED_240_ANSWERS
    )
    base_correction_fixed.add_argument(
        "--features", type=Path, default=DEFAULT_FIXED_240_FEATURES
    )
    base_correction_fixed.add_argument(
        "--feature-source-blind",
        type=Path,
        default=DEFAULT_FIXED_240_FEATURE_SOURCE_BLIND,
    )
    base_correction_fixed.add_argument(
        "--base-predictions", type=Path, default=DEFAULT_FIXED_240_BASE_PREDICTIONS
    )
    base_correction_fixed.add_argument(
        "--output", type=Path, default=DEFAULT_BASE_CORRECTION_FIXED_240_RESULT
    )
    reworked_boolean_fixed = subparsers.add_parser(
        "validate-reworked-boolean-correction-fixed-240"
    )
    reworked_boolean_fixed.add_argument(
        "--development-result",
        type=Path,
        default=DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
    )
    reworked_boolean_fixed.add_argument(
        "--blind", type=Path, default=DEFAULT_FIXED_240_BLIND
    )
    reworked_boolean_fixed.add_argument(
        "--answers", type=Path, default=DEFAULT_FIXED_240_ANSWERS
    )
    reworked_boolean_fixed.add_argument(
        "--replay", type=Path, default=DEFAULT_FIXED_240_REPLAY
    )
    reworked_boolean_fixed.add_argument(
        "--features", type=Path, default=DEFAULT_FIXED_240_FEATURES
    )
    reworked_boolean_fixed.add_argument(
        "--feature-source-blind",
        type=Path,
        default=DEFAULT_FIXED_240_FEATURE_SOURCE_BLIND,
    )
    reworked_boolean_fixed.add_argument(
        "--feature-source-answers",
        type=Path,
        default=DEFAULT_FIXED_240_FEATURE_SOURCE_ANSWERS,
    )
    reworked_boolean_fixed.add_argument(
        "--base-predictions", type=Path, default=DEFAULT_FIXED_240_BASE_PREDICTIONS
    )
    reworked_boolean_fixed.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REWORKED_BOOLEAN_CORRECTION_FIXED_240_RESULT,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "seal-holdout-guard":
        result = seal_holdout_guard(args.replay, args.output)
    elif args.command == "initialize":
        result = initialize_manifest(args.guard, args.output, args.exports, args.wiki)
    elif args.command == "generate-c0":
        result = generate_c0(
            args.manifest, args.guard, args.checkpoint, args.exports,
            args.volcengine_config, args.max_calls, args.retry_failed, args.relationship,
            args.shard_index, args.shard_count,
        )
    elif args.command == "upgrade-c0-input-contract":
        result = upgrade_c0_checkpoint_input_contract(
            args.manifest, args.guard, args.source, args.target, args.exports,
            args.shard_index, args.shard_count,
        )
    elif args.command == "merge-c0-checkpoints":
        result = merge_c0_checkpoints(
            args.manifest, args.guard, args.target, args.sources
        )
    elif args.command == "merge-owned-c0-shards":
        result = merge_owned_c0_shards(
            args.manifest, args.guard, args.target, args.sources
        )
    elif args.command == "freeze-pairs":
        result = freeze_pairs(
            args.manifest, args.guard, args.checkpoint, args.exports,
            args.replay, args.blind, args.answers
        )
    elif args.command == "extract-features":
        result = extract_features(
            args.blind, args.checkpoint, args.max_cases, args.retry_failed, args.workers,
            args.timeout_seconds, args.model, args.reasoning_effort,
        )
    elif args.command == "train":
        result = train_learning_curve(args.blind, args.answers, args.features, args.output)
    elif args.command == "train-reply-bubble-count":
        result = train_learning_curve(
            args.blind,
            args.answers,
            args.features,
            args.output,
            include_reply_bubble_count=True,
        )
    elif args.command == "train-shared-mlp":
        result = train_shared_mlp_learning_curve(
            args.blind, args.answers, args.features, args.output
        )
    elif args.command == "train-explicit-crosses":
        result = train_explicit_cross_learning_curve(
            args.blind, args.answers, args.features, args.output
        )
    elif args.command == "train-deep-shared-mlp":
        result = train_deep_shared_mlp_learning_curve(
            args.blind, args.answers, args.features, args.output
        )
    elif args.command == "train-boolean-context-crosses":
        result = train_boolean_context_cross_learning_curve(
            args.blind, args.answers, args.replay, args.features, args.output
        )
    elif args.command == "train-refined-boolean-context":
        result = train_refined_boolean_context_learning_curve(
            args.blind, args.answers, args.replay, args.features, args.output
        )
    elif args.command == "train-group-pattern-boolean":
        result = train_group_pattern_boolean_candidate(
            args.blind, args.answers, args.replay, args.features, args.output
        )
    elif args.command == "generate-group-pattern-holdout":
        result = generate_group_pattern_holdout(
            args.manifest,
            args.checkpoint,
            args.volcengine_config,
            args.max_calls,
            args.retry_failed,
            args.relationship,
            args.shard_index,
            args.shard_count,
        )
    elif args.command == "merge-group-pattern-holdout-checkpoints":
        result = merge_group_pattern_holdout_checkpoints(
            args.manifest, args.target, args.sources
        )
    elif args.command == "prepare-effective-group-pattern-holdout":
        result = prepare_effective_group_pattern_holdout(
            args.source_manifest,
            args.candidate_manifests,
            args.development_manifest,
            args.source_checkpoints,
            args.output_manifest,
            args.output_checkpoint,
        )
    elif args.command == "freeze-group-pattern-holdout":
        result = freeze_group_pattern_holdout(
            args.manifest,
            args.checkpoint,
            args.replay,
            args.blind,
            args.answers,
        )
    elif args.command == "validate-group-pattern-holdout":
        result = validate_group_pattern_holdout(
            args.development_blind,
            args.development_answers,
            args.development_replay,
            args.development_features,
            args.holdout_blind,
            args.holdout_answers,
            args.holdout_replay,
            args.holdout_features,
            args.output,
        )
    elif args.command == "build-stable-judge-validation":
        result = build_stable_judge_validation(
            args.development_blind,
            args.development_answers,
            args.reference_source,
            args.reference_output,
            args.blind_output,
            args.answers_output,
        )
    elif args.command == "run-stable-judge-validation":
        result = run_stable_judge_validation(
            args.blind,
            args.answers,
            args.reference,
            args.predictions,
            args.audit,
            args.profile,
            max_cases=args.max_cases,
            workers=args.workers,
        )
    elif args.command == "compare-stable-judge-classifier":
        result = compare_stable_judge_classifier(
            args.blind,
            args.answers,
            args.reference,
            args.stable_predictions,
            args.learning_curve,
            args.profile,
            args.output,
        )
    elif args.command == "train-base-correction-development":
        result = train_base_correction_development(
            args.blind,
            args.answers,
            args.features,
            args.validation_blind,
            args.base_predictions,
            args.output,
            include_group_participation_cross=args.include_group_participation_cross,
        )
    elif args.command == "train-reworked-boolean-correction-development":
        result = train_reworked_boolean_correction_development(
            args.blind,
            args.answers,
            args.replay,
            args.features,
            args.validation_blind,
            args.base_predictions,
            args.output,
        )
    elif args.command == "validate-base-correction-fixed-240":
        result = validate_base_correction_fixed_240(
            args.development_result,
            args.blind,
            args.answers,
            args.features,
            args.base_predictions,
            args.output,
            args.feature_source_blind,
        )
    elif args.command == "validate-reworked-boolean-correction-fixed-240":
        result = validate_reworked_boolean_correction_fixed_240(
            args.development_result,
            args.blind,
            args.answers,
            args.replay,
            args.features,
            args.base_predictions,
            args.output,
            args.feature_source_blind,
            args.feature_source_answers,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
