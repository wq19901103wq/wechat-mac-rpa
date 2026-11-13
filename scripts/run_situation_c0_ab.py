#!/usr/bin/env python3
"""Run a resumable C0 A/B on fresh verified-human development cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_persona_few_shots import Candidate  # noqa: E402
from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _export_payloads,
    _load_env,
    _parse_api_judge_response,
    _run_c0,
    _write_private,
    _event_has_tool_call,
    assert_reference_isolated,
    load_volcengine_credentials,
    rebuild_generation_inputs,
    reference_leakage_details,
)
from scripts.run_clean_human_c0_baseline import (  # noqa: E402
    PROFILE,
    REFERENCE,
    _blind_case,
)
from scripts.run_human_ai_judge_benchmark_v2 import build_messages, load_profile  # noqa: E402
from scripts.run_kimi_human_ai_judge_v2 import _run_codex  # noqa: E402
from src.reply.c0_config import current_c0_baseline_id, load_current_c0  # noqa: E402
from src.reply.c0_factory import create_current_c0_clients  # noqa: E402
from src.reply.few_shot import PersonaFewShotRetriever, _situation_profile, _situation_tags  # noqa: E402
from src.reply.generator import ReplyGenerator  # noqa: E402


DEFAULT_DEVELOPMENT = ROOT / "data/reports/private/c0_development_pool_v1/manifest.json"
DEFAULT_EXPOSED = ROOT / "data/reports/private/situation_few_shot_retrieval_v7/results.json"
DEFAULT_PREVIOUS_AB = ROOT / "data/reports/private/situation_c0_ab_v1/manifest.json"
DEFAULT_FIXED_TEST = ROOT / "data/reports/private/fixed_c0_holdout_final/manifest.json"
DEFAULT_FEW_SHOT = ROOT / "data/reports/private/situation_few_shot_library_v1/persona_examples.jsonl"
DEFAULT_DEVELOPMENT80_RUN = ROOT / "data/reports/private/situation_c0_latest_turn_form_fixed80_ab_v2/manifest.json"
DEFAULT_DEVELOPMENT80_FORCE_REPLY_PERSONA = ROOT / "data/reports/private/situation_c0_latest_turn_form_fixed80_ab_v2/persona_force_reply.shard-0-of-4.md"
DEFAULT_FROZEN_FORMAL_BASELINE_RUN = (
    ROOT / "data/reports/private/"
    "situation_c0_group_explicit_self_stance_action_formal240_160g80p_v2"
)
DEFAULT_FROZEN_FORMAL_SOURCE_MANIFESTS = (
    ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json",
    ROOT / "data/reports/private/adversarial_hidden_holdout_v1/manifest.json",
)
DEFAULT_NAMED_CONTEXT_FEW_SHOT = ROOT / "data/reports/private/situation_few_shot_library_named_context_v1/persona_examples.jsonl"
DEFAULT_LONG_NAMED_CONTEXT_FEW_SHOT = ROOT / "data/reports/private/situation_few_shot_library_named_context_10_v1/persona_examples.jsonl"
DEFAULT_CANDIDATE_FEW_SHOT = ROOT / "data/reports/private/situation_few_shot_library_expanded_holdout_excluded_v1/persona_examples.jsonl"
DEFAULT_EXPORTS = ROOT / "data/exports/b"
DEFAULT_OUTPUT = ROOT / "data/reports/private/situation_c0_clarification_ab_v1"
DEVELOPMENT80_FORCE_REPLY_SHA256 = "5b39ecbac546c45a7088a63993beb54f68e232affe5e372bbd43bf9bbc1dfbb4"
DEVELOPMENT80_NAMED_CONTEXT_SHA256 = "e46befb78cb73f6eeaa4fd429ad993b26ebd848f6154ff1d4ee1cfcdead08f8e"
FORMAL_GROUP_TURBO_MODEL = "doubao-seed-2.1-turbo"
SOURCE_CASE_IDS = {
    "human_8d7d164c54dd",
    "human_cd29db47a3c1",
    "human_98dc7316e369",
    "human_3b95fbaa1097",
    "human_23f9cd47fb7c",
    "human_5f42c5731af4",
    "human_fd952ce20970",
    "human_a975966dbf1f",
}
CURRENT_BASELINE_MODES = frozenset({
    "latest-turn-situation",
    "latest-turn-same-chat-sparse",
    "few-shot-count-5",
    "response-move-before-relevance",
    "semantic-topic-weight-8",
    "unresolved-fact-choice",
    "no-restatement-action",
    "expanded-pool",
    "latest-turn-form",
    "information-gap-response-move",
    "current-stance-continuity",
    "plain-response-before-styling",
    "group-participant-structure",
    "group-context-window-10",
    "group-speaker-target-attribution",
    "group-explicit-third-party-guard",
    "group-relevant-self-evidence",
    "group-self-context-freshness-boundary",
    "group-adjacent-self-response",
    "group-unified-history",
    "group-unified-relevant-self",
    "group-target-action-contract",
    "group-public-question-specific-followup",
    "group-instant-self-correction",
    "group-visible-self-stance-continuity",
    "group-pending-self-action",
    "group-pattern-participation",
    "group-current-event-self-role-continuity",
    "group-explicit-self-stance-action",
    "group-answer-authority-rewrite",
    "group-feature-plan-before-generation",
    "skill-snapshot",
    "generation-model",
})
PAIRED_COUNT_MODES = frozenset({
    "few-shot-count-5",
    "current-stance-continuity",
    "group-speaker-target-attribution",
    "group-explicit-third-party-guard",
    "group-relevant-self-evidence",
    "group-self-context-freshness-boundary",
    "group-adjacent-self-response",
    "group-unified-history",
    "group-unified-relevant-self",
    "group-target-action-contract",
    "group-public-question-specific-followup",
    "group-instant-self-correction",
    "group-visible-self-stance-continuity",
    "group-pending-self-action",
    "group-pattern-participation",
    "group-current-event-self-role-continuity",
    "group-explicit-self-stance-action",
    "group-answer-authority-rewrite",
    "group-feature-plan-before-generation",
    "skill-snapshot",
    "generation-model",
})


class FewShotInjectionMismatch(RuntimeError):
    """The real generation request did not receive the frozen few-shot count."""


class FeatureExtractionOnlyClient:
    """Route the group feature contract to a candidate model, keep planning unchanged."""

    _FEATURE_PROMPT_PREFIX = "分析群聊回复的事实边界。"

    def __init__(self, feature_client: Any, plan_client: Any):
        self.feature_client = feature_client
        self.plan_client = plan_client

    def chat(self, messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        prompt = str((messages or [{}])[-1].get("content") or "")
        client = (
            self.feature_client
            if prompt.startswith(self._FEATURE_PROMPT_PREFIX)
            else self.plan_client
        )
        return client.chat(messages=messages, **kwargs)


class CodexExecClient:
    """QwenClient-compatible adapter for one isolated Codex CLI generation."""

    def __init__(self, model: str, reasoning_effort: str = "low"):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.last_elapsed_ms: int | None = None
        self.last_audit: dict[str, Any] = {}

    def chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        message: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
        raise_on_error: bool = False,
        max_retries: int | None = None,
    ) -> str:
        del user_id, temperature, max_tokens, response_format, max_retries
        if tools:
            raise ValueError("Codex C0 generation does not allow tools")
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message or ""})
        prompt = (
            "Do not use tools. Produce only the requested assistant response. "
            "The following JSON array is the complete ordered message sequence; "
            "obey each message according to its role and preserve its order.\n"
            + json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        )
        try:
            with tempfile.TemporaryDirectory(prefix="codex-c0-generation-") as raw_dir:
                temp = Path(raw_dir)
                last_message = temp / "last_message.txt"
                command = [
                    "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                    "--sandbox", "read-only", "--cd", str(temp), "--skip-git-repo-check",
                    "--json", "--output-last-message", str(last_message),
                    "-m", self.model,
                    "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                    "-",
                ]
                started = time.monotonic()
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout or 180,
                )
                self.last_elapsed_ms = round((time.monotonic() - started) * 1000)
                events = []
                parse_errors = []
                for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        parse_errors.append({"line": line_number, "error": str(exc), "text": line})
                self.last_audit = {
                    "argv": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "events": events,
                    "event_parse_errors": parse_errors,
                    "message_roles": [str(row.get("role") or "") for row in messages],
                }
                if completed.returncode:
                    raise RuntimeError(f"Codex CLI failed: {completed.stderr[-2000:]}")
                if parse_errors:
                    raise ValueError("Codex CLI returned malformed event JSON")
                if any(_event_has_tool_call(event) for event in events):
                    raise ValueError("Codex C0 isolation failed: tool call found in event log")
                if not last_message.exists():
                    raise ValueError("Codex CLI did not write a final response")
                return last_message.read_text(encoding="utf-8")
        except Exception:
            if raise_on_error:
                raise
            return ""


def _require_actual_few_shots(
    result: dict[str, Any], expected_count: int, case_id: str, condition: str
) -> None:
    trace = result.get("trace") or {}
    ids = list(trace.get("few_shot_ids") or [])
    content = str(trace.get("few_shot_content") or "")
    if len(ids) != expected_count or not content:
        raise FewShotInjectionMismatch(
            f"{case_id}:{condition} expected {expected_count} injected few-shot examples, "
            f"got {len(ids)} with content={'present' if content else 'empty'}"
        )


class LeaveOneOutRetriever:
    def __init__(self, retriever: PersonaFewShotRetriever, max_chars: int = 2500):
        self.retriever = retriever
        self.max_chars = max_chars
        self.exclude_ids: set[str] = set()
        self.current_chat_id: str | None = None
        self.last_trace: dict[str, Any] = {}

    def is_approved(self) -> bool:
        return self.retriever.is_approved()

    def retrieve(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        requested = set(kwargs.pop("exclude_ids", set()) or set())
        kwargs["exclude_ids"] = requested | self.exclude_ids
        if self.current_chat_id and not kwargs.get("chat_id"):
            kwargs["chat_id"] = self.current_chat_id
        query = str(kwargs.get("query") or (args[0] if args else ""))
        limit = int(kwargs.get("limit", 0))
        if limit > 0:
            expanded = {**kwargs, "limit": max(limit * 10, 30)}
            ranked = self.retriever.retrieve(*args, **expanded)
            rows = []
            for row in ranked:
                candidate_rows = [*rows, row]
                _, rendered_ids = self.retriever.render_selected(
                    candidate_rows,
                    max_chars=self.max_chars,
                )
                if rendered_ids == [str(value["id"]) for value in candidate_rows]:
                    rows.append(row)
                if len(rows) == limit:
                    break
        else:
            rows = self.retriever.retrieve(*args, **kwargs)
        self.last_trace = {
            "query": query,
            "query_profile": _situation_profile(
                query,
                str(kwargs.get("interaction_context") or ""),
                self.retriever.enable_clarification,
                self.retriever.enable_information_gap_response_move,
            ),
            "information_gap_feature_enabled": self.retriever.enable_information_gap_response_move,
            "selected_ids": [str(row["id"]) for row in rows],
        }
        return rows

    def render_selected(self, *args: Any, **kwargs: Any) -> tuple[str, list[str]]:
        return self.retriever.render_selected(*args, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> tuple[str, list[str]]:
        return self.retriever.render(*args, **kwargs)

    def trace(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.last_trace))


class PairedCountSelection:
    """Rank once per case, then expose fixed prefixes for count A/B."""

    def __init__(self, retriever: PersonaFewShotRetriever, max_count: int):
        self.retriever = retriever
        self.max_count = max_count
        self.rows: list[dict[str, Any]] | None = None
        self.trace: dict[str, Any] = {}

    def reset(self) -> None:
        self.rows = None
        self.trace = {}

    def restore(self, ids: list[str], rows_by_id: dict[str, dict[str, Any]]) -> None:
        if len(ids) != self.max_count:
            raise ValueError(
                f"saved paired few-shot count is {len(ids)}, expected {self.max_count}"
            )
        missing = [example_id for example_id in ids if example_id not in rows_by_id]
        if missing:
            raise ValueError(f"saved paired few-shot ids are missing from the pool: {missing}")
        self.rows = [rows_by_id[example_id] for example_id in ids]
        self.trace = {"ranked_ids": list(ids), "restored_from_checkpoint": True}

    def select(self, query: str, kwargs: dict[str, Any]) -> None:
        if self.rows is not None:
            return
        self.rows = self.retriever.retrieve(query, **{**kwargs, "limit": self.max_count})
        self.trace = {"ranked_ids": [str(row["id"]) for row in self.rows]}


class PairedCountRetriever:
    def __init__(self, state: PairedCountSelection):
        self.state = state
        self.exclude_ids: set[str] = set()
        self.current_chat_id: str | None = None

    def is_approved(self) -> bool:
        return self.state.retriever.is_approved()

    def retrieve(self, query: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        requested = set(kwargs.pop("exclude_ids", set()) or set())
        kwargs["exclude_ids"] = requested | self.exclude_ids
        if self.current_chat_id and not kwargs.get("chat_id"):
            kwargs["chat_id"] = self.current_chat_id
        limit = int(kwargs.pop("limit", 0))
        self.state.select(query, kwargs)
        return list((self.state.rows or [])[:limit])

    def render_selected(self, *args: Any, **kwargs: Any) -> tuple[str, list[str]]:
        return self.state.retriever.render_selected(*args, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> tuple[str, list[str]]:
        return self.state.retriever.render(*args, **kwargs)

    def trace(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.state.trace))


def _load_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def _exact_pair_fingerprint(context: Any, reply: Any) -> str:
    context_list = context if isinstance(context, list) else [context]
    reply_list = reply if isinstance(reply, list) else [reply]
    payload = json.dumps([context_list, reply_list], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_message(value: Any) -> str:
    return re.sub(r"\s+", "", html.unescape(str(value or ""))).strip()


def _message_sequences(row: dict[str, Any], max_length: int) -> set[tuple[str, ...]]:
    context = row.get("context") or []
    reply = row.get("reply") or []
    context_values = context if isinstance(context, list) else [context]
    reply_values = reply if isinstance(reply, list) else [reply]
    messages = [
        normalized
        for value in [*context_values, *reply_values]
        if (normalized := _normalized_message(value))
    ]
    return {
        tuple(messages[start:start + length])
        for length in range(1, min(max_length, len(messages)) + 1)
        for start in range(len(messages) - length + 1)
    }


def _timeline_overlaps(
    row: dict[str, Any],
    target_messages: list[str],
) -> bool:
    context = row.get("context") or []
    reply = row.get("reply") or []
    context_values = context if isinstance(context, list) else [context]
    reply_values = reply if isinstance(reply, list) else [reply]
    historical_pairs, historical_long_messages = _timeline_overlap_keys(
        [*context_values, *reply_values]
    )
    target_pairs, target_long_messages = _timeline_overlap_keys(target_messages)
    return bool(
        historical_pairs & target_pairs
        or historical_long_messages & target_long_messages
    )


def _timeline_overlap_keys(
    messages: list[Any],
) -> tuple[set[tuple[str, str]], set[str]]:
    normalized_messages = tuple(
        normalized
        for value in messages
        if (normalized := _normalized_message(value))
    )
    return (
        {
            (normalized_messages[index], normalized_messages[index + 1])
            for index in range(len(normalized_messages) - 1)
        },
        {message for message in normalized_messages if len(message) >= 12},
    )


def _stable_source_id(row: dict[str, Any]) -> str | None:
    for key in ("original_message_id", "source_message_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return None


def _per_case_overlap_ids(
    cases: list[dict[str, Any]],
    few_shot_rows: dict[str, dict[str, Any]],
    full_context_by_case: dict[str, list[str]] | None = None,
) -> dict[str, set[str]]:
    source_ids: dict[str, set[str]] = {}
    pair_ids: dict[str, set[str]] = {}
    max_reply_length = max((
        len(reply) if isinstance((reply := case.get("reply") or []), list) else 1
        for case in cases
    ), default=0)
    reply_sequence_ids: dict[tuple[str, ...], set[str]] = {}
    timeline_pair_ids_by_chat: dict[
        str, dict[tuple[str, str], set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    timeline_long_ids_by_chat: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for example_id, row in few_shot_rows.items():
        if source_id := _stable_source_id(row):
            source_ids.setdefault(source_id, set()).add(example_id)
        fingerprint = _exact_pair_fingerprint(row.get("context", []), row.get("reply", []))
        pair_ids.setdefault(fingerprint, set()).add(example_id)
        for sequence in _message_sequences(row, max_reply_length):
            reply_sequence_ids.setdefault(sequence, set()).add(example_id)
        if chat_id := str(row.get("chat_id") or ""):
            context = row.get("context") or []
            reply = row.get("reply") or []
            context_values = context if isinstance(context, list) else [context]
            reply_values = reply if isinstance(reply, list) else [reply]
            timeline_pairs, timeline_long_messages = _timeline_overlap_keys(
                [*context_values, *reply_values]
            )
            for pair in timeline_pairs:
                timeline_pair_ids_by_chat[chat_id][pair].add(example_id)
            for message in timeline_long_messages:
                timeline_long_ids_by_chat[chat_id][message].add(example_id)

    overlaps: dict[str, set[str]] = {}
    for case in cases:
        case_id = str(case["id"])
        matched = {case_id}
        if source_id := _stable_source_id(case):
            matched.update(source_ids.get(source_id, set()))
        fingerprint = _exact_pair_fingerprint(case.get("context", []), case.get("reply", []))
        matched.update(pair_ids.get(fingerprint, set()))
        reply = case.get("reply") or []
        reply_values = reply if isinstance(reply, list) else [reply]
        reply_sequence = tuple(
            normalized
            for value in reply_values
            if (normalized := _normalized_message(value))
        )
        if reply_sequence:
            matched.update(reply_sequence_ids.get(reply_sequence, set()))
        chat_id = str(case.get("chat_id") or "")
        full_context = (full_context_by_case or {}).get(case_id) or []
        if chat_id and full_context:
            timeline_pairs, timeline_long_messages = _timeline_overlap_keys(full_context)
            for pair in timeline_pairs:
                matched.update(timeline_pair_ids_by_chat[chat_id].get(pair, set()))
            for message in timeline_long_messages:
                matched.update(timeline_long_ids_by_chat[chat_id].get(message, set()))
        overlaps[case_id] = matched
    return overlaps


def _per_case_overlap_ids_from_pools(
    cases: list[dict[str, Any]],
    pools: list[dict[str, dict[str, Any]]],
    full_context_by_case: dict[str, list[str]] | None = None,
) -> dict[str, set[str]]:
    overlap_maps = [
        _per_case_overlap_ids(cases, rows, full_context_by_case)
        for rows in pools
    ]
    return {
        str(case["id"]): set().union(*(
            overlap_map[str(case["id"])] for overlap_map in overlap_maps
        ))
        for case in cases
    }


def _drop_contaminated_generation_entries(
    entries: dict[str, dict[str, Any]],
    forbidden_by_case: dict[str, set[str]],
) -> int:
    removed = 0
    for key in list(entries):
        case_id = key.split(":", 1)[0]
        value = entries[key]
        # A frozen formal baseline was generated under the historical isolation
        # policy recorded with that run.  Do not silently delete it merely
        # because newer checks became stricter; newly generated entries still
        # must satisfy the current policy below.
        if value.get("protected_frozen_baseline") or value.get("reused_frozen_baseline"):
            continue
        few_shot_ids = set(((value.get("result") or {}).get("trace") or {}).get("few_shot_ids") or [])
        if value.get("status") == "scorable" and (
            bool(few_shot_ids & forbidden_by_case[case_id])
            or not _has_nonempty_reply(value.get("result") or {})
        ):
            del entries[key]
            removed += 1
    return removed


def _assert_generation_entries_isolated(
    entries: dict[str, dict[str, Any]],
    forbidden_by_case: dict[str, set[str]],
) -> None:
    leaks = []
    for key, value in entries.items():
        if value.get("status") != "scorable":
            continue
        if value.get("protected_frozen_baseline") or value.get("reused_frozen_baseline"):
            continue
        case_id = key.split(":", 1)[0]
        few_shot_ids = set(
            ((value.get("result") or {}).get("trace") or {}).get("few_shot_ids") or []
        )
        overlap = sorted(few_shot_ids & forbidden_by_case[case_id])
        if overlap:
            leaks.append({"entry": key, "few_shot_ids": overlap})
    if leaks:
        raise ValueError(
            "generation isolation failed: test-related few-shot reached the saved output: "
            + json.dumps(leaks, ensure_ascii=False, sort_keys=True)
        )


class PairedNamedContextSelection:
    """Rank once, then expose identical IDs with old or named-context rendering."""

    def __init__(
        self,
        retriever: PersonaFewShotRetriever,
        named_by_id: dict[str, dict[str, Any]],
        max_chars: int,
        baseline_by_id: dict[str, dict[str, Any]] | None = None,
    ):
        self.retriever = retriever
        self.named_by_id = named_by_id
        self.baseline_by_id = baseline_by_id
        self.max_chars = max_chars
        self.case: dict[str, Any] = {}
        self.old_rows: list[dict[str, Any]] | None = None
        self.named_rows: list[dict[str, Any]] | None = None
        self.trace: dict[str, Any] = {}

    def reset(self, case: dict[str, Any]) -> None:
        self.case = case
        self.old_rows = None
        self.named_rows = None
        self.trace = {"render_ms": {}}

    def _same_event(self, row: dict[str, Any]) -> bool:
        if str(row.get("chat_id") or "") != str(self.case.get("chat_id") or ""):
            return False
        try:
            return abs(float(row.get("timestamp")) - float(self.case.get("timestamp"))) <= 3600
        except (TypeError, ValueError):
            return False

    def select(self, query: str, limit: int, kwargs: dict[str, Any]) -> None:
        if self.old_rows is not None:
            return
        started = time.perf_counter()
        ranked = self.retriever.retrieve(
            query,
            **{**kwargs, "limit": max(limit * 10, 30)},
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        same_event_ids = [str(row["id"]) for row in ranked if self._same_event(row)]
        ranked_rows = [row for row in ranked if not self._same_event(row)][:limit]
        old_rows = [
            (self.baseline_by_id or {}).get(str(row["id"]), row)
            for row in ranked_rows
        ]
        missing_ids = [str(row["id"]) for row in ranked_rows if str(row["id"]) not in self.named_by_id]
        missing_baseline_ids = [
            str(row["id"])
            for row in ranked_rows
            if self.baseline_by_id is not None and str(row["id"]) not in self.baseline_by_id
        ]
        named_rows = [self.named_by_id.get(str(row["id"]), row) for row in ranked_rows]

        # Long named contexts may hit the prompt budget sooner. Use the largest
        # shared prefix that renders completely in both conditions.
        shared_count = len(old_rows)
        while shared_count:
            old_ids = [str(row["id"]) for row in old_rows[:shared_count]]
            _, rendered_old = self.retriever.render(old_rows[:shared_count], max_chars=self.max_chars)
            _, rendered_named = self.retriever.render(named_rows[:shared_count], max_chars=self.max_chars)
            if rendered_old == old_ids and rendered_named == old_ids:
                break
            shared_count -= 1
        self.old_rows = old_rows[:shared_count]
        self.named_rows = named_rows[:shared_count]
        self.trace.update({
            "selected_ids": [str(row["id"]) for row in self.old_rows],
            "same_event_excluded_ids": same_event_ids,
            "missing_named_ids": missing_ids,
            "missing_baseline_ids": missing_baseline_ids,
            "retrieval_ms": round(retrieval_ms, 3),
        })


class PairedNamedContextRetriever:
    def __init__(self, state: PairedNamedContextSelection, variant: str):
        self.state = state
        self.variant = variant
        self.exclude_ids: set[str] = set()
        self.current_chat_id: str | None = None

    def is_approved(self) -> bool:
        return self.state.retriever.is_approved()

    def retrieve(self, query: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        requested = set(kwargs.pop("exclude_ids", set()) or set())
        kwargs["exclude_ids"] = requested | self.exclude_ids
        if self.current_chat_id and not kwargs.get("chat_id"):
            kwargs["chat_id"] = self.current_chat_id
        limit = int(kwargs.pop("limit", 0))
        self.state.select(query, limit, kwargs)
        rows = self.state.named_rows if self.variant == "candidate" else self.state.old_rows
        return list(rows or [])

    def render(self, rows: list[dict[str, Any]], max_chars: int = 2500) -> tuple[str, list[str]]:
        started = time.perf_counter()
        rendered = self.state.retriever.render(rows, max_chars=max_chars)
        self.state.trace["render_ms"][self.variant] = round((time.perf_counter() - started) * 1000, 3)
        self.state.trace[f"rendered_{self.variant}_ids"] = list(rendered[1])
        return rendered

    def render_selected(self, rows: list[dict[str, Any]], max_chars: int = 2500) -> tuple[str, list[str]]:
        return self.render(rows, max_chars=max_chars)

    def trace(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.state.trace))


def _stable_key(row: dict[str, Any], candidate_mode: str = "clarification") -> str:
    selection_mode = (
        "active-thread"
        if candidate_mode in {"response-action", "latest-request-first"}
        else candidate_mode
    )
    return hashlib.sha256((f"situation-c0-{selection_mode}-ab-v1:" + str(row["id"])).encode()).hexdigest()


def _latest_request_kind(context: list[Any]) -> str | None:
    """Classify multi-turn cases that can test whether the latest request wins."""
    messages = [str(value).strip() for value in context if str(value).strip()]
    if len(messages) < 3:
        return None
    latest = messages[-1]
    compact = re.sub(r"[\s，。！？!?、~～…]+", "", latest)
    if compact in {"在吗", "人呢", "还在吗", "干嘛呢", "干啥呢"}:
        return None
    if re.search(r"多少|几个|几号|几点|哪天|什么时候|多久|日期|时间|比例|计划|安排", latest):
        return "number-time-plan"
    if re.search(r"帮(?:我|忙)?|麻烦|发(?:我|一下)|给我|看(?:下|一下)|查(?:下|一下)|告诉我|记得", latest):
        return "help-request"
    if re.search(r"要不要|能不能|可不可以|可以(?:吗|不)|一起|选哪个|选哪|去不去|来不来", latest):
        return "choice-invitation"
    if re.search(r"到(?:了|哪)|在路上|出发|回来|回去|来(?:了|没)|忙完|弄完|好了吗", latest):
        return "status"
    if re.search(r"[?？]|为什么|为啥|怎么|咋|什么|哪(?:个|里|儿)|谁|是否|有没有|是不是|行吗|好吗|对吗|吗$|么$", latest):
        return "direct-question"
    return None


def _excluded_case_ids(exposed: dict[str, Any], previous_abs: list[dict[str, Any]], fixed_test: dict[str, Any]) -> set[str]:
    return (
        SOURCE_CASE_IDS
        | {str(row["case_id"]) for row in exposed.get("cases", [])}
        | {str(case_id) for case_id in exposed.get("case_ids", [])}
        | {
            str(case_id)
            for previous_ab in previous_abs
            for case_id in previous_ab.get("case_ids", [])
        }
        | {str(row["internal_id"]) for row in fixed_test.get("cases", [])}
    )


def _select_cases(rows: list[dict[str, Any]], excluded: set[str], limit: int, candidate_mode: str = "clarification") -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["id"]) in excluded:
            continue
        request_kind = _latest_request_kind(row.get("context") or [])
        if candidate_mode == "latest-request-first" and request_kind is None:
            continue
        tags = sorted(_situation_tags("\n".join(str(value) for value in row.get("context") or [])))
        situation = request_kind or (tags[0] if tags else "untagged")
        relation = "group" if row.get("relationship") == "group" else "private"
        buckets[(relation, situation)].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: _stable_key(row, candidate_mode))

    relation_limit = {"group": limit // 2, "private": limit - limit // 2}
    selected: list[dict[str, Any]] = []
    for relation in ("group", "private"):
        keys = sorted(key for key in buckets if key[0] == relation)
        while sum(("group" if row.get("relationship") == "group" else "private") == relation for row in selected) < relation_limit[relation]:
            moved = False
            for key in keys:
                if buckets[key] and sum(("group" if row.get("relationship") == "group" else "private") == relation for row in selected) < relation_limit[relation]:
                    selected.append(buckets[key].pop(0))
                    moved = True
            if not moved:
                break
    return selected


def _candidate(row: dict[str, Any]) -> Candidate:
    return Candidate(
        relation=str(row.get("relationship") or "acquaintance"),
        chat_id=str(row["chat_id"]),
        context=list(row.get("context") or []),
        replies=list(row.get("reply") or []),
        score=0.0,
        timestamp=int(row["timestamp"]),
        source_provenance=str(row.get("source_provenance") or "before_automation_cutoff"),
    )


def _force_reply_persona_text() -> str:
    source = (ROOT / "data/persona.md").read_text(encoding="utf-8")
    start = source.index("  <reply_decision>")
    end = source.index("  </reply_decision>", start) + len("  </reply_decision>")
    source = (
        source[:start]
        + "  <reply_decision>\n"
          "    <rule>本轮为离线回复风格实验；只要输入可读，必须生成非空 replies，不评估是否应该回复。</rule>\n"
          "  </reply_decision>"
        + source[end:]
    )
    source = source.replace(
        "      <rule>replies 只完成 response_target；若无法避开 forbidden_materials，宁可输出空 replies。</rule>",
        "      <rule>replies 只完成 response_target；若材料受限，使用不引入受限材料的简短非空回复。</rule>",
    ).replace(
        "      <rule>不回复时输出空 replies。</rule>",
        "      <rule>本轮实验的 replies 必须非空。</rule>",
    )
    return source


def _force_reply_persona(output: Path, shard_suffix: str = "") -> Path:
    path = output / f"persona_force_reply{shard_suffix}.md"
    _write_private(path, _force_reply_persona_text())
    return path


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "relationship": dict(Counter("group" if row.get("relationship") == "group" else "private" for row in rows)),
        "situation": dict(Counter(
            (sorted(_situation_tags("\n".join(str(value) for value in row.get("context") or []))) or ["untagged"])[0]
            for row in rows
        )),
    }


def _seed_scorable_entries(
    entries: dict[str, Any],
    run_case_ids: list[str],
    checkpoints: list[Path],
    repetitions: int = 1,
    conditions: tuple[str, ...] = ("baseline", "candidate"),
    forbidden_by_case: dict[str, set[str]] | None = None,
) -> int:
    allowed = {
        _entry_key(case_id, condition, repetition, repetitions)
        for case_id in run_case_ids
        for condition in conditions
        for repetition in range(1, repetitions + 1)
    }
    imported = 0
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            continue
        source = json.loads(checkpoint.read_text(encoding="utf-8"))
        for key, value in (source.get("entries") or {}).items():
            case_id = key.rsplit(":", 1)[0]
            few_shot_ids = set(((value.get("result") or {}).get("trace") or {}).get("few_shot_ids") or [])
            if (
                key in allowed
                and value.get("status") == "scorable"
                and not few_shot_ids.intersection((forbidden_by_case or {}).get(case_id, {case_id}))
                and entries.get(key, {}).get("status") != "scorable"
            ):
                entries[key] = value
                imported += 1
    return imported


def _seed_baseline_from_candidate_entries(
    entries: dict[str, Any],
    run_case_ids: list[str],
    checkpoints: list[Path],
    repetitions: int = 1,
    forbidden_by_case: dict[str, set[str]] | None = None,
) -> int:
    imported = 0
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            continue
        source = json.loads(checkpoint.read_text(encoding="utf-8"))
        for case_id in run_case_ids:
            for repetition in range(1, repetitions + 1):
                source_key = _entry_key(case_id, "candidate", repetition, repetitions)
                target_key = _entry_key(case_id, "baseline", repetition, repetitions)
                value = (source.get("entries") or {}).get(source_key) or {}
                few_shot_ids = set(((value.get("result") or {}).get("trace") or {}).get("few_shot_ids") or [])
                if (
                    value.get("status") == "scorable"
                    and not few_shot_ids.intersection((forbidden_by_case or {}).get(case_id, {case_id}))
                    and entries.get(target_key, {}).get("status") != "scorable"
                ):
                    entries[target_key] = value
                    imported += 1
    return imported


def _case_ids_from_results(path: Path) -> list[str]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("case_ids"):
        return [str(case_id) for case_id in source["case_ids"]]
    return [str(row["case_id"]) for row in source.get("cases") or []]


def _generation_model_for_condition(
    args: argparse.Namespace,
    condition: str,
    relationship: str,
    config: dict[str, Any] | None = None,
) -> str:
    relationship = "group" if relationship == "group" else "private"
    config = config or load_current_c0()
    if (
        args.candidate_mode == "generation-model"
        and condition == "candidate"
        and (not args.group_only_candidate or relationship == "group")
    ):
        return args.generation_model
    return str(config[relationship]["model"])


def _generation_timeout_for_condition(
    args: argparse.Namespace,
    condition: str,
    relationship: str,
    config: dict[str, Any] | None = None,
) -> float:
    relationship = "group" if relationship == "group" else "private"
    config = config or load_current_c0()
    if (
        args.candidate_mode == "generation-model"
        and condition == "candidate"
        and (not args.group_only_candidate or relationship == "group")
        and args.generation_timeout_seconds is not None
    ):
        return float(args.generation_timeout_seconds)
    return float(config[relationship]["timeout_seconds"])


def _feature_plan_model_for_condition(
    args: argparse.Namespace,
    condition: str,
    relationship: str,
) -> str | None:
    """Return the auxiliary model only for the candidate group feature-plan path."""
    if (
        args.candidate_mode == "group-feature-plan-before-generation"
        and condition == "candidate"
        and relationship == "group"
    ):
        return str(args.feature_extraction_model)
    return None


def _reuse_private_baseline_for_group_only_change(
    entries: dict[str, Any],
    case_id: str,
    row: dict[str, Any],
    condition: str,
    repetition: int,
    repetitions: int,
    candidate_mode: str,
    *,
    group_only_candidate: bool = False,
    judge: bool = False,
) -> bool:
    if (
        (
            candidate_mode not in {
                "group-participant-structure",
                "group-context-window-10",
                "group-speaker-target-attribution",
                "group-explicit-third-party-guard",
                "group-relevant-self-evidence",
                "group-self-context-freshness-boundary",
                "group-adjacent-self-response",
                "group-unified-history",
                "group-unified-relevant-self",
                "group-target-action-contract",
                "group-public-question-specific-followup",
                "group-instant-self-correction",
                "group-visible-self-stance-continuity",
                "group-pending-self-action",
                "group-pattern-participation",
                "group-current-event-self-role-continuity",
                "group-explicit-self-stance-action",
                "group-answer-authority-rewrite",
                "group-feature-plan-before-generation",
                "skill-snapshot",
            }
            and not (candidate_mode == "generation-model" and group_only_candidate)
        )
        or row.get("relationship") == "group"
        or condition != "candidate"
    ):
        return False
    baseline = entries.get(_entry_key(case_id, "baseline", repetition, repetitions)) or {}
    if (judge and baseline.get("error")) or (not judge and baseline.get("status") != "scorable"):
        return False
    if baseline.get("protected_frozen_baseline"):
        reused = copy.deepcopy(baseline)
        reused.pop("protected_frozen_baseline", None)
        reused["reused_frozen_baseline"] = True
        entries[_entry_key(case_id, condition, repetition, repetitions)] = reused
    else:
        entries[_entry_key(case_id, condition, repetition, repetitions)] = baseline
    return True


def _group_target_action_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether the real replay activates the candidate-only contract."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return ReplyGenerator._group_target_action_contract(to_reply, all_messages) is not None


def _group_public_question_specific_followup_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether replay has a public, unresolved, concrete question."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_public_question_specific_followup_contract(
            to_reply, all_messages
        )
        is not None
    )


def _group_instant_self_correction_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether the real replay exposes an adjacent self turn to the candidate."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_instant_self_correction_contract(to_reply, all_messages)
        is not None
    )


def _group_visible_self_stance_continuity_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether replay has one strict adjacent self-to-same-speaker exchange."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_visible_self_stance_continuity_contract(
            to_reply, all_messages
        )
        is not None
    )


def _group_pending_self_action_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether replay has one strict adjacent pending self action."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_pending_self_action_contract(to_reply, all_messages)
        is not None
    )


def _group_pattern_participation_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether replay contains a supported multi-speaker group pattern."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_pattern_participation_contract(to_reply, all_messages)
        is not None
    )


def _group_current_event_self_role_continuity_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether replay proves a prior self turn belongs to the current event."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_current_event_self_role_continuity_contract(
            to_reply, all_messages
        )
        is not None
    )


def _group_explicit_self_stance_action_contract_applies(
    source: Candidate,
    payload: dict[str, Any],
) -> bool | None:
    """Return whether replay exposes a recent chronological self/other exchange."""
    try:
        to_reply, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
    except Exception:
        return None
    return (
        ReplyGenerator._group_explicit_self_stance_action_contract(
            to_reply, all_messages
        )
        is not None
    )


def _reuse_group_baseline_for_nontriggering_target_action_contract(
    entries: dict[str, Any],
    case_id: str,
    row: dict[str, Any],
    condition: str,
    repetition: int,
    repetitions: int,
    candidate_mode: str,
    contract_applies: bool | None,
    *,
    judge: bool = False,
) -> bool:
    """Keep non-triggering group cases identical instead of resampling Turbo."""
    if (
        candidate_mode not in {
            "group-target-action-contract",
            "group-public-question-specific-followup",
            "group-instant-self-correction",
            "group-visible-self-stance-continuity",
            "group-pending-self-action",
            "group-pattern-participation",
            "group-current-event-self-role-continuity",
            "group-explicit-self-stance-action",
            "group-answer-authority-rewrite",
        }
        or row.get("relationship") != "group"
        or condition != "candidate"
        or contract_applies is not False
    ):
        return False
    baseline = entries.get(_entry_key(case_id, "baseline", repetition, repetitions)) or {}
    if (judge and (not baseline or baseline.get("error"))) or (
        not judge and baseline.get("status") != "scorable"
    ):
        return False
    if baseline.get("protected_frozen_baseline"):
        reused = copy.deepcopy(baseline)
        reused.pop("protected_frozen_baseline", None)
        reused["reused_frozen_baseline"] = True
        entries[_entry_key(case_id, condition, repetition, repetitions)] = reused
    else:
        entries[_entry_key(case_id, condition, repetition, repetitions)] = baseline
    return True


def _seed_judge_entries(
    entries: dict[str, Any],
    run_case_ids: list[str],
    checkpoints: list[Path],
    repetitions: int = 1,
    conditions: tuple[str, ...] = ("baseline", "candidate"),
) -> int:
    allowed = {
        _entry_key(case_id, condition, repetition, repetitions)
        for case_id in run_case_ids
        for condition in conditions
        for repetition in range(1, repetitions + 1)
    }
    imported = 0
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            continue
        source = json.loads(checkpoint.read_text(encoding="utf-8"))
        for key, value in (source.get("entries") or {}).items():
            if key in allowed and not value.get("error") and (key not in entries or entries[key].get("error")):
                entries[key] = value
                imported += 1
    return imported


def _seed_baseline_from_candidate_judge_entries(
    entries: dict[str, Any],
    run_case_ids: list[str],
    checkpoints: list[Path],
    repetitions: int = 1,
) -> int:
    imported = 0
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            continue
        source = json.loads(checkpoint.read_text(encoding="utf-8"))
        for case_id in run_case_ids:
            for repetition in range(1, repetitions + 1):
                source_key = _entry_key(case_id, "candidate", repetition, repetitions)
                target_key = _entry_key(case_id, "baseline", repetition, repetitions)
                value = (source.get("entries") or {}).get(source_key) or {}
                if not value.get("error") and value:
                    entries[target_key] = value
                    imported += 1
    return imported


def _seed_split_baseline_entries(
    entries: dict[str, Any],
    run_case_ids: list[str],
    selected_by_id: dict[str, dict[str, Any]],
    checkpoints: list[Path],
    *,
    judge: bool = False,
    forbidden_by_case: dict[str, set[str]] | None = None,
) -> int:
    """Reuse adopted private outputs and retained group outputs as one baseline."""
    imported = 0
    sources: dict[str, Any] = {}
    for checkpoint in checkpoints:
        if checkpoint.exists():
            sources.update((json.loads(checkpoint.read_text(encoding="utf-8")).get("entries") or {}))
    for case_id in run_case_ids:
        relation = "group" if selected_by_id[case_id].get("relationship") == "group" else "private"
        source_condition = "baseline" if relation == "group" else "candidate"
        value = sources.get(f"{case_id}:{source_condition}") or {}
        if judge:
            usable = bool(value) and not value.get("error")
        else:
            few_shot_ids = set(((value.get("result") or {}).get("trace") or {}).get("few_shot_ids") or [])
            usable = (
                value.get("status") == "scorable"
                and not few_shot_ids.intersection((forbidden_by_case or {}).get(case_id, {case_id}))
            )
        target_key = f"{case_id}:baseline"
        if usable and (target_key not in entries or (judge and entries[target_key].get("error"))):
            entries[target_key] = value
            imported += 1
    return imported


def _seed_private_candidate_entries(
    entries: dict[str, Any],
    run_case_ids: list[str],
    selected_by_id: dict[str, dict[str, Any]],
    checkpoints: list[Path],
    *,
    forbidden_by_case: dict[str, set[str]] | None = None,
) -> int:
    """Reuse candidate private outputs and baseline group outputs as one candidate."""
    imported = 0
    sources: dict[str, Any] = {}
    for checkpoint in checkpoints:
        if checkpoint.exists():
            sources.update((json.loads(checkpoint.read_text(encoding="utf-8")).get("entries") or {}))
    for case_id in run_case_ids:
        relation = "group" if selected_by_id[case_id].get("relationship") == "group" else "private"
        source_condition = "baseline" if relation == "group" else "candidate"
        value = sources.get(f"{case_id}:{source_condition}") or {}
        few_shot_ids = set(((value.get("result") or {}).get("trace") or {}).get("few_shot_ids") or [])
        target_key = f"{case_id}:candidate"
        if (
            value.get("status") == "scorable"
            and not few_shot_ids.intersection((forbidden_by_case or {}).get(case_id, {case_id}))
            and entries.get(target_key, {}).get("status") != "scorable"
        ):
            entries[target_key] = value
            imported += 1
    return imported


def _reuse_group_baseline_judge_for_private_only_candidate(
    entries: dict[str, Any],
    case_id: str,
    row: dict[str, Any],
    condition: str,
    repetition: int,
    repetitions: int,
    private_only_candidate: bool,
) -> bool:
    """Reuse one Judge decision when the group candidate reply is the baseline reply."""
    if (
        not private_only_candidate
        or row.get("relationship") != "group"
        or condition != "candidate"
    ):
        return False
    source_key = _entry_key(case_id, "baseline", repetition, repetitions)
    target_key = _entry_key(case_id, "candidate", repetition, repetitions)
    source = entries.get(source_key) or {}
    if not source or source.get("error"):
        return False
    entries[target_key] = source
    return True


def _entry_key(case_id: str, condition: str, repetition: int, repetitions: int) -> str:
    if repetitions == 1:
        return f"{case_id}:{condition}"
    return f"{case_id}:{condition}:{repetition}"


def _has_nonempty_reply(result: dict[str, Any]) -> bool:
    return any(str(reply).strip() for reply in (result.get("replies") or []))


def _reference_without_case_leaks(
    reference: dict[str, Any], case: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filtered = copy.deepcopy(reference)
    removed: list[dict[str, Any]] = []
    while leaks := reference_leakage_details(filtered, [case]):
        indexes: dict[str, set[int]] = {"examples": set(), "scenario_few_shots": set()}
        for leak in leaks:
            match = re.match(
                r"reference\.(examples|scenario_few_shots)\[(\d+)\]",
                leak["reference_path"],
            )
            if not match:
                raise RuntimeError(f"cannot remove contamination outside reference examples: {leak}")
            indexes[match.group(1)].add(int(match.group(2)))
        for section, section_indexes in indexes.items():
            rows = filtered.get(section) or []
            for index in sorted(section_indexes, reverse=True):
                removed.append({"section": section, "index": index, "id": rows[index].get("id")})
                del rows[index]
    assert_reference_isolated(filtered, [case])
    return filtered, removed


def _generation_schedule(repetitions: int) -> list[tuple[str, int]]:
    if repetitions == 1:
        return [("baseline", 1), ("candidate", 1)]
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    schedule = []
    for repetition in range(1, repetitions + 1):
        conditions = ("baseline", "candidate") if repetition % 2 else ("candidate", "baseline")
        schedule.extend((condition, repetition) for condition in conditions)
    return schedule


def _few_shot_count_for_condition(args: argparse.Namespace, condition: str) -> int:
    if args.candidate_mode == "few-shot-count-5" and condition == "candidate":
        return 5
    return args.few_shot_count


def _few_shot_path_for_condition(args: argparse.Namespace, condition: str) -> Path:
    if args.candidate_mode == "expanded-pool" and condition == "candidate":
        return args.candidate_few_shot
    return args.few_shot


def _information_gap_response_move_for_condition(
    candidate_mode: str, condition: str
) -> bool:
    if candidate_mode == "information-gap-response-move":
        return condition == "candidate"
    return True


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT.resolve())),
        "rows": sum(1 for line in resolved.open(encoding="utf-8") if line.strip()),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


_SKILL_SNAPSHOT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _load_skill_snapshots(
    specs: list[str],
    option_name: str,
) -> dict[str, dict[str, str]]:
    snapshots: dict[str, dict[str, str]] = {}
    for raw_spec in specs:
        name, separator, raw_path = str(raw_spec).partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"{option_name} must use NAME=PATH")
        if not _SKILL_SNAPSHOT_NAME_RE.fullmatch(name):
            raise ValueError(f"{option_name} has an invalid skill name: {name!r}")
        if name in snapshots:
            raise ValueError(f"{option_name} repeats skill {name!r}")
        source_path = Path(raw_path).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"{option_name} must reference a UTF-8 regular file: {source_path}")
        try:
            content = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{option_name} must reference a UTF-8 file: {source_path}") from exc
        try:
            display_path = str(source_path.relative_to(ROOT.resolve()))
        except ValueError:
            display_path = str(source_path)
        snapshots[name] = {
            "source_path": display_path,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content": content,
        }
    return snapshots


def _skill_snapshot_overrides(args: argparse.Namespace) -> dict[str, dict[str, dict[str, str]]]:
    if args.candidate_mode != "skill-snapshot":
        if args.baseline_skill_snapshot or args.candidate_skill_snapshot:
            raise ValueError("skill snapshots require --candidate-mode skill-snapshot")
        return {"baseline": {}, "candidate": {}}
    baseline = _load_skill_snapshots(
        args.baseline_skill_snapshot,
        "--baseline-skill-snapshot",
    )
    candidate = _load_skill_snapshots(
        args.candidate_skill_snapshot,
        "--candidate-skill-snapshot",
    )
    if set(baseline) != {"group_banter"} or set(candidate) != {"group_banter"}:
        raise ValueError(
            "skill-snapshot requires exactly one group_banter snapshot for baseline and candidate"
        )
    if baseline["group_banter"]["content"] == candidate["group_banter"]["content"]:
        raise ValueError("skill-snapshot baseline and candidate bodies must differ")
    return {"baseline": baseline, "candidate": candidate}


def _formal_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "id": str(row["internal_id"]),
        "relationship": "group" if row.get("chat_type") == "group" else "private",
    }


def _file_sha256_provenance(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _is_frozen_formal_baseline_target(args: argparse.Namespace) -> bool:
    return (
        args.dataset == "formal240"
        and args.candidate_mode == "group-public-question-specific-followup"
    )


def _load_frozen_formal_baseline(
    args: argparse.Namespace,
    baseline_id: str,
) -> dict[str, Any] | None:
    """Recover the one approved formal baseline without ever generating it again."""
    if not _is_frozen_formal_baseline_target(args):
        return None

    frozen_run = Path(
        getattr(args, "frozen_formal_baseline_run", None)
        or DEFAULT_FROZEN_FORMAL_BASELINE_RUN
    ).resolve()
    source_manifest_values = getattr(args, "frozen_formal_source_manifest", None)
    source_manifests = tuple(
        Path(value).resolve()
        for value in (
            source_manifest_values
            if source_manifest_values
            else DEFAULT_FROZEN_FORMAL_SOURCE_MANIFESTS
        )
    )
    frozen_manifest_path = frozen_run / "manifest.json"
    if not frozen_manifest_path.is_file():
        raise ValueError(f"frozen formal baseline manifest is missing: {frozen_manifest_path}")
    if not source_manifests or any(not path.is_file() for path in source_manifests):
        raise ValueError("every frozen formal source manifest must exist")

    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen_manifest.get("current_c0_baseline_id") != baseline_id:
        raise ValueError("frozen formal baseline is not bound to the adopted current C0")
    case_ids = [str(case_id) for case_id in frozen_manifest.get("case_ids") or []]
    if len(case_ids) != 240 or len(set(case_ids)) != 240:
        raise ValueError("frozen formal baseline must contain exactly 240 unique case ids")

    source_cases: dict[str, dict[str, Any]] = {}
    for source_manifest_path in source_manifests:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        for source_row in source_manifest.get("cases") or []:
            row = _formal_case(source_row)
            case_id = str(row["id"])
            if case_id in source_cases:
                raise ValueError(f"frozen formal source manifests repeat case id: {case_id}")
            source_cases[case_id] = row
    missing_cases = [case_id for case_id in case_ids if case_id not in source_cases]
    if missing_cases:
        raise ValueError(
            "frozen formal baseline case ids cannot be recovered from the source manifests: "
            + ", ".join(missing_cases[:5])
        )
    cases = [copy.deepcopy(source_cases[case_id]) for case_id in case_ids]
    distribution = _distribution(cases)
    if distribution["relationship"] != {"group": 160, "private": 80}:
        raise ValueError(
            "frozen formal baseline must contain 160 group and 80 private cases"
        )

    baseline_entries: dict[str, dict[str, Any]] = {}
    checkpoint_paths = sorted(frozen_run.glob("generation_checkpoint*.json"))
    if not checkpoint_paths:
        raise ValueError("frozen formal baseline has no generation checkpoints")
    expected_keys = {case_id: _entry_key(case_id, "baseline", 1, 1) for case_id in case_ids}
    for checkpoint_path in checkpoint_paths:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("current_c0_baseline_id") != baseline_id:
            raise ValueError(
                f"frozen formal baseline checkpoint is bound to another C0: {checkpoint_path}"
            )
        for case_id, expected_key in expected_keys.items():
            entry = (checkpoint.get("entries") or {}).get(expected_key)
            if entry is None:
                continue
            if case_id in baseline_entries:
                raise ValueError(f"frozen formal baseline repeats case: {case_id}")
            if entry.get("status") != "scorable" or not _has_nonempty_reply(entry.get("result") or {}):
                raise ValueError(
                    f"frozen formal baseline is not scorable with a reply: {case_id}"
                )
            baseline_entries[case_id] = copy.deepcopy(entry)
    missing_entries = [case_id for case_id in case_ids if case_id not in baseline_entries]
    if missing_entries:
        raise ValueError(
            "frozen formal baseline is missing checkpoint entries: "
            + ", ".join(missing_entries[:5])
        )

    return {
        "cases": cases,
        "baseline_entries": baseline_entries,
        "distribution": distribution,
        "provenance": {
            "run_manifest": _file_sha256_provenance(frozen_manifest_path),
            "source_manifests": [
                _file_sha256_provenance(path) for path in source_manifests
            ],
            "generation_checkpoints": [
                _file_sha256_provenance(path) for path in checkpoint_paths
            ],
        },
    }


def _import_frozen_formal_baseline_entries(
    entries: dict[str, dict[str, Any]],
    run_case_ids: list[str],
    frozen_bundle: dict[str, Any],
) -> int:
    """Import canonical baseline replies and reject any generated substitute."""
    imported = 0
    for case_id in run_case_ids:
        key = _entry_key(case_id, "baseline", 1, 1)
        canonical = copy.deepcopy(frozen_bundle["baseline_entries"][case_id])
        canonical["protected_frozen_baseline"] = True
        existing = entries.get(key)
        if existing is None:
            entries[key] = canonical
            imported += 1
            continue
        if not existing.get("protected_frozen_baseline"):
            raise ValueError(
                f"output contains a regenerated formal baseline and cannot be resumed: {key}"
            )
        if existing != canonical:
            raise ValueError(
                f"output frozen formal baseline does not match its canonical source: {key}"
            )
    return imported


def _payload_for_case(
    row: dict[str, Any],
    dataset: str,
    exports: dict[str, dict[str, Any]],
    source_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any] | None:
    if dataset != "formal240":
        return exports.get(_candidate(row).chat_id)
    source_value = row.get("source_path")
    if not source_value:
        return None
    source_path = Path(str(source_value))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path = source_path.resolve()
    if source_path not in source_cache:
        source_cache[source_path] = json.loads(source_path.read_text(encoding="utf-8"))
    payload = source_cache[source_path]
    if payload.get("messages") is not None:
        return payload
    anchor_server_id = str(row.get("anchor_server_id") or "")
    target_reply = "".join(row.get("reply") or []) if isinstance(row.get("reply"), list) else str(row.get("reply") or "")
    target_seconds = float(row.get("timestamp") or 0)
    target_seconds = target_seconds / 1000 if target_seconds > 10_000_000_000 else target_seconds
    for source in payload.get("source_files") or []:
        nested_path = Path(str(source.get("path") or ""))
        if not nested_path.is_absolute():
            nested_path = ROOT / nested_path
        nested_path = nested_path.resolve()
        if nested_path not in source_cache:
            source_cache[nested_path] = json.loads(nested_path.read_text(encoding="utf-8"))
        nested = source_cache[nested_path]
        if any(
            (
                bool(anchor_server_id)
                and str(message.get("serverId") or message.get("platformMessageId") or "") == anchor_server_id
            )
            or (
                not anchor_server_id
                and bool(message.get("isSend"))
                and (
                    float(message.get("createTime") or 0)
                    / (1000 if float(message.get("createTime") or 0) > 10_000_000_000 else 1)
                ) == target_seconds
                and str(message.get("content") or "") == target_reply
            )
            for message in nested.get("messages") or []
        ):
            return nested
    return payload


def _full_generation_contexts(
    cases: list[dict[str, Any]],
    dataset: str,
    exports: dict[str, dict[str, Any]],
    source_cache: dict[Path, dict[str, Any]],
) -> dict[str, list[str]]:
    contexts: dict[str, list[str]] = {}
    for row in cases:
        case_id = str(row["id"])
        source = _candidate(row)
        payload = _payload_for_case(row, dataset, exports, source_cache)
        if payload is None:
            raise ValueError(f"cannot isolate few-shots without source timeline: {case_id}")
        _, all_messages, _ = rebuild_generation_inputs(
            source,
            payload,
            compact_verified_turn=True,
        )
        contexts[case_id] = [str(message.text or "") for message in all_messages]
    return contexts


def _formal_preflight(args: argparse.Namespace) -> None:
    private_ranking = args.candidate_mode == "latest-turn-form" and args.private_only_candidate
    group_pending_self_action = args.candidate_mode == "group-pending-self-action"
    group_public_question_specific_followup = (
        args.candidate_mode == "group-public-question-specific-followup"
    )
    group_pattern_participation = args.candidate_mode == "group-pattern-participation"
    group_current_event_self_role_continuity = (
        args.candidate_mode == "group-current-event-self-role-continuity"
    )
    group_explicit_self_stance_action = (
        args.candidate_mode == "group-explicit-self-stance-action"
    )
    group_skill_snapshot = args.candidate_mode == "skill-snapshot"
    group_turbo = (
        args.candidate_mode == "generation-model"
        and args.group_only_candidate
        and args.generation_provider == "volcengine"
        and args.generation_model == FORMAL_GROUP_TURBO_MODEL
    )
    if (
        not private_ranking
        and not group_pending_self_action
        and not group_public_question_specific_followup
        and not group_pattern_participation
        and not group_current_event_self_role_continuity
        and not group_explicit_self_stance_action
        and not group_skill_snapshot
        and not group_turbo
    ):
        raise ValueError(
            "formal240 allows private-only latest-turn-form, group-pending-self-action, "
            "group-public-question-specific-followup, "
            "group-pattern-participation, "
            "group-current-event-self-role-continuity, "
            "group-explicit-self-stance-action, "
            "group_banter skill-snapshot, "
            "or group-only doubao-seed-2.1-turbo"
        )
    if args.repetitions != 1:
        raise ValueError("formal240 requires repetitions=1")
    if args.max_cases != 240:
        raise ValueError("formal240 requires max-cases=240")
    if args.case_id or args.case_ids_from_results:
        raise ValueError("formal240 does not allow partial case selection")
    if args.generation_provider != "volcengine":
        raise ValueError("formal240 must use the same Volcengine generation provider as development80")
    if args.few_shot_count != 3:
        raise ValueError("formal240 must use the same three few-shot examples per case as development80")
    if args.few_shot_max_chars != 2500:
        raise ValueError("formal240 must use the same 2500-character few-shot prompt budget as development80")
    if args.named_context_few_shot.resolve() != DEFAULT_NAMED_CONTEXT_FEW_SHOT.resolve():
        raise ValueError("formal240 must use the same named-context rendering library as development80")
    if hashlib.sha256(args.named_context_few_shot.read_bytes()).hexdigest() != DEVELOPMENT80_NAMED_CONTEXT_SHA256:
        raise ValueError("the named-context rendering library changed after development80")
    if hashlib.sha256(DEFAULT_DEVELOPMENT80_FORCE_REPLY_PERSONA.read_bytes()).hexdigest() != DEVELOPMENT80_FORCE_REPLY_SHA256:
        raise ValueError("the forced-reply prompt changed after development80")
    development_manifest = json.loads(DEFAULT_DEVELOPMENT80_RUN.read_text(encoding="utf-8"))
    development_library = development_manifest["conditions"]["retrieval_libraries"]["candidate"]
    base = ROOT / development_library["path"]
    base_fingerprint = _file_fingerprint(base)
    if base_fingerprint != development_library:
        raise ValueError("the saved 80-case development pool no longer matches its manifest")
    if args.few_shot.resolve() != base.resolve() or _file_fingerprint(args.few_shot) != base_fingerprint:
        raise ValueError("formal240 must use the exact same complete few-shot pool as development80")


def _select_dataset(
    args: argparse.Namespace,
    frozen_bundle: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    allow_exposed_recheck = getattr(args, "allow_exposed_case_recheck", False)
    if allow_exposed_recheck:
        if args.dataset != "development80":
            raise ValueError("exposed-case recheck is only allowed for development80")
        if not args.case_id or args.case_ids_from_results:
            raise ValueError("exposed-case recheck requires explicit --case-id values only")

    if args.dataset == "formal240":
        _formal_preflight(args)
        if _is_frozen_formal_baseline_target(args):
            if frozen_bundle is None:
                raise ValueError("protected formal target requires a frozen baseline bundle")
            return (
                copy.deepcopy(frozen_bundle["cases"]),
                "approved frozen 240-case C0 baseline; candidate-only group change",
            )
        fixed_test = json.loads(args.fixed_test.read_text(encoding="utf-8"))
        cases = [_formal_case(row) for row in fixed_test.get("cases") or []]
        if len(cases) != 240 or len({row["id"] for row in cases}) != 240:
            raise ValueError(f"formal240 manifest must contain 240 unique cases, got {len(cases)}")
        return cases, "fixed 240 cases; only the dataset differs from development80"

    development = json.loads(args.development.read_text(encoding="utf-8"))
    exposed = json.loads(args.exposed.read_text(encoding="utf-8"))
    previous_paths = args.previous_ab or [DEFAULT_PREVIOUS_AB]
    previous_abs = [json.loads(path.read_text(encoding="utf-8")) for path in previous_paths]
    fixed_test = json.loads(args.fixed_test.read_text(encoding="utf-8"))
    excluded = _excluded_case_ids(exposed, previous_abs, fixed_test)
    development_cases = development["cases"]
    fixed_case_ids = _case_ids_from_results(args.case_ids_from_results) if args.case_ids_from_results else []
    if fixed_case_ids:
        rows_by_id = {str(row["id"]): row for row in development_cases}
        missing = [case_id for case_id in fixed_case_ids if case_id not in rows_by_id or case_id in excluded]
        if missing:
            raise ValueError(f"fixed case ids are not available in the development set: {missing}")
        selected = [rows_by_id[case_id] for case_id in fixed_case_ids[:args.max_cases]]
    elif args.case_id:
        requested_order = list(dict.fromkeys(args.case_id))
        requested = set(requested_order)
        if allow_exposed_recheck:
            rows_by_id = {str(row["id"]): row for row in development_cases}
            selected = [rows_by_id[case_id] for case_id in requested_order[:args.max_cases] if case_id in rows_by_id]
        else:
            development_cases = [row for row in development_cases if str(row["id"]) in requested]
            selected = _select_cases(development_cases, excluded, args.max_cases, args.candidate_mode)
            selected = [row for row in selected if str(row["id"]) in requested]
        missing = requested - {str(row["id"]) for row in selected}
        if missing:
            raise ValueError(f"requested case ids are not available in the selected development set: {sorted(missing)}")
    else:
        selected = _select_cases(development_cases, excluded, args.max_cases, args.candidate_mode)
    if allow_exposed_recheck:
        return selected, "causal recheck of explicitly named exposed development cases; excluded from new development and formal results"
    return selected, "development cases selected from the frozen development pool"


def run(args: argparse.Namespace) -> None:
    current_c0 = load_current_c0()
    baseline_id = current_c0_baseline_id()
    adopted_few_shot = Path(current_c0["few_shot_path"])
    if args.few_shot.resolve() != adopted_few_shot.resolve():
        raise ValueError("baseline few-shot path must match the adopted current C0")
    if args.few_shot_count != current_c0["few_shot"]["count"]:
        raise ValueError("baseline few-shot count must match the adopted current C0")
    if args.few_shot_max_chars != current_c0["few_shot"]["max_chars"]:
        raise ValueError("baseline few-shot max chars must match the adopted current C0")
    skill_snapshots = _skill_snapshot_overrides(args)
    seed_paths = (
        args.seed_generation_checkpoint
        + args.seed_candidate_generation_checkpoint
        + args.seed_baseline_from_candidate_generation_checkpoint
        + args.seed_judge_checkpoint
        + args.seed_baseline_from_candidate_judge_checkpoint
        + args.seed_split_baseline_generation_checkpoint
        + args.seed_private_candidate_generation_checkpoint
        + args.seed_split_baseline_judge_checkpoint
    )
    for seed_path in seed_paths:
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        if seed.get("current_c0_baseline_id") != baseline_id:
            raise ValueError(f"seed checkpoint is not bound to the adopted current C0 baseline: {seed_path}")
    frozen_bundle = _load_frozen_formal_baseline(args, baseline_id)
    selected, selection_description = _select_dataset(
        args,
        frozen_bundle=frozen_bundle,
    )
    exports = _export_payloads(args.exports)
    source_payload_cache: dict[Path, dict[str, Any]] = {}
    full_context_by_case = _full_generation_contexts(
        selected,
        args.dataset,
        exports,
        source_payload_cache,
    )
    few_shot_rows_by_id = _load_rows_by_id(args.few_shot)
    leak_check_paths = {
        args.few_shot,
        _few_shot_path_for_condition(args, "candidate"),
        args.named_context_few_shot,
    }
    if args.candidate_mode == "group-context-window-10":
        leak_check_paths.add(args.long_named_context_few_shot)
    per_case_overlap_ids = _per_case_overlap_ids_from_pools(
        selected,
        [_load_rows_by_id(path) for path in leak_check_paths],
        full_context_by_case,
    )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be within shard-count")
    if getattr(args, "preflight_only", False):
        preflight = {
            "status": "passed",
            "dataset": args.dataset,
            "cases": len(selected),
            "few_shot": _file_fingerprint(args.few_shot),
            "generation_provider": args.generation_provider,
            "candidate_mode": args.candidate_mode,
            "few_shot_count": args.few_shot_count,
            "skill_snapshots": {
                condition: {
                    name: {key: value for key, value in snapshot.items() if key != "content"}
                    for name, snapshot in snapshots.items()
                }
                for condition, snapshots in skill_snapshots.items()
            },
        }
        if frozen_bundle is not None:
            preflight["frozen_formal_baseline"] = {
                "baseline_imported_or_reused": len(selected),
                "baseline_generation": 0,
                **frozen_bundle["distribution"]["relationship"],
            }
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    selected_by_id = {str(row["id"]): row for row in selected}
    manifest_path = args.output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "current_c0_baseline_id": baseline_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": (
            "formal 240-case forced-reply C0 A/B"
            if args.dataset == "formal240"
            else "development-only forced-reply C0 A/B; not formal 240-case testing"
        ),
        "dataset": args.dataset,
        "causal_recheck": bool(getattr(args, "allow_exposed_case_recheck", False)),
        "selection": selection_description if args.dataset == "formal240" else (
            "fresh verified-human cases with at least three context messages and a substantive latest request, "
            "balanced group/private and round-robin by request kind; excludes prior exposed development, "
            "prior A/B, source cases, and all fixed-240 ids"
            if args.candidate_mode == "latest-request-first"
            else "fresh verified-human cases, balanced group/private and round-robin by situation; excludes prior exposed development, prior A/B, source cases, and all fixed-240 ids"
        ),
        "force_reply": True,
        "repetitions_per_condition": args.repetitions,
        "generation_order_per_case": [
            f"{'B' if condition == 'baseline' else 'C'}{repetition}"
            for condition, repetition in _generation_schedule(args.repetitions)
        ],
        "conditions": {
            "shared_human_library": (
                None if args.candidate_mode == "expanded-pool" else str(args.few_shot.resolve().relative_to(ROOT.resolve()))
            ),
            "retrieval_libraries": {
                "baseline": _file_fingerprint(args.few_shot),
                "candidate": _file_fingerprint(_few_shot_path_for_condition(args, "candidate")),
            },
            "baseline": "current C0 baseline",
            "candidate": (
                f"group chats use {args.generation_model} instead of {VOLCENGINE_DSV4F_MODEL}; private chats reuse the adopted baseline output"
                if args.candidate_mode == "generation-model" and args.group_only_candidate
                else f"the current C0 configuration generated by {args.generation_model} instead of {VOLCENGINE_DSV4F_MODEL}"
                if args.candidate_mode == "generation-model"
                else "private chats add latest-turn-form ranking; group chats reuse the baseline output"
                if args.private_only_candidate
                else
                "private chats reuse the baseline output; group chats use the frozen candidate group_banter skill body"
                if args.candidate_mode == "skill-snapshot"
                else
                "private chats keep the adopted latest-turn-form ranking; group chats add participant-structure ranking based on who spoke, who was addressed, and whether the speaker joined the recent exchange"
                if args.candidate_mode == "group-participant-structure"
                else "private chats are unchanged; group chats keep the same three examples and expand each example from at most five to at most ten continuous named messages"
                if args.candidate_mode == "group-context-window-10"
                else "private chats are unchanged; before writing a group-chat reply, identify who said the latest message, who it addresses, and whose stance each statement represents"
                if args.candidate_mode == "group-speaker-target-attribution"
                else "private chats are unchanged; only a group message that explicitly @-names and introduces another person's identity or role adds a reminder not to answer as that person"
                if args.candidate_mode == "group-explicit-third-party-guard"
                else "private chats are unchanged; group chats may use a recent self message only when the current unread directly responds to that fact, judgment, attitude, or constraint"
                if args.candidate_mode == "group-relevant-self-evidence"
                else "private chats are unchanged; group chats keep used jokes closed but may use a directly answered current self fact, state, plan, constraint, or judgment as background"
                if args.candidate_mode == "group-self-context-freshness-boundary"
                else "private chats are unchanged; when a group member actually responds to the immediately preceding self message, that one self fact, state, judgment, or plan may remain background even if the response is a short reaction"
                if args.candidate_mode == "group-adjacent-self-response"
                else "private chats are unchanged; group-chat history keeps self and other messages together in their original chronological order, with every speaker and role still explicit"
                if args.candidate_mode == "group-unified-history"
                else "private chats are unchanged; group history stays chronological and the model must select the one role=self message that the current unread actually answers, or none"
                if args.candidate_mode == "group-unified-relevant-self"
                else "private chats are unchanged; when one other-person message immediately follows the speaker's own message, the model may prioritize an obvious one-word self-correction over mechanically answering the intervening message"
                if args.candidate_mode == "group-instant-self-correction"
                else "private chats are unchanged; only a strict uninterrupted exchange of one self text followed by one or two texts from the same group member restores that local timeline so the model can identify the addressed stance and reply action"
                if args.candidate_mode == "group-visible-self-stance-continuity"
                else "private chats are unchanged; only a strict adjacent group exchange with a pending stance challenge or explicit choice restores the one relevant self message and names the reply action"
                if args.candidate_mode == "group-pending-self-action"
                else "private chats are unchanged; only a visible multi-speaker group pattern that includes the unread turn adds guidance to participate in its content without commenting on the pattern itself"
                if args.candidate_mode == "group-pattern-participation"
                else "private chats are unchanged; only when the complete timeline proves the speaker still participates in the current concrete event, restore the single most relevant self message to identify the reply target and action"
                if args.candidate_mode == "group-current-event-self-role-continuity"
                else "private chats are unchanged; only a recent chronological group exchange containing the speaker's own words asks the model to identify the reply target, interaction action, existing self stance, and genuinely new evidence before replying"
                if args.candidate_mode == "group-explicit-self-stance-action"
                else "private chats are unchanged; triggered group replies first receive the target/action answer-authority contract, then a narrow review removes only unsupported concrete confirmations, denials, or commitments while preserving questions, clarification, banter, uncertainty, and natural avoidance"
                if args.candidate_mode == "group-answer-authority-rewrite"
                else "private chats are unchanged; group messages with reliable address or interaction signals identify the likely addressee and interaction need, but treat neither as answer evidence and require independent support before confirming facts, states, plans, intentions, rules, advice, or conclusions"
                if args.candidate_mode == "group-target-action-contract"
                else "private chats reuse the baseline output; each group-chat prompt is first analyzed into validated visible interaction features and a short fact-bounded reply plan, then the same Turbo model produces the final reply; invalid, unsupported, failed, or timed-out auxiliary output falls back to the unchanged final generation prompt"
                if args.candidate_mode == "group-feature-plan-before-generation"
                else
                "five situation-ranked verified-human examples instead of three"
                if args.candidate_mode == "few-shot-count-5"
                else "the same ranking logic searches the expanded frozen-holdout-excluded human-example pool"
                if args.candidate_mode == "expanded-pool"
                else "the exact same selected examples rendered with real speaker names and up to five continuous chat messages"
                if args.candidate_mode == "named-context-render"
                else "after matching the same situation, examples with the right way to respond rank before examples that only share the topic or wording"
                if args.candidate_mode == "response-move-before-relevance"
                else "the dense semantic topic-similarity weight is increased from 4.0 to 8.0"
                if args.candidate_mode == "semantic-topic-weight-8"
                else "when a question offers multiple numeric factual answers without direct evidence, do not guess one"
                if args.candidate_mode == "unresolved-fact-choice"
                else "replace restating an already-complete fact, arrangement, or conclusion with one context-grounded conversational action"
                if args.candidate_mode == "no-restatement-action"
                else "when the source, role, basis, or required detail is missing, prefer verified-human examples that verify or ask for it"
                if args.candidate_mode == "information-gap-response-move"
                else "preserve the speaker's explicitly stated current stance while answering the latest turn"
                if args.candidate_mode == "current-stance-continuity"
                else "write the needed plain conversational response before deciding whether the live exchange supports styling it as a joke"
                if args.candidate_mode == "plain-response-before-styling"
                else "clarification-first situation matching enabled"
                if args.candidate_mode == "clarification"
                else (
                    "recent self replies may be used only to continue the speaker's active thread"
                    if args.candidate_mode == "active-thread"
                    else (
                        "precise conversational-setup matching for human examples"
                        if args.candidate_mode == "precise-situation"
                        else (
                            "the latest request addressed to the speaker must be handled before banter or older topics"
                            if args.candidate_mode == "latest-request-first"
                            else (
                                "after matching the situation and response move, prefer verified human examples from the current chat"
                                if args.candidate_mode == "same-chat-after-situation"
                                else (
                                    "match human examples to the situation and reply move in the latest message"
                                    if args.candidate_mode == "latest-turn-situation"
                                    else (
                                        "use the latest message to match the situation and reply move, then prefer the current chat; sparse turns avoid concrete examples"
                                        if args.candidate_mode == "latest-turn-same-chat-sparse"
                                        else "situation-matched human examples ground the reply action and brevity"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "only_variable": (
                f"group-chat C0 generation model only: baseline {VOLCENGINE_DSV4F_MODEL}; candidate {args.generation_model}; private outputs and Judge decisions are reused; prompts, context, few-shot pool, ranking, selected examples, generation parameters, and Judge are identical"
                if args.candidate_mode == "generation-model" and args.group_only_candidate
                else f"C0 generation model only: baseline {VOLCENGINE_DSV4F_MODEL}; candidate {args.generation_model}; dataset, prompts, context, few-shot pool, ranking, selected examples, generation parameters, and Judge are identical"
                if args.candidate_mode == "generation-model"
                else "private-chat few-shot ordering only: candidate adds latest-turn-form ranking; every group-chat candidate output is the baseline output"
                if args.private_only_candidate
                else
                "one frozen group_banter skill body only: baseline and candidate inject the separately recorded bodies; private outputs reuse baseline, while model, current C0 configuration, prompts, context, few-shot IDs and order, generation parameters, and Judge are identical"
                if args.candidate_mode == "skill-snapshot"
                else
                "group-chat few-shot ordering only: baseline ignores participant structure; candidate adds five exact participant-structure matches at +1.5 each and mismatches at -1.5 each; private-chat retrieval is identical"
                if args.candidate_mode == "group-participant-structure"
                else "group-chat few-shot context length only: the same IDs and order use at most five versus at most ten continuous named messages; private chats are identical"
                if args.candidate_mode == "group-context-window-10"
                else "group-chat generation instruction only: the candidate must attribute each statement to its speaker, identify the latest addressee, and avoid answering as another participant; private chats, few-shot pool, ranking, count, and rendering are identical"
                if args.candidate_mode == "group-speaker-target-attribution"
                else "group-chat generation instruction only, and only on an explicit @-named identity or role introduction; every non-triggering prompt path and all private chats are identical"
                if args.candidate_mode == "group-explicit-third-party-guard"
                else "group-chat prompt treatment of recent self messages only: forbidden background versus context allowed only when the unread directly depends on that exact self evidence; private chats, few-shot retrieval, model, and all other rules are identical"
                if args.candidate_mode == "group-relevant-self-evidence"
                else "group-chat freshness boundary only: used jokes, metaphors, number gags, person labels, and invented relationships remain forbidden, while a directly answered current self fact, state, plan, constraint, or judgment may remain background; private chats, few-shot retrieval, model, and all other rules are identical"
                if args.candidate_mode == "group-self-context-freshness-boundary"
                else "one group-chat trigger boundary only: baseline requires an explicit denial, correction, answer, challenge, defence, or rebuttal; candidate also accepts a short reaction only when it immediately follows and concretely responds to one self fact, state, judgment, or plan; ordinary reactions, older self history, private chats, few-shot retrieval, model, and all other rules are identical"
                if args.candidate_mode == "group-adjacent-self-response"
                else "group-chat history layout only: baseline separates self messages into consumed_self_replies; candidate places the same selected messages once in one chronological history with role=self or role=other, while keeping the prohibition on copying old self wording and jokes; private chats, selected messages, few-shot retrieval, model, and all other rules are identical"
                if args.candidate_mode == "group-unified-history"
                else "one group-chat context treatment only: candidate preserves one chronological history and requires selecting exactly one directly answered role=self message or none; it never creates a separate self-history section; private chats, selected messages, few-shot retrieval, model, and Judge are identical"
                if args.candidate_mode == "group-unified-relevant-self"
                else "one group-chat prompt contract only: when exactly one unread other-person text immediately follows a visible self text, candidate may output only an obvious local homophone, near-homophone, or one-character correction to that self text; uncertain cases keep the original generation path; non-triggering group prompts and all private outputs are identical; model, context, few-shot IDs and order, and Judge are unchanged"
                if args.candidate_mode == "group-instant-self-correction"
                else "one group-chat context treatment only: for a strict uninterrupted tail of one self text followed by one or two non-empty texts from the same member, candidate restores only that local chronological exchange and may use the self text solely when the unread concretely continues its fact, judgment, preference, or situation; all other self history stays forbidden, and non-triggering group prompts plus every private output are identical"
                if args.candidate_mode == "group-visible-self-stance-continuity"
                else "one group-chat pending-action treatment only: a strict adjacent exchange opens exactly one self message only when the same member either questions that specific stance or explicitly decides a choice requested by the speaker; non-triggering group prompts, every private output, context, few-shot IDs and order, model, and Judge are identical"
                if args.candidate_mode == "group-pending-self-action"
                else "one group-chat interaction-pattern treatment only: candidate guidance appears only when the unread turn joins a non-generic exact repeat, a shared first-person stance, or multi-speaker deferral; non-triggering group prompts, every private output, context, few-shot IDs and order, generation model, and Judge are unchanged"
                if args.candidate_mode == "group-pattern-participation"
                else "one group-chat context treatment only: when the complete timeline proves the speaker remains a participant, actor, questioner, or stance holder in the current concrete event, candidate restores exactly one most relevant self message; non-triggering group prompts, every private output, model, few-shot IDs and order, timeline, generation parameters, and Judge are unchanged"
                if args.candidate_mode == "group-current-event-self-role-continuity"
                else "one group-chat reasoning contract only: on a recent chronological self/other exchange, candidate identifies the current reply target, interaction action, prior self stance, and new evidence; it preserves the stance unless new evidence supports a natural update and may not copy old wording or invent facts; non-triggering group prompts and every private output are reused, while model, timeline, few-shot IDs and order, generation parameters, and Judge are unchanged"
                if args.candidate_mode == "group-explicit-self-stance-action"
                else "one group-chat answer-authority treatment only: candidate adds the same target/action contract and, only when it triggers, a narrow post-generation review that may delete unsupported concrete confirmations, denials, or commitments but must preserve the reply action and relational tone; non-triggering group prompts, every private output, context, few-shot IDs and order, generation model, and Judge are unchanged"
                if args.candidate_mode == "group-answer-authority-rewrite"
                else "one group-chat answer-authority boundary only: candidate keeps the same likely addressee and interaction-need guidance but states that neither is answer evidence; confirming a fact, state, plan, intention, group rule, concrete advice, or conclusion requires direct support outside the target message from an earlier self message, trusted memory, or a tool result; clarification, uncertainty, relevant thread continuation, and relational responses remain allowed; non-triggering group prompts and every private output are identical; model, context, few-shot IDs and order, and Judge are unchanged"
                if args.candidate_mode == "group-target-action-contract"
                else "group-chat pre-generation reasoning only: the candidate runs feature extraction and a short plan against the same final prompt, then appends only validated visible evidence, target/action, and fact boundaries before the unchanged final Turbo generation; every private candidate output reuses baseline, and all context, few-shot IDs and order, generation model, parameters, and Judge are unchanged; auxiliary failure leaves the prompt unchanged"
                if args.candidate_mode == "group-feature-plan-before-generation"
                else
                "the maximum number of situation-ranked few-shot examples entering the prompt: baseline 3, candidate 5"
                if args.candidate_mode == "few-shot-count-5"
                else "few-shot retrieval pool only: baseline 80,530 rows, candidate 82,093 rows; ranking, count, rendering, prompt, and model are unchanged"
                if args.candidate_mode == "expanded-pool"
                else "few-shot presentation only: old short context without names versus the same IDs and order with speaker names and continuous context"
                if args.candidate_mode == "named-context-render"
                else "few-shot ordering only: expected way of responding first, existing relevance score second"
                if args.candidate_mode == "response-move-before-relevance"
                else "few-shot ordering only: dense semantic topic-similarity weight 4.0 versus 8.0"
                if args.candidate_mode == "semantic-topic-weight-8"
                else "whether the reply may select a numeric factual option that is not directly supported by current context, trusted Wiki, or tools"
                if args.candidate_mode == "unresolved-fact-choice"
                else "whether an already-complete fact, arrangement, or conclusion is answered with one context-grounded action instead of being restated"
                if args.candidate_mode == "no-restatement-action"
                else "few-shot ordering only: whether missing source, role, basis, or required details prefer verified-human examples that verify or ask"
                if args.candidate_mode == "information-gap-response-move"
                else "whether an explicitly stated current stance is preserved instead of being weakened, reversed, or dropped"
                if args.candidate_mode == "current-stance-continuity"
                else "whether unsupported polishing into antithesis, idiom, metaphor, slogan, internet template, or a complete punchline is rolled back to the plain response"
                if args.candidate_mode == "plain-response-before-styling"
                else "whether an underspecified situation retrieves真人 examples that ask for the missing detail first"
                if args.candidate_mode == "clarification"
                else (
                    "whether recent self replies are continuity context instead of forbidden-only material"
                    if args.candidate_mode == "active-thread"
                    else (
                        "whether human examples match the same setup, relationship, and expected response instead of merely sharing words"
                        if args.candidate_mode == "precise-situation"
                        else (
                            "whether the first practical response handles the latest request addressed to the speaker"
                            if args.candidate_mode == "latest-request-first"
                            else (
                                "whether equally suitable examples from the current group or private chat are selected first, with cross-chat fallback"
                                if args.candidate_mode == "same-chat-after-situation"
                                else (
                                    "whether situation and reply-move matching use the latest message instead of older messages in the context"
                                    if args.candidate_mode == "latest-turn-situation"
                                    else (
                                        "whether latest-message situation matching plus current-chat priority improves example choice without letting sparse turns inherit concrete topics"
                                        if args.candidate_mode == "latest-turn-same-chat-sparse"
                                        else "whether the reply follows the matched human example's conversational action instead of upgrading it into a joke, explanation, summary, or advice"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "generation_scope": "offline causal A/B with tools and self-refine disabled equally for both conditions",
            "generation_models": {
                condition: {
                    relationship: _generation_model_for_condition(
                        args, condition, relationship, current_c0
                    )
                    for relationship in ("private", "group")
                }
                for condition in ("baseline", "candidate")
            },
            "generation_providers": {
                "baseline": "volcengine" if args.generation_provider == "codex" else args.generation_provider,
                "candidate": args.generation_provider,
            },
            "few_shot_count": {
                "baseline": _few_shot_count_for_condition(args, "baseline"),
                "candidate": _few_shot_count_for_condition(args, "candidate"),
            },
            "feature_extraction_model": (
                {"baseline": None, "candidate": args.feature_extraction_model}
                if args.candidate_mode == "group-feature-plan-before-generation"
                else None
            ),
            "named_context_library": (
                str(args.named_context_few_shot.resolve().relative_to(ROOT.resolve()))
                if args.candidate_mode in {"named-context-render", "group-context-window-10"}
                else None
            ),
            "long_named_context_library": (
                str(args.long_named_context_few_shot.resolve().relative_to(ROOT.resolve()))
                if args.candidate_mode == "group-context-window-10"
                else None
            ),
            "skill_snapshots": skill_snapshots,
        },
        "fixed_240_excluded": args.dataset == "development80",
        "per_case_few_shot_overlap_excluded": True,
        "distribution": _distribution(selected),
        "case_ids": list(selected_by_id),
    }
    if frozen_bundle is not None:
        manifest["frozen_formal_baseline"] = {
            "imported_or_reused": len(selected),
            "baseline_generation": 0,
            "provenance": frozen_bundle["provenance"],
        }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("case_ids") != manifest["case_ids"]
            or previous.get("conditions") != manifest["conditions"]
            or previous.get("repetitions_per_condition", 1) != args.repetitions
            or previous.get("current_c0_baseline_id") != baseline_id
            or previous.get("frozen_formal_baseline")
            != manifest.get("frozen_formal_baseline")
        ):
            raise ValueError("existing output manifest does not match this run")
        manifest = previous
    else:
        _write_private(manifest_path, manifest)
    run_case_ids = manifest["case_ids"][args.shard_index::args.shard_count]
    shard_suffix = "" if args.shard_count == 1 else f".shard-{args.shard_index}-of-{args.shard_count}"

    _load_env()
    if args.dataset == "formal240":
        persona_path = args.output / f"persona_force_reply{shard_suffix}.md"
        _write_private(persona_path, DEFAULT_DEVELOPMENT80_FORCE_REPLY_PERSONA.read_text(encoding="utf-8"))
    else:
        persona_path = _force_reply_persona(args.output, shard_suffix)
    os.environ["PERSONA_PATH"] = str(persona_path.resolve())
    os.environ["ENABLE_PERSONA_FEW_SHOTS"] = "1"
    os.environ["PERSONA_FEW_SHOT_COUNT"] = str(max(
        _few_shot_count_for_condition(args, "baseline"),
        _few_shot_count_for_condition(args, "candidate"),
    ))
    os.environ["PERSONA_FEW_SHOT_MAX_CHARS"] = str(args.few_shot_max_chars)
    os.environ["ENABLE_REACT_TOOLS"] = "0"
    os.environ["ENABLE_SELF_REFINE"] = "0"
    from src.memory import MemoryEngine
    from src.utils.qwen_client import QwenClient

    private_client, group_client = create_current_c0_clients(config=current_c0)
    clients = {
        "baseline": {"private": private_client, "group": group_client},
        "candidate": {"private": private_client, "group": group_client},
    }
    if args.candidate_mode == "generation-model":
        if args.generation_provider == "volcengine":
            api_key, base_url = load_volcengine_credentials(DEFAULT_VOLCENGINE_CONFIG)
            candidate_client = VolcengineAnthropicClient(api_key, base_url, args.generation_model)
        elif args.generation_provider == "codex":
            candidate_client = CodexExecClient(
                args.generation_model,
                reasoning_effort=args.codex_reasoning_effort,
            )
        else:
            raise ValueError("generation-model comparison requires volcengine or codex")
        if args.group_only_candidate:
            clients["candidate"] = {"private": private_client, "group": candidate_client}
        else:
            clients["candidate"] = {"private": candidate_client, "group": candidate_client}
    elif args.generation_provider != "volcengine":
        raise ValueError("feature comparisons must use the adopted current C0 provider")
    group_feature_extraction_client = None
    if args.candidate_mode == "group-feature-plan-before-generation":
        api_key, base_url = load_volcengine_credentials(DEFAULT_VOLCENGINE_CONFIG)
        group_feature_extraction_client = FeatureExtractionOnlyClient(
            VolcengineAnthropicClient(api_key, base_url, args.feature_extraction_model),
            clients["candidate"]["group"],
        )
    memory = MemoryEngine(llm_client=clients["baseline"]["private"])
    group_candidate_contract_by_case: dict[str, bool | None] = {}
    if args.candidate_mode in {
        "group-target-action-contract",
        "group-public-question-specific-followup",
        "group-instant-self-correction",
        "group-visible-self-stance-continuity",
        "group-pending-self-action",
        "group-pattern-participation",
        "group-current-event-self-role-continuity",
        "group-explicit-self-stance-action",
        "group-answer-authority-rewrite",
    }:
        for case_id in run_case_ids:
            row = selected_by_id[case_id]
            if row.get("relationship") != "group":
                continue
            source = _candidate(row)
            payload = _payload_for_case(row, args.dataset, exports, source_payload_cache)
            if payload is None:
                group_candidate_contract_by_case[case_id] = None
            elif args.candidate_mode == "group-public-question-specific-followup":
                group_candidate_contract_by_case[case_id] = (
                    _group_public_question_specific_followup_contract_applies(
                        source, payload
                    )
                )
            elif args.candidate_mode == "group-instant-self-correction":
                group_candidate_contract_by_case[case_id] = (
                    _group_instant_self_correction_contract_applies(source, payload)
                )
            elif args.candidate_mode == "group-visible-self-stance-continuity":
                group_candidate_contract_by_case[case_id] = (
                    _group_visible_self_stance_continuity_contract_applies(
                        source, payload
                    )
                )
            elif args.candidate_mode == "group-pending-self-action":
                group_candidate_contract_by_case[case_id] = (
                    _group_pending_self_action_contract_applies(source, payload)
                )
            elif args.candidate_mode == "group-pattern-participation":
                group_candidate_contract_by_case[case_id] = (
                    _group_pattern_participation_contract_applies(source, payload)
                )
            elif args.candidate_mode == "group-current-event-self-role-continuity":
                group_candidate_contract_by_case[case_id] = (
                    _group_current_event_self_role_continuity_contract_applies(
                        source, payload
                    )
                )
            elif args.candidate_mode == "group-explicit-self-stance-action":
                group_candidate_contract_by_case[case_id] = (
                    _group_explicit_self_stance_action_contract_applies(
                        source, payload
                    )
                )
            else:
                group_candidate_contract_by_case[case_id] = (
                    _group_target_action_contract_applies(source, payload)
                )
    current_baseline_mode = args.candidate_mode in CURRENT_BASELINE_MODES
    baseline_retriever = PersonaFewShotRetriever(
        args.few_shot,
        render_path=args.named_context_few_shot if current_baseline_mode else None,
        enable_clarification=False,
        enable_precise_situation=current_baseline_mode or args.candidate_mode in {
            "response-action",
            "same-chat-after-situation",
        },
        enable_information_gap_response_move=_information_gap_response_move_for_condition(
            args.candidate_mode, "baseline"
        ),
        prefer_latest_turn_form=args.candidate_mode in {
            "group-participant-structure",
            "group-context-window-10",
        },
    )
    paired_state: PairedNamedContextSelection | PairedCountSelection | None = None
    if args.candidate_mode == "named-context-render":
        paired_state = PairedNamedContextSelection(
            baseline_retriever,
            _load_rows_by_id(args.named_context_few_shot),
            args.few_shot_max_chars,
        )
        baseline = PairedNamedContextRetriever(paired_state, "baseline")
        candidate = PairedNamedContextRetriever(paired_state, "candidate")
    elif args.candidate_mode in PAIRED_COUNT_MODES:
        paired_state = PairedCountSelection(
            baseline_retriever,
            max(
                _few_shot_count_for_condition(args, "baseline"),
                _few_shot_count_for_condition(args, "candidate"),
            ),
        )
        baseline = PairedCountRetriever(paired_state)
        candidate = PairedCountRetriever(paired_state)
    else:
        baseline = LeaveOneOutRetriever(baseline_retriever, args.few_shot_max_chars)
        candidate = LeaveOneOutRetriever(PersonaFewShotRetriever(
            _few_shot_path_for_condition(args, "candidate"),
            render_path=args.named_context_few_shot if current_baseline_mode else None,
            enable_clarification=args.candidate_mode == "clarification",
            enable_precise_situation=args.candidate_mode in {
                "precise-situation",
                "response-action",
                "same-chat-after-situation",
                "latest-turn-same-chat-sparse",
                "semantic-topic-weight-8",
                "unresolved-fact-choice",
                "expanded-pool",
                "current-stance-continuity",
                "information-gap-response-move",
                "plain-response-before-styling",
                "latest-turn-form",
                "group-participant-structure",
                "group-context-window-10",
                "group-speaker-target-attribution",
                "group-explicit-third-party-guard",
                "group-self-context-freshness-boundary",
                "group-adjacent-self-response",
            },
            prioritize_latest_turn=args.candidate_mode in {
                "latest-turn-situation",
                "latest-turn-same-chat-sparse",
            },
            prefer_same_chat_after_situation=args.candidate_mode in {
                "same-chat-after-situation",
                "latest-turn-same-chat-sparse",
            },
            avoid_concrete_examples_for_sparse_query=(
                args.candidate_mode == "latest-turn-same-chat-sparse"
            ),
            prefer_response_move_before_relevance=(
                args.candidate_mode == "response-move-before-relevance"
            ),
            increase_semantic_topic_weight=(
                args.candidate_mode == "semantic-topic-weight-8"
            ),
            prefer_latest_turn_form=(
                args.candidate_mode == "latest-turn-form"
            ),
            prefer_group_participant_structure=(
                args.candidate_mode == "group-participant-structure"
            ),
            enable_information_gap_response_move=_information_gap_response_move_for_condition(
                args.candidate_mode, "candidate"
            ),
        ), args.few_shot_max_chars)
    retrievers: dict[str, dict[str, Any]] = {
        "baseline": {"group": baseline, "private": baseline},
        "candidate": {"group": candidate, "private": candidate},
    }
    if args.private_only_candidate:
        retrievers["candidate"]["group"] = baseline
    if args.candidate_mode == "group-participant-structure":
        group_baseline = LeaveOneOutRetriever(PersonaFewShotRetriever(
            args.few_shot,
            render_path=args.named_context_few_shot,
            enable_clarification=False,
            enable_precise_situation=True,
            enable_information_gap_response_move=_information_gap_response_move_for_condition(
                args.candidate_mode, "baseline"
            ),
            prefer_latest_turn_form=False,
            prefer_group_participant_structure=False,
        ), args.few_shot_max_chars)
        group_candidate = LeaveOneOutRetriever(PersonaFewShotRetriever(
            _few_shot_path_for_condition(args, "candidate"),
            render_path=args.named_context_few_shot,
            enable_clarification=False,
            enable_precise_situation=True,
            enable_information_gap_response_move=_information_gap_response_move_for_condition(
                args.candidate_mode, "candidate"
            ),
            prefer_latest_turn_form=False,
            prefer_group_participant_structure=True,
        ), args.few_shot_max_chars)
        retrievers["baseline"] = {"group": group_baseline, "private": baseline}
        retrievers["candidate"] = {"group": group_candidate, "private": baseline}
    elif args.candidate_mode == "group-context-window-10":
        group_ranker = PersonaFewShotRetriever(
            args.few_shot,
            enable_clarification=False,
            enable_precise_situation=True,
            enable_information_gap_response_move=_information_gap_response_move_for_condition(
                args.candidate_mode, "baseline"
            ),
            prefer_latest_turn_form=False,
            prefer_group_participant_structure=False,
        )
        paired_state = PairedNamedContextSelection(
            group_ranker,
            _load_rows_by_id(args.long_named_context_few_shot),
            args.few_shot_max_chars,
            baseline_by_id=_load_rows_by_id(args.named_context_few_shot),
        )
        group_baseline = PairedNamedContextRetriever(paired_state, "baseline")
        group_candidate = PairedNamedContextRetriever(paired_state, "candidate")
        retrievers["baseline"] = {"group": group_baseline, "private": baseline}
        retrievers["candidate"] = {"group": group_candidate, "private": baseline}
    generation_path = args.output / f"generation_checkpoint{shard_suffix}.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.exists()
        else {"current_c0_baseline_id": baseline_id, "entries": {}}
    )
    if generation.get("current_c0_baseline_id") != baseline_id:
        raise ValueError("generation checkpoint is not bound to the adopted current C0 baseline")
    entries = generation.setdefault("entries", {})
    imported_baseline_count = 0
    if frozen_bundle is not None:
        imported_baseline_count = _import_frozen_formal_baseline_entries(
            entries,
            run_case_ids,
            frozen_bundle,
        )
    _drop_contaminated_generation_entries(entries, per_case_overlap_ids)
    _seed_scorable_entries(
        entries,
        run_case_ids,
        args.seed_generation_checkpoint,
        args.repetitions,
        conditions=("baseline",) if args.seed_baseline_only else ("baseline", "candidate"),
        forbidden_by_case=per_case_overlap_ids,
    )
    _seed_scorable_entries(
        entries,
        run_case_ids,
        args.seed_candidate_generation_checkpoint,
        args.repetitions,
        conditions=("candidate",),
        forbidden_by_case=per_case_overlap_ids,
    )
    _seed_baseline_from_candidate_entries(
        entries,
        run_case_ids,
        args.seed_baseline_from_candidate_generation_checkpoint,
        args.repetitions,
        forbidden_by_case=per_case_overlap_ids,
    )
    _seed_split_baseline_entries(
        entries,
        run_case_ids,
        selected_by_id,
        args.seed_split_baseline_generation_checkpoint,
        forbidden_by_case=per_case_overlap_ids,
    )
    _seed_private_candidate_entries(
        entries,
        run_case_ids,
        selected_by_id,
        args.seed_private_candidate_generation_checkpoint,
        forbidden_by_case=per_case_overlap_ids,
    )
    for key in list(entries):
        if entries[key].get("status") in {"generation_failed", "generation_error"}:
            del entries[key]
    generation["generation_provider"] = args.generation_provider
    generation["generation_models"] = {
        condition: {
            relationship: getattr(condition_client, "model", type(condition_client).__name__)
            for relationship, condition_client in relationship_clients.items()
        }
        for condition, relationship_clients in clients.items()
    }
    generation["generation_providers"] = {
        "baseline": "volcengine" if args.generation_provider == "codex" else args.generation_provider,
        "candidate": args.generation_provider,
    }
    generation["generation_endpoint_hosts"] = {
        "baseline": "ark.cn-beijing.volces.com" if args.generation_provider in {"volcengine", "codex"} else "api.deepseek.com",
        "candidate": "codex-cli" if args.generation_provider == "codex" else (
            "ark.cn-beijing.volces.com" if args.generation_provider == "volcengine" else "api.deepseek.com"
        ),
    }
    if frozen_bundle is not None:
        generation["baseline_imported_or_reused"] = len(run_case_ids)
        generation["baseline_imported_this_write"] = imported_baseline_count
        generation["baseline_generation"] = 0
        generation["frozen_formal_baseline_provenance"] = frozen_bundle["provenance"]
    _write_private(generation_path, generation)
    try:
        if args.phase in {"generate", "all"}:
            for case_id in run_case_ids:
                row = selected_by_id[case_id]
                if isinstance(paired_state, PairedNamedContextSelection):
                    paired_state.reset(row)
                elif isinstance(paired_state, PairedCountSelection):
                    paired_state.reset()
                source = _candidate(row)
                payload = _payload_for_case(
                    row,
                    args.dataset,
                    exports,
                    source_payload_cache,
                )
                for condition, repetition in _generation_schedule(args.repetitions):
                    relation_key = "group" if source.relation == "group" else "private"
                    retriever = retrievers[condition][relation_key]
                    condition_few_shot_count = _few_shot_count_for_condition(args, condition)
                    key = _entry_key(case_id, condition, repetition, args.repetitions)
                    if frozen_bundle is not None and condition == "baseline":
                        if entries.get(key, {}).get("protected_frozen_baseline"):
                            continue
                        raise RuntimeError(
                            "protected formal baseline was not imported; refusing generation: "
                            + key
                        )
                    if _reuse_private_baseline_for_group_only_change(
                        entries,
                        case_id,
                        row,
                        condition,
                        repetition,
                        args.repetitions,
                        args.candidate_mode,
                        group_only_candidate=args.group_only_candidate,
                    ):
                        _write_private(generation_path, generation)
                        continue
                    if _reuse_group_baseline_for_nontriggering_target_action_contract(
                        entries,
                        case_id,
                        row,
                        condition,
                        repetition,
                        args.repetitions,
                        args.candidate_mode,
                        group_candidate_contract_by_case.get(case_id),
                    ):
                        _write_private(generation_path, generation)
                        continue
                    if (
                        args.private_only_candidate
                        and relation_key == "group"
                        and condition == "candidate"
                    ):
                        baseline_entry = entries.get(
                            _entry_key(case_id, "baseline", repetition, args.repetitions)
                        ) or {}
                        if baseline_entry.get("status") == "scorable":
                            entries[key] = baseline_entry
                            _write_private(generation_path, generation)
                            continue
                    if key in entries and entries[key].get("status") == "scorable":
                        if (
                            condition == "baseline"
                            and isinstance(paired_state, PairedCountSelection)
                            and paired_state.rows is None
                        ):
                            trace = ((entries[key].get("result") or {}).get("trace") or {})
                            paired_trace = trace.get("paired_few_shot") or {}
                            saved_ids = list(
                                paired_trace.get("ranked_ids")
                                or trace.get("few_shot_ids")
                                or []
                            )
                            paired_state.restore(saved_ids, few_shot_rows_by_id)
                        continue
                    if payload is None:
                        entries[key] = {"status": "missing_export"}
                        _write_private(generation_path, generation)
                        continue
                    retriever.exclude_ids = per_case_overlap_ids[case_id]
                    retriever.current_chat_id = source.chat_id
                    try:
                        result = _run_c0(
                            source,
                            payload,
                            clients[condition][relation_key],
                            memory,
                            generation_timeout_seconds=_generation_timeout_for_condition(
                                args, condition, relation_key, current_c0
                            ),
                            persona_few_shot_retriever=retriever,
                            persona_few_shot_count=condition_few_shot_count,
                            persona_few_shot_max_chars=args.few_shot_max_chars,
                            force_reply=True,
                            enable_active_thread_continuation=(
                                args.candidate_mode == "active-thread" and condition == "candidate"
                            ),
                            enable_response_action_grounding=(
                                current_baseline_mode or args.candidate_mode in {
                                    "response-action",
                                    "latest-request-first",
                                    "same-chat-after-situation",
                                }
                            ),
                            enable_latest_request_first=(
                                current_baseline_mode
                                or (args.candidate_mode == "latest-request-first" and condition == "candidate")
                            ),
                            enable_unresolved_fact_choice_guard=(
                                args.candidate_mode == "unresolved-fact-choice"
                                and condition == "candidate"
                            ),
                            enable_no_restatement_action=(
                                args.candidate_mode == "no-restatement-action"
                                and condition == "candidate"
                            ),
                            enable_plain_response_before_styling=(
                                args.candidate_mode == "plain-response-before-styling"
                                and condition == "candidate"
                            ),
                            enable_current_stance_continuity=(
                                args.candidate_mode == "current-stance-continuity"
                                and condition == "candidate"
                            ),
                            enable_group_speaker_target_attribution=(
                                args.candidate_mode == "group-speaker-target-attribution"
                                and condition == "candidate"
                            ),
                            enable_group_explicit_third_party_guard=(
                                args.candidate_mode == "group-explicit-third-party-guard"
                                and condition == "candidate"
                            ),
                            enable_group_unified_history=(
                                args.candidate_mode == "group-unified-history"
                                and condition == "candidate"
                            ),
                            enable_group_unified_relevant_self=(
                                args.candidate_mode == "group-unified-relevant-self"
                                and condition == "candidate"
                            ),
                            enable_group_relevant_self_evidence=(
                                args.candidate_mode == "group-adjacent-self-response"
                                or (
                                    args.candidate_mode in {
                                        "group-relevant-self-evidence",
                                        "group-self-context-freshness-boundary",
                                    }
                                    and condition == "candidate"
                                )
                            ),
                            enable_group_self_context_freshness_boundary=(
                                args.candidate_mode == "group-adjacent-self-response"
                                or (
                                    args.candidate_mode == "group-self-context-freshness-boundary"
                                    and condition == "candidate"
                                )
                            ),
                            enable_group_adjacent_self_response=(
                                args.candidate_mode == "group-adjacent-self-response"
                                and condition == "candidate"
                            ),
                            enable_group_target_action_contract=(
                                args.candidate_mode in {
                                    "group-target-action-contract",
                                    "group-answer-authority-rewrite",
                                }
                                and condition == "candidate"
                            ),
                            enable_group_public_question_specific_followup=(
                                args.candidate_mode
                                == "group-public-question-specific-followup"
                                and condition == "candidate"
                            ),
                            enable_group_instant_self_correction=(
                                args.candidate_mode == "group-instant-self-correction"
                                and condition == "candidate"
                            ),
                            enable_group_visible_self_stance_continuity=(
                                args.candidate_mode
                                == "group-visible-self-stance-continuity"
                                and condition == "candidate"
                            ),
                            enable_group_pending_self_action=(
                                args.candidate_mode == "group-pending-self-action"
                                and condition == "candidate"
                            ),
                            enable_group_pattern_participation=(
                                args.candidate_mode == "group-pattern-participation"
                                and condition == "candidate"
                            ),
                            enable_group_current_event_self_role_continuity=(
                                args.candidate_mode
                                == "group-current-event-self-role-continuity"
                                and condition == "candidate"
                            ),
                            enable_group_explicit_self_stance_action=(
                                args.candidate_mode
                                == "group-explicit-self-stance-action"
                                and condition == "candidate"
                            ),
                            enable_group_answer_authority_rewrite=(
                                args.candidate_mode == "group-answer-authority-rewrite"
                                and condition == "candidate"
                            ),
                            enable_group_feature_plan_before_generation=(
                                args.candidate_mode == "group-feature-plan-before-generation"
                                and condition == "candidate"
                            ),
                            group_feature_plan_llm_client=(
                                group_feature_extraction_client
                                if _feature_plan_model_for_condition(
                                    args, condition, relation_key
                                )
                                is not None
                                else None
                            ),
                            group_prefer_participant_structure=(
                                args.candidate_mode == "group-participant-structure"
                                and condition == "candidate"
                            ),
                            enable_precise_situation_matching=(
                                current_baseline_mode
                                or args.candidate_mode == "response-action"
                                or (args.candidate_mode == "precise-situation" and condition == "candidate")
                                or args.candidate_mode == "same-chat-after-situation"
                            ),
                            compact_verified_turn=True,
                            skill_content_overrides={
                                name: snapshot["content"]
                                for name, snapshot in skill_snapshots[condition].items()
                            } or None,
                        )
                        if args.dataset == "formal240":
                            _require_actual_few_shots(
                                result, condition_few_shot_count, case_id, condition
                            )
                        if hasattr(retriever, "trace"):
                            result.setdefault("trace", {})["paired_few_shot"] = retriever.trace()
                        status = (
                            "scorable"
                            if _has_nonempty_reply(result) and not result.get("generation_failed")
                            else "generation_failed"
                        )
                        entries[key] = {"status": status, "result": result}
                    except FewShotInjectionMismatch:
                        raise
                    except Exception as exc:
                        entries[key] = {"status": "generation_error", "error": f"{type(exc).__name__}: {exc}"}
                    _write_private(generation_path, generation)
    finally:
        memory.shutdown()

    _assert_generation_entries_isolated(entries, per_case_overlap_ids)

    judge_path = args.output / f"judge_checkpoint{shard_suffix}.json"
    judge = (
        json.loads(judge_path.read_text(encoding="utf-8"))
        if judge_path.exists()
        else {"current_c0_baseline_id": baseline_id, "entries": {}}
    )
    if judge.get("current_c0_baseline_id") != baseline_id:
        raise ValueError("judge checkpoint is not bound to the adopted current C0 baseline")
    judge_entries = judge.setdefault("entries", {})
    for key in list(judge_entries):
        if (entries.get(key) or {}).get("status") != "scorable":
            del judge_entries[key]
    _seed_judge_entries(
        judge_entries,
        run_case_ids,
        args.seed_judge_checkpoint,
        args.repetitions,
        conditions=("baseline",) if args.seed_baseline_only else ("baseline", "candidate"),
    )
    _seed_baseline_from_candidate_judge_entries(
        judge_entries,
        run_case_ids,
        args.seed_baseline_from_candidate_judge_checkpoint,
        args.repetitions,
    )
    _seed_split_baseline_entries(
        judge_entries,
        run_case_ids,
        selected_by_id,
        args.seed_split_baseline_judge_checkpoint,
        judge=True,
    )
    _write_private(judge_path, judge)
    if args.phase in {"judge", "all"}:
        profile = load_profile(PROFILE)
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        for case_id in run_case_ids:
            row = selected_by_id[case_id]
            relationship = "group" if row.get("relationship") == "group" else "private"
            if args.judge_relationship != "all" and relationship != args.judge_relationship:
                continue
            for condition, repetition in _generation_schedule(args.repetitions):
                key = _entry_key(case_id, condition, repetition, args.repetitions)
                if _reuse_private_baseline_for_group_only_change(
                    judge_entries,
                    case_id,
                    row,
                    condition,
                    repetition,
                    args.repetitions,
                    args.candidate_mode,
                    group_only_candidate=args.group_only_candidate,
                    judge=True,
                ):
                    _write_private(judge_path, judge)
                    continue
                if _reuse_group_baseline_for_nontriggering_target_action_contract(
                    judge_entries,
                    case_id,
                    row,
                    condition,
                    repetition,
                    args.repetitions,
                    args.candidate_mode,
                    group_candidate_contract_by_case.get(case_id),
                    judge=True,
                ):
                    _write_private(judge_path, judge)
                    continue
                if _reuse_group_baseline_judge_for_private_only_candidate(
                    judge_entries,
                    case_id,
                    row,
                    condition,
                    repetition,
                    args.repetitions,
                    args.private_only_candidate,
                ):
                    _write_private(judge_path, judge)
                    continue
                entry = entries.get(key) or {}
                if entry.get("status") != "scorable" or (key in judge_entries and not judge_entries[key].get("error")):
                    continue
                result = entry["result"]
                replay = {
                    "case_id": key,
                    "relationship": "group" if row.get("relationship") == "group" else "private",
                    "context_original": result.get("context_original") or row.get("context") or [],
                    "human_reply_original": row.get("reply") or [],
                    "ai_reply_original": result["replies"],
                    "c0_trace": {"generation_trace": result.get("trace") or {}},
                }
                blind, human_option = _blind_case(replay)
                try:
                    case_reference, removed_reference_rows = _reference_without_case_leaks(
                        reference, blind
                    )
                    raw, audit, elapsed_ms, tool_calls = _run_codex(
                        build_messages(blind, case_reference, profile),
                        model=args.judge_model,
                        reasoning_effort=args.judge_reasoning_effort,
                        timeout_seconds=args.judge_timeout_seconds,
                    )
                    parsed = _parse_api_judge_response(raw)
                    judge_entries[key] = {
                        **parsed,
                        "correct": parsed.get("human_option") == human_option,
                        "actual_human_option": human_option,
                        "elapsed_ms": elapsed_ms,
                        "tool_calls": tool_calls,
                        "audit": audit,
                        "removed_reference_rows": removed_reference_rows,
                    }
                except Exception as exc:
                    judge_entries[key] = {"error": f"{type(exc).__name__}: {exc}"}
                _write_private(judge_path, judge)

    cases = []
    for case_id in run_case_ids:
        row = selected_by_id[case_id]
        conditions = {}
        for condition in ("baseline", "candidate"):
            runs = []
            for repetition in range(1, args.repetitions + 1):
                key = _entry_key(case_id, condition, repetition, args.repetitions)
                entry = entries.get(key) or {}
                result = entry.get("result") or {}
                trace = result.get("trace") or {}
                runs.append({
                    "repetition": repetition,
                    "status": entry.get("status", "not_run"),
                    "reply": result.get("replies") or [],
                    "duration_seconds": result.get("duration_seconds"),
                    "few_shot_ids": trace.get("few_shot_ids") or [],
                    "few_shot_content": trace.get("few_shot_content") or "",
                    "paired_few_shot": trace.get("paired_few_shot") or {},
                    "judge": judge_entries.get(key),
                    "error": entry.get("error") or (judge_entries.get(key) or {}).get("error"),
                })
            conditions[condition] = {"runs": runs}
        cases.append({
            "case_id": case_id,
            "relationship": "group" if row.get("relationship") == "group" else "private",
            "context": row.get("context") or [],
            "human_reply": row.get("reply") or [],
            "conditions": conditions,
        })
    result = {"schema_version": 1, "created_at": datetime.now().astimezone().isoformat(), "manifest": manifest, "cases": cases}
    _write_private(args.output / f"results{shard_suffix}.json", result)
    print(json.dumps({
        "distribution": manifest["distribution"],
        "generation_status": dict(Counter(value.get("status", "not_run") for value in entries.values())),
        "judge_completed": sum(not value.get("error") for value in judge_entries.values()),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("development80", "formal240"), default="development80")
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--exposed", type=Path, default=DEFAULT_EXPOSED)
    parser.add_argument("--previous-ab", type=Path, action="append")
    parser.add_argument("--fixed-test", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument(
        "--frozen-formal-baseline-run",
        type=Path,
        help="approved formal baseline run to import; required by the protected target mode",
    )
    parser.add_argument(
        "--frozen-formal-source-manifest",
        type=Path,
        action="append",
        default=None,
        help="source formal manifest used to recover frozen cases; may repeat",
    )
    parser.add_argument("--few-shot", type=Path, default=DEFAULT_FEW_SHOT)
    parser.add_argument("--candidate-few-shot", type=Path, default=DEFAULT_CANDIDATE_FEW_SHOT)
    parser.add_argument("--named-context-few-shot", type=Path, default=DEFAULT_NAMED_CONTEXT_FEW_SHOT)
    parser.add_argument("--long-named-context-few-shot", type=Path, default=DEFAULT_LONG_NAMED_CONTEXT_FEW_SHOT)
    parser.add_argument("--few-shot-max-chars", type=int, default=2500)
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-ids-from-results", type=Path)
    parser.add_argument("--allow-exposed-case-recheck", action="store_true")
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument(
        "--baseline-skill-snapshot",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="frozen baseline skill body; only valid with --candidate-mode skill-snapshot",
    )
    parser.add_argument(
        "--candidate-skill-snapshot",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="frozen candidate skill body; only valid with --candidate-mode skill-snapshot",
    )
    parser.add_argument("--repetitions", type=int, choices=(1, 3, 6, 12), default=1)
    parser.add_argument("--generation-timeout-seconds", type=float)
    parser.add_argument("--generation-provider", choices=("volcengine", "official", "codex"), default="volcengine")
    parser.add_argument("--generation-model", default=VOLCENGINE_DSV4F_MODEL)
    parser.add_argument(
        "--feature-extraction-model",
        "--feature-plan-model",
        dest="feature_extraction_model",
        default=VOLCENGINE_DSV4F_MODEL,
        help="auxiliary model for group feature extraction only; short planning stays on current C0",
    )
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--seed-generation-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-candidate-generation-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-baseline-from-candidate-generation-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-judge-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-baseline-from-candidate-judge-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-split-baseline-generation-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-private-candidate-generation-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-split-baseline-judge-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--seed-baseline-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--phase", choices=("generate", "judge", "all"), default="all")
    parser.add_argument("--judge-relationship", choices=("all", "group", "private"), default="all")
    parser.add_argument("--judge-model", default="codex-cli-default")
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--judge-timeout-seconds", type=int, default=360)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--private-only-candidate", action="store_true")
    parser.add_argument("--group-only-candidate", action="store_true")
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "clarification",
            "active-thread",
            "response-action",
            "latest-request-first",
            "precise-situation",
            "same-chat-after-situation",
            "latest-turn-situation",
            "latest-turn-same-chat-sparse",
            "named-context-render",
            "few-shot-count-5",
            "response-move-before-relevance",
            "semantic-topic-weight-8",
            "unresolved-fact-choice",
            "no-restatement-action",
            "expanded-pool",
            "latest-turn-form",
            "information-gap-response-move",
            "current-stance-continuity",
            "plain-response-before-styling",
            "group-participant-structure",
            "group-context-window-10",
            "group-speaker-target-attribution",
            "group-explicit-third-party-guard",
            "group-relevant-self-evidence",
            "group-self-context-freshness-boundary",
            "group-adjacent-self-response",
            "group-unified-history",
            "group-unified-relevant-self",
            "group-target-action-contract",
            "group-public-question-specific-followup",
            "group-instant-self-correction",
            "group-visible-self-stance-continuity",
            "group-pending-self-action",
            "group-pattern-participation",
            "group-current-event-self-role-continuity",
            "group-explicit-self-stance-action",
            "group-answer-authority-rewrite",
            "group-feature-plan-before-generation",
            "skill-snapshot",
            "generation-model",
        ),
        default="clarification",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
