#!/usr/bin/env python3
"""Build and score a private human-vs-AI reply discrimination benchmark."""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import difflib
import hashlib
import html
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_persona_few_shots import Candidate, extract_candidates  # noqa: E402
from src.badcase.judge_config import current_judge_baseline_id, load_current_judge  # noqa: E402
from src.models.base import SenderType  # noqa: E402
from src.reply.c0_config import current_c0_baseline_id  # noqa: E402

CUTOFF = 1774972800.0  # 2026-04-01 00:00:00 Asia/Shanghai
PROTECTED_C0_SHA256 = "625b47803fff55a2e18914c5e734fcb2f01ae7c16e5180fdebdc817badd2656b"
PRIVATE_MODE = 0o600
DEFAULT_VOLCENGINE_CONFIG = Path.home() / ".cc-connect/config.toml"
VOLCENGINE_DSV4F_MODEL = "deepseek-v4-flash"
JUDGE_CHAT_SECTIONS = ("history", "unread", "consumed_self_replies")
JUDGE_FACTS = [
    {
        "fact": "本人老家是江苏泰州，属于淮扬菜地区。",
        "source": "user_confirmed_2026-08-17",
    },
]


def load_volcengine_credentials(config_path: Path) -> tuple[str, str]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project = next(row for row in config.get("projects", []) if row.get("name") == "feishu-connect")
    agent = project["agent"]
    provider = next(row for row in agent.get("providers", []) if row.get("name") == agent.get("provider"))
    credential = provider.get("api_key") or provider.get("auth_token")
    base_url = provider.get("base_url")
    if not credential or not base_url:
        raise ValueError("feishu-connect provider is missing its credential or base_url")
    return str(credential), str(base_url).rstrip("/")


class VolcengineAnthropicClient:
    """QwenClient-compatible adapter for Volcengine's Anthropic endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str, session: Any = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.resolved_model = ""
        self.last_thinking = ""
        self.last_request_body: dict[str, Any] = {}
        self.last_response_payload: dict[str, Any] = {}
        self.last_status_code: int | None = None
        self.last_elapsed_ms: int | None = None
        if session is None:
            session = requests.Session()
            session.trust_env = False
        self.session = session

    @staticmethod
    def _tool_definitions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        converted = []
        for row in tools or []:
            function = row.get("function", {})
            converted.append({
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            })
        return converted

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        def append(role: str, blocks: list[dict[str, Any]]) -> None:
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        for row in messages:
            role = row.get("role")
            content = row.get("content") or ""
            if role == "system":
                system_parts.append(str(content))
                continue
            if role == "tool":
                append("user", [{
                    "type": "tool_result",
                    "tool_use_id": row.get("tool_call_id", ""),
                    "content": str(content),
                }])
                continue

            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            if role == "assistant":
                for tool_call in row.get("tool_calls") or []:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments") or "{}"
                    try:
                        tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        tool_input = {"raw": arguments}
                    blocks.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": tool_input,
                    })
            append("assistant" if role == "assistant" else "user", blocks or [{"type": "text", "text": ""}])
        return "\n\n".join(system_parts), converted

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
    ) -> Any:
        del user_id, response_format
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message or ""})
        system, anthropic_messages = self._messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1000,
            "temperature": 0.7 if temperature is None else temperature,
        }
        if system:
            body["system"] = system
        converted_tools = self._tool_definitions(tools)
        if converted_tools:
            body["tools"] = converted_tools
        attempts = max(1, (max_retries or 0) + 1)
        try:
            response = None
            started = time.monotonic()
            self.last_request_body = body
            for attempt in range(attempts):
                response = self.session.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "content-type": "application/json",
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json=body,
                    timeout=timeout or 500,
                )
                if response.status_code < 500 or attempt + 1 >= attempts:
                    break
            assert response is not None
            self.last_elapsed_ms = round((time.monotonic() - started) * 1000)
            self.last_status_code = response.status_code
            response.raise_for_status()
            payload = response.json()
            self.last_response_payload = payload
            self.resolved_model = str(payload.get("model") or "")
            text_parts = []
            thinking_parts = []
            tool_calls = []
            for block in payload.get("content") or []:
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") == "thinking":
                    thinking_parts.append(str(block.get("thinking") or ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(SimpleNamespace(
                        id=str(block.get("id") or ""),
                        type="function",
                        function=SimpleNamespace(
                            name=str(block.get("name") or ""),
                            arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                        ),
                    ))
            self.last_thinking = "\n".join(thinking_parts)
            text = "\n".join(part for part in text_parts if part)
            if tool_calls:
                return SimpleNamespace(
                    content=text,
                    tool_calls=tool_calls,
                    reasoning_content=self.last_thinking,
                )
            return text
        except Exception:
            if raise_on_error:
                raise
            return ""


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


def official_judge_metadata(config: dict[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    correction = config["correction"]
    sidecars = config["metadata_sidecars"]
    return {
        "official_judge_baseline_id": current_judge_baseline_id(config["config_path"]),
        "current_c0_baseline_id": current_c0_baseline_id(),
        "prediction_type": config["prediction_type"],
        "provider": execution["provider"],
        "model": execution["model"],
        "reasoning_effort": execution["reasoning_effort"],
        "timeout_seconds": execution["timeout_seconds"],
        "codex_cli_version": execution["codex_cli_version"],
        "correction_artifact_path": correction["artifact"]["path"],
        "correction_artifact_sha256": correction["artifact"]["sha256"],
        "correction_model_name": correction["selected_model_name"],
        "correction_threshold": correction["flip_confidence_threshold"],
        "metadata_answers_path": sidecars["answers"]["path"],
        "metadata_answers_sha256": sidecars["answers"]["sha256"],
        "metadata_replay_path": sidecars["replay"]["path"],
        "metadata_replay_sha256": sidecars["replay"]["sha256"],
    }


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("relationship"),
        row.get("chat_id"),
        tuple(row.get("context") or []),
        tuple(row.get("reply") or []),
    )


def _candidate_identity(candidate: Candidate) -> tuple[Any, ...]:
    return (candidate.relation, candidate.chat_id, tuple(candidate.context), tuple(candidate.replies))


def _few_shot_identities(data_dir: Path) -> set[tuple[Any, ...]]:
    identities: set[tuple[Any, ...]] = set()
    for path in sorted(data_dir.glob("few_shot*/persona_examples.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                identities.add(_identity(json.loads(line)))
    return identities


def _chat_names(export_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for path in sorted(export_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        normalized = path.stem
        for prefix in ("私聊_", "群聊_", "曾经的好友_"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        chat_id = f"chat_{hashlib.sha256(normalized.strip().encode('utf-8')).hexdigest()[:10]}"
        display_name = str(payload.get("session", {}).get("displayName") or normalized).strip()
        if display_name:
            names[chat_id] = display_name
    return names


def _case_id(candidate: Candidate) -> str:
    payload = json.dumps(_candidate_identity(candidate), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _nonce(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def _overlap_text(value: Any) -> str:
    return re.sub(r"\s+", "", html.unescape(str(value or ""))).strip()


def _reference_strings(value: Any, path: str = "reference") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "reference_sha256":
                rows.extend(_reference_strings(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_reference_strings(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        rows.append((path, value))
    return rows


def _message_content(value: Any) -> str:
    if isinstance(value, dict):
        return _overlap_text(value.get("content") or value.get("text") or "")
    return _overlap_text(value)


def _xml_messages(value: str) -> list[str]:
    if "<message" not in value:
        return []
    try:
        root = ET.fromstring(f"<root>{value}</root>")
    except ET.ParseError:
        return []
    messages = []
    for element in root.iter("message"):
        text = html.unescape("".join(element.itertext())).strip()
        # Rendered histories prefix each message with ``speaker (time)：``.
        if "）：" in text:
            text = text.split("）：", 1)[1]
        normalized = _overlap_text(text)
        if normalized:
            messages.append(normalized)
    return messages


def _reference_message_sequences(reference: dict[str, Any], relationship: str) -> list[tuple[str, list[str]]]:
    sequences: list[tuple[str, list[str]]] = []
    for section in ("examples", "scenario_few_shots"):
        for index, row in enumerate(reference.get(section) or []):
            relationships = row.get("relationships") or []
            if section == "scenario_few_shots" and relationships and relationship not in relationships:
                continue
            messages: list[str] = []
            for value in row.get("context") or []:
                extracted = _xml_messages(str(value))
                messages.extend(extracted or ([_message_content(value)] if _message_content(value) else []))
            sequences.append((f"reference.{section}[{index}].context", messages))
    return sequences


def reference_leakage_details(
    reference: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    minimum_standalone_chars: int = 8,
) -> list[dict[str, Any]]:
    """Return exact leaks from the reference material actually eligible for each case.

    A standalone answer match is checked only from eight normalized characters to
    avoid treating common one-word replies as leaks.  Short answers are still
    caught when the test context plus answer occurs as one continuous message
    sequence inside a longer reference history.
    """
    leaks: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id") or case.get("id") or "")
        relationship = str(case.get("chat_type") or case.get("relationship") or "")
        context = [
            message
            for value in case.get("context") or case.get("context_original") or []
            if (message := _message_content(value))
        ]
        options = []
        if case.get("human_reply") is not None:
            options.append(("human_reply", case.get("human_reply") or []))
        for key in ("option_A", "option_B"):
            if case.get(key) is not None:
                options.append((key, case.get(key) or []))

        eligible_reference = dict(reference)
        eligible_reference["scenario_few_shots"] = [
            row for row in reference.get("scenario_few_shots") or []
            if not row.get("relationships") or relationship in row.get("relationships", [])
        ]
        strings = _reference_strings(eligible_reference)
        sequences = _reference_message_sequences(reference, relationship)
        seen: set[tuple[str, str, str]] = set()
        for option_name, raw_option in options:
            if isinstance(raw_option, str):
                raw_option = [raw_option]
            option = [message for value in raw_option if (message := _message_content(value))]
            for message in option:
                for path, value in strings:
                    normalized_value = _overlap_text(value)
                    labeled_answer = any(
                        marker in path for marker in (".human_reply", ".option_A", ".option_B")
                    )
                    if (labeled_answer and message == normalized_value) or (
                        len(message) >= minimum_standalone_chars and message in normalized_value
                    ):
                        key = (option_name, "answer_text", path)
                        if key not in seen:
                            seen.add(key)
                            leaks.append({
                                "case_id": case_id,
                                "option": option_name,
                                "kind": "answer_text",
                                "reference_path": path,
                            })
            target = context + option
            if not context or not option:
                continue
            for path, messages in sequences:
                width = len(target)
                if any(messages[start:start + width] == target for start in range(len(messages) - width + 1)):
                    key = (option_name, "context_and_answer_sequence", path)
                    if key not in seen:
                        seen.add(key)
                        leaks.append({
                            "case_id": case_id,
                            "option": option_name,
                            "kind": "context_and_answer_sequence",
                            "reference_path": path,
                        })
    return leaks


def assert_reference_isolated(reference: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    leaks = reference_leakage_details(reference, cases)
    if leaks:
        raise ValueError(
            "reference isolation failed: target reply or context found in reference pack: "
            + json.dumps(leaks, ensure_ascii=False, sort_keys=True)
        )


def build_judge_reference_pack(
    candidates: list[Candidate],
    *,
    excluded_chat_ids: set[str],
    excluded_replies: set[str],
    limit: int = 32,
) -> dict[str, Any]:
    """Freeze human style evidence from chats that cannot appear in the test set."""
    eligible = []
    for candidate in candidates:
        reply = _normalized_reply(candidate.replies)
        if (
            candidate.chat_id in excluded_chat_ids
            or not reply
            or reply in excluded_replies
            or candidate.source_provenance not in {"before_automation_cutoff", "explicit_human_marker"}
        ):
            continue
        eligible.append(candidate)
    eligible.sort(key=lambda row: hashlib.sha256(_case_id(row).encode()).hexdigest())
    selected = []
    used_chats = set()
    for candidate in eligible:
        if candidate.chat_id in used_chats:
            continue
        used_chats.add(candidate.chat_id)
        selected.append({
            "relationship": "group" if candidate.relation == "group" else "private",
            "context": list(candidate.context),
            "human_reply": list(candidate.replies),
        })
        if len(selected) >= limit:
            break
    pack = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "human_style_reference_pack",
        "human_before": "2026-04-01T00:00:00+08:00",
        "isolation": {
            "whole_test_chats_excluded": True,
            "excluded_chat_count": len(excluded_chat_ids),
            "excluded_chat_ids_sha256": _json_sha256(sorted(excluded_chat_ids)),
            "exact_test_replies_excluded": True,
        },
        "facts": JUDGE_FACTS,
        "examples": selected,
    }
    pack["reference_sha256"] = _json_sha256(pack)
    return pack


def validate_reference_isolation(
    reference: dict[str, Any],
    test_candidates: list[Candidate],
) -> None:
    expected_chat_hash = _json_sha256(sorted({row.chat_id for row in test_candidates}))
    isolation = reference.get("isolation") or {}
    if isolation.get("excluded_chat_ids_sha256") != expected_chat_hash:
        raise ValueError("reference isolation failed: excluded test chat hash mismatch")
    assert_reference_isolated(reference, [{
        "case_id": _case_id(row),
        "relationship": "group" if row.relation == "group" else "private",
        "context": list(row.context),
        "human_reply": list(row.replies),
    } for row in test_candidates])


def extract_judge_chat_context(user_prompt: str) -> list[str]:
    """Render C0's chat evidence without exposing its generation-control structure."""
    parsed: dict[str, list[tuple[int, int, ET.Element]]] = {}
    for name in JUDGE_CHAT_SECTIONS:
        opening = f"<{name}"
        closing = f"</{name}>"
        start = user_prompt.find(opening)
        opening_end = user_prompt.find(">", start + len(opening)) if start >= 0 else -1
        end = user_prompt.find(closing, opening_end + 1) if opening_end >= 0 else -1
        if start >= 0 and opening_end >= 0 and end >= 0:
            block = user_prompt[start:end + len(closing)]
            try:
                root = ET.fromstring(block)
            except ET.ParseError:
                return []
            messages = []
            for position, message in enumerate(root.findall("message")):
                try:
                    index = int(message.attrib.get("index", position))
                except ValueError:
                    index = position
                messages.append((index, position, message))
            parsed[name] = messages

    if not parsed:
        return []

    def render_message(message: ET.Element) -> str:
        neutral = ET.Element("message")
        for attribute in ("role", "type"):
            if message.attrib.get(attribute):
                neutral.set(attribute, message.attrib[attribute])
        neutral.text = "".join(message.itertext())
        return ET.tostring(neutral, encoding="unicode", short_empty_elements=False)

    history = sorted(
        parsed.get("history", []) + parsed.get("consumed_self_replies", []),
        key=lambda row: (row[0], row[1]),
    )
    unread = sorted(parsed.get("unread", []), key=lambda row: (row[0], row[1]))
    sections = []
    if history:
        sections.append(
            "<conversation_history>\n"
            + "\n".join(render_message(message) for _, _, message in history)
            + "\n</conversation_history>"
        )
    if unread:
        sections.append(
            "<messages_to_reply>\n"
            + "\n".join(render_message(message) for _, _, message in unread)
            + "\n</messages_to_reply>"
        )
    return sections


def select_holdout_candidates(
    candidates: list[Candidate],
    excluded: set[tuple[Any, ...]],
    per_relation: int,
    excluded_case_ids: set[str] | None = None,
) -> list[Candidate]:
    excluded_case_ids = excluded_case_ids or set()
    eligible = [
        candidate
        for candidate in candidates
        if _candidate_identity(candidate) not in excluded
        and _case_id(candidate) not in excluded_case_ids
        and 2 <= len("".join(candidate.replies)) <= 80
        and 2 <= len("".join(candidate.context)) <= 240
        and candidate.timestamp
        and candidate.timestamp / (1000 if candidate.timestamp > 10_000_000_000 else 1) < CUTOFF
    ]
    eligible.sort(key=lambda item: hashlib.sha256(_case_id(item).encode()).hexdigest())
    selected: list[Candidate] = []
    chats: Counter[tuple[str, str]] = Counter()
    relations: Counter[str] = Counter()
    for candidate in eligible:
        relation = "group" if candidate.relation == "group" else "private"
        if relations[relation] >= per_relation or chats[(relation, candidate.chat_id)]:
            continue
        selected.append(candidate)
        relations[relation] += 1
        chats[(relation, candidate.chat_id)] += 1
        if relations["group"] >= per_relation and relations["private"] >= per_relation:
            break
    return selected


def _previously_exposed_case_ids(report_dir: Path) -> set[str]:
    case_ids: set[str] = set()
    for path in report_dir.glob("human_ai_judge*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for case in payload.get("cases", []):
            case_id = case.get("case_id")
            if case_id:
                case_ids.add(str(case_id))
    return case_ids


def build_blind_packet(
    cases: list[dict[str, Any]], batch_id: str | None = None, reference_sha256: str = ""
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_id = batch_id or _nonce("batch")
    blind_cases = []
    answer_cases = []
    human_positions = ["A"] * ((len(cases) + 1) // 2) + ["B"] * (len(cases) // 2)
    secrets.SystemRandom().shuffle(human_positions)
    for case, human_is in zip(cases, human_positions):
        public_case_id = _nonce("case")
        ai_is = "B" if human_is == "A" else "A"
        trace = case.get("c0_trace", {}).get("generation_trace", {})
        context = extract_judge_chat_context(str(trace.get("user_prompt") or ""))
        if not context:
            context = list(case["context_original"])
        options = {
            human_is: list(case["human_reply_original"]),
            ai_is: list(case["ai_reply_original"]),
        }
        blind_cases.append({
            "case_id": public_case_id,
            "relationship": case["relationship"],
            "context_original": context,
            "option_A": options["A"],
            "option_B": options["B"],
        })
        answer_cases.append({
            "case_id": public_case_id,
            "source_case_id": case.get("case_id", ""),
            "human_option": human_is,
        })
    blind = {
        "schema_version": 2,
        "batch_id": batch_id,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "human_ai_reply_discrimination_blind",
        "reference_sha256": reference_sha256,
        "instruction": "Given the context, identify which option is the pre-2026-04 human reply. Judge personal likeness, contextual fit and natural humor; do not infer from option order.",
        "cases": blind_cases,
    }
    answers = {
        "schema_version": 2,
        "batch_id": batch_id,
        "privacy": "LOCAL PRIVATE ANSWER KEY: do not open before judging",
        "benchmark_kind": "human_ai_reply_discrimination_answer_key",
        "reference_sha256": reference_sha256,
        "cases": answer_cases,
    }
    answers["blind_sha256"] = _json_sha256(blind)
    return blind, answers


def write_benchmark_artifacts(
    cases: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    minimum_valid: int,
    replay_path: Path,
    blind_path: Path,
    answers_path: Path,
    reference_sha256: str = "",
) -> None:
    if len(cases) < minimum_valid:
        raise RuntimeError(
            f"insufficient valid cases: got {len(cases)}, require at least {minimum_valid}; "
            "existing benchmark artifacts were not changed"
        )
    blind, answers = build_blind_packet(cases, reference_sha256=reference_sha256)
    report = dict(report)
    report["batch_id"] = blind["batch_id"]
    report["blind_sha256"] = answers["blind_sha256"]
    _write_private(replay_path, report)
    _write_private(blind_path, blind)
    _write_private(answers_path, answers)


def _event_has_tool_call(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    event_type = str(event.get("type", "")).lower()
    item_type = str(item.get("type", "")).lower()
    blocked = ("tool_call", "function_call", "command_execution", "mcp_tool", "web_search")
    return any(marker in event_type or marker in item_type for marker in blocked)


def validate_judge_events(events: list[dict[str, Any]]) -> None:
    if any(_event_has_tool_call(event) for event in events):
        raise ValueError("judge isolation failed: tool call found in Codex event log")


def score_predictions(
    answer_key: dict[str, Any],
    predictions: dict[str, Any],
    judge_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if answer_key.get("schema_version") == 2:
        if predictions.get("batch_id") != answer_key.get("batch_id"):
            raise ValueError("prediction batch_id does not match answer key")
        if predictions.get("blind_sha256") != answer_key.get("blind_sha256"):
            raise ValueError("prediction blind_sha256 does not match answer key")
        if judge_events is None:
            raise ValueError("Codex Judge event log is required for schema v2 scoring")
        validate_judge_events(judge_events)
        if predictions.get("judge_events_sha256") != _json_sha256(judge_events):
            raise ValueError("prediction judge_events_sha256 does not match event log")
        if predictions.get("reference_sha256", "") != answer_key.get("reference_sha256", ""):
            raise ValueError("prediction reference_sha256 does not match answer key")
    answers = {case["case_id"]: case["human_option"] for case in answer_key.get("cases", [])}
    rows = []
    for prediction in predictions.get("cases", []):
        case_id = prediction.get("case_id")
        choice = prediction.get("human_option")
        if case_id not in answers or choice not in {"A", "B"}:
            continue
        rows.append({
            "case_id": case_id,
            "predicted": choice,
            "actual": answers[case_id],
            "correct": choice == answers[case_id],
            "confidence": prediction.get("confidence"),
            "reason": prediction.get("reason"),
        })
    correct = sum(row["correct"] for row in rows)
    return {
        "status": "completed" if len(rows) == len(answers) else "incomplete",
        "answered": len(rows),
        "total": len(answers),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "chance_accuracy": 0.5,
        "cases": rows,
    }


def score_api_judge_predictions(
    blind: dict[str, Any],
    answer_key: dict[str, Any],
    predictions: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Score isolated API calls only after judging has fully completed."""
    blind_sha = _json_sha256(blind)
    if predictions.get("batch_id") != answer_key.get("batch_id"):
        raise ValueError("prediction batch_id does not match answer key")
    if blind_sha != answer_key.get("blind_sha256") or blind_sha != predictions.get("blind_sha256"):
        raise ValueError("blind hash mismatch")
    if predictions.get("reference_sha256") != answer_key.get("reference_sha256"):
        raise ValueError("prediction reference_sha256 does not match answer key")
    if audit.get("invocation_id") != predictions.get("invocation_id"):
        raise ValueError("audit invocation_id does not match predictions")
    if audit.get("blind_sha256") != blind_sha:
        raise ValueError("audit blind hash mismatch")

    calls = audit.get("calls") or []
    cases = blind.get("cases") or []
    expected_ids = [case.get("case_id") for case in cases]
    call_ids = [call.get("case_id") for call in calls]
    if call_ids != expected_ids:
        raise ValueError("audit does not contain exactly one ordered call per case")
    if any(
        call.get("other_test_case_ids_in_request")
        or call.get("answer_fields_in_request")
        or call.get("tools_sent")
        or call.get("status_code") != 200
        for call in calls
    ):
        raise ValueError("API Judge isolation audit failed")

    result = score_predictions(
        {**answer_key, "schema_version": 1},
        predictions,
    )
    result.update({
        "schema_version": 1,
        "benchmark_kind": "independent_api_human_ai_reply_discrimination_result",
        "invocation_id": predictions.get("invocation_id"),
        "blind_sha256": blind_sha,
        "reference_sha256": predictions.get("reference_sha256"),
        "isolation_verified": True,
        "surface_baselines": score_surface_baselines(blind, answer_key),
    })
    return result


def _surface_features(reply: list[str]) -> dict[str, int]:
    text = "".join(reply)
    return {
        "characters": len(text),
        "messages": len(reply),
        "emoji": sum(1 for character in text if ord(character) > 0xFFFF),
        "punctuation": sum(1 for character in text if character in "，。！？!?、,.~～…：:；;"),
        "haha": sum(text.count(token) for token in ("哈哈", "hh", "笑死")),
    }


def score_surface_baselines(
    blind: dict[str, Any], answer_key: dict[str, Any]
) -> dict[str, Any]:
    answers = {row["case_id"]: row["human_option"] for row in answer_key.get("cases", [])}
    rules = {
        "shorter_characters": ("characters", "min"),
        "fewer_messages": ("messages", "min"),
        "fewer_punctuation": ("punctuation", "min"),
        "more_haha": ("haha", "max"),
    }
    totals = {name: 0 for name in rules}
    correct = {name: 0 for name in rules}
    case_rows = []
    for case in blind.get("cases", []):
        actual = answers.get(case.get("case_id"))
        features = {
            option: _surface_features(case.get(f"option_{option}") or [])
            for option in ("A", "B")
        }
        guesses = {}
        for name, (feature, direction) in rules.items():
            left, right = features["A"][feature], features["B"][feature]
            if left == right or actual not in {"A", "B"}:
                guesses[name] = None
                continue
            guess = "A" if ((left < right) if direction == "min" else (left > right)) else "B"
            guesses[name] = guess
            totals[name] += 1
            correct[name] += guess == actual
        case_rows.append({"case_id": case.get("case_id"), "features": features, "guesses": guesses})
    answer_count = Counter(answers.values())
    return {
        "position_baselines": {
            "always_A": answer_count["A"] / len(answers) if answers else None,
            "always_B": answer_count["B"] / len(answers) if answers else None,
        },
        "rules": {
            name: {
                "correct": correct[name],
                "decided": totals[name],
                "accuracy": correct[name] / totals[name] if totals[name] else None,
            }
            for name in rules
        },
        "cases": case_rows,
    }


def _export_payloads(export_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(export_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        normalized = path.stem
        for prefix in ("私聊_", "群聊_", "曾经的好友_"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        chat_id = f"chat_{hashlib.sha256(normalized.strip().encode('utf-8')).hexdigest()[:10]}"
        payloads[chat_id] = payload
    return payloads


def _target_index(candidate: Candidate, payload: dict[str, Any]) -> int:
    target_seconds = candidate.timestamp / (1000 if candidate.timestamp > 10_000_000_000 else 1)
    for index, row in enumerate(payload.get("messages", [])):
        row_seconds = float(row.get("createTime") or 0)
        row_seconds = row_seconds / 1000 if row_seconds > 10_000_000_000 else row_seconds
        if row.get("isSend") and row_seconds == target_seconds:
            return index
    raise ValueError(f"target reply not found for {candidate.chat_id} at {candidate.timestamp}")


def _verified_context_matches(expected: str, replayed: str) -> bool:
    def clean(value: str) -> str:
        value = re.sub(r"\\u(?:00a0|200[0-9a-f]|202f)", " ", value, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", value).strip()

    expected = clean(expected)
    replayed = clean(replayed)
    if "[联系人]" not in expected:
        return expected == replayed
    pattern = "^" + ".+?".join(re.escape(part) for part in expected.split("[联系人]")) + "$"
    return re.fullmatch(pattern, replayed) is not None


_NON_TEXT_CONTEXT_MARKERS = {
    "[位置]",
    "[语音]",
}


def _align_equal_timestamp_messages(messages: list[Any], expected: list[str]) -> list[Any]:
    aligned = list(messages)
    start = 0
    while start < len(aligned):
        end = start + 1
        while end < len(aligned) and aligned[end].create_time == aligned[start].create_time:
            end += 1
        if end - start > 1:
            remaining = aligned[start:end]
            ordered = []
            for text in expected[start:end]:
                match = next(
                    (index for index, message in enumerate(remaining) if _verified_context_matches(text, message.text)),
                    None,
                )
                if match is None:
                    break
                ordered.append(remaining.pop(match))
            if not remaining:
                aligned[start:end] = ordered
        start = end
    return aligned


def _latest_verified_text_subsequence(
    messages: list[Any], expected: list[str],
) -> tuple[list[Any], set[int]] | None:
    """Match a frozen compact turn while rejecting hidden self or trailing text."""
    text_rows = [
        (index, message)
        for index, message in enumerate(messages)
        if message.message_type == "text"
    ]
    if not expected or not text_rows:
        return None
    positions: list[int] = []
    cursor = len(text_rows)
    for text in reversed(expected):
        match = next(
            (
                position
                for position in range(cursor - 1, -1, -1)
                if _verified_context_matches(text, text_rows[position][1].text)
            ),
            None,
        )
        if match is None:
            return None
        positions.append(match)
        cursor = match
    positions.reverse()
    if positions[-1] != len(text_rows) - 1:
        return None
    selected_positions = set(positions)
    omitted_positions = set(range(positions[0], positions[-1] + 1)) - selected_positions
    if any(
        text_rows[position][1].sender_type is SenderType.SELF
        for position in omitted_positions
    ):
        return None
    return (
        [text_rows[position][1] for position in positions],
        {text_rows[position][0] for position in omitted_positions},
    )


def rebuild_generation_inputs(
    candidate: Candidate,
    payload: dict[str, Any],
    compact_verified_turn: bool = False,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Rebuild the generator input from the real export, excluding the target reply."""
    from src.models.base import ChatMessage
    from src.utils.xml_utils import _extract_xml_text

    target_index = _target_index(candidate, payload)
    payload_messages = payload.get("messages", [])
    target_row = payload_messages[target_index]
    target_seconds = float(target_row.get("createTime") or 0)
    target_seconds = target_seconds / 1000 if target_seconds > 10_000_000_000 else target_seconds
    timed_rows = []
    for index, row in enumerate(payload_messages):
        row_seconds = float(row.get("createTime") or 0)
        row_seconds = row_seconds / 1000 if row_seconds > 10_000_000_000 else row_seconds
        timed_rows.append((index, row, row_seconds))
    nonzero_times = [row_seconds for _, _, row_seconds in timed_rows if row_seconds]
    ascending = len(nonzero_times) < 2 or nonzero_times[0] <= nonzero_times[-1]
    prior_rows = [
        (index, row, row_seconds)
        for index, row, row_seconds in timed_rows
        if row_seconds < target_seconds
        or (
            row_seconds == target_seconds
            and ((ascending and index < target_index) or (not ascending and index > target_index))
        )
    ]
    prior_rows.sort(key=lambda item: (item[2], item[0] if ascending else -item[0]))
    session = payload.get("session", {})
    chat_name = str(session.get("displayName") or session.get("nickname") or candidate.chat_id)
    chatroom_id = str(session.get("wxid") or "") or None
    is_group = candidate.relation == "group"
    all_messages = []
    for _, row, _ in prior_rows:
        local_type = int(row.get("localType") or 0)
        content = str(row.get("content") or "")
        if local_type != 1 and content.startswith("<"):
            content = _extract_xml_text(content) or content
        is_send = bool(row.get("isSend"))
        sender = "自己" if is_send else (
            str(row.get("senderDisplayName") or row.get("senderUsername") or "对方")
            if is_group else chat_name
        )
        create_time = int(row.get("createTime") or 0)
        message = ChatMessage(
            text=content,
            sender=sender,
            sender_type=SenderType.SELF if is_send else SenderType.OTHER,
            chat_name=chat_name,
            chatroom_id=chatroom_id,
            message_type="text" if local_type == 1 else "other",
            local_id=row.get("localId"),
            server_id=str(row.get("serverId") or row.get("platformMessageId") or ""),
            create_time=create_time,
            raw_type=local_type,
            sender_wxid=str(row.get("senderUsername") or ""),
        )
        all_messages.append(message)
    if compact_verified_turn:
        verified_context = [
            str(text)
            for text in candidate.context
            if str(text) not in _NON_TEXT_CONTEXT_MARKERS
        ]
        expected_count = len(verified_context)
        if expected_count < 1:
            raise ValueError(f"verified context is empty for {candidate.chat_id}")
        verified_messages = [
            message for message in all_messages if message.message_type == "text"
        ][-expected_count:]
        verified_messages = _align_equal_timestamp_messages(verified_messages, verified_context)
        replay_context = [message.text for message in verified_messages]
        matches_verified = len(replay_context) == len(verified_context) and all(
            _verified_context_matches(expected, replayed)
            for expected, replayed in zip(verified_context, replay_context)
        )
        if not matches_verified:
            strict_messages = [
                message
                for message in all_messages
                if (
                    message.create_time / (1000 if message.create_time > 10_000_000_000 else 1)
                ) < target_seconds
            ]
            strict_verified = [
                message for message in strict_messages if message.message_type == "text"
            ][-expected_count:]
            strict_verified = _align_equal_timestamp_messages(strict_verified, verified_context)
            strict_replay = [message.text for message in strict_verified]
            if len(strict_replay) == len(verified_context) and all(
                _verified_context_matches(expected, replayed)
                for expected, replayed in zip(verified_context, strict_replay)
            ):
                all_messages = strict_messages
                verified_messages = strict_verified
                replay_context = strict_replay
        compact_omitted_text_count = 0
        if len(replay_context) != len(verified_context) or not all(
            _verified_context_matches(expected, replayed)
            for expected, replayed in zip(verified_context, replay_context)
        ):
            subsequence = _latest_verified_text_subsequence(all_messages, verified_context)
            if subsequence is not None:
                verified_messages, omitted_indexes = subsequence
                all_messages = [
                    message
                    for index, message in enumerate(all_messages)
                    if index not in omitted_indexes
                ]
                replay_context = [message.text for message in verified_messages]
                compact_omitted_text_count = len(omitted_indexes)
        if len(replay_context) != len(verified_context) or not all(
            _verified_context_matches(expected, replayed)
            for expected, replayed in zip(verified_context, replay_context)
        ):
            raise ValueError(
                f"verified context mismatch for {candidate.chat_id}: "
                f"expected={verified_context!r}, replayed={replay_context!r}"
            )
        boundary = max(
            (
                index
                for index, message in enumerate(verified_messages)
                if message.sender_type is SenderType.SELF
            ),
            default=-1,
        )
        to_reply = [
            message
            for message in verified_messages[boundary + 1:]
            if message.sender_type is SenderType.OTHER
        ]
    else:
        compact_omitted_text_count = 0
        boundary = max(
            (index for index, message in enumerate(all_messages) if message.sender_type is SenderType.SELF),
            default=-1,
        )
        to_reply = [
            message for message in all_messages[boundary + 1:]
            if message.sender_type is SenderType.OTHER
        ]
    if not to_reply:
        raise ValueError(f"no incoming messages before target reply for {candidate.chat_id}")
    metadata = {
        "chat_name": chat_name,
        "chatroom_id": chatroom_id,
        "target_index": target_index,
        "history_count": len(all_messages),
        "unread_count": len(to_reply),
        "compact_verified_turn": compact_verified_turn,
        "compact_omitted_text_count": compact_omitted_text_count,
        "self_history_count": sum(m.sender_type is SenderType.SELF for m in all_messages),
        "distinct_other_senders": len({m.sender for m in all_messages if m.sender_type is SenderType.OTHER}),
    }
    return to_reply, all_messages, metadata


def _normalized_reply(value: list[str] | str) -> str:
    text = "".join(value) if isinstance(value, list) else value
    return "".join(character.lower() for character in text if character.isalnum())


def detect_degenerate_case(human_reply: list[str], output: dict[str, Any]) -> list[str]:
    human = _normalized_reply(human_reply)
    ai = _normalized_reply(output.get("replies", []))
    reasons = []
    injected = json.dumps(output.get("trace", {}), ensure_ascii=False)
    if human and human in _normalized_reply(injected):
        reasons.append("target human reply appeared in C0 prompt/few-shot/tool trace")
    if human and ai:
        similarity = difflib.SequenceMatcher(None, human, ai).ratio()
        if human == ai or (min(len(human), len(ai)) >= 6 and similarity >= 0.9):
            reasons.append(f"C0 reproduced the human reply (similarity={similarity:.3f})")
    return reasons


def _load_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _run_c0(
    candidate: Candidate,
    payload: dict[str, Any],
    llm_client: Any = None,
    memory_engine: Any = None,
    generation_timeout_seconds: float = 20.0,
    persona_few_shot_retriever: Any | None = None,
    persona_few_shot_count: int | None = None,
    persona_few_shot_max_chars: int | None = None,
    force_reply: bool = False,
    enable_active_thread_continuation: bool = False,
    enable_response_action_grounding: bool = False,
    enable_latest_request_first: bool = False,
    enable_unresolved_fact_choice_guard: bool = False,
    enable_no_restatement_action: bool = False,
    enable_plain_response_before_styling: bool = False,
    enable_current_stance_continuity: bool = False,
    enable_group_speaker_target_attribution: bool = False,
    enable_group_explicit_third_party_guard: bool = False,
    enable_group_unified_history: bool = False,
    enable_group_unified_relevant_self: bool = False,
    enable_group_relevant_self_evidence: bool = False,
    enable_group_self_context_freshness_boundary: bool = False,
    enable_group_adjacent_self_response: bool = False,
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
    group_feature_plan_llm_client: Any | None = None,
    group_prefer_participant_structure: bool = False,
    enable_precise_situation_matching: bool = False,
    compact_verified_turn: bool = False,
    reply_generator: Any | None = None,
    few_shot_exclude_ids: set[str] | None = None,
    skill_content_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    from src.reply.generator import ReplyGenerator
    from src.tools.tool_registry import ToolRegistry

    timestamp = candidate.timestamp / (1000 if candidate.timestamp > 10_000_000_000 else 1)
    evaluation_time = datetime.fromtimestamp(timestamp)
    to_reply, all_messages, input_metadata = rebuild_generation_inputs(
        candidate,
        payload,
        compact_verified_turn=compact_verified_turn,
    )
    started = time.monotonic()
    generator = reply_generator
    if generator is None:
        generator = ReplyGenerator(
            llm_client=llm_client,
            memory_engine=memory_engine,
            tool_registry=ToolRegistry(),
            judge_worker=None,
            enable_timestamps=True,
            enable_active_thread_continuation=enable_active_thread_continuation,
            enable_response_action_grounding=enable_response_action_grounding,
            enable_latest_request_first=enable_latest_request_first,
            enable_unresolved_fact_choice_guard=enable_unresolved_fact_choice_guard,
            enable_no_restatement_action=enable_no_restatement_action,
            enable_plain_response_before_styling=enable_plain_response_before_styling,
            enable_current_stance_continuity=enable_current_stance_continuity,
            enable_group_speaker_target_attribution=enable_group_speaker_target_attribution,
            enable_group_explicit_third_party_guard=enable_group_explicit_third_party_guard,
            enable_group_unified_history=enable_group_unified_history,
            enable_group_unified_relevant_self=enable_group_unified_relevant_self,
            enable_group_relevant_self_evidence=enable_group_relevant_self_evidence,
            enable_group_self_context_freshness_boundary=enable_group_self_context_freshness_boundary,
            enable_group_adjacent_self_response=enable_group_adjacent_self_response,
            enable_group_target_action_contract=enable_group_target_action_contract,
            enable_group_public_question_specific_followup=enable_group_public_question_specific_followup,
            enable_group_instant_self_correction=enable_group_instant_self_correction,
            enable_group_visible_self_stance_continuity=enable_group_visible_self_stance_continuity,
            enable_group_pending_self_action=enable_group_pending_self_action,
            enable_group_pattern_participation=enable_group_pattern_participation,
            enable_group_current_event_self_role_continuity=enable_group_current_event_self_role_continuity,
            enable_group_explicit_self_stance_action=enable_group_explicit_self_stance_action,
            enable_group_answer_authority_rewrite=enable_group_answer_authority_rewrite,
            enable_group_feature_plan_before_generation=enable_group_feature_plan_before_generation,
            group_feature_plan_llm_client=group_feature_plan_llm_client,
            group_prefer_participant_structure=group_prefer_participant_structure,
            enable_precise_situation_matching=enable_precise_situation_matching,
            generation_timeout_seconds=generation_timeout_seconds,
            skill_content_overrides=skill_content_overrides,
        )
    if persona_few_shot_retriever is not None:
        generator.persona_few_shot_retriever = persona_few_shot_retriever
        generator.group_persona_few_shot_retriever = persona_few_shot_retriever
        generator.persona_few_shot_ready = persona_few_shot_retriever.is_approved()
    if persona_few_shot_count is not None:
        generator.persona_few_shot_count = persona_few_shot_count
    if persona_few_shot_max_chars is not None:
        generator.persona_few_shot_max_chars = persona_few_shot_max_chars
    replies = generator.generate(
        to_reply,
        all_messages,
        is_group=candidate.relation == "group",
        tick_id=0,
        enable_reply_restraint=not force_reply,
        evaluation_time=evaluation_time,
        few_shot_exclude_ids=few_shot_exclude_ids,
    )
    return {
        "replies": replies,
        "generation_failed": bool(generator.last_generation_failed),
        "force_reply": force_reply,
        "few_shot_ids": list(generator.last_few_shot_ids),
        "duration_seconds": round(time.monotonic() - started, 3),
        "input_metadata": input_metadata,
        "context_original": [message.text for message in to_reply],
        "trace": {
            "few_shot_ids": list(generator.last_few_shot_ids),
            "few_shot_content": generator.last_few_shot_content,
            "system_prompt": generator.last_system_prompt,
            "user_prompt": generator.last_user_prompt,
            "tools_context": generator.last_tools_context,
            "loaded_skills": list(generator.last_loaded_skills),
            "skill_content": generator.last_skill_injected_content,
            "tool_calls": list(generator.last_tool_calls),
            "llm_calls": list(generator.last_llm_calls),
            "latest_request_grounding": dict(generator.last_latest_request_grounding),
            "current_stance_grounding": dict(generator.last_current_stance_grounding),
            "group_answer_authority_review": dict(
                generator.last_group_answer_authority_review
            ),
            "generation_trace": list(generator.last_generation_trace),
        },
    }


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_codex_judge_prompt(
    blind: dict[str, Any], reference: dict[str, Any] | None = None
) -> str:
    reference = reference or {"facts": [], "examples": []}
    return (
        "你是独立盲测 Judge。只能依据下面提供的匿名题目判断哪个选项更像该聊天中真人本人回复。"
        "参考资料来自与测试聊天完全隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。"
        "参考资料只用于理解稳定事实和表达习惯，不得把参考示例当作测试答案。"
        "参考资料和聊天上下文没有提供的个人事实一律视为未知，不得自行判定真实或编造。"
        "你没有工具，也不得尝试读取文件或外部资料。输出严格 JSON："
        "{\"cases\":[{\"case_id\":\"...\",\"human_option\":\"A或B\","
        "\"confidence\":0到1,\"reason\":\"简短理由\"}]}。\n\n"
        "<reference_pack>\n"
        + json.dumps({"facts": reference.get("facts", []), "examples": reference.get("examples", [])}, ensure_ascii=False)
        + "\n</reference_pack>\n\n<blind_cases>\n"
        + json.dumps(blind, ensure_ascii=False)
        + "\n</blind_cases>"
    )


def build_api_judge_messages(
    case: dict[str, Any],
    reference: dict[str, Any] | None = None,
    judge_profile: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    reference = reference or {"facts": [], "examples": []}
    profile_mode = str((judge_profile or {}).get("mode") or "replace")
    if judge_profile is not None and profile_mode not in {"replace", "append_to_builtin"}:
        raise ValueError(f"unsupported Judge profile mode: {profile_mode}")
    if judge_profile is None or profile_mode == "append_to_builtin":
        appended_instruction = str(judge_profile["instructions"]) if judge_profile is not None else ""
        system = (
            "你是独立盲测 Judge。判断匿名回复 A、B 中哪个更像该聊天中真人本人回复。"
            "你只会收到当前这一道题，不能参考其他测试题。"
            "参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。"
            "参考资料只用于理解稳定事实和表达习惯，不得把参考示例当作测试答案。"
            "参考资料和当前聊天没有提供的个人事实一律视为未知，不得自行判定真实或编造。"
            + appended_instruction
            + "不得调用工具或外部资料。只输出一个 JSON 对象："
            "{\"human_option\":\"A或B\",\"confidence\":0到1,\"reason\":\"简短理由\"}。\n\n"
            "<reference_pack>\n"
            + json.dumps(
                {"facts": reference.get("facts", []), "examples": reference.get("examples", [])},
                ensure_ascii=False,
            )
            + "\n</reference_pack>"
        )
    else:
        system = (
            "你是独立盲测 Judge。你只会收到当前这一道题，不能参考其他测试题。"
            "参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。"
            "参考资料只用于理解稳定事实和表达习惯，不得把参考示例当作测试答案。"
            + str(judge_profile["instructions"])
            + "不得调用工具或外部资料。只输出一个 JSON 对象："
            "{\"option_a_thread\":\"A具体承接的问题、状态或并行话题\","
            "\"option_a_counterevidence\":\"A不像真人的最强反证\","
            "\"option_b_thread\":\"B具体承接的问题、状态或并行话题\","
            "\"option_b_counterevidence\":\"B不像真人的最强反证\","
            "\"human_option\":\"A或B\",\"confidence\":0到1,\"reason\":\"简短理由\"}。\n\n"
            "<reference_pack>\n"
            + json.dumps(
                {"facts": reference.get("facts", []), "examples": reference.get("examples", [])},
                ensure_ascii=False,
            )
            + "\n</reference_pack>"
        )
    user = "<blind_case>\n" + json.dumps(case, ensure_ascii=False) + "\n</blind_case>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_shared_single_case_prompt(
    case: dict[str, Any],
    reference: dict[str, Any] | None = None,
    judge_profile: dict[str, Any] | None = None,
) -> str:
    messages = build_api_judge_messages(case, reference, judge_profile)
    return (
        "<judge_instructions>\n"
        + messages[0]["content"]
        + "\n</judge_instructions>\n\n"
        + messages[1]["content"]
    )


def _parse_api_judge_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        try:
            if start < 0 or end <= start:
                raise json.JSONDecodeError("missing JSON object", text, 0)
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            option = re.search(r'"human_option"\s*:\s*"([AB])"', text)
            if not option:
                raise ValueError("API Judge did not return an unambiguous human_option")
            confidence = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text)
            reason = re.search(r'"reason"\s*:\s*"(.*)"\s*\}?\s*$', text, re.DOTALL)
            payload = {
                "human_option": option.group(1),
                "confidence": float(confidence.group(1)) if confidence else None,
                "reason": reason.group(1) if reason else "（原始响应 JSON 格式损坏，详见审计）",
            }
    if payload.get("human_option") not in {"A", "B"}:
        raise ValueError("API Judge returned an invalid human_option")
    parsed = {
        "human_option": payload["human_option"],
        "confidence": payload.get("confidence"),
        "reason": str(payload.get("reason") or ""),
    }
    if "selected_reply" in payload:
        parsed["selected_reply"] = payload.get("selected_reply")
    for field in (
        "option_a_thread",
        "option_a_person_specific_evidence",
        "option_a_counterevidence",
        "option_b_thread",
        "option_b_person_specific_evidence",
        "option_b_counterevidence",
        "decisive_evidence",
        "uncertainty",
    ):
        if field in payload:
            parsed[field] = str(payload.get(field) or "")
    return parsed


def _response_for_audit(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(payload, ensure_ascii=False))
    for block in sanitized.get("content") or []:
        if block.get("type") == "thinking":
            block.clear()
            block.update({"type": "thinking", "redacted": True})
    return sanitized


def run_api_judge_independently(
    blind_path: Path,
    predictions_path: Path,
    audit_path: Path,
    reference_path: Path,
    config_path: Path,
) -> None:
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_sha = str(reference.get("reference_sha256") or "")
    if blind.get("reference_sha256", "") != reference_sha:
        raise ValueError("Judge reference pack does not match blind packet")
    api_key, base_url = load_volcengine_credentials(config_path)
    endpoint = f"{base_url.rstrip('/')}/v1/messages"
    invocation_id = _nonce("api_judge")
    cases = []
    calls = []
    for index, case in enumerate(blind.get("cases") or [], start=1):
        prompt = build_shared_single_case_prompt(case, reference)
        messages = [{"role": "user", "content": prompt}]
        client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
        recorded_at = datetime.now(timezone.utc).isoformat()
        raw = client.chat(
            messages=messages,
            temperature=0,
            max_tokens=300,
            timeout=120,
            raise_on_error=True,
            max_retries=1,
        )
        prediction = _parse_api_judge_response(raw)
        cases.append({"case_id": case["case_id"], **prediction})
        request_body = client.last_request_body
        serialized_request = json.dumps(request_body, ensure_ascii=False)
        calls.append({
            "index": index,
            "case_id": case["case_id"],
            "recorded_at_utc": recorded_at,
            "transport": "direct_https_api",
            "conversation_state": "none; new client and one request for this case",
            "endpoint": endpoint,
            "headers": {
                "content-type": "application/json",
                "x-api-key": "[REDACTED]",
                "anthropic-version": "2023-06-01",
            },
            "request_body": request_body,
            "shared_prompt": prompt,
            "shared_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request_sha256": _json_sha256(request_body),
            "other_test_case_ids_in_request": [
                other["case_id"]
                for other in blind.get("cases") or []
                if other["case_id"] != case["case_id"]
                and other["case_id"] in serialized_request
            ],
            "answer_fields_in_request": [
                marker for marker, present in (
                    ("human_option=A_or_B", bool(re.search(r'\"human_option\"\s*:\s*\"[AB]\"', serialized_request))),
                    ("source_case_id", '"source_case_id"' in serialized_request),
                    ("actual", '"actual"' in serialized_request),
                    ("correct", '"correct"' in serialized_request),
                )
                if present
            ],
            "tools_sent": bool(request_body.get("tools")),
            "status_code": client.last_status_code,
            "elapsed_ms": client.last_elapsed_ms,
            "resolved_model": client.resolved_model,
            "response": _response_for_audit(client.last_response_payload),
            "parsed_prediction": prediction,
        })
    predictions = {
        "schema_version": 3,
        "benchmark_kind": "independent_api_human_ai_reply_discrimination_predictions",
        "invocation_id": invocation_id,
        "batch_id": blind["batch_id"],
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": reference_sha,
        "requested_model": VOLCENGINE_DSV4F_MODEL,
        "cases": cases,
    }
    audit = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "record_kind": "independent_api_human_ai_judge_invocation_audit",
        "invocation_id": invocation_id,
        "isolation": {
            "codex_used": False,
            "one_api_request_per_case": True,
            "new_client_per_case": True,
            "shared_conversation_state": False,
            "answer_key_loaded_during_judging": False,
            "tools_available": False,
        },
        "blind_sha256": predictions["blind_sha256"],
        "reference_sha256": reference_sha,
        "calls": calls,
    }
    _write_private(predictions_path, predictions)
    _write_private(audit_path, audit)


def run_codex_judge_independently(
    blind_path: Path,
    predictions_path: Path,
    audit_path: Path,
    reference_path: Path,
    max_cases: int | None = None,
    workers: int = 1,
    judge_profile_path: Path | None = None,
    only_case_ids: set[str] | None = None,
    dynamic_few_shot_manifest_path: Path | None = None,
    baseline_predictions_path: Path | None = None,
    baseline_audit_path: Path | None = None,
    dynamic_few_shot_mode: str = "append",
    prior_dynamic_few_shot_manifest_path: Path | None = None,
    prior_predictions_path: Path | None = None,
    prior_audit_path: Path | None = None,
    presentation: str = "canonical",
    official_metadata: dict[str, Any] | None = None,
) -> dict[str, int]:
    official_metadata = dict(official_metadata or {})
    if official_metadata:
        required_official_fields = {
            "official_judge_baseline_id",
            "current_c0_baseline_id",
            "prediction_type",
            "provider",
            "model",
            "reasoning_effort",
            "timeout_seconds",
            "codex_cli_version",
            "correction_artifact_path",
            "correction_artifact_sha256",
            "correction_model_name",
            "correction_threshold",
            "metadata_answers_path",
            "metadata_answers_sha256",
            "metadata_replay_path",
            "metadata_replay_sha256",
        }
        if set(official_metadata) != required_official_fields:
            raise ValueError("official Judge metadata is incomplete or contains unsupported fields")
        if official_metadata["official_judge_baseline_id"] != current_judge_baseline_id():
            raise ValueError("official Judge baseline ID does not match the adopted Judge")
        if official_metadata["current_c0_baseline_id"] != current_c0_baseline_id():
            raise ValueError("official Judge is not bound to the currently adopted C0 baseline")
        if official_metadata["provider"] != "codex_cli":
            raise ValueError("official Judge provider must be codex_cli")
        if official_metadata["prediction_type"] != "llm_boolean_features_with_logistic_correction":
            raise ValueError("official Judge must use the configured logistic correction")
        if any(
            official_metadata[field] in (None, "")
            for field in ("model", "reasoning_effort", "codex_cli_version")
        ):
            raise ValueError("official Judge runtime metadata cannot be empty")
        if not isinstance(official_metadata["timeout_seconds"], int) or official_metadata["timeout_seconds"] <= 0:
            raise ValueError("official Judge timeout_seconds must be a positive integer")
        if any(
            path is not None
            for path in (
                dynamic_few_shot_manifest_path,
                baseline_predictions_path,
                baseline_audit_path,
                prior_dynamic_few_shot_manifest_path,
                prior_predictions_path,
                prior_audit_path,
            )
        ):
            raise ValueError("official Judge cannot use dynamic few-shot or historical result reuse")
        if presentation != "canonical":
            raise ValueError("official Judge requires canonical presentation")
        if judge_profile_path is None:
            raise ValueError("official Judge requires its adopted Judge profile")
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_sha = str(reference.get("reference_sha256") or "")
    if blind.get("reference_sha256", "") != reference_sha:
        raise ValueError("Judge reference pack does not match blind packet")
    blind_sha = _json_sha256(blind)
    if presentation not in {"canonical", "swapped"}:
        raise ValueError("presentation must be canonical or swapped")
    formal_judge = None
    context_metadata_by_case: dict[str, dict[str, Any]] = {}
    if official_metadata:
        # These imports are deliberately lazy: this evaluator is also used by
        # standalone experiments, which must not silently load the formal model.
        from scripts.label_few_shot_ranker_samples import load_formal_judge, score_formal_judge_pair
        from scripts.run_judge_feature_classifier_experiment import (
            feature_extractor_prompt,
            load_context_metadata,
            run_codex_feature_extractor,
        )

        artifact_path = PROJECT_ROOT / str(official_metadata["correction_artifact_path"])
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != official_metadata["correction_artifact_sha256"]:
            raise ValueError("official Judge correction artifact sha256 mismatch")
        formal_judge = load_formal_judge(artifact_path)
        if formal_judge.artifact_sha256 != official_metadata["correction_artifact_sha256"]:
            raise ValueError("formal Judge artifact identity does not match official metadata")
        answers_path = PROJECT_ROOT / str(official_metadata["metadata_answers_path"])
        replay_path = PROJECT_ROOT / str(official_metadata["metadata_replay_path"])
        if hashlib.sha256(answers_path.read_bytes()).hexdigest() != official_metadata["metadata_answers_sha256"]:
            raise ValueError("official Judge answers sidecar sha256 mismatch")
        if hashlib.sha256(replay_path.read_bytes()).hexdigest() != official_metadata["metadata_replay_sha256"]:
            raise ValueError("official Judge replay sidecar sha256 mismatch")
        context_metadata_by_case = load_context_metadata(
            blind,
            json.loads(answers_path.read_text(encoding="utf-8")),
            json.loads(replay_path.read_text(encoding="utf-8")),
        )
    dynamic_examples_by_case: dict[str, list[dict[str, Any]]] = {}
    manifest = None
    dynamic_few_shot_manifest_sha256 = None
    if dynamic_few_shot_mode not in {"append", "replace", "replace_on_match"}:
        raise ValueError("dynamic few-shot mode must be append, replace, or replace_on_match")
    if dynamic_few_shot_mode in {"replace", "replace_on_match"} and dynamic_few_shot_manifest_path is None:
        raise ValueError("dynamic few-shot replacement mode requires a manifest")
    if dynamic_few_shot_manifest_path is not None:
        manifest_bytes = dynamic_few_shot_manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        dynamic_few_shot_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest.get("blind_sha256") != blind_sha:
            raise ValueError("dynamic few-shot manifest does not match blind packet")
        if manifest.get("reference_sha256") != reference_sha:
            raise ValueError("dynamic few-shot manifest does not match Judge reference pack")
        blind_relationships = {
            str(case["case_id"]): str(case.get("relationship") or "")
            for case in blind.get("cases") or []
        }
        manifest_cases = manifest.get("cases") or []
        if len(manifest_cases) != len(blind_relationships):
            raise ValueError("dynamic few-shot manifest must cover every blind case exactly once")
        allowed_example_fields = {"id", "relationship", "context", "human_reply"}
        for row in manifest_cases:
            if set(row) != {"case_id", "relationship", "examples"}:
                raise ValueError("dynamic few-shot manifest case contains unsupported fields")
            case_id = str(row.get("case_id") or "")
            relationship = str(row.get("relationship") or "")
            if case_id not in blind_relationships or case_id in dynamic_examples_by_case:
                raise ValueError("dynamic few-shot manifest has an unknown or duplicate case")
            if relationship != blind_relationships[case_id]:
                raise ValueError("dynamic few-shot manifest relationship does not match blind case")
            examples = row.get("examples")
            if not isinstance(examples, list) or not 0 <= len(examples) <= 10:
                raise ValueError("dynamic few-shot manifest requires zero to ten examples per case")
            for example in examples:
                if set(example) != allowed_example_fields:
                    raise ValueError("dynamic few-shot example contains unsupported fields")
                if str(example.get("relationship") or "") != relationship:
                    raise ValueError("dynamic few-shot example crosses private/group boundary")
                if not str(example.get("id") or ""):
                    raise ValueError("dynamic few-shot example requires an id")
                for field in ("context", "human_reply"):
                    value = example.get(field)
                    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
                        raise ValueError(f"dynamic few-shot example {field} must be a non-empty string list")
            if len({str(example["id"]) for example in examples}) != len(examples):
                raise ValueError("dynamic few-shot example ids must be unique within each case")
            dynamic_examples_by_case[case_id] = examples
        if set(dynamic_examples_by_case) != set(blind_relationships):
            raise ValueError("dynamic few-shot manifest does not cover the blind cases")
    prior_paths = (
        prior_dynamic_few_shot_manifest_path,
        prior_predictions_path,
        prior_audit_path,
    )
    if any(path is not None for path in prior_paths) and not all(path is not None for path in prior_paths):
        raise ValueError("prior candidate reuse requires manifest, predictions, and audit files")
    if all(path is not None for path in prior_paths) and dynamic_few_shot_manifest_path is None:
        raise ValueError("prior candidate reuse requires a current dynamic few-shot manifest")
    judge_profile = None
    judge_profile_name = "builtin_default"
    judge_profile_sha256 = None
    if judge_profile_path is not None:
        profile_bytes = judge_profile_path.read_bytes()
        judge_profile = json.loads(profile_bytes)
        if not judge_profile.get("name") or not judge_profile.get("instructions"):
            raise ValueError("Judge profile requires name and instructions")
        judge_profile_name = str(judge_profile["name"])
        judge_profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    if (baseline_predictions_path is None) != (baseline_audit_path is None):
        raise ValueError("baseline reuse requires both prediction and audit files")
    if baseline_predictions_path is not None and dynamic_few_shot_manifest_path is None:
        raise ValueError("baseline reuse requires a dynamic few-shot manifest")
    if baseline_predictions_path is not None and dynamic_few_shot_mode == "replace":
        raise ValueError("dynamic few-shot replace mode cannot reuse baseline cases")
    baseline_predictions = None
    baseline_audit = None
    baseline_predictions_sha256 = None
    baseline_audit_sha256 = None
    if baseline_predictions_path is not None and baseline_audit_path is not None:
        baseline_prediction_bytes = baseline_predictions_path.read_bytes()
        baseline_audit_bytes = baseline_audit_path.read_bytes()
        baseline_predictions = json.loads(baseline_prediction_bytes)
        baseline_audit = json.loads(baseline_audit_bytes)
        baseline_predictions_sha256 = hashlib.sha256(baseline_prediction_bytes).hexdigest()
        baseline_audit_sha256 = hashlib.sha256(baseline_audit_bytes).hexdigest()
        baseline_expected = {
            "batch_id": blind["batch_id"],
            "blind_sha256": blind_sha,
            "reference_sha256": reference_sha,
            "judge_profile_name": judge_profile_name,
            "judge_profile_sha256": judge_profile_sha256,
        }
        baseline_actual = dict(baseline_predictions)
        baseline_actual.setdefault("judge_profile_name", "builtin_default")
        baseline_actual.setdefault("judge_profile_sha256", None)
        if any(baseline_actual.get(field) != value for field, value in baseline_expected.items()):
            raise ValueError("baseline Judge predictions do not match this benchmark")
        if baseline_predictions.get("dynamic_few_shot_manifest_sha256") is not None:
            raise ValueError("baseline Judge predictions must not use dynamic few-shot examples")
        if baseline_audit.get("invocation_id") != baseline_predictions.get("invocation_id"):
            raise ValueError("baseline prediction and audit invocation ids do not match")
        baseline_audit_expected = {
            "blind_sha256": blind_sha,
            "reference_sha256": reference_sha,
            "judge_profile_name": judge_profile_name,
            "judge_profile_sha256": judge_profile_sha256,
        }
        baseline_audit_actual = dict(baseline_audit)
        baseline_audit_actual.setdefault("judge_profile_name", "builtin_default")
        baseline_audit_actual.setdefault("judge_profile_sha256", None)
        if any(
            baseline_audit_actual.get(field) != value
            for field, value in baseline_audit_expected.items()
        ):
            raise ValueError("baseline Judge audit does not match this benchmark")
    prior_predictions = None
    prior_audit = None
    prior_examples_by_case: dict[str, list[dict[str, Any]]] = {}
    prior_predictions_sha256 = None
    prior_audit_sha256 = None
    prior_manifest_sha256 = None
    if all(path is not None for path in prior_paths):
        prior_manifest_bytes = prior_dynamic_few_shot_manifest_path.read_bytes()
        prior_prediction_bytes = prior_predictions_path.read_bytes()
        prior_audit_bytes = prior_audit_path.read_bytes()
        prior_manifest = json.loads(prior_manifest_bytes)
        prior_predictions = json.loads(prior_prediction_bytes)
        prior_audit = json.loads(prior_audit_bytes)
        prior_manifest_sha256 = hashlib.sha256(prior_manifest_bytes).hexdigest()
        prior_predictions_sha256 = hashlib.sha256(prior_prediction_bytes).hexdigest()
        prior_audit_sha256 = hashlib.sha256(prior_audit_bytes).hexdigest()
        prior_examples_by_case = {
            str(row["case_id"]): row.get("examples") or []
            for row in prior_manifest.get("cases") or []
        }
        if set(prior_examples_by_case) != set(dynamic_examples_by_case):
            raise ValueError("prior dynamic few-shot manifest does not cover this benchmark")
        prior_expected = {
            "batch_id": blind["batch_id"],
            "blind_sha256": blind_sha,
            "reference_sha256": reference_sha,
            "judge_profile_name": judge_profile_name,
            "judge_profile_sha256": judge_profile_sha256,
            "dynamic_few_shot_manifest_sha256": prior_manifest_sha256,
            "dynamic_few_shot_mode": prior_predictions.get("dynamic_few_shot_mode", "append"),
        }
        prior_mode = str(prior_expected["dynamic_few_shot_mode"])
        if prior_mode != dynamic_few_shot_mode and not (
            prior_mode == "replace" and dynamic_few_shot_mode == "replace_on_match"
        ):
            raise ValueError("prior candidate mode cannot be reused by the current mode")
        prior_actual = dict(prior_predictions)
        prior_actual.setdefault("judge_profile_name", "builtin_default")
        prior_actual.setdefault("judge_profile_sha256", None)
        if any(prior_actual.get(field) != value for field, value in prior_expected.items()):
            raise ValueError("prior candidate predictions do not match this benchmark")
        if prior_audit.get("invocation_id") != prior_predictions.get("invocation_id"):
            raise ValueError("prior candidate prediction and audit invocation ids do not match")
        if any(prior_audit.get(field) != value for field, value in prior_expected.items() if field != "batch_id"):
            raise ValueError("prior candidate audit does not match this benchmark")
    if predictions_path.exists() != audit_path.exists():
        raise ValueError("resume requires both prediction and audit files")
    if predictions_path.exists():
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected = {
            "batch_id": blind["batch_id"],
            "blind_sha256": blind_sha,
            "reference_sha256": reference_sha,
            "judge_profile_name": judge_profile_name,
            "judge_profile_sha256": judge_profile_sha256,
            "dynamic_few_shot_manifest_sha256": dynamic_few_shot_manifest_sha256,
            "dynamic_few_shot_mode": dynamic_few_shot_mode,
            "baseline_predictions_sha256": baseline_predictions_sha256,
            "baseline_audit_sha256": baseline_audit_sha256,
            "prior_dynamic_few_shot_manifest_sha256": prior_manifest_sha256,
            "prior_predictions_sha256": prior_predictions_sha256,
            "prior_audit_sha256": prior_audit_sha256,
            "presentation": presentation,
        }
        expected.update(official_metadata)
        actual = dict(predictions)
        actual.setdefault("judge_profile_name", "builtin_default")
        actual.setdefault("judge_profile_sha256", None)
        actual.setdefault("dynamic_few_shot_manifest_sha256", None)
        actual.setdefault("dynamic_few_shot_mode", "append")
        actual.setdefault("baseline_predictions_sha256", None)
        actual.setdefault("baseline_audit_sha256", None)
        actual.setdefault("prior_dynamic_few_shot_manifest_sha256", None)
        actual.setdefault("prior_predictions_sha256", None)
        actual.setdefault("prior_audit_sha256", None)
        actual.setdefault("presentation", "canonical")
        if any(actual.get(field) != value for field, value in expected.items()):
            raise ValueError("existing Codex Judge output does not match this benchmark")
        if audit.get("invocation_id") != predictions.get("invocation_id"):
            raise ValueError("prediction and audit invocation ids do not match")
        audit_profile = {
            "judge_profile_name": audit.get("judge_profile_name", "builtin_default"),
            "judge_profile_sha256": audit.get("judge_profile_sha256"),
            "dynamic_few_shot_manifest_sha256": audit.get("dynamic_few_shot_manifest_sha256"),
            "dynamic_few_shot_mode": audit.get("dynamic_few_shot_mode", "append"),
            "baseline_predictions_sha256": audit.get("baseline_predictions_sha256"),
            "baseline_audit_sha256": audit.get("baseline_audit_sha256"),
            "prior_dynamic_few_shot_manifest_sha256": audit.get("prior_dynamic_few_shot_manifest_sha256"),
            "prior_predictions_sha256": audit.get("prior_predictions_sha256"),
            "prior_audit_sha256": audit.get("prior_audit_sha256"),
            "presentation": audit.get("presentation", "canonical"),
        }
        audit_profile.update({field: audit.get(field) for field in official_metadata})
        if any(
            audit_profile[field] != value
            for field, value in expected.items()
            if field.startswith("judge_profile_")
            or field == "dynamic_few_shot_manifest_sha256"
            or field == "dynamic_few_shot_mode"
            or field.startswith("baseline_")
            or field.startswith("prior_")
            or field == "presentation"
            or field in official_metadata
        ):
            raise ValueError("existing Codex Judge audit uses a different Judge configuration")
    else:
        invocation_id = _nonce("codex_api_judge")
        predictions = {
            "schema_version": 3,
            "benchmark_kind": "independent_codex_human_ai_reply_discrimination_predictions",
            "invocation_id": invocation_id,
            "batch_id": blind["batch_id"],
            "blind_sha256": blind_sha,
            "reference_sha256": reference_sha,
            "judge_profile_name": judge_profile_name,
            "judge_profile_sha256": judge_profile_sha256,
            "presentation": presentation,
            "cases": [],
        }
        predictions.update(official_metadata)
        if dynamic_few_shot_manifest_sha256 is not None:
            predictions["dynamic_few_shot_manifest_sha256"] = dynamic_few_shot_manifest_sha256
            predictions["dynamic_few_shot_mode"] = dynamic_few_shot_mode
        if baseline_predictions_sha256 is not None:
            predictions["baseline_predictions_sha256"] = baseline_predictions_sha256
            predictions["baseline_audit_sha256"] = baseline_audit_sha256
        if prior_predictions_sha256 is not None:
            predictions["prior_dynamic_few_shot_manifest_sha256"] = prior_manifest_sha256
            predictions["prior_predictions_sha256"] = prior_predictions_sha256
            predictions["prior_audit_sha256"] = prior_audit_sha256
        audit = {
            "schema_version": 1,
            "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
            "record_kind": "independent_codex_human_ai_judge_invocation_audit",
            "invocation_id": invocation_id,
            "isolation": {
                "one_process_per_case": True,
                "ephemeral": True,
                "shared_conversation_state": False,
                "answer_sidecar_loaded_for_metadata_lookup": bool(official_metadata),
                "human_label_exposed_to_llm_or_logistic": False,
                "tool_calls_allowed": False,
            },
            "blind_sha256": blind_sha,
            "reference_sha256": reference_sha,
            "judge_profile_name": judge_profile_name,
            "judge_profile_sha256": judge_profile_sha256,
            "presentation": presentation,
            "calls": [],
        }
        audit.update(official_metadata)
        if dynamic_few_shot_manifest_sha256 is not None:
            audit["dynamic_few_shot_manifest_sha256"] = dynamic_few_shot_manifest_sha256
            audit["dynamic_few_shot_mode"] = dynamic_few_shot_mode
        if baseline_predictions_sha256 is not None:
            audit["baseline_predictions_sha256"] = baseline_predictions_sha256
            audit["baseline_audit_sha256"] = baseline_audit_sha256
        if prior_predictions_sha256 is not None:
            audit["prior_dynamic_few_shot_manifest_sha256"] = prior_manifest_sha256
            audit["prior_predictions_sha256"] = prior_predictions_sha256
            audit["prior_audit_sha256"] = prior_audit_sha256
    cases = predictions["cases"]
    calls = audit["calls"]
    completed_case_ids = {row["case_id"] for row in cases}
    all_case_ids = [case["case_id"] for case in blind.get("cases") or []]
    if baseline_predictions is not None and baseline_audit is not None:
        baseline_cases = baseline_predictions.get("cases") or []
        baseline_calls = baseline_audit.get("calls") or []
        if [row.get("case_id") for row in baseline_cases] != all_case_ids:
            raise ValueError("baseline Judge predictions do not cover blind cases in order")
        if [row.get("case_id") for row in baseline_calls] != all_case_ids:
            raise ValueError("baseline Judge audit does not cover blind cases in order")
        if [row.get("index") for row in baseline_calls] != list(range(1, len(all_case_ids) + 1)):
            raise ValueError("baseline Judge audit indices do not match blind case order")
        baseline_case_by_id = {row["case_id"]: row for row in baseline_cases}
        baseline_call_by_id = {row["case_id"]: row for row in baseline_calls}
        reused_case_ids = [
            case_id for case_id in all_case_ids
            if not dynamic_examples_by_case[case_id] and case_id not in completed_case_ids
        ]
        for case_id in reused_case_ids:
            cases.append(copy.deepcopy(baseline_case_by_id[case_id]))
            reused_call = copy.deepcopy(baseline_call_by_id[case_id])
            reused_call["reused_from_baseline"] = True
            reused_call["baseline_invocation_id"] = baseline_predictions["invocation_id"]
            calls.append(reused_call)
        completed_case_ids.update(reused_case_ids)
    if prior_predictions is not None and prior_audit is not None:
        prior_cases = prior_predictions.get("cases") or []
        prior_calls = prior_audit.get("calls") or []
        if [row.get("case_id") for row in prior_cases] != all_case_ids:
            raise ValueError("prior candidate predictions do not cover blind cases in order")
        if [row.get("case_id") for row in prior_calls] != all_case_ids:
            raise ValueError("prior candidate audit does not cover blind cases in order")
        prior_case_by_id = {row["case_id"]: row for row in prior_cases}
        prior_call_by_id = {row["case_id"]: row for row in prior_calls}
        reused_case_ids = [
            case_id for case_id in all_case_ids
            if dynamic_examples_by_case[case_id]
            and dynamic_examples_by_case[case_id] == prior_examples_by_case[case_id]
            and case_id not in completed_case_ids
        ]
        for case_id in reused_case_ids:
            cases.append(copy.deepcopy(prior_case_by_id[case_id]))
            reused_call = copy.deepcopy(prior_call_by_id[case_id])
            reused_call["reused_from_prior_candidate"] = True
            reused_call["prior_invocation_id"] = prior_predictions["invocation_id"]
            calls.append(reused_call)
        completed_case_ids.update(reused_case_ids)
    if only_case_ids:
        missing_case_ids = only_case_ids.difference(all_case_ids)
        if missing_case_ids:
            raise ValueError(f"unknown Codex Judge case IDs: {sorted(missing_case_ids)}")
    case_order = {case_id: index for index, case_id in enumerate(all_case_ids)}
    calls.sort(key=lambda row: row["index"])
    cases.sort(key=lambda row: case_order[row["case_id"]])
    _write_private(predictions_path, predictions)
    _write_private(audit_path, audit)
    pending = [
        (index, case)
        for index, case in enumerate(blind.get("cases") or [], start=1)
        if case["case_id"] not in completed_case_ids
        and (not only_case_ids or case["case_id"] in only_case_ids)
    ]
    if max_cases is not None:
        pending = pending[:max_cases]

    def run_case(index: int, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        displayed_case = copy.deepcopy(case)
        if presentation == "swapped":
            displayed_case["option_A"], displayed_case["option_B"] = (
                displayed_case["option_B"],
                displayed_case["option_A"],
            )
        case_reference = reference
        dynamic_example_ids: list[str] = []
        if dynamic_examples_by_case:
            case_reference = copy.deepcopy(reference)
            dynamic_examples = dynamic_examples_by_case[case["case_id"]]
            if dynamic_few_shot_mode == "replace" or (
                dynamic_few_shot_mode == "replace_on_match" and dynamic_examples
            ):
                case_reference["examples"] = dynamic_examples
            else:
                case_reference.setdefault("examples", []).extend(dynamic_examples)
            dynamic_example_ids = [str(row["id"]) for row in dynamic_examples]
        prompt = build_shared_single_case_prompt(displayed_case, case_reference, judge_profile)
        with tempfile.TemporaryDirectory(prefix="human-ai-judge-single-") as temp_dir:
            temp = Path(temp_dir)
            last_message = temp / "last_message.json"
            command = [
                "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "--sandbox", "workspace-write", "--cd", str(temp), "--skip-git-repo-check",
                "--json", "--output-last-message", str(last_message), "-",
            ]
            if official_metadata:
                command[2:2] = [
                    "--model", str(official_metadata["model"]),
                    "--config", f'model_reasoning_effort="{official_metadata["reasoning_effort"]}"',
                ]
            recorded_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            run_kwargs: dict[str, Any] = {
                "input": prompt,
                "text": True,
                "capture_output": True,
                "check": False,
            }
            if official_metadata:
                run_kwargs["timeout"] = official_metadata["timeout_seconds"]
            completed = subprocess.run(command, **run_kwargs)
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
            raw = last_message.read_text(encoding="utf-8") if last_message.exists() else ""
            tool_count = sum(_event_has_tool_call(event) for event in events)
            serialized_prompt = prompt
            call = {
                "index": index,
                "case_id": case["case_id"],
                "recorded_at_utc": recorded_at,
                "transport": "codex_exec_ephemeral_single_case",
                "conversation_state": "none; new ephemeral process and temp directory",
                "argv": command,
                "shared_prompt": prompt,
                "shared_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "other_test_case_ids_in_request": [
                    other for other in all_case_ids
                    if other != case["case_id"] and other in serialized_prompt
                ],
                "answer_fields_in_request": [
                    marker for marker, present in (
                        ("human_option=A_or_B", bool(re.search(r'\"human_option\"\s*:\s*\"[AB]\"', serialized_prompt))),
                        ("source_case_id", '"source_case_id"' in serialized_prompt),
                        ("actual", '"actual"' in serialized_prompt),
                        ("correct", '"correct"' in serialized_prompt),
                    ) if present
                ],
                "tools_sent": bool(tool_count),
                "tool_call_count": tool_count,
                "returncode": completed.returncode,
                "status_code": 200 if completed.returncode == 0 else None,
                "elapsed_ms": elapsed_ms,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "events": events,
                "event_parse_errors": parse_errors,
                "raw_response": raw,
            }
            if dynamic_example_ids:
                call["dynamic_few_shot_example_ids"] = dynamic_example_ids
            if completed.returncode:
                raise RuntimeError(f"Codex Judge failed for {case['case_id']}: {completed.stderr[-2000:]}")
            if parse_errors or tool_count:
                raise ValueError(f"Codex Judge isolation failed for {case['case_id']}")
            displayed_prediction = _parse_api_judge_response(raw)
            prediction = copy.deepcopy(displayed_prediction)
            if presentation == "swapped":
                prediction["human_option"] = "B" if displayed_prediction["human_option"] == "A" else "A"
                for a_field, b_field in (
                    ("option_a_thread", "option_b_thread"),
                    ("option_a_person_specific_evidence", "option_b_person_specific_evidence"),
                    ("option_a_counterevidence", "option_b_counterevidence"),
                ):
                    if a_field in prediction or b_field in prediction:
                        prediction[a_field], prediction[b_field] = (
                            displayed_prediction.get(b_field, ""),
                            displayed_prediction.get(a_field, ""),
                        )
            prediction["displayed_human_option"] = displayed_prediction["human_option"]
            if official_metadata:
                assert formal_judge is not None
                feature_prompt = feature_extractor_prompt(case)
                llm_features, feature_audit = run_codex_feature_extractor(
                    feature_prompt,
                    timeout_seconds=official_metadata["timeout_seconds"],
                    model=official_metadata["model"],
                    reasoning_effort=official_metadata["reasoning_effort"],
                )
                probability_a = score_formal_judge_pair(
                    formal_judge,
                    llm_features,
                    list(case["option_A"]),
                    list(case["option_B"]),
                    context_metadata_by_case[case["case_id"]],
                )
                small_model_option = "A" if probability_a >= 0.5 else "B"
                small_model_confidence = max(probability_a, 1.0 - probability_a)
                base_human_option = prediction["human_option"]
                correction_applied = (
                    small_model_option != base_human_option
                    and small_model_confidence >= official_metadata["correction_threshold"]
                )
                prediction["base_human_option"] = base_human_option
                prediction["small_model_probability_a"] = probability_a
                prediction["small_model_human_option"] = small_model_option
                prediction["small_model_confidence"] = small_model_confidence
                prediction["correction_applied"] = correction_applied
                if correction_applied:
                    prediction["human_option"] = small_model_option
                call["feature_extraction"] = {
                    **feature_audit,
                    "model": official_metadata["model"],
                    "reasoning_effort": official_metadata["reasoning_effort"],
                    "feature_schema_sha256": formal_judge.schema_hash,
                    "answer_label_used": False,
                }
            call["displayed_prediction"] = displayed_prediction
            call["parsed_prediction"] = prediction
            return call, {"case_id": case["case_id"], **prediction}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_case, index, case) for index, case in pending]
        for future in concurrent.futures.as_completed(futures):
            try:
                call, prediction = future.result()
            except Exception as exc:
                if official_metadata:
                    raise RuntimeError(f"official Codex Judge case failed: {exc}") from exc
                print(f"Codex Judge case failed; leaving it pending for resume: {exc}", file=sys.stderr)
                continue
            calls.append(call)
            cases.append(prediction)
            calls.sort(key=lambda row: row["index"])
            cases.sort(key=lambda row: case_order[row["case_id"]])
            _write_private(predictions_path, predictions)
            _write_private(audit_path, audit)
    return {"completed": len(cases), "total": len(all_case_ids)}


def run_codex_judge(
    blind_path: Path,
    predictions_path: Path,
    events_path: Path,
    audit_path: Path | None = None,
    reference_path: Path | None = None,
) -> None:
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path else {}
    reference_sha = str(reference.get("reference_sha256") or "")
    if blind.get("reference_sha256", "") != reference_sha:
        raise ValueError("Judge reference pack does not match blind packet")
    prompt = build_codex_judge_prompt(blind, reference)
    audit_path = audit_path or events_path.with_name("human_ai_judge_audit.json")
    with tempfile.TemporaryDirectory(prefix="human-ai-judge-") as temp_dir:
        temp = Path(temp_dir)
        last_message = temp / "last_message.json"
        command = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "workspace-write", "--cd", str(temp), "--skip-git-repo-check",
            "--json", "--output-last-message", str(last_message), "-",
        ]
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        events = []
        event_parse_errors = []
        for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                event_parse_errors.append({"line": line_number, "error": str(exc)})
        last_message_raw = last_message.read_text(encoding="utf-8") if last_message.exists() else ""
        _write_private(audit_path, {
            "schema_version": 1,
            "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
            "record_kind": "human_ai_judge_invocation_audit",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_status": "captured_at_invocation",
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "argv": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "last_message": last_message_raw,
            "events": events,
            "event_parse_errors": event_parse_errors,
            "tool_call_count": sum(_event_has_tool_call(event) for event in events),
            "blind_sha256": _json_sha256(blind),
            "reference_sha256": reference_sha,
        })
        if completed.returncode:
            detail = completed.stderr.strip()[-2000:]
            raise RuntimeError(
                f"Codex Judge failed with exit code {completed.returncode}: {detail}"
            )
        if event_parse_errors:
            raise ValueError("Codex Judge emitted a non-JSON event line")
        validate_judge_events(events)
        predictions = json.loads(last_message_raw)
    predictions["schema_version"] = 2
    predictions["batch_id"] = blind["batch_id"]
    predictions["blind_sha256"] = _json_sha256(blind)
    predictions["judge_events_sha256"] = _json_sha256(events)
    predictions["reference_sha256"] = reference_sha
    _write_private(predictions_path, predictions)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8")
    events_path.chmod(PRIVATE_MODE)


def _assert_c0() -> str:
    digest = hashlib.sha256((PROJECT_ROOT / "src/reply/generator.py").read_bytes()).hexdigest()
    if digest != PROTECTED_C0_SHA256:
        raise ValueError(f"protected C0 mismatch: {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="真人回复与当前系统回复匿名二选一 Judge 盲测")
    parser.add_argument("--per-relation", type=int, default=80)
    parser.add_argument("--max-valid", type=int, default=40)
    parser.add_argument("--min-valid", type=int, default=30)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--reblind", action="store_true")
    parser.add_argument("--score", type=Path)
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--run-api-judge", action="store_true")
    parser.add_argument("--run-codex-api-judge", action="store_true")
    parser.add_argument("--official-judge", action="store_true")
    parser.add_argument("--codex-max-cases", type=int)
    parser.add_argument("--codex-workers", type=int, default=1)
    parser.add_argument("--codex-judge-profile", type=Path)
    parser.add_argument("--codex-only-case-id", action="append")
    parser.add_argument("--codex-dynamic-few-shot-manifest", type=Path)
    parser.add_argument(
        "--codex-dynamic-few-shot-mode",
        choices=("append", "replace", "replace_on_match"),
        default="append",
    )
    parser.add_argument("--codex-baseline-predictions", type=Path)
    parser.add_argument("--codex-baseline-audit", type=Path)
    parser.add_argument("--codex-prior-dynamic-few-shot-manifest", type=Path)
    parser.add_argument("--codex-prior-predictions", type=Path)
    parser.add_argument("--codex-prior-audit", type=Path)
    parser.add_argument("--presentation", choices=("canonical", "swapped"), default="canonical")
    parser.add_argument("--score-api-judge", action="store_true")
    parser.add_argument("--judge-events", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_events.jsonl")
    parser.add_argument("--judge-audit", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_audit.json")
    parser.add_argument("--api-judge-audit", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_api_independent_audit.json")
    parser.add_argument("--api-judge-predictions", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_api_independent_predictions.json")
    parser.add_argument("--api-judge-result", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_api_independent_result.json")
    parser.add_argument("--codex-api-judge-audit", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_codex_independent_audit.json")
    parser.add_argument("--codex-api-judge-predictions", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_codex_independent_predictions.json")
    parser.add_argument("--exports", type=Path, default=PROJECT_ROOT / "data/exports/b")
    parser.add_argument("--wiki-dir", type=Path, default=PROJECT_ROOT / "data/memory/wiki/users")
    parser.add_argument("--replay", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_replay.json")
    parser.add_argument("--blind", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_blind.json")
    parser.add_argument("--answers", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_answers.json")
    parser.add_argument("--reference", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_reference.json")
    parser.add_argument("--result", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_result.json")
    parser.add_argument("--generation-checkpoint", type=Path, default=PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_generation_checkpoint.json")
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    args = parser.parse_args()

    official_metadata = None
    if args.official_judge:
        if not args.run_codex_api_judge:
            parser.error("--official-judge requires --run-codex-api-judge")
        if args.codex_judge_profile is not None:
            parser.error("--official-judge does not allow --codex-judge-profile")
        if args.codex_dynamic_few_shot_manifest is not None or args.codex_dynamic_few_shot_mode != "append":
            parser.error("--official-judge does not allow dynamic few-shot options")
        if any(
            path is not None
            for path in (
                args.codex_baseline_predictions,
                args.codex_baseline_audit,
                args.codex_prior_dynamic_few_shot_manifest,
                args.codex_prior_predictions,
                args.codex_prior_audit,
            )
        ):
            parser.error("--official-judge does not allow historical result reuse")
        if args.presentation != "canonical":
            parser.error("--official-judge requires --presentation canonical")
        official_config = load_current_judge(require_runnable=True)
        args.codex_judge_profile = Path(official_config["profile_path"])
        args.blind = PROJECT_ROOT / official_config["dataset"]["path"]
        args.answers = PROJECT_ROOT / official_config["metadata_sidecars"]["answers"]["path"]
        args.replay = PROJECT_ROOT / official_config["metadata_sidecars"]["replay"]["path"]
        args.reference = PROJECT_ROOT / official_config["reference"]["path"]
        official_metadata = official_judge_metadata(official_config)

    if args.min_valid < 1 or args.min_valid > args.max_valid:
        parser.error("--min-valid must be between 1 and --max-valid")

    if args.run_judge:
        run_codex_judge(
            args.blind,
            args.score or args.result.with_name("human_ai_judge_revised_predictions.json"),
            args.judge_events,
            args.judge_audit,
            args.reference,
        )
        return

    if args.run_api_judge:
        run_api_judge_independently(
            args.blind,
            args.api_judge_predictions,
            args.api_judge_audit,
            args.reference,
            args.volcengine_config,
        )
        return

    if args.run_codex_api_judge:
        progress = run_codex_judge_independently(
            args.blind,
            args.codex_api_judge_predictions,
            args.codex_api_judge_audit,
            args.reference,
            args.codex_max_cases,
            args.codex_workers,
            args.codex_judge_profile,
            set(args.codex_only_case_id or []),
            args.codex_dynamic_few_shot_manifest,
            args.codex_baseline_predictions,
            args.codex_baseline_audit,
            args.codex_dynamic_few_shot_mode,
            args.codex_prior_dynamic_few_shot_manifest,
            args.codex_prior_predictions,
            args.codex_prior_audit,
            args.presentation,
            official_metadata,
        )
        print(json.dumps(progress, ensure_ascii=False))
        return

    if args.score_api_judge:
        score = score_api_judge_predictions(
            json.loads(args.blind.read_text(encoding="utf-8")),
            json.loads(args.answers.read_text(encoding="utf-8")),
            json.loads(args.api_judge_predictions.read_text(encoding="utf-8")),
            json.loads(args.api_judge_audit.read_text(encoding="utf-8")),
        )
        _write_private(args.api_judge_result, score)
        print(json.dumps(
            {key: score[key] for key in ("status", "answered", "total", "correct", "accuracy")},
            ensure_ascii=False,
        ))
        return

    if args.score:
        events = _load_json_lines(args.judge_events) if args.judge_events.exists() else None
        score = score_predictions(
            json.loads(args.answers.read_text(encoding="utf-8")),
            json.loads(args.score.read_text(encoding="utf-8")),
            events,
        )
        score["surface_baselines"] = score_surface_baselines(
            json.loads(args.blind.read_text(encoding="utf-8")),
            json.loads(args.answers.read_text(encoding="utf-8")),
        )
        _write_private(args.result, score)
        print(json.dumps({key: score[key] for key in ("status", "answered", "total", "correct", "accuracy")}, ensure_ascii=False))
        return

    if args.reblind:
        report = json.loads(args.replay.read_text(encoding="utf-8"))
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        cases = report.get("cases") or []
        write_benchmark_artifacts(
            cases,
            report,
            minimum_valid=args.min_valid,
            replay_path=args.replay,
            blind_path=args.blind,
            answers_path=args.answers,
            reference_sha256=reference["reference_sha256"],
        )
        print(json.dumps({"reblinded_cases": len(cases)}, ensure_ascii=False))
        return

    c0_before = _assert_c0()
    candidates = extract_candidates(args.exports, args.wiki_dir, human_before=CUTOFF)
    excluded = _few_shot_identities(PROJECT_ROOT / "data")
    exposed_case_ids = _previously_exposed_case_ids(args.blind.parent)
    selected = select_holdout_candidates(
        candidates,
        excluded,
        args.per_relation,
        exposed_case_ids,
    )
    selected_chat_ids = {row.chat_id for row in selected}
    selected_replies = {_normalized_reply(row.replies) for row in selected}
    reference = build_judge_reference_pack(
        candidates,
        excluded_chat_ids=selected_chat_ids,
        excluded_replies=selected_replies,
    )
    validate_reference_isolation(reference, selected)
    chat_names = _chat_names(args.exports)
    export_payloads = _export_payloads(args.exports)
    validation = {
        "c0_generator_sha256": c0_before,
        "raw_candidate_count": len(candidates),
        "few_shot_exact_rows_excluded": len(excluded),
        "previously_exposed_cases_excluded": len(exposed_case_ids),
        "selected_holdout_count": len(selected),
        "selected_relationships": dict(Counter("group" if row.relation == "group" else "private" for row in selected)),
        "selected_chat_names_resolved": sum(row.chat_id in chat_names for row in selected),
        "target_reply_passed_to_generator": False,
        "bot_constructed": False,
        "memory_enabled": True,
        "judge_worker_enabled": False,
        "action_tools_registered": False,
        "reference_example_count": len(reference["examples"]),
        "reference_whole_test_chats_excluded": True,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    _load_env()
    os.environ["ENABLE_SELF_REFINE"] = "0"
    os.environ["ENABLE_FACT_CHECK"] = "1"
    from src.memory import MemoryEngine

    api_key, base_url = load_volcengine_credentials(args.volcengine_config)
    llm_client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    memory_engine = MemoryEngine(llm_client=llm_client)
    checkpoint = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "c0_generator_sha256": c0_before,
        "entries": {},
    }
    if args.generation_checkpoint.exists():
        loaded_checkpoint = json.loads(args.generation_checkpoint.read_text(encoding="utf-8"))
        if loaded_checkpoint.get("c0_generator_sha256") != c0_before:
            raise ValueError("generation checkpoint belongs to a different C0 version")
        checkpoint = loaded_checkpoint
    entries = checkpoint.setdefault("entries", {})
    replay_cases = []
    invalid_cases = []
    try:
        for candidate in selected:
            source_case_id = _case_id(candidate)
            cached = entries.get(source_case_id)
            if cached:
                if cached.get("status") == "valid":
                    replay_cases.append(cached["replay_case"])
                elif cached.get("status") == "degenerate":
                    invalid_cases.append({"source_case_id": source_case_id, "reasons": cached.get("reasons", [])})
                if len(replay_cases) >= args.max_valid:
                    break
                continue
            payload = export_payloads.get(candidate.chat_id)
            if not payload:
                continue
            try:
                output = _run_c0(candidate, payload, llm_client, memory_engine)
            except Exception as exc:
                entries[source_case_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                _write_private(args.generation_checkpoint, checkpoint)
                continue
            if output["generation_failed"] or not output["replies"]:
                entries[source_case_id] = {
                    "status": "generation_failed" if output["generation_failed"] else "silent",
                    "duration_seconds": output["duration_seconds"],
                    "input_metadata": output["input_metadata"],
                }
                _write_private(args.generation_checkpoint, checkpoint)
                continue
            degenerate_reasons = detect_degenerate_case(candidate.replies, output)
            if degenerate_reasons:
                entries[source_case_id] = {"status": "degenerate", "reasons": degenerate_reasons}
                invalid_cases.append({"source_case_id": source_case_id, "reasons": degenerate_reasons})
                _write_private(args.generation_checkpoint, checkpoint)
                continue
            replay_case = {
                "case_id": source_case_id,
                "relationship": "group" if candidate.relation == "group" else "private",
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
            }
            entries[source_case_id] = {"status": "valid", "replay_case": replay_case}
            replay_cases.append(replay_case)
            _write_private(args.generation_checkpoint, checkpoint)
            if len(replay_cases) >= args.max_valid:
                break
    finally:
        memory_engine.shutdown()
    report = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "human_ai_reply_discrimination_replay",
        "generator_label": "protected C0 ReplyGenerator + Volcengine Anthropic adapter + current memory/few-shot configuration",
        "backend": {
            "provider": "volcengine",
            "requested_model": llm_client.model,
            "resolved_model": llm_client.resolved_model,
            "protocol": "Anthropic messages adapter for the production ReplyGenerator interface",
        },
        "validation": validation,
        "invalid_degenerate_cases": invalid_cases,
        "generation_outcomes": dict(Counter(row.get("status", "unknown") for row in entries.values())),
        "cases": replay_cases,
        "c0_sha256_after": _assert_c0(),
    }
    write_benchmark_artifacts(
        replay_cases,
        report,
        minimum_valid=args.min_valid,
        replay_path=args.replay,
        blind_path=args.blind,
        answers_path=args.answers,
        reference_sha256=reference["reference_sha256"],
    )
    _write_private(args.reference, reference)
    print(json.dumps({"valid_cases": len(replay_cases), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
