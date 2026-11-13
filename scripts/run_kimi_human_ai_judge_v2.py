#!/usr/bin/env python3
"""Run isolated Codex, Kimi Code, and direct-API Judge comparisons."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from scripts.evaluate_human_ai_judge import (
    PRIVATE_MODE,
    PROJECT_ROOT,
    _event_has_tool_call,
    _load_env,
    _parse_api_judge_response,
    _write_private,
    assert_reference_isolated,
    reference_leakage_details,
)
from scripts.run_human_ai_judge_benchmark_v2 import (
    PRIVATE_DIR,
    build_messages,
    load_profile,
    profile_sha256,
    swap_case,
    verify_frozen_artifacts,
)

KIMI_MODEL = "kimi-for-coding"
KIMI_CLI_MODEL = "kimi-code/kimi-for-coding"
KIMI_API_BASE_URL = "https://api.kimi.com/coding/v1"
CODEX_MODEL = "codex-cli-default"
CODEX_TIMEOUT_SECONDS = 360


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_dataset_shape(
    blind: dict[str, Any],
    *,
    expected_total: int | None = None,
    expected_private: int | None = None,
    expected_group: int | None = None,
) -> dict[str, int]:
    cases = list(blind.get("cases", []))
    counts = {
        "total": len(cases),
        "private": sum(row.get("relationship") == "private" for row in cases),
        "group": sum(row.get("relationship") == "group" for row in cases),
    }
    expected = {
        "total": expected_total,
        "private": expected_private,
        "group": expected_group,
    }
    mismatches = [
        f"{key}={counts[key]} (expected {value})"
        for key, value in expected.items()
        if value is not None and counts[key] != value
    ]
    if mismatches:
        raise ValueError("dataset shape mismatch: " + ", ".join(mismatches))
    return counts


def _response_dump(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return json.loads(response.json())


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


def _api_key() -> str:
    _load_env()
    value = os.getenv("KIMICODE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not value:
        raise ValueError("KIMICODE_API_KEY or OPENAI_API_KEY is required")
    return value


def _run_api(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], int, list[str]]:
    started = time.monotonic()
    client = OpenAI(api_key=_api_key(), base_url=KIMI_API_BASE_URL, timeout=120)
    response = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=messages,
        temperature=1,
        max_tokens=8192,
    )
    elapsed_ms = round((time.monotonic() - started) * 1000)
    message = response.choices[0].message
    raw = message.content or ""
    tool_calls = [str(row) for row in (message.tool_calls or [])]
    return raw, _response_dump(response), elapsed_ms, tool_calls


def _run_cli(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], int, list[str]]:
    with tempfile.TemporaryDirectory(prefix="kimi-judge-single-") as raw_dir:
        temp = Path(raw_dir)
        (temp / "system.md").write_text(messages[0]["content"], encoding="utf-8")
        (temp / "agent.yaml").write_text(
            "version: 1\n"
            "agent:\n"
            "  name: isolated-judge\n"
            "  system_prompt_path: system.md\n"
            "  tools: []\n"
            "  subagents: {}\n",
            encoding="utf-8",
        )
        command = [
            "kimi",
            "--work-dir", str(temp),
            "--agent-file", str(temp / "agent.yaml"),
            "--model", KIMI_CLI_MODEL,
            "--print",
            "--input-format", "text",
            "--output-format", "stream-json",
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            input=messages[1]["content"],
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
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
                parse_errors.append({"line": line_number, "error": str(exc), "text": line})
        tool_events = []
        final_texts = []
        for event in events:
            if event.get("role") == "assistant" and isinstance(event.get("content"), str):
                final_texts.append(event["content"])
            for block in event.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    final_texts.append(block["text"])
                if "tool" in str(block.get("type") or "").lower():
                    tool_events.append(block)
            if event.get("type") in {"assistant", "message", "result"}:
                for key in ("content", "text", "result", "message"):
                    value = event.get(key)
                    if isinstance(value, str) and value.strip():
                        final_texts.append(value)
        raw = final_texts[-1] if final_texts else completed.stdout.strip()
        audit = {
            "argv": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "events": events,
            "event_parse_errors": parse_errors,
            "agent_spec": {
                "system_prompt_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
                "tools": [],
                "subagents": {},
            },
        }
        if completed.returncode:
            raise RuntimeError(f"Kimi Code failed: {completed.stderr[-2000:]}")
        return raw, audit, elapsed_ms, [str(row) for row in tool_events]


def _run_codex(
    messages: list[dict[str, str]],
    *,
    model: str = CODEX_MODEL,
    reasoning_effort: str | None = None,
    timeout_seconds: int = CODEX_TIMEOUT_SECONDS,
) -> tuple[str, dict[str, Any], int, list[str]]:
    prompt = f"<system>\n{messages[0]['content']}\n</system>\n<user>\n{messages[1]['content']}\n</user>"
    with tempfile.TemporaryDirectory(prefix="codex-v2-judge-single-") as raw_dir:
        temp = Path(raw_dir)
        last_message = temp / "last_message.json"
        command = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "read-only", "--cd", str(temp), "--skip-git-repo-check",
            "--json",
        ]
        if model != CODEX_MODEL:
            command.extend(["--model", model])
        if reasoning_effort is not None:
            command.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
        command.extend(["--output-last-message", str(last_message), "-"])
        started = time.monotonic()
        completed = subprocess.run(
            command, input=prompt, text=True, capture_output=True, check=False,
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
                parse_errors.append({"line": line_number, "error": str(exc), "text": line})
        tool_events = [event for event in events if _event_has_tool_call(event)]
        raw = last_message.read_text(encoding="utf-8") if last_message.exists() else ""
        audit = {
            "argv": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "events": events,
            "event_parse_errors": parse_errors,
            "prompt_transport": "system and user text wrapped in one Codex CLI input",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "timeout_seconds": timeout_seconds,
        }
        if completed.returncode:
            raise RuntimeError(f"Codex CLI failed: {completed.stderr[-2000:]}")
        if parse_errors:
            raise ValueError("Codex CLI returned malformed event JSON")
        return raw, audit, elapsed_ms, [str(row) for row in tool_events]


def run(
    provider: str,
    split: str,
    presentation: str,
    profile_path: Path,
    reference_path: Path,
    output_path: Path,
    audit_path: Path,
    case_offset: int,
    max_cases: int | None,
    codex_model: str = CODEX_MODEL,
    codex_reasoning_effort: str | None = None,
    codex_timeout_seconds: int = CODEX_TIMEOUT_SECONDS,
    *,
    freeze_path: Path = PRIVATE_DIR / "freeze.json",
    case_ids: list[str] | None = None,
    expected_total: int | None = None,
    expected_private: int | None = None,
    expected_group: int | None = None,
) -> dict[str, Any]:
    profile = load_profile(profile_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    blind_path, _ = verify_frozen_artifacts(split, freeze, artifact_dir=freeze_path.parent)
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if blind.get("reference_sha256") != reference.get("reference_sha256"):
        raise ValueError("reference pack does not match blind packet")
    dataset_counts = _validate_dataset_shape(
        blind,
        expected_total=expected_total,
        expected_private=expected_private,
        expected_group=expected_group,
    )
    all_cases = list(blind.get("cases", []))
    selected_case_ids = list(dict.fromkeys(case_ids or []))
    if selected_case_ids:
        cases_by_id = {str(row["case_id"]): row for row in all_cases}
        missing = [case_id for case_id in selected_case_ids if case_id not in cases_by_id]
        if missing:
            raise ValueError(f"case ids not found in selected dataset: {missing}")
        selected_cases = [cases_by_id[case_id] for case_id in selected_case_ids]
    else:
        selected_cases = all_cases[case_offset:]
    model = codex_model if provider == "codex-cli" else KIMI_MODEL
    identity = {
        "split": split,
        "presentation": presentation,
        "provider": provider,
        "model": model,
        "reasoning_effort": codex_reasoning_effort if provider == "codex-cli" else None,
        "timeout_seconds": codex_timeout_seconds if provider == "codex-cli" else None,
        "profile_sha256": profile_sha256(profile),
        "blind_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
        "reference_sha256": reference.get("reference_sha256"),
        "reference_file_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "case_offset": case_offset,
        "freeze_sha256": freeze.get("freeze_sha256"),
        "dataset_counts": dataset_counts,
        "selected_case_ids": selected_case_ids or None,
    }
    tools_configured = provider == "codex-cli"
    predictions = {"schema_version": 1, **identity, "cases": []}
    audit = {
        "schema_version": 1,
        **identity,
        "isolation": {
            "one_case_per_process_or_client": True,
            "shared_conversation_state": False,
            "answer_key_loaded_during_judging": False,
            "tools_configured": tools_configured,
            "tool_policy": "any actual tool use invalidates the case" if tools_configured else "no tools configured",
        },
        "profile": profile,
        "calls": [],
    }
    if output_path.exists() or audit_path.exists():
        if not output_path.exists() or not audit_path.exists():
            raise ValueError("resume requires both prediction and audit files")
        predictions = json.loads(output_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if any(predictions.get(key) != value for key, value in identity.items()):
            raise ValueError("existing output identity does not match")
        audit["isolation"]["tools_configured"] = tools_configured
        audit["isolation"]["tool_policy"] = (
            "any actual tool use invalidates the case" if tools_configured else "no tools configured"
        )
        for call in audit.get("calls", []):
            call["tools_configured"] = tools_configured
    completed_ids = {str(row["case_id"]) for row in predictions["cases"]}
    calls_this_time = 0
    for canonical_case in selected_cases:
        case_id = str(canonical_case["case_id"])
        if case_id in completed_ids:
            continue
        if max_cases is not None and calls_this_time >= max_cases:
            break
        presented = swap_case(canonical_case) if presentation == "swapped" else canonical_case
        case_reference, removed_reference_rows = _reference_without_case_leaks(reference, canonical_case)
        messages = build_messages(presented, case_reference, profile)
        recorded_at = datetime.now(timezone.utc).isoformat()
        raw = ""
        response_audit: dict[str, Any] = {}
        elapsed_ms = 0
        tool_events: list[str] = []
        error = None
        try:
            if provider == "codex-cli":
                raw, response_audit, elapsed_ms, tool_events = _run_codex(
                    messages,
                    model=codex_model,
                    reasoning_effort=codex_reasoning_effort,
                    timeout_seconds=codex_timeout_seconds,
                )
            elif provider == "kimi-code":
                raw, response_audit, elapsed_ms, tool_events = _run_cli(messages)
            else:
                raw, response_audit, elapsed_ms, tool_events = _run_api(messages)
            parsed = _parse_api_judge_response(raw)
        except Exception as exc:
            parsed = {"human_option": None, "confidence": None, "reason": "invalid response"}
            error = f"{type(exc).__name__}: {exc}"
        human_option = parsed["human_option"]
        if presentation == "swapped" and human_option in {"A", "B"}:
            human_option = "B" if human_option == "A" else "A"
        predictions["cases"].append({
            "case_id": case_id,
            "human_option": human_option,
            "presented_human_option": parsed["human_option"],
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"),
            "invalid": human_option not in {"A", "B"},
        })
        audit["calls"].append({
            "case_id": case_id,
            "recorded_at_utc": recorded_at,
            "request_messages": messages,
            "request_sha256": _sha256(messages),
            "case_reference_sha256": _sha256(case_reference),
            "removed_reference_rows": removed_reference_rows,
            "answer_fields_in_request": [],
            "other_test_case_ids_in_request": [],
            "tools_configured": tools_configured,
            "tool_events": tool_events,
            "elapsed_ms": elapsed_ms,
            "raw_response": raw,
            "response": response_audit,
            "error": error,
        })
        _write_private(output_path, predictions)
        _write_private(audit_path, audit)
        calls_this_time += 1
    _write_private(output_path, predictions)
    _write_private(audit_path, audit)
    return {**identity, "completed": len(predictions["cases"]), "total": len(selected_cases)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("codex-cli", "kimi-code", "kimi-api"), required=True)
    parser.add_argument("--split", choices=("dev", "validation", "final"), default="validation")
    parser.add_argument("--presentation", choices=("canonical", "swapped"), default="canonical")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=PRIVATE_DIR / "judge_reference.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--codex-model", default=CODEX_MODEL)
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument("--codex-timeout-seconds", type=int, default=CODEX_TIMEOUT_SECONDS)
    parser.add_argument("--freeze-file", type=Path, default=PRIVATE_DIR / "freeze.json")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-private", type=int)
    parser.add_argument("--expected-group", type=int)
    args = parser.parse_args()
    result = run(
        args.provider, args.split, args.presentation, args.profile, args.reference,
        args.output, args.audit, args.case_offset, args.max_cases,
        args.codex_model, args.codex_reasoning_effort, args.codex_timeout_seconds,
        freeze_path=args.freeze_file,
        case_ids=args.case_ids,
        expected_total=args.expected_total,
        expected_private=args.expected_private,
        expected_group=args.expected_group,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
