"""Offline replay harness for the current reply generator.

The harness reads archived inputs from SQLite, calls only ReplyGenerator, and
never constructs WeChatBot or registers action-capable tools.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.models.base import ChatMessage, SenderType


PRIVATE_MODE = 0o600
TERMINAL_STATUSES = {"completed", "failed"}
REPLAY_SOURCE_PATHS = (
    "src/reply/generator.py",
    "src/reply/few_shot.py",
    "src/utils/qwen_client.py",
    "data/persona.md",
    "data/few_shot/persona_examples.jsonl",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sender_type(value: Any) -> SenderType:
    try:
        return SenderType(str(value or "other").lower())
    except ValueError:
        return SenderType.UNKNOWN


def _chat_message(item: dict[str, Any], chat_name: str) -> ChatMessage:
    return ChatMessage(
        text=str(item.get("text") or ""),
        sender=str(item.get("sender") or "对方"),
        sender_type=_sender_type(item.get("sender_type")),
        chat_name=chat_name,
        chatroom_id=item.get("chatroom_id"),
        is_at_me=bool(item.get("is_at_me")),
        timestamp=item.get("timestamp"),
        replied=bool(item.get("replied")),
        reply_text=str(item.get("reply_text") or ""),
        reply_time=item.get("reply_time"),
        message_type=str(item.get("message_type") or item.get("type") or "text"),
        image_description=str(item.get("image_description") or ""),
        image_text=str(item.get("image_text") or ""),
        is_image_duplicate=bool(item.get("is_image_duplicate")),
        quoted_text=str(item.get("quoted_text") or ""),
        account=str(item.get("account") or ""),
        local_id=item.get("local_id"),
        server_id=item.get("server_id"),
        create_time=item.get("create_time"),
        raw_type=item.get("raw_type"),
        sender_wxid=item.get("sender_wxid"),
    )


def _message_list(raw: str | None, chat_name: str) -> list[ChatMessage]:
    try:
        values = json.loads(raw or "[]")
    except json.JSONDecodeError:
        values = []
    if not isinstance(values, list):
        return []
    return [_chat_message(item, chat_name) for item in values if isinstance(item, dict)]


def load_replay_cases(
    db_path: Path, scenario_path: Path, selection_path: Path
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve selected case IDs to their original private tick inputs."""
    scenarios = _json_object(scenario_path)
    selection = _json_object(selection_path)
    if scenarios.get("source_fingerprint") != selection.get("source_fingerprint"):
        raise ValueError("scenario and selection fingerprints differ")

    selected_ids = [case["case_id"] for case in selection.get("cases", [])]
    scenario_by_id = {case["case_id"]: case for case in scenarios.get("cases", [])}
    missing = [case_id for case_id in selected_ids if case_id not in scenario_by_id]
    if missing:
        raise ValueError(f"selected cases missing from scenario packet: {missing[:3]}")

    source_ids = [int(scenario_by_id[case_id]["source_db_id"]) for case_id in selected_ids]
    placeholders = ",".join("?" for _ in source_ids)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"""
            SELECT id, tick_id, created_at, chat_name, is_group,
                   session_input_messages_json, session_output_unreplied_json
            FROM tick_log WHERE id IN ({placeholders})
            """,
            source_ids,
        ).fetchall()
    finally:
        connection.close()
    rows_by_id = {int(row["id"]): row for row in rows}

    result = []
    for case_id in selected_ids:
        scenario = scenario_by_id[case_id]
        row = rows_by_id.get(int(scenario["source_db_id"]))
        if row is None:
            raise ValueError(f"source tick missing for case {case_id}")
        chat_name = str(row["chat_name"] or "")
        evaluation_time = datetime.fromisoformat(str(row["created_at"]))
        result.append(
            {
                "case_id": case_id,
                "source_db_id": int(row["id"]),
                "source_tick_id": int(row["tick_id"]),
                "is_group": bool(row["is_group"]),
                "evaluation_time": evaluation_time,
                "all_messages": _message_list(row["session_input_messages_json"], chat_name),
                "unreplied": _message_list(row["session_output_unreplied_json"], chat_name),
            }
        )
    return str(scenarios.get("source_fingerprint") or ""), result


def system_fingerprint(project_root: Path, client: Any) -> dict[str, Any]:
    files = [project_root / path for path in REPLAY_SOURCE_PATHS]
    files.extend(sorted((project_root / "skills").glob("*/SKILL.md")))
    digest = hashlib.sha256()
    for path in files:
        if path.exists():
            digest.update(str(path.relative_to(project_root)).encode())
            digest.update(path.read_bytes())
    config = {
        "model": str(getattr(client, "model", "unknown")),
        "enable_self_refine": os.environ.get("ENABLE_SELF_REFINE", "1"),
        "enable_fact_check": os.environ.get("ENABLE_FACT_CHECK", "1"),
        "enable_persona_few_shots": os.environ.get("ENABLE_PERSONA_FEW_SHOTS", "0"),
        "persona_few_shot_count": os.environ.get("PERSONA_FEW_SHOT_COUNT", "8"),
        "tool_profile": "isolated_empty_registry",
        "memory_profile": "disabled_for_safe_replay",
        "judge_worker": "disabled",
    }
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return {"code_prompt_sha256": digest.hexdigest(), "config_sha256": config_digest, **config}


def private_source_snapshot(project_root: Path) -> dict[str, str]:
    """Archive exact replay inputs locally so a hash is not mistaken for restorable source."""
    paths = [project_root / path for path in REPLAY_SOURCE_PATHS]
    paths.extend(sorted((project_root / "skills").glob("*/SKILL.md")))
    return {
        str(path.relative_to(project_root)): path.read_text(encoding="utf-8")
        for path in paths
        if path.exists()
    }


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


def replay_cases(
    cases: list[dict[str, Any]],
    generator_factory: Callable[[], Any],
    output_path: Path,
    source_fingerprint: str,
    system: dict[str, Any],
    *,
    limit: int | None = None,
    workers: int = 1,
    retry_generation_failures: bool = False,
    source_snapshot: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run independent cases with a private, resumable checkpoint."""
    run_key = hashlib.sha256(
        json.dumps([source_fingerprint, system], sort_keys=True).encode()
    ).hexdigest()
    if output_path.exists():
        packet = _json_object(output_path)
        if packet.get("run_key") != run_key:
            raise ValueError("existing replay checkpoint belongs to another system fingerprint")
    else:
        packet = {
            "schema_version": 1,
            "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
            "benchmark_kind": "current_reply_system_replay",
            "status": "running",
            "source_fingerprint": source_fingerprint,
            "run_key": run_key,
            "system": system,
            "source_snapshot": source_snapshot,
            "limitations": [
                "Action-capable tools and memory are disabled in the safe replay profile.",
                "AI provisional review is not Human Gold.",
            ],
            "cases": [],
        }
    if retry_generation_failures:
        packet["cases"] = [
            item
            for item in packet.get("cases", [])
            if item.get("status") != "failed" and not item.get("generation_failed")
        ]
    completed = {
        item["case_id"]
        for item in packet.get("cases", [])
        if item.get("status") in TERMINAL_STATUSES
    }
    pending = [case for case in cases if case["case_id"] not in completed]
    if limit is not None:
        pending = pending[:limit]

    def run_one(case: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result: dict[str, Any] = {
            "case_id": case["case_id"],
            "source_tick_id": case["source_tick_id"],
            "evaluation_time": case["evaluation_time"].isoformat(),
        }
        try:
            generator = generator_factory()
            replies = generator.generate(
                case["unreplied"],
                case["all_messages"],
                is_group=case["is_group"],
                tick_id=case["source_tick_id"],
                evaluation_time=case["evaluation_time"],
            )
            result.update(
                {
                    "status": "completed",
                    "action": "reply" if replies else "silence",
                    "replies": replies,
                    "generation_failed": bool(generator.last_generation_failed),
                    "active_llm": generator.last_active_llm,
                    "loaded_skills": generator.last_loaded_skills,
                    "self_refine": {
                        "applied": generator.last_self_refine_applied,
                        "decision": generator.last_feedback_decision,
                        "issues": generator.last_feedback_issues,
                        "iterate_count": generator.last_iterate_count,
                    },
                    "llm_call_count": len(generator.last_llm_calls),
                    "tool_call_count": len(generator.last_tool_calls),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "action": None,
                    "replies": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        return result

    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, case): case["case_id"] for case in pending}
        for future in as_completed(futures):
            packet["cases"].append(future.result())
            _write_private(output_path, packet)

    terminal = sum(
        item.get("status") in TERMINAL_STATUSES for item in packet.get("cases", [])
    )
    packet["status"] = "completed" if terminal == len(cases) else "partial"
    packet["completed_cases"] = terminal
    packet["expected_cases"] = len(cases)
    _write_private(output_path, packet)
    return packet


def public_replay_report(packet: dict[str, Any]) -> dict[str, Any]:
    cases = packet.get("cases", [])
    return {
        "schema_version": 1,
        "benchmark_kind": "current_reply_system_replay",
        "status": packet.get("status"),
        "source_fingerprint": packet.get("source_fingerprint"),
        "system": packet.get("system"),
        "expected_cases": packet.get("expected_cases"),
        "completed_cases": packet.get("completed_cases"),
        "successful_cases": sum(item.get("status") == "completed" for item in cases),
        "failed_cases": sum(item.get("status") == "failed" for item in cases),
        "reply_cases": sum(item.get("action") == "reply" for item in cases),
        "silence_cases": sum(item.get("action") == "silence" for item in cases),
        "generation_failures": sum(bool(item.get("generation_failed")) for item in cases),
        "quality_status": "ai_provisional_review_pending",
        "eligible_for_automatic_ab": False,
        "interpretation": (
            "This is the current generator's safe offline replay result, not a quality score. "
            "AI provisional review must be completed before it becomes a rough baseline."
        ),
    }
