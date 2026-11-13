"""Build a fixed, private benchmark for the current reply system.

Historical Bot replies are evidence attached to a scenario, never the score of
the current system.  A current-system baseline exists only after the current
system has replayed the fixed scenarios and humans have reviewed those runs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.benchmarks.reply_history_quality import (
    Message,
    _json_list,
    _parse_messages,
    _parse_replies,
    _private_message,
    _style_profile,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReplyOpportunity:
    db_id: int
    tick_id: int
    created_at: str
    chat_key: str
    is_group: bool
    input_messages: tuple[Message, ...]
    target_messages: tuple[Message, ...]
    historical_should_reply: bool
    historical_send_success: bool
    historical_replies: tuple[str, ...]
    historical_human_is_badcase: int | None
    historical_human_badcase_type: str | None
    historical_human_notes: str | None


def _opportunity_key(opportunity: ReplyOpportunity) -> str:
    targets = [
        [message.role, message.text, message.timestamp, message.content_type]
        for message in opportunity.target_messages
    ]
    # The chat identity is hashed with the target messages and never published.
    payload = [opportunity.chat_key, targets]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_reply_opportunities(db_path: Path) -> tuple[list[ReplyOpportunity], int]:
    """Read every archived reply opportunity without starting the Bot."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        total_ticks = connection.execute("SELECT COUNT(*) FROM tick_log").fetchone()[0]
        rows = connection.execute(
            """
            SELECT id, tick_id, created_at, chat_name, is_group,
                   session_input_messages_json, session_output_unreplied_json,
                   should_reply, send_success, replies_sent_json,
                   human_is_badcase, human_badcase_type, human_notes
            FROM tick_log
            ORDER BY created_at, id
            """
        ).fetchall()
    finally:
        connection.close()

    opportunities = []
    for row in rows:
        target_messages = _parse_messages(row["session_output_unreplied_json"])
        if not target_messages:
            continue
        opportunities.append(
            ReplyOpportunity(
                db_id=row["id"],
                tick_id=row["tick_id"],
                created_at=row["created_at"] or "",
                chat_key=row["chat_name"] or "",
                is_group=bool(row["is_group"]),
                input_messages=_parse_messages(row["session_input_messages_json"]),
                target_messages=target_messages,
                historical_should_reply=bool(row["should_reply"]),
                historical_send_success=bool(row["send_success"]),
                historical_replies=_parse_replies(row["replies_sent_json"]),
                historical_human_is_badcase=row["human_is_badcase"],
                historical_human_badcase_type=row["human_badcase_type"],
                historical_human_notes=row["human_notes"],
            )
        )
    return opportunities, total_ticks


def deduplicate_opportunities(
    opportunities: list[ReplyOpportunity],
) -> list[ReplyOpportunity]:
    """Keep the earliest archived view of each real reply opportunity."""
    unique: dict[str, ReplyOpportunity] = {}
    for opportunity in opportunities:
        unique.setdefault(_opportunity_key(opportunity), opportunity)
    return list(unique.values())


def _human_reference(opportunities: list[ReplyOpportunity]) -> dict[str, Any]:
    seen: dict[str, set[tuple[str, str, float | None, str]]] = defaultdict(set)
    group_messages: list[Message] = []
    private_messages: list[Message] = []
    for opportunity in opportunities:
        destination = group_messages if opportunity.is_group else private_messages
        for message in opportunity.input_messages:
            key = (message.role, message.text, message.timestamp, message.content_type)
            if key in seen[opportunity.chat_key]:
                continue
            seen[opportunity.chat_key].add(key)
            destination.append(message)
    return {
        "source": "deduplicated real historical chat messages from all archived opportunities",
        "group": _style_profile(group_messages),
        "private": _style_profile(private_messages),
        "usage": "style reference only; not an automatic quality label",
    }


def build_fixed_scenario_packet(
    opportunities: list[ReplyOpportunity], total_ticks: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a public manifest and private fixed-scenario review packet."""
    unique = deduplicate_opportunities(opportunities)
    source_fingerprint = hashlib.sha256(
        "\n".join(_opportunity_key(item) for item in unique).encode()
    ).hexdigest()
    cases = []
    for opportunity in unique:
        case_id = hashlib.sha256(
            f"{source_fingerprint}:{_opportunity_key(opportunity)}".encode()
        ).hexdigest()[:20]
        if opportunity.historical_send_success and opportunity.historical_replies:
            historical_action = "replied"
        elif opportunity.historical_should_reply:
            historical_action = "generated_silence_or_send_failed"
        else:
            historical_action = "filtered_before_generation"
        cases.append(
            {
                "case_id": case_id,
                "source_db_id": opportunity.db_id,
                "source_tick_id": opportunity.tick_id,
                "recorded_at": opportunity.created_at,
                "is_group": opportunity.is_group,
                "pre_context": [
                    _private_message(message) for message in opportunity.input_messages[-20:]
                ],
                "target_messages": [
                    _private_message(message) for message in opportunity.target_messages
                ],
                "historical_system_evidence": {
                    "action": historical_action,
                    "replies": list(opportunity.historical_replies),
                    "human_badcase_label": (
                        {
                            "is_badcase": bool(opportunity.historical_human_is_badcase),
                            "type": opportunity.historical_human_badcase_type,
                            "notes": opportunity.historical_human_notes,
                        }
                        if opportunity.historical_human_is_badcase is not None
                        else None
                    ),
                    "usage": "historical comparison evidence only; not the current-system result",
                },
                "current_system_runs": [],
                "human_review": None,
                "judge_fact_safety": None,
            }
        )

    action_counts = Counter(
        case["historical_system_evidence"]["action"] for case in cases
    )
    private_packet = {
        "schema_version": SCHEMA_VERSION,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "fixed_current_reply_system_replay",
        "source_fingerprint": source_fingerprint,
        "rubric": {
            "action": ["reply", "silence"],
            "experience_grade": ["good", "acceptable", "bad", "critical"],
            "dimensions": [
                "reply_decision",
                "context_understanding",
                "human_likeness_and_relationship_fit",
                "information_emotion_or_momentum_value",
                "stale_joke_or_repetition",
                "fact_identity_and_boundary",
            ],
            "judge_usage": "fact and safety hints only; excluded from experience grade",
        },
        "cases": cases,
    }
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_kind": "fixed_current_reply_system_replay",
        "source_fingerprint": source_fingerprint,
        "scope": {
            "database_ticks": total_ticks,
            "archived_reply_opportunities": len(opportunities),
            "deduplicated_fixed_scenarios": len(cases),
            "group_scenarios": sum(case["is_group"] for case in cases),
            "private_scenarios": sum(not case["is_group"] for case in cases),
        },
        "historical_system_evidence": {
            "action_counts": dict(sorted(action_counts.items())),
            "usage": "test material and comparison evidence only",
        },
        "human_chat_reference": _human_reference(opportunities),
        "current_system_baseline": {
            "status": "replay_pending",
            "completed_runs": 0,
            "human_reviewed_scenarios": 0,
            "score": None,
        },
        "interpretation": (
            "This manifest defines the fixed historical scenarios. It is not a quality score. "
            "The current-system baseline is created only after replay and human review."
        ),
    }
    generated_at = datetime.now().astimezone().isoformat()
    public_manifest["generated_at"] = generated_at
    private_packet["generated_at"] = generated_at
    return public_manifest, private_packet


def generate_fixed_scenarios(db_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    opportunities, total_ticks = load_reply_opportunities(db_path)
    return build_fixed_scenario_packet(opportunities, total_ticks)

