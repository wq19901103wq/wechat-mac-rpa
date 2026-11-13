"""Deterministic, stratified sampling for the private Human Gold review set."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Any


def _candidate_action(case: dict[str, Any]) -> str | None:
    action = case["historical_system_evidence"]["action"]
    if action == "replied":
        return "reply"
    if action == "filtered_before_generation":
        return "silence"
    return None


def build_human_gold_packet(
    scenario_packet: dict[str, Any], sample_size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample across action and chat type; never synthesize a human label."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    for case in scenario_packet.get("cases", []):
        action = _candidate_action(case)
        if action is None:
            excluded[case["historical_system_evidence"]["action"]] += 1
            continue
        chat_type = "group" if case["is_group"] else "private"
        strata[(action, chat_type)].append(case)

    queues = {
        key: deque(sorted(cases, key=lambda case: case["case_id"]))
        for key, cases in sorted(strata.items())
    }
    selected = []
    while queues and len(selected) < sample_size:
        for key in list(queues):
            if queues[key]:
                selected.append(queues[key].popleft())
                if len(selected) == sample_size:
                    break
            if not queues[key]:
                del queues[key]

    review_cases = []
    for case in selected:
        action = _candidate_action(case)
        evidence = case["historical_system_evidence"]
        review_cases.append(
            {
                "case_id": case["case_id"],
                "is_group": case["is_group"],
                "pre_context": case["pre_context"],
                "target_messages": case["target_messages"],
                "candidate": {
                    "action": action,
                    "replies": evidence["replies"] if action == "reply" else [],
                    "source": "historical system evidence used only as a candidate to review",
                },
                "human_review": {
                    "status": "pending",
                    "action": None,
                    "experience_grade": None,
                    "reason": None,
                    "relationship_and_context_fit": None,
                    "information_emotion_or_momentum_value": None,
                    "stale_joke_or_repetition": None,
                    "fact_identity_or_boundary_risk": None,
                },
            }
        )

    selected_counts = Counter(
        (
            case["candidate"]["action"],
            "group" if case["is_group"] else "private",
        )
        for case in review_cases
    )
    generated_at = datetime.now().astimezone().isoformat()
    private_packet = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "human_gold_reply_quality",
        "source_fingerprint": scenario_packet.get("source_fingerprint"),
        "selection": "deterministic round-robin across reply/silence and group/private",
        "cases": review_cases,
        "generated_at": generated_at,
    }
    public_manifest = {
        "schema_version": 1,
        "benchmark_kind": "human_gold_reply_quality",
        "source_fingerprint": scenario_packet.get("source_fingerprint"),
        "requested_cases": sample_size,
        "selected_cases": len(review_cases),
        "selected_strata": {
            f"{action}_{chat_type}": count
            for (action, chat_type), count in sorted(selected_counts.items())
        },
        "excluded_ambiguous_actions": dict(sorted(excluded.items())),
        "completed_human_reviews": 0,
        "status": "human_review_pending",
        "generated_at": generated_at,
    }
    return public_manifest, private_packet
