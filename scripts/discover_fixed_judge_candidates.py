#!/usr/bin/env python3
"""Seal unexposed fixed-set candidates without model calls or answer output."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import build_adversarial_hidden_holdout as stage1
from scripts import build_adversarial_stage2_holdout as stage2
from scripts.audit_judge_split_exposure import collect_exposure
from scripts.discover_youzi_judge_candidates import (
    _load_reply_exposure,
    _walk_reply_strings,
)
from scripts.private_case_config import load_private_case_config


ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIR = ROOT / "data/reports/private/judge_fixed_candidates_v2"
MANIFEST_PATH = PRIVATE_DIR / "private_manifest.json"
PUBLIC_SUMMARY_PATH = PRIVATE_DIR / "public_summary.json"
YOUZI_CANDIDATES_PATH = (
    ROOT
    / "data/reports/private/adversarial_stage2_youzi_quota_v1/"
    "discovered_candidates_v2/manifest.json"
)
REFERENCE_PATH = (
    ROOT
    / "data/reports/private/human_ai_judge_benchmark_v2/"
    "judge_reference_scenario_few_shot_candidate_v3.json"
)
OLD_STAGE2_PATH = ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json"

CATEGORY_TARGETS = {
    "specified_common": 20,
    "other_group_complex": 62,
    "other_group": 58,
    "private": 80,
}
YOUZI_TARGET = 20
ORDINARY_CHAT_CAP = 3
COMPLEX_CHAT_CAP = 6
COMMON_EXTRA_SOURCES = (
    (
        "common_202602",
        ROOT
        / "data/private_benchmarks/judge_bot_adversarial_sources/20260822_account20b2"
        / "correct_common_prosperity_20260201_20260228.json",
    ),
    (
        "common_20260301_03",
        ROOT
        / "data/private_benchmarks/judge_bot_adversarial_sources/20260822_account20b2"
        / "correct_common_prosperity_20260301_20260303.json",
    ),
)


def _category(case: dict[str, Any]) -> str:
    if case.get("specified_group") == "correct_common_prosperity_with_luxixi":
        return "specified_common"
    if case.get("chat_type") == "private":
        return "private"
    if case.get("complex_large_group"):
        return "other_group_complex"
    return "other_group"


def _tail_is_clean(messages: list[dict[str, Any]], index: int, event_keys: list[int]) -> bool:
    event = event_keys[index]
    previous = [
        messages[cursor]
        for cursor in range(max(0, index - 3), index)
        if event_keys[cursor] == event
    ]
    return bool(previous) and all(
        int(row.get("localType") or 0) == 1
        and stage1._safe_text(str(row.get("content") or ""))
        for row in previous
    )


def _ordinary_candidates(exposed_fingerprints: set[str]) -> list[dict[str, Any]]:
    candidates = stage2._ordinary_candidates(exposed_fingerprints, per_chat_cap=10_000)
    clean: list[dict[str, Any]] = []
    cache: dict[str, tuple[list[dict[str, Any]], list[int]]] = {}
    for case in candidates:
        source_path = str(case["source_path"])
        if source_path not in cache:
            payload = json.loads((ROOT / source_path).read_text(encoding="utf-8"))
            messages = payload.get("messages") or []
            cache[source_path] = messages, stage1._event_keys(messages)
        messages, event_keys = cache[source_path]
        index = int(case["message_index"])
        if _tail_is_clean(messages, index, event_keys):
            clean.append(case)
    return clean


def _common_candidates(exposed_fingerprints: set[str]) -> list[dict[str, Any]]:
    original_source = stage2.COMMON_SOURCE
    cases = _common_candidates_from_source(exposed_fingerprints, original_source)
    try:
        for namespace, source in COMMON_EXTRA_SOURCES:
            if source.exists():
                cases.extend(_common_candidates_from_source(exposed_fingerprints, source, namespace))
    finally:
        stage2.COMMON_SOURCE = original_source
    return cases


def _common_candidates_from_source(
    exposed_fingerprints: set[str], source: Path, namespace: str | None = None
) -> list[dict[str, Any]]:
    config = load_private_case_config('common_talker')
    stage2.COMMON_SOURCE = source
    cases = _attach_common_event_numbers(
        stage2._common_candidates(
            exposed_fingerprints,
            candidate_filter=lambda event, index: _tail_is_clean(
                event, index, [0] * len(event)
            ),
        ),
        source,
    )
    if namespace is None:
        return cases
    return [
        {
            **case,
            "internal_id": hashlib.sha256(
                f"{config['common_talker']}:{namespace}:"
                f"{case['event_cluster_source_number']}:{case['timestamp']}:{case['reply']}".encode()
            ).hexdigest()[:20],
            "event_cluster": hashlib.sha256(
                f"{config['common_talker']}:{namespace}:"
                f"{case['event_cluster_source_number']}".encode()
            ).hexdigest()[:20],
        }
        for case in cases
    ]


def _attach_common_event_numbers(
    cases: list[dict[str, Any]], source: Path | None = None
) -> list[dict[str, Any]]:
    """Resolve the private event ordinal without changing the stable event hash."""
    if all("event_cluster_source_number" in case for case in cases):
        return cases
    config = load_private_case_config('common_talker')
    raw = json.loads((source or stage2.COMMON_SOURCE).read_text(encoding="utf-8"))
    by_hash = {
        hashlib.sha256(f"{config['common_talker']}:{event_number}".encode()).hexdigest()[:20]: event_number
        for event_number, _event in stage2._event_rows(raw.get("messages") or [])
    }
    return [
        {**case, "event_cluster_source_number": by_hash[str(case["event_cluster"])]}
        for case in cases
    ]


def filter_unexposed(
    cases: list[dict[str, Any]],
    *,
    exposed_case_ids: set[str],
    exposed_events: set[str],
    exposed_reply_fingerprints: set[str],
    comparison_replies: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for case in sorted(cases, key=lambda row: (str(row["event_cluster"]), str(row["internal_id"]))):
        reply = str(case["reply"])
        if str(case["internal_id"]) in exposed_case_ids:
            counts["case_id_exposed"] += 1
            continue
        if str(case["event_cluster"]) in exposed_events:
            counts["event_exposed"] += 1
            continue
        if stage2._fingerprint(reply) in exposed_reply_fingerprints:
            counts["reply_exposed"] += 1
            continue
        if any(stage2._near_duplicate(reply, old) for old in comparison_replies):
            counts["reply_near_duplicate"] += 1
            continue
        accepted.append(case)
    return accepted, dict(counts)


def select_fixed_non_youzi(
    cases: list[dict[str, Any]],
    *,
    seed: str,
    blocked_replies: list[str] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        buckets[_category(case)][str(case["chat_id"])].append(case)
    for chats in buckets.values():
        for rows in chats.values():
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}|{row['internal_id']}|{row['event_cluster']}".encode()
                ).hexdigest()
            )

    selected: list[dict[str, Any]] = []
    selected_replies: list[str] = []
    blocked_replies = blocked_replies or []
    used_events: set[str] = set()
    for category, target in CATEGORY_TARGETS.items():
        chats = buckets.get(category, {})
        chat_ids = sorted(
            chats,
            key=lambda chat_id: hashlib.sha256(f"{seed}|{category}|{chat_id}".encode()).hexdigest(),
        )
        picked = 0
        while picked < target:
            progressed = False
            for chat_id in chat_ids:
                if category == "specified_common":
                    cap = target
                elif category == "other_group_complex":
                    cap = COMPLEX_CHAT_CAP
                else:
                    cap = ORDINARY_CHAT_CAP
                already = sum(
                    _category(row) == category and str(row["chat_id"]) == chat_id
                    for row in selected
                )
                if already >= cap:
                    continue
                while chats[chat_id]:
                    row = chats[chat_id].pop(0)
                    if str(row["event_cluster"]) in used_events:
                        continue
                    if any(
                        stage2._near_duplicate(str(row["reply"]), reply)
                        for reply in blocked_replies + selected_replies
                    ):
                        continue
                    selected.append(row)
                    selected_replies.append(str(row["reply"]))
                    used_events.add(str(row["event_cluster"]))
                    picked += 1
                    progressed = True
                    break
                if picked >= target:
                    break
            if not progressed:
                break
    return selected


def _atomic_private_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def build(*, seed: str = "judge-fixed-candidates-v2") -> tuple[dict[str, Any], dict[str, Any]]:
    roots = [
        ROOT / "data/reports/private",
        ROOT / "data/few_shot",
        ROOT / "data/few_shot_v2",
        ROOT / "data/few_shot_v3",
        ROOT / "data/few_shot_v4",
        ROOT / "data/few_shot_v5",
        ROOT / "data/few_shot_verified_humor",
        ROOT / "docs/03-guides",
    ]
    ignored = {MANIFEST_PATH, PUBLIC_SUMMARY_PATH, YOUZI_CANDIDATES_PATH}
    exposed_case_ids, exposed_events, scanned_files = collect_exposure(roots, ignored=ignored)
    exposed_fingerprints, exposed_replies = _load_reply_exposure(roots, ignored=ignored)

    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    old_stage2 = json.loads(OLD_STAGE2_PATH.read_text(encoding="utf-8"))
    comparison_replies = list(_walk_reply_strings(reference)) + [
        str(case["reply"]) for case in old_stage2.get("cases") or []
    ]
    raw_candidates = _ordinary_candidates(exposed_fingerprints)
    clean_common = _common_candidates(exposed_fingerprints)
    raw_candidates.extend(clean_common)
    candidates, exclusions = filter_unexposed(
        raw_candidates,
        exposed_case_ids=exposed_case_ids,
        exposed_events=exposed_events,
        exposed_reply_fingerprints=exposed_fingerprints,
        comparison_replies=[],
    )
    selected = select_fixed_non_youzi(
        candidates,
        seed=seed,
        blocked_replies=comparison_replies,
    )
    selected = [
        {key: value for key, value in case.items() if key != "event_cluster_source_number"}
        for case in selected
    ]
    category_counts = {
        category: sum(_category(case) == category for case in selected)
        for category in CATEGORY_TARGETS
    }
    gaps = {
        category: max(0, target - category_counts[category])
        for category, target in CATEGORY_TARGETS.items()
    }

    youzi = json.loads(YOUZI_CANDIDATES_PATH.read_text(encoding="utf-8"))
    youzi_count = len(youzi.get("cases") or [])
    non_youzi_ready = not any(gaps.values())
    status = "non_youzi_sealed_youzi_pending" if non_youzi_ready else "insufficient_non_youzi_capacity"
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest = {
        "schema_version": 1,
        "status": status,
        "sealed_at": now,
        "seed": seed,
        "selection_policy": {
            "model_calls": 0,
            "one_case_per_event": True,
            "tail_three_messages_text_only": True,
            "case_event_and_reply_exposure_excluded": True,
            "reference_and_old_answer_near_duplicates_excluded": True,
            "ordinary_chat_cap": ORDINARY_CHAT_CAP,
            "complex_group_chat_cap": COMPLEX_CHAT_CAP,
            "wrong_same_name_common_prosperity_excluded": True,
            "youzi_samples_may_not_be_used_for_tuning_or_acceptance_until_complete": True,
        },
        "audit": {
            "artifact_files_with_identifiers": len(scanned_files),
            "exposed_case_ids": len(exposed_case_ids),
            "exposed_event_ids": len(exposed_events),
            "exposed_reply_fingerprints": len(exposed_fingerprints),
            "exclusions": exclusions,
        },
        "coverage": {
            "selected_non_youzi": len(selected),
            "category_counts": category_counts,
            "category_gaps": gaps,
            "distinct_events": len({str(case["event_cluster"]) for case in selected}),
            "distinct_chats": len({str(case["chat_id"]) for case in selected}),
            "sealed_youzi_slots": youzi_count,
            "youzi_target": YOUZI_TARGET,
            "youzi_pending_new_data": max(0, YOUZI_TARGET - youzi_count),
            "future_total_when_youzi_complete": len(selected) + YOUZI_TARGET,
        },
        "youzi_source": str(YOUZI_CANDIDATES_PATH.relative_to(ROOT)),
        "cases": selected,
    }
    _atomic_private_write(MANIFEST_PATH, manifest)
    public = {
        "schema_version": 1,
        "status": status,
        "sealed_at": now,
        "coverage": manifest["coverage"],
        "checks": manifest["selection_policy"],
        "privacy": "sample text, answers, case IDs, event IDs and chat IDs are omitted",
    }
    _atomic_private_write(PUBLIC_SUMMARY_PATH, public)
    return manifest, public


def main() -> None:
    _manifest, public = build()
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
