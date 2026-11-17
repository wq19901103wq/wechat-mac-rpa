#!/usr/bin/env python3
"""Discover unexposed Youzi-group Judge candidates without model calls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from scripts.audit_judge_split_exposure import collect_exposure
from scripts.build_adversarial_stage2_holdout import _near_duplicate
from scripts.private_case_config import load_private_case_config
from scripts.seal_youzi_stage2_quota import (
    CUTOFF_EXCLUSIVE,
    EVENT_GAP_SECONDS,
    _event_hash,
    _events,
    _fingerprint,
    _private_message,
    load_source,
)


ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIR = ROOT / "data/reports/private/adversarial_stage2_youzi_quota_v1/discovered_candidates_v2"
MANIFEST_PATH = PRIVATE_DIR / "manifest.json"
PUBLIC_SUMMARY_PATH = PRIVATE_DIR / "public_summary.json"
REFERENCE_PATH = (
    ROOT
    / "data/reports/private/human_ai_judge_benchmark_v2/"
    "judge_reference_scenario_few_shot_candidate_v3.json"
)
MEDIA_MARKERS = ("[图片]", "[表情]", "[表情包]", "[视频]")


def _walk_reply_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"reply", "human_reply", "option_A", "option_B"}:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, list):
                    yield from (str(part) for part in item if isinstance(part, str))
            yield from _walk_reply_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_reply_strings(item)


def _load_reply_exposure(
    roots: Iterable[Path], *, ignored: set[Path]
) -> tuple[set[str], list[str]]:
    fingerprints: set[str] = set()
    replies: list[str] = []
    ignored = {path.resolve() for path in ignored}
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in paths:
            if path.resolve() in ignored:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for reply in _walk_reply_strings(payload):
                if reply.strip():
                    fingerprints.add(_fingerprint(reply))
                    replies.append(reply)
    return fingerprints, replies


def _clean_text_tail(event: list[dict[str, Any]], index: int) -> bool:
    tail = event[max(0, index - 3) : index]
    return bool(tail) and all(
        int(row.get("localType") or 0) == 1
        and not any(marker in str(row.get("content") or "") for marker in MEDIA_MARKERS)
        for row in tail
    )


def discover_candidates(
    messages: list[dict[str, Any]],
    *,
    exposed_case_ids: set[str],
    exposed_events: set[str],
    exposed_reply_fingerprints: set[str],
    comparison_replies: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    config = load_private_case_config('self_sender')
    cases: list[dict[str, Any]] = []
    selected_replies: list[str] = []
    exclusions = {
        "event_already_exposed": 0,
        "no_identity_eligible_reply": 0,
        "no_clean_text_tail": 0,
        "case_id_already_exposed": 0,
        "reply_already_exposed": 0,
        "reply_near_reference_or_selected": 0,
    }
    for event in _events(messages):
        event_hash = _event_hash(event)
        event_cluster = "youzi_" + event_hash[:20]
        if event_hash in exposed_events or event_cluster in exposed_events:
            exclusions["event_already_exposed"] += 1
            continue

        eligible: list[tuple[int, dict[str, Any]]] = []
        for index, message in enumerate(event):
            reply = str(message.get("content") or "").strip()
            if (
                message.get("isSend")
                and str(message.get("senderUsername") or "") == config['self_sender']
                and int(message.get("localType") or 0) == 1
                and reply
                and int(message.get("createTime") or 0) < CUTOFF_EXCLUSIVE
                and any(not row.get("isSend") for row in event[:index])
            ):
                eligible.append((index, message))
        if not eligible:
            exclusions["no_identity_eligible_reply"] += 1
            continue

        clean = [(index, message) for index, message in eligible if _clean_text_tail(event, index)]
        if not clean:
            exclusions["no_clean_text_tail"] += 1
            continue

        chosen: tuple[int, dict[str, Any]] | None = None
        for index, message in clean:
            internal_id = hashlib.sha256(f"youzi:{event_hash}".encode()).hexdigest()[:20]
            reply = str(message.get("content") or "").strip()
            if internal_id in exposed_case_ids:
                exclusions["case_id_already_exposed"] += 1
                break
            if _fingerprint(reply) in exposed_reply_fingerprints:
                exclusions["reply_already_exposed"] += 1
                continue
            if any(
                _near_duplicate(reply, previous)
                for previous in comparison_replies + selected_replies
            ):
                exclusions["reply_near_reference_or_selected"] += 1
                continue
            chosen = index, message
            break
        if chosen is None:
            continue

        index, message = chosen
        reply = str(message.get("content") or "").strip()
        internal_id = hashlib.sha256(f"youzi:{event_hash}".encode()).hexdigest()[:20]
        selected_replies.append(reply)
        cases.append(
            {
                "internal_id": internal_id,
                "event_cluster": event_cluster,
                "anchor_server_id": str(message.get("serverId") or ""),
                "event_hash": event_hash,
                "timestamp": int(message.get("createTime") or 0),
                "context": [_private_message(row) for row in event[:index]],
                "human_reply": _private_message(message),
                "review": {
                    "identity_verified": True,
                    "complete_pre_reply_event_context": True,
                    "case_and_event_exposure_excluded": True,
                    "tail_three_messages_text_only": True,
                    "reference_and_selected_near_duplicate_excluded": True,
                    "human_semantic_review_pending": True,
                    "model_screening_used": False,
                },
            }
        )
    return cases, exclusions


def _atomic_private_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    messages, source_files = load_source()
    roots = [
        ROOT / "data/reports/private/human_ai_judge_benchmark_v2",
        *sorted((ROOT / "data/reports/private").glob("strict_historical_judge_acceptance_v*")),
        ROOT / "data/reports/private/adversarial_stage2_holdout_v1",
    ]
    ignored = {MANIFEST_PATH, PUBLIC_SUMMARY_PATH}
    exposed_case_ids, exposed_events, scanned_files = collect_exposure(
        roots, ignored=ignored
    )
    exposed_fingerprints, exposed_replies = _load_reply_exposure(
        roots, ignored=ignored
    )
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    reference_replies = list(_walk_reply_strings(reference))
    cases, exclusions = discover_candidates(
        messages,
        exposed_case_ids=exposed_case_ids,
        exposed_events=exposed_events,
        exposed_reply_fingerprints=exposed_fingerprints,
        comparison_replies=reference_replies,
    )
    manifest = {
        "schema_version": 1,
        "status": "machine_filtered_candidates_pending_human_review",
        "selection_policy": {
            "event_gap_seconds": EVENT_GAP_SECONDS,
            "one_case_per_event": True,
            "verified_human_identity": True,
            "cutoff_inclusive": "2026-07-09T23:59:59+08:00",
            "tail_three_messages_text_only": True,
            "case_and_event_exposure_excluded": True,
            "reference_and_few_shot_near_duplicates_excluded": True,
            "model_calls": 0,
        },
        "source_files": source_files,
        "audit": {
            "artifact_files_with_identifiers": len(scanned_files),
            "exposed_case_ids": len(exposed_case_ids),
            "exposed_event_ids": len(exposed_events),
            "exposed_reply_fingerprints": len(exposed_fingerprints),
            "exclusion_counts": exclusions,
        },
        "cases": cases,
    }
    _atomic_private_write(MANIFEST_PATH, manifest)
    public = {
        "schema_version": 1,
        "status": manifest["status"],
        "coverage": {
            "machine_filtered_unexposed_events": len(cases),
            "target_events": 20,
            "gap_before_human_review": max(0, 20 - len(cases)),
        },
        "checks": manifest["selection_policy"],
        "privacy": "sample text, contexts, answers and internal identifiers are omitted",
    }
    _atomic_private_write(PUBLIC_SUMMARY_PATH, public)
    return manifest, public


def main() -> None:
    _manifest, public = build()
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
