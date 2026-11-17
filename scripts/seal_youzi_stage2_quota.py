#!/usr/bin/env python3
"""Seal the reviewed Youzi-group Stage 2 quota without model calls."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.private_case_config import load_private_case_config  # noqa: E402


SOURCE_DIR = (
    ROOT
    / "data/private_benchmarks/judge_bot_adversarial_sources/20260818_account20b2"
)
SOURCE_GLOB = "youzi_group_2_page5k_*.json"
PRIVATE_DIR = ROOT / "data/reports/private/adversarial_stage2_youzi_quota_v1"
MANIFEST_PATH = PRIVATE_DIR / "manifest.json"
PUBLIC_SUMMARY_PATH = PRIVATE_DIR / "public_summary.json"
EVENT_GAP_SECONDS = 30 * 60
CUTOFF_EXCLUSIVE = int(
    datetime(2026, 7, 10, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
)

EXPOSURE_ROOTS = (
    ROOT / "data/reports/private",
    ROOT / "data/few_shot",
    ROOT / "data/few_shot_v2",
    ROOT / "data/few_shot_v3",
    ROOT / "data/few_shot_v4",
    ROOT / "data/few_shot_v5",
    ROOT / "data/few_shot_verified_humor",
    ROOT / "data/debug/prompts",
    ROOT / "data/benchmark_history",
    ROOT / "data/memory/wiki",
    ROOT / "docs/03-guides",
)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE).lower()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def collect_exposed_fingerprints() -> tuple[set[str], int]:
    fingerprints: set[str] = set()
    files_scanned = 0
    ignored = {MANIFEST_PATH.resolve(), PUBLIC_SUMMARY_PATH.resolve()}
    for base in EXPOSURE_ROOTS:
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if not path.is_file() or path.resolve() in ignored:
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".log"}:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            files_scanned += 1
            values: Iterable[str]
            if path.suffix.lower() == ".json":
                try:
                    values = _walk_strings(json.loads(raw))
                except json.JSONDecodeError:
                    values = raw.splitlines()
            elif path.suffix.lower() == ".jsonl":
                rows: list[str] = []
                for line in raw.splitlines():
                    try:
                        rows.extend(_walk_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        rows.append(line)
                values = rows
            else:
                values = raw.splitlines()
            for value in values:
                normalized = _normalize(value)
                if 2 <= len(normalized) <= 300:
                    fingerprints.add(
                        hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    )
    return fingerprints, files_scanned


def _message_key(message: dict[str, Any]) -> str:
    server_id = str(message.get("serverId") or "").strip()
    if server_id:
        return f"server:{server_id}"
    return "fallback:" + hashlib.sha256(
        json.dumps(
            {
                "localId": message.get("localId"),
                "createTime": message.get("createTime"),
                "senderUsername": message.get("senderUsername"),
                "content": message.get("content"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def load_source() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    config = load_private_case_config('target_chat')
    unique: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, str]] = []
    paths = sorted(SOURCE_DIR.glob(SOURCE_GLOB))
    if not paths:
        raise RuntimeError("Youzi source pages are missing")
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if str(payload.get("talker")) != config['target_chat']:
            raise RuntimeError(f"unexpected talker in {path.name}")
        source_files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        for message in payload.get("messages") or []:
            unique.setdefault(_message_key(message), message)
    messages = sorted(
        unique.values(),
        key=lambda row: (
            int(row.get("createTime") or 0),
            int(row.get("sortSeq") or 0),
            str(row.get("serverId") or ""),
        ),
    )
    return messages, source_files


def _events(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    events: list[list[dict[str, Any]]] = []
    previous_timestamp = 0
    for message in messages:
        timestamp = int(message.get("createTime") or 0)
        if not events or (
            previous_timestamp
            and timestamp
            and timestamp - previous_timestamp > EVENT_GAP_SECONDS
        ):
            events.append([])
        events[-1].append(message)
        if timestamp:
            previous_timestamp = timestamp
    return events


def _event_hash(event: list[dict[str, Any]]) -> str:
    stable_keys = [_message_key(message) for message in event]
    return hashlib.sha256("\n".join(stable_keys).encode("utf-8")).hexdigest()


def _private_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_id": str(message.get("serverId") or ""),
        "timestamp": int(message.get("createTime") or 0),
        "sender": str(message.get("senderUsername") or ""),
        "is_send": bool(message.get("isSend")),
        "local_type": message.get("localType"),
        "content": str(message.get("content") or ""),
    }


def select_cases(
    messages: list[dict[str, Any]], exposed: set[str]
) -> list[dict[str, Any]]:
    config = load_private_case_config('excluded_exposed_anchors', 'selected_anchors', 'self_sender')
    selected = set(config['selected_anchors'])
    excluded = set(config['excluded_exposed_anchors'])
    if selected & excluded:
        raise RuntimeError("selected and excluded anchors overlap")

    resolved: dict[str, tuple[list[dict[str, Any]], int]] = {}
    for event in _events(messages):
        for index, message in enumerate(event):
            anchor = str(message.get("serverId") or "")
            if anchor in selected:
                resolved[anchor] = (event, index)
    missing = selected - set(resolved)
    if missing:
        raise RuntimeError(f"selected anchors missing from source: {sorted(missing)}")

    cases: list[dict[str, Any]] = []
    event_hashes: set[str] = set()
    normalized_replies: set[str] = set()
    for anchor in config['selected_anchors']:
        event, index = resolved[anchor]
        message = event[index]
        timestamp = int(message.get("createTime") or 0)
        reply = str(message.get("content") or "").strip()
        event_hash = _event_hash(event)
        normalized_reply = _normalize(reply)
        if not (
            message.get("isSend")
            and str(message.get("senderUsername") or "") == config['self_sender']
            and int(message.get("localType") or 0) == 1
            and reply
            and timestamp < CUTOFF_EXCLUSIVE
        ):
            raise RuntimeError(f"anchor {anchor} failed identity/date/text checks")
        if not any(not row.get("isSend") for row in event[:index]):
            raise RuntimeError(f"anchor {anchor} has no incoming context")
        if event_hash in event_hashes:
            raise RuntimeError(f"multiple selected replies in event {event_hash}")
        if normalized_reply in normalized_replies:
            raise RuntimeError(f"duplicate selected reply at anchor {anchor}")
        if _fingerprint(reply) in exposed:
            raise RuntimeError(f"selected reply is already exposed: {anchor}")
        event_hashes.add(event_hash)
        normalized_replies.add(normalized_reply)
        cases.append(
            {
                "anchor_server_id": anchor,
                "event_hash": event_hash,
                "timestamp": timestamp,
                "context": [_private_message(row) for row in event[:index]],
                "human_reply": _private_message(message),
                "review": {
                    "identity_verified": True,
                    "complete_pre_reply_event_context": True,
                    "exact_reply_exposure_excluded": True,
                    "not_direct_group_member_repeat": "human_reviewed",
                    "semantic_near_duplicate": "human_reviewed",
                    "model_screening_used": False,
                },
            }
        )
    return sorted(cases, key=lambda row: (row["timestamp"], row["anchor_server_id"]))


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
    config = load_private_case_config('excluded_exposed_anchors')
    messages, source_files = load_source()
    exposed, files_scanned = collect_exposed_fingerprints()
    cases = select_cases(messages, exposed)
    source_max_timestamp = max(int(row.get("createTime") or 0) for row in messages)
    selected_timestamps = [row["timestamp"] for row in cases]
    if len(cases) != 20:
        raise RuntimeError(f"expected 20 selected events, got {len(cases)}")

    manifest = {
        "schema_version": 1,
        "status": "quota_sealed_stage_2_not_built",
        "selection_policy": {
            "group": "Youzi group 2",
            "cutoff_inclusive": "2026-07-09T23:59:59+08:00",
            "source_snapshot_latest_date": datetime.fromtimestamp(
                source_max_timestamp, ZoneInfo("Asia/Shanghai")
            ).date().isoformat(),
            "event_gap_seconds": EVENT_GAP_SECONDS,
            "one_case_per_event": True,
            "human_identity": "isSend=1 and verified self sender",
            "exact_reply_exposure_files_scanned": files_scanned,
            "exact_reply_exposure_fingerprint_count": len(exposed),
            "semantic_near_duplicate_check": "human_reviewed",
            "model_calls": 0,
        },
        "source_files": source_files,
        "coverage": {
            "selected_events": len(cases),
            "distinct_event_hashes": len({row["event_hash"] for row in cases}),
            "earliest_selected_date": datetime.fromtimestamp(
                min(selected_timestamps), ZoneInfo("Asia/Shanghai")
            ).date().isoformat(),
            "latest_selected_date": datetime.fromtimestamp(
                max(selected_timestamps), ZoneInfo("Asia/Shanghai")
            ).date().isoformat(),
        },
        "excluded_exposed_anchor_count": len(config['excluded_exposed_anchors']),
        "excluded_exposed_anchors": list(config['excluded_exposed_anchors']),
        "cases": cases,
    }
    _atomic_private_write(MANIFEST_PATH, manifest)
    manifest_sha256 = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    public = {
        "schema_version": 1,
        "status": "youzi_quota_sealed_stage_2_not_built",
        "coverage": {
            "selected_events": 20,
            "target_events": 20,
            "quota_gap": 0,
            "source_snapshot_latest_date": manifest["selection_policy"][
                "source_snapshot_latest_date"
            ],
            "latest_selected_date": manifest["coverage"]["latest_selected_date"],
        },
        "manifest_sha256": manifest_sha256,
        "privacy": "sample text, context, answers and internal identifiers are private",
        "checks": {
            "verified_human_identity": True,
            "complete_context_reproducible": True,
            "one_case_per_event": True,
            "known_exact_reply_exposure_excluded": True,
            "semantic_near_duplicate_check": "human_reviewed",
            "model_calls": 0,
            "bot_started": False,
            "c0_modified": False,
        },
        "stage_2_overall_ready": False,
    }
    _atomic_private_write(PUBLIC_SUMMARY_PATH, public)
    return manifest, public


def main() -> None:
    _manifest, public = build()
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
