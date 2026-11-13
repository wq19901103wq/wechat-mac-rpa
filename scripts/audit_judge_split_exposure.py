#!/usr/bin/env python3
"""Audit frozen Judge split IDs against prior evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _identifiers(payload: Any) -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        case_id = payload.get("case_id")
        if isinstance(case_id, str):
            yield "case", case_id
        for field in ("event_id", "event_cluster"):
            event_id = payload.get(field)
            if isinstance(event_id, str):
                yield "event", event_id
        for value in payload.values():
            yield from _identifiers(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _identifiers(value)


def collect_exposure(
    roots: Iterable[Path], *, ignored: set[Path] | None = None
) -> tuple[set[str], set[str], list[str]]:
    ignored = {path.resolve() for path in (ignored or set())}
    exposed_ids: set[str] = set()
    exposed_events: set[str] = set()
    scanned_files: list[str] = []
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
            identifiers = set(_identifiers(payload))
            if identifiers:
                exposed_ids.update(value for kind, value in identifiers if kind == "case")
                exposed_events.update(value for kind, value in identifiers if kind == "event")
                scanned_files.append(str(path))
    return exposed_ids, exposed_events, scanned_files


def audit_exposure(
    ledger: dict[str, Any],
    exposed_ids: set[str],
    exposed_events: set[str] | None = None,
) -> dict[str, Any]:
    exposed_events = exposed_events or set()
    splits: dict[str, Any] = {}
    for name, split in ledger["splits"].items():
        ids = split["case_ids"]
        events = split.get("event_clusters") or []
        case_overlap = [case_id for case_id in ids if case_id in exposed_ids]
        event_overlap = [event_id for event_id in events if event_id in exposed_events]
        splits[name] = {
            "count": len(ids),
            "previously_exposed": len(case_overlap),
            "unexposed": len(ids) - len(case_overlap),
            "case_id_previously_exposed": len(case_overlap),
            "event_cluster_previously_exposed": len(event_overlap),
            "eligible_as_unseen": not case_overlap and not event_overlap,
        }
    return {
        "status": (
            "clean" if all(row["eligible_as_unseen"] for row in splits.values())
            else "contaminated"
        ),
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/reports/private/judge_dataset_splits_v1/private_ledger.json"),
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("data/reports/private"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/private/judge_dataset_splits_v1/exposure_audit.json"),
    )
    args = parser.parse_args()

    roots = [
        args.reports_root / "human_ai_judge_benchmark_v2",
        *sorted(args.reports_root.glob("strict_historical_judge_acceptance_v*")),
    ]
    exposed_ids, exposed_events, scanned_files = collect_exposure(roots)

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    result = audit_exposure(ledger, exposed_ids, exposed_events)
    result["ledger"] = str(args.ledger)
    result["prior_artifact_files_with_case_ids"] = len(scanned_files)
    result["distinct_previously_exposed_case_ids"] = len(exposed_ids)
    result["distinct_previously_exposed_event_clusters"] = len(exposed_events)
    result["exclusion_summary"] = {
        "case_ids": len(exposed_ids),
        "event_clusters": len(exposed_events),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "clean" else 2


if __name__ == "__main__":
    raise SystemExit(main())
