#!/usr/bin/env python3
"""Build an offline-only persona view from pre-cutoff rows traceable to raw exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_persona_few_shots import extract_candidates  # noqa: E402

CUTOFF = 1774972800.0  # 2026-04-01 00:00:00 Asia/Shanghai
PRIVATE_MODE = 0o600


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("relationship"),
        row.get("chat_id"),
        tuple(row.get("context") or []),
        tuple(row.get("reply") or []),
    )


def _candidate_identity(candidate: Any) -> tuple[Any, ...]:
    return (
        candidate.relation,
        candidate.chat_id,
        tuple(candidate.context),
        tuple(candidate.replies),
    )


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(PRIVATE_MODE)


def build_view(
    source_path: Path,
    export_dir: Path,
    wiki_dir: Path,
    output_dir: Path,
    expected_matches: int | None = None,
) -> dict[str, Any]:
    raw_candidates = extract_candidates(export_dir, wiki_dir, human_before=CUTOFF)
    raw_by_identity = {_candidate_identity(candidate): candidate for candidate in raw_candidates}
    source_rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    verified_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, str]] = []
    for row in source_rows:
        candidate = raw_by_identity.get(_identity(row))
        if candidate is None:
            excluded_rows.append({"id": str(row.get("id") or ""), "reason": "not_traceable_to_raw_pre_cutoff_export"})
            continue
        timestamp_seconds = candidate.timestamp / 1000 if candidate.timestamp > 10_000_000_000 else candidate.timestamp
        if not timestamp_seconds or timestamp_seconds >= CUTOFF:
            excluded_rows.append({"id": str(row.get("id") or ""), "reason": "timestamp_not_before_cutoff"})
            continue
        verified_rows.append({
            **row,
            "timestamp": candidate.timestamp,
            "source_provenance": "before_automation_cutoff",
            "verification": "exact_context_reply_chat_match_to_raw_export",
        })

    if expected_matches is not None and len(verified_rows) != expected_matches:
        raise ValueError(f"expected {expected_matches} verified rows, got {len(verified_rows)}")

    jsonl_content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verified_rows)
    examples_path = output_dir / "persona_examples.jsonl"
    _write_private(examples_path, jsonl_content)
    examples_sha256 = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()
    report = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "offline same-chat style A/B only",
        "review_status": "approved",
        "approval_scope": "mechanically traceable pre-cutoff human-source evidence; not manual style approval",
        "cutoff_epoch": CUTOFF,
        "source_path": str(source_path.relative_to(PROJECT_ROOT)),
        "source_row_count": len(source_rows),
        "raw_candidate_count": len(raw_candidates),
        "verified_count": len(verified_rows),
        "excluded_count": len(excluded_rows),
        "excluded": excluded_rows,
        "unique_chat_count": len({row["chat_id"] for row in verified_rows}),
        "relationship_counts": dict(sorted(Counter(row["relationship"] for row in verified_rows).items())),
        "source_provenance_counts": dict(sorted(Counter(row["source_provenance"] for row in verified_rows).items())),
        "examples_sha256": examples_sha256,
        "verification_rule": "exact relationship/chat_id/context/reply match against locally re-extracted records before cutoff",
    }
    _write_private(output_dir / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="构建仅供离线实验使用的已验证同聊天真人样例视图")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/few_shot_v5/persona_examples.jsonl")
    parser.add_argument("--exports", type=Path, default=PROJECT_ROOT / "data/exports/b")
    parser.add_argument("--wiki-dir", type=Path, default=PROJECT_ROOT / "data/memory/wiki/users")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/style_same_chat_v1",
    )
    parser.add_argument("--expected-matches", type=int, default=973)
    args = parser.parse_args()
    report = build_view(args.source, args.exports, args.wiki_dir, args.output, args.expected_matches)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
