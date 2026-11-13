#!/usr/bin/env python3
"""Build the formal-240 C0 retrieval pool by only removing holdout overlaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data/reports/private/situation_few_shot_library_v1/persona_examples.jsonl"
DEFAULT_FIXED = ROOT / "data/reports/private/fixed_c0_holdout_final/manifest.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/situation_few_shot_library_formal240_v1/persona_examples.jsonl"


def _normalized_pair(context: Any, reply: Any) -> str:
    context_list = context if isinstance(context, list) else [context]
    reply_list = reply if isinstance(reply, list) else [reply]
    return json.dumps([context_list, reply_list], ensure_ascii=False, separators=(",", ":"))


def _exact_pair_fingerprint(context: Any, reply: Any) -> str:
    return hashlib.sha256(_normalized_pair(context, reply).encode("utf-8")).hexdigest()


def _stable_source_id(row: dict[str, Any]) -> str | None:
    """Return only an ID explicitly tied to the original source message."""
    for key in ("original_message_id", "source_message_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base: Path, fixed: Path, output: Path) -> dict[str, Any]:
    base_rows = _load_jsonl(base)
    fixed_rows = json.loads(fixed.read_text(encoding="utf-8")).get("cases", [])
    fixed_source_ids = {
        source_id for row in fixed_rows if (source_id := _stable_source_id(row)) is not None
    }
    fixed_pair_fingerprints = {
        _exact_pair_fingerprint(row.get("context", []), row.get("reply", []))
        for row in fixed_rows
    }

    base_ids = {str(row["id"]) for row in base_rows}
    if len(base_ids) != len(base_rows):
        raise ValueError("base pool contains duplicate ids")
    source_id_excluded = [
        row
        for row in base_rows
        if (source_id := _stable_source_id(row)) is not None and source_id in fixed_source_ids
    ]
    pair_excluded = [
        row for row in base_rows
        if _exact_pair_fingerprint(row.get("context", []), row.get("reply", []))
        in fixed_pair_fingerprints
    ]
    pair_excluded_ids = {
        str(row["id"]) for row in pair_excluded
    }
    excluded_ids = {str(row["id"]) for row in source_id_excluded} | pair_excluded_ids
    rows = [row for row in base_rows if str(row["id"]) not in excluded_ids]

    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("formal pool contains duplicate ids")
    if not {str(row["id"]) for row in rows}.issubset(base_ids):
        raise ValueError("formal pool added rows that are not in the development pool")
    if len(rows) > len(base_rows):
        raise ValueError("formal pool cannot be larger than the development pool")
    if any(
        (source_id := _stable_source_id(row)) is not None and source_id in fixed_source_ids
        for row in rows
    ):
        raise ValueError("formal pool contains a frozen-240 original source message id")
    if any(
        _exact_pair_fingerprint(row.get("context", []), row.get("reply", []))
        in fixed_pair_fingerprints
        for row in rows
    ):
        raise ValueError("formal pool contains a frozen-240 context+reply pair")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    output.write_text(payload, encoding="utf-8")
    examples_sha256 = _sha256(output)
    relationship_counts = Counter(
        "group" if row.get("relationship") == "group" else "private" for row in rows
    )
    report = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "formal-240 C0 retrieval pool; development pool minus exact holdout overlaps",
        "review_status": "approved",
        "approval_scope": "offline C0 feature experiment only",
        "base": str(base.relative_to(ROOT)),
        "base_sha256": _sha256(base),
        "frozen_holdout": str(fixed.relative_to(ROOT)),
        "counts": {
            "base_input": len(base_rows),
            "fixed_source_id_excluded": len(source_id_excluded),
            "fixed_pair_excluded_unique_ids": len(pair_excluded_ids),
            "fixed_pair_matches": len(pair_excluded),
            "excluded_unique_ids": len(excluded_ids),
            "output": len(rows),
            "net_change_from_base": len(rows) - len(base_rows),
            "relationship": dict(sorted(relationship_counts.items())),
        },
        "leakage_exclusion": {
            "primary": "exact original_message_id or source_message_id equality",
            "fallback": "SHA256 of exact normalized complete context plus human reply",
            "derived_ids_not_compared": ["id", "internal_id"],
            "fixed_source_ids_available": len(fixed_source_ids),
            "fixed_pair_fingerprints": len(fixed_pair_fingerprints),
        },
        "examples_sha256": examples_sha256,
        "output_sha256": examples_sha256,
    }
    output.with_name("report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.fixed, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
