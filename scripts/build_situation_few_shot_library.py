#!/usr/bin/env python3
"""Build a private situation-tagged few-shot library from the C0 development pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reply.few_shot import _situation_profile, _situation_tags  # noqa: E402

DEFAULT_SOURCE = ROOT / "data/reports/private/c0_development_pool_v1/manifest.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/situation_few_shot_library_v1"
PRIVATE_MODE = 0o600


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(PRIVATE_MODE)


def build(source: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    situations: Counter[str] = Counter()
    relationships: Counter[str] = Counter()
    relationship_situations: Counter[str] = Counter()
    for source_row in payload.get("cases", []):
        row = dict(source_row)
        tags = sorted(_situation_tags("\n".join(str(value) for value in row.get("context") or [])))
        row["situation_tags"] = tags
        row["situation_profile"] = _situation_profile(
            "\n".join(str(value) for value in row.get("context") or [])
        )
        row["verification"] = "inherited_from_holdout_excluded_verified_human_development_pool"
        rows.append(row)
        relationship = str(row.get("relationship") or "")
        relationships[relationship] += 1
        for tag in tags:
            situations[tag] += 1
            relationship_situations[f"{relationship}:{tag}"] += 1

    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    examples_path = output_dir / "persona_examples.jsonl"
    _write_private(examples_path, content)
    report = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "offline C0 situation-first retrieval experiment only",
        "review_status": "approved",
        "approval_scope": "mechanically inherited verified-human provenance; not production activation",
        "source": str(source.relative_to(ROOT)),
        "source_holdout_exclusion": payload.get("holdout_exclusion"),
        "source_exact_holdout_overlap": payload.get("counts", {}).get("exact_holdout_overlap"),
        "row_count": len(rows),
        "relationship_counts": dict(sorted(relationships.items())),
        "situation_counts": dict(sorted(situations.items())),
        "relationship_situation_counts": dict(sorted(relationship_situations.items())),
        "untagged_count": sum(not row["situation_tags"] for row in rows),
        "examples_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    _write_private(output_dir / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
