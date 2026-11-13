#!/usr/bin/env python3
"""Assemble a group-chat Judge transfer check from existing reply pairs only."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_strict_historical_judge_acceptance import (
    ROOT,
    _tail_has_unresolved_media,
    _with_full_judge_context,
    _write_json,
)


OUT_DIR = ROOT / "data/reports/private/strict_historical_judge_acceptance_v5_group_balanced"
SOURCES = {
    "v4_set2_volcengine": ROOT / "data/reports/private/strict_historical_judge_acceptance_v4_set2_volcengine/generation_results.json",
    "v4_set1": ROOT / "data/reports/private/strict_historical_judge_acceptance_v4_set1/generation_results.json",
    "v1": ROOT / "data/reports/private/strict_historical_judge_acceptance_v1/generation_results.json",
}
CASE_SOURCES = {
    "1769cc9a6c3e0281d564": "v4_set2_volcengine",
    "2a9f236838be47309c33": "v4_set2_volcengine",
    "8ff89fd5a1b4b8ec4418": "v4_set2_volcengine",
    "c4c50ec8a4ff9379eba2": "v4_set2_volcengine",
    "ca44c1afc500e1e98b76": "v4_set2_volcengine",
    "643c3f7c55ce965261f8": "v4_set1",
    "be939fc6d6a87cad5f77": "v4_set1",
    "d417b65bb84974c22b2a": "v1",
    "d59c02cffc75412a236b": "v4_set1",
    "fb3f57f3ebc76719d2f1": "v4_set1",
    "335c3cac5b2128c00a7d": "v4_set2_volcengine",
    "34aa40c43da8bc8a13d4": "v4_set2_volcengine",
    "5bbbfde2a38d3335b55e": "v4_set2_volcengine",
    "a198417a21e545b79068": "v4_set2_volcengine",
    "b43f2a531f54edc0e1ae": "v4_set2_volcengine",
}
REUSED_YOUZI_IDS = {
    "643c3f7c55ce965261f8",
    "be939fc6d6a87cad5f77",
    "d417b65bb84974c22b2a",
    "d59c02cffc75412a236b",
    "fb3f57f3ebc76719d2f1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble() -> dict[str, Any]:
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}
    by_source = {
        name: {str(row["case_id"]): row for row in payload["cases"]}
        for name, payload in payloads.items()
    }
    source_hashes = {name: _sha256(path) for name, path in SOURCES.items()}
    cases = []
    for case_id, source_name in CASE_SOURCES.items():
        if case_id not in by_source[source_name]:
            raise RuntimeError(f"missing {case_id} in {source_name}")
        row = copy.deepcopy(by_source[source_name][case_id])
        if not row.get("validity", {}).get("formally_scorable"):
            raise RuntimeError(f"unscorable existing pair: {case_id}")
        if _tail_has_unresolved_media(row):
            raise RuntimeError(f"unresolved tail media: {case_id}")
        if case_id == "d417b65bb84974c22b2a":
            full_context_row = copy.deepcopy(by_source["v4_set1"][case_id])
            if any(
                row[field] != full_context_row[field]
                for field in ("human_reply", "timestamp", "chat_name")
            ):
                raise RuntimeError(f"full-context identity mismatch: {case_id}")
            row["context"] = full_context_row["context"]
            row["generation"]["metadata"]["judge_history_count"] = len(row["context"])
        else:
            row = _with_full_judge_context(row)
        row["transfer_check_source"] = {
            "source_file": str(SOURCES[source_name].relative_to(ROOT)),
            "source_sha256": source_hashes[source_name],
            "judge_context_source_file": (
                str(SOURCES["v4_set1"].relative_to(ROOT))
                if case_id == "d417b65bb84974c22b2a"
                else str(SOURCES[source_name].relative_to(ROOT))
            ),
            "never_previously_judged": case_id not in REUSED_YOUZI_IDS,
            "reused_youzi_case": case_id in REUSED_YOUZI_IDS,
        }
        cases.append(row)
    counts = {}
    for row in cases:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    expected = {
        "correct_common_prosperity_with_luxixi": 5,
        "youzi_group": 5,
        "other_complex_large_group": 5,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected group balance: {counts}")
    return {
        "schema_version": 1,
        "status": "complete",
        "evaluation": {
            "evaluation_kind": "group_chat_transfer_check",
            "case_count": 15,
            "never_judged_count": 10,
            "reused_youzi_count": 5,
            "freeze_eligible": False,
            "uses_existing_reply_pairs_only": True,
            "source_files": [
                {"path": str(path.relative_to(ROOT)), "sha256": source_hashes[name]}
                for name, path in SOURCES.items()
            ],
            "independent_pool_audit": {
                "status": "insufficient_for_balanced_validation",
                "audited_at": "2026-08-21",
                "rule": "Deduplicate all strict-historical generation cases by case_id, then exclude every case found in an actual Judge result plus unscorable pairs.",
                "groups": {
                    "correct_common_prosperity_with_luxixi": {
                        "candidates": 71,
                        "previously_judged": 21,
                        "unseen_c0_silent": 47,
                        "unseen_answer_leak": 1,
                        "unseen_c0_reproduced_human": 0,
                        "unseen_usable_reply_pairs": 2,
                    },
                    "youzi_group": {
                        "candidates": 18,
                        "previously_judged": 9,
                        "unseen_c0_silent": 9,
                        "unseen_answer_leak": 0,
                        "unseen_c0_reproduced_human": 0,
                        "unseen_usable_reply_pairs": 0,
                    },
                    "other_complex_large_group": {
                        "candidates": 248,
                        "previously_judged": 68,
                        "unseen_c0_silent": 164,
                        "unseen_answer_leak": 14,
                        "unseen_c0_reproduced_human": 1,
                        "unseen_usable_reply_pairs": 1,
                    },
                },
                "usable_unseen_case_ids": [
                    "6477a4c3eb37a99ecd9f",
                    "7601b9167d87e9845488",
                    "aea42cfa97a3b38e2b28",
                ],
            },
        },
        "cases": cases,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = assemble()
    _write_json(OUT_DIR / "generation_results.json", output)
    print(json.dumps(output["evaluation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
