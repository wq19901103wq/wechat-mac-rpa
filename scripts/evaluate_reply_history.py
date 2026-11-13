#!/usr/bin/env python3
"""Generate an offline public baseline and local private review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.reply_history_quality import generate_evaluation  # noqa: E402


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "dirty": None}


def _write_json(path: Path, value: dict, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if private:
        path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线生成历史回复质量基线（不调用 API）")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "cases.db")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "reply_quality_baseline.json",
    )
    parser.add_argument(
        "--private-output", type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "private" / "reply_quality_review_cases.json",
    )
    args = parser.parse_args()
    report, packet = generate_evaluation(args.db)
    report["provenance"] = {
        **_git_state(),
        "evaluator_sha256": hashlib.sha256(
            (PROJECT_ROOT / "src" / "benchmarks" / "reply_history_quality.py").read_bytes()
        ).hexdigest(),
    }
    _write_json(args.output, report)
    _write_json(args.private_output, packet, private=True)
    scope = report["scope"]
    print(f"Public baseline: {args.output}")
    print(f"Private review packet: {args.private_output}")
    print(f"Coverage: {scope['archived_review_cases']}/{scope['sent_ticks']} ticks")
    print(f"Status: {report['baseline_status']}; {report['release_decision']}")


if __name__ == "__main__":
    main()
