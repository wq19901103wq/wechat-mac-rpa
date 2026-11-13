#!/usr/bin/env python3
"""Prepare a private Human Gold review packet without calling a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.reply_quality_gold import build_human_gold_packet  # noqa: E402


def _write(path: Path, value: dict, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if private:
        path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="准备回复质量真人 Gold 待标包")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json",
    )
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/reply_quality_gold.json",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    args = parser.parse_args()
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    manifest, packet = build_human_gold_packet(scenarios, args.sample_size)
    _write(args.output, manifest)
    _write(args.private_output, packet, private=True)
    print(f"Public manifest: {args.output}")
    print(f"Private Human Gold packet: {args.private_output}")
    print(f"Selected cases: {manifest['selected_cases']}")
    print("Status: human_review_pending")


if __name__ == "__main__":
    main()
