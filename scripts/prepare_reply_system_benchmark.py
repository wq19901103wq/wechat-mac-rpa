#!/usr/bin/env python3
"""Prepare fixed replay scenarios without starting the Bot or calling a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.reply_system_quality import generate_fixed_scenarios  # noqa: E402


def _write_json(path: Path, value: dict, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if private:
        path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="准备当前回复系统固定评测场景（不调用模型）")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "cases.db")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "reply_system_benchmark.json",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "private" / "reply_system_scenarios.json",
    )
    args = parser.parse_args()
    manifest, packet = generate_fixed_scenarios(args.db)
    _write_json(args.output, manifest)
    _write_json(args.private_output, packet, private=True)
    print(f"Public manifest: {args.output}")
    print(f"Private scenarios: {args.private_output}")
    print(f"Fixed scenarios: {manifest['scope']['deduplicated_fixed_scenarios']}")
    print("Current-system baseline: replay_pending")


if __name__ == "__main__":
    main()

