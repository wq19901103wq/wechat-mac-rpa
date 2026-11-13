#!/usr/bin/env python3
"""Replay the current reply generator without starting WeChat Bot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="安全离线回放当前回复系统")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-generation-failures", action="store_true")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/cases.db")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/current_reply_system_replay.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data/reports/current_reply_system_replay.json",
    )
    args = parser.parse_args()
    _load_env()

    from src.benchmarks.reply_system_replay import (
        load_replay_cases,
        private_source_snapshot,
        public_replay_report,
        replay_cases,
        system_fingerprint,
    )
    from src.reply.generator import ReplyGenerator
    from src.tools.tool_registry import ToolRegistry
    from src.utils.qwen_client import QwenClient

    client = QwenClient()
    fingerprint, cases = load_replay_cases(args.db, args.scenarios, args.selection)
    system = system_fingerprint(PROJECT_ROOT, client)
    source_snapshot = private_source_snapshot(PROJECT_ROOT)

    def generator_factory() -> ReplyGenerator:
        return ReplyGenerator(
            llm_client=QwenClient(),
            memory_engine=None,
            tool_registry=ToolRegistry(),
            judge_worker=None,
        )

    packet = replay_cases(
        cases,
        generator_factory,
        args.private_output,
        fingerprint,
        system,
        limit=args.limit,
        workers=args.workers,
        retry_generation_failures=args.retry_generation_failures,
        source_snapshot=source_snapshot,
    )
    report = public_replay_report(packet)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
