#!/usr/bin/env python3
"""Apply local AI rough labels to the Human Gold sample without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.reply_quality_provisional import (  # noqa: E402
    apply_provisional_annotations,
)


def _write(path: Path, value: dict, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if private:
        path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并回复质量 AI 粗标")
    parser.add_argument(
        "--packet",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_quality_provisional_annotations.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data/reports/reply_quality_provisional.json",
    )
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["cases"]
    packet, report = apply_provisional_annotations(packet, annotations)
    _write(args.packet, packet, private=True)
    _write(args.report, report)
    print(f"Rough labels: {report['cases']}")
    print(f"Grade counts: {report['grade_counts']}")
    print("Status: ai_provisional_not_human_ground_truth")


if __name__ == "__main__":
    main()
