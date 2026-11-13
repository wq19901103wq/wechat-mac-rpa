#!/usr/bin/env python3
"""Launch the local Human Gold review page. This does not start the Bot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="回复质量 Human Gold 本地评审页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--access-token",
        help="临时访问口令；启用后首次访问需在 URL 中提供 ?token=口令",
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    args = parser.parse_args()
    os.environ["REPLY_QUALITY_GOLD_PACKET"] = str(args.packet.resolve())
    if args.access_token:
        os.environ["REPLY_QUALITY_REVIEW_TOKEN"] = args.access_token
    print(f"Human Gold review: http://{args.host}:{args.port}")
    print("This review server does not start the Bot or call a model.")
    uvicorn.run(
        "src.benchmarks.reply_quality_review_app:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
