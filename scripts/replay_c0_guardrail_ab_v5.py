#!/usr/bin/env python3
"""Offline C0 round-5 A/B: one final-response validation pass on exposed cases."""

from __future__ import annotations

import argparse
import html
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import replay_c0_guardrail_ab_v2 as round2  # noqa: E402
from scripts import replay_c0_guardrail_ab_v3 as round3  # noqa: E402
from scripts.replay_c0_guardrail_ab import CASE_IDS, _option_map, _serialize  # noqa: E402
from scripts.replay_current_reply_system import _load_env  # noqa: E402
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402


FINAL_VALIDATION_RULE = (
    "<rule>写出候选回复后，发送前只做一次终稿核对：回复应承接最后一个仍在进行的话题；"
    "若某句话与上下文或可信历史中的明确事实冲突、把未知平台行为说成确定事实、重新捞起已被后续消息覆盖的旧话题，"
    "或在对方没有要求解释时使用解释性陈述，就删掉该句或改成简短且诚实的不确定表达。"
    "不要借核对新增事实、理由、建议或玩笑；核对后直接输出最终回复。</rule>"
)


def _install_round5_candidate(generator: Any) -> None:
    round2._install_candidate(generator)
    original = generator._build_user_prompt

    def candidate_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original(*args, **kwargs)
        return prompt.replace(
            "</reply_guard>", FINAL_VALIDATION_RULE + "\n</reply_guard>", 1
        )

    generator._build_user_prompt = MethodType(candidate_prompt, generator)


def _render(payload: dict[str, Any]) -> str:
    page = round2._render(payload)
    page = page.replace("C0 第二轮匿名 A/B", "C0 第五轮匿名 A/B")
    page = page.replace(
        "当前冻结 C0 对比第二轮小步候选。",
        "当前第二轮 C0 对比第五轮单变量终稿核对候选。",
    )
    experience = (payload.get("round_analysis") or {}).get("iteration_experience") or []
    if experience:
        items = "".join(
            f"<li>{html.escape(str(item), quote=True)}</li>" for item in experience
        )
        page = page.replace(
            "</ol></section>",
            f"</ol><h3>本轮沉淀的迭代经验</h3><ul>{items}</ul></section>",
            1,
        )
    return page


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--round2-results",
        type=Path,
        default=ROOT / "data/reports/private/c0_guardrail_ab_v2/results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/reports/private/c0_guardrail_ab_v5",
    )
    parser.add_argument("--render-existing", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()

    if args.render_existing:
        payload = json.loads(args.render_existing.read_text(encoding="utf-8"))
        if args.reviews:
            reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
            for row in payload["cases"]:
                row["review"] = reviews["cases"][row["case_id"]]
            labels = ("candidate_win", "baseline_win", "tie")
            payload["review_summary"] = {
                label: sum(
                    row["review"]["classification"] == label
                    for row in payload["cases"]
                )
                for label in labels
            }
            payload["round_analysis"] = reviews["round_analysis"]
            payload["status"] = "complete"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "index.html").write_text(_render(payload), encoding="utf-8")
        print(json.dumps({"status": payload["status"], **payload["scope"]}, ensure_ascii=False))
        return 0

    _load_env()
    fingerprint, all_cases = load_replay_cases(
        ROOT / "data/cases.db",
        ROOT / "data/reports/private/reply_system_scenarios.json",
        ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    by_id = {case["case_id"]: case for case in all_cases}
    cases = [by_id[case_id] for case_id in CASE_IDS]
    baseline_payload = json.loads(args.round2_results.read_text(encoding="utf-8"))
    baseline_rows = {row["case_id"]: row for row in baseline_payload["cases"]}

    original_installer = round3._install_round3_candidate
    round3._install_round3_candidate = _install_round5_candidate
    try:
        candidates: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    round3._run_candidate,
                    case,
                    "volcengine",
                    round3.DEFAULT_VOLCENGINE_CONFIG,
                ): case["case_id"]
                for case in cases
            }
            for index, future in enumerate(as_completed(futures), 1):
                case_id = futures[future]
                candidates[case_id] = future.result()
                print(f"[{index}/{len(futures)}] {case_id} candidate", flush=True)
    finally:
        round3._install_round3_candidate = original_installer

    output = []
    for case in cases:
        output.append({
            "case_id": case["case_id"],
            "chat_name": case["all_messages"][-1].chat_name,
            "is_group": case["is_group"],
            "context": [_serialize(message) for message in case["all_messages"]],
            "blind_mapping": _option_map("v5:" + case["case_id"]),
            "runs": {
                "baseline": baseline_rows[case["case_id"]]["runs"]["candidate"],
                "candidate": candidates[case["case_id"]],
            },
        })
    failed = [
        (row["case_id"], version)
        for row in output
        for version, result in row["runs"].items()
        if result.get("status") != "completed" or result.get("generation_failed")
    ]
    payload = {
        "schema_version": 5,
        "status": "generation_failed" if failed else "pending_manual_review",
        "source_fingerprint": fingerprint,
        "generation_provider": "volcengine",
        "scope": {
            "total": len(output),
            "group": sum(row["is_group"] for row in output),
            "private": sum(not row["is_group"] for row in output),
        },
        "single_variable": "final_response_validation_pass",
        "baseline": "round2_candidate",
        "candidate": "round2_candidate_plus_final_response_validation_pass",
        "failed_runs": failed,
        "cases": output,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "failed_runs": failed}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
