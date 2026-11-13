#!/usr/bin/env python3
"""Offline C0 round-4 A/B: test one evidence-gating rule on exposed cases."""

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


EVIDENCE_GATE_RULE = (
    "<rule>只有当前上下文明确给出依据时，才可陈述平台或系统状态、因果关系、他人动机或风险判断；"
    "没有依据时直接简短回应当前问题，不自行补充机制、猜测动机，也不在对方未追问原因时主动扩写泛化解释。</rule>"
)

ROUND_ANALYSIS = {
    "decision": "待人工完整上下文复核后决定。",
    "confirmed_gains": [],
    "remaining_blocker": "待复核。",
    "regressions_to_track": [
        "过度收紧导致该回答的明确问题被静默或漏答",
        "把真人可合理掌握的私人信息误当作无依据内容删除",
        "亲密私聊被压缩成生硬、模板化的短句",
    ],
    "how_directions_were_found": [
        "冻结 Judge 只提名容易被识别为 AI 的案例，不把 Judge 输出当事实",
        "人工查看完整上下文，聚类出机制幻觉、无依据动机猜测和无人追问时主动解释三类重复问题",
        "第三轮抽象双关规则没有修复目标案例并引入回归，因此回到第二轮基线",
        "本轮只增加一条可核对的证据门控规则，在同一批已曝光训练案例上做单变量比较",
        "人工匿名比较后检查净胜负和新增严重回归，固定测试集不参与调参",
    ],
}


def _install_round4_candidate(generator: Any) -> None:
    round2._install_candidate(generator)
    original = generator._build_user_prompt

    def candidate_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original(*args, **kwargs)
        return prompt.replace(
            "</reply_guard>", EVIDENCE_GATE_RULE + "\n</reply_guard>", 1
        )

    generator._build_user_prompt = MethodType(candidate_prompt, generator)


def _render(payload: dict[str, Any]) -> str:
    page = round2._render(payload)
    page = page.replace("C0 第二轮匿名 A/B", "C0 第四轮匿名 A/B")
    page = page.replace(
        "当前冻结 C0 对比第二轮小步候选。",
        "当前第二轮 C0 对比第四轮单变量证据门控候选。",
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
        default=ROOT / "data/reports/private/c0_guardrail_ab_v4",
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
    round3._install_round3_candidate = _install_round4_candidate
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
            "blind_mapping": _option_map("v4:" + case["case_id"]),
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
        "schema_version": 4,
        "status": "generation_failed" if failed else "pending_manual_review",
        "source_fingerprint": fingerprint,
        "generation_provider": "volcengine",
        "scope": {
            "total": len(output),
            "group": sum(row["is_group"] for row in output),
            "private": sum(not row["is_group"] for row in output),
        },
        "single_variable": "context_evidence_gate",
        "baseline": "round2_candidate",
        "candidate": "round2_candidate_plus_context_evidence_gate",
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
