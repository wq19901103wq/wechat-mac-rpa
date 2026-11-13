#!/usr/bin/env python3
"""Offline C0 round-3 A/B: preserve round-2 gains and test local wordplay handling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import replay_c0_guardrail_ab_v2 as round2  # noqa: E402
from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    load_volcengine_credentials,
)
from scripts.replay_c0_guardrail_ab import CASE_IDS, _option_map, _serialize  # noqa: E402
from scripts.replay_current_reply_system import _load_env  # noqa: E402
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402


WORDPLAY_RULE = (
    "<rule>遇到多义词、谐音或拆词接龙时，先用紧邻对话判断实际意图；"
    "若当前短句是在承接前一句的双关，不要被身份资料的字面答案带离当前语义。"
    "玩笑中的临时自称不得写回长期身份事实。</rule>"
)

ROUND_ANALYSIS = {
    "decision": "第三轮候选不采用，继续保留第二轮 C0。目标双关案例未修复，且引入机制幻觉和泛化解释回归。",
    "confirmed_gains": [
        "部分场景能更直接承接紧邻消息，例如回应“急于撇清”",
        "亲密私聊中有两条回复更短、更自然地承接情绪",
    ],
    "remaining_blocker": (
        "“你属什么”仍未识别历史真人回复里的“属什么 / 不属于你”双关；"
        "候选改成“属保密”只是回避字面答案，不能证明规则真正生效。"
    ),
    "regressions_to_track": [
        "不要把正常事实问题强行当作玩笑，也不要用回避答案代替理解双关",
        "继续禁止编造平台 beta、系统重置、任务清空等未知机制",
        "群聊无人追问时不要主动展开周期股、低 PE 等泛化解释",
        "双关规则需要先识别具体触发词及上下句语义关系，再决定是否启用，不能全局追加一句抽象规则",
    ],
    "how_directions_were_found": [
        "冻结 Judge 只负责提名容易被识别为 AI 的案例，不把 Judge 输出当事实",
        "回到完整历史上下文和真人原始回复，纠正了上一轮把该案例归因为私人事实冲突的错误",
        "对同一批已暴露训练案例只增加一条双关语义规则，复用第二轮结果作为基线",
        "人工逐条比较完整上下文，先看目标案例是否真正修复，再检查事实问题、群聊插话和私聊展开是否回归",
        "目标案例没有修复且出现严重新回归，因此拒绝候选；失败本身记录为下一轮方向发现证据",
    ],
}


def _install_round3_candidate(generator: Any) -> None:
    round2._install_candidate(generator)
    original = generator._build_user_prompt

    def candidate_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original(*args, **kwargs)
        return prompt.replace(
            "</reply_guard>", WORDPLAY_RULE + "\n</reply_guard>", 1
        )

    generator._build_user_prompt = MethodType(candidate_prompt, generator)


def _run_candidate(case: dict[str, Any], provider: str, config_path: Path) -> dict[str, Any]:
    from src.reply.generator import ReplyGenerator
    from src.tools.tool_registry import ToolRegistry
    from src.utils.qwen_client import QwenClient

    started = time.monotonic()
    if provider == "volcengine":
        api_key, base_url = load_volcengine_credentials(config_path)
        client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    else:
        client = QwenClient()
    generator = ReplyGenerator(
        llm_client=client,
        memory_engine=None,
        tool_registry=ToolRegistry(),
        judge_worker=None,
    )
    _install_round3_candidate(generator)
    try:
        replies = generator.generate(
            case["unreplied"],
            case["all_messages"],
            is_group=case["is_group"],
            tick_id=case["source_tick_id"],
            evaluation_time=case["evaluation_time"],
        )
        return {
            "status": "completed",
            "replies": replies,
            "generation_failed": bool(generator.last_generation_failed),
            "model": client.model,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "replies": [],
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.monotonic() - started, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--provider", choices=("official", "volcengine"), default="volcengine")
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument(
        "--round2-results",
        type=Path,
        default=ROOT / "data/reports/private/c0_guardrail_ab_v2/results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/reports/private/c0_guardrail_ab_v3",
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
            payload["round_analysis"] = ROUND_ANALYSIS
            payload["status"] = "complete"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        page = round2._render(payload).replace("C0 第二轮匿名 A/B", "C0 第三轮匿名 A/B")
        page = page.replace(
            "当前冻结 C0 对比第二轮小步候选。",
            "当前第二轮 C0 对比第三轮单变量双关语义候选。",
        )
        (args.out_dir / "index.html").write_text(page, encoding="utf-8")
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
    round2_payload = json.loads(args.round2_results.read_text(encoding="utf-8"))
    round2_rows = {row["case_id"]: row for row in round2_payload["cases"]}

    candidates: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_candidate, case, args.provider, args.volcengine_config): case["case_id"]
            for case in cases
        }
        for index, future in enumerate(as_completed(futures), 1):
            case_id = futures[future]
            candidates[case_id] = future.result()
            print(f"[{index}/{len(futures)}] {case_id} candidate", flush=True)

    output = []
    for case in cases:
        prior = round2_rows[case["case_id"]]
        output.append(
            {
                "case_id": case["case_id"],
                "chat_name": case["all_messages"][-1].chat_name,
                "is_group": case["is_group"],
                "context": [_serialize(message) for message in case["all_messages"]],
                "blind_mapping": _option_map("v3:" + case["case_id"]),
                "runs": {
                    "baseline": prior["runs"]["candidate"],
                    "candidate": candidates[case["case_id"]],
                },
            }
        )

    failed = [
        (row["case_id"], version)
        for row in output
        for version, result in row["runs"].items()
        if result.get("status") != "completed" or result.get("generation_failed")
    ]
    payload = {
        "schema_version": 3,
        "status": "generation_failed" if failed else "pending_manual_review",
        "source_fingerprint": fingerprint,
        "generation_provider": args.provider,
        "scope": {
            "total": len(output),
            "group": sum(row["is_group"] for row in output),
            "private": sum(not row["is_group"] for row in output),
        },
        "single_variable": "local_wordplay_semantic_priority",
        "baseline": "round2_candidate",
        "candidate": "round2_candidate_plus_wordplay_rule",
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
