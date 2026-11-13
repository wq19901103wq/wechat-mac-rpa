#!/usr/bin/env python3
"""Offline C0 round-2 A/B on exposed training cases; never starts the bot."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_c0_guardrail_ab import CASE_IDS, _option_map, _serialize  # noqa: E402
from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    load_volcengine_credentials,
)
from scripts.replay_current_reply_system import _load_env  # noqa: E402
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402


CANDIDATE_RULES = (
    "<rule>不要把上下文中已经显而易见的信息换一种说法重复解释；除非对方明确表示不理解或要求解释，否则这类解释必须删掉。</rule>",
    "<rule>可信历史与候选说法冲突时必须以可信历史为准；不得编造属相、身份、关系、经历或其他私人事实。</rule>",
    "<rule>不得编造系统、平台或任务机制，例如不存在的数量上限、自动顶替、重置清空、自动失效或后台处理规则；来源没有说明时不要自行解释机制。</rule>",
    "<rule>不得把玩笑写成会真实履行的付款、购买、寄送、预约、代办或到场承诺，也不得虚构已经执行的细节；上下文明显在玩笑时可以自然接话，不要因出现行动词就机械拒绝。</rule>",
)
GROUP_RULE = (
    "<rule>群聊优先短而克制，不强行插话、不重复解释、不堆多个包袱；"
    "但不得为了硬凑一句话而漏答明确问题，必要信息可以自然说完整。</rule>"
)

ROUND_ANALYSIS = {
    "decision": "候选有净收益，但暂不整体替换 C0；下一轮只修可信私人事实约束。",
    "confirmed_gains": [
        "减少复用已经出现过的旧梗",
        "减少对显然信息和无人追问机制的重复解释",
        "未知系统机制时承认不知道，不再自行编造",
        "能在当前话题内自然接话，并完成明确身份纠正",
    ],
    "remaining_blocker": "可信历史明确为属狗时，两版仍回答属马，说明私人事实约束尚未真正作用于生成。",
    "regressions_to_track": [
        "回捞已被最新消息覆盖的旧支线",
        "亲密私聊中过度展开，弱化短句的自然情绪回应",
        "把一次性语义扩大成长期或普遍表述",
    ],
    "how_directions_were_found": [
        "先由冻结 Judge 提名容易被识别为 AI 的案例，不把 Judge 结论当事实",
        "人工在完整上下文中匿名比较旧版与候选",
        "把胜负理由按旧梗、重复解释、事实冲突、话题锚定和语气展开聚类",
        "只把跨案例复现或存在可靠历史冲突的问题列为迭代方向，难分样本不强行总结",
        "每轮只选择一个稳定问题做小幅修改，再用未参与调优的固定测试集验证",
    ],
}


def _install_candidate(generator: Any) -> None:
    original = generator._build_user_prompt

    def candidate_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original(*args, **kwargs)
        is_group = bool(kwargs.get("is_group", args[2] if len(args) > 2 else False))
        rules = list(CANDIDATE_RULES)
        if is_group:
            rules.append(GROUP_RULE)
        addition = "\n".join(rules) + "\n"
        return prompt.replace("</reply_guard>", addition + "</reply_guard>", 1)

    generator._build_user_prompt = MethodType(candidate_prompt, generator)


def _run(
    case: dict[str, Any], version: str, provider: str, config_path: Path
) -> dict[str, Any]:
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
    if version == "candidate":
        _install_candidate(generator)
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


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _render(payload: dict[str, Any]) -> str:
    cards = []
    for row in payload["cases"]:
        review = row.get("review") or {}
        verdict = review.get("classification", "pending")
        reason = review.get("reason", "待人工查看完整上下文后复核。")
        context = "".join(
            f'<div class="msg {"self" if item["role"] == "self" else "other"}"><b>{_e(item["speaker"])}</b><span>{_e(item["content"])}</span></div>'
            for item in row["context"]
        )
        options = []
        for label in ("A", "B"):
            version = row["blind_mapping"][label]
            result = row["runs"][version]
            reply = " / ".join(result.get("replies") or []) or "（静默）"
            options.append(f'<div class="option"><b>回复 {label}</b><p>{_e(reply)}</p></div>')
        cards.append(
            f'<article class="case" data-tags="{"group" if row["is_group"] else "private"} {verdict}">'
            f'<h2>{_e(row["chat_name"])}</h2><p class="verdict {verdict}">{_e(verdict)}：{_e(reason)}</p>'
            f'<div class="context">{context}</div>{"".join(options)}'
            f'<details><summary>揭晓版本</summary><p>A = {_e(row["blind_mapping"]["A"])}；B = {_e(row["blind_mapping"]["B"])}</p></details></article>'
        )
    summary = payload.get("review_summary") or {"status": "pending_manual_review"}
    analysis = payload.get("round_analysis") or {}
    analysis_html = ""
    if analysis:
        gains = "".join(f"<li>{_e(item)}</li>" for item in analysis["confirmed_gains"])
        regressions = "".join(f"<li>{_e(item)}</li>" for item in analysis["regressions_to_track"])
        discovery = "".join(f"<li>{_e(item)}</li>" for item in analysis["how_directions_were_found"])
        analysis_html = (
            f'<section class="analysis"><h2>本轮结论</h2><p><b>{_e(analysis["decision"])}</b></p>'
            f'<h3>确认改善</h3><ul>{gains}</ul><h3>当前阻塞项</h3><p>{_e(analysis["remaining_blocker"])}</p>'
            f'<h3>已记录的后续方向</h3><ul>{regressions}</ul>'
            f'<h3>这些方向是怎么找到的</h3><ol>{discovery}</ol></section>'
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>C0 第二轮匿名 A/B</title><style>
body{{margin:0;background:#f4f5f7;color:#202124;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:760px;margin:auto;padding:16px}}h1{{font-size:23px;margin:4px 0 8px}}.summary{{background:#fff4d8;border-radius:10px;padding:10px}}.analysis{{background:white;border-left:4px solid #2878d0;border-radius:12px;padding:14px;margin:12px 0}}.analysis h2,.analysis h3{{margin:7px 0}}.analysis ul,.analysis ol{{padding-left:22px}}.filters{{position:sticky;top:0;background:#f4f5f7;padding:8px 0;z-index:2}}button{{border:1px solid #ccd2d8;background:white;border-radius:18px;padding:8px 13px;margin:3px}}.case{{background:white;border-radius:14px;padding:14px;margin:12px 0;box-shadow:0 2px 10px #0000000d}}h2{{font-size:17px}}.verdict{{border-radius:8px;padding:8px 10px;background:#eef1f4}}.context{{background:#eef1f4;border-radius:10px;padding:8px;max-height:52vh;overflow:auto}}.msg{{display:flex;gap:8px;padding:6px;border-bottom:1px solid #dde1e5}}.msg b{{min-width:76px;color:#53606d}}.msg.self b{{color:#176f45}}.option{{border:1px solid #d8dde3;border-radius:10px;padding:10px;margin-top:10px}}.hidden{{display:none}}</style></head><body><main>
<h1>C0 第二轮匿名 A/B</h1><p>20 个已暴露训练案例；当前冻结 C0 对比第二轮小步候选。Judge 仅提名，人工完整上下文复核。</p><p class="summary">候选胜 {_e(summary.get("candidate_win"))} · 基线胜 {_e(summary.get("baseline_win"))} · 持平 {_e(summary.get("tie"))}</p><p>Bot 始终关闭，未发送任何消息；固定测试集未用于调参。</p>{analysis_html}<div class="filters"><button data-filter="all">全部</button><button data-filter="candidate_win">候选胜</button><button data-filter="baseline_win">基线胜</button><button data-filter="tie">持平</button><button data-filter="group">群聊</button><button data-filter="private">私聊</button></div>{''.join(cards)}
</main><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.querySelectorAll('.case').forEach(c=>c.classList.toggle('hidden',b.dataset.filter!=='all'&&!c.dataset.tags.split(' ').includes(b.dataset.filter))))</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--provider", choices=("official", "volcengine"), default="volcengine")
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/reports/private/c0_guardrail_ab_v2")
    parser.add_argument("--render-existing", type=Path)
    parser.add_argument("--retry-existing", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_existing:
        payload = json.loads(args.render_existing.read_text(encoding="utf-8"))
    else:
        _load_env()
        fingerprint, all_cases = load_replay_cases(
            ROOT / "data/cases.db",
            ROOT / "data/reports/private/reply_system_scenarios.json",
            ROOT / "data/reports/private/reply_quality_gold_cases.json",
        )
        by_id = {case["case_id"]: case for case in all_cases}
        cases = [by_id[case_id] for case_id in CASE_IDS]
        if args.retry_existing:
            payload = json.loads(args.retry_existing.read_text(encoding="utf-8"))
            rows_by_id = {row["case_id"]: row for row in payload["cases"]}
            pending = [
                (by_id[row["case_id"]], version)
                for row in payload["cases"]
                for version, result in row["runs"].items()
                if result.get("status") != "completed" or result.get("generation_failed")
            ]
            futures = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for case, version in pending:
                    futures[
                        pool.submit(
                            _run, case, version, args.provider, args.volcengine_config
                        )
                    ] = (case["case_id"], version)
                for index, future in enumerate(as_completed(futures), 1):
                    case_id, version = futures[future]
                    rows_by_id[case_id]["runs"][version] = future.result()
                    print(f"[{index}/{len(futures)}] {case_id} {version}", flush=True)
            payload["generation_provider"] = args.provider
        else:
            results = {case["case_id"]: {} for case in cases}
            futures = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for case in cases:
                    for version in ("baseline", "candidate"):
                        futures[
                            pool.submit(
                                _run, case, version, args.provider, args.volcengine_config
                            )
                        ] = (case["case_id"], version)
                for index, future in enumerate(as_completed(futures), 1):
                    case_id, version = futures[future]
                    results[case_id][version] = future.result()
                    print(f"[{index}/{len(futures)}] {case_id} {version}", flush=True)
            output = []
            for case in cases:
                context = [_serialize(message) for message in case["all_messages"]]
                output.append({
                    "case_id": case["case_id"],
                    "chat_name": case["all_messages"][-1].chat_name,
                    "is_group": case["is_group"],
                    "context": context,
                    "blind_mapping": _option_map(case["case_id"]),
                    "runs": results[case["case_id"]],
                })
            payload = {
                "schema_version": 2,
                "status": "pending_manual_review",
                "source_fingerprint": fingerprint,
                "generation_provider": args.provider,
                "scope": {"total": len(output), "group": sum(row["is_group"] for row in output), "private": sum(not row["is_group"] for row in output)},
                "cases": output,
            }
    failed_runs = [
        (row["case_id"], version)
        for row in payload["cases"]
        for version, result in row["runs"].items()
        if result.get("status") != "completed" or result.get("generation_failed")
    ]
    if failed_runs:
        payload["status"] = "generation_failed"
        payload["failed_runs"] = failed_runs
        (args.out_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": payload["status"], "failed_runs": len(failed_runs)}, ensure_ascii=False))
        return 2
    payload.pop("failed_runs", None)
    if payload.get("status") == "generation_failed":
        payload["status"] = "pending_manual_review"
    if args.reviews:
        reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
        for row in payload["cases"]:
            row["review"] = reviews["cases"][row["case_id"]]
        labels = ("candidate_win", "baseline_win", "tie")
        payload["review_summary"] = {
            label: sum(row["review"]["classification"] == label for row in payload["cases"])
            for label in labels
        }
        payload["round_analysis"] = ROUND_ANALYSIS
        payload["status"] = "complete"
    (args.out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (args.out_dir / "index.html").write_text(_render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["scope"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
