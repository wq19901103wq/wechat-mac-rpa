#!/usr/bin/env python3
"""Offline A/B for the first C0 guardrail iteration; never constructs the bot."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_current_reply_system import _load_env  # noqa: E402
from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    load_volcengine_credentials,
)
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402


CASE_IDS = (
    "00b4a6bfdee914f498e2", "05c3b5f96ae199f1ed1c", "028479e4792a63b9dd64",
    "0344721ce96c1a046e99", "07b4b59487d8a430ad0f", "1237fd162c96df2bcef2",
    "136c6cb4d37afdfc34d3", "1375d5a9ad3caae76a2c", "0dfffa16e6b46dc17e46",
    "1394ae54261080cd92ee", "0e281ad9f57d50716e71", "149443b99ba56d2af31e",
    "15f03ea1550042cb5267", "166b6926e757e32862f4", "1b4902a8f6433cdc1194",
    "1c0c2271472a7549f1c9", "28ed94e582bae0f82c29", "3fda086471ee97b00c8c",
    "5bddcd3d7bb60ddce1ce", "896b288cb9d301358fbe",
)

REMOVED_BASELINE_RULES = (
    "<rule>不要把上下文中已经显而易见的信息换一种说法重复解释；除非对方明确表示不理解或要求解释，否则这类解释必须删掉。</rule>\n",
    "<rule>不得声称自己已经或将要付款、购买、寄送、预约、代办、到场等现实行动，除非可信来源明确表明该行动真实发生或已由本人决定。</rule>\n",
    "<rule>群聊默认只回复一条、且只用一句话；只有对方明确要求多个答案，或一句话确实无法避免歧义时才可以增加。</rule>\n",
)

REVIEW = {
    "00b4a6bfdee914f498e2": ("candidate_win", "候选避免了没有必要的“生四个”。"),
    "05c3b5f96ae199f1ed1c": ("candidate_win", "候选避免了无关的“过户流程发我”。"),
    "028479e4792a63b9dd64": ("tie", "两版都静默。"),
    "0344721ce96c1a046e99": ("tie", "两版都静默。"),
    "07b4b59487d8a430ad0f": ("tie", "两版都静默。"),
    "1237fd162c96df2bcef2": ("candidate_win", "候选避免对已经结束的支线继续插话。"),
    "136c6cb4d37afdfc34d3": ("baseline_win", "候选“我 王芊”出现得突兀，也错过了自然接话时机。"),
    "1375d5a9ad3caae76a2c": ("candidate_win", "候选避免了没有依据的 HBM 解释。"),
    "0dfffa16e6b46dc17e46": ("candidate_win", "候选的玩笑更贴合“还没加好友就分财产”的当前语境。"),
    "1394ae54261080cd92ee": ("candidate_win", "候选直接纠正 3.6 元与 3.6 万的误读。"),
    "0e281ad9f57d50716e71": ("tie", "两版完全相同。"),
    "149443b99ba56d2af31e": ("tie", "两版都静默。"),
    "15f03ea1550042cb5267": ("tie", "两版都静默。"),
    "166b6926e757e32862f4": ("tie", "两版都静默。"),
    "1b4902a8f6433cdc1194": ("tie", "两种表达都能成立，没有稳定优劣。"),
    "1c0c2271472a7549f1c9": ("baseline_win", "候选声称属兔，与可信历史中的“我属狗的”冲突。"),
    "28ed94e582bae0f82c29": ("baseline_win", "候选编造了任务上限和重置机制的解释。"),
    "3fda086471ee97b00c8c": ("candidate_win", "候选避免在没有使用依据时武断评价 VPN 网站。"),
    "5bddcd3d7bb60ddce1ce": ("candidate_win", "候选对母亲的回应更自然、更贴近当前情绪。"),
    "896b288cb9d301358fbe": ("baseline_win", "旧版顺势追问地铁更自然，候选接话较弱。"),
}

REVIEW_LABELS = {
    "candidate_win": "候选胜",
    "baseline_win": "基线胜",
    "tie": "持平",
}


def _serialize(message: Any) -> dict[str, Any]:
    return {
        "time": message.timestamp,
        "speaker": message.sender,
        "role": message.sender_type.value,
        "content": message.text,
        "unread": False,
    }


def _install_baseline(generator: Any) -> None:
    original = generator._build_user_prompt

    def baseline_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original(*args, **kwargs)
        for rule in REMOVED_BASELINE_RULES:
            prompt = prompt.replace(rule, "")
        return prompt

    generator._build_user_prompt = MethodType(baseline_prompt, generator)


def _run(case: dict[str, Any], version: str, provider: str, config_path: Path) -> dict[str, Any]:
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
    if version == "baseline":
        _install_baseline(generator)
    try:
        replies = generator.generate(
            case["unreplied"], case["all_messages"], is_group=case["is_group"],
            tick_id=case["source_tick_id"], evaluation_time=case["evaluation_time"],
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
            "status": "failed", "replies": [],
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.monotonic() - started, 3),
        }


def _option_map(case_id: str) -> dict[str, str]:
    baseline_is_a = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0
    return {"A": "baseline" if baseline_is_a else "candidate", "B": "candidate" if baseline_is_a else "baseline"}


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _render(payload: dict[str, Any]) -> str:
    counts = {key: 0 for key in REVIEW_LABELS}
    cards = []
    for row in payload["cases"]:
        review = row.get("review") or {"classification": "tie", "reason": "尚未复核。"}
        classification = review["classification"]
        counts[classification] += 1
        mapping = row["blind_mapping"]
        context = "".join(
            f'<div class="msg {"self" if item["role"] == "self" else "other"}"><b>{_e(item["speaker"])}</b><span>{_e(item["content"])}</span></div>'
            for item in row["context"]
        )
        options = []
        for label in ("A", "B"):
            result = row["runs"][mapping[label]]
            reply = " / ".join(result.get("replies") or []) or "（静默）"
            options.append(f'<div class="option"><b>回复 {label}</b><p>{_e(reply)}</p></div>')
        cards.append(
            f'<article class="case" data-tags="{"group" if row["is_group"] else "private"} {classification}">'
            f'<h2>{_e(row["chat_name"])}</h2><p class="verdict {classification}">{_e(REVIEW_LABELS[classification])}：{_e(review["reason"])}</p>'
            f'<div class="context">{context}</div>{"".join(options)}'
            f'<details><summary>揭晓版本</summary><p>A = {_e(mapping["A"])}；B = {_e(mapping["B"])}</p></details></article>'
        )
    group = {key: sum(row["is_group"] and row["review"]["classification"] == key for row in payload["cases"]) for key in REVIEW_LABELS}
    private = {key: sum(not row["is_group"] and row["review"]["classification"] == key for row in payload["cases"]) for key in REVIEW_LABELS}
    summary = " · ".join(f"{REVIEW_LABELS[key]} {counts[key]}" for key in REVIEW_LABELS)
    group_summary = " / ".join(str(group[key]) for key in REVIEW_LABELS)
    private_summary = " / ".join(str(private[key]) for key in REVIEW_LABELS)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>C0 第一轮匿名 A/B</title><style>
body{{margin:0;background:#f4f5f7;color:#202124;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:760px;margin:auto;padding:16px}}h1{{font-size:23px;margin:4px 0 8px}}.lede{{color:#59636e}}.summary{{font-weight:700}}.warning{{background:#fff4d8;border-radius:10px;padding:10px}}.filters{{position:sticky;top:0;background:#f4f5f7;padding:8px 0;z-index:2}}button{{border:1px solid #ccd2d8;background:white;border-radius:18px;padding:8px 13px;margin:3px 3px 3px 0}}.case{{background:white;border-radius:14px;padding:14px;margin:12px 0;box-shadow:0 2px 10px #0000000d}}h2{{font-size:17px;margin:0 0 10px}}.verdict{{border-radius:8px;padding:8px 10px}}.candidate_win{{background:#e8f5ec;color:#176f45}}.baseline_win{{background:#fdecec;color:#a12d2d}}.tie{{background:#eef1f4;color:#53606d}}.context{{background:#eef1f4;border-radius:10px;padding:8px;max-height:52vh;overflow:auto}}.msg{{display:flex;gap:8px;padding:6px;border-bottom:1px solid #dde1e5}}.msg b{{min-width:76px;color:#53606d}}.msg.self b{{color:#176f45}}.option{{border:1px solid #d8dde3;border-radius:10px;padding:10px;margin-top:10px}}.option b{{color:#3858a8}}.option p{{margin:6px 0 0;font-size:16px}}details{{margin-top:10px;color:#68727d}}.hidden{{display:none}}</style></head><body><main>
<h1>C0 第一轮匿名 A/B</h1><p class="lede">20 个已暴露训练案例；14 群聊 + 6 私聊。完整历史上下文、真实昵称，A/B 顺序按案例打乱。</p><p class="summary">总计：{_e(summary)}</p><p class="lede">群聊（候选胜 / 基线胜 / 平）：{_e(group_summary)}；私聊：{_e(private_summary)}。</p><p class="warning"><b>结论：</b>群聊局部净收益，但私聊出现严重退步，候选整包不采用，当前 C0 保持不变。</p><p class="lede">这是训练案例上的人工诊断，Judge 仅作辅助、不作为事实或金标。两组回复均由同一火山托管模型生成，唯一实验变量是三条候选 Prompt 规则；这不是当前正式生产端点的严格对照。Bot 始终关闭，未发送任何消息。</p><div class="filters"><button data-filter="all">全部</button><button data-filter="candidate_win">候选胜</button><button data-filter="baseline_win">基线胜</button><button data-filter="tie">持平</button><button data-filter="group">群聊</button><button data-filter="private">私聊</button></div>{''.join(cards)}
</main><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.querySelectorAll('.case').forEach(c=>c.classList.toggle('hidden',b.dataset.filter!=='all'&&!c.dataset.tags.split(' ').includes(b.dataset.filter))))</script></body></html>'''


def _add_review(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload["cases"]:
        classification, reason = REVIEW[row["case_id"]]
        row["review"] = {"classification": classification, "reason": reason, "reviewer": "codex_manual"}
    counts = {key: sum(row["review"]["classification"] == key for row in payload["cases"]) for key in REVIEW_LABELS}
    payload["review_summary"] = {
        **counts,
        "group": {key: sum(row["is_group"] and row["review"]["classification"] == key for row in payload["cases"]) for key in REVIEW_LABELS},
        "private": {key: sum(not row["is_group"] and row["review"]["classification"] == key for row in payload["cases"]) for key in REVIEW_LABELS},
        "decision": "candidate_not_adopted",
        "reason": "group gains are net positive, but private-chat factual and causal-fabrication regressions block adoption",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--provider", choices=("official", "volcengine"), default="official")
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/reports/private/c0_guardrail_ab_v1")
    parser.add_argument("--render-existing", type=Path)
    args = parser.parse_args()
    if args.render_existing:
        payload = _add_review(json.loads(args.render_existing.read_text(encoding="utf-8")))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.out_dir / "index.html").write_text(_render(payload), encoding="utf-8")
        print(json.dumps(payload["review_summary"], ensure_ascii=False))
        return 0
    _load_env()
    fingerprint, all_cases = load_replay_cases(
        ROOT / "data/cases.db",
        ROOT / "data/reports/private/reply_system_scenarios.json",
        ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    by_id = {case["case_id"]: case for case in all_cases}
    missing = [case_id for case_id in CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"missing cases: {missing}")
    cases = [by_id[case_id] for case_id in CASE_IDS]
    output = []
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case in cases:
            for version in ("baseline", "candidate"):
                futures[pool.submit(_run, case, version, args.provider, args.volcengine_config)] = (case["case_id"], version)
        results: dict[str, dict[str, Any]] = {case["case_id"]: {} for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            case_id, version = futures[future]
            results[case_id][version] = future.result()
            print(f"[{index}/{len(futures)}] {case_id} {version}", flush=True)
    for case in cases:
        context = [_serialize(message) for message in case["all_messages"]]
        unread_ids = {id(message) for message in case["unreplied"]}
        for item, message in zip(context, case["all_messages"], strict=True):
            item["unread"] = id(message) in unread_ids
        output.append({
            "case_id": case["case_id"], "chat_name": case["all_messages"][-1].chat_name,
            "is_group": case["is_group"], "context": context,
            "blind_mapping": _option_map(case["case_id"]), "runs": results[case["case_id"]],
        })
    payload = _add_review({
        "schema_version": 1, "status": "complete", "source_fingerprint": fingerprint,
        "generation_provider": args.provider,
        "scope": {"total": len(output), "group": sum(row["is_group"] for row in output), "private": sum(not row["is_group"] for row in output)},
        "only_difference": "candidate adds three C0 guardrails: no redundant obvious explanations; no unsupported real-world commitments; one short group reply by default",
        "cases": output,
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (args.out_dir / "index.html").write_text(_render(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", **payload["scope"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
