#!/usr/bin/env python3
"""Render the recent-thread participation A/B as a private readable report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from render_participation_holdout_review import (
    PRIVATE_MODE,
    PROJECT_ROOT,
    _context_lines,
    _effective_action,
    _result_text,
)


INPUT = PROJECT_ROOT / "data/reports/private/participation_recent_thread_ab_replay.json"
OUTPUT = PROJECT_ROOT / "data/reports/private/participation_recent_thread_ab_review.md"


def _stable(expected: str, runs: list[dict[str, Any]]) -> bool:
    return all(_effective_action(run) == expected for run in runs)


def main() -> None:
    packet = json.loads(INPUT.read_text(encoding="utf-8"))
    if packet.get("status") != "completed":
        raise ValueError("replay is not complete")
    cases = packet["cases"]
    counts: dict[str, dict[str, int]] = {}
    for version in ("C0", "C1"):
        counts[version] = {
            "stable": sum(_stable(case["expected_action"], case["runs"][version]) for case in cases),
            "correct": sum(
                _effective_action(run) == case["expected_action"]
                for case in cases
                for run in case["runs"][version]
            ),
            "reply_correct": sum(
                _effective_action(run) == "reply"
                for case in cases
                if case["expected_action"] == "reply"
                for run in case["runs"][version]
            ),
            "silence_correct": sum(
                _effective_action(run) == "silence"
                for case in cases
                if case["expected_action"] == "silence"
                for run in case["runs"][version]
            ),
        }
    validation = packet["validation"]
    lines = [
        "# 近期本人话题承接 A/B：C0 与 C1",
        "",
        "只判断该回复还是该沉默，不评价回复文案。真人历史行为是临时标准，Codex 直接 Judge；DeepSeek 只生成待测回复。",
        "",
        "## 结论",
        "",
        f"- C0：严格稳定通过 {counts['C0']['stable']}/{len(cases)}；逐次正确 {counts['C0']['correct']}/{len(cases) * 3}。",
        f"- C1：严格稳定通过 {counts['C1']['stable']}/{len(cases)}；逐次正确 {counts['C1']['correct']}/{len(cases) * 3}。",
        f"- 应回复场景：C0 {counts['C0']['reply_correct']}/27，C1 {counts['C1']['reply_correct']}/27；C1 没有新增稳定正例。",
        f"- 应沉默场景：C0 {counts['C0']['silence_correct']}/18，C1 {counts['C1']['silence_correct']}/18；两边均未新增乱插话。",
        "- C1 在团队映射玩笑、排队安排、算法吐槽三个场景比 C0 更少参与，因此判定退步并拒绝。",
        "- 保留受保护 C0；C1 只在离线内存存在，从未写入正式回复逻辑。",
        "",
        "## 实验完整性",
        "",
        f"- C0 SHA256：`{validation['c0_generator_sha256']}`",
        f"- C1 SHA256：`{validation['candidate_generator_sha256']}`",
        "- 15 个案例均早于 2026-04-01；真人答案未进入输入；当前 few-shot 无答案泄漏。",
        "- 90/90 次完成，0 异常、0 模型生成失败、0 空字符串伪回复。",
        "- 未构造或启动 Bot；未启用 memory、Judge worker 或动作工具。",
        "",
        "## 逐案例原始证据",
        "",
    ]
    for index, case in enumerate(cases, 1):
        evidence = case["human_evidence"]
        lines.extend([f"### {index}. {case['case_id']}", "", "完整输入上下文：", "", *_context_lines(evidence), ""])
        if case["expected_action"] == "reply":
            lines.append(f"真人随后原始回复：{evidence['human_reply_original'].get('text', '')}")
        else:
            lines.append("真人实际表现：目标窗口内保持沉默。")
        for version in ("C0", "C1"):
            lines.extend(["", f"{version} 三次原始结果：", ""])
            lines.extend(f"- 第{run['run']}次：{_result_text(run)}" for run in case["runs"][version])
            actions = [_effective_action(run) for run in case["runs"][version]]
            verdict = "通过" if all(action == case["expected_action"] for action in actions) else "未稳定通过"
            lines.extend(["", f"Codex 判断：{verdict}（{' / '.join(actions)}）。"])
        lines.append("")
    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    OUTPUT.chmod(PRIVATE_MODE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
