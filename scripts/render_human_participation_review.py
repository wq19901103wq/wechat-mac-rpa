#!/usr/bin/env python3
"""Render the private participation replay as a mobile-friendly Markdown review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_MODE = 0o600


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _result_text(result: dict[str, Any]) -> str:
    action = result.get("action") or "failed"
    replies = " / ".join(result.get("replies") or [])
    suffix = f"：{replies}" if replies else ""
    return f"{action}{suffix}（{result.get('duration_seconds', '?')}s）"


def _judge(expected: str, runs: list[dict[str, Any]]) -> tuple[str, str]:
    actions = [run.get("action") for run in runs]
    if any(action in {None, "generation_failed"} for action in actions):
        return "失败", "存在异常或生成失败，不能把空结果算作正确沉默。"
    if all(action == expected for action in actions):
        return "通过", f"3 次都稳定选择 {expected}，与真人历史行为一致。"
    if len(set(actions)) > 1:
        return "不稳定", f"3 次参与决定不一致：{' / '.join(map(str, actions))}。"
    return "未通过", f"3 次都选择 {actions[0]}，但真人历史行为是 {expected}。"


def _context_lines(evidence: dict[str, Any]) -> list[str]:
    values = evidence.get("context_original") or []
    lines = []
    for value in values:
        if isinstance(value, dict):
            who = "王芊" if value.get("is_self") else value.get("sender", "未知")
            lines.append(
                f"- {value.get('time', '')}｜{who}｜{value.get('type', '')}：{value.get('text', '')}"
            )
        else:
            lines.append(f"- 群友：{value}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/human_participation_ab_replay.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/human_participation_ab_review.md",
    )
    args = parser.parse_args()
    packet = _load(args.input)
    if packet.get("status") != "completed":
        raise ValueError("replay is not complete")

    counts = {"C0": 0, "P1": 0}
    stable_silence = {"C0": 0, "P1": 0}
    per_run_correct = {"C0": 0, "P1": 0}
    lines = [
        "# 真人参与判断：C0 与 P1 离线复核",
        "",
        "本报告只评价系统该回复还是该沉默，不评价回复措辞。真人历史行为是临时参照，Codex 直接判定；DeepSeek 只生成待测回复。",
        "",
        "共 12 个 2026-04 前真人场景，真人实际回复 6 个、真人实际沉默 6 个。C0 与 P1 各回放 3 次，共 72 次；异常和生成失败均为 0。",
        "",
        "重要限制：部分真人回复场景已存在于当前 few-shot 素材中，模型可能直接检索并复现。因此本组只能作为参与判断 sanity set，不能证明对未见场景的泛化或表达风格学习效果。",
        "",
        "## 汇总结论",
        "",
    ]

    case_reviews = []
    for case in packet.get("cases", []):
        expected = case["expected_action"]
        review = {}
        for version in ("C0", "P1"):
            runs = case["runs"][version]
            verdict, reason = _judge(expected, runs)
            review[version] = (verdict, reason)
            if verdict == "通过":
                counts[version] += 1
                if expected == "silence":
                    stable_silence[version] += 1
            per_run_correct[version] += sum(run.get("action") == expected for run in runs)
        case_reviews.append((case, review))

    lines.extend(
        [
            f"- C0：严格稳定通过 {counts['C0']}/12；逐次正确 {per_run_correct['C0']}/36；6 个核心沉默场景稳定通过 {stable_silence['C0']}/6。",
            f"- P1：严格稳定通过 {counts['P1']}/12；逐次正确 {per_run_correct['P1']}/36；6 个核心沉默场景稳定通过 {stable_silence['P1']}/6。",
            "- 结论：P1 没有改善参与判断，并在 `reply_verified_01` 从稳定回复退化为一次漏回；不采纳 P1，继续保留 C0。",
            "- 共同缺口：两个版本都稳定漏回 `去哪吃`、`这样装是上面没有水管的对吧`、`好` 这三种真人会自然参与、但没有直接点名或问 Bot 的场景。",
            "",
            "## 逐案例原始证据",
            "",
        ]
    )

    for index, (case, review) in enumerate(case_reviews, 1):
        evidence = case["human_evidence"]
        human_replies = evidence.get("human_reply_original") or []
        lines.extend(
            [
                f"### {index}. {case['case_id']}",
                "",
                "聊天上下文：",
                "",
                *_context_lines(evidence),
                "",
            ]
        )
        if case["expected_action"] == "reply":
            lines.append(f"真人原始回复：{' / '.join(human_replies)}")
        else:
            lines.append(
                "真人实际表现：目标窗口内保持沉默（王芊发送消息数为 0）。"
            )
        lines.extend(["", "C0 三次：", ""])
        lines.extend(
            f"- 第{run['run']}次：{_result_text(run)}" for run in case["runs"]["C0"]
        )
        lines.extend(
            [
                "",
                f"Codex 判断：{review['C0'][0]}。{review['C0'][1]}",
                "",
                "P1 三次：",
                "",
            ]
        )
        lines.extend(
            f"- 第{run['run']}次：{_result_text(run)}" for run in case["runs"]["P1"]
        )
        lines.extend(
            [
                "",
                f"Codex 判断：{review['P1'][0]}。{review['P1'][1]}",
                "",
            ]
        )
        if evidence.get("limitation"):
            lines.extend([f"证据限制：{evidence['limitation']}", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    args.output.chmod(PRIVATE_MODE)
    print(args.output)


if __name__ == "__main__":
    main()
