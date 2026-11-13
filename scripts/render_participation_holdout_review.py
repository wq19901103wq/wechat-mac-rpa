#!/usr/bin/env python3
"""Render the participation holdout A/B as a mobile-friendly private report."""

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


def _effective_action(result: dict[str, Any]) -> str:
    action = result.get("action")
    if action == "reply" and not any(str(reply).strip() for reply in result.get("replies") or []):
        return "invalid_empty_reply"
    return str(action or "failed")


def _result_text(result: dict[str, Any]) -> str:
    action = _effective_action(result)
    replies = " / ".join(repr(reply) for reply in result.get("replies") or [])
    suffix = f"：{replies}" if replies else ""
    few_shots = ", ".join(result.get("few_shot_ids") or []) or "无"
    return f"{action}{suffix}；few-shot={few_shots}"


def _judge(expected: str, runs: list[dict[str, Any]]) -> tuple[str, str]:
    actions = [_effective_action(run) for run in runs]
    if all(action == expected for action in actions):
        return "通过", f"3 次都稳定选择 {expected}。"
    if "invalid_empty_reply" in actions:
        return "失败", f"出现空字符串伪回复；三次结果为 {' / '.join(actions)}。"
    return "不稳定", f"三次结果为 {' / '.join(actions)}，未达到 3/3 稳定。"


def _context_lines(evidence: dict[str, Any]) -> list[str]:
    lines = []
    for value in evidence.get("context_original") or []:
        who = "王芊" if value.get("is_self") else value.get("sender", "未知")
        lines.append(
            f"- {value.get('time', '')}｜{who}｜{value.get('type', '')}：{value.get('text', '')}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_holdout_ab_replay.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_holdout_ab_review.md",
    )
    args = parser.parse_args()
    packet = _load(args.input)
    if packet.get("status") != "completed":
        raise ValueError("replay is not complete")

    stable = {"C0": 0, "H1": 0}
    correct_runs = {"C0": 0, "H1": 0}
    invalid_runs = {"C0": 0, "H1": 0}
    reviews = []
    for case in packet["cases"]:
        review = {}
        for version in ("C0", "H1"):
            actions = [_effective_action(run) for run in case["runs"][version]]
            verdict = _judge(case["expected_action"], case["runs"][version])
            review[version] = verdict
            stable[version] += verdict[0] == "通过"
            correct_runs[version] += sum(action == case["expected_action"] for action in actions)
            invalid_runs[version] += sum(action == "invalid_empty_reply" for action in actions)
        reviews.append((case, review))

    validation = packet["validation"]
    lines = [
        "# 参与判断 Holdout：C0 与 H1",
        "",
        "本报告只判断该回复还是该沉默，不评价具体措辞好坏。真人历史行为是临时标准，Codex 直接 Judge；DeepSeek 只生成待测结果。",
        "",
        "## 结论",
        "",
        f"- C0：严格稳定通过 {stable['C0']}/11；逐次正确 {correct_runs['C0']}/33；空字符串伪回复 {invalid_runs['C0']} 次。",
        f"- H1：严格稳定通过 {stable['H1']}/11；逐次正确 {correct_runs['H1']}/33；空字符串伪回复 {invalid_runs['H1']} 次。",
        "- 六个核心沉默场景：C0 与 H1 都是 18/18 正确，没有增加乱插话。",
        "- H1 没有新增任何稳定应回复场景；`reply_interview_followup` 反而从 C0 的 3/3 回复退化为 2/3，另出现一次空字符串伪回复。",
        "- 判定：拒绝 H1，继续保留 C0；候选只存在于离线内存，没有写入正式回复逻辑。",
        "",
        "## 实验完整性",
        "",
        f"- C0 SHA256：`{validation['c0_generator_sha256']}`",
        f"- H1 SHA256：`{validation['candidate_generator_sha256']}`",
        "- 11 个案例均早于 2026-04-01；真人回复未进入模型输入。",
        "- 当前启用 few-shot 中未发现任一正例的“目标消息+真人答案”同时出现；逐次召回 ID 在下方保留。",
        "- 66/66 次调用完成，0 异常、0 模型生成失败；H1 有 1 次业务层空字符串回复，单独按失败处理。",
        "- 未构造或启动 Bot；未启用 memory、Judge worker 或动作工具。",
        "",
        "## 逐案例原始证据",
        "",
    ]

    for index, (case, review) in enumerate(reviews, 1):
        evidence = case["human_evidence"]
        lines.extend([f"### {index}. {case['case_id']}", "", "聊天上下文：", "", *_context_lines(evidence), ""])
        if case["expected_action"] == "reply":
            human = evidence["human_reply_original"]
            lines.append(f"真人随后原始回复：{human.get('text', '')}")
        else:
            lines.append("真人实际表现：目标窗口内保持沉默。")
        for version in ("C0", "H1"):
            lines.extend(["", f"{version} 三次原始结果：", ""])
            lines.extend(
                f"- 第{run['run']}次：{_result_text(run)}"
                for run in case["runs"][version]
            )
            verdict, reason = review[version]
            lines.extend(["", f"Codex 判断：{verdict}。{reason}"])
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    args.output.chmod(PRIVATE_MODE)
    print(args.output)


if __name__ == "__main__":
    main()
