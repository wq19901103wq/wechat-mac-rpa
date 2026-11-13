#!/usr/bin/env python3
"""Build the one-case, leak-audited group-queue Judge few-shot manifest."""

from __future__ import annotations

import argparse
import copy
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_judge_dynamic_few_shots import (  # noqa: E402
    _contexts_overlap,
    _normalized_messages,
    extract_timeline,
)
from scripts.evaluate_human_ai_judge import _json_sha256, _write_private  # noqa: E402


REPORT_DIR = ROOT / "data/reports/private/human_ai_judge_benchmark_v3_current_c0_f977633e9391c0f4"
TARGET_CASE_ID = "case_8edbf4a18421c7bce8ee9f68"
PROTECTION_CASE_IDS = {
    "case_3218ed5a029c181b6f6b8ba9",
    "case_6d177802c158c9e0f22d8305",
    "case_ee2c91a094f817f33e474e43",
    "case_e4f2c171af41d8953d4edf08",
}
EXPECTED_BLIND_SHA256 = "f9b7bd17ca207dd9505363660b5b4187eef985366106b67d994781b71a1e6f33"
EXPECTED_REFERENCE_SHA256 = "bbb9cf146ade297e80d8fb67cc7e77da1ced07a248bdbcb6be717b97d9119dbb"
EXPECTED_POOL_SHA256 = "279a178f00bff170ea81a510893f6a1000aa83d0e4b6064851864eec494bc8da"
HISTORY_CUTOFF = 1774972800

REPEAT_EXPORT = ROOT / "data/exports/b/群聊_腾讯亚洲购房买买买.json"
RIFF_EXPORT = ROOT / "data/exports/b/群聊_打工人和退休干部们.json"

REPEAT_WINDOW = (
    ("5236419268632461206", 0, "Thinker", "为什么人家都发财了"),
    ("7696491650584431329", 0, "幸福的矿工-卢卡斯-施梦圜", "为什么人家都发财了"),
    ("8856856170938778002", 1, "王芊", "为什么人家都发财了"),
)
RIFF_WINDOW = (
    ("2092032555287924563", 0, "解俊杰-thd-谢伟", "以前羡慕住在金融家的，现在还得羡慕住在招商外滩玺的"),
    ("534385648482356579", 0, "杨天宇", "以前羡慕住在金融家的，现在还得羡慕住在招商外滩玺的"),
    ("4860669581607879820", 0, "榊原中二-汪亦茂", "以前羡慕住在金融家的，现在还得羡慕住在招商外滩玺的"),
    ("4708170946530707888", 1, "王芊", "以前羡慕住在金融家的，现在还得羡慕住在金融家的"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: Any) -> str:
    return re.sub(r"\s+", "", html.unescape(str(value or ""))).strip()


def _without_speaker(value: str) -> str:
    return re.sub(r"^[^：:\n]{1,80}[：:]\s*", "", value, count=1).strip()


def _validate_window(path: Path, expected: tuple[tuple[str, int, str, str], ...]) -> list[dict[str, Any]]:
    messages = list(_load(path).get("messages") or [])
    target_id = expected[-1][0]
    matching_indices = [
        index for index, row in enumerate(messages)
        if str(row.get("platformMessageId") or "") == target_id
    ]
    if len(matching_indices) != 1:
        raise ValueError(f"{path.name}: expected one message {target_id}, found {len(matching_indices)}")
    end = matching_indices[0] + 1
    rows = messages[end - len(expected):end]
    if len(rows) != len(expected):
        raise ValueError(f"{path.name}: historical queue does not have the required predecessors")
    for row, (message_id, is_send, sender, content) in zip(rows, expected):
        actual = (
            str(row.get("platformMessageId") or ""),
            int(row.get("isSend") or 0),
            str(row.get("senderDisplayName") or ""),
            _plain(row.get("content")),
        )
        wanted = (message_id, is_send, sender, _plain(content))
        if actual != wanted:
            raise ValueError(f"{path.name}: historical queue changed at {message_id}")
        if int(row.get("createTime") or 0) >= HISTORY_CUTOFF:
            raise ValueError(f"{path.name}: historical example is not earlier than the fixed cutoff")
    return rows


def _has_two_member_terminal_queue(timeline: list[dict[str, str]]) -> bool:
    if len(timeline) < 2:
        return False
    left, right = timeline[-2:]
    return (
        left.get("role") == "other"
        and right.get("role") == "other"
        and bool(left.get("sender"))
        and bool(right.get("sender"))
        and left["sender"] != right["sender"]
        and _plain(left.get("text")) == _plain(right.get("text"))
    )


def _example_from_window(
    *, example_id: str, rows: list[dict[str, Any]], reply_text: str
) -> dict[str, Any]:
    return {
        "id": example_id,
        "relationship": "group",
        "context": [
            f"{row['senderDisplayName']}：{str(row.get('content') or '').strip()}"
            for row in rows[:-1]
        ],
        "human_reply": [f"我：{reply_text}"],
    }


def _leakage_counts(
    examples: list[dict[str, Any]], target: dict[str, Any], timeline: list[dict[str, str]]
) -> dict[str, int]:
    target_messages = [message["text"] for message in timeline]
    option_pairs = {
        _normalized_messages(option)
        for option in (target.get("option_A") or [], target.get("option_B") or [])
    }
    pair_overlap = 0
    long_message_overlap = 0
    reply_overlap = 0
    target_normalized = _normalized_messages(target_messages)
    target_pairs = {
        target_normalized[index:index + 2]
        for index in range(len(target_normalized) - 1)
    }
    target_long = {message for message in target_normalized if len(message) >= 12}
    for example in examples:
        context = [_without_speaker(value) for value in example["context"]]
        reply = [_without_speaker(value) for value in example["human_reply"]]
        historical = _normalized_messages(context + reply)
        historical_pairs = {
            historical[index:index + 2]
            for index in range(len(historical) - 1)
        }
        pair_overlap += int(bool(historical_pairs & target_pairs))
        long_message_overlap += int(bool({value for value in historical if len(value) >= 12} & target_long))
        reply_overlap += int(_normalized_messages(reply) in option_pairs)
        if _contexts_overlap(context, reply, [target_messages]):
            raise ValueError(f"dynamic example {example['id']} overlaps the target context")
    return {
        "examples_with_consecutive_two_message_overlap": pair_overlap,
        "examples_with_complete_message_overlap_at_least_12_chars": long_message_overlap,
        "examples_with_exact_option_reply": reply_overlap,
    }


def build(report_dir: Path) -> tuple[Path, Path]:
    blind = _load(report_dir / "dev_blind.json")
    answers = _load(report_dir / "dev_answers.json")
    reference = _load(report_dir / "judge_reference.json")
    base_manifest = _load(report_dir / "judge_dynamic_few_shot_behavior_v7_manifest.json")
    if _json_sha256(blind) != EXPECTED_BLIND_SHA256:
        raise ValueError("fixed blind packet changed")
    if str(reference.get("reference_sha256") or "") != EXPECTED_REFERENCE_SHA256:
        raise ValueError("fixed Judge reference changed")
    if base_manifest.get("blind_sha256") != EXPECTED_BLIND_SHA256:
        raise ValueError("base manifest blind hash changed")
    if base_manifest.get("reference_sha256") != EXPECTED_REFERENCE_SHA256:
        raise ValueError("base manifest reference hash changed")
    if base_manifest.get("pool_sha256") != EXPECTED_POOL_SHA256:
        raise ValueError("base manifest historical pool changed")

    blind_cases = list(blind.get("cases") or [])
    answer_cases = list(answers.get("cases") or [])
    if len(blind_cases) != 160 or len(answer_cases) != 160:
        raise ValueError("fixed benchmark must contain exactly 160 cases and 160 answers")
    relationship_counts = {
        relationship: sum(row.get("relationship") == relationship for row in blind_cases)
        for relationship in ("private", "group")
    }
    if relationship_counts != {"private": 80, "group": 80}:
        raise ValueError(f"fixed relationship split changed: {relationship_counts}")
    case_by_id = {str(row["case_id"]): row for row in blind_cases}
    if len(case_by_id) != 160 or TARGET_CASE_ID not in case_by_id:
        raise ValueError("fixed benchmark case ids changed")
    if not PROTECTION_CASE_IDS.issubset(case_by_id):
        raise ValueError("one or more protection cases are missing")
    target = case_by_id[TARGET_CASE_ID]
    timeline = extract_timeline(list(target.get("context_original") or []))
    if target.get("relationship") != "group" or not _has_two_member_terminal_queue(timeline):
        raise ValueError("target case no longer ends in a two-member exact queue")

    repeat_rows = _validate_window(REPEAT_EXPORT, REPEAT_WINDOW)
    riff_rows = _validate_window(RIFF_EXPORT, RIFF_WINDOW)
    examples = [
        _example_from_window(
            example_id="group_queue_repeat_20200429_8856856170938778002",
            rows=repeat_rows,
            reply_text="为什么人家都发财了",
        ),
        _example_from_window(
            example_id="group_queue_riff_20250128_4708170946530707888",
            rows=riff_rows,
            reply_text="以前羡慕住在金融家的，现在还得羡慕住在金融家的",
        ),
    ]
    leakage = _leakage_counts(examples, target, timeline)
    if any(leakage.values()):
        raise ValueError(f"queue few-shot leakage audit failed: {leakage}")

    manifest = copy.deepcopy(base_manifest)
    for row in manifest.get("cases") or []:
        row["examples"] = examples if row.get("case_id") == TARGET_CASE_ID else []
    triggered = [row for row in manifest["cases"] if row.get("examples")]
    private_triggered = [row for row in triggered if row.get("relationship") == "private"]
    non_target_triggered = [row for row in triggered if row.get("case_id") != TARGET_CASE_ID]
    if len(triggered) != 1 or private_triggered or non_target_triggered:
        raise ValueError("queue few-shot manifest triggered outside the single approved target")
    if any(
        next(row for row in manifest["cases"] if row["case_id"] == case_id)["examples"]
        for case_id in PROTECTION_CASE_IDS
    ):
        raise ValueError("a protection case unexpectedly received a dynamic example")

    output_manifest = report_dir / "judge_group_queue_few_shot_manifest.json"
    output_audit = report_dir / "judge_group_queue_few_shot_leakage_audit.json"
    audit = {
        "schema_version": 1,
        "kind": "judge_group_queue_few_shot_leakage_audit",
        "blind_sha256": EXPECTED_BLIND_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "pool_sha256": EXPECTED_POOL_SHA256,
        "total_case_count": len(blind_cases),
        "private_case_count": relationship_counts["private"],
        "group_case_count": relationship_counts["group"],
        "triggered_case_count": len(triggered),
        "private_triggered_case_count": len(private_triggered),
        "non_target_triggered_case_count": len(non_target_triggered),
        "target_case_id": TARGET_CASE_ID,
        "target_terminal_queue_member_count": 2,
        "target_dynamic_example_count": len(examples),
        "protection_case_count": len(PROTECTION_CASE_IDS),
        "protection_cases_with_dynamic_examples": 0,
        **leakage,
        "example_ids": [example["id"] for example in examples],
    }
    _write_private(output_manifest, manifest)
    _write_private(output_audit, audit)
    return output_manifest, output_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    manifest, audit = build(args.report_dir)
    print(json.dumps({"manifest": str(manifest), "audit": str(audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
