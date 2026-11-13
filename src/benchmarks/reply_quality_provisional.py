"""Merge AI provisional annotations without treating them as Human Gold."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any


GRADES = {"good", "acceptable", "bad", "critical"}
ACTIONS = {"reply", "silence"}


def apply_provisional_annotations(
    packet: dict[str, Any], annotations: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {item["case_id"]: item for item in annotations}
    cases = packet.get("cases", [])
    case_ids = {case["case_id"] for case in cases}
    if set(by_id) != case_ids:
        raise ValueError("粗标 case_id 必须与待标包完全一致")

    grades: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    confidence_bands: Counter[str] = Counter()
    for case in cases:
        annotation = by_id[case["case_id"]]
        if annotation.get("experience_grade") not in GRADES:
            raise ValueError("粗标体验等级无效")
        if annotation.get("recommended_action") not in ACTIONS:
            raise ValueError("粗标回复决策无效")
        if not str(annotation.get("reason") or "").strip():
            raise ValueError("粗标必须说明理由")
        case["ai_provisional"] = {
            **annotation,
            "status": "ai_provisional_not_human_ground_truth",
        }
        grades[annotation["experience_grade"]] += 1
        actions[annotation["recommended_action"]] += 1
        confidence_bands[annotation.get("confidence", "low")] += 1

    generated_at = datetime.now().astimezone().isoformat()
    packet["ai_provisional_generated_at"] = generated_at
    report = {
        "schema_version": 1,
        "benchmark_kind": "reply_quality_ai_provisional",
        "source_fingerprint": packet.get("source_fingerprint"),
        "cases": len(cases),
        "grade_counts": {
            grade: grades[grade]
            for grade in ("good", "acceptable", "bad", "critical")
        },
        "recommended_action_counts": dict(sorted(actions.items())),
        "confidence_counts": dict(sorted(confidence_bands.items())),
        "status": "ai_provisional_not_human_ground_truth",
        "eligible_to_calibrate_judge": False,
        "interpretation": (
            "粗标依据完整上下文、历史真人表达方式和已有历史信号；"
            "它用于形成首轮问题地图，不能代替 Human Gold 或放行 Judge/A/B。"
        ),
        "generated_at": generated_at,
    }
    return packet, report
