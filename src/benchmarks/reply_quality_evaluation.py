"""Metrics and hard gates for the reply-quality data flywheel.

This module never calls a model.  It evaluates already collected human Gold,
Judge predictions, and anonymous pairwise A/B decisions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


GRADES = ("good", "acceptable", "bad", "critical")
GRADE_INDEX = {grade: index for index, grade in enumerate(GRADES)}
PREFERENCES = ("control", "treatment", "tie")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _required_thresholds(policy: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    return [name for name in names if policy.get(name) is None]


def evaluate_judge_calibration(
    gold_cases: list[dict[str, Any]],
    judge_predictions: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Compare Judge outputs with human Gold without inventing missing labels."""
    gold = {
        case["case_id"]: case
        for case in gold_cases
        if case.get("human_review", {}).get("status") == "completed"
        and case["human_review"].get("action") in {"reply", "silence"}
        and case["human_review"].get("experience_grade") in GRADES
    }
    predictions = {case["case_id"]: case for case in judge_predictions}
    matched_ids = sorted(gold.keys() & predictions.keys())

    action_matches = 0
    grade_matches = 0
    adjacent_grade_matches = 0
    human_bad = 0
    predicted_bad = 0
    bad_true_positive = 0
    human_critical = 0
    critical_detected = 0
    disagreements = Counter()

    for case_id in matched_ids:
        review = gold[case_id]["human_review"]
        prediction = predictions[case_id]
        if prediction.get("action") == review.get("action"):
            action_matches += 1
        else:
            disagreements["reply_decision"] += 1

        human_grade = review.get("experience_grade")
        predicted_grade = prediction.get("experience_grade")
        if human_grade == predicted_grade:
            grade_matches += 1
        else:
            disagreements["experience_grade"] += 1
        if human_grade in GRADE_INDEX and predicted_grade in GRADE_INDEX:
            if abs(GRADE_INDEX[human_grade] - GRADE_INDEX[predicted_grade]) <= 1:
                adjacent_grade_matches += 1

        is_human_bad = human_grade in {"bad", "critical"}
        is_predicted_bad = predicted_grade in {"bad", "critical"}
        human_bad += is_human_bad
        predicted_bad += is_predicted_bad
        bad_true_positive += is_human_bad and is_predicted_bad
        human_critical += human_grade == "critical"
        critical_detected += human_grade == "critical" and predicted_grade == "critical"

    total = len(matched_ids)
    metrics = {
        "action_accuracy": _ratio(action_matches, total),
        "grade_exact_accuracy": _ratio(grade_matches, total),
        "grade_adjacent_accuracy": _ratio(adjacent_grade_matches, total),
        "bad_precision": _ratio(bad_true_positive, predicted_bad),
        "bad_recall": _ratio(bad_true_positive, human_bad),
        "critical_recall": _ratio(critical_detected, human_critical),
    }
    required = (
        "min_gold_cases",
        "min_action_accuracy",
        "min_grade_adjacent_accuracy",
        "min_bad_precision",
        "min_bad_recall",
        "min_critical_recall",
    )
    missing_thresholds = _required_thresholds(policy, required)
    failures = []
    if not missing_thresholds:
        if total < policy["min_gold_cases"]:
            failures.append("insufficient_gold_cases")
        for metric_name, threshold_name in (
            ("action_accuracy", "min_action_accuracy"),
            ("grade_adjacent_accuracy", "min_grade_adjacent_accuracy"),
            ("bad_precision", "min_bad_precision"),
            ("bad_recall", "min_bad_recall"),
            ("critical_recall", "min_critical_recall"),
        ):
            value = metrics[metric_name]
            if value is None or value < policy[threshold_name]:
                failures.append(metric_name)

    if not gold:
        status = "human_gold_pending"
    elif not matched_ids:
        status = "judge_predictions_pending"
    elif missing_thresholds:
        status = "thresholds_pending"
    elif failures:
        status = "failed"
    else:
        status = "passed"

    return {
        "benchmark_kind": "judge_quality",
        "status": status,
        "gold_cases": len(gold),
        "matched_cases": total,
        "metrics": metrics,
        "disagreements": dict(sorted(disagreements.items())),
        "missing_thresholds": missing_thresholds,
        "failures": failures,
        "eligible_for_automatic_ab": status == "passed",
    }


def evaluate_ab(
    comparisons: list[dict[str, Any]],
    judge_calibration: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate anonymous pairwise A/B decisions behind the Judge gate."""
    valid = [item for item in comparisons if item.get("preference") in PREFERENCES]
    preferences = Counter(item["preference"] for item in valid)
    total = len(valid)
    treatment_critical = sum(bool(item.get("treatment_critical")) for item in valid)
    control_critical = sum(bool(item.get("control_critical")) for item in valid)
    metrics = {
        "treatment_win_rate": _ratio(preferences["treatment"], total),
        "control_win_rate": _ratio(preferences["control"], total),
        "tie_rate": _ratio(preferences["tie"], total),
        "treatment_regression_rate": _ratio(preferences["control"], total),
        "treatment_critical_count": treatment_critical,
        "control_critical_count": control_critical,
    }
    required = ("min_comparisons", "min_treatment_win_rate", "max_regression_rate")
    missing_thresholds = _required_thresholds(policy, required)
    failures = []
    if judge_calibration.get("status") != "passed":
        status = "judge_not_calibrated"
    elif not valid:
        status = "comparisons_pending"
    elif missing_thresholds:
        status = "thresholds_pending"
    else:
        if total < policy["min_comparisons"]:
            failures.append("insufficient_comparisons")
        if metrics["treatment_win_rate"] < policy["min_treatment_win_rate"]:
            failures.append("treatment_win_rate")
        if metrics["treatment_regression_rate"] > policy["max_regression_rate"]:
            failures.append("treatment_regression_rate")
        if treatment_critical > control_critical:
            failures.append("critical_regression")
        status = "rejected" if failures else "accepted"

    return {
        "benchmark_kind": "reply_system_ab",
        "status": status,
        "comparisons": total,
        "metrics": metrics,
        "missing_thresholds": missing_thresholds,
        "failures": failures,
        "eligible_to_keep_change": status == "accepted",
    }
