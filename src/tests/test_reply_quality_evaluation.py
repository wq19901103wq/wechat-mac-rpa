from src.benchmarks.reply_quality_evaluation import (
    evaluate_ab,
    evaluate_judge_calibration,
)


JUDGE_POLICY = {
    "min_gold_cases": 2,
    "min_action_accuracy": 1.0,
    "min_grade_adjacent_accuracy": 1.0,
    "min_bad_precision": 1.0,
    "min_bad_recall": 1.0,
    "min_critical_recall": 1.0,
}


def _gold(case_id, action, grade):
    return {
        "case_id": case_id,
        "human_review": {
            "status": "completed",
            "action": action,
            "experience_grade": grade,
        },
    }


def test_judge_cannot_pass_without_human_gold_or_thresholds():
    assert evaluate_judge_calibration([], [], JUDGE_POLICY)["status"] == "human_gold_pending"
    result = evaluate_judge_calibration(
        [_gold("one", "reply", "good")],
        [{"case_id": "one", "action": "reply", "experience_grade": "good"}],
        {},
    )
    assert result["status"] == "thresholds_pending"
    assert not result["eligible_for_automatic_ab"]


def test_judge_reports_human_experience_agreement_and_risk_recall():
    gold = [_gold("one", "reply", "good"), _gold("two", "silence", "critical")]
    predictions = [
        {"case_id": "one", "action": "reply", "experience_grade": "good"},
        {"case_id": "two", "action": "silence", "experience_grade": "critical"},
    ]
    result = evaluate_judge_calibration(gold, predictions, JUDGE_POLICY)
    assert result["status"] == "passed"
    assert result["metrics"]["action_accuracy"] == 1.0
    assert result["metrics"]["critical_recall"] == 1.0
    assert result["eligible_for_automatic_ab"]


def test_ab_is_blocked_until_judge_passes():
    result = evaluate_ab(
        [{"preference": "treatment"}],
        {"status": "failed"},
        {"min_comparisons": 1, "min_treatment_win_rate": 0.5, "max_regression_rate": 0.5},
    )
    assert result["status"] == "judge_not_calibrated"
    assert not result["eligible_to_keep_change"]


def test_ab_rejects_critical_regression_even_when_treatment_wins():
    comparisons = [
        {"preference": "treatment", "treatment_critical": True},
        {"preference": "treatment"},
        {"preference": "control"},
    ]
    policy = {
        "min_comparisons": 3,
        "min_treatment_win_rate": 0.6,
        "max_regression_rate": 0.4,
    }
    result = evaluate_ab(comparisons, {"status": "passed"}, policy)
    assert result["status"] == "rejected"
    assert "critical_regression" in result["failures"]
