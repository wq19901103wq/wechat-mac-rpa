from pathlib import Path

from src.benchmarks.ocr_quality import (
    OCRBenchmarkCase,
    build_summary,
    compute_metrics,
    evaluate_case,
)


def _case(name: str = "case", cohort: str = "representative") -> OCRBenchmarkCase:
    return OCRBenchmarkCase(
        name=name,
        cohort=cohort,
        source="private-real",
        screenshot_path=Path("unused.png"),
        expected={
            "chat_name": "测试群（3）",
            "chat_list": [{"nickname": "测试群", "unread_count": "2"}],
            "messages": [
                {"sender": "对方", "text": "下午三点开会", "check_mode": "exact"},
                {"sender": "自己", "text": "收到", "check_mode": "exact"},
            ],
        },
    )


def test_perfect_case_passes_strict_rules():
    result = evaluate_case(_case(), {
        "chat_name": "测试群 (3)",
        "chat_list": [{"nickname": "测试群", "unread_count": "2"}],
        "messages": [
            {"sender": "对方", "text": "下午三点开会"},
            {"sender": "自己", "text": "收到"},
        ],
    })

    assert result.passed is True
    assert compute_metrics([result])["text_accuracy"] == 1.0


def test_one_sender_error_fails_whole_case():
    result = evaluate_case(_case(), {
        "chat_name": "测试群 (3)",
        "chat_list": [{"nickname": "测试群", "unread_count": "2"}],
        "messages": [
            {"sender": "自己", "text": "下午三点开会"},
            {"sender": "自己", "text": "收到"},
        ],
    })

    assert result.sender_accuracy == 0.5
    assert result.text_accuracy == 1.0
    assert result.passed is False


def test_summary_keeps_representative_and_regression_separate():
    representative = evaluate_case(_case("normal"), {
        "chat_name": "测试群 (3)",
        "chat_list": [{"nickname": "测试群", "unread_count": "2"}],
        "messages": [
            {"sender": "对方", "text": "下午三点开会"},
            {"sender": "自己", "text": "收到"},
        ],
    })
    regression_case = _case("hard", "regression")
    regression = evaluate_case(regression_case, {})
    summary = build_summary(
        [_case("normal"), regression_case],
        [representative, regression],
        [],
    )

    assert summary["cohorts"]["representative"]["pass_rate"] == 1.0
    assert summary["cohorts"]["regression"]["pass_rate"] == 0.0
