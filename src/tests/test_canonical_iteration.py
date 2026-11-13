import pytest

from scripts.run_canonical_iteration import preflight


def test_c0_preflight_uses_adopted_c0_without_model_calls():
    result = preflight("c0", "development", "one switch only")

    assert result["status"] == "ready"
    assert result["model_calls_started"] == 0
    assert result["current_c0_baseline_id"].startswith("c0-")
    assert result["next_command"] == "scripts/run_situation_c0_ab.py"


def test_fixed_test_is_marked_independent_validation_only():
    result = preflight("c0", "fixed_test", "one switch only")

    assert result["dataset_purpose"] == "independent_validation_only"


def test_judge_preflight_fails_before_model_calls_when_240_baseline_is_incomplete():
    with pytest.raises(RuntimeError, match="baseline requirements are missing"):
        preflight("judge", "development", "one switch only")
