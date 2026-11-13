import json

import pytest

from src.badcase.judge_config import (
    current_judge_baseline_id,
    load_current_judge,
    require_official_judge_runnable,
)


def test_current_judge_is_pending_240_direct_llm_and_bound_to_current_c0():
    config = load_current_judge()

    assert config["prediction_type"] == "direct_llm_pairwise"
    assert config["status"] == "pending_regeneration"
    assert config["dataset"] == {
        "path": None,
        "sha256": None,
        "total_cases": 240,
        "private_cases": None,
        "group_cases": None,
    }
    assert config["official_judge_baseline"] is False
    assert config["runnable_as_official"] is False
    with pytest.raises(RuntimeError, match="unavailable until the 240-case baseline is adopted"):
        current_judge_baseline_id()


def test_current_judge_fails_closed_while_official_baseline_is_incomplete():
    with pytest.raises(RuntimeError, match="baseline requirements are missing"):
        require_official_judge_runnable(load_current_judge())


def test_official_judge_private_and_group_counts_must_sum_to_240(tmp_path):
    config = load_current_judge().copy()
    config.pop("config_path", None)
    config.pop("profile_path", None)
    config["dataset"] = dict(config["dataset"], private_cases=80, group_cases=80)
    path = tmp_path / "invalid-counts.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="must sum to 240"):
        load_current_judge(path)


def test_feature_classifier_config_is_rejected_as_official(tmp_path):
    path = tmp_path / "feature.json"
    path.write_text(json.dumps({"experiment_type": "feature_classifier_experiment"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot be loaded as the official Judge"):
        load_current_judge(path)
