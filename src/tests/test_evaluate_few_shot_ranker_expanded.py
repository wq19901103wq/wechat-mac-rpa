import hashlib
import json

import pytest

from scripts.evaluate_few_shot_ranker_expanded import evaluate


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(sample_id, target_id, relationship, candidate_id):
    return {
        "sample_id": sample_id,
        "target": {
            "case_id": target_id,
            "relationship": relationship,
            "context": ["私密上下文不得进入结果"],
        },
        "candidate": {"id": candidate_id, "reply": ["私密回复不得进入结果"]},
    }


def _label(sample_id, relationship, candidate_id, label, probability, status="ok"):
    if status != "ok":
        return {"sample_id": sample_id, "status": status, "error_type": "C0GenerationError"}
    return {
        "sample_id": sample_id,
        "relationship": relationship,
        "injected_candidate_id": candidate_id,
        "label": label,
        "probability": probability,
        "status": "ok",
    }


def _sources(tmp_path, selections, samples, labels, variant="xgboost"):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    selections_path = plan_dir / "selections.jsonl"
    _write_jsonl(selections_path, selections)
    (plan_dir / "manifest.json").write_text(json.dumps({
        "kind": f"few_shot_ranker_expanded_{variant}_selection_plan",
        "files": {"selections_sha256": _sha(selections_path)},
    }))

    sample_path = tmp_path / "samples.jsonl"
    _write_jsonl(sample_path, samples)
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    label_path = label_dir / "labels.jsonl"
    _write_jsonl(label_path, labels)
    (label_dir / "manifest.json").write_text(json.dumps({
        "samples_path": str(sample_path),
        "samples_sha256": _sha(sample_path),
        "judge_config": {"baseline_id": "judge-current"},
    }))
    return selections_path, (sample_path,), (label_path,)


def test_expanded_evaluation_reuses_successful_retry_and_blocks_incomplete_coverage(tmp_path):
    selections = [
        {
            "target_id": "group-improved", "relationship": "group",
            "production_candidate_id": "p1", "xgboost_candidate_id": "x1",
        },
        {
            "target_id": "private-same", "relationship": "private",
            "production_candidate_id": "p2", "xgboost_candidate_id": "p2",
        },
        {
            "target_id": "group-missing", "relationship": "group",
            "production_candidate_id": "p3", "xgboost_candidate_id": "x3",
        },
    ]
    samples = [
        _sample("production:p1", "group-improved", "group", "p1"),
        _sample("xgboost:x1", "group-improved", "group", "x1"),
        _sample("production:p2", "private-same", "private", "p2"),
        _sample("production:p3", "group-missing", "group", "p3"),
        _sample("xgboost:x3", "group-missing", "group", "x3"),
    ]
    labels = [
        _label("production:p1", "group", "p1", 0, 0.2),
        _label("xgboost:x1", "group", "x1", 0, 0.1, status="error"),
        _label("xgboost:x1", "group", "x1", 1, 0.8),
        _label("xgboost:x1", "group", "x1", 0, 0.1, status="error"),
        _label("production:p2", "private", "p2", 1, 0.6),
        _label("production:p3", "group", "p3", 0, 0.3),
        _label("xgboost:x3", "group", "x3", 0, 0.1, status="error"),
    ]
    paths = _sources(tmp_path, selections, samples, labels)

    result = evaluate(
        *paths,
        candidate_variant="xgboost",
        c0_baseline_id="c0-current",
        judge_baseline_id="judge-current",
    )

    assert result["coverage"]["complete_targets"] == 2
    assert result["coverage"]["excluded_targets"] == [{
        "target_id": "group-missing",
        "relationship": "group",
        "missing_production_label": False,
        "missing_candidate_label": True,
        "reason": "incomplete_verified_context_replay",
    }]
    assert result["metrics"]["overall"]["net_human_like_gain"] == 1
    assert result["metrics"]["overall"]["improved"] == 1
    assert result["promotion_gate"]["directional_gate_passed"] is True
    assert result["promotion_gate"]["eligible_for_quality_review"] is False
    assert result["promotion_gate"]["decision"] == "hold_incomplete_paired_coverage"
    assert "私密" not in json.dumps(result, ensure_ascii=False)


def test_expanded_evaluation_advances_only_complete_non_regressing_result(tmp_path):
    selections = [
        {
            "target_id": "group", "relationship": "group",
            "production_candidate_id": "pg", "lr_candidate_id": "lg",
        },
        {
            "target_id": "private", "relationship": "private",
            "production_candidate_id": "pp", "lr_candidate_id": "lp",
        },
    ]
    samples = [
        _sample("production:pg", "group", "group", "pg"),
        _sample("lr:lg", "group", "group", "lg"),
        _sample("production:pp", "private", "private", "pp"),
        _sample("lr:lp", "private", "private", "lp"),
    ]
    labels = [
        _label("production:pg", "group", "pg", 0, 0.2),
        _label("lr:lg", "group", "lg", 1, 0.7),
        _label("production:pp", "private", "pp", 1, 0.6),
        _label("lr:lp", "private", "lp", 1, 0.8),
    ]
    paths = _sources(tmp_path, selections, samples, labels, variant="lr")

    result = evaluate(
        *paths,
        candidate_variant="lr",
        c0_baseline_id="c0-current",
        judge_baseline_id="judge-current",
    )

    assert result["promotion_gate"]["coverage_complete"] is True
    assert result["promotion_gate"]["eligible_for_quality_review"] is True
    assert result["promotion_gate"]["decision"] == "advance_to_explicit_quality_review"


def test_expanded_evaluation_rejects_conflicting_pair_labels(tmp_path):
    selections = [{
        "target_id": "target", "relationship": "group",
        "production_candidate_id": "same", "xgboost_candidate_id": "same",
    }]
    samples = [
        _sample("production:same", "target", "group", "same"),
        _sample("xgboost:same", "target", "group", "same"),
    ]
    labels = [
        _label("production:same", "group", "same", 0, 0.2),
        _label("xgboost:same", "group", "same", 1, 0.8),
    ]
    paths = _sources(tmp_path, selections, samples, labels)

    with pytest.raises(ValueError, match="conflicting labels"):
        evaluate(
            *paths,
            candidate_variant="xgboost",
            c0_baseline_id="c0-current",
            judge_baseline_id="judge-current",
        )
