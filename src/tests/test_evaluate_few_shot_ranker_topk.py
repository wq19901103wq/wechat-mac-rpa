import hashlib
import json

import pytest

import scripts.evaluate_few_shot_ranker_topk as evaluator


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(sample_id, variant, candidate_ids):
    return {
        "sample_id": sample_id,
        "evaluation_variant": variant,
        "target": {"case_id": "target-1", "relationship": "group"},
        "candidates": [{"id": candidate_id} for candidate_id in candidate_ids],
        "generation": {
            "few_shot_count": 3,
            "static_few_shot_count": 0,
            "injected_candidate_ids": candidate_ids,
        },
    }


def _fixture(tmp_path):
    selections_path = tmp_path / "selections.jsonl"
    samples_path = tmp_path / "samples.jsonl"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    labels_path = labels_dir / "labels.jsonl"
    production_ids = ["candidate-1", "candidate-2", "candidate-3"]
    lr_ids = ["candidate-3", "candidate-1", "candidate-4"]
    _write_jsonl(selections_path, [{
        "target_id": "target-1",
        "relationship": "group",
        "selection_count": 3,
        "production_candidate_ids": production_ids,
        "lr_candidate_ids": lr_ids,
        "same_top3": False,
    }])
    _write_jsonl(samples_path, [
        _arm("production-arm", "production", production_ids),
        _arm("lr-arm", "lr", lr_ids),
    ])
    _write_jsonl(labels_path, [
        {
            "sample_id": "production-arm", "relationship": "group", "status": "ok",
            "label": 0, "probability": 0.2, "injected_candidate_ids": production_ids,
        },
        {
            "sample_id": "lr-arm", "relationship": "group", "status": "ok",
            "label": 1, "probability": 0.8, "injected_candidate_ids": lr_ids,
        },
    ])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "kind": "few_shot_ranker_expanded_lr_top3_selection_plan",
        "protocol": {"selection_count": 3},
        "counts": {"expanded_evaluation_targets": 1, "excluded_targets": []},
        "files": {"selections_sha256": _sha256(selections_path)},
    }), encoding="utf-8")
    (labels_dir / "manifest.json").write_text(json.dumps({
        "samples_path": str(samples_path),
        "samples_sha256": _sha256(samples_path),
        "judge_config": {"baseline_id": "judge-current"},
        "c0_config": {"baseline_id": "c0-current"},
        "parameters": {"bound_few_shot_count": 3},
    }), encoding="utf-8")
    return selections_path, samples_path, labels_path


def test_evaluate_top3_compares_full_arms_by_sample_id(tmp_path):
    selections_path, samples_path, labels_path = _fixture(tmp_path)

    result = evaluator.evaluate(
        selections_path,
        (samples_path,),
        (labels_path,),
        candidate_variant="lr",
        selection_count=3,
        c0_baseline_id="c0-current",
        judge_baseline_id="judge-current",
    )

    assert result["coverage"]["complete_targets"] == 1
    assert result["metrics"]["overall"]["net_human_like_gain"] == 1
    assert result["protocol"]["ranker_training_uses_single_candidate_labels"] is True
    assert result["learning_gate"]["directional_signal_positive"] is True


def test_evaluate_top3_rejects_label_with_different_injection_order(tmp_path):
    selections_path, samples_path, labels_path = _fixture(tmp_path)
    rows = evaluator._read_jsonl(labels_path)
    rows[1]["injected_candidate_ids"] = list(reversed(rows[1]["injected_candidate_ids"]))
    _write_jsonl(labels_path, rows)

    with pytest.raises(ValueError, match="does not match"):
        evaluator.evaluate(
            selections_path,
            (samples_path,),
            (labels_path,),
            candidate_variant="lr",
            selection_count=3,
            c0_baseline_id="c0-current",
            judge_baseline_id="judge-current",
        )


def _formal1000_fixture(tmp_path, *, hard_errors=0, with_quality=True):
    selections_path = tmp_path / "selections.jsonl"
    samples_path = tmp_path / "samples.jsonl"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    labels_path = labels_dir / "labels.jsonl"
    selections = []
    samples = []
    labels = []
    for index in range(1000):
        target_id = f"target-{index}"
        relationship = "group" if index < 667 else "private"
        production_ids = [f"p-{index}-{slot}" for slot in range(3)]
        changed = index == 0
        lr_ids = [f"lr-{index}-{slot}" for slot in range(3)] if changed else production_ids
        selections.append({
            "target_id": target_id,
            "relationship": relationship,
            "selection_count": 3,
            "production_candidate_ids": production_ids,
            "lr_candidate_ids": lr_ids,
            "same_top3": not changed,
        })
        production_sample_id = f"production-{index}"
        samples.append({
            "sample_id": production_sample_id,
            "evaluation_variant": "production",
            "target": {"case_id": target_id, "relationship": relationship},
            "candidates": [{"id": value} for value in production_ids],
            "generation": {
                "few_shot_count": 3, "static_few_shot_count": 0,
                "injected_candidate_ids": production_ids,
            },
        })
        labels.append({
            "sample_id": production_sample_id, "relationship": relationship, "status": "ok",
            "label": 0, "probability": 0.2, "injected_candidate_ids": production_ids,
        })
        if changed:
            samples.append({
                "sample_id": "lr-0",
                "evaluation_variant": "lr",
                "target": {"case_id": target_id, "relationship": relationship},
                "candidates": [{"id": value} for value in lr_ids],
                "generation": {
                    "few_shot_count": 3, "static_few_shot_count": 0,
                    "injected_candidate_ids": lr_ids,
                },
            })
            labels.append({
                "sample_id": "lr-0", "relationship": relationship, "status": "ok",
                "label": 1, "probability": 0.8, "injected_candidate_ids": lr_ids,
            })
    _write_jsonl(selections_path, selections)
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "kind": "few_shot_ranker_expanded_lr_top3_selection_plan",
        "protocol": {"selection_count": 3},
        "counts": {"expanded_evaluation_targets": 1000, "excluded_targets": []},
        "files": {"selections_sha256": _sha256(selections_path)},
    }), encoding="utf-8")
    (labels_dir / "manifest.json").write_text(json.dumps({
        "samples_path": str(samples_path),
        "samples_sha256": _sha256(samples_path),
        "judge_config": {"baseline_id": "judge-current"},
        "c0_config": {"baseline_id": "c0-current"},
        "parameters": {"bound_few_shot_count": 3},
    }), encoding="utf-8")
    quality_path = tmp_path / "quality.json"
    if with_quality:
        quality_path.write_text(json.dumps({
            "status": "completed",
            "reviewed_target_ids": ["target-0"],
            "new_hard_fact_errors": hard_errors,
            "selections_sha256": _sha256(selections_path),
        }), encoding="utf-8")
    return selections_path, samples_path, labels_path, quality_path


def test_formal1000_promotes_only_with_exact_coverage_and_clean_quality_review(tmp_path):
    selections, samples, labels, quality = _formal1000_fixture(tmp_path)

    result = evaluator.evaluate(
        selections, (samples,), (labels,), candidate_variant="lr", selection_count=3,
        c0_baseline_id="c0-current", judge_baseline_id="judge-current",
        evaluation_scope="formal1000", quality_review_path=quality,
    )

    assert result["coverage"]["complete_relationship_counts"] == {"group": 667, "private": 333}
    assert result["learning_gate"]["formal_promotion_eligible"] is True
    assert result["learning_gate"]["decision"] == "promote_update_default_c0"


def test_formal1000_missing_quality_review_blocks_promotion(tmp_path):
    selections, samples, labels, _quality = _formal1000_fixture(tmp_path, with_quality=False)

    result = evaluator.evaluate(
        selections, (samples,), (labels,), candidate_variant="lr", selection_count=3,
        c0_baseline_id="c0-current", judge_baseline_id="judge-current",
        evaluation_scope="formal1000",
    )

    assert result["learning_gate"]["formal_promotion_eligible"] is False
    assert result["learning_gate"]["decision"] == "hold_pending_complete_quality_review"


def test_formal1000_new_hard_fact_error_blocks_promotion(tmp_path):
    selections, samples, labels, quality = _formal1000_fixture(tmp_path, hard_errors=1)

    result = evaluator.evaluate(
        selections, (samples,), (labels,), candidate_variant="lr", selection_count=3,
        c0_baseline_id="c0-current", judge_baseline_id="judge-current",
        evaluation_scope="formal1000", quality_review_path=quality,
    )

    assert result["learning_gate"]["formal_promotion_eligible"] is False
    assert result["learning_gate"]["decision"] == "reject_new_hard_fact_errors"
