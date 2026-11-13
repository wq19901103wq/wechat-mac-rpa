import json

import pytest

from scripts.evaluate_few_shot_ranker import (
    CANDIDATE_ACTION_FEATURE_NAMES,
    InsufficientDataError,
    LabeledSample,
    _add_production_trace_features,
    _conservative_scores,
    _top_rank_limited_scores,
    evaluate,
    evaluate_train_dev,
    load_labeled_samples,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _dataset(private_targets, group_targets):
    samples = []
    labels = []
    for relationship, count in (("private", private_targets), ("group", group_targets)):
        for target_index in range(count):
            target_id = f"{relationship}-{target_index:02d}"
            for candidate_index in range(8):
                sample_id = f"{target_id}-candidate-{candidate_index}"
                samples.append({
                    "sample_id": sample_id,
                    "target": {"case_id": target_id, "relationship": relationship},
                    "candidate": {"id": f"candidate-{candidate_index}"},
                    "candidate_rank": candidate_index + 1,
                    "bootstrap_retrieval_score": 8 - candidate_index,
                    "local_features": {
                        "context_is_private": relationship == "private",
                        "candidate_matches_action": candidate_index == 7,
                    },
                })
                labels.append({
                    "sample_id": sample_id,
                    "status": "ok",
                    "label": int(candidate_index == 7),
                })
    return samples, labels


def _add_production_trace(samples):
    for sample in samples:
        rank = sample["candidate_rank"]
        sample["production_retrieval_features"] = {
            "production_lexical_score": float(9 - rank),
            "production_length_similarity": float(9 - rank),
            "production_semantic_similarity": float(9 - rank),
            "production_profile_compatibility": float(9 - rank),
            "production_group_participant_match_count": float(9 - rank),
            "production_pre_diversity_rank": float(rank),
            "production_response_move_exact_match": rank == 8,
            "production_precise_situation_match": rank == 7,
            "production_multi_message_reply_coverage": rank == 6,
            "production_diversity_skips_before_selection": 1.0 if rank == 5 else 0.0,
        }


def _target_action_rows(private_targets=8, group_targets=8):
    return [
        {
            "target_id": f"{relationship}-{target_index:02d}",
            "status": "ok",
            "features": {
                "needs_direct_response": True,
                "needs_clarification": False,
                "needs_light_reaction": False,
                "needs_coordination": False,
                    "requires_factual_boundary": False,
                    "has_explicit_addressee": False,
                    "is_natural_closure_or_confirmation": False,
                    "still_requires_interaction_progress": False,
                "is_natural_closure_or_confirmation": False,
                "still_requires_interaction_progress": False,
            },
        }
        for relationship, count in (("private", private_targets), ("group", group_targets))
        for target_index in range(count)
    ]


def _candidate_action_rows(samples, *, include_all=True):
    rows = []
    for sample in samples:
        candidate = sample["candidate"]
        candidate_id = candidate["id"]
        if any(row["candidate_id"] == candidate_id for row in rows):
            continue
        features = {name: False for name in CANDIDATE_ACTION_FEATURE_NAMES}
        if sample["candidate_rank"] == 8:
            features.update({
                "candidate_input_needs_direct_response": True,
                "candidate_reply_directly_addresses_request": True,
            })
        # This reply looks direct, but its historical input did not require a
        # direct response. It must not count as demonstrated action evidence.
        if sample["candidate_rank"] == 7:
            features["candidate_reply_directly_addresses_request"] = True
        rows.append({
            "candidate_id": candidate_id,
            "relationship": candidate.get("relationship") or (
                "acquaintance" if sample["target"]["relationship"] == "private" else "group"
            ),
            "status": "ok",
            "features": features,
        })
    return rows if include_all else rows[:-1]


def _add_crossable_local_features(samples):
    for sample in samples:
        sample["local_features"].update({
            "candidate_reply_is_short": False,
            "candidate_reply_has_question": False,
            "candidate_reply_is_multi_bubble": False,
            "candidate_same_chat": False,
            "context_multiple_other_speakers": False,
        })


def test_evaluate_uses_target_disjoint_balanced_folds(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)

    result = evaluate(samples_path, labels_path)

    assert result["dataset"]["targets"] == 16
    assert result["dataset"]["labeled_samples"] == 128
    assert all(fold == {"private": 2, "group": 2} for fold in result["fold_target_counts"].values())
    # The baseline is production top-1.  The learner is allowed to fully
    # re-rank that production top-8 by its out-of-fold probability.
    assert result["ranking_metrics"]["top1_all"]["bootstrap"] == 0.0
    assert result["ranking_metrics"]["top1_all"]["learned"] == 1.0
    assert result["ranking_metrics"]["judge_detection_top1_all"]["bootstrap"] == 1.0
    assert result["ranking_metrics"]["judge_detection_top1_all"]["learned"] == 0.0
    assert result["ranking_metrics"]["mrr"]["delta"] == pytest.approx(0.875)
    assert result["dataset"]["ranking_feature_names"] == [
        "candidate_is_production_rank_2_or_3",
        "candidate_is_production_rank_4_to_8",
        "candidate_is_production_top1",
        "candidate_matches_action",
    ]
    assert result["dataset"]["excluded_target_constant_feature_names"] == [
        "context_is_private"
    ]
    assert result["dataset"]["pairwise_preference_count"] == 112
    assert result["top1_diagnostics"]["private-00"] == {
        "relationship": "private",
        "production": {"candidate_rank": 1, "label": 0},
        "pairwise": {"candidate_rank": 8, "label": 1},
    }


def test_evaluate_train_dev_fits_train_and_scores_only_disjoint_dev_targets(tmp_path):
    train_samples, train_labels = _dataset(private_targets=8, group_targets=0)
    dev_samples, dev_labels = _dataset(private_targets=0, group_targets=8)
    for sample in dev_samples:
        target_id = sample["target"]["case_id"]
        sample["target"]["case_id"] = "dev-" + target_id
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path, train_labels_path, dev_samples_path, dev_labels_path,
    )

    assert result["train_dataset"]["targets"] == 8
    assert result["dev_dataset"]["targets"] == 8
    assert result["ranking_metrics"]["judge_detection_top1_all"] == {
        "bootstrap": 1.0,
        "learned": 0.0,
        "delta": -1.0,
        "ci95_low": -1.0,
        "ci95_high": -1.0,
    }
    assert result["ranking_metrics_by_relationship"]["group"]["top1_all"]["delta"] == 1.0


def test_group_only_train_dev_keeps_private_production_order(tmp_path):
    train_samples, train_labels = _dataset(private_targets=8, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=8, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        learned_relationship="group",
    )

    assert result["protocol"]["private_ranking_when_group_only"] == (
        "production retriever returned candidate rank"
    )
    assert result["train_dataset"]["fit_relationship_targets"] == {"group": 8}
    assert result["ranking_metrics_by_relationship"]["group"]["judge_detection_top1_all"]["delta"] == -1.0
    assert result["ranking_metrics_by_relationship"]["private"]["judge_detection_top1_all"]["delta"] == 0.0
    assert all(
        diagnostic["learned"]["candidate_rank"] == 1
        for target_id, diagnostic in result["top1_diagnostics"].items()
        if diagnostic["relationship"] == "private"
    )


def test_private_only_train_dev_keeps_group_production_order(tmp_path):
    train_samples, train_labels = _dataset(private_targets=8, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=8, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        learned_relationship="private",
    )

    assert result["protocol"]["group_ranking_when_private_only"] == (
        "production retriever returned candidate rank"
    )
    assert result["train_dataset"]["fit_relationship_targets"] == {"private": 8}
    assert result["ranking_metrics_by_relationship"]["private"]["judge_detection_top1_all"]["delta"] == -1.0
    assert result["ranking_metrics_by_relationship"]["group"]["judge_detection_top1_all"]["delta"] == 0.0
    assert all(
        diagnostic["learned"]["candidate_rank"] == 1
        for target_id, diagnostic in result["top1_diagnostics"].items()
        if diagnostic["relationship"] == "group"
    )


def test_group_only_train_dev_xgboost_keeps_private_production_order(tmp_path):
    pytest.importorskip("xgboost")
    train_samples, train_labels = _dataset(private_targets=8, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=8, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        learned_relationship="group",
        model_kind="xgboost",
    )

    assert result["protocol"]["model"].startswith("XGBRanker(")
    assert result["coefficients"] is None
    assert result["feature_importances"]
    assert all(
        diagnostic["learned"]["candidate_rank"] == 1
        for diagnostic in result["top1_diagnostics"].values()
        if diagnostic["relationship"] == "private"
    )


def test_train_dev_can_add_fully_labeled_incomplete_target_to_pairwise_fit_only(tmp_path):
    train_samples, train_labels = _dataset(private_targets=0, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=0, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    for rank in range(1, 6):
        sample_id = f"incomplete-group-{rank}"
        train_samples.append({
            "sample_id": sample_id,
            "target": {"case_id": "incomplete-group", "relationship": "group"},
            "candidate": {"id": sample_id},
            "candidate_rank": rank,
            "local_features": {
                "context_is_private": False,
                "candidate_matches_action": rank <= 3,
            },
        })
        train_labels.append({"sample_id": sample_id, "status": "ok", "label": int(rank <= 3)})
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        learned_relationship="group",
        include_incomplete_train_pairwise_targets=True,
    )

    assert result["train_dataset"]["fit_targets"] == 9
    assert result["train_dataset"]["pairwise_preference_count"] == 62
    assert result["dev_dataset"]["targets"] == 8
    assert result["train_dataset"]["included_incomplete_pairwise_training_targets"] == [{
        "target_id": "incomplete-group",
        "relationship": "group",
        "source_candidate_count": 5,
        "successful_label_count": 5,
        "positive_count": 3,
        "negative_count": 2,
        "pairwise_preference_count": 6,
        "reason": "complete_labels_but_incomplete_production_top8_pairwise_training_only",
    }]


def test_group_only_trace_features_do_not_require_private_trace(tmp_path):
    train_samples, train_labels = _dataset(private_targets=8, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=8, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    _add_production_trace([
        sample for sample in train_samples if sample["target"]["relationship"] == "group"
    ])
    _add_production_trace([
        sample for sample in dev_samples if sample["target"]["relationship"] == "group"
    ])
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        use_production_trace_features=True,
        learned_relationship="group",
    )

    assert "candidate_production_lexical_score_best" in result["dev_dataset"]["ranking_feature_names"]
    assert all(
        diagnostic["learned"]["candidate_rank"] == 1
        for diagnostic in result["top1_diagnostics"].values()
        if diagnostic["relationship"] == "private"
    )


def test_group_only_train_dev_joins_action_and_trace_features_without_touching_private(tmp_path):
    train_samples, train_labels = _dataset(private_targets=8, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=8, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    _add_production_trace([
        sample for sample in train_samples if sample["target"]["relationship"] == "group"
    ])
    _add_production_trace([
        sample for sample in dev_samples if sample["target"]["relationship"] == "group"
    ])
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    train_target_features_path = tmp_path / "train_targets.jsonl"
    train_candidate_features_path = tmp_path / "train_candidates.jsonl"
    dev_target_features_path = tmp_path / "dev_targets.jsonl"
    dev_candidate_features_path = tmp_path / "dev_candidates.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    def group_target_rows(samples):
        return [
            {
                "target_id": sample["target"]["case_id"],
                "status": "ok",
                "features": {
                    "needs_direct_response": True,
                    "needs_clarification": False,
                    "needs_light_reaction": False,
                    "needs_coordination": False,
                    "requires_factual_boundary": False,
                    "has_explicit_addressee": False,
                    "is_natural_closure_or_confirmation": False,
                    "still_requires_interaction_progress": False,
                },
            }
            for sample in samples
            if sample["target"]["relationship"] == "group" and sample["candidate_rank"] == 1
        ]

    _write_jsonl(train_target_features_path, group_target_rows(train_samples))
    _write_jsonl(
        train_candidate_features_path,
        _candidate_action_rows([
            sample for sample in train_samples if sample["target"]["relationship"] == "group"
        ]),
    )
    _write_jsonl(dev_target_features_path, group_target_rows(dev_samples))
    _write_jsonl(
        dev_candidate_features_path,
        _candidate_action_rows([
            sample for sample in dev_samples if sample["target"]["relationship"] == "group"
        ]),
    )

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        learned_relationship="group",
        replacement_policy="production_top3_only",
        use_production_trace_features=True,
        train_target_features_path=train_target_features_path,
        train_candidate_features_path=train_candidate_features_path,
        dev_target_features_path=dev_target_features_path,
        dev_candidate_features_path=dev_candidate_features_path,
    )

    assert "target_needs_direct_response__candidate_direct_request_answered" in (
        result["dev_dataset"]["ranking_feature_names"]
    )
    assert "candidate_production_lexical_score_best" in result["dev_dataset"]["ranking_feature_names"]
    assert result["protocol"]["candidate_action_features"] is True
    assert result["sources"]["train_target_features"]["path"] == str(train_target_features_path)
    assert result["sources"]["dev_candidate_features"]["path"] == str(dev_candidate_features_path)
    assert all(
        diagnostic["learned"]["candidate_rank"] == 1
        for diagnostic in result["top1_diagnostics"].values()
        if diagnostic["relationship"] == "private"
    )


def test_train_dev_top3_policy_blocks_rank4_to_8_replacements(tmp_path):
    train_samples, train_labels = _dataset(private_targets=0, group_targets=8)
    dev_samples, dev_labels = _dataset(private_targets=0, group_targets=8)
    for sample in dev_samples:
        sample["target"]["case_id"] = "dev-" + sample["target"]["case_id"]
        sample["sample_id"] = "dev-" + sample["sample_id"]
    for label in dev_labels:
        label["sample_id"] = "dev-" + label["sample_id"]
    train_samples_path = tmp_path / "train_samples.jsonl"
    train_labels_path = tmp_path / "train_labels.jsonl"
    dev_samples_path = tmp_path / "dev_samples.jsonl"
    dev_labels_path = tmp_path / "dev_labels.jsonl"
    _write_jsonl(train_samples_path, train_samples)
    _write_jsonl(train_labels_path, train_labels)
    _write_jsonl(dev_samples_path, dev_samples)
    _write_jsonl(dev_labels_path, dev_labels)

    result = evaluate_train_dev(
        train_samples_path,
        train_labels_path,
        dev_samples_path,
        dev_labels_path,
        replacement_policy="production_top3_only",
    )

    assert result["protocol"]["replacement_policy"] == "production_top3_only"
    assert result["ranking_metrics"]["judge_detection_top1_all"]["delta"] == 0.0
    assert all(
        diagnostic["learned"]["candidate_rank"] == 1
        for diagnostic in result["top1_diagnostics"].values()
    )


def test_top_rank_limited_scores_only_selects_from_production_prefix():
    rows = [
        LabeledSample(
            sample_id=f"sample-{rank}",
            target_id="target",
            candidate_id=f"candidate-{rank}",
            relationship="group",
            candidate_relationship="group",
            candidate_rank=rank,
            features={},
            label=0,
        )
        for rank in range(1, 9)
    ]
    scores, diagnostics = _top_rank_limited_scores(
        {"target": rows},
        {row.sample_id: float(row.candidate_rank) for row in rows},
        max_replacement_rank=3,
    )

    assert max(rows, key=lambda row: scores[row.sample_id]).candidate_rank == 3
    assert diagnostics["target"]["selected_candidate_rank"] == 3


def test_production_trace_features_are_target_relative_booleans(tmp_path):
    samples, labels = _dataset(private_targets=1, group_targets=1)
    _add_production_trace(samples)
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)

    rows = _add_production_trace_features(load_labeled_samples(samples_path, labels_path))
    rank1 = next(row for row in rows if row.target_id == "private-00" and row.candidate_rank == 1)
    rank3 = next(row for row in rows if row.target_id == "private-00" and row.candidate_rank == 3)
    rank4 = next(row for row in rows if row.target_id == "private-00" and row.candidate_rank == 4)
    rank5 = next(row for row in rows if row.target_id == "private-00" and row.candidate_rank == 5)
    rank8 = next(row for row in rows if row.target_id == "private-00" and row.candidate_rank == 8)

    assert rank1.features["candidate_production_lexical_score_best"]
    assert rank3.features["candidate_production_lexical_score_top3"]
    assert not rank4.features["candidate_production_lexical_score_top3"]
    assert rank1.features["candidate_production_pre_diversity_top1"]
    assert rank5.features["candidate_production_diversity_displaced"]
    assert rank8.features["candidate_production_response_move_exact_match"]


def test_evaluate_ignores_legacy_handwritten_retrieval_scores(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    for sample in samples:
        sample["bootstrap_retrieval_score"] = 1
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)

    result = evaluate(samples_path, labels_path)

    assert result["ranking_metrics"]["top1_all"]["bootstrap"] == 0.0
    assert result["ranking_metrics"]["top1_all"]["learned"] == 1.0
    assert result["ranking_metrics"]["judge_detection_top1_all"]["learned"] == 0.0


def test_evaluate_requires_bounded_minimum_target_coverage(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=7)
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)

    with pytest.raises(InsufficientDataError, match="missing private=0, group=1"):
        evaluate(samples_path, labels_path)


def test_evaluate_combines_disjoint_label_indexes_and_rejects_overlap(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    samples_path = tmp_path / "samples.jsonl"
    first_labels_path = tmp_path / "first.jsonl"
    second_labels_path = tmp_path / "second.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(first_labels_path, labels[:64])
    _write_jsonl(second_labels_path, labels[64:])

    result = evaluate(samples_path, (first_labels_path, second_labels_path))
    assert result["dataset"]["labeled_samples"] == 128

    _write_jsonl(second_labels_path, labels[63:])
    with pytest.raises(ValueError, match="multiple label indexes"):
        evaluate(samples_path, (first_labels_path, second_labels_path))


def test_load_labeled_samples_keeps_success_over_later_transient_error(tmp_path):
    samples, labels = _dataset(private_targets=1, group_targets=0)
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels + [{
        "sample_id": "private-00-candidate-0",
        "status": "error",
        "error_type": "FeatureExtractionError",
    }])

    rows = load_labeled_samples(samples_path, labels_path)

    assert len(rows) == 8
    assert next(row for row in rows if row.candidate_rank == 1).label == 0


def test_load_labeled_samples_adds_private_question_cross_locally(tmp_path):
    samples, labels = _dataset(private_targets=1, group_targets=1)
    for sample in samples:
        sample["local_features"]["candidate_reply_has_question"] = (
            sample["candidate_rank"] == 2
        )
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)

    rows = load_labeled_samples(samples_path, labels_path)

    private_question = next(
        row for row in rows
        if row.target_id == "private-00" and row.candidate_rank == 2
    )
    group_question = next(
        row for row in rows
        if row.target_id == "group-00" and row.candidate_rank == 2
    )
    assert private_question.features["context_is_private__candidate_reply_has_question"]
    assert not group_question.features["context_is_private__candidate_reply_has_question"]


def test_evaluate_uses_target_facts_only_as_candidate_crosses(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    for sample in samples:
        candidate_index = sample["candidate_rank"] - 1
        sample["local_features"].update({
            "candidate_reply_is_short": candidate_index == 0,
            "candidate_reply_has_question": False,
            "candidate_reply_is_multi_bubble": False,
            "candidate_same_chat": False,
            "context_multiple_other_speakers": False,
        })
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    target_features_path = tmp_path / "target_features.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_features_path, [
        {
            "target_id": f"{relationship}-{target_index:02d}",
            "status": "ok",
            "features": {
                "needs_direct_response": relationship == "private",
                "needs_clarification": False,
                "needs_light_reaction": False,
                "needs_coordination": False,
                    "requires_factual_boundary": False,
                    "has_explicit_addressee": False,
                    "is_natural_closure_or_confirmation": False,
                    "still_requires_interaction_progress": False,
            },
        }
        for relationship in ("private", "group")
        for target_index in range(8)
    ])

    result = evaluate(samples_path, labels_path, target_features_path=target_features_path)

    assert result["dataset"]["target_feature_cross_count"] == 5
    assert "needs_direct_response" not in result["dataset"]["ranking_feature_names"]
    assert "target_needs_direct_response__not_reply_is_short" in result["dataset"]["ranking_feature_names"]


def test_evaluate_explicitly_excludes_targets_without_a_full_top8(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    incomplete_target = "private-incomplete"
    for candidate_index in range(5):
        sample_id = f"{incomplete_target}-candidate-{candidate_index}"
        samples.append({
            "sample_id": sample_id,
            "target": {"case_id": incomplete_target, "relationship": "private"},
            "candidate": {"id": f"candidate-{candidate_index}"},
            "candidate_rank": candidate_index + 1,
            "local_features": {
                "context_is_private": True,
                "candidate_matches_action": candidate_index == 4,
            },
        })
        labels.append({"sample_id": sample_id, "status": "ok", "label": 0})
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    target_features_path = tmp_path / "target_features.jsonl"
    for sample in samples:
        sample["local_features"].update({
            "candidate_reply_is_short": False,
            "candidate_reply_has_question": False,
            "candidate_reply_is_multi_bubble": False,
            "candidate_same_chat": False,
            "context_multiple_other_speakers": False,
        })
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_features_path, [
        {
            "target_id": f"{relationship}-{target_index:02d}",
            "status": "ok",
            "features": {
                "needs_direct_response": False,
                "needs_clarification": False,
                "needs_light_reaction": False,
                "needs_coordination": False,
                "requires_factual_boundary": False,
                "has_explicit_addressee": False,
                "is_natural_closure_or_confirmation": False,
                "still_requires_interaction_progress": False,
            },
        }
        for relationship in ("private", "group")
        for target_index in range(8)
    ])

    result = evaluate(samples_path, labels_path, target_features_path=target_features_path)

    assert result["dataset"]["targets"] == 16
    assert result["dataset"]["labeled_samples"] == 128
    assert result["dataset"]["excluded_incomplete_targets"] == [{
        "target_id": incomplete_target,
        "relationship": "private",
        "source_candidate_count": 5,
        "successful_label_count": 5,
        "reason": "source_candidate_count_not_8",
    }]


def test_conservative_policy_keeps_production_top1_without_action_match(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)

    result = evaluate(
        samples_path,
        labels_path,
        replacement_policy="conservative_action_match",
    )
    assert result["ranking_metrics"]["top1_all"]["bootstrap"] == 0.0
    assert result["ranking_metrics"]["top1_all"]["learned"] == 0.0
    assert all(
        value["reason"] == "retain_production_top1_no_supported_action_match"
        for value in result["replacement_diagnostics"].values()
    )


def test_top3_conservative_policy_rejects_a_supported_rank4_override():
    target_id = "target-1"
    production = LabeledSample(
        sample_id="target-1-rank-1", target_id=target_id, candidate_id="rank-1",
        relationship="private", candidate_relationship="acquaintance", candidate_rank=1,
        features={}, label=0,
    )
    rank4_match = LabeledSample(
        sample_id="target-1-rank-4", target_id=target_id, candidate_id="rank-4",
        relationship="private", candidate_relationship="acquaintance", candidate_rank=4,
        features={
            "target_needs_direct_response__candidate_direct_request_answered": True,
        }, label=1,
    )
    groups = {target_id: [production, rank4_match]}
    raw_scores = {production.sample_id: 0.0, rank4_match.sample_id: 1.0}

    unrestricted, _ = _conservative_scores(groups, raw_scores)
    top3_only, diagnostics = _conservative_scores(
        groups, raw_scores, max_replacement_rank=3,
    )

    assert unrestricted[rank4_match.sample_id] == 1.0
    assert top3_only[production.sample_id] == 1.0
    assert diagnostics[target_id]["reason"] == "retain_production_top1_no_supported_action_match"


def test_conservative_policy_replaces_only_with_completed_historical_action_match(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    _add_crossable_local_features(samples)
    for sample in samples:
        target_id = sample["target"]["case_id"]
        candidate_rank = sample["candidate_rank"]
        sample["candidate"]["id"] = f"{target_id}-candidate-{candidate_rank}"
        sample["candidate"]["relationship"] = (
            "acquaintance" if sample["target"]["relationship"] == "private" else "group"
        )
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    target_features_path = tmp_path / "target_features.jsonl"
    candidate_features_path = tmp_path / "candidate_features.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_features_path, _target_action_rows())
    _write_jsonl(candidate_features_path, _candidate_action_rows(samples))

    result = evaluate(
        samples_path,
        labels_path,
        target_features_path=target_features_path,
        candidate_features_path=candidate_features_path,
        replacement_policy="conservative_action_match",
    )

    assert result["ranking_metrics"]["judge_detection_top1_all"] == {
        "bootstrap": 1.0,
        "learned": 0.0,
        "delta": -1.0,
        "ci95_low": -1.0,
        "ci95_high": -1.0,
    }
    assert result["dataset"]["candidate_action_compound_count"] == 6
    for diagnostic in result["replacement_diagnostics"].values():
        assert diagnostic["selected_candidate_rank"] == 8
        assert diagnostic["reason"] == "supported_action_match"
        assert diagnostic["action_match_features"] == [
            "target_needs_direct_response__candidate_direct_request_answered"
        ]


def test_light_reaction_replaces_only_when_it_is_the_target_primary_action(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    _add_crossable_local_features(samples)
    for sample in samples:
        target_id = sample["target"]["case_id"]
        candidate_rank = sample["candidate_rank"]
        sample["candidate"]["id"] = f"{target_id}-candidate-{candidate_rank}"
        sample["candidate"]["relationship"] = (
            "acquaintance" if sample["target"]["relationship"] == "private" else "group"
        )
    target_rows = _target_action_rows()
    candidate_rows = _candidate_action_rows(samples)
    for target in target_rows:
        # A light reply may be socially acceptable here, but the real primary
        # action remains answering the direct request.
        target["features"]["needs_light_reaction"] = False
    for candidate in candidate_rows:
        candidate["features"]["candidate_input_needs_direct_response"] = False
        candidate["features"]["candidate_reply_directly_addresses_request"] = False
        if candidate["candidate_id"].endswith("-candidate-8"):
            candidate["features"].update({
                "candidate_input_needs_light_reaction": True,
                "candidate_reply_is_light_reaction": True,
            })
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    target_features_path = tmp_path / "target_features.jsonl"
    candidate_features_path = tmp_path / "candidate_features.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_features_path, target_rows)
    _write_jsonl(candidate_features_path, candidate_rows)

    no_primary_light_reaction = evaluate(
        samples_path,
        labels_path,
        target_features_path=target_features_path,
        candidate_features_path=candidate_features_path,
        replacement_policy="conservative_action_match",
    )
    assert all(
        value["reason"] == "retain_production_top1_no_supported_action_match"
        for value in no_primary_light_reaction["replacement_diagnostics"].values()
    )

    for target in target_rows:
        target["features"]["needs_direct_response"] = False
        target["features"]["needs_light_reaction"] = True
    _write_jsonl(target_features_path, target_rows)
    primary_light_reaction = evaluate(
        samples_path,
        labels_path,
        target_features_path=target_features_path,
        candidate_features_path=candidate_features_path,
        replacement_policy="conservative_action_match",
    )
    assert all(
        value["selected_candidate_rank"] == 8
        and value["action_match_features"] == [
            "target_needs_light_reaction__candidate_light_reaction_used"
        ]
        for value in primary_light_reaction["replacement_diagnostics"].values()
    )


def test_candidate_action_features_fail_closed_when_a_selected_candidate_is_missing(tmp_path):
    samples, labels = _dataset(private_targets=8, group_targets=8)
    _add_crossable_local_features(samples)
    for sample in samples:
        target_id = sample["target"]["case_id"]
        candidate_rank = sample["candidate_rank"]
        sample["candidate"]["id"] = f"{target_id}-candidate-{candidate_rank}"
        sample["candidate"]["relationship"] = (
            "acquaintance" if sample["target"]["relationship"] == "private" else "group"
        )
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    target_features_path = tmp_path / "target_features.jsonl"
    candidate_features_path = tmp_path / "candidate_features.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_features_path, _target_action_rows())
    _write_jsonl(candidate_features_path, _candidate_action_rows(samples, include_all=False))

    with pytest.raises(ValueError, match="candidate feature candidates differ from samples"):
        evaluate(
            samples_path,
            labels_path,
            target_features_path=target_features_path,
            candidate_features_path=candidate_features_path,
            replacement_policy="conservative_action_match",
        )
