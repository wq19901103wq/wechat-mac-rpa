import scripts.prepare_few_shot_ranker_expanded_evaluation as prepare


def test_disabled_relationship_reuses_production_top3_in_order():
    selections = [{
        "target_id": "target-1",
        "relationship": "private",
        "selection_count": 3,
        "production_sample_ids": ["sample-1", "sample-2", "sample-3"],
        "production_candidate_ids": ["candidate-1", "candidate-2", "candidate-3"],
        "production_candidate_ranks": [1, 2, 3],
        "lr_sample_ids": ["sample-3", "sample-1", "sample-4"],
        "lr_candidate_ids": ["candidate-3", "candidate-1", "candidate-4"],
        "lr_candidate_ranks": [3, 1, 4],
        "lr_scores": [0.9, 0.8, 0.7],
        "same_top3": False,
    }]
    scores = {"sample-1": 0.3, "sample-2": 0.2, "sample-3": 0.1}

    result = prepare._apply_relationship_policy(
        selections,
        scores,
        model_kind="lr",
        selection_count=3,
        ranker_enabled=False,
    )

    assert result[0]["lr_sample_ids"] == result[0]["production_sample_ids"]
    assert result[0]["lr_candidate_ids"] == result[0]["production_candidate_ids"]
    assert result[0]["lr_candidate_ranks"] == result[0]["production_candidate_ranks"]
    assert result[0]["lr_scores"] == [0.3, 0.2, 0.1]
    assert result[0]["same_top3"] is True
    assert selections[0]["same_top3"] is False
