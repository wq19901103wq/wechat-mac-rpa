import json

import scripts.build_few_shot_ranker_expanded_label_samples as builder


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(sample_id, candidate_id, rank):
    return {
        "sample_id": sample_id,
        "candidate_rank": rank,
        "target": {
            "case_id": "target-1",
            "relationship": "group",
            "chat_id": "chat-1",
            "timestamp": 1,
            "context": ["target-context"],
            "context_messages": [],
        },
        "candidate": {
            "id": candidate_id,
            "relationship": "group",
            "chat_id": "historical-chat",
            "timestamp": 0,
            "context": ["candidate-context"],
            "reply": [candidate_id],
        },
        "generation": {
            "few_shot_count": 1,
            "static_few_shot_count": 0,
            "injected_candidate_id": candidate_id,
        },
    }


def test_build_top3_binds_each_arm_in_frozen_order(tmp_path):
    samples_path = tmp_path / "source.jsonl"
    selections_path = tmp_path / "selections.jsonl"
    output_dir = tmp_path / "output"
    sources = [
        _source("sample-1", "candidate-1", 1),
        _source("sample-2", "candidate-2", 2),
        _source("sample-3", "candidate-3", 3),
        _source("sample-4", "candidate-4", 4),
    ]
    selection = {
        "target_id": "target-1",
        "relationship": "group",
        "selection_count": 3,
        "production_sample_ids": ["sample-1", "sample-2", "sample-3"],
        "production_candidate_ids": ["candidate-1", "candidate-2", "candidate-3"],
        "production_candidate_ranks": [1, 2, 3],
        "lr_sample_ids": ["sample-3", "sample-1", "sample-4"],
        "lr_candidate_ids": ["candidate-3", "candidate-1", "candidate-4"],
        "lr_candidate_ranks": [3, 1, 4],
        "lr_scores": [3.0, 2.0, 1.0],
        "same_top3": False,
    }
    _write_jsonl(samples_path, sources)
    _write_jsonl(selections_path, [selection])

    manifest = builder.build(
        selections_path=selections_path,
        samples_path=samples_path,
        output_dir=output_dir,
        candidate_variant="lr",
    )

    rows = builder._read_jsonl(output_dir / "samples.jsonl")
    assert [row["evaluation_variant"] for row in rows] == ["production", "lr"]
    assert [candidate["id"] for candidate in rows[0]["candidates"]] == [
        "candidate-1", "candidate-2", "candidate-3"
    ]
    assert [candidate["id"] for candidate in rows[1]["candidates"]] == [
        "candidate-3", "candidate-1", "candidate-4"
    ]
    assert rows[0]["generation"]["injected_candidate_ids"] == [
        "candidate-1", "candidate-2", "candidate-3"
    ]
    assert manifest["protocol"]["bound_few_shot_count"] == 3
    assert manifest["counts"]["samples"] == 2


def test_build_top1_keeps_legacy_row_shape_and_id(tmp_path):
    samples_path = tmp_path / "source.jsonl"
    selections_path = tmp_path / "selections.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(samples_path, [_source("sample-1", "candidate-1", 1)])
    _write_jsonl(selections_path, [{
        "target_id": "target-1",
        "relationship": "group",
        "production_sample_id": "sample-1",
        "production_candidate_id": "candidate-1",
        "production_candidate_rank": 1,
        "lr_sample_id": "sample-1",
        "lr_candidate_id": "candidate-1",
        "lr_candidate_rank": 1,
        "lr_score": 1.0,
        "same_top1": True,
    }])

    manifest = builder.build(
        selections_path=selections_path,
        samples_path=samples_path,
        output_dir=output_dir,
        candidate_variant="lr",
    )

    row = builder._read_jsonl(output_dir / "samples.jsonl")[0]
    assert row["sample_id"] == "production:sample-1"
    assert "candidate" in row and "candidates" not in row
    assert manifest["protocol"]["single_few_shot_only"] is True
