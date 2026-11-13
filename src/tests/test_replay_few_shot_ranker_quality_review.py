import hashlib
import json

import scripts.replay_few_shot_ranker_quality_review as review


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_quality_replay_saves_exact_paired_outputs_and_resumes(tmp_path, monkeypatch):
    selections_dir = tmp_path / "plan"
    selections_dir.mkdir()
    selections_path = selections_dir / "selections.jsonl"
    _write_jsonl(selections_path, [{
        "target_id": "target-1",
        "relationship": "group",
        "same_top1": False,
        "production_sample_id": "sample-production",
        "lr_sample_id": "sample-lr",
    }])
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps({
        "candidate_variant": "lr",
        "current_defaults": {
            "c0_baseline_id": "c0-current",
            "judge_baseline_id": "judge-current",
        },
        "sources": {
            "selections": {
                "sha256": hashlib.sha256(selections_path.read_bytes()).hexdigest(),
            },
        },
        "paired_outcomes": {"regressed_target_ids": ["target-1"]},
    }))
    target = {
        "case_id": "target-1", "chat_id": "chat-1", "relationship": "group",
        "timestamp": 100, "context": ["问题"],
        "context_messages": [{"sender": "甲", "text": "问题"}],
    }
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        {
            "sample_id": "sample-production", "target": target,
            "candidate": {"id": "production", "relationship": "group", "context": ["旧问题"], "reply": ["旧回复"]},
        },
        {
            "sample_id": "sample-lr", "target": target,
            "candidate": {"id": "lr", "relationship": "group", "context": ["另一个问题"], "reply": ["另一个回复"]},
        },
    ])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "splits": {"validation": [{"case_id": "target-1", "chat_id": "chat-1", "replies": ["真人"]}]},
    }))
    monkeypatch.setattr(review, "current_c0_baseline_id", lambda: "c0-current")
    monkeypatch.setattr(review, "current_judge_baseline_id", lambda: "judge-current")
    monkeypatch.setattr(
        review, "load_current_judge",
        lambda **_kwargs: {"required_current_c0_baseline_id": "c0-current"},
    )
    calls = []

    def run(_target, _payload, *, persona_few_shot_retriever, **_kwargs):
        candidate = persona_few_shot_retriever.retrieve(limit=1)[0]
        calls.append(candidate["id"])
        return {
            "few_shot_ids": [candidate["id"]],
            "replies": [f"生成-{candidate['id']}"],
            "generation_failed": False,
            "duration_seconds": 0.1,
        }

    output_dir = tmp_path / "out"
    result = review.replay(
        evaluation_path=evaluation_path,
        selections_path=selections_path,
        samples_path=samples_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        payloads={"chat-1": {"messages": []}},
        generator=object(),
        c0_runner=run,
    )

    assert calls == ["production", "lr"]
    assert result["summary"]["complete_pairs"] == 1
    assert result["cases"][0]["production"]["generated_reply"] == ["生成-production"]
    assert result["cases"][0]["lr"]["generated_reply"] == ["生成-lr"]
    assert (output_dir / "generation.jsonl").stat().st_mode & 0o777 == 0o600

    resumed = review.replay(
        evaluation_path=evaluation_path,
        selections_path=selections_path,
        samples_path=samples_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        payloads={"chat-1": {"messages": []}},
        generator=object(),
        c0_runner=run,
        resume=True,
    )
    assert calls == ["production", "lr"]
    assert resumed["summary"]["complete_pairs"] == 1


def test_quality_replay_binds_top3_in_frozen_order_from_cases_manifest(tmp_path, monkeypatch):
    selections_path = tmp_path / "selections.jsonl"
    production_ids = ["p1", "p2", "p3"]
    lr_ids = ["p3", "p1", "p4"]
    _write_jsonl(selections_path, [{
        "target_id": "target-1", "relationship": "group", "selection_count": 3,
        "same_top3": False,
        "production_sample_ids": [f"s-{value}" for value in production_ids],
        "production_candidate_ids": production_ids,
        "lr_sample_ids": [f"s-{value}" for value in lr_ids],
        "lr_candidate_ids": lr_ids,
    }])
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps({
        "candidate_variant": "lr",
        "current_defaults": {
            "c0_baseline_id": "c0-current", "judge_baseline_id": "judge-current",
            "few_shot_count": 3,
        },
        "sources": {"selections": {"sha256": hashlib.sha256(selections_path.read_bytes()).hexdigest()}},
        "paired_outcomes": {"improved_target_ids": ["target-1"]},
    }), encoding="utf-8")
    target = {
        "case_id": "target-1", "chat_id": "chat-1", "relationship": "group",
        "timestamp": 100, "context": ["问题"],
    }
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        {
            "sample_id": f"s-{candidate_id}", "target": target,
            "candidate": {
                "id": candidate_id, "relationship": "group",
                "context": [f"问题-{candidate_id}"], "reply": [f"回复-{candidate_id}"],
            },
        }
        for candidate_id in ("p1", "p2", "p3", "p4")
    ])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "cases": [{
            "id": "target-1", "chat_id": "chat-1", "relationship": "group",
            "reply": ["真人"],
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(review, "current_c0_baseline_id", lambda: "c0-current")
    monkeypatch.setattr(review, "current_judge_baseline_id", lambda: "judge-current")
    monkeypatch.setattr(
        review, "load_current_judge",
        lambda **_kwargs: {"required_current_c0_baseline_id": "c0-current"},
    )
    calls = []

    def run(_target, _payload, *, persona_few_shot_retriever, persona_few_shot_count, **_kwargs):
        rows = persona_few_shot_retriever.retrieve(limit=persona_few_shot_count)
        ids = [row["id"] for row in rows]
        calls.append(ids)
        return {"few_shot_ids": ids, "replies": ["生成"], "generation_failed": False}

    result = review.replay(
        evaluation_path=evaluation_path, selections_path=selections_path,
        samples_path=samples_path, manifest_path=manifest_path,
        output_dir=tmp_path / "out", payloads={"chat-1": {}},
        generator=object(), c0_runner=run, scope="improved",
    )

    assert calls == [production_ids, lr_ids]
    assert result["cases"][0]["production"]["candidate_ids"] == production_ids
    assert result["cases"][0]["lr"]["candidate_ids"] == lr_ids
    assert result["protocol"]["bound_few_shot_count_per_arm"] == 3
