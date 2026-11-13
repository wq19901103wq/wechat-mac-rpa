import json

import pytest

from scripts.build_human_ai_judge_benchmark_v2 import candidate_record
from scripts.build_human_ai_judge_benchmark_v3 import (
    TARGETS,
    _reference_without_case_leaks,
    build_relation_queues,
    merge_checkpoint_data,
    relation_shard_rows,
    selected_cases,
    should_generate_case,
    validate_manifest,
)
from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_human_ai_judge import _json_sha256, assert_reference_isolated


def _candidate(chat_id: str, index: int, relation: str) -> Candidate:
    return Candidate(
        relation=relation,
        chat_id=chat_id,
        context=[f"{chat_id}上下文{index}"],
        replies=[f"{chat_id}回复{index}"],
        score=1,
        timestamp=index + 1,
    )


def test_balanced_queues_do_not_allow_one_relation_to_fill_the_other():
    candidates = [_candidate(f"private-{i}", i, "private") for i in range(160)]
    candidates += [_candidate(f"group-{i}", i, "group") for i in range(79)]
    with pytest.raises(ValueError, match="group has 79 eligible candidates; need exactly 80"):
        build_relation_queues(candidates)


def test_balanced_queues_put_distinct_chats_before_second_case():
    candidates = []
    for relation in ("private", "group"):
        for chat_index in range(80):
            candidates.extend([
                _candidate(f"{relation}-{chat_index}", chat_index * 2, relation),
                _candidate(f"{relation}-{chat_index}", chat_index * 2 + 1, relation),
            ])
    queues = build_relation_queues(candidates)
    assert {relation: len(queues[relation]) for relation in TARGETS} == {
        "private": 160,
        "group": 160,
    }
    for relation in TARGETS:
        assert len({row.chat_id for row in queues[relation][:80]}) == 80


def test_selected_cases_freezes_exactly_80_private_and_80_group():
    candidates = {
        relation: [_candidate(f"{relation}-{i}", i, relation) for i in range(90)]
        for relation in TARGETS
    }
    manifest = {
        "queues": {
            relation: [candidate_record(row) for row in rows]
            for relation, rows in candidates.items()
        }
    }
    entries = {}
    for relation, rows in candidates.items():
        for row in rows:
            case_id = candidate_record(row)["case_id"]
            entries[case_id] = {
                "relationship": relation,
                "status": "valid",
                "replay_case": {
                    "case_id": case_id,
                    "source_chat_id": row.chat_id,
                    "relationship": relation,
                },
            }
    selected = selected_cases(manifest, {"entries": entries})
    assert len(selected) == 160
    assert sum(row["relationship"] == "private" for row in selected) == 80
    assert sum(row["relationship"] == "group" for row in selected) == 80
    assert len({row["case_id"] for row in selected}) == 160


def test_manifest_rejects_wrong_quota_and_duplicate_case(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_human_ai_judge_benchmark_v3.current_c0_sha256", lambda: "c0"
    )
    monkeypatch.setattr(
        "scripts.build_human_ai_judge_benchmark_v3.current_c0_baseline_id", lambda: "baseline"
    )
    rows = {
        relation: [candidate_record(_candidate(f"{relation}-{i}", i, relation)) for i in range(80)]
        for relation in TARGETS
    }
    manifest = {
        "protocol": {"targets": TARGETS, "force_reply": True},
        "sources": {"c0_generator_sha256": "c0", "current_c0_baseline_id": "baseline"},
        "queues": rows,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    validate_manifest(manifest)

    tampered = json.loads(json.dumps(manifest))
    tampered["queues"]["group"][0] = tampered["queues"]["private"][0]
    tampered["manifest_sha256"] = _json_sha256({
        key: value for key, value in tampered.items() if key != "manifest_sha256"
    })
    with pytest.raises(ValueError, match="different relationship|more than once"):
        validate_manifest(tampered)


def test_manifest_requires_forced_c0_replies(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_human_ai_judge_benchmark_v3.current_c0_sha256", lambda: "c0"
    )
    monkeypatch.setattr(
        "scripts.build_human_ai_judge_benchmark_v3.current_c0_baseline_id", lambda: "baseline"
    )
    rows = {
        relation: [candidate_record(_candidate(f"{relation}-{i}", i, relation)) for i in range(80)]
        for relation in TARGETS
    }
    manifest = {
        "protocol": {"targets": TARGETS},
        "sources": {"c0_generator_sha256": "c0", "current_c0_baseline_id": "baseline"},
        "queues": rows,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    with pytest.raises(ValueError, match="forced C0 replies"):
        validate_manifest(manifest)


def test_reference_leak_check_rejects_target_reply_and_context_sequence():
    case = {
        "case_id": "case-one",
        "context_original": ["老王：发财了"],
        "option_A": ["卧槽 你自己也发财了"],
        "option_B": ["恭喜发财"],
    }
    reference = {
        "examples": [{
            "context": ["老王：发财了"],
            "human_reply": ["卧槽 你自己也发财了"],
        }]
    }
    with pytest.raises(ValueError, match="reference isolation failed"):
        assert_reference_isolated(reference, [case])


def test_reference_filter_removes_all_matching_examples_and_rehashes():
    cases = [
        {
            "case_id": "private-case",
            "relationship": "private",
            "context_original": ["老王：发财了"],
            "option_A": ["卧槽 你自己也发财了"],
            "option_B": ["恭喜发财"],
        },
        {
            "case_id": "group-case",
            "relationship": "group",
            "context_original": ["小周：今天吃什么"],
            "option_A": ["火锅走起朋友们"],
            "option_B": ["都可以"],
        },
    ]
    reference = {
        "examples": [{"id": "example-1", "human_reply": ["卧槽 你自己也发财了"]}],
        "scenario_few_shots": [
            {
                "id": "scenario-1",
                "relationships": ["group"],
                "human_reply": ["火锅走起朋友们"],
            }
        ],
        "reference_sha256": "old-sha",
    }

    filtered, removed = _reference_without_case_leaks(reference, cases)

    assert filtered["examples"] == []
    assert filtered["scenario_few_shots"] == []
    assert {(row["section"], row["id"]) for row in removed} == {
        ("examples", "example-1"),
        ("scenario_few_shots", "scenario-1"),
    }
    assert filtered["reference_sha256"] == _json_sha256({
        key: value for key, value in filtered.items() if key != "reference_sha256"
    })
    assert_reference_isolated(filtered, cases)


@pytest.mark.parametrize(
    ("entry", "retry_failed", "expected"),
    [
        (None, False, True),
        ({"status": "valid"}, True, False),
        ({"status": "silent"}, True, False),
        ({"status": "generation_failed"}, False, False),
        ({"status": "generation_failed"}, True, True),
        ({"status": "error"}, True, True),
    ],
)
def test_should_generate_case_only_retries_explicit_failures(entry, retry_failed, expected):
    assert should_generate_case(entry, retry_failed=retry_failed) is expected


def test_merge_checkpoint_data_requires_same_run_and_rejects_conflicts():
    target = {
        "manifest_sha256": "manifest",
        "current_c0_baseline_id": "baseline",
        "c0_generator_sha256": "c0",
        "entries": {"private": {"status": "valid"}},
    }
    source = {
        "manifest_sha256": "manifest",
        "current_c0_baseline_id": "baseline",
        "c0_generator_sha256": "c0",
        "entries": {"group": {"status": "valid"}},
    }
    assert set(merge_checkpoint_data(target, source)["entries"]) == {"private", "group"}

    source["entries"] = {"private": {"status": "error"}}
    with pytest.raises(ValueError, match="entry conflict"):
        merge_checkpoint_data(target, source)


def test_relation_shards_are_disjoint_and_preserve_the_queue():
    rows = [{"case_id": str(index)} for index in range(17)]
    shards = [relation_shard_rows(rows, index, 4) for index in range(4)]
    assert {row["case_id"] for shard in shards for row in shard} == {
        row["case_id"] for row in rows
    }
    assert sum(len(shard) for shard in shards) == len(rows)
    assert relation_shard_rows(rows, 0, 4) == rows[::4]
