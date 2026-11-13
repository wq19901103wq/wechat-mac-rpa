import json

import pytest

from scripts.build_human_ai_judge_benchmark_v2 import (
    NEW_TARGETS,
    allocate_chat_splits,
    candidate_record,
    surface_difficulty,
    validate_chat_isolation,
    validate_manifest,
)
from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_human_ai_judge import _json_sha256


def _candidate(chat_id: str, timestamp: int, relation: str = "private") -> Candidate:
    return Candidate(
        relation=relation,
        chat_id=chat_id,
        context=[f"上下文{timestamp}"],
        replies=[f"回复{timestamp}"],
        score=1,
        timestamp=timestamp,
    )


def test_allocate_chat_splits_never_crosses_chats_and_uses_second_cases_later():
    candidates = []
    for index in range(30):
        relation = "group" if index % 3 == 0 else "private"
        candidates.extend([
            _candidate(f"chat-{index}", index * 10 + 1, relation),
            _candidate(f"chat-{index}", index * 10 + 2, relation),
        ])

    queues = allocate_chat_splits(candidates, excluded_chat_ids={"chat-0"})
    records = {split: [candidate_record(row) for row in rows] for split, rows in queues.items()}
    validate_chat_isolation(records, {"existing-dev-chat"})

    split_by_chat = {}
    for split, rows in records.items():
        first_duplicate_index = next(
            (i for i, row in enumerate(rows) if sum(r["chat_id"] == row["chat_id"] for r in rows[:i]) > 0),
            len(rows),
        )
        assert len({row["chat_id"] for row in rows[:first_duplicate_index]}) == first_duplicate_index
        for row in rows:
            assert split_by_chat.setdefault(row["chat_id"], split) == split
    assert "chat-0" not in split_by_chat


def test_validate_chat_isolation_rejects_overlap():
    row = candidate_record(_candidate("same-chat", 1))
    with pytest.raises(ValueError, match="chat isolation failed"):
        validate_chat_isolation({"dev": [row], "validation": [row], "final": []}, set())


def test_validate_manifest_rejects_hash_or_c0_tampering():
    manifest = {
        "sources": {"c0_generator_sha256": "wrong"},
        "existing_dev": {"chat_ids": []},
        "queues": {"dev": [], "validation": [], "final": []},
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    with pytest.raises(ValueError, match="protected C0"):
        validate_manifest(manifest)

    manifest["sources"]["c0_generator_sha256"] = (
        "625b47803fff55a2e18914c5e734fcb2f01ae7c16e5180fdebdc817badd2656b"
    )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_manifest(manifest)


def test_surface_difficulty_is_auditable_and_answer_independent():
    hard = {"human_reply_original": ["哈哈确实"], "ai_reply_original": ["哈哈也是"]}
    ordinary = {"human_reply_original": ["行"], "ai_reply_original": ["我理解你的意思，这件事情可以慢慢讨论"]}
    assert surface_difficulty(hard) == "hard_surface"
    assert surface_difficulty(ordinary) == "ordinary"


def test_protocol_new_targets_sum_to_150():
    assert NEW_TARGETS == {"dev": 20, "validation": 30, "final": 100}
    assert sum(NEW_TARGETS.values()) == 150
