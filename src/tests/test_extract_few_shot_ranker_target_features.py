import json

from scripts.extract_few_shot_ranker_target_features import (
    TARGET_FEATURE_NAMES,
    _resumable_rows,
    extract,
    write_output,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extracts_one_target_only_feature_record_per_target(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        {
            "target": {
                "case_id": "private-1",
                "relationship": "private",
                "context_messages": [{"sender": "甲", "text": "这个怎么处理？"}],
            },
            "candidate": {"reply": ["candidate text that must not be sent"]},
        },
        {
            "target": {
                "case_id": "private-1",
                "relationship": "private",
                "context_messages": [{"sender": "甲", "text": "这个怎么处理？"}],
            },
            "candidate": {"reply": ["another candidate text"]},
        },
    ])
    requests = []

    class Client:
        model = "deepseek-v4-flash"

        def chat(self, **kwargs):
            requests.append(kwargs)
            return json.dumps({name: False for name in TARGET_FEATURE_NAMES})

    rows, model = extract(samples_path, client_factory=Client, workers=1)

    assert model == "deepseek-v4-flash"
    assert rows == [{
        "target_id": "private-1",
        "relationship": "private",
        "status": "ok",
        "features": {name: False for name in TARGET_FEATURE_NAMES},
        "attempts": 1,
    }]
    prompt = requests[0]["messages"][1]["content"]
    assert "这个怎么处理？" in prompt
    assert "candidate text" not in prompt
    assert len(requests) == 1

    manifest = write_output(tmp_path / "out", rows, samples_path=samples_path, model=model)
    assert manifest["protocol"]["candidate_llm_calls"] == 0
    assert manifest["counts"]["ok"] == 1
    assert manifest["counts"]["error"] == 0
    assert (tmp_path / "out" / "targets.jsonl").exists()


def test_records_a_failed_target_after_bounded_retries(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [{
        "target": {
            "case_id": "group-1",
            "relationship": "group",
            "context_messages": [{"sender": "甲", "text": "在吗"}],
        },
    }])

    class FailingClient:
        model = "deepseek-v4-flash"

        def chat(self, **kwargs):
            raise TimeoutError("provider timed out")

    rows, _ = extract(samples_path, client_factory=FailingClient, workers=1, max_attempts=1)

    assert rows == [{
        "target_id": "group-1",
        "relationship": "group",
        "status": "error",
        "error": "TimeoutError: provider timed out",
        "attempts": 1,
    }]


def test_resume_retains_valid_successes_and_retries_only_other_targets(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        {
            "target": {
                "case_id": "group-1",
                "relationship": "group",
                "context_messages": [{"sender": "甲", "text": "在吗"}],
            },
        },
        {
            "target": {
                "case_id": "private-1",
                "relationship": "private",
                "context_messages": [{"sender": "乙", "text": "怎么处理？"}],
            },
        },
    ])
    output_dir = tmp_path / "out"
    write_output(output_dir, [
        {
            "target_id": "group-1",
            "relationship": "group",
            "status": "ok",
            "features": {name: False for name in TARGET_FEATURE_NAMES},
            "attempts": 1,
        },
        {
            "target_id": "private-1",
            "relationship": "private",
            "status": "error",
            "error": "TimeoutError: provider timed out",
            "attempts": 3,
        },
    ], samples_path=samples_path, model="deepseek-v4-flash")

    retained, retry_ids = _resumable_rows(output_dir, samples_path)
    requests = []

    class Client:
        model = "deepseek-v4-flash"

        def chat(self, **kwargs):
            requests.append(kwargs)
            return json.dumps({name: False for name in TARGET_FEATURE_NAMES})

    retried, _ = extract(samples_path, client_factory=Client, workers=1, target_ids=retry_ids)

    assert [row["target_id"] for row in retained] == ["group-1"]
    assert [row["target_id"] for row in retried] == ["private-1"]
    assert "怎么处理？" in requests[0]["messages"][1]["content"]
    assert "在吗" not in requests[0]["messages"][1]["content"]


def test_resume_can_limit_target_extraction_to_group_targets(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        {"target": {"case_id": "group-1", "relationship": "group", "context_messages": [{"sender": "甲", "text": "在吗"}]}},
        {"target": {"case_id": "private-1", "relationship": "private", "context_messages": [{"sender": "乙", "text": "怎么处理？"}]}},
    ])

    output_dir = tmp_path / "out"
    write_output(output_dir, [{
        "target_id": "private-1",
        "relationship": "private",
        "status": "ok",
        "features": {name: False for name in TARGET_FEATURE_NAMES},
        "attempts": 1,
    }], samples_path=samples_path, model="deepseek-v4-flash")

    retained, retry_ids = _resumable_rows(
        output_dir, samples_path, target_relationship="group",
    )

    assert retained == []
    assert retry_ids == {"group-1"}
