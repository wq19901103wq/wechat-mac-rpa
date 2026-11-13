import json

from scripts.extract_few_shot_ranker_candidate_features import (
    CANDIDATE_FEATURE_NAMES,
    _resumable_rows,
    extract,
    write_output,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sample(candidate_id, context, reply):
    return {
        "target": {"case_id": "target", "relationship": "private"},
        "candidate": {
            "id": candidate_id,
            "relationship": "acquaintance",
            "context": context,
            "reply": reply,
        },
    }


def test_extracts_one_candidate_only_feature_record_per_unique_candidate(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        _sample("candidate-1", ["这个怎么处理？"], ["我先确认一下"]),
        _sample("candidate-1", ["这个怎么处理？"], ["我先确认一下"]),
    ])
    requests = []

    class Client:
        model = "deepseek-v4-flash"

        def chat(self, **kwargs):
            requests.append(kwargs)
            return json.dumps({name: False for name in CANDIDATE_FEATURE_NAMES})

    rows, model = extract(samples_path, client_factory=Client, workers=1)

    assert model == "deepseek-v4-flash"
    assert rows == [{
        "candidate_id": "candidate-1",
        "relationship": "acquaintance",
        "status": "ok",
        "features": {name: False for name in CANDIDATE_FEATURE_NAMES},
        "attempts": 1,
    }]
    prompt = requests[0]["messages"][1]["content"]
    assert "这个怎么处理？" in prompt
    assert "我先确认一下" in prompt
    assert "target" not in prompt
    assert len(requests) == 1
    assert requests[0]["timeout"] == 20

    manifest = write_output(tmp_path / "out", rows, samples_path=samples_path, model=model)
    assert manifest["protocol"]["target_llm_calls"] == 0
    assert manifest["protocol"]["candidate_llm_calls"] == 1
    assert (tmp_path / "out" / "candidates.jsonl").exists()


def test_resume_retains_valid_successes_and_retries_only_other_candidates(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [
        _sample("candidate-1", ["在吗"], ["在"]),
        _sample("candidate-2", ["怎么处理？"], ["我确认下"]),
    ])
    output_dir = tmp_path / "out"
    write_output(output_dir, [
        {
            "candidate_id": "candidate-1",
            "relationship": "acquaintance",
            "status": "ok",
            "features": {name: False for name in CANDIDATE_FEATURE_NAMES},
            "attempts": 1,
        },
        {
            "candidate_id": "candidate-2",
            "relationship": "acquaintance",
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
            return json.dumps({name: False for name in CANDIDATE_FEATURE_NAMES})

    retried, _ = extract(samples_path, client_factory=Client, workers=1, candidate_ids=retry_ids)

    assert [row["candidate_id"] for row in retained] == ["candidate-1"]
    assert [row["candidate_id"] for row in retried] == ["candidate-2"]
    assert "怎么处理？" in requests[0]["messages"][1]["content"]
    assert "在吗" not in requests[0]["messages"][1]["content"]


def test_reuses_one_client_for_retries_in_the_same_worker(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, [_sample("candidate-1", ["怎么处理？"], ["我确认下"])])
    created = []

    class Client:
        model = "deepseek-v4-flash"

        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return json.dumps({name: False for name in CANDIDATE_FEATURE_NAMES})

    def factory():
        client = Client()
        created.append(client)
        return client

    rows, _ = extract(samples_path, client_factory=factory, workers=1, max_attempts=2)

    assert len(created) == 1
    assert created[0].calls == 2
    assert rows[0]["status"] == "ok"
    assert rows[0]["attempts"] == 2


def test_resume_can_limit_candidates_to_group_targets(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    group = _sample("group-candidate", ["在吗"], ["在"])
    group["target"] = {"case_id": "group-target", "relationship": "group"}
    private = _sample("private-candidate", ["怎么处理？"], ["我确认下"])
    _write_jsonl(samples_path, [group, private])

    output_dir = tmp_path / "out"
    write_output(output_dir, [{
        "candidate_id": "private-candidate",
        "relationship": "acquaintance",
        "status": "ok",
        "features": {name: False for name in CANDIDATE_FEATURE_NAMES},
        "attempts": 1,
    }], samples_path=samples_path, model="deepseek-v4-flash")

    retained, retry_ids = _resumable_rows(
        output_dir, samples_path, target_relationship="group",
    )

    assert retained == []
    assert retry_ids == {"group-candidate"}
