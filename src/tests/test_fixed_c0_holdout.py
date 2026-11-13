from scripts import run_fixed_c0_holdout as formal_ab


def test_formal_forced_reply_reserves_independent_fallback_budget(monkeypatch):
    observed = {}
    clock = {"now": 1_000.0}

    def fake_parent(instance, messages, tools, deadline, chat_name):
        observed["main_deadline"] = deadline
        clock["now"] = deadline
        return [], messages + [{"role": "tool", "content": "large trace"}]

    class FakeClient:
        def chat(self, **kwargs):
            observed["fallback_timeout"] = kwargs["timeout"]
            observed["fallback_messages"] = kwargs["messages"]
            return '{"replies":["我看看"]}'

    monkeypatch.setattr(
        formal_ab.AdoptedBaselineHistoricalGenerator,
        "_react_generate",
        fake_parent,
    )
    monkeypatch.setattr(formal_ab.time, "time", lambda: clock["now"])
    generator = object.__new__(formal_ab.ForcedReplyAdoptedBaselineHistoricalGenerator)
    generator.llm_client = FakeClient()
    generator.last_generation_trace = []
    generator.last_generation_failed = True
    generator.last_raw_response = ""

    replies, _ = generator._react_generate(
        [{"role": "user", "content": "test"}], [], 1_180.0, "test"
    )

    assert observed["main_deadline"] == 1_090.0
    assert observed["fallback_timeout"] == 90.0
    assert all(message["role"] != "tool" for message in observed["fallback_messages"])
    assert replies == ["我看看"]


def test_formal_generators_isolate_only_the_two_accumulated_features(monkeypatch):
    captured = []

    def fake_init(instance, *args, **kwargs):
        captured.append(dict(kwargs))
        instance.enable_information_gap_response_move = kwargs["enable_information_gap_response_move"]
        instance.prefer_latest_turn_form = kwargs["prefer_latest_turn_form"]

    monkeypatch.setattr(
        formal_ab.ForcedReplyAdoptedBaselineHistoricalGenerator,
        "__init__",
        fake_init,
    )

    baseline = formal_ab.FormalBaselineGenerator()
    candidate = formal_ab.CumulativeCandidateGenerator()

    assert not baseline.enable_information_gap_response_move
    assert not baseline.prefer_latest_turn_form
    assert candidate.enable_information_gap_response_move
    assert candidate.prefer_latest_turn_form
    assert captured == [
        {
            "enable_information_gap_response_move": False,
            "prefer_latest_turn_form": False,
        },
        {
            "enable_information_gap_response_move": True,
            "prefer_latest_turn_form": True,
        },
    ]


def test_formal_checkpoint_keys_keep_versions_separate():
    assert formal_ab._entry_id("case-1", "baseline") == "case-1:baseline"
    assert formal_ab._entry_id("case-1", "candidate") == "case-1:candidate"


def test_formal_generation_uses_compact_verified_turn_for_both_versions(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        model = "deepseek-v4-flash"

    class FakeMemory:
        def __init__(self, llm_client):
            self.llm_client = llm_client

        def shutdown(self):
            return None

    def fake_generate_case(source, client, memory, timeout, **kwargs):
        calls.append(kwargs)
        return {
            "context": [],
            "human_reply": ["真人"],
            "c0_reply": ["候选"],
            "generation": {},
            "validity": {"formally_scorable": True, "exclusions": []},
        }

    monkeypatch.setattr(formal_ab, "_generation_client", lambda *args: FakeClient())
    monkeypatch.setattr(formal_ab, "MemoryEngine", FakeMemory)
    monkeypatch.setattr(formal_ab, "generate_case", fake_generate_case)
    monkeypatch.setattr(formal_ab, "_write_results", lambda *args: {})
    monkeypatch.setattr(formal_ab, "_experiment_persona", lambda *args: formal_ab.PERSONA)
    monkeypatch.setattr(formal_ab, "_locked_config", lambda *args: {"generation_model": "deepseek-v4-flash"})

    formal_ab.run_generation(
        tmp_path,
        [{"internal_id": "case-1"}],
        None,
        180.0,
        True,
        None,
        formal_ab.FEW_SHOTS,
    )

    assert [call["compact_verified_turn"] for call in calls] == [True, True]
    assert [call["generator_cls"] for call in calls] == [
        formal_ab.FormalBaselineGenerator,
        formal_ab.CumulativeCandidateGenerator,
    ]
    assert formal_ab.os.environ["PERSONA_FEW_SHOT_PATH"] == str(formal_ab.FEW_SHOTS)
    assert formal_ab.os.environ["GROUP_PERSONA_FEW_SHOT_PATH"] == str(formal_ab.FEW_SHOTS)


def test_parallel_generation_writes_each_case_and_version_once(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        model = "deepseek-v4-flash"

    class FakeMemory:
        def __init__(self, llm_client):
            self.llm_client = llm_client

        def shutdown(self):
            return None

    def fake_generate_case(source, client, memory, timeout, **kwargs):
        condition = (
            "baseline"
            if kwargs["generator_cls"] is formal_ab.FormalBaselineGenerator
            else "candidate"
        )
        calls.append((source["internal_id"], condition))
        return {
            "context": [],
            "human_reply": ["真人"],
            "c0_reply": [f"{source['internal_id']}-{condition}"],
            "generation": {},
            "validity": {"formally_scorable": True, "exclusions": []},
        }

    monkeypatch.setattr(formal_ab, "_generation_client", lambda *args: FakeClient())
    monkeypatch.setattr(formal_ab, "MemoryEngine", FakeMemory)
    monkeypatch.setattr(formal_ab, "generate_case", fake_generate_case)
    monkeypatch.setattr(formal_ab, "_write_results", lambda *args: {})
    monkeypatch.setattr(formal_ab, "_experiment_persona", lambda *args: formal_ab.PERSONA)
    monkeypatch.setattr(formal_ab, "_locked_config", lambda *args: {"generation_model": "deepseek-v4-flash"})

    sources = [{"internal_id": f"case-{index}"} for index in range(8)]
    formal_ab.run_generation(
        tmp_path,
        sources,
        None,
        180.0,
        True,
        None,
        formal_ab.FEW_SHOTS,
        workers=4,
    )

    checkpoint = formal_ab._load(tmp_path / "generation_checkpoint.json", {})
    assert len(checkpoint["entries"]) == 16
    assert len(calls) == 16
    assert len(set(calls)) == 16
    formal_ab._require_complete_generation(sources, checkpoint["entries"])


def test_complete_generation_rejects_a_missing_version():
    sources = [{"internal_id": "case-1"}]
    entries = {
        "case-1:baseline": {"status": "scorable"},
    }

    try:
        formal_ab._require_complete_generation(sources, entries)
    except RuntimeError as exc:
        assert "scorable=1/2" in str(exc)
        assert "missing=1" in str(exc)
    else:
        raise AssertionError("missing candidate must block the formal Judge run")


def _paired_case(baseline_correct: bool, candidate_correct: bool) -> dict:
    return {
        "chat_type": "group",
        "complex_large_group": False,
        "specified_group": None,
        "media_dependency": False,
        "baseline": {"judge": {"correct": baseline_correct}},
        "candidate": {"judge": {"correct": candidate_correct}},
    }


def test_summary_uses_lower_candidate_discrimination_as_improvement():
    summary = formal_ab._summary([
        _paired_case(True, False),
        _paired_case(False, True),
        _paired_case(True, True),
    ])

    assert summary["baseline_judge_correct"] == 2
    assert summary["candidate_judge_correct"] == 2
    assert summary["improved"] == 1
    assert summary["regressed"] == 1
    assert summary["unchanged"] == 1
    assert summary["net_improvement"] == 0
    assert summary["status"] == "partial"


def test_summary_is_complete_only_with_all_240_paired_cases():
    cases = [_paired_case(True, False) for _ in range(240)]

    assert formal_ab._summary(cases)["status"] == "complete"
    assert formal_ab._summary(cases[:239])["status"] == "partial"
