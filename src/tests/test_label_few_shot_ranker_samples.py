import json
import stat

import numpy as np
import pytest

import scripts.label_few_shot_ranker_samples as labeler


def _judge(*, names=("feature",), coefficients=(1.0,)):
    return labeler.FormalJudge(
        feature_names=names,
        coefficients=np.asarray(coefficients, dtype=float),
        context_schema={"known_group_names": (), "known_group_members": ()},
        schema_hash="schema-hash",
        artifact_sha256="artifact-hash",
    )


def _judge_run(tmp_path):
    return labeler.JudgeRun(
        config_path=tmp_path / "current_judge.json",
        config_sha256="config-hash",
        baseline_id="judge-baseline-id",
        artifact_path=tmp_path / "judge.json",
        model="configured-feature-model",
        reasoning_effort="configured-reasoning",
        timeout_seconds=321,
        is_experimental_override=False,
        required_current_c0_baseline_id=labeler.current_c0_baseline_id(),
    )


def _sample():
    return {
        "sample_id": "sample-id",
        "generation": {
            "few_shot_count": 1,
            "static_few_shot_count": 0,
            "injected_candidate_id": "bound-candidate",
        },
        "target": {
            "case_id": "case-id",
            "relationship": "group",
            "chat_id": "chat-id",
            "timestamp": 1_700_000_000,
            "context": ["原始上下文秘密"],
            "context_messages": [{"sender": "错误的发送者", "text": "原始上下文秘密"}],
        },
        "candidate": {
            "id": "bound-candidate",
            "relationship": "group",
            "chat_id": "historical-chat",
            "timestamp": 1_600_000_000,
            "context": ["历史示例秘密"],
            "reply": ["历史回复秘密"],
        },
    }


def _manifest_case():
    return {
        "case_id": "case-id",
        "chat_id": "chat-id",
        "replies": ["真人回复秘密"],
        "reply_messages": [],
    }


def _context_xml():
    return [
        "<context>"
        "<message role='other' type='text'>甲（2026-01-01 10:00）：第一句</message>"
        "<message role='self' type='text'>我的回复</message>"
        "<message role='other' type='text'>乙（2026-01-01 10:01）：最后一个问题？</message>"
        "</context>"
    ]


def _c0_user_prompt():
    return (
        "<history>\n"
        '<message index="1" role="other" type="text">甲（2026-01-01 10:00）：第一句</message>\n'
        '<message index="2" role="self" type="text">我的回复</message>\n'
        "</history>\n"
        "<unread>\n"
        '<message index="3" role="other" type="text">乙（2026-01-01 10:01）：最后一个问题？</message>\n'
        "</unread>"
    )


def test_fixed_retriever_accepts_only_its_bound_candidate():
    retriever = labeler.FixedOneFewShotRetriever({"id": "bound", "context": ["x"], "reply": ["y"]})

    assert retriever.retrieve(limit=1) == [{"id": "bound", "context": ["x"], "reply": ["y"]}]
    with pytest.raises(ValueError, match="limit=1"):
        retriever.retrieve(limit=8)
    with pytest.raises(ValueError, match="unbound"):
        retriever.render_selected([{"id": "other"}])


def test_label_one_binds_three_few_shots_in_frozen_order(monkeypatch):
    sample = _sample()
    candidates = [
        {**sample["candidate"], "id": f"candidate-{index}"}
        for index in range(1, 4)
    ]
    sample.pop("candidate")
    sample["candidates"] = candidates
    sample["generation"] = {
        "few_shot_count": 3,
        "static_few_shot_count": 0,
        "injected_candidate_ids": [candidate["id"] for candidate in candidates],
    }

    def c0_runner(_target, _payload, *, persona_few_shot_retriever, persona_few_shot_count, **_kwargs):
        assert persona_few_shot_count == 3
        selected = persona_few_shot_retriever.retrieve(limit=3)
        assert [row["id"] for row in selected] == [candidate["id"] for candidate in candidates]
        return {
            "few_shot_ids": [candidate["id"] for candidate in candidates],
            "replies": ["生成回复秘密"],
            "context_original": _context_xml(),
            "input_metadata": {},
            "trace": {"user_prompt": _c0_user_prompt()},
        }

    monkeypatch.setattr(labeler, "score_formal_judge_pair", lambda *_args: 0.75)
    row = labeler.label_one(
        sample,
        manifest_by_case_id={"case-id": _manifest_case()},
        payloads={"chat-id": {"payload": "private"}},
        judge=_judge(),
        feature_runner=lambda *_args: ({}, {}),
        c0_runner=c0_runner,
        c0_generator=object(),
    )

    assert row["label"] == 1
    assert row["injected_candidate_ids"] == [candidate["id"] for candidate in candidates]
    assert "injected_candidate_id" not in row


def test_pair_score_is_complementary_when_a_and_b_swap(monkeypatch):
    monkeypatch.setattr(labeler, "validate_option_features", lambda value: value)
    monkeypatch.setattr(
        labeler,
        "vectorize_boolean_options",
        lambda features: (
            np.asarray([features["option_A"]["value"]], dtype=float),
            np.asarray([features["option_B"]["value"]], dtype=float),
            ["feature"],
        ),
    )
    monkeypatch.setattr(
        labeler,
        "vectorize_observable_option_booleans",
        lambda _reply, _metadata: (np.asarray([], dtype=float), []),
    )
    monkeypatch.setattr(
        labeler,
        "vectorize_group_pattern_option_boolean",
        lambda _reply, _metadata: (np.asarray([], dtype=float), []),
    )
    monkeypatch.setattr(
        labeler,
        "vectorize_context_booleans",
        lambda _metadata, _schema: (np.asarray([], dtype=float), []),
    )
    monkeypatch.setattr(
        labeler,
        "expand_boolean_option_with_refined_context",
        lambda vector, names, _context, _context_names: (vector, names),
    )
    features = {"option_A": {"value": 1}, "option_B": {"value": 0}}

    forward = labeler.score_formal_judge_pair(_judge(coefficients=(2.0,)), features, ["a"], ["b"], {})
    backward = labeler.score_formal_judge_pair(
        _judge(coefficients=(2.0,)),
        {"option_A": {"value": 0}, "option_B": {"value": 1}},
        ["b"],
        ["a"],
        {},
    )

    assert forward > 0.5
    assert forward + backward == pytest.approx(1.0)


def test_label_one_uses_c0_prompt_xml_context_and_half_probability_is_positive(monkeypatch):
    received = {}

    def c0_runner(_target, _payload, *, persona_few_shot_retriever, persona_few_shot_count, **_kwargs):
        assert persona_few_shot_count == 1
        assert persona_few_shot_retriever.retrieve(limit=1)[0]["id"] == "bound-candidate"
        return {
            "few_shot_ids": ["bound-candidate"],
            "replies": ["生成回复秘密"],
            "context_original": ["正文含有裸 & 和 <，不能当 XML 解析"],
            "input_metadata": {"chat_name": "测试群"},
            "trace": {"user_prompt": _c0_user_prompt()},
        }

    def score(_judge, _features, _reply_a, _reply_b, metadata):
        received.update(metadata)
        return 0.5

    monkeypatch.setattr(labeler, "score_formal_judge_pair", score)
    row = labeler.label_one(
        _sample(),
        manifest_by_case_id={"case-id": _manifest_case()},
        payloads={"chat-id": {"payload": "private"}},
        judge=_judge(),
        feature_runner=lambda *_args: ({}, {}),
        c0_runner=c0_runner,
        c0_generator=object(),
    )

    assert row["label"] == 1
    assert row["probability"] == 0.5
    assert set(row) == {
        "sample_id", "relationship", "label", "probability", "injected_candidate_id",
        "schema_hash", "duration_seconds", "status",
    }
    assert received["recent_speakers"][:3] == ("乙", "__self__", "甲")
    assert received["latest_message_has_question_mark"] is True
    assert received["recent_group_other_texts"][-1].endswith("最后一个问题？")


def test_formal_judge_context_reuses_escaped_prompt_context_not_raw_message_text():
    context = labeler.formal_judge_context({
        "context_original": ["正文有裸 & 和 <，不应被解析"],
        "trace": {"user_prompt": _c0_user_prompt()},
    })

    assert labeler.message_text_timeline(context) == (
        "甲（2026-01-01 10:00）：第一句",
        "我的回复",
        "乙（2026-01-01 10:01）：最后一个问题？",
    )
    assert labeler.speaker_timeline(context) == ("甲", "__self__", "乙")


def test_label_one_reports_safe_generation_and_feature_failure_categories():
    common = {
        "manifest_by_case_id": {"case-id": _manifest_case()},
        "payloads": {"chat-id": {"payload": "private"}},
        "judge": _judge(),
        "c0_generator": object(),
    }

    with pytest.raises(labeler.C0GenerationError, match="C0 replay failed"):
        labeler.label_one(
            _sample(),
            c0_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private")),
            **common,
        )

    with pytest.raises(labeler.FeatureExtractionError, match="Judge feature extraction failed"):
        labeler.label_one(
            _sample(),
            c0_runner=lambda *_args, **_kwargs: {
                "few_shot_ids": ["bound-candidate"],
                "replies": ["generated"],
                "context_original": _context_xml(),
                "input_metadata": {},
                "trace": {"user_prompt": _c0_user_prompt()},
            },
            feature_runner=lambda *_args: (_ for _ in ()).throw(RuntimeError("private")),
            **common,
        )


def test_label_one_reports_safe_binding_and_response_shape_categories():
    common = {
        "manifest_by_case_id": {"case-id": _manifest_case()},
        "payloads": {"chat-id": {"payload": "private"}},
        "judge": _judge(),
        "c0_generator": object(),
    }

    with pytest.raises(labeler.FewShotBindingError, match="bound few-shot"):
        labeler.label_one(
            _sample(),
            c0_runner=lambda *_args, **_kwargs: {"few_shot_ids": []},
            **common,
        )

    with pytest.raises(labeler.C0ResponseShapeError, match="invalid shape"):
        labeler.label_one(
            _sample(),
            c0_runner=lambda *_args, **_kwargs: {
                "few_shot_ids": ["bound-candidate"],
                "replies": "not-a-list",
                "context_original": _context_xml(),
            },
            **common,
        )


def test_malformed_or_schema_incompatible_judge_artifact_fails_closed(tmp_path):
    artifact = tmp_path / "judge.json"
    artifact.write_text(json.dumps({"selected_model_name": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen pairwise boolean"):
        labeler.load_formal_judge(artifact)


def test_resolve_judge_run_uses_adopted_config_and_rejects_silent_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "current_judge.json"
    config_path.write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "artifact.json"
    config = {
        "execution": {
            "model": "configured-model",
            "reasoning_effort": "low",
            "timeout_seconds": 360,
        },
        "correction": {"artifact": {"path": str(artifact_path)}},
    }
    monkeypatch.setattr(labeler, "load_current_judge", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(labeler, "current_judge_baseline_id", lambda *_args, **_kwargs: "judge-test")

    resolved = labeler.resolve_judge_run(judge_config_path=config_path)

    assert resolved.artifact_path == artifact_path
    assert resolved.model == "configured-model"
    assert resolved.reasoning_effort == "low"
    assert resolved.timeout_seconds == 360
    assert resolved.baseline_id == "judge-test"
    assert resolved.is_experimental_override is False
    with pytest.raises(ValueError, match="allow_experimental_judge_override"):
        labeler.resolve_judge_run(judge_config_path=config_path, feature_model="other-model")


def test_label_manifest_records_resolved_judge_execution(tmp_path, monkeypatch):
    sample = _sample()
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "judge-manifest.json"
    manifest_path.write_text(json.dumps({"splits": {"train": [_manifest_case()]}}), encoding="utf-8")
    monkeypatch.setattr(labeler, "resolve_judge_run", lambda **_kwargs: _judge_run(tmp_path))
    monkeypatch.setattr(labeler, "load_formal_judge", lambda _path: _judge())
    seen = {}
    monkeypatch.setattr(
        labeler,
        "label_one",
        lambda row, **kwargs: seen.update(kwargs) or {
            "sample_id": row["sample_id"], "relationship": "group", "label": 1,
            "probability": 0.5, "injected_candidate_id": "bound-candidate",
            "schema_hash": "schema-hash", "duration_seconds": 0.001, "status": "ok",
        },
    )

    result = labeler.label(
        samples_path=samples_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        payloads={},
        c0_generator=object(),
        limit=1,
    )

    assert seen["feature_timeout_seconds"] == 321
    assert seen["feature_model"] == "configured-feature-model"
    assert seen["feature_reasoning_effort"] == "configured-reasoning"
    assert result["judge_config"]["path"] == str(tmp_path / "current_judge.json")
    assert result["judge_config"]["sha256"] == "config-hash"
    assert result["judge_config"]["baseline_id"] == "judge-baseline-id"
    assert result["parameters"]["experimental_judge_override"] is False


def test_label_accepts_cases_manifest_and_normalizes_private_reply(tmp_path, monkeypatch):
    sample = _sample()
    sample["target"]["relationship"] = "private"
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "cases": [{
            "id": "case-id", "chat_id": "chat-id", "relationship": "acquaintance",
            "reply": ["真人回复秘密"],
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(labeler, "resolve_judge_run", lambda **_kwargs: _judge_run(tmp_path))
    monkeypatch.setattr(labeler, "load_formal_judge", lambda _path: _judge())
    seen = {}
    monkeypatch.setattr(
        labeler,
        "label_one",
        lambda row, **kwargs: seen.update(kwargs["manifest_by_case_id"]["case-id"]) or {
            "sample_id": row["sample_id"], "relationship": "private", "label": 1,
            "probability": 0.5, "injected_candidate_id": "bound-candidate",
            "schema_hash": "schema-hash", "duration_seconds": 0.001, "status": "ok",
        },
    )

    labeler.label(
        samples_path=samples_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        payloads={},
        c0_generator=object(),
        limit=1,
    )

    assert seen["case_id"] == "case-id"
    assert seen["relationship"] == "private"
    assert seen["replies"] == ["真人回复秘密"]


def test_label_rejects_a_non_judge_manifest_before_loading_the_judge(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(_sample(), ensure_ascii=False) + "\n", encoding="utf-8")
    wrong_manifest_path = tmp_path / "sample-manifest.json"
    wrong_manifest_path.write_text(json.dumps({"kind": "single_few_shot_samples"}), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty splits mapping"):
        labeler.label(
            samples_path=samples_path,
            manifest_path=wrong_manifest_path,
            output_dir=tmp_path / "out",
            payloads={},
            limit=1,
        )


def test_durable_output_is_safe_and_resume_only_retries_errors(tmp_path, monkeypatch):
    sample = _sample()
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    judge_manifest_path = tmp_path / "judge-manifest.json"
    judge_manifest_path.write_text(
        json.dumps({"splits": {"train": [_manifest_case()]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    fsync_calls = []
    attempts = []

    monkeypatch.setattr(labeler, "load_formal_judge", lambda _path: _judge())
    monkeypatch.setattr(labeler, "resolve_judge_run", lambda **_kwargs: _judge_run(tmp_path))
    monkeypatch.setattr(labeler.os, "fsync", lambda fd: fsync_calls.append(fd))

    def fail(*_args, **_kwargs):
        attempts.append("fail")
        raise ValueError("生成回复秘密、prompt秘密、异常上下文秘密")

    monkeypatch.setattr(labeler, "label_one", fail)
    labeler.label(
        samples_path=samples_path,
        manifest_path=judge_manifest_path,
        output_dir=output_dir,
        payloads={},
        c0_generator=object(),
        limit=1,
    )
    labeler.label(
        samples_path=samples_path,
        manifest_path=judge_manifest_path,
        output_dir=output_dir,
        payloads={},
        c0_generator=object(),
        limit=1,
        resume=True,
    )

    def succeed(*_args, **_kwargs):
        attempts.append("success")
        return {
            "sample_id": "sample-id",
            "relationship": "group",
            "label": 1,
            "probability": 0.5,
            "injected_candidate_id": "bound-candidate",
            "schema_hash": "schema-hash",
            "duration_seconds": 0.001,
            "status": "ok",
        }

    monkeypatch.setattr(labeler, "label_one", succeed)
    labeler.label(
        samples_path=samples_path,
        manifest_path=judge_manifest_path,
        output_dir=output_dir,
        payloads={},
        limit=1,
        resume=True,
        retry_errors=True,
    )

    output = (output_dir / "labels.jsonl").read_text(encoding="utf-8")
    manifest = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert attempts == ["fail", "success"]
    assert fsync_calls
    assert "生成回复秘密" not in output + manifest
    assert "prompt秘密" not in output + manifest
    assert "异常上下文秘密" not in output + manifest
    assert stat.S_IMODE((output_dir / "labels.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((output_dir / "manifest.json").stat().st_mode) == 0o600
    results = json.loads(manifest)["results"]
    assert results["append_only_records"] == 2
    assert results["unique_samples"] == 1
    assert results["latest_status_counts"] == {"ok": 1}


def test_label_reuses_one_c0_generator_across_pending_samples(tmp_path, monkeypatch):
    first = _sample()
    second = _sample()
    second["sample_id"] = "second-sample-id"
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (first, second)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": [_manifest_case()]}}, ensure_ascii=False), encoding="utf-8"
    )
    generator = object()
    factory_calls = []
    seen_generators = []
    monkeypatch.setattr(labeler, "load_formal_judge", lambda _path: _judge())
    monkeypatch.setattr(labeler, "resolve_judge_run", lambda **_kwargs: _judge_run(tmp_path))
    monkeypatch.setattr(
        labeler,
        "create_current_c0_generator",
        lambda **_kwargs: factory_calls.append("created") or generator,
    )
    monkeypatch.setattr(
        labeler,
        "label_one",
        lambda row, **kwargs: seen_generators.append(kwargs["c0_generator"]) or {
            "sample_id": row["sample_id"], "relationship": "group", "label": 1,
            "probability": 0.5, "injected_candidate_id": "bound-candidate",
            "schema_hash": "schema-hash", "duration_seconds": 0.001, "status": "ok",
        },
    )

    labeler.label(
        samples_path=samples_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        payloads={},
        limit=2,
    )

    assert factory_calls == ["created"]
    assert seen_generators == [generator, generator]


def test_resume_keeps_an_earlier_success_after_a_later_error(tmp_path, monkeypatch):
    sample = _sample()
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    judge_manifest_path = tmp_path / "judge-manifest.json"
    judge_manifest_path.write_text(
        json.dumps({"splits": {"train": [_manifest_case()]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "labels.jsonl").write_text(
        "\n".join((
            json.dumps({"sample_id": "sample-id", "status": "ok", "label": 1}),
            json.dumps({"sample_id": "sample-id", "status": "error", "error_type": "ValueError"}),
        )) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(labeler, "load_formal_judge", lambda _path: _judge())
    monkeypatch.setattr(labeler, "resolve_judge_run", lambda **_kwargs: _judge_run(tmp_path))

    result = labeler.label(
        samples_path=samples_path,
        manifest_path=judge_manifest_path,
        output_dir=output_dir,
        payloads={},
        limit=1,
        resume=True,
        retry_errors=True,
    )

    assert result["results"]["requested_this_run"] == 0
    assert result["results"]["latest_status_counts"] == {"ok": 1}


def test_label_restricts_a_worker_to_requested_target_case_ids(tmp_path, monkeypatch):
    selected = _sample()
    ignored = _sample()
    ignored["sample_id"] = "ignored-sample"
    ignored["target"] = {**ignored["target"], "case_id": "ignored-case"}
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (selected, ignored)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": [_manifest_case()]}}, ensure_ascii=False), encoding="utf-8"
    )
    judge_path = tmp_path / "judge.json"
    judge_path.write_text("{}", encoding="utf-8")
    seen = []

    monkeypatch.setattr(labeler, "load_formal_judge", lambda _path: _judge())
    monkeypatch.setattr(labeler, "resolve_judge_run", lambda **_kwargs: _judge_run(tmp_path))
    monkeypatch.setattr(
        labeler,
        "label_one",
        lambda row, **_kwargs: seen.append(row["sample_id"]) or {
            "sample_id": row["sample_id"], "relationship": "group", "label": 1,
            "probability": 0.5, "injected_candidate_id": "bound-candidate",
            "schema_hash": "schema-hash", "duration_seconds": 0.001, "status": "ok",
        },
    )
    labeler.label(
        samples_path=samples_path,
        manifest_path=manifest_path,
        judge_path=judge_path,
        output_dir=tmp_path / "out",
        payloads={},
        c0_generator=object(),
        all_rows=True,
        target_case_ids=("case-id",),
    )

    assert seen == ["sample-id"]
