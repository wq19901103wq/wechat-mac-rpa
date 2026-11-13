import argparse
import hashlib
import json

import pytest

from scripts import run_situation_c0_ab as runner


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_args(tmp_path, formal_pool, fixed):
    return argparse.Namespace(
        dataset="formal240",
        candidate_mode="latest-turn-form",
        private_only_candidate=True,
        repetitions=1,
        max_cases=240,
        case_id=[],
        case_ids_from_results=None,
        few_shot=formal_pool,
        few_shot_count=3,
        few_shot_max_chars=2500,
        named_context_few_shot=tmp_path / "named.jsonl",
        generation_provider="volcengine",
        fixed_test=fixed,
        output=tmp_path / "must-not-exist-before-preflight",
        shard_count=1,
        shard_index=0,
        preflight_only=False,
    )


def _freeze_test_persona(monkeypatch, tmp_path):
    persona = "fixed test persona"
    path = tmp_path / "frozen-persona.md"
    path.write_text(persona, encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_DEVELOPMENT80_FORCE_REPLY_PERSONA", path)
    monkeypatch.setattr(
        runner,
        "DEVELOPMENT80_FORCE_REPLY_SHA256",
        hashlib.sha256(persona.encode("utf-8")).hexdigest(),
    )


def _build_frozen_formal_baseline(tmp_path, baseline_id="c0-test"):
    run = tmp_path / "frozen-run"
    source_one = tmp_path / "source-one.json"
    source_two = tmp_path / "source-two.json"
    case_rows = [
        {
            "internal_id": f"case-{index}",
            "chat_type": "group" if index < 160 else "private",
        }
        for index in range(240)
    ]
    source_one.write_text(json.dumps({"cases": case_rows[:120]}), encoding="utf-8")
    source_two.write_text(json.dumps({"cases": case_rows[120:]}), encoding="utf-8")
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "current_c0_baseline_id": baseline_id,
        "case_ids": [row["internal_id"] for row in case_rows],
    }), encoding="utf-8")
    for shard_index in range(4):
        entries = {}
        for row in case_rows[shard_index * 60:(shard_index + 1) * 60]:
            entries[f'{row["internal_id"]}:baseline'] = {
                "status": "scorable",
                "result": {"replies": [f'reply {row["internal_id"]}']},
            }
        (run / f"generation_checkpoint.shard-{shard_index}-of-4.json").write_text(
            json.dumps({"current_c0_baseline_id": baseline_id, "entries": entries}),
            encoding="utf-8",
        )
    return run, (source_one, source_two), case_rows


def _frozen_target_args(run, sources):
    return argparse.Namespace(
        dataset="formal240",
        candidate_mode="group-public-question-specific-followup",
        frozen_formal_baseline_run=run,
        frozen_formal_source_manifest=list(sources),
    )


def test_frozen_formal_baseline_recovers_canonical_160_group_80_private(tmp_path):
    run, sources, case_rows = _build_frozen_formal_baseline(tmp_path)

    bundle = runner._load_frozen_formal_baseline(
        _frozen_target_args(run, sources),
        "c0-test",
    )

    assert [row["id"] for row in bundle["cases"]] == [
        row["internal_id"] for row in case_rows
    ]
    assert bundle["distribution"]["relationship"] == {"group": 160, "private": 80}
    assert len(bundle["baseline_entries"]) == 240
    assert len(bundle["provenance"]["generation_checkpoints"]) == 4


def test_frozen_formal_baseline_rejects_wrong_id_or_invalid_reply(tmp_path):
    run, sources, _ = _build_frozen_formal_baseline(tmp_path)
    args = _frozen_target_args(run, sources)

    with pytest.raises(ValueError, match="not bound"):
        runner._load_frozen_formal_baseline(args, "another-c0")

    checkpoint = run / "generation_checkpoint.shard-0-of-4.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["entries"]["case-0:baseline"]["result"]["replies"] = []
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not scorable with a reply"):
        runner._load_frozen_formal_baseline(args, "c0-test")


def test_import_frozen_formal_baseline_rejects_any_substitute():
    bundle = {
        "baseline_entries": {
            "case-1": {"status": "scorable", "result": {"replies": ["原回复"]}},
        },
    }
    entries = {}

    assert runner._import_frozen_formal_baseline_entries(
        entries, ["case-1"], bundle
    ) == 1
    assert entries["case-1:baseline"]["protected_frozen_baseline"] is True
    assert entries["case-1:baseline"]["result"]["replies"] == ["原回复"]

    with pytest.raises(ValueError, match="regenerated formal baseline"):
        runner._import_frozen_formal_baseline_entries(
            {"case-1:baseline": {"status": "scorable"}}, ["case-1"], bundle
        )

    with pytest.raises(ValueError, match="does not match"):
        runner._import_frozen_formal_baseline_entries(
            {
                "case-1:baseline": {
                    "status": "scorable",
                    "result": {"replies": ["另一条"]},
                    "protected_frozen_baseline": True,
                },
            },
            ["case-1"],
            bundle,
        )


def test_frozen_target_selects_canonical_cases_not_fixed_test(monkeypatch, tmp_path):
    run, sources, _ = _build_frozen_formal_baseline(tmp_path)
    args = _frozen_target_args(run, sources)
    args.fixed_test = tmp_path / "wrong-fixed-test.json"
    args.fixed_test.write_text(json.dumps({"cases": []}), encoding="utf-8")
    bundle = runner._load_frozen_formal_baseline(args, "c0-test")
    monkeypatch.setattr(runner, "_formal_preflight", lambda ignored: None)

    cases, description = runner._select_dataset(args, frozen_bundle=bundle)

    assert len(cases) == 240
    assert description.startswith("approved frozen 240-case")


def test_formal240_reuses_exact_development_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    base = tmp_path / "base.jsonl"
    development_manifest = tmp_path / "development80.json"
    fixed = tmp_path / "fixed240.json"
    _write_jsonl(base, [{"id": "a"}, {"id": "b"}])
    development_manifest.write_text(json.dumps({
        "conditions": {"retrieval_libraries": {"candidate": {
            "path": "base.jsonl", "rows": 2, "sha256": _sha256(base)
        }}}
    }), encoding="utf-8")
    fixed.write_text(json.dumps({"cases": [
        {"internal_id": f"case-{index}", "chat_type": "group" if index < 120 else "private"}
        for index in range(240)
    ]}), encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_DEVELOPMENT80_RUN", development_manifest)
    args = _formal_args(tmp_path, base, fixed)
    args.named_context_few_shot.write_text("named\n", encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_NAMED_CONTEXT_FEW_SHOT", args.named_context_few_shot)
    monkeypatch.setattr(runner, "DEVELOPMENT80_NAMED_CONTEXT_SHA256", _sha256(args.named_context_few_shot))
    _freeze_test_persona(monkeypatch, tmp_path)

    cases, description = runner._select_dataset(args)

    assert len(cases) == 240
    assert cases[0]["id"] == "case-0"
    assert cases[0]["relationship"] == "group"
    assert cases[-1]["relationship"] == "private"
    assert description == "fixed 240 cases; only the dataset differs from development80"


def test_formal240_rejects_any_pool_other_than_development_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    base = tmp_path / "base.jsonl"
    formal_pool = tmp_path / "formal" / "persona_examples.jsonl"
    development_manifest = tmp_path / "development80.json"
    fixed = tmp_path / "fixed240.json"
    _write_jsonl(base, [{"id": "a"}])
    _write_jsonl(formal_pool, [{"id": "a"}])
    development_manifest.write_text(json.dumps({
        "conditions": {"retrieval_libraries": {"candidate": {
            "path": "base.jsonl", "rows": 1, "sha256": _sha256(base)
        }}}
    }), encoding="utf-8")
    fixed.write_text(json.dumps({"cases": []}), encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_DEVELOPMENT80_RUN", development_manifest)
    args = _formal_args(tmp_path, formal_pool, fixed)
    args.named_context_few_shot.write_text("named\n", encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_NAMED_CONTEXT_FEW_SHOT", args.named_context_few_shot)
    monkeypatch.setattr(runner, "DEVELOPMENT80_NAMED_CONTEXT_SHA256", _sha256(args.named_context_few_shot))
    _freeze_test_persona(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="exact same complete few-shot pool"):
        runner._formal_preflight(args)

    assert not args.output.exists()


def test_each_case_excludes_only_its_own_overlapping_few_shots():
    cases = [
        {"id": "case-a", "context": ["same a"], "reply": ["reply a"]},
        {"id": "case-b", "context": ["same b"], "reply": ["reply b"]},
    ]
    pool = {
        "example-a": {"id": "example-a", "context": ["same a"], "reply": ["reply a"]},
        "example-b": {"id": "example-b", "context": ["same b"], "reply": ["reply b"]},
        "example-c": {"id": "example-c", "context": ["other"], "reply": ["other reply"]},
    }

    excluded = runner._per_case_overlap_ids(cases, pool)

    assert excluded["case-a"] == {"case-a", "example-a"}
    assert excluded["case-b"] == {"case-b", "example-b"}
    assert "example-b" not in excluded["case-a"]
    assert "example-a" not in excluded["case-b"]


def test_exposed_case_recheck_requires_explicit_development_case_ids(tmp_path):
    args = argparse.Namespace(
        dataset="formal240",
        allow_exposed_case_recheck=True,
        case_id=["case-a"],
        case_ids_from_results=None,
    )
    with pytest.raises(ValueError, match="only allowed for development80"):
        runner._select_dataset(args)

    args.dataset = "development80"
    args.case_id = []
    with pytest.raises(ValueError, match="requires explicit --case-id"):
        runner._select_dataset(args)


def test_exposed_case_recheck_selects_only_named_old_cases(tmp_path):
    development = tmp_path / "development.json"
    exposed = tmp_path / "exposed.json"
    fixed = tmp_path / "fixed.json"
    previous = tmp_path / "previous.json"
    development.write_text(json.dumps({"cases": [
        {"id": "old-a", "relationship": "group", "context": ["a"]},
        {"id": "fresh-b", "relationship": "group", "context": ["b"]},
    ]}), encoding="utf-8")
    exposed.write_text(json.dumps({"case_ids": ["old-a"]}), encoding="utf-8")
    fixed.write_text(json.dumps({"cases": []}), encoding="utf-8")
    previous.write_text(json.dumps({"cases": []}), encoding="utf-8")
    args = argparse.Namespace(
        dataset="development80",
        allow_exposed_case_recheck=True,
        case_id=["old-a"],
        case_ids_from_results=None,
        development=development,
        exposed=exposed,
        previous_ab=[previous],
        fixed_test=fixed,
        max_cases=1,
        candidate_mode="group-self-context-freshness-boundary",
    )

    cases, description = runner._select_dataset(args)

    assert [row["id"] for row in cases] == ["old-a"]
    assert description.startswith("causal recheck")


def test_formal240_rejects_development_generation_condition_changes(monkeypatch, tmp_path):
    args = _formal_args(tmp_path, tmp_path / "wrong-pool.jsonl", tmp_path / "fixed.json")
    args.generation_provider = "official"

    with pytest.raises(ValueError, match="Volcengine"):
        runner._formal_preflight(args)

    args.generation_provider = "volcengine"
    args.few_shot_count = 5
    with pytest.raises(ValueError, match="three few-shot"):
        runner._formal_preflight(args)


def test_formal240_allows_group_pending_self_action(monkeypatch, tmp_path):
    base = tmp_path / "base.jsonl"
    development_manifest = tmp_path / "development80.json"
    named = tmp_path / "named.jsonl"
    _write_jsonl(base, [{"id": "a"}])
    development_manifest.write_text(json.dumps({
        "conditions": {"retrieval_libraries": {"candidate": {
            "path": "base.jsonl", "rows": 1, "sha256": _sha256(base)
        }}}
    }), encoding="utf-8")
    named.write_text("named\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DEFAULT_DEVELOPMENT80_RUN", development_manifest)
    monkeypatch.setattr(runner, "DEFAULT_NAMED_CONTEXT_FEW_SHOT", named)
    monkeypatch.setattr(runner, "DEVELOPMENT80_NAMED_CONTEXT_SHA256", _sha256(named))
    _freeze_test_persona(monkeypatch, tmp_path)
    args = _formal_args(tmp_path, base, tmp_path / "fixed.json")
    args.candidate_mode = "group-pending-self-action"
    args.named_context_few_shot = named

    runner._formal_preflight(args)


def test_formal240_allows_group_public_question_specific_followup(monkeypatch, tmp_path):
    base = tmp_path / "base.jsonl"
    development_manifest = tmp_path / "development80.json"
    named = tmp_path / "named.jsonl"
    _write_jsonl(base, [{"id": "a"}])
    development_manifest.write_text(json.dumps({
        "conditions": {"retrieval_libraries": {"candidate": {
            "path": "base.jsonl", "rows": 1, "sha256": _sha256(base)
        }}}
    }), encoding="utf-8")
    named.write_text("named\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DEFAULT_DEVELOPMENT80_RUN", development_manifest)
    monkeypatch.setattr(runner, "DEFAULT_NAMED_CONTEXT_FEW_SHOT", named)
    monkeypatch.setattr(runner, "DEVELOPMENT80_NAMED_CONTEXT_SHA256", _sha256(named))
    _freeze_test_persona(monkeypatch, tmp_path)
    args = _formal_args(tmp_path, base, tmp_path / "fixed.json")
    args.candidate_mode = "group-public-question-specific-followup"
    args.named_context_few_shot = named

    runner._formal_preflight(args)


def test_formal240_allows_group_explicit_self_stance_action(monkeypatch, tmp_path):
    base = tmp_path / "base.jsonl"
    development_manifest = tmp_path / "development80.json"
    named = tmp_path / "named.jsonl"
    _write_jsonl(base, [{"id": "a"}])
    development_manifest.write_text(json.dumps({
        "conditions": {"retrieval_libraries": {"candidate": {
            "path": "base.jsonl", "rows": 1, "sha256": _sha256(base)
        }}}
    }), encoding="utf-8")
    named.write_text("named\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DEFAULT_DEVELOPMENT80_RUN", development_manifest)
    monkeypatch.setattr(runner, "DEFAULT_NAMED_CONTEXT_FEW_SHOT", named)
    monkeypatch.setattr(runner, "DEVELOPMENT80_NAMED_CONTEXT_SHA256", _sha256(named))
    _freeze_test_persona(monkeypatch, tmp_path)
    args = _formal_args(tmp_path, base, tmp_path / "fixed.json")
    args.candidate_mode = "group-explicit-self-stance-action"
    args.named_context_few_shot = named

    runner._formal_preflight(args)


def test_formal240_requires_three_few_shots_in_the_actual_request():
    good = {
        "trace": {
            "few_shot_ids": ["one", "two", "three"],
            "few_shot_content": "three rendered examples",
        }
    }
    runner._require_actual_few_shots(good, 3, "case-1", "candidate")

    with pytest.raises(runner.FewShotInjectionMismatch, match="got 0"):
        runner._require_actual_few_shots(
            {"trace": {"few_shot_ids": [], "few_shot_content": ""}},
            3,
            "case-1",
            "candidate",
        )


def test_formal240_loads_each_cases_exact_source_path(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "sources" / "chat.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"messages": [{"content": "原始消息"}]}), encoding="utf-8")
    row = {
        "internal_id": "case-1",
        "chat_id": "not-present-in-exports",
        "source_path": "sources/chat.json",
    }
    cache = {}

    payload = runner._payload_for_case(row, "formal240", {}, cache)

    assert payload == {"messages": [{"content": "原始消息"}]}
    assert len(cache) == 1


def test_formal240_resolves_source_manifest_to_raw_export(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps({"messages": [{"serverId": "target", "content": "原始消息"}]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"source_files": [{"path": "raw.json"}]}),
        encoding="utf-8",
    )
    row = {"source_path": "manifest.json", "anchor_server_id": "target"}

    payload = runner._payload_for_case(row, "formal240", {}, {})

    assert payload == {"messages": [{"serverId": "target", "content": "原始消息"}]}
