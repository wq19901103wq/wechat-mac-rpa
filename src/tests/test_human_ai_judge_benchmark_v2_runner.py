import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_kimi_human_ai_judge_v2 import (
    _reference_without_case_leaks,
    _run_codex,
    _validate_dataset_shape,
    main as run_judge_main,
)

from scripts.evaluate_human_ai_judge import _parse_api_judge_response

from scripts.run_human_ai_judge_benchmark_v2 import (
    BASELINE_PROFILE,
    build_messages,
    claim_final_run,
    compare_presentations,
    load_resumable_run,
    profile_sha256,
    resolve_selected_reply,
    score_run,
    swap_case,
    validate_request_body,
    verify_frozen_artifacts,
)


V3_PROFILE_PATH = (
    Path(__file__).parents[2]
    / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v3_authorship_calibrated.json"
)
BALANCED160_DIR = (
    Path(__file__).parents[2]
    / "data/reports/private/human_ai_judge_benchmark_v3_balanced160"
)
BOUNDARY_PROFILE_PATH = (
    Path(__file__).parents[2]
    / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v2_boundary_candidate.json"
)


def _case():
    return {
        "case_id": "case-one",
        "relationship": "private",
        "context_original": ["在吗"],
        "option_A": ["在"],
        "option_B": ["您好，请问有什么可以帮助您？"],
    }


def test_reference_filter_removes_only_the_current_case_example():
    case = _case()
    reference = {
        "scenario_few_shots": [
            {"id": "leak", "context_original": ["在吗"], "option_A": ["在"], "option_B": ["您好，请问有什么可以帮助您？"]},
            {"id": "safe", "context_original": ["吃了吗"], "option_A": ["吃了"], "option_B": ["还没"]},
        ]
    }

    filtered, removed = _reference_without_case_leaks(reference, case)

    assert [row["id"] for row in filtered["scenario_few_shots"]] == ["safe"]
    assert removed == [{"section": "scenario_few_shots", "index": 0, "id": "leak"}]
    assert [row["id"] for row in reference["scenario_few_shots"]] == ["leak", "safe"]


def test_codex_judge_records_explicit_sol_xhigh_configuration(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["timeout"] = kwargs["timeout"]
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            '{"human_option":"A","confidence":0.8,"reason":"test"}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_kimi_human_ai_judge_v2.subprocess.run", fake_run)
    _, audit, _, _ = _run_codex(
        [{"content": "system"}, {"content": "user"}],
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        timeout_seconds=600,
    )

    assert captured["command"][captured["command"].index("--model") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="xhigh"' in captured["command"]
    assert captured["timeout"] == 600
    assert audit["model"] == "gpt-5.6-sol"
    assert audit["reasoning_effort"] == "xhigh"
    assert audit["timeout_seconds"] == 600


def test_cli_forwards_explicit_codex_configuration(monkeypatch, tmp_path):
    captured = {"kwargs": {}}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"completed": 0, "total": 0}

    monkeypatch.setattr("scripts.run_kimi_human_ai_judge_v2.run", fake_run)
    monkeypatch.setattr("sys.argv", [
        "run_kimi_human_ai_judge_v2.py",
        "--provider", "codex-cli",
        "--profile", str(tmp_path / "profile.json"),
        "--output", str(tmp_path / "output.json"),
        "--audit", str(tmp_path / "audit.json"),
        "--codex-model", "gpt-5.6-sol",
        "--codex-reasoning-effort", "xhigh",
        "--codex-timeout-seconds", "600",
        "--freeze-file", str(tmp_path / "freeze.json"),
        "--case-id", "case-one",
        "--expected-total", "160",
        "--expected-private", "80",
        "--expected-group", "80",
    ])

    run_judge_main()

    assert captured["args"][-3:] == ("gpt-5.6-sol", "xhigh", 600)
    assert captured["kwargs"] == {
        "freeze_path": tmp_path / "freeze.json",
        "case_ids": ["case-one"],
        "expected_total": 160,
        "expected_private": 80,
        "expected_group": 80,
    }


def test_balanced160_dataset_is_exactly_80_private_and_80_group():
    blind = json.loads((BALANCED160_DIR / "dev_blind.json").read_text(encoding="utf-8"))

    assert _validate_dataset_shape(
        blind,
        expected_total=160,
        expected_private=80,
        expected_group=80,
    ) == {"total": 160, "private": 80, "group": 80}


def test_dataset_shape_gate_rejects_old_50_case_set():
    blind = {"cases": [
        {"relationship": "private"} for _ in range(25)
    ] + [
        {"relationship": "group"} for _ in range(25)
    ]}

    with pytest.raises(ValueError, match=r"total=50 .*private=25 .*group=25"):
        _validate_dataset_shape(
            blind,
            expected_total=160,
            expected_private=80,
            expected_group=80,
        )


def test_prompt_is_single_case_toolless_and_has_no_answer():
    messages = build_messages(_case(), {"facts": [], "examples": []}, BASELINE_PROFILE)
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "case-one" in serialized
    assert "不得调用工具" in serialized
    assert '"source_case_id"' not in serialized
    assert '"human_option": "A"' not in serialized
    assert '"human_option": "B"' not in serialized


def test_explicit_thread_candidate_requires_symmetric_thread_and_counterevidence():
    profile = {
        **BASELINE_PROFILE,
        "response_schema": "explicit_thread_attribution_v1",
    }
    messages = build_messages(_case(), {"facts": [], "examples": []}, profile)
    prompt = messages[0]["content"]
    assert '"option_a_thread"' in prompt
    assert '"option_a_counterevidence"' in prompt
    assert '"option_b_thread"' in prompt
    assert '"option_b_counterevidence"' in prompt


def test_evidence_gate_candidate_requires_person_specific_evidence_and_uncertainty():
    profile = {
        **BASELINE_PROFILE,
        "response_schema": "person_specific_evidence_gate_v1",
    }
    prompt = build_messages(_case(), {"facts": [], "examples": []}, profile)[0]["content"]
    assert '"option_a_person_specific_evidence"' in prompt
    assert '"option_b_person_specific_evidence"' in prompt
    assert '"decisive_evidence"' in prompt
    assert '"uncertainty"' in prompt


def test_content_anchored_candidate_requires_verbatim_selected_reply():
    profile = {
        **BASELINE_PROFILE,
        "response_schema": "content_anchored_evidence_gate_v2",
    }
    prompt = build_messages(_case(), {"facts": [], "examples": []}, profile)[0]["content"]
    assert '"selected_reply"' in prompt


def test_pattern_cards_are_loaded_only_for_matching_chat_type():
    profile = {
        **BASELINE_PROFILE,
        "pattern_cards": [
            {"id": "group-only", "relationships": ["group"]},
            {"id": "private-only", "relationships": ["private"]},
        ],
    }
    group_case = {**_case(), "relationship": "group"}
    prompt = build_messages(group_case, {"facts": [], "examples": []}, profile)[0]["content"]
    assert "group-only" in prompt
    assert "private-only" not in prompt


def test_scenario_few_shots_are_complete_labeled_and_relationship_scoped():
    profile = {
        **BASELINE_PROFILE,
        "decision_priority": ["P1 先核对事实", "P2 再看表达"],
    }
    reference = {
        "facts": [],
        "examples": [],
        "scenario_few_shots": [
            {
                "id": "group-shot",
                "relationships": ["group"],
                "context": ["完整群聊上下文"],
                "option_A": ["真人回复"],
                "option_B": ["AI 回复"],
                "human_option": "A",
                "reason": "区分理由",
                "boundary": "适用边界",
            },
            {
                "id": "private-shot",
                "relationships": ["private"],
                "context": ["完整私聊上下文"],
                "option_A": ["AI 回复"],
                "option_B": ["真人回复"],
                "human_option": "B",
                "reason": "另一个理由",
                "boundary": "另一个边界",
            },
        ],
    }
    group_case = {**_case(), "relationship": "group"}
    prompt = build_messages(group_case, reference, profile)[0]["content"]
    assert "P1 先核对事实" in prompt
    assert "完整群聊上下文" in prompt
    assert "真人回复" in prompt
    assert '"human_option": "A"' in prompt
    assert "区分理由" in prompt
    assert "适用边界" in prompt
    assert "private-shot" not in prompt


def test_decision_gates_are_included_in_reference_pack():
    reference = {
        "facts": [],
        "examples": [],
        "decision_gates": [
            {"name": "obvious_restatement_veto", "rule": "重复显然信息必须扣分"}
        ],
    }
    prompt = build_messages(_case(), reference, BASELINE_PROFILE)[0]["content"]
    assert '"decision_gates"' in prompt
    assert "obvious_restatement_veto" in prompt
    assert "重复显然信息必须扣分" in prompt


def test_selected_reply_maps_content_and_rejects_label_mismatch():
    case = _case()
    assert resolve_selected_reply(case, {
        "selected_reply": ["在"], "human_option": "A",
    }) == ("A", None)
    assert resolve_selected_reply(case, {
        "selected_reply": ["在"], "human_option": "B",
    }) == (None, "selected_reply_label_mismatch")
    assert resolve_selected_reply(case, {
        "human_option": "A",
    }) == (None, "selected_reply_missing")


def test_judge_parser_preserves_explicit_thread_trace_without_changing_legacy_shape():
    explicit = _parse_api_judge_response(json.dumps({
        "option_a_thread": "承接前一条",
        "option_a_counterevidence": "表达偏完整",
        "option_b_thread": "回到较早线程",
        "option_b_counterevidence": "跳转略突兀",
        "human_option": "B",
        "confidence": 0.61,
        "reason": "B 的跨线程承接更符合本人",
    }, ensure_ascii=False))
    assert explicit["option_b_thread"] == "回到较早线程"
    legacy = _parse_api_judge_response(
        '{"human_option":"A","confidence":0.5,"reason":"证据接近"}'
    )
    assert set(legacy) == {"human_option", "confidence", "reason"}


def test_judge_parser_preserves_person_specific_evidence_gate_trace():
    parsed = _parse_api_judge_response(json.dumps({
        "option_a_person_specific_evidence": "无",
        "option_b_person_specific_evidence": "本人常用称呼",
        "decisive_evidence": "B的称呼习惯",
        "uncertainty": "两边都能承接上下文",
        "human_option": "B",
        "confidence": 0.64,
        "reason": "只有B提供弱的本人特异证据",
    }, ensure_ascii=False))
    assert parsed["option_b_person_specific_evidence"] == "本人常用称呼"
    assert parsed["decisive_evidence"] == "B的称呼习惯"
    assert parsed["uncertainty"] == "两边都能承接上下文"


def test_judge_parser_preserves_selected_reply_as_message_list():
    parsed = _parse_api_judge_response(json.dumps({
        "selected_reply": ["第一条", "第二条"],
        "human_option": "A",
        "confidence": 0.6,
        "reason": "证据较弱",
    }, ensure_ascii=False))
    assert parsed["selected_reply"] == ["第一条", "第二条"]


def test_swap_case_only_swaps_options():
    case = _case()
    swapped = swap_case(case)
    assert swapped["option_A"] == case["option_B"]
    assert swapped["option_B"] == case["option_A"]
    assert swapped["context_original"] == case["context_original"]


def test_request_audit_rejects_tools_other_cases_and_answers():
    clean = {"messages": [{"content": "case-one"}]}
    validate_request_body(clean, "case-one", ["case-two"])
    with pytest.raises(ValueError, match="tools"):
        validate_request_body({**clean, "tools": [{}]}, "case-one", [])
    with pytest.raises(ValueError, match="another"):
        validate_request_body({"messages": [{"content": "case-one case-two"}]}, "case-one", ["case-two"])
    with pytest.raises(ValueError, match="answer metadata"):
        validate_request_body({"messages": [{"content": 'case-one {"actual":"A"}'}]}, "case-one", [])


def test_final_claim_is_resumable_but_rejects_new_design(tmp_path):
    claim = tmp_path / "claim.json"
    output = tmp_path / "predictions.json"
    first = claim_final_run(
        claim,
        freeze_sha256="benchmark",
        prompt_sha256="prompt-one",
        model="model",
        presentation="canonical",
        output_path=output,
    )
    second = claim_final_run(
        claim,
        freeze_sha256="benchmark",
        prompt_sha256="prompt-one",
        model="model",
        presentation="canonical",
        output_path=output,
    )
    assert first == second
    with pytest.raises(ValueError, match="already claimed"):
        claim_final_run(
            claim,
            freeze_sha256="benchmark",
            prompt_sha256="prompt-two",
            model="model",
            presentation="canonical",
            output_path=output,
        )


def test_nonfinal_run_resumes_only_with_matching_identity(tmp_path):
    output = tmp_path / "predictions.json"
    audit = tmp_path / "audit.json"
    identity = {"split": "dev", "profile_sha256": "profile", "presentation": "canonical"}
    output.write_text(json.dumps({**identity, "invocation_id": "same", "cases": []}), encoding="utf-8")
    audit.write_text(json.dumps({"invocation_id": "same", "calls": []}), encoding="utf-8")
    loaded = load_resumable_run(output, audit, identity)
    assert loaded is not None
    assert loaded[0]["invocation_id"] == "same"

    with pytest.raises(ValueError, match="profile_sha256"):
        load_resumable_run(output, audit, {**identity, "profile_sha256": "other"})


def test_frozen_artifacts_are_read_from_the_freeze_directory(tmp_path):
    blind = tmp_path / "dev_blind.json"
    answers = tmp_path / "dev_answers.json"
    blind.write_text("blind", encoding="utf-8")
    answers.write_text("answers", encoding="utf-8")
    from scripts.build_human_ai_judge_benchmark_v2 import stable_file_sha256

    freeze = {"artifact_sha256": {"dev": {
        "blind": stable_file_sha256(blind),
        "answers": stable_file_sha256(answers),
    }}}
    assert verify_frozen_artifacts("dev", freeze, tmp_path) == (blind, answers)


def test_profile_hash_changes_with_one_hypothesis():
    changed = dict(BASELINE_PROFILE)
    changed["instructions"] += "新增一条规则"
    assert profile_sha256(changed) != profile_sha256(BASELINE_PROFILE)


def test_v3_profile_separates_authorship_from_reply_quality_and_calibrates_confidence():
    profile = json.loads(V3_PROFILE_PATH.read_text(encoding="utf-8"))
    messages = build_messages(_case(), {"facts": [], "examples": []}, profile)
    prompt = messages[0]["content"]
    assert "本人特异证据" in prompt
    assert "通用好回复证据" in prompt
    assert "不得超过 0.75" in prompt
    assert "跨线程解释" in prompt
    assert '"human_option":"A或B"' in prompt


def test_boundary_candidate_treats_reply_shape_as_neutral_and_caps_ambiguous_confidence():
    profile = json.loads(BOUNDARY_PROFILE_PATH.read_text(encoding="utf-8"))
    messages = build_messages(_case(), {"facts": [], "examples": []}, profile)
    prompt = messages[0]["content"]
    assert "全部是中性特征" in prompt
    assert "更完整、更理想" in prompt
    assert "更短、更碎、更俏皮" in prompt
    assert "承接最后一句、较早线程" in prompt
    assert "confidence 不得超过 0.62" in prompt
    assert "不得为了支持选择而编造区别" in prompt


def test_compare_presentations_uses_canonicalized_choices(tmp_path):
    shared = {
        "split": "dev",
        "blind_sha256": "blind",
        "reference_sha256": "reference",
        "profile_sha256": "profile",
        "model": "model",
    }
    canonical = {**shared, "presentation": "canonical", "cases": [
        {"case_id": "one", "human_option": "A"},
        {"case_id": "two", "human_option": "B"},
    ]}
    swapped = {**shared, "presentation": "swapped", "cases": [
        {"case_id": "one", "human_option": "A"},
        {"case_id": "two", "human_option": "A"},
    ]}
    canonical_path = tmp_path / "canonical.json"
    swapped_path = tmp_path / "swapped.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    swapped_path.write_text(json.dumps(swapped), encoding="utf-8")
    result = compare_presentations(canonical_path, swapped_path)
    assert result == {"status": "completed", "compared": 2, "agreed": 1, "agreement": 0.5}
