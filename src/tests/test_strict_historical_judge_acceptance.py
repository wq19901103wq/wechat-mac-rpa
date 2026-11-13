import json
from datetime import datetime

import pytest

from scripts.run_strict_historical_judge_acceptance import (
    _candidate_event_id,
    _direct_reference_overlap,
    _discard_cases_from_id,
    _generation_client,
    _hidden_stratum_quotas,
    _length_bucket_quotas,
    _pending_generation_cases,
    _present_sealed,
    _scorable_counts,
    _scorable_length_counts,
    _validate_judge_request,
    judge_hidden,
    render_html,
    seal_fair_replacements,
    seal_hidden,
    summarize_position_swap,
    generate_case,
)
from scripts.run_human_ai_judge_benchmark_v2 import BASELINE_PROFILE, build_messages


def _generated(case_id: str, human: str, ai: str, group: str = "other_group") -> dict:
    return {
        "case_id": case_id,
        "event_id": "event-" + case_id,
        "group": group,
        "chat_type": "group",
        "context": [{"content": "前文", "role": "other", "speaker": "甲", "time": None}],
        "human_reply": [human],
        "c0_reply": [ai],
        "validity": {"formally_scorable": True},
    }


def _candidate(case_id: str, group: str, event: str | None = None) -> dict:
    return {
        "internal_id": case_id,
        "event_cluster": event or "event-" + case_id,
        "source_path": "source.json",
        "timestamp": 1,
        "reply": "真人-" + case_id,
        "context": ["前文-" + case_id],
        "context_messages": [{"content": "前文-" + case_id}],
        "specified_group": group if group in {"correct_common_prosperity_with_luxixi", "youzi_group"} else None,
        "complex_large_group": group == "other_complex_large_group",
        "chat_type": "private" if group == "private" else "group",
    }


def test_generate_case_allows_human_text_that_already_appeared_in_visible_history(monkeypatch):
    source = {
        **_candidate("repeated", "other_group"),
        "reply": "前文-repeated",
        "timestamp": 1,
    }

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.last_tool_calls = []
            self.last_generation_failed = False
            self.last_generation_trace = []
            self.last_llm_calls = []
            self.last_llm_messages = []
            self.last_raw_response = '{"replies":["前文-repeated"]}'
            self.last_thinking = ""
            self.last_fact_check = {}
            self.last_loaded_skills = []
            self.last_skill_injected_content = ""
            self.last_self_refine_applied = False
            self.last_feedback_decision = ""
            self.last_feedback_issues = []
            self.last_feedback_raw = ""
            self.last_iterate_count = 0
            self.last_iterate_raw = ""
            self.last_iterate_skipped_no_budget = False
            self.last_few_shot_content = ""
            self.last_few_shot_ids = []
            self.last_tools_context = ""
            self.last_system_prompt = "系统"
            prompt_time = datetime.fromtimestamp(source["timestamp"]).strftime("%Y-%m-%d %H:%M")
            self.last_user_prompt = f"<current_time>{prompt_time}</current_time>前文-repeated"

        def generate(self, *args, **kwargs):
            return ["前文-repeated"]

    monkeypatch.setattr(
        "scripts.run_strict_historical_judge_acceptance.strict_build_inputs",
        lambda row: (
            [],
            [],
            {},
            {"chat_name": "测试群"},
        ),
    )
    monkeypatch.setattr(
        "scripts.run_strict_historical_judge_acceptance._case_context",
        lambda messages, unread: [{"content": "前文-repeated"}],
    )
    result = generate_case(source, type("LLM", (), {"model": "fake", "base_url": ""})(), None, 10, FakeGenerator)
    assert result["validity"]["answer_in_visible_context"] is True
    assert result["validity"]["answer_in_c0_trace"] is False
    assert result["validity"]["c0_reproduced_human"] is True
    assert result["validity"]["formally_scorable"] is True


def test_generate_case_keeps_clock_and_side_effect_findings_as_diagnostics(monkeypatch):
    source = {
        **_candidate("diagnostics", "other_group"),
        "reply": "真人回复",
        "timestamp": 1,
    }

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.last_tool_calls = [{"tool_name": "send_file"}]
            self.last_generation_failed = False
            self.last_generation_trace = []
            self.last_llm_calls = []
            self.last_llm_messages = []
            self.last_raw_response = '{"replies":["候选回复"]}'
            self.last_thinking = ""
            self.last_fact_check = {}
            self.last_loaded_skills = []
            self.last_skill_injected_content = ""
            self.last_self_refine_applied = False
            self.last_feedback_decision = ""
            self.last_feedback_issues = []
            self.last_feedback_raw = ""
            self.last_iterate_count = 0
            self.last_iterate_raw = ""
            self.last_iterate_skipped_no_budget = False
            self.last_few_shot_content = ""
            self.last_few_shot_ids = []
            self.last_tools_context = ""
            self.last_system_prompt = "系统"
            self.last_user_prompt = "没有历史时间标签"

        def generate(self, *args, **kwargs):
            return ["候选回复"]

    monkeypatch.setattr(
        "scripts.run_strict_historical_judge_acceptance.strict_build_inputs",
        lambda row: ([], [], {}, {"chat_name": "测试群"}),
    )
    monkeypatch.setattr(
        "scripts.run_strict_historical_judge_acceptance._case_context",
        lambda messages, unread: [{"content": "前文"}],
    )

    result = generate_case(
        source,
        type("LLM", (), {"model": "fake", "base_url": ""})(),
        None,
        10,
        FakeGenerator,
    )

    assert result["validity"]["formally_scorable"] is True
    assert result["validity"]["exclusions"] == []
    assert result["validity"]["historical_prompt_clock_ok"] is False
    assert result["validity"]["side_effect_tool_calls"] == [{"tool_name": "send_file"}]


def test_direct_reference_overlap_requires_same_context_and_reply():
    case = _generated("one", "真人回答", "AI 回答")
    reference = {"examples": [{"context": ["前文"], "human_reply": ["真人回答"]}]}
    assert _direct_reference_overlap(case, reference)
    assert not _direct_reference_overlap(case, {"examples": [{"context": ["别的前文"], "human_reply": ["真人回答"]}]})


def test_judge_request_contains_only_one_sealed_blind_case():
    blind = {
        "case_id": "case-one",
        "chat_type": "group",
        "chat_name": "匿名群聊",
        "context": [],
        "option_A": ["甲"],
        "option_B": ["乙"],
    }
    messages = build_messages(blind, {"facts": [], "examples": []}, BASELINE_PROFILE)
    _validate_judge_request(messages, blind, ["case-two"])
    polluted = json.loads(json.dumps(messages, ensure_ascii=False))
    polluted[1]["content"] += "case-two"
    with pytest.raises(RuntimeError, match="differs|another"):
        _validate_judge_request(polluted, blind, ["case-two"])


def test_swapped_presentation_only_exchanges_options_and_answer():
    sealed = {
        "blind_cases": [{
            "case_id": "one",
            "chat_type": "group",
            "chat_name": "匿名群聊",
            "context": [{"speaker": "甲", "content": "前文"}],
            "option_A": ["真人"],
            "option_B": ["机器"],
        }],
        "answers": [{"case_id": "one", "human_option": "A"}],
        "unchanged": {"marker": True},
    }
    canonical = _present_sealed(sealed, "canonical")
    swapped = _present_sealed(sealed, "swapped")
    assert canonical is sealed
    assert swapped["blind_cases"][0]["option_A"] == ["机器"]
    assert swapped["blind_cases"][0]["option_B"] == ["真人"]
    assert swapped["answers"][0]["human_option"] == "B"
    assert swapped["blind_cases"][0]["context"] == sealed["blind_cases"][0]["context"]
    assert swapped["unchanged"] == sealed["unchanged"]
    assert sealed["blind_cases"][0]["option_A"] == ["真人"]
    assert sealed["answers"][0]["human_option"] == "A"


def test_judge_resume_does_not_call_completed_case(monkeypatch, tmp_path):
    profile = tmp_path / "profile.json"
    reference = tmp_path / "reference.json"
    profile.write_text(json.dumps(BASELINE_PROFILE, ensure_ascii=False), encoding="utf-8")
    reference.write_text('{"facts": [], "examples": []}', encoding="utf-8")
    sealed = {
        "blind_cases": [
            {"case_id": case_id, "chat_type": "group", "chat_name": "匿名群聊", "context": [], "option_A": ["真人"], "option_B": ["机器"]}
            for case_id in ("done", "pending")
        ],
        "answers": [
            {"case_id": case_id, "human_option": "A"}
            for case_id in ("done", "pending")
        ],
    }
    profile_sha = __import__("hashlib").sha256(profile.read_bytes()).hexdigest()
    reference_sha = __import__("hashlib").sha256(reference.read_bytes()).hexdigest()
    (tmp_path / "judge_results.json").write_text(json.dumps({
        "profile_sha256": profile_sha,
        "reference_sha256": reference_sha,
        "presentation": "canonical",
        "cases": [{"case_id": "done", "correct": True}],
    }), encoding="utf-8")
    calls = []

    def fake_run(messages):
        calls.append(messages)
        return ('{"human_option":"A","confidence":0.5,"reason":"判断"}', {}, 1, [])

    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance._run_codex", fake_run)
    judged = judge_hidden(sealed, profile, reference, tmp_path, resume=True)
    assert [row["case_id"] for row in judged["cases"]] == ["done", "pending"]
    assert len(calls) == 1


def test_judge_rejects_reference_leak_before_model_call(monkeypatch, tmp_path):
    profile = tmp_path / "profile.json"
    reference = tmp_path / "reference.json"
    profile.write_text(json.dumps(BASELINE_PROFILE, ensure_ascii=False), encoding="utf-8")
    reference.write_text(json.dumps({
        "facts": [],
        "examples": [],
        "scenario_few_shots": [{
            "relationships": ["group"],
            "context": ["独立上下文"],
            "option_A": ["这是真人完整回复而且已经泄漏"],
        }],
    }, ensure_ascii=False), encoding="utf-8")
    sealed = {
        "blind_cases": [{
            "case_id": "leaked", "chat_type": "group", "chat_name": "匿名群聊",
            "context": [], "option_A": ["这是真人完整回复而且已经泄漏"], "option_B": ["候选"],
        }],
        "answers": [{"case_id": "leaked", "human_option": "A"}],
    }
    calls = []
    monkeypatch.setattr(
        "scripts.run_strict_historical_judge_acceptance._run_codex",
        lambda messages: calls.append(messages),
    )
    with pytest.raises(ValueError, match="reference isolation failed"):
        judge_hidden(sealed, profile, reference, tmp_path)
    assert calls == []


def test_sealed_blind_options_remain_position_balanced(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)
    cases = []
    for group, quota in _hidden_stratum_quotas(10).items():
        cases.extend(
            _generated(f"{group}-{i}", f"人{group}-{i}", f"机{group}-{i}", group)
            for i in range(quota)
        )
    sealed = seal_hidden({"cases": cases}, 10)
    assert sum(row["human_option"] == "A" for row in sealed["answers"]) == 5


def test_group_balanced_selection_excludes_tail_media():
    groups = (
        "correct_common_prosperity_with_luxixi",
        "youzi_group",
        "other_complex_large_group",
    )
    cases = []
    for group in groups:
        for index in range(3):
            case = _generated(
                f"{group}-{index}",
                f"人{group}-{index}",
                f"机{group}-{index}",
                group,
            )
            cases.append(case)
        media_case = _generated(
            f"{group}-media",
            f"人{group}-media",
            f"机{group}-media",
            group,
        )
        media_case["context"][-1]["content"] = "[图片]"
        cases.append(media_case)

    sealed = seal_hidden(
        {"cases": cases},
        6,
        selection_mode="group-balanced",
    )

    assert sealed["selected_by_stratum"] == {
        "correct_common_prosperity_with_luxixi": 2,
        "youzi_group": 2,
        "other_complex_large_group": 2,
        "other_group": 0,
        "private": 0,
    }
    assert sealed["rules"]["group_chat_balanced"] is True
    assert sealed["rules"]["unresolved_tail_media_excluded"] is True
    assert sealed["unresolved_tail_media_excluded_count"] == 3
    assert not any(row["case_id"].endswith("-media") for row in sealed["answers"])
    for blind, answer in zip(sealed["blind_cases"], sealed["answers"], strict=True):
        assert blind[f"option_{answer['human_option']}"] == [f"人{answer['case_id']}"]


def test_hidden_100_uses_fixed_stage2_proportional_strata(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)
    expected = {
        "correct_common_prosperity_with_luxixi": 9,
        "youzi_group": 8,
        "other_complex_large_group": 26,
        "other_group": 24,
        "private": 33,
    }
    cases = []
    for group, quota in expected.items():
        cases.extend(
            _generated(f"{group}-{i}", f"人{group}-{i}", f"机{group}-{i}", group)
            for i in range(quota + 2)
        )
    sealed = seal_hidden({"cases": cases}, 100)
    assert sealed["stratum_quotas"] == expected
    assert sealed["selected_by_stratum"] == expected
    assert len(sealed["blind_cases"]) == 100
    assert sum(row["human_option"] == "A" for row in sealed["answers"]) == 50


def test_hidden_stratum_shortage_fails_instead_of_reallocating(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)
    cases = []
    for group, quota in _hidden_stratum_quotas(10).items():
        count = quota - 1 if group == "private" else quota + 1
        cases.extend(
            _generated(f"{group}-{i}", f"人{group}-{i}", f"机{group}-{i}", group)
            for i in range(count)
        )
    with pytest.raises(RuntimeError, match="hidden stratum capacity shortage"):
        seal_hidden({"cases": cases}, 10)


def test_length_balanced_hidden_set_excludes_old_cases_and_records_strata(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)
    cases = []
    for group, quota in _hidden_stratum_quotas(100).items():
        for index in range(quota + 2):
            cases.append(_generated(f"short-{group}-{index}", "短", "这是明显更长的机器回复", group))
            cases.append(_generated(f"equal-{group}-{index}", "差不多长", "机器长度", group))
            cases.append(_generated(f"long-{group}-{index}", "这是一条明显更长的真人回复", "短", group))
    excluded = {
        "short-correct_common_prosperity_with_luxixi-0",
        "equal-correct_common_prosperity_with_luxixi-0",
        "long-correct_common_prosperity_with_luxixi-0",
    }
    sealed = seal_hidden(
        {"cases": cases}, 100, excluded_case_ids=excluded, length_balanced=True
    )
    selected_ids = {row["case_id"] for row in sealed["answers"]}
    assert not selected_ids & excluded
    assert sealed["length_bucket_quotas"] == _length_bucket_quotas(100)
    assert sealed["selected_by_length_bucket"] == _length_bucket_quotas(100)
    assert sum(sealed["selected_by_stratum"].values()) == 100
    assert sealed["stratum_quotas"] == _hidden_stratum_quotas(100)
    assert sealed["selected_by_stratum"] == _hidden_stratum_quotas(100)
    assert sealed["rules"]["stage2_proportional_stratified_selection"] is True
    assert sum(row["human_option"] == "A" for row in sealed["answers"]) == 50


def test_length_balanced_hidden_set_fails_when_bucket_lacks_capacity(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)
    cases = [
        _generated(f"short-{index}", "短", "这是明显更长的机器回复")
        for index in range(12)
    ]
    with pytest.raises(RuntimeError, match="length-bucket capacity shortage"):
        seal_hidden({"cases": cases}, 9, length_balanced=True)


def test_exploratory_unseen_excludes_tail_media_and_records_non_freeze_scope(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)
    cases = []
    for index in range(12):
        cases.append(_generated(f"short-{index}", "短", "这是明显更长的机器回复", "private"))
    for index in range(3):
        cases.append(_generated(f"equal-{index}", "差不多长", "机器长度", "other_group"))
    cases.append(_generated("long", "这是一条明显更长的真人回复", "短", "private"))
    media = _generated("media", "短", "这是明显更长的机器回复", "private")
    media["context"].append({"content": "[图片]", "role": "other", "speaker": "甲", "time": None})
    cases.append(media)

    sealed = seal_hidden(
        {"cases": cases},
        10,
        excluded_case_ids={"short-0"},
        selection_mode="exploratory-unseen",
    )

    selected = {row["case_id"] for row in sealed["answers"]}
    assert "media" not in selected
    assert "short-0" not in selected
    assert sealed["selected_by_length_bucket"] == {
        "human_shorter": 6,
        "near_equal": 3,
        "human_longer": 1,
    }
    assert sealed["selected_by_stratum"] == {
        "correct_common_prosperity_with_luxixi": 0,
        "youzi_group": 0,
        "other_complex_large_group": 0,
        "other_group": 3,
        "private": 7,
    }
    assert sealed["rules"]["independent_unseen"] is True
    assert sealed["rules"]["private_heavy"] is True
    assert sealed["rules"]["not_freeze_eligible"] is True
    assert sealed["rules"]["stage2_proportional_stratified_selection"] is False
    assert sealed["unresolved_tail_media_excluded_count"] == 1
    assert len(sealed["selected_case_ids_sha256"]) == 64
    assert sum(row["human_option"] == "A" for row in sealed["answers"]) == 5


def test_resume_skips_completed_case_and_event():
    candidates = [
        _candidate("done", "other_group", "same-event"),
        _candidate("same-event-new-id", "other_group", "same-event"),
        _candidate("pending", "other_group"),
    ]
    completed = [_generated("done", "真人-done", "机器", "other_group")]
    pending = _pending_generation_cases(
        candidates, completed, {group: 2 for group in _hidden_stratum_quotas(100)}, {"examples": []}
    )
    assert [row["internal_id"] for row in pending] == ["pending"]


def test_pending_generation_skips_previously_exposed_case():
    candidates = [
        _candidate("old", "other_group"),
        _candidate("new", "other_group"),
    ]
    pending = _pending_generation_cases(
        candidates,
        [],
        None,
        {"examples": []},
        {"old"},
    )
    assert [case["internal_id"] for case in pending] == ["new"]


def test_discard_cases_from_id_removes_polluted_case_and_tail():
    cases = [{"case_id": "ok-1"}, {"case_id": "bad"}, {"case_id": "bad-tail"}]
    assert _discard_cases_from_id(cases, "bad") == [{"case_id": "ok-1"}]
    with pytest.raises(RuntimeError, match="not found"):
        _discard_cases_from_id(cases, "missing")


def test_volcengine_generation_client_keeps_dsv4f_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.run_strict_historical_judge_acceptance.load_volcengine_credentials",
        lambda path: ("key", "https://volc.example"),
    )
    client = _generation_client("volcengine", tmp_path / "config.toml")
    assert client.model == "deepseek-v4-flash"
    assert client.base_url == "https://volc.example"


def test_full_stratum_is_not_generated_again():
    completed = [_generated("done", "真人-done", "机器", "other_group")]
    quotas = {group: 2 for group in _hidden_stratum_quotas(100)}
    quotas["other_group"] = 1
    pending = _pending_generation_cases(
        [_candidate("group-new", "other_group"), _candidate("private-new", "private")],
        completed,
        quotas,
        {"examples": []},
    )
    assert [row["internal_id"] for row in pending] == ["private-new"]


def test_scorable_counts_excludes_reference_overlap():
    overlapping = _generated("overlap", "真人回答", "机器", "other_group")
    clean = _generated("clean", "另一个真人回答", "机器", "other_group")
    reference = {"examples": [{"context": ["前文"], "human_reply": ["真人回答"]}]}
    assert _scorable_counts([overlapping, clean], reference)["other_group"] == 1


def test_all_full_strata_have_no_pending_cases():
    quotas = {group: 1 for group in _hidden_stratum_quotas(100)}
    completed = [
        _generated(group, "真人-" + group, "机器", group)
        for group in quotas
    ]
    candidates = [_candidate("new-" + group, group) for group in quotas]
    assert _pending_generation_cases(candidates, completed, quotas, {"examples": []}) == []


def test_length_counts_exclude_previously_exposed_cases():
    cases = [
        _generated("old-short", "短", "这是明显更长的机器回复"),
        _generated("new-equal", "差不多长", "机器长度"),
        _generated("new-long", "这是一条明显更长的真人回复", "短"),
    ]
    assert _scorable_length_counts(cases, {"examples": []}, {"old-short"}) == {
        "human_shorter": 0,
        "near_equal": 1,
        "human_longer": 1,
    }


def test_length_generation_pending_ignores_old_stratum_fullness():
    completed = [_generated("done", "真人-done", "机器", "other_group")]
    pending = _pending_generation_cases(
        [_candidate("done", "other_group"), _candidate("new", "other_group")],
        completed,
        None,
        {"examples": []},
    )
    assert [row["internal_id"] for row in pending] == ["new"]


def test_render_html_exposes_independent_mobile_filters():
    case = _generated("one", "短", "这是一条更长的机器回复")
    case["chat_name"] = "测试群"
    case["generation"] = {
        "metadata": {"generator_history_count": 3, "event_context_count": 1}
    }
    judged = {
        "cases": [{
            "case_id": "one",
            "valid": True,
            "correct": False,
            "confidence": 0.9,
            "selected_option": "B",
            "human_option": "A",
            "reason": "更完整",
            "error_category": "length_or_completeness",
            "blind_case": {"option_A": ["短"], "option_B": ["这是一条更长的机器回复"]},
        }]
    }
    rendered = render_html({"cases": [case]}, judged)
    assert 'id="result-filter"' in rendered
    assert 'id="context-filter"' in rendered
    assert 'id="group-filter"' in rendered
    assert 'id="length-filter"' in rendered
    assert 'id="cause-filter"' in rendered
    assert 'data-high-confidence-wrong="true"' in rendered
    assert 'data-cause="context_scope_mismatch"' in rendered
    assert "本轮验收无效" in rendered


def test_render_html_uses_explicit_judge_history_count_for_alignment():
    case = _generated("aligned", "真人", "机器")
    case["chat_name"] = "测试群"
    case["generation"] = {
        "metadata": {
            "generator_history_count": 1000,
            "event_context_count": 1,
            "judge_history_count": 1000,
        }
    }
    judged = {
        "cases": [{
            "case_id": "aligned",
            "valid": True,
            "correct": True,
            "confidence": 0.7,
            "selected_option": "A",
            "human_option": "A",
            "reason": "判断",
            "blind_case": {"option_A": ["真人"], "option_B": ["机器"]},
        }]
    }
    rendered = render_html({"cases": [case]}, judged)
    assert "上下文范围：C0 1000 条 / Judge 1000 条 · 一致" in rendered
    assert 'data-context="equal"' in rendered
    assert "公平得分（仅验收 Judge）" in rendered
    assert "本轮验收无效" not in rendered


def test_render_html_exposes_manual_misjudgment_review_filter():
    case = _generated("bc38ee57b06a9b4358b5", "中大是什么", "我还以为真只能吃中饭 吓一跳")
    case["chat_name"] = "测试群"
    case["generation"] = {"metadata": {"generator_history_count": 2, "judge_history_count": 2}}
    judged = {"cases": [{
        "case_id": case["case_id"],
        "valid": True,
        "correct": False,
        "confidence": 0.84,
        "selected_option": "B",
        "human_option": "A",
        "reason": "判断",
        "blind_case": {"option_A": case["human_reply"], "option_B": case["c0_reply"]},
    }]}
    rendered = render_html({"cases": [case]}, judged)
    assert 'id="review-filter"' in rendered
    assert 'data-review="intrinsically_hard"' in rendered
    assert "题目本身难分" in rendered
    assert "5 个题目本身难分、12 个判断逻辑明显错误" in rendered


def test_position_swap_summary_compares_selected_reply_identity():
    judged = {"cases": [
        {"case_id": "stable-human", "valid": True, "correct": True},
        {"case_id": "stable-c0", "valid": True, "correct": False},
        {"case_id": "changed", "valid": True, "correct": True},
    ]}
    swapped = {"cases": [
        {"case_id": "stable-human", "valid": True, "correct": True},
        {"case_id": "stable-c0", "valid": True, "correct": False},
        {"case_id": "changed", "valid": True, "correct": False},
    ]}
    assert summarize_position_swap(judged, swapped) == {
        "total": 3,
        "swapped_correct": 1,
        "identity_consistent": 2,
        "correct_to_wrong": 1,
        "wrong_to_correct": 0,
        "both_wrong": 1,
    }


def test_fair_replacements_exclude_old_and_unfair_and_restore_balances(monkeypatch, tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"examples": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.run_strict_historical_judge_acceptance.REFERENCE", reference)

    old_cases = []
    old_answers = []
    old_blind = []
    fair_groups = (
        ["correct_common_prosperity_with_luxixi"] * 2
        + ["youzi_group"]
        + ["other_complex_large_group"] * 22
        + ["other_group"] * 21
        + ["private"] * 36
    )
    unfair_groups = ["other_group"] * 9 + ["private"] * 9
    fair_buckets = ["human_shorter"] * 22 + ["near_equal"] * 29 + ["human_longer"] * 31
    unfair_buckets = ["human_shorter", "near_equal", "human_longer"] * 6
    for index in range(100):
        bucket = (fair_buckets + unfair_buckets)[index]
        human, ai = {
            "human_shorter": ("短", "这是明显更长的机器回复"),
            "near_equal": ("差不多长", "机器长度"),
            "human_longer": ("这是一条明显更长的真人回复", "短"),
        }[bucket]
        case = _generated(f"old-{index}", human, ai, (fair_groups + unfair_groups)[index])
        case["generation"] = {"metadata": {"generator_history_count": 10, "judge_history_count": 10 if index < 82 else 2}}
        old_cases.append(case)
        old_answers.append({"case_id": case["case_id"], "human_option": "A" if index < 37 or 82 <= index < 95 else "B"})
        old_blind.append({"case_id": case["case_id"], "chat_type": case["chat_type"], "chat_name": "匿名", "context": [], "option_A": [human], "option_B": [ai]})

    new_cases = []
    matrix = {"other_group": (6, 2, 1), "private": (6, 3, 0)}
    for group, counts in matrix.items():
        for bucket, count in zip(("human_shorter", "near_equal", "human_longer"), counts, strict=True):
            for index in range(count):
                human, ai = {
                    "human_shorter": ("短", "这是明显更长的机器回复"),
                    "near_equal": ("差不多长", "机器长度"),
                    "human_longer": ("这是一条明显更长的真人回复", "短"),
                }[bucket]
                case = _generated(f"new-{group}-{bucket}-{index}", human, ai, group)
                case["generation"] = {"metadata": {"generator_history_count": 20, "judge_history_count": 20}}
                new_cases.append(case)
    unfair_new = _generated("new-unfair", "短", "这是明显更长的机器回复", "other_group")
    unfair_new["generation"] = {"metadata": {"generator_history_count": 20, "judge_history_count": 1}}

    original = {
        "answers": old_answers,
        "blind_cases": old_blind,
        "selected_by_stratum": {
            "correct_common_prosperity_with_luxixi": 2,
            "youzi_group": 1,
            "other_complex_large_group": 22,
            "other_group": 30,
            "private": 45,
        },
        "selected_by_length_bucket": {
            "human_shorter": 34,
            "near_equal": 34,
            "human_longer": 32,
        },
    }
    replacement, combined = seal_fair_replacements({"cases": old_cases + [unfair_new] + new_cases}, original)
    replacement_ids = {row["case_id"] for row in replacement["answers"]}
    assert len(replacement_ids) == 18
    assert not replacement_ids & {row["case_id"] for row in old_answers}
    assert "new-unfair" not in replacement_ids
    assert combined["selected_by_stratum"] == {
        "correct_common_prosperity_with_luxixi": 2,
        "youzi_group": 1,
        "other_complex_large_group": 22,
        "other_group": 30,
        "private": 45,
    }
    assert combined["selected_by_length_bucket"] == {
        "human_shorter": 34,
        "near_equal": 34,
        "human_longer": 32,
    }
    assert sum(row["human_option"] == "A" for row in combined["answers"]) == 50
