from argparse import Namespace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.evaluate_human_ai_judge import _run_c0, assert_reference_isolated
from scripts.run_situation_c0_ab import (
    CURRENT_BASELINE_MODES,
    CodexExecClient,
    FeatureExtractionOnlyClient,
    PAIRED_COUNT_MODES,
    PairedCountRetriever,
    PairedCountSelection,
    PairedNamedContextRetriever,
    PairedNamedContextSelection,
    _entry_key,
    _case_ids_from_results,
    _generation_schedule,
    _generation_model_for_condition,
    _generation_timeout_for_condition,
    _feature_plan_model_for_condition,
    _has_nonempty_reply,
    _information_gap_response_move_for_condition,
    _few_shot_count_for_condition,
    _few_shot_path_for_condition,
    _latest_request_kind,
    _per_case_overlap_ids,
    _per_case_overlap_ids_from_pools,
    _reference_without_case_leaks,
    _drop_contaminated_generation_entries,
    _assert_generation_entries_isolated,
    _reuse_private_baseline_for_group_only_change,
    _group_instant_self_correction_contract_applies,
    _group_pattern_participation_contract_applies,
    _group_pending_self_action_contract_applies,
    _group_visible_self_stance_continuity_contract_applies,
    _group_current_event_self_role_continuity_contract_applies,
    _group_explicit_self_stance_action_contract_applies,
    _reuse_group_baseline_for_nontriggering_target_action_contract,
    _select_cases,
    _seed_judge_entries,
    _seed_scorable_entries,
    _seed_baseline_from_candidate_entries,
    _seed_baseline_from_candidate_judge_entries,
    _seed_split_baseline_entries,
    _seed_private_candidate_entries,
    _timeline_overlaps,
    _reuse_group_baseline_judge_for_private_only_candidate,
    _skill_snapshot_overrides,
)
from src.reply.few_shot import PersonaFewShotRetriever
from src.models.base import ChatMessage, SenderType


def test_offline_c0_accepts_group_unified_relevant_self_flag():
    assert "enable_group_unified_relevant_self" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_target_action_contract_flag():
    assert "enable_group_target_action_contract" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_public_question_specific_followup_flag():
    assert (
        "enable_group_public_question_specific_followup"
        in inspect.signature(_run_c0).parameters
    )


def test_group_public_question_specific_followup_is_paired_current_baseline_mode():
    assert "group-public-question-specific-followup" in CURRENT_BASELINE_MODES
    assert "group-public-question-specific-followup" in PAIRED_COUNT_MODES


def test_offline_c0_accepts_group_instant_self_correction_flag():
    assert "enable_group_instant_self_correction" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_visible_self_stance_continuity_flag():
    assert "enable_group_visible_self_stance_continuity" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_pending_self_action_flag():
    assert "enable_group_pending_self_action" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_pattern_participation_flag():
    assert "enable_group_pattern_participation" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_current_event_self_role_continuity_flag():
    assert "enable_group_current_event_self_role_continuity" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_explicit_self_stance_action_flag():
    assert "enable_group_explicit_self_stance_action" in inspect.signature(_run_c0).parameters


def test_group_explicit_self_stance_action_inherits_current_c0_and_paired_few_shots():
    assert "group-explicit-self-stance-action" in CURRENT_BASELINE_MODES
    assert "group-explicit-self-stance-action" in PAIRED_COUNT_MODES


def test_group_current_event_self_role_continuity_inherits_current_c0_and_paired_few_shots():
    assert "group-current-event-self-role-continuity" in CURRENT_BASELINE_MODES
    assert "group-current-event-self-role-continuity" in PAIRED_COUNT_MODES


def test_group_pattern_participation_inherits_current_c0_and_paired_few_shots():
    assert "group-pattern-participation" in CURRENT_BASELINE_MODES
    assert "group-pattern-participation" in PAIRED_COUNT_MODES


def test_offline_c0_accepts_group_answer_authority_rewrite_flag():
    assert "enable_group_answer_authority_rewrite" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_group_feature_plan_before_generation_flag():
    assert "enable_group_feature_plan_before_generation" in inspect.signature(_run_c0).parameters


def test_offline_c0_accepts_frozen_skill_content_override():
    assert "skill_content_overrides" in inspect.signature(_run_c0).parameters


def test_skill_snapshot_inherits_current_c0_and_reuses_private_baseline_result():
    assert "skill-snapshot" in CURRENT_BASELINE_MODES
    assert "skill-snapshot" in PAIRED_COUNT_MODES

    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "skill-snapshot",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_skill_snapshot_requires_distinct_paired_group_banter_files(tmp_path):
    baseline = tmp_path / "baseline.md"
    candidate = tmp_path / "candidate.md"
    baseline.write_text("old group rule", encoding="utf-8")
    candidate.write_text("new group rule", encoding="utf-8")
    args = Namespace(
        candidate_mode="skill-snapshot",
        baseline_skill_snapshot=[f"group_banter={baseline}"],
        candidate_skill_snapshot=[f"group_banter={candidate}"],
    )

    snapshots = _skill_snapshot_overrides(args)

    assert snapshots["baseline"]["group_banter"]["content"] == "old group rule"
    assert snapshots["candidate"]["group_banter"]["content"] == "new group rule"
    assert snapshots["baseline"]["group_banter"]["sha256"] != snapshots["candidate"]["group_banter"]["sha256"]

    args.candidate_skill_snapshot = [f"group_banter={baseline}"]
    with pytest.raises(ValueError, match="bodies must differ"):
        _skill_snapshot_overrides(args)


@pytest.mark.parametrize(
    ("candidate_mode", "baseline_specs", "candidate_specs", "message"),
    [
        ("skill-snapshot", ["not-an-assignment"], [], "NAME=PATH"),
        ("skill-snapshot", ["bad/name=x"], ["group_banter=x"], "invalid skill name"),
        ("skill-snapshot", ["group_banter=missing"], ["group_banter=missing"], "regular file"),
        ("clarification", [], ["group_banter=missing"], "require --candidate-mode"),
    ],
)
def test_skill_snapshot_rejects_invalid_or_wrong_mode_inputs(
    candidate_mode,
    baseline_specs,
    candidate_specs,
    message,
):
    args = Namespace(
        candidate_mode=candidate_mode,
        baseline_skill_snapshot=baseline_specs,
        candidate_skill_snapshot=candidate_specs,
    )

    with pytest.raises(ValueError, match=message):
        _skill_snapshot_overrides(args)


def test_group_feature_plan_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-feature-plan-before-generation",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_feature_extraction_model_changes_only_feature_call():
    experiment = Namespace(
        candidate_mode="group-feature-plan-before-generation",
        feature_extraction_model="doubao-seed-2.0-lite",
        generation_model="deepseek-v4-flash",
        generation_timeout_seconds=None,
        group_only_candidate=False,
    )
    config = {
        "private": {"model": "deepseek-v4-flash", "timeout_seconds": 20},
        "group": {"model": "doubao-seed-2.1-turbo", "timeout_seconds": 360},
    }

    assert _feature_plan_model_for_condition(experiment, "baseline", "group") is None
    assert _feature_plan_model_for_condition(experiment, "candidate", "private") is None
    assert _feature_plan_model_for_condition(experiment, "candidate", "group") == "doubao-seed-2.0-lite"
    assert _generation_model_for_condition(experiment, "candidate", "group", config) == "doubao-seed-2.1-turbo"


def test_feature_extraction_only_client_keeps_short_plan_on_current_c0():
    class Client:
        def __init__(self, label):
            self.label = label

        def chat(self, **_kwargs):
            return self.label

    client = FeatureExtractionOnlyClient(Client("lite"), Client("turbo"))

    assert client.chat(messages=[{"role": "user", "content": "分析群聊回复的事实边界。test"}]) == "lite"
    assert client.chat(messages=[{"role": "user", "content": "你是群聊回复前的短计划器。test"}]) == "turbo"


def test_generation_model_experiment_changes_candidate_model_only():
    experiment = Namespace(
        candidate_mode="generation-model",
        generation_model="doubao-seed-2.1-turbo",
        generation_timeout_seconds=180,
        group_only_candidate=True,
    )
    config = {
        "private": {"model": "deepseek-v4-flash", "timeout_seconds": 20},
        "group": {"model": "doubao-seed-2.1-turbo", "timeout_seconds": 360},
    }

    assert _generation_model_for_condition(experiment, "baseline", "private", config) == "deepseek-v4-flash"
    assert _generation_model_for_condition(experiment, "baseline", "group", config) == "doubao-seed-2.1-turbo"
    assert _generation_model_for_condition(experiment, "candidate", "private", config) == "deepseek-v4-flash"
    assert _generation_model_for_condition(experiment, "candidate", "group", config) == "doubao-seed-2.1-turbo"
    assert _generation_timeout_for_condition(experiment, "baseline", "private", config) == 20
    assert _generation_timeout_for_condition(experiment, "baseline", "group", config) == 360
    assert _generation_timeout_for_condition(experiment, "candidate", "private", config) == 20
    assert _generation_timeout_for_condition(experiment, "candidate", "group", config) == 180


def test_codex_generation_client_preserves_order_and_isolation(tmp_path):
    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "第一句"},
        {"role": "assistant", "content": "第二句"},
        {"role": "user", "content": "第三句"},
    ]

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"replies":["收到"]}', encoding="utf-8")
        transported = json.loads(kwargs["input"].split("\n", 1)[1])
        assert transported == messages
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed"}\n', stderr="")

    client = CodexExecClient("gpt-5.6-luna", reasoning_effort="low")
    with patch("scripts.run_situation_c0_ab.subprocess.run", side_effect=fake_run) as run:
        reply = client.chat(messages=messages, timeout=180, raise_on_error=True)

    command = run.call_args.args[0]
    assert command[:2] == ["codex", "exec"]
    assert ["-m", "gpt-5.6-luna"] == command[command.index("-m"):command.index("-m") + 2]
    assert 'model_reasoning_effort="low"' in command
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert reply == '{"replies":["收到"]}'


def test_codex_generation_client_rejects_tool_event():
    def fake_run(command, **_kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("不应采用", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"command_execution"}}\n',
            stderr="",
        )

    client = CodexExecClient("gpt-5.6-luna")
    with patch("scripts.run_situation_c0_ab.subprocess.run", side_effect=fake_run):
        try:
            client.chat(messages=[{"role": "user", "content": "回复"}], raise_on_error=True)
        except ValueError as exc:
            assert "tool call" in str(exc)
        else:
            raise AssertionError("Codex tool event was accepted")


def test_overlap_excludes_answer_sequence_inside_longer_few_shot_context():
    cases = [{
        "id": "case",
        "context": ["短上下文"],
        "reply": ["卧槽", "你自己也发财了"],
    }]
    rows = {
        "adjacent": {
            "id": "adjacent",
            "context": ["更早的话", "卧槽", "你自己也发财了", "后续消息"],
            "reply": ["遗产分一下"],
        },
        "unrelated": {
            "id": "unrelated",
            "context": ["更早的话", "卧槽", "后续消息"],
            "reply": ["你自己也发财了"],
        },
    }

    overlaps = _per_case_overlap_ids(cases, rows)

    assert overlaps["case"] == {"case", "adjacent"}


def test_overlap_excludes_scalar_formal_reply_inside_multi_reply_few_shot():
    cases = [{
        "id": "case",
        "context": ["还在呀", "累不累"],
        "reply": "累的半死不活的",
    }]
    rows = {
        "leak": {
            "id": "leak",
            "context": ["还在呀", "累不累"],
            "reply": ["累的半死不活的", "刚到家"],
        },
    }

    overlaps = _per_case_overlap_ids(cases, rows)

    assert overlaps["case"] == {"case", "leak"}


def test_timeline_overlap_treats_scalar_context_and_reply_as_messages():
    row = {
        "context": "最近还天天在家看安家",
        "reply": "庆余年应该是在病毒之前就看完了好么",
    }

    assert _timeline_overlaps(
        row,
        [
            "以前我不怎么看电视剧",
            "最近还天天在家看安家",
            "庆余年应该是在病毒之前就看完了好么",
            "不是的",
        ],
    )
    assert not _timeline_overlaps(row, ["最", "近", "还", "天", "天"])


def test_same_chat_full_timeline_excludes_partial_overlap_but_keeps_nonoverlap():
    cases = [{
        "id": "human_b5a19d5739ad",
        "chat_id": "chat_11944220c1",
        "context": ["安家跟弱智你不觉着么", "我以前没看过"],
        "reply": ["更"],
    }]
    rows = {
        "human_b1fae1f0f3a8": {
            "id": "human_b1fae1f0f3a8",
            "chat_id": "chat_11944220c1",
            "context": [
                "以前我不怎么看电视剧 都是这个新冠病毒闷的 我居然看完了庆余年",
                "最近还天天在家看安家",
            ],
            "reply": ["庆余年应该是在病毒之前就看完了好么"],
        },
        "same-chat-nonoverlap": {
            "id": "same-chat-nonoverlap",
            "chat_id": "chat_11944220c1",
            "context": ["另一年聊旅行", "国庆去哪里"],
            "reply": ["新疆"],
        },
    }
    full_context = {
        "human_b5a19d5739ad": [
            "以前我不怎么看电视剧 都是这个新冠病毒闷的 我居然看完了庆余年",
            "最近还天天在家看安家",
            "庆余年应该是在病毒之前就看完了好么",
            "不是的",
            "安家跟弱智你不觉着么",
            "我以前没看过",
        ],
    }

    overlaps = _per_case_overlap_ids(cases, rows, full_context)

    assert overlaps["human_b5a19d5739ad"] == {
        "human_b5a19d5739ad",
        "human_b1fae1f0f3a8",
    }


def test_generation_isolation_rejects_saved_test_related_few_shot():
    entries = {
        "case:baseline": {
            "status": "scorable",
            "result": {"trace": {"few_shot_ids": ["safe", "leak"]}},
        },
    }

    try:
        _assert_generation_entries_isolated(entries, {"case": {"case", "leak"}})
    except ValueError as exc:
        assert "case:baseline" in str(exc)
        assert "leak" in str(exc)
    else:
        raise AssertionError("contaminated saved generation was accepted")


def test_judge_reference_removes_only_case_leak_without_mutating_source():
    reference = {
        "examples": [
            {"id": "leak", "context": ["别的话"], "human_reply": ["这是目标真人回复"]},
            {"id": "safe", "context": ["安全上下文"], "human_reply": ["安全回复"]},
        ],
        "scenario_few_shots": [],
    }
    case = {
        "case_id": "case",
        "relationship": "group",
        "context": ["当前上下文"],
        "option_A": ["这是目标真人回复"],
        "option_B": ["候选回复"],
    }

    filtered, removed = _reference_without_case_leaks(reference, case)

    assert removed == [{"section": "examples", "index": 0, "id": "leak"}]
    assert [row["id"] for row in filtered["examples"]] == ["safe"]
    assert [row["id"] for row in reference["examples"]] == ["leak", "safe"]
    assert_reference_isolated(filtered, [case])


def test_overlap_excludes_same_stable_source_id_and_exact_pair():
    cases = [{
        "id": "case",
        "original_message_id": "source-1",
        "context": ["上下文"],
        "reply": ["真人回复"],
    }]
    rows = {
        "same-source": {
            "id": "same-source",
            "original_message_id": "source-1",
            "context": ["不同上下文"],
            "reply": ["不同回复"],
        },
        "same-pair": {
            "id": "same-pair",
            "context": ["上下文"],
            "reply": ["真人回复"],
        },
    }

    overlaps = _per_case_overlap_ids(cases, rows)

    assert overlaps["case"] == {"case", "same-source", "same-pair"}


def test_overlap_merges_expanded_candidate_pool_matches():
    cases = [{"id": "case", "context": ["问题"], "reply": ["真人答案"]}]
    ranking_pool = {"rank-only": {"id": "rank-only", "context": ["别的问题"], "reply": ["别的答案"]}}
    expanded_pool = {"expanded-leak": {"id": "expanded-leak", "context": ["问题"], "reply": ["真人答案"]}}

    overlaps = _per_case_overlap_ids_from_pools(cases, [ranking_pool, expanded_pool])

    assert overlaps["case"] == {"case", "expanded-leak"}


def test_overlap_from_named_render_pool_excludes_same_id_from_ranking():
    cases = [{"id": "case", "context": ["问题"], "reply": ["真人答案"]}]
    ranking_pool = {"shared-id": {"id": "shared-id", "context": ["别的问题"], "reply": ["别的答案"]}}
    named_render_pool = {"shared-id": {"id": "shared-id", "context": ["问题"], "reply": ["真人答案"]}}

    overlaps = _per_case_overlap_ids_from_pools(cases, [ranking_pool, named_render_pool])

    assert overlaps["case"] == {"case", "shared-id"}


def test_contaminated_checkpoint_entry_is_removed_before_reuse():
    entries = {
        "case:baseline": {
            "status": "scorable",
            "result": {"replies": ["回复"], "trace": {"few_shot_ids": ["leak"]}},
        },
        "case:candidate": {
            "status": "scorable",
            "result": {"replies": ["回复"], "trace": {"few_shot_ids": ["clean"]}},
        },
    }

    removed = _drop_contaminated_generation_entries(entries, {"case": {"leak"}})

    assert removed == 1
    assert set(entries) == {"case:candidate"}


def test_information_gap_feature_is_isolated_when_tested_and_retained_after_adoption():
    assert not _information_gap_response_move_for_condition(
        "information-gap-response-move", "baseline"
    )
    assert _information_gap_response_move_for_condition(
        "information-gap-response-move", "candidate"
    )
    assert _information_gap_response_move_for_condition("few-shot-count-5", "baseline")
    assert _information_gap_response_move_for_condition("few-shot-count-5", "candidate")


def test_group_participant_feature_keeps_private_latest_turn_form_out_of_group():
    private = PersonaFewShotRetriever(
        Path("unused"),
        prefer_latest_turn_form=True,
        prefer_group_participant_structure=False,
    )
    group = PersonaFewShotRetriever(
        Path("unused"),
        prefer_latest_turn_form=False,
        prefer_group_participant_structure=True,
    )

    assert private.prefer_latest_turn_form is True
    assert private.prefer_group_participant_structure is False
    assert group.prefer_latest_turn_form is False
    assert group.prefer_group_participant_structure is True


def test_group_participant_experiment_uses_identical_private_retriever():
    private = object()
    group_baseline = object()
    group_candidate = object()
    retrievers = {
        "baseline": {"group": group_baseline, "private": private},
        "candidate": {"group": group_candidate, "private": private},
    }

    assert retrievers["baseline"]["private"] is retrievers["candidate"]["private"]
    assert retrievers["baseline"]["group"] is not retrievers["candidate"]["group"]


def test_group_participant_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-participant-structure",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_only_generation_model_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "generation-model",
        group_only_candidate=True,
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_only_generation_model_does_not_reuse_group_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "generation-model",
        group_only_candidate=True,
    )

    assert reused is False
    assert "case-1:candidate" not in entries


def test_generation_model_without_group_only_does_not_reuse_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "generation-model",
    )

    assert reused is False
    assert "case-1:candidate" not in entries


def test_group_speaker_target_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-speaker-target-attribution",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_target_action_contract_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-target-action-contract",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_public_question_followup_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-public-question-specific-followup",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_instant_self_correction_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-instant-self-correction",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_visible_self_stance_continuity_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-visible-self-stance-continuity",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_pending_self_action_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-pending-self-action",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_pattern_participation_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-pattern-participation",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_current_event_self_role_continuity_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-current-event-self-role-continuity",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_explicit_self_stance_action_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-explicit-self-stance-action",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_answer_authority_rewrite_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-answer-authority-rewrite",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_nontriggering_group_target_action_contract_reuses_generation_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-target-action-contract",
        False,
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_public_question_followup_reuses_only_nontriggering_group_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-public-question-specific-followup",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline

    triggering_entries = {"case-1:baseline": baseline}
    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        triggering_entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-public-question-specific-followup",
        True,
    ) is False
    assert "case-1:candidate" not in triggering_entries


def test_group_instant_self_correction_reuses_only_nontriggering_group_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-instant-self-correction",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline

    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    triggering_entries = {"case-1:baseline": baseline}
    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        triggering_entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-instant-self-correction",
        True,
    ) is False
    assert "case-1:candidate" not in triggering_entries


def test_group_visible_self_stance_continuity_reuses_only_nontriggering_group_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-visible-self-stance-continuity",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline

    triggering_entries = {"case-1:baseline": baseline}
    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        triggering_entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-visible-self-stance-continuity",
        True,
    ) is False
    assert "case-1:candidate" not in triggering_entries


def test_group_pending_self_action_reuses_only_nontriggering_group_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-pending-self-action",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline

    triggering_entries = {"case-1:baseline": baseline}
    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        triggering_entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-pending-self-action",
        True,
    ) is False
    assert "case-1:candidate" not in triggering_entries


def test_group_pattern_participation_reuses_only_nontriggering_group_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-pattern-participation",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline


def test_group_current_event_self_role_continuity_reuses_only_nontriggering_group_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-current-event-self-role-continuity",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline


def test_group_current_event_self_role_continuity_activation_uses_rebuilt_timeline():
    source = SimpleNamespace(relation="group")
    previous = ChatMessage(
        sender="我",
        text="我争取 但是8点下班",
        sender_type=SenderType.SELF,
        chat_name="测试群",
    )
    first = ChatMessage(
        sender="小王",
        text="那就九点开始",
        sender_type=SenderType.OTHER,
        chat_name="测试群",
    )
    incoming = ChatMessage(
        sender="小李",
        text="等你下班过来",
        sender_type=SenderType.OTHER,
        chat_name="测试群",
    )

    with patch(
        "scripts.run_situation_c0_ab.rebuild_generation_inputs",
        return_value=([incoming], [previous, first, incoming], {}),
    ) as rebuild:
        assert _group_current_event_self_role_continuity_contract_applies(
            source, {"messages": []}
        ) is True

    rebuild.assert_called_once_with(
        source, {"messages": []}, compact_verified_turn=True
    )

    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    triggering_entries = {"case-1:baseline": baseline}
    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        triggering_entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-current-event-self-role-continuity",
        True,
    ) is False
    assert "case-1:candidate" not in triggering_entries


def test_group_explicit_self_stance_action_activation_uses_rebuilt_timeline():
    source = SimpleNamespace(relation="group")
    previous = ChatMessage(
        sender="我",
        text="感觉现在不想买，主要没有用车需求",
        sender_type=SenderType.SELF,
        chat_name="测试群",
    )
    incoming = ChatMessage(
        sender="小李",
        text="那你先给我当三年司机练练手",
        sender_type=SenderType.OTHER,
        chat_name="测试群",
    )

    with patch(
        "scripts.run_situation_c0_ab.rebuild_generation_inputs",
        return_value=([incoming], [previous, incoming], {}),
    ) as rebuild:
        assert _group_explicit_self_stance_action_contract_applies(
            source, {"messages": []}
        ) is True

    rebuild.assert_called_once_with(
        source, {"messages": []}, compact_verified_turn=True
    )

    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}
    assert _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-explicit-self-stance-action",
        False,
    ) is True
    assert entries["case-1:candidate"] is baseline


def test_group_instant_self_correction_replay_trigger_uses_rebuilt_message_identity():
    previous = ChatMessage(
        sender="我",
        sender_type=SenderType.SELF,
        message_type="text",
        text="接到",
        chat_name="测试群",
    )
    incoming = ChatMessage(
        sender="小王",
        sender_type=SenderType.OTHER,
        message_type="text",
        text="插话",
        chat_name="测试群",
    )
    with patch(
        "scripts.run_situation_c0_ab.rebuild_generation_inputs",
        return_value=([incoming], [previous, incoming], {}),
    ):
        assert _group_instant_self_correction_contract_applies(
            SimpleNamespace(), {}
        ) is True


def test_group_visible_self_stance_continuity_replay_trigger_uses_rebuilt_timeline():
    previous = ChatMessage(
        sender="我",
        sender_type=SenderType.SELF,
        message_type="text",
        text="没事 一起还了 主要利息太高",
        chat_name="测试群",
    )
    unread = [
        ChatMessage(
            sender="小王",
            sender_type=SenderType.OTHER,
            message_type="text",
            text="5还利息高？",
            chat_name="测试群",
        ),
        ChatMessage(
            sender="小王",
            sender_type=SenderType.OTHER,
            message_type="text",
            text="比银行便宜吧？",
            chat_name="测试群",
        ),
    ]
    with patch(
        "scripts.run_situation_c0_ab.rebuild_generation_inputs",
        return_value=(unread, [previous, *unread], {}),
    ):
        assert _group_visible_self_stance_continuity_contract_applies(
            SimpleNamespace(), {}
        ) is True


def test_group_pending_self_action_replay_trigger_uses_rebuilt_timeline():
    previous = ChatMessage(
        sender="我",
        sender_type=SenderType.SELF,
        message_type="text",
        text="solo还能有位置么",
        chat_name="测试群",
    )
    incoming = ChatMessage(
        sender="小王",
        sender_type=SenderType.OTHER,
        message_type="text",
        text="好的，那我不找了，就1吧",
        chat_name="测试群",
    )
    with patch(
        "scripts.run_situation_c0_ab.rebuild_generation_inputs",
        return_value=([incoming], [previous, incoming], {}),
    ):
        assert _group_pending_self_action_contract_applies(
            SimpleNamespace(), {}
        ) is True


def test_group_pattern_participation_replay_trigger_uses_rebuilt_timeline():
    previous = ChatMessage(
        sender="小王",
        sender_type=SenderType.OTHER,
        message_type="text",
        text="事到如今最对不起的就是西西",
        chat_name="测试群",
    )
    incoming = ChatMessage(
        sender="小李",
        sender_type=SenderType.OTHER,
        message_type="text",
        text="事到如今最对不起的就是西西",
        chat_name="测试群",
    )
    with patch(
        "scripts.run_situation_c0_ab.rebuild_generation_inputs",
        return_value=([incoming], [previous, incoming], {}),
    ):
        assert _group_pattern_participation_contract_applies(
            SimpleNamespace(), {}
        ) is True


def test_group_instant_self_correction_core_case_tails_both_trigger():
    tails = [
        ("房价看板块 不看接到", "不是这个同济融创玫瑰公馆吗？"),
        ("安家跟弱智你不觉着么", "我以前没看过"),
    ]

    for previous_text, incoming_text in tails:
        previous = ChatMessage(
            sender="我",
            sender_type=SenderType.SELF,
            message_type="text",
            text=previous_text,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="群友",
            sender_type=SenderType.OTHER,
            message_type="text",
            text=incoming_text,
            chat_name="测试群",
        )
        with patch(
            "scripts.run_situation_c0_ab.rebuild_generation_inputs",
            return_value=([incoming], [previous, incoming], {}),
        ):
            assert _group_instant_self_correction_contract_applies(
                SimpleNamespace(), {}
            ) is True


def test_nontriggering_group_answer_authority_rewrite_reuses_generation_baseline():
    baseline = {"status": "scorable", "result": {"replies": ["收到"]}}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-answer-authority-rewrite",
        False,
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_triggering_group_answer_authority_rewrite_does_not_reuse_generation_baseline():
    entries = {"case-1:baseline": {"status": "scorable"}}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-answer-authority-rewrite",
        True,
    )

    assert reused is False
    assert "case-1:candidate" not in entries


def test_triggering_group_target_action_contract_does_not_reuse_generation_baseline():
    entries = {"case-1:baseline": {"status": "scorable"}}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-target-action-contract",
        True,
    )

    assert reused is False
    assert "case-1:candidate" not in entries


def test_unknown_group_target_action_contract_state_does_not_reuse_generation_baseline():
    entries = {"case-1:baseline": {"status": "scorable"}}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-target-action-contract",
        None,
    )

    assert reused is False
    assert "case-1:candidate" not in entries


def test_nontriggering_group_target_action_contract_requires_scorable_generation_baseline():
    entries = {"case-1:baseline": {"status": "generation_failed"}}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-target-action-contract",
        False,
    )

    assert reused is False
    assert "case-1:candidate" not in entries


def test_nontriggering_group_target_action_contract_reuses_judge_baseline():
    baseline = {"correct": True, "confidence": 0.8}
    entries = {"case-1:baseline": baseline}

    reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
        entries,
        "case-1",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        "group-target-action-contract",
        False,
        judge=True,
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_nontriggering_group_target_action_contract_rejects_missing_or_failed_judge_baseline():
    for entries in ({}, {"case-1:baseline": {"error": "timeout"}}):
        reused = _reuse_group_baseline_for_nontriggering_target_action_contract(
            entries,
            "case-1",
            {"relationship": "group"},
            "candidate",
            1,
            1,
            "group-target-action-contract",
            False,
            judge=True,
        )

        assert reused is False
        assert "case-1:candidate" not in entries


def test_explicit_third_party_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-explicit-third-party-guard",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_relevant_self_evidence_experiment_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-relevant-self-evidence",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_self_context_freshness_boundary_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-self-context-freshness-boundary",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_group_adjacent_self_response_reuses_private_baseline_result():
    baseline = {"status": "scorable", "result": {"replies": ["问下"]}}
    entries = {"case-1:baseline": baseline, "case-1:candidate": {"status": "scorable"}}

    reused = _reuse_private_baseline_for_group_only_change(
        entries,
        "case-1",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        "group-adjacent-self-response",
    )

    assert reused is True
    assert entries["case-1:candidate"] is baseline


def test_expanded_pool_changes_only_candidate_retrieval_path(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    expanded = tmp_path / "expanded.jsonl"
    experiment = Namespace(
        candidate_mode="expanded-pool",
        few_shot=baseline,
        candidate_few_shot=expanded,
    )

    assert _few_shot_path_for_condition(experiment, "baseline") == baseline
    assert _few_shot_path_for_condition(experiment, "candidate") == expanded


def test_empty_reply_is_not_scorable():
    assert not _has_nonempty_reply({"replies": []})
    assert not _has_nonempty_reply({"replies": [""]})
    assert not _has_nonempty_reply({"replies": ["  "]})
    assert _has_nonempty_reply({"replies": ["问下谁发的"]})


def test_few_shot_count_candidate_changes_only_candidate_limit():
    count_experiment = Namespace(candidate_mode="few-shot-count-5", few_shot_count=3)
    existing_experiment = Namespace(candidate_mode="latest-request-first", few_shot_count=3)

    assert _few_shot_count_for_condition(count_experiment, "baseline") == 3
    assert _few_shot_count_for_condition(count_experiment, "candidate") == 5
    assert _few_shot_count_for_condition(existing_experiment, "baseline") == 3
    assert _few_shot_count_for_condition(existing_experiment, "candidate") == 3


def test_repeated_ab_uses_exact_interleaved_order_and_distinct_checkpoint_keys():
    assert _generation_schedule(3) == [
        ("baseline", 1),
        ("candidate", 1),
        ("candidate", 2),
        ("baseline", 2),
        ("baseline", 3),
        ("candidate", 3),
    ]


def test_hard_error_attribution_supports_six_interleaved_repetitions():
    schedule = _generation_schedule(6)

    assert len(schedule) == 12
    assert schedule[:4] == [
        ("baseline", 1),
        ("candidate", 1),
        ("candidate", 2),
        ("baseline", 2),
    ]
    assert schedule[-4:] == [
        ("baseline", 5),
        ("candidate", 5),
        ("candidate", 6),
        ("baseline", 6),
    ]


def test_fixed_case_ids_preserve_results_order(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('{"cases":[{"case_id":"two"},{"case_id":"one"}]}', encoding="utf-8")

    assert _case_ids_from_results(path) == ["two", "one"]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"case_ids":["three","four"]}', encoding="utf-8")
    assert _case_ids_from_results(manifest_path) == ["three", "four"]


def test_seed_baseline_only_does_not_import_old_candidate(tmp_path):
    generation_path = tmp_path / "generation.json"
    generation_path.write_text(
        '{"entries":{"case:baseline":{"status":"scorable","result":{"trace":{"few_shot_ids":[]}}},'
        '"case:candidate":{"status":"scorable","result":{"trace":{"few_shot_ids":[]}}}}}',
        encoding="utf-8",
    )
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(
        '{"entries":{"case:baseline":{"correct":true},"case:candidate":{"correct":false}}}',
        encoding="utf-8",
    )
    generation_entries = {}
    judge_entries = {}

    assert _seed_scorable_entries(
        generation_entries, ["case"], [generation_path], conditions=("baseline",)
    ) == 1
    assert _seed_judge_entries(
        judge_entries, ["case"], [judge_path], conditions=("baseline",)
    ) == 1
    assert set(generation_entries) == {"case:baseline"}
    assert set(judge_entries) == {"case:baseline"}
    assert [_entry_key("case", condition, repetition, 3) for condition, repetition in _generation_schedule(3)] == [
        "case:baseline:1",
        "case:candidate:1",
        "case:candidate:2",
        "case:baseline:2",
        "case:baseline:3",
        "case:candidate:3",
    ]


def test_seed_new_baseline_from_previous_candidate_without_regeneration(tmp_path):
    checkpoint = tmp_path / "generation.json"
    checkpoint.write_text(
        '{"entries":{"case:baseline":{"status":"scorable","result":{"replies":["old"]}},'
        '"case:candidate":{"status":"scorable","result":{"replies":["adopted"],"trace":{"few_shot_ids":[]}}}}}',
        encoding="utf-8",
    )
    entries = {}

    assert _seed_baseline_from_candidate_entries(entries, ["case"], [checkpoint]) == 1
    assert entries["case:baseline"]["result"]["replies"] == ["adopted"]
    assert "case:candidate" not in entries


def test_seed_new_baseline_judge_from_previous_candidate_without_rejudging(tmp_path):
    checkpoint = tmp_path / "judge.json"
    checkpoint.write_text(
        '{"entries":{"case:baseline":{"correct":false},'
        '"case:candidate":{"correct":true,"confidence":0.7}}}',
        encoding="utf-8",
    )
    entries = {"case:baseline": {"correct": False}}

    assert _seed_baseline_from_candidate_judge_entries(entries, ["case"], [checkpoint]) == 1
    assert entries["case:baseline"]["correct"] is True
    assert entries["case:baseline"]["confidence"] == 0.7
    assert "case:candidate" not in entries


def test_seed_split_baseline_uses_old_group_and_adopted_private(tmp_path):
    generation = tmp_path / "generation.json"
    generation.write_text(
        '{"entries":{'
        '"group:baseline":{"status":"scorable","result":{"replies":["group-old"],"trace":{"few_shot_ids":[]}}},'
        '"group:candidate":{"status":"scorable","result":{"replies":["group-rejected"],"trace":{"few_shot_ids":[]}}},'
        '"private:baseline":{"status":"scorable","result":{"replies":["private-old"],"trace":{"few_shot_ids":[]}}},'
        '"private:candidate":{"status":"scorable","result":{"replies":["private-adopted"],"trace":{"few_shot_ids":[]}}}}}',
        encoding="utf-8",
    )
    selected = {
        "group": {"relationship": "group"},
        "private": {"relationship": "acquaintance"},
    }
    entries = {}

    assert _seed_split_baseline_entries(
        entries, ["group", "private"], selected, [generation]
    ) == 2
    assert entries["group:baseline"]["result"]["replies"] == ["group-old"]
    assert entries["private:baseline"]["result"]["replies"] == ["private-adopted"]


def test_seed_private_candidate_uses_candidate_private_and_baseline_group(tmp_path):
    generation = tmp_path / "generation.json"
    generation.write_text(
        '{"entries":{'
        '"group:baseline":{"status":"scorable","result":{"replies":["group-old"],"trace":{"few_shot_ids":[]}}},'
        '"group:candidate":{"status":"scorable","result":{"replies":["group-changed"],"trace":{"few_shot_ids":[]}}},'
        '"private:baseline":{"status":"scorable","result":{"replies":["private-old"],"trace":{"few_shot_ids":[]}}},'
        '"private:candidate":{"status":"scorable","result":{"replies":["private-new"],"trace":{"few_shot_ids":[]}}}}}',
        encoding="utf-8",
    )
    selected = {
        "group": {"relationship": "group"},
        "private": {"relationship": "acquaintance"},
    }
    entries = {}

    assert _seed_private_candidate_entries(
        entries, ["group", "private"], selected, [generation]
    ) == 2
    assert entries["group:candidate"]["result"]["replies"] == ["group-old"]
    assert entries["private:candidate"]["result"]["replies"] == ["private-new"]


def test_seed_private_candidate_rejects_output_using_forbidden_few_shot(tmp_path):
    generation = tmp_path / "generation.json"
    generation.write_text(
        '{"entries":{'
        '"private:candidate":{"status":"scorable","result":{"replies":["leaked"],'
        '"trace":{"few_shot_ids":["same-source"]}}}}}',
        encoding="utf-8",
    )
    entries = {}

    assert _seed_private_candidate_entries(
        entries,
        ["private"],
        {"private": {"relationship": "private"}},
        [generation],
        forbidden_by_case={"private": {"same-source"}},
    ) == 0
    assert entries == {}


def test_private_only_candidate_reuses_group_baseline_judge_decision():
    baseline = {"correct": True, "confidence": 0.8}
    entries = {"group:baseline": baseline}

    assert _reuse_group_baseline_judge_for_private_only_candidate(
        entries,
        "group",
        {"relationship": "group"},
        "candidate",
        1,
        1,
        True,
    ) is True
    assert entries["group:candidate"] is baseline


def test_private_only_candidate_does_not_reuse_private_judge_decision():
    entries = {"private:baseline": {"correct": True}}

    assert _reuse_group_baseline_judge_for_private_only_candidate(
        entries,
        "private",
        {"relationship": "private"},
        "candidate",
        1,
        1,
        True,
    ) is False
    assert "private:candidate" not in entries


def _row(case_id: str, relationship: str, context: list[str]) -> dict:
    return {"id": case_id, "relationship": relationship, "context": context}


def test_latest_request_kind_requires_multi_turn_substantive_request():
    assert _latest_request_kind(["前文", "还在聊", "在吗"]) is None
    assert _latest_request_kind(["前文", "还在聊", "现金和股票比例多少"]) == "number-time-plan"
    assert _latest_request_kind(["前文", "开玩笑", "帮我看下这个呗"]) == "help-request"
    assert _latest_request_kind(["旧问题", "调侃", "晚上要不要一起吃饭"]) == "choice-invitation"
    assert _latest_request_kind(["旧话题", "哈哈", "你出发了吗"]) == "status"
    assert _latest_request_kind(["旧话题", "哈哈", "这个最后怎么处理？"]) == "direct-question"


def test_latest_request_selection_excludes_short_and_ping_only_cases():
    rows = [
        _row("short", "group", ["这个怎么处理？"]),
        _row("ping", "group", ["旧话题", "哈哈", "人呢"]),
        _row("group-question", "group", ["旧话题", "哈哈", "这个怎么处理？"]),
        _row("private-help", "acquaintance", ["旧话题", "哈哈", "帮我看下这个"]),
    ]

    selected = _select_cases(rows, set(), 2, "latest-request-first")

    assert {row["id"] for row in selected} == {"group-question", "private-help"}


class _FakeRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def is_approved(self):
        return True

    def retrieve(self, query, **kwargs):
        self.calls += 1
        return self.rows[: kwargs["limit"]]

    def render(self, rows, max_chars=2500):
        ids = []
        parts = []
        for row in rows:
            text = "|".join(message.get("text", "") for message in row.get("context_messages", [])) or "|".join(row["context"])
            if len("\n".join(parts + [text])) > max_chars:
                break
            parts.append(text)
            ids.append(row["id"])
        return "\n".join(parts), ids

    def render_selected(self, rows, max_chars=2500):
        return self.render(rows, max_chars=max_chars)


def test_count_pair_ranks_once_and_keeps_fixed_prefixes_across_repetitions_and_retries():
    rows = [
        {"id": str(index), "context": [str(index)], "reply": [str(index)]}
        for index in range(1, 7)
    ]
    inner = _FakeRetriever(rows)
    state = PairedCountSelection(inner, max_count=5)
    baseline = PairedCountRetriever(state)
    candidate = PairedCountRetriever(state)

    baseline_first = baseline.retrieve("first query", limit=3, exclude_ids={"case"})
    candidate_first = candidate.retrieve("different query", limit=5, exclude_ids={"case"})
    candidate_retry = candidate.retrieve("grounding retry query", limit=5, exclude_ids={"case"})
    candidate_second = candidate.retrieve("third query", limit=5, exclude_ids={"case"})
    baseline_second = baseline.retrieve("fourth query", limit=3, exclude_ids={"case"})

    assert inner.calls == 1
    assert [row["id"] for row in baseline_first] == ["1", "2", "3"]
    assert [row["id"] for row in candidate_first] == ["1", "2", "3", "4", "5"]
    assert candidate_retry == candidate_first
    assert candidate_second == candidate_first
    assert baseline_second == baseline_first


def test_count_pair_restores_exact_checkpoint_ids_without_reranking():
    rows = [
        {"id": str(index), "context": [str(index)], "reply": [str(index)]}
        for index in range(1, 5)
    ]
    inner = _FakeRetriever(list(reversed(rows)))
    state = PairedCountSelection(inner, max_count=3)

    state.restore(["2", "1", "3"], {row["id"]: row for row in rows})
    selected = PairedCountRetriever(state).retrieve("new query", limit=3)

    assert inner.calls == 0
    assert [row["id"] for row in selected] == ["2", "1", "3"]
    assert state.trace == {
        "ranked_ids": ["2", "1", "3"],
        "restored_from_checkpoint": True,
    }


def test_count_pair_rejects_incomplete_checkpoint_ids():
    state = PairedCountSelection(_FakeRetriever([]), max_count=3)

    try:
        state.restore(["1", "2"], {})
    except ValueError as exc:
        assert str(exc) == "saved paired few-shot count is 2, expected 3"
    else:
        raise AssertionError("incomplete checkpoint must not silently rerank")


def test_named_context_pair_ranks_once_and_keeps_identical_ids():
    old = [
        {"id": "adjacent", "chat_id": "g", "timestamp": 900, "context": ["skip"], "reply": ["x"]},
        {"id": "one", "chat_id": "other", "timestamp": 1, "context": ["old one"], "reply": ["x"]},
        {"id": "two", "chat_id": "other", "timestamp": 2, "context": ["old two"], "reply": ["y"]},
    ]
    named = {
        "one": {**old[1], "context_messages": [{"sender": "小王", "text": "named one"}]},
        "two": {**old[2], "context_messages": [{"sender": "小李", "text": "named two"}]},
    }
    inner = _FakeRetriever(old)
    state = PairedNamedContextSelection(inner, named, max_chars=100)
    state.reset({"chat_id": "g", "timestamp": 1000})
    baseline = PairedNamedContextRetriever(state, "baseline")
    candidate = PairedNamedContextRetriever(state, "candidate")

    baseline_rows = baseline.retrieve("query", limit=2, exclude_ids={"case"})
    candidate_rows = candidate.retrieve("query", limit=2, exclude_ids={"case"})
    baseline_prompt, baseline_ids = baseline.render_selected(baseline_rows)
    candidate_prompt, candidate_ids = candidate.render_selected(candidate_rows)

    assert inner.calls == 1
    assert baseline_ids == candidate_ids == ["one", "two"]
    assert "old one" in baseline_prompt
    assert "named one" in candidate_prompt
    assert state.trace["same_event_excluded_ids"] == ["adjacent"]


def test_named_context_pair_uses_largest_shared_prompt_prefix():
    old = [
        {"id": "one", "chat_id": "a", "timestamp": 1, "context": ["a"], "reply": ["x"]},
        {"id": "two", "chat_id": "b", "timestamp": 2, "context": ["b"], "reply": ["y"]},
    ]
    named = {
        "one": {**old[0], "context_messages": [{"sender": "甲", "text": "short"}]},
        "two": {**old[1], "context_messages": [{"sender": "乙", "text": "x" * 30}]},
    }
    state = PairedNamedContextSelection(_FakeRetriever(old), named, max_chars=20)
    state.reset({"chat_id": "case", "timestamp": 10})
    baseline = PairedNamedContextRetriever(state, "baseline")
    candidate = PairedNamedContextRetriever(state, "candidate")

    assert [row["id"] for row in baseline.retrieve("query", limit=2)] == ["one"]
    assert [row["id"] for row in candidate.retrieve("query", limit=2)] == ["one"]


def test_named_context_pair_can_compare_short_and_long_named_contexts():
    ranked = [
        {"id": "one", "chat_id": "a", "timestamp": 1, "context": ["rank"], "reply": ["x"]},
        {"id": "two", "chat_id": "b", "timestamp": 2, "context": ["rank"], "reply": ["y"]},
    ]
    short = {
        row["id"]: {
            **row,
            "context_messages": [{"sender": "甲", "text": f"短-{row['id']}"}],
        }
        for row in ranked
    }
    long = {
        row["id"]: {
            **row,
            "context_messages": [
                {"sender": "甲", "text": f"前-{row['id']}"},
                {"sender": "乙", "text": f"后-{row['id']}"},
            ],
        }
        for row in ranked
    }
    state = PairedNamedContextSelection(
        _FakeRetriever(ranked), long, max_chars=500, baseline_by_id=short
    )
    state.reset({"chat_id": "case", "timestamp": 10})
    baseline = PairedNamedContextRetriever(state, "baseline")
    candidate = PairedNamedContextRetriever(state, "candidate")

    baseline_rows = baseline.retrieve("query", limit=2)
    candidate_rows = candidate.retrieve("query", limit=2)
    baseline_prompt, baseline_ids = baseline.render(baseline_rows)
    candidate_prompt, candidate_ids = candidate.render(candidate_rows)

    assert baseline_ids == candidate_ids == ["one", "two"]
    assert "短-one" in baseline_prompt and "前-one" not in baseline_prompt
    assert "前-one" in candidate_prompt and "后-one" in candidate_prompt


def test_exact_move_same_chat_gate_requires_an_exact_response_move(tmp_path):
    examples = tmp_path / "persona_examples.jsonl"
    examples.write_text(
        "\n".join(
                [
                    '{"id":"same-wrong","chat_id":"current","relationship":"group","context":["有些东西不知道怎么选，要不要一起看"],"reply":["比如"],"source_provenance":"explicit_human_marker","situation_tags":["invitation_request"],"situation_profile":{"expected_response_move":"clarify_detail"}}',
                    '{"id":"same-right","chat_id":"current","relationship":"group","context":["一起吃饭吗"],"reply":["几点"],"source_provenance":"explicit_human_marker"}',
                    '{"id":"cross-right","chat_id":"other","relationship":"group","context":["要不要出来吃"],"reply":["去哪"],"source_provenance":"explicit_human_marker"}',
            ]
        ),
        encoding="utf-8",
    )
    retriever = PersonaFewShotRetriever(
        examples,
        prefer_same_chat_after_exact_move=True,
    )

    selected = retriever.retrieve(
        "晚上要不要一起吃饭",
        chat_name="",
        chat_id="current",
        is_group=True,
        relationship="group",
        limit=3,
    )

    assert [row["id"] for row in selected][:2] == ["same-right", "cross-right"]
