import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_human_ai_judge import (
    VolcengineAnthropicClient,
    _candidate_identity,
    _chat_names,
    _json_sha256,
    build_judge_reference_pack,
    build_api_judge_messages,
    build_shared_single_case_prompt,
    build_codex_judge_prompt,
    build_blind_packet,
    detect_degenerate_case,
    extract_judge_chat_context,
    rebuild_generation_inputs,
    run_codex_judge,
    run_codex_judge_independently,
    main as judge_main,
    _parse_api_judge_response,
    score_surface_baselines,
    score_predictions,
    score_api_judge_predictions,
    select_holdout_candidates,
    validate_judge_events,
    validate_reference_isolation,
    reference_leakage_details,
    write_benchmark_artifacts,
)
from src.reply.c0_config import current_c0_baseline_id


def test_api_judge_prompt_contains_only_one_blind_case():
    case = {"case_id": "case-one", "context_original": ["上下文"], "option_A": ["甲"], "option_B": ["乙"]}
    messages = build_api_judge_messages(case, {"facts": [], "examples": []})
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "case-one" in serialized
    assert "human_option" in serialized
    assert "source_case_id" not in serialized
    assert _parse_api_judge_response('{"human_option":"B","confidence":0.7,"reason":"更自然"}')["human_option"] == "B"


def test_independent_codex_judge_resumes_without_repeating_completed_cases(tmp_path, monkeypatch):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [
            {"case_id": "one", "context_original": ["甲"], "option_A": ["真"], "option_B": ["假"]},
            {"case_id": "two", "context_original": ["乙"], "option_A": ["真"], "option_B": ["假"]},
        ],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    invoked_prompts = []

    def fake_run(command, input, **kwargs):
        invoked_prompts.append(input)
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text(
            '{"human_option":"A","confidence":0.8,"reason":"更自然"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.evaluate_human_ai_judge.subprocess.run", fake_run)

    first = run_codex_judge_independently(
        blind_path, predictions_path, audit_path, reference_path, max_cases=1
    )
    invocation_id = json.loads(predictions_path.read_text(encoding="utf-8"))["invocation_id"]
    second = run_codex_judge_independently(
        blind_path, predictions_path, audit_path, reference_path, max_cases=1
    )
    saved = json.loads(predictions_path.read_text(encoding="utf-8"))

    assert first == {"completed": 1, "total": 2}
    assert second == {"completed": 2, "total": 2}
    assert saved["invocation_id"] == invocation_id
    assert [row["case_id"] for row in saved["cases"]] == ["one", "two"]
    assert len(invoked_prompts) == 2


def test_official_judge_missing_runtime_fails_before_creating_outputs(tmp_path, monkeypatch):
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_human_ai_judge.py",
        "--run-codex-api-judge",
        "--official-judge",
        "--codex-api-judge-predictions", str(predictions_path),
        "--codex-api-judge-audit", str(audit_path),
    ])

    with pytest.raises(RuntimeError, match="baseline requirements are missing"):
        judge_main()

    assert not predictions_path.exists()
    assert not audit_path.exists()


@pytest.mark.parametrize("forbidden", [
    ["--codex-judge-profile", "custom.json"],
    ["--codex-dynamic-few-shot-manifest", "dynamic.json"],
    ["--codex-baseline-predictions", "old.json", "--codex-baseline-audit", "old-audit.json"],
    [
        "--codex-prior-dynamic-few-shot-manifest", "old-manifest.json",
        "--codex-prior-predictions", "old.json",
        "--codex-prior-audit", "old-audit.json",
    ],
    ["--presentation", "swapped"],
])
def test_official_judge_rejects_experiment_overrides(forbidden, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "evaluate_human_ai_judge.py",
        "--run-codex-api-judge",
        "--official-judge",
        *forbidden,
    ])

    with pytest.raises(SystemExit) as exc:
        judge_main()

    assert exc.value.code == 2


def test_official_judge_records_runtime_and_passes_it_to_codex(tmp_path, monkeypatch):
    blind = {
        "batch_id": "official-one",
        "reference_sha256": "ref",
        "cases": [
            {"case_id": "one", "context_original": ["甲"], "option_A": ["真"], "option_B": ["假"]},
        ],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    profile_path = tmp_path / "profile.json"
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    profile_path.write_text(
        json.dumps({"name": "official-profile", "instructions": "只比较两条回复。"}),
        encoding="utf-8",
    )
    runtime = {
        "official_judge_baseline_id": "judge-test",
        "current_c0_baseline_id": current_c0_baseline_id(),
        "provider": "codex_cli",
        "model": "test-model",
        "reasoning_effort": "xhigh",
        "timeout_seconds": 900,
        "codex_cli_version": "test-version",
    }
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text(
            '{"human_option":"A","confidence":0.8,"reason":"更自然"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.evaluate_human_ai_judge.subprocess.run", fake_run)
    monkeypatch.setattr(
        "scripts.evaluate_human_ai_judge.current_judge_baseline_id",
        lambda: "judge-test",
    )
    run_codex_judge_independently(
        blind_path,
        predictions_path,
        audit_path,
        reference_path,
        judge_profile_path=profile_path,
        official_metadata=runtime,
    )

    assert observed["command"][2:6] == [
        "--model", "test-model", "--config", 'model_reasoning_effort="xhigh"',
    ]
    assert observed["kwargs"]["timeout"] == 900
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for key, value in runtime.items():
        assert predictions[key] == value
        assert audit[key] == value


def test_independent_codex_judge_swaps_options_and_maps_choice_back(tmp_path, monkeypatch):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [
            {"case_id": "one", "context_original": ["上下文"], "option_A": ["原甲"], "option_B": ["原乙"]},
        ],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    def fake_run(command, input, **kwargs):
        assert input.index("原乙") < input.index("原甲")
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text(
            '{"human_option":"A","confidence":0.8,"reason":"更自然"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.evaluate_human_ai_judge.subprocess.run", fake_run)
    run_codex_judge_independently(
        blind_path,
        predictions_path,
        audit_path,
        reference_path,
        presentation="swapped",
    )

    saved = json.loads(predictions_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert saved["presentation"] == "swapped"
    assert saved["cases"][0]["displayed_human_option"] == "A"
    assert saved["cases"][0]["human_option"] == "B"
    assert audit["presentation"] == "swapped"
    assert audit["calls"][0]["displayed_prediction"]["human_option"] == "A"
    assert audit["calls"][0]["parsed_prediction"]["human_option"] == "B"


def test_independent_codex_judge_can_run_only_one_named_case(tmp_path, monkeypatch):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [
            {"case_id": "one", "context_original": ["甲"], "option_A": ["真"], "option_B": ["假"]},
            {"case_id": "two", "context_original": ["乙"], "option_A": ["真"], "option_B": ["假"]},
        ],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    def fake_run(command, input, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text(
            '{"human_option":"A","confidence":0.8,"reason":"更自然"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.evaluate_human_ai_judge.subprocess.run", fake_run)

    progress = run_codex_judge_independently(
        blind_path,
        predictions_path,
        audit_path,
        reference_path,
        only_case_ids={"two"},
    )
    saved = json.loads(predictions_path.read_text(encoding="utf-8"))

    assert progress == {"completed": 1, "total": 2}
    assert [row["case_id"] for row in saved["cases"]] == ["two"]


def test_independent_codex_judge_rejects_unknown_named_case(tmp_path):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [{"case_id": "one", "context_original": ["甲"], "option_A": ["真"], "option_B": ["假"]}],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown Codex Judge case IDs"):
        run_codex_judge_independently(
            blind_path,
            tmp_path / "predictions.json",
            tmp_path / "audit.json",
            reference_path,
            only_case_ids={"missing"},
        )


def test_api_judge_prompt_has_output_schema_but_no_answer_value():
    case = {"case_id": "case-one", "context_original": ["上下文"], "option_A": ["甲"], "option_B": ["乙"]}
    messages = build_api_judge_messages(case)
    serialized = json.dumps(messages, ensure_ascii=False)
    system_prompt = messages[0]["content"]
    assert '"human_option"' in system_prompt
    assert not re.search(r'"human_option"\s*:\s*"[AB]"', system_prompt)
    for leaked_field in ('"source_case_id"', '"actual"', '"correct"'):
        assert leaked_field not in serialized


def test_shared_single_case_prompt_is_one_neutral_payload():
    case = {"case_id": "case-one", "context_original": ["上下文"], "option_A": ["甲"], "option_B": ["乙"]}
    prompt = build_shared_single_case_prompt(case, {"facts": [], "examples": []})
    assert prompt.count("case-one") == 1
    assert "<judge_instructions>" in prompt and "<blind_case>" in prompt
    assert not re.search(r'"human_option"\s*:\s*"[AB]"', prompt)


def test_default_shared_prompt_is_unchanged_when_profile_is_omitted():
    case = {"case_id": "case-one", "context_original": ["上下文"], "option_A": ["甲"], "option_B": ["乙"]}
    reference = {"facts": [], "examples": []}
    assert build_shared_single_case_prompt(case, reference, None) == build_shared_single_case_prompt(case, reference)


def test_append_profile_changes_only_the_single_instruction():
    case = {"case_id": "case-one", "context_original": ["上下文"], "option_A": ["甲"], "option_B": ["乙"]}
    reference = {"facts": [], "examples": []}
    instruction = "不得仅因某个选项没有直接回答最后一句就判它更像 AI。"
    baseline = build_shared_single_case_prompt(case, reference)
    candidate = build_shared_single_case_prompt(
        case,
        reference,
        {"name": "single-rule", "mode": "append_to_builtin", "instructions": instruction},
    )
    assert candidate.replace(instruction, "", 1) == baseline
    assert candidate.count(instruction) == 1
    for field in ("option_a_thread", "option_a_counterevidence", "option_b_thread", "option_b_counterevidence"):
        assert field not in candidate


def test_thread_attribution_profile_requires_symmetric_threads_without_leaking_answer():
    case = {"case_id": "case-one", "context_original": ["上下文"], "option_A": ["甲"], "option_B": ["乙"]}
    profile = {
        "name": "thread-attribution",
        "instructions": "先判断 A、B 分别承接完整时间线中的哪个问题、状态或并行话题；不得默认真人一定回应最后一句。",
    }
    prompt = build_shared_single_case_prompt(case, {"facts": [], "examples": []}, profile)
    for field in ("option_a_thread", "option_a_counterevidence", "option_b_thread", "option_b_counterevidence"):
        assert field in prompt
    assert "不得默认真人一定回应最后一句" in prompt
    assert not re.search(r'"human_option"\s*:\s*"[AB]"', prompt)
    for leaked_field in ('"source_case_id"', '"actual"', '"correct"'):
        assert leaked_field not in prompt


def test_independent_codex_judge_refuses_to_resume_with_a_different_profile(tmp_path, monkeypatch):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [{"case_id": "one", "context_original": ["甲"], "option_A": ["真"], "option_B": ["假"]}],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    profile_path = tmp_path / "profile.json"
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    profile_path.write_text(json.dumps({"name": "candidate", "instructions": "先判断承接话题。"}), encoding="utf-8")

    def fake_run(command, input, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text('{"human_option":"A","confidence":0.8,"reason":"更自然"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.evaluate_human_ai_judge.subprocess.run", fake_run)
    run_codex_judge_independently(blind_path, predictions_path, audit_path, reference_path)

    with pytest.raises(ValueError, match="does not match"):
        run_codex_judge_independently(
            blind_path, predictions_path, audit_path, reference_path, judge_profile_path=profile_path
        )
from src.models.base import SenderType


def _candidate(chat_id: str, relation: str = "group", timestamp: int = 1_700_000_003) -> Candidate:
    return Candidate(
        relation=relation,
        chat_id=chat_id,
        context=["你看这个"],
        replies=["看到了"],
        score=1,
        timestamp=timestamp,
    )


def test_holdout_excludes_exact_few_shot_and_uses_unique_chats():
    excluded = _candidate("excluded")
    candidates = [excluded, _candidate("a"), _candidate("a"), _candidate("b")]
    selected = select_holdout_candidates(candidates, {_candidate_identity(excluded)}, per_relation=2)
    assert {row.chat_id for row in selected} == {"a", "b"}


def test_chat_names_restore_production_display_name(tmp_path):
    export = tmp_path / "私聊_文件名.json"
    export.write_text('{"session":{"displayName":"当前备注名"}}', encoding="utf-8")
    expected_id = "chat_" + hashlib.sha256("文件名".encode()).hexdigest()[:10]
    assert _chat_names(tmp_path)[expected_id] == "当前备注名"


def test_volcengine_adapter_translates_tools_and_tool_results():
    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "model": "deepseek-v4-flash-resolved",
                "content": [
                    {"type": "thinking", "thinking": "先查记忆"},
                    {"type": "tool_use", "id": "tool-1", "name": "search_memory", "input": {"query": "甲"}},
                ],
            }

    class Session:
        def post(self, url, headers, json, timeout):
            self.request = {"url": url, "headers": headers, "json": json, "timeout": timeout}
            return Response()

    session = Session()
    client = VolcengineAnthropicClient("secret", "https://example.test/api/coding", "deepseek-v4-flash", session)
    result = client.chat(
        messages=[
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "old-tool",
                "type": "function",
                "function": {"name": "search_memory", "arguments": '{"query":"乙"}'},
            }]},
            {"role": "tool", "tool_call_id": "old-tool", "content": "结果"},
        ],
        tools=[{"type": "function", "function": {
            "name": "search_memory",
            "description": "检索",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }}],
    )

    body = session.request["json"]
    assert session.request["url"] == "https://example.test/api/coding/v1/messages"
    assert body["system"] == "系统"
    assert body["tools"][0]["input_schema"]["properties"]["query"]["type"] == "string"
    assert body["messages"][-1]["content"][0] == {
        "type": "tool_result", "tool_use_id": "old-tool", "content": "结果",
    }
    assert result.tool_calls[0].function.name == "search_memory"
    assert json.loads(result.tool_calls[0].function.arguments) == {"query": "甲"}
    assert client.last_thinking == "先查记忆"
    assert client.resolved_model == "deepseek-v4-flash-resolved"


def test_volcengine_adapter_ignores_machine_proxy_by_default():
    client = VolcengineAnthropicClient("secret", "https://example.test", "model")
    assert client.session.trust_env is False


def test_volcengine_adapter_returns_text_without_exposing_thinking():
    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"model": "resolved", "content": [
                {"type": "thinking", "thinking": "内部推理"},
                {"type": "text", "text": '{"replies":["行"]}'},
            ]}

    session = SimpleNamespace(post=lambda *args, **kwargs: Response())
    client = VolcengineAnthropicClient("secret", "https://example.test", "model", session)
    assert client.chat(messages=[{"role": "user", "content": "回复"}]) == '{"replies":["行"]}'
    assert client.last_thinking == "内部推理"


def test_generation_inputs_keep_self_and_group_senders_but_exclude_target():
    candidate = _candidate("group", timestamp=1_700_000_004)
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localId": 1, "localType": 1, "createTime": 1_700_000_001, "isSend": True, "content": "我前面说过"},
            {"localId": 2, "localType": 1, "createTime": 1_700_000_002, "isSend": False, "senderDisplayName": "甲", "senderUsername": "wx_a", "content": "第一句"},
            {"localId": 3, "localType": 1, "createTime": 1_700_000_003, "isSend": False, "senderDisplayName": "乙", "senderUsername": "wx_b", "content": "第二句"},
            {"localId": 4, "localType": 1, "createTime": 1_700_000_004, "isSend": True, "content": "真人答案"},
            {"localId": 5, "localType": 1, "createTime": 1_700_000_005, "isSend": False, "senderDisplayName": "甲", "content": "未来消息"},
        ],
    }

    to_reply, all_messages, metadata = rebuild_generation_inputs(candidate, payload)

    assert [message.text for message in all_messages] == ["我前面说过", "第一句", "第二句"]
    assert [message.sender_type for message in all_messages] == [
        SenderType.SELF,
        SenderType.OTHER,
        SenderType.OTHER,
    ]
    assert [message.sender for message in to_reply] == ["甲", "乙"]
    assert [message.text for message in to_reply] == ["第一句", "第二句"]
    assert metadata["self_history_count"] == 1
    assert metadata["distinct_other_senders"] == 2
    assert "真人答案" not in json.dumps([message.__dict__ for message in all_messages], ensure_ascii=False, default=str)


def test_generation_inputs_can_limit_unread_to_compact_verified_human_turn():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["当前问题", "当前补充"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_010,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localId": 1, "localType": 1, "createTime": 1_700_000_001, "isSend": True, "content": "很早的回复"},
            {"localId": 2, "localType": 1, "createTime": 1_700_000_002, "isSend": False, "senderDisplayName": "甲", "content": "旧话题一"},
            {"localId": 3, "localType": 1, "createTime": 1_700_000_003, "isSend": False, "senderDisplayName": "乙", "content": "旧话题二"},
            {"localId": 4, "localType": 1, "createTime": 1_700_000_008, "isSend": False, "senderDisplayName": "丙", "content": "当前问题"},
            {"localId": 5, "localType": 1, "createTime": 1_700_000_009, "isSend": False, "senderDisplayName": "丙", "content": "当前补充"},
            {"localId": 6, "localType": 1, "createTime": 1_700_000_010, "isSend": True, "content": "真人答案"},
        ],
    }

    normal_unread, _, normal_metadata = rebuild_generation_inputs(candidate, payload)
    compact_unread, all_messages, compact_metadata = rebuild_generation_inputs(
        candidate,
        payload,
        compact_verified_turn=True,
    )

    assert [message.text for message in normal_unread] == ["旧话题一", "旧话题二", "当前问题", "当前补充"]
    assert [message.text for message in compact_unread] == ["当前问题", "当前补充"]
    assert [message.text for message in all_messages] == ["很早的回复", "旧话题一", "旧话题二", "当前问题", "当前补充"]
    assert normal_metadata["compact_verified_turn"] is False
    assert compact_metadata["compact_verified_turn"] is True
    assert compact_metadata["unread_count"] == 2


def test_generation_inputs_reject_compact_turn_that_does_not_match_verified_context():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["实际并不存在的当前问题"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_003,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localId": 1, "localType": 1, "createTime": 1_700_000_002, "isSend": False, "senderDisplayName": "甲", "content": "真实当前问题"},
            {"localId": 2, "localType": 1, "createTime": 1_700_000_003, "isSend": True, "content": "真人答案"},
        ],
    }

    with pytest.raises(ValueError, match="verified context mismatch"):
        rebuild_generation_inputs(candidate, payload, compact_verified_turn=True)


def test_generation_inputs_rebuild_frozen_subsequence_without_interleaved_other_text():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["冻结问题", "冻结补充"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_005,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localId": 1, "localType": 1, "createTime": 1_700_000_001, "isSend": True, "content": "更早本人消息"},
            {"localId": 2, "localType": 1, "createTime": 1_700_000_002, "isSend": False, "senderDisplayName": "甲", "content": "冻结问题"},
            {"localId": 3, "localType": 1, "createTime": 1_700_000_003, "isSend": False, "senderDisplayName": "乙", "content": "冻结集未包含的夹入消息"},
            {"localId": 4, "localType": 1, "createTime": 1_700_000_004, "isSend": False, "senderDisplayName": "甲", "content": "冻结补充"},
            {"localId": 5, "localType": 1, "createTime": 1_700_000_005, "isSend": True, "content": "真人答案"},
        ],
    }

    to_reply, all_messages, metadata = rebuild_generation_inputs(
        candidate, payload, compact_verified_turn=True,
    )

    assert [message.text for message in to_reply] == ["冻结问题", "冻结补充"]
    assert [message.text for message in all_messages] == ["更早本人消息", "冻结问题", "冻结补充"]
    assert metadata["compact_omitted_text_count"] == 1


def test_generation_inputs_reject_frozen_subsequence_that_would_hide_self_text():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["冻结问题", "冻结补充"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_004,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localId": 1, "localType": 1, "createTime": 1_700_000_001, "isSend": False, "senderDisplayName": "甲", "content": "冻结问题"},
            {"localId": 2, "localType": 1, "createTime": 1_700_000_002, "isSend": True, "content": "不能隐藏的本人消息"},
            {"localId": 3, "localType": 1, "createTime": 1_700_000_003, "isSend": False, "senderDisplayName": "甲", "content": "冻结补充"},
            {"localId": 4, "localType": 1, "createTime": 1_700_000_004, "isSend": True, "content": "真人答案"},
        ],
    }

    with pytest.raises(ValueError, match="verified context mismatch"):
        rebuild_generation_inputs(candidate, payload, compact_verified_turn=True)


def test_generation_inputs_accept_known_export_formatting_and_redaction_differences():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["@韭菜三号 小建", "问下 @[联系人]"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_003,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localId": 1, "localType": 1, "createTime": 1_700_000_001, "isSend": False, "senderDisplayName": "甲", "content": "@韭菜三号\\u2005小建"},
            {"localId": 2, "localType": 1, "createTime": 1_700_000_002, "isSend": False, "senderDisplayName": "乙", "content": "问下 @王芊"},
            {"localId": 3, "localType": 1, "createTime": 1_700_000_003, "isSend": True, "content": "真人答案"},
        ],
    }

    to_reply, _, _ = rebuild_generation_inputs(candidate, payload, compact_verified_turn=True)

    assert [message.text for message in to_reply] == ["@韭菜三号\\u2005小建", "问下 @王芊"]


def test_generation_inputs_validate_mixed_speakers_in_reverse_order_and_skip_media():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["前一个问题", "我先回答", "最后追问"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_010,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localType": 1, "createTime": 1_700_000_010, "isSend": True, "content": "真人答案"},
            {"localType": 1, "createTime": 1_700_000_009, "isSend": False, "content": "最后追问"},
            {"localType": 49, "createTime": 1_700_000_008, "isSend": False, "content": "<msg><title>媒体消息</title></msg>"},
            {"localType": 1, "createTime": 1_700_000_007, "isSend": True, "content": "我先回答"},
            {"localType": 1, "createTime": 1_700_000_006, "isSend": False, "content": "前一个问题"},
            {"localType": 1, "createTime": 1_700_000_005, "isSend": False, "content": "更早内容"},
        ],
    }

    to_reply, all_messages, metadata = rebuild_generation_inputs(
        candidate,
        payload,
        compact_verified_turn=True,
    )

    assert [message.text for message in to_reply] == ["最后追问"]
    assert [message.text for message in all_messages if message.message_type == "text"] == [
        "更早内容",
        "前一个问题",
        "我先回答",
        "最后追问",
    ]
    assert metadata["unread_count"] == 1


def test_generation_inputs_ignore_media_markers_in_verified_context():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["前一个问题", "[语音]", "我先回答", "[位置]", "最后追问"],
        replies=["真人答案"],
        score=1,
        timestamp=1_700_000_010,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localType": 1, "createTime": 1_700_000_010, "isSend": True, "content": "真人答案"},
            {"localType": 1, "createTime": 1_700_000_009, "isSend": False, "content": "最后追问"},
            {"localType": 48, "createTime": 1_700_000_008, "isSend": False, "content": "<msg><location/></msg>"},
            {"localType": 1, "createTime": 1_700_000_007, "isSend": True, "content": "我先回答"},
            {"localType": 34, "createTime": 1_700_000_006, "isSend": False, "content": "<msg><voice/></msg>"},
            {"localType": 1, "createTime": 1_700_000_005, "isSend": False, "content": "前一个问题"},
        ],
    }

    to_reply, _, metadata = rebuild_generation_inputs(
        candidate,
        payload,
        compact_verified_turn=True,
    )

    assert [message.text for message in to_reply] == ["最后追问"]
    assert metadata["unread_count"] == 1


def test_generation_inputs_use_verified_order_for_equal_timestamps():
    candidate = Candidate(
        relation="group",
        chat_id="group",
        context=["first", "second"],
        replies=["reply"],
        score=1,
        timestamp=20,
    )
    payload = {
        "session": {"displayName": "测试群", "wxid": "room@chatroom"},
        "messages": [
            {"localType": 1, "createTime": 20, "isSend": True, "content": "reply"},
            {"localType": 1, "createTime": 10, "isSend": False, "content": "first"},
            {"localType": 1, "createTime": 10, "isSend": False, "content": "second"},
        ],
    }

    to_reply, _, _ = rebuild_generation_inputs(candidate, payload, compact_verified_turn=True)

    assert [message.text for message in to_reply] == ["first", "second"]


def test_blind_packet_hides_source_identity_and_separates_answer_key():
    blind, answers = build_blind_packet([
        {
            "case_id": "stable-source-id",
            "relationship": "private",
            "context_original": ["在吗"],
            "human_reply_original": ["咋了"],
            "ai_reply_original": ["我在，请问有什么事？"],
        }
    ])
    case = blind["cases"][0]
    assert case["case_id"].startswith("case_")
    assert case["case_id"] != "stable-source-id"
    assert "stable-source-id" not in json.dumps(blind, ensure_ascii=False)
    assert "human_option" not in case
    actual = answers["cases"][0]["human_option"]
    assert case[f"option_{actual}"] == ["咋了"]
    assert answers["blind_sha256"] == _json_sha256(blind)


def test_blind_packet_neutralizes_c0_chat_sections_without_losing_messages():
    prompt = """<session>会话</session>
<context><participant_memory>未来资料</participant_memory></context>
<history><message index="2" role="other" type="text">前文</message></history>
<unread><message index="1" role="other" type="text" handled="false">问题</message></unread>
<consumed_self_replies reuse="forbidden"><message index="1" role="self" type="text">失联48小时</message></consumed_self_replies>
<rules><rule>生成规则</rule></rules>
<few_shot_examples>风格答案</few_shot_examples>"""
    extracted = extract_judge_chat_context(prompt)
    assert extracted == [
        '<conversation_history>\n<message role="self" type="text">失联48小时</message>\n'
        '<message role="other" type="text">前文</message>\n</conversation_history>',
        '<messages_to_reply>\n<message role="other" type="text">问题</message>\n</messages_to_reply>',
    ]

    blind, _ = build_blind_packet([{
        "case_id": "source",
        "relationship": "private",
        "context_original": ["不应使用的旧上下文"],
        "human_reply_original": ["真人"],
        "ai_reply_original": ["候选"],
        "c0_trace": {"generation_trace": {"user_prompt": prompt}},
    }])
    serialized = json.dumps(blind["cases"][0]["context_original"], ensure_ascii=False)
    assert "前文" in serialized and "问题" in serialized and "失联48小时" in serialized
    for excluded in ("未来资料", "生成规则", "风格答案", "不应使用的旧上下文"):
        assert excluded not in serialized
    for leaked_hint in ("consumed_self_replies", 'reuse=\\"forbidden\\"', 'handled=\\"false\\"'):
        assert leaked_hint not in serialized


def test_reference_pack_excludes_entire_test_chats_and_target_replies():
    test = _candidate("test-chat")
    same_chat = _candidate("test-chat")
    same_chat.replies = ["同聊天参考"]
    duplicate = _candidate("other-duplicate")
    duplicate.replies = list(test.replies)
    reference_row = _candidate("reference-chat")
    reference_row.replies = ["独立真人参考"]
    reference_row.source_provenance = "before_automation_cutoff"
    pack = build_judge_reference_pack(
        [test, same_chat, duplicate, reference_row],
        excluded_chat_ids={test.chat_id},
        excluded_replies={"".join(test.replies)},
        limit=10,
    )

    validate_reference_isolation(pack, [test])
    serialized = json.dumps(pack, ensure_ascii=False)
    assert "独立真人参考" in serialized
    assert "同聊天参考" not in serialized
    assert "看到了" not in serialized
    assert pack["isolation"]["whole_test_chats_excluded"] is True
    assert pack["facts"][0]["source"] == "user_confirmed_2026-08-17"


def test_reference_isolation_rejects_target_reply_leak():
    test = _candidate("test-chat")
    pack = build_judge_reference_pack(
        [_candidate("reference-chat")],
        excluded_chat_ids={test.chat_id},
        excluded_replies=set(),
    )
    pack["examples"].append({"human_reply": list(test.replies)})
    with pytest.raises(ValueError, match="target reply"):
        validate_reference_isolation(pack, [test])


def test_reference_isolation_scans_scenario_context_options_reason_and_boundary():
    base = {
        "case_id": "target",
        "chat_type": "group",
        "context": [{"content": "现在进度怎么样"}],
        "option_A": ["大概什么时间点可以重新上门调整"],
        "option_B": ["收到，我问一下"],
    }
    for field in ("option_A", "reason", "boundary"):
        scenario = {
            "relationships": ["group"],
            "context": ["独立上下文"],
            "option_A": ["另一个真人答案"],
            "option_B": ["另一个 AI 答案"],
            "reason": "独立理由",
            "boundary": "独立边界",
        }
        scenario[field] = base["option_A"] if field == "option_A" else base["option_A"][0]
        leaks = reference_leakage_details({"scenario_few_shots": [scenario]}, [base])
        assert any(row["reference_path"].endswith(field) or f".{field}[" in row["reference_path"] for row in leaks)


def test_reference_isolation_rejects_context_and_answer_inside_long_history():
    case = {
        "case_id": "target",
        "chat_type": "group",
        "context": [{"content": "王主管走一下自由流程"}, {"content": "收到"}],
        "option_A": ["大概什么时间点可以重新上门调整"],
        "option_B": ["好的"],
    }
    history = (
        '<conversation_history><message role="other">甲（2026-01-01 10:00）：前文</message>'
        '<message role="other">乙（2026-01-01 10:01）：王主管走一下自由流程</message>'
        '<message role="other">甲（2026-01-01 10:02）：收到</message>'
        '<message role="self">我（2026-01-01 10:03）：大概什么时间点可以重新上门调整</message>'
        '</conversation_history>'
    )
    leaks = reference_leakage_details({"scenario_few_shots": [{
        "relationships": ["group"], "context": [history],
    }]}, [case])
    assert any(row["kind"] == "context_and_answer_sequence" for row in leaks)


def test_reference_isolation_ignores_common_short_reply_without_context_match():
    case = {
        "case_id": "short", "chat_type": "private", "context": ["在吗"],
        "option_A": ["好的"], "option_B": ["嗯"],
    }
    reference = {"examples": [{"context": ["别的话题里普通出现好的两个字"], "human_reply": ["另一个回答"]}]}
    assert reference_leakage_details(reference, [case]) == []


def test_judge_prompt_contains_reference_but_no_answer_or_source_identity():
    blind, _ = build_blind_packet([{
        "case_id": "secret-source-id",
        "relationship": "private",
        "context_original": ["在吗"],
        "human_reply_original": ["咋了"],
        "ai_reply_original": ["我在"],
    }], reference_sha256="ref-hash")
    reference = {
        "facts": [{"fact": "本人老家是江苏泰州", "source": "user_confirmed"}],
        "examples": [{"relationship": "private", "context": ["吃啥"], "human_reply": ["随便"]}],
    }
    prompt = build_codex_judge_prompt(blind, reference)
    assert "本人老家是江苏泰州" in prompt
    assert "随便" in prompt
    assert "secret-source-id" not in prompt
    assert '"source_case_id"' not in prompt
    assert '"human_option": "A"' not in prompt
    assert '"human_option": "B"' not in prompt


def test_surface_baselines_report_only_non_ties():
    blind = {"cases": [
        {"case_id": "one", "option_A": ["短"], "option_B": ["明显更长的回复"]},
        {"case_id": "two", "option_A": ["一", "二"], "option_B": ["一样"]},
    ]}
    answers = {"cases": [
        {"case_id": "one", "human_option": "A"},
        {"case_id": "two", "human_option": "B"},
    ]}
    result = score_surface_baselines(blind, answers)
    assert result["rules"]["shorter_characters"]["correct"] == 1
    assert result["rules"]["shorter_characters"]["decided"] == 1
    assert result["rules"]["fewer_messages"]["correct"] == 1
    assert result["rules"]["fewer_messages"]["decided"] == 1
    assert result["position_baselines"] == {"always_A": 0.5, "always_B": 0.5}


def test_blind_packet_balances_human_option_positions():
    cases = [
        {
            "case_id": f"source-{index}",
            "relationship": "private",
            "context_original": ["在吗"],
            "human_reply_original": [f"真人-{index}"],
            "ai_reply_original": [f"AI-{index}"],
        }
        for index in range(30)
    ]
    _, answers = build_blind_packet(cases)
    counts = Counter(row["human_option"] for row in answers["cases"])
    assert counts == {"A": 15, "B": 15}


@pytest.mark.parametrize("trace_field", ["system_prompt", "user_prompt", "few_shot_content", "tool_calls"])
def test_target_answer_in_any_generation_trace_invalidates_case(trace_field):
    output = {"replies": ["另一条回复"], "trace": {trace_field: "这里出现真人答案"}}
    reasons = detect_degenerate_case(["真人答案"], output)
    assert any("appeared" in reason for reason in reasons)


def test_near_copy_of_human_reply_invalidates_case():
    output = {"replies": ["这个答案确实就是这样啊"], "trace": {}}
    reasons = detect_degenerate_case(["这个答案确实就是这样呀"], output)
    assert any("reproduced" in reason for reason in reasons)


def test_score_predictions_reports_accuracy_for_legacy_packet():
    answers = {"cases": [{"case_id": "one", "human_option": "B"}, {"case_id": "two", "human_option": "A"}]}
    result = score_predictions(answers, {"cases": [{"case_id": "one", "human_option": "B"}]})
    assert result["status"] == "incomplete"
    assert result["accuracy"] == 1.0
    assert result["correct"] == 1


def test_api_judge_parser_recovers_unambiguous_option_from_malformed_json():
    from scripts.evaluate_human_ai_judge import _parse_api_judge_response

    result = _parse_api_judge_response(
        '{"human_option":"B","confidence":0.8,"reason":"像本人说的"话""}'
    )

    assert result["human_option"] == "B"
    assert result["confidence"] == 0.8
    assert result["reason"] == '像本人说的"话"'


def test_score_api_judge_predictions_requires_one_isolated_call_per_case():
    blind = {"batch_id": "batch", "cases": [
        {"case_id": "one", "option_A": ["甲"], "option_B": ["乙"]},
        {"case_id": "two", "option_A": ["丙"], "option_B": ["丁"]},
    ]}
    blind_sha = _json_sha256(blind)
    answers = {
        "batch_id": "batch", "blind_sha256": blind_sha, "reference_sha256": "ref",
        "cases": [{"case_id": "one", "human_option": "A"}, {"case_id": "two", "human_option": "B"}],
    }
    predictions = {
        "batch_id": "batch", "blind_sha256": blind_sha, "reference_sha256": "ref", "invocation_id": "run",
        "cases": [{"case_id": "one", "human_option": "A"}, {"case_id": "two", "human_option": "A"}],
    }
    calls = [{
        "case_id": case_id, "other_test_case_ids_in_request": [], "answer_fields_in_request": [],
        "tools_sent": False, "status_code": 200,
    } for case_id in ("one", "two")]
    audit = {"invocation_id": "run", "blind_sha256": blind_sha, "calls": calls}

    result = score_api_judge_predictions(blind, answers, predictions, audit)
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["isolation_verified"] is True

    audit["calls"][0]["other_test_case_ids_in_request"] = ["two"]
    with pytest.raises(ValueError, match="isolation"):
        score_api_judge_predictions(blind, answers, predictions, audit)


def test_schema_v2_rejects_batch_hash_and_event_log_mismatches():
    blind, answers = build_blind_packet([
        {
            "case_id": "source",
            "relationship": "private",
            "context_original": ["在吗"],
            "human_reply_original": ["咋了"],
            "ai_reply_original": ["我在"],
        }
    ])
    events = [{"type": "item.completed", "item": {"type": "message"}}]
    base = {
        "batch_id": answers["batch_id"],
        "blind_sha256": answers["blind_sha256"],
        "judge_events_sha256": _json_sha256(events),
        "cases": [{"case_id": blind["cases"][0]["case_id"], "human_option": "A"}],
    }
    for key, bad_value in (("batch_id", "wrong"), ("blind_sha256", "wrong"), ("judge_events_sha256", "wrong")):
        predictions = dict(base)
        predictions[key] = bad_value
        with pytest.raises(ValueError):
            score_predictions(answers, predictions, events)


def test_judge_event_log_with_tool_call_is_rejected():
    with pytest.raises(ValueError, match="tool call"):
        validate_judge_events([{"type": "item.completed", "item": {"type": "command_execution"}}])


def test_run_codex_judge_uses_ephemeral_restricted_invocation(monkeypatch, tmp_path):
    blind_path = tmp_path / "blind.json"
    predictions_path = tmp_path / "predictions.json"
    events_path = tmp_path / "events.jsonl"
    audit_path = tmp_path / "audit.json"
    blind = {"schema_version": 2, "batch_id": "batch_test", "cases": []}
    blind_path.write_text(json.dumps(blind), encoding="utf-8")
    captured = {}

    def fake_run(command, input, text, capture_output, check):
        captured["command"] = command
        captured["prompt"] = input
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump({"cases": []}, file)
        event = {"type": "item.completed", "item": {"type": "message"}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(event) + "\n", stderr="")

    monkeypatch.setattr("scripts.evaluate_human_ai_judge.subprocess.run", fake_run)
    run_codex_judge(blind_path, predictions_path, events_path, audit_path)

    command = captured["command"]
    for argument in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "read-only",
        "--skip-git-repo-check",
        "--json",
    ):
        assert argument in command
    assert command[command.index("--cd") + 1].startswith("/tmp/") or "human-ai-judge-" in command[command.index("--cd") + 1]
    assert str(blind_path) not in captured["prompt"]
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert predictions["batch_id"] == "batch_test"
    assert predictions["blind_sha256"] == _json_sha256(blind)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["evidence_status"] == "captured_at_invocation"
    assert audit["prompt"] == build_codex_judge_prompt(blind)
    assert audit["argv"] == command
    assert audit["stdout"]
    assert audit["stderr"] == ""
    assert audit["tool_call_count"] == 0
    assert audit_path.stat().st_mode & 0o777 == 0o600


def test_insufficient_cases_do_not_overwrite_existing_artifacts(tmp_path):
    paths = [tmp_path / name for name in ("replay.json", "blind.json", "answers.json")]
    for path in paths:
        path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(RuntimeError, match="insufficient valid cases"):
        write_benchmark_artifacts(
            [],
            {"cases": []},
            minimum_valid=1,
            replay_path=paths[0],
            blind_path=paths[1],
            answers_path=paths[2],
        )

    assert [path.read_text(encoding="utf-8") for path in paths] == ["sentinel"] * 3
