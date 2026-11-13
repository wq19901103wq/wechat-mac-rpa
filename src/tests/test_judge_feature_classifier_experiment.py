import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_judge_feature_classifier_experiment as experiment
from scripts.build_human_ai_judge_benchmark_v2 import candidate_record
from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_human_ai_judge import _json_sha256


def _candidate(
    case: int,
    *,
    relation: str = "acquaintance",
    chat_id: str | None = None,
    context: list[str] | None = None,
    replies: list[str] | None = None,
) -> Candidate:
    return Candidate(
        relation=relation,
        chat_id=chat_id or f"chat-{case}",
        context=context or [f"上下文消息-{case}-足够长且彼此不同"],
        replies=replies or [f"真人回复-{case}"],
        score=1.0,
        timestamp=1_700_000_000 + case,
    )


def _option(seed: int) -> dict:
    option = {field: seed % 4 for field in experiment.INTEGER_FIELDS}
    option.update({field: bool(seed % 2) for field in experiment.BOOLEAN_FIELDS})
    option.update({field: values[seed % len(values)] for field, values in experiment.ENUM_FIELDS.items()})
    return option


def _features(left: int = 3, right: int = 0) -> dict:
    return {"option_A": _option(left), "option_B": _option(right)}


def _guard(signatures: list[dict] | None = None) -> dict:
    guard = {
        "schema_version": 1,
        "kind": "judge_feature_classifier_holdout_hash_guard",
        "privacy": "hashes only",
        "case_count": 160,
        "relationship_counts": {"private": 80, "group": 80},
        "signatures": signatures or [],
    }
    guard["guard_sha256"] = _json_sha256(guard)
    return guard


def test_feature_classifier_results_are_not_marked_as_official_judge(tmp_path):
    path = tmp_path / "result.json"

    experiment._write_feature_classifier_result(path, {"accuracy": 0.75})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["experiment_type"] == "feature_classifier_experiment"
    assert saved["official_judge_baseline"] is False
    assert saved["accuracy"] == 0.75


def _stable_validation_packets() -> tuple[dict, dict, dict]:
    reference = {
        "schema_version": 1,
        "batch_id": experiment.STABLE_VALIDATION_BATCH_ID,
        "facts": [],
        "examples": [],
    }
    reference["reference_sha256"] = _json_sha256(reference)
    cases = []
    answer_cases = []
    for index in range(160):
        relationship = "private" if index < 80 else "group"
        case_id = f"stable-{index:03d}"
        cases.append({
            "case_id": case_id,
            "split": "validation",
            "relationship": relationship,
            "context": [f"固定上下文-{index}"],
            "option_A": [f"固定候选甲-{index}"],
            "option_B": [f"固定候选乙-{index}"],
        })
        answer_cases.append({
            "case_id": case_id,
            "split": "validation",
            "human_option": "A",
        })
    blind = {
        "schema_version": 1,
        "batch_id": experiment.STABLE_VALIDATION_BATCH_ID,
        "reference_sha256": reference["reference_sha256"],
        "cases": cases,
    }
    answers = {
        "schema_version": 1,
        "batch_id": experiment.STABLE_VALIDATION_BATCH_ID,
        "reference_sha256": reference["reference_sha256"],
        "cases": answer_cases,
    }
    answers["blind_sha256"] = _json_sha256(blind)
    return reference, blind, answers


def test_protocol_uses_exact_expanded_sample_counts():
    assert experiment.TRAIN_TARGETS == {"private": 600, "group": 600}
    assert experiment.VALIDATION_TARGETS == {"private": 80, "group": 80}
    assert experiment.TRAIN_TOTAL == 1200
    assert experiment.VALIDATION_TOTAL == 160
    assert experiment.LEARNING_CURVE_TOTALS == (480, 800, 1200)


def test_holdout_guard_contains_hashes_only_and_requires_80_per_relation(tmp_path):
    replay = {
        "cases": [
            {
                "case_id": f"private-{index}",
                "source_chat_id": f"private-chat-{index}",
                "relationship": "private",
                "timestamp": 1_600_000_000 + index,
                "context_original": [f"私聊上下文明文-{index}"],
                "human_reply_original": [f"私聊答案明文-{index}"],
            }
            for index in range(80)
        ]
        + [
            {
                "case_id": f"group-{index}",
                "source_chat_id": f"group-chat-{index}",
                "relationship": "group",
                "timestamp": 1_700_000_000 + index,
                "context_original": [f"群聊上下文明文-{index}"],
                "human_reply_original": [f"群聊答案明文-{index}"],
            }
            for index in range(80)
        ]
    }
    replay_path = tmp_path / "replay.json"
    output_path = tmp_path / "guard.json"
    replay_path.write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")

    guard = experiment.seal_holdout_guard(replay_path, output_path)

    serialized = output_path.read_text(encoding="utf-8")
    assert guard["case_count"] == 160
    assert guard["relationship_counts"] == {"private": 80, "group": 80}
    assert "私聊上下文明文" not in serialized
    assert "群聊答案明文" not in serialized
    experiment.validate_guard(guard)


def test_same_chat_nonoverlapping_history_is_allowed():
    older = _candidate(1, chat_id="same-chat", context=["较早且不重叠的完整消息"])
    current = _candidate(2, chat_id="same-chat", context=["当前且不重叠的完整消息"])

    assert experiment.signatures_overlap(
        experiment.event_signature(older), experiment.event_signature(current)
    ) == []


def test_same_chat_pair_or_long_message_overlap_is_rejected():
    long_message = "这是长度达到十二个字符以上的同一条消息"
    left = _candidate(1, chat_id="same-chat", context=["第一句", "第二句", long_message])
    pair = _candidate(2, chat_id="same-chat", context=["第一句", "第二句", "另一句"])
    long_only = _candidate(3, chat_id="same-chat", context=["开头", long_message])

    pair_reasons = experiment.signatures_overlap(
        experiment.event_signature(left), experiment.event_signature(pair)
    )
    long_reasons = experiment.signatures_overlap(
        experiment.event_signature(left), experiment.event_signature(long_only)
    )
    assert "overlapping_message_pair" in pair_reasons
    assert "overlapping_long_message" in long_reasons


def test_few_shot_exclusion_covers_overlap_and_answer_leak_without_blocking_chat(tmp_path):
    rows = [
        {
            "id": "same-time",
            "relationship": "private",
            "chat_id": "same-chat",
            "timestamp": 1_700_000_001,
            "context": ["不重合消息"],
            "reply": ["其他回复"],
        },
        {
            "id": "same-pair",
            "relationship": "private",
            "chat_id": "same-chat",
            "timestamp": 1_600_000_000,
            "context": ["第一句", "第二句", "其他尾句"],
            "reply": ["另一回复"],
        },
        {
            "id": "same-long-message",
            "relationship": "private",
            "chat_id": "same-chat",
            "timestamp": 1_600_000_001,
            "context": ["这是一条超过十二个字符的重合消息"],
            "reply": ["再一回复"],
        },
        {
            "id": "same-answer-other-chat",
            "relationship": "private",
            "chat_id": "other-chat",
            "timestamp": 1_600_000_002,
            "context": ["完全不同的上下文"],
            "reply": ["目标真人回复"],
        },
        {
            "id": "same-chat-nonoverlap",
            "relationship": "private",
            "chat_id": "same-chat",
            "timestamp": 1_600_000_003,
            "context": ["较早且完全不重合的聊天事件"],
            "reply": ["无关回复"],
        },
    ]
    path = tmp_path / "few_shots.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    candidate = _candidate(
        1,
        chat_id="same-chat",
        context=["第一句", "第二句", "这是一条超过十二个字符的重合消息"],
        replies=["目标真人回复"],
    )

    excluded = experiment.few_shot_exclude_ids(
        candidate, experiment.load_few_shot_exclusion_index(path)
    )

    assert excluded == {
        "same-time", "same-pair", "same-long-message", "same-answer-other-chat"
    }


def test_few_shot_exclusion_normalizes_answer_case(tmp_path):
    path = tmp_path / "few_shots.jsonl"
    path.write_text(
        json.dumps({
            "id": "case-folded-answer",
            "relationship": "group",
            "chat_id": "historical-chat",
            "timestamp": 1_600_000_000,
            "context": ["无关历史上下文"],
            "reply": ["OK"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    candidate = _candidate(
        1,
        chat_id="target-chat",
        context=["当前上下文"],
        replies=["ok"],
    )

    excluded = experiment.few_shot_exclude_ids(
        candidate, experiment.load_few_shot_exclusion_index(path)
    )

    assert excluded == {"case-folded-answer"}


def test_input_leak_diagnostic_ignores_output_and_current_case_fields():
    output = {
        "few_shot_ids": ["few-shot-1"],
        "trace": {
            "few_shot_content": "示例里没有答案",
            "system_prompt": "系统规则",
            "llm_calls": [{"response_preview": "目标真人回复"}],
            "user_prompt": "本题上下文里有人说目标真人回复",
            "latest_request_grounding": {"reply_after": ["目标真人回复"]},
        },
    }

    diagnostic = experiment.injected_material_diagnostic(["目标真人回复"], output)

    assert diagnostic == {"material_types": [], "few_shot_ids": []}


def test_input_leak_diagnostic_requires_exact_labeled_short_reply():
    generic = {
        "few_shot_ids": ["few-shot-1"],
        "trace": {"few_shot_content": "规则里顺带提到好的这个短词"},
    }
    labeled = {
        "few_shot_ids": ["few-shot-2"],
        "trace": {
            "few_shot_content": (
                "<example><context><message>怎么样</message></context>"
                "<response><message>好的</message></response></example>"
            )
        },
    }

    assert experiment.injected_material_diagnostic(["好的"], generic) == {
        "material_types": [],
        "few_shot_ids": [],
    }
    assert experiment.injected_material_diagnostic(["好的"], labeled) == {
        "material_types": ["few_shot"],
        "few_shot_ids": ["few-shot-2"],
    }


def test_input_leak_diagnostic_flags_long_unlabeled_reference_substring():
    output = {
        "trace": {"system_prompt": "禁止直接输出这是长度超过八个字符的目标答案"},
    }

    diagnostic = experiment.injected_material_diagnostic(
        ["这是长度超过八个字符的目标答案"], output
    )

    assert diagnostic == {
        "material_types": ["system_prompt"],
        "few_shot_ids": [],
    }


def test_experiment_degenerate_detection_keeps_real_reply_reproduction():
    output = {"replies": ["完全一样的真人回复"], "trace": {"llm_calls": []}}

    assert experiment.detect_experiment_degenerate_case(
        ["完全一样的真人回复"], output
    ) == ["C0 reproduced the human reply (similarity=1.000)"]


def test_diagnostic_client_records_empty_and_exception_without_content():
    class Response:
        status_code = 400

        @staticmethod
        def json():
            return {
                "error": {
                    "type": "invalid_request_error",
                    "code": "BadRequest",
                    "message": "max_tokens must be less than the maximum allowed value",
                }
            }

    class Client:
        model = "test-model"

        def __init__(self):
            self.calls = 0
            self.last_status_code = None

        def chat(self, **kwargs):
            assert kwargs["raise_on_error"] is True
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content="", tool_calls=[])
            self.last_status_code = 400
            error = TimeoutError("secret response and prompt must not be stored")
            error.response = Response()
            raise error

    client = experiment.DiagnosticChatClient(Client())

    messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "不能保存的请求"},
    ]
    assert client.chat(messages=messages, max_tokens=512).content == ""
    assert client.chat(messages=messages, tools=[{"secret": "不能保存"}], max_tokens=10000) == ""
    calls = client.diagnostic_calls()

    assert calls[0]["stage"] == "skill_router"
    assert calls[0]["attempt"] == 1
    assert calls[0]["returned_empty_content"] is True
    assert calls[0]["exception_type"] is None
    assert calls[0]["request_message_count"] == 2
    assert calls[0]["request_character_count"] == 9
    assert calls[0]["system_prompt_character_count"] == 2
    assert calls[0]["tool_definition_count"] == 0
    assert calls[0]["max_tokens"] == 512
    assert calls[1]["stage"] == "generation"
    assert calls[1]["attempt"] == 1
    assert calls[1]["returned_empty_content"] is True
    assert calls[1]["exception_type"] == "TimeoutError"
    assert calls[1]["http_status_code"] == 400
    assert calls[1]["tool_definition_count"] == 1
    assert calls[1]["server_error_type"] == "invalid_request_error"
    assert calls[1]["server_error_code"] == "BadRequest"
    assert calls[1]["server_error_category"] == "max_tokens_limit"
    assert "secret" not in json.dumps(calls)
    assert "不能保存" not in json.dumps(calls, ensure_ascii=False)


def test_c0_failure_diagnostic_marks_budget_exhausted_before_first_call():
    client = experiment.DiagnosticChatClient(SimpleNamespace())

    diagnostic = experiment.c0_failure_diagnostic(
        {"generation_failed": True}, client
    )

    assert diagnostic == {
        "llm_call_count": 0,
        "llm_calls": [],
        "budget_exhausted_before_first_llm_call": True,
    }


def test_feature_request_contains_one_anonymous_case_and_no_answer_metadata():
    case = {
        "case_id": "must-not-leak",
        "source_case_id": "must-not-leak-either",
        "human_option": "B",
        "relationship": "group",
        "context_original": ["群聊上下文"],
        "option_A": ["候选甲"],
        "option_B": ["候选乙"],
    }

    prompt = experiment.feature_extractor_prompt(case)

    assert "must-not-leak" not in prompt
    assert '"human_option"' not in prompt
    assert '"source_case_id"' not in prompt
    assert prompt.count("<case>") == 1
    assert prompt.count("</case>") == 1
    assert '"option_A"' in prompt and '"option_B"' in prompt
    assert "answer_now" in prompt


def test_feature_schema_is_strict_and_ab_symmetric():
    parsed = experiment.parse_feature_response(json.dumps(_features()))
    assert set(parsed) == {"option_A", "option_B"}
    assert set(parsed["option_A"]) == experiment.OPTION_FIELDS

    invalid = _features()
    invalid["option_A"]["free_text_reason"] = "真人"
    with pytest.raises(ValueError, match="fixed schema"):
        experiment.parse_feature_response(json.dumps(invalid))


def test_feature_response_json_schema_matches_fixed_feature_schema():
    schema = experiment.feature_response_json_schema()

    assert schema["required"] == ["option_A", "option_B"]
    for option in ("option_A", "option_B"):
        option_schema = schema["properties"][option]
        assert set(option_schema["required"]) == experiment.OPTION_FIELDS
        assert option_schema["additionalProperties"] is False


def test_feature_extractor_retries_only_a_schema_failure(monkeypatch):
    responses = [
        '{"option_A": {}}',
        json.dumps(_features()),
    ]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(responses.pop(0), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    features, audit = experiment.run_codex_feature_extractor(
        "只输出 JSON", timeout_seconds=30, model=None, reasoning_effort=None
    )

    assert features == _features()
    assert len(calls) == 2
    assert all("--output-schema" in command for command in calls)
    assert audit["attempt_count"] == 2
    assert audit["schema_retry_count"] == 1


def test_swapping_options_strictly_negates_feature_vector():
    features = _features(3, 0)
    vector, names = experiment.vectorize_pair(features)
    swapped, swapped_names = experiment.vectorize_pair(
        {"option_A": features["option_B"], "option_B": features["option_A"]}
    )

    assert names == swapped_names
    assert np.array_equal(swapped, -vector)


def test_boolean_option_encoding_uses_only_threshold_and_one_hot_bits():
    vector, names = experiment.vectorize_boolean_option(_option(2))

    assert set(vector.tolist()) <= {0.0, 1.0}
    for field in experiment.INTEGER_FIELDS:
        indexes = [names.index(f"{field}>={threshold}") for threshold in (1, 2, 3)]
        assert vector[indexes].tolist() == [1.0, 1.0, 0.0]


def test_speaker_timeline_preserves_self_and_other_message_order():
    context = [
        '<conversation_history>'
        '<message role="other">甲（2026-08-28 10:00）：第一句</message>'
        '<message role="self">我（2026-08-28 10:01）：第二句</message>'
        '</conversation_history>',
        '<latest_messages>'
        '<message role="other">乙（2026-08-28 10:02）：第三句</message>'
        '</latest_messages>',
    ]

    assert experiment.speaker_timeline(context) == ("甲", "__self__", "乙")


def test_context_metadata_uses_answer_source_id_instead_of_packet_position():
    blind = {
        "cases": [
            {
                "case_id": "blind-first",
                "split": "train",
                "relationship": "group",
                "context_original": [
                    '<conversation_history><message role="other">甲（2026-08-28 10:00）：一</message></conversation_history>'
                ],
            },
            {
                "case_id": "blind-second",
                "split": "validation",
                "relationship": "private",
                "context_original": [
                    '<conversation_history><message role="self">我（2026-08-28 10:00）：二</message></conversation_history>'
                ],
            },
        ]
    }
    answers = {
        "blind_sha256": _json_sha256(blind),
        "cases": [
            {
                "case_id": "blind-first",
                "source_case_id": "source-z",
                "split": "train",
            },
            {
                "case_id": "blind-second",
                "source_case_id": "source-a",
                "split": "validation",
            },
        ],
    }
    replay = {
        "cases": [
            {
                "case_id": "source-a",
                "relationship": "private",
                "source_chat_id": "private-chat",
                "c0_trace": {"input_metadata": {"chat_name": "私聊联系人"}},
            },
            {
                "case_id": "source-z",
                "relationship": "group",
                "source_chat_id": "group-chat",
                "c0_trace": {"input_metadata": {"chat_name": "测试群聊"}},
            },
        ]
    }

    metadata = experiment.load_context_metadata(blind, answers, replay)

    assert metadata["blind-first"]["source_chat_id"] == "group-chat"
    assert metadata["blind-first"]["group_name"] == "测试群聊"
    assert metadata["blind-second"]["source_chat_id"] == "private-chat"


def test_context_schema_uses_training_rows_only():
    metadata = {
        "train": {
            "split": "train",
            "relationship": "group",
            "source_chat_id": "train-group",
            "group_name": "训练群",
            "recent_speakers": ("甲",),
        },
        "validation": {
            "split": "validation",
            "relationship": "group",
            "source_chat_id": "validation-group",
            "group_name": "验证群",
            "recent_speakers": ("乙",),
        },
    }

    schema = experiment.build_context_boolean_schema(
        metadata,
        group_name_min_train_cases=1,
        group_member_min_train_occurrences=1,
    )

    assert schema["known_group_names"] == ("训练群",)
    assert schema["known_group_members"] == (("train-group", "甲"),)


def test_context_encoding_contains_boolean_identity_timeline_and_switch_bits():
    schema = {
        "known_group_names": ("测试群聊",),
        "known_group_members": (("group-chat", "甲"),),
    }
    metadata = {
        "relationship": "group",
        "source_chat_id": "group-chat",
        "group_name": "测试群聊",
        "recent_speakers": ("甲", "__self__", "甲"),
    }

    vector, names = experiment.vectorize_context_booleans(metadata, schema)
    encoded = dict(zip(names, vector.tolist()))

    assert set(vector.tolist()) <= {0.0, 1.0}
    assert encoded["is_private"] == 0.0
    assert encoded["is_group"] == 1.0
    assert encoded["is_private"] + encoded["is_group"] == 1.0
    assert encoded[experiment._private_category_name("group_name", "测试群聊")] == 1.0
    assert encoded["recent_1_is_other"] == 1.0
    assert encoded["recent_2_is_self"] == 1.0
    assert encoded["recent_1_same_speaker_as_3"] == 1.0
    assert encoded["latest_turn_switches_speaker"] == 1.0
    member_name = experiment._private_category_name(
        "recent_1_group_member", "group-chat\0甲"
    )
    assert encoded[member_name] == 1.0


def test_boolean_context_crosses_are_binary_and_condition_each_option():
    option_a = np.asarray([1.0, 0.0])
    option_b = np.asarray([0.0, 1.0])
    context = np.asarray([1.0, 0.0])

    expanded_a, names = experiment.expand_boolean_option_with_context(
        option_a, ["x", "y"], context, ["private", "group"]
    )
    expanded_b, other_names = experiment.expand_boolean_option_with_context(
        option_b, ["x", "y"], context, ["private", "group"]
    )

    assert names == other_names
    assert set(expanded_a.tolist() + expanded_b.tolist()) <= {0.0, 1.0}
    assert expanded_a.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert expanded_b.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    coefficients = np.asarray([0.2, -0.1, 0.8, -0.4, 2.0, 2.0])
    probability = experiment.expit((expanded_a - expanded_b) @ coefficients)
    swapped = experiment.expit((expanded_b - expanded_a) @ coefficients)
    assert probability + swapped == pytest.approx(1.0, abs=1e-12)


def test_observable_option_features_are_clear_booleans():
    values, names = experiment.vectorize_observable_option_booleans(
        ["@甲 你看12？", "哈哈"],
        {
            "relationship": "group",
            "source_chat_id": "group-chat",
            "recent_speakers": ("甲", "__self__"),
        },
    )
    encoded = dict(zip(names, values.tolist()))

    assert names == list(experiment.OBSERVABLE_OPTION_BOOLEAN_NAMES)
    assert set(values.tolist()) <= {0.0, 1.0}
    assert encoded["reply_bubble_count>=2"] == 1.0
    assert encoded["reply_contains_question_mark"] == 1.0
    assert encoded["reply_contains_at_mention"] == 1.0
    assert encoded["reply_contains_digit"] == 1.0
    assert encoded["reply_mentions_latest_speaker"] == 1.0
    assert encoded["reply_contains_laughter_marker"] == 1.0


def test_group_pattern_feature_recognizes_repeated_confirmation():
    metadata = {
        "relationship": "group",
        "recent_group_other_texts": (
            "甲（2026-08-28 10:00）：确认",
            "乙（2026-08-28 10:01）：确认",
        ),
    }

    assert experiment.reply_reuses_active_group_pattern(["确认"], metadata)


def test_group_pattern_feature_recognizes_distinctive_sentence_skeleton():
    metadata = {
        "relationship": "group",
        "recent_group_other_texts": (
            "甲（2026-08-28 10:00）：只要是牛奶，我就要。不挑不捡，只求速度！！",
        ),
    }

    assert experiment.reply_reuses_active_group_pattern(
        ["同 只要是吃的 我就要"], metadata
    )
    assert not experiment.reply_reuses_active_group_pattern(
        ["危机感拉满"], metadata
    )


def test_group_pattern_feature_rejects_ordinary_partial_overlap_and_private_chat():
    group_metadata = {
        "relationship": "group",
        "recent_group_other_texts": (
            "甲（2026-08-28 10:00）：今天下午天气不错",
        ),
    }
    private_metadata = {
        "relationship": "private",
        "recent_group_other_texts": ("确认", "确认"),
    }

    assert not experiment.reply_reuses_active_group_pattern(
        ["今天下午去开会吧"], group_metadata
    )
    assert not experiment.reply_reuses_active_group_pattern(["确认"], private_metadata)


def test_group_pattern_vector_is_boolean():
    values, names = experiment.vectorize_group_pattern_option_boolean(
        ["确认"],
        {
            "relationship": "group",
            "recent_group_other_texts": ("确认", "确认"),
        },
    )

    assert names == list(experiment.GROUP_PATTERN_OPTION_BOOLEAN_NAMES)
    assert values.tolist() == [1.0]


def test_group_pattern_pair_matrix_is_ab_symmetric_and_keeps_72d_baseline():
    baseline_names = ["base_boolean", "observable_boolean"]
    candidate_names = [
        *baseline_names,
        *experiment.GROUP_PATTERN_OPTION_BOOLEAN_NAMES,
    ]
    row = (
        "case",
        "group",
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([0.0]),
        np.asarray([]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([1.0]),
        np.asarray([0.0]),
        1,
    )
    swapped = (
        "case",
        "group",
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([]),
        np.asarray([1.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        0,
    )

    baseline = experiment._pair_variant_matrix(
        [row], "refined_boolean_option_only", baseline_names, []
    ).toarray()
    candidate = experiment._pair_variant_matrix(
        [row], "refined_boolean_group_pattern", candidate_names, []
    ).toarray()
    swapped_candidate = experiment._pair_variant_matrix(
        [swapped], "refined_boolean_group_pattern", candidate_names, []
    ).toarray()

    assert baseline.shape == (1, 2)
    assert candidate.shape == (1, 3)
    assert np.array_equal(swapped_candidate, -candidate)


def test_group_pattern_holdout_manifest_is_fixed_240_with_80_private_160_group():
    manifest = experiment._load_group_pattern_holdout_manifest(
        experiment.DEFAULT_GROUP_PATTERN_HOLDOUT_MANIFEST
    )

    relationships = [
        "group" if row.get("chat_type") == "group" else "private"
        for row in manifest["cases"]
    ]
    assert len(manifest["cases"]) == 240
    assert relationships.count("private") == 80
    assert relationships.count("group") == 160


def test_group_pattern_holdout_candidate_keeps_string_reply_as_one_bubble():
    candidate = experiment._group_pattern_holdout_candidate({
        "chat_type": "group",
        "chat_id": "group-chat",
        "context": ["上下文"],
        "reply": "真人完整回复",
        "timestamp": 1_700_000_000,
    })

    assert candidate.replies == ["真人完整回复"]


def test_group_pattern_holdout_checkpoint_binds_current_c0(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(experiment, "current_c0_baseline_id", lambda: "c0-current")
    monkeypatch.setattr(experiment, "current_c0_sha256", lambda: "generator-current")

    checkpoint = experiment._new_group_pattern_holdout_checkpoint(manifest_path)

    assert checkpoint["current_c0_baseline_id"] == "c0-current"
    assert checkpoint["c0_generator_sha256"] == "generator-current"
    assert checkpoint["generation_input_contract"] == {
        "compact_verified_turn": True
    }


def _group_pattern_holdout_row(
    case_id: str,
    *,
    relationship: str,
    chat_id: str | None = None,
    context: list[str] | None = None,
    reply: str | None = None,
) -> dict:
    return {
        "internal_id": case_id,
        "chat_type": "group" if relationship == "group" else "private",
        "chat_id": chat_id or f"chat-{case_id}",
        "context": context or [f"上下文-{case_id}-足够长且唯一"],
        "reply": reply or f"真人回复-{case_id}",
        "timestamp": 1_700_000_000 + sum(map(ord, case_id)),
    }


def test_effective_group_pattern_holdout_keeps_valid_and_replaces_only_safe_cases():
    original_private = _group_pattern_holdout_row(
        "original-private", relationship="private"
    )
    original_group = _group_pattern_holdout_row(
        "original-group", relationship="group"
    )
    overlapping = _group_pattern_holdout_row(
        "overlapping-private",
        relationship="private",
        chat_id=original_private["chat_id"],
        context=original_private["context"],
    )
    replacement = _group_pattern_holdout_row(
        "replacement-private", relationship="private"
    )

    selected, audit = experiment._select_effective_group_pattern_holdout_cases(
        [original_private, original_group],
        [overlapping, replacement],
        [],
        {"original-private"},
        lambda row: True,
        targets={"private": 1, "group": 1},
    )

    assert [row["internal_id"] for row in selected] == [
        "original-group",
        "replacement-private",
    ]
    assert audit["replacements"] == {"private": 1}
    assert audit["rejected_candidates"] == {"private:overlap": 1}


def test_effective_group_pattern_holdout_checkpoint_only_reuses_exact_valid_entry():
    row = _group_pattern_holdout_row("case-one", relationship="group")
    candidate = experiment._group_pattern_holdout_candidate(row)
    compatible = {
        "status": "valid",
        "relationship": "group",
        "replay_case": {
            "case_id": row["internal_id"],
            "source_chat_id": candidate.chat_id,
            "relationship": candidate.relation,
            "timestamp": candidate.timestamp,
            "human_reply_original": candidate.replies,
        },
    }

    assert experiment._compatible_group_pattern_holdout_entry(row, compatible)
    assert not experiment._compatible_group_pattern_holdout_entry(
        row, {**compatible, "status": "generation_failed"}
    )
    changed = json.loads(json.dumps(compatible))
    changed["replay_case"]["human_reply_original"] = ["另一条答案"]
    assert not experiment._compatible_group_pattern_holdout_entry(row, changed)


def test_effective_group_pattern_holdout_manifest_uses_canonical_checksum(tmp_path):
    cases = [
        _group_pattern_holdout_row(f"private-{index}", relationship="private")
        for index in range(80)
    ] + [
        _group_pattern_holdout_row(f"group-{index}", relationship="group")
        for index in range(160)
    ]
    manifest = {
        "schema_version": 1,
        "kind": "judge_group_pattern_holdout_effective_v1",
        "source_manifest_sha256": experiment.GROUP_PATTERN_HOLDOUT_MANIFEST_SHA256,
        "cases": cases,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    path = tmp_path / "effective.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert experiment._load_group_pattern_holdout_manifest(path) == manifest


def test_holdout_boolean_rows_require_frozen_training_schema():
    blind = {"cases": []}
    answers = {"blind_sha256": _json_sha256(blind), "cases": []}
    checkpoint = {
        "blind_sha256": _json_sha256(blind),
        "entries": {},
    }

    with pytest.raises(ValueError, match="frozen training context schema"):
        experiment.load_boolean_context_rows(
            blind,
            answers,
            {"cases": []},
            checkpoint,
            expected_split_counts={"holdout": 0},
        )


def test_holdout_boolean_rows_do_not_build_schema(monkeypatch):
    blind = {"cases": []}
    answers = {"blind_sha256": _json_sha256(blind), "cases": []}
    checkpoint = {
        "blind_sha256": _json_sha256(blind),
        "entries": {},
    }
    monkeypatch.setattr(
        experiment,
        "build_context_boolean_schema",
        lambda metadata: pytest.fail("holdout must not build its own schema"),
    )

    rows, *_ = experiment.load_boolean_context_rows(
        blind,
        answers,
        {"cases": []},
        checkpoint,
        context_schema={"known_group_names": (), "known_group_members": ()},
        expected_split_counts={"holdout": 0},
    )

    assert rows == {"holdout": []}


def test_refined_context_crosses_are_whitelisted_not_cartesian():
    group_name = experiment._private_category_name("group_name", "测试群聊")
    option_names = [
        "tone=playful",
        "reply_bubble_count>=2",
        "addressee_fit>=2",
        "reply_mentions_latest_speaker",
        "unrelated_feature",
    ]
    context_names = [
        "is_group",
        "recent_1_is_other",
        group_name,
    ]

    crosses = experiment.refined_boolean_context_crosses(
        option_names, context_names
    )

    assert ("is_group", "tone=playful") in crosses
    assert ("recent_1_is_other", "addressee_fit>=2") in crosses
    assert (group_name, "reply_bubble_count>=2") in crosses
    assert (group_name, "unrelated_feature") not in crosses
    assert len(crosses) < len(option_names) * len(context_names)


def test_refined_pair_matrix_is_ab_symmetric_and_label_independent():
    option_names = ["tone=playful", "reply_bubble_count>=2"]
    context_names = ["is_group"]
    left = np.asarray([1.0])
    right = np.asarray([0.0])
    observable_left = np.asarray([0.0])
    observable_right = np.asarray([1.0])
    context = np.asarray([1.0])
    row = (
        "case",
        "group",
        np.asarray([0.0]),
        np.asarray([0.0]),
        left,
        right,
        context,
        observable_left,
        observable_right,
        1,
    )
    swapped = (
        "case",
        "group",
        np.asarray([0.0]),
        np.asarray([0.0]),
        right,
        left,
        context,
        observable_right,
        observable_left,
        0,
    )
    relabeled = (*row[:-1], 0)

    matrix = experiment._pair_variant_matrix(
        [row],
        "refined_boolean_context_cross",
        option_names,
        context_names,
    ).toarray()
    swapped_matrix = experiment._pair_variant_matrix(
        [swapped],
        "refined_boolean_context_cross",
        option_names,
        context_names,
    ).toarray()
    relabeled_matrix = experiment._pair_variant_matrix(
        [relabeled],
        "refined_boolean_context_cross",
        option_names,
        context_names,
    ).toarray()
    _, feature_names = experiment.expand_boolean_option_with_refined_context(
        np.zeros(len(option_names)),
        option_names,
        np.zeros(len(context_names)),
        context_names,
    )
    coefficients = np.arange(1, matrix.shape[1] + 1, dtype=float)
    probability = experiment.expit(matrix[0] @ coefficients)
    swapped_probability = experiment.expit(swapped_matrix[0] @ coefficients)

    assert matrix.shape[1] == len(feature_names)
    assert np.array_equal(swapped_matrix, -matrix)
    assert np.array_equal(relabeled_matrix, matrix)
    assert probability + swapped_probability == pytest.approx(1.0, abs=1e-12)


def test_refined_decision_rejects_crosses_against_refined_boolean_baseline():
    variants = {
        "legacy_numeric_42": {"validation": {"correct": 131}},
        "boolean_option_only": {"validation": {"correct": 130}},
        "refined_boolean_option_only": {"validation": {"correct": 142}},
        "refined_boolean_context_cross": {"validation": {"correct": 140}},
    }
    context_increment = {
        "net": -2,
        "relationships": {
            "private": {"net": -5},
            "group": {"net": 3},
        },
    }
    routing_validation = {
        "correct": 145,
        "total": 160,
        "relationships": {
            "private": {"correct": 73, "total": 80},
            "group": {"correct": 72, "total": 80},
        },
    }

    summary = experiment._refined_boolean_decision_summary(
        variants, context_increment, routing_validation
    )

    assert summary["best_single_shared_variant"]["variant"] == (
        "refined_boolean_option_only"
    )
    assert summary["context_cross_candidate"] == {
        "baseline_variant": "refined_boolean_option_only",
        "variant": "refined_boolean_context_cross",
        "comparison": context_increment,
        "decision": "reject",
    }
    assert summary["best_observed_routing_candidate"]["validation"] == (
        routing_validation
    )
    assert summary["best_observed_routing_candidate"]["status"] == (
        "requires_independent_validation"
    )
    assert summary["best_observed_routing_candidate"]["adopted"] is False
    assert summary["decision"] == "retain_refined_boolean_option_only"


def test_reply_bubble_count_difference_is_ab_symmetric():
    case = {"option_A": ["第一条", "第二条"], "option_B": ["一条"]}
    swapped = {"option_A": case["option_B"], "option_B": case["option_A"]}

    difference = experiment.reply_bubble_count_difference(case)

    assert difference == 1.0
    assert experiment.reply_bubble_count_difference(swapped) == -difference


def test_feature_matrix_can_add_reply_bubble_count_without_changing_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 1)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 1)
    cases = [
        {
            "case_id": "train",
            "split": "train",
            "relationship": "private",
            "option_A": ["一", "二"],
            "option_B": ["一"],
        },
        {
            "case_id": "validation",
            "split": "validation",
            "relationship": "group",
            "option_A": ["一"],
            "option_B": ["一", "二", "三"],
        },
    ]
    blind = {"cases": cases}
    answers = {
        "blind_sha256": _json_sha256(blind),
        "cases": [
            {"case_id": row["case_id"], "split": row["split"], "human_option": "A"}
            for row in cases
        ],
    }
    checkpoint = {
        "blind_sha256": _json_sha256(blind),
        "entries": {
            row["case_id"]: {"status": "valid", "features": _features()}
            for row in cases
        },
    }

    rows, names = experiment.load_feature_matrix(
        blind, answers, checkpoint, include_reply_bubble_count=True
    )

    assert names[-1] == "reply_bubble_count"
    assert rows["train"][0][2][-1] == 1.0
    assert rows["validation"][0][2][-1] == -2.0


def test_shared_mlp_keeps_options_separate_and_is_ab_symmetric():
    features = _features(3, 0)
    left, right, names = experiment.vectorize_options(features)
    hidden_units = 2
    parameters = np.zeros(hidden_units * len(names) + hidden_units + hidden_units)
    parameters[0] = 1.0
    parameters[hidden_units * len(names)] = -0.5
    parameters[-2:] = [1.0, -0.5]
    model = {
        "parameters": parameters,
        "mean": np.zeros(len(names)),
        "scale": np.ones(len(names)),
        "input_dim": len(names),
        "hidden_units": hidden_units,
    }

    probability = experiment.shared_pair_mlp_probability(
        model, left[None, :], right[None, :]
    )[0]
    swapped = experiment.shared_pair_mlp_probability(
        model, right[None, :], left[None, :]
    )[0]

    assert len(names) == 42
    assert 0.0 < probability < 1.0
    assert probability + swapped == pytest.approx(1.0, abs=1e-12)


def test_explicit_crosses_keep_raw_features_and_add_each_pair_once():
    values = np.asarray([2.0, 3.0, 5.0])

    expanded, names = experiment.expand_option_with_pairwise_crosses(
        values, ["first", "second", "third"]
    )

    assert names == [
        "first", "second", "third",
        "first*second", "first*third", "second*third",
    ]
    assert expanded.tolist() == [2.0, 3.0, 5.0, 6.0, 10.0, 15.0]


def test_explicit_cross_pair_score_is_ab_symmetric():
    features = _features(3, 0)
    left, right, names = experiment.vectorize_options(features)
    expanded_left, expanded_names = experiment.expand_option_with_pairwise_crosses(
        left, names
    )
    expanded_right, right_names = experiment.expand_option_with_pairwise_crosses(
        right, names
    )
    coefficients = np.linspace(-0.5, 0.5, num=len(expanded_names))

    probability = experiment.expit((expanded_left - expanded_right) @ coefficients)
    swapped = experiment.expit((expanded_right - expanded_left) @ coefficients)

    assert expanded_names == right_names
    assert len(names) == 42
    assert len(expanded_names) == 903
    assert probability + swapped == pytest.approx(1.0, abs=1e-12)


def test_deep_shared_mlp_keeps_options_separate_and_is_ab_symmetric():
    features = _features(3, 0)
    left, right, names = experiment.vectorize_options(features)
    hidden_units = (3, 2)
    parameter_count = (
        hidden_units[0] * len(names)
        + hidden_units[0]
        + hidden_units[1] * hidden_units[0]
        + hidden_units[1]
        + hidden_units[1]
    )
    parameters = np.linspace(-0.2, 0.2, num=parameter_count)
    model = {
        "parameters": parameters,
        "mean": np.zeros(len(names)),
        "scale": np.ones(len(names)),
        "input_dim": len(names),
        "hidden_units": hidden_units,
    }

    probability = experiment.deep_shared_pair_mlp_probability(
        model, left[None, :], right[None, :]
    )[0]
    swapped = experiment.deep_shared_pair_mlp_probability(
        model, right[None, :], left[None, :]
    )[0]

    assert 0.0 < probability < 1.0
    assert probability + swapped == pytest.approx(1.0, abs=1e-12)


def test_small_model_matrix_contains_only_numeric_feature_differences(monkeypatch):
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 2)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 2)
    cases = [
        {"case_id": "train-a", "split": "train", "relationship": "private"},
        {"case_id": "train-b", "split": "train", "relationship": "group"},
        {"case_id": "validation-a", "split": "validation", "relationship": "private"},
        {"case_id": "validation-b", "split": "validation", "relationship": "group"},
    ]
    blind = {"cases": cases}
    answers = {
        "blind_sha256": _json_sha256(blind),
        "cases": [
            {"case_id": row["case_id"], "split": row["split"], "human_option": "A"}
            for row in cases
        ],
    }
    checkpoint = {
        "blind_sha256": _json_sha256(blind),
        "entries": {
            row["case_id"]: {"status": "valid", "features": _features()}
            for row in cases
        },
    }

    rows, _ = experiment.load_feature_matrix(blind, answers, checkpoint)

    for split_rows in rows.values():
        for case_id, relationship, vector, label in split_rows:
            assert isinstance(case_id, str)
            assert relationship in {"private", "group"}
            assert vector.dtype.kind == "f"
            assert all(isinstance(value, np.floating) for value in vector)
            assert label == 1


def test_human_option_b_flips_label_to_zero(monkeypatch):
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 1)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 1)
    blind = {
        "cases": [
            {"case_id": "train", "split": "train", "relationship": "private"},
            {"case_id": "validation", "split": "validation", "relationship": "group"},
        ]
    }
    answers = {
        "blind_sha256": _json_sha256(blind),
        "cases": [
            {"case_id": "train", "split": "train", "human_option": "A"},
            {"case_id": "validation", "split": "validation", "human_option": "B"},
        ],
    }
    checkpoint = {
        "blind_sha256": _json_sha256(blind),
        "entries": {
            case_id: {"status": "valid", "features": _features()}
            for case_id in ("train", "validation")
        },
    }

    rows, _ = experiment.load_feature_matrix(blind, answers, checkpoint)

    assert rows["train"][0][3] == 1
    assert rows["validation"][0][3] == 0


def test_split_gate_rejects_wrong_counts():
    with pytest.raises(ValueError, match="train must contain exactly 1200"):
        experiment.validate_split_records({"train": [], "validation": []}, _guard())


def test_initialize_deduplicates_answers_across_private_and_group(tmp_path, monkeypatch):
    candidates = [
        _candidate(1, replies=["跨关系重复答案"]),
        _candidate(2),
        _candidate(3),
        _candidate(4, relation="group", replies=["跨关系重复答案"]),
        _candidate(5, relation="group"),
        _candidate(6, relation="group"),
    ]
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 2)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 2)
    monkeypatch.setattr(experiment, "TRAIN_TARGETS", {"private": 1, "group": 1})
    monkeypatch.setattr(experiment, "VALIDATION_TARGETS", {"private": 1, "group": 1})
    monkeypatch.setattr(experiment, "eligible_candidates", lambda *args: (candidates, {}))
    monkeypatch.setattr(
        experiment,
        "round_robin",
        lambda rows, relation: [
            row for row in rows
            if ("group" if row.relation == "group" else "private") == relation
        ],
    )
    monkeypatch.setattr(experiment, "load_current_c0", lambda: {"few_shot_path": "fixed.json"})
    monkeypatch.setattr(experiment, "current_c0_baseline_id", lambda: "c0-test")
    monkeypatch.setattr(experiment, "current_c0_sha256", lambda: "generator-test")
    guard_path = tmp_path / "guard.json"
    output_path = tmp_path / "manifest.json"
    guard_path.write_text(json.dumps(_guard(), ensure_ascii=False), encoding="utf-8")

    manifest = experiment.initialize_manifest(
        guard_path, output_path, tmp_path / "exports", tmp_path / "wiki"
    )

    group_validation = [
        row for row in manifest["splits"]["validation"]
        if row["relationship"] == "group"
    ]
    assert group_validation[0]["replies"] == ["真人回复-5"]


def test_split_gate_rejects_holdout_and_train_validation_event_overlap(monkeypatch):
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 2)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 2)
    monkeypatch.setattr(experiment, "TRAIN_TARGETS", {"private": 1, "group": 1})
    monkeypatch.setattr(experiment, "VALIDATION_TARGETS", {"private": 1, "group": 1})
    shared_long = "这是一条用于验证事件重合的长消息"
    validation_private = _candidate(
        1, chat_id="shared-chat", context=["验证前文", shared_long]
    )
    validation_group = _candidate(2, relation="group")
    train_private = _candidate(
        3, chat_id="shared-chat", context=[shared_long, "训练后文"]
    )
    train_group = _candidate(4, relation="group")
    overlapping_splits = {
        "validation": [candidate_record(validation_private), candidate_record(validation_group)],
        "train": [candidate_record(train_private), candidate_record(train_group)],
    }

    with pytest.raises(ValueError, match="overlapping event"):
        experiment.validate_split_records(overlapping_splits, _guard())

    clean_splits = {
        "validation": [candidate_record(_candidate(11)), candidate_record(_candidate(12, relation="group"))],
        "train": [candidate_record(_candidate(13)), candidate_record(_candidate(14, relation="group"))],
    }
    holdout_signature = experiment.event_signature(_candidate(11))
    with pytest.raises(ValueError, match="overlaps fixed holdout"):
        experiment.validate_split_records(clean_splits, _guard([holdout_signature]))


def test_feature_checkpoint_resumes_without_reprocessing_valid_case(tmp_path, monkeypatch):
    blind = {
        "cases": [
            {
                "case_id": "already-valid",
                "split": "train",
                "relationship": "private",
                "context_original": ["上下文一"],
                "option_A": ["甲"],
                "option_B": ["乙"],
            },
            {
                "case_id": "pending",
                "split": "train",
                "relationship": "group",
                "context_original": ["上下文二"],
                "option_A": ["丙"],
                "option_B": ["丁"],
            },
        ]
    }
    blind_path = tmp_path / "blind.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    blind_path.write_text(json.dumps(blind, ensure_ascii=False), encoding="utf-8")
    checkpoint = experiment._new_feature_checkpoint(blind)
    checkpoint["entries"]["already-valid"] = {
        "status": "valid",
        "features": _features(),
    }
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    prompts: list[str] = []

    def fake_extract(prompt, timeout_seconds, model, reasoning_effort):
        prompts.append(prompt)
        return _features(), {"duration_seconds": 0.01}

    monkeypatch.setattr(experiment, "run_codex_feature_extractor", fake_extract)

    status = experiment.extract_features(
        blind_path,
        checkpoint_path,
        max_cases=None,
        retry_failed=False,
        workers=1,
        timeout_seconds=30,
        model=None,
        reasoning_effort=None,
    )

    assert status["processed_this_run"] == 1
    assert len(prompts) == 1
    assert "上下文二" in prompts[0]
    assert "上下文一" not in prompts[0]


def test_c0_shards_are_disjoint_and_cover_every_row():
    rows = [("train", {"case_id": f"case-{index}"}) for index in range(17)]
    shards = [experiment.shard_rows(rows, index, 4) for index in range(4)]

    case_ids = [row[1]["case_id"] for shard in shards for row in shard]
    assert len(case_ids) == 17
    assert len(set(case_ids)) == 17
    assert set(case_ids) == {f"case-{index}" for index in range(17)}
    assert [len(shard) for shard in shards] == [5, 4, 4, 4]

    with pytest.raises(ValueError, match="shard index"):
        experiment.shard_rows(rows, 4, 4)


def test_current_c0_runner_forces_compact_verified_turn(monkeypatch):
    captured: dict = {}

    def fake_run_c0(candidate, payload, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(experiment, "_run_c0", fake_run_c0)

    result = experiment.run_current_c0_for_judge_experiment(
        _candidate(1), {"messages": []}, object(), {"few-shot-1"}
    )

    assert result == {"ok": True}
    assert captured["compact_verified_turn"] is True
    assert captured["force_reply"] is True
    assert captured["few_shot_exclude_ids"] == {"few-shot-1"}


@pytest.mark.parametrize("contract", [None, {"compact_verified_turn": False}])
def test_c0_checkpoint_rejects_missing_or_wrong_input_contract(
    tmp_path, monkeypatch, contract
):
    manifest = {"manifest_sha256": "manifest"}
    checkpoint = {
        "schema_version": 1,
        "manifest_sha256": "manifest",
        "current_c0_baseline_id": "c0",
        "entries": {},
    }
    if contract is not None:
        checkpoint["generation_input_contract"] = contract
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    monkeypatch.setattr(experiment, "current_c0_baseline_id", lambda: "c0")

    with pytest.raises(ValueError, match="generation input contract"):
        experiment._load_c0_checkpoint(path, manifest)


@pytest.mark.parametrize(
    ("old_context", "old_history_count", "expected"),
    [
        (["当前消息"], 3, True),
        (["错误消息"], 3, False),
        (["当前消息"], 4, False),
    ],
)
def test_effective_input_compatibility_requires_exact_context_and_history_count(
    monkeypatch, old_context, old_history_count, expected
):
    monkeypatch.setattr(
        experiment,
        "rebuild_generation_inputs",
        lambda candidate, payload, compact_verified_turn: (
            [SimpleNamespace(text="当前消息")],
            [object(), object(), object()],
            {"unread_count": 1},
        ),
    )
    entry = {
        "replay_case": {
            "context_original": old_context,
            "c0_trace": {
                "input_metadata": {
                    "unread_count": 1,
                    "history_count": old_history_count,
                }
            },
        }
    }

    compatible, diagnostic = experiment.effective_input_compatibility(
        entry, _candidate(1), {}
    )

    assert compatible is expected
    assert diagnostic["required_history_count"] == 3


def test_pre_freeze_audit_checks_effective_contract_and_leaks(monkeypatch, tmp_path):
    validation = candidate_record(_candidate(1))
    train = candidate_record(_candidate(2))
    manifest = {"splits": {"validation": [validation], "train": [train]}}

    def entry(row, compact):
        return {
            "status": "valid",
            "replay_case": {
                "source_chat_id": row["chat_id"],
                "relationship": row["relationship"],
                "timestamp": row["timestamp"],
                "human_reply_original": row["replies"],
                "ai_reply_original": ["这是不同的候选回复"],
                "c0_trace": {
                    "few_shot_ids": [],
                    "input_metadata": {"compact_verified_turn": compact},
                    "generation_trace": {},
                },
            },
        }

    checkpoint = {
        "entries": {
            validation["case_id"]: entry(validation, False),
            train["case_id"]: entry(train, True),
        }
    }
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 1)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 1)
    monkeypatch.setattr(
        experiment, "_export_payloads", lambda path: {"chat-1": {}, "chat-2": {}}
    )
    monkeypatch.setattr(
        experiment, "effective_input_compatibility", lambda *args: (True, {})
    )

    result = experiment.audit_c0_checkpoint_for_freeze(
        manifest, checkpoint, tmp_path
    )

    assert result["checked_cases"] == 2
    assert result["input_contract_equivalent"] == 2
    assert result["trace_compact_verified_turn"] == {"false": 1, "true": 1}
    assert result["reference_material_leaks"] == 0


def test_pre_freeze_audit_rejects_reference_material_leak(monkeypatch, tmp_path):
    row = candidate_record(_candidate(1, replies=["目标真人回复足够长"]))
    manifest = {"splits": {"validation": [row], "train": []}}
    checkpoint = {
        "entries": {
            row["case_id"]: {
                "status": "valid",
                "replay_case": {
                    "source_chat_id": row["chat_id"],
                    "relationship": row["relationship"],
                    "timestamp": row["timestamp"],
                    "human_reply_original": row["replies"],
                    "ai_reply_original": ["另一条回复"],
                    "c0_trace": {
                        "few_shot_ids": [],
                        "input_metadata": {"compact_verified_turn": True},
                        "generation_trace": {"system_prompt": "目标真人回复足够长"},
                    },
                },
            }
        }
    }
    monkeypatch.setattr(experiment, "TRAIN_TOTAL", 0)
    monkeypatch.setattr(experiment, "VALIDATION_TOTAL", 1)
    monkeypatch.setattr(experiment, "_export_payloads", lambda path: {"chat-1": {}})
    monkeypatch.setattr(
        experiment, "effective_input_compatibility", lambda *args: (True, {})
    )

    with pytest.raises(ValueError, match="reference material"):
        experiment.audit_c0_checkpoint_for_freeze(manifest, checkpoint, tmp_path)


def test_upgrade_c0_checkpoint_only_copies_owned_compatible_entries(
    tmp_path, monkeypatch
):
    rows = [
        {**candidate_record(_candidate(index)), "case_id": f"case-{index}"}
        for index in range(4)
    ]
    manifest = {
        "manifest_sha256": "manifest",
        "splits": {"validation": rows[:2], "train": rows[2:]},
    }
    source = {
        "schema_version": 1,
        "manifest_sha256": "manifest",
        "current_c0_baseline_id": "c0",
        "entries": {
            f"case-{index}": {
                "status": "valid",
                "replay_case": {"source": index},
            }
            for index in range(4)
        },
    }
    manifest_path = tmp_path / "manifest.json"
    guard_path = tmp_path / "guard.json"
    source_path = tmp_path / "source.json"
    target_path = tmp_path / "target.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    guard_path.write_text(json.dumps(_guard()), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(experiment, "validate_manifest", lambda *args: None)
    monkeypatch.setattr(experiment, "current_c0_baseline_id", lambda: "c0")
    monkeypatch.setattr(
        experiment,
        "_export_payloads",
        lambda path: {row["chat_id"]: {} for row in rows},
    )
    monkeypatch.setattr(
        experiment,
        "effective_input_compatibility",
        lambda entry, candidate, payload: (
            candidate.chat_id == "chat-0",
            {"previous_history_count": 1, "required_history_count": 2},
        ),
    )

    result = experiment.upgrade_c0_checkpoint_input_contract(
        manifest_path,
        guard_path,
        source_path,
        target_path,
        tmp_path,
        shard_index=0,
        shard_count=2,
    )

    checkpoint = json.loads(target_path.read_text(encoding="utf-8"))
    assert set(checkpoint["entries"]) == {"case-0", "case-2"}
    assert checkpoint["entries"]["case-0"] == source["entries"]["case-0"]
    assert checkpoint["entries"]["case-2"]["status"] == "input_contract_mismatch"
    assert checkpoint["generation_input_contract"] == {
        "compact_verified_turn": True
    }
    assert result == {
        "owned_cases": 2,
        "checkpoint_entries": 2,
        "outcomes": {"input_contract_mismatch": 1, "reused_valid": 1},
    }


def test_c0_checkpoint_merge_allows_exact_duplicate_and_rejects_conflict():
    base = {
        "schema_version": 1,
        "manifest_sha256": "manifest",
        "current_c0_baseline_id": "c0",
        "generation_input_contract": experiment.C0_GENERATION_INPUT_CONTRACT,
        "entries": {"case-1": {"status": "valid"}},
    }
    compatible = {
        **base,
        "entries": {
            "case-1": {"status": "valid"},
            "case-2": {"status": "error"},
        },
    }

    merged = experiment.merge_c0_checkpoint_data(base, compatible)
    assert set(merged["entries"]) == {"case-1", "case-2"}

    conflict = {**compatible, "entries": {"case-1": {"status": "error"}}}
    with pytest.raises(ValueError, match="entry conflict: case-1"):
        experiment.merge_c0_checkpoint_data(base, conflict)


def test_c0_checkpoint_merge_rejects_different_manifest():
    base = {
        "schema_version": 1,
        "manifest_sha256": "manifest-a",
        "current_c0_baseline_id": "c0",
        "generation_input_contract": experiment.C0_GENERATION_INPUT_CONTRACT,
        "entries": {},
    }
    source = {**base, "manifest_sha256": "manifest-b"}

    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        experiment.merge_c0_checkpoint_data(base, source)


def test_owned_c0_shard_merge_uses_only_the_manifest_owner(tmp_path, monkeypatch):
    manifest = {
        "manifest_sha256": "manifest",
        "splits": {
            "validation": [
                {"case_id": "case-0", "relationship": "private"},
                {"case_id": "case-1", "relationship": "group"},
            ],
            "train": [
                {"case_id": "case-2", "relationship": "private"},
                {"case_id": "case-3", "relationship": "group"},
            ],
        },
    }
    guard = _guard()
    manifest_path = tmp_path / "manifest.json"
    guard_path = tmp_path / "guard.json"
    target_path = tmp_path / "merged.json"
    source_paths = [tmp_path / "shard-0.json", tmp_path / "shard-1.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    monkeypatch.setattr(experiment, "validate_manifest", lambda *args: None)
    monkeypatch.setattr(experiment, "current_c0_baseline_id", lambda: "c0")
    base = {
        "schema_version": 1,
        "manifest_sha256": "manifest",
        "current_c0_baseline_id": "c0",
        "generation_input_contract": experiment.C0_GENERATION_INPUT_CONTRACT,
    }
    source_paths[0].write_text(json.dumps({
        **base,
        "entries": {
            "case-0": {"status": "valid", "owner": 0},
            "case-1": {"status": "error", "owner": "stale"},
            "case-2": {"status": "valid", "owner": 0},
            "case-3": {"status": "error", "owner": "stale"},
        },
    }), encoding="utf-8")
    source_paths[1].write_text(json.dumps({
        **base,
        "entries": {
            "case-0": {"status": "error", "owner": "stale"},
            "case-1": {"status": "valid", "owner": 1},
            "case-2": {"status": "error", "owner": "stale"},
            "case-3": {"status": "valid", "owner": 1},
        },
    }), encoding="utf-8")

    experiment.merge_owned_c0_shards(
        manifest_path, guard_path, target_path, source_paths
    )

    merged = json.loads(target_path.read_text(encoding="utf-8"))
    assert {case_id: row["owner"] for case_id, row in merged["entries"].items()} == {
        "case-0": 0,
        "case-1": 1,
        "case-2": 0,
        "case-3": 1,
    }


def test_balanced_learning_subset_keeps_exact_private_group_counts():
    vector = np.asarray([1.0])
    rows = [
        (f"private-{index}", "private", vector, index % 2) for index in range(600)
    ] + [
        (f"group-{index}", "group", vector, index % 2) for index in range(600)
    ]

    for total in experiment.LEARNING_CURVE_TOTALS:
        selected = experiment.balanced_learning_subset(rows, total)
        assert len(selected) == total
        assert sum(row[1] == "private" for row in selected) == total // 2
        assert sum(row[1] == "group" for row in selected) == total // 2


def test_stable_validation_projection_preserves_fixed_cases_exactly():
    reference, validation_blind, validation_answers = _stable_validation_packets()
    train_case = {
        "case_id": "train-only",
        "split": "train",
        "relationship": "private",
        "context": ["训练上下文"],
        "option_A": ["训练候选甲"],
        "option_B": ["训练候选乙"],
    }
    development_blind = {
        **validation_blind,
        "batch_id": "development",
        "reference_sha256": "development-reference",
        "cases": [train_case, *validation_blind["cases"]],
    }
    development_answers = {
        "schema_version": 1,
        "batch_id": "development",
        "reference_sha256": "development-reference",
        "cases": [
            {"case_id": "train-only", "split": "train", "human_option": "B"},
            *validation_answers["cases"],
        ],
    }
    development_answers["blind_sha256"] = _json_sha256(development_blind)

    cases, answers = experiment._validation_projection(
        development_blind, development_answers
    )

    assert cases == validation_blind["cases"]
    assert answers == validation_answers["cases"]
    assert [row["relationship"] for row in cases] == [
        *("private" for _ in range(80)),
        *("group" for _ in range(80)),
    ]
    assert reference["batch_id"] == experiment.STABLE_VALIDATION_BATCH_ID


def test_reference_repair_deletes_complete_leaked_example(monkeypatch):
    reference = {
        "schema_version": 1,
        "batch_id": "old",
        "facts": [],
        "examples": [{"id": "keep"}, {"id": "leaked"}, {"id": "also-keep"}],
    }

    def fake_leaks(payload, cases):
        if any(row.get("id") == "leaked" for row in payload.get("examples", [])):
            return [{"reference_path": "reference.examples[1].human_reply[0]"}]
        return []

    monkeypatch.setattr(experiment, "reference_leakage_details", fake_leaks)
    monkeypatch.setattr(experiment, "assert_reference_isolated", lambda *args: None)

    repaired, audit = experiment.repair_reference_for_cases(
        reference, [{"case_id": "one"}], batch_id="fixed"
    )

    assert [row["id"] for row in repaired["examples"]] == ["keep", "also-keep"]
    assert repaired["batch_id"] == "fixed"
    assert audit == {
        "initial_leak_count": 1,
        "removed_example_indexes": [1],
        "removed_example_count": 1,
        "remaining_leak_count": 0,
    }


def test_reference_repair_refuses_non_example_leak(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "reference_leakage_details",
        lambda *args: [{"reference_path": "reference.facts[0].content"}],
    )

    with pytest.raises(ValueError, match="not safely removable"):
        experiment.repair_reference_for_cases(
            {"facts": [{"content": "leak"}], "examples": []},
            [{"case_id": "one"}],
            batch_id="fixed",
        )


@pytest.mark.parametrize("broken_field", ["batch", "reference", "blind"])
def test_stable_validation_artifacts_require_batch_and_hash_binding(broken_field):
    reference, blind, answers = _stable_validation_packets()
    if broken_field == "batch":
        answers["batch_id"] = "wrong"
    elif broken_field == "reference":
        blind["reference_sha256"] = "wrong"
        answers["blind_sha256"] = _json_sha256(blind)
    else:
        answers["blind_sha256"] = "wrong"

    with pytest.raises(ValueError):
        experiment.validate_stable_validation_artifacts(reference, blind, answers)


@pytest.mark.parametrize("failure", ["duplicate", "out_of_order", "audit_mismatch"])
def test_stable_judge_resume_rejects_invalid_partial_checkpoint(tmp_path, failure):
    reference, blind, _ = _stable_validation_packets()
    expected = {
        "batch_id": blind["batch_id"],
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": reference["reference_sha256"],
        "judge_profile_name": experiment.STABLE_PROFILE_NAME,
        "judge_profile_sha256": experiment.STABLE_PROFILE_SHA256,
    }
    ids = [blind["cases"][0]["case_id"], blind["cases"][1]["case_id"]]
    if failure == "duplicate":
        ids = [ids[0], ids[0]]
    elif failure == "out_of_order":
        ids.reverse()
    predictions = {
        **expected,
        "invocation_id": "run-1",
        "cases": [{"case_id": case_id, "human_option": "A"} for case_id in ids],
    }
    audit_ids = ids if failure != "audit_mismatch" else ids[:1]
    audit = {
        **{key: value for key, value in expected.items() if key != "batch_id"},
        "invocation_id": "run-1",
        "calls": [
            {"case_id": case_id, "index": blind["cases"].index(next(
                row for row in blind["cases"] if row["case_id"] == case_id
            )) + 1}
            for case_id in audit_ids
        ],
    }
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError):
        experiment._validate_stable_judge_resume(
            predictions_path, audit_path, reference, blind
        )


def test_compare_stable_judge_classifier_is_case_aligned_and_applies_all_gates(
    tmp_path, monkeypatch
):
    reference, blind, answers = _stable_validation_packets()
    paths = {
        "reference": tmp_path / "reference.json",
        "blind": tmp_path / "blind.json",
        "answers": tmp_path / "answers.json",
        "stable": tmp_path / "stable.json",
        "curve": tmp_path / "curve.json",
        "profile": tmp_path / "profile.json",
        "output": tmp_path / "comparison.json",
    }
    for key, payload in (
        ("reference", reference), ("blind", blind), ("answers", answers)
    ):
        paths[key].write_text(json.dumps(payload), encoding="utf-8")
    paths["profile"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(experiment, "validate_stable_judge_profile", lambda path: {})
    stable_rows = []
    classifier_rows = []
    for index, case in enumerate(blind["cases"]):
        stable_prediction = "B" if index in {0, 80} else "A"
        classifier_prediction = "A" if index in {0, 80} else stable_prediction
        stable_rows.append({"case_id": case["case_id"], "human_option": stable_prediction})
        classifier_rows.append({
            "case_id": case["case_id"],
            "relationship": case["relationship"],
            "predicted_human_option": classifier_prediction,
            "human_option": "A",
            "probability_A": 0.9 if classifier_prediction == "A" else 0.1,
        })
    stable = {
        "batch_id": blind["batch_id"],
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": reference["reference_sha256"],
        "judge_profile_name": experiment.STABLE_PROFILE_NAME,
        "judge_profile_sha256": experiment.STABLE_PROFILE_SHA256,
        "cases": stable_rows,
    }
    curve = {"learning_curve": [{"training_total": 1200, "predictions": classifier_rows}]}
    paths["stable"].write_text(json.dumps(stable), encoding="utf-8")
    paths["curve"].write_text(json.dumps(curve), encoding="utf-8")

    result = experiment.compare_stable_judge_classifier(
        paths["blind"], paths["answers"], paths["reference"], paths["stable"],
        paths["curve"], paths["profile"], paths["output"],
    )

    assert result["scores"]["stable_judge"]["overall"]["correct"] == 158
    assert result["scores"]["feature_classifier"]["overall"]["correct"] == 160
    assert result["scores"]["feature_classifier_net_change"] == {
        "overall": 2, "private": 1, "group": 1
    }
    assert result["changed_cases"]["corrected_count"] == 2
    assert result["changed_cases"]["new_error_count"] == 0
    assert result["decision"] == "adopt"
    assert all(row["passed"] for row in result["promotion_gate"].values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"stable_covered": 159},
        {"overall_net": 0},
    ],
)
def test_each_promotion_gate_failure_rejects(overrides):
    values = {
        "stable_covered": 160,
        "classifier_covered": 160,
        "overall_net": 1,
    }
    values.update(overrides)
    gates = experiment.promotion_gate_results(**values)

    assert sum(not row["passed"] for row in gates.values()) == 1
    assert not all(row["passed"] for row in gates.values())


def test_base_prediction_features_are_symmetric_boolean_differences():
    option_a = experiment._base_prediction_difference_features("A", 0.80)
    option_b = experiment._base_prediction_difference_features("B", 0.80)

    assert option_a.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    np.testing.assert_array_equal(option_b, -option_a)


def test_group_participation_cross_is_one_boolean_per_option():
    option_a = _option(1)
    option_b = _option(1)
    option_a["depends_on_unstated_fact"] = True
    option_a["group_pattern_alignment"] = 1
    option_b["depends_on_unstated_fact"] = False
    option_b["group_pattern_alignment"] = 3

    left = experiment._group_participation_fact_pattern_value(
        option_a, "group", ("other", "__self__")
    )
    right = experiment._group_participation_fact_pattern_value(
        option_b, "group", ("other", "__self__")
    )

    assert left - right == 1.0
    assert experiment._group_participation_fact_pattern_value(
        option_a, "private", ("other", "__self__")
    ) == 0.0
    assert experiment._group_participation_fact_pattern_value(
        option_a, "group", ("other",)
    ) == 0.0


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_base_prediction_feature_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="between 0 and 1"):
        experiment._base_prediction_difference_features("A", confidence)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "out_of_order"])
def test_base_prediction_features_require_complete_ordered_case_ids(failure):
    blind = {
        "cases": [{"case_id": "one"}, {"case_id": "two"}],
    }
    rows = [
        {"case_id": "one", "human_option": "A", "confidence": 0.8},
        {"case_id": "two", "human_option": "B", "confidence": 0.7},
    ]
    if failure == "missing":
        rows = rows[:1]
    elif failure == "duplicate":
        rows[1]["case_id"] = "one"
    else:
        rows.reverse()
    predictions = {
        "blind_sha256": _json_sha256(blind),
        "judge_profile_name": experiment.STABLE_PROFILE_NAME,
        "judge_profile_sha256": experiment.STABLE_PROFILE_SHA256,
        "presentation": "canonical",
        "cases": rows,
    }

    with pytest.raises(ValueError):
        experiment._load_base_prediction_features(blind, predictions)


def test_base_correction_defaults_to_base_and_only_flips_on_strong_disagreement():
    corrected, flipped = experiment._apply_base_correction(
        np.asarray([1, 0, 1]),
        np.asarray([0.20, 0.55, 0.90]),
        threshold=0.75,
    )

    assert corrected.tolist() == [0, 0, 1]
    assert flipped.tolist() == [True, False, False]


def test_base_correction_promotes_on_overall_net_improvement():
    rows = [
        ("private-correction-1", "private", np.asarray([0.0]), "validation"),
        ("private-correction-2", "private", np.asarray([0.0]), "validation"),
        ("group-new-error", "group", np.asarray([0.0]), "validation"),
        ("group-unchanged", "group", np.asarray([0.0]), "validation"),
    ]
    evaluation = experiment._evaluate_base_correction(
        rows,
        np.asarray([1, 1, 1, 1]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.9, 0.9, 0.9, 0.9]),
        np.asarray([0.9, 0.9, 0.1, 0.9]),
        threshold=0.8,
    )

    assert evaluation["scores"]["net_change"] == {
        "overall": 1, "private": 2, "group": -1
    }
    assert evaluation["changed_cases"]["new_error_count"] == 1
    assert evaluation["changed_cases"]["high_confidence_new_error_count"] == 1
    assert isinstance(evaluation["cases"][2]["high_confidence_new_error"], bool)
    json.dumps(evaluation)
    assert evaluation["passed"] is True


def test_base_correction_fits_once_on_1200_and_only_scores_development(
    monkeypatch, tmp_path
):
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("blind", "answers", "features", "validation", "base", "result")
    }
    feature_names = [f"feature-{index}" for index in range(73)]
    rows = [
        (
            f"case-{index}",
            "private" if index % 2 == 0 else "group",
            np.zeros(73) if index < 1200 else np.full(73, 9.0),
            "train" if index < 1200 else "validation",
        )
        for index in range(1360)
    ]
    validation_cases = [{"case_id": row[0]} for row in rows[1200:]]
    payloads = {
        paths["blind"]: {},
        paths["answers"]: {},
        paths["features"]: {},
        paths["validation"]: {"cases": validation_cases},
        paths["base"]: {},
    }
    fit_calls = []

    class FakeLogisticRegression:
        def __init__(self, **kwargs):
            self.coef_ = np.zeros((1, 73))

        def fit(self, x, y):
            fit_calls.append((x.copy(), y.copy()))
            return self

        def decision_function(self, x):
            return np.full(len(x), 2.0)

    monkeypatch.setattr(experiment, "_load_json", lambda path: payloads[path])
    monkeypatch.setattr(
        experiment,
        "load_base_correction_feature_rows",
        lambda blind, checkpoint, **kwargs: (rows, feature_names),
    )
    monkeypatch.setattr(
        experiment,
        "_labels_for_base_correction_rows",
        lambda rows, blind, answers: np.ones(len(rows), dtype=int),
    )
    monkeypatch.setattr(
        experiment,
        "_load_base_prediction_features",
        lambda blind, predictions: (
            np.zeros((160, 5)),
            np.zeros(160, dtype=int),
            np.full(160, 0.9),
            [f"base-{index}" for index in range(5)],
        ),
    )
    monkeypatch.setattr(experiment, "LogisticRegression", FakeLogisticRegression)
    monkeypatch.setattr(experiment, "_write_feature_classifier_result", lambda *args: None)

    result = experiment.train_base_correction_development(
        paths["blind"],
        paths["answers"],
        paths["features"],
        paths["validation"],
        paths["base"],
        paths["result"],
        include_group_participation_cross=True,
    )

    assert len(fit_calls) == 1
    assert fit_calls[0][0].shape == (1200, 73)
    assert not fit_calls[0][0].any()
    assert result["training"] == {
        "training_cases": 1200,
        "development_cases": 160,
        "development_used_for_model_fit": False,
        "development_used_for_threshold_selection": True,
        "model_fit_count": 1,
        "threshold_grid": list(experiment.BASE_CORRECTION_THRESHOLD_GRID),
    }
    assert result["feature_count"] == 73
    assert result["final_model"]["feature_names"] == feature_names


def _reworked_boolean_row(index, *, split, left=1.0, right=0.0):
    relationship = "private" if index % 2 == 0 else "group"
    return (
        f"{split}-{index}",
        relationship,
        np.asarray([]),
        np.asarray([]),
        np.asarray([left]),
        np.asarray([right]),
        np.asarray([
            float(relationship == "private"),
            float(relationship == "group"),
        ]),
        np.zeros(len(experiment.OBSERVABLE_OPTION_BOOLEAN_NAMES)),
        np.zeros(len(experiment.OBSERVABLE_OPTION_BOOLEAN_NAMES)),
        np.zeros(len(experiment.GROUP_PATTERN_OPTION_BOOLEAN_NAMES)),
        np.zeros(len(experiment.GROUP_PATTERN_OPTION_BOOLEAN_NAMES)),
        1,
    )


def test_reworked_boolean_shared_tree_is_ab_symmetric():
    row = _reworked_boolean_row(0, split="validation")
    swapped = (
        *row[:4],
        row[5],
        row[4],
        row[6],
        row[8],
        row[7],
        row[10],
        row[9],
        0,
    )

    class FirstFeatureScorer:
        def decision_function(self, values):
            return values[:, 0]

    probability = experiment._shared_tree_pair_probability(
        FirstFeatureScorer(), [row]
    )[0]
    swapped_probability = experiment._shared_tree_pair_probability(
        FirstFeatureScorer(), [swapped]
    )[0]

    assert probability + swapped_probability == pytest.approx(1.0, abs=1e-12)


def test_reworked_boolean_correction_uses_1200_train_160_development_and_no_240(
    monkeypatch, tmp_path
):
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "blind",
            "answers",
            "replay",
            "features",
            "validation",
            "base",
            "result",
        )
    }
    training_rows = [
        _reworked_boolean_row(index, split="train") for index in range(1200)
    ]
    validation_rows = [
        _reworked_boolean_row(index, split="validation") for index in range(160)
    ]
    validation_blind = {
        "cases": [{"case_id": row[0]} for row in validation_rows]
    }
    payloads = {
        paths["blind"]: {},
        paths["answers"]: {},
        paths["replay"]: {},
        paths["features"]: {},
        paths["validation"]: validation_blind,
        paths["base"]: {},
    }
    loaded_paths = []
    fit_shapes = []

    def fake_load(path):
        loaded_paths.append(path)
        if path not in payloads:
            pytest.fail(f"unexpected artifact opened: {path}")
        return payloads[path]

    class FakeLogisticRegression:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, values, labels):
            fit_shapes.append(("linear", values.shape, labels.shape))
            self.coef_ = np.zeros((1, values.shape[1]))
            return self

        def decision_function(self, values):
            return np.full(values.shape[0], 2.0)

    class FakeHistGradientBoostingClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, values, labels):
            fit_shapes.append(("tree", values.shape, labels.shape))
            return self

        def decision_function(self, values):
            return values[:, 0] * 2.0

    monkeypatch.setattr(experiment, "_load_json", fake_load)
    monkeypatch.setattr(
        experiment,
        "load_boolean_context_rows",
        lambda *args, **kwargs: (
            {"train": training_rows, "validation": validation_rows},
            [],
            ["context_fit>=1"],
            ["is_private", "is_group"],
            {"known_group_names": (), "known_group_members": ()},
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_load_base_prediction_features",
        lambda blind, predictions: (
            np.zeros((160, 5)),
            np.zeros(160, dtype=int),
            np.full(160, 0.9),
            [f"base-{index}" for index in range(5)],
        ),
    )
    monkeypatch.setattr(experiment, "LogisticRegression", FakeLogisticRegression)
    monkeypatch.setattr(
        experiment,
        "HistGradientBoostingClassifier",
        FakeHistGradientBoostingClassifier,
    )

    result = experiment.train_reworked_boolean_correction_development(
        paths["blind"],
        paths["answers"],
        paths["replay"],
        paths["features"],
        paths["validation"],
        paths["base"],
        paths["result"],
    )

    assert loaded_paths == [
        paths["blind"],
        paths["answers"],
        paths["replay"],
        paths["features"],
        paths["validation"],
        paths["base"],
    ]
    assert fit_shapes[0][0] == "linear"
    assert fit_shapes[0][1][0] == 1200
    assert fit_shapes[0][2] == (1200,)
    assert fit_shapes[1] == ("tree", (2400, 14), (2400,))
    assert result["training"] == {
        "training_cases": 1200,
        "development_cases": 160,
        "development_used_for_model_fit": False,
        "development_used_for_model_and_threshold_selection": True,
        "model_candidates_compared": 2,
        "threshold_grid": list(experiment.BASE_CORRECTION_THRESHOLD_GRID),
    }
    assert [row["name"] for row in result["model_candidates"]] == [
        "symmetric_logistic_refined_boolean_crosses",
        "shared_hist_gradient_boosting_boolean_scorer",
    ]
    assert result["correction_policy"] == {
        "large_model_base_used_as_small_model_feature": False,
        "large_model_base_feature_omission_reason": (
            "the frozen 1,200 training cases do not have predictions from the same "
            "official Base configuration; no training feature was fabricated"
        ),
        "large_model_base_used_as_default_prediction": True,
        "flip_only_on_small_model_disagreement_above_threshold": True,
        "promotion_rule": "overall net improvement on the 160-case development set",
    }
    assert result["selected_development_result"]["scores"]["net_change"]["overall"] > 0
    assert result["decision"] == "eligible_for_next_stage"
    assert result["entered_fixed_240"] is False
    assert result["fixed_240_answers_read"] is False
    assert result["online_components_modified"] == []
    saved = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert saved["official_judge_baseline"] is False


def test_rejected_development_cannot_open_fixed_240(tmp_path):
    development = tmp_path / "development.json"
    development.write_text(json.dumps({"decision": "reject"}), encoding="utf-8")

    with pytest.raises(ValueError, match="did not allow"):
        experiment.validate_base_correction_fixed_240(
            development,
            tmp_path / "blind.json",
            tmp_path / "answers.json",
            tmp_path / "features.json",
            tmp_path / "base.json",
            tmp_path / "result.json",
        )


def test_feature_checkpoint_accepts_metadata_only_blind_packet_change():
    source = {"batch_id": "fixed", "reference_sha256": "", "cases": [{"case_id": "1"}]}
    current = {
        "batch_id": "fixed",
        "reference_sha256": "filled-later",
        "cases": [{"case_id": "1"}],
    }
    checkpoint = {"blind_sha256": experiment._json_sha256(source)}

    compatibility = experiment._validate_feature_checkpoint_blind_compatibility(
        current, checkpoint, source
    )

    assert compatibility["mode"] == "metadata_only_packet_change"
    assert compatibility["cases_exactly_equal"] is True


def test_feature_checkpoint_rejects_actual_blind_case_change():
    source = {"batch_id": "fixed", "cases": [{"case_id": "1", "option_A": "old"}]}
    current = {"batch_id": "fixed", "cases": [{"case_id": "1", "option_A": "new"}]}
    checkpoint = {"blind_sha256": experiment._json_sha256(source)}

    with pytest.raises(ValueError, match="cases differ"):
        experiment._validate_feature_checkpoint_blind_compatibility(
            current, checkpoint, source
        )


def test_fixed_240_answers_are_loaded_after_predictions(monkeypatch, tmp_path):
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("development", "blind", "answers", "features", "base", "result")
    }
    original_names = [f"feature-{index}" for index in range(72)]
    development = {
        "decision": "enter_fixed_240",
        "training": {
            "training_cases": 1200,
            "development_cases": 160,
            "development_used_for_model_fit": False,
            "development_used_for_threshold_selection": True,
            "model_fit_count": 1,
        },
        "final_model": {
            "threshold": 0.9,
            "feature_names": original_names,
            "coefficients": [0.0] * 72,
        },
    }
    rows = [
        (
            f"case-{index}",
            "private" if index < 80 else "group",
            np.zeros(72),
            "holdout",
        )
        for index in range(240)
    ]
    loaded = []
    payloads = {
        paths["development"]: development,
        paths["blind"]: {"cases": []},
        paths["features"]: {
            "blind_sha256": experiment._json_sha256({"cases": []})
        },
        paths["base"]: {},
        paths["answers"]: {},
    }

    def fake_load(path):
        loaded.append(path)
        return payloads[path]

    monkeypatch.setattr(experiment, "_load_json", fake_load)
    monkeypatch.setattr(
        experiment,
        "load_base_correction_feature_rows",
        lambda blind, checkpoint, **kwargs: (rows, original_names),
    )
    monkeypatch.setattr(
        experiment,
        "_load_base_prediction_features",
        lambda blind, predictions: (
            np.zeros((240, 5)),
            np.ones(240, dtype=int),
            np.full(240, 0.9),
            [f"base-{index}" for index in range(5)],
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_labels_for_base_correction_rows",
        lambda rows, blind, answers: np.ones(240, dtype=int),
    )
    monkeypatch.setattr(experiment, "_write_feature_classifier_result", lambda *args: None)

    experiment.validate_base_correction_fixed_240(
        paths["development"],
        paths["blind"],
        paths["answers"],
        paths["features"],
        paths["base"],
        paths["result"],
    )

    assert loaded.index(paths["answers"]) > loaded.index(paths["base"])


@pytest.mark.parametrize(
    "command",
    [
        "train-refined-boolean-context",
        "train-group-pattern-boolean",
        "prepare-effective-group-pattern-holdout",
        "build-stable-judge-validation",
        "run-stable-judge-validation",
        "compare-stable-judge-classifier",
        "train-base-correction-development",
        "train-reworked-boolean-correction-development",
        "validate-base-correction-fixed-240",
        "validate-reworked-boolean-correction-fixed-240",
    ],
)
def test_stable_judge_cli_commands_parse(command):
    argv = [command]
    if command == "prepare-effective-group-pattern-holdout":
        argv.extend(["--source-checkpoints", "checkpoint.json"])
    args = experiment.build_parser().parse_args(argv)

    assert args.command == command


def test_base_correction_cli_can_enable_one_group_participation_cross():
    args = experiment.build_parser().parse_args([
        "train-base-correction-development",
        "--include-group-participation-cross",
    ])

    assert args.include_group_participation_cross is True
