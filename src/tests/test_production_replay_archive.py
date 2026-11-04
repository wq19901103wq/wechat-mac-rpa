import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.benchmarks.production_replay_archive import (
    DecisionPointCutoff,
    ProductionReplayArchive,
    RecordingLLMClient,
    ReplaySender,
    ReplaySideEffectSink,
    guarded_tool_registry,
)
from src.models.base import ChatMessage, PerceptionResult, SenderType
from src.session.global_store import ChatState
from src.tools.tool_registry import ToolRegistry


def _store():
    message = ChatMessage("你好", "示例成员", SenderType.OTHER, "示例交流群")
    state = ChatState("group-1", "示例交流群", True, [message])
    return SimpleNamespace(max_messages=1000, chats={"示例交流群": state})


def test_archive_preserves_state_screenshot_and_pre_action_cutoff(tmp_path: Path):
    screenshot = tmp_path / "source.png"
    screenshot.write_bytes(b"not-a-real-image")
    archive = ProductionReplayArchive(tmp_path / "archive", "session-1", {"mode": "ocr"})
    archive.capture_session_start(_store())
    archive.begin_tick(7, _store())
    result = PerceptionResult("示例交流群", [], [], str(screenshot), is_group=True)
    archive.capture_perception(7, result)
    debug = SimpleNamespace(reply_generation_trace=[], bot_reply_text="哈哈")
    archive.capture_decision_point(7, debug, ["哈哈"])
    archive.finish_tick(7, debug)

    tick_dir = tmp_path / "archive" / "session-1" / "ticks" / "7"
    meta = json.loads((tick_dir / "tick.json").read_text(encoding="utf-8"))
    assert meta["decision_point_captured"] is True
    assert meta["eligible_for_formal_replay"] is True
    assert (tick_dir / "screenshot.png").read_bytes() == b"not-a-real-image"
    with gzip.open(tick_dir / "state_before.json.gz", "rt", encoding="utf-8") as handle:
        state = json.load(handle)
    assert state["chats"]["示例交流群"]["messages"][0]["sender"] == "示例成员"
    assert len((tmp_path / "archive" / "session-1" / "manifest.jsonl").read_text().splitlines()) == 1


def test_side_effect_sink_records_and_never_calls_external_action(tmp_path: Path):
    sink = ReplaySideEffectSink(tmp_path / "effects.jsonl")
    sender = ReplaySender(sink)
    with pytest.raises(DecisionPointCutoff):
        sender.send("候选回复", chat_name="示例交流群")
    record = json.loads((tmp_path / "effects.jsonl").read_text(encoding="utf-8"))
    assert record["kind"] == "send_text"
    assert record["payload"]["text"] == "候选回复"


def test_guarded_tool_cutoff_is_not_swallowed_by_tool_execute(tmp_path: Path):
    original = Mock()
    registry = ToolRegistry()
    registry.register("lookup", "查询", {"type": "object"}, original)
    guarded = guarded_tool_registry(
        registry,
        ReplaySideEffectSink(tmp_path / "effects.jsonl"),
    )

    with pytest.raises(DecisionPointCutoff):
        guarded.get("lookup").execute('{"query":"泰州"}')

    original.assert_not_called()
    record = json.loads((tmp_path / "effects.jsonl").read_text(encoding="utf-8"))
    assert record["kind"] == "tool_execution"
    assert record["payload"]["name"] == "lookup"


def test_tool_using_tick_is_not_formal_replay_eligible(tmp_path: Path):
    archive = ProductionReplayArchive(tmp_path / "archive", "session-2", {})
    archive.begin_tick(8, _store())
    debug = SimpleNamespace(
        reply_generation_trace=[{"type": "tool_execution", "tool_name": "send_file"}]
    )
    archive.capture_decision_point(8, debug, ["给你发了"])
    archive.finish_tick(8, debug)
    meta = json.loads(
        (tmp_path / "archive" / "session-2" / "ticks" / "8" / "tick.json").read_text()
    )
    assert meta["eligible_for_formal_replay"] is False
    assert meta["ineligible_reason"] == "generation_executed_tools"


class _FakeLLMClient:
    def __init__(self, response=None, error=None):
        self.model = "fake-model"
        self.base_url = "https://user:secret@example.test/v1?api_key=secret#fragment"
        self.api_key = "must-not-be-archived"
        self.response = response
        self.error = error

    def chat(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


class _FakeRegistry:
    def to_openai_schemas(self):
        return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


def _generator(client):
    return SimpleNamespace(
        llm_client=client,
        _fact_check_client=None,
        session_memory=SimpleNamespace(_sessions={"示例交流群": {"marker": "完整会话记忆"}}),
        tool_registry=_FakeRegistry(),
        enable_time_awareness=True,
        enable_reply_restraint=True,
        enable_unread_dedup=True,
        enable_timestamps=True,
        enable_react_tools=True,
        enable_self_refine=True,
        enable_fact_check=True,
        enable_persona_few_shots=True,
        persona_few_shot_count=8,
        persona_few_shot_max_chars=2500,
        persona_few_shot_allow_unreviewed=False,
        last_system_prompt="system-完整",
        last_user_prompt="user-完整",
        last_raw_response="raw-完整",
        last_llm_messages=[],
        last_few_shot_ids=["human-1"],
        last_few_shot_content="真人原话",
        last_loaded_skills=["group_banter"],
        last_skill_injected_content="skill 原文",
        last_tools_context="缓存原文",
        last_feedback_raw="feedback 原文",
        last_iterate_raw="iterate 原文",
    )


def _read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_exact_llm_call_capture_is_complete_and_sanitized(tmp_path: Path):
    long_prompt = "上下文" * 6000
    long_response = "回复" * 3000
    response = SimpleNamespace(
        content=long_response,
        reasoning_content="完整思考",
        tool_calls=[{"function": {"name": "lookup", "arguments": '{"query":"完整参数"}'}}],
    )
    archive = ProductionReplayArchive(tmp_path / "archive", "session-exact", {})
    generator = _generator(_FakeLLMClient(response=response))
    archive.install_generator(generator)
    assert isinstance(generator.llm_client, RecordingLLMClient)
    archive.begin_tick(1, _store())
    archive.capture_generator_pre_state(1, generator)
    returned = generator.llm_client.chat(messages=[{"role": "user", "content": long_prompt}])
    assert returned is response
    archive.capture_decision_point(
        1,
        SimpleNamespace(reply_generation_trace=[]),
        [long_response],
        generator=generator,
    )
    archive.finish_tick(1, SimpleNamespace(reply_generation_trace=[]))

    tick_dir = tmp_path / "archive" / "session-exact" / "ticks" / "1"
    call = _read_gzip_json(tick_dir / "llm_calls" / "001.json.gz")
    assert call["request"]["kwargs"]["messages"][0]["content"] == long_prompt
    assert call["response"]["content"] == long_response
    assert call["response"]["tool_calls"][0]["function"]["arguments"] == '{"query":"完整参数"}'
    assert call["client"]["base_url"] == "https://example.test/v1"
    assert "must-not-be-archived" not in json.dumps(call, ensure_ascii=False)
    assert "api_key=secret" not in json.dumps(call, ensure_ascii=False)
    meta = json.loads((tick_dir / "tick.json").read_text(encoding="utf-8"))
    assert meta["eligible_for_formal_replay"] is True
    decision = _read_gzip_json(tick_dir / "decision_point.json.gz")
    assert decision["exact_generator_state"]["few_shot_content"] == "真人原话"
    assert decision["exact_generator_state"]["session_memory_after"]["sessions"]["示例交流群"]["marker"] == "完整会话记忆"
    before = _read_gzip_json(tick_dir / "generator_before.json.gz")
    assert before["session_memory"]["sessions"]["示例交流群"]["marker"] == "完整会话记忆"
    assert before["tool_schemas"][0]["function"]["name"] == "lookup"


def test_dependency_snapshots_are_copied_and_hashed(tmp_path: Path):
    source = tmp_path / "prompt.md"
    source.write_text("不可变依赖", encoding="utf-8")
    archive = ProductionReplayArchive(tmp_path / "archive", "session-deps", {})
    archive.capture_session_start(_store(), [source])
    evidence = json.loads(
        (tmp_path / "archive" / "session-deps" / "session_start_evidence.json").read_text()
    )
    record = evidence["source_files"][0]
    snapshot = tmp_path / "archive" / "session-deps" / record["snapshot"]
    assert snapshot.read_text(encoding="utf-8") == "不可变依赖"
    assert record["sha256"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()


def test_missing_exact_llm_call_is_not_formal_replay_eligible(tmp_path: Path):
    archive = ProductionReplayArchive(tmp_path / "archive", "session-missing", {})
    generator = _generator(_FakeLLMClient(response="unused"))
    archive.install_generator(generator)
    archive.begin_tick(1, _store())
    archive.capture_decision_point(
        1,
        SimpleNamespace(reply_generation_trace=[]),
        [],
        generator=generator,
    )
    meta = json.loads(
        (tmp_path / "archive" / "session-missing" / "ticks" / "1" / "tick.json").read_text()
    )
    assert meta["eligible_for_formal_replay"] is False
    assert meta["ineligible_reason"] == "missing_exact_llm_call_capture"


def test_recording_client_reraises_without_changing_exception(tmp_path: Path):
    archive = ProductionReplayArchive(tmp_path / "archive", "session-error", {})
    expected = RuntimeError("fake failure")
    generator = _generator(_FakeLLMClient(error=expected))
    archive.install_generator(generator)
    archive.begin_tick(1, _store())
    with pytest.raises(RuntimeError) as raised:
        generator.llm_client.chat(messages=[{"role": "user", "content": "test"}])
    assert raised.value is expected
    archive.finish_tick(1, SimpleNamespace(reply_generation_trace=[]))
    call = _read_gzip_json(
        tmp_path / "archive" / "session-error" / "ticks" / "1" / "llm_calls" / "001.json.gz"
    )
    assert call["error"] == "RuntimeError: fake failure"
