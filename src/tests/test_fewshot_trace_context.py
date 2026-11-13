from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.compare_fewshot_trace_context import replay_body, replace_examples, run_selected, select_excerpts, shorten_examples, worker


def test_shortening_preserves_reply_identity_and_everything_outside_context():
    example = ('<example id="source"><context>\n'
               '<message sender="A">old &amp; unrelated</message>\n'
               '<message sender="B">recent\nmultiline</message>\n'
               '<message sender="A">latest</message>\n'
               '</context><response><message>original reply</message></response></example>')
    original = "unchanged prefix" + example + "unchanged suffix"
    shortened, counts = shorten_examples(original, 2)
    assert shortened == original.replace('\n<message sender="A">old &amp; unrelated</message>', '', 1)
    assert counts == [{"id": "source", "before": 3, "after": 2}]


def test_short_examples_stay_byte_identical():
    text = '<example id="x"><context><message>hi</message></context><response>ok</response></example>'
    assert shorten_examples(text, 3)[0] == text
    with pytest.raises(ValueError):
        shorten_examples(text, 0)


def test_tool_followup_trace_is_rejected_without_silently_dropping_requests():
    event = {"kind": "llm", "status": "ok", "request": {"body": {"messages": []}}}
    trace = {"operations": [{"kind": "generate", "events": [event]}]}
    result = replay_body(trace)
    result["body"]["messages"].append("changed")
    assert trace["operations"][0]["events"][0]["request"]["body"]["messages"] == []
    trace["operations"][0]["events"].append(deepcopy(event))
    with pytest.raises(ValueError):
        replay_body(trace)


def test_top_level_trace_and_replacement_preserve_frozen_request():
    old = "\n".join(f'<example id="old{i}"><context/><response/></example>' for i in range(3))
    new = "\n".join(f'<example id="manual-excerpt-{i}"><context/><response/></example>' for i in range(3))
    request = {"protocol": "anthropic", "body": {"system": "frozen", "model": "frozen",
               "messages": [{"role": "user", "content": "history\n<style_examples>\n" + old + "\n</style_examples>\nunread"}]}}
    trace = {"events": [{"kind": "llm", "status": "ok", "request": request}]}
    candidate = replace_examples(replay_body(trace), "<style_examples>" + new + "</style_examples>")
    restored = deepcopy(candidate)
    restored["body"]["messages"][0]["content"] = request["body"]["messages"][0]["content"]
    assert restored == request
    assert candidate["body"]["messages"][0]["content"].startswith("history\n<style_examples>\n")
    assert candidate["body"]["messages"][0]["content"].endswith("\n</style_examples>\nunread")
    spaced = "<style_examples>\n" + new + "\n</style_examples>"
    assert replace_examples(request, spaced) == candidate
    assert trace["events"][0]["request"] == request
    trace["events"].append(deepcopy(trace["events"][0]))
    with pytest.raises(ValueError):
        replay_body(trace)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_selected_examples_need_not_fill_top_three(count):
    old = "\n".join(f'<example id="old{i}"/>' for i in range(3))
    old = old.replace('/>', '></example>')
    request = {"body": {"messages": [{"content": "before\n" + old + "\nafter"}]}}
    selected = "\n".join(f'<example id="new{i}"></example>' for i in range(count))
    candidate = replace_examples(request, "<style_examples>" + selected + "\n</style_examples>")
    text = candidate["body"]["messages"][0]["content"]
    assert text.startswith("before\n") and text.endswith("\nafter")
    assert text.count('<example id="new') == count
    if count == 0:
        assert text == "before\n\nafter"


def test_examples_keep_complete_context_and_reply_with_sender_aliases(tmp_path):
    pool = tmp_path / "pool.jsonl"
    source = {"id": "x", "content_sha256": "hash", "source_provenance": "before_automation_cutoff", "relationship": "private",
              "source_span": {"order_verified": False},
              "context_messages": [{"sender": "self name", "is_self": True, "text": "咋样"},
                                   {"sender": "private name", "is_self": False, "text": "哎\n忧伤"}],
              "reply": ["咋了", "说说看呢"]}
    pool.write_text(json.dumps(source) + "\n")
    selection = {"id": "x", "content_sha256": "hash", "context_indices": [0, 1], "reply_indices": [0, 1]}
    rows, evidence = select_excerpts(pool, [selection])
    assert rows == [{"id": "manual-excerpt-x", "context_messages": [
        {"sender": "本人", "is_self": True, "text": "咋样"},
        {"sender": "对方", "is_self": False, "text": "哎\n忧伤"}], "reply": ["咋了", "说说看呢"]}]
    assert "private name" not in json.dumps(evidence)
    assert evidence[0]["source_order_verified"] is False
    with pytest.raises(ValueError):
        select_excerpts(pool, [{**selection, "content_sha256": "changed"}])
    with pytest.raises(ValueError):
        select_excerpts(pool, [{**selection, "context_indices": [-1]}])


@pytest.mark.parametrize("context_indices,reply_indices", [([0, 1], [0, 1]), ([1, 2], [0, 1]),
    ([0, 2], [0, 1]), ([0, 1, 2], [0]), ([0, 1, 2], [1])])
def test_rejects_omitted_context_and_partial_self_reply(tmp_path, context_indices, reply_indices):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"id": "x", "content_sha256": "hash", "relationship": "private",
        "context_messages": [{"sender": "other", "is_self": False, "text": text}
                             for text in ["前因", "哇", "猪猪真厉害"]],
        "reply": ["一般一般", "下一句"]}) + "\n")
    with pytest.raises(ValueError, match="complete source"):
        select_excerpts(pool, [{"id": "x", "content_sha256": "hash", "context_indices": context_indices,
                               "reply_indices": reply_indices}])


def test_rejects_group_instead_of_merging_distinct_speakers(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"id": "x", "content_sha256": "hash", "relationship": "group",
        "context_messages": [{"sender": "A", "is_self": False, "text": "a"},
                             {"sender": "B", "is_self": False, "text": "b"}], "reply": ["ok"]}) + "\n")
    with pytest.raises(ValueError, match="private conversation"):
        select_excerpts(pool, [{"id": "x", "content_sha256": "hash", "context_indices": [0, 1],
                               "reply_indices": [0]}])


def test_worker_retains_raw_on_parse_failure_and_disables_retry(monkeypatch):
    calls = []
    class Client:
        def __init__(self, settings, config):
            calls.append((settings, config))

        def chat(self, messages, **kwargs):
            return "raw malformed output"

    def fail_parse(raw):
        raise ValueError("bad json")

    monkeypatch.setattr(sys, "path", list(sys.path))
    for name, module in {
        "src.infrastructure.config": SimpleNamespace(load_settings=lambda: {"llm": {"max_retries": 3}}),
        "src.iteration": SimpleNamespace(versions=SimpleNamespace(switch_instance=lambda instance: None)),
        "src.generator.generator": SimpleNamespace(ReplyGenerator=SimpleNamespace(_parse=fail_parse, _validate=lambda *args: None)),
        "src.llm": SimpleNamespace(ChatClient=Client),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    result = worker({"repository": "/tmp", "instance": "test", "timeout": 20, "client_config": {},
                     "request": {"protocol": "anthropic", "body": {"model": "frozen", "messages": []}}})
    assert result["status"] == "parse_error"
    assert result["raw"] == "raw malformed output"
    assert result["error"] == "ValueError"
    assert result["chat_elapsed_seconds"] >= 0
    assert len(calls) == 1
    assert calls[0][0]["llm"]["max_retries"] == 0
    assert calls[0][1]["timeout_seconds"] == 20


@pytest.mark.parametrize("timeout,total,arm", [(21, 60, "candidate"), (20, 61, "candidate"), (20, 60, "both")])
def test_selected_replay_rejects_unbounded_or_baseline_generation(timeout, total, arm):
    with pytest.raises(ValueError):
        run_selected(SimpleNamespace(timeout=timeout, total_seconds=total, arm=arm))
