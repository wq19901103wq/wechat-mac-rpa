"""Contextual model review of the complete draft before sending."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path


_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review.md"
_CONTEXT_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review_context.md"
_VERIFY_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review_verify.md"
_CATEGORIES = {"ai_identity", "hallucination", "identity_confusion", "repetition",
               "no_information", "other_ai_trait"}


class _RecordingClient:
    def __init__(self, client, messages, on_messages):
        self._client, self._messages = client, messages
        self._on_messages = on_messages

    def __getattr__(self, name):
        return getattr(self._client, name)

    def chat(self, messages, *args, **kwargs):
        self._messages[:] = json.loads(json.dumps(messages, ensure_ascii=False))
        if self._on_messages is not None:
            self._on_messages(self._messages)
        return self._client.chat(messages, *args, **kwargs)


@contextmanager
def capture_generation(clients, on_messages=None):
    messages = []
    original = dict(clients)
    try:
        clients.update({key: _RecordingClient(client, messages, on_messages) for key, client in original.items()})
        yield messages
    finally:
        clients.update(original)


class ContextPrefetch:
    def __init__(self, pool, client, online_input, target_indices):
        self.pool, self.client = pool, client
        self.input = json.loads(json.dumps(dict(conversation=online_input, target_indices=target_indices),
                                           ensure_ascii=False))
        self.context = copy_context()
        self.prompt = _CONTEXT_PROMPT.read_text(encoding="utf-8")
        self.future = None

    def start(self, messages):
        if self.future is None:
            self.payload = {**self.input, "generation_material": json.loads(json.dumps(messages, ensure_ascii=False))}
            self.future = self.pool.submit(self.context.run, _analyze_context, self.client, self.prompt, self.payload)

    def matches(self, client, prompt, payload):
        return (self.future is not None and client is self.client and prompt == self.prompt
                and payload == self.payload)


def _analyze_context(client, prompt, payload):
    started = time.monotonic()
    raw = client.chat([dict(role="system", content=prompt),
                       dict(role="user", content=json.dumps(payload, ensure_ascii=False))], json_mode=True)
    context = json.loads(raw)
    if (not isinstance(context, dict) or not isinstance(context.get("events"), list)
            or not isinstance(context.get("targets"), list)):
        raise ValueError("回复审查返回无效上下文分析")
    return dict(analysis=context, raw=raw, model=client.model_name, seconds=time.monotonic() - started,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def review_replies(client, replies, online_input, target_indices, generation_messages, *, parallel=False, prefetch=None):
    started = time.monotonic()
    if not replies or not generation_messages:
        raise ValueError("回复审查缺少草稿或本轮生成依据")
    prompt = _PROMPT.read_text(encoding="utf-8")
    payload = dict(conversation=online_input, target_indices=target_indices,
                   generation_material=generation_messages, replies=replies)
    context_prompt = _CONTEXT_PROMPT.read_text(encoding="utf-8")
    context_input = {key: value for key, value in payload.items() if key != "replies"}
    prefetched = prefetch is not None and prefetch.matches(client, context_prompt, context_input)
    context_review = (prefetch.future.result() if prefetched else
                      _analyze_context(client, context_prompt, context_input))
    context_review = {**context_review, "prefetched": prefetched, "wait_seconds": time.monotonic() - started}
    payload["context_analysis"] = context_review["analysis"]
    verify_prompt = _VERIFY_PROMPT.read_text(encoding="utf-8")
    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            primary = pool.submit(copy_context().run, _review_once, client, prompt, payload, replies)
            verification = pool.submit(copy_context().run, _review_once, client, verify_prompt, payload, replies)
            result = primary.result()
            if result["decision"] == "allow":
                verified = verification.result()
                result = {**verified, "reviews": [result, verified]}
    else:
        result = _review_once(client, prompt, payload, replies)
        if result["decision"] == "allow":
            verified = _review_once(client, verify_prompt, payload, replies)
            result = {**verified, "reviews": [result, verified]}
    return {**result, "context_review": context_review,
            "parallel": parallel, "seconds": time.monotonic() - started}


def _review_once(client, prompt, payload, replies):
    started = time.monotonic()
    raw = client.chat([dict(role="system", content=prompt),
                       dict(role="user", content=json.dumps(payload, ensure_ascii=False))], json_mode=True)
    result = json.loads(raw)
    if (not isinstance(result, dict) or result.get("decision") not in ("allow", "block")
            or not isinstance(result.get("issues"), list)
            or (result["decision"] == "allow") != (not result["issues"])):
        raise ValueError("回复审查返回无效决定")
    for issue in result["issues"]:
        if (not isinstance(issue, dict) or issue.get("category") not in _CATEGORIES
                or type(issue.get("reply_index")) is not int
                or not 0 <= issue["reply_index"] < len(replies)
                or not isinstance(issue.get("reason"), str) or not issue["reason"].strip()
                or not isinstance(issue.get("quote"), str) or not issue["quote"].strip()
                or issue["quote"] not in replies[issue["reply_index"]]):
            raise ValueError("回复审查返回无效问题依据")
    return {**result, "model": client.model_name,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw": raw,
            "seconds": time.monotonic() - started}
