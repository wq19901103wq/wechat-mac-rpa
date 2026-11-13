"""Contextual model review of the complete draft before sending."""

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path


_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review.md"
_logger = logging.getLogger(__name__)
_CONTEXT_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review_context.md"
_VERIFY_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review_verify.md"
_SHARED_PROMPT = Path(__file__).resolve().parents[2] / "prompts/reply_review_shared.md"
_CATEGORIES = {"ai_identity", "hallucination", "identity_confusion", "repetition",
               "no_information", "other_ai_trait"}


def shared_review_prompt():
    return _SHARED_PROMPT.read_text(encoding="utf-8") + "\n" + "\n".join(
        f"<stage name=\"{stage}\">\n{path.read_text(encoding='utf-8')}\n</stage>"
        for stage, path in (("context", _CONTEXT_PROMPT), ("primary", _PROMPT), ("verify", _VERIFY_PROMPT)))


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
    def __init__(self, pool, client, online_input, target_indices, *, reference_prefix=False):
        self.pool, self.client = pool, client
        self.reference_prefix = reference_prefix
        self.input = json.loads(json.dumps(dict(conversation=online_input, target_indices=target_indices),
                                           ensure_ascii=False))
        self.context = copy_context()
        self.prompt = _CONTEXT_PROMPT.read_text(encoding="utf-8")
        self.future = None

    def start(self, messages):
        if self.future is None:
            self.payload = {**self.input, "generation_material": json.loads(json.dumps(messages, ensure_ascii=False))}
            if self.reference_prefix:
                self.payload = {"generation_material": self.payload["generation_material"], **self.payload}
            self.future = self.pool.submit(self.context.run, _analyze_context, self.client, self.prompt, self.payload)

    def matches(self, client, prompt, payload):
        return (self.future is not None and client is self.client and prompt == self.prompt
                and list(payload) == list(self.payload) and payload == self.payload)


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


def review_replies(client, replies, online_input, target_indices, generation_messages, *, parallel=False, prefetch=None,
                   prompt_overrides=None, direct_context=False, context_client=None, reference_prefix=False,
                   shared_prefix=False):
    started = time.monotonic()
    if not replies or not generation_messages:
        raise ValueError("回复审查缺少草稿或本轮生成依据")
    if type(reference_prefix) is not bool:
        raise ValueError("资料前缀选项必须为布尔值")
    overrides = prompt_overrides if prompt_overrides is not None else {}
    if (not isinstance(overrides, dict) or overrides.keys() - {"primary", "context", "verify"}
            or any(not isinstance(value, str) or not value.strip() for value in overrides.values())):
        raise ValueError("回复审查提示覆盖无效")
    if type(direct_context) is not bool or (direct_context and (
            parallel or prefetch is not None or "context" in overrides)):
        raise ValueError("直接原文审查要求串行且不使用上下文预取或摘要提示覆盖")
    if context_client is not None and (parallel or prefetch is not None or direct_context):
        raise ValueError("独立上下文模型要求串行且不使用预取或直接原文审查")
    if type(shared_prefix) is not bool or (shared_prefix and (
            parallel or prefetch is not None or overrides or direct_context or context_client is not None
            or reference_prefix)):
        raise ValueError("共同前缀要求独立串行审查且不覆盖提示、模型或资料顺序")
    prompt = overrides.get("primary", _PROMPT.read_text(encoding="utf-8"))
    common_prompt = shared_review_prompt() if shared_prefix else None
    if shared_prefix:
        prompt = common_prompt
    payload = dict(conversation=online_input, target_indices=target_indices,
                   generation_material=generation_messages, replies=replies)
    if reference_prefix or shared_prefix:
        payload = {"generation_material": generation_messages, **payload}
    if direct_context:
        context_review = dict(skipped=True, seconds=0.0, prefetched=False, wait_seconds=0.0)
    else:
        context_prompt = overrides.get("context", _CONTEXT_PROMPT.read_text(encoding="utf-8"))
        context_input = {key: value for key, value in payload.items() if key != "replies"}
        if shared_prefix:
            context_prompt = common_prompt
            context_input["review_stage"] = "context"
        prefetched = prefetch is not None and prefetch.matches(client, context_prompt, context_input)
        context_review = (prefetch.future.result() if prefetched else
                          _analyze_context(context_client if context_client is not None else client,
                                           context_prompt, context_input))
        context_review = {**context_review, "prefetched": prefetched, "wait_seconds": time.monotonic() - started}
        payload["context_analysis"] = context_review["analysis"]
    verify_prompt = overrides.get("verify", _VERIFY_PROMPT.read_text(encoding="utf-8"))
    if shared_prefix:
        verify_prompt = common_prompt
        payload["review_stage"] = "primary"
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
            verify_payload = {**payload, "review_stage": "verify"} if shared_prefix else payload
            verified = _review_once(client, verify_prompt, verify_payload, replies)
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
                or not isinstance(issue.get("quote"), str) or not issue["quote"].strip()):
            raise ValueError("回复审查返回无效问题依据")
        quote = issue["quote"]
        target = replies[issue["reply_index"]]
        # 引用比对做空白归一化：仅空白差异不再视为无效依据；
        # 真实不符仍拒绝整个判定（fail-closed，不放行未证实的引用）
        if (quote not in target
                and re.sub(r"\s+", "", quote) not in re.sub(r"\s+", "", target)):
            raise ValueError("回复审查返回无效问题依据")
    return {**result, "model": client.model_name,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw": raw,
            "seconds": time.monotonic() - started}
