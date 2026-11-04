"""Shared-checkout digital-human drafts; this module has no sending capability."""

import hashlib
import json
import logging
from functools import partial
import math
import re
import time
import uuid
from pathlib import Path

from src.models.base import ChatMessage, SenderType
from src.reply.dh_runtime import WorkerRuntime
from src.reply.exposure_guard import capture_generation, review_replies


_logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]


def _load_runtime(repository):
    entry = repository / "src/generator/draft.py"
    if not entry.is_file():
        raise DigitalHumanError("digital-human 主仓库草稿入口缺失；检查 repository 配置")
    return partial(WorkerRuntime, repository)


class DigitalHumanError(RuntimeError):
    pass


def load_digital_human_config(path: Path | None = None) -> dict:
    path = path or _ROOT / "config/digital_human.json"
    if not path.is_file():
        raise DigitalHumanError("缺少 DH 本机配置；请复制 config/digital_human.json.example 为 config/digital_human.json 并填写实例")
    config = json.loads(path.read_text(encoding="utf-8"))
    repository = Path(config["repository"])
    if not repository.is_absolute():
        repository = _ROOT / repository
    instance, timeout = config["instance"], config["timeout_seconds"]
    if (not isinstance(instance, str) or not instance.strip()
            or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise DigitalHumanError("digital-human 配置需要明确实例和有限正数超时")
    result = {"repository": repository.resolve(), "instance": instance, "timeout": timeout}
    if config.get("review_client") is not None:
        if config["review_client"] not in ("private", "group"):
            raise DigitalHumanError("回复审查客户端必须为 private 或 group")
        result["review_client"] = config["review_client"]
    for key in ("instances_root", "settings_file", "env_file"):
        if config.get(key):
            value = Path(config[key]).expanduser()
            result[key] = (value if value.is_absolute() else _ROOT / value).resolve()
    return result


def input_digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _message_text(message: ChatMessage) -> str:
    text = message.text
    if message.message_type != "text":
        label = {"image": "图片", "sticker": "表情包", "mixed": "图文",
                 "link_card": "链接卡片", "video": "视频"}.get(message.message_type)
        if label is None:
            raise DigitalHumanError(f"未知消息类型：{message.message_type}")
        parts = [f"[{label}]"]
        if message.image_description.strip():
            parts.append(f"描述：{message.image_description}")
        if text.strip():
            parts.append(text)
        if message.image_text.strip() and message.image_text != text:
            parts.append(f"识别文字：{message.image_text}")
        text = "\n".join(parts)
    if message.quoted_text.strip():
        text = f"「引用：{message.quoted_text}」\n{text}"
    return text


def messages_input(messages: list[ChatMessage], *, source_chat_id: str | None = None, is_group: bool) -> dict:
    observed = any(message.observed_at is not None for message in messages)
    if not messages or type(is_group) is not bool or (not observed and not source_chat_id):
        raise DigitalHumanError("需要非空文本上下文、明确聊天类型和原始聊天 ID")
    chat_name = messages[0].chat_name
    context = []
    for message in messages:
        if message.chat_name != chat_name or (source_chat_id and message.chatroom_id and message.chatroom_id != source_chat_id):
            raise DigitalHumanError("上下文不能混合不同聊天")
        if message.sender_type not in (SenderType.SELF, SenderType.OTHER):
            raise DigitalHumanError("发言人角色必须明确，不能把 UNKNOWN 当作 OTHER")
        text = _message_text(message)
        timestamp = message.observed_at if message.observed_at is not None else message.create_time
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp <= 0
                or not text.strip() or not message.sender.strip()):
            raise DigitalHumanError("消息需要有效时间、文本和发言人")
        context.append({"text": text, "sender": message.sender,
                        "is_self": message.sender_type == SenderType.SELF, "timestamp": timestamp})
        if observed:
            context[-1]["time_basis"] = "first_observed" if message.observed_at is not None else "recorded"
    value = {"kind": "observed" if observed else "messages", "chat_type": "group" if is_group else "private",
             "chat_name": chat_name, "context": context}
    if not observed:
        value["source_chat_id"] = source_chat_id
    return value


class DigitalHumanDraftClient:
    def __init__(self, repository: Path, instance: str, *, timeout: float = 600,
                 instances_root=None, settings_file=None, env_file=None, review_client=None):
        started = time.perf_counter()
        if (not isinstance(instance, str) or not instance or type(timeout) not in (int, float)
                or timeout <= 0 or not math.isfinite(timeout)):
            raise DigitalHumanError("需要明确实例和有限正数超时")
        if review_client is not None and review_client not in ("private", "group"):
            raise DigitalHumanError("回复审查客户端必须为 private 或 group")
        repository = repository.resolve()
        self.instance, self.timeout = instance, timeout
        self.review_client = review_client
        self._runtime = None
        try:
            runtime = _load_runtime(repository)
            self._runtime = runtime(instance, timeout=timeout,
                instances_root=instances_root or repository / "instances",
                settings_file=settings_file, env_file=env_file or repository / ".env")
            self.binding = dict(self._runtime.binding)
            if (self.binding.get("instance") != instance
                    or not all(isinstance(self.binding.get(key), str) and self.binding[key]
                               for key in ("generator", "data", "config_sha256", "sources_sha256"))):
                raise DigitalHumanError("DH 初始化失败或实例绑定不符")
            self.startup_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            self.close()
            raise DigitalHumanError(f"DH 初始化失败：{exc}") from exc

    def draft(self, online_input: dict, *, target_indices=None) -> dict:
        started = time.perf_counter()
        self._review_messages = []
        if self._runtime is None:
            raise DigitalHumanError("DH 草稿客户端已关闭")
        request_id = uuid.uuid4().hex
        try:
            payload = json.dumps({"request_id": request_id, "input": online_input}, ensure_ascii=False,
                                 allow_nan=False).encode() + b"\n"
            if len(payload) > 4 * 1024 * 1024:
                raise DigitalHumanError("DH 输入超过协议限制")
            clients = getattr(getattr(self._runtime, "generator", None), "_llm", {})
            with capture_generation(clients) as generation_messages:
                if isinstance(self._runtime, WorkerRuntime) and target_indices is not None:
                    response = self._runtime.draft(json.loads(payload), review=dict(
                        client=self.review_client or online_input["chat_type"], target_indices=target_indices))
                else:
                    response = self._runtime.draft(json.loads(payload))
            if not isinstance(response, dict):
                raise DigitalHumanError("DH 返回无效结果对象")
            self._review_messages = response.pop("generation_messages", generation_messages)
            if response.get("request_id") != request_id or response.get("binding") != self.binding:
                raise DigitalHumanError("DH 响应身份或版本绑定不符")
            if response.get("status") == "error":
                raise DigitalHumanError("DH 生成失败：" + str(response.get("error", "unknown")))
            replies = response.get("replies")
            if (response.get("draft_only") is not True or response.get("sent") is not False
                    or not isinstance(replies, list) or len(replies) > 3
                    or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
                    or response.get("status") != ("draft" if replies else "silence")
                    or response.get("input_payload_sha256") != input_digest(online_input)
                    or not isinstance(response.get("input_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", response["input_sha256"])):
                raise DigitalHumanError("DH 草稿结果未通过协议核验")
            response["rpa_timing_ms"] = {"startup": self.startup_ms,
                                         "draft": (time.perf_counter() - started) * 1000}
            return response
        except Exception as exc:
            self.close()
            if isinstance(exc, DigitalHumanError):
                raise
            raise DigitalHumanError(f"DH 生成失败：{exc}") from exc

    def review(self, replies, online_input, target_indices):
        if self._runtime is None:
            raise DigitalHumanError("DH 草稿客户端已关闭")
        name = self.review_client or online_input["chat_type"]
        if isinstance(self._runtime, WorkerRuntime):
            return self._runtime.call("review", client=name, review=dict(replies=replies,
                online_input=online_input, target_indices=target_indices,
                generation_messages=self._review_messages))
        client = self._runtime.generator._llm[name]
        return review_replies(client, replies, online_input, target_indices, self._review_messages)

    def close(self):
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DigitalHumanReplyGenerator:
    def __init__(self, repository: Path, instance: str, *, timeout: float = 900, tool_registry=None,
                 instances_root=None, settings_file=None, env_file=None, review_client=None):
        self.repository, self.instance, self.timeout = repository, instance, timeout
        self.runtime_config = {key: value for key, value in dict(instances_root=instances_root,
            settings_file=settings_file, env_file=env_file, review_client=review_client).items() if value is not None}
        self.tool_registry = tool_registry
        self.llm_client = None
        self._client = None
        self.last_generation_failed = False
        self.last_exposure_block = None
        self.last_user_prompt = ""
        self.last_raw_response = ""
        self.last_generation_trace = []
        self.last_active_llm = "digital-human"
        self._client = DigitalHumanDraftClient(self.repository, self.instance, timeout=self.timeout,
                                               **self.runtime_config)

    @staticmethod
    def text_for_logging(text):
        return text

    @staticmethod
    def messages_for_logging(messages):
        return messages

    def generate(self, messages: list[ChatMessage], context: list[ChatMessage], *, is_group: bool = False,
                 tick_id: int = 0) -> list[str]:
        self.last_generation_failed = True
        self.last_exposure_block = None
        self.last_user_prompt = ""
        self.last_raw_response = ""
        self.last_generation_trace = []
        self.last_active_llm = "digital-human"
        try:
            if not messages:
                raise DigitalHumanError("缺少待回复消息")
            history = context or messages
            target_positions = [next((i for i, row in enumerate(history) if row is target), None)
                                for target in messages]
            if any(position is None for position in target_positions):
                raise DigitalHumanError("待回复消息不在当前上下文中")
            observed = any(row.observed_at is not None for row in history[-200:])
            if not observed:
                history = history[:max(target_positions) + 1]
            if min(target_positions) < len(history) - 200:
                raise DigitalHumanError("待回复消息超过 DH 的 200 条上下文限制")
            source_ids = {row.chatroom_id for row in history if row.chatroom_id}
            if len(source_ids) > 1 or (not observed and
                    (not source_ids or next(iter(source_ids)).startswith("_chats_"))):
                raise DigitalHumanError("需要唯一的原始聊天 ID；不能使用 OCR 合成 ID")
            online_input = messages_input(history[-200:], source_chat_id=next(iter(source_ids), None), is_group=is_group)
            target_indices = [position - max(0, len(history) - 200) for position in target_positions]
            cutoff = min(online_input["context"][index]["timestamp"] for index in target_indices) - 24 * 60 * 60
            retained = [index for index, row in enumerate(online_input["context"])
                        if index >= min(target_indices) or row["timestamp"] >= cutoff]
            online_input["context"] = [online_input["context"][index] for index in retained]
            target_indices = [retained.index(index) for index in target_indices]
            self.last_user_prompt = json.dumps(online_input, ensure_ascii=False)
            if self._client is None:
                self._client = DigitalHumanDraftClient(self.repository, self.instance, timeout=self.timeout,
                                                       **self.runtime_config)
            response = self._client.draft(online_input)
            self.last_raw_response = json.dumps(response, ensure_ascii=False)
            binding = response["binding"]
            self.last_active_llm = f"digital-human:{binding['instance']}/{binding['generator']}/{binding['data']}"
            self.last_generation_trace = [{"type": "digital_human_draft", "tick_id": tick_id,
                                           "response": response, "capture_scope": "draft_protocol"}]
            if response["replies"]:
                started = time.perf_counter()
                try:
                    review = self._client.review(response["replies"], online_input, target_indices)
                    self.last_generation_trace.append({"type": "reply_review", "tick_id": tick_id, **review,
                        "latency_ms": (time.perf_counter() - started) * 1000})
                    if review["decision"] == "block":
                        self.last_exposure_block = {"rule": ",".join(dict.fromkeys(
                            issue["category"] for issue in review["issues"])),
                            "issues": review["issues"], "replies": list(response["replies"])}
                        _logger.warning("大模型回复审查拦截，整轮不发送: %s", self.last_exposure_block)
                        self.last_generation_failed = False
                        return []
                except Exception as exc:
                    self.last_generation_trace.append({"type": "reply_review_error", "tick_id": tick_id,
                        "error": str(exc), "latency_ms": (time.perf_counter() - started) * 1000})
                    _logger.exception("大模型回复审查失败，本轮不发送，保留未读等待重试")
                    return []
            self.last_generation_failed = False
            return response["replies"]
        except Exception as exc:
            self.close()
            self.last_raw_response = json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
            self.last_generation_trace = [{"type": "digital_human_error", "tick_id": tick_id, "error": str(exc)}]
            _logger.exception("digital-human 生成失败，保留未回复消息；不回退旧生成器")
            return []

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
