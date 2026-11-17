#!/usr/bin/env python3
"""Run a small production-shaped historical C0 vs human reply audit."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import (  # noqa: E402
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _load_env,
    _parse_api_judge_response,
    build_api_judge_messages,
    load_volcengine_credentials,
)
from scripts.private_case_config import load_private_case_config  # noqa: E402
from scripts.seal_youzi_stage2_quota import load_source as load_youzi_source  # noqa: E402
from src.memory.engine import MemoryEngine  # noqa: E402
from src.models.base import ChatMessage, SenderType  # noqa: E402
from src.reply.generator import ReplyGenerator  # noqa: E402
from src.tools.builtin_tools import register_builtin_tools  # noqa: E402
from src.tools.tool_registry import ToolRegistry  # noqa: E402
from src.utils.qwen_client import QwenClient  # noqa: E402
from src.utils.xml_utils import _extract_xml_text  # noqa: E402

OUT_DIR = ROOT / "data/reports/private/current_c0_historical_ab_v1"
MANIFEST = ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json"
SIDE_EFFECT_TOOLS = {
    "tuya_control_device",
    "tuya_set_temperature",
    "print3d_scale_model",
    "print3d_update_support",
    "send_file",
}


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def _norm(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value or "", flags=re.UNICODE).lower()


def _message_text(row: dict[str, Any]) -> str:
    content = str(row.get("content") or row.get("parsedContent") or "").strip()
    local_type = int(row.get("localType") or 0)
    if content.startswith("<"):
        extracted = _extract_xml_text(content)
        if extracted:
            return extracted
    if content and not content.startswith("<"):
        return content
    return {3: "[图片]", 43: "[视频]", 47: "[表情]", 49: "[链接/引用]"}.get(local_type, "[非文本消息]")


def _sort_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        messages,
        key=lambda row: (
            int(row.get("createTime") or 0),
            int(row.get("sortSeq") or 0),
            str(row.get("serverId") or row.get("platformMessageId") or ""),
        ),
    )


def _load_cases(
    manifest_path: Path, case_ids: tuple[str, ...] | None
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case_ids is None:
        return payload["cases"]
    by_id = {row["internal_id"]: row for row in payload["cases"]}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise RuntimeError(f"holdout cases missing: {missing}")
    return [by_id[case_id] for case_id in case_ids]


def _source(case: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    source_path = str(case["source_path"])
    if "adversarial_stage2_youzi_quota_v1" in source_path:
        config = load_private_case_config('target_chat', 'target_chat_name')
        messages, _ = load_youzi_source()
        return messages, config['target_chat_name'], config['target_chat']
    path = ROOT / source_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    session = payload.get("session") or {}
    if case.get("specified_group") == "correct_common_prosperity_with_luxixi":
        chat_name = load_private_case_config('common_chat_name')['common_chat_name']
    else:
        chat_name = str(session.get("displayName") or session.get("nickname") or path.stem.removeprefix("群聊_"))
    chatroom_id = str(payload.get("talker") or session.get("wxid") or case.get("chat_id") or "")
    return _sort_messages(payload.get("messages") or []), chat_name, chatroom_id


def _target_index(messages: list[dict[str, Any]], case: dict[str, Any]) -> int:
    candidates = [
        index
        for index, row in enumerate(messages)
        if int(row.get("createTime") or 0) == int(case["timestamp"]) and bool(row.get("isSend"))
    ]
    exact = [index for index in candidates if _norm(_message_text(messages[index])) == _norm(str(case["reply"]))]
    if exact:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"cannot locate target {case['internal_id']} ({len(candidates)} timestamp candidates)")


def _sender(row: dict[str, Any], mapping: dict[str, str]) -> tuple[str, SenderType]:
    if bool(row.get("isSend")):
        return "本人", SenderType.SELF
    display_name = str(row.get("senderDisplayName") or "").strip()
    if display_name:
        return display_name, SenderType.OTHER
    identity = str(row.get("senderUsername") or "未知成员")
    return identity, SenderType.OTHER


def _chat_message(
    row: dict[str, Any], chat_name: str, chatroom_id: str, mapping: dict[str, str]
) -> ChatMessage:
    sender, sender_type = _sender(row, mapping)
    timestamp = int(row.get("createTime") or 0)
    local_type = int(row.get("localType") or 0)
    return ChatMessage(
        text=_message_text(row),
        sender=sender,
        sender_type=sender_type,
        chat_name=chat_name,
        chatroom_id=chatroom_id,
        timestamp=datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else None,
        message_type="text" if local_type == 1 else "other",
        local_id=row.get("localId"),
        server_id=str(row.get("serverId") or row.get("platformMessageId") or ""),
        create_time=timestamp,
        raw_type=local_type,
        sender_wxid=str(row.get("senderUsername") or ""),
    )


def _match_context(
    prefix: list[ChatMessage], context_rows: list[dict[str, Any]]
) -> tuple[list[ChatMessage], str]:
    wanted = [
        (_norm(_message_text(row)), str(row.get("speaker") or "") == "self")
        for row in context_rows
        if _message_text(row) != "[非文本消息]"
    ]
    wanted = [value for value in wanted if value[0]]
    if not wanted:
        return [], "manifest context is empty"
    positions: list[int] = []
    cursor = len(prefix) - 1
    for value, is_self in reversed(wanted):
        found = -1
        for index in range(cursor, max(-1, cursor - 180), -1):
            same_role = (prefix[index].sender_type is SenderType.SELF) == is_self
            if _norm(prefix[index].text) == value and same_role:
                found = index
                break
        if found < 0:
            return [], f"could not align context item: {value[:30]}"
        positions.append(found)
        cursor = found - 1
    positions.reverse()
    return [prefix[index] for index in positions], "aligned to raw source"


def _build_inputs(case: dict[str, Any]) -> tuple[list[ChatMessage], list[ChatMessage], dict[str, Any]]:
    rows, chat_name, chatroom_id = _source(case)
    target_index = _target_index(rows, case)
    mapping: dict[str, str] = {}
    raw_prefix = [
        _chat_message(row, chat_name, chatroom_id, mapping)
        for row in rows[:target_index]
        if int(row.get("localType") or 0) != 10000
    ]
    context, alignment = _match_context(raw_prefix, case.get("context_messages") or [])
    if not context:
        context = [message for message in raw_prefix[-80:] if message.sender_type is SenderType.OTHER]
        alignment += "; fallback to last 80 raw messages"
    all_messages = raw_prefix[-1000:]
    last_self = max(
        (index for index, message in enumerate(context) if message.sender_type is SenderType.SELF),
        default=-1,
    )
    incoming = [message for message in context[last_self + 1 :] if message.sender_type is SenderType.OTHER]
    incoming_ids = {id(message) for message in incoming}
    to_reply = [message for message in all_messages if id(message) in incoming_ids]
    if not to_reply:
        raise RuntimeError(f"no incoming context for {case['internal_id']}")
    return to_reply, all_messages, {
        "chat_name": chat_name,
        "chatroom_id": chatroom_id,
        "target_index": target_index,
        "raw_history_before_target": len(raw_prefix),
        "generator_history_count": len(all_messages),
        "unread_count": len(to_reply),
        "context_alignment": alignment,
        "distinct_other_senders": len({message.sender for message in all_messages if message.sender_type is SenderType.OTHER}),
        "source_path": case["source_path"],
    }


def _safe_registry(evaluation_time: datetime) -> ToolRegistry:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    if registry.has("get_current_time"):
        registry.get("get_current_time").func = lambda: evaluation_time.strftime("%Y-%m-%d %H:%M:%S")
    for name in SIDE_EFFECT_TOOLS:
        if registry.has(name):
            registry.get(name).func = lambda _name=name, **_kwargs: f"[历史离线评测已阻断副作用工具：{_name}]"
    return registry


def _trace(generator: ReplyGenerator) -> dict[str, Any]:
    return _jsonable({
        "system_prompt": generator.last_system_prompt,
        "user_prompt": generator.last_user_prompt,
        "tools_context": generator.last_tools_context,
        "few_shot_ids": generator.last_few_shot_ids,
        "few_shot_content": generator.last_few_shot_content,
        "loaded_skills": generator.last_loaded_skills,
        "skill_content": generator.last_skill_injected_content,
        "raw_response": generator.last_raw_response,
        "llm_calls": generator.last_llm_calls,
        "tool_calls": generator.last_tool_calls,
        "generation_trace": generator.last_generation_trace,
        "llm_messages": generator.last_llm_messages,
        "self_refine": {
            "applied": generator.last_self_refine_applied,
            "decision": generator.last_feedback_decision,
            "issues": generator.last_feedback_issues,
            "feedback_raw": generator.last_feedback_raw,
            "iterate_count": generator.last_iterate_count,
            "iterate_raw": generator.last_iterate_raw,
            "iterate_skipped_no_budget": generator.last_iterate_skipped_no_budget,
        },
        "fact_check": generator.last_fact_check,
    })


def _generate_case(
    case: dict[str, Any], llm: Any, memory: MemoryEngine,
    generation_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    to_reply, all_messages, metadata = _build_inputs(case)
    evaluation_time = datetime.fromtimestamp(int(case["timestamp"]))
    generator = ReplyGenerator(
        llm_client=llm,
        memory_engine=memory,
        tool_registry=_safe_registry(evaluation_time),
        judge_worker=None,
        enable_timestamps=True,
        generation_timeout_seconds=generation_timeout_seconds,
    )
    started = time.monotonic()
    replies = generator.generate(
        to_reply,
        all_messages,
        is_group=case.get("chat_type") == "group",
        tick_id=0,
        evaluation_time=evaluation_time,
    )
    trace = _trace(generator)
    human = str(case["reply"])
    trace_text = json.dumps(trace, ensure_ascii=False)
    normalized_human = _norm(human)
    normalized_ai = _norm("".join(replies))
    similarity = difflib.SequenceMatcher(None, normalized_human, normalized_ai).ratio() if normalized_human and normalized_ai else 0.0
    side_effect_calls = [
        row for row in generator.last_tool_calls
        if str(row.get("tool_name") or row.get("name") or "") in SIDE_EFFECT_TOOLS
    ]
    leakage = bool(normalized_human and normalized_human in _norm(trace_text))
    reproduced = bool(normalized_human and normalized_ai and (normalized_human == normalized_ai or (min(len(normalized_human), len(normalized_ai)) >= 6 and similarity >= 0.9)))
    context_view = [
        {
            "time": message.timestamp,
            "speaker": message.sender,
            "role": message.sender_type.value,
            "content": message.text,
            "unread": message in to_reply,
        }
        for message in all_messages[-80:]
    ]
    return {
        "case_id": case["internal_id"],
        "group": (
            "youzi_group"
            if str(case.get("specified_group") or "").startswith("youzi_group")
            else case.get("specified_group") or "other_complex_large_group"
        ),
        "chat_name": metadata["chat_name"],
        "chat_type": case.get("chat_type"),
        "complex_large_group": bool(case.get("complex_large_group")),
        "participation_bucket": (
            "must_reply" if case.get("chat_type") == "private" else "optional_reply"
        ),
        "timestamp": int(case["timestamp"]),
        "human_reply": [human],
        "c0_reply": replies,
        "context": context_view,
        "unread_context": [
            {"time": message.timestamp, "speaker": message.sender, "content": message.text}
            for message in to_reply
        ],
        "generation": {
            "failed": bool(generator.last_generation_failed),
            "duration_seconds": round(time.monotonic() - started, 3),
            "timeout_budget_seconds": generation_timeout_seconds,
            "model": str(getattr(llm, "model", "") or os.environ.get("LLM_MODEL", "deepseek-v4-flash")),
            "base_url_host": re.sub(
                r"^https?://([^/]+).*$",
                r"\1",
                str(getattr(llm, "base_url", "") or os.environ.get("LLM_BASE_URL", "")),
            ),
            "current_generator_sha256": hashlib.sha256((ROOT / "src/reply/generator.py").read_bytes()).hexdigest(),
            "metadata": metadata,
            "trace": trace,
        },
        "validity": {
            "answer_in_c0_trace": leakage,
            "c0_reproduced_human": reproduced,
            "reply_similarity": round(similarity, 4),
            "side_effect_tool_calls": _jsonable(side_effect_calls),
            "formally_scorable": not generator.last_generation_failed and not leakage and not reproduced and not side_effect_calls,
            "production_differences": [
                "历史单步回放：生成前使用原始历史导出，生成后不发送、不写生产状态",
                "get_current_time 返回题目发生时间，避免历史时间穿越",
                "有副作用工具保留生产 schema，但执行被阻断；若模型调用则该题不正式计分",
                "当前生成器的“最近30分钟”筛选内部仍使用运行时钟；脚本未修改 C0",
            ],
        },
    }


def _blind(case: dict[str, Any]) -> tuple[dict[str, Any], str]:
    human_option = "A" if int(hashlib.sha256(case["case_id"].encode()).hexdigest(), 16) % 2 == 0 else "B"
    options = {
        human_option: case["human_reply"],
        "B" if human_option == "A" else "A": case["c0_reply"],
    }
    anonymous_speakers: dict[str, str] = {}
    context = []
    for row in case["context"]:
        if row.get("role") == SenderType.SELF.value:
            speaker = "本人"
        elif case["chat_type"] == "private":
            speaker = "对方"
        else:
            original = str(row.get("speaker") or "未知成员")
            if original not in anonymous_speakers:
                anonymous_speakers[original] = f"成员{len(anonymous_speakers) + 1}"
            speaker = anonymous_speakers[original]
        context.append({"time": row["time"], "speaker": speaker, "content": row["content"]})
    return {
        "case_id": case["case_id"],
        "chat_type": case["chat_type"],
        "chat_name": "匿名群聊" if case["chat_type"] == "group" else "匿名私聊",
        "context": context,
        "option_A": options["A"],
        "option_B": options["B"],
    }, human_option


def _judge(case: dict[str, Any], client: VolcengineAnthropicClient) -> dict[str, Any]:
    blind, human_option = _blind(case)
    messages = build_api_judge_messages(blind, {"facts": [], "examples": []})
    started = time.monotonic()
    raw = client.chat(messages=messages, temperature=0, max_tokens=400, timeout=500, raise_on_error=True, max_retries=0)
    raw_text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
    parsed = _parse_api_judge_response(raw_text)
    return {
        "human_option": human_option,
        "selected_option": parsed["human_option"],
        "correct": parsed["human_option"] == human_option,
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "model_requested": client.model,
        "model_resolved": client.resolved_model,
        "blind_case": blind,
        "request_messages": messages,
        "request_body": _jsonable(client.last_request_body),
        "raw_response": raw_text,
        "response_payload": _jsonable(client.last_response_payload),
        "isolation": {
            "one_case_per_request": True,
            "identity_labels_absent": True,
            "answer_key_absent": True,
            "c0_prompt_absent": True,
            "tools_disabled": True,
        },
    }


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pretty(value: Any) -> str:
    return _esc(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _review_speaker(value: Any, mapping: dict[str, str]) -> str:
    speaker = str(value or "").strip()
    if speaker == "本人":
        return speaker
    if speaker and not re.match(r"^(?:wxid_|gh_|\d+@openim$|.+@chatroom$)", speaker):
        return speaker
    key = speaker or "missing"
    if key not in mapping:
        mapping[key] = f"未知成员{len(mapping) + 1}"
    return mapping[key]


def _render(data: dict[str, Any]) -> str:
    cards = []
    for index, case in enumerate(data["cases"], 1):
        judge = case.get("judge") or {}
        blind = judge.get("blind_case") or {}
        review_speakers: dict[str, str] = {}
        context_lines = "".join(
            f'<div class="chat-row {"self" if row["role"] == "self" else "other"}"><div class="chat-speaker"><b>{_esc(_review_speaker(row["speaker"], review_speakers))}</b><span>{_esc(row["time"] or "")}</span></div><div class="chat-bubble">{_esc(row["content"])}</div></div>'
            for row in case["context"]
        )
        option_a = "<br>".join(_esc(x) for x in blind.get("option_A") or ["（空回复）"])
        option_b = "<br>".join(_esc(x) for x in blind.get("option_B") or ["（空回复）"])
        validity = case["validity"]
        status = "可计分" if validity["formally_scorable"] else "排除"
        participation = "明确应回复" if case["participation_bucket"] == "must_reply" else "可回可不回"
        outcome = (
            "生成失败" if case["generation"]["failed"]
            else "C0 实际回复" if case["c0_reply"]
            else "C0 沉默"
        )
        fact_check = case["generation"]["trace"].get("fact_check") or {}
        fact_label = (
            "触发并改写" if fact_check.get("rewrite_applied")
            else "触发并拦截" if fact_check.get("caused_feedback_fail")
            else "已核查通过" if fact_check.get("attempted") and fact_check.get("status") == "passed"
            else "调用异常" if str(fact_check.get("status", "")).endswith("error")
            else "生产配置关闭（Self-Refine 未启用）" if fact_check.get("status") == "self_refine_disabled"
            else "主生成未完成，尚未到核查" if fact_check.get("status") == "not_reached"
            else f"已跳过：{fact_check.get('skipped_reason')}" if fact_check.get("skipped_reason")
            else "未触发"
        )
        cards.append(f"""
        <article class="case" data-participation="{_esc(case['participation_bucket'])}" data-outcome="{_esc(outcome)}">
          <header><div><small>#{index} · {_esc(case['chat_name'])}</small><h2>{_esc(datetime.fromtimestamp(case['timestamp']).strftime('%Y-%m-%d %H:%M'))}</h2></div><span class="pill">{participation} · {outcome} · {status}</span></header>
          <h3>完整聊天上下文（{len(case['context'])} 条）</h3><div class="chat-view">{context_lines}</div>
          <div class="options">
            <section><b>回复 A</b><p>{option_a}</p></section>
            <section><b>回复 B</b><p>{option_b}</p></section>
          </div>
          <div class="judge"><b>Judge 选 {_esc(judge.get('selected_option', '未运行'))}</b><span>置信度 {_esc(judge.get('confidence', '-'))}</span><p>{_esc(judge.get('reason', ''))}</p></div>
          <details><summary>揭晓身份与有效性</summary><div class="audit"><p><b>真人：</b>{'<br>'.join(_esc(x) for x in case['human_reply'])}</p><p><b>当前 C0：</b>{'<br>'.join(_esc(x) for x in case['c0_reply']) or '（空回复）'}</p><pre>{_pretty(validity)}</pre></div></details>
          <details><summary>事实核查：{_esc(fact_label)}</summary><div class="audit"><h3>是否触发、判断与改写结果</h3><pre>{_pretty(fact_check)}</pre></div></details>
          <details><summary>C0 原始提示词、few-shot 与调用链</summary><div class="audit"><h3>System prompt</h3><pre>{_esc(case['generation']['trace']['system_prompt'])}</pre><h3>User prompt</h3><pre>{_esc(case['generation']['trace']['user_prompt'])}</pre><h3>Few-shot</h3><pre>{_esc(case['generation']['trace']['few_shot_content'])}</pre><h3>LLM calls</h3><pre>{_pretty(case['generation']['trace']['llm_calls'])}</pre><h3>Tool calls</h3><pre>{_pretty(case['generation']['trace']['tool_calls'])}</pre></div></details>
          <details><summary>Judge 原始提示词与 API 调用</summary><div class="audit"><h3>Messages</h3><pre>{_pretty(judge.get('request_messages'))}</pre><h3>Request body</h3><pre>{_pretty(judge.get('request_body'))}</pre><h3>Raw response</h3><pre>{_esc(judge.get('raw_response', ''))}</pre><h3>隔离检查</h3><pre>{_pretty(judge.get('isolation'))}</pre></div></details>
        </article>""")
    summary = data["summary"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>当前 C0 历史离线 A/B</title><style>
    :root{{--bg:#0d1117;--card:#161b22;--line:#30363d;--text:#f0f6fc;--muted:#8b949e;--accent:#58a6ff;--self:#164e2e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:920px;margin:auto;padding:14px;min-width:0}}h1{{font-size:22px;margin:4px 0}}h2{{font-size:17px;margin:2px 0}}h3{{font-size:13px;color:var(--muted);margin:16px 0 6px}}.lede{{color:var(--muted);margin:6px 0 16px}}.summary{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:14px}}.summary div,.case{{background:var(--card);border:1px solid var(--line);border-radius:12px}}.summary div{{padding:12px;text-align:center}}.summary b{{display:block;font-size:20px;color:var(--accent)}}.summary span{{font-size:12px;color:var(--muted)}}.controls{{position:sticky;top:6px;z-index:10;display:flex;align-items:center;gap:8px;background:rgba(13,17,23,.96);backdrop-filter:blur(8px);border:1px solid var(--line);border-radius:12px;padding:10px;margin:14px 0}}.controls button{{appearance:none;border:1px solid var(--accent);background:var(--card);color:var(--text);border-radius:8px;padding:9px 13px;font-size:14px}}.controls button:disabled{{opacity:.35}}.progress{{flex:1;text-align:center;font-size:13px;color:var(--muted)}}.case{{display:none;padding:14px;margin:12px 0}}.case.active{{display:block}}header{{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}}small,header span,.judge span{{color:var(--muted)}}.pill{{border:1px solid #d29922;border-radius:999px;padding:4px 9px;font-size:12px}}.chat-view{{display:flex;flex-direction:column;gap:8px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px;min-width:0}}.chat-row{{display:flex;flex-direction:column;align-items:flex-start;max-width:90%}}.chat-row.self{{align-self:flex-end;align-items:flex-end}}.chat-speaker{{display:flex;gap:8px;font-size:11px;color:var(--muted);margin:0 6px 3px}}.chat-bubble{{background:#21262d;border-radius:4px 12px 12px 12px;padding:7px 10px;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}}.chat-row.self .chat-bubble{{background:var(--self);border-radius:12px 4px 12px 12px}}.options{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}}.options section{{border:1px solid var(--line);border-radius:10px;padding:11px;background:var(--bg)}}.options p{{margin:8px 0 0;font-size:16px;white-space:pre-wrap;overflow-wrap:anywhere}}.judge{{margin-top:10px;padding:10px;border-left:3px solid #d29922;background:#211d16;border-radius:5px}}.judge span{{float:right;font-size:12px}}.judge p{{margin:5px 0 0}}details{{border-top:1px solid var(--line);margin-top:12px;padding-top:10px}}summary{{cursor:pointer;font-weight:650;color:var(--accent)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;max-height:38rem;overflow:auto;background:#080b10;border:1px solid var(--line);border-radius:8px;padding:10px;font-size:11px}}@media(max-width:620px){{main{{padding:10px;padding-bottom:calc(18px + env(safe-area-inset-bottom))}}.summary{{grid-template-columns:1fr 1fr}}.options{{grid-template-columns:1fr}}.case{{padding:11px}}.controls button{{min-width:72px;padding:9px 10px}}.progress{{font-size:12px}}details pre{{max-height:60vh}}}}
    </style></head><body><main><h1>当前 C0 × 真人历史离线 A/B</h1><p class="lede">历史完整聊天生成前对齐；生成后停止，不发送。页面显示历史导出的真实昵称；Judge 收到的是重新匿名后的成员名。</p><div class="summary"><div><b>{summary['total']}</b><span>固定样本</span></div><div><b>{summary['participation']['must_reply']}</b><span>明确应回复</span></div><div><b>{summary['participation']['optional_reply']}</b><span>可回可不回</span></div><div><b>{summary['outcomes']['generation_failed']}</b><span>生成失败</span></div><div><b>{summary['outcomes']['c0_silent']}</b><span>C0 沉默</span></div><div><b>{summary['outcomes']['c0_replied']}</b><span>C0 实际回复</span></div><div><b>{summary['judge_correct']}/{summary['judge_scored']}</b><span>非空回复 Judge</span></div><div><b>{summary['fact_check']['attempted']}</b><span>事实核查实际调用</span></div><div><b>{summary['fact_check']['issues_found']}</b><span>发现问题并拦截</span></div><div><b>{summary['fact_check']['rewrite_applied']}</b><span>完成改写</span></div><div><b>{summary['fact_check']['disabled']}</b><span>生产配置关闭</span></div><div><b>{summary['fact_check']['not_reached']}</b><span>主生成未完成</span></div><div><b>{summary['fact_check']['errors']}</b><span>核查调用异常</span></div></div><p class="lede">覆盖：群聊 {summary['coverage']['group_chat']}（复杂大群 {summary['coverage']['complex_large_group']}），私聊 {summary['coverage']['private_chat']}；正确共同富裕群 {summary['coverage']['correct_common_prosperity_with_luxixi']}，柚子群 {summary['coverage']['youzi_group']}。指定群不在这套已封存 Stage 1 中，未伪装成已覆盖。</p><nav class="controls" aria-label="题目导航"><button id="prev" type="button">上一题</button><span class="progress" id="progress">准备中</span><button id="next" type="button">下一题</button></nav><div id="deck">{''.join(cards)}</div></main><script>(()=>{{const cards=[...document.querySelectorAll('.case')];const progress=document.getElementById('progress');const prev=document.getElementById('prev');const next=document.getElementById('next');let index=0;function show(value){{if(!cards.length){{progress.textContent='0 / 0';prev.disabled=next.disabled=true;return}}index=Math.max(0,Math.min(value,cards.length-1));cards.forEach((card,i)=>card.classList.toggle('active',i===index));progress.textContent=`${{index+1}} / ${{cards.length}}`;prev.disabled=index===0;next.disabled=index===cards.length-1;window.scrollTo({{top:document.querySelector('.controls').offsetTop-6,behavior:'smooth'}})}}prev.addEventListener('click',()=>show(index-1));next.addEventListener('click',()=>show(index+1));show(0)}})();</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="reuse already generated cases")
    parser.add_argument("--all-cases", action="store_true", help="evaluate every case in the manifest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-indices", help="1-based comma-separated manifest positions")
    parser.add_argument("--generation-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--generation-provider",
        choices=("official", "volcengine"),
        default="official",
        help="provider used only for C0 generation",
    )
    parser.add_argument("--skip-judge", action="store_true", help="skip the auxiliary external Judge")
    args = parser.parse_args()
    _load_env()
    cases = _load_cases(
        args.manifest,
        None if args.all_cases else tuple(load_private_case_config('case_ids')['case_ids']),
    )
    if args.case_indices:
        selected_indices = {int(value) for value in args.case_indices.split(",")}
        cases = [case for position, case in enumerate(cases, 1) if position in selected_indices]
    if args.limit is not None:
        cases = cases[: args.limit]
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    results_path = out_dir / "results.json"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        existing = {row["case_id"]: row for row in prior.get("cases") or []}
    if args.generation_provider == "volcengine":
        api_key, base_url = load_volcengine_credentials(Path.home() / ".cc-connect/config.toml")
        llm = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    else:
        llm = QwenClient()
    memory = MemoryEngine(llm_client=llm)
    try:
        judge_client = None
        if not args.skip_judge:
            api_key, base_url = load_volcengine_credentials(Path.home() / ".cc-connect/config.toml")
            judge_client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
        output_cases = []
        for index, source_case in enumerate(cases, 1):
            case_id = source_case["internal_id"]
            row = existing.get(case_id)
            retrying_failed_case = bool(row and row.get("generation", {}).get("failed"))
            prior_timeout = float((row or {}).get("generation", {}).get("timeout_budget_seconds") or 20.0)
            needs_generation = row is None or (
                retrying_failed_case and prior_timeout < args.generation_timeout_seconds
            )
            if needs_generation:
                print(f"[{index}/{len(cases)}] generating {case_id}", flush=True)
                row = _generate_case(
                    source_case,
                    llm,
                    memory,
                    generation_timeout_seconds=args.generation_timeout_seconds,
                )
                row["generation"]["retry_of_failed_case"] = retrying_failed_case
            else:
                validity = row["validity"]
                validity["formally_scorable"] = not (
                    validity["answer_in_c0_trace"]
                    or validity["c0_reproduced_human"]
                    or validity["side_effect_tool_calls"]
                )
            if (
                row["validity"]["formally_scorable"]
                and row["c0_reply"]
                and judge_client is not None
                and not row.get("judge")
                and not row.get("judge_error")
            ):
                print(f"[{index}/{len(cases)}] judging {case_id}", flush=True)
                try:
                    row["judge"] = _judge(row, judge_client)
                    row.pop("judge_error", None)
                except Exception as exc:
                    row["judge_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[{index}/{len(cases)}] judge failed: {row['judge_error']}", flush=True)
            output_cases.append(row)
            remaining_existing = [
                existing[candidate["internal_id"]]
                for candidate in cases[index:]
                if candidate["internal_id"] in existing
            ]
            partial = {
                "schema_version": 1,
                "status": "running",
                "created_at": datetime.now().astimezone().isoformat(),
                "cases": output_cases + remaining_existing,
            }
            _write_json(results_path, partial)
    finally:
        memory.shutdown()
    scored = [row for row in output_cases if row["validity"]["formally_scorable"]]
    judge_scored = [row for row in scored if row.get("judge")]
    summary = {
        "total": len(output_cases),
        "formally_scorable": len(scored),
        "excluded": len(output_cases) - len(scored),
        "judge_scored": len(judge_scored),
        "judge_correct": sum(bool(row["judge"]["correct"]) for row in judge_scored),
        "participation": {
            "must_reply": sum(row["participation_bucket"] == "must_reply" for row in output_cases),
            "optional_reply": sum(row["participation_bucket"] == "optional_reply" for row in output_cases),
        },
        "outcomes": {
            "generation_failed": sum(bool(row["generation"]["failed"]) for row in output_cases),
            "c0_silent": sum(not row["generation"]["failed"] and not row["c0_reply"] for row in output_cases),
            "c0_replied": sum(not row["generation"]["failed"] and bool(row["c0_reply"]) for row in output_cases),
        },
        "fact_check": {
            "attempted": sum(bool(row["generation"]["trace"].get("fact_check", {}).get("attempted")) for row in output_cases),
            "issues_found": sum(bool(row["generation"]["trace"].get("fact_check", {}).get("caused_feedback_fail")) for row in output_cases),
            "rewrite_applied": sum(bool(row["generation"]["trace"].get("fact_check", {}).get("rewrite_applied")) for row in output_cases),
            "disabled": sum(row["generation"]["trace"].get("fact_check", {}).get("status") == "self_refine_disabled" for row in output_cases),
            "not_reached": sum(row["generation"]["trace"].get("fact_check", {}).get("status") == "not_reached" for row in output_cases),
            "errors": sum(str(row["generation"]["trace"].get("fact_check", {}).get("status", "")).endswith("error") for row in output_cases),
        },
        "coverage": {
            "group_chat": sum(row["chat_type"] == "group" for row in output_cases),
            "private_chat": sum(row["chat_type"] == "private" for row in output_cases),
            "complex_large_group": sum(row["complex_large_group"] for row in output_cases),
            "correct_common_prosperity_with_luxixi": sum(row["group"] == "correct_common_prosperity_with_luxixi" for row in output_cases),
            "youzi_group": sum(row["group"] == "youzi_group" for row in output_cases),
        },
        "interpretation": "Judge is auxiliary only; this page does not use Judge alone to declare C0 better or worse than the human replies.",
    }
    final = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "cases": output_cases,
    }
    _write_json(results_path, final)
    (out_dir / "index.html").write_text(_render(final), encoding="utf-8")
    (out_dir / "index.html").chmod(0o600)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
