#!/usr/bin/env python3
"""Build a read-only review of ambiguous self messages near Bot sends."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data/private_benchmarks/judge_bot_adversarial_sources/20260818_account20b2"
SOURCE_GLOB = "youzi_group_2_page5k_*.json"
MEMBERS_PATH = SOURCE_DIR / "youzi_group_2_members.json"
LOG_PATHS = (ROOT / "data/logs/execution.jsonl.1", ROOT / "data/logs/execution.jsonl")
OUTPUT_DIR = ROOT / "data/reports/private/youzi_identity_review_v1"
TARGET_CHAT = "50570746383@chatroom"
TARGET_CHAT_NAME = "柚子群2"
SELF_SENDER = "wxid_qxlscnesk92m21"
EVENT_GAP_SECONDS = 30 * 60
COLLISION_SECONDS = 10
MIN_QUALIFIED_EVENTS = 7
TZ = ZoneInfo("Asia/Shanghai")
PENDING_CONFIRMATION_SERVER_IDS = (
    "5278612712177403526",
    "2039022965924810276",
)

USER_CONFIRMED_HUMAN_SERVER_IDS = frozenset(
    {
        "3732567670665303461",
        "843420406975241680",
        "9137219767804178700",
        "1626499216596393015",
        "7126116260531224334",
        "5798129490978776406",
        "6400981979698023102",
        "3292358293486618907",
        "3945254183450218897",
        "8471931083934576241",
        "8856833826520054690",
        "615155421457512356",
        "5278612712177403526",
        "2039022965924810276",
    }
)
KNOWN_EXPOSED_SERVER_IDS = USER_CONFIRMED_HUMAN_SERVER_IDS
USER_CONFIRMATION = {
    "status": "用户已确认真人",
    "confirmed_on": "2026-08-19",
    "source": "用户明确确认：柚子群中显示为王芊的气泡均为真人手发；Bot 只使用不爱说话身份在该群回复",
    "scope": "适用于目标柚子群中 senderUsername 为王芊账号的气泡；不爱说话身份不得作为真人标准；不确认邻近 Bot 日志文本",
}


def is_user_confirmed_human(message: dict[str, Any]) -> bool:
    return (
        str(message.get("senderUsername") or "") == SELF_SENDER
        and bool(message.get("isSend"))
    )


def _message_key(message: dict[str, Any]) -> str:
    server_id = str(message.get("serverId") or "").strip()
    if server_id:
        return f"server:{server_id}"
    stable = {
        "localId": message.get("localId"),
        "createTime": message.get("createTime"),
        "senderUsername": message.get("senderUsername"),
        "content": message.get("content"),
    }
    return "fallback:" + hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_messages(source_dir: Path = SOURCE_DIR) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    paths = sorted(source_dir.glob(SOURCE_GLOB))
    if not paths:
        raise RuntimeError("Youzi source pages are missing")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("talker")) != TARGET_CHAT:
            raise RuntimeError(f"unexpected talker in {path.name}")
        for message in payload.get("messages") or []:
            unique.setdefault(_message_key(message), message)
    return sorted(
        unique.values(),
        key=lambda row: (
            int(row.get("createTime") or 0),
            int(row.get("sortSeq") or 0),
            str(row.get("serverId") or ""),
        ),
    )


def load_members(path: Path = MEMBERS_PATH) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {SELF_SENDER: "我"}
    for member in payload.get("members") or []:
        wxid = str(member.get("wxid") or "")
        if not wxid:
            continue
        result[wxid] = next(
            (
                str(member.get(key) or "").strip()
                for key in ("groupNickname", "displayName", "nickname")
                if str(member.get(key) or "").strip()
            ),
            wxid,
        )
    return result


def _parse_log_time(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)
    return parsed.timestamp()


def load_successful_sends(log_paths: tuple[Path, ...] = LOG_PATHS) -> list[dict[str, Any]]:
    ticks: dict[str, dict[str, Any]] = {}
    for path in log_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tick_id = str(row.get("tick_id") or "")
                if not tick_id:
                    continue
                event = str(row.get("event") or "")
                if event not in {"layout", "messages", "decision", "send"}:
                    continue
                tick = ticks.setdefault(tick_id, {"tick_id": tick_id})
                tick[event] = row

    sends: list[dict[str, Any]] = []
    for tick in ticks.values():
        layout = tick.get("layout") or {}
        decision = tick.get("decision") or {}
        send = tick.get("send") or {}
        if layout.get("chat_name") != TARGET_CHAT_NAME or send.get("success") is not True:
            continue
        reply_text = str(decision.get("reply_text") or "").strip()
        if not reply_text or not send.get("ts"):
            continue
        sends.append(
            {
                "tick_id": tick["tick_id"],
                "timestamp": _parse_log_time(str(send["ts"])),
                "observed_at": str(send["ts"]),
                "reply_text": reply_text,
                "send_success": True,
                "send_preview": str(send.get("text_preview") or ""),
                "decision_reason": str(decision.get("reason") or ""),
                "production_context": (tick.get("messages") or {}).get("message_details") or [],
            }
        )
    return sorted(sends, key=lambda row: row["timestamp"])


def split_events(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    events: list[list[dict[str, Any]]] = []
    previous = 0
    for message in messages:
        timestamp = int(message.get("createTime") or 0)
        if not events or (previous and timestamp and timestamp - previous > EVENT_GAP_SECONDS):
            events.append([])
        events[-1].append(message)
        if timestamp:
            previous = timestamp
    return events


def _event_hash(event: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(_message_key(message) for message in event).encode("utf-8")
    ).hexdigest()


def build_candidates(
    messages: list[dict[str, Any]],
    sends: list[dict[str, Any]],
    members: dict[str, str],
    collision_seconds: int = COLLISION_SECONDS,
) -> list[dict[str, Any]]:
    event_by_key: dict[str, list[dict[str, Any]]] = {}
    for event in split_events(messages):
        for message in event:
            event_by_key[_message_key(message)] = event

    candidates: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("senderUsername") or "") != SELF_SENDER:
            continue
        if not bool(message.get("isSend")) or int(message.get("localType") or 0) != 1:
            continue
        content = str(message.get("content") or "").strip()
        timestamp = int(message.get("createTime") or 0)
        if not content or not timestamp:
            continue
        server_id = str(message.get("serverId") or "")
        confirmed = is_user_confirmed_human(message)
        nearest = min(sends, key=lambda row: abs(float(row["timestamp"]) - timestamp)) if sends else None
        delta = float(nearest["timestamp"]) - timestamp if nearest else None
        if not confirmed and (delta is None or abs(delta) > collision_seconds):
            continue
        context = []
        event = event_by_key[_message_key(message)]
        for item in event:
            sender = str(item.get("senderUsername") or "")
            context.append(
                {
                    "server_id": str(item.get("serverId") or ""),
                    "timestamp": int(item.get("createTime") or 0),
                    "sender": members.get(sender, sender or "未知成员"),
                    "is_self": sender == SELF_SENDER,
                    "content": _display_content(item),
                }
            )
        candidates.append(
            {
                "case_id": f"youzi-identity-{len(candidates) + 1:02d}",
                "identity_status": USER_CONFIRMATION["status"] if confirmed else "身份无法确认",
                "identity_confirmation": USER_CONFIRMATION if confirmed else None,
                "event_hash": _event_hash(event),
                "wechat_message": {
                    "server_id": server_id,
                    "timestamp": timestamp,
                    "observed_at": _format_ts(timestamp),
                    "content": content,
                    "sender": "我",
                    "account_evidence": (
                        "用户已明确确认：目标柚子群内王芊账号气泡均为真人手发"
                        if confirmed
                        else "isSend=1 + 本人 senderUsername，仅证明从本人微信账号发出"
                    ),
                },
                "bot_send": (
                    {**nearest, "delta_seconds": round(delta, 3)}
                    if nearest is not None and abs(delta) <= collision_seconds
                    else None
                ),
                "event_context": context,
            }
        )
    return candidates


def build_pending_confirmation_candidates(
    messages: list[dict[str, Any]],
    members: dict[str, str],
    server_ids: tuple[str, ...] = PENDING_CONFIRMATION_SERVER_IDS,
) -> list[dict[str, Any]]:
    event_by_key: dict[str, list[dict[str, Any]]] = {}
    message_by_server_id: dict[str, dict[str, Any]] = {}
    for event in split_events(messages):
        for message in event:
            event_by_key[_message_key(message)] = event
            server_id = str(message.get("serverId") or "")
            if server_id:
                message_by_server_id[server_id] = message

    candidates = []
    for server_id in server_ids:
        if server_id in USER_CONFIRMED_HUMAN_SERVER_IDS:
            continue
        message = message_by_server_id.get(server_id)
        if message is None:
            raise RuntimeError(f"pending confirmation message is missing: {server_id}")
        event = event_by_key[_message_key(message)]
        context = []
        for item in event:
            sender = str(item.get("senderUsername") or "")
            context.append(
                {
                    "server_id": str(item.get("serverId") or ""),
                    "timestamp": int(item.get("createTime") or 0),
                    "sender": members.get(sender, sender or "未知成员"),
                    "is_self": sender == SELF_SENDER,
                    "content": _display_content(item),
                }
            )
        timestamp = int(message.get("createTime") or 0)
        candidates.append(
            {
                "server_id": server_id,
                "identity_status": "身份尚未确认，不计分",
                "event_hash": _event_hash(event),
                "timestamp": timestamp,
                "observed_at": _format_ts(timestamp),
                "sender": members.get(SELF_SENDER, "我"),
                "content": _display_content(message),
                "account_evidence": "isSend=1 只证明从本人微信账号发出，不能证明是真人手发",
                "event_context": context,
            }
        )
    return candidates


def assess_qualification(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed = [
        case for case in candidates if case["identity_status"] == USER_CONFIRMATION["status"]
    ]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for case in confirmed:
        by_event.setdefault(case["event_hash"], []).append(case)

    selected_ids: list[str] = []
    exposed_ids: list[str] = []
    duplicate_event_ids: list[str] = []
    for event_cases in by_event.values():
        unexposed = [
            case
            for case in event_cases
            if case["wechat_message"]["server_id"] not in KNOWN_EXPOSED_SERVER_IDS
        ]
        exposed_ids.extend(
            case["wechat_message"]["server_id"]
            for case in event_cases
            if case["wechat_message"]["server_id"] in KNOWN_EXPOSED_SERVER_IDS
        )
        if not unexposed:
            continue
        selected_ids.append(unexposed[0]["wechat_message"]["server_id"])
        duplicate_event_ids.extend(
            case["wechat_message"]["server_id"] for case in unexposed[1:]
        )

    selected = set(selected_ids)
    exposed = set(exposed_ids)
    duplicate = set(duplicate_event_ids)
    for case in candidates:
        server_id = case["wechat_message"]["server_id"]
        if case["identity_status"] != USER_CONFIRMATION["status"]:
            status = "身份未确认，不可入题"
        elif server_id in exposed:
            status = "真人身份已确认，但原话已暴露，不可作独立题"
        elif server_id in duplicate:
            status = "真人身份已确认，但同一事件只保留一题"
        elif server_id in selected:
            status = "真人身份已确认，可作为独立题样本基础"
        else:
            status = "真人身份已确认，但当前不可入题"
        case["qualification_status"] = status
        case["selected_as_event_representative"] = server_id in selected

    qualified_count = len(selected_ids)
    return {
        "confirmed_bubbles": len(confirmed),
        "distinct_confirmed_events": len(by_event),
        "known_exposed_bubbles": len(exposed_ids),
        "qualified_independent_event_representatives": qualified_count,
        "minimum_required_before_sealing": MIN_QUALIFIED_EVENTS,
        "gap": max(0, MIN_QUALIFIED_EVENTS - qualified_count),
        "ready_to_seal": qualified_count >= MIN_QUALIFIED_EVENTS,
        "selected_server_ids": selected_ids,
        "exposed_server_ids": exposed_ids,
        "same_event_not_selected_server_ids": duplicate_event_ids,
        "model_calls": 0,
        "c0_outputs_generated": 0,
    }


def _display_content(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "").strip()
    if content and not content.startswith("<?xml"):
        return content
    raw = str(message.get("rawContent") or "")
    parts = []
    for tag in ("title", "des"):
        match = re.search(rf"<{tag}>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>", raw, re.S)
        if match and match.group(1).strip():
            parts.append(html.unescape(match.group(1).strip()))
    if parts:
        return " / ".join(parts)
    kind = int(message.get("localType") or 0)
    return {3: "[图片]", 34: "[语音]", 43: "[视频]", 49: "[引用或分享消息]"}.get(kind, "[非文本消息]")


def _format_ts(timestamp: int | float) -> str:
    return datetime.fromtimestamp(timestamp, TZ).strftime("%Y-%m-%d %H:%M:%S")


def render_report(
    candidates: list[dict[str, Any]],
    pending_candidates: list[dict[str, Any]] | None = None,
) -> str:
    qualification = assess_qualification(candidates)
    pending_cards = []
    for index, case in enumerate(pending_candidates or [], 1):
        messages = []
        for item in case["event_context"]:
            side = " self" if item["is_self"] else ""
            messages.append(
                f'<div class="msg{side}"><div class="who">{html.escape(item["sender"])} · '
                f'{html.escape(_format_ts(item["timestamp"]))}</div>'
                f'<div>{html.escape(item["content"])}</div></div>'
            )
        pending_cards.append(
            f'''<section class="case pending" id="pending-{index}">
              <div class="case-head"><b>待确认候选 {index}</b><span class="status">身份尚未确认，不计分</span></div>
              <div class="evidence human"><div class="label">本人账号中出现的候选气泡</div>
                <div class="time">{html.escape(case["observed_at"])} · serverId {html.escape(case["server_id"])}</div>
                <div class="who">{html.escape(case["sender"])}</div>
                <div class="reply">{html.escape(case["content"])}</div>
                <div class="note">{html.escape(case["account_evidence"])}</div></div>
              <details><summary>查看完整事件上下文（{len(case["event_context"])} 条）</summary>
                <div class="chat">{''.join(messages)}</div></details>
            </section>'''
        )
    cards = []
    public_candidates = [
        case
        for case in candidates
        if case["wechat_message"]["server_id"] in KNOWN_EXPOSED_SERVER_IDS
    ]
    for index, case in enumerate(public_candidates, 1):
        bubble = case["wechat_message"]
        bot = case["bot_send"]
        messages = []
        for item in case["event_context"]:
            side = " self" if item["is_self"] else ""
            messages.append(
                f'<div class="msg{side}"><div class="who">{html.escape(item["sender"])} · '
                f'{html.escape(_format_ts(item["timestamp"]))}</div>'
                f'<div>{html.escape(item["content"])}</div></div>'
            )
        production = []
        for item in (bot or {}).get("production_context") or []:
            text = str(item.get("text") or item.get("image_desc") or "[非文本消息]")
            production.append(
                f'<div><b>{html.escape(str(item.get("sender") or "未知"))}</b>：{html.escape(text)}</div>'
            )
        if bot:
            bot_evidence = f'''<div class="evidence bot"><div class="label">最近的 Bot 成功发送日志</div>
                <div class="time">{html.escape(bot["observed_at"])} · tick {html.escape(bot["tick_id"])} · 时间差 {bot["delta_seconds"]:+g} 秒</div>
                <div class="reply">{html.escape(bot["reply_text"])}</div>
                <div class="note">send.success=true；不等于已证明对应上方微信气泡</div></div>'''
        else:
            bot_evidence = '<div class="evidence bot"><div class="label">Bot 发送日志</div><div class="note">该已确认真人气泡附近没有成功发送日志；身份仅来自用户按 serverId 的明确确认。</div></div>'
        confirmed = case["identity_status"] == USER_CONFIRMATION["status"]
        cards.append(
            f'''<section class="case" id="case-{index}">
              <div class="case-head"><b>#{index}</b><span class="status {'confirmed' if confirmed else ''}">{html.escape(case["identity_status"])}</span></div>
              <div class="qualification">{html.escape(case["qualification_status"])}</div>
              <div class="evidence human"><div class="label">微信中实际出现的本人气泡</div>
                <div class="time">{html.escape(bubble["observed_at"])} · serverId {html.escape(bubble["server_id"])}</div>
                <div class="reply">{html.escape(bubble["content"])}</div>
                <div class="note">{html.escape(bubble["account_evidence"])}</div></div>
              {bot_evidence}
              <details><summary>查看完整微信事件上下文（{len(case["event_context"])} 条）</summary>
                <div class="chat">{''.join(messages)}</div></details>
              <details><summary>查看 Bot 当时看到的生产上下文与发送链</summary>
                <div class="trace">{''.join(production) or '<div>没有对应的 Bot 发送链</div>'}
                <hr><div><b>决策理由：</b>{html.escape((bot or {}).get("decision_reason") or "未记录")}</div>
                <div><b>发送预览：</b>{html.escape((bot or {}).get("send_preview") or "未记录")}</div>
                <div><b>send.success：</b>{'true' if bot else '无对应记录'}</div></div></details>
            </section>'''
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>柚子群身份人工确认</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--yellow:#d29922;--blue:#58a6ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}
.wrap{{max-width:760px;margin:auto;padding:16px}}h1{{font-size:22px;margin:0 0 8px}}.intro{{color:var(--muted);margin:0 0 16px}}
.summary,.case{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}}
.summary b{{font-size:22px;color:var(--yellow)}}.case-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.status{{color:#fff;background:#7d4e00;border-radius:999px;padding:3px 9px;font-size:12px}}.status.confirmed{{background:#1f6f3d}}.qualification{{color:var(--muted);font-size:13px;margin-bottom:8px}}.evidence{{border-left:4px solid;padding:10px 12px;margin:9px 0;background:#0d1117;border-radius:7px}}
.human{{border-color:var(--blue)}}.bot{{border-color:var(--yellow)}}.label{{font-weight:700}}.time,.note,.who{{color:var(--muted);font-size:12px}}.reply{{font-size:17px;margin:5px 0;white-space:pre-wrap}}
details{{border-top:1px solid var(--line);padding-top:10px;margin-top:10px}}summary{{cursor:pointer;color:var(--blue)}}.chat,.trace{{margin-top:10px}}
.msg{{background:#22272e;border-radius:9px;padding:8px 10px;margin:7px 15% 7px 0;white-space:pre-wrap}}.msg.self{{margin-left:15%;margin-right:0;background:#17385d}}.trace div{{margin:5px 0}}hr{{border:0;border-top:1px solid var(--line);margin:10px 0}}
@media(max-width:600px){{.wrap{{padding:10px}}.case{{padding:12px}}.msg{{margin-right:7%}}.msg.self{{margin-left:7%}}.reply{{font-size:16px}}}}
</style></head><body><main class="wrap"><h1>柚子群2 · 身份人工确认</h1>
<p class="intro">用途仅是核实“本人账号消息”究竟是真人手发还是 Bot 代发。本页不运行 Judge，也没有生成 C0 对照。</p>
<div class="summary"><b>{qualification['confirmed_bubbles']}</b> 条已确认真人气泡 · {qualification['distinct_confirmed_events']} 个独立事件 · {qualification['qualified_independent_event_representatives']} 个含未暴露真人回复的事件<br><span class="note">身份规则已满足当前柚子群基础门槛。未公开内容继续保持隐藏；本页不形成 Judge 成绩。</span></div>
<h2>待你确认的全新事件</h2>
<p class="intro">下面只展示候选证据。请按 serverId 确认是否为你真人手发；未确认前不会进入题集。</p>
{''.join(pending_cards) if pending_cards else '<div class="summary">当前没有待确认候选。</div>'}
<h2>此前已公开确认的气泡</h2>
<p class="intro">只展示此前已经公开的 14 条；其余真人原文不在网页展开，避免污染隐藏复验。</p>
{''.join(cards) if cards else '<div class="summary">没有找到符合条件的候选。</div>'}</main></body></html>'''


def write_outputs(
    candidates: list[dict[str, Any]],
    output_dir: Path = OUTPUT_DIR,
    pending_candidates: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    qualification = assess_qualification(candidates)
    manifest = {
        "schema_version": 2,
        "chat_name": TARGET_CHAT_NAME,
        "talker": TARGET_CHAT,
        "identity_policy": "in the target Youzi group, Wang Qian account bubbles are user-confirmed human-authored; the Bu Ai Shuo Hua identity is the Bot identity",
        "user_confirmation": USER_CONFIRMATION,
        "collision_seconds": COLLISION_SECONDS,
        "candidate_count": len(candidates),
        "pending_confirmation_count": len(pending_candidates or []),
        "pending_confirmation_cases": pending_candidates or [],
        "qualification": qualification,
        "stage2_manifest_updated": False,
        "judge_run": False,
        "c0_modified": False,
        "bot_started": False,
        "cases": candidates,
    }
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.html"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(candidates, pending_candidates), encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    os.chmod(report_path, 0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    messages = load_messages()
    members = load_members()
    candidates = build_candidates(messages, load_successful_sends(), members)
    pending_candidates = build_pending_confirmation_candidates(messages, members)
    write_outputs(candidates, args.output_dir, pending_candidates)
    print(json.dumps({"candidate_count": len(candidates), "pending_confirmation_count": len(pending_candidates), "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
