#!/usr/bin/env python3
"""Replay human participation cases without constructing or starting WeChat Bot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PRIVATE_MODE = 0o600
REPLAY_ENV = {
    "ENABLE_SELF_REFINE": "0",
    "ENABLE_FACT_CHECK": "1",
    "ENABLE_PERSONA_FEW_SHOTS": "1",
    "PERSONA_FEW_SHOT_COUNT": "3",
}


def _load_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


def _jsonl_row(path: Path, source_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") == source_id:
            return row
    raise ValueError(f"missing {source_id} in {path}")


def _message_type(raw_type: str, content: str) -> str:
    if raw_type == "图片消息":
        return "image"
    if raw_type == "动画表情":
        return "sticker"
    if raw_type == "视频消息":
        return "video"
    if "[链接]" in content or raw_type in {"小程序", "链接消息"}:
        return "link_card"
    return "text"


def _display_message(item: dict[str, Any]) -> dict[str, Any]:
    content = str(item.get("content") or "").lstrip("\n")
    return {
        "time": item.get("formattedTime"),
        "sender": item.get("senderDisplayName") or "未知",
        "is_self": bool(item.get("isSend")),
        "type": item.get("type") or "未知",
        "text": content or f"[{item.get('type') or '媒体消息'}]",
    }


def _chat_message(item: dict[str, Any], chat_name: str):
    from src.models.base import ChatMessage, SenderType

    content = str(item.get("content") or "").lstrip("\n")
    return ChatMessage(
        text=content,
        sender=str(item.get("senderDisplayName") or "未知"),
        sender_type=SenderType.SELF if item.get("isSend") else SenderType.OTHER,
        chat_name=chat_name,
        timestamp=str(item.get("formattedTime") or ""),
        message_type=_message_type(str(item.get("type") or ""), content),
        local_id=item.get("localId"),
        create_time=item.get("createTime"),
        raw_type=item.get("localType"),
        sender_wxid=item.get("senderUsername"),
    )


def _build_jsonl_case(spec: dict[str, Any]) -> dict[str, Any]:
    from src.models.base import ChatMessage, SenderType

    source_path = PROJECT_ROOT / spec["source"]
    row = _jsonl_row(source_path, spec["source_id"])
    raw_timestamp = row.get("timestamp")
    if raw_timestamp:
        evaluation_time = datetime.fromtimestamp(int(raw_timestamp))
        time_basis = "source_timestamp"
    else:
        evaluation_time = datetime(2026, 3, 31, 12, 0, 0)
        time_basis = "fixed_pre_cutoff_time_source_snippet_has_no_timestamp"
    context = [str(text) for text in row.get("context", [])]
    start = evaluation_time - timedelta(seconds=max(0, len(context) - 1))
    messages = [
        ChatMessage(
            text=text,
            sender="群友（原始摘录未保留姓名）",
            sender_type=SenderType.OTHER,
            chat_name=str(row.get("chat_id") or "历史真人群聊摘录"),
            timestamp=(start + timedelta(seconds=index)).isoformat(sep=" "),
        )
        for index, text in enumerate(context)
    ]
    return {
        "case_id": spec["case_id"],
        "expected_action": "reply",
        "is_group": row.get("relationship") == "group",
        "evaluation_time": evaluation_time,
        "all_messages": messages,
        "unreplied": list(messages),
        "human_evidence": {
            "decision": "reply",
            "context_original": context,
            "human_reply_original": list(row.get("reply", [])),
            "source": spec["source"],
            "source_id": spec["source_id"],
            "source_provenance": row.get("source_provenance"),
            "time_basis": time_basis,
            "limitation": "Curated source preserves the necessary message text but not original sender names.",
        },
    }


def _build_window_case(spec: dict[str, Any]) -> dict[str, Any]:
    source_path = PROJECT_ROOT / spec["source"]
    packet = _json(source_path)
    selected = [
        item
        for item in packet.get("messages", [])
        if spec["start"] <= str(item.get("formattedTime") or "") <= spec["end"]
    ]
    if not selected:
        raise ValueError(f"empty historical window: {spec['case_id']}")
    after_unreplied = [
        item
        for item in selected
        if str(item.get("formattedTime") or "") >= spec["unreplied_start"]
    ]
    self_after_target = [item for item in after_unreplied if item.get("isSend")]
    if self_after_target:
        raise ValueError(f"human reply exists in silence window: {spec['case_id']}")
    session = packet.get("session") or {}
    chat_name = str(session.get("displayName") or session.get("name") or source_path.stem)
    return {
        "case_id": spec["case_id"],
        "expected_action": "silence",
        "is_group": True,
        "evaluation_time": datetime.strptime(spec["end"], "%Y-%m-%d %H:%M:%S"),
        "all_messages": [_chat_message(item, chat_name) for item in selected],
        "unreplied": [_chat_message(item, chat_name) for item in after_unreplied],
        "human_evidence": {
            "decision": "silence",
            "source": spec["source"],
            "window_start": spec["start"],
            "unreplied_start": spec["unreplied_start"],
            "window_end": spec["end"],
            "human_messages_in_target_window": 0,
            "context_original": [_display_message(item) for item in selected],
        },
    }


def build_cases(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(manifest_path)
    cases = []
    for spec in manifest.get("cases", []):
        if spec.get("source_kind") == "jsonl":
            case = _build_jsonl_case(spec)
        elif spec.get("source_kind") == "weflow_window":
            case = _build_window_case(spec)
        else:
            raise ValueError(f"unsupported source kind: {spec.get('source_kind')}")
        if case["expected_action"] != spec.get("expected_action"):
            raise ValueError(f"expected action mismatch: {spec['case_id']}")
        cases.append(case)
    if len(cases) != 12:
        raise ValueError(f"expected 12 cases, got {len(cases)}")
    decisions = [case["expected_action"] for case in cases]
    if decisions.count("reply") != 6 or decisions.count("silence") != 6:
        raise ValueError("case set must contain 6 replies and 6 silences")
    return manifest, cases


def _generator_class(source: str, version: str):
    module_name = f"src.reply._participation_eval_{version}"
    module = types.ModuleType(module_name)
    module.__file__ = str(PROJECT_ROOT / "src/reply/generator.py")
    module.__package__ = "src.reply"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.ReplyGenerator


def _run_one(generator_class, case: dict[str, Any]) -> dict[str, Any]:
    from src.tools.tool_registry import ToolRegistry
    from src.utils.qwen_client import QwenClient

    started = time.monotonic()
    result: dict[str, Any] = {"status": "failed", "action": None, "replies": []}
    try:
        client = QwenClient()
        generator = generator_class(
            llm_client=client,
            memory_engine=None,
            tool_registry=ToolRegistry(),
            judge_worker=None,
        )
        replies = generator.generate(
            case["unreplied"],
            case["all_messages"],
            is_group=case["is_group"],
            tick_id=0,
            evaluation_time=case["evaluation_time"],
        )
        generation_failed = bool(generator.last_generation_failed)
        result.update(
            {
                "status": "completed",
                "action": (
                    "generation_failed"
                    if generation_failed
                    else ("reply" if replies else "silence")
                ),
                "replies": replies,
                "generation_failed": generation_failed,
                "model": client.model,
                "active_llm": generator.last_active_llm,
                "loaded_skills": generator.last_loaded_skills,
                "llm_call_count": len(generator.last_llm_calls),
                "tool_call_count": len(generator.last_tool_calls),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="离线回放真人参与判断案例")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/human_participation_cases.json",
    )
    parser.add_argument(
        "--p1-report",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_system_p1_participation_run1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/human_participation_ab_replay.json",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    _load_env()
    manifest, cases = build_cases(args.manifest)

    c0_source = (PROJECT_ROOT / "src/reply/generator.py").read_text(encoding="utf-8")
    c0_sha = _sha256_text(c0_source)
    if c0_sha != manifest["c0_generator_sha256"]:
        raise ValueError("current generator no longer matches protected C0")
    p1_packet = _json(args.p1_report)
    p1_source = p1_packet.get("source_snapshot", {}).get("src/reply/generator.py")
    if not isinstance(p1_source, str) or not p1_source:
        raise ValueError("P1 generator source snapshot is missing")
    p1_sha = _sha256_text(p1_source)

    for key, value in REPLAY_ENV.items():
        os.environ[key] = value
    model = str(p1_packet.get("system", {}).get("model") or "deepseek-v4-flash")
    os.environ["LLM_MODEL"] = model

    validation = {
        "case_count": len(cases),
        "reply_cases": sum(case["expected_action"] == "reply" for case in cases),
        "silence_cases": sum(case["expected_action"] == "silence" for case in cases),
        "c0_generator_sha256": c0_sha,
        "p1_generator_sha256": p1_sha,
        "model": model,
        "config": REPLAY_ENV,
        "bot_constructed": False,
        "action_tools_registered": False,
        "memory_enabled": False,
        "judge_worker_enabled": False,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    classes = {
        "C0": _generator_class(c0_source, "c0"),
        "P1": _generator_class(p1_source, "p1"),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "human_participation_ab_replay",
        "scope": "participation decision only; reply wording is not judged",
        "status": "running",
        "validation": validation,
        "cases": [
            {
                "case_id": case["case_id"],
                "expected_action": case["expected_action"],
                "human_evidence": case["human_evidence"],
                "runs": {"C0": [], "P1": []},
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {item["case_id"]: item for item in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for version in ("C0", "P1"):
            for case in cases:
                result = _run_one(classes[version], case)
                result["run"] = run_number
                report_by_id[case["case_id"]]["runs"][version].append(result)
                _write_private(args.output, report)
    report["status"] = "completed"
    report["completed_runs"] = args.runs * len(cases) * len(classes)
    report["expected_runs"] = args.runs * len(cases) * len(classes)
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
