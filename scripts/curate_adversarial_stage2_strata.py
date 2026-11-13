#!/usr/bin/env python3
"""Curate Stage 2 difficulty strata with one isolated API call per hidden case."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_adversarial_stage2_holdout import MANIFEST_PATH, PRIVATE_DIR
from scripts.evaluate_human_ai_judge import (
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _json_sha256,
    _response_for_audit,
    _write_private,
    load_volcengine_credentials,
)

LABELS_PATH = PRIVATE_DIR / "strata_labels.json"
AUDIT_PATH = PRIVATE_DIR / "strata_curation_audit.json"
INPUT_SNAPSHOT_NAME = "input_manifest.json"

STRATA = (
    "multi_speaker_target_resolution",
    "topic_switch_or_parallel_threads",
    "cross_message_reference",
    "fact_grounding",
    "humor_or_emotional_response",
    "real_world_commitment_boundary",
)

PROFILE = {
    "schema_version": 1,
    "name": "stage2_strata_curator_v1",
    "model": VOLCENGINE_DSV4F_MODEL,
    "temperature": 0,
    "definitions": {
        "multi_speaker_target_resolution": "上下文有至少两位其他参与者，且理解回复需要判断本人是在接谁、回应哪位或避免对象混淆；只有多人出现但对象毫无难度不算。",
        "topic_switch_or_parallel_threads": "上下文发生明显话题切换，或同时存在两个以上子话题，理解回复需要选对当前话题；单一话题自然推进不算。",
        "cross_message_reference": "回复承接的关键信息不只在紧邻上一条，而要跨过至少一条消息理解追问、指代、回扣或省略内容。",
        "fact_grounding": "判断或生成合适回复需要忠实使用上下文中的具体事实，或需要克制地承认未知以避免事实编造；普通闲聊中任何一句都含事实不自动算。",
        "humor_or_emotional_response": "真人回复的主要作用包含自然幽默、调侃、情绪共鸣、安慰、兴奋、吐槽或关系感，而非只传递中性信息。",
        "real_world_commitment_boundary": "上下文或回复涉及转账、红包、付款、购买、寄送、预约、代办、到场等现实行动承诺，或玩笑与真实承诺容易混淆；泛泛建议和纯事实讨论不算。",
    },
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _load_input_snapshot(manifest_path: Path, output_dir: Path) -> tuple[bytes, str]:
    source_bytes = manifest_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    snapshot_path = output_dir / INPUT_SNAPSHOT_NAME
    if snapshot_path.exists():
        snapshot_bytes = snapshot_path.read_bytes()
        if hashlib.sha256(snapshot_bytes).hexdigest() != source_sha256:
            raise ValueError("source manifest changed since this curation run was created")
        return snapshot_bytes, source_sha256
    _write_bytes_atomic(snapshot_path, source_bytes)
    return source_bytes, source_sha256


def _verify_input_unchanged(manifest_path: Path, expected_sha256: str) -> None:
    if _file_sha256(manifest_path) != expected_sha256:
        raise ValueError("source manifest changed during curation")


def _acquire_output_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".curation.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another curation process is writing {output_dir}") from None
    return handle


def build_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
        "labels": {name: "boolean" for name in STRATA},
        "evidence": {name: "简短中文依据；若为 false 说明缺少哪个必要条件" for name in STRATA},
    }
    system = (
        "你是独立的数据分层整理者，只标注当前一个真实聊天事件。"
        "你无法访问其他题目、答案、文件、工具或外部资料，也不要猜测参与者身份。"
        "严格按定义判断，不要为了凑配额放宽标准。每个标签可以交叉为 true。\n"
        "定义：\n" + json.dumps(PROFILE["definitions"], ensure_ascii=False, indent=2)
        + "\n只输出一个符合以下结构的 JSON 对象，不要输出 markdown：\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    packet = {
        "context": case["context_messages"],
        "human_reply": {"speaker": "self", "content": case["reply"]},
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]


def parse_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("curator response contains no JSON object")
        payload = json.loads(text[start : end + 1])
    labels = payload.get("labels")
    evidence = payload.get("evidence")
    if not isinstance(labels, dict) or not isinstance(evidence, dict):
        raise ValueError("curator response is missing labels or evidence")
    if set(labels) != set(STRATA) or set(evidence) != set(STRATA):
        raise ValueError("curator response has unexpected strata keys")
    if any(type(labels[name]) is not bool for name in STRATA):
        raise ValueError("all curator labels must be booleans")
    if any(not str(evidence[name]).strip() for name in STRATA):
        raise ValueError("all curator evidence fields must be non-empty")
    return {
        "labels": {name: labels[name] for name in STRATA},
        "evidence": {name: str(evidence[name]).strip() for name in STRATA},
    }


def validate_request_body(body: dict[str, Any], case: dict[str, Any]) -> None:
    if body.get("tools"):
        raise ValueError("curator isolation failed: tools were sent")
    if body.get("temperature") != 0:
        raise ValueError("curator isolation failed: temperature is not zero")
    user_texts: list[str] = []
    for message in body.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            user_texts.append(content)
        elif isinstance(content, list):
            user_texts.extend(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    if len(user_texts) != 1:
        raise ValueError("curator isolation failed: request must contain exactly one user text")
    try:
        sent_packet = json.loads(user_texts[0])
    except json.JSONDecodeError as exc:
        raise ValueError("curator request does not contain a valid single-case packet") from exc
    expected_packet = {
        "context": case["context_messages"],
        "human_reply": {"speaker": "self", "content": case["reply"]},
    }
    if sent_packet != expected_packet:
        raise ValueError("curator request does not contain the exact single-case packet")
    serialized = json.dumps(body, ensure_ascii=False)
    forbidden_values = (
        str(case.get("internal_id") or ""),
        str(case.get("chat_id") or ""),
        str(case.get("event_cluster") or ""),
        str(case.get("source_path") or ""),
    )
    if any(value and value in serialized for value in forbidden_values):
        raise ValueError("curator request leaked hidden identity metadata")


def _new_artifacts(manifest_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    invocation_id = f"stage2_curator_{secrets.token_hex(12)}"
    common = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "manifest_sha256": manifest_sha256,
        "profile_sha256": _json_sha256(PROFILE),
        "model": VOLCENGINE_DSV4F_MODEL,
    }
    labels = {**common, "status": "in_progress", "cases": []}
    audit = {
        **common,
        "privacy": "LOCAL PRIVATE: contains hidden real chat text; never publish or commit",
        "isolation": {
            "one_case_per_request": True,
            "fresh_client_per_case": True,
            "shared_conversation_state": False,
            "tools_available": False,
            "other_cases_available": False,
            "bot_or_judge_identity_available": False,
        },
        "profile": PROFILE,
        "calls": [],
    }
    return labels, audit


def _load_or_create(
    manifest_sha256: str,
    labels_path: Path = LABELS_PATH,
    audit_path: Path = AUDIT_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if labels_path.exists() != audit_path.exists():
        raise ValueError("resume requires both label and audit files")
    if not labels_path.exists():
        return _new_artifacts(manifest_sha256)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "manifest_sha256": manifest_sha256,
        "profile_sha256": _json_sha256(PROFILE),
        "model": VOLCENGINE_DSV4F_MODEL,
    }
    for field, value in expected.items():
        if labels.get(field) != value or audit.get(field) != value:
            raise ValueError(f"existing curation artifact has a different {field}")
    if labels.get("invocation_id") != audit.get("invocation_id"):
        raise ValueError("label and audit invocation ids do not match")
    return labels, audit


def aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {name: sum(bool(row["labels"][name]) for row in rows) for name in STRATA}


def _curate_case(api_key: str, base_url: str, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    internal_id = str(case["internal_id"])
    client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    raw = ""
    parsed = None
    error = None
    try:
        raw = client.chat(
            messages=build_messages(case),
            temperature=0,
            max_tokens=1200,
            timeout=180,
            raise_on_error=True,
            max_retries=1,
        )
        validate_request_body(client.last_request_body, case)
        parsed = parse_response(raw)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    call = {
        "internal_id": internal_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "request_sha256": _json_sha256(client.last_request_body),
        "request_body": client.last_request_body,
        "response": _response_for_audit(client.last_response_payload),
        "status_code": client.last_status_code,
        "resolved_model": client.resolved_model,
        "tools_sent": bool(client.last_request_body.get("tools")),
        "raw_response": raw,
        "error": error,
    }
    return call, parsed


def run(
    config_path: Path,
    max_cases: int | None = None,
    workers: int = 1,
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Path = PRIVATE_DIR,
) -> dict[str, Any]:
    labels_path = output_dir / "strata_labels.json"
    audit_path = output_dir / "strata_curation_audit.json"
    lock_handle = _acquire_output_lock(output_dir)
    try:
        manifest_bytes, manifest_sha256 = _load_input_snapshot(manifest_path, output_dir)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        cases = manifest.get("cases") or []
        if not cases or any(not row.get("context_messages") for row in cases):
            raise ValueError("manifest is not ready for isolated strata curation")
        labels, audit = _load_or_create(manifest_sha256, labels_path, audit_path)
        completed = {str(row["internal_id"]) for row in labels["cases"]}
        api_key, base_url = load_volcengine_credentials(config_path)
        pending = [case for case in cases if str(case["internal_id"]) not in completed]
        if max_cases is not None:
            pending = pending[:max_cases]
        attempted = len(pending)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_curate_case, api_key, base_url, case): case for case in pending
            }
            for future in as_completed(futures):
                case = futures[future]
                internal_id = str(case["internal_id"])
                call, parsed = future.result()
                audit["calls"].append(call)
                if parsed is not None:
                    labels["cases"].append({"internal_id": internal_id, **parsed})
                    completed.add(internal_id)
                labels["aggregate"] = aggregate(labels["cases"])
                labels["status"] = (
                    "completed" if len(labels["cases"]) == len(cases) else "in_progress"
                )
                _verify_input_unchanged(manifest_path, manifest_sha256)
                _write_private(labels_path, labels)
                _write_private(audit_path, audit)
        return {
            "status": labels["status"],
            "completed": len(labels["cases"]),
            "total": len(cases),
            "aggregate": labels.get("aggregate", {}),
            "attempted_this_run": attempted,
            "errors": sum(bool(row.get("error")) for row in audit["calls"]),
            "model": VOLCENGINE_DSV4F_MODEL,
        }
    finally:
        lock_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE_DIR)
    args = parser.parse_args()
    print(json.dumps(run(
        args.config,
        args.max_cases,
        args.workers,
        args.manifest,
        args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
