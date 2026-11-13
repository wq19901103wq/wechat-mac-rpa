#!/usr/bin/env python3
"""Run a resumable C0-vs-verified-human baseline with frozen Codex Judge v6."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_human_ai_judge import (
    DEFAULT_VOLCENGINE_CONFIG,
    PROJECT_ROOT,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _export_payloads,
    _load_env,
    _parse_api_judge_response,
    _run_c0,
    _write_private,
    detect_degenerate_case,
    extract_judge_chat_context,
    load_volcengine_credentials,
)
from scripts.run_human_ai_judge_benchmark_v2 import build_messages, load_profile
from scripts.run_kimi_human_ai_judge_v2 import _run_codex
from scripts.replay_same_chat_style_ab import RelevantSameChatRetriever


PRIVATE_DIR = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/reports/private/clean_human_c0_baseline_v1"
PROFILE = PRIVATE_DIR / "judge_profile_v2_scenario_few_shot_candidate_v6.json"
REFERENCE = PRIVATE_DIR / "judge_reference_scenario_few_shot_candidate_v6.json"
EXPORTS = PROJECT_ROOT / "data/exports/b"
SPLITS = ("dev", "validation", "final")
MEDIA_MARKERS = ("[图片]", "[动画表情]", "[视频]", "<img", "refermsg", "[文件]")


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or "")


def _stable_order(row: dict[str, Any]) -> str:
    return hashlib.sha256(("clean-human-c0-v1:" + _case_id(row)).encode()).hexdigest()


def _load_verified_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in SPLITS:
        payload = json.loads((PRIVATE_DIR / f"{split}_replay.json").read_text(encoding="utf-8"))
        for row in payload.get("cases", []):
            case_id = _case_id(row)
            if not case_id or case_id in seen:
                continue
            seen.add(case_id)
            rows.append({**row, "historical_split": split})
    return rows


def _build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted((row for row in rows if row.get("relationship") == "group"), key=_stable_order)
    privates = sorted((row for row in rows if row.get("relationship") != "group"), key=_stable_order)
    development = groups[:10] + privates[:10]
    reserved = groups[10:16] + privates[10:34]
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "180 previously exposed but verified-human Judge cases",
        "selection_rule": "sha256(clean-human-c0-v1:case_id), stratified before C0 generation",
        "development_case_ids": [_case_id(row) for row in development],
        "reserved_validation_case_ids": [_case_id(row) for row in reserved],
        "counts": {
            "all": len(rows),
            "development": len(development),
            "development_group": sum(row.get("relationship") == "group" for row in development),
            "development_private": sum(row.get("relationship") != "group" for row in development),
            "reserved_validation": len(reserved),
            "reserved_group": sum(row.get("relationship") == "group" for row in reserved),
            "reserved_private": sum(row.get("relationship") != "group" for row in reserved),
        },
    }


def _candidate(row: dict[str, Any]) -> Candidate:
    return Candidate(
        relation=str(row.get("relationship") or "private"),
        chat_id=str(row["source_chat_id"]),
        context=list(row.get("context_original") or []),
        replies=list(row.get("human_reply_original") or []),
        score=0.0,
        timestamp=int(row["timestamp"]),
        source_provenance="verified_human_benchmark",
    )


def _has_media_tail(row: dict[str, Any]) -> bool:
    tail = "\n".join(str(value) for value in (row.get("context_original") or [])[-3:]).lower()
    return any(marker.lower() in tail for marker in MEDIA_MARKERS)


def _blind_case(replay: dict[str, Any]) -> tuple[dict[str, Any], str]:
    human_is = "A" if int(hashlib.sha256(replay["case_id"].encode()).hexdigest(), 16) % 2 == 0 else "B"
    ai_is = "B" if human_is == "A" else "A"
    trace = replay.get("c0_trace", {}).get("generation_trace", {})
    context = extract_judge_chat_context(str(trace.get("user_prompt") or ""))
    if not context:
        context = list(replay.get("context_original") or [])
    options = {
        human_is: replay["human_reply_original"],
        ai_is: replay["ai_reply_original"],
    }
    return ({
        "case_id": replay["case_id"],
        "relationship": replay["relationship"],
        "context_original": context,
        "option_A": options["A"],
        "option_B": options["B"],
    }, human_is)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


def _render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    cards = []
    for row in payload.get("cases", []):
        status = row["status"]
        judge = row.get("judge") or {}
        confidence = judge.get("confidence")
        wrong_class = " wrong" if judge and not judge.get("correct") else ""
        context = row.get("blind_case", {}).get("context_original") or row.get("source_context") or []
        cards.append(f'''<article class="case{wrong_class}">
          <div class="meta"><b>{html.escape(row['case_id'])}</b> · {html.escape(row['relationship'])} · {html.escape(status)}</div>
          <details open><summary>完整上下文</summary><pre>{html.escape(chr(10).join(context))}</pre></details>
          <div class="grid"><section><h3>真人原回复</h3><pre>{html.escape(chr(10).join(row.get('human_reply') or []))}</pre></section>
          <section><h3>当前 C0 回复</h3><pre>{html.escape(chr(10).join(row.get('c0_reply') or []))}</pre></section></div>
          <p>Judge：{html.escape(str(judge.get('human_option') or '—'))}；正确={html.escape(str(judge.get('correct') if judge else '—'))}；置信度={html.escape(str(confidence if confidence is not None else '—'))}</p>
          <p>{html.escape(str(judge.get('reason') or ''))}</p>
        </article>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>纯真人上下文 C0 基线</title><style>
    :root{{color-scheme:dark}}body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0b1020;color:#e7ecf5;margin:0;padding:14px}}main{{max-width:980px;margin:auto}}
    .summary,.case{{background:#151c31;border:1px solid #2a3554;border-radius:14px;padding:14px;margin:12px 0}}.case.wrong{{border-color:#b85c5c}}.meta{{color:#9fb0d0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1427;padding:10px;border-radius:9px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}h3{{font-size:14px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
    </style></head><body><main><h1>纯真人上下文 C0 基线</h1><div class="summary"><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></div>{''.join(cards)}</main></body></html>'''


def run(
    output_dir: Path,
    max_cases: int | None,
    generation_provider: str,
    generation_timeout_seconds: float,
    case_set: str,
    persona_few_shot_path: Path | None,
    retrieval_policy: str,
    persona_style_rule: str | None,
    force_reply: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.json"
    rows = _load_verified_rows()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _build_manifest(rows)
        _write_private(manifest_path, manifest)
    by_id = {_case_id(row): row for row in rows}
    selected_key = "development_case_ids" if case_set == "development" else "reserved_validation_case_ids"
    selected_ids = manifest[selected_key][:max_cases] if max_cases else manifest[selected_key]

    checkpoint_path = output_dir / "generation_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {"entries": {}}
    entries = checkpoint.setdefault("entries", {})
    previous_policy = checkpoint.get("retrieval_policy")
    if entries and previous_policy not in {None, retrieval_policy}:
        raise ValueError(f"output already contains retrieval_policy={previous_policy!r}")
    _load_env()
    if persona_style_rule or force_reply:
        source_persona = (PROJECT_ROOT / "data/persona.md").read_text(encoding="utf-8")
        if force_reply:
            start = source_persona.index("  <reply_decision>")
            end = source_persona.index("  </reply_decision>", start) + len("  </reply_decision>")
            source_persona = (
                source_persona[:start]
                + "  <reply_decision>\n"
                  "    <rule>本轮为离线回复风格实验；只要输入可读，必须生成非空 replies，不评估是否应该回复。</rule>\n"
                  "  </reply_decision>"
                + source_persona[end:]
            )
            source_persona = source_persona.replace(
                "      <rule>replies 只完成 response_target；若无法避开 forbidden_materials，宁可输出空 replies。</rule>",
                "      <rule>replies 只完成 response_target；若材料受限，使用不引入受限材料的简短非空回复。</rule>",
            ).replace(
                "      <rule>不回复时输出空 replies。</rule>",
                "      <rule>本轮实验的 replies 必须非空。</rule>",
            )
        marker = "    <content>\n"
        if persona_style_rule and marker not in source_persona:
            raise ValueError("persona style content marker not found")
        experiment_persona = source_persona
        if persona_style_rule:
            experiment_persona = experiment_persona.replace(
                marker,
                marker + f"      <rule>{persona_style_rule}</rule>\n",
                1,
            )
        experiment_persona_path = output_dir / "persona_candidate.md"
        _write_private(experiment_persona_path, experiment_persona)
        os.environ["PERSONA_PATH"] = str(experiment_persona_path.resolve())
    if persona_few_shot_path is not None:
        os.environ["ENABLE_PERSONA_FEW_SHOTS"] = "1"
        os.environ["PERSONA_FEW_SHOT_PATH"] = str(persona_few_shot_path.resolve())
    from src.memory import MemoryEngine
    from src.utils.qwen_client import QwenClient
    if generation_provider == "volcengine":
        api_key, base_url = load_volcengine_credentials(DEFAULT_VOLCENGINE_CONFIG)
        client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    else:
        client = QwenClient()
    previous_provider = checkpoint.get("generation_provider")
    if previous_provider != generation_provider:
        for case_id in list(entries):
            if entries[case_id].get("status") in {"generation_failed", "generation_error"}:
                del entries[case_id]
    checkpoint["generation_provider"] = generation_provider
    checkpoint["generation_model"] = client.model
    checkpoint["generation_endpoint_host"] = "ark.cn-beijing.volces.com" if generation_provider == "volcengine" else "api.deepseek.com"
    checkpoint["case_set"] = case_set
    checkpoint["persona_few_shot_path"] = os.environ.get("PERSONA_FEW_SHOT_PATH")
    checkpoint["retrieval_policy"] = retrieval_policy
    checkpoint["persona_style_rule"] = persona_style_rule
    checkpoint["force_reply"] = force_reply
    checkpoint["retrieval_max_examples"] = 2 if retrieval_policy == "relevant-same-chat" else None
    checkpoint["retrieval_threshold"] = (
        {"minimum_bigram_overlap": 2, "minimum_dice": 0.06}
        if retrieval_policy == "relevant-same-chat" else None
    )
    _write_private(checkpoint_path, checkpoint)
    memory = MemoryEngine(llm_client=client)
    exports = _export_payloads(EXPORTS)
    experimental_retriever = (
        RelevantSameChatRetriever(persona_few_shot_path)
        if retrieval_policy == "relevant-same-chat" and persona_few_shot_path is not None
        else None
    )
    try:
        for case_id in selected_ids:
            if case_id in entries and not (
                entries[case_id].get("status") in {"generation_failed", "generation_error"}
                and float(entries[case_id].get("generation_timeout_seconds") or 20.0) < generation_timeout_seconds
            ):
                continue
            row = by_id[case_id]
            if _has_media_tail(row):
                entries[case_id] = {"status": "excluded_media_tail"}
                _write_private(checkpoint_path, checkpoint)
                continue
            candidate = _candidate(row)
            payload = exports.get(candidate.chat_id)
            if payload is None:
                entries[case_id] = {"status": "excluded_missing_export"}
                _write_private(checkpoint_path, checkpoint)
                continue
            try:
                generated = _run_c0(
                    candidate,
                    payload,
                    client,
                    memory,
                    generation_timeout_seconds=generation_timeout_seconds,
                    persona_few_shot_retriever=experimental_retriever,
                    persona_few_shot_count=2 if experimental_retriever is not None else None,
                    force_reply=force_reply,
                )
                if generated["generation_failed"]:
                    entries[case_id] = {
                        "status": "generation_failed",
                        "generation_timeout_seconds": generation_timeout_seconds,
                        "generation": generated,
                    }
                elif not generated["replies"]:
                    entries[case_id] = {"status": "silent", "generation": generated}
                else:
                    reasons = detect_degenerate_case(candidate.replies, generated)
                    if reasons:
                        entries[case_id] = {"status": "excluded_degenerate", "reasons": reasons, "generation": generated}
                    else:
                        replay = {
                            "case_id": case_id,
                            "relationship": candidate.relation,
                            "context_original": generated["context_original"],
                            "human_reply_original": candidate.replies,
                            "ai_reply_original": generated["replies"],
                            "c0_trace": {
                                "generation_trace": generated["trace"],
                                "input_metadata": generated["input_metadata"],
                                "duration_seconds": generated.get("duration_seconds"),
                            },
                        }
                        entries[case_id] = {"status": "scorable", "replay": replay}
            except Exception as exc:
                entries[case_id] = {"status": "generation_error", "error": f"{type(exc).__name__}: {exc}"}
            _write_private(checkpoint_path, checkpoint)
    finally:
        memory.shutdown()

    profile = load_profile(PROFILE)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    judge_path = output_dir / "judge_checkpoint.json"
    judge_checkpoint = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {"entries": {}}
    judge_entries = judge_checkpoint.setdefault("entries", {})
    for case_id in selected_ids:
        generated_entry = entries.get(case_id, {})
        if generated_entry.get("status") != "scorable" or (
            case_id in judge_entries and not judge_entries[case_id].get("error")
        ):
            continue
        blind, human_option = _blind_case(generated_entry["replay"])
        try:
            raw, audit, elapsed_ms, tool_calls = _run_codex(build_messages(blind, reference, profile))
            parsed = _parse_api_judge_response(raw)
            judge_entries[case_id] = {
                **parsed,
                "correct": parsed["human_option"] == human_option,
                "actual_human_option": human_option,
                "elapsed_ms": elapsed_ms,
                "tool_calls": tool_calls,
                "audit": audit,
                "blind_case": blind,
            }
        except Exception as exc:
            judge_entries[case_id] = {"error": f"{type(exc).__name__}: {exc}"}
        _write_private(judge_path, judge_checkpoint)

    cases = []
    for case_id in selected_ids:
        row = by_id[case_id]
        entry = entries.get(case_id, {"status": "not_run"})
        judge = judge_entries.get(case_id)
        replay = entry.get("replay") or {}
        generation = entry.get("generation") or {}
        generation_trace = generation.get("trace") or replay.get("c0_trace", {}).get("generation_trace") or {}
        few_shot_content = str(generation_trace.get("few_shot_content") or "")
        cases.append({
            "case_id": case_id,
            "relationship": row.get("relationship"),
            "status": entry.get("status"),
            "source_context": row.get("context_original") or [],
            "human_reply": row.get("human_reply_original") or [],
            "c0_reply": replay.get("ai_reply_original") or [],
            "few_shot_ids": replay.get("c0_trace", {}).get("generation_trace", {}).get("few_shot_ids") or [],
            "few_shot_rendered_chars": len(few_shot_content),
            "generation_duration_seconds": generation.get("duration_seconds") or replay.get("c0_trace", {}).get("duration_seconds"),
            "blind_case": (judge or {}).get("blind_case") or {},
            "judge": None if not judge or judge.get("error") else {key: judge.get(key) for key in ("human_option", "actual_human_option", "correct", "confidence", "reason")},
            "error": entry.get("error") or (judge or {}).get("error"),
        })
    judged = [row for row in cases if row.get("judge")]
    generation_durations = [
        float(row["generation_duration_seconds"])
        for row in cases
        if row.get("generation_duration_seconds") is not None
    ]
    few_shot_counts = [len(row["few_shot_ids"]) for row in cases if row["status"] == "scorable"]
    few_shot_chars = [row["few_shot_rendered_chars"] for row in cases if row["status"] == "scorable"]
    all_processed = len(cases) == len(selected_ids) and all(row["status"] != "not_run" for row in cases)
    all_scorable_judged = all(row["status"] != "scorable" or row.get("judge") for row in cases)
    summary = {
        "status": "complete" if all_processed and all_scorable_judged else "partial",
        "c0_model": client.model,
        "c0_generation_provider": generation_provider,
        "c0_endpoint_host": checkpoint["generation_endpoint_host"],
        "case_set": case_set,
        "persona_few_shot_path": checkpoint.get("persona_few_shot_path"),
        "retrieval_policy": retrieval_policy,
        "retrieval_max_examples": checkpoint.get("retrieval_max_examples"),
        "retrieval_threshold": checkpoint.get("retrieval_threshold"),
        "judge": "Codex Judge v5 (frozen)",
        "selected": len(selected_ids),
        "status_counts": dict(Counter(row["status"] for row in cases)),
        "scorable_judged": len(judged),
        "judge_correct_human_identifications": sum(bool(row["judge"]["correct"]) for row in judged),
        "judge_discrimination_accuracy": (sum(bool(row["judge"]["correct"]) for row in judged) / len(judged)) if judged else None,
        "high_confidence_correct_c0_identifications": sum(bool(row["judge"]["correct"]) and float(row["judge"].get("confidence") or 0) >= 0.8 for row in judged),
        "generation_rt_seconds": {
            "count": len(generation_durations),
            "p50": _percentile(generation_durations, 0.50),
            "p95": _percentile(generation_durations, 0.95),
        },
        "few_shot_selected_count_distribution": dict(Counter(few_shot_counts)),
        "few_shot_rendered_chars": {
            "p50": _percentile([float(value) for value in few_shot_chars], 0.50),
            "p95": _percentile([float(value) for value in few_shot_chars], 0.95),
        },
        "group_selected": sum(row["relationship"] == "group" for row in cases),
        "private_selected": sum(row["relationship"] != "group" for row in cases),
        "interpretation": "Judge is frozen and auxiliary; errors and reasons require human review before changing C0.",
    }
    result = {"schema_version": 1, "created_at": datetime.now().astimezone().isoformat(), "summary": summary, "manifest": manifest, "cases": cases}
    _write_private(output_dir / "results.json", result)
    index = output_dir / "index.html"
    index.write_text(_render(result), encoding="utf-8")
    index.chmod(0o600)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--generation-provider", choices=("official", "volcengine"), default="official")
    parser.add_argument("--generation-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--case-set", choices=("development", "reserved"), default="development")
    parser.add_argument("--persona-few-shot-path", type=Path)
    parser.add_argument("--retrieval-policy", choices=("default", "relevant-same-chat"), default="default")
    parser.add_argument("--persona-style-rule")
    parser.add_argument("--force-reply", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        args.max_cases,
        args.generation_provider,
        args.generation_timeout_seconds,
        args.case_set,
        args.persona_few_shot_path,
        args.retrieval_policy,
        args.persona_style_rule,
        args.force_reply,
    )


if __name__ == "__main__":
    main()
