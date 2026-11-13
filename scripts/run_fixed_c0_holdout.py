#!/usr/bin/env python3
"""Run the formal baseline and cumulative C0 on the frozen 240 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    _load_env,
    _parse_api_judge_response,
    _write_private,
)
from scripts.run_human_ai_judge_benchmark_v2 import build_messages, load_profile  # noqa: E402
from scripts.run_kimi_human_ai_judge_v2 import _run_codex  # noqa: E402
from scripts.run_strict_historical_judge_acceptance import (  # noqa: E402
    HistoricalPromptReplyGenerator,
    _blind_case,
    _generation_client,
    _group_key,
    generate_case,
)
from src.memory.engine import MemoryEngine  # noqa: E402


MANIFEST = ROOT / "data/reports/private/fixed_c0_holdout_final/manifest.json"
OUTPUT_DIR = ROOT / "data/reports/private/fixed_c0_formal_ab_v2"
PROFILE = ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_profile_v2_scenario_few_shot_candidate_v6.json"
REFERENCE = ROOT / "data/reports/private/human_ai_judge_benchmark_v2/judge_reference_scenario_few_shot_candidate_v6.json"
FEW_SHOTS = ROOT / "data/few_shot_verified_humor/persona_examples.jsonl"
PERSONA = ROOT / "data/persona.md"
BASELINE_RULE = (
    "<rule>history 或工具结果中的单次经历只能证明那一次发生过，不能据此断言当前普遍允许、必须满足某条件、"
    "存在固定流程或保证结果。回答“能不能、是否允许、怎么操作”时，只有当前直接规则或政策证据才能肯定；"
    "只有历史个例时不得直接回答“可以”或把个例改写成操作规则，应保留不确定性或建议向负责方确认。</rule>"
)
MEDIA_TAIL_MARKERS = ("<img", "cdnmidimgurl", "[图片]", "[表情]", "[视频]", "<type>3</type>")
FORMAL_MAIN_REACT_MAX_SECONDS = 90.0
FORMAL_FALLBACK_RESERVE_SECONDS = 60.0


class AdoptedBaselineHistoricalGenerator(HistoricalPromptReplyGenerator):
    """Current production generator with the unadopted candidate-v2 rule removed."""

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        if BASELINE_RULE not in prompt:
            raise RuntimeError("candidate-v2 rule marker missing; cannot prove adopted baseline prompt")
        prompt = prompt.replace(BASELINE_RULE, "", 1)
        if BASELINE_RULE in prompt:
            raise RuntimeError("candidate-v2 rule occurs more than once")
        return prompt


class ForcedReplyAdoptedBaselineHistoricalGenerator(AdoptedBaselineHistoricalGenerator):
    """Offline-only variant that removes reply-suppression rules from the request."""

    _REPLACEMENTS = {
        "后续图片、表情、链接或闲聊不得把它盖掉；若明确指向别人则不参与。":
            "后续图片、表情、链接或闲聊不得把它盖掉；若明确指向别人，改为围绕当前可读话题简短回应。",
        "<rule>群聊中明确 @ 其他人、称呼其他人或延续他人之间的问答，且没有同时指向你时，默认 replies 为空；普通陈述也不必句句参与。</rule>":
            "<rule>本轮是离线风格实验；群聊没有指向你时，也围绕最近的可读话题生成一条简短非空回复。</rule>",
        "<rule>连续图片、表情、视频或链接若没有点名你、向你提问或邀请你参与，默认 replies 为空；不要逐条描述，也不要为了接梗强行参与。</rule>":
            "<rule>连续媒体消息本身不可读时不要猜图；若同时存在可读文字，围绕文字生成一条简短非空回复。</rule>",
        "<rule>本轮主要语义是要求停止交流时，replies 必须为空。</rule>":
            "<rule>本轮主要语义是要求停止交流时，只生成一条简短结束回复，不继续追击。</rule>",
        "<rule>即使 unread 再次引用这些内容，也不得 callback；宁可字面回应、换全新角度或不回复。</rule>":
            "<rule>即使 unread 再次引用这些内容，也不得 callback；应字面回应或换全新角度。</rule>",
        "<rule>若 unread 只是复述或抱怨已用素材且没有独立新内容，replies 必须为空；明确提问、请求解释或继续对话不适用此条。</rule>":
            "<rule>若 unread 只是复述或抱怨已用素材且没有独立新内容，做一句简短字面回应；不得复用旧素材。</rule>",
    }

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        for old, new in self._REPLACEMENTS.items():
            prompt = prompt.replace(old, new)
        return prompt

    def _react_generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        deadline: float,
        chat_name: str = "",
    ) -> tuple[list[str], list[dict[str, Any]]]:
        now = time.time()
        remaining = max(0.0, deadline - now)
        reserve = min(
            FORMAL_FALLBACK_RESERVE_SECONDS,
            max(0.0, remaining - 1.0),
        )
        main_budget = min(
            FORMAL_MAIN_REACT_MAX_SECONDS,
            max(1.0, remaining - reserve),
        )
        main_deadline = min(deadline, now + main_budget)
        replies, final_messages = super()._react_generate(
            messages, tools, main_deadline, chat_name
        )
        if replies or deadline - time.time() < 1.0:
            return replies, final_messages

        # Do not resend the potentially huge tool trace: it can consume the
        # entire reserved fallback window before producing any reply.
        fallback_messages = list(messages)
        fallback_messages.append({
            "role": "user",
            "content": (
                "离线评测要求本轮必须回复。不要再调用工具；只依据现有上下文，"
                "输出一条简短、非空且不编造事实的 JSON：{\"replies\":[\"...\"]}"
            ),
        })
        try:
            raw = self.llm_client.chat(
                messages=fallback_messages,
                tools=None,
                max_tokens=512,
                timeout=max(1.0, deadline - time.time()),
                response_format={"type": "json_object"},
                max_retries=0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            fallback_replies = self._parse_replies(text)
            self.last_generation_trace.append({
                "type": "forced_nonempty_fallback",
                "timestamp": time.time(),
                "content": text or "",
            })
            if fallback_replies:
                self.last_generation_failed = False
                self.last_raw_response = text
                return fallback_replies, fallback_messages + [{"role": "assistant", "content": text}]
        except Exception as exc:
            self.last_generation_trace.append({
                "type": "forced_nonempty_fallback_error",
                "timestamp": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
            })
        return [], final_messages


class FormalBaselineGenerator(ForcedReplyAdoptedBaselineHistoricalGenerator):
    """Last formal baseline, before the two accumulated retrieval changes."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["enable_information_gap_response_move"] = False
        kwargs["prefer_latest_turn_form"] = False
        super().__init__(*args, **kwargs)


class CumulativeCandidateGenerator(ForcedReplyAdoptedBaselineHistoricalGenerator):
    """Candidate that reached the +4/80 formal-test gate."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["enable_information_gap_response_move"] = True
        kwargs["prefer_latest_turn_form"] = True
        super().__init__(*args, **kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_media_tail(source: dict[str, Any]) -> bool:
    context = source.get("context_messages") or source.get("context") or []
    tail = "\n".join(
        str(row if isinstance(row, str) else row.get("content") or "")
        for row in context[-3:]
    ).lower()
    return any(marker.lower() in tail for marker in MEDIA_TAIL_MARKERS)


def _load(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _experiment_persona(output_dir: Path, force_reply: bool, persona_style_rule: str | None) -> Path:
    source = PERSONA.read_text(encoding="utf-8")
    if force_reply:
        start = source.index("  <reply_decision>")
        end = source.index("  </reply_decision>", start) + len("  </reply_decision>")
        source = (
            source[:start]
            + "  <reply_decision>\n"
              "    <rule>本轮为离线回复风格实验；只要输入可读，必须生成非空 replies，不评估是否应该回复。</rule>\n"
              "  </reply_decision>"
            + source[end:]
        )
        source = source.replace(
            "      <rule>replies 只完成 response_target；若无法避开 forbidden_materials，宁可输出空 replies。</rule>",
            "      <rule>replies 只完成 response_target；若材料受限，使用不引入受限材料的简短非空回复。</rule>",
        ).replace(
            "      <rule>不回复时输出空 replies。</rule>",
            "      <rule>本轮实验的 replies 必须非空。</rule>",
        )
        replacements = {
            '<case type="normal_ending">本轮消息的主要语义功能是结束交流或要求停止时，即使措辞不友好，也输出空 replies，不追击。</case>':
                '<case type="normal_ending">本轮消息的主要语义功能是结束交流或要求停止时，只生成一句简短结束回复，不追击。</case>',
            "没有新鲜角度时短促回应或不回。": "没有新鲜角度时做短促字面回应。",
            "给态度、新信息、具体动作、必要追问，或选择不回。": "给态度、新信息、具体动作、必要追问，或做简短字面回应。",
            "若 unread 除追问、复述或抱怨 ineligible 素材外没有独立的新内容，输出空 replies。":
                "若 unread 除追问、复述或抱怨 ineligible 素材外没有独立的新内容，做一句简短字面回应。",
            "没有 eligible 新角度时，直接回应 unread 的字面内容或不回，不要从 history 或 memory 找梗。":
                "没有 eligible 新角度时，直接回应 unread 的字面内容，不要从 history 或 memory 找梗。",
            "若无法避开 forbidden_materials，宁可输出空 replies。":
                "若无法避开 forbidden_materials，改用不引入受限材料的简短非空回复。",
        }
        for old, new in replacements.items():
            source = source.replace(old, new)
    if persona_style_rule:
        marker = "    <content>\n"
        if marker not in source:
            raise RuntimeError("persona style content marker missing")
        source = source.replace(marker, marker + f"      <rule>{persona_style_rule}</rule>\n", 1)
    path = output_dir / "persona_experiment.md"
    _write_private(path, source)
    return path


def _locked_config(
    force_reply: bool, persona_style_rule: str | None, persona_few_shot_path: Path
) -> dict[str, Any]:
    return {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "generation_provider": "volcengine",
        "generation_model": "deepseek-v4-flash",
        "persona_few_shot_path": str(persona_few_shot_path.relative_to(ROOT)),
        "persona_few_shot_sha256": _sha256(persona_few_shot_path),
        "persona_few_shot_rows": sum(1 for line in persona_few_shot_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "group_persona_few_shot_path": str(persona_few_shot_path.relative_to(ROOT)),
        "generator_variants": {
            "baseline": {
                "information_gap_response_move": False,
                "prefer_latest_turn_form": False,
            },
            "candidate": {
                "information_gap_response_move": True,
                "prefer_latest_turn_form": True,
            },
        },
        "force_reply": force_reply,
        "persona_style_rule": persona_style_rule,
        "judge": "Codex Judge v5 (frozen)",
        "judge_profile": str(PROFILE.relative_to(ROOT)),
        "judge_profile_sha256": _sha256(PROFILE),
        "judge_reference": str(REFERENCE.relative_to(ROOT)),
        "judge_reference_sha256": _sha256(REFERENCE),
    }


def _ensure_config(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    prior = checkpoint.get("locked_config")
    if prior is not None and prior != config:
        raise RuntimeError("checkpoint configuration differs from the frozen 240-case run")
    checkpoint["locked_config"] = config


def _condition_result(
    generation: dict[str, Any] | None, judge: dict[str, Any] | None
) -> dict[str, Any]:
    generation = generation or {"status": "not_run"}
    generated = generation.get("case") or {}
    clean_judge = None
    if judge and not judge.get("error"):
        clean_judge = {
            key: judge.get(key)
            for key in ("human_option", "actual_human_option", "correct", "confidence", "reason", "elapsed_ms")
        }
    return {
        "status": generation.get("status"),
        "c0_reply": generated.get("c0_reply") or [],
        "generation": generated.get("generation") or {},
        "validity": generated.get("validity") or generation.get("validity") or {},
        "judge": clean_judge,
        "judge_error": (judge or {}).get("error"),
    }


def _entry_id(case_id: str, condition: str) -> str:
    if condition not in {"baseline", "candidate"}:
        raise ValueError(f"unknown formal A/B condition: {condition}")
    return f"{case_id}:{condition}"


def _case_result(
    source: dict[str, Any], generation_entries: dict[str, Any], judge_entries: dict[str, Any]
) -> dict[str, Any]:
    prefix = str(source["internal_id"])
    baseline_generation = generation_entries.get(_entry_id(prefix, "baseline")) or {}
    candidate_generation = generation_entries.get(_entry_id(prefix, "candidate")) or {}
    generated = baseline_generation.get("case") or candidate_generation.get("case") or {}
    return {
        "case_id": str(source["internal_id"]),
        "chat_id": str(source.get("chat_id") or ""),
        "chat_type": str(source.get("chat_type") or ""),
        "group": _group_key(source),
        "complex_large_group": bool(source.get("complex_large_group")),
        "specified_group": source.get("specified_group"),
        "media_dependency": _has_media_tail(source),
        "context": generated.get("context") or source.get("context_messages") or source.get("context") or [],
        "human_reply": generated.get("human_reply") or [str(source.get("reply") or "")],
        "baseline": _condition_result(
            baseline_generation, judge_entries.get(_entry_id(prefix, "baseline"))
        ),
        "candidate": _condition_result(
            candidate_generation, judge_entries.get(_entry_id(prefix, "candidate"))
        ),
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [
        row for row in cases
        if row["baseline"].get("judge") and row["candidate"].get("judge")
    ]
    baseline_correct = sum(bool(row["baseline"]["judge"].get("correct")) for row in paired)
    candidate_correct = sum(bool(row["candidate"]["judge"].get("correct")) for row in paired)
    improved = sum(
        bool(row["baseline"]["judge"].get("correct"))
        and not bool(row["candidate"]["judge"].get("correct"))
        for row in paired
    )
    regressed = sum(
        not bool(row["baseline"]["judge"].get("correct"))
        and bool(row["candidate"]["judge"].get("correct"))
        for row in paired
    )

    def segment(name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        segment_judged = [
            row for row in selected
            if row["baseline"].get("judge") and row["candidate"].get("judge")
        ]
        base_correct = sum(bool(row["baseline"]["judge"].get("correct")) for row in segment_judged)
        cand_correct = sum(bool(row["candidate"]["judge"].get("correct")) for row in segment_judged)
        return {
            "name": name,
            "reserved": len(selected),
            "paired": len(segment_judged),
            "baseline_judge_correct": base_correct,
            "candidate_judge_correct": cand_correct,
            "baseline_judge_discrimination_accuracy": base_correct / len(segment_judged) if segment_judged else None,
            "candidate_judge_discrimination_accuracy": cand_correct / len(segment_judged) if segment_judged else None,
        }

    return {
        "status": "complete" if len(cases) == 240 and len(paired) == 240 else "partial",
        "reserved_total": len(cases),
        "scoreable_target": sum(not row.get("media_dependency") for row in cases),
        "paired": len(paired),
        "baseline_judge_correct": baseline_correct,
        "candidate_judge_correct": candidate_correct,
        "baseline_judge_discrimination_accuracy": baseline_correct / len(paired) if paired else None,
        "candidate_judge_discrimination_accuracy": candidate_correct / len(paired) if paired else None,
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(paired) - improved - regressed,
        "net_improvement": improved - regressed,
        "segments": {
            "group": segment("群聊", [row for row in cases if row["chat_type"] == "group"]),
            "private": segment("私聊", [row for row in cases if row["chat_type"] != "group"]),
            "complex_large_group": segment("复杂大群", [row for row in cases if row["complex_large_group"]]),
            "common_prosperity": segment("共同富裕群", [row for row in cases if row["specified_group"] == "correct_common_prosperity_with_luxixi"]),
            "youzi": segment("柚子群", [row for row in cases if str(row.get("specified_group") or "").startswith("youzi_group")]),
        },
        "interpretation": "固定240题不参与调优；同一冻结Judge分别辨别真人回复与两版C0回复。",
    }


def _write_results(
    output_dir: Path,
    sources: list[dict[str, Any]],
    generation_entries: dict[str, Any],
    judge_entries: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    cases = [
        _case_result(row, generation_entries, judge_entries)
        for row in sources
    ]
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now().astimezone().isoformat(),
        "locked_config": config,
        "summary": _summary(cases),
        "cases": cases,
    }
    _write_private(output_dir / "results.json", payload)
    return payload


def run_generation(
    output_dir: Path,
    sources: list[dict[str, Any]],
    limit: int | None,
    timeout: float,
    force_reply: bool,
    persona_style_rule: str | None,
    persona_few_shot_path: Path,
    workers: int = 1,
) -> None:
    if not force_reply:
        raise RuntimeError("formal 240-case A/B requires force_reply=true for both versions")
    checkpoint_path = output_dir / "generation_checkpoint.json"
    checkpoint = _load(checkpoint_path, {"entries": {}})
    config = _locked_config(force_reply, persona_style_rule, persona_few_shot_path)
    _ensure_config(checkpoint, config)
    entries = checkpoint.setdefault("entries", {})
    selected = sources[:limit] if limit else sources
    _load_env()
    os.environ["ENABLE_PERSONA_FEW_SHOTS"] = "1"
    os.environ["PERSONA_FEW_SHOT_PATH"] = str(persona_few_shot_path)
    os.environ["GROUP_PERSONA_FEW_SHOT_PATH"] = str(persona_few_shot_path)
    os.environ.pop("PERSONA_FEW_SHOT_RENDER_PATH", None)
    os.environ.pop("GROUP_PERSONA_FEW_SHOT_RENDER_PATH", None)
    os.environ["ENABLE_REACT_TOOLS"] = "0"
    os.environ["ENABLE_SELF_REFINE"] = "0"
    if force_reply or persona_style_rule:
        os.environ["PERSONA_PATH"] = str(_experiment_persona(output_dir, force_reply, persona_style_rule).resolve())
    generator_classes = {
        "baseline": FormalBaselineGenerator,
        "candidate": CumulativeCandidateGenerator,
    }
    pending = []
    for source in selected:
        case_id = str(source["internal_id"])
        conditions = [
            (condition, generator_cls)
            for condition, generator_cls in generator_classes.items()
            if entries.get(_entry_id(case_id, condition), {}).get("status") != "scorable"
        ]
        if conditions:
            pending.append((source, conditions))

    local = threading.local()
    states: list[MemoryEngine] = []
    states_lock = threading.Lock()

    def generate_source(task: tuple[dict[str, Any], list[tuple[str, type]]]) -> dict[str, Any]:
        source, conditions = task
        if not hasattr(local, "client"):
            local.client = _generation_client("volcengine", DEFAULT_VOLCENGINE_CONFIG)
            if str(local.client.model) != config["generation_model"]:
                raise RuntimeError(f"unexpected generation model: {local.client.model}")
            local.memory = MemoryEngine(llm_client=local.client)
            with states_lock:
                states.append(local.memory)
        generated_entries = {}
        case_id = str(source["internal_id"])
        for condition, generator_cls in conditions:
            entry_id = _entry_id(case_id, condition)
            try:
                generated = generate_case(
                    source,
                    local.client,
                    local.memory,
                    timeout,
                    generator_cls=generator_cls,
                    force_reply=True,
                    compact_verified_turn=True,
                )
                status = "scorable" if generated["validity"]["formally_scorable"] else "excluded_unscorable"
                generated_entries[entry_id] = {
                    "status": status,
                    "case": generated,
                    "validity": generated["validity"],
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                status = (
                    "excluded_unscorable"
                    if "no independent incoming messages after the last self message" in str(exc)
                    else "generation_error"
                )
                generated_entries[entry_id] = {"status": status, "error": error}
        return generated_entries

    def save(generated_entries: dict[str, Any]) -> None:
        entries.update(generated_entries)
        checkpoint["updated_at"] = datetime.now().astimezone().isoformat()
        _write_private(checkpoint_path, checkpoint)
        judge_entries = _load(output_dir / "judge_checkpoint.json", {"entries": {}}).get("entries", {})
        _write_results(output_dir, sources, entries, judge_entries, config)

    try:
        if workers <= 1:
            for task in pending:
                save(generate_source(task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(generate_source, task) for task in pending]
                for future in as_completed(futures):
                    save(future.result())
    finally:
        for memory in states:
            memory.shutdown()


def _require_complete_generation(sources: list[dict[str, Any]], entries: dict[str, Any]) -> None:
    expected = {
        _entry_id(str(source["internal_id"]), condition)
        for source in sources
        for condition in ("baseline", "candidate")
    }
    scorable = {key for key, value in entries.items() if value.get("status") == "scorable"}
    missing = expected - scorable
    unexpected = set(entries) - expected
    if missing or unexpected:
        raise RuntimeError(
            f"formal generation incomplete: scorable={len(expected - missing)}/{len(expected)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def run_judge(
    output_dir: Path,
    sources: list[dict[str, Any]],
    limit: int | None,
    force_reply: bool,
    persona_style_rule: str | None,
    persona_few_shot_path: Path,
    workers: int = 1,
) -> None:
    if not force_reply:
        raise RuntimeError("formal 240-case A/B requires force_reply=true for both versions")
    config = _locked_config(force_reply, persona_style_rule, persona_few_shot_path)
    generation_checkpoint = _load(output_dir / "generation_checkpoint.json", {"entries": {}})
    _ensure_config(generation_checkpoint, config)
    generation_entries = generation_checkpoint.get("entries", {})
    if limit is None:
        _require_complete_generation(sources, generation_entries)
    judge_path = output_dir / "judge_checkpoint.json"
    checkpoint = _load(judge_path, {"entries": {}})
    _ensure_config(checkpoint, config)
    entries = checkpoint.setdefault("entries", {})
    profile = load_profile(PROFILE)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    selected = sources[:limit] if limit else sources
    pending: list[tuple[str, dict[str, Any], int]] = []
    for position, source in enumerate(sources):
        if source not in selected:
            continue
        case_id = str(source["internal_id"])
        for condition_index, condition in enumerate(("baseline", "candidate")):
            entry_id = _entry_id(case_id, condition)
            generated = generation_entries.get(entry_id) or {}
            if (
                generated.get("status") != "scorable"
                or entry_id in entries and not entries[entry_id].get("error")
            ):
                continue
            pending.append((entry_id, generated["case"], position * 2 + condition_index))

    def judge_one(task: tuple[str, dict[str, Any], int]) -> tuple[str, dict[str, Any]]:
        entry_id, case, blind_position = task
        blind, human_option = _blind_case(case, blind_position)
        try:
            raw, audit, elapsed_ms, tool_calls = _run_codex(build_messages(blind, reference, profile))
            parsed = _parse_api_judge_response(raw)
            result = {
                    **parsed,
                    "correct": parsed["human_option"] == human_option,
                    "actual_human_option": human_option,
                    "elapsed_ms": elapsed_ms,
                    "tool_calls": tool_calls,
                    "audit": audit,
                    "blind_case": blind,
                }
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return entry_id, result

    def save(result: tuple[str, dict[str, Any]]) -> None:
        entry_id, value = result
        entries[entry_id] = value
        checkpoint["updated_at"] = datetime.now().astimezone().isoformat()
        _write_private(judge_path, checkpoint)
        _write_results(output_dir, sources, generation_entries, entries, config)

    if workers <= 1:
        for task in pending:
            save(judge_one(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(judge_one, task) for task in pending]
            for future in as_completed(futures):
                save(future.result())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("generate", "judge", "all", "render"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--generation-workers", type=int, default=1)
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument("--force-reply", action="store_true", default=True)
    parser.add_argument("--persona-style-rule")
    parser.add_argument("--persona-few-shot-path", type=Path, default=FEW_SHOTS)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    persona_few_shot_path = (
        args.persona_few_shot_path
        if args.persona_few_shot_path.is_absolute()
        else ROOT / args.persona_few_shot_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = list(manifest.get("cases") or [])
    if len(sources) != 240:
        raise RuntimeError(f"fixed manifest must contain 240 cases, got {len(sources)}")
    if args.phase in {"generate", "all"}:
        run_generation(
            output_dir,
            sources,
            args.limit,
            args.generation_timeout_seconds,
            args.force_reply,
            args.persona_style_rule,
            persona_few_shot_path,
            args.generation_workers,
        )
    if args.phase in {"judge", "all"}:
        run_judge(
            output_dir,
            sources,
            args.limit,
            args.force_reply,
            args.persona_style_rule,
            persona_few_shot_path,
            args.judge_workers,
        )
    generation = _load(output_dir / "generation_checkpoint.json", {"entries": {}})
    judge = _load(output_dir / "judge_checkpoint.json", {"entries": {}})
    result = _write_results(
        output_dir,
        sources,
        generation.get("entries", {}),
        judge.get("entries", {}),
        _locked_config(args.force_reply, args.persona_style_rule, persona_few_shot_path),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
