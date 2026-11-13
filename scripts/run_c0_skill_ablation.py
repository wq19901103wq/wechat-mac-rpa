#!/usr/bin/env python3
"""Offline, resumable C0 Skill ablations on the frozen 240-case Judge set.

The script never changes the adopted C0/Judge configuration.  For one Skill it
replays only the frozen cases where that Skill was actually injected, preserves
the original A/B orientation, and reruns only downstream feature/Judge work
whose C0 reply changed.  All artifacts stay under ``c0_skill_ablation``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_human_ai_judge import run_codex_judge_independently
from scripts.run_judge_feature_classifier_experiment import (
    DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
    _json_sha256,
    _write_private,
    extract_features,
    validate_reworked_boolean_correction_fixed_240,
)
from src.reply.c0_config import current_c0_baseline_id, load_current_c0
from src.reply.c0_factory import create_current_c0_clients


BASE_DIR = PROJECT_ROOT / "data/reports/private/judge_feature_classifier_v1"
OUTPUT_ROOT = PROJECT_ROOT / "data/reports/private/c0_skill_ablation"
MANIFEST = BASE_DIR / "group_pattern_holdout_effective_v3_manifest.json"
C0_CHECKPOINT = BASE_DIR / "group_pattern_holdout_effective_v3_c0_checkpoint.json"
REPLAY = BASE_DIR / "group_pattern_holdout_effective_v3_replay.json"
FEATURE_BLIND = BASE_DIR / "group_pattern_holdout_effective_v3_blind.json"
FEATURE_ANSWERS = BASE_DIR / "group_pattern_holdout_effective_v3_answers.json"
FEATURES = BASE_DIR / "group_pattern_holdout_effective_v3_features.json"
DIRECT_BLIND = BASE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_blind.json"
DIRECT_ANSWERS = BASE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_answers.json"
DIRECT_PREDICTIONS = BASE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_predictions.json"
DIRECT_AUDIT = BASE_DIR / "group_pattern_holdout_effective_v3_direct_llm_base_audit.json"
BASE_RESULT = BASE_DIR / "reworked_boolean_correction_fixed_240_result.json"
JUDGE_REFERENCE = (
    PROJECT_ROOT
    / "data/reports/private/human_ai_judge_benchmark_v3_current_c0_f977633e9391c0f4/judge_reference.json"
)
JUDGE_PROFILE = (
    PROJECT_ROOT
    / "data/reports/private/human_ai_judge_benchmark_v3_balanced160/judge_profile_single_rule_full_timeline_v1.json"
)
BASELINE_FORMAL_CORRECT = 209
BASELINE_DIRECT_CORRECT = 196


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_once(source: Path, target: Path) -> None:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _paths(skill: str) -> dict[str, Path]:
    root = OUTPUT_ROOT / skill
    return {
        "root": root,
        "checkpoint": root / "c0_checkpoint.json",
        "replay": root / "replay.json",
        "feature_blind": root / "feature_blind.json",
        "feature_answers": root / "feature_answers.json",
        "features": root / "features.json",
        "direct_blind": root / "direct_blind.json",
        "direct_answers": root / "direct_answers.json",
        "direct_predictions": root / "direct_predictions.json",
        "direct_audit": root / "direct_audit.json",
        "formal_result": root / "formal_result.json",
        "result": root / "result.json",
    }


def _validate_sources() -> None:
    required = [
        MANIFEST, C0_CHECKPOINT, REPLAY, FEATURE_BLIND, FEATURE_ANSWERS,
        FEATURES, DIRECT_BLIND, DIRECT_ANSWERS, DIRECT_PREDICTIONS,
        DIRECT_AUDIT, BASE_RESULT, DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
        JUDGE_REFERENCE, JUDGE_PROFILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required fixed-240 artifact is missing: " + ", ".join(missing))
    base = _load(BASE_RESULT)
    actual = int(base["evaluation"]["scores"]["conservative_correction"]["overall"]["correct"])
    if actual != BASELINE_FORMAL_CORRECT:
        raise ValueError(f"unexpected frozen formal baseline: {actual}/240, expected 209/240")


def _skill_case_ids() -> dict[str, list[str]]:
    checkpoint = _load(C0_CHECKPOINT)
    output: dict[str, list[str]] = {}
    for case_id, entry in (checkpoint.get("entries") or {}).items():
        trace = (((entry.get("replay_case") or {}).get("c0_trace") or {}).get("generation_trace") or {})
        for skill in trace.get("loaded_skills") or []:
            output.setdefault(str(skill), []).append(str(case_id))
    return {skill: sorted(ids) for skill, ids in sorted(output.items())}


def _status() -> dict[str, Any]:
    _validate_sources()
    hits = _skill_case_ids()
    configured = {
        path.stem
        for path in (PROJECT_ROOT / "data/skills").glob("*.md")
        if path.is_file()
    }
    names = sorted(set(hits) | configured)
    skills = []
    for skill in names:
        result_path = _paths(skill)["result"]
        existing = _load(result_path) if result_path.exists() else None
        skills.append({
            "skill": skill,
            "fixed_240_hits": len(hits.get(skill, [])),
            "status": (existing or {}).get("status", "pending"),
            "decision": (existing or {}).get("decision"),
        })
    return {
        "kind": "c0_skill_ablation_status",
        "baseline": {"formal_judge": "209/240", "direct_judge": "196/240"},
        "skills": skills,
        "output_root": str(OUTPUT_ROOT),
    }


def _prepare_candidate(skill: str, source_case_ids: list[str]) -> dict[str, Path]:
    paths = _paths(skill)
    paths["root"].mkdir(parents=True, exist_ok=True)
    checkpoint_was_missing = not paths["checkpoint"].exists()
    for key, source in {
        "checkpoint": C0_CHECKPOINT,
        "replay": REPLAY,
        "feature_blind": FEATURE_BLIND,
        "feature_answers": FEATURE_ANSWERS,
        "features": FEATURES,
        "direct_blind": DIRECT_BLIND,
        "direct_answers": DIRECT_ANSWERS,
        "direct_predictions": DIRECT_PREDICTIONS,
        "direct_audit": DIRECT_AUDIT,
    }.items():
        _copy_once(source, paths[key])

    checkpoint = _load(paths["checkpoint"])
    entries = checkpoint.get("entries") or {}
    # Only remove the affected entries when the candidate is first initialized.
    # Subsequent invocations must retain already replayed cases so a timeout can
    # resume rather than regenerate a different stochastic sample.
    if checkpoint_was_missing:
        for case_id in source_case_ids:
            entries.pop(case_id, None)
        checkpoint["entries"] = entries
    elif any(case_id in entries for case_id in source_case_ids) and checkpoint.get("generation_mode") != "historical_prompt_skill_ablation":
        raise ValueError(
            "existing candidate contains affected cases with unknown generation provenance; "
            f"remove {paths['root']} before rerunning"
        )
    checkpoint["generation_mode"] = "historical_prompt_skill_ablation"
    checkpoint["disabled_skill"] = skill
    _write_private(paths["checkpoint"], checkpoint)
    return paths


def _remove_active_skill_block(user_prompt: str, skill: str) -> str:
    """Remove exactly the historical active-Skill block, and nothing else."""
    pattern = (
        r"<active_skill\s+name=['\"]" + re.escape(skill)
        + r"['\"]>.*?</active_skill>\s*"
    )
    ablated, replacements = re.subn(pattern, "", user_prompt, count=1, flags=re.DOTALL)
    if replacements != 1:
        raise ValueError(
            f"historical prompt must contain exactly one active_skill block for {skill}; "
            f"found {replacements}"
        )
    return ablated


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _parse_replies(text: str) -> list[str]:
    """Mirror the frozen generator's reply parser for a historical replay."""
    if not text or not text.strip():
        return []
    payload = _extract_json_object(text)
    if payload is not None:
        replies = payload.get("replies")
        if not isinstance(replies, list) or not all(isinstance(reply, str) for reply in replies):
            return []
        return [
            reply.strip() for reply in replies
            if reply.strip() not in ("收到", "好的", "嗯", "OK", "1")
        ][:3]
    stripped = text.strip()
    if "{" in stripped or '"replies"' in stripped:
        return []
    for separator in ("\n\n", "\n"):
        pieces = [piece.strip().replace("\n", " ") for piece in text.split(separator) if piece.strip()]
        if len(pieces) > 1:
            return [piece for piece in pieces if piece not in ("收到", "好的", "嗯", "OK", "1")][:3]
    return [stripped.replace("\n", " ")] if stripped not in ("收到", "好的", "嗯", "OK", "1") else []


def _replay_historical_prompt(
    source_entry: dict[str, Any], skill: str, client: Any, timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Regenerate one C0 response from the frozen prompt with one Skill removed.

    The 240 checkpoint predates the current generator source hash.  Calling the
    current ReplyGenerator here would change more than the selected Skill, so
    this intentionally replays the archived system/user prompts directly.
    """
    source_replay = source_entry.get("replay_case") or {}
    source_trace = ((source_replay.get("c0_trace") or {}).get("generation_trace") or {})
    system_prompt = source_trace.get("system_prompt")
    original_user_prompt = source_trace.get("user_prompt")
    if not isinstance(system_prompt, str) or not system_prompt or not isinstance(original_user_prompt, str):
        raise ValueError("source replay has no complete historical prompt")
    user_prompt = _remove_active_skill_block(original_user_prompt, skill)
    loaded_skills = [
        str(name) for name in source_trace.get("loaded_skills") or [] if str(name) != skill
    ]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    llm_calls: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + timeout_seconds
    raw_outputs: list[str] = []
    status = "generation_failed"
    error = "exhausted_retries"

    # This is the no-tools path of ReplyGenerator._react_generate: the frozen
    # C0 baseline made at most three attempts, keeping each prior assistant
    # response in the message stream.
    for attempt in range(1, 4):
        call_started = time.monotonic()
        try:
            remaining = max(1.0, deadline - time.monotonic())
            raw = client.chat(
                messages=messages,
                max_tokens=10000,
                timeout=remaining,
                raise_on_error=True,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            text = str(text or "")
            raw_outputs.append(text)
            llm_calls.append({
                "attempt": attempt,
                "elapsed": round(time.monotonic() - started, 2),
                "messages_count": len(messages),
                "has_tools": False,
                "has_tool_calls": False,
                "response_preview": text[:500],
                "elapsed_ms": round((time.monotonic() - call_started) * 1000),
            })
            messages.append({"role": "assistant", "content": text})
            replies = _parse_replies(text)
            if replies:
                status = "valid"
                error = ""
                break
            payload = _extract_json_object(text) if text else None
            if isinstance(payload, dict) and payload.get("replies") == []:
                status = "silent"
                error = "model_returned_empty_replies"
                break
            if isinstance(payload, dict) and isinstance(payload.get("replies"), list) and any(
                isinstance(reply, str) and reply.strip() for reply in payload["replies"]
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "刚才的 replies 只有会被过滤的占位回复。"
                        "请改为与当前上下文有关、包含实际含义的非空回复，并继续严格输出 JSON。"
                    ),
                })
            error = "invalid_or_empty_model_output"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            llm_calls.append({
                "attempt": attempt,
                "elapsed": round(time.monotonic() - started, 2),
                "messages_count": len(messages),
                "error": error,
                "elapsed_ms": round((time.monotonic() - call_started) * 1000),
            })

    trace = copy.deepcopy(source_trace)
    trace.update({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "loaded_skills": loaded_skills,
        "skill_content": _remove_active_skill_block(str(source_trace.get("skill_content") or ""), skill)
        if skill in [str(name) for name in source_trace.get("loaded_skills") or []]
        else str(source_trace.get("skill_content") or ""),
        "llm_calls": llm_calls,
        "skill_ablation": {
            "disabled_skill": skill,
            "generation_mode": "historical_prompt_skill_ablation",
            "original_user_prompt_sha256": _json_sha256(original_user_prompt),
            "ablated_user_prompt_sha256": _json_sha256(user_prompt),
            "raw_output_previews": [text[:500] for text in raw_outputs],
        },
    })
    metadata = {
        "status": status,
        "error": error,
        "duration_seconds": round(time.monotonic() - started, 3),
        "replies": replies if status == "valid" else [],
        "trace": trace,
    }
    return (copy.deepcopy(source_replay) if status == "valid" else None), metadata


def _generation_status(
    source_case_ids: list[str], candidate_entries: dict[str, Any], calls: int,
) -> dict[str, Any]:
    scoped = [candidate_entries.get(case_id) or {} for case_id in source_case_ids]
    outcomes = Counter(str(entry.get("status") or "pending") for entry in scoped)
    return {
        "generation_mode": "historical_prompt_skill_ablation",
        "api_calls_this_run": calls,
        "required": len(source_case_ids),
        "outcomes": dict(sorted(outcomes.items())),
        "pending": len(source_case_ids) - int(outcomes.get("valid", 0)),
    }


def _generate(skill: str, paths: dict[str, Path], source_case_ids: list[str], max_calls: int | None) -> dict[str, Any]:
    config = load_current_c0()
    source = _load(C0_CHECKPOINT)
    checkpoint = _load(paths["checkpoint"])
    if source.get("current_c0_baseline_id") != current_c0_baseline_id():
        raise ValueError("current adopted C0 configuration no longer matches the frozen 240 source")
    private_client, group_client = create_current_c0_clients(config=config)
    calls = 0
    entries = checkpoint.setdefault("entries", {})
    for case_id in source_case_ids:
        if (entries.get(case_id) or {}).get("status") == "valid":
            continue
        if max_calls is not None and calls >= max_calls:
            break
        source_entry = (source.get("entries") or {}).get(case_id)
        if not source_entry or source_entry.get("status") != "valid":
            raise ValueError(f"frozen source case is not valid: {case_id}")
        relationship = str(source_entry.get("relationship") or "")
        if relationship not in {"private", "group"}:
            raise ValueError(f"frozen source case has invalid relationship: {case_id}")
        calls += 1
        try:
            replay, output = _replay_historical_prompt(
                source_entry,
                skill,
                group_client if relationship == "group" else private_client,
                int(config[relationship]["timeout_seconds"]),
            )
            if replay is None:
                entries[case_id] = {
                    "split": "holdout",
                    "relationship": relationship,
                    "status": output["status"],
                    "duration_seconds": output["duration_seconds"],
                    "failure_diagnostic": output["error"],
                    "generation_trace": output["trace"],
                }
            else:
                replay["ai_reply_original"] = output["replies"]
                c0_trace = replay.setdefault("c0_trace", {})
                c0_trace["duration_seconds"] = output["duration_seconds"]
                c0_trace["generation_trace"] = output["trace"]
                entries[case_id] = {
                    "split": "holdout",
                    "relationship": relationship,
                    "status": "valid",
                    "replay_case": replay,
                }
        except Exception as exc:
            entries[case_id] = {
                "split": "holdout",
                "relationship": relationship,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_private(paths["checkpoint"], checkpoint)
    return _generation_status(source_case_ids, entries, calls)


def _changed_source_ids(paths: dict[str, Path], source_case_ids: list[str]) -> list[str]:
    base_entries = _load(C0_CHECKPOINT)["entries"]
    candidate_entries = _load(paths["checkpoint"])["entries"]
    changed: list[str] = []
    for source_id in source_case_ids:
        before = ((base_entries.get(source_id) or {}).get("replay_case") or {})
        after = ((candidate_entries.get(source_id) or {}).get("replay_case") or {})
        if not after or (after.get("ai_reply_original") != before.get("ai_reply_original")):
            changed.append(source_id)
    return changed


def _update_replay(paths: dict[str, Path]) -> None:
    candidate_entries = _load(paths["checkpoint"])["entries"]
    replay = _load(paths["replay"])
    cases = []
    for row in replay.get("cases") or []:
        source_id = str(row["case_id"])
        candidate = (candidate_entries.get(source_id) or {}).get("replay_case")
        if not candidate:
            raise ValueError(f"candidate checkpoint lacks valid replay for {source_id}")
        cases.append(copy.deepcopy(candidate))
    replay["cases"] = cases
    replay["skill_ablation"] = {"disabled_skill": paths["root"].name}
    _write_private(paths["replay"], replay)


def _replace_ai_options(
    blind: dict[str, Any], answers: dict[str, Any], candidate_entries: dict[str, Any],
    changed_source_ids: set[str],
) -> set[str]:
    source_by_public = {
        str(row["case_id"]): str(row["source_case_id"])
        for row in answers.get("cases") or []
    }
    public_changed: set[str] = set()
    for case in blind.get("cases") or []:
        public_id = str(case["case_id"])
        source_id = source_by_public.get(public_id)
        if source_id not in changed_source_ids:
            continue
        entry = candidate_entries.get(source_id) or {}
        replay = entry.get("replay_case") or {}
        ai_reply = replay.get("ai_reply_original")
        if not isinstance(ai_reply, list) or not ai_reply:
            raise ValueError(f"changed source case has no valid AI reply: {source_id}")
        human_option = next(
            (str(row["human_option"]) for row in answers["cases"] if str(row["case_id"]) == public_id),
            None,
        )
        if human_option not in {"A", "B"}:
            raise ValueError(f"answer key lacks public case {public_id}")
        ai_option = "B" if human_option == "A" else "A"
        case[f"option_{ai_option}"] = ai_reply
        public_changed.add(public_id)
    return public_changed


def _update_packets(paths: dict[str, Path], changed_source_ids: list[str]) -> set[str]:
    candidate_entries = _load(paths["checkpoint"])["entries"]
    source_ids = set(changed_source_ids)
    feature_blind = _load(paths["feature_blind"])
    feature_answers = _load(paths["feature_answers"])
    direct_blind = _load(paths["direct_blind"])
    direct_answers = _load(paths["direct_answers"])
    changed_feature = _replace_ai_options(
        feature_blind, feature_answers, candidate_entries, source_ids
    )
    changed_direct = _replace_ai_options(
        direct_blind, direct_answers, candidate_entries, source_ids
    )
    if changed_feature != changed_direct:
        raise ValueError("feature and direct packets map changed source cases differently")
    if feature_blind.get("cases") != direct_blind.get("cases"):
        raise ValueError("feature and direct blind packets diverged")
    feature_answers["blind_sha256"] = _json_sha256(feature_blind)
    direct_answers["blind_sha256"] = _json_sha256(direct_blind)
    _write_private(paths["feature_blind"], feature_blind)
    _write_private(paths["feature_answers"], feature_answers)
    _write_private(paths["direct_blind"], direct_blind)
    _write_private(paths["direct_answers"], direct_answers)
    return changed_feature


def _public_case_ids(paths: dict[str, Path], changed_source_ids: list[str]) -> set[str]:
    source_to_public = {
        str(row["source_case_id"]): str(row["case_id"])
        for row in _load(paths["direct_answers"]).get("cases") or []
    }
    return {
        source_to_public[source_id]
        for source_id in changed_source_ids
        if source_id in source_to_public
    }


def _packets_already_contain_candidate(
    paths: dict[str, Path], changed_source_ids: list[str], public_changed: set[str],
) -> bool:
    """Whether an interrupted run already wrote this candidate into both packets."""
    candidate_entries = _load(paths["checkpoint"]).get("entries") or {}
    answers = {
        str(row["case_id"]): row for row in _load(paths["direct_answers"]).get("cases") or []
    }
    for key in ("feature_blind", "direct_blind"):
        cases = {
            str(row["case_id"]): row for row in _load(paths[key]).get("cases") or []
        }
        if set(cases) != set(answers):
            return False
        for public_id in public_changed:
            answer = answers.get(public_id) or {}
            source_id = str(answer.get("source_case_id") or "")
            if source_id not in changed_source_ids:
                return False
            ai_reply = ((candidate_entries.get(source_id) or {}).get("replay_case") or {}).get("ai_reply_original")
            human_option = str(answer.get("human_option") or "")
            ai_option = "B" if human_option == "A" else "A"
            if not isinstance(ai_reply, list) or cases[public_id].get(f"option_{ai_option}") != ai_reply:
                return False
    return True


def _invalidate_changed_downstream(paths: dict[str, Path], public_changed: set[str]) -> None:
    feature_blind = _load(paths["feature_blind"])
    features = _load(paths["features"])
    features["blind_sha256"] = _json_sha256(feature_blind)
    features["entries"] = {
        case_id: row for case_id, row in (features.get("entries") or {}).items()
        if str(case_id) not in public_changed
    }
    _write_private(paths["features"], features)

    direct_blind = _load(paths["direct_blind"])
    for key in ("direct_predictions", "direct_audit"):
        payload = _load(paths[key])
        payload["blind_sha256"] = _json_sha256(direct_blind)
        rows_key = "cases" if key == "direct_predictions" else "calls"
        payload[rows_key] = [
            row for row in payload.get(rows_key) or []
            if str(row.get("case_id")) not in public_changed
        ]
        _write_private(paths[key], payload)


def _score_direct(paths: dict[str, Path]) -> dict[str, Any]:
    answers = {str(row["case_id"]): row for row in _load(paths["direct_answers"])["cases"]}
    blind = {str(row["case_id"]): row for row in _load(paths["direct_blind"])["cases"]}
    predictions = _load(paths["direct_predictions"])["cases"]
    if {str(row["case_id"]) for row in predictions} != set(answers):
        raise ValueError("direct Judge predictions do not cover the fixed 240")
    buckets: dict[str, dict[str, int]] = {
        "overall": {"correct": 0, "total": 0},
        "private": {"correct": 0, "total": 0},
        "group": {"correct": 0, "total": 0},
    }
    for row in predictions:
        case_id = str(row["case_id"])
        relation = str(blind[case_id]["relationship"])
        correct = str(row["human_option"]) == str(answers[case_id]["human_option"])
        for bucket in ("overall", relation):
            buckets[bucket]["total"] += 1
            buckets[bucket]["correct"] += int(correct)
    for value in buckets.values():
        value["accuracy"] = value["correct"] / value["total"]
    return buckets


def _formal_changed_cases(result: dict[str, Any]) -> dict[str, list[str]]:
    base_rows = {
        str(row["case_id"]): row
        for row in _load(BASE_RESULT)["evaluation"]["cases"]
    }
    candidate_rows = {
        str(row["case_id"]): row
        for row in result["evaluation"]["cases"]
    }
    corrected, regressed = [], []
    for case_id, current in candidate_rows.items():
        before = base_rows[case_id]
        if bool(current["final_correct"]) and not bool(before["final_correct"]):
            corrected.append(case_id)
        if bool(before["final_correct"]) and not bool(current["final_correct"]):
            regressed.append(case_id)
    return {"corrected": sorted(corrected), "regressed": sorted(regressed)}


def _inherit_result(skill: str, source_case_ids: list[str], generation: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "kind": "c0_skill_ablation_fixed_240",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skill": skill,
        "status": "completed_no_reply_change",
        "fixed_240_skill_hits": len(source_case_ids),
        "replayed_cases": len(source_case_ids),
        "changed_ai_replies": 0,
        "generation": generation,
        "direct_judge": {"correct": BASELINE_DIRECT_CORRECT, "total": 240, "net_change": 0},
        "formal_judge": {"correct": BASELINE_FORMAL_CORRECT, "total": 240, "net_change": 0},
        "decision": "remove",
        "reason": "disabling the Skill did not change any frozen C0 reply",
        "online_components_modified": [],
    }
    _write_private(_paths(skill)["result"], result)
    return result


def run_skill(skill: str, max_calls: int | None, feature_workers: int) -> dict[str, Any]:
    _validate_sources()
    source_case_ids = _skill_case_ids().get(skill, [])
    if not source_case_ids:
        result = {
            "schema_version": 1,
            "kind": "c0_skill_ablation_fixed_240",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "skill": skill,
            "status": "not_evaluable_no_fixed_240_hits",
            "fixed_240_skill_hits": 0,
            "decision": None,
            "online_components_modified": [],
        }
        _write_private(_paths(skill)["result"], result)
        return result
    paths = _prepare_candidate(skill, source_case_ids)
    generation = _generate(skill, paths, source_case_ids, max_calls)
    pending = int(generation.get("pending") or 0)
    if pending:
        return {
            "skill": skill,
            "status": "generation_incomplete",
            "generation": generation,
            "rerun": f"python scripts/run_c0_skill_ablation.py --skill {skill}",
        }
    changed_source_ids = _changed_source_ids(paths, source_case_ids)
    if not changed_source_ids:
        return _inherit_result(skill, source_case_ids, generation)
    # A prior invocation can finish scoring but be interrupted while serializing
    # the summary.  Reuse those complete artifacts rather than invalidating and
    # recomputing the affected cases.
    if paths["formal_result"].exists():
        existing_features = _load(paths["features"])
        existing_predictions = _load(paths["direct_predictions"])
        if (
            len(existing_features.get("entries") or {}) == 240
            and len(existing_predictions.get("cases") or []) == 240
        ):
            formal = _load(paths["formal_result"])
            answers = _load(paths["direct_answers"])
            source_by_public = {
                str(row["source_case_id"]): str(row["case_id"])
                for row in answers.get("cases") or []
            }
            public_changed = {
                source_by_public[source_id] for source_id in changed_source_ids
                if source_id in source_by_public
            }
            direct = _score_direct(paths)
            formal_scores = formal["evaluation"]["scores"]["conservative_correction"]
            formal_correct = int(formal_scores["overall"]["correct"])
            result = {
                "schema_version": 1,
                "kind": "c0_skill_ablation_fixed_240",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "skill": skill,
                "status": "completed",
                "fixed_240_skill_hits": len(source_case_ids),
                "replayed_cases": len(source_case_ids),
                "changed_ai_replies": len(changed_source_ids),
                "changed_source_case_ids": changed_source_ids,
                "changed_public_case_ids": sorted(public_changed),
                "generation": generation,
                "feature_extraction": {"status": "reused_complete_artifact", "required": 240},
                "direct_judge": {"scores": direct, "net_change": int(direct["overall"]["correct"]) - BASELINE_DIRECT_CORRECT},
                "formal_judge": {"scores": formal_scores, "net_change": formal_correct - BASELINE_FORMAL_CORRECT, "changed_cases": _formal_changed_cases(formal)},
                "decision": "retain" if formal_correct > BASELINE_FORMAL_CORRECT else "remove",
                "decision_rule": "retain only when disabling the Skill raises formal Judge accuracy above 209/240",
                "artifacts": {key: str(path) for key, path in paths.items() if key != "root"},
                "online_components_modified": [],
            }
            _write_private(paths["result"], result)
            return result
    public_changed = _public_case_ids(paths, changed_source_ids)
    _update_replay(paths)
    if not _packets_already_contain_candidate(paths, changed_source_ids, public_changed):
        public_changed = _update_packets(paths, changed_source_ids)
        _invalidate_changed_downstream(paths, public_changed)
    feature_status = extract_features(
        paths["feature_blind"], paths["features"], None, False, feature_workers,
        180, None, None,
    )
    if int(feature_status["outcomes"].get("valid", 0)) != int(feature_status["required"]):
        return {
            "skill": skill,
            "status": "feature_extraction_incomplete",
            "changed_ai_replies": len(changed_source_ids),
            "feature_status": feature_status,
        }
    judge_status = run_codex_judge_independently(
        paths["direct_blind"], paths["direct_predictions"], paths["direct_audit"],
        JUDGE_REFERENCE, workers=1, judge_profile_path=JUDGE_PROFILE,
        only_case_ids=public_changed,
    )
    if int(judge_status["completed"]) != 240:
        return {
            "skill": skill,
            "status": "direct_judge_incomplete",
            "changed_ai_replies": len(changed_source_ids),
            "judge_status": judge_status,
        }
    formal = validate_reworked_boolean_correction_fixed_240(
        DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
        paths["direct_blind"], paths["direct_answers"], paths["replay"], paths["features"],
        paths["direct_predictions"], paths["formal_result"], paths["feature_blind"],
        paths["feature_answers"],
    )
    direct = _score_direct(paths)
    formal_scores = formal["evaluation"]["scores"]["conservative_correction"]
    formal_correct = int(formal_scores["overall"]["correct"])
    result = {
        "schema_version": 1,
        "kind": "c0_skill_ablation_fixed_240",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skill": skill,
        "status": "completed",
        "fixed_240_skill_hits": len(source_case_ids),
        "replayed_cases": len(source_case_ids),
        "changed_ai_replies": len(changed_source_ids),
        "changed_source_case_ids": changed_source_ids,
        "changed_public_case_ids": sorted(public_changed),
        "generation": generation,
        "feature_extraction": feature_status,
        "direct_judge": {
            "scores": direct,
            "net_change": int(direct["overall"]["correct"]) - BASELINE_DIRECT_CORRECT,
        },
        "formal_judge": {
            "scores": formal_scores,
            "net_change": formal_correct - BASELINE_FORMAL_CORRECT,
            "changed_cases": _formal_changed_cases(formal),
        },
        # Per task rule: only an accuracy increase after removal protects a Skill.
        "decision": "retain" if formal_correct > BASELINE_FORMAL_CORRECT else "remove",
        "decision_rule": "retain only when disabling the Skill raises formal Judge accuracy above 209/240",
        "artifacts": {key: str(path) for key, path in paths.items() if key != "root"},
        "online_components_modified": [],
    }
    _write_private(paths["result"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="show frozen-set Skill coverage and progress")
    parser.add_argument("--skill", help="one Skill to ablate")
    parser.add_argument("--all", action="store_true", help="run every Skill that hit the fixed 240")
    parser.add_argument("--max-calls", type=int, default=None, help="cap C0 regenerations for this invocation")
    parser.add_argument("--feature-workers", type=int, default=4)
    args = parser.parse_args()
    if sum(bool(value) for value in (args.status, args.skill, args.all)) != 1:
        parser.error("choose exactly one of --status, --skill, or --all")
    if args.status:
        print(json.dumps(_status(), ensure_ascii=False, indent=2))
        return 0
    if args.skill:
        print(json.dumps(run_skill(args.skill, args.max_calls, args.feature_workers), ensure_ascii=False, indent=2))
        return 0
    summary = []
    for skill, case_ids in _skill_case_ids().items():
        summary.append(run_skill(skill, args.max_calls, args.feature_workers))
    print(json.dumps({"kind": "c0_skill_ablation_all", "results": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
