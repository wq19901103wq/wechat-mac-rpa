#!/usr/bin/env python3
"""Evaluate one frozen C0 A/B reply set without touching defaults.

The candidate replies have already been generated on the fixed 160-group / 80-
private set.  This runner only creates isolated Judge packets, evaluates the
new baseline once, and reuses its unchanged private/group judgments when the
candidate is identical.  It never writes ``config/current_c0.json`` or starts
the online bot.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
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


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "data/reports/private/"
    "situation_c0_group_public_question_specific_followup_formal240_160g80p_v2"
)
SOURCE_ROOT = DEFAULT_SOURCE_ROOT
OUTPUT_ROOT = SOURCE_ROOT / "judge_evaluation"
FROZEN_BASELINE_ROOT = (
    PROJECT_ROOT
    / "data/reports/private/"
    "situation_c0_group_explicit_self_stance_action_formal240_160g80p_v2"
)
FROZEN_JUDGE_ROOT = FROZEN_BASELINE_ROOT / "judge_evaluation/baseline"

FEATURE_BLIND = FROZEN_JUDGE_ROOT / "feature_blind.json"
FEATURE_ANSWERS = FROZEN_JUDGE_ROOT / "feature_answers.json"
DIRECT_BLIND = FROZEN_JUDGE_ROOT / "direct_blind.json"
DIRECT_ANSWERS = FROZEN_JUDGE_ROOT / "direct_answers.json"
REPLAY = FROZEN_JUDGE_ROOT / "replay.json"
JUDGE_REFERENCE = (
    PROJECT_ROOT
    / "data/reports/private/human_ai_judge_benchmark_v3_current_c0_f977633e9391c0f4/judge_reference.json"
)
JUDGE_PROFILE = (
    PROJECT_ROOT
    / "data/reports/private/human_ai_judge_benchmark_v3_balanced160"
    / "judge_profile_single_rule_full_timeline_v1.json"
)

CONDITIONS = ("candidate",)
EXPECTED_FROZEN_C0_BASELINE_ID = "c0-a67799c5ac488f5d"
EXPECTED_CHANGED_SOURCE_IDS = {
    "82b1d486909971663d8b",
    "d5df6307a75c07e85f47",
    "98cbc343f6c8791d81ee",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_private(path, value)


def _paths(condition: str) -> dict[str, Path]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    root = OUTPUT_ROOT / condition
    return {
        "root": root,
        "feature_blind": root / "feature_blind.json",
        "feature_answers": root / "feature_answers.json",
        "direct_blind": root / "direct_blind.json",
        "direct_answers": root / "direct_answers.json",
        "replay": root / "replay.json",
        "features": root / "features.json",
        "direct_predictions": root / "direct_predictions.json",
        "direct_audit": root / "direct_audit.json",
        "formal_result": root / "formal_result.json",
    }


def _required_sources() -> list[Path]:
    return [
        SOURCE_ROOT / "generation_checkpoint.json",
        *(FROZEN_BASELINE_ROOT / f"results.shard-{index}-of-4.json" for index in range(4)),
        FEATURE_BLIND,
        FEATURE_ANSWERS,
        DIRECT_BLIND,
        DIRECT_ANSWERS,
        REPLAY,
        FROZEN_JUDGE_ROOT / "features.json",
        FROZEN_JUDGE_ROOT / "direct_predictions.json",
        FROZEN_JUDGE_ROOT / "direct_audit.json",
        FROZEN_JUDGE_ROOT / "formal_result.json",
        JUDGE_REFERENCE,
        JUDGE_PROFILE,
        DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
    ]


def _load_c0_results() -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in _required_sources() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))
    checkpoint = _load(SOURCE_ROOT / "generation_checkpoint.json")
    if checkpoint.get("current_c0_baseline_id") != EXPECTED_FROZEN_C0_BASELINE_ID:
        raise ValueError("candidate checkpoint is not bound to the frozen C0 baseline")
    if checkpoint.get("baseline_generation") != 0:
        raise ValueError("candidate checkpoint must not regenerate the frozen baseline")
    if checkpoint.get("baseline_imported_or_reused") != 240:
        raise ValueError("candidate checkpoint must reuse all 240 frozen baseline replies")
    checkpoint_entries = checkpoint.get("entries")
    if not isinstance(checkpoint_entries, dict):
        raise ValueError("candidate checkpoint does not contain entries")

    cases: list[dict[str, Any]] = []
    for index in range(4):
        payload = _load(FROZEN_BASELINE_ROOT / f"results.shard-{index}-of-4.json")
        shard_cases = payload.get("cases")
        if not isinstance(shard_cases, list):
            raise ValueError(f"result shard {index} does not contain cases")
        cases.extend(shard_cases)
    by_source: dict[str, dict[str, Any]] = {}
    for case in cases:
        source_id = str(case.get("case_id") or "")
        relationship = str(case.get("relationship") or "")
        if not source_id or source_id in by_source:
            raise ValueError(f"duplicate or missing C0 source case id: {source_id}")
        if relationship not in {"group", "private"}:
            raise ValueError(f"invalid relationship for {source_id}: {relationship}")
        baseline_run = ((case.get("conditions") or {}).get("baseline") or {}).get("runs")
        baseline_run = baseline_run[0] if isinstance(baseline_run, list) and baseline_run else None
        baseline_reply = baseline_run.get("reply") if isinstance(baseline_run, dict) else None
        if (
            not isinstance(baseline_run, dict)
            or baseline_run.get("status") != "scorable"
            or not isinstance(baseline_reply, list)
            or not baseline_reply
            or not all(isinstance(message, str) and message for message in baseline_reply)
        ):
            raise ValueError(f"frozen baseline has no valid scorable reply for {source_id}")
        normalized: dict[str, Any] = {
            "source_case_id": source_id,
            "relationship": relationship,
            "context": case.get("context"),
            "human_reply": case.get("human_reply"),
            "baseline": list(baseline_reply),
        }
        for condition in ("baseline", "candidate"):
            entry = checkpoint_entries.get(f"{source_id}:{condition}")
            result = entry.get("result") if isinstance(entry, dict) else None
            reply = result.get("replies") if isinstance(result, dict) else None
            if (
                not isinstance(entry, dict)
                or entry.get("status") != "scorable"
                or not isinstance(reply, list)
                or not reply
                or not all(isinstance(message, str) and message for message in reply)
            ):
                raise ValueError(f"candidate checkpoint {condition} has no valid scorable reply for {source_id}")
            if condition == "baseline":
                if list(reply) != normalized["baseline"]:
                    raise ValueError(f"candidate baseline differs from frozen baseline for {source_id}")
            else:
                normalized["candidate"] = list(reply)
        by_source[source_id] = normalized
    relationships = Counter(row["relationship"] for row in by_source.values())
    if len(by_source) != 240 or relationships != Counter({"group": 160, "private": 80}):
        raise ValueError(f"C0 results must have 160 group + 80 private cases, got {relationships}")
    private_changed = [
        source_id for source_id, row in by_source.items()
        if row["relationship"] == "private" and row["baseline"] != row["candidate"]
    ]
    if private_changed:
        raise ValueError("candidate must reuse baseline output for private cases: " + ", ".join(private_changed))
    changed_sources = {
        source_id for source_id, row in by_source.items()
        if row["baseline"] != row["candidate"]
    }
    if changed_sources != EXPECTED_CHANGED_SOURCE_IDS:
        raise ValueError(
            "candidate changed sources differ from this frozen single-rule evaluation: "
            + ", ".join(sorted(changed_sources))
        )
    return by_source


def _answer_mapping(answers: dict[str, Any]) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for row in answers.get("cases") or []:
        public_id = str(row.get("case_id") or "")
        source_id = str(row.get("source_case_id") or "")
        human_option = str(row.get("human_option") or "")
        if not public_id or not source_id or human_option not in {"A", "B"} or public_id in mapping:
            raise ValueError("invalid or duplicate answer mapping")
        mapping[public_id] = (source_id, human_option)
    if len(mapping) != 240:
        raise ValueError(f"answer packet must cover 240 cases, got {len(mapping)}")
    return mapping


def _replace_all_ai_options(
    blind: dict[str, Any],
    answers: dict[str, Any],
    c0_results: dict[str, dict[str, Any]],
    condition: str,
) -> None:
    mapping = _answer_mapping(answers)
    case_ids = [str(row.get("case_id") or "") for row in blind.get("cases") or []]
    if len(case_ids) != 240 or set(case_ids) != set(mapping):
        raise ValueError("blind and answers do not cover the same 240 cases")
    for case in blind["cases"]:
        public_id = str(case["case_id"])
        source_id, human_option = mapping[public_id]
        source = c0_results.get(source_id)
        if source is None:
            raise ValueError(f"C0 result lacks source case: {source_id}")
        if str(case.get("relationship") or "") != source["relationship"]:
            raise ValueError(f"relationship mismatch for {public_id}")
        ai_option = "B" if human_option == "A" else "A"
        if not isinstance(case.get(f"option_{human_option}"), list):
            raise ValueError(f"blind human option is malformed for {public_id}")
        case[f"option_{ai_option}"] = list(source[condition])


def _updated_replay(c0_results: dict[str, dict[str, Any]], condition: str) -> dict[str, Any]:
    replay = copy.deepcopy(_load(REPLAY))
    rows = replay.get("cases") or []
    if len(rows) != 240:
        raise ValueError("frozen replay does not contain 240 cases")
    seen: set[str] = set()
    for row in rows:
        source_id = str(row.get("case_id") or "")
        source = c0_results.get(source_id)
        if source is None or source_id in seen:
            raise ValueError(f"frozen replay has invalid case id: {source_id}")
        seen.add(source_id)
        row["ai_reply_original"] = list(source[condition])
    if set(c0_results) != seen:
        raise ValueError("frozen replay and C0 result source cases differ")
    replay["c0_candidate_evaluation"] = {
        "condition": condition,
        "source_results": str(SOURCE_ROOT),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return replay


def _write_condition_packets(c0_results: dict[str, dict[str, Any]], condition: str) -> None:
    paths = _paths(condition)
    feature_blind = copy.deepcopy(_load(FEATURE_BLIND))
    feature_answers = copy.deepcopy(_load(FEATURE_ANSWERS))
    direct_blind = copy.deepcopy(_load(DIRECT_BLIND))
    direct_answers = copy.deepcopy(_load(DIRECT_ANSWERS))
    if feature_answers.get("cases") != direct_answers.get("cases"):
        raise ValueError("fixed feature and direct answer keys differ")
    _replace_all_ai_options(feature_blind, feature_answers, c0_results, condition)
    _replace_all_ai_options(direct_blind, direct_answers, c0_results, condition)
    if feature_blind.get("cases") != direct_blind.get("cases"):
        raise ValueError("feature and direct packets diverged after replacing C0 replies")
    feature_answers["blind_sha256"] = _json_sha256(feature_blind)
    direct_answers["blind_sha256"] = _json_sha256(direct_blind)
    _write(paths["feature_blind"], feature_blind)
    _write(paths["feature_answers"], feature_answers)
    _write(paths["direct_blind"], direct_blind)
    _write(paths["direct_answers"], direct_answers)
    _write(paths["replay"], _updated_replay(c0_results, condition))


def _changed_public_ids(c0_results: dict[str, dict[str, Any]]) -> set[str]:
    mapping = _answer_mapping(_load(DIRECT_ANSWERS))
    source_to_public = {source_id: public_id for public_id, (source_id, _) in mapping.items()}
    changed_sources = {
        source_id for source_id, row in c0_results.items()
        if row["baseline"] != row["candidate"]
    }
    if any(c0_results[source_id]["relationship"] != "group" for source_id in changed_sources):
        raise ValueError("only group replies may change in this candidate")
    missing = changed_sources.difference(source_to_public)
    if missing:
        raise ValueError("C0 changed source is absent from Judge answer key: " + ", ".join(sorted(missing)))
    return {source_to_public[source_id] for source_id in changed_sources}


def _ensure_candidate_reuse(changed_public_ids: set[str]) -> None:
    candidate = _paths("candidate")
    baseline_predictions = _load(FROZEN_JUDGE_ROOT / "direct_predictions.json")
    baseline_audit = _load(FROZEN_JUDGE_ROOT / "direct_audit.json")
    feature_checkpoint = _load(FROZEN_JUDGE_ROOT / "features.json")
    required_ids = [
        str(row["case_id"])
        for row in _load(FROZEN_JUDGE_ROOT / "direct_blind.json").get("cases") or []
    ]
    candidate_ids = [str(row["case_id"]) for row in _load(candidate["direct_blind"]).get("cases") or []]
    if candidate_ids != required_ids:
        raise ValueError("candidate Judge packet differs from frozen baseline case order")
    if [str(row.get("case_id") or "") for row in baseline_predictions.get("cases") or []] != required_ids:
        raise ValueError("baseline direct Judge is incomplete or unordered")
    if [str(row.get("case_id") or "") for row in baseline_audit.get("calls") or []] != required_ids:
        raise ValueError("baseline direct Judge audit is incomplete or unordered")
    if set((feature_checkpoint.get("entries") or {})) != set(required_ids):
        raise ValueError("baseline feature extraction is incomplete")

    candidate_blind_sha = _json_sha256(_load(candidate["direct_blind"]))
    candidate_feature_sha = _json_sha256(_load(candidate["feature_blind"]))
    unchanged = set(required_ids).difference(changed_public_ids)

    if candidate["direct_predictions"].exists() != candidate["direct_audit"].exists():
        raise ValueError("candidate direct Judge resume files are incomplete")
    if not candidate["direct_predictions"].exists():
        prediction = copy.deepcopy(baseline_predictions)
        audit = copy.deepcopy(baseline_audit)
        invocation_id = f"c0_candidate_{uuid.uuid4().hex}"
        prediction["invocation_id"] = invocation_id
        prediction["blind_sha256"] = candidate_blind_sha
        prediction["cases"] = [
            row for row in prediction["cases"] if str(row.get("case_id") or "") in unchanged
        ]
        audit["invocation_id"] = invocation_id
        audit["blind_sha256"] = candidate_blind_sha
        audit["calls"] = [
            {
                **row,
                "reused_from_frozen_baseline": True,
                "baseline_invocation_id": baseline_predictions["invocation_id"],
            }
            for row in audit["calls"]
            if str(row.get("case_id") or "") in unchanged
        ]
        _write(candidate["direct_predictions"], prediction)
        _write(candidate["direct_audit"], audit)

    if not candidate["features"].exists():
        checkpoint = copy.deepcopy(feature_checkpoint)
        checkpoint["blind_sha256"] = candidate_feature_sha
        checkpoint["entries"] = {
            case_id: row for case_id, row in checkpoint["entries"].items()
            if case_id in unchanged
        }
        _write(candidate["features"], checkpoint)


def _run_condition(
    condition: str,
    *,
    workers: int,
    feature_timeout_seconds: int,
    only_case_ids: set[str] | None = None,
) -> None:
    paths = _paths(condition)
    extract_features(
        blind_path=paths["feature_blind"],
        checkpoint_path=paths["features"],
        max_cases=None,
        retry_failed=True,
        workers=workers,
        timeout_seconds=feature_timeout_seconds,
        model=None,
        reasoning_effort=None,
    )
    run_codex_judge_independently(
        blind_path=paths["direct_blind"],
        predictions_path=paths["direct_predictions"],
        audit_path=paths["direct_audit"],
        reference_path=JUDGE_REFERENCE,
        max_cases=None,
        workers=workers,
        judge_profile_path=JUDGE_PROFILE,
        only_case_ids=only_case_ids,
    )
    result = validate_reworked_boolean_correction_fixed_240(
        development_result_path=DEFAULT_REWORKED_BOOLEAN_CORRECTION_DEVELOPMENT_RESULT,
        blind_path=paths["direct_blind"],
        answers_path=paths["direct_answers"],
        replay_path=paths["replay"],
        feature_checkpoint_path=paths["features"],
        base_predictions_path=paths["direct_predictions"],
        result_path=paths["formal_result"],
        feature_source_blind_path=paths["feature_blind"],
        feature_source_answers_path=paths["feature_answers"],
    )
    if result.get("fixed_240", {}).get("total") != 240:
        raise ValueError(f"{condition} formal result is incomplete")


def _scores(formal_result: dict[str, Any]) -> dict[str, Any]:
    return formal_result["evaluation"]["scores"]


def _paired_report(c0_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = _paths("candidate")
    baseline_result = _load(FROZEN_JUDGE_ROOT / "formal_result.json")
    candidate_result = _load(candidate["formal_result"])
    baseline_cases = {
        str(row["case_id"]): row for row in baseline_result["evaluation"].get("cases") or []
    }
    candidate_cases = {
        str(row["case_id"]): row for row in candidate_result["evaluation"].get("cases") or []
    }
    mapping = _answer_mapping(_load(DIRECT_ANSWERS))
    if set(baseline_cases) != set(mapping) or set(candidate_cases) != set(mapping):
        raise ValueError("formal results do not cover the fixed 240 cases")

    changes: list[dict[str, Any]] = []
    for public_id, (source_id, _) in mapping.items():
        source = c0_results[source_id]
        base_case = baseline_cases[public_id]
        candidate_case = candidate_cases[public_id]
        base_correct = bool(base_case["final_correct"])
        candidate_correct = bool(candidate_case["final_correct"])
        if base_correct and not candidate_correct:
            effect = "improved_for_c0"
        elif not base_correct and candidate_correct:
            effect = "worsened_for_c0"
        else:
            effect = "unchanged_judge_outcome"
        if source["baseline"] != source["candidate"] or effect != "unchanged_judge_outcome":
            changes.append({
                "public_case_id": public_id,
                "source_case_id": source_id,
                "relationship": source["relationship"],
                "effect": effect,
                "context": source["context"],
                "human_reply": source["human_reply"],
                "baseline_reply": source["baseline"],
                "candidate_reply": source["candidate"],
                "baseline_judge": {
                    "prediction": base_case["final_prediction"],
                    "correct": base_correct,
                    "base_prediction": base_case["base_prediction"],
                },
                "candidate_judge": {
                    "prediction": candidate_case["final_prediction"],
                    "correct": candidate_correct,
                    "base_prediction": candidate_case["base_prediction"],
                },
                "risk_labels": [],
                "audit_status": "review_required_for_action_or_fact_boundary",
            })
    score_delta: dict[str, dict[str, int]] = {}
    baseline_scores = _scores(baseline_result)
    candidate_scores = _scores(candidate_result)
    for judge_component in ("large_model_base", "conservative_correction"):
        score_delta[judge_component] = {
            bucket: (
                int(candidate_scores[judge_component][bucket]["correct"])
                - int(baseline_scores[judge_component][bucket]["correct"])
            )
            for bucket in ("overall", "group", "private")
        }
    improvements = [row for row in changes if row["effect"] == "improved_for_c0"]
    regressions = [row for row in changes if row["effect"] == "worsened_for_c0"]
    report = {
        "schema_version": 1,
        "kind": "c0_candidate_paired_judge_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "candidate": "one frozen C0 candidate rule only",
            "formal_cases": {"total": 240, "group": 160, "private": 80},
            "private_outputs_reused": True,
            "default_c0_changed": False,
            "judge_accuracy_interpretation": "a lower Judge correct count means C0 is harder to distinguish and therefore improves",
        },
        "reply_changes": {
            "total": sum(row["baseline"] != row["candidate"] for row in c0_results.values()),
            "group": sum(
                row["relationship"] == "group" and row["baseline"] != row["candidate"]
                for row in c0_results.values()
            ),
            "private": 0,
        },
        "scores": {"baseline": baseline_scores, "candidate": candidate_scores, "correct_count_delta": score_delta},
        "paired_outcomes": {
            "improved_for_c0": len(improvements),
            "worsened_for_c0": len(regressions),
            "net_improvement_for_c0": len(improvements) - len(regressions),
            "improved_case_ids": [row["public_case_id"] for row in improvements],
            "worsened_case_ids": [row["public_case_id"] for row in regressions],
        },
        "audit_notice": (
            "Judge discrimination is not a factual-safety oracle. Each changed reply and every paired "
            "outcome change is retained below for explicit review of answering for others, forced signup, "
            "invented personal facts, intervention in non-self tasks, and damage to group banter/formation."
        ),
        "changed_or_outcome_changed_cases": changes,
        "recommendation": "pending_explicit_safety_audit",
    }
    _write(OUTPUT_ROOT / "paired_report.json", report)
    return report


def _prepare(c0_results: dict[str, dict[str, Any]]) -> set[str]:
    _write_condition_packets(c0_results, "candidate")
    changed = _changed_public_ids(c0_results)
    metadata = {
        "schema_version": 1,
        "kind": "c0_candidate_judge_evaluation_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(SOURCE_ROOT),
        "frozen_baseline_root": str(FROZEN_BASELINE_ROOT),
        "cases": {"total": 240, "group": 160, "private": 80},
        "changed_public_case_count": len(changed),
        "changed_public_case_ids": sorted(changed),
        "only_variable": "one frozen C0 candidate rule",
        "default_c0_changed": False,
    }
    _write(OUTPUT_ROOT / "manifest.json", metadata)
    return changed


def main() -> None:
    global SOURCE_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="candidate C0 checkpoint directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="isolated Judge evaluation directory (defaults to SOURCE_ROOT/judge_evaluation)",
    )
    parser.add_argument("--prepare-only", action="store_true", help="build packets and stop before Judge calls")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    SOURCE_ROOT = args.source_root.resolve()
    OUTPUT_ROOT = (args.output_root or SOURCE_ROOT / "judge_evaluation").resolve()
    c0_results = _load_c0_results()
    changed = _prepare(c0_results)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "changed_public_cases": len(changed)}, ensure_ascii=False))
        return
    _ensure_candidate_reuse(changed)
    _run_condition(
        "candidate",
        workers=args.workers,
        feature_timeout_seconds=args.feature_timeout_seconds,
        only_case_ids=changed,
    )
    report = _paired_report(c0_results)
    print(json.dumps({
        "completed": True,
        "changed_public_cases": len(changed),
        "paired_outcomes": report["paired_outcomes"],
        "correct_count_delta": report["scores"]["correct_count_delta"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
