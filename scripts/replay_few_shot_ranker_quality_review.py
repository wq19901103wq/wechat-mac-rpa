#!/usr/bin/env python3
"""Replay frozen ranker arms into a private human-quality review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import _run_c0  # noqa: E402
from scripts.label_few_shot_ranker_samples import (  # noqa: E402
    DEFAULT_EXPORT_DIRS,
    FixedFewShotRetriever,
    FixedOneFewShotRetriever,
    _target_candidate,
    load_export_payloads,
)
from src.badcase.judge_config import current_judge_baseline_id, load_current_judge  # noqa: E402
from src.reply.c0_config import current_c0_baseline_id  # noqa: E402
from src.reply.c0_factory import create_current_c0_generator  # noqa: E402


DEFAULT_EVALUATION = (
    ROOT / "data/reports/private/few_shot_ranker_lr_expanded480_evaluation_v2/result.json"
)
DEFAULT_SELECTIONS = (
    ROOT / "data/reports/private/few_shot_ranker_lr_expanded480_plan/selections.jsonl"
)
DEFAULT_SAMPLES = (
    ROOT / "data/reports/private/few_shot_ranker_samples_v8_xgboost_expanded480/samples.jsonl"
)
DEFAULT_MANIFEST = ROOT / "data/reports/private/judge_feature_classifier_v1/manifest.json"
DEFAULT_OUTPUT = (
    ROOT / "data/reports/private/few_shot_ranker_lr_expanded480_quality_review_regressions_v1"
)
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
SCOPES = ("regressed", "improved", "all_changed")
C0Runner = Callable[..., dict[str, Any]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_private(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, PRIVATE_MODE)


def _append_private(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, PRIVATE_MODE)


def _selected_rows(
    evaluation: Mapping[str, Any],
    selections: list[dict[str, Any]],
    *,
    scope: str,
    selection_count: int,
) -> list[dict[str, Any]]:
    if scope == "regressed":
        selected_ids = set((evaluation.get("paired_outcomes") or {}).get("regressed_target_ids") or [])
    elif scope == "improved":
        selected_ids = set((evaluation.get("paired_outcomes") or {}).get("improved_target_ids") or [])
    elif scope == "all_changed":
        same_key = f"same_top{selection_count}"
        selected_ids = {
            str(row.get("target_id") or "")
            for row in selections
            if row.get(same_key) is False
        }
    else:
        raise ValueError("unsupported review scope")
    result = [row for row in selections if str(row.get("target_id") or "") in selected_ids]
    if len(result) != len(selected_ids) or not result:
        raise ValueError("review scope does not match the frozen selection plan")
    return sorted(result, key=lambda row: str(row["target_id"]))


def _latest_records(path: Path) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    if not path.exists():
        return latest
    for row in _read_jsonl(path):
        entry_id = str(row.get("entry_id") or "")
        if not entry_id:
            raise ValueError("quality replay checkpoint has an entry without an id")
        previous = latest.get(entry_id)
        if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
            latest[entry_id] = row
    return latest


def _review_results(
    selections: list[dict[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
    *,
    candidate_variant: str,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for selection in selections:
        target_id = str(selection["target_id"])
        production = latest.get(f"{target_id}:production")
        candidate = latest.get(f"{target_id}:{candidate_variant}")
        if not production or production.get("status") != "ok":
            errors.append({"target_id": target_id, "condition": "production"})
        if not candidate or candidate.get("status") != "ok":
            errors.append({"target_id": target_id, "condition": candidate_variant})
        if (
            not production
            or production.get("status") != "ok"
            or not candidate
            or candidate.get("status") != "ok"
        ):
            continue
        if (
            production.get("target_context") != candidate.get("target_context")
            or production.get("human_reply") != candidate.get("human_reply")
        ):
            raise ValueError("paired quality replays do not share the same target input")
        cases.append({
            "target_id": target_id,
            "relationship": str(selection["relationship"]),
            "target_context": production["target_context"],
            "human_reply": production["human_reply"],
            "production": {
                "generated_reply": production["generated_reply"],
                "candidate_ids": production["candidate_ids"],
                "bound_few_shots": production["bound_few_shots"],
                "generation_failed": production["generation_failed"],
            },
            candidate_variant: {
                "generated_reply": candidate["generated_reply"],
                "candidate_ids": candidate["candidate_ids"],
                "bound_few_shots": candidate["bound_few_shots"],
                "generation_failed": candidate["generation_failed"],
            },
        })
    return {
        "schema_version": 1,
        "kind": "few_shot_ranker_independent_human_quality_replay",
        "candidate_variant": candidate_variant,
        "summary": {
            "selected_targets": len(selections),
            "complete_pairs": len(cases),
            "failed_arms": len(errors),
            "relationship_targets": dict(sorted(Counter(
                str(row["relationship"]) for row in cases
            ).items())),
        },
        "errors": errors,
        "cases": cases,
    }


def replay(
    *,
    evaluation_path: Path = DEFAULT_EVALUATION,
    selections_path: Path = DEFAULT_SELECTIONS,
    samples_path: Path = DEFAULT_SAMPLES,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
    export_dirs: tuple[Path, ...] = DEFAULT_EXPORT_DIRS,
    scope: str = "regressed",
    candidate_variant: str = "lr",
    shard_index: int | None = None,
    shard_count: int | None = None,
    resume: bool = False,
    retry_errors: bool = False,
    c0_runner: C0Runner = _run_c0,
    generator: Any | None = None,
    payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard index and count must be supplied together")
    if shard_count is not None and (shard_count < 1 or shard_index is None or not 0 <= shard_index < shard_count):
        raise ValueError("shard values must satisfy 0 <= shard index < shard count")
    if retry_errors and not resume:
        raise ValueError("retry-errors requires resume")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("candidate_variant") != candidate_variant:
        raise ValueError("evaluation and quality candidate variants differ")
    selection_count = int((evaluation.get("current_defaults") or {}).get("few_shot_count") or 1)
    if selection_count < 1:
        raise ValueError("evaluation has an invalid few-shot count")
    c0_id = current_c0_baseline_id()
    judge_id = current_judge_baseline_id()
    judge_config = load_current_judge(require_runnable=True)
    if str(judge_config.get("required_current_c0_baseline_id") or "") != c0_id:
        raise ValueError("current Judge is not bound to the current C0")
    frozen_defaults = evaluation.get("current_defaults") or {}
    if (
        frozen_defaults.get("c0_baseline_id") != c0_id
        or frozen_defaults.get("judge_baseline_id") != judge_id
        or int(frozen_defaults.get("few_shot_count") or 1) != selection_count
    ):
        raise ValueError("quality replay defaults differ from the frozen evaluation")
    if str((evaluation.get("sources") or {}).get("selections", {}).get("sha256") or "") != _sha256_file(
        selections_path
    ):
        raise ValueError("quality replay selection plan differs from the evaluation")

    selections = _selected_rows(
        evaluation,
        _read_jsonl(selections_path),
        scope=scope,
        selection_count=selection_count,
    )
    if shard_count is not None and shard_index is not None:
        selections = [
            row for position, row in enumerate(selections)
            if position % shard_count == shard_index
        ]
    source_rows = _read_jsonl(samples_path)
    sources = {str(row.get("sample_id") or ""): row for row in source_rows}
    if len(sources) != len(source_rows) or "" in sources:
        raise ValueError("expanded sample source has duplicate or missing ids")
    judge_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(judge_manifest.get("splits"), dict) and judge_manifest["splits"]:
        manifest_rows = [
            dict(row) for rows in judge_manifest["splits"].values() for row in rows
        ]
    elif isinstance(judge_manifest.get("cases"), list) and judge_manifest["cases"]:
        manifest_rows = []
        for row in judge_manifest["cases"]:
            normalized = dict(row)
            normalized["case_id"] = str(row.get("case_id") or row.get("id") or "")
            normalized["relationship"] = (
                "private" if row.get("relationship") == "acquaintance" else row.get("relationship")
            )
            normalized["replies"] = row.get("replies") or row.get("reply") or []
            manifest_rows.append(normalized)
    else:
        raise ValueError("quality replay manifest must contain non-empty splits or cases")
    manifest_by_case_id = {str(row["case_id"]): row for row in manifest_rows}
    if payloads is None:
        payloads, export_hashes = load_export_payloads(export_dirs)
    else:
        payloads = {str(key): dict(value) for key, value in payloads.items()}
        export_hashes = {}

    output_dir.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    os.chmod(output_dir, PRIVATE_DIR_MODE)
    checkpoint_path = output_dir / "generation.jsonl"
    if checkpoint_path.exists() and not resume:
        raise ValueError("quality replay output already exists; use --resume")
    latest = _latest_records(checkpoint_path)
    active_generator = generator
    written = Counter()
    for selection in selections:
        target_id = str(selection["target_id"])
        for condition in ("production", candidate_variant):
            entry_id = f"{target_id}:{condition}"
            previous = latest.get(entry_id)
            if previous and (
                previous.get("status") == "ok"
                or (previous.get("status") == "error" and not retry_errors)
            ):
                continue
            if selection_count == 1:
                sample_ids = [str(selection[f"{condition}_sample_id"])]
                candidate_ids = []
            else:
                sample_ids = [str(value) for value in selection[f"{condition}_sample_ids"]]
                candidate_ids = [str(value) for value in selection[f"{condition}_candidate_ids"]]
            if (
                len(sample_ids) != selection_count
                or (candidate_ids and len(candidate_ids) != selection_count)
                or any(not value for value in sample_ids + candidate_ids)
            ):
                raise ValueError("selection has an incomplete frozen few-shot arm")
            source_arm = [sources.get(sample_id) for sample_id in sample_ids]
            if any(source is None for source in source_arm):
                raise ValueError("selection references a missing expanded sample")
            arm_rows = [source for source in source_arm if source is not None]
            target = arm_rows[0]["target"]
            candidates = [source["candidate"] for source in arm_rows]
            source_candidate_ids = [str(candidate.get("id") or "") for candidate in candidates]
            if not candidate_ids:
                candidate_ids = source_candidate_ids
            if (
                any(source["target"] != target for source in arm_rows)
                or str(target.get("case_id") or "") != target_id
                or source_candidate_ids != candidate_ids
            ):
                raise ValueError("selection and expanded sample frozen arm differ")
            manifest_case = manifest_by_case_id.get(target_id)
            payload = payloads.get(str(target.get("chat_id") or ""))
            if manifest_case is None or payload is None:
                raise ValueError("quality replay target source is unavailable")
            if active_generator is None:
                active_generator = create_current_c0_generator(enable_persona_few_shots=True)
            try:
                output = c0_runner(
                    _target_candidate(target, manifest_case),
                    dict(payload),
                    persona_few_shot_retriever=(
                        FixedOneFewShotRetriever(candidates[0])
                        if selection_count == 1 else FixedFewShotRetriever(candidates)
                    ),
                    persona_few_shot_count=selection_count,
                    compact_verified_turn=True,
                    reply_generator=active_generator,
                )
                if output.get("few_shot_ids") != candidate_ids:
                    raise RuntimeError("quality replay did not retain the bound few-shot")
                record = {
                    "entry_id": entry_id,
                    "target_id": target_id,
                    "relationship": str(target["relationship"]),
                    "condition": condition,
                    "candidate_ids": candidate_ids,
                    "target_context": target.get("context_messages") or target.get("context") or [],
                    "human_reply": manifest_case.get("replies") or [],
                    "generated_reply": output.get("replies") or [],
                    "generation_failed": bool(output.get("generation_failed")),
                    "bound_few_shots": [
                        {
                            "relationship": str(candidate.get("relationship") or ""),
                            "context": candidate.get("context") or [],
                            "reply": candidate.get("reply") or [],
                        }
                        for candidate in candidates
                    ],
                    "duration_seconds": output.get("duration_seconds"),
                    "status": "ok",
                }
            except Exception as exc:
                record = {
                    "entry_id": entry_id,
                    "target_id": target_id,
                    "condition": condition,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            _append_private(checkpoint_path, record)
            latest[entry_id] = record
            written[str(record["status"])] += 1

    result = _review_results(selections, latest, candidate_variant=candidate_variant)
    result.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "shard": {"index": shard_index, "count": shard_count},
        "protocol": {
            "independent_quality_replay": True,
            "reuses_metric_outcomes": False,
            "bound_few_shot_count_per_arm": selection_count,
            "static_few_shots": 0,
            "current_c0_baseline_id": c0_id,
            "current_judge_baseline_id": judge_id,
            "judge_calls": 0,
        },
        "sources": {
            "evaluation": {"path": _recorded_path(evaluation_path), "sha256": _sha256_file(evaluation_path)},
            "selections": {"path": _recorded_path(selections_path), "sha256": _sha256_file(selections_path)},
            "samples": {"path": _recorded_path(samples_path), "sha256": _sha256_file(samples_path)},
            "judge_manifest": {"path": _recorded_path(manifest_path), "sha256": _sha256_file(manifest_path)},
            "export_payload_sha256": dict(sorted(export_hashes.items())),
        },
        "written_this_run": dict(written),
    })
    _write_private(output_dir / "results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exports", type=Path, action="append", dest="export_dirs")
    parser.add_argument("--scope", choices=SCOPES, default="regressed")
    parser.add_argument("--candidate-variant", default="lr")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    result = replay(
        evaluation_path=args.evaluation,
        selections_path=args.selections,
        samples_path=args.samples,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        export_dirs=tuple(args.export_dirs) if args.export_dirs else DEFAULT_EXPORT_DIRS,
        scope=args.scope,
        candidate_variant=args.candidate_variant,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        resume=args.resume,
        retry_errors=args.retry_errors,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
