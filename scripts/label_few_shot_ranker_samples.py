#!/usr/bin/env python3
"""Generate private, leak-safe labels for single-few-shot ranker samples.

Each row replays C0 with exactly the one candidate bound to that row.  The
frozen formal Judge then compares that reply (A) with the target's historical
human reply (B).  A row is positive when the formal Judge gives A a probability
of at least 0.5 of being the human reply.

The JSONL output is deliberately an audit index, not a conversation archive:
it contains identifiers, labels and hashes only.  Chat text, generated replies,
few-shots, prompts and traces must remain in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_persona_few_shots import Candidate  # noqa: E402
from scripts.evaluate_human_ai_judge import extract_judge_chat_context, _run_c0  # noqa: E402
from scripts.run_judge_feature_classifier_experiment import (  # noqa: E402
    GROUP_PATTERN_RECENT_MESSAGE_LIMIT,
    RECENT_SPEAKER_POSITIONS,
    expand_boolean_option_with_refined_context,
    feature_extractor_prompt,
    group_other_text_timeline,
    message_text_timeline,
    run_codex_feature_extractor,
    speaker_timeline,
    validate_option_features,
    vectorize_boolean_options,
    vectorize_context_booleans,
    vectorize_group_pattern_option_boolean,
    vectorize_observable_option_booleans,
)
from src.badcase.judge_config import current_judge_baseline_id, load_current_judge  # noqa: E402
from src.reply.c0_config import DEFAULT_CONFIG_PATH as DEFAULT_C0_CONFIG_PATH  # noqa: E402
from src.reply.c0_config import current_c0_baseline_id  # noqa: E402
from src.reply.c0_factory import create_current_c0_generator  # noqa: E402
from src.reply.few_shot import PersonaFewShotRetriever  # noqa: E402


DEFAULT_SAMPLES = ROOT / "data/reports/private/few_shot_ranker_samples_v2/samples.jsonl"
DEFAULT_MANIFEST = ROOT / "data/reports/private/judge_feature_classifier_v1/manifest.json"
DEFAULT_JUDGE_CONFIG = ROOT / "config/current_judge.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/few_shot_ranker_labels_v2"
DEFAULT_EXPORT_DIRS = (ROOT / "data/exports/b",)
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _safe_error_type(exc: BaseException) -> str:
    """Never put exception text in a durable record: it can contain chat text."""
    if isinstance(exc, FeatureExtractionError):
        cause = exc.__cause__
        if isinstance(cause, TimeoutError):
            return "FeatureExtractionTimeout"
        if isinstance(cause, RuntimeError):
            return "FeatureExtractionRuntimeError"
        if isinstance(cause, ValueError):
            return "FeatureExtractionFormatError"
    return type(exc).__name__


@dataclass(frozen=True)
class FormalJudge:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    context_schema: dict[str, tuple[Any, ...]]
    schema_hash: str
    artifact_sha256: str


@dataclass(frozen=True)
class JudgeRun:
    """Resolved formal-Judge inputs recorded with every durable labeling run."""

    config_path: Path
    config_sha256: str
    baseline_id: str
    artifact_path: Path
    model: str
    reasoning_effort: str
    timeout_seconds: int
    is_experimental_override: bool
    required_current_c0_baseline_id: str


class C0GenerationError(RuntimeError):
    """Safe category for a failed C0 replay; original details remain in memory."""


class FewShotBindingError(RuntimeError):
    """Safe category when the replay does not retain its bound demonstration."""


class C0ResponseShapeError(RuntimeError):
    """Safe category for an incomplete replay result before Judge scoring."""


class FeatureExtractionError(RuntimeError):
    """Safe category for a failed Judge feature extraction."""


class JudgeContextError(RuntimeError):
    """Safe category for context recovery before Judge feature extraction."""


class JudgeFeaturePromptError(RuntimeError):
    """Safe category for formal Judge feature-request construction."""


class JudgeMetadataError(RuntimeError):
    """Safe category for formal Judge observable-context construction."""


class JudgeScoringError(RuntimeError):
    """Safe category for frozen formal Judge scoring."""


class FixedFewShotRetriever:
    """Retriever used only for label collection; it cannot silently add examples."""

    def __init__(self, candidates: Iterable[Mapping[str, Any]]):
        self._candidates = [dict(candidate) for candidate in candidates]
        self._ids = [str(candidate.get("id") or "") for candidate in self._candidates]
        if not self._candidates or any(not candidate_id for candidate_id in self._ids):
            raise ValueError("bound few-shot candidate has no id")
        if len(self._ids) != len(set(self._ids)):
            raise ValueError("bound few-shot candidate ids are not unique")

    def is_approved(self) -> bool:
        return True

    def retrieve(self, *, limit: int = 8, **_: Any) -> list[dict[str, Any]]:
        if limit != len(self._candidates):
            raise ValueError(f"bound few-shot labeling requires limit={len(self._candidates)}")
        return [dict(candidate) for candidate in self._candidates]

    def render_selected(
        self, rows: list[dict[str, Any]], max_chars: int = 2500
    ) -> tuple[str, list[str]]:
        if [str(row.get("id") or "") for row in rows] != self._ids:
            raise ValueError("few-shot labeling received an unbound candidate")
        return PersonaFewShotRetriever.render(rows, max_chars=max_chars)


class FixedOneFewShotRetriever(FixedFewShotRetriever):
    def __init__(self, candidate: Mapping[str, Any]):
        super().__init__([candidate])


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_judge_run(
    *,
    judge_config_path: Path = DEFAULT_JUDGE_CONFIG,
    judge_path: Path | None = None,
    feature_timeout_seconds: int | None = None,
    feature_model: str | None = None,
    feature_reasoning_effort: str | None = None,
    allow_experimental_judge_override: bool = False,
) -> JudgeRun:
    """Resolve the adopted Judge by default; permit drift only with an explicit opt-in."""
    config_path = judge_config_path.resolve()
    config = load_current_judge(config_path, require_runnable=True)
    execution = dict(config["execution"])
    artifact = dict((config["correction"] or {})["artifact"])
    configured_artifact_path = _resolve_project_path(str(artifact["path"])).resolve()
    overrides_requested = any(
        value is not None
        for value in (judge_path, feature_timeout_seconds, feature_model, feature_reasoning_effort)
    )
    if overrides_requested and not allow_experimental_judge_override:
        raise ValueError(
            "Judge overrides require allow_experimental_judge_override=True; "
            "the default must use config/current_judge.json"
        )
    resolved_artifact_path = (judge_path or configured_artifact_path).resolve()
    timeout_seconds = (
        feature_timeout_seconds
        if feature_timeout_seconds is not None
        else execution["timeout_seconds"]
    )
    model = feature_model if feature_model is not None else execution["model"]
    reasoning_effort = (
        feature_reasoning_effort
        if feature_reasoning_effort is not None
        else execution["reasoning_effort"]
    )
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("feature timeout must be a positive integer")
    if not isinstance(model, str) or not model:
        raise ValueError("feature model must be a non-empty string")
    if not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise ValueError("feature reasoning effort must be a non-empty string")
    return JudgeRun(
        config_path=config_path,
        config_sha256=_sha256_bytes(config_path.read_bytes()),
        baseline_id=current_judge_baseline_id(config_path),
        artifact_path=resolved_artifact_path,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        is_experimental_override=(
            resolved_artifact_path != configured_artifact_path
            or timeout_seconds != execution["timeout_seconds"]
            or model != execution["model"]
            or reasoning_effort != execution["reasoning_effort"]
        ),
        required_current_c0_baseline_id=str(
            config.get("required_current_c0_baseline_id") or ""
        ),
    )


def load_formal_judge(path: Path) -> FormalJudge:
    """Load the frozen pairwise Judge and fail closed on any schema drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("final_model") or {}
    feature_system = payload.get("feature_system") or {}
    names = tuple(str(value) for value in model.get("feature_names") or ())
    coefficients = np.asarray(model.get("coefficients") or (), dtype=float)
    raw_schema = feature_system.get("context_schema")
    if (
        payload.get("selected_model_name") != "symmetric_logistic_refined_boolean_crosses"
        or model.get("model") != "LogisticRegression"
        or model.get("fit_intercept") is not False
        or not names
        or len(names) != len(coefficients)
        or int(model.get("input_feature_count") or -1) != len(names)
        or not isinstance(raw_schema, dict)
    ):
        raise ValueError("formal Judge artifact is not the frozen pairwise boolean model")
    context_schema = {
        str(key): tuple(value) for key, value in raw_schema.items()
        if isinstance(value, list)
    }
    if set(context_schema) != {"known_group_names", "known_group_members"}:
        raise ValueError("formal Judge context schema is incomplete")
    schema_payload = {
        "feature_names": names,
        "context_schema": context_schema,
        "model": model.get("model"),
        "fit_intercept": model.get("fit_intercept"),
    }
    return FormalJudge(
        feature_names=names,
        coefficients=coefficients,
        context_schema=context_schema,
        schema_hash=_json_sha256(schema_payload),
        artifact_sha256=_sha256_bytes(path.read_bytes()),
    )


def probability_from_vectors(diff: np.ndarray, coefficients: np.ndarray) -> float:
    if diff.ndim != 1 or coefficients.ndim != 1 or len(diff) != len(coefficients):
        raise ValueError("Judge vector and coefficient dimensions differ")
    logit = float(np.dot(diff, coefficients))
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def context_metadata(
    target: Mapping[str, Any],
    output_metadata: Mapping[str, Any],
    schema: Mapping[str, tuple[Any, ...]],
    context_original: list[str],
) -> dict[str, Any]:
    """Match the frozen Judge's metadata construction from the exact XML context."""
    message_texts = message_text_timeline(context_original)
    if not message_texts:
        raise ValueError("C0 returned no textual context for Judge metadata")
    latest = message_texts[-1]
    return {
        "relationship": str(target.get("relationship") or ""),
        "source_chat_id": str(target.get("chat_id") or "__unknown_chat__"),
        "group_name": str(output_metadata.get("chat_name") or "__unknown_group_name__"),
        "recent_speakers": tuple(reversed(speaker_timeline(context_original)))[:RECENT_SPEAKER_POSITIONS],
        "latest_message_has_question_mark": any(marker in latest for marker in ("?", "？")),
        "latest_message_has_at_mention": "@" in latest,
        "recent_group_other_texts": group_other_text_timeline(context_original)[-GROUP_PATTERN_RECENT_MESSAGE_LIMIT:],
        # Passing schema through makes the construction's frozen dependency explicit.
        "_frozen_schema_keys": tuple(sorted(schema)),
    }


def formal_judge_context(output: Mapping[str, Any]) -> list[str]:
    """Reuse the exact neutral XML context that the formal Judge evaluates.

    `_run_c0()` keeps raw message bodies in ``context_original`` for replay
    compatibility.  Those bodies are not XML fragments and therefore cannot be
    passed to the frozen Judge's context-vector functions.  The actual C0 user
    prompt already contains escaped, role- and type-labelled chat messages; use
    the existing neutral extractor so feature extraction and metadata see the
    same input as the formal Judge baseline.
    """
    trace = output.get("trace")
    prompt = trace.get("user_prompt") if isinstance(trace, Mapping) else None
    if not isinstance(prompt, str):
        raise ValueError("C0 replay did not retain a Judge-compatible chat context")
    context = extract_judge_chat_context(prompt)
    if not context:
        raise ValueError("C0 replay has no parseable Judge-compatible chat context")
    return context


def score_formal_judge_pair(
    judge: FormalJudge,
    llm_features: Mapping[str, Any],
    reply_a: list[str],
    reply_b: list[str],
    metadata: Mapping[str, Any],
) -> float:
    """Return P(A is the more human-like reply), with strict A/B symmetry."""
    features = {
        "option_A": validate_option_features(llm_features.get("option_A")),
        "option_B": validate_option_features(llm_features.get("option_B")),
    }
    option_a, option_b, names = vectorize_boolean_options(features)
    observed_a, observed_names = vectorize_observable_option_booleans(reply_a, dict(metadata))
    observed_b, other_observed_names = vectorize_observable_option_booleans(reply_b, dict(metadata))
    group_a, group_names = vectorize_group_pattern_option_boolean(reply_a, dict(metadata))
    group_b, other_group_names = vectorize_group_pattern_option_boolean(reply_b, dict(metadata))
    if observed_names != other_observed_names or group_names != other_group_names:
        raise ValueError("A/B observable feature schemas differ")
    option_names = [*names, *observed_names, *group_names]
    vector_a = np.concatenate((option_a, observed_a, group_a))
    vector_b = np.concatenate((option_b, observed_b, group_b))
    context, context_names = vectorize_context_booleans(dict(metadata), judge.context_schema)
    expanded_a, expanded_names = expand_boolean_option_with_refined_context(
        vector_a, option_names, context, context_names
    )
    expanded_b, swapped_names = expand_boolean_option_with_refined_context(
        vector_b, option_names, context, context_names
    )
    if expanded_names != swapped_names or tuple(expanded_names) != judge.feature_names:
        raise ValueError("formal Judge artifact and current feature schema differ")
    probability = probability_from_vectors(expanded_a - expanded_b, judge.coefficients)
    swapped_probability = probability_from_vectors(expanded_b - expanded_a, judge.coefficients)
    if not math.isclose(probability + swapped_probability, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("formal Judge A/B probabilities are not complementary")
    return probability


def _target_candidate(target: Mapping[str, Any], manifest_case: Mapping[str, Any]) -> Candidate:
    relation = "group" if str(target.get("relationship")) == "group" else "acquaintance"
    return Candidate(
        relation=relation,
        chat_id=str(target["chat_id"]),
        context=[str(item) for item in target.get("context") or ()],
        replies=[str(item) for item in manifest_case.get("replies") or ()],
        score=float(manifest_case.get("score") or 0),
        timestamp=int(target["timestamp"]),
        context_messages=[dict(item) for item in target.get("context_messages") or ()],
        reply_messages=[dict(item) for item in manifest_case.get("reply_messages") or ()],
    )


def _validate_sample(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    generation = sample.get("generation") or {}
    if isinstance(sample.get("candidate"), Mapping):
        candidates = [dict(sample["candidate"])]
        injected_ids = [str(generation.get("injected_candidate_id") or "")]
    else:
        raw_candidates = sample.get("candidates")
        if not isinstance(raw_candidates, list) or not all(
            isinstance(candidate, Mapping) for candidate in raw_candidates
        ):
            raise ValueError("sample has no bound few-shot candidates")
        candidates = [dict(candidate) for candidate in raw_candidates]
        raw_ids = generation.get("injected_candidate_ids")
        injected_ids = [str(value) for value in raw_ids] if isinstance(raw_ids, list) else []
    candidate_ids = [str(candidate.get("id") or "") for candidate in candidates]
    if (
        not candidate_ids
        or any(not candidate_id for candidate_id in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or generation.get("few_shot_count") != len(candidates)
        or generation.get("static_few_shot_count") != 0
        or injected_ids != candidate_ids
    ):
        raise ValueError("sample is not a bound few-shot sample")
    return candidates


FeatureRunner = Callable[[str, int, str | None, str | None], tuple[dict[str, Any], dict[str, Any]]]
C0Runner = Callable[..., dict[str, Any]]


def label_one(
    sample: Mapping[str, Any],
    *,
    manifest_by_case_id: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    judge: FormalJudge,
    feature_runner: FeatureRunner = run_codex_feature_extractor,
    c0_runner: C0Runner = _run_c0,
    c0_generator: Any | None = None,
    feature_timeout_seconds: int = 60,
    feature_model: str | None = None,
    feature_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Generate and score one row.  The returned dict contains no conversation text."""
    candidate_rows = _validate_sample(sample)
    sample_id = str(sample["sample_id"])
    target = sample["target"]
    case_id = str(target["case_id"])
    manifest_case = manifest_by_case_id.get(case_id)
    if manifest_case is None:
        raise ValueError("target case is absent from the Judge manifest")
    if str(manifest_case.get("chat_id") or "") != str(target.get("chat_id") or ""):
        raise ValueError("target case and Judge manifest chat ids differ")
    chat_id = str(target["chat_id"])
    payload = payloads.get(chat_id)
    if payload is None:
        raise ValueError("target export payload is unavailable")
    target_candidate = _target_candidate(target, manifest_case)
    retriever = FixedFewShotRetriever(candidate_rows)
    candidate_ids = [str(candidate["id"]) for candidate in candidate_rows]
    started = time.monotonic()
    generator = c0_generator or create_current_c0_generator(enable_persona_few_shots=True)
    try:
        output = c0_runner(
            target_candidate,
            dict(payload),
            persona_few_shot_retriever=retriever,
            persona_few_shot_count=len(candidate_rows),
            compact_verified_turn=True,
            reply_generator=generator,
        )
    except Exception as exc:
        raise C0GenerationError("C0 replay failed") from exc
    injected = output.get("few_shot_ids")
    if injected != candidate_ids:
        raise FewShotBindingError("C0 did not use exactly the bound few-shots")
    reply_a = output.get("replies")
    reply_b = manifest_case.get("replies")
    context_original = output.get("context_original")
    if (
        not isinstance(reply_a, list) or not all(isinstance(value, str) for value in reply_a)
        or not isinstance(reply_b, list) or not all(isinstance(value, str) for value in reply_b)
        or not isinstance(context_original, list) or not all(isinstance(value, str) for value in context_original)
    ):
        raise C0ResponseShapeError("C0 or manifest reply has an invalid shape")
    try:
        judge_context = formal_judge_context(output)
    except Exception as exc:
        raise JudgeContextError("Judge context recovery failed") from exc
    feature_case = {
        "relationship": str(target["relationship"]),
        "context_original": judge_context,
        "option_A": reply_a,
        "option_B": reply_b,
    }
    try:
        prompt = feature_extractor_prompt(feature_case)
    except Exception as exc:
        raise JudgeFeaturePromptError("Judge feature prompt construction failed") from exc
    try:
        llm_features, _audit = feature_runner(
            prompt, feature_timeout_seconds, feature_model, feature_reasoning_effort
        )
    except Exception as exc:
        raise FeatureExtractionError("Judge feature extraction failed") from exc
    try:
        metadata = context_metadata(
            target,
            output.get("input_metadata") or {},
            judge.context_schema,
            judge_context,
        )
    except Exception as exc:
        raise JudgeMetadataError("Judge metadata construction failed") from exc
    try:
        probability = score_formal_judge_pair(judge, llm_features, reply_a, reply_b, metadata)
    except Exception as exc:
        raise JudgeScoringError("formal Judge scoring failed") from exc
    # Do not retain `output`, `prompt`, `feature_case`, trace, or feature audit beyond this scope.
    result = {
        "sample_id": sample_id,
        "relationship": str(target["relationship"]),
        "label": int(probability >= 0.5),
        "probability": round(probability, 12),
        "schema_hash": judge.schema_hash,
        "duration_seconds": round(time.monotonic() - started, 3),
        "status": "ok",
    }
    if len(candidate_ids) == 1:
        result["injected_candidate_id"] = candidate_ids[0]
    else:
        result["injected_candidate_ids"] = candidate_ids
    return result


def append_jsonl_durable(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, PRIVATE_MODE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_no} is not an object")
        records.append(row)
    return records


def load_export_payloads(export_dirs: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load both export roots; duplicate hashed chats must be byte-equivalent payloads."""
    payloads: dict[str, dict[str, Any]] = {}
    payload_hashes: dict[str, str] = {}
    for export_dir in export_dirs:
        for path in sorted(export_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            normalized = path.stem
            for prefix in ("私聊_", "群聊_", "曾经的好友_"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            chat_id = "chat_" + _sha256_bytes(normalized.strip().encode("utf-8"))[:10]
            canonical = _json_sha256(payload)
            previous = payload_hashes.get(chat_id)
            if previous is not None and previous != canonical:
                raise ValueError("duplicate export chat has conflicting payloads")
            payloads[chat_id] = payload
            payload_hashes[chat_id] = canonical
    return payloads, payload_hashes


def _existing_status(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in records:
        sample_id = str(row.get("sample_id") or "")
        status = str(row.get("status") or "")
        if sample_id and status in {"ok", "error"}:
            if status == "ok" or sample_id not in result:
                result[sample_id] = status
    return result


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, PRIVATE_MODE)


def label(
    *,
    samples_path: Path = DEFAULT_SAMPLES,
    manifest_path: Path = DEFAULT_MANIFEST,
    judge_config_path: Path = DEFAULT_JUDGE_CONFIG,
    judge_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    export_dirs: Iterable[Path] = DEFAULT_EXPORT_DIRS,
    resume: bool = False,
    retry_errors: bool = False,
    limit: int | None = None,
    all_rows: bool = False,
    target_case_ids: Iterable[str] = (),
    shard_index: int | None = None,
    shard_count: int | None = None,
    feature_runner: FeatureRunner = run_codex_feature_extractor,
    c0_runner: C0Runner = _run_c0,
    c0_generator: Any | None = None,
    payloads: Mapping[str, Mapping[str, Any]] | None = None,
    feature_timeout_seconds: int | None = None,
    feature_model: str | None = None,
    feature_reasoning_effort: str | None = None,
    allow_experimental_judge_override: bool = False,
) -> dict[str, Any]:
    if (limit is None) == (not all_rows):
        raise ValueError("choose exactly one of limit or all_rows")
    if retry_errors and not resume:
        raise ValueError("retry_errors requires resume")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be supplied together")
    if shard_count is not None and (shard_count < 1 or shard_index is None or not 0 <= shard_index < shard_count):
        raise ValueError("shard values must satisfy 0 <= shard_index < shard_count")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, PRIVATE_DIR_MODE)
    labels_path = output_dir / "labels.jsonl"
    run_manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("output directory is non-empty; use resume")
    existing = _existing_status(_load_jsonl(labels_path)) if labels_path.exists() else {}
    selected_case_ids = {str(value) for value in target_case_ids if str(value)}
    samples = _load_jsonl(samples_path)
    if selected_case_ids:
        samples = [
            sample for sample in samples
            if str((sample.get("target") or {}).get("case_id") or "") in selected_case_ids
        ]
        if not samples:
            raise ValueError("no samples match the requested target case ids")
    if shard_count is not None and shard_index is not None:
        # Partition by target, never by individual candidate: variants for the
        # same target must remain together for a later paired comparison.
        target_ids = sorted({str((sample.get("target") or {}).get("case_id") or "") for sample in samples})
        if "" in target_ids:
            raise ValueError("samples contain a target without a case id")
        shard_target_ids = {
            target_id for position, target_id in enumerate(target_ids)
            if position % shard_count == shard_index
        }
        samples = [
            sample for sample in samples
            if str((sample.get("target") or {}).get("case_id") or "") in shard_target_ids
        ]
    few_shot_counts = {
        int((sample.get("generation") or {}).get("few_shot_count") or 0)
        for sample in samples
    }
    if len(few_shot_counts) != 1 or next(iter(few_shot_counts)) < 1:
        raise ValueError("label input must use one positive bound few-shot count")
    bound_few_shot_count = next(iter(few_shot_counts))
    effective_target_case_count = len({
        str((sample.get("target") or {}).get("case_id") or "") for sample in samples
    })
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    cases = manifest.get("cases")
    if isinstance(splits, Mapping) and splits:
        manifest_rows = [dict(row) for rows in splits.values() for row in rows]
    elif isinstance(cases, list) and cases:
        manifest_rows = []
        for row in cases:
            normalized = dict(row)
            normalized["case_id"] = str(row.get("case_id") or row.get("id") or "")
            relationship = str(row.get("relationship") or "")
            normalized["relationship"] = (
                "private" if relationship == "acquaintance" else relationship
            )
            normalized["replies"] = row.get("replies") or row.get("reply") or []
            manifest_rows.append(normalized)
    else:
        raise ValueError("Judge manifest must contain a non-empty splits mapping or cases list")
    if not manifest_rows:
        raise ValueError("Judge manifest contains no cases")
    manifest_by_case_id = {str(row.get("case_id")): row for row in manifest_rows}
    judge_run = resolve_judge_run(
        judge_config_path=judge_config_path,
        judge_path=judge_path,
        feature_timeout_seconds=feature_timeout_seconds,
        feature_model=feature_model,
        feature_reasoning_effort=feature_reasoning_effort,
        allow_experimental_judge_override=allow_experimental_judge_override,
    )
    c0_config_path = DEFAULT_C0_CONFIG_PATH.resolve()
    c0_baseline_id = current_c0_baseline_id(c0_config_path)
    if judge_run.required_current_c0_baseline_id != c0_baseline_id:
        raise ValueError("current Judge is not bound to the current C0")
    judge = load_formal_judge(judge_run.artifact_path)
    if payloads is None:
        payloads, export_hashes = load_export_payloads(tuple(export_dirs))
    else:
        payloads = {str(key): dict(value) for key, value in payloads.items()}
        export_hashes = {str(key): _json_sha256(value) for key, value in payloads.items()}
    pending = []
    for sample in samples:
        previous = existing.get(str(sample.get("sample_id") or ""))
        if previous == "ok" or (previous == "error" and not retry_errors):
            continue
        pending.append(sample)
    if limit is not None:
        pending = pending[:limit]
    created = Counter()
    generator = c0_generator
    for sample in pending:
        if generator is None:
            generator = create_current_c0_generator(enable_persona_few_shots=True)
        try:
            row = label_one(
                sample,
                manifest_by_case_id=manifest_by_case_id,
                payloads=payloads,
                judge=judge,
                feature_runner=feature_runner,
                c0_runner=c0_runner,
                c0_generator=generator,
                feature_timeout_seconds=judge_run.timeout_seconds,
                feature_model=judge_run.model,
                feature_reasoning_effort=judge_run.reasoning_effort,
            )
        except Exception as exc:  # durable error records must not reveal exception text
            row = {
                "sample_id": str(sample.get("sample_id") or "__unknown_sample__"),
                "status": "error",
                "error_type": _safe_error_type(exc),
            }
        append_jsonl_durable(labels_path, row)
        created[str(row["status"])] += 1
    final_records = _load_jsonl(labels_path) if labels_path.exists() else []
    latest_records_by_sample: dict[str, Mapping[str, Any]] = {}
    for row in final_records:
        sample_id = str(row.get("sample_id") or "")
        previous = latest_records_by_sample.get(sample_id)
        if sample_id and (
            previous is None
            or row.get("status") == "ok"
            or previous.get("status") != "ok"
        ):
            latest_records_by_sample[sample_id] = row
    safe_manifest = {
        "schema_version": 1,
        "kind": (
            "single_few_shot_ranker_labels"
            if bound_few_shot_count == 1
            else f"top{bound_few_shot_count}_few_shot_ranker_labels"
        ),
        "samples_path": _recorded_path(samples_path),
        "samples_sha256": _sha256_bytes(samples_path.read_bytes()),
        "judge_config": {
            "path": _recorded_path(judge_run.config_path),
            "sha256": judge_run.config_sha256,
            "baseline_id": judge_run.baseline_id,
        },
        "c0_config": {
            "path": _recorded_path(c0_config_path),
            "sha256": _sha256_bytes(c0_config_path.read_bytes()),
            "baseline_id": c0_baseline_id,
        },
        "judge_path": _recorded_path(judge_run.artifact_path),
        "judge_artifact_sha256": judge.artifact_sha256,
        "judge_schema_hash": judge.schema_hash,
        "manifest_path": _recorded_path(manifest_path),
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "export_payload_sha256": dict(sorted(export_hashes.items())),
        "parameters": {
            "single_few_shot_only": bound_few_shot_count == 1,
            "bound_few_shot_count": bound_few_shot_count,
            "static_few_shot_count": 0,
            "label_positive_probability_gte": 0.5,
            "feature_timeout_seconds": judge_run.timeout_seconds,
            "feature_model": judge_run.model,
            "feature_reasoning_effort": judge_run.reasoning_effort,
            "experimental_judge_override": judge_run.is_experimental_override,
            "target_case_count": effective_target_case_count,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
        "results": {
            "requested_this_run": len(pending),
            "written_this_run": dict(created),
            "append_only_records": len(final_records),
            "unique_samples": len(latest_records_by_sample),
            "latest_status_counts": dict(
                Counter(
                    str(row.get("status") or "unknown")
                    for row in latest_records_by_sample.values()
                )
            ),
        },
    }
    _write_manifest(run_manifest_path, safe_manifest)
    return safe_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--judge-config", type=Path, default=DEFAULT_JUDGE_CONFIG)
    parser.add_argument("--judge", type=Path, help="Experimental override for the correction artifact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exports", type=Path, action="append", dest="export_dirs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true", dest="all_rows")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--case-id", action="append", dest="target_case_ids")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--feature-timeout-seconds", type=int, help="Experimental Judge execution override")
    parser.add_argument("--feature-model", help="Experimental Judge execution override")
    parser.add_argument("--feature-reasoning-effort", help="Experimental Judge execution override")
    parser.add_argument(
        "--allow-experimental-judge-override",
        action="store_true",
        help="Required before any --judge or --feature-* override can change the formal Judge",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = label(
        samples_path=args.samples,
        manifest_path=args.manifest,
        judge_config_path=args.judge_config,
        judge_path=args.judge,
        output_dir=args.output_dir,
        export_dirs=tuple(args.export_dirs) if args.export_dirs else DEFAULT_EXPORT_DIRS,
        resume=args.resume,
        retry_errors=args.retry_errors,
        limit=args.limit,
        all_rows=args.all_rows,
        target_case_ids=tuple(args.target_case_ids or ()),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        feature_timeout_seconds=args.feature_timeout_seconds,
        feature_model=args.feature_model,
        feature_reasoning_effort=args.feature_reasoning_effort,
        allow_experimental_judge_override=args.allow_experimental_judge_override,
    )
    # Do not print manifest content: it lists private hashed identifiers.
    print(json.dumps({"latest_status_counts": result["results"]["latest_status_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
