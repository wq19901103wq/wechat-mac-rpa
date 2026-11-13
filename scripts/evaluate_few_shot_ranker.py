#!/usr/bin/env python3
"""Compare a local single-few-shot reranker against production retrieval.

The evaluator only consumes the existing private sample and label indexes.  It
never loads exports, conversation text, C0, or the Judge.  Model selection is
fixed in this file: labels are split by *target*, stratified by relationship,
so candidates for one target cannot occur in both fitting and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "data/reports/private/few_shot_ranker_samples_v2/samples.jsonl"
DEFAULT_LABELS = ROOT / "data/reports/private/few_shot_ranker_labels_v2/labels.jsonl"
DEFAULT_OUTPUT = ROOT / "data/reports/private/few_shot_ranker_ranker_eval_v3/result.json"
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
FOLDS = 4
MIN_TARGETS_PER_RELATIONSHIP = 8
EXPECTED_CANDIDATES_PER_TARGET = 8
BOOTSTRAP_ITERATIONS = 2_000

# A production top-1 is deliberately not displaced by surface resemblance.
# These names reserve the only allowed override path for independently
# extracted, semantically specific action matches.  Each feature requires both
# an historical input action and evidence that the historical reply actually
# carried it out; a reply's wording alone can never qualify.
STRONG_REPLACEMENT_FEATURES = (
    "target_needs_direct_response__candidate_direct_request_answered",
    "target_needs_clarification__candidate_ambiguity_clarified",
    "target_needs_light_reaction__candidate_light_reaction_used",
    "target_needs_coordination__candidate_coordination_advanced",
    "target_requires_factual_boundary__candidate_factual_boundary_respected",
    "target_has_explicit_addressee__candidate_addressee_answered",
    "target_is_natural_closure_or_confirmation__candidate_reply_is_natural_closure_or_confirmation",
    "target_still_requires_interaction_progress__candidate_reply_moves_interaction_forward",
)

CANDIDATE_ACTION_FEATURE_NAMES = (
    "candidate_input_needs_direct_response",
    "candidate_input_needs_clarification",
    "candidate_input_needs_light_reaction",
    "candidate_input_needs_coordination",
    "candidate_input_requires_factual_boundary",
    "candidate_input_has_explicit_addressee",
    "candidate_reply_directly_addresses_request",
    "candidate_reply_asks_for_clarification",
    "candidate_reply_is_light_reaction",
    "candidate_reply_advances_coordination",
    "candidate_reply_respects_factual_boundary",
    "candidate_reply_addresses_current_addressee",
    "candidate_reply_is_natural_closure_or_confirmation",
    "candidate_reply_moves_interaction_forward",
)

# A candidate only supplies transferable evidence when its historical input
# actually called for an action and 我's real reply demonstrably completed that
# same action.  Target facts are joined only after this compound fact exists.
ACTION_COMPOUND_FEATURES = (
    (
        "needs_direct_response",
        "candidate_input_needs_direct_response",
        "candidate_reply_directly_addresses_request",
        "candidate_direct_request_answered",
    ),
    (
        "needs_clarification",
        "candidate_input_needs_clarification",
        "candidate_reply_asks_for_clarification",
        "candidate_ambiguity_clarified",
    ),
    (
        "needs_light_reaction",
        "candidate_input_needs_light_reaction",
        "candidate_reply_is_light_reaction",
        "candidate_light_reaction_used",
    ),
    (
        "needs_coordination",
        "candidate_input_needs_coordination",
        "candidate_reply_advances_coordination",
        "candidate_coordination_advanced",
    ),
    (
        "requires_factual_boundary",
        "candidate_input_requires_factual_boundary",
        "candidate_reply_respects_factual_boundary",
        "candidate_factual_boundary_respected",
    ),
    (
        "has_explicit_addressee",
        "candidate_input_has_explicit_addressee",
        "candidate_reply_addresses_current_addressee",
        "candidate_addressee_answered",
    ),
)

# These reply facts do not require an equivalent historical input fact: a
# closing or progress-making reply is transferable only when the current
# target calls for the same interaction state. Candidate facts are extracted
# offline; online ranking still needs one target-only model call.
TARGET_REPLY_FEATURE_CROSSES = (
    (
        "is_natural_closure_or_confirmation",
        "candidate_reply_is_natural_closure_or_confirmation",
    ),
    (
        "still_requires_interaction_progress",
        "candidate_reply_moves_interaction_forward",
    ),
)

# Named units for target/candidate action ablations. A unit always contains
# both the completed historical action fact and its matching current-target
# cross; it is never valid to enable a cross without its candidate fact.
# ``all`` retains the existing feature schema exactly.
ACTION_CROSS_NAMES = (
    "direct_response",
    "clarification",
    "light_reaction",
    "coordination",
    "factual_boundary",
    "explicit_addressee",
    "natural_closure",
    "interaction_progress",
)

ACTION_NAME_BY_COMPOUND_FEATURE = {
    "candidate_direct_request_answered": "direct_response",
    "candidate_ambiguity_clarified": "clarification",
    "candidate_light_reaction_used": "light_reaction",
    "candidate_coordination_advanced": "coordination",
    "candidate_factual_boundary_respected": "factual_boundary",
    "candidate_addressee_answered": "explicit_addressee",
}

# These are fixed before the first target-level feature evaluation.  Every
# right-hand term is a local candidate fact, so online ranking still needs at
# most one model call to understand the target, never one call per candidate.
TARGET_FEATURE_CROSSES = (
    ("needs_light_reaction", "candidate_reply_is_short", False),
    ("needs_clarification", "candidate_reply_has_question", False),
    ("needs_direct_response", "candidate_reply_is_short", True),
    ("needs_coordination", "candidate_reply_is_multi_bubble", False),
    ("context_multiple_other_speakers", "candidate_same_chat", False),
)

# Keep this deliberately small and local: it tests whether an otherwise useful
# historical question is over-preferred specifically in a private chat.  Both
# source facts already exist in the retrieval record, so scoring every
# candidate needs no additional model call.
LOCAL_FEATURE_CROSSES = (
    ("context_is_private", "candidate_reply_has_question"),
)

# The production retriever already computes these signals locally for every
# candidate.  Keep their use in the ranker boolean and target-relative: raw
# scores from different conversations are not comparable, while "best" and
# "top three" retain the original retrieval information needed to decide
# whether a learned re-order is warranted.  No LLM call is introduced here.
PRODUCTION_TRACE_RANK_FACTS = (
    "production_lexical_score",
    "production_length_similarity",
    "production_semantic_similarity",
    "production_profile_compatibility",
    "production_group_participant_match_count",
)


class InsufficientDataError(ValueError):
    """The first target-disjoint comparison cannot yet be run."""


@dataclass(frozen=True)
class LabeledSample:
    sample_id: str
    target_id: str
    candidate_id: str
    relationship: str
    candidate_relationship: str
    candidate_rank: int
    features: Mapping[str, bool]
    label: int
    retrieval_features: Mapping[str, Any] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _labels_paths(value: Path | Iterable[Path]) -> tuple[Path, ...]:
    if isinstance(value, Path):
        return (value,)
    paths = tuple(value)
    if not paths:
        raise ValueError("at least one label index is required")
    return paths


def _load_target_features(
    path: Path,
    expected_target_ids: set[str],
    *,
    allow_extra_targets: bool = False,
) -> dict[str, Mapping[str, bool]]:
    """Read one previously-extracted, target-only boolean record per target."""
    rows = _read_jsonl(path)
    values: dict[str, Mapping[str, bool]] = {}
    required_names = {
        "needs_direct_response",
        "needs_clarification",
        "needs_light_reaction",
        "needs_coordination",
        "requires_factual_boundary",
        "has_explicit_addressee",
        "is_natural_closure_or_confirmation",
        "still_requires_interaction_progress",
    }
    for row in rows:
        target_id = str(row.get("target_id") or "")
        features = row.get("features")
        if row.get("status") != "ok" or not target_id or not isinstance(features, dict):
            raise ValueError("target feature record is incomplete")
        if target_id in values:
            raise ValueError("target feature source has a duplicate target")
        if set(features) != required_names or any(type(features[name]) is not bool for name in required_names):
            raise ValueError("target feature source has an invalid boolean schema")
        values[target_id] = {name: features[name] for name in required_names}
    if not expected_target_ids <= set(values) or (
        not allow_extra_targets and set(values) != expected_target_ids
    ):
        missing = sorted(expected_target_ids - set(values))
        extra = sorted(set(values) - expected_target_ids)
        raise ValueError(f"target feature targets differ from samples: missing={missing}, extra={extra}")
    return values


def _load_candidate_action_features(
    path: Path,
    expected_candidates: Mapping[str, str],
) -> dict[str, Mapping[str, bool]]:
    """Read one precomputed action record for each candidate used in evaluation.

    Extraction intentionally covers every candidate in the source index, while
    an evaluation may exclude incomplete targets.  Extra records are therefore
    allowed.  Missing, duplicate, mismatched-relationship, failed, or malformed
    records for a selected candidate fail closed.
    """
    values: dict[str, Mapping[str, bool]] = {}
    required_names = set(CANDIDATE_ACTION_FEATURE_NAMES)
    for row in _read_jsonl(path):
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id not in expected_candidates:
            continue
        if candidate_id in values:
            raise ValueError("candidate feature source has a duplicate candidate")
        features = row.get("features")
        if row.get("status") != "ok" or not isinstance(features, dict):
            raise ValueError("candidate feature record is incomplete")
        if str(row.get("relationship") or "") != expected_candidates[candidate_id]:
            raise ValueError("candidate feature relationship differs from samples")
        if set(features) != required_names or any(
            type(features[name]) is not bool for name in required_names
        ):
            raise ValueError("candidate feature source has an invalid boolean schema")
        values[candidate_id] = {name: features[name] for name in CANDIDATE_ACTION_FEATURE_NAMES}
    missing = sorted(set(expected_candidates) - set(values))
    if missing:
        raise ValueError(f"candidate feature candidates differ from samples: missing={missing}")
    return values


def _crossed_features(
    local_features: Mapping[str, bool], target_features: Mapping[str, bool],
) -> dict[str, bool]:
    """Add only target-fact × candidate-fact interactions, never bare target facts."""
    result = dict(local_features)
    for target_name, candidate_name, negate_candidate in TARGET_FEATURE_CROSSES:
        if candidate_name not in local_features:
            raise ValueError(f"local feature required by target cross is absent: {candidate_name}")
        candidate_value = bool(local_features[candidate_name])
        if negate_candidate:
            candidate_value = not candidate_value
            candidate_name = "not_" + candidate_name.removeprefix("candidate_")
        target_value = (
            bool(local_features[target_name])
            if target_name.startswith("context_") else bool(target_features[target_name])
        )
        result[f"target_{target_name}__{candidate_name}"] = (
            target_value and candidate_value
        )
    return result


def _add_local_feature_crosses(local_features: Mapping[str, bool]) -> dict[str, bool]:
    """Add declared context × candidate fact crosses when both facts exist."""
    result = dict(local_features)
    for context_name, candidate_name in LOCAL_FEATURE_CROSSES:
        if context_name in local_features and candidate_name in local_features:
            result[f"{context_name}__{candidate_name}"] = (
                bool(local_features[context_name]) and bool(local_features[candidate_name])
            )
    return result


def load_labeled_samples(
    samples_path: Path,
    labels_path: Path | Iterable[Path],
    *,
    target_features_path: Path | None = None,
) -> list[LabeledSample]:
    """Join isolated append-only label indexes to candidates.

    Rows can be retried within one index.  Any successful record wins over a
    later transient failure; among successful records the latest one wins.  A
    sample may not occur in more than one index: that would make a sharded run
    ambiguous and fail closed instead of silently choosing one worker's row.
    """
    samples = {str(row.get("sample_id") or ""): row for row in _read_jsonl(samples_path)}
    latest_labels: dict[str, Mapping[str, Any]] = {}
    source_by_sample: dict[str, Path] = {}
    for path in _labels_paths(labels_path):
        latest_in_source: dict[str, Mapping[str, Any]] = {}
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                previous = latest_in_source.get(sample_id)
                if (
                    previous is None
                    or row.get("status") == "ok"
                    or previous.get("status") != "ok"
                ):
                    latest_in_source[sample_id] = row
        for sample_id, row in latest_in_source.items():
            previous_source = source_by_sample.get(sample_id)
            if previous_source is not None:
                raise ValueError(
                    "sample occurs in multiple label indexes; sharded outputs must be disjoint"
                )
            source_by_sample[sample_id] = path
            latest_labels[sample_id] = row

    target_features = None
    if target_features_path is not None:
        expected_target_ids = {
            str(row.get("target", {}).get("case_id") or "") for row in samples.values()
        }
        if "" in expected_target_ids:
            raise ValueError("sample is missing a target id")
        target_features = _load_target_features(target_features_path, expected_target_ids)

    result: list[LabeledSample] = []
    for sample_id, label_row in latest_labels.items():
        if label_row.get("status") != "ok":
            continue
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError("label references an unknown sample")
        raw_label = label_row.get("label")
        if raw_label not in {0, 1}:
            raise ValueError("successful label is not binary")
        target = sample.get("target") or {}
        candidate = sample.get("candidate") or {}
        features = sample.get("local_features")
        relationship = str(target.get("relationship") or "")
        if not isinstance(features, dict) or relationship not in {"private", "group"}:
            raise ValueError("sample has an invalid local-feature schema")
        candidate_rank = int(sample.get("candidate_rank") or 0)
        candidate_relationship = str(candidate.get("relationship") or (
            "acquaintance" if relationship == "private" else "group"
        ))
        if candidate_relationship not in {"acquaintance", "group"}:
            raise ValueError("sample has an invalid candidate relationship")
        local_features = {str(name): bool(value) for name, value in features.items()}
        # Production retrieval is already a useful ranking signal.  It must be
        # available to the learned score itself, rather than only resolving a
        # tie after a full re-ordering.  Preserve its ordinal meaning in
        # coarse, mutually-exclusive boolean bands: a rank-5 retrieval result
        # must not be treated as equally plausible as rank-2 merely because
        # both are not top-1.
        local_features["candidate_is_production_top1"] = candidate_rank == 1
        local_features["candidate_is_production_rank_2_or_3"] = candidate_rank in {2, 3}
        local_features["candidate_is_production_rank_4_to_8"] = 4 <= candidate_rank <= 8
        local_features = _add_local_feature_crosses(local_features)
        retrieval_features = sample.get("production_retrieval_features") or {}
        if not isinstance(retrieval_features, dict):
            raise ValueError("sample has an invalid production retrieval trace")
        result.append(LabeledSample(
            sample_id=sample_id,
            target_id=str(target.get("case_id") or ""),
            candidate_id=str(candidate.get("id") or ""),
            relationship=relationship,
            candidate_relationship=candidate_relationship,
            candidate_rank=candidate_rank,
            features=(
                _crossed_features(local_features, target_features[str(target.get("case_id") or "")])
                if target_features is not None
                else local_features
            ),
            label=int(raw_label),
            retrieval_features=retrieval_features,
        ))
    if not result:
        raise InsufficientDataError("there are no successful labels")
    if any(not row.target_id or not row.candidate_id or row.candidate_rank < 1 for row in result):
        raise ValueError("labeled sample has an incomplete ranking identity")
    return result


def _by_target(rows: Iterable[LabeledSample]) -> dict[str, list[LabeledSample]]:
    grouped: dict[str, list[LabeledSample]] = defaultdict(list)
    for row in rows:
        grouped[row.target_id].append(row)
    return dict(grouped)


def _complete_target_rows(
    samples_path: Path, rows: list[LabeledSample],
) -> tuple[list[LabeledSample], list[dict[str, Any]]]:
    """Keep only fully labeled production top-8 targets.

    The source index deliberately contains a small number of targets whose
    retriever returned fewer than eight leak-safe candidates.  They are useful
    label-collection evidence but are not a fair top-8 ranking comparison.
    Record each exclusion so the reported denominator is auditable instead of
    silently treating a smaller candidate set as an easier ranking task.
    """
    source_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in _read_jsonl(samples_path):
        target = sample.get("target") or {}
        target_id = str(target.get("case_id") or "")
        relationship = str(target.get("relationship") or "")
        candidate = sample.get("candidate") or {}
        if not target_id or relationship not in {"private", "group"}:
            raise ValueError("source sample has an invalid target identity")
        if not str(sample.get("sample_id") or "") or not str(candidate.get("id") or ""):
            raise ValueError("source sample has an incomplete candidate identity")
        source_groups[target_id].append(sample)

    labeled_groups = _by_target(rows)
    selected: list[LabeledSample] = []
    excluded: list[dict[str, Any]] = []
    for target_id, source_rows in sorted(source_groups.items()):
        relationships = {str((sample.get("target") or {}).get("relationship") or "") for sample in source_rows}
        if len(relationships) != 1:
            raise ValueError("source samples have a target with mixed relationships")
        relationship = next(iter(relationships))
        source_sample_ids = {str(sample.get("sample_id") or "") for sample in source_rows}
        source_candidate_ids = {
            str((sample.get("candidate") or {}).get("id") or "") for sample in source_rows
        }
        labeled_rows = labeled_groups.get(target_id, [])
        labeled_sample_ids = {row.sample_id for row in labeled_rows}
        reason: str | None = None
        if len(source_rows) != EXPECTED_CANDIDATES_PER_TARGET:
            reason = "source_candidate_count_not_8"
        elif len(source_sample_ids) != len(source_rows) or len(source_candidate_ids) != len(source_rows):
            raise ValueError("source samples have duplicate ranking identities")
        elif len(labeled_rows) != EXPECTED_CANDIDATES_PER_TARGET:
            reason = "successful_label_count_not_8"
        elif labeled_sample_ids != source_sample_ids:
            raise ValueError("successful labels do not match the source candidates")
        elif {row.relationship for row in labeled_rows} != {relationship}:
            raise ValueError("successful labels have a relationship mismatch")
        if reason is not None:
            excluded.append({
                "target_id": target_id,
                "relationship": relationship,
                "source_candidate_count": len(source_rows),
                "successful_label_count": len(labeled_rows),
                "reason": reason,
            })
            continue
        selected.extend(labeled_rows)
    if not selected:
        raise InsufficientDataError("no target has a complete labeled production top-8")
    return selected, excluded


def _add_target_feature_crosses(
    rows: list[LabeledSample], target_features_path: Path,
) -> list[LabeledSample]:
    """Add target-only feature crosses after incomplete targets were excluded."""
    target_features = _load_target_features(
        target_features_path,
        {row.target_id for row in rows},
        allow_extra_targets=True,
    )
    return [
        replace(
            row,
            features=_crossed_features(row.features, target_features[row.target_id]),
        )
        for row in rows
    ]


def _add_candidate_action_crosses(
    rows: list[LabeledSample],
    *,
    target_features_path: Path,
    candidate_features_path: Path,
    action_cross_names: frozenset[str] | None = None,
) -> list[LabeledSample]:
    """Join offline candidate actions, then cross only completed actions to target facts."""
    selected_names = action_cross_names or frozenset(ACTION_CROSS_NAMES)
    unknown_names = selected_names - set(ACTION_CROSS_NAMES)
    if unknown_names:
        raise ValueError(f"unknown candidate action crosses: {sorted(unknown_names)}")
    target_features = _load_target_features(
        target_features_path,
        {row.target_id for row in rows},
        allow_extra_targets=True,
    )
    expected_candidates: dict[str, str] = {}
    for row in rows:
        previous = expected_candidates.setdefault(row.candidate_id, row.candidate_relationship)
        if previous != row.candidate_relationship:
            raise ValueError("candidate has inconsistent relationships across samples")
    candidate_features = _load_candidate_action_features(
        candidate_features_path, expected_candidates,
    )
    result: list[LabeledSample] = []
    for row in rows:
        candidate = candidate_features[row.candidate_id]
        features = dict(row.features)
        for target_name, input_name, reply_name, compound_name in ACTION_COMPOUND_FEATURES:
            action_name = ACTION_NAME_BY_COMPOUND_FEATURE[compound_name]
            if action_name not in selected_names:
                continue
            compound_value = bool(candidate[input_name]) and bool(candidate[reply_name])
            features[compound_name] = compound_value
            features[f"target_{target_name}__{compound_name}"] = (
                bool(target_features[row.target_id][target_name]) and compound_value
            )
        for target_name, reply_name in TARGET_REPLY_FEATURE_CROSSES:
            action_name = (
                "natural_closure"
                if target_name == "is_natural_closure_or_confirmation"
                else "interaction_progress"
            )
            if action_name not in selected_names:
                continue
            features[f"target_{target_name}__{reply_name}"] = (
                bool(target_features[row.target_id][target_name])
                and bool(candidate[reply_name])
            )
        result.append(replace(row, features=features))
    return result


def _add_production_trace_features(rows: list[LabeledSample]) -> list[LabeledSample]:
    """Add target-relative boolean facts from the production retrieval trace.

    The trace is reconstructed offline from the exact production top-8.  It
    contains no generated reply or Judge information.  A feature says only
    that a candidate was best (or within the best three, including ties) on a
    signal already used by production; it never compares raw values between
    unrelated conversations.
    """
    result: list[LabeledSample] = []
    for target_rows in _by_target(rows).values():
        traces = [row.retrieval_features for row in target_rows]
        if any(not trace for trace in traces):
            raise ValueError("production trace features were requested but a selected sample has no trace")
        for fact in PRODUCTION_TRACE_RANK_FACTS:
            if any(not isinstance(trace.get(fact), (int, float)) for trace in traces):
                raise ValueError(f"production trace is missing numeric fact: {fact}")
        for trace in traces:
            for fact in (
                "production_pre_diversity_rank",
                "production_response_move_exact_match",
                "production_precise_situation_match",
                "production_multi_message_reply_coverage",
                "production_diversity_skips_before_selection",
            ):
                if fact not in trace or not isinstance(trace[fact], (bool, int, float)):
                    raise ValueError(f"production trace is missing fact: {fact}")

        top_values = {
            fact: max(float(trace[fact]) for trace in traces)
            for fact in PRODUCTION_TRACE_RANK_FACTS
        }
        top3_values = {
            fact: sorted((float(trace[fact]) for trace in traces), reverse=True)[min(2, len(traces) - 1)]
            for fact in PRODUCTION_TRACE_RANK_FACTS
        }
        for row in target_rows:
            trace = row.retrieval_features
            features = dict(row.features)
            for fact in PRODUCTION_TRACE_RANK_FACTS:
                short_name = fact.removeprefix("production_")
                value = float(trace[fact])
                features[f"candidate_production_{short_name}_best"] = value == top_values[fact]
                features[f"candidate_production_{short_name}_top3"] = value >= top3_values[fact]
            pre_diversity_rank = float(trace["production_pre_diversity_rank"])
            features.update({
                "candidate_production_pre_diversity_top1": pre_diversity_rank == 1.0,
                "candidate_production_pre_diversity_rank_2_or_3": pre_diversity_rank in {2.0, 3.0},
                "candidate_production_pre_diversity_rank_4_or_lower": pre_diversity_rank >= 4.0,
                "candidate_production_response_move_exact_match": bool(trace["production_response_move_exact_match"]),
                "candidate_production_precise_situation_match": bool(trace["production_precise_situation_match"]),
                "candidate_production_multi_message_reply_coverage": bool(trace["production_multi_message_reply_coverage"]),
                "candidate_production_diversity_displaced": (
                    float(trace["production_diversity_skips_before_selection"]) > 0.0
                ),
            })
            result.append(replace(row, features=features))
    return result


def _ranking_feature_names(
    groups: Mapping[str, list[LabeledSample]],
) -> tuple[str, ...]:
    """Keep only facts that can distinguish candidates for at least one target.

    Relationship and current-context facts are useful only when crossed with a
    candidate fact.  In their bare form they are constant across a target's
    top-8, so a ranker cannot use them to choose one candidate over another.
    Dropping them here prevents a global classifier from accidentally learning
    a relationship-level label prevalence instead of a within-target order.
    """
    all_names = tuple(sorted({name for rows in groups.values() for row in rows for name in row.features}))
    return tuple(
        name for name in all_names
        if any(len({bool(row.features[name]) for row in rows}) > 1 for rows in groups.values())
    )


def validate_first_comparison(rows: list[LabeledSample]) -> tuple[dict[str, list[LabeledSample]], tuple[str, ...]]:
    """Require the bounded, balanced first comparison before fitting anything."""
    groups = _by_target(rows)
    relationship_targets: dict[str, list[str]] = defaultdict(list)
    for target_id, target_rows in groups.items():
        relationships = {row.relationship for row in target_rows}
        if len(relationships) != 1:
            raise ValueError("a target has mixed relationships")
        if len(target_rows) != EXPECTED_CANDIDATES_PER_TARGET:
            raise InsufficientDataError(
                "each target needs all single-few-shot candidates labeled before comparison"
            )
        if len({row.candidate_id for row in target_rows}) != len(target_rows):
            raise ValueError("a target has duplicate candidates")
        relationship_targets[next(iter(relationships))].append(target_id)
    missing = {
        relationship: max(0, MIN_TARGETS_PER_RELATIONSHIP - len(relationship_targets[relationship]))
        for relationship in ("private", "group")
    }
    if any(missing.values()):
        raise InsufficientDataError(
            "need at least eight fully labeled targets for both private and group; "
            f"missing private={missing['private']}, group={missing['group']}"
        )
    all_feature_names = tuple(sorted({name for row in rows for name in row.features}))
    if not all_feature_names:
        raise ValueError("no local features were found")
    if any(set(row.features) != set(all_feature_names) for row in rows):
        raise ValueError("local-feature schema drift across samples")
    feature_names = _ranking_feature_names(groups)
    if not feature_names:
        raise InsufficientDataError("no candidate-varying features are available for ranking")
    return groups, feature_names


def _folds(groups: Mapping[str, list[LabeledSample]]) -> dict[str, int]:
    """Deterministic relation-stratified folds, assigned only at target level."""
    target_relationship = {
        target_id: rows[0].relationship for target_id, rows in groups.items()
    }
    result: dict[str, int] = {}
    for relationship in ("private", "group"):
        target_ids = sorted(
            target_id for target_id, value in target_relationship.items()
            if value == relationship
        )
        for index, target_id in enumerate(target_ids):
            result[target_id] = index % FOLDS
    return result


def _feature_matrix(rows: list[LabeledSample], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[float(row.features[name]) for name in feature_names] for row in rows],
        dtype=float,
    )


def _pairwise_training_matrix(
    rows: Iterable[LabeledSample], feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric positive-vs-negative pairs within each target only.

    Every train record says only that a positive candidate should outrank a
    negative candidate from the *same* target.  Symmetric swapped rows remove
    the need for an intercept and make target-wide feature constants cancel.
    """
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    for target_rows in _by_target(rows).values():
        positives = [row for row in target_rows if row.label]
        negatives = [row for row in target_rows if not row.label]
        for positive in positives:
            positive_vector = _feature_matrix([positive], feature_names)[0]
            for negative in negatives:
                negative_vector = _feature_matrix([negative], feature_names)[0]
                difference = positive_vector - negative_vector
                matrices.extend((difference, -difference))
                labels.extend((1, 0))
    if not matrices:
        raise InsufficientDataError("no target has both positive and negative candidates")
    return np.asarray(matrices, dtype=float), np.asarray(labels, dtype=int)


def _fit_train_dev_ranker(
    fit_rows: list[LabeledSample],
    scored_rows: list[LabeledSample],
    feature_names: tuple[str, ...],
    *,
    model_kind: str,
) -> tuple[dict[str, float], Any, int]:
    """Fit one fixed ranker and score unseen candidate rows.

    LR consumes symmetric within-target differences.  XGBoost instead consumes
    the same raw candidates grouped by target, using its pairwise ranking
    objective.  Both paths therefore learn only candidate preferences within a
    target; neither can use one target's label prevalence to rank another.
    """
    pair_matrix, pair_labels = _pairwise_training_matrix(fit_rows, feature_names)
    preference_count = len(pair_matrix) // 2
    if model_kind == "lr":
        model = LogisticRegression(
            C=0.25,
            fit_intercept=False,
            max_iter=2_000,
            random_state=0,
        )
        model.fit(pair_matrix, pair_labels)
        scores = _feature_matrix(scored_rows, feature_names) @ model.coef_[0]
    elif model_kind == "xgboost":
        try:
            from xgboost import XGBRanker
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError(
                "XGBoost comparison requires xgboost==2.1.4; install requirements.txt first"
            ) from exc
        fit_groups = _by_target(fit_rows)
        ordered_rows = [
            row
            for target_id in sorted(fit_groups)
            for row in sorted(fit_groups[target_id], key=lambda value: value.candidate_rank)
        ]
        group_sizes = [len(fit_groups[target_id]) for target_id in sorted(fit_groups)]
        # Fixed shallow/regularised parameters are deliberately a single
        # contrast to LR, not a tuning sweep over this small data set.
        model = XGBRanker(
            objective="rank:pairwise",
            n_estimators=24,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=10.0,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=0,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(
            _feature_matrix(ordered_rows, feature_names),
            np.asarray([row.label for row in ordered_rows], dtype=float),
            group=group_sizes,
        )
        scores = model.predict(_feature_matrix(scored_rows, feature_names))
    else:
        raise ValueError("model kind must be lr or xgboost")
    return (
        {row.sample_id: float(score) for row, score in zip(scored_rows, scores)},
        model,
        preference_count,
    )


def _ranker_model_description(model_kind: str) -> str:
    if model_kind == "lr":
        return "pairwise LogisticRegression(C=0.25, fit_intercept=False, max_iter=2000, random_state=0)"
    if model_kind == "xgboost":
        return (
            "XGBRanker(objective=rank:pairwise, n_estimators=24, max_depth=2, "
            "learning_rate=0.05, min_child_weight=5, reg_alpha=1.0, reg_lambda=10.0, "
            "subsample=1.0, colsample_bytree=1.0, random_state=0, n_jobs=1, tree_method=hist)"
        )
    raise ValueError("model kind must be lr or xgboost")


def _rank_metrics(
    groups: Mapping[str, list[LabeledSample]],
    scores: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Measure a ranking; original retrieval rank breaks only model-score ties."""
    per_target: dict[str, dict[str, float]] = {}
    for target_id, rows in groups.items():
        positives = sum(row.label for row in rows)
        ranked = sorted(
            rows,
            key=lambda row: (
                -scores[row.sample_id],
                row.candidate_rank,
                row.candidate_id,
            ),
        )
        selected = ranked[0]
        top1_passes = float(selected.label)
        target_metrics = {
            "top1_all": top1_passes,
            "judge_detection_top1_all": 1.0 - top1_passes,
            "selected_candidate_rank": float(selected.candidate_rank),
            "selected_candidate_label": top1_passes,
        }
        if positives:
            first_positive = next(index for index, row in enumerate(ranked, start=1) if row.label)
            target_metrics.update({
                "top1": top1_passes,
                "mrr": 1.0 / first_positive,
                "first_positive_rank": float(first_positive),
            })
        per_target[target_id] = target_metrics
    if not any("top1" in value for value in per_target.values()):
        raise InsufficientDataError("no target contains a positive candidate")
    summary = {
        name: float(np.mean([value[name] for value in per_target.values() if name in value]))
        for name in ("top1_all", "judge_detection_top1_all", "top1", "mrr", "first_positive_rank")
    }
    summary["all_targets"] = float(len(per_target))
    summary["eligible_targets"] = float(sum("top1" in value for value in per_target.values()))
    return summary, per_target


def _conservative_scores(
    groups: Mapping[str, list[LabeledSample]],
    raw_scores: Mapping[str, float],
    *,
    max_replacement_rank: int | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Keep production top-1 unless an action match has learned support.

    Surface or provenance facts may still contribute to the learned score, but
    cannot independently change the returned few-shot.  This makes the safe
    no-action-feature outcome auditable: it must select the production rank-1
    candidate for every target instead of performing a weaker full re-rank.
    """
    selection_scores: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for target_id, rows in groups.items():
        production = next(row for row in rows if row.candidate_rank == 1)
        eligible = [
            row for row in rows
            if row.candidate_rank != 1
            and (max_replacement_rank is None or row.candidate_rank <= max_replacement_rank)
            and raw_scores[row.sample_id] > raw_scores[production.sample_id]
            and any(bool(row.features.get(name)) for name in STRONG_REPLACEMENT_FEATURES)
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda row: (-raw_scores[row.sample_id], row.candidate_rank, row.candidate_id),
            )
            evidence = [
                name for name in STRONG_REPLACEMENT_FEATURES
                if bool(selected.features.get(name))
            ]
            reason = "supported_action_match"
        else:
            selected = production
            evidence = []
            reason = "retain_production_top1_no_supported_action_match"
        for row in rows:
            selection_scores[row.sample_id] = float(row is selected)
        diagnostics[target_id] = {
            "selected_candidate_rank": selected.candidate_rank,
            "replaced_production_top1": selected is not production,
            "reason": reason,
            "action_match_features": evidence,
            "score_margin_vs_production": float(
                raw_scores[selected.sample_id] - raw_scores[production.sample_id]
            ),
        }
    return selection_scores, diagnostics


def _top_rank_limited_scores(
    groups: Mapping[str, list[LabeledSample]],
    raw_scores: Mapping[str, float],
    *,
    max_replacement_rank: int,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Re-rank only the production prefix and preserve the rest verbatim."""
    selection_scores: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for target_id, rows in groups.items():
        production = next(row for row in rows if row.candidate_rank == 1)
        eligible = [row for row in rows if row.candidate_rank <= max_replacement_rank]
        selected = min(
            eligible,
            key=lambda row: (-raw_scores[row.sample_id], row.candidate_rank, row.candidate_id),
        )
        for row in rows:
            selection_scores[row.sample_id] = float(row is selected)
        diagnostics[target_id] = {
            "selected_candidate_rank": selected.candidate_rank,
            "replaced_production_top1": selected is not production,
            "reason": f"pairwise_rerank_within_production_top_{max_replacement_rank}",
            "score_margin_vs_production": float(
                raw_scores[selected.sample_id] - raw_scores[production.sample_id]
            ),
        }
    return selection_scores, diagnostics


def _paired_bootstrap_interval(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    metric: str,
) -> dict[str, float]:
    target_ids = sorted(
        key for key in set(baseline) & set(candidate)
        if metric in baseline[key] and metric in candidate[key]
    )
    if not target_ids:
        raise InsufficientDataError(f"no targets support metric {metric}")
    diffs = np.asarray([candidate[key][metric] - baseline[key][metric] for key in target_ids])
    generator = np.random.default_rng(0)
    means = np.asarray([
        float(np.mean(generator.choice(diffs, size=len(diffs), replace=True)))
        for _ in range(BOOTSTRAP_ITERATIONS)
    ])
    return {
        "delta": float(np.mean(diffs)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _metric_comparison(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Summarize the paired production-vs-ranker result for one target set."""
    result: dict[str, dict[str, float]] = {}
    for metric in ("top1_all", "judge_detection_top1_all", "top1", "mrr", "first_positive_rank"):
        result[metric] = {
            "bootstrap": float(np.mean([value[metric] for value in baseline.values() if metric in value])),
            "learned": float(np.mean([value[metric] for value in candidate.values() if metric in value])),
            **_paired_bootstrap_interval(baseline, candidate, metric),
        }
    return result


def _holdout_rows(
    samples_path: Path,
    labels_path: Path | Iterable[Path],
) -> tuple[list[LabeledSample], list[dict[str, Any]]]:
    """Return only fully labelled top-8 targets, without OOF balance constraints."""
    rows = load_labeled_samples(samples_path, labels_path)
    rows, excluded = _complete_target_rows(samples_path, rows)
    groups = _by_target(rows)
    if len(groups) < MIN_TARGETS_PER_RELATIONSHIP:
        raise InsufficientDataError(
            "need at least eight fully labeled targets for a holdout comparison"
        )
    feature_schema = set(rows[0].features)
    if any(set(row.features) != feature_schema for row in rows):
        raise ValueError("local-feature schema drift across samples")
    return rows, excluded


def _training_rows(
    samples_path: Path,
    labels_path: Path | Iterable[Path],
    *,
    include_incomplete_pairwise_targets: bool,
) -> tuple[list[LabeledSample], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return fair top-8 rows, optionally adding auditable pairwise-only rows.

    An incomplete source target can never enter a ranking metric.  It can,
    however, contribute to a pairwise fit when every candidate it did return
    was labeled and it contains both outcomes.  This is deliberately opt-in
    and reported separately, so development denominators remain unchanged.
    """
    all_rows = load_labeled_samples(samples_path, labels_path)
    complete_rows, excluded = _complete_target_rows(samples_path, all_rows)
    if not include_incomplete_pairwise_targets:
        return complete_rows, excluded, []

    source_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in _read_jsonl(samples_path):
        target = sample.get("target") or {}
        target_id = str(target.get("case_id") or "")
        if target_id:
            source_groups[target_id].append(sample)
    labeled_groups = _by_target(all_rows)
    extra_rows: list[LabeledSample] = []
    included: list[dict[str, Any]] = []
    for record in excluded:
        target_id = str(record["target_id"])
        source_rows = source_groups[target_id]
        labeled_rows = labeled_groups.get(target_id, [])
        source_sample_ids = {str(sample.get("sample_id") or "") for sample in source_rows}
        labeled_sample_ids = {row.sample_id for row in labeled_rows}
        positives = sum(row.label for row in labeled_rows)
        negatives = len(labeled_rows) - positives
        if (
            len(source_rows) >= 2
            and len(labeled_rows) == len(source_rows)
            and labeled_sample_ids == source_sample_ids
            and positives
            and negatives
        ):
            extra_rows.extend(labeled_rows)
            included.append({
                "target_id": target_id,
                "relationship": record["relationship"],
                "source_candidate_count": len(source_rows),
                "successful_label_count": len(labeled_rows),
                "positive_count": positives,
                "negative_count": negatives,
                "pairwise_preference_count": positives * negatives,
                "reason": "complete_labels_but_incomplete_production_top8_pairwise_training_only",
            })
    return complete_rows + extra_rows, excluded, included


def evaluate_train_dev(
    train_samples_path: Path,
    train_labels_path: Path | Iterable[Path],
    dev_samples_path: Path,
    dev_labels_path: Path | Iterable[Path],
    *,
    use_production_trace_features: bool = False,
    learned_relationship: str = "all",
    replacement_policy: str = "full_rerank",
    train_target_features_path: Path | None = None,
    train_candidate_features_path: Path | None = None,
    dev_target_features_path: Path | None = None,
    dev_candidate_features_path: Path | None = None,
    candidate_action_crosses: Iterable[str] | None = None,
    include_incomplete_train_pairwise_targets: bool = False,
    model_kind: str = "lr",
) -> dict[str, Any]:
    """Fit once on train targets, then compare rankings only on disjoint dev targets.

    This is the first-round protocol for the single-few-shot ranker.  It is
    intentionally separate from ``evaluate()``, whose target-disjoint folds
    are a development diagnostic rather than an independent comparison.
    """
    if learned_relationship not in {"all", "group", "private"}:
        raise ValueError("learned relationship must be all, group, or private")
    if replacement_policy not in {"full_rerank", "production_top3_only"}:
        raise ValueError("train/dev replacement policy must be full_rerank or production_top3_only")
    if model_kind not in {"lr", "xgboost"}:
        raise ValueError("model kind must be lr or xgboost")
    action_feature_paths = (
        train_target_features_path,
        train_candidate_features_path,
        dev_target_features_path,
        dev_candidate_features_path,
    )
    selected_action_crosses = (
        frozenset(candidate_action_crosses)
        if candidate_action_crosses is not None else frozenset(ACTION_CROSS_NAMES)
    )
    unknown_action_crosses = selected_action_crosses - set(ACTION_CROSS_NAMES)
    if unknown_action_crosses:
        raise ValueError(f"unknown candidate action crosses: {sorted(unknown_action_crosses)}")
    if not selected_action_crosses and any(path is not None for path in action_feature_paths):
        raise ValueError("candidate action features require at least one selected action cross")
    if any(path is not None for path in action_feature_paths) and not all(
        path is not None for path in action_feature_paths
    ):
        raise ValueError(
            "train/dev action features require train and dev target/candidate feature files together"
        )
    train_rows, train_excluded, train_included_incomplete = _training_rows(
        train_samples_path,
        train_labels_path,
        include_incomplete_pairwise_targets=include_incomplete_train_pairwise_targets,
    )
    dev_rows, dev_excluded = _holdout_rows(dev_samples_path, dev_labels_path)
    if all(path is not None for path in action_feature_paths):
        # When only group ranking is learned, action facts are intentionally
        # extracted only for group examples.  Private ranking is preserved
        # exactly, so requiring private features would add cost without ever
        # affecting a score.
        action_relationship = learned_relationship if learned_relationship in {"group", "private"} else None
        for rows, target_path, candidate_path in (
            (train_rows, train_target_features_path, train_candidate_features_path),
            (dev_rows, dev_target_features_path, dev_candidate_features_path),
        ):
            assert target_path is not None and candidate_path is not None
            action_rows = [
                row for row in rows
                if action_relationship is None or row.relationship == action_relationship
            ]
            action_by_sample = {
                row.sample_id: row
                for row in _add_candidate_action_crosses(
                    action_rows,
                    target_features_path=target_path,
                    candidate_features_path=candidate_path,
                    action_cross_names=selected_action_crosses,
                )
            }
            rows[:] = [action_by_sample.get(row.sample_id, row) for row in rows]
    if use_production_trace_features:
        # In group-only mode private chats retain production rank exactly, so
        # they never need the optional trace.  This permits an independent
        # group-only replay without re-creating private traces that cannot
        # affect either fitting or scoring.
        trace_relationship = learned_relationship if learned_relationship in {"group", "private"} else None
        for rows in (train_rows, dev_rows):
            trace_rows = [
                row for row in rows
                if trace_relationship is None or row.relationship == trace_relationship
            ]
            traced_rows = {
                row.sample_id: row for row in _add_production_trace_features(trace_rows)
            }
            rows[:] = [traced_rows.get(row.sample_id, row) for row in rows]
    train_groups = _by_target(train_rows)
    dev_groups = _by_target(dev_rows)
    shared_targets = set(train_groups) & set(dev_groups)
    if shared_targets:
        raise ValueError("train and dev target sets overlap")
    fit_rows = [
        row for row in train_rows
        if learned_relationship == "all" or row.relationship == learned_relationship
    ]
    scored_rows = [
        row for row in dev_rows
        if learned_relationship == "all" or row.relationship == learned_relationship
    ]
    fit_groups = _by_target(fit_rows)
    if not fit_rows:
        raise InsufficientDataError(f"no train rows for learned relationship: {learned_relationship}")
    if not scored_rows:
        raise InsufficientDataError(f"no dev rows for learned relationship: {learned_relationship}")
    train_feature_schema = set(fit_rows[0].features)
    if any(set(row.features) != train_feature_schema for row in fit_rows):
        raise ValueError("train ranking-feature schema drift across fitted rows")
    dev_feature_schema = set(scored_rows[0].features)
    if any(set(row.features) != dev_feature_schema for row in scored_rows):
        raise ValueError("dev ranking-feature schema drift across scored rows")
    if train_feature_schema != dev_feature_schema:
        raise ValueError("train and dev ranking-feature schemas differ")
    feature_names = _ranking_feature_names(fit_groups)
    if not feature_names:
        raise InsufficientDataError("no train candidate-varying features are available for ranking")
    learned_scores, model, preference_count = _fit_train_dev_ranker(
        fit_rows,
        scored_rows,
        feature_names,
        model_kind=model_kind,
    )
    raw_scores = {
        row.sample_id: (
            learned_scores[row.sample_id]
            if learned_relationship == "all" or row.relationship == learned_relationship
            else -float(row.candidate_rank)
        )
        for row in dev_rows
    }
    if replacement_policy == "full_rerank":
        scores = raw_scores
        replacement_diagnostics = {
            target_id: {
                "selected_candidate_rank": int(min(
                    target_rows,
                    key=lambda row: (-scores[row.sample_id], row.candidate_rank, row.candidate_id),
                ).candidate_rank),
                "replacement_policy": "full_rerank",
            }
            for target_id, target_rows in dev_groups.items()
        }
    else:
        learned_groups = {
            target_id: rows for target_id, rows in dev_groups.items()
            if learned_relationship == "all" or rows[0].relationship == learned_relationship
        }
        learned_scores_limited, replacement_diagnostics = _top_rank_limited_scores(
            learned_groups,
            raw_scores,
            max_replacement_rank=3,
        )
        scores = {
            row.sample_id: (
                learned_scores_limited[row.sample_id]
                if row.target_id in learned_groups else -float(row.candidate_rank)
            )
            for row in dev_rows
        }
        for target_id in dev_groups:
            if target_id not in replacement_diagnostics:
                replacement_diagnostics[target_id] = {
                    "selected_candidate_rank": 1,
                    "replacement_policy": "preserve_production_order",
                }
    production_scores = {row.sample_id: -float(row.candidate_rank) for row in dev_rows}
    baseline_metrics, baseline_targets = _rank_metrics(dev_groups, production_scores)
    ranker_metrics, ranker_targets = _rank_metrics(dev_groups, scores)
    metrics = _metric_comparison(baseline_targets, ranker_targets)
    target_relationship = {
        target_id: target_rows[0].relationship for target_id, target_rows in dev_groups.items()
    }
    by_relationship: dict[str, dict[str, dict[str, float]]] = {}
    for relationship in sorted(set(target_relationship.values())):
        target_ids = {
            target_id for target_id, value in target_relationship.items() if value == relationship
        }
        by_relationship[relationship] = _metric_comparison(
            {target_id: baseline_targets[target_id] for target_id in target_ids},
            {target_id: ranker_targets[target_id] for target_id in target_ids},
        )
    diagnostics = {
        target_id: {
            "relationship": target_relationship[target_id],
            "production": {
                "candidate_rank": int(baseline_targets[target_id]["selected_candidate_rank"]),
                "label": int(baseline_targets[target_id]["selected_candidate_label"]),
            },
            "learned": {
                "candidate_rank": int(ranker_targets[target_id]["selected_candidate_rank"]),
                "label": int(ranker_targets[target_id]["selected_candidate_label"]),
            },
        }
        for target_id in sorted(dev_groups)
    }
    return {
        "schema_version": 1,
        "kind": "single_few_shot_ranker_train_dev_pairwise_evaluation",
        "sources": {
            "train_samples": {
                "path": _recorded_path(train_samples_path),
                "sha256": _sha256_file(train_samples_path),
            },
            "train_label_indexes": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in _labels_paths(train_labels_path)
            ],
            "dev_samples": {
                "path": _recorded_path(dev_samples_path),
                "sha256": _sha256_file(dev_samples_path),
            },
            "dev_label_indexes": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in _labels_paths(dev_labels_path)
            ],
            "train_target_features": (
                {
                    "path": _recorded_path(train_target_features_path),
                    "sha256": _sha256_file(train_target_features_path),
                }
                if train_target_features_path is not None else None
            ),
            "train_candidate_features": (
                {
                    "path": _recorded_path(train_candidate_features_path),
                    "sha256": _sha256_file(train_candidate_features_path),
                }
                if train_candidate_features_path is not None else None
            ),
            "dev_target_features": (
                {
                    "path": _recorded_path(dev_target_features_path),
                    "sha256": _sha256_file(dev_target_features_path),
                }
                if dev_target_features_path is not None else None
            ),
            "dev_candidate_features": (
                {
                    "path": _recorded_path(dev_candidate_features_path),
                    "sha256": _sha256_file(dev_candidate_features_path),
                }
                if dev_candidate_features_path is not None else None
            ),
        },
        "protocol": {
            "single_few_shot_only": True,
            "static_few_shot_count": 0,
            "baseline_ranking": "production retriever returned candidate rank",
            "candidate_ranking": (
                "pairwise learned score across the production top-8 for every relationship, then original candidate rank"
                if learned_relationship == "all" and replacement_policy == "full_rerank" else
                f"pairwise learned score across the production top-8 for {learned_relationship} targets; "
                "other relationships retain the production rank"
                if replacement_policy == "full_rerank" else
                "pairwise learned score only within the production top-3; lower-ranked candidates retain the production order"
            ),
            "primary_metric": "judge_detection_top1_all; lower is better",
            "train_dev_target_disjoint": True,
            "learned_relationship": learned_relationship,
            "private_ranking_when_group_only": (
                "production retriever returned candidate rank"
                if learned_relationship == "group" else None
            ),
            "group_ranking_when_private_only": (
                "production retriever returned candidate rank"
                if learned_relationship == "private" else None
            ),
            "production_trace_features": use_production_trace_features,
            "candidate_action_features": all(path is not None for path in action_feature_paths),
            "candidate_action_crosses": sorted(selected_action_crosses),
            "replacement_policy": replacement_policy,
            "include_incomplete_train_pairwise_targets": include_incomplete_train_pairwise_targets,
            "model": _ranker_model_description(model_kind),
            "selection": "fixed before the independent dev comparison; no parameter search",
        },
        "train_dataset": {
            "labeled_samples": len(train_rows),
            "targets": len(train_groups),
            "relationship_targets": dict(sorted(Counter(
                values[0].relationship for values in train_groups.values()
            ).items())),
            "excluded_incomplete_targets": train_excluded,
            "included_incomplete_pairwise_training_targets": train_included_incomplete,
            "fit_labeled_samples": len(fit_rows),
            "fit_targets": len(fit_groups),
            "fit_relationship_targets": dict(sorted(Counter(
                values[0].relationship for values in fit_groups.values()
            ).items())),
            "pairwise_preference_count": preference_count,
        },
        "dev_dataset": {
            "labeled_samples": len(dev_rows),
            "targets": len(dev_groups),
            "relationship_targets": dict(sorted(Counter(
                values[0].relationship for values in dev_groups.values()
            ).items())),
            "excluded_incomplete_targets": dev_excluded,
            "ranking_feature_count": len(feature_names),
            "ranking_feature_names": list(feature_names),
        },
        "ranking_metrics": metrics,
        "ranking_metrics_by_relationship": by_relationship,
        "top1_diagnostics": diagnostics,
        "replacement_diagnostics": replacement_diagnostics,
        "coefficients": (
            {name: float(value) for name, value in zip(feature_names, model.coef_[0])}
            if model_kind == "lr" else None
        ),
        "feature_importances": (
            {name: float(value) for name, value in zip(feature_names, model.feature_importances_)}
            if model_kind == "xgboost" else None
        ),
        "raw_dev_metrics": {"production": baseline_metrics, "learned": ranker_metrics},
    }


def evaluate(
    samples_path: Path = DEFAULT_SAMPLES,
    labels_path: Path | Iterable[Path] = DEFAULT_LABELS,
    *,
    target_features_path: Path | None = None,
    candidate_features_path: Path | None = None,
    replacement_policy: str = "full_rerank",
) -> dict[str, Any]:
    if replacement_policy not in {
        "full_rerank",
        "conservative_action_match",
        "conservative_action_match_top3",
    }:
        raise ValueError(
            "replacement policy must be full_rerank, conservative_action_match, "
            "or conservative_action_match_top3"
        )
    labels_paths = _labels_paths(labels_path)
    if candidate_features_path is not None and target_features_path is None:
        raise ValueError("candidate action features require target features")
    rows = load_labeled_samples(samples_path, labels_path)
    rows, excluded_incomplete_targets = _complete_target_rows(samples_path, rows)
    if target_features_path is not None:
        rows = _add_target_feature_crosses(rows, target_features_path)
    if candidate_features_path is not None:
        rows = _add_candidate_action_crosses(
            rows,
            target_features_path=target_features_path,
            candidate_features_path=candidate_features_path,
        )
    groups, feature_names = validate_first_comparison(rows)
    fold_by_target = _folds(groups)
    scores: dict[str, float] = {}
    coefficient_rows: list[np.ndarray] = []
    for fold in range(FOLDS):
        train = [row for row in rows if fold_by_target[row.target_id] != fold]
        test = [row for row in rows if fold_by_target[row.target_id] == fold]
        pair_matrix, pair_labels = _pairwise_training_matrix(train, feature_names)
        model = LogisticRegression(
            C=0.25,
            fit_intercept=False,
            max_iter=2_000,
            random_state=0,
        )
        model.fit(pair_matrix, pair_labels)
        coefficient_rows.append(model.coef_[0])
        for row, score in zip(test, _feature_matrix(test, feature_names) @ model.coef_[0]):
            scores[row.sample_id] = float(score)
    if set(scores) != {row.sample_id for row in rows}:
        raise RuntimeError("out-of-fold predictions do not cover every labeled sample")

    production_scores = {row.sample_id: -float(row.candidate_rank) for row in rows}
    baseline_metrics, baseline_targets = _rank_metrics(groups, production_scores)
    if replacement_policy == "full_rerank":
        selected_scores = scores
    else:
        selected_scores, replacement_diagnostics = _conservative_scores(
            groups,
            scores,
            max_replacement_rank=3 if replacement_policy == "conservative_action_match_top3" else None,
        )
    ranker_metrics, ranker_targets = _rank_metrics(
        groups,
        selected_scores,
    )
    if replacement_policy == "full_rerank":
        replacement_diagnostics = {
            target_id: {
                "selected_candidate_rank": int(ranker_targets[target_id]["selected_candidate_rank"]),
                "replaced_production_top1": int(ranker_targets[target_id]["selected_candidate_rank"]) != 1,
                "reason": "full_pairwise_rerank",
                "action_match_features": [],
                "score_margin_vs_production": float(
                    scores[next(row.sample_id for row in groups[target_id]
                                if row.candidate_rank == int(ranker_targets[target_id]["selected_candidate_rank"]))]
                    - scores[next(row.sample_id for row in groups[target_id] if row.candidate_rank == 1)]
                ),
            }
            for target_id in sorted(groups)
        }
    target_relationship = {target_id: target_rows[0].relationship for target_id, target_rows in groups.items()}
    split_counts = {
        str(fold): {
            relationship: sum(1 for target_id, assigned in fold_by_target.items()
                              if assigned == fold and target_relationship[target_id] == relationship)
            for relationship in ("private", "group")
        }
        for fold in range(FOLDS)
    }
    metrics = {}
    for metric in ("top1_all", "judge_detection_top1_all", "top1", "mrr", "first_positive_rank"):
        metrics[metric] = {
            "bootstrap": baseline_metrics[metric],
            "learned": ranker_metrics[metric],
            **_paired_bootstrap_interval(baseline_targets, ranker_targets, metric),
        }
    top1_diagnostics = {
        target_id: {
            "relationship": target_relationship[target_id],
            "production": {
                "candidate_rank": int(baseline_targets[target_id]["selected_candidate_rank"]),
                "label": int(baseline_targets[target_id]["selected_candidate_label"]),
            },
            "pairwise": {
                "candidate_rank": int(ranker_targets[target_id]["selected_candidate_rank"]),
                "label": int(ranker_targets[target_id]["selected_candidate_label"]),
            },
        }
        for target_id in sorted(groups)
    }
    coefficient_mean = np.mean(np.asarray(coefficient_rows), axis=0)
    all_feature_names = tuple(sorted(rows[0].features))
    excluded_feature_names = tuple(
        name for name in all_feature_names if name not in set(feature_names)
    )
    full_pair_matrix, _ = _pairwise_training_matrix(rows, feature_names)
    return {
        "schema_version": 6,
        "kind": "single_few_shot_production_top8_pairwise_reranker_target_disjoint_evaluation",
        "sources": {
            "samples_path": _recorded_path(samples_path),
            "samples_sha256": _sha256_file(samples_path),
            "label_indexes": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in labels_paths
            ],
            "target_features": (
                {"path": _recorded_path(target_features_path), "sha256": _sha256_file(target_features_path)}
                if target_features_path is not None else None
            ),
            "candidate_features": (
                {"path": _recorded_path(candidate_features_path), "sha256": _sha256_file(candidate_features_path)}
                if candidate_features_path is not None else None
            ),
        },
        "protocol": {
            "single_few_shot_only": True,
            "static_few_shot_count": 0,
            "candidate_features": (
                "target-varying local booleans, production-rank bands, local target crosses, "
                "and offline historical input×reply action facts crossed to target facts"
                if candidate_features_path is not None else
                "target-varying local booleans, production-rank bands, and fixed target-fact × local-candidate-fact crosses"
                if target_features_path is not None else "target-varying local booleans and production-rank bands"
            ),
            "baseline_ranking": "production retriever returned candidate rank",
            "candidate_ranking": (
                "pairwise learned score across the production top-8, then original candidate rank"
                if replacement_policy == "full_rerank" else
                "retain production top-1 unless a higher-scored alternate has a declared semantic action-match feature"
                + (" and lies in production top-3" if replacement_policy == "conservative_action_match_top3" else "")
            ),
            "replacement_policy": replacement_policy,
            "strong_replacement_features": list(STRONG_REPLACEMENT_FEATURES),
            "primary_metric": "judge_detection_top1_all; lower is better",
            "target_disjoint": True,
            "folds": FOLDS,
            "minimum_targets_per_relationship": MIN_TARGETS_PER_RELATIONSHIP,
            "model": "pairwise LogisticRegression(C=0.25, fit_intercept=False, max_iter=2000, random_state=0)",
            "selection": "fixed before evaluation; no parameter search",
        },
        "dataset": {
            "labeled_samples": len(rows),
            "targets": len(groups),
            "excluded_incomplete_targets": excluded_incomplete_targets,
            "relationship_targets": dict(sorted(Counter(
                values[0].relationship for values in groups.values()
            ).items())),
            "label_counts": dict(sorted(Counter(row.label for row in rows).items())),
            "local_feature_count": len(all_feature_names),
            "target_feature_cross_count": len(TARGET_FEATURE_CROSSES) if target_features_path is not None else 0,
            "candidate_action_compound_count": (
                len(ACTION_COMPOUND_FEATURES) if candidate_features_path is not None else 0
            ),
            "candidate_action_target_cross_count": (
                len(ACTION_COMPOUND_FEATURES) if candidate_features_path is not None else 0
            ),
            "candidate_reply_target_cross_count": (
                len(TARGET_REPLY_FEATURE_CROSSES) if candidate_features_path is not None else 0
            ),
            "ranking_feature_count": len(feature_names),
            "ranking_feature_names": list(feature_names),
            "excluded_target_constant_feature_names": list(excluded_feature_names),
            "pairwise_preference_count": len(full_pair_matrix) // 2,
            "targets_with_positive_candidate": int(baseline_metrics["eligible_targets"]),
        },
        "fold_target_counts": split_counts,
        "ranking_metrics": metrics,
        "top1_diagnostics": top1_diagnostics,
        "replacement_diagnostics": replacement_diagnostics,
        "mean_fold_coefficients": {
            name: float(value) for name, value in zip(feature_names, coefficient_mean)
        },
    }


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, PRIVATE_DIR_MODE)
    path.write_text(json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, PRIVATE_MODE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--labels", type=Path, action="append")
    parser.add_argument("--train-samples", type=Path)
    parser.add_argument("--train-labels", type=Path, action="append")
    parser.add_argument("--dev-samples", type=Path)
    parser.add_argument("--dev-labels", type=Path, action="append")
    parser.add_argument("--target-features", type=Path)
    parser.add_argument("--candidate-features", type=Path)
    parser.add_argument("--train-target-features", type=Path)
    parser.add_argument("--train-candidate-features", type=Path)
    parser.add_argument("--dev-target-features", type=Path)
    parser.add_argument("--dev-candidate-features", type=Path)
    parser.add_argument(
        "--candidate-action-crosses",
        default="all",
        help=(
            "comma-separated action units to add when candidate action features are supplied; "
            "use 'all' for the established full schema"
        ),
    )
    parser.add_argument(
        "--include-incomplete-train-pairwise-targets",
        action="store_true",
        help=(
            "fit on fully labeled incomplete train targets only when they contain both outcomes; "
            "never includes them in development metrics"
        ),
    )
    parser.add_argument(
        "--use-production-trace-features",
        action="store_true",
        help="add target-relative boolean facts from the existing production retrieval trace",
    )
    parser.add_argument(
        "--learned-relationship",
        choices=("all", "group", "private"),
        default="all",
        help="fit and apply the learned ranking only to this relationship; a single relationship preserves the other production order",
    )
    parser.add_argument(
        "--replacement-policy",
        choices=(
            "full_rerank",
            "production_top3_only",
            "conservative_action_match",
            "conservative_action_match_top3",
        ),
        default="full_rerank",
    )
    parser.add_argument(
        "--model",
        choices=("lr", "xgboost"),
        default="lr",
        help="fixed ranker family for a train/dev comparison; default keeps the established LR",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate_action_crosses = (
        ACTION_CROSS_NAMES if args.candidate_action_crosses == "all" else tuple(
            name for name in args.candidate_action_crosses.split(",") if name
        )
    )
    holdout_values = (args.train_samples, args.train_labels, args.dev_samples, args.dev_labels)
    if any(value is not None for value in holdout_values):
        if not all(value is not None for value in holdout_values):
            raise ValueError("--train-samples, --train-labels, --dev-samples, and --dev-labels are required together")
        if args.target_features or args.candidate_features or args.replacement_policy not in {
            "full_rerank", "production_top3_only",
        }:
            raise ValueError("train/dev evaluation uses the train/dev action-feature arguments and this replacement policy")
        result = evaluate_train_dev(
            args.train_samples,
            tuple(args.train_labels),
            args.dev_samples,
            tuple(args.dev_labels),
            use_production_trace_features=args.use_production_trace_features,
            learned_relationship=args.learned_relationship,
            replacement_policy=args.replacement_policy,
            train_target_features_path=args.train_target_features,
            train_candidate_features_path=args.train_candidate_features,
            dev_target_features_path=args.dev_target_features,
            dev_candidate_features_path=args.dev_candidate_features,
            candidate_action_crosses=candidate_action_crosses,
            include_incomplete_train_pairwise_targets=args.include_incomplete_train_pairwise_targets,
            model_kind=args.model,
        )
    else:
        if args.use_production_trace_features or args.learned_relationship != "all":
            raise ValueError("production trace features and learned relationship are currently supported only by train/dev evaluation")
        result = evaluate(
            args.samples,
            tuple(args.labels) if args.labels else DEFAULT_LABELS,
            target_features_path=args.target_features,
            candidate_features_path=args.candidate_features,
            replacement_policy=args.replacement_policy,
        )
    write_result(args.output, result)
    metrics = result["ranking_metrics"]
    print(json.dumps({
        "judge_detection_top1_delta": metrics["judge_detection_top1_all"]["delta"],
        "top1_delta": metrics["top1"]["delta"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
