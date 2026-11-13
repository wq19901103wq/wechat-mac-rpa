#!/usr/bin/env python3
"""Read-only consistency checks for the canonical Judge/C0 workflow."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "docs/03-guides/JUDGE_C0_CANONICAL_WORKFLOW.json"
CANONICAL_DOC = ROOT / "docs/03-guides/JUDGE_C0_CANONICAL_WORKFLOW.md"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def require_historical_notice(relative: str) -> None:
    path = ROOT / relative
    require(path.exists(), f"superseded document missing: {relative}")
    if path.suffix == ".json":
        notice = json.loads(path.read_text(encoding="utf-8")).get("authority_notice") or {}
        require(notice.get("status") == "historical_only", f"historical notice missing: {relative}")
        require(notice.get("canonical_workflow") == str(CANONICAL_DOC.relative_to(ROOT)), f"canonical link missing: {relative}")
    else:
        opening = path.read_text(encoding="utf-8")[:500]
        require("历史口径" in opening and "JUDGE_C0_CANONICAL_WORKFLOW" in opening, f"historical notice missing: {relative}")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(config["authority"] == "canonical", "workflow config is not canonical")
    require(config["data_model"]["official_sets"] == ["development", "fixed_test"], "official set model changed")
    require(not config["data_model"]["additional_official_test_sets_allowed"], "additional test sets enabled")

    unified = config["unified_execution"]
    require(unified["status"] == "active", "unified execution is not active")
    require(unified["entrypoint"] == "scripts/run_canonical_iteration.py", "canonical entrypoint changed")
    require((ROOT / unified["entrypoint"]).is_file(), "canonical entrypoint is missing")
    require(unified["adopted_c0_config"] == "config/current_c0.json", "adopted C0 config entry changed")
    require(unified["production_and_experiments_share_factory"], "production and experiments no longer share the C0 factory")
    require(unified["judge_artifacts_must_bind_current_c0_baseline_id"], "Judge artifacts are not bound to current C0")
    require(unified["missing_or_mismatched_baseline_id_fails_before_result_use"], "stale Judge result rejection disabled")
    require(not unified["prior_judge_results_from_old_or_unknown_c0_valid"], "old-C0 Judge results became valid")

    from scripts.run_canonical_iteration import preflight
    from src.badcase.judge_config import (
        current_judge_baseline_id,
        load_current_judge,
        missing_official_baseline_fields,
        require_official_judge_runnable,
    )
    from src.reply.c0_config import current_c0_baseline_id

    baseline_id = current_c0_baseline_id()
    require(unified["current_c0_baseline_id"] == baseline_id, "canonical workflow C0 baseline ID drifted")
    require(config["judge"]["required_current_c0_baseline_id"] == baseline_id, "Judge required C0 baseline ID drifted")
    require(not config["judge"]["prior_development_scores_and_candidate_decisions_valid"], "invalid old Judge conclusions became active")
    official = config["judge"]["official_baseline"]
    require(official["config"] == "config/current_judge.json", "official Judge config entry changed")
    require((ROOT / official["config"]).is_file(), "official Judge config is missing")
    judge = load_current_judge(ROOT / official["config"])
    require(judge["status"] == "pending_regeneration", "official Judge must remain pending until the 240-case baseline is complete")
    require(config["judge"]["current"] is None, "a pending Judge was recorded as current")
    require(official["id"] is None, "a pending Judge was assigned an official baseline ID")
    try:
        current_judge_baseline_id(ROOT / official["config"])
    except RuntimeError:
        pass
    else:
        require(False, "pending 240-case Judge unexpectedly produced an official baseline ID")
    require(judge["required_current_c0_baseline_id"] == baseline_id, "official Judge is not bound to current C0")
    require(judge["prediction_type"] == "direct_llm_pairwise", "official Judge is not a direct LLM A/B comparison")
    require(judge["official_judge_baseline"] is False, "pending Judge was marked as the official baseline")
    require(judge.get("experiment_type") is None, "an experiment is configured as the official Judge")
    require(official["status"] == judge["status"], "documented Judge status drifted")
    require(official["total_cases"] == judge["dataset"]["total_cases"] == 240, "official Judge target is not 240")
    require(official["private_cases"] is judge["dataset"]["private_cases"] is None, "private allocation was guessed")
    require(official["group_cases"] is judge["dataset"]["group_cases"] is None, "group allocation was guessed")
    require(official["dataset_path"] is judge["dataset"]["path"] is None, "official Judge dataset path was guessed")
    require(official["dataset_sha256"] is judge["dataset"]["sha256"] is None, "official Judge dataset hash was guessed")
    require(official["profile_name"] == judge["profile"]["name"], "official Judge profile name drifted")
    require(official["profile_sha256"] == judge["profile"]["sha256"], "official Judge profile hash drifted")
    require(official["required_current_c0_baseline_id"] == baseline_id, "documented Judge C0 binding drifted")
    require(official["runnable_as_official"] is judge["runnable_as_official"], "documented Judge runnable state drifted")
    missing = missing_official_baseline_fields(judge)
    require(official["missing_official_baseline_fields"] == missing, "documented missing Judge baseline fields drifted")
    require(official["unknown_parameters_must_not_be_guessed"], "unknown Judge parameters may be guessed")
    require(not official["feature_classifier_experiments_are_official_judge"], "feature-classifier experiments became official")
    if official["runnable_as_official"]:
        require(not missing, "official Judge is marked runnable with missing parameters")
        require_official_judge_runnable(judge)
    else:
        require(bool(missing), "official Judge is blocked without a recorded missing parameter")
        try:
            require_official_judge_runnable(judge)
        except RuntimeError:
            pass
        else:
            require(False, "official Judge did not fail closed")

    c0_preflight = preflight("c0", "development", "validator-smoke-test")
    require(c0_preflight["status"] == "ready", "C0 canonical preflight is not ready")
    require(c0_preflight["model_calls_started"] == 0, "C0 preflight started a model call")
    require(c0_preflight["current_c0_baseline_id"] == baseline_id, "C0 preflight used another baseline")
    if not official["runnable_as_official"]:
        try:
            preflight("judge", "development", "validator-smoke-test")
        except RuntimeError:
            pass
        else:
            require(False, "Judge canonical preflight did not fail closed")

    feature_experiment_source = (ROOT / "scripts/run_judge_feature_classifier_experiment.py").read_text(encoding="utf-8")
    require('"experiment_type": "feature_classifier_experiment"' in feature_experiment_source, "small-model results lack an experiment marker")
    require('"official_judge_baseline": False' in feature_experiment_source, "small-model results can masquerade as official Judge results")
    require(config["judge"]["next_baseline"] == {
        "status": "pending_regeneration",
        "private_cases": None,
        "group_cases": None,
        "total_cases": 240,
        "fixed_c0_240_used_for_training_or_rule_selection": False,
    }, "next Judge baseline contract changed")
    require(config["judge"]["distinct_240_sets"]["interchangeable"] is False, "different 240-case datasets became interchangeable")

    development = config["development"]
    require(sha256(development["manifest"]) == development["manifest_sha256_at_review"], "development manifest changed")
    development_manifest = load(development["manifest"])
    counts = development_manifest["counts"]
    require(counts["development_pool"] == development["current_total"], "development total drifted")
    require(counts["relationships"]["group"] == development["current_group"], "development group count drifted")
    require(counts["relationships"]["acquaintance"] == development["current_private_or_acquaintance"], "development private count drifted")
    require(counts["exact_holdout_overlap"] == 0, "development overlaps the fixed test")

    fixed = config["fixed_test"]
    require(sha256(fixed["historical_source_manifest"]) == fixed["historical_source_manifest_sha256"], "historical 240-case manifest changed")
    historical = load(fixed["historical_results"])["summary"]
    require(historical["reserved_total"] == fixed["historical_reserved"], "historical reserved count drifted")
    require(historical["judged"] == fixed["historical_scorable"], "historical scorable count drifted")
    require(historical["segments"]["group"]["scorable"] == fixed["historical_group_scorable"], "historical group count drifted")
    require(historical["segments"]["private"]["scorable"] == fixed["historical_private_scorable"], "historical private count drifted")

    target = fixed["target"]
    require(target["scorable_total"] == target["group"] + target["private"], "target group/private do not sum to total")
    require(target["scorable_total"] == 240 and target["group"] == 200 and target["private"] == 40, "locked target distribution changed")
    require(config["c0"]["forced_reply_in_development"] and config["c0"]["forced_reply_in_fixed_test"], "forced-reply scope changed")
    require(config["c0"]["few_shot_selection"] == "situation_relationship_reply_action", "few-shot matching changed")
    require(config["c0"]["few_shot_count_max"] == 3 and config["c0"]["exclude_current_case"], "few-shot limit or self-exclusion changed")
    protocol = config["c0"]["experiment_protocol"]
    require(protocol["trigger_precheck_cases"] == 3 and protocol["trigger_precheck_required"] == 3, "trigger precheck counts changed")
    require(protocol["trigger_precheck_is_quality_evidence"] is False, "trigger precheck cannot become quality evidence")
    require(protocol["fixed_total_generation_count"] is False, "total generation count must remain adaptive")
    require(protocol["generate_only_new_candidate_on_development"] is True, "old development versions may not be regenerated by default")
    require(protocol["reuse_saved_development_outputs"] is True, "saved development outputs must be reused")
    require(protocol["rerun_full_development_before_fixed_test"] is False, "full development rerun before fixed test is forbidden")
    require(protocol["record_every_version_development_outputs"] is True, "every development version must be recorded")
    require(protocol["compare_incremental_to_iteration_baseline"] is True, "incremental baseline comparison disabled")
    require(protocol["compare_cumulative_to_official_baseline"] is True, "cumulative official-baseline comparison disabled")
    require(protocol["sum_intermediate_net_wins"] is False, "intermediate net wins may not be summed")
    require(protocol["global_feature_large_development_triggered"] == 80 and protocol["global_feature_large_development_untriggered"] == 0, "global feature development split changed")
    require(protocol["initial_runs_per_arm"] == 1 and protocol["repeat_on_judge_flip_per_arm"] == 3, "adaptive repeat rules changed")
    require(protocol["single_arm_failure_max_retries"] == 2, "single-arm retry rule changed")
    require(protocol["large_development_ab_total"] == 80, "large development A/B total changed")
    require(protocol["large_development_triggered"] == 40 and protocol["large_development_untriggered"] == 40, "large development A/B strata changed")
    require(protocol["large_development_required_pairs"] == 80, "large development completion gate changed")
    require(protocol["large_development_triggered_net_wins_minimum"] == 1, "development benefit gate changed")
    require(protocol["large_development_control_net_losses_maximum"] == 1, "development regression gate changed")
    require(protocol["paired_test"] == "two_sided_exact_sign_test" and protocol["statistical_stability_p_max"] == 0.05, "paired stability rule changed")
    acceptance = config["c0"]["acceptance"]
    require(acceptance["iteration_baseline_net_wins_minimum"] == 1, "iteration-baseline gate changed")
    require(acceptance["fixed_test_entry_cumulative_net_wins_minimum"] == 4, "fixed-test entry gain changed")
    require(acceptance["fixed_test_entry_development_pairs"] == 80, "fixed-test entry comparison size changed")
    require(acceptance["judge_identification_rate_must_strictly_decrease"], "C0 improvement gate changed")
    require(acceptance["new_verified_hard_contradictions"] == 0, "hard-contradiction gate changed")
    require(acceptance["fixed_test_required_pairs"] == 240, "fixed-test completion gate changed")
    require(not acceptance["human_ab_is_gate"] and not acceptance["rt_is_gate"], "non-gating diagnostics became hard gates")
    hard = config["hard_contradiction"]
    require(hard["first_review_repeats_per_arm"] == 6 and hard["second_review_repeats_per_arm"] == 12, "hard-error repeat counts changed")
    require(hard["attribution_test"] == "one_sided_fisher_candidate_greater" and hard["attribution_p_max"] == 0.05, "hard-error attribution rule changed")
    require(config["reporting"]["single_mobile_page"], "multiple result pages enabled")
    require(config["reporting"]["url"] == "https://judge.wq19901103wqwechat.cn/benchmark/reply/current-c0", "fixed result URL changed")
    require(config["reporting"]["public_result_basis"] == "fixed_test_only", "development sample scores became public results")
    require(not config["reporting"]["development_sample_scores_visible"], "development sample scores became visible")
    require(config["reporting"]["fixed_test_incomplete_label_required_until_240_effective"], "incomplete fixed test may be presented as complete")
    require(config["reporting"]["judge_outcome_filters_required"] == ["judge_correct_identified_c0", "judge_wrong_treated_c0_as_human"], "Judge result filters changed")
    require(config["bot"] == {"start_allowed": False, "send_allowed": False}, "Bot safety lock changed")

    for relative in config["superseded_documents"]:
        require_historical_notice(relative)

    if fixed["status"] == "frozen":
        require(bool(fixed["final_manifest"]), "frozen test has no final manifest")
        require(bool(fixed["final_manifest_sha256"]), "frozen test has no final hash")
        require(sha256(fixed["final_manifest"]) == fixed["final_manifest_sha256"], "frozen test manifest changed")

    print("PASS: canonical Judge/C0 workflow is internally consistent")
    print(f"STATUS: C0 fixed test = {fixed['status']}; target = 240 effective (200 group / 40 private)")
    print("STATUS: official Judge = pending_regeneration; target = 240 (private/group allocation pending)")


if __name__ == "__main__":
    main()
