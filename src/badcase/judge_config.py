"""Load and verify the adopted official LLM-feature-plus-logistic Judge config."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.reply.c0_config import current_c0_baseline_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/current_judge.json"
OFFICIAL_JUDGE_TOTAL_CASES = 240
REQUIRED_EXECUTION_FIELDS = (
    "model",
    "reasoning_effort",
    "timeout_seconds",
    "codex_cli_version",
)
REQUIRED_PROFILE_FIELDS = ("path", "name", "sha256")
REQUIRED_DATASET_FIELDS = ("path", "sha256", "private_cases", "group_cases")
REQUIRED_REFERENCE_FIELDS = ("path", "sha256")
REQUIRED_CORRECTION_FIELDS = (
    "selected_model_name",
    "model",
    "input_feature_count",
    "fit_intercept",
    "regularization",
    "C",
    "flip_confidence_threshold",
)
OFFICIAL_PREDICTION_TYPE = "llm_boolean_features_with_logistic_correction"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def missing_official_execution_fields(config: dict[str, Any]) -> list[str]:
    execution = config.get("execution") or {}
    return [f"execution.{field}" for field in REQUIRED_EXECUTION_FIELDS if execution.get(field) in (None, "")]


def missing_official_baseline_fields(config: dict[str, Any]) -> list[str]:
    profile = config.get("profile") or {}
    dataset = config.get("dataset") or {}
    reference = config.get("reference") or {}
    correction = config.get("correction") or {}
    artifact = correction.get("artifact") or {}
    sidecars = config.get("metadata_sidecars") or {}
    missing = [f"profile.{field}" for field in REQUIRED_PROFILE_FIELDS if profile.get(field) in (None, "")]
    missing.extend(f"dataset.{field}" for field in REQUIRED_DATASET_FIELDS if dataset.get(field) in (None, ""))
    missing.extend(f"reference.{field}" for field in REQUIRED_REFERENCE_FIELDS if reference.get(field) in (None, ""))
    missing.extend(
        f"metadata_sidecars.{sidecar}.{field}"
        for sidecar in ("answers", "replay")
        for field in ("path", "sha256")
        if (sidecars.get(sidecar) or {}).get(field) in (None, "")
    )
    missing.extend(
        f"correction.{field}" for field in REQUIRED_CORRECTION_FIELDS if correction.get(field) in (None, "")
    )
    missing.extend(
        f"correction.artifact.{field}" for field in ("path", "sha256") if artifact.get(field) in (None, "")
    )
    missing.extend(missing_official_execution_fields(config))
    return missing


def _verify_file(path_value: Any, expected_sha256: Any, description: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"official Judge {description} path is missing")
    path = _resolve(path_value)
    if not path.is_file():
        raise RuntimeError(f"official Judge {description} is missing: {path}")
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"official Judge {description} sha256 mismatch")
    return path


def _verify_correction_artifact(config: dict[str, Any]) -> None:
    correction = config["correction"]
    artifact = correction["artifact"]
    path = _verify_file(artifact["path"], artifact["sha256"], "correction artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("final_model") or {}
    expected = {
        "selected_model_name": "symmetric_logistic_refined_boolean_crosses",
        "model": "LogisticRegression",
        "input_feature_count": 524,
        "fit_intercept": False,
        "regularization": "L2",
        "C": 1.0,
    }
    if any(correction.get(field) != value for field, value in expected.items()):
        raise RuntimeError("official Judge correction configuration is not the frozen logistic model")
    if any(
        (payload.get(field) if field == "selected_model_name" else model.get(field)) != value
        for field, value in expected.items()
    ):
        raise RuntimeError("official Judge correction artifact does not match the frozen logistic model")
    threshold = correction.get("flip_confidence_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.5 <= threshold <= 1:
        raise RuntimeError("official Judge flip_confidence_threshold must be between 0.5 and 1")


def require_official_judge_runnable(config: dict[str, Any]) -> None:
    missing = missing_official_baseline_fields(config)
    if missing:
        raise RuntimeError(
            "official Judge 240-case baseline is not runnable because baseline requirements are missing: "
            + ", ".join(missing)
        )
    if config.get("status") != "adopted" or config.get("official_judge_baseline") is not True:
        raise RuntimeError("official Judge 240-case baseline has not been adopted")
    if config.get("runnable_as_official") is not True:
        raise RuntimeError("official Judge is disabled until runnable_as_official=true")
    timeout = (config.get("execution") or {}).get("timeout_seconds")
    if not isinstance(timeout, int) or timeout <= 0:
        raise RuntimeError("official Judge timeout_seconds must be a positive integer")


@lru_cache(maxsize=4)
def load_current_judge(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    require_runnable: bool = False,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_type") == "feature_classifier_experiment":
        raise RuntimeError("feature-classifier experiments cannot be loaded as the official Judge")
    if config.get("schema_version") != 1 or config.get("status") != "adopted":
        raise RuntimeError("current Judge config must use schema_version=1 and status=adopted")
    if config.get("prediction_type") != OFFICIAL_PREDICTION_TYPE:
        raise RuntimeError("official Judge must use LLM boolean features with frozen logistic correction")
    if config.get("required_current_c0_baseline_id") != current_c0_baseline_id():
        raise RuntimeError("official Judge is not bound to the currently adopted C0 baseline")
    if (config.get("execution") or {}).get("provider") != "codex_cli":
        raise RuntimeError("official Judge provider must be codex_cli")

    dataset = config.get("dataset") or {}
    if dataset.get("total_cases") != OFFICIAL_JUDGE_TOTAL_CASES:
        raise RuntimeError("official Judge dataset must contain exactly 240 cases")
    private_cases = dataset.get("private_cases")
    group_cases = dataset.get("group_cases")
    if private_cases is not None or group_cases is not None:
        if not isinstance(private_cases, int) or not isinstance(group_cases, int):
            raise RuntimeError("official Judge private/group counts must both be integers")
        if private_cases < 0 or group_cases < 0 or private_cases + group_cases != OFFICIAL_JUDGE_TOTAL_CASES:
            raise RuntimeError("official Judge private/group counts must sum to 240")

    dataset_path_value = dataset.get("path")
    if dataset_path_value not in (None, ""):
        _verify_file(dataset_path_value, dataset.get("sha256"), "dataset")

    profile = config.get("profile") or {}
    profile_path_value = profile.get("path")
    profile_path: Path | None = None
    if profile_path_value not in (None, ""):
        profile_path = _verify_file(profile_path_value, profile.get("sha256"), "profile")
        profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_payload.get("name") != profile.get("name") or profile.get("name") != config.get("name"):
            raise RuntimeError("official Judge profile name mismatch")

    if config.get("missing_official_baseline_fields") != missing_official_baseline_fields(config):
        raise RuntimeError("official Judge missing-field record is stale")
    if config.get("official_judge_baseline") is not True:
        raise RuntimeError("adopted Judge config is not marked as the official Judge baseline")
    if missing_official_baseline_fields(config):
        raise RuntimeError("adopted official Judge config is incomplete")
    _verify_file((config.get("reference") or {}).get("path"), (config.get("reference") or {}).get("sha256"), "reference")
    for sidecar in (config.get("metadata_sidecars") or {}).values():
        _verify_file(sidecar.get("path"), sidecar.get("sha256"), "metadata sidecar")
    _verify_correction_artifact(config)
    if require_runnable:
        require_official_judge_runnable(config)

    result = dict(config)
    result["config_path"] = str(path)
    result["profile_path"] = str(profile_path) if profile_path is not None else None
    return result


def current_judge_baseline_id(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = dict(load_current_judge(config_path))
    if config.get("status") != "adopted" or config.get("official_judge_baseline") is not True:
        raise RuntimeError("official Judge baseline ID is unavailable until the 240-case baseline is adopted")
    config.pop("config_path", None)
    config.pop("profile_path", None)
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "judge-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def clear_current_judge_cache() -> None:
    load_current_judge.cache_clear()
