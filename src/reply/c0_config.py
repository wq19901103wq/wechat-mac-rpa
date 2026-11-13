"""Load and verify the one adopted C0 baseline."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/current_c0.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"current C0 {label} is missing: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"current C0 {label} sha256 mismatch: expected {expected_sha256}, got {actual}"
        )


@lru_cache(maxsize=4)
def load_current_c0(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "adopted":
        raise RuntimeError("current C0 config must be schema_version=1 and status=adopted")

    private = config.get("private", {})
    group = config.get("group", {})
    if private.get("model") != "deepseek-v4-flash" or private.get("timeout_seconds") != 20:
        raise RuntimeError("current C0 private baseline must be deepseek-v4-flash / 20 seconds")
    if group.get("model") != "doubao-seed-2.1-turbo" or group.get("timeout_seconds") != 360:
        raise RuntimeError("current C0 group baseline must be doubao-seed-2.1-turbo / 360 seconds")
    execution = config.get("execution", {})
    if execution != {"enable_react_tools": False, "enable_self_refine": False}:
        raise RuntimeError("current C0 execution baseline must disable ReAct tools and self-refine")

    skills = config.get("skills")
    if not isinstance(skills, dict):
        raise RuntimeError("current C0 skills must be an object")
    disabled_skills = skills.get("disabled")
    if not isinstance(disabled_skills, list) or not all(isinstance(name, str) for name in disabled_skills):
        raise RuntimeError("current C0 skills.disabled must be a list of strings")
    normalized_disabled_skills = [name.strip() for name in disabled_skills]
    if any(not name for name in normalized_disabled_skills):
        raise RuntimeError("current C0 skills.disabled cannot contain blank names")
    if len(normalized_disabled_skills) != len(set(normalized_disabled_skills)):
        raise RuntimeError("current C0 skills.disabled cannot contain duplicates")
    for name in normalized_disabled_skills:
        if not (PROJECT_ROOT / "skills" / name / "SKILL.md").is_file():
            raise RuntimeError(f"current C0 disabled skill is unknown: {name}")

    few_shot = config.get("few_shot", {})
    if (
        few_shot.get("enabled") is not True
        or few_shot.get("count") != 3
        or few_shot.get("max_chars") != 2500
        or few_shot.get("rows") != 80530
    ):
        raise RuntimeError("current C0 few-shot baseline must be enabled / 3 rows / 2500 chars / 80530 pool rows")
    few_shot_path = _resolve(str(few_shot.get("path", "")))
    report_path = _resolve(str(few_shot.get("report_path", "")))
    evidence = config.get("adoption_evidence", {})
    evidence_path = _resolve(str(evidence.get("path", "")))
    _verify_file(few_shot_path, str(few_shot.get("sha256", "")), "few-shot pool")
    with few_shot_path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    if rows != 80530:
        raise RuntimeError(f"current C0 few-shot row count mismatch: expected 80530, got {rows}")
    _verify_file(report_path, str(few_shot.get("report_sha256", "")), "few-shot report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("review_status") != few_shot.get("required_review_status"):
        raise RuntimeError("current C0 few-shot report is not approved")
    _verify_file(evidence_path, str(evidence.get("sha256", "")), "adoption evidence")

    result = dict(config)
    result["config_path"] = str(path)
    result["few_shot_path"] = str(few_shot_path)
    result["skills"] = {"disabled": normalized_disabled_skills}
    return result


def current_c0_baseline_id(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = dict(load_current_c0(config_path))
    config.pop("config_path", None)
    config.pop("few_shot_path", None)
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "c0-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def clear_current_c0_cache() -> None:
    load_current_c0.cache_clear()
