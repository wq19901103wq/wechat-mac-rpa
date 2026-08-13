"""OCR Quality benchmark：私有真实 fixture、缓存运行和分层指标。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private_benchmarks" / "ocr"
FIXTURE_DIR = PRIVATE_ROOT / "fixtures"
CACHE_DIR = PRIVATE_ROOT / "cache"


@dataclass
class OCRBenchmarkCase:
    name: str
    cohort: str
    source: str
    screenshot_path: Path
    expected: dict[str, Any]
    description: str = ""


@dataclass
class OCRBenchmarkResult:
    case_name: str
    cohort: str
    source: str
    screenshot_path: Path
    expected_chat_name: str
    actual_chat_name: str
    chat_name_match: bool
    expected_message_count: int
    actual_message_count: int
    message_count_match: bool
    sender_correct: int
    sender_total: int
    sender_accuracy: float
    text_correct: int
    text_total: int
    text_accuracy: float
    chat_list_correct: int
    chat_list_total: int
    chat_list_accuracy: float
    message_details: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    error: str = ""


def _normalize_chat_name(value: Any) -> str:
    text = str(value or "").replace("（", "(").replace("）", ")")
    return text.replace(" ", "").replace("\u3000", "").strip()


def _normalize_text(value: Any) -> str:
    text = str(value or "")
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _text_matches(expected: str, actual: str, mode: str = "similarity") -> bool:
    expected_text = _normalize_text(expected)
    actual_text = _normalize_text(actual)
    if mode == "exact":
        return expected_text == actual_text
    return SequenceMatcher(None, expected_text, actual_text).ratio() >= 0.8


def load_cases(fixture_dir: Path = FIXTURE_DIR) -> list[OCRBenchmarkCase]:
    """加载 Git 忽略目录中的真实截图与同名 Ground Truth JSON。"""
    cases: list[OCRBenchmarkCase] = []
    if not fixture_dir.exists():
        return cases
    metadata_paths = list(fixture_dir.glob("*.json"))
    metadata_paths.extend((fixture_dir / "legacy" / "errors").glob("error_*.json"))
    for metadata_path in sorted(metadata_paths):
        if metadata_path.stem in {"prescreen_annotations", "small_scene", "medium_scene", "large_scene"}:
            continue
        screenshot_path = metadata_path.with_suffix(".png")
        if not screenshot_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "messages" not in metadata:
            continue
        messages = []
        for message in metadata.get("messages", []):
            text = str(message.get("text", ""))
            check_mode = str(message.get("check_mode", message.get("check", "exact")))
            if check_mode == "exact" and ("**" in text or "__" in text):
                check_mode = "similarity"
            messages.append({
                "sender": str(message.get("sender", "")),
                "text": text,
                "check_mode": check_mode,
            })
        name = metadata_path.stem
        cases.append(OCRBenchmarkCase(
            name=name,
            cohort="regression" if name.startswith("regression_") else "representative",
            source="private-real",
            screenshot_path=screenshot_path,
            expected={
                "chat_name": metadata.get("chat_name", ""),
                "chat_list": metadata.get("chat_list", []),
                "messages": messages,
            },
            description=str(metadata.get("description", "")),
        ))
    return cases


def evaluate_case(case: OCRBenchmarkCase, api_result: dict[str, Any]) -> OCRBenchmarkResult:
    """按历史严格标准评估：名称、数量、sender 全对，text 至少 80%。"""
    expected_messages = case.expected.get("messages", [])
    actual_messages = api_result.get("messages", []) if isinstance(api_result, dict) else []
    expected_count = len(expected_messages)
    actual_count = len(actual_messages)
    aligned_total = max(expected_count, actual_count)
    sender_correct = 0
    text_correct = 0
    details = []

    for index in range(aligned_total):
        expected = expected_messages[index] if index < expected_count else {}
        actual = actual_messages[index] if index < actual_count else {}
        sender_ok = str(expected.get("sender", "")) == str(actual.get("sender", ""))
        text_ok = _text_matches(
            str(expected.get("text", "")),
            str(actual.get("text", "")),
            str(expected.get("check_mode", "similarity")),
        )
        sender_correct += int(sender_ok)
        text_correct += int(text_ok)
        details.append({
            "index": index,
            "expected_sender": expected.get("sender", ""),
            "actual_sender": actual.get("sender", ""),
            "sender_ok": sender_ok,
            "expected_text": expected.get("text", ""),
            "actual_text": actual.get("text", ""),
            "text_ok": text_ok,
        })

    sender_accuracy = sender_correct / aligned_total if aligned_total else 1.0
    text_accuracy = text_correct / aligned_total if aligned_total else 1.0
    expected_chat_name = str(case.expected.get("chat_name", ""))
    actual_chat_name = str(api_result.get("chat_name", ""))
    chat_name_match = (
        _normalize_chat_name(expected_chat_name) == _normalize_chat_name(actual_chat_name)
    )
    message_count_match = expected_count == actual_count
    expected_chat_list = case.expected.get("chat_list", [])
    actual_chat_list = api_result.get("chat_list", [])
    actual_by_name = {
        _normalize_chat_name(item.get("nickname", "")): str(item.get("unread_count", ""))
        for item in actual_chat_list
    }
    chat_list_correct = sum(
        actual_by_name.get(_normalize_chat_name(item.get("nickname", "")))
        == str(item.get("unread_count", ""))
        for item in expected_chat_list
    )
    chat_list_total = len(expected_chat_list)
    chat_list_accuracy = (
        chat_list_correct / chat_list_total if chat_list_total else 1.0
    )
    passed = (
        chat_name_match
        and message_count_match
        and sender_accuracy == 1.0
        and text_accuracy >= 0.8
    )
    return OCRBenchmarkResult(
        case_name=case.name,
        cohort=case.cohort,
        source=case.source,
        screenshot_path=case.screenshot_path,
        expected_chat_name=expected_chat_name,
        actual_chat_name=actual_chat_name,
        chat_name_match=chat_name_match,
        expected_message_count=expected_count,
        actual_message_count=actual_count,
        message_count_match=message_count_match,
        sender_correct=sender_correct,
        sender_total=aligned_total,
        sender_accuracy=sender_accuracy,
        text_correct=text_correct,
        text_total=aligned_total,
        text_accuracy=text_accuracy,
        chat_list_correct=chat_list_correct,
        chat_list_total=chat_list_total,
        chat_list_accuracy=chat_list_accuracy,
        message_details=details,
        passed=passed,
    )


def _cache_path(case: OCRBenchmarkCase, model: str) -> Path:
    digest = hashlib.sha256()
    digest.update(case.screenshot_path.read_bytes())
    digest.update(json.dumps(case.expected, sort_keys=True, ensure_ascii=False).encode())
    digest.update(model.encode())
    return CACHE_DIR / f"{case.name}_{digest.hexdigest()[:16]}.json"


def _legacy_cache_path(case: OCRBenchmarkCase) -> Path:
    digest = hashlib.md5(case.screenshot_path.read_bytes(), usedforsecurity=False).hexdigest()
    return FIXTURE_DIR / "ocr_cache" / f"{digest}.json"


def run_benchmark(
    use_api: bool = False,
    model: str = "qwen3.6-flash",
    cases: list[OCRBenchmarkCase] | None = None,
) -> tuple[list[OCRBenchmarkResult], list[str]]:
    """运行私有 OCR benchmark，返回已评估结果和缺少缓存的 case 名。"""
    selected_cases = cases if cases is not None else load_cases()
    results = []
    missing_cache = []
    client = None
    if use_api:
        from src.perception.smart_pipeline import _QwenAPIClient

        client = _QwenAPIClient(model=model)

    for case in selected_cases:
        cache_path = _cache_path(case, model)
        if use_api:
            assert client is not None
            try:
                api_result = client.recognize(str(case.screenshot_path))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(api_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                results.append(OCRBenchmarkResult(
                    case_name=case.name,
                    cohort=case.cohort,
                    source=case.source,
                    screenshot_path=case.screenshot_path,
                    expected_chat_name=str(case.expected.get("chat_name", "")),
                    actual_chat_name="",
                    chat_name_match=False,
                    expected_message_count=len(case.expected.get("messages", [])),
                    actual_message_count=0,
                    message_count_match=False,
                    sender_correct=0,
                    sender_total=len(case.expected.get("messages", [])),
                    sender_accuracy=0.0,
                    text_correct=0,
                    text_total=len(case.expected.get("messages", [])),
                    text_accuracy=0.0,
                    chat_list_correct=0,
                    chat_list_total=len(case.expected.get("chat_list", [])),
                    chat_list_accuracy=0.0,
                    passed=False,
                    error=str(exc),
                ))
                continue
        elif cache_path.exists() or _legacy_cache_path(case).exists():
            readable_cache = cache_path if cache_path.exists() else _legacy_cache_path(case)
            api_result = json.loads(readable_cache.read_text(encoding="utf-8"))
        else:
            missing_cache.append(case.name)
            continue
        results.append(evaluate_case(case, api_result))
    return results, missing_cache


def compute_metrics(results: list[OCRBenchmarkResult]) -> dict[str, Any]:
    """计算字段指标；整 case 通过率只在同一 cohort 内解释。"""
    evaluated = [result for result in results if not result.error]
    if not evaluated:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": None,  # nosec B105 - metric value, not a credential
            "chat_name_accuracy": None,
            "message_count_accuracy": None,
            "sender_accuracy": None,
            "text_accuracy": None,
            "chat_list_accuracy": None,
            "api_errors": len(results),
        }
    total = len(evaluated)
    passed = sum(result.passed for result in evaluated)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total,
        "chat_name_accuracy": sum(result.chat_name_match for result in evaluated) / total,
        "message_count_accuracy": sum(result.message_count_match for result in evaluated) / total,
        "sender_accuracy": sum(result.sender_accuracy for result in evaluated) / total,
        "text_accuracy": sum(result.text_accuracy for result in evaluated) / total,
        "chat_list_accuracy": sum(result.chat_list_accuracy for result in evaluated) / total,
        "api_errors": len(results) - total,
    }


def build_summary(
    cases: list[OCRBenchmarkCase],
    results: list[OCRBenchmarkResult],
    missing_cache: list[str],
) -> dict[str, Any]:
    """分别汇总代表性与回归场景，避免混成一个“准确率”。"""
    cohorts = {}
    for cohort in ("representative", "regression"):
        cohort_cases = [case for case in cases if case.cohort == cohort]
        cohort_results = [result for result in results if result.cohort == cohort]
        cohorts[cohort] = {
            "configured_cases": len(cohort_cases),
            **compute_metrics(cohort_results),
        }
    return {
        "status": "available" if len(results) == len(cases) and not missing_cache else "unavailable",
        "configured_cases": len(cases),
        "evaluated_cases": len(results),
        "missing_cache": missing_cache,
        "cohorts": cohorts,
        "field_metrics_all": compute_metrics(results),
    }
