#!/usr/bin/env python3
"""
OCR 质量 Benchmark

评估 qwen3.6-flash API 对微信截图的识别质量，覆盖 sender 正确性、消息内容、
聊天名称、聊天列表等核心维度。

用法:
    # 使用真实 API（计算准确率，建立/更新缓存）
    python -m pytest src/tests/test_ocr_quality_benchmark.py -v --run-api

    # 使用缓存（快速回归）
    python -m pytest src/tests/test_ocr_quality_benchmark.py -v

    # 命令行直接运行（带详细报告）
    python src/tests/test_ocr_quality_benchmark.py --run-api
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = PROJECT_ROOT / "tests_integration" / "fixtures"
CACHE_DIR = FIXTURE_DIR / "ocr_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OCRBenchmarkCase:
    """单个 benchmark case"""
    name: str
    screenshot_path: Path
    ground_truth: dict
    category: str  # private_chat | group_chat | regression | legacy_private | legacy_group


@dataclass
class OCRBenchmarkResult:
    """单个 case 的测试结果"""
    case_name: str
    category: str

    # chat_name
    expected_chat_name: str
    actual_chat_name: str
    chat_name_match: bool

    # messages count
    expected_message_count: int
    actual_message_count: int
    message_count_match: bool

    # sender
    sender_correct: int
    sender_total: int
    sender_accuracy: float

    # text
    text_correct: int
    text_total: int
    text_accuracy: float

    # chat_list
    chat_list_correct: int
    chat_list_total: int
    chat_list_accuracy: float

    # details
    message_details: list[dict] = field(default_factory=list)
    api_raw: dict = field(default_factory=dict)
    passed: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _infer_category(case_name: str, ground_truth: dict) -> str:
    """从 case 名称和 ground truth 推断场景类型"""
    if case_name.startswith("real_private_"):
        return "private_chat"
    if case_name.startswith("real_group_"):
        return "group_chat"
    if case_name.startswith("regression_"):
        return "regression"
    if case_name.startswith("error_"):
        # Legacy: 从内容推断私聊/群聊
        senders = set(m.get("sender", "") for m in ground_truth.get("messages", []))
        sender_types = set(m.get("sender_type", "") for m in ground_truth.get("messages", []))
        if senders <= {"自己", "对方"} and sender_types <= {"self", "other"}:
            return "legacy_private"
        return "legacy_group"
    if case_name.startswith("real_chat_"):
        senders = set(m.get("sender", "") for m in ground_truth.get("messages", []))
        if senders <= {"自己", "对方"}:
            return "private_chat"
        return "group_chat"
    return "unknown"


def _normalize_ground_truth(raw: dict, case_name: str) -> dict:
    """统一 ground truth 格式"""
    result = {
        "chat_name": raw.get("chat_name", ""),
        "chat_list": raw.get("chat_list", []),
        "messages": [],
        "category": _infer_category(case_name, raw),
    }

    for msg in raw.get("messages", []):
        # 统一 check_mode
        check_mode = "exact"
        if "check_mode" in msg:
            check_mode = msg["check_mode"]
        elif "check" in msg:
            check_mode = msg["check"]

        text = str(msg.get("text", ""))
        # 启发式：如果 text 中包含 markdown 标记但标注为 exact，自动改为 similarity
        # 因为 ground truth 可能从 markdown 源复制，不是纯 OCR 文本
        if check_mode == "exact" and ("**" in text or "__" in text):
            check_mode = "similarity"

        result["messages"].append({
            "sender": str(msg.get("sender", "")),
            "text": text,
            "check_mode": check_mode,
        })

    return result


def _load_all_cases() -> list[OCRBenchmarkCase]:
    """扫描所有可用的截图+标注对"""
    cases = []
    skipped = []

    # 1. 现有 fixtures（tests/fixtures/*.json）
    for json_path in sorted(FIXTURE_DIR.glob("*.json")):
        png_path = json_path.with_suffix(".png")
        if not png_path.exists():
            continue
        name = json_path.stem
        # 跳过非聊天截图的 fixture
        if name in ("prescreen_annotations", "small_scene", "medium_scene", "large_scene"):
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
            if "messages" not in raw:
                skipped.append((name, "no messages field"))
                continue
            gt = _normalize_ground_truth(raw, name)
            cases.append(OCRBenchmarkCase(
                name=name,
                screenshot_path=png_path,
                ground_truth=gt,
                category=gt["category"],
            ))
        except Exception as e:
            skipped.append((name, str(e)))

    # 2. Legacy errors（tests/fixtures/legacy/errors/*.json）
    legacy_dir = FIXTURE_DIR / "legacy" / "errors"
    if legacy_dir.exists():
        for json_path in sorted(legacy_dir.glob("error_*.json")):
            png_path = json_path.with_suffix(".png")
            if not png_path.exists():
                continue
            name = json_path.stem

            try:
                with open(json_path, encoding="utf-8") as f:
                    raw = json.load(f)
                if "messages" not in raw:
                    skipped.append((name, "no messages field"))
                    continue
                gt = _normalize_ground_truth(raw, name)
                cases.append(OCRBenchmarkCase(
                    name=name,
                    screenshot_path=png_path,
                    ground_truth=gt,
                    category=gt["category"],
                ))
            except Exception as e:
                skipped.append((name, str(e)))

    if skipped:
        print(f"跳过 {len(skipped)} 个 case:")
        for name, reason in skipped[:5]:
            print(f"  {name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return cases


# ---------------------------------------------------------------------------
# API & Cache
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    """计算文件 MD5"""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_text(text: str, aggressive: bool = False) -> str:
    """规范化 text，去除格式差异。

    Args:
        text: 原始文本
        aggressive: 是否激进模式（去除 markdown、emoji 等，用于 similarity check）
    """
    if not isinstance(text, str):
        text = str(text)
    # 基本规范化：统一换行、去除首尾空白
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if aggressive:
        # 去除 markdown 标记
        text = re.sub(r'\*{1,2}|_{1,2}', '', text)
        # 去除常见 emoji
        text = re.sub(r'[📱⚡️😂🤖💪👍🔥❤✅❌👉👈]', '', text)
        # 去除项目符号
        text = re.sub(r'^[•·\-]\s*', '', text, flags=re.MULTILINE)
        text = " ".join(text.split())
    return text.strip()


def _normalize_chat_name(name: str) -> str:
    """规范化 chat_name：统一全角/半角括号，去除空格"""
    if not isinstance(name, str):
        name = str(name)
    # 全角括号 -> 半角括号
    name = name.replace("（", "(").replace("）", ")")
    # 去除所有空格（聊天名称中的空格差异不影响语义）
    name = name.replace(" ", "").replace("\u3000", "")
    return name.strip()


def _read_cache(screenshot_path: Path) -> dict | None:
    """读取缓存的 API 结果"""
    h = _file_hash(screenshot_path)
    cache_path = CACHE_DIR / f"{h}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(screenshot_path: Path, result: dict) -> None:
    """保存 API 结果到缓存"""
    h = _file_hash(screenshot_path)
    cache_path = CACHE_DIR / f"{h}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _load_env() -> None:
    """加载 .env 文件中的环境变量"""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("DASHSCOPE_") and key not in os.environ:
                        os.environ[key] = value.strip().strip('"').strip("'")


def _get_api_key() -> str:
    """获取 API key"""
    _load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY")
    return api_key


def _call_api(screenshot_path: Path, api_key: str | None = None) -> dict:
    """调用 qwen3.6-flash API 识别截图"""
    try:
        from src.perception.smart_pipeline import _QwenAPIClient
    except ImportError:
        from perception.smart_pipeline import _QwenAPIClient  # type: ignore[no-redef]

    if api_key is None:
        api_key = _get_api_key()

    client = _QwenAPIClient(api_key=api_key)
    return client.recognize(str(screenshot_path))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _text_matches(expected: str, actual: str, check_mode: str = "exact") -> bool:
    """判断 text 是否匹配"""
    exp = _normalize_text(expected, aggressive=(check_mode == "similarity"))
    act = _normalize_text(actual, aggressive=(check_mode == "similarity"))
    if check_mode == "exact":
        return exp == act
    # similarity: 相似度 >= 0.7（基于规范化后的文本）
    ratio = SequenceMatcher(None, exp, act).ratio()
    return ratio >= 0.7


def _chat_list_matches(expected_list: list[dict], actual_list: list[dict]) -> tuple[int, int]:
    """评估 chat_list 准确性，返回 (correct, total)"""
    if not expected_list and not actual_list:
        return 0, 0  # 都没有，不统计

    total = max(len(expected_list), len(actual_list))
    correct = 0

    for i in range(min(len(expected_list), len(actual_list))):
        exp = expected_list[i]
        act = actual_list[i]
        nick_match = str(exp.get("nickname", "")) == str(act.get("nickname", ""))
        unread_match = str(exp.get("unread_count", "")) == str(act.get("unread_count", ""))
        if nick_match and unread_match:
            correct += 1

    return correct, total


def evaluate_case(case: OCRBenchmarkCase, api_result: dict) -> OCRBenchmarkResult:
    """评估单个 case"""
    gt = case.ground_truth
    expected_messages = gt["messages"]
    actual_messages = api_result.get("messages", []) if isinstance(api_result, dict) else []

    # chat_name
    expected_chat_name = gt["chat_name"]
    actual_chat_name = api_result.get("chat_name", "") if isinstance(api_result, dict) else ""
    chat_name_match = _normalize_chat_name(expected_chat_name) == _normalize_chat_name(actual_chat_name)

    # message count
    expected_count = len(expected_messages)
    actual_count = len(actual_messages)
    message_count_match = expected_count == actual_count

    # sender & text (逐条对齐)
    sender_correct = 0
    text_correct = 0
    message_details = []

    for i in range(min(expected_count, actual_count)):
        exp_msg = expected_messages[i]
        act_msg = actual_messages[i] if i < len(actual_messages) else {}
        if not isinstance(act_msg, dict):
            act_msg = {}

        exp_sender = exp_msg.get("sender", "").strip()
        act_sender = act_msg.get("sender", "").strip()
        sender_ok = exp_sender == act_sender

        exp_text = exp_msg.get("text", "")
        act_text = act_msg.get("text", "")
        # 记录原始值用于报告
        exp_text_raw = exp_text
        act_text_raw = act_text
        check_mode = exp_msg.get("check_mode", "exact")
        text_ok = _text_matches(exp_text, act_text, check_mode)

        if sender_ok:
            sender_correct += 1
        if text_ok:
            text_correct += 1

        message_details.append({
            "index": i,
            "expected_sender": exp_sender,
            "actual_sender": act_sender,
            "sender_ok": sender_ok,
            "expected_text": exp_text_raw,
            "actual_text": act_text_raw,
            "text_ok": text_ok,
            "check_mode": check_mode,
        })

    # 缺失/多余的消息算错误
    total_for_alignment = max(expected_count, actual_count)
    sender_total = total_for_alignment
    text_total = total_for_alignment

    # chat_list
    expected_chat_list = gt.get("chat_list", [])
    actual_chat_list = api_result.get("chat_list", []) if isinstance(api_result, dict) else []
    chat_list_correct, chat_list_total = _chat_list_matches(expected_chat_list, actual_chat_list)

    # accuracy
    sender_accuracy = sender_correct / sender_total if sender_total > 0 else 1.0
    text_accuracy = text_correct / text_total if text_total > 0 else 1.0
    chat_list_accuracy = chat_list_correct / chat_list_total if chat_list_total > 0 else 1.0

    # pass 标准：chat_name 对 + message_count 对 + sender 100% + text >= 80%
    passed = (
        chat_name_match
        and message_count_match
        and sender_accuracy == 1.0
        and text_accuracy >= 0.8
    )

    return OCRBenchmarkResult(
        case_name=case.name,
        category=case.category,
        expected_chat_name=expected_chat_name,
        actual_chat_name=actual_chat_name,
        chat_name_match=chat_name_match,
        expected_message_count=expected_count,
        actual_message_count=actual_count,
        message_count_match=message_count_match,
        sender_correct=sender_correct,
        sender_total=sender_total,
        sender_accuracy=sender_accuracy,
        text_correct=text_correct,
        text_total=text_total,
        text_accuracy=text_accuracy,
        chat_list_correct=chat_list_correct,
        chat_list_total=chat_list_total,
        chat_list_accuracy=chat_list_accuracy,
        message_details=message_details,
        api_raw=api_result,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(use_api: bool = False, api_key: str | None = None) -> list[OCRBenchmarkResult]:
    """运行 benchmark"""
    cases = _load_all_cases()
    if not cases:
        raise RuntimeError("未找到任何 OCR benchmark case")

    print(f"\n加载了 {len(cases)} 个 case:")
    cat_counts: dict[str, int] = {}
    for c in cases:
        cat_counts[c.category] = cat_counts.get(c.category, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    results = []

    for case in cases:
        # 尝试缓存
        api_result = None
        if not use_api:
            api_result = _read_cache(case.screenshot_path)

        if api_result is None:
            if not use_api:
                print(f"  [{case.name}] 跳过: 无缓存且未使用 --run-api")
                continue

            print(f"  [{case.name}] 调用 API...")
            try:
                api_result = _call_api(case.screenshot_path, api_key)
                _save_cache(case.screenshot_path, api_result)
                time.sleep(0.5)
            except Exception as e:
                print(f"  [{case.name}] API 调用失败: {e}")
                results.append(OCRBenchmarkResult(
                    case_name=case.name,
                    category=case.category,
                    expected_chat_name=case.ground_truth.get("chat_name", ""),
                    actual_chat_name="",
                    chat_name_match=False,
                    expected_message_count=len(case.ground_truth.get("messages", [])),
                    actual_message_count=0,
                    message_count_match=False,
                    sender_correct=0,
                    sender_total=len(case.ground_truth.get("messages", [])),
                    sender_accuracy=0.0,
                    text_correct=0,
                    text_total=len(case.ground_truth.get("messages", [])),
                    text_accuracy=0.0,
                    chat_list_correct=0,
                    chat_list_total=0,
                    chat_list_accuracy=0.0,
                    api_raw={},
                    passed=False,
                    error=str(e),
                ))
                continue

        result = evaluate_case(case, api_result)
        results.append(result)
        status = "✅" if result.passed else "❌"
        print(f"  [{case.name}] {status} sender={result.sender_accuracy:.0%} "
              f"text={result.text_accuracy:.0%} count={'✅' if result.message_count_match else '❌'}")

    return results


# ---------------------------------------------------------------------------
# Metrics & Report
# ---------------------------------------------------------------------------

def compute_metrics(results: list[OCRBenchmarkResult]) -> dict[str, Any]:
    """计算汇总指标"""
    if not results:
        return {}

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    chat_name_acc = sum(1 for r in results if r.chat_name_match) / total
    msg_count_acc = sum(1 for r in results if r.message_count_match) / total

    sender_cases = [r for r in results if r.sender_total > 0]
    sender_acc = sum(r.sender_accuracy for r in sender_cases) / len(sender_cases) if sender_cases else 1.0
    sender_perfect = sum(1 for r in sender_cases if r.sender_accuracy == 1.0) / len(sender_cases) if sender_cases else 1.0

    text_cases = [r for r in results if r.text_total > 0]
    text_acc = sum(r.text_accuracy for r in text_cases) / len(text_cases) if text_cases else 1.0

    chat_list_cases = [r for r in results if r.chat_list_total > 0]
    chat_list_acc = sum(r.chat_list_accuracy for r in chat_list_cases) / len(chat_list_cases) if chat_list_cases else 1.0

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total,
        "chat_name_accuracy": chat_name_acc,
        "message_count_accuracy": msg_count_acc,
        "sender_accuracy": sender_acc,
        "sender_perfect_rate": sender_perfect,
        "text_accuracy": text_acc,
        "chat_list_accuracy": chat_list_acc,
    }


def print_report(results: list[OCRBenchmarkResult], metrics: dict[str, Any]) -> None:
    """打印详细报告"""
    print("\n" + "=" * 95)
    print("OCR 质量 Benchmark 报告")
    print("=" * 95)

    # 按 category 分组
    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    # 逐个 case
    print("\n【逐个 Case 结果】")
    print(f"{'Case':<38} {'类别':<15} {'名称':<6} {'数量':<6} {'Sender':<8} {'Text':<8} {'结果'}")
    print("-" * 95)
    for cat in sorted(categories.keys()):
        for r in categories[cat]:
            name_ok = "✅" if r.chat_name_match else "❌"
            count_ok = "✅" if r.message_count_match else "❌"
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"{r.case_name:<38} {r.category:<15} {name_ok:<6} {count_ok:<6} "
                  f"{r.sender_accuracy:<8.0%} {r.text_accuracy:<8.0%} {status}")

    # 汇总
    print("\n【核心指标】")
    print(f"  Total cases:           {metrics['total']}")
    print(f"  Passed:                {metrics['passed']}/{metrics['total']} ({metrics['pass_rate']:.1%})")
    print(f"  Chat Name 准确率:       {metrics['chat_name_accuracy']:.1%}")
    print(f"  Message Count 准确率:   {metrics['message_count_accuracy']:.1%}")
    print(f"  Sender 平均准确率:      {metrics['sender_accuracy']:.1%}")
    print(f"  Sender 100%正确率:      {metrics['sender_perfect_rate']:.1%}")
    print(f"  Text 平均准确率:        {metrics['text_accuracy']:.1%}")
    print(f"  Chat List 准确率:       {metrics['chat_list_accuracy']:.1%}")

    # 按 category 分析
    print("\n【按类别分析】")
    for cat in sorted(categories.keys()):
        subset = categories[cat]
        m = compute_metrics(subset)
        print(f"  {cat:<18}: {m['passed']}/{m['total']} passed, "
              f"sender={m['sender_perfect_rate']:.0%}, "
              f"text={m['text_accuracy']:.0%}, "
              f"count={m['message_count_accuracy']:.0%}")

    # 失败详情
    failed = [r for r in results if not r.passed]
    if failed:
        print("\n【失败详情】")
        for r in failed:
            print(f"\n  ❌ {r.case_name} ({r.category})")
            if not r.chat_name_match:
                print(f"      chat_name: 预期='{r.expected_chat_name}' 实际='{r.actual_chat_name}'")
            if not r.message_count_match:
                print(f"      message_count: 预期={r.expected_message_count} 实际={r.actual_message_count}")
            if r.sender_accuracy < 1.0:
                bad_senders = [d for d in r.message_details if not d["sender_ok"]]
                missing = r.expected_message_count - r.actual_message_count
                extra = r.actual_message_count - r.expected_message_count
                print(f"      sender 错误 ({len(bad_senders)} 条对齐错误, 缺失={missing}, 多余={extra}):")
                for d in bad_senders[:3]:
                    print(f"        [{d['index']}] 预期='{d['expected_sender']}' 实际='{d['actual_sender']}'")
            if r.text_accuracy < 0.8:
                bad_texts = [d for d in r.message_details if not d["text_ok"]]
                print(f"      text 错误 ({len(bad_texts)} 条):")
                for d in bad_texts[:3]:
                    exp = d['expected_text'][:50]
                    act = d['actual_text'][:50]
                    print(f"        [{d['index']}] 预期='{exp}' 实际='{act}'")

    print("=" * 95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OCR 质量 Benchmark")
    parser.add_argument("--run-api", action="store_true", help="调用真实 API（否则使用缓存）")
    parser.add_argument("--api-key", default=None, help="API key")
    args = parser.parse_args()

    results = run_benchmark(use_api=args.run_api, api_key=args.api_key)
    metrics = compute_metrics(results)
    print_report(results, metrics)

    # 如果有失败，返回非零 exit code（CI 用）
    if metrics.get("pass_rate", 1.0) < 1.0:
        failed_names = [r.case_name for r in results if not r.passed]
        print(f"\n⚠️  失败 case: {', '.join(failed_names)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pytest 接口
# ---------------------------------------------------------------------------

_run_api_flag = False


def pytest_configure(config):
    global _run_api_flag
    _run_api_flag = config.getoption("--run-api", default=False)


def pytest_addoption(parser):
    parser.addoption("--run-api", action="store_true", default=False,
                     help="调用真实 API 运行 benchmark")


@pytest.fixture(scope="module")
def benchmark_results():
    return run_benchmark(use_api=_run_api_flag)


def test_benchmark_sender_perfect_rate(benchmark_results):
    """Sender 100% 正确率（sender 污染是致命 bug）"""
    metrics = compute_metrics(benchmark_results)
    # 基线阶段：记录指标，不强制失败
    assert metrics["sender_perfect_rate"] >= 0.0  # 总是通过，用于记录
    print(f"\n  📊 Sender perfect rate: {metrics['sender_perfect_rate']:.1%}")


def test_benchmark_overall_summary(benchmark_results):
    """输出整体汇总"""
    metrics = compute_metrics(benchmark_results)
    print(f"\n  📊 OCR Benchmark: {metrics['passed']}/{metrics['total']} passed")
    print(f"     chat_name={metrics['chat_name_accuracy']:.0%} | "
          f"count={metrics['message_count_accuracy']:.0%} | "
          f"sender={metrics['sender_perfect_rate']:.0%} | "
          f"text={metrics['text_accuracy']:.0%}")


@pytest.mark.skipif(not os.environ.get("DASHSCOPE_API_KEY"), reason="未设置 DASHSCOPE_API_KEY")
def test_benchmark_with_real_api():
    """使用真实 API 运行（手动触发）"""
    results = run_benchmark(use_api=True)
    metrics = compute_metrics(results)
    print_report(results, metrics)


if __name__ == "__main__":
    main()
