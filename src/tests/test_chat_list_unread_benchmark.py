#!/usr/bin/env python3
"""
未读角标识别 benchmark 测试框架。

用法:
    # 使用真实 API（计算 precision/recall）
    python -m pytest src/tests/test_chat_list_unread_benchmark.py -v --run-api

    # 使用缓存的 API 结果（快速回归）
    python -m pytest src/tests/test_chat_list_unread_benchmark.py -v

    # 命令行直接运行（带详细报告）
    python src/tests/test_chat_list_unread_benchmark.py --run-api

Fixture 目录结构:
    src/tests/fixtures/unread_badge/
        case_XXX_name/               # 每个 case 一个目录
            screenshot.png           # 截图
            ground_truth.json        # 标注
            api_result.json          # (可选) 缓存的 API 结果

ground_truth.json 字段:
    - target_nickname: 目标聊天昵称
    - has_unread: bool, 实际是否有未读
    - unread_count: str, 实际未读数（无则空字符串）
    - category: "true_positive" | "false_positive" | "true_negative" | "false_negative"
    - avatar_type: "single" | "single_icon" | "group_mosaic"
    - notes: str, 人工备注
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "unread_badge"


@dataclass
class BenchmarkResult:
    """单个 case 的测试结果"""
    case_name: str
    target_nickname: str
    ground_truth_has_unread: bool
    ground_truth_count: str
    predicted_has_unread: bool
    predicted_count: str
    avatar_type: str
    notes: str
    api_raw: dict = field(default_factory=dict)
    passed: bool = False


def _load_fixture_cases() -> list[Path]:
    """加载所有 fixture case 目录"""
    if not FIXTURE_DIR.exists():
        return []
    return sorted([d for d in FIXTURE_DIR.iterdir() if d.is_dir() and d.name.startswith("case_")])


def _read_ground_truth(case_dir: Path) -> dict:
    """读取 ground truth"""
    gt_path = case_dir / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def _read_cached_api_result(case_dir: Path) -> dict | None:
    """读取缓存的 API 结果"""
    cache_path = case_dir / "api_result.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def _save_api_result(case_dir: Path, result: dict) -> None:
    """保存 API 结果到缓存"""
    cache_path = case_dir / "api_result.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _call_api_recognize(screenshot_path: Path, api_key: str | None = None) -> list[dict]:
    """调用真实 API 识别截图，返回 chat_list"""
    # 使用项目中的 _QwenAPIClient（定义在 smart_pipeline.py）
    try:
        from src.perception.smart_pipeline import _QwenAPIClient
    except ImportError:
        try:
            # Fallback for direct execution
            from perception.smart_pipeline import _QwenAPIClient  # type: ignore[no-redef]
        except ImportError:
            # Absolute import with sys.path
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from perception.smart_pipeline import _QwenAPIClient  # type: ignore[no-redef]

    if api_key is None:
        api_key = (os.environ.get("DASHSCOPE_API_KEY")
                   or os.environ.get("OPENCLAW_API_KEY")
                   or os.environ.get("QWEN_API_KEY"))
        # Try reading from .env file
        if not api_key:
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("DASHSCOPE_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
                            break
        if not api_key:
            raise RuntimeError("未设置 API key。请设置 DASHSCOPE_API_KEY 环境变量或在 .env 文件中配置")

    client = _QwenAPIClient(api_key=api_key)
    result = client.recognize(str(screenshot_path))

    # result 可能是 dict 或 list
    if isinstance(result, dict):
        return result.get("chat_list", [])
    elif isinstance(result, list):
        return result
    return []


def _find_target_in_api_result(chat_list: list[dict], target_nickname: str) -> dict | None:
    """在 API 结果中查找目标昵称对应的 item。

    优先策略：
    1. 精确匹配（或截断后的精确前缀匹配）
    2. 昵称长度最接近 target 的匹配（避免 "王芊" 先匹配到 "W1han、王芊"）
    """
    candidates = []
    for item in chat_list:
        nick = item.get("nickname", "")
        score = 0
        # 精确匹配
        if nick == target_nickname:
            score = 1000
        # 截断精确前缀匹配（如 "王芊 @ai开发小..." vs "王芊 @ai开发小分队"）
        elif nick.endswith("...") and len(nick) > 3:
            nick_prefix = nick[:-3]
            if target_nickname.startswith(nick_prefix):
                score = 900
        # 互相包含
        elif target_nickname in nick or nick in target_nickname:
            score = 100
        # 部分包含（长度 >= 3 的子串）
        elif len(nick) >= 3 and nick in target_nickname:
            score = 50

        if score > 0:
            # 额外加分：昵称长度越接近 target，越可能是正确匹配
            length_diff = abs(len(nick) - len(target_nickname))
            score -= length_diff  # 长度差异越小，分数越高
            candidates.append((score, item))

    if not candidates:
        return None
    # 按分数降序排列，返回最高分的
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _apply_unread_filter(raw_count: str) -> str:
    """应用与 smart_pipeline 相同的过滤逻辑"""
    if not raw_count:
        return ""
    # 时间戳过滤
    if ":" in raw_count:
        return ""
    # 汉字过滤
    if any("\u4e00" <= c <= "\u9fff" for c in raw_count):
        return ""
    # 非数字过滤
    if not raw_count.isdigit():
        return ""
    # 超过99过滤
    if int(raw_count) > 99:
        return ""
    return raw_count


def run_benchmark(use_api: bool = False, api_key: str | None = None) -> list[BenchmarkResult]:
    """
    运行 benchmark，返回所有 case 的结果。

    Args:
        use_api: 是否调用真实 API。False 则使用缓存结果。
        api_key: API key（use_api=True 时需要）
    """
    case_dirs = _load_fixture_cases()
    if not case_dirs:
        raise RuntimeError(f"未找到 fixture cases: {FIXTURE_DIR}")

    results: list[BenchmarkResult] = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        gt = _read_ground_truth(case_dir)
        target_nickname = gt["target_nickname"]
        gt_has_unread = gt["has_unread"]
        gt_count = gt.get("unread_count", "")
        avatar_type = gt.get("avatar_type", "unknown")
        notes = gt.get("notes", "")

        # 获取 API 结果
        api_result: dict[str, Any] | None = None
        if use_api:
            screenshot_path = case_dir / "screenshot.png"
            print(f"  [{case_name}] 调用 API 识别: {screenshot_path.name}")
            try:
                chat_list = _call_api_recognize(screenshot_path, api_key)
                api_result = {"chat_list": chat_list}
                _save_api_result(case_dir, api_result)
                time.sleep(0.5)  # 避免速率限制
            except Exception as e:
                print(f"  [{case_name}] API 调用失败: {e}")
                api_result = {"chat_list": [], "error": str(e)}
                _save_api_result(case_dir, api_result)
        else:
            api_result = _read_cached_api_result(case_dir)

        if api_result is None:
            print(f"  [{case_name}] 跳过: 无缓存结果且未使用 --run-api")
            continue

        chat_list = api_result.get("chat_list", [])
        target_item = _find_target_in_api_result(chat_list, target_nickname)

        if target_item:
            raw_count = target_item.get("unread_count", "")
            predicted_count = _apply_unread_filter(raw_count)
        else:
            raw_count = ""
            predicted_count = ""

        predicted_has_unread = bool(predicted_count)
        passed = (predicted_has_unread == gt_has_unread and
                  (not gt_has_unread or predicted_count == gt_count))

        results.append(BenchmarkResult(
            case_name=case_name,
            target_nickname=target_nickname,
            ground_truth_has_unread=gt_has_unread,
            ground_truth_count=gt_count,
            predicted_has_unread=predicted_has_unread,
            predicted_count=predicted_count,
            avatar_type=avatar_type,
            notes=notes,
            api_raw=api_result,
            passed=passed,
        ))

    return results


def compute_metrics(results: list[BenchmarkResult]) -> dict[str, Any]:
    """计算 precision, recall, F1"""
    tp = sum(1 for r in results if r.ground_truth_has_unread and r.predicted_has_unread)
    fp = sum(1 for r in results if not r.ground_truth_has_unread and r.predicted_has_unread)
    fn = sum(1 for r in results if r.ground_truth_has_unread and not r.predicted_has_unread)
    tn = sum(1 for r in results if not r.ground_truth_has_unread and not r.predicted_has_unread)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
    }


def print_report(results: list[BenchmarkResult], metrics: dict[str, Any]) -> None:
    """打印详细报告"""
    print("\n" + "=" * 70)
    print("未读角标识别 Benchmark 报告")
    print("=" * 70)

    print("\n【逐个 Case 结果】")
    print(f"{'Case':<30} {'昵称':<20} {'期望':<8} {'预测':<8} {'结果':<6} {'头像类型':<12} {'备注'}")
    print("-" * 110)
    for r in results:
        expect = f"{r.ground_truth_count or '无'}" if r.ground_truth_has_unread else "无"
        pred = f"{r.predicted_count}" if r.predicted_has_unread else "无"
        status = "✅ PASS" if r.passed else "❌ FAIL"
        nick = r.target_nickname[:18]
        print(f"{r.case_name:<30} {nick:<20} {expect:<8} {pred:<8} {status:<6} {r.avatar_type:<12} {r.notes[:30]}")

    print("\n【指标汇总】")
    print(f"  Total cases:   {metrics['total']}")
    print(f"  TP (正确检出):  {metrics['tp']}")
    print(f"  FP (误报):      {metrics['fp']}")
    print(f"  FN (漏报):      {metrics['fn']}")
    print(f"  TN (正确否定):  {metrics['tn']}")
    print(f"  Precision:     {metrics['precision']:.2%}")
    print(f"  Recall:        {metrics['recall']:.2%}")
    print(f"  F1 Score:      {metrics['f1']:.2%}")
    print(f"  Accuracy:      {metrics['accuracy']:.2%}")
    print(f"  Passed:        {metrics['passed']}/{metrics['total']}")

    print("\n【按头像类型分析】")
    avatar_types = set(r.avatar_type for r in results)
    for at in sorted(avatar_types):
        subset = [r for r in results if r.avatar_type == at]
        m = compute_metrics(subset)
        print(f"  {at:<15}: Precision={m['precision']:.0%} Recall={m['recall']:.0%} "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="未读角标识别 Benchmark")
    parser.add_argument("--run-api", action="store_true", help="调用真实 API（否则使用缓存）")
    parser.add_argument("--api-key", default=None, help="API key（默认从环境变量读取）")
    parser.add_argument("--threshold-precision", type=float, default=0.0,
                        help="Precision 阈值，低于此值返回非零 exit code")
    parser.add_argument("--threshold-recall", type=float, default=0.0,
                        help="Recall 阈值，低于此值返回非零 exit code")
    args = parser.parse_args()

    results = run_benchmark(use_api=args.run_api, api_key=args.api_key)
    metrics = compute_metrics(results)
    print_report(results, metrics)

    # 返回 exit code
    exit_code = 0
    if args.threshold_precision > 0 and metrics["precision"] < args.threshold_precision:
        print(f"\n⚠️  Precision {metrics['precision']:.2%} 低于阈值 {args.threshold_precision:.2%}")
        exit_code = 1
    if args.threshold_recall > 0 and metrics["recall"] < args.threshold_recall:
        print(f"\n⚠️  Recall {metrics['recall']:.2%} 低于阈值 {args.threshold_recall:.2%}")
        exit_code = 1
    if metrics["passed"] < metrics["total"]:
        print(f"\n⚠️  有 {metrics['total'] - metrics['passed']} 个 case 未通过")
        # 不强制 exit 1，因为 benchmark 目的就是记录失败

    sys.exit(exit_code)


# ============== Pytest 接口 ==============

@pytest.fixture(scope="module")
def benchmark_results():
    """Pytest fixture: 运行 benchmark（使用缓存）"""
    return run_benchmark(use_api=False)


def test_benchmark_all_cases_passed(benchmark_results):
    """所有 case 都应通过（当前预期会失败，用于记录 baseline）"""
    failed = [r for r in benchmark_results if not r.passed]
    if failed:
        names = ", ".join(r.case_name for r in failed)
        pytest.fail(f"以下 case 未通过: {names}")


def test_benchmark_precision(benchmark_results):
    """Precision 不应低于 50%"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["precision"] >= 0.5, f"Precision 过低: {metrics['precision']:.1%}"


def test_benchmark_recall(benchmark_results):
    """Recall 应为 100%（不能有漏报）"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["recall"] == 1.0, f"Recall 不足: {metrics['recall']:.1%}，有 {metrics['fn']} 个漏报"


@pytest.mark.skipif(not os.environ.get("OPENCLAW_API_KEY") and not os.environ.get("QWEN_API_KEY"),
                    reason="未设置 API key")
def test_benchmark_with_real_api():
    """使用真实 API 运行 benchmark（手动触发）"""
    results = run_benchmark(use_api=True)
    metrics = compute_metrics(results)
    print_report(results, metrics)
    assert metrics["precision"] >= 0.5, f"Precision 过低: {metrics['precision']:.1%}"
    assert metrics["recall"] == 1.0, f"Recall 不足: {metrics['recall']:.1%}"


if __name__ == "__main__":
    main()
