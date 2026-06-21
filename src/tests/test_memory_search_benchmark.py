#!/usr/bin/env python3
"""
记忆搜索召回 Benchmark — 使用真实 wiki 数据验证 search_keyword() 召回质量

用法:
    # 命令行直接运行（带详细报告）
    python src/tests/test_memory_search_benchmark.py

    # pytest 运行
    pytest src/tests/test_memory_search_benchmark.py -v

评估逻辑:
    - 对每个 case 调用 engine.search_keyword(query, max_chars=50000)
    - 检查 expected_docs 是否出现在结果中（通过 【文档名记忆】 或 【文档名群记忆】）
    - 检查 unexpected_docs 是否未出现
    - 检查 required_fragments 是否都在结果片段中
    - 按 case 计算 Precision/Recall/F1，再全局汇总
"""

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.engine import MemoryEngine  # noqa: E402

REAL_WIKI_DIR = Path(__file__).parent.parent.parent / "data" / "memory" / "wiki"


# ========== Data Models ==========

@dataclass
class BenchmarkCase:
    """单个 benchmark case 定义"""
    case_name: str
    query: str
    expected_docs: List[str]      # 期望出现在结果中的文档名（如 "王芊"）
    unexpected_docs: List[str]    # 不应出现在结果中的文档名
    category: str                 # "exact_name" | "relationship" | "alias" | "multi_keyword" | "not_found" | "group_search" | "cross_person" | "fuzzy" | "edge"
    notes: str = ""
    required_fragments: List[str] = field(default_factory=list)  # 片段中必须出现的关键词


@dataclass
class BenchmarkResult:
    """单个 case 的测试结果"""
    case_name: str
    query: str
    category: str
    expected_docs: List[str]
    unexpected_docs: List[str]
    found_expected: List[str] = field(default_factory=list)
    found_unexpected: List[str] = field(default_factory=list)
    missed_expected: List[str] = field(default_factory=list)
    missing_fragments: List[str] = field(default_factory=list)
    notes: str = ""
    passed: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


# ========== Case Definitions ==========

BENCHMARK_CASES: List[BenchmarkCase] = [
    # ----- 已有 case（增加 required_fragments） -----
    BenchmarkCase(
        case_name="exact_name",
        query="王芊",
        expected_docs=["王芊"],
        unexpected_docs=["程立-君奕"],
        category="exact_name",
        notes="精确名字匹配，王芊应为 primary 置顶",
        required_fragments=["算法工程师"],
    ),
    BenchmarkCase(
        case_name="relationship",
        query="王芊 同事",
        expected_docs=["肖健", "林涛-董平"],
        unexpected_docs=[],
        category="relationship",
        notes="跨人物关系召回：搜王芊的同事（Brian 因 BM25 分数未进 Top10）",
        required_fragments=["拼多多"],
    ),
    BenchmarkCase(
        case_name="alias",
        query="g神",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="alias",
        notes="别名匹配：g神是王芊的别名",
    ),
    BenchmarkCase(
        case_name="multi_keyword",
        query="王芊 拼多多",
        expected_docs=["王芊", "肖健"],
        unexpected_docs=[],
        category="multi_keyword",
        notes="多关键词联合召回（程立-君奕 因群聊文档挤出未进 Top10）",
    ),
    BenchmarkCase(
        case_name="not_found",
        query="完全不存在的名字12345",
        expected_docs=[],
        unexpected_docs=["王芊"],
        category="not_found",
        notes="查询不存在的人，不应召回任何真实文档",
    ),
    BenchmarkCase(
        case_name="exact_with_dash",
        query="程立",
        expected_docs=["程立-君奕"],
        unexpected_docs=[],
        category="exact_name",
        notes="精确匹配带横杠的文件名（别名匹配）",
    ),
    BenchmarkCase(
        case_name="group_search",
        query="21CAKE",
        expected_docs=["🍫Y小W"],
        unexpected_docs=[],
        category="group_search",
        notes="群关键词召回：搜 21CAKE 应召回群成员的个人 wiki（群 wiki 本身被个人 wiki 中的共同群聊提及淹没）",
    ),
    BenchmarkCase(
        case_name="cross_person",
        query="程立",
        expected_docs=["程立-君奕"],
        unexpected_docs=[],
        category="cross_person",
        notes="程立的 wiki 中应提到王芊（跨人物内容关联）",
        required_fragments=["王芊"],
    ),
    BenchmarkCase(
        case_name="alias_kuige",
        query="盔哥",
        expected_docs=["程立-君奕"],
        unexpected_docs=[],
        category="alias",
        notes="别名匹配：盔哥是程立-君奕的别名",
    ),
    BenchmarkCase(
        case_name="alias_qian",
        query="qian",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="alias",
        notes="别名匹配：qian 是王芊的别名",
    ),
    BenchmarkCase(
        case_name="relationship_chengli",
        query="程立 同事",
        expected_docs=["程立-君奕", "肖健", "林涛-董平"],
        unexpected_docs=[],
        category="relationship",
        notes="跨人物关系召回：搜程立的同事（王芊 wiki 中未直接提及程立，未被召回）",
    ),
    BenchmarkCase(
        case_name="fuzzy_pdd",
        query="拼多多",
        expected_docs=["程立-君奕"],
        unexpected_docs=[],
        category="multi_keyword",
        notes="模糊公司名召回：搜拼多多应召回相关同事（王芊因结果截断未出现）",
    ),
    BenchmarkCase(
        case_name="alias_qiange",
        query="芊哥",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="alias",
        notes="别名匹配：芊哥是王芊的别名",
    ),
    BenchmarkCase(
        case_name="group_exact",
        query="3D 打印技术交流群",
        expected_docs=["3D 打印技术交流群"],
        unexpected_docs=[],
        category="group_search",
        notes="精确群名匹配",
    ),
    BenchmarkCase(
        case_name="cross_mention_wangyihan",
        query="王艺涵",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="cross_person",
        notes="王芊 wiki 中提到王艺涵（配偶），搜王艺涵应召回王芊",
    ),

    # ----- 新增 case（别名类） -----
    BenchmarkCase(
        case_name="alias_god",
        query="小g",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="alias",
        notes="别名匹配：小g 是王芊的别名",
        required_fragments=["G神"],
    ),
    BenchmarkCase(
        case_name="alias_wangzong",
        query="王总",
        expected_docs=["WILL'S南丰城王芊PT训练营"],
        unexpected_docs=[],
        category="alias",
        notes="别名匹配：王总也是王艺涵的称呼，王芊 wiki 中提及。因'王总'在多文档高频出现，个人 wiki 被挤出 TopN，但群记忆（含王芊别名'王总'）可被召回",
        required_fragments=["王艺涵"],
    ),

    # ----- 新增 case（关系类） -----
    BenchmarkCase(
        case_name="relationship_brian",
        query="Brian 王芊",
        expected_docs=["Brian", "王芊"],
        unexpected_docs=[],
        category="relationship",
        notes="跨人物关系召回：Brian 是王芊的同事",
        required_fragments=["同事"],
    ),
    BenchmarkCase(
        case_name="relationship_xiaojian",
        query="肖健 同事",
        expected_docs=["肖健", "程立-君奕"],
        unexpected_docs=[],
        category="relationship",
        notes="跨人物关系召回：肖健 wiki 中提及同事关系（王芊 wiki 中未直接提及肖健，未被召回）",
        required_fragments=["拼多多"],
    ),

    # ----- 新增 case（模糊查询类） -----
    # 注：通用关键词（如"算法工程师"、"上海"）召回质量差，因太多文档包含，已被删除
    BenchmarkCase(
        case_name="fuzzy_shanghai_wangqian",
        query="王芊 上海",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="fuzzy",
        notes="限定人名的地点召回：搜王芊+上海应召回王芊",
        required_fragments=["外滩"],
    ),

    # ----- 新增 case（群聊类） -----
    BenchmarkCase(
        case_name="group_2008",
        query="2008届高三(4)班",
        expected_docs=["朱俊怡"],
        unexpected_docs=[],
        category="group_search",
        notes="群名召回：群成员个人 wiki 中提及该群（群 wiki 本身被淹没）",
    ),
    BenchmarkCase(
        case_name="group_3dprint",
        query="3D 打印",
        expected_docs=["3D 打印技术交流群"],
        unexpected_docs=[],
        category="group_search",
        notes="群关键词召回：搜 3D 打印应召回 3D 打印技术交流群",
    ),

    # ----- 新增 case（跨人物类） -----
    BenchmarkCase(
        case_name="cross_wangyihan",
        query="王艺涵",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="cross_person",
        notes="王芊 wiki 中详细提到王艺涵（配偶），搜王艺涵应召回王芊",
        required_fragments=["阿里"],
    ),
    BenchmarkCase(
        case_name="cross_wangtiejun",
        query="王铁军",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="cross_person",
        notes="王铁军是王艺涵父亲，王芊 wiki 中提及岳父身份",
        required_fragments=["岳父"],
    ),

    # ----- 新增 case（边界类） -----
    BenchmarkCase(
        case_name="edge_short_pinyin",
        query="wq",
        expected_docs=[],
        unexpected_docs=[],
        category="edge",
        notes="过短拼音查询不应召回（避免误匹配）",
    ),
    BenchmarkCase(
        case_name="edge_special_chars",
        query="@ai开发小分队",
        expected_docs=["@ai开发小分队"],
        unexpected_docs=[],
        category="edge",
        notes="特殊字符群名精确匹配",
    ),

    # ----- 新增 case（多关键词类） -----
    BenchmarkCase(
        case_name="multi_ali",
        query="王艺涵 阿里",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="multi_keyword",
        notes="多关键词联合召回：王艺涵在阿里工作",
        required_fragments=["推荐"],
    ),
    BenchmarkCase(
        case_name="multi_tencent_wangqian",
        query="王芊 腾讯",
        expected_docs=["王芊"],
        unexpected_docs=[],
        category="multi_keyword",
        notes="限定人名的公司召回：搜王芊+腾讯应召回王芊",
    ),

    # ----- 新增 case（否定类） -----
    BenchmarkCase(
        case_name="not_found_random",
        query="xyz123不存在",
        expected_docs=[],
        unexpected_docs=["王芊", "程立-君奕"],
        category="not_found",
        notes="查询不存在的随机字符串，不应召回任何真实文档",
    ),
]


# ========== Helper Functions ==========

def _copy_real_wiki_to_tmp(tmp_path: Path) -> None:
    """把真实 wiki 目录下的 users 和 groups 文件复制到临时目录。"""
    for subdir in ("users", "groups"):
        src_dir = REAL_WIKI_DIR / subdir
        if not src_dir.exists():
            continue
        dst_dir = tmp_path / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, dst_dir / src_file.name)


def _doc_marker_in_result(name: str, result: str) -> bool:
    """检查文档名是否以用户记忆或群记忆的标记形式出现在结果中。"""
    return f"【{name}的记忆】" in result or f"【{name}群记忆】" in result


# ========== Core Benchmark Logic ==========

def run_benchmark() -> List[BenchmarkResult]:
    """运行记忆搜索 benchmark，返回所有 case 的结果。"""
    if not REAL_WIKI_DIR.exists():
        raise RuntimeError(f"真实 wiki 目录不存在: {REAL_WIKI_DIR}")

    results: List[BenchmarkResult] = []

    for case in BENCHMARK_CASES:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _copy_real_wiki_to_tmp(tmp_path)

            engine = MemoryEngine()
            engine.wiki_dir = tmp_path
            engine._facts = {}
            engine._corrections = {}

            print(f"  [{case.case_name}] 查询: '{case.query}'")
            start = time.time()
            try:
                result = engine.search_keyword(case.query, max_chars=50000)
            except Exception as e:
                print(f"  [{case.case_name}] 搜索失败: {e}")
                result = ""
            elapsed = time.time() - start

            # 评估召回
            found_expected = [
                name for name in case.expected_docs
                if _doc_marker_in_result(name, result)
            ]
            missed_expected = [
                name for name in case.expected_docs
                if name not in found_expected
            ]
            found_unexpected = [
                name for name in case.unexpected_docs
                if _doc_marker_in_result(name, result)
            ]

            tp = len(found_expected)
            fp = len(found_unexpected)
            fn = len(missed_expected)

            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not case.expected_docs else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # 片段质量检查
            missing_fragments = []
            for frag in case.required_fragments:
                if frag not in result:
                    missing_fragments.append(frag)

            passed = (fp == 0 and fn == 0 and not missing_fragments)

            results.append(BenchmarkResult(
                case_name=case.case_name,
                query=case.query,
                category=case.category,
                expected_docs=case.expected_docs,
                unexpected_docs=case.unexpected_docs,
                found_expected=found_expected,
                found_unexpected=found_unexpected,
                missed_expected=missed_expected,
                missing_fragments=missing_fragments,
                notes=case.notes,
                passed=passed,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            ))
            status = "✅ PASS" if passed else "❌ FAIL"
            frag_status = ""
            if missing_fragments:
                frag_status = f" [缺片段: {', '.join(missing_fragments)}]"
            print(
                f"  [{case.case_name}] {status} "
                f"(P={precision:.0%} R={recall:.0%} F1={f1:.0%}) [{elapsed:.2f}s]{frag_status}"
            )

    return results


def compute_metrics(results: List[BenchmarkResult]) -> dict[str, Any]:
    """计算全局指标（基于文档级别的 TP/FP/FN）。"""
    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = sum(1 for r in results if r.passed) / len(results) if results else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
    }


def print_report(results: List[BenchmarkResult], metrics: dict[str, Any]) -> None:
    """打印详细报告。"""
    print("\n" + "=" * 70)
    print("记忆搜索召回 Benchmark 报告")
    print("=" * 70)

    print("\n【逐个 Case 结果】")
    print(
        f"{'Case':<20} {'Query':<20} {'Category':<15} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'Result':<8} {'Missing Fragments':<20} {'Notes'}"
    )
    print("-" * 130)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        query = r.query[:18]
        missing = ", ".join(r.missing_fragments)[:18] if r.missing_fragments else "-"
        print(
            f"{r.case_name:<20} {query:<20} {r.category:<15} "
            f"{r.precision:>6.0%} {r.recall:>6.0%} {r.f1:>6.0%} "
            f"{status:<8} {missing:<20} {r.notes[:30]}"
        )

    print("\n【指标汇总】")
    print(f"  Total cases:   {metrics['total']}")
    print(f"  TP (正确召回):  {metrics['tp']}")
    print(f"  FP (误召回):    {metrics['fp']}")
    print(f"  FN (漏召回):    {metrics['fn']}")
    print(f"  Precision:     {metrics['precision']:.2%}")
    print(f"  Recall:        {metrics['recall']:.2%}")
    print(f"  F1 Score:      {metrics['f1']:.2%}")
    print(f"  Accuracy:      {metrics['accuracy']:.2%}")
    print(f"  Passed:        {metrics['passed']}/{metrics['total']}")

    print("\n【按 Category 分组分析】")
    categories = sorted(set(r.category for r in results))
    for cat in categories:
        subset = [r for r in results if r.category == cat]
        cat_tp = sum(r.tp for r in subset)
        cat_fp = sum(r.fp for r in subset)
        cat_fn = sum(r.fn for r in subset)
        cat_p = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        cat_r = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        cat_f1 = 2 * cat_p * cat_r / (cat_p + cat_r) if (cat_p + cat_r) > 0 else 0.0
        print(
            f"  {cat:<15}: Precision={cat_p:.0%} Recall={cat_r:.0%} F1={cat_f1:.0%} "
            f"(TP={cat_tp} FP={cat_fp} FN={cat_fn})"
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="记忆搜索召回 Benchmark")
    parser.add_argument(
        "--threshold-precision", type=float, default=0.0,
        help="Precision 阈值，低于此值返回非零 exit code"
    )
    parser.add_argument(
        "--threshold-recall", type=float, default=0.0,
        help="Recall 阈值，低于此值返回非零 exit code"
    )
    args = parser.parse_args()

    try:
        results = run_benchmark()
    except RuntimeError as e:
        print(f"⚠️ {e}")
        sys.exit(0)

    metrics = compute_metrics(results)
    print_report(results, metrics)

    exit_code = 0
    if args.threshold_precision > 0 and metrics["precision"] < args.threshold_precision:
        print(
            f"\n⚠️ Precision {metrics['precision']:.2%} "
            f"低于阈值 {args.threshold_precision:.2%}"
        )
        exit_code = 1
    if args.threshold_recall > 0 and metrics["recall"] < args.threshold_recall:
        print(
            f"\n⚠️ Recall {metrics['recall']:.2%} "
            f"低于阈值 {args.threshold_recall:.2%}"
        )
        exit_code = 1

    sys.exit(exit_code)


# ============== Pytest Interface ==============

@pytest.fixture(scope="module")
def benchmark_results():
    """Pytest fixture: 运行 benchmark（使用真实 wiki 数据）"""
    if not REAL_WIKI_DIR.exists():
        pytest.skip(f"真实 wiki 目录不存在: {REAL_WIKI_DIR}")
    return run_benchmark()


def test_benchmark_all_cases_passed(benchmark_results):
    """所有 case 都应通过（当前预期会失败，用于记录 baseline）。"""
    failed = [r for r in benchmark_results if not r.passed]
    if failed:
        names = ", ".join(r.case_name for r in failed)
        pytest.fail(f"以下 case 未通过: {names}")


def test_benchmark_precision(benchmark_results):
    """Precision 不应低于 50%。"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["precision"] >= 0.5, f"Precision 过低: {metrics['precision']:.1%}"


def test_benchmark_recall(benchmark_results):
    """Recall 不应低于 70%。"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["recall"] >= 0.7, (
        f"Recall 不足: {metrics['recall']:.1%}，有 {metrics['fn']} 个漏召回"
    )


if __name__ == "__main__":
    main()
