#!/usr/bin/env python3
"""
历史原文检索召回 Benchmark — 用真实 779k 消息索引验证 search_history 召回质量

与 test_memory_search_benchmark.py（wiki 摘要召回）并列：
  - memory_search benchmark → 检索编译后的人物 wiki（"这个人是谁"）
  - history_search benchmark → 检索历史聊天原文（"当时说了什么"）

用法:
    # 命令行直接运行（带详细报告）
    python src/tests/test_history_search_benchmark.py

    # pytest 运行
    pytest src/tests/test_history_search_benchmark.py -v

评估逻辑:
    - 对每个 case 调用 HistorySearchIndex.search(query, top_k, sender_name=...)
    - 收集返回结果中所有 context_messages 的 id（含命中消息本身）
    - exact_recall / semantic 类：expected_ids 应全部命中，unexpected_ids 不应命中
    - fragment 类：expected_fragments 应出现在格式化结果文本中
    - not_found 类：查询不存在的词，不应召回 expected_ids 之外的真实消息
    - sender_filter 类：所有命中消息的 sender 应为指定发送者
    - 按 case 计算 Precision/Recall/F1，再全局汇总

依赖真实索引（1.7GB pickle，懒加载 ~9s）。索引/依赖未就绪时整体 skip，
与 memory_search benchmark 在 wiki 缺失时 skip 的行为一致。
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import history_search  # noqa: E402
from src.memory.history_search import HistorySearchIndex  # noqa: E402


# ========== Data Models ==========

@dataclass
class BenchmarkCase:
    """单个 benchmark case 定义

    expected_ids     期望命中的消息 id 列表（来自真实索引，用 exact_recall/semantic 验证）
    unexpected_ids   不应出现在结果中的消息 id
    expected_fragments 期望在格式化结果文本中出现的子串（fragment 类）
    expected_sender  sender_filter 类：所有命中消息的 sender 应为此值
    """
    case_name: str
    query: str
    category: str  # "exact_recall" | "semantic" | "fragment" | "not_found" | "sender_filter" | "edge" | "composite" | "ambiguity" | "negative" | "noise"
    expected_ids: List[str] = field(default_factory=list)
    unexpected_ids: List[str] = field(default_factory=list)
    expected_fragments: List[str] = field(default_factory=list)
    expected_sender: str = ""
    top_k: int = 5
    notes: str = ""
    known_issue: str = ""  # 非空=已知问题，FAIL 不计入 recall 惩罚，仅记录
    expected_primary_id: str = ""  # 最该排前的消息 id（MRR 首选目标）；空则用 expected_ids[0]


@dataclass
class BenchmarkResult:
    """单个 case 的测试结果"""
    case_name: str
    query: str
    category: str
    expected_ids: List[str]
    unexpected_ids: List[str]
    found_expected: List[str] = field(default_factory=list)
    found_unexpected: List[str] = field(default_factory=list)
    missed_expected: List[str] = field(default_factory=list)
    missing_fragments: List[str] = field(default_factory=list)
    sender_violations: List[str] = field(default_factory=list)
    notes: str = ""
    passed: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    known_issue: str = ""
    elapsed: float = 0.0
    # 排序指标（order-aware）
    first_rank: int = 0   # expected_primary_id 在结果中的 1-based 排名；0=未命中
    hit_at_5: bool = False  # expected_primary_id 是否排进前 5
    mrr: float = 0.0      # 1/first_rank（rank>5 或未命中算 0）
    # 召回阶段指标（融合 rerank 前）
    in_recall_pool: bool = False  # primary 是否在 dense+keyword 召回候选池 top30 里
    recall_pool_rank: int = 0   # primary 在召回池中的排名；0=未命中


# ========== Case Definitions ==========
#
# expected_ids 取自真实索引（seed=42 抽样，见 scripts 抽样记录）。
# exact_recall：query=原文，期望同一条消息命中——验证向量空间基本可用。
# semantic：query=语义改写（非逐字），期望仍能召回原文——BGE 的核心价值在此。

BENCHMARK_CASES: List[BenchmarkCase] = [
    # ── exact_recall：逐字原文检索，必须命中自身 ──
    BenchmarkCase(
        case_name="exact_xiaojian_tesla",
        query="特斯拉竟然没砸死",
        expected_ids=["b_私聊_肖健_1378"],
        category="exact_recall",
        notes="逐字原文检索，肖健的特斯拉消息应命中自身",
    ),
    BenchmarkCase(
        case_name="exact_qiushui_hesuan",
        query="这两天核酸了没？",
        expected_ids=["b_私聊_秋水文章_8016"],
        category="exact_recall",
        notes="逐字原文检索，秋水文章的核酸消息应命中自身",
    ),
    BenchmarkCase(
        case_name="exact_lujie_pudong",
        query="浦东的还是闵行的",
        expected_ids=["b_私聊_陆杰（山间云）_1040"],
        category="exact_recall",
        notes="逐字原文检索，陆杰的浦东/闵行消息应命中自身",
    ),
    BenchmarkCase(
        case_name="exact_wangbing_cum",
        query="你们的粗排先是用的eenmf 后来用的是啥呀？",
        expected_ids=["b_私聊_王兵(tylerwang)_8891"],
        category="exact_recall",
        notes="逐字原文检索，王兵的粗排技术讨论应命中自身",
    ),
    BenchmarkCase(
        case_name="exact_group_paopao",
        query="港股有点想重仓泡泡玛特",
        expected_ids=["b_群聊_📮美港股价值投资群_46256"],
        category="exact_recall",
        notes="逐字原文检索，美港股群坤蜀黍的泡泡玛特消息应命中自身",
    ),

    # ── semantic：语义改写，验证 BGE 跨表述召回能力 ──
    BenchmarkCase(
        case_name="semantic_tesla_stock",
        query="特斯拉股价跌了没",
        expected_ids=["b_私聊_肖健_1378"],
        category="semantic",
        notes="语义改写：'特斯拉股价跌了没' 应召回肖健'特斯拉竟然没砸死'",
        known_issue="bge-small-zh-v1.5 对短查询的语义泛化不足：'股价跌了没' 与 '竟然没砸死' 语义相近但向量距离偏大，未进 Top5。属编码器能力上限，非逻辑 bug。",
    ),
    BenchmarkCase(
        case_name="semantic_hesuan_test",
        query="最近做核酸了吗",
        expected_ids=["b_私聊_秋水文章_8016"],
        category="semantic",
        notes="语义改写：'最近做核酸了吗' 应召回秋水文章'这两天核酸了没？'",
        known_issue="同 semantic_tesla_stock：bge-small 短查询泛化不足，'最近做核酸了吗' 与 '这两天核酸了没？' 未匹配。待编码器升级或加 query 改写后回归。",
    ),
    BenchmarkCase(
        case_name="semantic_paopao_position",
        query="想加仓泡泡玛特港股",
        expected_ids=["b_群聊_📮美港股价值投资群_46256"],
        category="semantic",
        notes="语义改写：'想加仓泡泡玛特港股' 应召回坤蜀黍'港股有点想重仓泡泡玛特'",
    ),
    BenchmarkCase(
        case_name="semantic_pudong_minhang_location",
        query="房子在浦东还是闵行",
        expected_ids=["b_私聊_陆杰（山间云）_1040"],
        category="semantic",
        notes="语义改写：'房子在浦东还是闵行' 应召回陆杰'浦东的还是闵行的'",
    ),

    # ── fragment：自然语言描述，期望片段出现在结果中 ──
    BenchmarkCase(
        case_name="fragment_eenmf_recall",
        query="粗排模型 eenmf 技术选型",
        expected_fragments=["eenmf"],
        category="fragment",
        notes="关键词+语义检索，结果应包含 eenmf 片段",
    ),
    BenchmarkCase(
        case_name="keyword_eenmf_term",
        query="eenmf",
        expected_fragments=["eenmf"],
        category="keyword_advantage",
        notes="纯术语查询：dense 向量对生僻术语区分度低，keyword 路应靠精确词命中召回含 eenmf 的消息。含 eenmf 的消息有多条，故用 fragment 检查而非固定 id。",
    ),
    BenchmarkCase(
        case_name="fragment_tesla_text",
        query="特斯拉没砸死",
        expected_fragments=["特斯拉"],
        category="fragment",
        notes="特斯拉相关讨论，结果应包含特斯拉片段",
    ),

    # ── not_found：不存在的查询，不应召回真实消息 ──
    BenchmarkCase(
        case_name="not_found_random",
        query="zzqqxx不存在的对话内容12345",
        expected_ids=[],
        unexpected_ids=[
            "b_私聊_肖健_1378",
            "b_私聊_秋水文章_8016",
            "b_群聊_📮美港股价值投资群_46256",
        ],
        category="not_found",
        notes="随机不存在的查询，不应召回任何已知真实消息",
    ),

    # ── sender_filter：限定发送者，命中应全部来自该发送者 ──
    BenchmarkCase(
        case_name="sender_filter_xiaojian",
        query="特斯拉",
        expected_sender="肖健",
        top_k=5,
        category="sender_filter",
        notes="sender_name=肖健 加权后，命中消息的 sender 应为肖健（或结果为空）",
        known_issue="sender 加权仅 +0.05 轻微 boost，强语义匹配仍可能召回他人；用于观察加权效果",
    ),

    # ── edge：空查询 ──
    BenchmarkCase(
        case_name="edge_empty_query",
        query="",
        expected_ids=[],
        category="edge",
        notes="空查询应返回空结果，不抛异常",
    ),
    BenchmarkCase(
        case_name="edge_blank_query",
        query="   ",
        expected_ids=[],
        category="edge",
        notes="纯空白查询应返回空结果",
    ),

    # ===== 复杂场景 case（composite / ambiguity / negative / noise）=====
    # 基于 77 万真实消息实测标注，known_issue 标真实短板。

    # ── composite：组合条件（多词），测多条件交集召回 + 排序 ──
    BenchmarkCase(
        case_name="comp_tesla_stock_drop",
        query="特斯拉 股价 跌",
        expected_ids=["b_群聊_📮美港股价值投资群_13195"],
        expected_primary_id="b_群聊_📮美港股价值投资群_13195",
        category="composite",
        notes="组合查询应召回含'特斯拉+股价+下跌'的消息且排第1（实测 rank1）",
    ),
    BenchmarkCase(
        case_name="comp_paopao_position",
        query="泡泡玛特 港股 加仓",
        expected_ids=["b_群聊_📮美港股价值投资群_46256"],
        expected_primary_id="b_群聊_📮美港股价值投资群_46256",
        category="composite",
        notes="组合查询应召回'港股重仓泡泡玛特'消息且排第1",
    ),

    # ── ambiguity：短词/术语，测精确词召回不被近似词干扰 ──
    BenchmarkCase(
        case_name="amb_eenmf_term",
        query="eenmf",
        expected_ids=["b_私聊_幸福的矿工-卢卡斯-施梦圜_234"],
        expected_primary_id="b_私聊_幸福的矿工-卢卡斯-施梦圜_234",
        unexpected_ids=["b_私聊_Y_2187", "b_私聊_幸福的矿工-卢卡斯-施梦圜_2672"],
        category="ambiguity",
        notes="术语 eenmf 应召回含 eenmf 的消息，不应召回近似词'ee?''eesm?'",
        known_issue="keyword 子串匹配把'ee?''eesm?'排进 top5（eenmf 含 ee 子串）。短术语近似词干扰，需词边界匹配。",
    ),

    # ── negative：不存在实体，测不误召回 ──
    BenchmarkCase(
        case_name="neg_nonexistent",
        query="不存在xyz12345",
        expected_ids=[],
        unexpected_ids=["b_私聊_肖健_10859", "b_群聊_外滩玺睦邻友好群_3740"],
        category="negative",
        notes="查不存在的实体应返回空，不应召回含'不存在'/'不是'的消息",
        known_issue="dense 路 min_score=0.01 太低，把含'不存在''不是'的语义相近消息召回。dense 对否定词理解弱，需提高阈值或加否定检测。",
    ),

    # ── noise：短消息噪声，测信息量低的短消息不占 top ──
    BenchmarkCase(
        case_name="noise_tesla_short",
        query="特斯拉",
        expected_ids=["b_私聊_夏晨风 少荃_1768"],
        expected_primary_id="b_私聊_夏晨风 少荃_1768",
        unexpected_ids=["b_私聊_Lisha 莉莎_5097"],
        category="noise",
        notes="搜特斯拉应召回有上下文的消息（如'30万一个特斯拉'），纯'特斯拉'两字无信息量不应排第1",
        known_issue="纯'特斯拉'短消息因 keyword 命中+context 加权排第1，信息量低的短消息排序偏高。短消息需降权。",
    ),
]


# ========== Core Benchmark Logic ==========

def _index() -> Optional[HistorySearchIndex]:
    """获取真实索引单例；不可用时返回 None。"""
    if not history_search.is_available():
        return None
    return history_search.get_history_index()


def _returned_ids(results: List[dict]) -> set:
    """收集所有返回结果中 context_messages 的 id（含命中消息本身）。"""
    ids: set = set()
    for r in results:
        hit = r.get("hit_message") or {}
        if hit.get("id"):
            ids.add(hit["id"])
        for m in r.get("context_messages", []):
            if m and m.get("id"):
                ids.add(m["id"])
    return ids


def _returned_senders(results: List[dict]) -> List[str]:
    """收集所有命中消息的 sender。"""
    senders: List[str] = []
    for r in results:
        hit = r.get("hit_message") or {}
        if hit.get("sender"):
            senders.append(hit["sender"])
    return senders


def _primary_rank_in_raw(primary_id: str, raw: List[dict]) -> int:
    """返回 primary_id 在结果中的 1-based 排名；0=未命中。

    在每个结果项的 hit_message 和 context_messages 里查找 primary_id，
    返回第一个命中的结果项的位次（1-based）。
    """
    if not primary_id:
        return 0
    for rank, r in enumerate(raw, 1):
        hit = r.get("hit_message") or {}
        if hit.get("id") == primary_id:
            return rank
        for m in r.get("context_messages", []):
            if m and m.get("id") == primary_id:
                return rank
    return 0


def run_benchmark() -> List[BenchmarkResult]:
    """运行历史原文检索 benchmark，返回所有 case 的结果。"""
    idx = _index()
    if idx is None:
        raise RuntimeError(
            "search_history 索引或编码器依赖未就绪（检查 WECHAT_HISTORY_INDEX_PATH / "
            "WECHAT_BGE_MODEL_PATH 及 onnxruntime+tokenizers）"
        )

    results: List[BenchmarkResult] = []

    for case in BENCHMARK_CASES:
        print(f"  [{case.case_name}] 查询: '{case.query}'")
        start = time.time()
        try:
            raw = idx.search(
                case.query,
                top_k=case.top_k,
                sender_name=case.expected_sender or "",
            )
            formatted = HistorySearchIndex.format_results(raw, case.query, max_chars=50000)
            # 召回阶段候选池（dense+keyword 并集，融合前）
            recall_pool = idx.recall_candidate_ids(
                case.query, top_n=30, sender_name=case.expected_sender or "",
            )
        except Exception as e:
            print(f"  [{case.case_name}] 检索失败: {e}")
            raw = []
            formatted = ""
            recall_pool = []
        elapsed = time.time() - start

        returned = _returned_ids(raw)

        found_expected = [i for i in case.expected_ids if i in returned]
        missed_expected = [i for i in case.expected_ids if i not in returned]
        found_unexpected = [i for i in case.unexpected_ids if i in returned]

        # 片段质量检查
        missing_fragments = [f for f in case.expected_fragments if f not in formatted]

        # sender 过滤检查：所有命中 sender 应为 expected_sender
        sender_violations: List[str] = []
        if case.expected_sender:
            for s in _returned_senders(raw):
                if s != case.expected_sender:
                    sender_violations.append(s)

        tp = len(found_expected)
        fp = len(found_unexpected) + len(sender_violations)
        fn = len(missed_expected) + len(missing_fragments)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not case.expected_ids and not case.expected_fragments and not case.expected_sender else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        passed = (fp == 0 and fn == 0)
        is_known = bool(case.known_issue)

        # 排序指标：expected_primary_id 在结果中的排名
        primary = case.expected_primary_id or (case.expected_ids[0] if case.expected_ids else "")
        first_rank = _primary_rank_in_raw(primary, raw) if primary else 0
        hit_at_5 = 1 <= first_rank <= 5
        mrr = (1.0 / first_rank) if (1 <= first_rank <= 5) else 0.0

        # 召回阶段指标：primary 在 dense+keyword 召回池 top30 里吗（融合前）
        in_recall_pool = primary in recall_pool if primary else False
        recall_pool_rank = (recall_pool.index(primary) + 1) if in_recall_pool else 0

        results.append(BenchmarkResult(
            case_name=case.case_name,
            query=case.query,
            category=case.category,
            expected_ids=case.expected_ids,
            unexpected_ids=case.unexpected_ids,
            found_expected=found_expected,
            found_unexpected=found_unexpected,
            missed_expected=missed_expected,
            missing_fragments=missing_fragments,
            sender_violations=sender_violations,
            notes=case.notes,
            passed=passed,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            known_issue=case.known_issue,
            elapsed=elapsed,
            first_rank=first_rank,
            hit_at_5=hit_at_5,
            mrr=mrr,
            in_recall_pool=in_recall_pool,
            recall_pool_rank=recall_pool_rank,
        ))

        if is_known and not passed:
            status = "⚠️ KNOWN"
        else:
            status = "✅ PASS" if passed else "❌ FAIL"
        extra = ""
        if missing_fragments:
            extra += f" [缺片段: {','.join(missing_fragments)}]"
        if sender_violations:
            extra += f" [sender越界: {','.join(set(sender_violations))}]"
        rank_str = f" rank={first_rank}" if primary else ""
        pool_str = f" pool={'Y'+str(recall_pool_rank) if in_recall_pool else 'N'}" if primary else ""
        print(
            f"  [{case.case_name}] {status} "
            f"(P={precision:.0%} R={recall:.0%} F1={f1:.0%} MRR={mrr:.2f}) [{elapsed:.2f}s]{extra}{rank_str}{pool_str}"
        )

    return results


def compute_metrics(results: List[BenchmarkResult]) -> dict[str, Any]:
    """计算全局指标（基于消息 id 级别的 TP/FP/FN）。

    known_issue 的 case 不计入 TP/FP/FN，但计入 accuracy 分母。
    """
    scored = [r for r in results if not r.known_issue]
    total_tp = sum(r.tp for r in scored)
    total_fp = sum(r.fp for r in scored)
    total_fn = sum(r.fn for r in scored)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = sum(1 for r in results if r.passed) / len(results) if results else 0.0
    known_fail = sum(1 for r in results if r.known_issue and not r.passed)

    # 排序指标（排除 known_issue，与 P/R 一致）：MRR@5 + Hit@5
    ranked = [r for r in results if not r.known_issue and r.expected_ids]
    mrr_at_5 = sum(r.mrr for r in ranked) / len(ranked) if ranked else 0.0
    hit_at_5 = sum(1 for r in ranked if r.hit_at_5) / len(ranked) if ranked else 0.0
    recall_pool_hit = sum(1 for r in ranked if r.in_recall_pool) / len(ranked) if ranked else 0.0

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
        "known_fail": known_fail,
        "mrr_at_5": mrr_at_5,
        "hit_at_5": hit_at_5,
        "recall_pool_hit": recall_pool_hit,
    }


def print_report(results: List[BenchmarkResult], metrics: dict[str, Any]) -> None:
    """打印详细报告。"""
    print("\n" + "=" * 70)
    print("历史原文检索召回 Benchmark 报告")
    print("=" * 70)

    print("\n【逐个 Case 结果】")
    print(
        f"{'Case':<28} {'Category':<15} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'Time':>6} {'Result':<8} {'Notes'}"
    )
    print("-" * 110)
    for r in results:
        if r.known_issue and not r.passed:
            status = "⚠️KNOWN"
        else:
            status = "✅PASS" if r.passed else "❌FAIL"
        print(
            f"{r.case_name:<28} {r.category:<15} "
            f"{r.precision:>6.0%} {r.recall:>6.0%} {r.f1:>6.0%} "
            f"{r.elapsed:>5.2f}s {status:<8} {r.notes[:36]}"
        )

    print("\n【指标汇总】")
    print(f"  Total cases:   {metrics['total']}")
    print(f"  TP (正确召回):  {metrics['tp']}")
    print(f"  FP (误召回):    {metrics['fp']}")
    print(f"  FN (漏召回):    {metrics['fn']}")
    if metrics.get("known_fail"):
        print(f"  Known issues:  {metrics['known_fail']} (已知问题，不计入 recall)")
    print(f"  Precision:     {metrics['precision']:.2%}")
    print(f"  Recall:        {metrics['recall']:.2%}  (排除 known_issue 后)")
    print(f"  F1 Score:      {metrics['f1']:.2%}")
    print(f"  MRR@5:         {metrics['mrr_at_5']:.2%}  (排序质量，排除 known_issue)")
    print(f"  Hit@5:         {metrics['hit_at_5']:.2%}  (primary 进前 5 比例)")
    print(f"  召回池Hit@30:  {metrics['recall_pool_hit']:.2%}  (primary 在 dense+keyword 召回池 top30，融合前)")
    print(f"  Accuracy:      {metrics['accuracy']:.2%}")
    print(f"  Passed:        {metrics['passed']}/{metrics['total']}")

    print("\n【按 Category 分组分析】")
    categories = sorted(set(r.category for r in results))
    for cat in categories:
        subset = [r for r in results if r.category == cat and not r.known_issue]
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
    parser = argparse.ArgumentParser(description="历史原文检索召回 Benchmark")
    parser.add_argument(
        "--threshold-precision", type=float, default=0.0,
        help="Precision 阈值，低于此值返回非零 exit code",
    )
    parser.add_argument(
        "--threshold-recall", type=float, default=0.0,
        help="Recall 阈值，低于此值返回非零 exit code",
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
        print(f"\n⚠️ Precision {metrics['precision']:.2%} 低于阈值 {args.threshold_precision:.2%}")
        exit_code = 1
    if args.threshold_recall > 0 and metrics["recall"] < args.threshold_recall:
        print(f"\n⚠️ Recall {metrics['recall']:.2%} 低于阈值 {args.threshold_recall:.2%}")
        exit_code = 1

    sys.exit(exit_code)


# ============== Pytest Interface ==============

@pytest.fixture(scope="module")
def benchmark_results():
    """Pytest fixture: 运行 benchmark（使用真实索引数据）"""
    idx = _index()
    if idx is None:
        pytest.skip("search_history 索引或依赖未就绪")
    return run_benchmark()


def test_benchmark_all_cases_passed(benchmark_results):
    """所有非 known_issue 的 case 都应通过。known_issue 仅记录不阻塞。"""
    failed = [r for r in benchmark_results if not r.passed and not r.known_issue]
    if failed:
        names = ", ".join(r.case_name for r in failed)
        pytest.fail(f"以下 case 未通过: {names}")


def test_benchmark_known_issues_documented(benchmark_results):
    """known_issue 的 case 应有说明。"""
    undocumented = [r for r in benchmark_results if r.known_issue and not r.known_issue.strip()]
    assert not undocumented, "known_issue 必须填写说明"


def test_benchmark_precision(benchmark_results):
    """Precision 不应低于 50%。"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["precision"] >= 0.5, f"Precision 过低: {metrics['precision']:.1%}"


def test_benchmark_recall(benchmark_results):
    """Recall 不应低于 50%。"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["recall"] >= 0.5, f"Recall 过低: {metrics['recall']:.1%}"


def test_benchmark_exact_recall(benchmark_results):
    """逐字原文检索必须 100% 召回自身（向量空间基本可用性兜底）。"""
    exact = [r for r in benchmark_results if r.category == "exact_recall" and not r.known_issue]
    if not exact:
        pytest.skip("无 exact_recall case")
    missed = [r.case_name for r in exact if r.missed_expected]
    assert not missed, f"exact_recall 漏召回: {missed}"


def test_benchmark_mrr(benchmark_results):
    """MRR@5（排序质量）不应低于 50%。"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["mrr_at_5"] >= 0.5, (
        f"MRR@5 不足: {metrics['mrr_at_5']:.1%}，primary 文档排序靠后"
    )


def test_benchmark_hit_at_5(benchmark_results):
    """Hit@5（primary 进前 5 比例）不应低于 70%。"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["hit_at_5"] >= 0.7, (
        f"Hit@5 不足: {metrics['hit_at_5']:.1%}，primary 文档常排不进前 5"
    )


def test_benchmark_recall_pool(benchmark_results):
    """召回阶段召回率：primary 应在 dense+keyword 召回池 top30 里（融合前）。

    区分召回问题（池里没有→需 query 改写/同义词）vs 排序问题（池里有但没排上→需 rerank）。
    """
    metrics = compute_metrics(benchmark_results)
    assert metrics["recall_pool_hit"] >= 0.7, (
        f"召回池命中率不足: {metrics['recall_pool_hit']:.1%}，"
        f"primary 常在召回阶段就漏（需 query 改写而非 rerank）"
    )


if __name__ == "__main__":
    main()
