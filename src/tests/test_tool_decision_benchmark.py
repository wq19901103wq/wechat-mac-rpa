#!/usr/bin/env python3
"""
Tool 调用决策 benchmark - search_memory 过度调用检测。

验证在给定对话场景下，LLM 是否正确地决定调用/不调用 search_memory。

用法:
    # 使用真实 API（计算 precision/recall）
    python -m pytest src/tests/test_tool_decision_benchmark.py -v --run-api

    # 命令行直接运行（带详细报告）
    python src/tests/test_tool_decision_benchmark.py --run-api

    # 使用缓存（快速回归）
    pytest src/tests/test_tool_decision_benchmark.py -v

Fixture 目录结构:
    src/tests/fixtures/tool_decision/
        {case_name}/
            llm_response.json   # 缓存的 LLM 响应

缓存内容格式:
    {"tool_calls": [...], "content": "...", "timestamp": ...}
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tests.test_reply_quality_benchmark import JudgeLLM, Rubric, RubricDimension  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tool_decision"

@dataclass
class BenchmarkCase:
    case_name: str
    user_message: str
    should_call_memory: bool
    category: str
    notes: str = ""
    rubric: Optional[Rubric] = None


@dataclass
class BenchmarkResult:
    case_name: str
    should_call: bool
    actually_called: bool
    called_tools: List[str]
    category: str
    passed: bool
    raw_response_preview: str
    error: str = ""
    pass_rate: float = 1.0
    n_runs: int = 1
    rubric_scores: Optional[Dict[str, Any]] = None
    evaluation_mode: str = "binary"


# 预定义的 benchmark cases
BENCHMARK_CASES: List[BenchmarkCase] = [
    # === 应该调用 search_memory ===
    BenchmarkCase(
        case_name="person_identity_wanghai",
        user_message="王海是谁？",
        should_call_memory=True,
        category="person_identity",
        notes="人物身份查询，必须调用 search_memory",
    ),
    BenchmarkCase(
        case_name="relationship_chengli_wangqian",
        user_message="程立和王芊什么关系？",
        should_call_memory=True,
        category="relationship",
        notes="跨人物关系查询，必须调用 search_memory",
    ),
    BenchmarkCase(
        case_name="background_wangqian_location",
        user_message="王芊住在哪里？",
        should_call_memory=True,
        category="background",
        notes="人物背景/地点查询，必须调用 search_memory",
    ),
    BenchmarkCase(
        case_name="attribute_wangyihan_mbti",
        user_message="王艺涵是什么MBTI？",
        should_call_memory=True,
        category="attribute",
        notes="人物属性（MBTI）查询，必须调用 search_memory",
    ),
    BenchmarkCase(
        case_name="group_members_ai_team",
        user_message="ai开发小分队里有哪些人？",
        should_call_memory=True,
        category="group_members",
        notes="群成员信息查询，涉及人物，必须调用 search_memory",
    ),
    BenchmarkCase(
        case_name="relationship_wangqian_spouse",
        user_message="王芊的老婆是谁？",
        should_call_memory=True,
        category="relationship",
        notes="人物关系查询，必须调用 search_memory",
    ),
    # === 不应该调用 search_memory ===
    BenchmarkCase(
        case_name="greeting_hello",
        user_message="你好",
        should_call_memory=False,
        category="greeting",
        notes="打招呼，不需要任何工具",
    ),
    BenchmarkCase(
        case_name="weather_today",
        user_message="今天天气怎么样？",
        should_call_memory=False,
        category="weather",
        notes="天气查询，应调用 get_weather",
    ),
    BenchmarkCase(
        case_name="stock_maotai",
        user_message="茅台股票多少了？",
        should_call_memory=False,
        category="stock",
        notes="股票查询，应调用 stock_query",
    ),
    BenchmarkCase(
        case_name="time_now",
        user_message="现在几点？",
        should_call_memory=False,
        category="time",
        notes="时间查询，应调用 get_current_time",
    ),
    BenchmarkCase(
        case_name="chat_joke",
        user_message="讲个笑话",
        should_call_memory=False,
        category="chat",
        notes="纯闲聊，不需要任何工具",
    ),
    BenchmarkCase(
        case_name="websearch_news",
        user_message="帮我搜一下今天的新闻",
        should_call_memory=False,
        category="websearch",
        notes="网页搜索，应调用 web_search",
    ),
    # === 对抗性/边界 case：涉及人物但应调其他 tool ===
    BenchmarkCase(
        case_name="adversarial_link",
        user_message="王芊刚才发的链接你看了吗？",
        should_call_memory=False,
        category="adversarial",
        notes="涉及人物但应调 browse_url，不应调 search_memory",
    ),
    BenchmarkCase(
        case_name="adversarial_websearch",
        user_message="帮我搜一下王芊的最新动态",
        should_call_memory=False,
        category="adversarial",
        notes="涉及人物但应调 web_search，不应调 search_memory",
    ),
    BenchmarkCase(
        case_name="adversarial_stock",
        user_message="王芊买的茅台涨了吗",
        should_call_memory=False,
        category="adversarial",
        notes="涉及人物但应调 stock_query，不应调 search_memory",
    ),
    BenchmarkCase(
        case_name="adversarial_statement",
        user_message="我也在拼多多上班",
        should_call_memory=False,
        category="adversarial",
        notes="陈述句，不是询问，不需要工具",
    ),
    BenchmarkCase(
        case_name="adversarial_vague",
        user_message="你知道王芊这个人吗",
        should_call_memory=False,
        category="adversarial",
        notes="简单知道，不涉及具体事实查询",
    ),
    # === 更多应调用 search_memory（表述更隐晦）===
    BenchmarkCase(
        case_name="person_identity_kuige",
        user_message="盔哥是谁？",
        should_call_memory=True,
        category="person_identity",
        notes="别名人物身份查询",
    ),
    BenchmarkCase(
        case_name="relationship_mother_in_law",
        user_message="王芊的岳母是谁？",
        should_call_memory=True,
        category="relationship",
        notes="人物关系查询（岳母）",
    ),
    BenchmarkCase(
        case_name="background_xiaohaige",
        user_message="小海哥和王芊什么关系？",
        should_call_memory=True,
        category="relationship",
        notes="跨人物关系查询",
    ),
    BenchmarkCase(
        case_name="career_wangyihan",
        user_message="王艺涵在阿里做什么的？",
        should_call_memory=True,
        category="background",
        notes="人物职业背景查询",
    ),
    BenchmarkCase(
        case_name="family_size",
        user_message="王芊家里几口人？",
        should_call_memory=True,
        category="background",
        notes="家庭背景查询",
    ),
    BenchmarkCase(
        case_name="group_purpose",
        user_message="ai开发小分队是干嘛的？",
        should_call_memory=True,
        category="group_members",
        notes="群背景查询",
    ),
    # === 更多不应调用 search_memory ===
    BenchmarkCase(
        case_name="emotion_haha",
        user_message="哈哈哈",
        should_call_memory=False,
        category="chat",
        notes="纯情绪表达，不需要工具",
    ),
    BenchmarkCase(
        case_name="polite_thanks",
        user_message="谢谢",
        should_call_memory=False,
        category="chat",
        notes="礼貌用语，不需要工具",
    ),
    BenchmarkCase(
        case_name="network_slang",
        user_message="6",
        should_call_memory=False,
        category="chat",
        notes="网络用语，不需要工具",
    ),
    BenchmarkCase(
        case_name="goodnight",
        user_message="晚安",
        should_call_memory=False,
        category="chat",
        notes="日常寒暄，不需要工具",
    ),
]


def _get_api_key() -> str | None:
    """从环境变量或 .env 文件读取 API key"""
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENCLAW_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DASHSCOPE_API_KEY="):
                        api_key = line.split("=", 1)[1]
                        break
                    elif line.startswith("DEEPSEEK_API_KEY="):
                        api_key = line.split("=", 1)[1]
                        break
    return api_key


def _read_cached_response(case_name: str) -> dict | None:
    """读取缓存的 LLM 响应"""
    cache_path = FIXTURE_DIR / case_name / "llm_response.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def _save_cached_response(case_name: str, data: dict) -> None:
    """保存 LLM 响应到缓存"""
    case_dir = FIXTURE_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_dir / "llm_response.json"
    with open(cache_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_tool_rubric(case: BenchmarkCase) -> Rubric:
    """为 tool 决策 case 构建 rubric"""
    if case.should_call_memory:
        return Rubric(
            instructions=f"评估 LLM 的 tool 决策。用户消息：'{case.user_message}'",
            dimensions=[
                RubricDimension(
                    name="应调用 search_memory",
                    description="用户消息涉及需要查询记忆的人物信息",
                    criteria="用户消息涉及人物身份、关系、背景、属性等需要查询本地 wiki 记忆的信息，应调用 search_memory",
                    required=True,
                ),
            ],
        )
    else:
        if case.category == "adversarial":
            return Rubric(
                instructions=f"评估 LLM 的 tool 决策（对抗性 case）。用户消息：'{case.user_message}'",
                dimensions=[
                    RubricDimension(
                        name="未过度调用 search_memory",
                        description="不应仅因出现人名就调用 search_memory",
                        criteria="不应仅因为消息中出现人名就条件反射地调用 search_memory。需要分析用户的真实意图：是查询人物信息，还是查询天气/股票/链接等其他内容",
                        required=True,
                    ),
                    RubricDimension(
                        name="正确工具选择",
                        description="如果用户意图需要工具，应选择正确的工具",
                        criteria=f"{case.notes}",
                        required=True,
                    ),
                ],
            )
        else:
            return Rubric(
                instructions=f"评估 LLM 的 tool 决策。用户消息：'{case.user_message}'",
                dimensions=[
                    RubricDimension(
                        name="不调用 search_memory",
                        description="用户消息不涉及需要查询记忆的人物信息",
                        criteria="用户消息不涉及需要查询记忆的人物信息，不应调用 search_memory",
                        required=True,
                    ),
                ],
            )


def _tool_calls_hash(tool_calls: List[dict]) -> str:
    """为 tool_calls 生成 hash，用于 judge 缓存 key"""
    return hashlib.sha256(json.dumps(tool_calls, sort_keys=True).encode()).hexdigest()[:16]


def _read_judge_cache(case_name: str, tool_calls: List[dict]) -> dict | None:
    cache_path = FIXTURE_DIR / case_name / f"judge_{_tool_calls_hash(tool_calls)}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def _save_judge_cache(case_name: str, tool_calls: List[dict], result: dict) -> None:
    case_dir = FIXTURE_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_dir / f"judge_{_tool_calls_hash(tool_calls)}.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _call_llm(case: BenchmarkCase, api_key: str | None = None) -> dict:
    """调用 ReplyGenerator 生成回复，从 last_tool_calls 提取 tool calls"""
    from src.models.base import ChatMessage, SenderType
    from src.reply.generator import ReplyGenerator
    from src.tools import ToolRegistry, register_builtin_tools
    from src.utils.qwen_client import QwenClient

    # 如果外部传入了 api_key，临时设置到环境变量
    env_backup = None
    if api_key:
        env_backup = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = api_key

    try:
        # 创建独立工具注册表
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # 注册 search_memory（实验必需）
        def _mock_search_memory(query: str = "") -> str:
            return f"[记忆搜索结果] {query}"
        registry.register(
            name="search_memory",
            description="搜索本地长期记忆。当你不确定某个人是谁、某件事的背景、或者某个关系时，调用此工具查询本地 wiki 记忆库。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词。必须是单个具体的人名、昵称或名词，不要组合多个词。",
                    },
                },
                "required": ["query"],
            },
            func=_mock_search_memory,
        )

        # 构造 ChatMessage
        msg = ChatMessage(
            text=case.user_message,
            sender="User",
            sender_type=SenderType.OTHER,
            chat_name="Benchmark",
        )

        llm_client = QwenClient(model="deepseek-v4-flash")
        reply_generator = ReplyGenerator(
            llm_client=llm_client,
            tool_registry=registry,
            judge_worker=None,
        )

        replies = reply_generator.generate(
            unreplied=[msg],
            all_messages=[msg],
            is_group=False,
        )
    finally:
        if env_backup is not None:
            os.environ["DASHSCOPE_API_KEY"] = env_backup
        elif api_key and "DASHSCOPE_API_KEY" in os.environ:
            del os.environ["DASHSCOPE_API_KEY"]

    # 从 last_tool_calls 提取 tool calls
    result: dict[str, Any] = {"tool_calls": [], "content": "", "timestamp": time.time()}
    tool_calls = reply_generator.last_tool_calls or []
    for tc in tool_calls:
        result["tool_calls"].append({
            "id": tc.get("tool_call_id", ""),
            "type": "function",
            "function": {
                "name": tc.get("tool_name", ""),
                "arguments": tc.get("arguments", ""),
            },
        })

    # 内容：取 replies 的拼接，或 raw_response
    if replies:
        result["content"] = " | ".join(replies)
    else:
        result["content"] = reply_generator.last_raw_response or ""

    return result


def run_benchmark(use_api: bool = False, api_key: str | None = None, n_runs: int = 1, use_judge: bool = True) -> list[BenchmarkResult]:
    """
    运行 benchmark，返回所有 case 的结果。

    Args:
        use_api: 是否调用真实 API。False 则使用缓存结果。
        api_key: API key（use_api=True 时需要，None 则从环境变量读取）
        n_runs: API 模式下每个 case 运行次数（稳定性测试），默认 1
        use_judge: 是否使用 Judge LLM 评估决策质量（对抗性 case 强制启用）
    """
    results: list[BenchmarkResult] = []

    if use_api and api_key is None:
        api_key = _get_api_key()

    judge = None

    for case in BENCHMARK_CASES:
        run_passes: list[bool] = []
        all_called_tools: list[str] = []
        last_raw_preview = ""
        last_error = ""

        if use_api:
            if not api_key:
                print(f"  [{case.case_name}] 跳过: 未设置 API key")
                results.append(
                    BenchmarkResult(
                        case_name=case.case_name,
                        should_call=case.should_call_memory,
                        actually_called=False,
                        called_tools=[],
                        category=case.category,
                        passed=False,
                        raw_response_preview="",
                        error="未设置 API key",
                        pass_rate=0.0,
                        n_runs=0,
                    )
                )
                continue

            for run in range(n_runs):
                print(f"  [{case.case_name}] 调用 API (run {run+1}/{n_runs}): {case.user_message[:30]}")
                try:
                    llm_response = _call_llm(case, api_key)
                    _save_cached_response(case.case_name, llm_response)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  [{case.case_name}] API 调用失败: {e}")
                    llm_response = {"tool_calls": [], "content": "", "timestamp": time.time(), "error": str(e)}
                    _save_cached_response(case.case_name, llm_response)

                tool_calls = llm_response.get("tool_calls", [])
                called_tools: List[str] = []
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    if name:
                        called_tools.append(name)

                actually_called_memory = "search_memory" in called_tools
                passed = actually_called_memory == case.should_call_memory
                run_passes.append(passed)
                all_called_tools.extend(called_tools)

                content = llm_response.get("content", "")
                raw_preview = content[:80] + "..." if len(content) > 80 else content
                if tool_calls:
                    tc_preview = ", ".join(called_tools)
                    raw_preview = f"[tools: {tc_preview}] {raw_preview}"
                last_raw_preview = raw_preview
                last_error = llm_response.get("error", "")

            pass_rate = sum(run_passes) / len(run_passes) if run_passes else 0.0
            passed = pass_rate >= 0.67  # 2/3 通过算合格
            # 去重工具列表用于展示
            unique_tools = list(dict.fromkeys(all_called_tools))

            results.append(
                BenchmarkResult(
                    case_name=case.case_name,
                    should_call=case.should_call_memory,
                    actually_called=bool([t for t in unique_tools if t == "search_memory"]),
                    called_tools=unique_tools,
                    category=case.category,
                    passed=passed,
                    raw_response_preview=last_raw_preview,
                    error=last_error,
                    pass_rate=pass_rate,
                    n_runs=len(run_passes),
                )
            )
        else:
            llm_response = _read_cached_response(case.case_name)
            if llm_response is None:
                print(f"  [{case.case_name}] 跳过: 无缓存结果且未使用 --run-api")
                results.append(
                    BenchmarkResult(
                        case_name=case.case_name,
                        should_call=case.should_call_memory,
                        actually_called=False,
                        called_tools=[],
                        category=case.category,
                        passed=False,
                        raw_response_preview="",
                        error="无缓存结果且未使用 --run-api",
                        pass_rate=0.0,
                        n_runs=0,
                    )
                )
                continue

            tool_calls = llm_response.get("tool_calls", [])
            called_tools: List[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                if name:
                    called_tools.append(name)

            actually_called_memory = "search_memory" in called_tools
            passed = actually_called_memory == case.should_call_memory
            content = llm_response.get("content", "")
            raw_preview = content[:80] + "..." if len(content) > 80 else content
            if tool_calls:
                tc_preview = ", ".join(called_tools)
                raw_preview = f"[tools: {tc_preview}] {raw_preview}"

            results.append(
                BenchmarkResult(
                    case_name=case.case_name,
                    should_call=case.should_call_memory,
                    actually_called=actually_called_memory,
                    called_tools=called_tools,
                    category=case.category,
                    passed=passed,
                    raw_response_preview=raw_preview,
                    error=llm_response.get("error", ""),
                    pass_rate=1.0 if passed else 0.0,
                    n_runs=1,
                )
            )

    # =============================================================
    # Judge 评估：对失败的 case 和对抗性 case 调用 Judge
    # =============================================================
    if use_judge:
        case_map = {c.case_name: c for c in BENCHMARK_CASES}
        for r in results:
            if r.error and "未设置 API key" in r.error:
                continue
            if r.error and "无缓存结果" in r.error:
                continue

            bc = case_map.get(r.case_name)
            if not bc:
                continue

            # 只对失败的 case 或对抗性 case 调用 Judge
            needs_judge = (not r.passed) or (bc.category == "adversarial")
            if not needs_judge:
                continue

            # 构建 tool_calls 用于缓存 key
            tool_calls_for_hash = [{"name": t} for t in r.called_tools]
            judge_cache = _read_judge_cache(r.case_name, tool_calls_for_hash)

            if judge_cache is not None:
                r.rubric_scores = judge_cache
                r.evaluation_mode = "rubric(cached)"
                print(f"  [{r.case_name}] 📋 Judge 缓存")
                continue

            if not _get_api_key():
                print(f"  [{r.case_name}] ⚠️ 无 API key，跳过 Judge")
                continue

            if judge is None:
                print("  [Judge] 初始化 deepseek-v4-pro...")
                judge = JudgeLLM(api_key=_get_api_key())

            rubric = _build_tool_rubric(bc)
            context = f"用户消息: {bc.user_message}\n期望: {'调用 search_memory' if bc.should_call_memory else '不调用 search_memory'}\n实际调用工具: {r.called_tools or '无'}"

            print(f"  [{r.case_name}] 🧑‍⚖️ Judge 评估...")
            rubric_scores = judge.evaluate(
                rubric=rubric,
                context=context,
                replies=[r.raw_response_preview or "(无文本回复)"],
                case_notes=bc.notes,
            )
            _save_judge_cache(r.case_name, tool_calls_for_hash, rubric_scores)
            r.rubric_scores = rubric_scores
            r.evaluation_mode = "rubric"
            time.sleep(0.3)

    return results


def compute_metrics(results: list[BenchmarkResult]) -> dict[str, Any]:
    """计算 precision, recall, F1, accuracy"""
    tp = sum(1 for r in results if r.should_call and r.actually_called)
    fp = sum(1 for r in results if not r.should_call and r.actually_called)
    fn = sum(1 for r in results if r.should_call and not r.actually_called)
    tn = sum(1 for r in results if not r.should_call and not r.actually_called)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
    }


def print_report(results: list[BenchmarkResult], metrics: dict[str, Any]) -> None:
    """打印详细报告"""
    print("\n" + "=" * 90)
    print("Tool 调用决策 Benchmark 报告 (search_memory)")
    print("=" * 90)

    print("\n【逐个 Case 结果】")
    print(
        f"{'Case':<40} {'期望':<6} {'实际':<6} {'结果':<8} {'通过率':>8} {'调用工具':<22} {'类别':<15}"
    )
    print("-" * 120)
    for r in results:
        expect = "是" if r.should_call else "否"
        actual = "是" if r.actually_called else "否"
        status = "✅ PASS" if r.passed else "❌ FAIL"
        rate_str = f"{r.pass_rate:.0%}({r.n_runs})" if r.n_runs > 1 else "-"
        tools_str = ", ".join(r.called_tools) if r.called_tools else "无"
        tools_str = tools_str[:19] + "..." if len(tools_str) > 22 else tools_str
        print(
            f"{r.case_name:<40} {expect:<6} {actual:<6} {status:<8} {rate_str:>8} {tools_str:<22} {r.category:<15}"
        )

    print("\n【指标汇总】")
    print(f"  Total cases:   {metrics['total']}")
    print(f"  TP (正确调用):  {metrics['tp']}")
    print(f"  FP (过度调用):  {metrics['fp']}")
    print(f"  FN (漏调用):    {metrics['fn']}")
    print(f"  TN (正确不调用): {metrics['tn']}")
    print(f"  Precision:     {metrics['precision']:.2%}")
    print(f"  Recall:        {metrics['recall']:.2%}")
    print(f"  F1 Score:      {metrics['f1']:.2%}")
    print(f"  Accuracy:      {metrics['accuracy']:.2%}")
    print(f"  Passed:        {metrics['passed']}/{metrics['total']}")

    # 按 should_call 分组
    print("\n【按期望调用分组】")
    should_call_cases = [r for r in results if r.should_call]
    should_not_call_cases = [r for r in results if not r.should_call]
    if should_call_cases:
        m = compute_metrics(should_call_cases)
        print(
            f"  应调用 search_memory:  Precision={m['precision']:.0%} Recall={m['recall']:.0%} "
            f"(TP={m['tp']} FN={m['fn']})"
        )
    if should_not_call_cases:
        m = compute_metrics(should_not_call_cases)
        print(
            f"  不应调用 search_memory: Precision={m['precision']:.0%} Recall={m['recall']:.0%} "
            f"(TN={m['tn']} FP={m['fp']})"
        )

    # 失败的 case 详情
    failed = [r for r in results if not r.passed]
    if failed:
        print("\n【失败 Case 详情】")
        for r in failed:
            expect = "应调用" if r.should_call else "不应调用"
            actual = "实际调用" if r.actually_called else "实际未调用"
            print(f"  ❌ {r.case_name} ({r.category}): {expect}，{actual}")
            print(f"      原始响应: {r.raw_response_preview}")
            if r.error:
                print(f"      错误: {r.error}")

    # Judge 评估详情（对抗性 case）
    judged = [r for r in results if r.rubric_scores]
    if judged:
        print("\n🧑‍⚖️ Judge 评估详情")
        for r in judged:
            print(f"\n  [{r.case_name}] ({r.evaluation_mode})")
            for d in r.rubric_scores.get("dimensions", []):
                icon = "✅" if d["score"] == "PASS" else "❌"
                req = "(必须)" if d.get("required", True) else "(参考)"
                print(f"     {icon} {d['name']} {req}: {d['reason']}")
            if r.rubric_scores.get("explanation"):
                print(f"     💡 {r.rubric_scores['explanation']}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Tool 调用决策 Benchmark")
    parser.add_argument("--run-api", action="store_true", help="调用真实 API（否则使用缓存）")
    parser.add_argument("--api-key", default=None, help="API key（默认从环境变量读取）")
    parser.add_argument(
        "--n-runs", type=int, default=1, help="每个 case 运行次数（稳定性测试，默认 1）"
    )
    parser.add_argument(
        "--threshold-precision", type=float, default=0.0, help="Precision 阈值"
    )
    parser.add_argument(
        "--threshold-recall", type=float, default=0.0, help="Recall 阈值"
    )
    parser.add_argument(
        "--threshold-accuracy", type=float, default=0.0, help="Accuracy 阈值"
    )
    args = parser.parse_args()

    results = run_benchmark(use_api=args.run_api, api_key=args.api_key, n_runs=args.n_runs)
    metrics = compute_metrics(results)
    print_report(results, metrics)

    exit_code = 0
    if args.threshold_precision > 0 and metrics["precision"] < args.threshold_precision:
        print(
            f"\n⚠️  Precision {metrics['precision']:.2%} 低于阈值 {args.threshold_precision:.2%}"
        )
        exit_code = 1
    if args.threshold_recall > 0 and metrics["recall"] < args.threshold_recall:
        print(
            f"\n⚠️  Recall {metrics['recall']:.2%} 低于阈值 {args.threshold_recall:.2%}"
        )
        exit_code = 1
    if args.threshold_accuracy > 0 and metrics["accuracy"] < args.threshold_accuracy:
        print(
            f"\n⚠️  Accuracy {metrics['accuracy']:.2%} 低于阈值 {args.threshold_accuracy:.2%}"
        )
        exit_code = 1

    sys.exit(exit_code)


# ============== Pytest 接口 ==============


def _has_api_key_or_cache() -> bool:
    """检查是否有 API key 或至少一个 case 的缓存"""
    if _get_api_key():
        return True
    for case in BENCHMARK_CASES:
        if _read_cached_response(case.case_name) is not None:
            return True
    return False


@pytest.fixture(scope="module")
def benchmark_results():
    """Pytest fixture: 运行 benchmark（使用缓存）"""
    return run_benchmark(use_api=False)


def test_all_cases_passed(benchmark_results):
    """所有 case 都应通过（当前预期可能失败，用于记录 baseline）"""
    failed = [r for r in benchmark_results if not r.passed]
    if failed:
        names = ", ".join(r.case_name for r in failed)
        pytest.fail(f"以下 case 未通过: {names}")


def test_precision_threshold(benchmark_results):
    """Precision 不应过低"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["precision"] >= 0.5, f"Precision 过低: {metrics['precision']:.1%}"


def test_recall_threshold(benchmark_results):
    """Recall 不应过低"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["recall"] >= 0.5, f"Recall 过低: {metrics['recall']:.1%}"


def test_accuracy_threshold(benchmark_results):
    """Accuracy 不应过低"""
    metrics = compute_metrics(benchmark_results)
    assert metrics["accuracy"] >= 0.5, f"Accuracy 过低: {metrics['accuracy']:.1%}"


@pytest.mark.skipif(not _has_api_key_or_cache(), reason="未设置 API key 且无缓存")
def test_with_real_api():
    """使用真实 API 运行 benchmark（手动触发）"""
    results = run_benchmark(use_api=True)
    metrics = compute_metrics(results)
    print_report(results, metrics)
    assert metrics["precision"] >= 0.5, f"Precision 过低: {metrics['precision']:.1%}"
    assert metrics["recall"] >= 0.5, f"Recall 过低: {metrics['recall']:.1%}"


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------
# Auto-generated cases
# -------------------------------------------------------------------------
# test commit from review

