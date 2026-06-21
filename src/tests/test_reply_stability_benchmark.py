#!/usr/bin/env python3
"""
Bot 回复稳定性 Benchmark — 直接用生产 case 的完整 prompt 调 LLM 生成 3 次

与线上一致：
- prompt: 使用 judge_quality case 的 full_user_prompt + full_system_prompt（生产数据）
- LLM: QwenClient（与 run_bot.py 相同）
- Judge: JudgeWorker._judge()（与生产 badcase 闭环相同）
- 每个 case 调 LLM 3 次 → 得到 3 个不同回复 → 每个 Judge 评一次 → 平均

用法:
    python src/tests/test_reply_stability_benchmark.py --run-api --n-generations 3
    pytest src/tests/test_reply_stability_benchmark.py -v              # 缓存回归
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reply_stability"
CACHE_DIR = FIXTURE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StabilityResult:
    case_name: str
    source: str
    notes: str
    context_msgs: list[str]
    full_user_prompt: str
    full_system_prompt: str
    replies: list[str] = field(default_factory=list)
    reply_scores: list[dict] = field(default_factory=list)
    avg_overall_score: float = 0
    overall_score_std: float = 0
    avg_dimensions: dict = field(default_factory=dict)
    cross_similarity: float = 0
    n_generations: int = 0
    error: str = ""


def _load_cases() -> list[StabilityResult]:
    """从生产 review_drafts 加载真实的 full_user_prompt + full_system_prompt。"""
    import json as _json
    cases = []
    drafts_dir = PROJECT_ROOT / "data" / "review_drafts" / "committed"

    for f in sorted(drafts_dir.glob("*.json")):
        if "test" in f.name:
            continue
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        up = d.get("full_user_prompt", "")
        sp = d.get("full_system_prompt", "")
        if not up:
            continue  # 没有真实 prompt，跳过

        judge = d.get("judge_result", {})
        conv = d.get("conversation", [])
        context_lines = []
        for m in conv[-15:]:
            sender = m.get("sender", "?")
            text = m.get("text", "")
            context_lines.append(f"{sender}: {text}")

        cases.append(StabilityResult(
            case_name=d["draft_id"].replace(":", "-")[:50],
            source=d.get("chat_name", ""),
            notes=f"{judge.get('badcase_type', '?')} | {judge.get('reason', '')[:150]}",
            context_msgs=context_lines,
            full_user_prompt=up,
            full_system_prompt=sp,
        ))

    # 也加载 pending 里有真实 prompt 的
    pending_dir = PROJECT_ROOT / "data" / "review_drafts" / "pending"
    for f in sorted(pending_dir.glob("*.json")):
        if "mock" in f.name:
            continue
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        up = d.get("full_user_prompt", "")
        sp = d.get("full_system_prompt", "")
        if not up:
            continue
        judge = d.get("judge_result", {})
        conv = d.get("conversation", [])
        context_lines = []
        for m in conv[-15:]:
            context_lines.append(f"{m.get('sender', '?')}: {m.get('text', '')}")
        cases.append(StabilityResult(
            case_name=d["draft_id"].replace(":", "-")[:50],
            source=d.get("chat_name", ""),
            notes=f"{judge.get('badcase_type', '?')} | {judge.get('reason', '')[:150]}",
            context_msgs=context_lines,
            full_user_prompt=up,
            full_system_prompt=sp,
        ))

    return cases


def _reply_similarity(replies: list[str]) -> float:
    if len(replies) < 2:
        return 0.0
    sims = []
    for i in range(len(replies)):
        for j in range(i + 1, len(replies)):
            wa = set(replies[i])
            wb = set(replies[j])
            if wa | wb:
                sims.append(len(wa & wb) / len(wa | wb))
    return statistics.mean(sims) if sims else 0.0


def run_benchmark(use_api: bool = False, api_key: str | None = None, n_generations: int = 3) -> list[StabilityResult]:
    cases = _load_cases()
    results: list[StabilityResult] = []
    tools: list[dict] = []

    if use_api:
        # 加载 .env（与 run_bot.py 相同）
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key

        import json as _json

        from src.badcase.judge_worker import JudgeWorker
        from src.memory import MemoryEngine
        from src.tools.builtin_tools import register_builtin_tools
        from src.tools.tool_registry import get_registry
        from src.utils.qwen_client import QwenClient

        llm = QwenClient()
        judge = JudgeWorker()

        # 注册工具（与线上一致）
        registry = get_registry()
        register_builtin_tools()
        # 注册 search_memory
        mem = MemoryEngine()
        def _search_memory(query: str = "") -> str:
            return mem.search_keyword(query)
        registry.register(
            name="search_memory",
            description="搜索本地长期记忆。当你不确定某个人是谁、某件事的背景、或者某个关系时，调用此工具查询本地 wiki 记忆库。",
            parameters={"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]},
            func=_search_memory,
        )
        tools = registry.to_openai_schemas()

    for case in cases:
        if not case.full_user_prompt:
            case.error = "无 full_user_prompt"
            results.append(case)
            continue

        all_runs = []

        if use_api:
            if not os.environ.get("DASHSCOPE_API_KEY"):
                case.error = "未设置 API key"
                results.append(case)
                continue

            print(f"  [{case.case_name}] 生成 {n_generations} 次...", end=" ")
            sys.stdout.flush()

            sp = case.full_system_prompt  # 直接用生产 prompt，不自己编

            for i in range(n_generations):
                msgs = []
                if sp:
                    msgs.append({"role": "system", "content": sp})
                msgs.append({"role": "user", "content": case.full_user_prompt})

                tool_log = []
                final_reply = ""
                raw = None
                max_rounds = 3
                try:
                    for round_i in range(max_rounds):
                        raw = llm.chat(messages=msgs, tools=tools, max_tokens=500, timeout=60)
                        # 检查是否返回了 tool_calls
                        if hasattr(raw, "tool_calls") and raw.tool_calls:
                            msgs.append({"role": "assistant", "content": raw.content or "",
                                         "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in raw.tool_calls]})
                            for tc in raw.tool_calls:
                                name = tc.function.name
                                args_str = tc.function.arguments
                                if registry.has(name):
                                    result = registry.get(name).execute(args_str)
                                else:
                                    result = f"工具 {name} 不存在"
                                tool_log.append({"name": name, "args": args_str, "result": str(result)[:500]})
                                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                            continue  # 继续下一轮，让 LLM 基于工具结果回复
                        else:
                            final_reply = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
                            break

                    if not final_reply and raw is not None:
                        final_reply = raw if isinstance(raw, str) else getattr(raw, "content", str(raw)) or "(空)"
                    # 解析 JSON 格式: {"replies": ["text1", "text2"]}
                    if final_reply.strip().startswith("{"):
                        try:
                            data = _json.loads(final_reply.strip())
                            replies_list = data.get("replies", [])
                            if replies_list:
                                final_reply = " | ".join(str(r) for r in replies_list)
                        except Exception:
                            pass  # 不是 JSON，保持原样
                    print("✓", end="", flush=True)
                except Exception as e:
                    final_reply = f"[ERROR: {e}]"
                    print("✗", end="", flush=True)

                all_runs.append({
                    "reply": final_reply,
                    "tool_log": tool_log,
                    "system_prompt": sp,  # 实际使用的 system prompt
                    "user_prompt": case.full_user_prompt,
                })
                if i < n_generations - 1:
                    time.sleep(0.5)
            print()

            # Judge each reply
            for run in all_runs:
                tick_data = {
                    "tick_id": case.case_name,
                    "session_input_messages": [],
                    "bot_reply_text": run["reply"],
                    "tool_calls": [{"function": {"name": t["name"], "arguments": t["args"]}} for t in run.get("tool_log", [])],
                    "full_user_prompt": case.full_user_prompt,
                    "full_system_prompt": case.full_system_prompt,
                    "full_tools_context": "",
                    "full_llm_messages": [],
                }
                try:
                    jr = judge._judge(tick_data)
                    run["judge"] = jr
                except Exception as e:
                    run["judge"] = {"error": str(e), "overall_score": 0, "dimensions": {}}

            _write_cache(case.case_name, all_runs)
        else:
            all_runs = _read_cache(case.case_name)
            if not all_runs:
                case.error = "无缓存"
                results.append(case)
                continue

        # 聚合
        replies = [r.get("reply", "") for r in all_runs]
        judges = [r.get("judge", {}) for r in all_runs]
        scores = [float(j.get("overall_score", 0)) for j in judges if "error" not in j]
        avg_score = round(statistics.mean(scores), 1) if scores else 0
        score_std = round(statistics.stdev(scores), 1) if len(scores) >= 2 else 0

        # 每次回复得分
        reply_scores = []
        for i, run in enumerate(all_runs):
            j = run.get("judge", {})
            dims = {}
            for name, dd in j.get("dimensions", {}).items():
                dims[name] = dd.get("score", 0)
            reply_scores.append({
                "reply": run.get("reply", "")[:500],
                "overall_score": j.get("overall_score", 0),
                "is_badcase": j.get("is_badcase", False),
                "dimensions": dims,
                "tool_log": run.get("tool_log", []),
            })

        # 平均维度
        dim_names = ["幻觉控制", "记忆召回", "幽默感", "逼格语气", "个性一致性", "简洁度", "上下文理解"]
        avg_dims = {}
        for name in dim_names:
            vals = [float(j.get("dimensions", {}).get(name, {}).get("score", 0))
                    for j in judges if "error" not in j]
            if vals:
                std_val = round(statistics.stdev(vals), 1) if len(vals) >= 2 and len(set(vals)) > 1 else 0
                avg_dims[name] = {
                    "score": round(statistics.mean(vals), 1),
                    "std": std_val,
                }

        case.replies = replies
        case.reply_scores = reply_scores
        case.avg_overall_score = avg_score
        case.overall_score_std = score_std
        case.avg_dimensions = avg_dims
        case.cross_similarity = round(_reply_similarity(replies), 2)
        case.n_generations = len(all_runs)
        results.append(case)

    return results


def _read_cache(case_name: str) -> list[dict]:
    cache_path = CACHE_DIR / f"{case_name}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data.get("runs", [])
    return []


def _write_cache(case_name: str, runs: list[dict]):
    cache_path = CACHE_DIR / f"{case_name}.json"
    cache_path.write_text(json.dumps({
        "n_runs": len(runs), "runs": runs
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def print_report(results: list[StabilityResult]):
    print("\n" + "=" * 70)
    print("🤖 Bot 回复稳定性（生产 prompt × 3 次生成）")
    print("=" * 70)

    valid = [r for r in results if not r.error]
    for r in valid:
        stable = "🟢" if r.overall_score_std < 3 else "🟡" if r.overall_score_std < 6 else "🔴"
        print(f"\n  [{r.case_name}] {stable} {r.avg_overall_score:.0f}±{r.overall_score_std:.0f}/35  sim={r.cross_similarity:.0%}")
        for i, rs in enumerate(r.reply_scores):
            print(f"    [{i+1}] {rs['overall_score']}/35 reply={rs['reply'][:100]}")
        for name, dd in r.avg_dimensions.items():
            std = f" ±{dd['std']:.1f}" if dd.get("std", 0) > 0.3 else ""
            bar = "▮" * round(dd["score"]) + "▯" * (5 - round(dd["score"]))
            print(f"    [{bar}] {name}: {dd['score']}/5{std}")

    if valid:
        avg_std = statistics.mean([r.overall_score_std for r in valid])
        print(f"\n📊 汇总: {len(valid)} cases, 平均波动 {avg_std:.1f}")


# =============================================================================
# pytest
# =============================================================================

@pytest.fixture(scope="module")
def stability_results():
    return run_benchmark(use_api=False)


def test_stability_score_variance(stability_results):
    valid = [r for r in stability_results if not r.error and r.overall_score_std > 0]
    if not valid:
        pytest.skip("无足够数据")
    avg_std = statistics.mean([r.overall_score_std for r in valid])
    assert avg_std < 8, f"回复评分波动过大: avg_std={avg_std:.1f}"


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bot 回复稳定性 Benchmark（生产 prompt）")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--n-generations", type=int, default=3)
    args = parser.parse_args()

    cases = _load_cases()
    print(f"🤖 Bot Reply Stability — {len(cases)} cases（生产 prompt）")
    print(f"   模式: {'真实 LLM' if args.run_api else '缓存回归'} × {args.n_generations} generations")
    for c in cases:
        has_prompt = "✓" if c.full_user_prompt else "✗ 无prompt"
        print(f"     [{c.case_name}] {has_prompt} ← {c.source}")
    print()

    results = run_benchmark(use_api=args.run_api, api_key=args.api_key, n_generations=args.n_generations)
    print_report(results)


if __name__ == "__main__":
    main()
