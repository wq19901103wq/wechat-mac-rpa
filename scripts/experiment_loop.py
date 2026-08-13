#!/usr/bin/env python3
"""
实验迭代闭环脚本 — 自动分析 → 生成新实验 → A/B → 循环

用法:
  python3 scripts/experiment_loop.py --exp no_time --all-labeled  # 单次实验+分析
  python3 scripts/experiment_loop.py --auto-iterate --rounds 3      # 自动迭代3轮
  python3 scripts/experiment_loop.py --compare 6 7                  # 对比两个实验
"""

import json, sqlite3, sys, subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "cases.db"


def analyze_experiment(exp_id: int) -> dict:
    """分析一个实验的结果，返回结构化分析。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    exp = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
    if not exp:
        conn.close()
        return {"error": f"Experiment {exp_id} not found"}
    exp = dict(exp)

    # 逐 tick 对比
    results = conn.execute("""
        SELECT c.tick_id,
               MAX(CASE WHEN c.config_name='control' THEN c.judge_score END) as c_score,
               MAX(CASE WHEN c.config_name='control' THEN c.judge_is_badcase END) as c_bc,
               MAX(CASE WHEN c.config_name='control' THEN c.bot_reply END) as c_reply,
               MAX(CASE WHEN c.config_name!='control' THEN c.judge_score END) as e_score,
               MAX(CASE WHEN c.config_name!='control' THEN c.judge_is_badcase END) as e_bc,
               MAX(CASE WHEN c.config_name!='control' THEN c.bot_reply END) as e_reply
        FROM experiment_results c
        WHERE c.experiment_id=?
        GROUP BY c.tick_id ORDER BY c.tick_id
    """, (exp_id,)).fetchall()

    improved = []   # 变好（高分组更低 badcase 或更高分）
    degraded = []   # 变差
    unchanged = []

    for r in results:
        d = dict(r)
        c_s, e_s = d["c_score"] or 0, d["e_score"] or 0
        c_bc, e_bc = d["c_bc"] or 0, d["e_bc"] or 0
        diff = e_s - c_s

        case_data = {
            "tick_id": d["tick_id"],
            "control_score": c_s, "exp_score": e_s,
            "control_bc": bool(c_bc), "exp_bc": bool(e_bc),
            "diff": diff, "control_reply": (d["c_reply"] or "")[:100],
            "exp_reply": (d["e_reply"] or "")[:100],
        }

        if e_bc < c_bc or diff > 3:
            improved.append(case_data)
        elif e_bc > c_bc or diff < -3:
            degraded.append(case_data)
        else:
            unchanged.append(case_data)

    dims = json.loads(exp["dimension_diffs_json"] or "{}")
    best_dim = max(dims.items(), key=lambda x: x[1]) if dims else ("?", 0)
    worst_dim = min(dims.items(), key=lambda x: x[1]) if dims else ("?", 0)

    conn.close()

    return {
        "exp_id": exp_id,
        "exp_name": exp["name"],
        "exp_description": exp.get("description", ""),
        "n_samples": exp["n_samples"],
        "control_badcase_rate": exp["control_badcase_rate"] or 0,
        "exp_badcase_rate": exp["exp_badcase_rate"] or 0,
        "control_avg_score": exp["control_avg_score"] or 0,
        "exp_avg_score": exp["exp_avg_score"] or 0,
        "summary": exp.get("summary", ""),
        "best_dimension": best_dim,
        "worst_dimension": worst_dim,
        "dimension_diffs": dims,
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
    }


def compare_experiments(id_a: int, id_b: int):
    """对比两个实验，找出差异。"""
    a = analyze_experiment(id_a)
    b = analyze_experiment(id_b)
    if "error" in a or "error" in b:
        print(f"Error: {a.get('error')} / {b.get('error')}")
        return

    print(f"📊 实验对比: [{id_a}] {a['exp_name']} vs [{id_b}] {b['exp_name']}")
    print(f"   样本: {a['n_samples']} vs {b['n_samples']}")
    print()
    print(f"   {'':20} | {'实验'+str(id_a):>12} | {'实验'+str(id_b):>12} | 差异")
    print(f"   {'Badcase率':20} | {a['exp_badcase_rate']:>11.0%} | {b['exp_badcase_rate']:>11.0%} | {b['exp_badcase_rate']-a['exp_badcase_rate']:>+.0%}")
    print(f"   {'均分':20} | {a['exp_avg_score']:>11.1f} | {b['exp_avg_score']:>11.1f} | {b['exp_avg_score']-a['exp_avg_score']:>+.1f}")
    print(f"   {'变好 case':20} | {len(a['improved']):>11} | {len(b['improved']):>11}")
    print(f"   {'变差 case':20} | {len(a['degraded']):>11} | {len(b['degraded']):>11}")
    print()

    # 维度对比
    print("   维度差异:")
    all_dims = set(list(a['dimension_diffs'].keys()) + list(b['dimension_diffs'].keys()))
    for dim in sorted(all_dims):
        da = a['dimension_diffs'].get(dim, 0)
        db_dim = b['dimension_diffs'].get(dim, 0)
        winner = "→" if abs(da) < 0.1 and abs(db_dim) < 0.1 else ("A" if da > db_dim else "B")
        print(f"   {dim:15}: A={da:+.1f} B={db_dim:+.1f} [{winner}]")


def print_iteration_insights(analysis: dict):
    """从实验结果提取迭代方向。"""
    print(f"\n🔍 迭代洞察 — {analysis['exp_name']}")
    print(f"   当前: badcase率={analysis['exp_badcase_rate']:.0%} 均分={analysis['exp_avg_score']:.1f}")
    print()

    if analysis['improved']:
        print(f"   ✅ 变好的 case ({len(analysis['improved'])}):")
        for c in analysis['improved'][:3]:
            print(f"      #{c['tick_id']:>5} +{c['diff']:+.0f}分 | {c['exp_reply'][:60]}")
        print()

    if analysis['degraded']:
        print(f"   ❌ 变差的 case ({len(analysis['degraded'])}):")
        for c in analysis['degraded'][:3]:
            print(f"      #{c['tick_id']:>5} {c['diff']:+.0f}分 | {c['exp_reply'][:60]}")
        print()

    print(f"   📐 维度分析:")
    dims = sorted(analysis['dimension_diffs'].items(), key=lambda x: -x[1])
    for dim, diff in dims:
        bar = "█" * max(0, int(diff * 5)) if diff > 0 else "░" * max(0, int(abs(diff) * 5))
        tag = "↑改善" if diff > 0.2 else ("↓退化" if diff < -0.2 else "—持平")
        print(f"      {dim:12} {diff:+.1f} {bar} {tag}")

    # 建议下一步
    print(f"\n   💡 下一步建议:")
    worst_dim = analysis['worst_dimension']
    if worst_dim[1] < -0.3:
        print(f"      1. 修复退化维度: {worst_dim[0]} ({worst_dim[1]:+.1f})")
    if analysis['degraded']:
        print(f"      2. 检查变差的 {len(analysis['degraded'])} 个 case，确认是偶发还是系统性问题")
    if analysis['exp_badcase_rate'] < analysis['control_badcase_rate']:
        print(f"      3. ✅ 实验组 badcase 率低于对照组，方向正确，继续迭代")
    else:
        print(f"      3. ⚠️ 实验组未改善，考虑调整策略或换方向")


def run_and_analyze(exp_name: str, all_labeled: bool = False, n_samples: int = 5):
    """运行实验 + 分析 + 打印洞察。"""
    cmd = ["python3", str(PROJECT_ROOT / "scripts" / "run_experiment.py"), "--exp", exp_name]
    if all_labeled:
        cmd.append("--all-labeled")
    else:
        cmd.extend(["--n-samples", str(n_samples)])

    print(f"🧪 运行实验: {exp_name}")
    subprocess.run(cmd)

    # 找到最新实验 ID
    conn = sqlite3.connect(str(DB_PATH))
    latest_id = conn.execute("SELECT MAX(id) FROM experiments").fetchone()[0]
    conn.close()

    analysis = analyze_experiment(latest_id)
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return None

    print_iteration_insights(analysis)
    print(f"\n📊 详细报告: http://localhost:8766/experiments/{latest_id}")
    return analysis


def auto_iterate(rounds: int = 3, n_samples: int = 0):
    """自动迭代：从 all_off 逐步开启功能，增量对比。n_samples=0 表示用全部已标注。"""
    exp_sequence = ["enable_time", "enable_restraint", "enable_search", "enable_all_p0"]
    results_history = []

    for i, exp_name in enumerate(exp_sequence[:rounds]):
        print(f"\n{'='*60}")
        print(f"🔄 第 {i+1}/{rounds} 轮: {exp_name}")
        print(f"{'='*60}")

        cmd = ["python3", str(PROJECT_ROOT / "scripts" / "run_experiment.py"), "--exp", exp_name]
        cmd.extend(["--n-samples", str(n_samples if n_samples > 0 else 30)])  # 从全部 tick 随机采样
        subprocess.run(cmd)

        conn = sqlite3.connect(str(DB_PATH))
        latest_id = conn.execute("SELECT MAX(id) FROM experiments").fetchone()[0]
        conn.close()

        analysis = analyze_experiment(latest_id)
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            continue
        print_iteration_insights(analysis)
        results_history.append(analysis)

        # 如果实验组显著优于 all_off 基线 → 功能有效
        if analysis['exp_badcase_rate'] < analysis['control_badcase_rate']:
            print(f"\n   ✅ {exp_name} 有效！badcase {analysis['exp_badcase_rate']:.0%} < 基线 {analysis['control_badcase_rate']:.0%}")
        elif analysis['exp_avg_score'] > analysis['control_avg_score'] + 1:
            print(f"\n   ✅ {exp_name} 有效！均分 +{analysis['exp_avg_score']-analysis['control_avg_score']:.1f}")
        else:
            print(f"\n   ⚠️ {exp_name} 未显著改善")

    print(f"\n{'='*60}")
    print(f"📊 {len(results_history)} 轮迭代完成")
    for i, r in enumerate(results_history):
        improvement = "✅" if r['exp_badcase_rate'] < r['control_badcase_rate'] else "—"
        print(f"  {improvement} {r['exp_name']}: exp={r['exp_badcase_rate']:.0%} baseline={r['control_badcase_rate']:.0%} 均分={r['exp_avg_score']:.1f}")

    return results_history


def _suggest_next_experiment(analysis: dict) -> str:
    """根据分析结果，生成下一轮实验名称。"""
    worst_dim, _ = analysis['worst_dimension']
    exp_name = analysis['exp_name']

    # 映射：最差维度 → 应启用的功能
    dim_to_enable = {
        "时间推理": "time_awareness",     # 需要时间感知
        "幻觉控制": "search_in_page",     # 需要搜索验证
        "回复必要性": "reply_restraint",   # 需要回复克制
        "信息准确性": "search_in_page",    # 需要搜索验证
        "上下文理解": "time_awareness",    # 需要时间戳
    }

    target = dim_to_enable.get(worst_dim, "")
    if not target:
        return exp_name

    # 如果当前实验关闭了某个功能，下一轮启用它
    # 如果当前是 all_off，下一轮只启用最差维度对应的功能
    if exp_name == "all_off":
        return target.replace("_awareness", "").replace("_in_page", "").replace("_restraint", "")

    # 组合改进：逐步叠加
    if exp_name == "no_time" and target == "reply_restraint":
        return "all_off"  # 反而全关更差，说明需要全部启用
    if exp_name == "no_restraint" and target == "time_awareness":
        return "no_time"
    if exp_name == "no_dedup" and target == "reply_restraint":
        return "no_restraint"

    return exp_name


def main():
    import argparse
    parser = argparse.ArgumentParser(description="实验迭代闭环")
    parser.add_argument("--exp", help="实验名称")
    parser.add_argument("--all-labeled", action="store_true")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--analyze", type=int, help="分析指定实验 ID")
    parser.add_argument("--compare", nargs=2, type=int, help="对比两个实验 ID")
    parser.add_argument("--auto-iterate", action="store_true", help="自动迭代")
    parser.add_argument("--rounds", type=int, default=3, help="迭代轮数")
    args = parser.parse_args()

    if args.auto_iterate:
        auto_iterate(args.rounds, args.n_samples)
        return

    if args.compare:
        compare_experiments(args.compare[0], args.compare[1])
        return

    if args.analyze:
        analysis = analyze_experiment(args.analyze)
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        print_iteration_insights(analysis)
        return

    if args.exp:
        run_and_analyze(args.exp, args.all_labeled, args.n_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
