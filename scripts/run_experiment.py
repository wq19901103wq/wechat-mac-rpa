#!/usr/bin/env python3
"""
A/B 实验 — 固定 Judge，变 Bot 参数，对比得分

用法:
    python3 scripts/run_experiment.py --exp time_awareness_off --all-labeled
    python3 scripts/run_experiment.py --exp reply_restraint_off --n-samples 10
"""

import json, os, sys, sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List

from src.models.base import ChatMessage

sys.path.insert(0, str(Path(__file__).parent.parent))

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cases.db"
RESULTS_DIR = PROJECT_ROOT / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Bot 参数配置
# =============================================================================

@dataclass
class BotConfig:
    """Bot 回复生成参数。"""
    name: str
    description: str = ""
    # Prompt 参数
    enable_time_awareness: bool = True
    enable_reply_restraint: bool = True
    enable_unread_dedup: bool = True
    enable_timestamps: bool = True
    # 工具参数
    enable_search_in_page: bool = True
    browse_truncate: int = 3000
    tool_result_truncate: int = 3000
    # 模型参数（可选切换）
    model: str = "deepseek-v4-flash"
    temperature: float = 0.7


# 基线 = 当前生产配置
CONTROL = BotConfig(name="control", description="当前生产配置（基线）")

# 实验组
BOT_EXPERIMENTS = {
    # ====== 消融实验：逐个关闭功能，衡量损失 ======
    "no_time": BotConfig(
        name="no_time",
        description="【消融】关闭时间感知：去掉每条消息的绝对时间标签（YYYY-MM-DD HH:MM），去掉会话头部的当前时间+星期+时段说明，去掉时间戳说明。预期：时间推理维度退化，把历史消息当现在发的。",
        enable_time_awareness=False, enable_timestamps=False,
    ),
    "no_restraint": BotConfig(
        name="no_restraint",
        description="【消融】关闭回复克制：删除 persona 中的'回复克制原则'（OK/表情/确认词不回复），删除回复去重提示（历史已回复的消息标记为可跳过）。预期：对表情包/OK消息过度回复增加。",
        enable_reply_restraint=False,
    ),
    "no_dedup": BotConfig(
        name="no_dedup",
        description="【消融】关闭未读去重：去掉未读消息后的'已回复可跳过'标记，去掉检查历史回复的逻辑。预期：重复回复已处理过的消息增加。",
        enable_unread_dedup=False,
    ),
    "no_search_page": BotConfig(
        name="no_search_page",
        description="【消融】关闭 search_in_page 工具：browse_url 后无法对页面内容进行关键词搜索，只能依赖截断的前3000字。预期：browse后信息不完整导致幻觉增加。",
        enable_search_in_page=False,
    ),
    "short_truncate": BotConfig(
        name="short_truncate",
        description="【消融】缩短截断：browse_url 返回1000字（正常12000），工具结果500字（正常12000）。模拟信息受限环境。预期：信息准确性退化。",
        browse_truncate=1000, tool_result_truncate=500,
    ),
    "all_off": BotConfig(
        name="all_off",
        description="【基线】关闭所有改进：无时间标签、无回复克制、无未读去重、无search_in_page、信息截断到最小值。回到项目初始状态。每次增量实验对比此基线。",
        enable_time_awareness=False, enable_timestamps=False,
        enable_reply_restraint=False, enable_unread_dedup=False,
        enable_search_in_page=False, browse_truncate=1000, tool_result_truncate=500,
    ),
    # ====== 增量实验：从基线逐步开启，量化每项收益 ======
    "enable_time": BotConfig(
        name="enable_time",
        description="【增量】仅开启时间感知（其余保持基线关闭状态）：给每条聊天消息注入绝对时间标签（YYYY-MM-DD HH:MM），在会话头注入'当前时间+星期+时段'，告诉Bot消息时间戳的含义。预期：减少时间误判（把历史消息当现在发的），幻觉控制改善。",
        enable_time_awareness=True, enable_timestamps=True,
        enable_reply_restraint=False, enable_unread_dedup=False, enable_search_in_page=False,
    ),
    "enable_restraint": BotConfig(
        name="enable_restraint",
        description="【增量】仅开启回复克制（其余保持基线关闭状态）：Persona增加'回复克制原则'（OK/表情包/确认词/已回复过的消息不回复），未读消息标记'历史中已有回复可跳过'。预期：减少过度回复，回复必要性维度改善。",
        enable_reply_restraint=True, enable_unread_dedup=True,
        enable_time_awareness=False, enable_timestamps=False, enable_search_in_page=False,
    ),
    "enable_search": BotConfig(
        name="enable_search",
        description="【增量】仅开启信息增强（其余保持基线关闭状态）：browse_url返回12000字（原3000），工具结果不截断（原500），开启search_in_page（browse后可在页面内关键词搜索，前后各200字上下文）。预期：信息准确性+幻觉控制双提升。",
        enable_search_in_page=True, browse_truncate=12000, tool_result_truncate=12000,
        enable_time_awareness=False, enable_timestamps=False,
        enable_reply_restraint=False, enable_unread_dedup=False,
    ),
    "enable_all_p0": BotConfig(
        name="enable_all_p0",
        description="【全开】同时开启所有P0改进：时间感知+回复克制+未读去重+search_in_page+信息不截断。验证功能组合的效果——是否存在互相抵消或1+1>2的协同效应。",
        enable_time_awareness=True, enable_timestamps=True,
        enable_reply_restraint=True, enable_unread_dedup=True,
        enable_search_in_page=True, browse_truncate=12000, tool_result_truncate=12000,
    ),
}


# =============================================================================
# Bot 回复生成（用相同 prompt 调 LLM 重新生成）
# =============================================================================

# 实验禁止调用的写操作工具（会改变现实世界）
_EXPERIMENT_WRITE_TOOL_BLACKLIST = {"tuya_control_device", "tuya_set_temperature"}


# =============================================================================
# ChatMessage 反序列化（从 tick_log JSON 还原）
# =============================================================================

def _deserialize_messages(json_str: str) -> List[ChatMessage]:
    """从 JSON 字符串反序列化 ChatMessage 列表。"""
    from src.models.base import ChatMessage, SenderType
    data = json.loads(json_str) if json_str else []
    messages = []
    for item in data:
        sender_type_str = item.get("sender_type", "other")
        try:
            sender_type = SenderType(sender_type_str)
        except ValueError:
            sender_type = SenderType.OTHER
        msg = ChatMessage(
            text=item.get("text", ""),
            sender=item.get("sender", ""),
            sender_type=sender_type,
            chat_name=item.get("chat_name", ""),
            is_at_me=item.get("is_at_me", False),
            timestamp=item.get("timestamp"),
            replied=item.get("replied", False),
            reply_text=item.get("reply_text", ""),
            reply_time=item.get("reply_time"),
            message_type=item.get("message_type", "text"),
            image_description=item.get("image_description", ""),
            image_text=item.get("image_text", ""),
            is_image_duplicate=item.get("is_image_duplicate", False),
            account=item.get("account", ""),
            local_id=item.get("local_id"),
            server_id=item.get("server_id"),
            create_time=item.get("create_time"),
            raw_type=item.get("raw_type"),
            sender_wxid=item.get("sender_wxid"),
        )
        messages.append(msg)
    return messages


def _generate_with_config(all_messages: List[ChatMessage], unreplied: List[ChatMessage],
                          is_group: bool, config: BotConfig) -> tuple[str, str, str]:
    """使用 ReplyGenerator 生成回复，返回 (reply_text, system_prompt, user_prompt)。"""
    from src.reply.generator import ReplyGenerator
    from src.tools import ToolRegistry, register_builtin_tools
    from src.utils.qwen_client import QwenClient

    # 创建独立的工具注册表（实验和生产隔离）
    registry = ToolRegistry()
    register_builtin_tools(registry)
    # 移除会改变现实世界的写操作工具
    for name in _EXPERIMENT_WRITE_TOOL_BLACKLIST:
        if registry.has(name):
            registry._tools.pop(name, None)

    llm_client = QwenClient(model=config.model)
    reply_generator = ReplyGenerator(
        llm_client=llm_client,
        tool_registry=registry,
        judge_worker=None,
        enable_time_awareness=config.enable_time_awareness,
        enable_reply_restraint=config.enable_reply_restraint,
        enable_unread_dedup=config.enable_unread_dedup,
        enable_timestamps=config.enable_timestamps,
    )

    replies = reply_generator.generate(
        unreplied=unreplied,
        all_messages=all_messages,
        is_group=is_group,
    )

    reply_text = " | ".join(replies) if replies else ""
    return reply_text, reply_generator.last_system_prompt, reply_generator.last_user_prompt


# =============================================================================
# Judge（固定不变）
# =============================================================================

def judge_reply(tick_data: dict, bot_reply: str) -> dict:
    """用统一的 Judge 评分。"""
    from src.badcase.judge_worker import JudgeWorker
    import json as _json

    worker = JudgeWorker()
    tc = _json.loads(tick_data.get("tool_calls_json", "[]") or "[]")
    sp = tick_data.get("system_prompt", "") or ""
    up = tick_data.get("user_prompt", "") or ""
    llm_msgs = [{"role": "system", "content": sp}, {"role": "user", "content": up}]

    tool_info = [{"tool": t.get("tool_name",""), "args": str(t.get("arguments",""))[:200], "result": str(t.get("result_preview",""))[:3000]} for t in tc]

    return worker._judge({
        "tick_id": tick_data.get("tick_id", 0),
        "chat_name": tick_data.get("chat_name", ""),
        "bot_reply_text": bot_reply,
        "tool_calls": tc,
        "tool_results_json": _json.dumps(tool_info, ensure_ascii=False),
        "full_user_prompt": up,
        "full_system_prompt": sp,
        "full_llm_messages": llm_msgs,
    })


# =============================================================================
# Runner
# =============================================================================

def run_experiment(exp_config: BotConfig, tick_ids: list):
    """跑实验：对每个 tick，基线 vs 实验组都生成回复，Judge 打分，对比。"""
    from src.badcase.case_db import get_db
    from src.models.base import ChatMessage

    control_results = []
    exp_results = []

    for tid in tick_ids:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT * FROM tick_log WHERE tick_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        conn.close()
        if not r:
            continue
        d = dict(r)

        # 反序列化消息（优先使用 session_input_messages_json）
        session_input_json = d.get("session_input_messages_json", "") or ""
        session_unreplied_json = d.get("session_output_unreplied_json", "") or ""
        if not session_input_json or not session_unreplied_json:
            print(f"  #{tid}: 跳过（无 session_input_messages_json 或 session_output_unreplied_json）")
            continue

        try:
            all_messages = _deserialize_messages(session_input_json)
            unreplied = _deserialize_messages(session_unreplied_json)
        except Exception as err:
            print(f"  #{tid}: 跳过（反序列化失败: {err}）")
            continue

        if not all_messages or not unreplied:
            print(f"  #{tid}: 跳过（空消息列表）")
            continue

        is_group = bool(d.get("is_group", 0))

        # 对照组：用当前生产配置（CONTROL）重新生成
        control_reply, control_sp, control_up = _generate_with_config(
            all_messages, unreplied, is_group, CONTROL
        )
        control_judge = judge_reply(d, control_reply)

        # 实验组：用实验配置重新生成
        exp_reply, exp_sp, exp_up = _generate_with_config(
            all_messages, unreplied, is_group, exp_config
        )
        exp_judge = judge_reply(d, exp_reply)

        control_results.append({"tick_id": tid, "reply": control_reply, "sp": control_sp, "up": control_up, "judge": control_judge})
        exp_results.append({"tick_id": tid, "reply": exp_reply, "sp": exp_sp, "up": exp_up, "judge": exp_judge})

        c_bc = "BAD" if control_judge.get("is_badcase") else "OK"
        e_bc = "BAD" if exp_judge.get("is_badcase") else "OK"
        c_s = control_judge.get("overall_score", 0)
        e_s = exp_judge.get("overall_score", 0)
        print(f"  #{tid}: baseline={c_bc}({c_s:.0f}) exp={e_bc}({e_s:.0f}) diff={e_s-c_s:+.0f}")

    # 过滤无效评分（Judge 空返回/解析失败）
    invalid_reasons = ("空返回", "JSON 解析失败")
    valid_pairs = [(c, e) for c, e in zip(control_results, exp_results)
                   if c["judge"].get("reason") not in invalid_reasons and e["judge"].get("reason") not in invalid_reasons]
    n_total = len(control_results)
    n_valid = len(valid_pairs)
    c_bad = sum(1 for c, e in valid_pairs if c["judge"].get("is_badcase"))
    e_bad = sum(1 for c, e in valid_pairs if e["judge"].get("is_badcase"))
    c_avg = sum(c["judge"].get("overall_score", 0) for c, e in valid_pairs) / n_valid if n_valid else 0
    e_avg = sum(e["judge"].get("overall_score", 0) for c, e in valid_pairs) / n_valid if n_valid else 0

    invalid_n = n_total - n_valid
    print(f"\n有效样本: {n_valid}/{n_total} ({invalid_n} 个无效已排除)")
    print(f"基线: badcase={c_bad}/{n_valid} ({c_bad/n_valid*100:.0f}%) 均分={c_avg:.1f}")
    print(f"实验: badcase={e_bad}/{n_valid} ({e_bad/n_valid*100:.0f}%) 均分={e_avg:.1f}")
    print(f"差异: badcase {c_bad-e_bad:+d} 均分 {e_avg-c_avg:+.1f}")

    # 维度对比（仅有效样本）
    dims = ["幻觉控制", "时间推理", "回复必要性", "信息准确性", "上下文理解"]
    print("维度差异（实验-基线）:")
    for dim in dims:
        c_dim = sum(c["judge"].get("dimensions", {}).get(dim, {}).get("score", 0) for c, e in valid_pairs) / n_valid if n_valid else 0
        e_dim = sum(e["judge"].get("dimensions", {}).get(dim, {}).get("score", 0) for c, e in valid_pairs) / n_valid if n_valid else 0
        diff = e_dim - c_dim
        bar = "█" * max(0, int(diff * 5)) if diff > 0 else "░" * max(0, int(abs(diff) * 5))
        print(f"  {dim}: {c_dim:.1f} → {e_dim:.1f} ({diff:+.1f}) {bar}")

    # 存入数据库
    db = get_db()
    conn = db._get_conn()
    conn.execute("""INSERT INTO experiments (name, description, n_samples,
        control_badcase_rate, exp_badcase_rate, control_avg_score, exp_avg_score,
        summary, dimension_diffs_json, is_improvement)
        VALUES (?,?,?,?,?,?,?,?,?,?)""", (
        exp_config.name, exp_config.description, n_valid,
        c_bad/n_valid if n_valid else 0, e_bad/n_valid if n_valid else 0, c_avg, e_avg,
        f"badcase {c_bad-e_bad:+d} 均分 {e_avg-c_avg:+.1f} (有效{n_valid}/{n_total})",
        json.dumps({dim: round(
            sum(e["judge"].get("dimensions", {}).get(dim, {}).get("score", 0) for c, e in valid_pairs) / n_valid -
            sum(c["judge"].get("dimensions", {}).get(dim, {}).get("score", 0) for c, e in valid_pairs) / n_valid, 1
        ) for dim in dims}, ensure_ascii=False),
        1 if e_avg > c_avg + 1 else 0,
    ))
    exp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for c, e in zip(control_results, exp_results):
        for r, cfg in [(c, "control"), (e, exp_config.name)]:
            conn.execute("""INSERT OR REPLACE INTO experiment_results
                (experiment_id, tick_id, config_name, bot_reply,
                 judge_is_badcase, judge_score, judge_dimensions_json, judge_reason,
                 system_prompt, user_prompt)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                exp_id, r["tick_id"], cfg, r["reply"][:500],
                1 if r["judge"].get("is_badcase") else 0,
                r["judge"].get("overall_score", 0),
                json.dumps(r["judge"].get("dimensions", {}), ensure_ascii=False),
                r["judge"].get("reason", ""),
                r.get("sp", ""), r.get("up", ""),
            ))
    conn.commit(); conn.close()
    print(f"实验 ID={exp_id} 已保存")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bot A/B 实验")
    parser.add_argument("--exp", required=True, help="实验名称: " + ", ".join(BOT_EXPERIMENTS.keys()))
    parser.add_argument("--tick-id", type=int, help="单个 tick ID")
    parser.add_argument("--all-labeled", action="store_true", help="所有人工标注的 tick")
    parser.add_argument("--n-samples", type=int, default=5, help="随机采样 N 个 tick")
    args = parser.parse_args()

    if args.exp not in BOT_EXPERIMENTS:
        print(f"未知实验: {args.exp}, 可用: {', '.join(BOT_EXPERIMENTS.keys())}")
        return

    exp_config = BOT_EXPERIMENTS[args.exp]

    if args.tick_id:
        tick_ids = [args.tick_id]
    elif args.all_labeled:
        conn = sqlite3.connect(str(DB_PATH))
        tick_ids = [r[0] for r in conn.execute("SELECT tick_id FROM tick_log WHERE human_is_badcase IS NOT NULL ORDER BY id").fetchall()]
        conn.close()
    else:
        conn = sqlite3.connect(str(DB_PATH))
        # 等距采样，跨实验可对比
        total = conn.execute("SELECT COUNT(*) FROM tick_log WHERE should_reply=1").fetchone()[0]
        step = max(1, total // args.n_samples)
        all_ids = [r[0] for r in conn.execute(
            "SELECT tick_id FROM tick_log WHERE should_reply=1 ORDER BY id LIMIT ?",
            (args.n_samples * step,),
        ).fetchall()[::step]]
        conn.close()
        tick_ids = all_ids

    print(f"实验: {exp_config.name} — {exp_config.description}")
    print(f"样本: {len(tick_ids)} 个 tick, 固定 Judge: deepseek-v4-pro\n")

    run_experiment(exp_config, tick_ids)


if __name__ == "__main__":
    main()
