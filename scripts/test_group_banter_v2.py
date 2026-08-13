#!/usr/bin/env python3
"""群聊互动 SKILL v2 效果验证：有限个真实群聊 case。

调用真实 LLM，观察"幽默技巧"段注入后的回复风格。
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.models.base import ChatMessage, SenderType
from src.utils.qwen_client import QwenClient
from src.reply.generator import ReplyGenerator


# 测试场景：每组的最后一条消息是"未读"，前面的都是历史
# generate() 会自动构建 [会话]/[对方信息]/[历史消息]/[未读消息] 结构
SCENES = [
    {
        "name": "装穷仇富",
        "is_group": True,
        "chat_name": "朋友们",
        "history": [
            ("大刘", "刚提了辆宝马 哈哈哈"),
        ],
        "unread": [
            ("大刘", "是不是"),
        ],
    },
    {
        "name": "被动玩笑 — 对方装到了",
        "is_group": True,
        "chat_name": "好友群",
        "history": [
            ("示例用户丁", "我今天又赚了一笔"),
        ],
        "unread": [
            ("示例用户丁", "你们说是不是？"),
        ],
    },
    {
        "name": "自嘲 — 被 cue 短板",
        "is_group": True,
        "chat_name": "AI 玩家",
        "history": [
            ("小张", "这次 3d 打印模型太复杂了"),
            ("小张", "没点耐心搞不定"),
        ],
        "unread": [
            ("小张", "@你 你搞过这个吗"),
        ],
    },
    {
        "name": "反诘接梗 — 对方吐槽",
        "is_group": True,
        "chat_name": "好友群",
        "history": [
            ("示例别名庚", "最近花钱花了好多"),
        ],
        "unread": [
            ("示例别名庚", "心里真不舒服啊"),
        ],
    },
    {
        "name": "跟风复读 — 队列型",
        "is_group": True,
        "chat_name": "股市交流",
        "history": [
            ("示例别名庚", "警惕资本主义打牌"),
            ("老李", "警惕资本主义打牌"),
        ],
        "unread": [
            ("大刘", "警惕资本主义打牌"),
        ],
    },
]


def make_messages(history, unread, chat_name):
    """构建 all_messages 列表，最后一个是 unreplied 如果 unread 不为空。"""
    all_msgs = []
    for sender, text in history:
        all_msgs.append(ChatMessage(
            text=text,
            sender=sender,
            sender_type=SenderType.OTHER,
            chat_name=chat_name,
            timestamp=time.time() - 60,  # 1 分钟前
        ))
    unreplied_msgs = []
    for sender, text in unread:
        msg = ChatMessage(
            text=text,
            sender=sender,
            sender_type=SenderType.OTHER,
            chat_name=chat_name,
        )
        unreplied_msgs.append(msg)
        all_msgs.append(msg)
    return unreplied_msgs, all_msgs


def main():
    llm = QwenClient()
    gen = ReplyGenerator(llm_client=llm)
    gen.enable_self_refine = True
    gen.enable_react_tools = True

    model = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
    print("=" * 60)
    print(f"🤖 模型: {model}")
    print("=" * 60)

    for scene in SCENES:
        name = scene["name"]
        unreplied, all_messages = make_messages(
            scene["history"], scene["unread"], scene["chat_name"]
        )

        print(f"\n{'─' * 60}")
        print(f"📋 场景: {name} ({scene['chat_name']})")
        print(f"{'─' * 60}")
        for s, t in scene["history"] + scene["unread"]:
            unread_tag = " 📩" if (s, t) in scene["unread"] else ""
            print(f"  {s}: {t}{unread_tag}")

        t0 = time.time()
        try:
            replies = gen.generate(
                unreplied=unreplied,
                all_messages=all_messages,
                is_group=scene["is_group"],
            )
            t1 = time.time()
            print(f"\n  ⏱  {t1-t0:.1f}s")
            if replies:
                for i, r in enumerate(replies, 1):
                    print(f"  📝 [{i}] {r}")
            else:
                print("  ⚠️  无回复")
            if gen.last_self_refine_applied:
                print(f"  🔍  Self-Refine: decision={gen.last_feedback_decision} iter={gen.last_iterate_count}")
            # 打印注入的 skill 内容（预览）
            injected = getattr(gen, "last_skill_injected_content", "")
            if injected:
                # 只打印 skill 名，不打印内容
                skill_names = [l.strip() for l in injected.split("\n") if l.strip().startswith("【")]
                print(f"  📋 注入 skill: {skill_names}")
        except Exception as e:
            t1 = time.time()
            print(f"\n  ❌ 异常 ({t1-t0:.1f}s): {e}")

    print(f"\n{'=' * 60}")
    print("✅ 测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
