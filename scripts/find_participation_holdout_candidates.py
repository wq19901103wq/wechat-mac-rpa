#!/usr/bin/env python3
"""List pre-cutoff real group replies that may exercise natural participation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CUTOFF = int(datetime(2026, 4, 1).timestamp())
SHARED_MARKERS = (
    "定吧", "这么定", "那就", "可以", "安排", "一起", "吃饭", "走吧",
    "出发", "确认", "报名", "订了", "定了", "约", "行程", "时间",
)
DIRECT_MARKERS = ("@王芊", "@芊", "芊芊", "芊哥", "王芊")


def main() -> None:
    rows = []
    paths = [
        *sorted((ROOT / "data/exports/main").glob("*.json")),
        *sorted((ROOT / "data/exports/b").glob("*.json")),
    ]
    for path in paths:
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet.get("session", {}).get("type") != "群聊":
            continue
        messages = packet.get("messages", [])
        for index, message in enumerate(messages):
            reply_time = int(message.get("createTime") or 0)
            reply = str(message.get("content") or "").strip()
            if (
                not message.get("isSend")
                or reply_time >= CUTOFF
                or message.get("type") != "文本消息"
                or not reply
                or len(reply) > 40
            ):
                continue
            preceding = []
            cursor = index - 1
            while cursor >= 0 and len(preceding) < 6:
                earlier = messages[cursor]
                if reply_time - int(earlier.get("createTime") or 0) > 900:
                    break
                if earlier.get("isSend"):
                    break
                if earlier.get("type") == "文本消息":
                    preceding.append(earlier)
                cursor -= 1
            preceding.reverse()
            if not preceding:
                continue
            texts = [str(item.get("content") or "").strip() for item in preceding]
            joined = " ".join(texts)
            direct = any(marker in joined for marker in DIRECT_MARKERS) or "?" in joined or "？" in joined
            shared = any(marker in joined for marker in SHARED_MARKERS)
            previous_self = messages[cursor] if cursor >= 0 else None
            already_participating = bool(
                previous_self
                and previous_self.get("isSend")
                and int(preceding[0].get("createTime") or 0)
                - int(previous_self.get("createTime") or 0)
                <= 900
            )
            if direct or not (shared or already_participating):
                continue
            rows.append({
                "time": message.get("formattedTime"),
                "source": str(path.relative_to(ROOT)),
                "local_id": message.get("localId"),
                "already_participating": already_participating,
                "shared_marker": shared,
                "previous_self": (
                    str(previous_self.get("content") or "").strip()
                    if already_participating else None
                ),
                "context": [
                    {
                        "sender": item.get("senderDisplayName"),
                        "text": str(item.get("content") or "").strip(),
                    }
                    for item in preceding
                ],
                "human_reply": reply,
            })
    for row in sorted(rows, key=lambda item: item["time"], reverse=True):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
