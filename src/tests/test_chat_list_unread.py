#!/usr/bin/env python3
"""ChatList 未读角标识别回归测试

用真实误判截图的 fixture 数据验证：
- 群聊拼贴头像内部图案不应被识别为未读角标
- 合并策略和过滤逻辑应能纠正 API/本地 OCR 的误判
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import ChatListItem, Rect
from src.perception.smart_pipeline import SmartPerceptionPipeline

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "unread_false_positive"


def _apply_unread_filter(api_chat_list: list[dict]) -> None:
    """与 _run_with_api 中一致的未读角标后处理过滤逻辑。"""
    for item in api_chat_list:
        raw = item.get("unread_count", "")
        if not raw:
            continue
        if ":" in raw:
            item["unread_count"] = ""
            continue
        if any("\u4e00" <= c <= "\u9fff" for c in raw):
            item["unread_count"] = ""
            continue
        if not raw.isdigit():
            item["unread_count"] = ""
            continue
        if int(raw) > 99:
            item["unread_count"] = ""
            continue


class TestChatListUnreadFalsePositive:
    """回归测试：群聊拼贴头像误判为未读角标"""

    @pytest.fixture
    def pipeline(self):
        """创建 SmartPipeline 实例，mock 所有 I/O 依赖（不调用真实 API/OCR）。"""
        with patch("src.perception.smart_pipeline.WindowCapture"), \
             patch("src.perception.smart_pipeline.VisionOCREngine"), \
             patch("src.perception.smart_pipeline.LayoutParser"):
            p = SmartPerceptionPipeline(profile=PROFILE_WECHAT_MAC_1760X1280)
            yield p

    @pytest.fixture
    def fixture_data(self):
        with open(FIXTURE_DIR / "tick_0129.json", encoding="utf-8") as f:
            return json.load(f)

    @pytest.mark.xfail(
        reason="已知问题：群聊拼贴头像被误判为未读角标，待合并策略/过滤规则修复后通过",
        strict=False,
    )
    def test_group_avatar_not_unread_badge(self, pipeline, fixture_data):
        """
        Fixture 截图中"王芊 @ai开发小分队"是群聊拼贴头像，头像外右上角
        没有红色圆形未读角标。期望 unread_count = ''。

        当前状态：API 误判为 "10"，本地 OCR 误判为 "1"，合并后得到 "10"，
        过滤逻辑未拦截。此测试先记录这一失败事实，待合并策略/过滤规则
        修复后通过。
        """
        # 1. 从 fixture 构建本地 Layout 识别的 chat_list
        local_items: list[ChatListItem] = []
        for name, unread in zip(
            fixture_data["layout_chat_list_nicknames"],
            fixture_data["layout_chat_list_unread"],
        ):
            local_items.append(
                ChatListItem(
                    nickname=name,
                    last_message_preview="",
                    unread_count=unread,
                    timestamp="",
                    rect=Rect(x=0, y=0, width=100, height=50),
                )
            )

        # 2. API 识别的 chat_list（来自旧 prompt 的误判结果）
        api_chat_list: list[dict] = fixture_data["api_chat_list"]

        # 3. 应用过滤逻辑
        _apply_unread_filter(api_chat_list)

        # 4. 合并（当前策略：API unread_count 覆盖本地）
        merged = pipeline._merge_chat_list(local_items, api_chat_list)

        # 5. 断言
        target = next(
            (item for item in merged if "ai开发小分队" in item.nickname),
            None,
        )
        assert target is not None, "未在 merged chat_list 中找到目标聊天"
        assert target.unread_count == "", (
            f"群聊拼贴头像被误判为未读角标: "
            f"nickname={target.nickname!r}, unread_count={target.unread_count!r}, "
            f"期望=''。API 识别={api_chat_list[1].get('unread_count')!r}, "
            f"本地识别={local_items[1].unread_count!r}"
        )
