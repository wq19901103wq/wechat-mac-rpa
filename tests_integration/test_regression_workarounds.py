#!/usr/bin/env python3
"""
暴露测试：验证当前未修复的问题，测试应失败，修复后才绿。

这些测试不是防护 workaround 回归，而是直接暴露现有 bug：
1. 多元素 cluster 的第一个元素被无条件当昵称 → 消息内容被吞（漏回）
2. clean_chat_name 不清洗数字前缀 → 但这不是 bug，是正确行为（OCR 问题应在 OCR 层解决）
3. _clean_reply 不过滤思考内容 → 也不是 bug，问题应在 prompt/LLM 层解决

所以此文件主要暴露问题 1。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layout.layout_parser import LayoutParser, UILayout
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.message.extractor import MessageExtractor
from src.models.base import OCRTextElement, Point, Rect
from src.reply.generator import ReplyGenerator


# ═══════════════════════════════════════════════════════════
# 1. 漏回：多元素 cluster 的第一个元素不应被无条件当昵称
# ═══════════════════════════════════════════════════════════


class TestMessageExtractionLeakage:
    """
    当前 _extract_other_messages 的逻辑：
        if len(cluster) > 1 and x in nickname_range:
            nickname = cluster[0].text      # ← 无条件把第一个元素当昵称
            msg_elems = cluster[1:]

    这会导致：如果 OCR 把两条短消息聚类在一起（y 间距 < 80），
    第一条消息被当作昵称，内容被吞掉。

    修复方向：用布局特征（垂直间距）区分昵称和消息，
    而不是无条件把 cluster[0] 当昵称。
    """

    def _make_layout(self, elements: list) -> UILayout:
        """构造 UILayout，message_candidates 为传入的元素列表。"""
        return UILayout(
            chat_name="测试群",
            chat_list_items=[],
            title_elements=[],
            input_elements=[],
            timestamp_elements=[],
            self_bubbles=[],
            message_candidates=elements,
        )

    def test_two_short_messages_clustered_both_extracted(self):
        """
        两条短消息 "怎么" 和 "在吗" 被 OCR 聚类在一起（y 间距 50 < 80），
        都应作为消息内容提取，不应把 "怎么" 当作昵称吞掉。

        当前行为："怎么" 被当昵称，只提取到 "在吗" → 漏回
        期望行为：两条都提取到 → 测试当前会失败
        """
        extractor = MessageExtractor(PROFILE_WECHAT_MAC_1760X1280)

        # x=750：在昵称范围(528~792)内，且 >700（避开头像噪声区）
        elem1 = OCRTextElement(
            text="怎么",
            bbox=Rect(730, 490, 40, 20),
            center=Point(750, 500),
            confidence=0.95,
        )
        elem2 = OCRTextElement(
            text="在吗",
            bbox=Rect(730, 540, 40, 20),
            center=Point(750, 550),
            confidence=0.95,
        )
        layout = self._make_layout([elem1, elem2])
        messages = extractor.extract(layout)

        texts = [m.text for m in messages]
        assert "怎么" in texts, f"'怎么' 被漏掉了，实际提取到 {texts}"
        assert "在吗" in texts, f"'在吗' 被漏掉了，实际提取到 {texts}"

    def test_avatar_digit_mistaken_as_nickname(self):
        """
        tick81 真实案例：头像上的未读数字 "10" 被当成昵称。

        左侧聊天列表中：
            "10"   x=182 y=403  ← 头像未读数字
            "什么需要的时候可以叫你" x=228 y=430 ← 消息预览

        y 差距 27px < threshold，被聚类到一起。
        cluster[0]="10" 在昵称 x 范围内 → 被当成昵称。
        结果：消息 sender="10"，而非真实发送者。

        期望：纯数字短文本（<=2位）不应被认定为昵称。
        """
        extractor = MessageExtractor(PROFILE_WECHAT_MAC_1760X1280)

        # 真实案例: "10" cx=701 在右侧消息区，_is_avatar_noise 不过滤（>700）
        # y 差距 15px 会被聚类，x=701 在昵称范围内 → 被当昵称
        elem_digit = OCRTextElement(
            text="10",
            bbox=Rect(690, 925, 22, 20),
            center=Point(701, 931),
            confidence=0.95,
        )
        elem_msg = OCRTextElement(
            text="什么需要的时候可以叫你",
            bbox=Rect(830, 940, 180, 20),
            center=Point(919, 946),
            confidence=0.95,
        )
        layout = self._make_layout([elem_digit, elem_msg])
        messages = extractor.extract(layout)

        # 修复后: "10" 和消息不再被误聚类，各自独立提取
        texts = {m.text for m in messages}
        assert "什么需要的时候可以叫你" in texts, (
            f"消息未被提取，实际: {texts}"
        )
        # 关键：任何消息的 sender 都不能是 "10"
        for msg in messages:
            assert msg.sender != "10", (
                f"头像数字 '10' 被错误识别为昵称，sender={msg.sender!r}, text={msg.text!r}"
            )

    def test_three_messages_clustered_all_extracted(self):
        """
        三条消息聚类在一起，第一条不应被当昵称。
        """
        extractor = MessageExtractor(PROFILE_WECHAT_MAC_1760X1280)

        elems = [
            OCRTextElement(text="嗯", bbox=Rect(730, 490, 30, 20), center=Point(750, 500), confidence=0.95),
            OCRTextElement(text="好的", bbox=Rect(730, 540, 40, 20), center=Point(750, 550), confidence=0.95),
            OCRTextElement(text="收到", bbox=Rect(730, 590, 40, 20), center=Point(750, 600), confidence=0.95),
        ]
        layout = self._make_layout(elems)
        messages = extractor.extract(layout)

        texts = [m.text for m in messages]
        assert "嗯" in texts, f"'嗯' 被漏掉了，实际提取到 {texts}"
        assert "好的" in texts, f"'好的' 被漏掉了，实际提取到 {texts}"
        assert "收到" in texts, f"'收到' 被漏掉了，实际提取到 {texts}"


# ═══════════════════════════════════════════════════════════
# 2. clean_chat_name 行为确认（不是 bug，是设计选择）
# ═══════════════════════════════════════════════════════════


class TestCleanChatNameBehavior:
    """
    clean_chat_name 不再清洗数字前缀/乱码前缀，这是有意的设计：
    OCR 合并问题应在 OCR 层解决，不应事后打补丁。

    这些测试是"行为文档"，验证当前设计，不是 bug 暴露。
    """

    def test_digit_prefix_preserved(self):
        """'10 10 示例用户甲' 保持原样。"""
        assert LayoutParser.clean_chat_name("10 10 示例用户甲") == "10 10 示例用户甲"

    def test_symbol_garbage_prefix_preserved(self):
        """'！岔站 示例用户甲 @ai' 保持原样。"""
        assert LayoutParser.clean_chat_name("！岔站 示例用户甲 @ai") == "！岔站 示例用户甲 @ai"

    def test_timestamp_suffix_removed(self):
        """时间戳后缀仍去除（时间戳是 UI 元素，不是昵称的一部分）。"""
        assert LayoutParser.clean_chat_name("示例聊天名甲昨天 22:26") == "示例聊天名甲"


# ═══════════════════════════════════════════════════════════
# 3. _clean_reply 行为确认（不是 bug，是设计选择）
# ═══════════════════════════════════════════════════════════


class TestCleanReplyBehavior:
    """
    _clean_reply 不再过滤思考内容，直接返回原文。
    思考混入问题应在 prompt/LLM 层解决（如 max_tokens=1024）。

    这些测试是"行为文档"，验证当前设计，不是 bug 暴露。
    """

    def test_thinking_content_passed_through(self):
        raw = "让我想想，用户可能是在问天气，我应该直接回答。明天晴。"
        assert ReplyGenerator()._clean_reply(raw) == raw

    def test_empty_returns_empty(self):
        assert ReplyGenerator()._clean_reply("") == ""
        assert ReplyGenerator()._clean_reply("   ") == ""
