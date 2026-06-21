#!/usr/bin/env python3
"""L3 MessageExtractor 单元测试"""

from src.layout.layout_parser import UILayout
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.message.extractor import MessageExtractor
from src.models.base import OCRTextElement, Point, Rect, SenderType


def make_element(text: str, cx: int, cy: int, confidence: float = 0.95) -> OCRTextElement:
    """快速构造 OCRTextElement，bbox 以 center 为中心生成 30x15 矩形

    面积 450 >= avatar_noise 阈值 300，避免在单元测试中被误过滤。
    """
    return OCRTextElement(
        text=text,
        bbox=Rect(x=cx - 15, y=cy - 7, width=30, height=15),
        center=Point(x=cx, y=cy),
        confidence=confidence,
    )


def make_layout(
    chat_name: str = "测试聊天",
    self_bubbles: list = None,
    message_candidates: list = None,
    timestamp_elements: list = None,
) -> UILayout:
    return UILayout(
        chat_name=chat_name,
        chat_list_items=[],
        title_elements=[],
        input_elements=[],
        timestamp_elements=timestamp_elements or [],
        self_bubbles=self_bubbles or [],
        message_candidates=message_candidates or [],
    )


class TestSelfMessageExtraction:
    """自己消息提取"""

    def test_single_self_bubble_single_text(self):
        """一个绿色气泡内有一条文本"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        bubble = Rect(x=1200, y=300, width=200, height=40)
        elem = make_element("你好", cx=1300, cy=320)
        layout = make_layout(self_bubbles=[bubble], message_candidates=[elem])

        messages = extractor.extract(layout)
        assert len(messages) == 1
        assert messages[0].text == "你好"
        assert messages[0].sender == "自己"
        assert messages[0].sender_type == SenderType.SELF
        assert messages[0].chat_name == "测试聊天"

    def test_self_bubble_merge_multiple_texts_sorted_by_y(self):
        """同一气泡内多条文本按 y 排序后合并"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        bubble = Rect(x=1200, y=300, width=200, height=80)
        elem_bottom = make_element("世界", cx=1300, cy=360)
        elem_top = make_element("你好", cx=1300, cy=320)
        layout = make_layout(
            self_bubbles=[bubble],
            message_candidates=[elem_bottom, elem_top]
        )

        messages = extractor.extract(layout)
        assert len(messages) == 1
        assert messages[0].text == "你好 世界"
        assert messages[0].sender_type == SenderType.SELF


class TestOtherMessageExtraction:
    """对方消息提取"""

    def test_cluster_two_close_elements_into_one_message(self):
        """两个 y 间距小于 threshold 的元素聚类为一条消息"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        # x=450 在消息区但不在昵称识别区内（昵称区约为 528~968）
        elem1 = make_element("今天", cx=450, cy=300)
        elem2 = make_element("天气不错", cx=450, cy=350)
        layout = make_layout(message_candidates=[elem1, elem2])

        messages = extractor.extract(layout)
        assert len(messages) == 1
        assert messages[0].text == "今天 天气不错"
        assert messages[0].sender == "对方"
        assert messages[0].sender_type == SenderType.OTHER

    def test_split_into_two_clusters_when_spacing_over_threshold(self):
        """y 间距超过 threshold 时分为两条消息"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        elem1 = make_element("第一条", cx=450, cy=300)
        elem2 = make_element("第二条", cx=450, cy=400)
        layout = make_layout(message_candidates=[elem1, elem2])

        messages = extractor.extract(layout)
        assert len(messages) == 2
        assert messages[0].text == "第一条"
        assert messages[1].text == "第二条"

    def test_nickname_detection_in_zone(self):
        """昵称在识别区域内时用作 sender"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        # nickname_x_min_ratio=0.30, nickname_x_max_ratio=0.55
        # window_width=1760 -> x 范围 528~968
        # 昵称面积需 >=1000 才能成为"有效昵称"，放宽 x_threshold 让消息聚类
        nickname = OCRTextElement(
            text="小明",
            bbox=Rect(x=660, y=285, width=80, height=30),
            center=Point(700, 300),
            confidence=0.95,
        )
        # 消息气泡在昵称区域右侧（真实布局）
        msg_elem = make_element("在吗", cx=1000, cy=340)
        layout = make_layout(message_candidates=[nickname, msg_elem])

        messages = extractor.extract(layout)
        assert len(messages) == 1
        assert messages[0].sender == "小明"
        assert messages[0].text == "在吗"
        assert messages[0].sender_type == SenderType.OTHER

    def test_nickname_out_of_zone_ignored(self):
        """昵称不在识别区域内时视为普通消息内容"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        # x=200 不在 528~968 范围内
        nickname = make_element("小明", cx=200, cy=300)
        msg_elem = make_element("在吗", cx=200, cy=340)
        layout = make_layout(message_candidates=[nickname, msg_elem])

        messages = extractor.extract(layout)
        assert len(messages) == 1
        assert messages[0].sender == "对方"
        assert messages[0].text == "小明 在吗"


class TestMixedLayout:
    """混合场景"""

    def test_mixed_self_and_other(self):
        """同时包含自己消息和对方消息"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        self_bubble = Rect(x=1200, y=300, width=200, height=40)
        self_elem = make_element("我发的", cx=1300, cy=320)

        other_elem = make_element("对方发的", cx=450, cy=400)
        layout = make_layout(
            self_bubbles=[self_bubble],
            message_candidates=[self_elem, other_elem]
        )

        messages = extractor.extract(layout)
        assert len(messages) == 2
        # 按 y 排序
        assert messages[0].sender_type == SenderType.SELF
        assert messages[0].text == "我发的"
        assert messages[1].sender_type == SenderType.OTHER
        assert messages[1].text == "对方发的"


class TestEdgeCases:
    """边界情况"""

    def test_empty_layout_returns_empty(self):
        """空布局返回空列表"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)
        layout = make_layout()
        assert extractor.extract(layout) == []

    def test_self_bubble_takes_priority_over_message_candidates(self):
        """落在 self_bubble 内的候选元素优先识别为 SELF"""
        profile = PROFILE_WECHAT_MAC_1760X1280
        extractor = MessageExtractor(profile)

        bubble = Rect(x=1200, y=300, width=200, height=40)
        elem = make_element("我发的", cx=1300, cy=320)
        layout = make_layout(
            self_bubbles=[bubble],
            message_candidates=[elem]
        )

        messages = extractor.extract(layout)
        assert len(messages) == 1
        assert messages[0].sender_type == SenderType.SELF
