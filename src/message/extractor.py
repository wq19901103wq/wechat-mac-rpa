#!/usr/bin/env python3
"""L3 Message Extractor - 从 UILayout 中提取结构化消息"""

import re
from typing import Any, Dict, List

from src.layout.layout_parser import TIMESTAMP_PATTERNS, UILayout
from src.layout.profile import LayoutProfile
from src.models.base import ChatMessage, OCRTextElement, Point, Rect, SenderType

# 微信系统通知/安全提示关键词模式
_SYSTEM_NOTICE_KEYWORDS = [
    "对方账号安全性未知",
    "涉及金钱交易务必电话确认",
    "保护个人财产和隐私安全",
    "系统提示",
    "微信官方安全提示",
    "安全提醒",
    "风险提示",
    "转账提醒",
    "谨防诈骗",
]


class MessageExtractor:
    def __init__(self, profile: LayoutProfile):
        self.profile = profile
        self.debug_info: Dict[str, Any] = {}

    def extract(self, layout: UILayout) -> List[ChatMessage]:
        messages: List[ChatMessage] = []

        # 1. 提取 SELF 消息
        self_messages = self._extract_self_messages(layout)
        messages.extend(self_messages)

        # 2. 提取 OTHER 消息
        other_messages = self._extract_other_messages(layout)
        messages.extend(other_messages)

        # 3. 按 y 坐标排序
        messages.sort(key=self._message_y_position)

        # 合并调试信息（不覆盖 _extract_other_messages 中设置的 clusters）
        self.debug_info.update({
            "self_messages": [{"text": m.text, "sender": m.sender} for m in self_messages],
            "other_messages": [{"text": m.text, "sender": m.sender, "sender_type": m.sender_type.value} for m in other_messages],
        })

        return messages

    def _extract_self_messages(self, layout: UILayout) -> List[ChatMessage]:
        """提取自己发送的消息（中心点落在 self_bubbles 内的文本）"""
        messages = []
        used = set()

        for bubble in layout.self_bubbles:
            texts_in_bubble = []
            for elem in layout.message_candidates:
                elem_id = id(elem)
                if elem_id in used:
                    continue
                if self._point_in_rect(elem.center, bubble):
                    texts_in_bubble.append(elem)
                    used.add(elem_id)

            if texts_in_bubble:
                texts_in_bubble.sort(key=lambda e: e.center.y)
                merged = " ".join(e.text for e in texts_in_bubble)
                messages.append(
                    ChatMessage(
                        text=merged,
                        sender="自己",
                        sender_type=SenderType.SELF,
                        chat_name=layout.chat_name,
                        source_elements=texts_in_bubble,
                    )
                )

        return messages

    def _extract_other_messages(self, layout: UILayout) -> List[ChatMessage]:
        """提取对方发送的消息（不在 self_bubbles 内的 message_candidates）"""
        # 过滤掉已用于 SELF 消息的元素
        used_self = set()
        for bubble in layout.self_bubbles:
            for elem in layout.message_candidates:
                if self._point_in_rect(elem.center, bubble):
                    used_self.add(id(elem))

        other_elems = [
            e for e in layout.message_candidates
            if id(e) not in used_self and not self._is_noise_candidate(e)
        ]

        # 过滤头像区域噪声（消息区左侧边缘的元素，如头像上的步数/未读数字）
        # 判断依据：布局特征（x 坐标偏左 + 面积小 + 置信度低），不依赖文本内容
        # 过滤头像区域噪声（消息区左侧边缘的元素，如头像上的步数/未读数字）
        # 判断依据：布局特征（x 坐标偏左 + 面积小 + 置信度低），不依赖文本内容
        avatar_noise_x_max = self.profile.left_boundary + 220
        def _is_avatar_noise(elem):
            if elem.center.x >= avatar_noise_x_max:
                return False
            area = elem.bbox.width * elem.bbox.height
            # 头像区域 + 面积极小 → 噪声
            if area < 300:
                return True
            # 头像区域 + 低置信度 + 面积不大 → 噪声（如 "1 10" area=464, conf=0.3）
            if elem.confidence < 0.6 and area < 1500:
                return True
            return False

        other_elems = [e for e in other_elems if not _is_avatar_noise(e)]

        if not other_elems:
            return []

        other_elems.sort(key=lambda e: e.center.y)

        # 按 y 坐标聚类，同时检查 x 一致性
        # 同一组（昵称+消息）的元素应在相近的 x 区域，
        # x 差距过大说明是头像噪声与消息的误聚类（tick81："10" cx=701 vs 消息 cx=919）
        clusters = [[other_elems[0]]]
        for elem in other_elems[1:]:
            last = clusters[-1][-1]
            y_gap = elem.center.y - last.center.y
            x_gap = abs(elem.center.x - last.center.x)
            # x 阈值：默认 7% 窗口宽度（1760px 下约 123px）
            # 昵称（x~750）和消息气泡（x~900-1050）之间差距可达 250px+
            # 如果当前 cluster 顶部是"有效昵称"（在昵称区域且面积足够大），则放宽到 25%
            x_threshold = self.profile.window_width * 0.07
            x_min = self.profile.window_width * self.profile.nickname_x_min_ratio
            x_max = self.profile.window_width * self.profile.nickname_x_max_ratio
            top = clusters[-1][0]
            top_area = top.bbox.width * top.bbox.height
            is_valid_nickname = x_min <= top.center.x <= x_max and top_area >= 1000
            if is_valid_nickname:
                x_threshold = self.profile.window_width * 0.25
            if y_gap < self.profile.message_cluster_threshold and x_gap < x_threshold:
                clusters[-1].append(elem)
            else:
                clusters.append([elem])

        messages = []
        for cluster in clusters:
            cluster.sort(key=lambda e: e.center.y)
            top = cluster[0]

            # 检查聚类顶部是否在昵称识别区域
            x_min = self.profile.window_width * self.profile.nickname_x_min_ratio
            x_max = self.profile.window_width * self.profile.nickname_x_max_ratio

            # 昵称判定：仅基于布局特征
            # cluster[0] 在昵称区域 且 cluster 中有元素超出昵称区域 → 是昵称+消息
            # cluster[0] 在昵称区域 但所有元素都在昵称区域内 → 拆分为独立短消息
            if len(cluster) > 1 and x_min <= top.center.x <= x_max:
                has_outside = any(
                    e.center.x < x_min or e.center.x > x_max
                    for e in cluster[1:]
                )
                if has_outside:
                    nickname = top.text
                    msg_elems = cluster[1:]
                    # 创建一条消息
                    self._append_message(messages, msg_elems, nickname, layout.chat_name)
                else:
                    # 所有元素都在昵称区域内：每条都是独立消息
                    for elem in cluster:
                        self._append_message(messages, [elem], "对方", layout.chat_name)
            else:
                nickname = "对方"
                msg_elems = cluster
                self._append_message(messages, msg_elems, nickname, layout.chat_name)

        self.debug_info["clusters"] = [
            {
                "texts": [e.text for e in c],
                "gap": c[1].center.y - c[0].center.y if len(c) > 1 else None,
                "top_x": c[0].center.x,
                "in_nick_range": x_min <= c[0].center.x <= x_max if len(c) > 0 else False,
                "nickname_assigned": (len(c) > 1 and x_min <= c[0].center.x <= x_max),
            }
            for c in clusters
        ]

        return messages

    @staticmethod
    def _point_in_rect(point: Point, rect: Rect) -> bool:
        return rect.x <= point.x <= rect.x + rect.width and rect.y <= point.y <= rect.y + rect.height

    @staticmethod
    def _is_system_notice(text: str, source_elements) -> bool:
        """判断是否为微信系统通知/安全提示"""
        text = text.strip()
        # 1. 文本内容匹配系统通知关键词
        for kw in _SYSTEM_NOTICE_KEYWORDS:
            if kw in text:
                return True
        # 2. 长度较长（>30字）且包含"安全""提示""保护""诈骗"等词
        if len(text) > 30 and any(k in text for k in ["安全", "提示", "保护", "诈骗", "转账", "风险"]):
            return True
        return False

    @staticmethod
    def _append_message(messages, msg_elems, nickname, chat_name):
        """将 msg_elems 合并为一条消息并追加到 messages 列表。"""
        if not msg_elems:
            return
        merged = " ".join(e.text for e in msg_elems)
        if MessageExtractor._is_system_notice(merged, msg_elems):
            sender = "系统"
            sender_type = SenderType.SYSTEM
        else:
            sender = nickname
            sender_type = SenderType.OTHER
        is_at_me = "@" in merged
        messages.append(
            ChatMessage(
                text=merged,
                sender=sender,
                sender_type=sender_type,
                chat_name=chat_name,
                is_at_me=is_at_me,
                source_elements=msg_elems,
            )
        )

    @staticmethod
    def _is_noise_candidate(elem: OCRTextElement, image_height: int = 1280) -> bool:
        """过滤掉明显不是消息内容的 OCR 噪声。

        判断依据：布局特征（位置、面积、置信度），不依赖文本内容。
        """
        text = elem.text.strip()
        if not text:
            return True
        # 时间戳模式（即使 LayoutParser 没识别到）— 时间戳有固定布局格式，保留
        if any(re.match(p, text) for p in TIMESTAMP_PATTERNS):
            return True

        area = elem.bbox.width * elem.bbox.height

        # 底部输入法候选框/图标噪声（y > 1150 且面积小）
        if elem.center.y > 1150 and area < 600:
            return True

        # 低置信度 + 面积小 → 图标/头像 OCR 误识别
        # 覆盖 "巧?" (area=903, conf=0.3) 和 "MipUxJ Z" (area=1668, conf=0.3)
        if elem.confidence < 0.4 and area < 2000:
            return True

        # 极低置信度 → 几乎一定是噪声
        if elem.confidence <= 0.3:
            return True

        return False

    @staticmethod
    def _message_y_position(msg: ChatMessage) -> int:
        """获取消息用于排序的 y 坐标"""
        if msg.source_elements:
            return msg.source_elements[0].center.y
        return 0
