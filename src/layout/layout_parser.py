#!/usr/bin/env python3
"""L3 Layout Parser - UI 布局分组"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from scipy import ndimage
import logging

from src.layout.profile import LayoutProfile
from src.models.base import ChatListItem, OCRTextElement, Rect

_logger = logging.getLogger(__name__)


TIMESTAMP_PATTERNS = [
    r"^\d{1,2}:\d{2}$",
    r"^昨天 \d{1,2}:\d{2}$",
    r"^星期[一二三四五六日—] \d{1,2}:\d{2}$",
    r"^星期[一二三四五六日—]$",
    r"^\d{4}/\d{2}/\d{2}$",
]


@dataclass
class UILayout:
    """UI 布局分组结果"""
    chat_name: str
    chat_list_items: List[ChatListItem]
    title_elements: List[OCRTextElement]
    input_elements: List[OCRTextElement]
    timestamp_elements: List[OCRTextElement]
    self_bubbles: List[Rect]
    message_candidates: List[OCRTextElement]
    is_service_account_list: bool = False


class LayoutParser:
    """将 OCR 元素按 UI 区域分组。只做分组，不做过滤。"""

    def __init__(self, profile: LayoutProfile):
        self.profile = profile
        self.debug_info: Dict[str, Any] = {}

    @staticmethod
    def clean_chat_name(text: str) -> str:
        """仅去除聊天列表中的时间戳后缀（如 '昨天 22:26'），时间戳是 UI 元素而非昵称。"""
        text = re.sub(r'(昨天|今天)\s+\d{1,2}[:：]\d{2}$', '', text)
        text = re.sub(r'\s+\d{1,2}[:：]\d{2}$', '', text)
        return text.strip()

    def parse(self, elements: List[OCRTextElement], image_path: str) -> UILayout:
        """
        将 OCR 元素分组为 UI 区域。

        Returns:
            UILayout: 包含各区域元素的完整布局描述
        """
        import time
        t_parse_start = time.time()
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        width, height = img.size

        # 动态调整绝对坐标，适配窗口尺寸变化
        scale_x = width / self.profile.window_width
        scale_y = height / self.profile.window_height
        self._scale_x = scale_x
        self._scale_y = scale_y
        self._scaled_left_boundary = int(self.profile.left_boundary * scale_x)
        self._scaled_chat_list_x_max = int(self.profile.chat_list_x_max * scale_x)
        self._scaled_title_y_max = int(self.profile.title_y_max * scale_y)
        bottom_margin = self.profile.window_height - self.profile.input_y_min
        self._scaled_input_y_min = int(height - bottom_margin) if bottom_margin > 0 else self.profile.input_y_min

        # 1. 左右分割
        left_elements = [e for e in elements if e.bbox.x < self._scaled_left_boundary]
        right_elements = [e for e in elements if e.bbox.x >= self._scaled_left_boundary]

        # 2. 标题栏（右侧上部）
        title_x_max = int(width * self.profile.title_x_max_ratio)
        title_elements = [
            e for e in right_elements
            if e.bbox.y < self._scaled_title_y_max and e.bbox.x < title_x_max
        ]

        # 3. 输入框（右侧底部）
        input_elements = [
            e for e in right_elements if e.bbox.y >= self._scaled_input_y_min
        ]

        # 4. 时间戳（右侧消息区中央，匹配正则）
        timestamp_elements = []
        for e in right_elements:
            if not any(re.match(p, e.text) for p in TIMESTAMP_PATTERNS):
                continue
            if e.bbox.y < self._scaled_title_y_max or e.bbox.y >= self._scaled_input_y_min:
                continue
            # 位于消息区中央
            left_central = int(width * 0.25)
            right_central = int(width * 0.75)
            if left_central <= e.center.x <= right_central:
                timestamp_elements.append(e)

        # 5. 绿色气泡检测
        t_bubble_start = time.time()
        self_bubbles = self._detect_self_bubbles(arr)
        t_bubble_ms = (time.time() - t_bubble_start) * 1000

        # 6. 聊天列表解析
        t_chatlist_start = time.time()
        chat_list_items = self._parse_chat_list(left_elements, image_path)
        t_chatlist_ms = (time.time() - t_chatlist_start) * 1000

        # 7. 消息候选区：右侧排除已分类的元素
        excluded = set(id(e) for e in title_elements + input_elements + timestamp_elements)
        message_candidates = [e for e in right_elements if id(e) not in excluded]

        # 提取 chat_name
        chat_name = self._extract_chat_name(title_elements, width)

        # 检测是否为服务号/订阅号/公众号列表（误点固定入口后会进入该视图）
        is_service_account_list = self._detect_service_account_list(title_elements)

        t_parse_ms = (time.time() - t_parse_start) * 1000
        print(f"[Perf][Layout] parse: {t_parse_ms:.0f}ms "
              f"bubbles={t_bubble_ms:.0f}ms chat_list={t_chatlist_ms:.0f}ms "
              f"elements={len(elements)} items={len(chat_list_items)}")

        self.debug_info = {
            "left_elements": [{"text": e.text, "x": e.bbox.x, "y": e.bbox.y} for e in left_elements],
            "right_elements": [{"text": e.text, "x": e.bbox.x, "y": e.bbox.y} for e in right_elements],
            "title_elements": [{"text": e.text, "x": e.bbox.x, "y": e.bbox.y} for e in title_elements],
            "input_elements": [{"text": e.text, "x": e.bbox.x, "y": e.bbox.y} for e in input_elements],
            "timestamp_elements": [{"text": e.text, "x": e.bbox.x, "y": e.bbox.y} for e in timestamp_elements],
            "message_candidates": [{"text": e.text, "cx": e.center.x, "cy": e.center.y} for e in message_candidates],
            "self_bubbles": [{"x": b.x, "y": b.y, "w": b.width, "h": b.height} for b in self_bubbles],
        }

        return UILayout(
            chat_name=chat_name,
            chat_list_items=chat_list_items,
            title_elements=title_elements,
            input_elements=input_elements,
            timestamp_elements=timestamp_elements,
            self_bubbles=self_bubbles,
            message_candidates=message_candidates,
            is_service_account_list=is_service_account_list,
        )

    def _detect_self_bubbles(self, arr: np.ndarray) -> List[Rect]:
        """通过颜色检测识别绿色气泡区域。

        优化策略：
        1. 只扫描右侧消息区，避免全图扫描
        2. 先粗筛（容差 15）再找连通区域，最后精筛（容差 35）
        """
        h, w = arr.shape[:2]
        # 裁剪到右侧消息区（使用动态坐标）
        left = getattr(self, '_scaled_left_boundary', self.profile.left_boundary)
        title_y = getattr(self, '_scaled_title_y_max', self.profile.title_y_max)
        input_y = getattr(self, '_scaled_input_y_min', self.profile.input_y_min)
        x1 = max(0, left)
        y1 = max(0, title_y)
        x2 = min(w, w)
        y2 = min(h, input_y)
        if x1 >= x2 or y1 >= y2:
            return []

        msg_region = arr[y1:y2, x1:x2, :]
        target = np.array(self.profile.self_green, dtype=int)

        # Step 1: 粗筛——高置信度绿色像素（容差 15）
        diff = np.abs(msg_region.astype(int) - target)
        strict_mask = np.all(diff < 15, axis=2)

        # 如果没有严格匹配，放宽到容差 35
        if not np.any(strict_mask):
            strict_mask = np.all(diff < self.profile.self_green_tolerance, axis=2)

        # Step 2: 连通区域标记（在稀疏 mask 上更快）
        labeled, num = ndimage.label(strict_mask)
        bubbles: List[Rect] = []
        for i in range(1, num + 1):
            ys, xs = np.where(labeled == i)
            pixel_count = len(xs)
            if pixel_count < self.profile.min_bubble_pixels:
                continue
            bx1, bx2 = int(xs.min()), int(xs.max())
            by1, by2 = int(ys.min()), int(ys.max())
            # 坐标加回裁剪偏移量
            bubbles.append(
                Rect(
                    x=bx1 + x1,
                    y=by1 + y1,
                    width=bx2 - bx1 + 1,
                    height=by2 - by1 + 1,
                )
            )
        return bubbles

    def _parse_chat_list(self, left_elements: List[OCRTextElement], image_path: str = "") -> List[ChatListItem]:
        """解析左侧聊天列表。"""
        # 过滤出列表区内的元素（排除分割线/顶部搜索栏等）
        # 搜索栏在 y<80，聊天列表项从 y>80 开始
        chat_list_x_max = getattr(self, '_scaled_chat_list_x_max', self.profile.chat_list_x_max)
        elems = [
            e for e in left_elements
            if e.bbox.x <= chat_list_x_max and e.center.y > 80
        ]
        elems.sort(key=lambda e: e.center.y)
        if not elems:
            return []

        # 昵称列：x >= 150 覆盖昵称（昵称可能在 x=150~230 范围内，如 "王芊 @ai"）
        scale_x = getattr(self, '_scale_x', 1.0)
        scale_y = getattr(self, '_scale_y', 1.0)
        nick_min = int(150 * scale_x)
        nick_max = int(chat_list_x_max * 0.95)
        nick_col = [e for e in elems if nick_min <= e.bbox.x <= nick_max]
        # 过滤头像区域噪声：未读角标/微信运动步数字体很小，
        # bbox 面积通常 200-700；而昵称/预览文字面积通常 > 1500。
        # 用面积阈值 1000 自然分割，比 isdigit() 更本质（不依赖文本内容）。
        min_nickname_area = 1000
        nick_col = [
            e for e in nick_col
            if (e.bbox.width * e.bbox.height) >= min_nickname_area
        ]
        nick_col.sort(key=lambda e: e.center.y)
        if not nick_col:
            return []

        # 按 y 间隔 < 50 分组（同一聊天项的昵称+预览）
        groups: List[List[OCRTextElement]] = []
        current = [nick_col[0]]
        for i in range(1, len(nick_col)):
            if nick_col[i].center.y - nick_col[i - 1].center.y < 50:
                current.append(nick_col[i])
            else:
                groups.append(current)
                current = [nick_col[i]]
        groups.append(current)

        # 先计算每个组的代表 y
        group_anchor_y = []
        for group in groups:
            group.sort(key=lambda e: e.center.y)
            anchor_y = group[0].center.y
            group_anchor_y.append(anchor_y)

        # 未读检测：OCR 数字 + 颜色检测
        unread_for_group: List[str] = [""] * len(groups)
        
        # 1) OCR 数字候选：只接受在头像右上角精确区域的小元素
        # 未读角标位于头像右上角，center.x 约 100-150（Retina），头像内部噪声 < 80，时间戳 > 180
        # 两位数字如 "39" 面积可达 560，故阈值放宽到 1000。
        unread_x_min = int(80 * scale_x)
        unread_x_max = int(170 * scale_x)
        unread_max_width = int(35 * scale_x)
        unread_max_height = int(30 * scale_y)
        unread_area_threshold = int(1000 * scale_x * scale_y)
        unread_candidates = [
            e for e in elems
            if unread_x_min <= e.center.x <= unread_x_max and (e.bbox.width * e.bbox.height) < unread_area_threshold
            and e.bbox.width < unread_max_width and e.bbox.height < unread_max_height
        ]
        for uc in unread_candidates:
            # 过滤不合理的未读数：非数字、>99、或为空
            text = uc.text.strip()
            if not text or not text.isdigit() or int(text) > 99:
                continue
            best_idx = -1
            best_dist = float('inf')
            for idx, anchor_y in enumerate(group_anchor_y):
                dist = abs(uc.center.y - anchor_y)
                if dist < best_dist and dist < 60:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0:
                unread_for_group[best_idx] = text
        
        # 2) 颜色检测补充：对每个聊天项的头像右上角精确区域检测红色 badge
        if image_path:
            try:
                img_arr = np.array(Image.open(image_path).convert("RGB"))
                for idx, anchor_y in enumerate(group_anchor_y):
                    if unread_for_group[idx]:
                        continue  # OCR 已检测到数字，跳过颜色检测
                    
                    # 头像顶部估算：昵称顶部 - 15
                    nick_top = groups[idx][0].bbox.y
                    avatar_top = nick_top - int(15 * scale_y)
                    # 头像在昵称左侧，宽度约 50-55px
                    nick_left = groups[idx][0].bbox.x
                    avatar_left = max(0, nick_left - int(55 * scale_x))
                    
                    # badge 在头像右上角，取头像右上 badge_size x badge_size 区域
                    badge_size = int(25 * max(scale_x, scale_y))
                    avatar_width = int(55 * scale_x)
                    y1 = max(0, avatar_top)
                    y2 = min(img_arr.shape[0], avatar_top + badge_size)
                    x1 = max(0, avatar_left + avatar_width - badge_size)
                    x2 = min(img_arr.shape[1], avatar_left + avatar_width)
                    
                    if y1 < y2 and x1 < x2:
                        region = img_arr[y1:y2, x1:x2, :].astype(int)
                        rr, gg, bb = region[:, :, 0], region[:, :, 1], region[:, :, 2]
                        # 严格红色条件
                        strict_red = (rr > 200) & (rr - gg > 50) & (rr - bb > 50)
                        red_pixels = np.sum(strict_red)
                        if red_pixels >= 50:
                            unread_for_group[idx] = "1"
            except Exception as e:
                _logger.warning("detect unread badge failed: %s", e)

        def _clean_nickname(text: str) -> str:
            """清理 OCR 产生的昵称污染：去掉噪声符号前缀和时间戳后缀。

            注意：不基于内容（如"是不是数字"）清洗，只去掉明显的符号前缀。
            头像数字/乱码应在 nick_col 阶段通过面积过滤排除，不在此处处理。
            """
            # 去掉开头到第一个中文字符之前的噪声符号（最多10个字符）
            noise_chars = set('！＠@#$%^&*()[]{}<>"\'，。、；：？！-~+=|\\/ \t\n\r0123456789')
            i = 0
            while i < min(10, len(text)) and text[i] in noise_chars:
                i += 1
            text = text[i:]
            # 情况C: '岔站 王芊' → 头像上的中文乱码前缀（2-3个无意义汉字）
            # 如果开头是2-3个中文+空格，且后面更长，去掉前缀
            m = re.match(r'^([\u4e00-\u9fa5]{1,3})\s+(.{4,})$', text)
            if m and m.group(1) in ('岔站', '收到', '搜索', '好的', '可以'):
                text = m.group(2)
            # 2. 去掉时间戳后缀，如 '王老板们昨天 22:26' → '王老板们'
            text = re.sub(r'(昨天|今天)\s+\d{1,2}[:：]\d{2}$', '', text)
            text = re.sub(r'\s+\d{1,2}[:：]\d{2}$', '', text)
            return text.strip()

        items: List[ChatListItem] = []
        for idx, group in enumerate(groups):
            group.sort(key=lambda e: e.center.y)
            nickname = _clean_nickname(group[0].text)
            last_message_preview = ""
            if len(group) > 1:
                last_message_preview = group[-1].text

            unread_count = unread_for_group[idx]
            timestamp = ""

            anchor_y = group_anchor_y[idx]
            y_min = anchor_y - 85
            y_max = anchor_y + 85

            # 时间戳
            for e in elems:
                if not (y_min - 20 <= e.center.y <= y_max + 20):
                    continue
                if e.bbox.x > chat_list_x_max * 0.70:
                    if e.text in TIMESTAMP_PATTERNS or (len(e.text) >= 4 and e.text[1] == ':' and e.text[:1].isdigit() and e.text[2:].isdigit()):
                        timestamp = e.text

            # 列表项的 rect
            all_elems = list(group)
            min_x = min(e.bbox.x for e in all_elems)
            min_y = min(e.bbox.y for e in all_elems)
            max_x = max(e.bbox.x + e.bbox.width for e in all_elems)
            max_y = max(e.bbox.y + e.bbox.height for e in all_elems)
            item_rect = Rect(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

            items.append(
                ChatListItem(
                    nickname=nickname,
                    last_message_preview=last_message_preview,
                    unread_count=unread_count,
                    timestamp=timestamp,
                    rect=item_rect,
                )
            )

        # 过滤微信固定入口，避免 Bot 误点击服务号/订阅号后无法返回
        _BLOCKED_ENTRIES = {"订阅号", "服务号", "公众号"}
        items = [item for item in items if item.nickname not in _BLOCKED_ENTRIES]

        self.debug_info["chat_list"] = {
            "groups": [[e.text for e in g] for g in groups],
            "group_anchor_y": group_anchor_y,
            "unread_for_group": unread_for_group,
            "items": [
                {"nickname": i.nickname, "preview": i.last_message_preview, "unread": i.unread_count}
                for i in items
            ],
        }

        return items

    def _extract_chat_name(self, title_elements: List[OCRTextElement], width: int) -> str:
        """从标题元素中提取聊天名称。"""
        if not title_elements:
            return ""

        # 标题噪声过滤：基于布局特征（位置在窗口右上角），不依赖文本内容
        # 窗口控制按钮（关闭/最小化/最大化）通常在右上角 x > width * 0.85
        right_edge_threshold = width * 0.85
        filtered = [
            e for e in title_elements
            if e.center.x < right_edge_threshold
            and not any(re.match(p, e.text) for p in TIMESTAMP_PATTERNS)
        ]
        candidates = filtered if filtered else title_elements
        best = max(candidates, key=lambda e: len(e.text))
        return self.clean_chat_name(best.text)

    _SERVICE_ACCOUNT_TITLES = {"服务号", "订阅号", "公众号"}

    def _detect_service_account_list(
        self, title_elements: List[OCRTextElement]
    ) -> bool:
        """检测当前是否为服务号/订阅号/公众号列表视图。"""
        for e in title_elements:
            text = e.text.strip()
            if text in self._SERVICE_ACCOUNT_TITLES:
                return True
        return False
