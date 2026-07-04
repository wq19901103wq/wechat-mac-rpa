#!/usr/bin/env python3
"""L3.5 Smart Vision Pipeline - 本地预判 + qwen3.6-flash API 兜底

架构:
    截图 → 像素差异判断 ──无变化──→ 本地 LayoutParser(chat_list) + 空 messages
                    │
                    └──有变化──→ 本地 LayoutParser(chat_list) + qwen3.6-flash(messages)

优势:
    - 92.6% 的 tick 无需调用 API（基于 69 张连续截图评测）
    - 消息提取准确率从本地 OCR 的 ~60% 提升到 qwen3.6-flash 的 ~83%
    - 群聊昵称识别、emoji、换行格式全部保留
"""

import base64
import hashlib

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from src.capture.window_capture import WeChatNotReadyError, WindowCapture
from src.layout.layout_parser import TIMESTAMP_PATTERNS, LayoutParser, UILayout
from src.layout.profile import LayoutProfile
from src.models.base import MEDIA_MESSAGE_TYPES, ChatListItem, ChatMessage, PerceptionResult, Rect, SenderType
from src.ocr.vision_ocr import VisionOCREngine
from src.utils.chat_utils import _is_group_chat_name

_logger = logging.getLogger("src.runtime.smart_pipeline")

# ---------------------------------------------------------------------------
# Qwen3.6-flash API 客户端（轻量封装，避免循环导入 benchmark 脚本）
# ---------------------------------------------------------------------------

QWEN_SYSTEM_PROMPT = """你是一位专精 UI 截图文字识别的 OCR 引擎。请仔细识别这张微信 Mac 版截图中的文字信息，并输出为 JSON。

截图包含以下区域：
1. 左侧聊天列表：每个条目包含头像、昵称、最后一条消息预览时间、未读角标数字（红色圆形背景）
2. 中间上方标题栏：当前聊天名称
3. 中间消息区域：消息按对话顺序从上到下排列

请严格按以下 JSON 格式输出（不要加 markdown 代码块，直接输出纯 JSON）：

{
  "chat_name": "当前聊天名称，如果没有则空字符串",
  "chat_list": [
    {"nickname": "昵称1", "unread_count": "未读数量，没有则为空字符串"},
    {"nickname": "昵称2", "unread_count": "3"}
  ],
  "messages": [
    {"sender": "自己", "text": "消息内容", "type": "text"},
    {"sender": "对方", "text": "私聊中对方的消息", "type": "text"},
    {"sender": "群成员昵称", "text": "群聊中对方的消息", "type": "text"},
    {"sender": "对方", "text": "", "type": "image", "image_description": "一只橘猫趴在键盘上", "image_text": "不想上班"},
    {"sender": "对方", "text": "", "type": "sticker", "image_description": "熊猫头流泪的表情包", "image_text": ""},
    {"sender": "群成员昵称", "text": "回复消息", "type": "text", "quoted_text": "被引用的消息内容"}
  ]
}

【关键识别规则 - 必须严格遵守】

1. 未读角标（unread_count）：
   - 【铁律】只有同时满足以下所有条件才是未读角标：
     a) 红色纯色圆形背景（无照片纹理、无头像内容）
     b) 白色数字居中
     c) 位于头像边界之外，浮在头像右上角
   - 【严禁 - 违反则识别失败】白底黑字、头像上的文字/数字、拼贴头像中的小头像，都不是未读角标。把这些误判为未读角标会导致整个识别结果作废，必须避免
   - 【严禁】左侧边栏（如微信图标）上的总未读数是全局的，绝不能把它当成某个具体聊天的未读角标
   - 预览消息右侧的时间戳（如"09:31"、"昨天"、"00:57"）不是未读角标
   - 如果没有红色圆形数字，unread_count 为空字符串""

2. 聊天列表（chat_list）：
   - 【重要】chat_list 必须严格按照截图中左侧列表从上到下的顺序排列，第一个就是截图中最顶部的条目
   - 左侧每个条目提取昵称，忽略头像区域的所有数字（除非是红色圆形未读角标）
   - 当前高亮选中的聊天条目也必须包含
   - 预览消息文字不要放入 nickname

3. 消息 sender 判断（这是最容易出错的地方，请仔细看对齐方向）：
   - 【铁律】对齐方向是判断 sender 的唯一依据，文本、图片、表情、链接卡片全部适用：
     - 靠右对齐 = "自己" 发送的，sender 必须填 "自己"
     - 靠左对齐 = 对方发送的，sender 填 "对方"（私聊）或群成员昵称（群聊）
   - 文本看气泡位置，图片/表情看图片到左右边缘的距离——离左边近=对方，离右边近=自己
   - 【绝对不能搞反】左边的消息不可能是自己发的，右边的消息不可能是对方发的
   - 【兜底规则】如果实在看不清对齐方向 → 不要猜，标"未知"
   - 【群聊 vs 私聊区分】
     - 群聊：消息上方会显示发送者昵称 → sender 必须填这个实际昵称
     - 私聊：只有两个人，消息上方不显示发送者昵称 → sender 必须填 "对方"
     - 【重要】群聊里可能存在一些比较长、看起来像普通消息的昵称（如"无论几点，都是饭点"、"人心中的成见是一顿大餐"）。这些文字是发送者身份标识，不是消息内容，不要输出为独立消息
   - 时间戳（如"昨天 21:58"、"11:34"、"00:22"）不是消息，不要输出

4. 消息 type 分类（重要新增，最容易出错）：
   - "text"：气泡内的所有内容。包括纯文字、emoji、气泡内的小表情符号
   - "image"：图片、照片、截图等非表情类的图像内容
   - "sticker"：仅指不在气泡内的独立大表情文件（微信表情商店下载的大表情、GIF 动图、熊猫头等）
   - "mixed"：图文混排（消息同时包含图片和文字）
   - "link_card"：链接卡片、文章分享、小程序卡片
   - 【绝对规则】白色/灰色/绿色气泡内的任何内容，一律识别为 text。sticker 只可能是脱离气泡的独立图像
   - 【常见错误】不要把气泡内的 emoji 或小表情错标为 sticker。例如"哈哈😂"、气泡内的"🐶"都是 text
   - 【常见错误】不要把对方发送的纯文字消息（即使很短或只含 emoji）错标为 image/sticker
   - 判断标准：先看消息是否有气泡包裹。有气泡 → text；无气泡的独立图像 → image/sticker
   - 区分 image 和 sticker：表情包通常尺寸较小、风格卡通、配简短文字；照片/截图通常尺寸较大、内容写实

5. 图片/表情识别（重要新增）：
   - 如果消息是图片或表情包，text 字段放图片上的文字（如有），没有则空字符串
   - image_description：详细描述图片/表情包的内容。例如：
     - 照片："夕阳下的海滩，天空呈现橙红色，有几只海鸥"
     - 表情包："一只熊猫头流泪，配文'我太难了'"
     - 截图："手机截图，显示微信聊天界面"
   - image_text：图片上叠加的文字（如表情包配字、截图中的文字、照片上的水印文字）
   - 【隐私保护】如果图片包含身份证、银行卡、地址、电话号码等隐私信息，image_description 简化为"[图片-包含隐私信息]"
   - 【隐私保护】如果图片包含裸露、暴力等不适宜内容，image_description 简化为"[图片]"
   - 链接卡片（link_card）的 image_description 描述卡片外观（如"分享了一篇公众号文章，标题为xxx"），image_text 提取卡片上的标题和摘要

6. 消息（messages）：
   - 包含所有消息：文字、图片、表情包、链接卡片
   - 排除所有时间戳
   - 按截图中从上到下顺序排列
   - 【新增】识别引用格式：微信 Mac 版引用消息时，主气泡外下方有灰色小字，左侧有竖线标识，显示被引用的消息内容
     - 如果消息包含引用，在 JSON 中加 "quoted_text" 字段，值为被引用区域的文字内容
     - 引用区域是纯文字 → quoted_text 填该文字
     - 引用区域显示 [图片] → quoted_text 填 "[图片]"
     - 引用区域显示 [表情] → quoted_text 填 "[表情]"
     - 引用区域是链接卡片 → quoted_text 填卡片标题/摘要
     - 没有引用 → 不加 quoted_text 字段

7. 头像过滤（极其重要，最容易出错的点）：
   - 聊天中每条消息旁边的小圆形头像是联系人照片，**绝对不是消息**，不要输出
   - 头像通常位于消息气泡的左侧（对方）或右侧（自己），紧贴消息区域边缘
   - 【铁律】任何小圆形、方形小图的头像照片，全部忽略，不算消息
   - 【常见错误】把对方的风景头像当成"图片消息"，把卡通头像当成"表情包"——这些都是头像，不是消息

8. 输入框过滤：
   - 截图最底部是输入框区域（有表情😊、文件📎、截图✂️、语音🎤按钮）
   - 输入框中的文字是未发送的草稿，不是已发送的消息，必须排除
   - 不要输出输入框中的任何内容

9. 只输出 JSON，不要任何解释文字。
"""


class _QwenAPIClient:
    """轻量级 qwen3.6-flash API 客户端，只保留核心调用逻辑。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen3.6-flash"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"),
        )

    def recognize(self, image_path: str) -> dict:
        raw, _, _, _ = self.recognize_with_debug(image_path)
        return raw

    def recognize_with_debug(self, image_path: str) -> tuple:
        """识别并返回 (parsed_result, prompt, raw_response, thinking)。"""
        b64 = self._image_to_base64(image_path)
        prompt = QWEN_SYSTEM_PROMPT
        messages: list[Any] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )
        msg = response.choices[0].message
        raw = msg.content or ""
        thinking = getattr(msg, "reasoning_content", "") or ""
        return self._extract_json(raw), prompt, raw, thinking

    @staticmethod
    def _image_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _extract_json(text: str) -> dict:
        from src.utils.json_extractor import extract_json
        return extract_json(text) or {}


# ---------------------------------------------------------------------------
# Image Description Dedup Tracker - 基于描述相似度的图片去重
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# SmartPerceptionPipeline
# ---------------------------------------------------------------------------

class SmartPerceptionPipeline:
    """智能感知管道：本地预判 + qwen3.6-flash API 兜底。

    与 VisionPipeline 接口完全兼容（duck typing），可直接替换。
    """

    # 消息区域像素差异阈值（0.001 = 0.1%）
    # 原值 0.005 导致大量无实质变化的截图触发 API，烧钱过快
    DEFAULT_PIXEL_DIFF_THRESHOLD = 0.001
    # 消息区域 ROI（相对坐标 x1, y1, x2, y2），排除左侧列表和底部输入框
    DEFAULT_MESSAGE_REGION = (0.35, 0.12, 0.95, 0.97)
    # 聊天列表区域（左侧边栏，用于检测新未读、列表滚动等变化）
    DEFAULT_CHAT_LIST_REGION = (0.0, 0.0, 0.35, 1.0)
    # 窗口最小有效尺寸（小于此值视为异常，如登录浮窗）
    MIN_WINDOW_WIDTH = 800
    MIN_WINDOW_HEIGHT = 600

    def __init__(
        self,
        profile: LayoutProfile,
        api_key: Optional[str] = None,
        pixel_diff_threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
        message_region: tuple = DEFAULT_MESSAGE_REGION,
        chat_list_region: tuple = DEFAULT_CHAT_LIST_REGION,
        always_use_api: bool = False,
    ):
        self.capture = WindowCapture()
        self.ocr = VisionOCREngine()
        self.layout = LayoutParser(profile)
        self.profile = profile

        # API 客户端（延迟初始化，失败时优雅降级）
        self._api_client: Optional[_QwenAPIClient] = None
        self._api_key = api_key
        self._api_available: Optional[bool] = None

        # 像素差异判断状态
        self.pixel_diff_threshold = pixel_diff_threshold
        self.message_region = message_region
        self.chat_list_region = chat_list_region
        self.always_use_api = always_use_api
        self._last_screenshot: Optional[Path] = None
        self._last_hash: Optional[str] = None
        self._last_perception: Optional[PerceptionResult] = None

        # 连续低差异计数：连续 N 帧差异 < 阈值，进入稳定模式进一步降低阈值
        self._consecutive_low_diff = 0
        self._stable_mode_threshold = pixel_diff_threshold * 0.5
        self._stable_mode_after = 3  # 连续 3 帧低差异后触发

        # 统计
        self.api_call_count = 0
        self.skip_count = 0
        self.local_fallback_count = 0

        # ===== WeFlow 集成：初始化 =====
        self._weflow_mode = os.getenv("WEFLOW_MODE", "ocr")
        self._weflow_pipeline = None
        if self._weflow_mode in ("weflow", "hybrid"):
            try:
                from .weflow_pipeline import WeFlowPipeline
                self._weflow_pipeline = WeFlowPipeline(profile)
                _logger.info(f"[SmartPipeline] WeFlow 模式: {self._weflow_mode}")
            except Exception as e:
                _logger.warning(f"[SmartPipeline] WeFlow 初始化失败，降级为 OCR: {e}")
                self._weflow_mode = "ocr"

    # -----------------------------------------------------------------------
    # 公共接口（与 VisionPipeline.perceive 签名一致）
    # -----------------------------------------------------------------------

    def perceive(self) -> Optional[PerceptionResult]:
        """执行完整视觉链路，带本地预判优化。

        Returns:
            PerceptionResult: 结构化结果
            None: 窗口捕获失败或尺寸异常
        """
        _logger.info("[SmartPipeline] perceive() 开始")

        # ===== WeFlow 分流 =====
        if self._weflow_mode in ("weflow", "hybrid") and self._weflow_pipeline:
            try:
                print("[SmartPipeline] calling weflow_pipeline.perceive()")
                result = self._weflow_pipeline.perceive()
                print(f"[SmartPipeline] weflow_pipeline.perceive() returned {result is not None}")
                if result is not None:
                    _logger.info(f"[SmartPipeline] WeFlow perceive 成功: {result.chat_name}, {len(result.messages)}条消息")
                    return result
                if self._weflow_mode == "weflow":
                    _logger.warning("[SmartPipeline] WeFlow 返回 None，且模式为 weflow-only")
                    return None
                _logger.warning("[SmartPipeline] WeFlow 返回 None，fallback 到 OCR")
            except Exception as e:
                _logger.warning(f"[SmartPipeline] WeFlow perceive 异常: {e}")
                if self._weflow_mode == "weflow":
                    return None

        # 1. 截图
        t_capture_start = time.time()
        try:
            capture_result = self.capture.capture()
            t_capture_ms = (time.time() - t_capture_start) * 1000
        except WeChatNotReadyError as e:
            _logger.warning(f"[SmartPipeline] 窗口捕获失败: {e}")
            return None
        except Exception as e:
            _logger.warning(f"[SmartPipeline] 窗口捕获异常: {e}")
            return None

        image_path = capture_result.image_path
        window_rect = capture_result.window_rect
        scale_factor = getattr(capture_result, "scale_factor", 1.0)
        _logger.info(
            f"[SmartPipeline] 截图成功: {Path(image_path).name}, "
            f"窗口={window_rect.width}x{window_rect.height}, scale={scale_factor}, "
            f"capture={t_capture_ms:.0f}ms"
        )

        # 2. 窗口尺寸检查（过滤登录浮窗等异常窗口）
        if window_rect.width < self.MIN_WINDOW_WIDTH or window_rect.height < self.MIN_WINDOW_HEIGHT:
            _logger.warning(
                f"[SmartPipeline] 窗口尺寸异常 ({window_rect.width}x{window_rect.height})，"
                f"小于最小阈值 ({self.MIN_WINDOW_WIDTH}x{self.MIN_WINDOW_HEIGHT})，"
                "可能处于登录/异常状态，跳过"
            )
            return None

        # 3. 像素差异判断（always_use_api 模式强制走 API）
        skip_api = False
        diff_ratio = None
        if self.always_use_api:
            _logger.info("[SmartPipeline] always_use_api=true，强制调用API（不跳过）")
        elif self._last_screenshot and self._last_screenshot.exists() and self._last_hash is not None:
            curr_hash = self._compute_hash(image_path)
            _logger.debug(
                f"[SmartPipeline] hash对比: prev={self._last_hash[:8]}... curr={curr_hash[:8]}..."
            )
            if curr_hash == self._last_hash:
                skip_api = True
                self._consecutive_low_diff += 1
                _logger.info(
                    "[SmartPipeline] 截图完全相同 (hash一致)，跳过API调用"
                )
            else:
                msg_diff = self._check_pixel_diff(str(self._last_screenshot), image_path, self.message_region)
                chat_diff = self._check_pixel_diff(str(self._last_screenshot), image_path, self.chat_list_region)
                diff_ratio = max(msg_diff, chat_diff)
                # 稳定模式：连续多帧低差异后，阈值临时降低 50%
                effective_threshold = self.pixel_diff_threshold
                if self._consecutive_low_diff >= self._stable_mode_after:
                    effective_threshold = self._stable_mode_threshold
                    _logger.info(
                        f"[SmartPipeline] 稳定模式已触发 (连续{self._consecutive_low_diff}帧低差异)，"
                        f"有效阈值降至 {effective_threshold:.6f}"
                    )
                skip_api = diff_ratio < effective_threshold
                if skip_api:
                    self._consecutive_low_diff += 1
                else:
                    self._consecutive_low_diff = 0
                _logger.info(
                    f"[SmartPipeline] 像素差异: msg={msg_diff:.6f} chat_list={chat_diff:.6f} "
                    f"max={diff_ratio:.6f} (阈值={effective_threshold}), "
                    f"决策={'跳过API' if skip_api else '调用API'}"
                )
        else:
            _logger.info("[SmartPipeline] 无历史截图，首次运行，调用API")
            self._consecutive_low_diff = 0

        self._last_screenshot = Path(image_path)
        self._last_hash = self._compute_hash(image_path)

        if skip_api:
            self.skip_count += 1
            _logger.info(
                f"[SmartPipeline] 本地跳过统计: skip_count={self.skip_count}, "
                f"api_count={self.api_call_count}, "
                f"skip_rate={self.skip_count/(self.skip_count+self.api_call_count)*100:.1f}%"
            )
            return self._run_local_only(image_path, window_rect, scale_factor)

        # 4. 有变化：本地 Layout + qwen3.6-flash API（并行）
        self.api_call_count += 1
        _logger.info(
            f"[SmartPipeline] 触发API调用: api_count={self.api_call_count}, "
            f"skip_count={self.skip_count}"
        )
        result = self._run_with_api(image_path, window_rect, scale_factor)
        _logger.info(f'[DEBUG] _run_with_api returned OK, msgs={len(result.messages) if result else 0}')
        return result

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 调试信息序列化
    # -----------------------------------------------------------------------

    @staticmethod
    def _serialize_ocr_element(e):
        return {
            "text": getattr(e, "text", ""),
            "x": getattr(e.bbox, "x", 0),
            "y": getattr(e.bbox, "y", 0),
            "w": getattr(e.bbox, "width", 0),
            "h": getattr(e.bbox, "height", 0),
            "cx": getattr(e.center, "x", 0),
            "cy": getattr(e.center, "y", 0),
            "confidence": getattr(e, "confidence", 0.0),
        }

    def _serialize_layout(self, layout) -> dict:
        if layout is None:
            return {}
        info = layout.debug_info if hasattr(layout, "debug_info") else {}
        result = {
            "layout_left_elements": info.get("left_elements", []),
            "layout_right_elements": info.get("right_elements", []),
            "layout_title_elements": [self._serialize_ocr_element(e) for e in getattr(layout, "title_elements", [])],
            "layout_input_elements": [self._serialize_ocr_element(e) for e in getattr(layout, "input_elements", [])],
            "layout_timestamp_elements": [self._serialize_ocr_element(e) for e in getattr(layout, "timestamp_elements", [])],
            "layout_chat_list_nicknames": [item.nickname for item in layout.chat_list_items],
            "layout_chat_list_unread": [item.unread_count for item in layout.chat_list_items],
            "layout_message_candidates": [self._serialize_ocr_element(e) for e in layout.message_candidates],
            "layout_self_bubbles": [
                {"x": r.x, "y": r.y, "w": r.width, "h": r.height}
                for r in layout.self_bubbles
            ],
        }
        # 添加 debug_info 中的其他字段
        if "chat_list" in info:
            result["layout_chat_list_groups"] = info["chat_list"].get("groups", [])
        return result

    def _build_debug_info(self, layout, api_prompt: str = "", api_response: str = "", extraction_messages=None, api_thinking: str = "") -> dict:
        info = self._serialize_layout(layout)
        info["api_prompt"] = api_prompt
        info["api_response"] = api_response
        info["api_thinking"] = api_thinking
        if extraction_messages is not None:
            info["extraction_messages"] = [
                {
                    "text": m.text,
                    "sender": m.sender,
                    "sender_type": m.sender_type.value if hasattr(m.sender_type, "value") else m.sender_type,
                    "chat_name": m.chat_name,
                    "type": m.message_type,
                    "image_description": m.image_description,
                    "image_text": m.image_text,

                }
                for m in extraction_messages
            ]
        return info

    def _extract_local_messages(
        self, layout: UILayout, chat_name: str
    ) -> list[ChatMessage]:
        """从本地 Layout 结果中粗略提取消息，用于本地路径检测变化。
        只做基本分类（self/other），不做复杂的群聊 sender 识别。"""
        messages: list[ChatMessage] = []
        input_texts = {e.text.strip() for e in layout.input_elements}

        for elem in layout.message_candidates:
            text = elem.text.strip()
            if not text:
                continue
            # 跳过时间戳
            if any(re.match(p, text) for p in TIMESTAMP_PATTERNS):
                continue
            # 跳过输入框内容
            if text in input_texts:
                continue

            # 判断 sender：中心点是否在 self_bubble 内
            is_self = False
            for bubble in layout.self_bubbles:
                if (
                    bubble.x <= elem.center.x <= bubble.x + bubble.width
                    and bubble.y <= elem.center.y <= bubble.y + bubble.height
                ):
                    is_self = True
                    break

            sender_type = SenderType.SELF if is_self else SenderType.OTHER
            sender = "自己" if is_self else chat_name

            messages.append(
                ChatMessage(
                    text=text,
                    sender=sender,
                    sender_type=sender_type,
                    chat_name=chat_name,
                    message_type="text",
                    source_elements=[elem],
                )
            )

        # 按 y 坐标排序（从上到下）
        messages.sort(
            key=lambda m: m.source_elements[0].center.y if m.source_elements else 0
        )
        return messages

    def _run_local_only(
        self, image_path: str, window_rect: Rect, scale_factor: float
    ) -> PerceptionResult:
        """无显著变化时：复用上次缓存，不跑 OCR。"""
        _logger.info("[SmartPipeline] 进入本地路径(跳过API+OCR)")
        if self._last_perception:
            _logger.info("[SmartPipeline] 复用上次感知结果(跳过OCR)")
            return PerceptionResult(
                chat_name=self._last_perception.chat_name,
                messages=[],
                chat_list_items=self._last_perception.chat_list_items,
                screenshot_path=image_path,
                is_group=self._last_perception.is_group,
                window_rect=window_rect,
                scale_factor=scale_factor,
                is_service_account_list=self._last_perception.is_service_account_list,
            )
        # 首次运行没有缓存，需要跑 OCR 建立基线
        _logger.info("[SmartPipeline] 首次运行，跑本地OCR建立缓存")
        t0 = time.time()
        elements = self.ocr.recognize(image_path)
        layout = self.layout.parse(elements, image_path)
        _logger.info(
            f"[SmartPipeline] 缓存建立: chat_name='{layout.chat_name}', "
            f"chat_list={len(layout.chat_list_items)}项, "
            f"耗时={(time.time()-t0)*1000:.0f}ms")
        debug_info = self._build_debug_info(layout)
        result = PerceptionResult(
            chat_name=layout.chat_name or "",
            messages=[],
            chat_list_items=layout.chat_list_items,
            screenshot_path=image_path,
            is_group=_is_group_chat_name(layout.chat_name or ""),
            window_rect=window_rect,
            scale_factor=scale_factor,
            debug_info=debug_info,
            is_service_account_list=layout.is_service_account_list,
        )
        self._last_perception = result
        return result

    def _build_chat_list_items_from_api(
        self, api_chat_list: list, window_width: int, window_height: int, chat_name: str
    ) -> list:
        """从 API 返回的 chat_list 构建 ChatListItem，使用基于索引的虚拟 rect。

        固定坐标基于 2x Retina（1738×1602）截图实测。非 Retina 或外接显示器上取值会偏移。
        此函数仅在本地 Layout 解析失败时作为兜底调用。
        """
        items = []
        is_expanded = bool(chat_name)
        sidebar_width = 55
        list_start_x = sidebar_width
        list_width = int(window_width * 0.35) if is_expanded else int(window_width * 0.85)
        list_start_y = 50
        item_height = 75

        for i, item in enumerate(api_chat_list):
            item_y = list_start_y + i * item_height
            rect = Rect(x=list_start_x, y=item_y, width=list_width, height=item_height)
            items.append(
                ChatListItem(
                    nickname=item.get("nickname", ""),
                    last_message_preview="",
                    unread_count=item.get("unread_count", ""),
                    timestamp="",
                    rect=rect,
                )
            )
        return items

    def _run_with_api(
        self, image_path: str, window_rect: Rect, scale_factor: float
    ) -> PerceptionResult:
        """有变化时：API 负责消息提取和昵称识别，本地 Layout 负责聊天列表定位。"""
        _logger.info("[SmartPipeline] 进入API路径")
        t0 = time.time()

        # 1. 并行跑本地 OCR + Layout（用于获取聊天列表的准确位置）
        local_t0 = time.time()
        try:
            local_result = self._run_local_pipeline(image_path)
            layout = local_result["layout"]
            local_chat_list = layout.chat_list_items
            local_ms = (time.time() - local_t0) * 1000
            _logger.info(f"[SmartPipeline] 本地Layout完成: chat_list={len(local_chat_list)}项, "
                         f"耗时={local_ms:.0f}ms (ocr+layout)")
        except Exception as e:
            _logger.warning(f"[SmartPipeline] 本地Layout失败: {e}")
            layout = None
            local_chat_list = []

        # 2. 调用 API（用于准确识别昵称、未读数、消息内容）
        api_result = self._run_api_pipeline(image_path)

        api_messages = api_result.get("messages", [])
        api_chat_name = api_result.get("chat_name", "")
        api_chat_list = api_result.get("chat_list", [])
        api_prompt = api_result.get("prompt", "")
        api_response = api_result.get("response", "")
        api_thinking = api_result.get("thinking", "")

        # 过滤误识别的未读角标（时间戳、群人数等）
        for item in api_chat_list:
            raw = item.get("unread_count", "")
            if not raw:
                continue
            # 包含冒号 → 时间戳（如"10:23"）
            if ":" in raw:
                item["unread_count"] = ""
                continue
            # 包含汉字 → 时间描述（如"昨天"）
            if any("\u4e00" <= c <= "\u9fff" for c in raw):
                item["unread_count"] = ""
                continue
            # 不是纯数字 → 误识别
            if not raw.isdigit():
                item["unread_count"] = ""
                continue
            # 数字过大 → 群人数等误识别（微信未读角标最大99）
            if int(raw) > 99:
                item["unread_count"] = ""
                continue

        # 3. 结合：本地 Layout 提供准确 rect，API 提供准确 nickname/unread_count
        t_merge_start = time.time()
        chat_list_items = self._merge_chat_list(local_chat_list, api_chat_list)
        messages = self._convert_api_messages(api_messages, api_chat_name)
        t_merge_ms = (time.time() - t_merge_start) * 1000

        total_ms = (time.time() - t0) * 1000
        _logger.info(
            f"[SmartPipeline] 完成: chat_name='{api_chat_name}', "
            f"messages={len(messages)}条, chat_list={len(chat_list_items)}项, "
            f"耗时={total_ms:.0f}ms merge={t_merge_ms:.0f}ms"
        )
        if messages:
            for i, m in enumerate(messages):
                preview = m.text[:40].replace(chr(10), '\\n')
                _logger.debug(f"  msg[{i}] sender={m.sender} type={m.sender_type.value} text='{preview}...'")

        debug_info = self._build_debug_info(layout, api_prompt, api_response, messages, api_thinking)
        debug_info["api_chat_list"] = api_chat_list
        result = PerceptionResult(
            chat_name=api_chat_name,
            messages=messages,
            chat_list_items=chat_list_items,
            screenshot_path=image_path,
            is_group=_is_group_chat_name(api_chat_name),
            window_rect=window_rect,
            scale_factor=scale_factor,
            debug_info=debug_info,
            is_service_account_list=getattr(layout, 'is_service_account_list', False) if layout else False,
        )
        self._last_perception = result
        return result

    def _merge_chat_list(
        self, local_chat_list: list, api_chat_list: list
    ) -> list:
        """结合本地 Layout 的准确位置和 API 的准确昵称/未读数。

        策略：
        1. 本地 chat_list 按 y 坐标排序（截图从上到下）
        2. API chat_list 理论上也应该从上到下，但不可靠
        3. 如果数量相同，按索引一一对应（本地提供 rect，API 提供 nickname/unread）
        4. 如果数量不同，用昵称模糊匹配
        """
        if not local_chat_list:
            # 本地 Layout 失败，回退到虚拟坐标。1738×1602 为 2x Retina 参考尺寸。
            _logger.warning("[SmartPipeline] 本地Layout无chat_list，回退到虚拟坐标")
            return self._build_chat_list_items_from_api(api_chat_list, 1738, 1602, "")

        # 本地按 y 坐标排序（确保从上到下）
        sorted_local = sorted(local_chat_list, key=lambda item: item.rect.y)

        # 如果数量相同，直接按索引对应
        if len(sorted_local) == len(api_chat_list):
            result = []
            for local_item, api_item in zip(sorted_local, api_chat_list):
                result.append(
                    ChatListItem(
                        nickname=api_item.get("nickname", local_item.nickname),
                        last_message_preview=local_item.last_message_preview,
                        unread_count=api_item.get("unread_count", local_item.unread_count),
                        timestamp=local_item.timestamp,
                        rect=local_item.rect,
                    )
                )
            return result

        # 数量不同：用昵称模糊匹配
        _logger.warning(
            f"[SmartPipeline] chat_list数量不匹配: 本地={len(sorted_local)}, API={len(api_chat_list)}, "
            f"使用昵称模糊匹配"
        )
        result = []
        for local_item in sorted_local:
            best_match = None
            best_score = 0.0
            for api_item in api_chat_list:
                nickname = api_item.get("nickname", "")
                score = self._lcs_similarity(local_item.nickname, nickname)
                if score > best_score:
                    best_score = score
                    best_match = api_item
            if best_match and best_score >= 0.5:
                result.append(
                    ChatListItem(
                        nickname=best_match.get("nickname", local_item.nickname),
                        last_message_preview=local_item.last_message_preview,
                        unread_count=best_match.get("unread_count", local_item.unread_count),
                        timestamp=local_item.timestamp,
                        rect=local_item.rect,
                    )
                )
            else:
                result.append(local_item)
        return result

    def _lcs_similarity(self, a: str, b: str) -> float:
        """最长公共子序列相似度。"""
        if not a or not b:
            return 0.0
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]
        return 2 * lcs_len / (m + n) if (m + n) > 0 else 0.0

    def _run_local_pipeline(self, image_path: str) -> dict:
        """本地 OCR + Layout 解析。"""
        elements = self.ocr.recognize(image_path)
        layout = self.layout.parse(elements, image_path)
        return {"layout": layout, "elements": elements}

    def _run_api_pipeline(self, image_path: str) -> dict:
        """调用 qwen3.6-flash API。"""
        client = self._get_api_client()
        if client is None:
            _logger.warning("[SmartPipeline] API客户端不可用，跳过API调用")
            return {}
        t0 = time.time()
        _logger.info(f"[SmartPipeline] API请求开始: model=qwen3.6-flash, image={Path(image_path).name}")
        try:
            raw, prompt, response_text, thinking = client.recognize_with_debug(image_path)
            latency_ms = (time.time() - t0) * 1000
            _logger.info(
                f"[SmartPipeline] API请求成功: latency={latency_ms:.0f}ms, "
                f"chat_name='{raw.get('chat_name', '')}', "
                f"messages={len(raw.get('messages', []))}, "
                f"chat_list={len(raw.get('chat_list', []))}, "
                f"thinking={len(thinking)}字"
            )
            return {
                "chat_name": raw.get("chat_name", ""),
                "messages": raw.get("messages", []),
                "chat_list": raw.get("chat_list", []),
                "prompt": prompt,
                "response": response_text,
                "thinking": thinking,
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            _logger.error(f"[SmartPipeline] API请求失败({latency_ms:.0f}ms): {e}")
            return {}

    def _get_api_client(self) -> Optional[_QwenAPIClient]:
        """延迟初始化 API 客户端，失败时返回 None。"""
        if self._api_available is False:
            return None
        if self._api_client is not None:
            return self._api_client
        try:
            self._api_client = _QwenAPIClient(api_key=self._api_key)
            self._api_available = True
            _logger.info("qwen3.6-flash API 客户端初始化成功")
            return self._api_client
        except Exception as e:
            self._api_available = False
            _logger.error(f"API 客户端初始化失败: {e}")
            return None

    # -----------------------------------------------------------------------
    # 像素差异计算
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_hash(path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()

    def _check_pixel_diff(self, prev_path: str, curr_path: str, region: tuple) -> float:
        """计算指定区域像素差异比例。"""
        try:
            prev = np.array(Image.open(prev_path).convert("RGB"), dtype=np.int16)
            curr = np.array(Image.open(curr_path).convert("RGB"), dtype=np.int16)
        except Exception as e:
            _logger.warning("[SmartPipeline] 像素 diff 计算失败: %s，视为有变化", e)
            return 1.0  # 出错时视为有变化

        if prev.shape != curr.shape:
            return 1.0  # 尺寸变化视为有变化

        h, w = curr.shape[:2]
        x1, y1, x2, y2 = region
        region_slice = (
            slice(int(y1 * h), int(y2 * h)),
            slice(int(x1 * w), int(x2 * w)),
        )

        diff = np.abs(curr[region_slice] - prev[region_slice])
        diff_mask = np.any(diff > 10, axis=2)  # RGB 任一通道差异 > 10
        diff_ratio = float(np.mean(diff_mask))
        return diff_ratio

    # -----------------------------------------------------------------------
    # 结果转换
    # -----------------------------------------------------------------------

    def _convert_api_messages(self, raw_messages: list, chat_name: str) -> list[ChatMessage]:
        """将 API 返回的 messages 转换为 ChatMessage 列表。

        支持图片/表情包识别和去重。
        私聊 sender 统一为 chat_name；群聊 sender 做校验防错。
        """
        is_group = _is_group_chat_name(chat_name)

        messages = []
        last_left_sender = "对方"  # 群聊中上一个左侧白色气泡的 sender

        for m in raw_messages:
            sender = m.get("sender", "")
            text = m.get("text", "")
            msg_type = m.get("type", "text") or "text"
            image_description = m.get("image_description", "") or ""
            image_text = m.get("image_text", "") or ""
            quoted_text = m.get("quoted_text", "") or ""

            if sender == "自己":
                sender_type = SenderType.SELF
            elif sender in ("对方", "未知", ""):
                sender_type = SenderType.OTHER
                sender = "未知" if is_group else "对方"
            else:
                # 群聊昵称或其他 sender
                sender_type = SenderType.OTHER

            # 私聊：sender 统一为 chat_name（对方昵称），避免下游看到"对方"
            if not is_group and sender_type == SenderType.OTHER:
                sender = chat_name

            # 群聊：sender 校验。API 有时把消息内容当成 sender（连续消息无昵称时）
            if is_group and sender_type == SenderType.OTHER:
                if sender == text and len(text) > 3:
                    sender = last_left_sender
                last_left_sender = sender

            # 图片/表情/链接卡片：允许 text 为空
            is_media = msg_type in MEDIA_MESSAGE_TYPES
            if not text and not is_media:
                continue

            # 从 API 返回提取时间戳（如果有），否则用当前时间
            msg_ts = m.get("timestamp", "") or m.get("time", "") or ""
            create_time = None
            if msg_ts:
                try:
                    from datetime import datetime
                    create_time = int(datetime.strptime(msg_ts.replace('  ',' '), "%Y-%m-%d %H:%M:%S").timestamp())
                except Exception as e:
                    _logger.warning("[SmartPipeline] 时间戳解析失败: %s (raw=%r)", e, msg_ts)
            if not create_time:
                import time
                create_time = int(time.time())

            messages.append(
                ChatMessage(
                    text=text,
                    sender=sender,
                    sender_type=sender_type,
                    chat_name=chat_name,
                    message_type=msg_type,
                    image_description=image_description,
                    image_text=image_text,
                    quoted_text=quoted_text,
                    timestamp=msg_ts,
                    create_time=create_time,
                )
            )
        return messages

    def get_stats(self) -> dict:
        """返回统计信息。"""
        total = self.api_call_count + self.skip_count
        return {
            "total_ticks": total,
            "api_calls": self.api_call_count,
            "skipped": self.skip_count,
            "local_fallbacks": self.local_fallback_count,
            "api_ratio": round(self.api_call_count / total, 3) if total > 0 else 0,
        }
