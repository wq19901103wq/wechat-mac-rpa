# API Surface 速查表（当前生产架构）

> 本文档描述的是当前实际代码的公共接口，可直接复制粘贴使用。
> 下划线前缀的类/方法为内部实现，不推荐外部调用。

---

## L1: Domain Models

### `src/models/base.py`

**定位**: L1 领域模型

**公共接口**:
```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

@dataclass
class OCRTextElement:
    text: str
    bbox: Rect
    center: Point
    confidence: float

class SenderType(Enum):
    SELF = "self"
    OTHER = "other"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ChatMessage:
    text: str
    sender: str
    sender_type: SenderType
    chat_name: str
    is_at_me: bool = False
    timestamp: Optional[str] = None
    source_elements: Optional[List[OCRTextElement]] = None
    replied: bool = False
    reply_text: str = ""
    reply_time: Optional[float] = None
    message_type: str = "text"
    image_description: str = ""
    image_text: str = ""
    is_image_duplicate: bool = False
    account: str = ""
    local_id: Optional[int] = None
    server_id: Optional[str] = None
    create_time: Optional[int] = None
    raw_type: Optional[int] = None
    sender_wxid: Optional[str] = None

@dataclass
class ActionResult:
    success: bool
    sent_text: Optional[str] = None
    error: Optional[str] = None

@dataclass
class ChatListItem:
    nickname: str
    last_message_preview: str
    unread_count: str
    timestamp: str
    rect: Rect

@dataclass
class PerceptionResult:
    chat_name: str
    messages: List[ChatMessage]
    chat_list_items: List[ChatListItem]
    screenshot_path: str
    is_group: bool = False
    window_rect: Optional[Rect] = None
    scale_factor: float = 1.0
    debug_info: Optional[dict] = None
```

---

## L2: Capture / OCR

### `src/capture/window_capture.py`

**定位**: L2 窗口截图

**公共接口**:
```python
from dataclasses import dataclass
from typing import Optional

class WindowNotFoundError(Exception):
    """未找到目标窗口时抛出"""

class WeChatNotReadyError(Exception):
    """微信窗口尺寸异常（未登录/需扫码）时抛出"""

class CaptureValidationError(Exception):
    """截图内容验证失败时抛出"""

@dataclass
class CaptureResult:
    image_path: str
    window_rect: Rect
    scale_factor: float

class WindowCapture:
    def __init__(
        self,
        output_path: str = None,
        min_effective_width: int = 800,
        min_effective_height: int = 600,
        login_handler: Optional["WeChatLoginHandler"] = None,
    ): ...

    def capture(self) -> CaptureResult:
        """查找并截图微信主窗口。"""
```

**内部实现（不推荐外部调用）**:
```python
class WindowCapture:
    def _find_window(self) -> Optional[tuple]: ...
    def _find_window_with_options(self, options: int) -> Optional[tuple]: ...
    def _is_effective_window(self, rect: Rect) -> bool: ...
    def _activate_wechat(self) -> None: ...
    def _get_scale_factor(self) -> float: ...
    def _to_screencapture_region(self, rect: Rect) -> str: ...
    def _do_capture(self, rect: Rect, window_id: int) -> None: ...
    def _validate_wechat_screenshot(self, image_path: str) -> bool: ...
```

---

### `src/ocr/vision_ocr.py`

**定位**: L2 OCR 引擎

**公共接口**:
```python
from typing import List

class VisionOCREngine:
    def __init__(self, language: str = "zh-Hans"): ...

    def recognize(self, image_path: str) -> List[OCRTextElement]:
        """识别图片中的所有文本，返回按 center.y 升序排列的元素列表。"""

    @property
    def image_width(self) -> int: ...

    @property
    def image_height(self) -> int: ...

class OCRElement(OCRTextElement):
    """Backward compatibility wrapper，增加 x/y/cx/cy/width/height 等属性。"""

# 兼容别名
VisionOCR = VisionOCREngine
```

---

## L3: Layout / Message

### `src/layout/profile.py`

**定位**: L2 布局配置

**公共接口**:
```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LayoutProfile:
    name: str
    window_width: int
    window_height: int
    left_boundary: int
    chat_list_x_max: int
    title_y_max: int
    title_x_max_ratio: float
    input_y_min: int
    self_green: Tuple[int, int, int]
    self_green_tolerance: int
    min_bubble_pixels: int
    nickname_x_min_ratio: float
    nickname_x_max_ratio: float
    nickname_y_offset_min: int
    nickname_y_offset_max: int
    message_cluster_threshold: int = 80

PROFILE_WECHAT_MAC_1760X1280: LayoutProfile
```

---

### `src/layout/layout_parser.py`

**定位**: L3 布局分组

**公共接口**:
```python
from dataclasses import dataclass
from typing import List

@dataclass
class UILayout:
    chat_name: str
    chat_list_items: List[ChatListItem]
    title_elements: List[OCRTextElement]
    input_elements: List[OCRTextElement]
    timestamp_elements: List[OCRTextElement]
    self_bubbles: List[Rect]
    message_candidates: List[OCRTextElement]

class LayoutParser:
    def __init__(self, profile: LayoutProfile): ...

    @staticmethod
    def clean_chat_name(text: str) -> str: ...

    def parse(self, elements: List[OCRTextElement], image_path: str) -> UILayout:
        """将 OCR 元素分组为 UI 区域。"""
```

**内部实现（不推荐外部调用）**:
```python
class LayoutParser:
    def _detect_self_bubbles(self, arr: np.ndarray) -> List[Rect]: ...
    def _parse_chat_list(self, left_elements: List[OCRTextElement], image_path: str = "") -> List[ChatListItem]: ...
    def _extract_chat_name(self, title_elements: List[OCRTextElement], width: int) -> str: ...
```

---

### `src/message/extractor.py`

**定位**: L3 消息提取

**公共接口**:
```python
from typing import List

class MessageExtractor:
    def __init__(self, profile: LayoutProfile): ...

    def extract(self, layout: UILayout) -> List[ChatMessage]:
        """从 UILayout 中提取结构化消息列表。"""
```

**内部实现（不推荐外部调用）**:
```python
class MessageExtractor:
    def _extract_self_messages(self, layout: UILayout) -> List[ChatMessage]: ...
    def _extract_other_messages(self, layout: UILayout) -> List[ChatMessage]: ...
    @staticmethod
    def _point_in_rect(point: Point, rect: Rect) -> bool: ...
    @staticmethod
    def _is_system_notice(text: str, source_elements) -> bool: ...
    @staticmethod
    def _append_message(messages, msg_elems, nickname, chat_name): ...
    @staticmethod
    def _is_noise_candidate(elem: OCRTextElement, image_height: int = 1280) -> bool: ...
    @staticmethod
    def _message_y_position(msg: ChatMessage) -> int: ...
```

---

## L3.5: Perception

### `src/perception/smart_pipeline.py`

**定位**: L3.5 智能感知管道（主力）

**公共接口**:
```python
from typing import Optional

class SmartPerceptionPipeline:
    DEFAULT_PIXEL_DIFF_THRESHOLD = 0.001
    DEFAULT_MESSAGE_REGION = (0.35, 0.12, 0.95, 0.97)
    MIN_WINDOW_WIDTH = 800
    MIN_WINDOW_HEIGHT = 600

    def __init__(
        self,
        profile: LayoutProfile,
        api_key: Optional[str] = None,
        pixel_diff_threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
        message_region: tuple = DEFAULT_MESSAGE_REGION,
        always_use_api: bool = False,
    ): ...

    def perceive(self) -> Optional[PerceptionResult]:
        """执行完整视觉链路，带本地预判优化。"""

    def get_stats(self) -> dict:
        """返回统计信息（total_ticks / api_calls / skipped / local_fallbacks / api_ratio）。"""
```

**内部实现（不推荐外部调用）**:
```python
class SmartPerceptionPipeline:
    def _serialize_ocr_element(self, e) -> dict: ...
    def _serialize_layout(self, layout) -> dict: ...
    def _build_debug_info(self, layout, api_prompt: str = "", api_response: str = "", extraction_messages=None) -> dict: ...
    def _extract_local_messages(self, layout: UILayout, chat_name: str) -> list[ChatMessage]: ...
    def _run_local_only(self, image_path: str, window_rect: Rect, scale_factor: float) -> PerceptionResult: ...
    def _build_chat_list_items_from_api(self, api_chat_list: list, window_width: int, window_height: int, chat_name: str) -> list: ...
    def _run_with_api(self, image_path: str, window_rect: Rect, scale_factor: float) -> PerceptionResult: ...
    def _merge_chat_list(self, local_chat_list: list, api_chat_list: list) -> list: ...
    def _lcs_similarity(self, a: str, b: str) -> float: ...
    def _run_local_pipeline(self, image_path: str) -> dict: ...
    def _run_api_pipeline(self, image_path: str) -> dict: ...
    def _get_api_client(self) -> Optional["_QwenAPIClient"]: ...
    @staticmethod
    def _compute_hash(path: str) -> str: ...
    def _check_pixel_diff(self, prev_path: str, curr_path: str) -> float: ...
    def _convert_api_messages(self, raw_messages: list, chat_name: str) -> list[ChatMessage]: ...
```

---

### `src/perception/vision_pipeline.py`

**定位**: L3.5 视觉感知管道（备用回退）

**公共接口**:
```python
from typing import Optional

class VisionPipeline:
    def __init__(self, profile: LayoutProfile): ...

    def perceive(self) -> Optional[PerceptionResult]:
        """执行完整视觉链路：截图 → OCR → 布局分组 → 消息提取。"""
```

---

## L4: Session / Reply / Action / Memory / Tools

### `src/session/global_store.py`

**定位**: L4 全局消息存储

**公共接口**:
```python
from typing import Dict, List, Optional, Tuple

@dataclass
class ChatState:
    chat_id: str
    chat_name: str
    is_group: bool = False
    messages: List[ChatMessage] = field(default_factory=list)

class GlobalStore:
    def __init__(self, max_messages: int = 200, state_file: str = "data/global_state.json"): ...

    def merge_tick(
        self,
        chat_name: str,
        messages: List[ChatMessage],
        mode: str = "ocr",
        is_group: bool = False,
    ) -> Tuple[ChatState, List[ChatMessage]]:
        """合并 tick 检测到的消息，返回 (state, 未回复的消息列表)。"""

    def mark_replied(self, chat_name: str, target_msg: ChatMessage, reply_text: str): ...

    def inject_history(
        self,
        chat_name: str,
        messages: List[ChatMessage],
        mode: str = "weflow",
        is_group: bool = False,
    ) -> int:
        """批量注入历史消息，返回注入的消息数量。"""

    def get_unreplied(self, chat_name: str) -> List[ChatMessage]: ...

    def last_reply_time(self, chat_name: str) -> Optional[float]: ...

    def reply_count(self, chat_name: str) -> int: ...

    def save(self): ...

    def save_screenshot(self, image_path: str, session_id: str = None) -> str:
        """保存截图到 data/screenshots/ 目录，返回保存后的路径。"""
```

**内部实现（不推荐外部调用）**:
```python
class GlobalStore:
    def _merge_tick_legacy(self, chat_name: str, messages: List[ChatMessage], is_group: bool = False) -> List[ChatMessage]: ...
    def _merge_tick_lcs(self, chat_name: str, messages: List[ChatMessage], is_group: bool = False) -> List[ChatMessage]: ...
    def _merge_tick_weflow(self, chat_name: str, messages: List[ChatMessage]) -> List[ChatMessage]: ...
    def _load(self): ...
    @staticmethod
    def _dict_to_msg(m: dict, chat_name: str) -> ChatMessage: ...
    def _load_sharded(self, index_file: Path): ...
    @staticmethod
    def _msg_to_dict(m: ChatMessage) -> dict: ...
```

---

### `src/reply/generator.py`

**定位**: L4 回复生成

**公共接口**:
```python
from typing import Any, Dict, List

class ReplyGenerator:
    def __init__(self, llm_client=None, complex_llm_client=None, memory_engine=None): ...

    def generate(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage], is_group: bool = False, tick_id: int = 0) -> List[str]:
        """生成回复内容，返回多条回复列表（最多3条）。"""
```

**内部实现（不推荐外部调用）**:
```python
class ReplyGenerator:
    def _submit_to_judge(self, tick_id: int, replies: List[str], unreplied: List[ChatMessage], all_messages: List[ChatMessage], is_group: bool): ...
    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]: ...
    def _parse_replies(self, text: str) -> List[str]: ...
    def _load_skill_manifest(self) -> List[Dict[str, str]]: ...
    def _load_skill_content(self, skill_name: str) -> str: ...
    def _route_skills(self, user_text: str) -> List[str]: ...
    def _load_skill_one_liners(self) -> str: ...
    def _system_prompt(self) -> str: ...
    def _build_tools_context(self, chat_name: str) -> str: ...
    def _hermes_system_prompt(self) -> str: ...
    @staticmethod
    def _format_message_line(m: ChatMessage) -> str: ...
    def _build_user_prompt(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage], is_group: bool = False) -> str: ...
    def _clean_reply(self, text: str) -> str: ...
    def _fallback_reply(self, msg: ChatMessage) -> str: ...
```

---

### `src/reply/policy.py`

**定位**: L4 回复决策

**公共接口**:
```python
from typing import Any

class ReplyPolicy:
    def __init__(self, require_at_in_group: bool = False): ...

    def should_reply(self, msg: ChatMessage, session: Any) -> bool:
        """代码层只做最基本的过滤（自己/系统消息不回复），其余交给 AI 自主决定。"""
```

---

### `src/reply/session_memory.py`

**定位**: L4 跨 tick 短期记忆（工具缓存）

**公共接口**:
```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class CachedToolResult:
    tool_name: str
    query: str
    result: str
    timestamp: float
    ttl_seconds: int

    @property
    def expired(self) -> bool: ...
    @property
    def age_seconds(self) -> int: ...
    @property
    def remain_seconds(self) -> int: ...
    def format_line(self) -> str: ...

@dataclass
class SessionSnapshot:
    chat_name: str
    is_group: bool = False
    last_active: float = field(default_factory=time.time)
    tool_cache: List[CachedToolResult] = field(default_factory=list)
    bot_replies: List[str] = field(default_factory=list)

    def add_tool_result(self, tool_name: str, query: str, result: str, ttl: int): ...
    def get_valid_cache(self) -> List[CachedToolResult]: ...
    def get_all_cache(self) -> List[CachedToolResult]: ...
    def cleanup_expired(self): ...
    def add_reply(self, reply: str): ...
    def get_recent_replies(self, n: int = 5) -> List[str]: ...

class SessionMemory:
    DEFAULT_TTL = {
        "web_search": 300,
        "stock_query": 60,
        "get_weather": 1800,
        "search_memory": 600,
        "get_current_time": 0,
    }

    def __init__(self): ...

    def get_or_create(self, chat_name: str, is_group: bool = False) -> SessionSnapshot: ...

    def add_tool_result(self, chat_name: str, tool_name: str, query: str, result: str): ...

    def add_reply(self, chat_name: str, reply: str): ...

    def get_cache_lines(self, chat_name: str, include_expired: bool = False) -> List[str]: ...

    def cleanup_stale_sessions(self, max_idle_seconds: int = 3600): ...
```

---

### `src/memory/engine.py`

**定位**: L4 长期记忆引擎（LLM Wiki）

**公共接口**:
```python
from typing import Dict, List, Optional

class MemoryEngine:
    def __init__(self, llm_client=None): ...

    def get_user_memory(self, user_name: str, max_chars: int = 2000) -> str:
        """读取用户 wiki（含别名合并 + 外挂 facts），返回压缩后的摘要。"""

    def get_group_memory(self, group_name: str, max_chars: int = 2000) -> str:
        """读取群聊 wiki（含外挂 corrections），返回压缩后的摘要。"""

    def update_user_wiki(self, user_name: str, chat_name: str, messages: List, bot_replies: List[str]) -> None:
        """把更新任务加入队列，后台异步执行。"""

    def update_group_wiki(self, group_name: str, chat_name: str, messages: List, bot_replies: List[str]) -> None:
        """把群聊 wiki 更新任务加入队列，后台异步执行。"""

    def shutdown(self) -> None:
        """关闭 worker 线程，等待队列清空。"""

    def search_keyword(self, keyword: str, max_chars: int = 6000) -> str:
        """BM25 搜索本地 wiki，返回最相关的 wiki 内容。"""

    def search_related_mentions(self, text: str, exclude_user: Optional[str] = None, max_files: int = 5) -> List[str]:
        """扫描文本中提到的人名，只加载这些人自己的 wiki。"""
```

**内部实现（不推荐外部调用）**:
```python
class MemoryEngine:
    def _load_overrides(self) -> None: ...
    def _resolve_alias(self, user_name: str) -> str: ...
    def _all_names_for(self, user_name: str) -> List[str]: ...
    def _user_wiki_path(self, user_name: str) -> Path: ...
    def _group_wiki_path(self, group_name: str) -> Path: ...
    def _load_wiki(self, path: Path) -> str: ...
    def _save_wiki(self, path: Path, content: str) -> None: ...
    def _save_prompt(self, path: Path, prompt: str) -> None: ...
    def _save_alias_suggestion(self, path: Path, aliases: List[str], is_group: bool = False) -> None: ...
    def _compress_wiki(self, wiki: str, max_chars: int) -> str: ...
    def _format_conversation(self, messages: List, bot_replies: List[str]) -> str: ...
    def _do_update(self, task: dict) -> None: ...
    def _try_generate_wiki(self, prompt: str, path: Path, is_group: bool = False) -> str: ...
    def _strip_llm_prefix(self, text: str) -> str: ...
    def _do_update_user(self, task: dict) -> None: ...
    def _do_update_group(self, task: dict) -> None: ...
    def _extract_aliases_from_user_wiki(self, wiki: str, user_name: str) -> List[str]: ...
    def _extract_aliases_from_group_wiki(self, wiki: str) -> Dict[str, List[str]]: ...
    def _merge_aliases(self, user_name: str, new_aliases: List[str]) -> None: ...
    def _do_merge_aliases(self, user_name: str, new_aliases: List[str]) -> None: ...
    def _expand_search_keywords(self, keyword: str) -> List[str]: ...
    def _extract_all_snippets(self, content: str, keywords: List[str], max_snippets: int = 2) -> List[str]: ...
    def _start_worker(self) -> None: ...
```

---

### `src/tools/tool_registry.py`

**定位**: L4 工具注册表

**公共接口**:
```python
from typing import Any, Callable, Dict, List

class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], func: Callable): ...

    def to_openai_schema(self) -> Dict: ...

    def execute(self, arguments: str) -> str: ...

class ToolRegistry:
    def __init__(self): ...

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable) -> Tool: ...

    def get(self, name: str) -> Tool: ...

    def has(self, name: str) -> bool: ...

    def list_tools(self) -> List[Tool]: ...

    def to_openai_schemas(self) -> List[Dict]: ...

def get_registry() -> ToolRegistry:
    """返回全局单例 ToolRegistry。"""
```

---

### `src/action/message_sender.py`

**定位**: L4 动作层 — 消息发送

**公共接口**:
```python
from abc import ABC, abstractmethod

class MessageSender(ABC):
    @abstractmethod
    def send(self, text: str) -> ActionResult: ...

    @abstractmethod
    def send_image(self, image_path: str) -> ActionResult: ...

    @abstractmethod
    def send_file(self, file_path: str) -> ActionResult: ...

class WeChatMessageSender(MessageSender):
    def send(self, text: str) -> ActionResult:
        """发送文本消息到当前微信聊天。"""

    def send_image(self, image_path: str) -> ActionResult:
        """预留：将图片复制到剪贴板后粘贴发送。"""

    def send_file(self, file_path: str) -> ActionResult:
        """预留：拖拽文件到输入框或复制到剪贴板后粘贴发送。"""
```

**内部实现（不推荐外部调用）**:
```python
class WeChatMessageSender(MessageSender):
    def _ensure_wechat_frontmost(self, max_retries: int = 3) -> tuple[bool, str]: ...
    def _focus_input(self) -> tuple[int, str]: ...
    def _pbcopy(self, text: str) -> tuple[int, str]: ...
    def _paste(self, delay: float) -> tuple[int, str]: ...
    def _clear_clipboard(self) -> None: ...
    def _verify(self) -> tuple[str, int, int]: ...
    def _clear_input(self) -> None: ...
    def _keystroke(self, text: str) -> tuple[int, str]: ...
    def _send_return(self) -> tuple[int, str]: ...
```

---

### `src/action/chat_list_clicker.py`

**定位**: L4 动作层 — 聊天列表点击

**公共接口**:
```python
from typing import List, Optional

class ChatListClicker:
    def __init__(self, window_rect: Rect, scale_factor: float = 2.0): ...

    def click_item(self, item: ChatListItem) -> bool:
        """点击聊天列表项的中心位置。"""

    def click_by_index(self, items: list[ChatListItem], index: int) -> bool: ...

    def click_first_unread(
        self, items: list[ChatListItem], exclude_nickname: Optional[str] = None
    ) -> Optional[ChatListItem]:
        """点击第一个有未读消息的聊天项（排除当前已打开的聊天），返回被点击的项。"""
```

---

### `src/action/login_recovery.py`

**定位**: L4 动作层 — 登录恢复

**公共接口**:
```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class LoginRecoveryStatus(Enum):
    SUCCESS = "success"
    NEEDS_PHONE_CONFIRM = "needs_phone_confirm"
    NEEDS_QRCODE = "needs_qrcode"
    NO_LOGIN_BUTTON = "no_login_button"

@dataclass
class LoginRecoveryResult:
    status: LoginRecoveryStatus
    message: str

class WeChatLoginHandler:
    def __init__(
        self,
        capture_output: str = "/tmp/wechat_login_capture.png",
        login_keywords: List[str] = None,
        min_effective_width: int = 800,
        min_effective_height: int = 600,
    ): ...

    def handle(self) -> LoginRecoveryResult:
        """主流程：查找窗口 → 截图 OCR → 点击登录按钮 → 检查恢复结果。"""
```

**内部实现（不推荐外部调用）**:
```python
class WeChatLoginHandler:
    def _find_window(self) -> Optional[Rect]: ...
    def _capture_window(self, rect: Rect) -> str: ...
    def _detect_login_button(self, elements: List[OCRTextElement]) -> Optional[Rect]: ...
    def _is_phone_confirm_state(self, elements: List[OCRTextElement]) -> bool: ...
    def _click_login_button(self, window_rect: Rect, btn_rect: Rect) -> bool: ...
```

---

## L5: Bot

### `src/bot/wechat_bot.py`

**定位**: L5 主循环编排

**公共接口**:
```python
from typing import Callable, Dict, Optional

class WeChatBot:
    def __init__(
        self,
        profile: LayoutProfile,
        on_message: Optional[Callable] = None,
        llm_client=None,
        complex_llm_client=None,
        debug_mode: bool = False,
        use_openclaw: bool = True,
        perception=None,
        enable_chat_switch: bool = True,
    ): ...

    def tick(self) -> None:
        """执行一轮：感知 → 去重 → 决策 → 回复。"""

    def save_sessions(self) -> None:
        """保存全局状态到磁盘（增量保存）。"""

    def run_auto(self, interval: float = 5.0) -> None:
        """自动运行主循环。"""

    def send_to_chat(self, chat_name: str, text: str) -> ActionResult:
        """外部系统调用此接口主动发消息到指定聊天。"""
```

**内部实现（不推荐外部调用）**:
```python
class WeChatBot:
    def _inject_weflow_history(self, weflow_pipeline) -> None: ...
    def _try_switch_to_unread_chat(self, result: PerceptionResult) -> str: ...
```

---

## LLM Clients

### `src/llm/openclaw_client.py`

**定位**: LLM 客户端 — OpenClaw / Kimi Code 代理

**公共接口**:
```python
from typing import Any, Dict, List, Optional

class OpenClawClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18790",
        api_key: Optional[str] = None,
        model: str = "kimi-for-coding",
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ): ...

    def chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None):
        """调用 OpenClaw 生成回复。返回 str 或 message 对象（含 tool_calls 时）。"""

    @classmethod
    def from_openclaw_config(cls, config_path: Optional[str] = None) -> "OpenClawClient":
        """从 OpenClaw 配置文件自动读取 base_url 和模型配置。"""
```

> 注：`src/llm/qwen_client.py` 当前不存在于代码库中。实际生产环境使用 `OpenClawClient` 作为默认 LLM 客户端，SmartPerceptionPipeline 内部通过 `_QwenAPIClient` 私有类调用 qwen3.6-flash API。
