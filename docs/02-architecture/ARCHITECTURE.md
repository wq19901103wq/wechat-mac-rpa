# 微信 Mac RPA 架构设计文档

> **重要提示：本文档描述当前生产架构的设计理念和分层结构。具体公共接口以 `API_SURFACE.md` 为准，模块索引以 `MODULE_INDEX.md` 为准。**
>
> 当前实际代码位于 `src/` 目录下，模块化架构（L1-L5）已全部落地。
>
> **文档分类**：
> - `API_SURFACE.md` — 当前实际代码的公共接口（可直接复制粘贴）
> - `MODULE_INDEX.md` — 按问题/文件索引（改代码前先看这个）
> - `ARCHITECTURE.md` — 架构设计理念、分层规则、设计决策（本文档）
>
> 目标：让任何 AI Agent 在 5 分钟内理解系统结构，并能独立修改任一模块。

---

## 一、架构总览

### 1.1 核心原则

1. **单一职责**：每个文件只做一件事
2. **依赖单向**：上层可调用下层，下层不可反向依赖
3. **配置与代码分离**：所有布局相关的边界常量提取到 `LayoutProfile`，会话/策略参数提取到对应 L4 模块
4. **测试即文档**：每个模块有独立单元测试

### 1.2 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Application                                       │
│  src/bot/wechat_bot.py                               │
│  主循环编排：perceive → session → policy → generate → action│
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Layer 4:    │    │  Layer 4:       │    │  Layer 4:    │
│  Session     │    │  Reply          │    │  Action      │
│  会话/去重   │    │  回复决策/生成  │    │  执行发送    │
└──────────────┘    └─────────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3.5: SmartPerceptionPipeline                         │
│  src/perception/smart_pipeline.py                    │
│  智能感知管道：本地预判 + API兜底，对 Bot 层隐藏视觉细节    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Layer 3:    │    │  Layer 3:       │    │  Layer 2:    │
│  Message     │    │  Layout         │    │  Capture     │
│  消息模型/   │    │  布局解析器     │    │  窗口截图    │
│  提取器      │    │                 │    │              │
└──────────────┘    └─────────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Layer 2:    │    │  Layer 2:       │    │              │
│  OCR         │    │  LayoutProfile  │    │              │
│  文字识别    │    │  布局配置       │    │              │
└──────────────┘    └─────────────────┘    └──────────────┘
        │                     │
        ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Domain Models                                     │
│  基础数据类型：Point, Rect, OCRTextElement,                 │
│  ChatMessage, ActionResult, PerceptionResult                │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 依赖规则

- **Domain (L1)** 不依赖任何其他层
- **Capture / OCR / LayoutProfile (L2)** 只依赖 L1
- **Message / Layout (L3)** 只依赖 L1-L2
- **VisionPipeline (L3.5)** 可依赖 L1-L3，但**对 L4-L5 隐藏内部细节**
- **Session / Reply / Action (L4)** 可依赖 L1 和 L3.5 的输出（`ChatMessage`, `PerceptionResult`），**不可直接依赖 L2-L3 的内部实现**
- **Bot (L5)** 只依赖 L1、L3.5、L4。**Bot 层禁止直接 import OCR/Layout/Capture**

**禁止**：
- 下层模块 `import` 上层模块
- Bot 层直接操作 `OCRTextElement`、`UILayout`、`CaptureResult`
- Session 层暴露视觉实现细节给 Bot 层

**层边界规则（防越界）**：
- **L3.5 感知层**：只做"截图 → 提取消息"，**禁止做去重、状态管理、回复决策**。去重是 L4 Session/GlobalStore 的职责
- **L4 会话层**：只做"去重 + 状态管理 + 持久化"。感知细节对 L4 隐藏
- **跨层重复功能 = Bug**：如果某功能在两层同时出现（如 SmartPipeline 里的 `ImageDedupTracker` 和 GlobalStore 的 `_is_fuzzy_duplicate`），说明边界混乱，必须迁移到正确层

---

## 二、各模块详细设计

### 2.1 Domain Models (L1)

**文件**: `src/models/base.py`

**职责**: 定义整个系统的基础数据结构。无业务逻辑，纯数据容器。

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

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
    """OCR 原始输出元素"""
    text: str
    bbox: Rect           # 外接矩形
    center: Point        # 中心点，用于位置判断
    confidence: float

class SenderType(Enum):
    SELF = "self"
    OTHER = "other"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ChatMessage:
    """领域模型：一条聊天消息"""
    text: str
    sender: str
    sender_type: SenderType
    chat_name: str
    is_at_me: bool = False
    timestamp: Optional[str] = None
    source_elements: Optional[List[OCRTextElement]] = None  # 溯源：仅供 L3 Extractor 构造消息和 L4 Session debug 使用，Bot 层禁止读取

@dataclass
class SentMessage:
    """记录由 Bot 自己发送的消息"""
    text: str
    sent_at: float

@dataclass
class ActionResult:
    success: bool
    sent_text: Optional[str] = None
    error: Optional[str] = None

@dataclass
class ChatListItem:
    """左侧聊天列表项，属于 Domain Model，被 Layout 和 UIInteractor 共用"""
    nickname: str
    last_message_preview: str
    unread_count: str
    timestamp: str
    rect: Rect  # 列表项在屏幕上的位置，供 UIInteractor 点击使用

@dataclass
class PerceptionResult:
    """VisionPipeline 的输出，对 Bot 层隐藏所有视觉实现细节"""
    chat_name: str
    messages: List[ChatMessage]
    chat_list_items: List[ChatListItem]
    screenshot_path: str
```

**设计要点**:
- `OCRTextElement` 是 OCR 层和 Layout 层的通用接口
- `ChatMessage.source_elements` 保留溯源能力，方便 debug
- `ActionResult` 记录发送动作的结果（成功/失败）

---

### 2.2 Capture (L2)

**文件**: `src/capture/window_capture.py`

**职责**: 找到微信窗口并截图。输出原始图片。

**接口**:

```python
class WindowCapture:
    def __init__(self, output_path: str = None,
                 min_effective_width: int = 800,
                 min_effective_height: int = 600):
        pass

    def capture(self) -> CaptureResult:
        """
        查找并截图微信主窗口。

        如果找到的最大窗口尺寸过小（< min_effective_width × min_effective_height），
        会先尝试 `osascript -e 'tell application "WeChat" to activate'` 激活微信，
        等待 2 秒后重试。重试后仍无效则抛出 `WeChatNotReadyError`。

        Returns:
            CaptureResult: 包含图片路径和窗口几何信息

        Raises:
            WindowNotFoundError: 未找到任何微信窗口
            WeChatNotReadyError: 窗口尺寸异常，可能需要扫码登录
        """
        pass

@dataclass
class CaptureResult:
    image_path: str
    window_rect: Rect
    scale_factor: float  # Retina 屏幕为 2.0，普通屏幕为 1.0
```

**实现细节**:
- 使用 `Quartz.CGWindowListCopyWindowInfo` 枚举窗口
- 过滤条件：`owner in ['WeChat', '微信']` 且 `width > 200, height > 200`
- 在多个窗口中选择**面积最大**的窗口
- 如果最大窗口仍小于 `min_effective_width × min_effective_height`（默认 800×600），自动激活微信并重试
- 重试无效时抛出 `WeChatNotReadyError`，提示可能需要扫码登录
- 使用 `screencapture -R` 命令截图
- 自动处理 Retina 屏幕缩放

**测试策略**:
- Mock Quartz 窗口列表，验证能正确识别主窗口
- 验证截图文件生成且尺寸与窗口声明一致

---

### 2.3 OCR (L2)

**文件**: `src/ocr/vision_ocr.py`

**职责**: 从图片中提取文本元素。不做任何过滤或解释。

**接口**:

```python
class VisionOCREngine:
    def recognize(self, image_path: str) -> List[OCRTextElement]:
        """
        识别图片中的所有文本。
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            OCRTextElement 列表，按 y 坐标从上到下排序
        """
        pass
```

**实现细节**:
- 使用 macOS Vision 框架 (`VNRecognizeTextRequest`)
- 将 Vision 的归一化坐标转换为像素坐标
- 输出按 `center.y` 升序排列

**约束**:
- 不过滤时间戳
- 不判断 sender_type
- 不做任何业务假设

**测试策略**:
- 用固定测试图片，验证输出元素数量和文本内容
- 验证坐标转换正确

---

### 2.4 LayoutProfile (L2)

**文件**: `src/layout/profile.py`

**职责**: 把写死的布局常量提取为配置对象。

**接口**:

```python
@dataclass
class LayoutProfile:
    """
    针对特定微信版本 + 分辨率的布局配置。
    当微信更新或窗口缩放异常时，优先调整此配置。
    """
    name: str
    window_width: int           # 适配窗口宽度
    window_height: int          # 适配窗口高度
    
    # 区域边界（像素）
    left_boundary: int          # 聊天列表右边界
    chat_list_x_max: int        # 聊天列表最大 x
    title_y_max: int            # 标题栏底部
    title_x_max_ratio: float    # 标题栏右侧比例上限
    input_y_min: int            # 输入框顶部
    
    # 颜色检测
    self_green: Tuple[int, int, int]
    self_green_tolerance: int
    min_bubble_pixels: int      # 气泡最小像素数
    message_cluster_threshold: int = 80  # 消息按 y 聚类的阈值（像素）
    
    # 昵称识别区域（相对坐标 0.0-1.0）
    nickname_x_min_ratio: float
    nickname_x_max_ratio: float
    nickname_y_offset_min: int
    nickname_y_offset_max: int

# 预配置实例
PROFILE_WECHAT_MAC_1760X1280 = LayoutProfile(
    name="wechat_mac_4.1.8_1760x1280",
    window_width=1760,
    window_height=1280,
    left_boundary=480,
    chat_list_x_max=360,
    title_y_max=95,
    title_x_max_ratio=0.95,
    input_y_min=1040,
    self_green=(176, 240, 167),
    self_green_tolerance=35,
    min_bubble_pixels=1000,
    message_cluster_threshold=80,
    nickname_x_min_ratio=0.30,
    nickname_x_max_ratio=0.45,
    nickname_y_offset_min=15,
    nickname_y_offset_max=50,
)
```

**设计要点**:
- 所有边界值集中在一处
- 支持多 profile，未来可自动检测窗口尺寸匹配对应 profile
- 修改布局阈值时，不需要改动业务代码

---

### 2.5 Layout Parser (L3)

**文件**: `src/layout/layout_parser.py`

**职责**: 把 OCR 元素按 UI 区域分组。输出 `UILayout`。

**关键设计**: **只做分组，不做过滤**。不判断"这是不是消息"。

**接口**:

```python
class LayoutParser:
    def __init__(self, profile: LayoutProfile):
        self.profile = profile
    
    def parse(self, elements: List[OCRTextElement], image_path: str) -> UILayout:
        """
        将 OCR 元素分组为 UI 区域。
        
        Returns:
            UILayout: 包含各区域元素的完整布局描述
        """
        pass

@dataclass
class UILayout:
    """UI 布局分组结果"""
    chat_name: str
    
    # 左侧聊天列表
    chat_list_items: List[ChatListItem]
    
    # 右侧区域分组
    title_elements: List[OCRTextElement]
    input_elements: List[OCRTextElement]
    timestamp_elements: List[OCRTextElement]
    self_bubbles: List[Rect]       # 绿色气泡区域
    message_candidates: List[OCRTextElement]  # 在消息区的所有元素
```

**分组逻辑**:

1. **左右分割**: `x < left_boundary` 为左侧（聊天列表），`x >= left_boundary` 为右侧（聊天内容区）。`chat_list_x_max` 是左侧列表的有效宽度上限（用于过滤聊天列表内的噪点），分割时主边界仍用 `left_boundary`
2. **标题栏**: 右侧中 `y < title_y_max` 且 `x < width * title_x_max_ratio` 的元素（width 从 image_path 对应图片的宽度获取）
3. **输入框**: 右侧中 `y >= input_y_min` 的元素
4. **时间戳**: 匹配预定义正则模式 `TIMESTAMP_PATTERNS` 且位于消息区中央的元素
5. **绿色气泡**: 通过颜色检测（`self_green`）识别，过滤小噪点
6. **消息候选区**: 右侧中排除上述分组后的剩余元素

**时间戳模式**:

```python
TIMESTAMP_PATTERNS = [
    r"^\d{1,2}:\d{2}$",                     # 12:34
    r"^昨天 \d{1,2}:\d{2}$",                 # 昨天 12:34
    r"^星期[一二三四五六日] \d{1,2}:\d{2}$",  # 星期一 12:34
    r"^\d{4}/\d{2}/\d{2}$",                  # 2024/01/15
]
```

**约束**:
- 不调用 LLM
- 不判断 sender_type
- 不做消息合并

**测试策略**:
- 用测试图片验证各区域元素数量
- 验证 `chat_name` 提取正确
- 验证绿色气泡数量和位置正确

---

### 2.6 Message Extractor (L3)

**文件**: `src/message/extractor.py`

**职责**: 从 `UILayout` 中提取结构化消息列表 `List[ChatMessage]`。

**接口**:

```python
class MessageExtractor:
    def __init__(self, profile: LayoutProfile):
        self.profile = profile
    
    def extract(self, layout: UILayout) -> List[ChatMessage]:
        """
        从 UI 布局中提取消息。
        
        逻辑：
        1. 先处理 self_bubbles 内的文本 → SELF 消息
        2. 再处理其他候选文本 → OTHER 消息
        3. 按 y 坐标排序
        
        Returns:
            ChatMessage 列表，按时间顺序排列
        """
        pass
```

**提取规则**:

**自己消息**:
- 文本中心点落在 `self_bubbles` 内的元素
- 同一气泡内的多个文本按 y 排序后合并
- `sender_type = SELF`

**对方消息**:
- 不在任何 self_bubble 内的 `message_candidates`
- 按 y 坐标聚类（间距 < `profile.message_cluster_threshold` 为一组）
- 检查聚类顶部是否有昵称（在 `nickname_x_min_ratio` ~ `nickname_x_max_ratio` 对应区域内）
- `sender_type = OTHER`

**系统消息**:
- 已提前在 Layout 阶段归入 `timestamp_elements`
- 这里不处理

**测试策略**:
- 验证消息数量与气泡数量匹配
- 验证 sender_type 识别正确率
- 验证消息按 y 坐标正确排序

---

### 2.7 Perception Pipeline (L3.5)

**文件**: 
- `src/perception/smart_pipeline.py` — 主力感知管道（本地预判 + API 兜底）
- `src/perception/vision_pipeline.py` — 纯本地 OCR 管道（备用回退）

**职责**: 将 Capture → OCR → Layout → Extract 的完整视觉链路封装为单一接口。对 Bot 层完全隐藏视觉实现细节。

#### SmartPerceptionPipeline（主力）

`SmartPerceptionPipeline` 采用"本地预判 + API 兜底"策略：

1. **本地预判**：先用本地 `VisionPipeline`（OCR + Layout + Extract）提取消息
2. **智能决策**：如果本地预判结果足够（有新消息、内容完整），直接返回
3. **API 兜底**：如果本地预判为空或不完整，调用 qwen3.x-flash 多模态 API 对截图进行端到端识别
4. **结果合并**：将 API 返回的消息与本地预判结果合并去重

**环境变量控制**：
- `USE_MULTIMODAL_OCR=true`（默认）：启用 SmartPerceptionPipeline
- `ALWAYS_USE_API=true`：禁用本地预判，每次 tick 都走 API
- `USE_MULTIMODAL_OCR=false`：回退到纯本地 VisionPipeline

```python
class SmartPerceptionPipeline:
    def __init__(self, profile: LayoutProfile, always_use_api: bool = False):
        self.local_pipeline = VisionPipeline(profile)
        self.api_client = _QwenAPIClient()  # qwen3.x-flash 多模态
        self.always_use_api = always_use_api
    
    def perceive(self) -> Optional[PerceptionResult]:
        """
        执行智能感知链路：
        1. 本地预判 → 如果结果可用则直接返回
        2. API 兜底 → 调用多模态 API 端到端识别
        3. 合并去重 → 返回最终结果
        
        Returns:
            PerceptionResult: 包含结构化消息列表、聊天名、截图路径
            None: 当 Capture 失败时返回 None，由 Bot 层跳过本轮
        """
        pass
```

#### VisionPipeline（备用）

纯本地 OCR 管道，不依赖任何外部 API：

```python
class VisionPipeline:
    def __init__(self, profile: LayoutProfile):
        self.capture = WindowCapture()
        self.ocr = VisionOCREngine()
        self.layout = LayoutParser(profile)
        self.extractor = MessageExtractor(profile)
    
    def perceive(self) -> Optional[PerceptionResult]:
        """
        执行完整视觉链路：截图 → OCR → 布局分组 → 消息提取。
        
        Returns:
            PerceptionResult: 包含结构化消息列表、聊天名、截图路径
            None: 当 Capture 失败（如未找到窗口）时返回 None，由 Bot 层跳过本轮
        """
        pass
```

**设计要点**:
- Bot 层禁止直接操作 `OCRTextElement`、`UILayout`、`CaptureResult`
- `SmartPerceptionPipeline` 是默认感知管道，通过 `run_bot.py` 中的 `_create_perception()` 创建
- `VisionPipeline` 仅在 SmartPerceptionPipeline 初始化失败或环境变量指定时作为回退
- **边界约束**：L3.5 只负责提取，不维护任何跨 tick 状态（如历史消息、去重索引）。"合并去重"仅指同一 tick 内本地结果与 API 结果的去重，不是持久化去重

---

### 2.8 GlobalStore (L4)

**文件**: `src/session/global_store.py`

**职责**: 全局消息存储 —— 管理所有聊天的消息历史、去重、回复状态和持久化。**这是防止循环发送和消息重复处理的关键层。**

**接口**:

```python
class GlobalStore:
    def __init__(self, max_messages: int = 200, state_file: str = "data/global_state.json"):
        self.chats: Dict[str, ChatState] = {}  # chat_name -> ChatState
        self.max_messages = max_messages
        self._state_file = Path(state_file)
    
    def merge_tick(self, chat_name: str, messages: List[ChatMessage]) -> Tuple[ChatState, List[ChatMessage]]:
        """
        合并 tick 检测到的消息，返回 (state, 未回复的消息列表).
        
        去重策略：LCS 序列对齐。
        1. 取历史末尾 50 条作为窗口
        2. 用 _match_single（二值匹配）做 DP 求最长公共子序列
        3. 回溯得到 tick 中匹配 history 的索引集合 matched
        4. matched 的最右端索引 max_matched 之后的 unmatched → 新消息
        5. max_matched 之前的 unmatched → 旧的（嵌在匹配序列中的跳过项，通常是 OCR 抖动）
        6. 如果 matched 为空（完全无序列匹配）→ 回退到逐条 _in_history 检查
        """
        pass
    
    def mark_replied(self, chat_name: str, target_msg: ChatMessage, reply_text: str):
        """标记单条消息已回复。"""
        pass
    
    def get_unreplied(self, chat_name: str) -> List[ChatMessage]:
        """获取某聊天中所有未回复的消息（按时间顺序）"""
        pass
    
    def save(self):
        """保存状态到磁盘（加锁保护读-改-写操作）"""
        pass
```

**去重算法**（`merge_tick` 核心逻辑）：

```python
def _match_single(a: ChatMessage, b: ChatMessage, chat_name: str) -> bool:
    """直接比较两条消息是否匹配（二值：匹配/不匹配）。"""
    # 1. 精确匹配（标准化 sender + 内容 hash）
    if _msg_id(chat_name, a) == _msg_id(chat_name, b):
        return True
    # 2. sender_type 或 message_type 不同 → 不匹配
    if a.sender_type != b.sender_type or a.message_type != b.message_type:
        return False
    # 3. 文字消息：SequenceMatcher >= 0.80
    if a.message_type == "text":
        return difflib.SequenceMatcher(None, _normalize_text(a.text), 
                                       _normalize_text(b.text)).ratio() >= 0.80
    # 4. 图片/表情/混合：2-gram Jaccard >= 0.08
    return _jaccard_2gram(a.image_description, b.image_description) >= 0.08

def _lcs_match(history, tick, chat_name) -> set:
    """LCS 序列对齐：返回 tick 中匹配 history 的索引集合。"""
    m, n = len(history), len(tick)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _match_single(history[i - 1], tick[j - 1], chat_name):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # 回溯找匹配的 tick 索引
    matched = set()
    i, j = m, n
    while i > 0 and j > 0:
        if _match_single(history[i - 1], tick[j - 1], chat_name):
            matched.add(j - 1)
            i -= 1; j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return matched

def merge_tick(self, chat_name, messages):
    # ... tick 内去重 ...
    
    history_window = state.messages[-50:]  # 取最近 50 条
    matched = _lcs_match(history_window, messages, chat_name)
    
    if not matched:
        # 完全无序列匹配，回退逐条检查
        new_messages = [msg for msg in messages if not _in_history(msg)]
    else:
        max_matched = max(matched)
        # max_matched 之前的 unmatched → 旧的（OCR 抖动导致 _match_single 失败）
        # max_matched 之后的 unmatched → 新的
        new_messages = [
            messages[i] for i in range(len(messages))
            if i not in matched and i > max_matched
        ]
    
    # 添加新消息到历史 + 持久化
    for msg in new_messages:
        state.messages.append(msg)
        state._msg_ids.add(_msg_id(chat_name, msg))
    
    # 裁剪旧消息
    if len(state.messages) > self.max_messages:
        removed = state.messages[:-self.max_messages]
        state.messages = state.messages[-self.max_messages:]
        for msg in removed:
            state._msg_ids.discard(_msg_id(chat_name, msg))
    
    unreplied = [msg for msg in state.messages 
                 if not msg.replied and msg.sender_type != SenderType.SELF]
    return state, unreplied
```

**模糊兜底**（`_is_fuzzy_duplicate`）：
- 文字消息：按长度动态阈值（短句 0.90，长句 0.80），用 SequenceMatcher 比较最近 10 条历史
- 图片消息：基于 `image_description` 做 2-gram Jaccard，阈值 0.08
- 跳过 Bot 自己发的消息（避免用 Bot 回复去重新消息）

**设计要点**:
- 去重核心在 GlobalStore，同时承担持久化职责
- LCS 序列对齐优于逐条独立判断：对 OCR/API 描述抖动更鲁棒（如 "说过" vs "said"、emoji 变化）
- `_msg_id` 使用标准化 sender（私聊对方统一为 chat_name，消除 API 昵称识别不稳定）
- 50 条 history_window 平衡了召回率和性能
- 持久化使用"写临时文件 + os.replace"的原子写入，外加 threading.Lock 保护

---

### 2.9 Reply Policy & Generator (L4)

**文件**: 
- `src/reply/policy.py` — 回复决策
- `src/reply/generator.py` — 回复生成

**职责**:
- `policy`: 决定是否回复
- `generator`: 生成回复内容

**接口**:

```python
class ReplyPolicy:
    def should_reply(self, msg: ChatMessage, session: ChatState) -> bool:
        """
        决策逻辑：
        1. 自己消息 → False
        2. 系统消息 → False
        3. 冷却期内同一聊天 → False
        4. 群聊且未@我 → False
        5. 其他 → True
        """
        pass

class ReplyGenerator:
    def __init__(self, llm_client=None, complex_llm_client=None, memory_engine=None):
        """
        Args:
            llm_client: 主 LLM 客户端（deepseek-v4-flash via dashscope）
            complex_llm_client: 复杂任务 LLM 客户端（OpenClaw/Hermes，连接 127.0.0.1:18790）
            memory_engine: 记忆引擎（可选，用于个性化回复）
        """
        self.llm_client = llm_client
        self.complex_llm_client = complex_llm_client
        self.memory_engine = memory_engine
    
    def generate(self, msg: ChatMessage, session: ChatState) -> List[str]:
        """
        调用 LLM 生成回复。
        
        Returns:
            List[str]: 可能包含多条回复（如需要分段发送）
            []: 当不应回复时返回空列表
        
        系统提示词固定：
        - 友好自然
        - 简洁（≤50字）
        - 群聊@时直接回答
        """
        pass
```

**约束**:
- `generator` 只做内容生成，不做发送决策
- 生成失败时返回空列表 `[]`（不回复，不使用兜底话术）
- `openclaw_client.py` 在 LLM 返回空响应时抛出 `RuntimeError`，由 `reply/generator.py` 的 ReplyGenerator 捕获并跳过回复
- ReplyGenerator 支持双模型路由：简单任务走 `llm_client`（OpenClaw/Kimi），复杂任务（带 skill/tool 匹配）走 `complex_llm_client`（Hermes）

---

### 2.10 Action Layer (L4)

**文件**: `src/action/message_sender.py`、`src/action/ui_interactor.py`

**职责**: 执行所有与微信窗口的交互操作，分为两类：
- `MessageSender`：内容输入（文本、图片、文件）
- `UIInteractor`：坐标/UI 操作（点击聊天项、切换聊天）

#### MessageSender

```python
class MessageSender:
    def send(self, text: str) -> ActionResult: ...
    def send_image(self, image_path: str) -> ActionResult: ...
    def send_file(self, file_path: str) -> ActionResult: ...
```

**当前实现（文本）**：基于 AppleScript 的全局键盘事件（`Command+V` 粘贴 + `Return` 发送）。只要微信窗口处于前台激活状态且光标在输入框中，就不需要知道输入框的像素坐标。

```python
class WeChatMessageSender(MessageSender):
    def send(self, text: str) -> ActionResult:
        try:
            # 确保微信窗口在前台，防止消息发到其他应用
            subprocess.run(['osascript', '-e', 'tell application "WeChat" to activate'], timeout=3, capture_output=True)
            time.sleep(0.1)
            
            subprocess.run(['pbcopy'], input=text.encode('utf-8'), timeout=2)
            time.sleep(0.15)
            script = '''
                tell application "System Events"
                    tell process "WeChat"
                        keystroke "v" using command down
                        delay 0.15
                        keystroke return
                    end tell
                end tell
            '''
            subprocess.run(['osascript', '-e', script], timeout=5, capture_output=True)
            return ActionResult(success=True, sent_text=text)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def send_image(self, image_path: str) -> ActionResult:
        """预留：将图片复制到剪贴板后 Command+V 粘贴发送。"""
        pass
    
    def send_file(self, file_path: str) -> ActionResult:
        """预留：拖拽文件到输入框或复制到剪贴板后粘贴发送。"""
        pass
```

**禁忌**:
- 不能用 `keystroke "a" using command down` 这类全选操作（中文 IME 会产生产拼音碎片）
- 不能用 `typewrite` 逐字符输入（同样受 IME 影响）

#### UIInteractor

```python
class UIInteractor:
    def click_chat_item(self, item: ChatListItem) -> bool: ...
    def click_input_box(self) -> bool: ...
```

**职责**：基于坐标进行鼠标点击操作，用于切换聊天或聚焦输入框。

**为什么当前发送文本不需要坐标，但还需要 `UIInteractor`？**
- 当前发送文本只需要键盘（光标已在输入框）
- 切换聊天、发送图片/文件后的聚焦、未来拖拽文件等场景**必须依赖坐标点击**
- `UIInteractor` 由 `VisionPipeline` 输出的 `ChatListItem` / `Rect` 驱动，Bot 层不直接接触坐标

#### WeChatLoginHandler

**文件**: `src/action/login_recovery.py`

**职责**: 当 `WindowCapture` 检测到微信窗口尺寸异常（未登录/浮窗）时，尝试自动恢复。

> **⚠️ 跨层协调器特例**：`login_recovery` 是系统中唯一的**跨层协调器**，它内部需要协调 L2 capture + L2 ocr + L4 action 三层的能力来完成恢复流程。因此它不可避免地会跨越 L2-L4 的边界。作为特例，它被放在 action 层（因为最终手段是 UI 点击），但**不允许任何下层模块反向依赖它**。
>
> 分层约束：
> - `WindowCapture` (L2) **禁止** import `WeChatLoginHandler`。窗口异常时只抛 `WeChatNotReadyError`，由上层处理恢复。
> - `perception` (L3.5) **禁止** import `WeChatLoginHandler`。感知失败时向上抛异常。
> - `WeChatBot` (L5) 统一捕获 `WeChatNotReadyError` → 调用 `WeChatLoginHandler.handle()` → 恢复成功后重试 perception。

**接口**:

```python
class LoginRecoveryStatus(Enum):
    SUCCESS = "success"
    NEEDS_PHONE_CONFIRM = "needs_phone_confirm"
    NEEDS_QRCODE = "needs_qrcode"
    NO_LOGIN_BUTTON = "no_login_button"

class WeChatLoginHandler:
    def __init__(self, capture_output: str = "/tmp/wechat_login_capture.png",
                 login_keywords: List[str] = None,
                 min_effective_width: int = 800,
                 min_effective_height: int = 600): ...
    def handle(self) -> LoginRecoveryResult: ...
```

**恢复流程**:
1. 查找微信窗口并截图
2. OCR 识别：
   - 如果已出现 **"需在手机上完成登录"**，直接返回 `NEEDS_PHONE_CONFIRM`
   - 如果检测到 **"登录" / "进入微信"** 等关键词，计算按钮坐标并尝试点击
3. 点击后等待 8 秒（给手机确认留足时间）
4. 再次检查窗口尺寸：
   - 窗口 ≥ 800×600 → `SUCCESS`
   - 窗口仍小但出现手机确认提示 → `NEEDS_PHONE_CONFIRM`
   - 窗口仍小且无提示 → `NEEDS_QRCODE`（提示用户手动点击或在手机上确认）

**点击实现**: 使用 AppleScript / Quartz / cliclick 做 best-effort 尝试，不保证 100% 成功（受 macOS 辅助功能和微信安全机制影响）。

---

### 2.11 Bot Orchestrator (L5)

**文件**: `src/bot/wechat_bot.py`

**职责**: 主循环编排，把各层串起来。

**接口**:

```python
class WeChatBot:
    def __init__(self, profile: LayoutProfile, on_message: Optional[Callable] = None):
        # Bot 层只依赖感知管道，禁止直接持有 Capture/OCR/Layout/Extractor
        self.perception = SmartPerceptionPipeline(profile)  # 主力：本地预判 + API 兜底
        # 回退：感知管道初始化失败时使用纯本地 VisionPipeline
        # self.perception = VisionPipeline(profile)
        self.global_store = GlobalStore()
        self.policy = ReplyPolicy()
        self.generator = ReplyGenerator(
            llm_client=DashscopeClient(model="deepseek-v4-flash"),     # 主模型
            complex_llm_client=OpenClawClient(url="http://127.0.0.1:18790"),  # 复杂任务
        )
        self.sender = WeChatMessageSender()
        self.on_message = on_message  # 预留：外部系统集成回调
        self.running = False
    
    def tick(self) -> None:
        """执行一轮：感知 → 去重 → 决策 → 回复"""
        pass
    
    def run_auto(self, interval: float = 5.0) -> None:
        while self.running:
            self.tick()
            time.sleep(interval)
    
    def _get_session(self, chat_name: str) -> ChatState:
        """获取或创建指定聊天的会话对象。"""
        pass
    
    def send_to_chat(self, chat_name: str, text: str) -> ActionResult:
        """预留：外部系统调用此接口主动发消息到指定聊天。"""
        pass
```

**主循环伪代码**:

```python
def tick(self) -> None:
    # 所有视觉细节对 Bot 隐藏，统一走 Pipeline
    result = self.perception.perceive()
    if result is None:
        # 未找到窗口或截图失败，跳过本轮
        return
    
    messages = result.messages
    chat_name = result.chat_name
    
    state, unreplied = self.global_store.merge_tick(chat_name, messages)
    
    if not unreplied:
        return
    
    # 推送新消息给外部系统（如 OpenClaw）
    for msg in unreplied:
        if self.on_message:
            self.on_message(msg, state)
    
    latest = unreplied[-1]
    should_send = self.policy.should_reply(latest, state)
    
    if should_send:
        reply = self.generator.generate(latest, state)
        if reply:
            action_result = self.sender.send(reply)
            if action_result.success:
                self.global_store.mark_replied(chat_name, latest, reply)
```

**运行一次的数据流**:

```
[Capture]        screenshot.png
    ↓
[OCR]            List<OCRTextElement>
    ↓
[LayoutParser]   UILayout
    ↓
[Extractor]      List<ChatMessage>
    ↓
[Session]        过滤为 new_messages
    ↓
[ReplyPolicy]    should_reply ?
    ↓
[Generator]      reply_text
    ↓
[Sender]         ActionResult
    ↓
[Session]        record_sent(reply)
```

---

### 2.12 Logging (L4)

**文件**: `src/logging/bot_logger.py`

**职责**: 记录 Bot 运行期事件，输出到 `execution.jsonl`。

**接口**:

```python
class BotLogger:
    def __init__(self, logs_dir: str = None, max_bytes: int = 5*1024*1024, backup_count: int = 3) -> None: ...
    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str, exc_info: bool = False) -> None: ...
    def critical(self, msg: str) -> None: ...
    def log_tick_start(self, tick_id: int, interval: float) -> None: ...
    def log_capture(self, tick_id: int, success: bool, window_info: dict = None, error: str = None) -> None: ...
    def log_ocr(self, tick_id: int, element_count: int, duration_ms: float, sample_texts: List[str]) -> None: ...
    def log_layout(self, tick_id: int, chat_name: str, title_elem_count: int, input_elem_count: int, timestamp_elem_count: int, self_bubble_count: int, message_candidate_count: int) -> None: ...
    def log_messages(self, tick_id: int, total_messages: int, new_messages: int, message_details: List[dict]) -> None: ...
    def log_decision(self, tick_id: int, should_reply: bool, reason: str, latest_text: str, reply_text: str = None, extra: dict = None) -> None: ...
    def log_send(self, tick_id: int, success: bool, text: str, error: str = None) -> None: ...
    def log_exception(self, tick_id: int, phase: str, exc: Exception) -> None: ...
    def log_stats(self, tick_id: int, stats: dict) -> None: ...
```

**设计要点**:
- 采用结构化 JSONL 格式，便于后续查询和分析
- 日志目录默认在 `logs/`，支持自动轮转

---

### 2.13 Storage (L4)

> ⚠️ **当前状态**: `src/storage/` 目录尚未创建，持久化功能由 `GlobalStore`（`src/session/global_store.py`）统一承担。以下接口为设计目标，实际实现以 `GlobalStore` 为准。

**目标文件**: `src/storage/chat_history.py`（待拆分）

**职责**: 持久化聊天历史记录，按 `chat_name` 分片存储。

**接口**:

```python
@dataclass
class HistoryRecord:
    text: str
    sender: str
    sender_type: str
    chat_name: str
    is_at_me: bool = False
    timestamp: str = ""
    message_hash: str = ""
    confidence: float = 0.0
    bubble_y: int = 0          # 用于位置关联和回声检测
    source: str = "ocr"
    tick_id: int = 0
    screenshot_path: str = ""

class ChatHistory:
    def __init__(self, storage_dir: str = None) -> None: ...
    def append_messages(self, chat_name: str, messages: List[ChatMessage], tick_id: int = 0, screenshot_path: str = "") -> List[HistoryRecord]: ...
    def get_messages(self, chat_name: Optional[str] = None, since: datetime = None, until: datetime = None, limit: int = 500) -> List[HistoryRecord]: ...
    def get_recent_chats(self, hours: float = 24.0, limit: int = 100) -> Dict[str, List[HistoryRecord]]: ...
    def get_last_message(self, chat_name: str) -> Optional[HistoryRecord]: ...
    def get_stats(self) -> dict: ...
    def export_chat(self, chat_name: str, output_path: str = None) -> str: ...
```

**设计要点**:
- 按 `chat_name` 分片为独立 jsonl 文件，避免单文件过大
- 不实现去重逻辑，去重由 `ChatState` 负责
- 详细设计见 `LOGGING_DESIGN.md`

---

## 四、关键设计决策

### 4.1 为什么去重放在 Session 而不是 Storage？

- **Storage** 负责持久化，关心的是"这条消息要不要存"
- **Session** 负责业务状态，关心的是"这条消息要不要回复"
- 循环发送的根因是"业务决策错误"，不是"存储错误"

### 4.2 为什么 LayoutParser 不做过滤？

- 时间戳是不是"消息"，取决于业务定义
- Layout 只负责"这是时间戳元素"，不决定"要不要忽略"
- 这样 MessageExtractor 可以灵活处理（比如某些场景需要保留时间戳）

### 4.3 为什么用 `ActionResult` 而不是 `last_reply_content` 字符串？

- 字符串匹配只能判断内容是否相同，无法记录发送时间和上下文
- `ActionResult` 包含 `sent_at`，支持基于时间窗口的回声检测
- 聊天滚动时 Y 坐标不可靠，因此去重放弃了坐标匹配，改用**窗口指纹 + 上下文序列**判断重复视图
- `ActionResult` 列表支持连续发送多条消息后的回声检测

### 4.4 为什么当前只支持单聊天循环？

- macOS 微信**没有稳定的原生快捷键**用于在不同聊天间切换（`Command+F` 搜索结果不可靠，上下箭头需要已知列表位置）
- 当前 `MessageSender` 只依赖键盘事件，不需要坐标；但切换聊天必须依赖**坐标点击**
- 因此多聊天支持需要新增 `UIInteractor`（坐标点击）+ 扩展 `VisionPipeline` 提供聊天列表坐标，属于明确的下一阶段扩展点

---

## 五、扩展点

### 5.1 支持其他 IM 软件

只需新增：
1. `LayoutProfile`（钉钉/飞书的布局配置）
2. `MessageSender`（不同软件的发送方式）

Capture、OCR、Session、Reply 全部复用。

### 5.2 支持多分辨率自动适配

```python
class ProfileSelector:
    def select(self, window_rect: Rect) -> LayoutProfile:
        """根据窗口尺寸自动匹配最接近的 profile"""
```

### 5.3 支持图片/语音消息识别

在 `MessageExtractor` 中增加：
- 图片检测（通过 OCR 的 `[图片]` 文本标记）
- 语音检测（通过 `[语音]` 文本标记 + 时长元素）

### 5.4 支持多聊天切换回复

当前架构仅支持"单聊天循环"。要支持多聊天，需要：

1. **Action 层新增 `UIInteractor`**：
   ```python
   class UIInteractor:
       def click_chat_item(self, item: ChatListItem) -> bool: ...
   ```
2. **Bot 层调整主循环**：
   - 先 `perception.perceive()` 获取左侧聊天列表（`result.chat_list_items`）
   - Bot 层决策后，通过 `ui_interactor.click_chat_item(item)` 切换到目标聊天
   - 再执行一次 `perception.perceive()` 获取该聊天最新消息，然后回复

**原则**：坐标操作完全封装在 Action `UIInteractor` 中；`VisionPipeline` 只负责感知，不承担任何交互动作。

### 5.5 支持 Bot 层外部集成（如 OpenClaw / MCP）

当前 `WeChatBot` 是自包含的闭环系统（`tick()` → `run_auto()`），没有外部集成接口。

要对接外部 Agent 系统（如 OpenClaw），需在 L5 Bot 层增加以下扩展：

1. **事件出口：`on_message` 回调**
   ```python
   class WeChatBot:
       def __init__(self, ..., on_message=None):
           self.on_message = on_message  # 推送新消息给外部系统
   ```
   在 `tick()` 中识别到 `new_messages` 后，通过 `on_message(msg, session)` 把事件流推出去。

2. **外部入口：`send_to_chat()` 主动发送**
   ```python
   def send_to_chat(self, chat_name: str, text: str) -> ActionResult:
       """外部系统调用此接口主动发消息到指定聊天。"""
   ```

3. **替换回复生成器**
   `ReplyGenerator` 的接口已经是抽象的，可通过注入不同的 LLM 客户端实现：
   - `llm_client`: 主模型（deepseek-v4-flash via dashscope，默认）
   - `complex_llm_client`: 复杂任务模型（OpenClaw → Kimi Code，连接 127.0.0.1:18790）
   - `memory_engine`: 可选，用于个性化回复

**更远的未来**：可把 `WeChatBot` 包装为 MCP Server，提供工具 `send_wechat_message` 和资源 `wechat://recent_messages/{chat_name}`。

---

## 六、错误处理策略

| 层级 | 错误场景 | 处理方式 |
|------|---------|---------|
| Capture | 未找到窗口 | 跳过本轮，sleep interval |
| OCR | 无文本识别 | 继续执行，返回空列表 |
| Layout | 无法提取 chat_name | 跳过回复，尝试切换到未读聊天 |
| Session | 检测到回声消息 | `filter_new()` 过滤掉，不回复 |
| Generator | LLM 调用失败 | 返回空列表 `[]`，不回复（无兜底话术） |
| Sender | AppleScript 失败 | 返回 `ActionResult(success=False)`，不重试 |

**原则**: 任何一层失败都不应该让整个系统崩溃，应该优雅降级。

---

## 七、文件结构（目标）

```
wechat-mac-rpa/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py              # Point, Rect, ChatMessage, etc.
│   ├── capture/
│   │   ├── __init__.py
│   │   └── window_capture.py
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── vision_ocr.py
│   ├── layout/
│   │   ├── __init__.py
│   │   ├── profile.py
│   │   └── layout_parser.py
│   ├── message/
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── session/
│   │   ├── __init__.py
│   │   └── global_store.py
│   ├── reply/
│   │   ├── __init__.py
│   │   ├── policy.py
│   │   └── generator.py
│   ├── action/
│   │   ├── __init__.py
│   │   ├── message_sender.py
│   │   └── ui_interactor.py
│   ├── perception/
│   │   ├── __init__.py
│   │   └── vision_pipeline.py   # 视觉感知管道
│   ├── bot/
│   │   ├── __init__.py
│   │   └── wechat_bot.py        # 主循环
│   ├── logging/
│   │   ├── __init__.py
│   │   └── bot_logger.py
│   └── storage/
│       ├── __init__.py
│       └── chat_history.py
├── tests/
│   ├── fixtures/
│   │   ├── errors/              # 错误用例库
│   │   └── current.png
│   ├── test_capture.py
│   ├── test_ocr.py
│   ├── test_layout.py
│   ├── test_message.py
│   ├── test_session.py
│   ├── test_sender.py
│   ├── test_bot.py
│   ├── test_logging.py          # BotLogger 测试
│   └── test_chat_history.py     # ChatHistory 测试
├── ARCHITECTURE.md              # 本文档
├── LOGGING_DESIGN.md            # 日志与历史记录设计
└── LESSONS_LEARNED.md
```

---

**最后更新**: 2026-05-03
**文档状态**: 已覆盖全部模块，重构已完成
**状态**: ✅ L1-L5 模块化架构已全部落地。核心架构更新：SmartPerceptionPipeline（本地预判+API兜底）、双模型架构（deepseek-v4-flash + OpenClaw/Hermes）、兜底话术废弃（返回空列表）
