# WeFlow HTTP API 集成技术方案

> 版本: v1.0  
> 日期: 2026-05-07  
> 目标: 在完全保留现有 OCR 架构的前提下，集成 WeFlow HTTP API 作为可选数据源，支持随时切换

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [现有架构回顾](#2-现有架构回顾)
3. [WeFlow API 能力边界](#3-weflow-api-能力边界)
4. [集成架构设计](#4-集成架构设计)
5. [模块级改造方案](#5-模块级改造方案)
6. [数据流设计](#6-数据流设计)
7. [配置与切换机制](#7-配置与切换机制)
8. [关键接口设计](#8-关键接口设计)
9. [兼容性与兜底策略](#9-兼容性与兜底策略)
10. [实施计划](#10-实施计划)
11. [风险清单](#11-风险清单)

---

## 1. 背景与目标

### 1.1 现状痛点

当前 RPA 基于截图 + OCR + Vision API 读取微信消息，存在以下问题：

| 痛点 | 影响 |
|------|------|
| OCR 文本准确率 ~83% | 群聊 sender 识别错误、短消息误识别 |
| 每次 tick 可能调用 qwen3.6-flash API | 成本高，延迟大（1-3s）|
| 消息深度仅 5~10 条（窗口可见） | Bot 上下文浅，无法参考历史对话 |
| emoji/特殊符号易丢失 | 消息内容不完整 |
| 复杂去重算法（LCS + 模糊匹配） | 代码复杂，维护成本高 |

### 1.2 WeFlow 能力

WeFlow 0.26.10 preview 提供本地 HTTP API（`127.0.0.1:5031`）：

- `GET /api/v1/contacts` — 19 个联系人/群聊列表（含昵称、wxid、类型）
- `GET /api/v1/messages?talker=xxx&limit=N&offset=N` — 结构化消息（含 `localId`, `isSend`, `senderUsername`, `content`, `createTime`）

### 1.3 目标

1. **零删除**：现有 OCR 代码（Capture/OCR/Layout/Extractor/SmartPipeline）完全保留
2. **零侵入**：现有代码不加任何 WeFlow 相关逻辑，通过配置切换
3. **随时切换**：一个环境变量即可在 OCR / WeFlow / Hybrid 三种模式间切换
4. **历史初始化**：WeFlow 模式下，首次打开聊天时可从数据库拉取历史消息作为上下文
5. **自动兜底**：WeFlow API 异常时自动 fallback 到 OCR，Bot 不中断

---

## 2. 现有架构回顾

### 2.1 模块职责

```
src/
├── capture/
│   └── window_capture.py       # L1: 截图（Quartz + screencapture）
├── ocr/
│   └── vision_ocr.py           # L2: OCR（macOS Vision 框架）
├── layout/
│   ├── layout_parser.py        # L3: 布局分组（气泡/列表/输入框）
│   └── profile.py              # L3: 布局配置文件
├── message/
│   └── extractor.py            # L3.5: 从布局提取消息
├── perception/
│   ├── vision_pipeline.py      # L3.5: 纯 OCR 流程（ perceive() → PerceptionResult ）
│   └── smart_pipeline.py       # L3.5: 智能管道（本地预判 + qwen3.6-flash 兜底）
├── session/
│   └── global_store.py         # L4: 跨 tick 去重 + 消息历史 + 回复状态
├── reply/
│   ├── policy.py               # L4: 回复策略（群聊 @ 检测等）
│   ├── generator.py            # L4: Prompt 构建 + LLM 调用
│   └── session_memory.py       # L4: 短期记忆（工具缓存）
├── memory/
│   └── engine.py               # L4: 长期记忆（LLM Wiki）
├── action/
│   ├── message_sender.py       # L5: 发送消息
│   ├── chat_list_clicker.py    # L5: 点击聊天列表
│   └── ui_interactor.py        # L5: UI 交互
└── bot/
    └── wechat_bot.py           # L6: 主循环编排（tick()）
```

### 2.2 当前 tick 流程

```
tick()
  ├── perceive() ──→ 截图 → OCR → Layout → Extract → PerceptionResult
  │                    ↑ SmartPipeline 在此做本地预判 + qwen3.6-flash 兜底
  │
  ├── merge_tick() ──→ GlobalStore 去重 → (ChatState, unreplied_messages)
  │                    ↑ 复杂算法：LCS + SequenceMatcher + Jaccard
  │
  ├── should_reply() ──→ ReplyPolicy 判断是否需要回复
  │
  ├── generate() ──→ ReplyGenerator 构建 prompt → LLM 生成回复
  │                    ↑ all_messages 只有截图里的 5~10 条
  │
  └── send() ──→ MessageSender 发送消息
```

### 2.3 PerceptionResult 结构

```python
@dataclass
class PerceptionResult:
    chat_name: str              # 当前聊天名称
    messages: List[ChatMessage] # 识别出的消息列表
    chat_list_items: List[ChatListItem]  # 左侧聊天列表
    screenshot_path: str        # 截图路径
    debug_info: dict            # 中间调试信息
```

这是**感知层与上层唯一的接口**。任何数据源（OCR/WeFlow）都必须输出此结构。

---

## 3. WeFlow API 能力边界

### 3.1 已验证接口

```
GET /api/v1/contacts?access_token={token}
  → { success: true, contacts: [ { username, displayName, nickname, type, alias, region } ] }

GET /api/v1/messages?access_token={token}&talker={wxid}&limit={N}&offset={M}
  → {
      success: true,
      talker: "wxid_xxx",
      count: 20,
      hasMore: true,
      messages: [
        {
          localId: 25,              # 数据库自增 ID，唯一且稳定
          serverId: "3818097325769990171",
          localType: 1,             # 1=text, 33=小程序, 66=名片, ...
          createTime: 1777792692,   # 秒级时间戳
          sortSeq: 1777792692000,
          isSend: 1,                # 1=自己, 0=对方
          senderUsername: "wxid_example_self",
          content: "消息文本",
          rawContent: "原始内容",
          parsedContent: ""
        }
      ]
    }
```

### 3.2 WeFlow 不能做什么

| 能力 | WeFlow 支持？ | 说明 |
|------|-------------|------|
| 读取消息内容 | ✅ | 文本/XML 均可 |
| 判断 isSend | ✅ | `isSend` 字段 |
| 获取 sender wxid | ✅ | `senderUsername` |
| 获取 sender 昵称 | ❌ | 只返回 wxid，昵称需查 contacts |
| 知道"当前打开的是谁" | ❌ | 不提供窗口状态 |
| 检测未读红点 | ❌ | 不提供 UI 状态 |
| 发送消息 | ❌ | 只读 API |
| 点击聊天列表 | ❌ | 无操作 API |
| 标记已回复 | ❌ | 微信数据库无此字段 |
| 多聊天同时监控 | ⚠️ 需轮询 | 无推送/通知机制 |

**结论**：WeFlow 是**纯消息数据源**，无法替代 Capture/Action/GlobalStore 的核心功能。

---

## 4. 集成架构设计

### 4.1 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                     抽象层：PerceptionResult                  │
│                  （与数据源无关的统一输出）                     │
└─────────────────────────────────────────────────────────────┘
                              ▲
           ┌──────────────────┴──────────────────┐
           │                                     │
    ┌──────▼──────┐                     ┌────────▼────────┐
    │ OCR Pipeline │                     │ WeFlow Pipeline │
    │  (现有代码)   │                     │   (新增代码)    │
    │              │                     │                 │
    │ Capture      │                     │ Capture (轻量)  │
    │   → OCR      │                     │   → 标题识别    │
    │   → Layout   │                     │                 │
    │   → Extract  │                     │ WeFlow API      │
    │   → API兜底  │                     │   → contacts    │
    │              │                     │   → messages    │
    │ (800+行代码) │                     │   → 历史初始化   │
    │   完全保留   │                     │                 │
    └──────────────┘                     └─────────────────┘
```

### 4.2 运行模式

通过环境变量 `WEFLOW_MODE` 控制：

| 模式 | 环境变量 | 消息来源 | 去重方式 | 适用场景 |
|------|---------|---------|---------|---------|
| `ocr` | 默认 / `WEFLOW_MODE=ocr` | SmartPipeline (截图+OCR+API) | 模糊匹配 (LCS) | WeFlow 不可用时 |
| `weflow` | `WEFLOW_MODE=weflow` | WeFlow API | `localId` 精确匹配 | 稳定运行期 |
| `hybrid` | `WEFLOW_MODE=hybrid` | WeFlow 优先，异常 fallback OCR | `localId` + 模糊匹配 | 过渡期/测试期 |

切换方式：
```bash
# 零代码改动切换
export WEFLOW_MODE=weflow
python run_bot.py

# 切回 OCR
export WEFLOW_MODE=ocr
python run_bot.py
```

---

## 5. 模块级改造方案

### 5.1 新增模块

#### 5.1.1 `perception/weflow_client.py`（已就绪 ✅）

- **状态**：已编写并测试通过
- **功能**：WeFlow HTTP API 客户端，封装 contacts/messages 接口
- **接口**：
  - `get_contacts()` → `List[WeFlowContact]`
  - `get_messages(talker, limit, offset)` → `(List[WeFlowMessage], has_more)`
  - `get_latest_messages(talker, limit)` → `List[WeFlowMessage]`
  - `health_check()` → `bool`

#### 5.1.2 `perception/weflow_pipeline.py`（新增）

- **功能**：WeFlow 模式下的感知管道，输出标准 `PerceptionResult`
- **职责**：
  1. 截图（仅用于识别当前聊天标题）
  2. 轻量 OCR（只识别标题栏 + 聊天列表，不识别消息区）
  3. 标题 → talker 映射（通过 contacts 缓存查 wxid）
  4. WeFlow API 拉取消息（默认 50 条，支持历史初始化）
  5. 转换为 `ChatMessage` 列表
  6. 组装 `PerceptionResult`

- **伪代码**：
```python
class WeFlowPipeline:
    def __init__(self, profile, weflow_client=None):
        self.capture = WindowCapture()          # 复用现有截图模块
        self.ocr = VisionOCREngine()            # 复用现有 OCR
        self.layout = LayoutParser(profile)     # 复用现有 Layout
        self.weflow = weflow_client or WeFlowClient()
        self._contacts_cache: List[WeFlowContact] = []
        self._contacts_ts: float = 0
        self._history_initialized: Set[str] = set()  # 已初始化历史的聊天

    def perceive(self) -> PerceptionResult:
        # Step 1: 截图
        capture_result = self.capture.capture()
        image_path = capture_result.image_path

        # Step 2: 轻量 OCR（标题 + 聊天列表）
        elements = self.ocr.recognize(image_path)
        layout = self.layout.parse(elements, image_path)

        chat_name = layout.chat_name
        if not chat_name:
            return None  # 未识别到聊天标题

        # Step 3: 标题 → talker 映射
        talker = self._resolve_talker(chat_name)
        if not talker:
            return None  # 找不到对应的 wxid

        # Step 4: WeFlow API 拉消息
        is_first_open = chat_name not in self._history_initialized
        limit = 100 if is_first_open else 20
        messages = self.weflow.get_messages(talker, limit=limit)

        if is_first_open:
            self._history_initialized.add(chat_name)

        # Step 5: 转换为 ChatMessage
        chat_messages = self._convert_to_chat_messages(messages, talker)

        # Step 6: 组装 PerceptionResult
        return PerceptionResult(
            chat_name=chat_name,
            messages=chat_messages,
            chat_list_items=layout.chat_list_items,
            screenshot_path=image_path,
            debug_info={"source": "weflow", "talker": talker, "limit": limit},
        )
```

#### 5.1.3 `session/global_store.py` — 增加 WeFlow 去重分支（修改）

- **改动点**：`merge_tick()` 方法增加模式判断
- **现有代码**：完全不改动，整体下移为 `_merge_tick_ocr()`
- **新增代码**：`_merge_tick_weflow()` — 基于 `localId` 的精确去重

```python
def merge_tick(self, chat_name, messages, mode="ocr") -> Tuple[ChatState, List[ChatMessage]]:
    if chat_name not in self.chats:
        self.chats[chat_name] = ChatState(...)

    if mode == "weflow" and messages and hasattr(messages[0], "local_id"):
        new_messages = self._merge_tick_weflow(chat_name, messages)
    else:
        new_messages = self._merge_tick_ocr(chat_name, messages)  # 原有代码

    # 后续逻辑不变（添加消息、裁剪、收集未读）
    ...

def _merge_tick_weflow(self, chat_name, messages) -> List[ChatMessage]:
    """WeFlow 精确去重：基于 localId"""
    state = self.chats[chat_name]
    seen_ids = {m.local_id for m in state.messages if hasattr(m, "local_id")}
    new_messages = [m for m in messages if getattr(m, "local_id", None) not in seen_ids]
    return new_messages
```

### 5.2 修改模块

#### 5.2.1 `perception/smart_pipeline.py` — 增加分流入口（修改）

- **改动量**：~20 行
- **策略**：`__init__` 末尾增加 WeFlow 初始化，`perceive()` 开头增加模式分流
- **现有代码**：完全不动，整体包装

```python
class SmartPerceptionPipeline:
    def __init__(self, ...):
        # ===== 原有代码完全保留 =====
        self.capture = WindowCapture(...)
        self.ocr = VisionOCREngine()
        self.layout = LayoutParser(profile)
        # ... 原有所有代码 ...

        # ===== 新增：WeFlow 初始化（5 行）=====
        self._weflow_mode = os.getenv("WEFLOW_MODE", "ocr")
        self._weflow_pipeline = None
        if self._weflow_mode in ("weflow", "hybrid"):
            try:
                from .weflow_pipeline import WeFlowPipeline
                self._weflow_pipeline = WeFlowPipeline(profile)
            except Exception as e:
                _logger.warning(f"WeFlow 初始化失败，降级为 OCR: {e}")
                self._weflow_mode = "ocr"

    def perceive(self) -> Optional[PerceptionResult]:
        # ===== 新增：模式分流（8 行）=====
        if self._weflow_mode in ("weflow", "hybrid") and self._weflow_pipeline:
            try:
                result = self._weflow_pipeline.perceive()
                if result is not None:
                    return result
                if self._weflow_mode == "weflow":
                    return None
                # hybrid 模式：WeFlow 失败则继续走 OCR
            except Exception as e:
                _logger.warning(f"WeFlow perceive 失败: {e}")
                if self._weflow_mode == "weflow":
                    return None

        # ===== 原有 OCR 代码完全不动 =====
        return self._perceive_ocr()

    def _perceive_ocr(self) -> Optional[PerceptionResult]:
        """原有 perceive() 代码整体下移，零改动"""
        # ... 原有 200+ 行代码原封不动 ...
```

#### 5.2.2 `bot/wechat_bot.py` — 传递模式参数（修改）

- **改动量**：~5 行
- **改动点**：`tick()` 中调用 `merge_tick()` 时传入模式参数

```python
def tick(self) -> None:
    # ... 原有代码 ...
    state, unreplied = self.global_store.merge_tick(
        chat_name, messages,
        mode=os.getenv("WEFLOW_MODE", "ocr")  # 新增参数
    )
    # ... 原有代码完全不变 ...
```

### 5.3 完全保留的模块（零改动）

| 模块 | 行数 | 保留原因 |
|------|------|---------|
| `capture/window_capture.py` | 277 | WeFlow 仍需截图确认窗口状态 |
| `ocr/vision_ocr.py` | 158 | WeFlow 仍需轻量 OCR 识别标题 |
| `layout/layout_parser.py` | 373 | WeFlow 仍需布局分组识别标题/列表 |
| `layout/profile.py` | — | 布局配置 |
| `message/extractor.py` | 264 | OCR 模式仍需要 |
| `reply/policy.py` | — | 策略与数据源无关 |
| `reply/generator.py` | 789 | 只消费 PerceptionResult，不感知来源 |
| `reply/session_memory.py` | 172 | 工具缓存与消息源无关 |
| `memory/engine.py` | 445 | 记忆更新逻辑不变 |
| `action/*.py` | ~450 | WeFlow 无操作 API |
| `models/base.py` | — | 数据模型 |
| `tools/*.py` | — | 工具注册 |
| `tests/*.py` | — | 测试用例 |

---

## 6. 数据流设计

### 6.1 OCR 模式（现有流程，完全保留）

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Capture │───→│   OCR   │───→│ Layout  │───→│ Extract │───→│  Global │
│ (截图)  │    │ (识别)  │    │ (分组)  │    │ (提消息)│    │ (去重)  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    │  LCS    │
                                                             │ 模糊匹配│
                                                             └────┬────┘
                                                                  │
                                                             ┌────▼────┐
                                                             │  Policy │
                                                             │ Generator│
                                                             │  Action │
                                                             └─────────┘
```

### 6.2 WeFlow 模式（新增流程）

```
┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐
│ Capture │───→│ 轻量 OCR    │───→│ 标题→talker │───→│ WeFlow  │
│ (截图)  │    │ (标题+列表) │    │  映射查询   │    │  API    │
└─────────┘    └─────────────┘    └─────────────┘    │ get_msgs│
                                                      └────┬────┘
                                                           │
                                                      ┌────▼────┐
                                                      │  Global │
                                                      │ localId │
                                                      │ 精确去重│
                                                      └────┬────┘
                                                           │
                                                      ┌────▼────┐
                                                      │  Policy │
                                                      │ Generator│
                                                      │  Action │
                                                      └─────────┘
```

### 6.3 关键差异对比

| 环节 | OCR 模式 | WeFlow 模式 |
|------|---------|------------|
| 截图范围 | 全窗口 | 全窗口（但只解析标题/列表）|
| OCR 调用 | 完整识别（含消息区） | 轻量识别（仅标题+列表） |
| 消息来源 | Extractor 从布局提取 | WeFlow API 从数据库读取 |
| 消息准确率 | ~83% | 100% |
| 消息深度 | 5~10 条 | 50~100 条（可配置） |
| 去重算法 | LCS + SequenceMatcher + Jaccard | `localId` 集合判断 |
| API 成本 | 可能调用 qwen3.6-flash | 0（本地 HTTP） |
| tick 延迟 | 1~5s（含 API 调用） | <500ms（纯本地） |

---

## 7. 配置与切换机制

### 7.1 环境变量

```bash
# 模式切换
export WEFLOW_MODE=ocr      # 纯 OCR（默认）
export WEFLOW_MODE=weflow   # 纯 WeFlow
export WEFLOW_MODE=hybrid   # WeFlow 优先，异常 fallback OCR

# WeFlow 连接配置（可选，有默认值）
export WEFLOW_HOST=127.0.0.1
export WEFLOW_PORT=5031
export WEFLOW_TOKEN=weflow_token_123

# WeFlow 行为配置
export WEFLOW_HISTORY_LIMIT=100   # 首次打开聊天拉取历史条数
export WEFLOW_TICK_LIMIT=20       # 普通 tick 拉取条数
```

### 7.2 配置文件（可选）

```json
// data/weflow_config.json
{
  "mode": "hybrid",
  "host": "127.0.0.1",
  "port": 5031,
  "token": "weflow_token_123",
  "history_limit": 100,
  "tick_limit": 20,
  "fallback_on_error": true,
  "fallback_on_empty": true
}
```

### 7.3 运行时切换

```python
# 运行时查询当前模式
from src.perception import get_perception_mode

mode = get_perception_mode()  # "ocr" | "weflow" | "hybrid"
```

---

## 8. 关键接口设计

### 8.1 WeFlowPipeline 完整接口

```python
class WeFlowPipeline:
    """WeFlow 感知管道，与 VisionPipeline/SmartPipeline 接口兼容"""

    def __init__(
        self,
        profile: LayoutProfile,
        weflow_client: Optional[WeFlowClient] = None,
        history_limit: int = 100,
        tick_limit: int = 20,
    ):
        ...

    def perceive(self) -> Optional[PerceptionResult]:
        """主入口，输出标准 PerceptionResult"""
        ...

    def health_check(self) -> bool:
        """WeFlow API 是否可用"""
        ...

    def _resolve_talker(self, chat_name: str) -> Optional[str]:
        """聊天名 → wxid 映射（含群聊名→@chatroom 解析）"""
        ...

    def _convert_to_chat_messages(
        self,
        weflow_messages: List[WeFlowMessage],
        talker: str,
    ) -> List[ChatMessage]:
        """WeFlowMessage → ChatMessage（兼容现有模型）"""
        ...

    def _enrich_sender_name(
        self,
        message: WeFlowMessage,
        contact: WeFlowContact,
    ) -> str:
        """wxid → 昵称映射（群聊场景）"""
        ...
```

### 8.2 ChatMessage 模型扩展（可选）

```python
# models/base.py — 增加可选字段，不影响现有代码
@dataclass
class ChatMessage:
    text: str
    sender: str
    sender_type: SenderType
    chat_name: str = ""
    is_at_me: bool = False
    replied: bool = False
    # ... 现有字段 ...

    # ===== 新增：WeFlow 专属字段（可选，不影响 OCR 流程）=====
    local_id: Optional[int] = None      # WeFlow localId，用于精确去重
    server_id: Optional[str] = None     # WeFlow serverId
    create_time: Optional[int] = None   # 秒级时间戳
    raw_type: Optional[int] = None      # WeFlow localType
```

### 8.3 GlobalStore 接口扩展

```python
class GlobalStore:
    # ===== 现有方法完全保留 =====
    def merge_tick(self, chat_name, messages, mode="ocr") -> Tuple[ChatState, List[ChatMessage]]:
        ...

    # ===== 新增：WeFlow 精确去重 =====
    def _merge_tick_weflow(self, chat_name, messages) -> List[ChatMessage]:
        """基于 localId 的精确去重"""
        ...

    # ===== 新增：历史消息批量注入 =====
    def inject_history(self, chat_name, messages: List[ChatMessage]):
        """从 WeFlow 批量注入历史消息（首次打开聊天时）"""
        ...
```

---

## 9. 兼容性与兜底策略

### 9.1 三层兜底

```
Layer 1: WeFlow API 调用成功 → 使用 WeFlow 数据
    ↓ 失败/超时
Layer 2: hybrid 模式 → fallback 到 OCR 流程
    ↓ OCR 也失败
Layer 3: 返回 None → tick 跳过，等待下一个 tick
```

### 9.2 异常场景处理

| 异常场景 | 处理策略 |
|---------|---------|
| WeFlow 进程未启动 | hybrid 模式 fallback OCR；weflow 模式返回 None |
| HTTP 连接超时（3s）| fallback OCR |
| API 返回空消息 | fallback OCR（可能数据库未同步） |
| API 返回格式异常 | fallback OCR + 日志告警 |
| 标题识别失败（找不到 talker）| fallback OCR |
| 群聊名映射失败 | fallback OCR |

### 9.3 数据兼容性

- `PerceptionResult` 结构不变 → `ReplyPolicy` / `ReplyGenerator` / `Action` 零改动
- `ChatMessage` 增加可选字段 → 现有代码 `hasattr` 判断，不报错
- `GlobalStore` 增加模式参数 → 默认 `mode="ocr"`，现有调用方式不变

---

## 10. 实施计划

### Phase 1: 基础设施（Day 1）

| 任务 | 文件 | 状态 |
|------|------|------|
| WeFlowClient 开发 | `weflow_client.py` | ✅ 已完成 |
| 群聊消息格式验证 | 测试脚本 | 📝 待做 |
| wxid → 昵称映射表 | contacts 缓存 | 📝 待做 |

### Phase 2: 感知层（Day 2）

| 任务 | 文件 | 改动量 |
|------|------|--------|
| WeFlowPipeline 开发 | `weflow_pipeline.py` | 新增 ~200 行 |
| SmartPipeline 分流 | `smart_pipeline.py` | ~20 行 |
| 标题 → talker 映射 | `weflow_pipeline.py` | ~30 行 |
| 历史消息初始化 | `weflow_pipeline.py` | ~20 行 |

### Phase 3: 存储层（Day 3）

| 任务 | 文件 | 改动量 |
|------|------|--------|
| GlobalStore localId 去重 | `global_store.py` | ~15 行 |
| 历史消息批量注入 | `global_store.py` | ~10 行 |
| Bot 传递模式参数 | `wechat_bot.py` | ~5 行 |

### Phase 4: 测试验证（Day 4-5）

| 任务 | 验证内容 |
|------|---------|
| 私聊消息读取 | 文本、emoji、小程序卡片 |
| 群聊消息读取 | sender 识别、@消息、多人连续发言 |
| 去重准确性 | 连续 tick 不重复、新消息不漏 |
| 历史初始化 | 首次打开聊天拉 100 条历史 |
| Fallback 测试 | 手动关闭 WeFlow，验证 OCR 接管 |
| 长时运行 | 24 小时不间断，对比 OCR/WeFlow 稳定性 |

---

## 11. 风险清单

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| WeFlow 更新后 API 变化 | 中 | 高 | 封装客户端，隔离变化；hybrid 模式自动 fallback |
| WeFlow 进程不稳定 | 中 | 高 | hybrid 模式；监控 health_check |
| 群聊 sender 只有 wxid 无昵称 | 高 | 中 | contacts 缓存做 wxid→昵称映射 |
| 消息时序错乱（createTime）| 低 | 中 | 以 `localId` 为主键，`createTime` 仅展示 |
| 数据库与 UI 不同步 | 低 | 低 | 微信数据库刷新有延迟，可接受 |
| macOS 升级后 WeFlow 失效 | 中 | 高 | 环境变量切回 `WEFLOW_MODE=ocr` |
| 代码复杂度增加 | 中 | 低 | 新增模块独立，原有代码零改动 |

---

## 附录 A: 群聊消息格式待验证项

```
测试群聊: 34933558648@chatroom / 20886562146@chatroom / 6738243824@chatroom

待确认:
1. 群聊消息的 senderUsername 是 wxid 还是昵称？
2. @消息的内容格式（content 中是否包含 @wxid 或 @昵称）？
3. 自己发的群聊消息 isSend=1，senderUsername 是什么？
4. 连续多人发言时，localId 是否连续递增？
5. 撤回消息是否还在数据库中？
```

---

## 附录 B: 文件变更总览

| 文件 | 动作 | 改动行数 | 说明 |
|------|------|---------|------|
| `perception/weflow_client.py` | 新增 | 265 | ✅ 已就绪 |
| `perception/weflow_pipeline.py` | 新增 | ~200 | Phase 2 |
| `perception/smart_pipeline.py` | 修改 | ~20 | `__init__` + `perceive()` 分流 |
| `session/global_store.py` | 修改 | ~25 | `_merge_tick_weflow` + `inject_history` |
| `bot/wechat_bot.py` | 修改 | ~5 | `merge_tick` 传 mode 参数 |
| `models/base.py` | 可选修改 | ~4 | ChatMessage 增加可选字段 |
| 其他所有文件 | 不动 | 0 | 完全保留 |

---

*方案完毕，等待确认后进入实施阶段。*
