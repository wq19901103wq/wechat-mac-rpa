# Models Module Spec

## 1. 模块职责
定义全项目共享的领域模型（Domain Models）。**本模块只包含数据定义，不含业务逻辑。**

## 2. 功能需求 (FR)

- **FR-1**: 提供基础几何类型 `Point`、`Rect`，用于屏幕坐标系。
- **FR-2**: `OCRTextElement` 承载 OCR 原始输出，包含文本、外接矩形 `bbox`、中心点 `center`、置信度。
- **FR-3**: `ChatMessage` 是核心消息模型，必须携带 `sender_type`（self/other/system/unknown）、回复状态（`replied`/`reply_text`/`reply_time`）、消息类型（text/image/sticker/mixed/link_card）。
- **FR-4**: `PerceptionResult` 是感知层的唯一输出，对 Bot 层隐藏所有视觉实现细节。
- **FR-5**: `ChatListItem` 供 Layout 层和 UIInteractor 共用，包含可点击区域 `rect`。

## 3. 非功能需求 (NFR)

- **NFR-1**: `Point` 和 `Rect` 使用 `frozen=True` dataclass，确保不可变。
- **NFR-2**: 新增字段必须为可选或有默认值，避免反序列化旧数据时崩溃。

## 4. 接口契约

### 输出
| 类型 | 说明 |
|------|------|
| `PerceptionResult` | `chat_name: str`, `is_group: bool`, `messages: List[ChatMessage]`, `chat_list_items: List[ChatListItem]`, `screenshot_path: str`, `window_rect: Optional[Rect]`, `scale_factor: float`, `debug_info: Optional[Dict]` |
| `ChatMessage` | `text`, `sender`, `sender_type: SenderType`, `chat_name`, `is_at_me`, `timestamp`, `replied`, `reply_text`, `reply_time`, `message_type`, `image_description`, `image_text`, `is_image_duplicate` |

## 5. 核心规则与约束

### 规则 1: `PerceptionResult.is_group` 是权威来源
`is_group` 由感知层统一判断，**下游模块（Bot/Session/Reply）必须直接使用此值，不得重新从 `chat_name` 推导。**

### 规则 2: `msg.chat_name` 和 `msg.sender` 是只读字段
存储层和 Bot 层不得修改 `ChatMessage.chat_name` 和 `ChatMessage.sender`。存储使用独立的 `session_key` 做查找。

例外：在 `merge_tick` 首次入库时，`chat_name` 会被标准化为归一化后的 session key（`_normalize_chat_name`）。此修改仅发生在消息首次进入 GlobalStore 时，后续不得再次修改。

### 规则 3: 新增 WeFlow 字段不影响现有代码
`local_id`、`server_id`、`create_time`、`raw_type`、`sender_wxid` 为 WeFlow 扩展字段，必须为 `Optional` 且有默认值。

## 6. 错误处理
本模块不抛异常。数据校验由调用方负责。

## 7. 依赖关系
- 被所有其他模块依赖（最底层）。
- 不依赖任何项目内模块。
