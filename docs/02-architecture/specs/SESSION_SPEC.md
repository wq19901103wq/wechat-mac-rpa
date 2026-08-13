# Session (GlobalStore) Module Spec

## 1. 模块职责
管理所有聊天的消息历史和回复状态，提供统一去重、增量持久化。

## 2. 功能需求 (FR)

- **FR-1**: `merge_tick(chat_name, messages)`：合并本轮感知到的消息，返回 `(ChatState, unreplied_messages)`。
- **FR-2**: 去重策略：
  - OCR 模式：LCS 序列对齐 + 模糊去重（`SequenceMatcher` + `Jaccard`）
  - WeFlow 模式：`local_id` 精确去重
- **FR-3**: 消息 ID 生成：基于 `chat_name + 标准化 sender + 内容指纹`。
- **FR-4**: 裁剪历史：每聊天最多保留 `max_messages` 条（默认 200），超限删除最旧消息。
- **FR-5**: `mark_replied()`：标记消息已回复（支持对象引用匹配 + text+sender 兜底）。
- **FR-6**: `inject_history()`：批量注入历史消息（WeFlow 初始化用），标记 `replied=True`，不去重不裁剪。
- **FR-7**: 增量持久化：只保存有变化的聊天（`_dirty` 标记），分片格式（每聊天一个 JSON 文件）。

## 3. 非功能需求 (NFR)

- **NFR-1**: 线程安全：所有状态修改通过 `threading.Lock` 保护。
- **NFR-2**: 加载兼容：优先加载分片格式，回退旧格式单 JSON。
- **NFR-3**: 保存原子性：先写 `.tmp` 文件，再 `os.replace`。

## 4. 接口契约

### 输入
```python
GlobalStore(max_messages: int = 200, state_file: str = "data/global_state.json")

merge_tick(
    chat_name: str,
    messages: List[ChatMessage],
    mode: str = "ocr",       # "ocr" | "weflow" | "hybrid"
    is_group: bool = False,  # 由感知层传入，不自行推导
) -> Tuple[ChatState, List[ChatMessage]]
```

### 输出
- `ChatState`: 包含 `chat_id`, `chat_name`, `is_group`, `messages`, `_msg_ids`
- `List[ChatMessage]`: 本轮新出现的、未回复的消息列表

## 5. 核心规则与约束

### 规则 1: `_normalize_sender` 不修改 `msg.sender`
标准化 sender 仅用于生成 `_msg_id`，**不得写入 `msg.sender` 字段**。`msg.sender` 保持感知层原始值。

### 规则 2: `is_group` 由调用方传入
`merge_tick` 和 `inject_history` 的 `is_group` 参数由 Bot 层从 `PerceptionResult` 传入。存储层不自行判断群聊状态。

### 规则 3: 模糊去重阈值按长度动态调整
| 消息长度 | 阈值 |
|---------|------|
| ≤3 | 0.90 |
| ≤8 | 0.85 |
| ≤20 | 0.82 |
| >20 | 0.80 |

### 规则 4: Bot 自己的消息不参与去重匹配
模糊去重时跳过 `sender_type == SELF` 的消息，避免拿 Bot 回复去重用户新消息。

### 规则 5: 图片/表情基于 `image_description` 去重
文字消息基于 `text`，图片/表情/混合消息基于 `message_type + image_description`。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| 加载失败 | 记录 warning，从空状态开始 |
| 保存失败 | 记录 warning/error，不阻断主流程 |

## 7. 依赖关系
- 依赖 `src.models.base`
- 依赖 `src.utils.chat_utils._is_group_chat_name`（仅用于 `_normalize_sender` 的兜底判断）
