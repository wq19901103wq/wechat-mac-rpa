# Layout Module Spec

## 1. 模块职责
将 OCR 元素按微信 UI 区域分组（左侧聊天列表、右侧标题栏/消息区/输入框）。**只做分组，不做过滤。**

## 2. 功能需求 (FR)

- **FR-1**: 基于 `LayoutProfile` 中的相对坐标，将元素分为 `left_elements`（左侧边栏+聊天列表）和 `right_elements`（右侧内容区）。
- **FR-2**: 从 `right_elements` 中提取：标题栏（顶部）、输入框（底部）、时间戳（中央匹配正则）。
- **FR-3**: 通过颜色检测识别绿色气泡区域（`self_bubbles`），用于后续判断"自己"发送的消息。
- **FR-4**: 解析左侧聊天列表，输出 `List[ChatListItem]`（含昵称、预览、未读数、时间戳、点击区域 `rect`）。
- **FR-5**: 动态适配窗口尺寸变化（基于 `scale_x` / `scale_y`）。
- **FR-6**: 提取 `chat_name`（标题栏最长文本，过滤窗口控制按钮噪声）。

## 3. 非功能需求 (NFR)

- **NFR-1**: 绿色气泡检测使用粗筛（容差 15）+ 连通区域 + 精筛（容差 35）策略，避免全图扫描。
- **NFR-2**: 聊天列表未读检测使用 OCR 数字 + 颜色检测双通道。

## 4. 接口契约

### 输入
```python
LayoutParser(profile: LayoutProfile)
parser.parse(elements: List[OCRTextElement], image_path: str) -> UILayout
```

### 输出
```python
UILayout(
    chat_name: str,
    chat_list_items: List[ChatListItem],
    title_elements: List[OCRTextElement],
    input_elements: List[OCRTextElement],
    timestamp_elements: List[OCRTextElement],
    self_bubbles: List[Rect],
    message_candidates: List[OCRTextElement],  # 右侧排除已分类后的剩余元素
)
```

## 5. 核心规则与约束

### 规则 1: `clean_chat_name` 只去时间戳后缀，不去群人数后缀
`clean_chat_name` 仅去除聊天列表中的时间戳后缀（如 `昨天 22:26`）。**群人数后缀（如 `（128）`）的去除由 `chat_utils._normalize_chat_name` 负责，不在本模块处理。**

### 规则 2: 未读角标与头像噪声分离
未读角标通过以下方式与头像噪声区分：
- 位置：头像右上角精确区域
- 尺寸：小面积阈值过滤
- 内容：纯数字且 ≤99
- 颜色：红色 badge 检测作为补充

### 规则 3: 时间戳不是消息
`TIMESTAMP_PATTERNS` 匹配的时间戳元素必须被排除在 `message_candidates` 之外。

## 6. 错误处理
本模块不抛异常。OCR 元素为空时返回空 `UILayout`。

## 7. 依赖关系
- 依赖 `src.models.base`
- 依赖 `src.layout.profile.LayoutProfile`
- 被 `src.perception` 调用
