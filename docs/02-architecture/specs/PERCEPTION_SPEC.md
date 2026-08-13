# Perception Module Spec

## 1. 模块职责
整合 Capture + OCR + Layout (+ API)，输出结构化的 `PerceptionResult`。是 Bot 层与视觉实现之间的唯一边界。

## 2. 功能需求 (FR)

- **FR-1**: 提供与 `VisionPipeline.perceive()` 完全兼容的接口（duck typing）。
- **FR-2**: 本地预判：通过像素差异判断截图是否有实质变化。消息区域或聊天列表区域任一有变化即触发 API；两区域均无变化时跳过 API 调用。
- **FR-3**: 有变化时：本地 Layout 提供聊天列表位置 + qwen3.6-flash API 提供消息/昵称/未读数识别。
- **FR-4**: 支持 WeFlow 分流模式（`weflow` / `hybrid` / `ocr`）。
- **FR-5**: 合并本地 Layout 和 API 的聊天列表结果（本地提供准确 `rect`，API 提供准确 `nickname`/`unread_count`）。
- **FR-6**: 过滤 API 误识别的未读角标（时间戳、群人数、非数字、>99）。
- **FR-7**: 窗口尺寸检查：小于 800x600 视为异常（登录浮窗），返回 None。

## 3. 非功能需求 (NFR)

- **NFR-1**: 像素差异阈值默认 0.001，同时应用于消息区域和聊天列表区域。
- **NFR-2**: 稳定模式：连续多帧低差异后，阈值临时降低 50%。
- **NFR-3**: API 客户端延迟初始化，失败时优雅降级为本地 OCR。
- **NFR-4**: 消息区域 ROI：`(0.35, 0.12, 0.95, 0.97)`；聊天列表区域 ROI：`(0.0, 0.0, 0.35, 1.0)`。

## 4. 接口契约

### 输入
```python
SmartPerceptionPipeline(
    profile: LayoutProfile,
    api_key: Optional[str] = None,
    pixel_diff_threshold: float = 0.001,
    message_region: tuple = (0.35, 0.12, 0.95, 0.97),
    chat_list_region: tuple = (0.0, 0.0, 0.35, 1.0),
    always_use_api: bool = False,
)
```

### 输出
```python
Optional[PerceptionResult]
# None: 窗口捕获失败或尺寸异常
```

## 5. 核心规则与约束

### 规则 1: `is_group` 在感知层唯一判断
```python
is_group = _is_group_chat_name(chat_name)  # 唯一实现点在 chat_utils
```
**下游模块不得重新判断。PerceptionResult 携带的 `is_group` 是权威值。**

### 规则 2: 私聊 sender 统一为 `chat_name`
在 `_convert_api_messages` 中：
```python
if not is_group and sender_type == SenderType.OTHER:
    sender = chat_name
```
确保私聊下游看到的是对方真实昵称，而非"对方"。

### 规则 3: 群聊 sender 校验防错
API 有时把消息内容当成 sender（连续消息无昵称时）。校验规则：
```python
if is_group and sender_type == SenderType.OTHER:
    if sender == text and len(text) > 3:
        sender = last_left_sender
```

### 规则 4: 输入框内容必须排除
API prompt 中明确要求排除截图最底部的输入框区域（未发送草稿）。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| 窗口捕获失败 | 返回 None，记录 warning |
| API 请求失败 | 降级为本地 OCR（空 messages） |
| WeFlow 失败且模式为 `weflow` | 返回 None |
| WeFlow 失败且模式为 `hybrid` | Fallback 到 OCR |

## 7. 依赖关系
- 依赖 `src.capture`, `src.ocr`, `src.layout`
- 依赖 `src.utils.chat_utils._is_group_chat_name`
- 依赖 `src.utils.xml_utils._extract_xml_text`
- 被 `src.bot.WeChatBot` 调用
