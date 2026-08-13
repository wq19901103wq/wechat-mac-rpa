# Action Module Spec

## 1. 模块职责
执行对微信窗口的 UI 操作：发送消息、点击聊天列表项、登录恢复。

## 2. 功能需求 (FR)

- **FR-1**: `WeChatMessageSender.send(text)`：通过 AppleScript 将文本发送到当前微信聊天。
- **FR-2**: 发送流程：保存原始剪贴板 → 激活微信 → pbcopy 文本 → 点击输入框获取焦点 → Command+V 粘贴 → 清空剪贴板 → verify（Command+A+C + pbpaste）→ Right Arrow 取消全选 → Return 发送 → 恢复原始剪贴板。
- **FR-3**: 粘贴验证：发送前必须验证输入框内容是否与预期文本匹配，不匹配则重试（3 次常规重试 + 2 次 fallback：更长 delay 粘贴、keystroke 逐字输入，最多 5 次）。
- **FR-4**: `ChatListClicker.click_item(item)`：根据 `ChatListItem.rect` 点击左侧聊天列表项。
- **FR-5**: `WeChatLoginHandler`：检测登录状态，尝试恢复（如点击扫码登录按钮）。

## 3. 非功能需求 (NFR)

- **NFR-1**: 每次发送必须恢复用户原始剪贴板内容。
- **NFR-2**: 激活微信重试 3 次，每次间隔 0.3 秒。
- **NFR-3**: 发送间隔：多条回复之间间隔 1.5 秒。

## 4. 接口契约

### 输入
```python
WeChatMessageSender().send(text: str) -> ActionResult

ChatListClicker(window_rect: Rect, scale_factor: float)
clicker.click_item(item: ChatListItem) -> bool
```

### 输出
```python
ActionResult(success: bool, sent_text: Optional[str], error: Optional[str])
```

## 5. 核心规则与约束

### 规则 1: Verify 前必须清空剪贴板
```python
# 关键：防止 pbpaste 读到的是旧剪贴板残留，而不是输入框内容
subprocess.run(["pbcopy"], input=b"", timeout=2)
```
这是 **sender false-success bug** 的修复：旧代码未清空剪贴板，`pbpaste` 读到的是之前 pbcopy 的内容，导致 verify 永远通过。

### 规则 2: Verify 后必须先 Right Arrow 再 Return
Verify 阶段使用了 Command+A 全选，直接按 Return 某些输入框会把选中内容替换成换行符。必须先 `key code 124`（Right Arrow）取消全选，再 `keystroke return`。

### 规则 3: 粘贴不匹配时清空输入框重试
```python
keystroke "a" using command down  # 全选
key code 51                       # Delete
```
然后重新 pbcopy + 粘贴。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| 激活微信失败 | 返回 `ActionResult(success=False)` |
| pbcopy 失败 | 返回失败 |
| 5 次粘贴验证均失败 | 返回失败，附带最后一次读取到的输入框内容 |
| 回车发送失败 | 返回失败 |

## 7. 依赖关系
- 依赖 `src.models.base.ActionResult, ChatListItem, Rect`
- 被 `src.bot.WeChatBot` 调用
