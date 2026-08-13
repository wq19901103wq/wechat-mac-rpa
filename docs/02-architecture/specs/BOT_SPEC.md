# Bot Module Spec

## 1. 模块职责
主循环编排器：协调感知层、会话层、回复层、动作层，执行 `tick()` → `perceive` → `merge` → `decide` → `reply` → `send` 的完整链路。

## 2. 功能需求 (FR)

- **FR-1**: `tick()`：执行一轮完整的感知-去重-决策-回复循环。
- **FR-2**: `run_auto(interval)`：持续运行 tick 循环，捕获所有未处理异常。
- **FR-3**: 聊天切换：当前聊天无未读时，自动切换到未读数最多的聊天（支持 WeFlow API 轮询和 OCR 角标两种检测方式）。
- **FR-4**: WeFlow 初始化：启动时若 `WEFLOW_MODE=weflow/hybrid`，注入全量历史到 GlobalStore，然后切回 OCR 模式运行。
- **FR-5**: 逐条发送回复，间隔 1.5 秒。
- **FR-6**: 发送后将 Bot 回复放入 `pending_self_messages`，等待感知层确认后入库，避免 WeFlow API 延迟导致上下文缺失。
- **FR-7**: 记忆更新：每轮回复后异步更新用户 wiki / 群 wiki。
- **FR-8**: `send_to_chat()`：外部系统调用，主动发消息。注意：实际直接调用 `sender.send(text)`，未先切换到目标聊天。
- **FR-9**: Debug 日志：每轮 tick 保存完整的 perception/session/reply/action debug 信息。

## 3. 非功能需求 (NFR)

- **NFR-1**: 聊天切换防抖：10 秒内不重复切换同一个目标。
- **NFR-2**: 全局状态增量保存：每轮 tick 结束后保存，只保存 `_dirty` 标记的聊天。
- **NFR-3**: 免回复列表：`{"腾讯新闻", "文件传输助手"}` 等系统账号不回复。

## 4. 接口契约

### 输入
```python
WeChatBot(
    profile: LayoutProfile,
    on_message: Optional[Callable] = None,
    llm_client=None,
    complex_llm_client=None,
    debug_mode: bool = False,
    use_openclaw: bool = True,
    perception=None,
    enable_chat_switch: bool = True,
)
```

### 输出
无直接返回值。副作用：发送消息、更新状态、保存日志。

## 5. 核心规则与约束

### 规则 1: `raw_chat_name` 保留原始 OCR 名称
```python
raw_chat_name = result.chat_name or ""          # 原始值，含群人数后缀
chat_name = _normalize_chat_name(raw_chat_name)  # 归一化值，用于 session key
is_group = _is_group_chat_name(raw_chat_name)    # 基于原始值判断
```
**`is_group` 只计算一次，传递给所有下游模块。**

### 规则 2: 标题栏识别失败时的安全行为
- 有消息但 `chat_name` 为空 → 跳过切换，避免误点当前聊天
- 无消息且 `chat_name` 为空 → 尝试切换到未读聊天

### 规则 3: 空回复也标记为已处理
如果 `replies` 为空（LLM 决策不回复），仍需调用 `mark_replied`，避免下一轮重复当成未读。

### 规则 4: 记忆更新区分私聊/群聊
- 私聊：只更新用户 wiki（`chat_name` 作为用户名）
- 群聊：同时更新群 wiki 和最后发言者 wiki

### 规则 5: 启动时自动同步 knowledge_source.md
通过 `scripts/sync_knowledge.py` 同步外挂知识到 wiki 格式。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| `perceive()` 返回 None | 记录日志，尝试切换未读聊天 |
| 发送失败 | 中断后续回复，记录错误 |
| tick 未捕获异常 | 记录 exception，不阻断循环 |
| 保存状态失败 | 记录 warning，不阻断 |

## 7. 依赖关系
- 依赖 `src.perception`, `src.session`, `src.reply`, `src.action`, `src.memory`, `src.logging`
- 依赖 `src.utils.chat_utils._is_group_chat_name, _normalize_chat_name`
