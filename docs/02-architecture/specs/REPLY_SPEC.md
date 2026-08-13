# Reply Module Spec

## 1. 模块职责
根据未读消息和历史上下文，生成 Bot 回复内容。支持工具调用、Skill 路由、双模型切换。

## 2. 功能需求 (FR)

- **FR-1**: `ReplyGenerator.generate(unreplied, all_messages, is_group)` → 返回 `List[str]`（0-3 条回复）。
- **FR-2**: 构建结构化 user prompt：会话信息 + `[我的信息]`（Bot wiki）+ `[对方信息]`（用户 wiki）+ `[相关背景]` + `[历史消息]` + `[未读消息]`。
- **FR-3**: 支持多轮工具调用（function calling），最多 10 轮，工具阶段总时间不超过 20 秒。
- **FR-4**: Skill 路由：根据用户消息内容，轻量 LLM 调用判断需要加载哪些 skill，将 skill 正文注入 prompt。
- **FR-5**: 模型切换：复杂任务（匹配 skill）走 `complex_llm_client`（hermes）；简单任务走 `llm_client`（deepseek）。
- **FR-6**: deepseek 可输出 `"use_hermes"` 信号，触发 fallback 到 hermes 重新生成（保留工具调用结果上下文）。
- **FR-7**: 解析 LLM 输出的 JSON `{"replies": [...]}`，过滤空字符串和敷衍词（收到、好的、嗯、OK、1）。
- **FR-8**: `ReplyPolicy.should_reply(msg, state)` 判断单条消息是否应回复。

## 3. 非功能需求 (NFR)

- **NFR-1**: deepseek 总超时 35 秒，hermes 总超时 600 秒。
- **NFR-2**: 每条回复简洁自然，deepseek 不超过 50 字，hermes 不超过 300 字。（目前仅依赖 prompt 约束，无代码级兜底。TODO：增加代码级截断）
- **NFR-3**: 所有 LLM 调用、工具调用、trace 记录到 debug 字段，供排查使用。

## 4. 接口契约

### 输入
```python
ReplyGenerator(
    llm_client=None,           # deepseek-v4-flash
    complex_llm_client=None,   # hermes（可选）
    memory_engine=None,        # 用于 search_memory 工具 + wiki 注入
)

generate(
    unreplied: List[ChatMessage],
    all_messages: List[ChatMessage],
    is_group: bool = False,
    tick_id: int = 0,
) -> List[str]
```

### 输出
`List[str]`：0-3 条回复文本。空列表表示"不回复"。

## 5. 核心规则与约束

### 规则 1: 私聊必须回复，群聊被@时必须回复
System prompt 强制规则：
- 私聊：`replies` 不得为空数组
- 群聊：被 `@` 时 `replies` 不得为空；没被 `@` 时 `replies` 必须为空

### 规则 2: `[我的信息]` 优先于 `[对方信息]`
`_build_user_prompt` 先注入 Bot 自己的 wiki（`林岚`），再注入对方 wiki。防止 LLM 被对方 wiki 淹没后混淆身份。

### 规则 3: 历史消息选择策略
取以下并集：
1. 最近 20 条消息
2. 最近 10 分钟内的消息
3. 最近 5 条 Bot 自己发的消息（防止被对方密集消息淹没）

上限 80 条，保持正序（旧在前）。

### 规则 4: 不知道就必须 search_memory
System prompt 强制：涉及人物相关事实信息时，**必须先调用 `search_memory`**；搜索后 wiki 中仍无记录，才回答"不知道"。严禁跳过搜索直接说不知道。

### 规则 5: Bot 自己的历史消息存在严重错误可能
System prompt 明确告知 LLM：历史消息中标记为"我："的内容可能存在严重错误或幻觉，**严禁当作事实引用**。与 wiki 冲突时以 wiki 为准。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| LLM 返回空 / 无效内容 | 重试 2 次，仍失败返回空列表 |
| JSON 解析失败 | 回退：把整段文本当作单条回复 |
| 工具调用超时 | `force_no_tools=True`，禁用工具后强制生成文本回复 |
| hermes fallback 也失败 | 返回空列表 |

## 7. 依赖关系
- 依赖 `src.models.base`
- 依赖 `src.tools.tool_registry`
- 依赖 `src.memory.engine.MemoryEngine`
- 依赖 `src.reply.session_memory.SessionMemory`
- 被 `src.bot.WeChatBot` 调用
