# ReAct + Self-Refine 回复生成重构设计

## 更新信息

- 日期：2026-07-05
- 作者：Kimi Code CLI
- 状态：已实现（feat/react-self-refine 分支）
- 关联 issue：回复质量 / 思考深度不足

## 背景

当前 `src/reply/generator.py` 基于单次 LLM 调用生成回复。`deepseek-v4-flash` 虽有 thinking 能力，但受限于：

1. `persona.md` 明确压制思考过程（“不要分析，不要推理”）。
2. 固定 `max_tokens=2000`，复杂场景思考空间不足。
3. 无显式质量校验环节，错误/幻觉无法在线修正。

用户要求提升回复质量，特别是增加思考深度。经实测，两步推理和 ReAct 模式可显著提升分析深度。本规格决定采用 **ReAct（推理+行动）+ Self-Refine（自检+修正）** 架构替代现有单次推理。

## 目标

1. 让模型在生成前能主动、深入思考（通过 `think` 工具）。
2. 让模型在必要时调用搜索/记忆工具补充信息。
3. 生成后自动检查质量，发现问题时自我修正。
4. 保持示例用户甲人设（简洁、幽默、略带傲娇）。
5. 总延迟控制在 **20s** 以内。
6. 支持一键切回原有单次推理模式（通过环境变量）。

## 非目标

- 不引入新的 LLM 模型，全部使用 `deepseek-v4-flash`。
- 不改变感知层、OCR、消息发送层。
- 不替换 JudgeWorker 离线审计机制。
- 不做长期记忆持久化改造。

## 架构

```
用户消息 + 历史上下文
        │
        ▼
┌─────────────────────────────────────┐
│ Call 1: ReAct 生成                    │
│   system: persona.md + 工具描述        │
│   user: 结构化上下文 + 格式指令          │
│   tools: think, search_memory, ...    │
│   max_tokens: 10000                   │
│   max_tool_calls: 10                  │
│   timeout: 动态剩余预算                 │
│   output: {"replies": [...]}         │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ Call 2: Feedback（Self-Refine）       │
│   追加 user: 质量检查指令 + 格式指令     │
│   max_tokens: 10000                   │
│   timeout: 动态剩余预算                 │
│   output: {"decision": "pass/fail",  │
│            "issues": [...]}          │
└─────────────────────────────────────┘
        │
        ├── decision=pass ──► 返回 replies
        │
        └── decision=fail ──► Call 3: Iterate
                追加 user: 修正指令 + 格式指令
                max_tokens: 10000
                timeout: 动态剩余预算
                output: {"replies": [...]}
```

### ReAct 循环与现有工具循环的关系

现有 `src/reply/generator.py` 第 358-488 行**已经是一个 ReAct 循环**：模型返回 `tool_calls` → 执行工具 → 把结果塞回 `messages` → 再调一次 LLM。本次改动不是替换循环，而是在现有循环上做增量改造：

1. **注册 `think` 工具**。
2. **在工具执行分支处理 `think`**：`think` 不调用外部服务，直接返回确认字符串。
3. **提高 `max_tokens`**：从 2000 提到 10000。
4. **增加 `max_tool_calls` 限制**：当前无明确上限，新设 10。
5. **循环结束后追加 Self-Refine**：Feedback + Iterate。

现有工具缓存、日志逻辑保持不变。

```python
def _react_generate(self, messages, tools, deadline):
    tool_call_count = 0
    while tool_call_count < MAX_TOOL_CALLS:
        response = self.llm_client.chat(messages=messages, tools=tools, timeout=deadline - time.time())
        msg = response.choices[0].message

        # 保存 assistant message（含 reasoning_content/tool_calls）
        assistant_msg = {"role": "assistant"}
        if getattr(msg, "content", None):
            assistant_msg["content"] = msg.content
        if getattr(msg, "reasoning_content", None):
            assistant_msg["reasoning_content"] = msg.reasoning_content
        if getattr(msg, "tool_calls", None):
            assistant_msg["tool_calls"] = [dict(tc) for tc in msg.tool_calls]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            # 最终回复
            return parse_replies(msg.content or ""), messages

        # 执行 tool_calls
        for tc in msg.tool_calls:
            if tc.function.name == "think":
                result = "思考已记录，继续生成回复。"
            else:
                result = self.tool_registry.execute(tc.function.name, tc.function.arguments)
                self.session_memory.add_tool_result(chat_name, tc.function.name, ...)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
        tool_call_count += len(msg.tool_calls)

    # 达到上限仍未输出 JSON，强制收尾
    messages.append({"role": "user", "content": "请直接输出最终回复 JSON。"})
    response = self.llm_client.chat(messages=messages, timeout=deadline - time.time())
    return parse_replies(response.choices[0].message.content or ""), messages
```

## 详细设计

### 1. 工具注册

在 `ReplyGenerator.__init__` 中注册 `think` 工具：

```python
self.tool_registry.register(
    name="think",
    description=(
        "在回复前停下来深入思考。用于需要深度推理、权衡多因素、分析意图、"
        "检查人设一致性的场景。此工具不获取新信息，只记录你的思考过程。"
        "在想清楚之前随时可以调用，次数不限。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "你的思考内容。尽情思考，越详细越好。"
            }
        },
        "required": ["thought"],
    },
    func=lambda thought: "思考已记录，继续生成回复。",
)
```

`think` 工具的执行函数不调用外部服务，仅返回确认字符串，让模型继续生成。

### 2. ReAct 生成阶段（Call 1）

输入：
- `system_prompt`: `persona.md` + 技能/工具描述
- `user_prompt`: 结构化上下文（聊天历史、对方信息、未读消息、已缓存数据）+ 格式指令
- `tools`: 当前已注册工具（含 `think`）
- `max_tokens`: 10000
- `max_tool_calls`: 10
- `timeout`: 20s 总预算减去已消耗时间

循环逻辑：
1. 调用 LLM。
2. 如果返回 `tool_calls`，执行工具，把结果加入 messages，继续循环。
3. 如果返回最终 JSON `{"replies": [...]}`，解析并结束。
4. 如果 `tool_calls` 次数达到 10 仍未输出 JSON，追加一条 user message：
   > “思考已经足够，请直接输出最终回复 JSON。”
   再调一次 LLM。如果仍然失败，返回 `{"replies": []}`。

格式指令追加在 user prompt 末尾：

```markdown
请直接输出 JSON：`{"replies": ["回复1", "回复2"]}`。
JSON 之外不要输出任何文字。
```

### 3. Feedback 阶段（Call 2）

在 Call 1 的完整 messages 末尾追加一条 user message：

```markdown
请作为质量检查员，评估以上 assistant 回复。

检查维度：
1. 诚实性 / 幻觉：回复中的事实、数字、事件是否有出处？是否编造？
2. 一致性：是否和上下文、工具结果矛盾？是否自我矛盾？
3. 人设符合度：是否像示例用户甲本人？是否保持简洁、幽默、略带傲娇？是否变成 AI 助手腔？
4. 称呼正确性：是否使用对方偏好称呼？（如示例用户申必须叫“示例用户申”）
5. 极简原则：是否啰嗦、重复、废话？
6. 格式正确性：是否是有效 JSON？结构是否为 {"replies": [...]}？
7. 安全性：是否泄露敏感信息？是否有不合适内容？

如果回复质量合格，输出：`{"decision": "pass"}`
如果发现问题，输出：`{"decision": "fail", "issues": ["问题1描述", "问题2描述"]}`

不要输出任何其他文字。
```

输入参数：
- `messages`: Call 1 完整 messages + Feedback user message
- `max_tokens`: 10000
- `temperature`: 0.3（检查任务，低温度更稳定）
- `timeout`: 动态剩余预算

输出解析：
- `decision=pass`：直接返回 Call 1 的 replies。
- `decision=fail`：进入 Call 3。
- 解析失败 / 超时 / 异常：降级返回 Call 1 的 replies。

### 4. Iterate 阶段（Call 3）

在 Feedback 的 messages 基础上追加：

```markdown
请基于以上反馈改进 assistant 的回复。

要求：
- 修复反馈中列出的所有问题。
- 保持示例用户甲本人语气，不要变成 AI 助手腔。
- 保持简洁，不要啰嗦。
- 输出 JSON：`{"replies": ["改进后的回复1", "改进后的回复2"]}`
- JSON 之外不要输出任何文字。
```

输入参数：
- `messages`: Call 2 messages + Iterate user message
- `max_tokens`: 10000
- `temperature`: 0.7
- `timeout`: 动态剩余预算

输出解析：
- 成功：返回新的 replies。
- 失败：降级返回 Call 1 的 replies。

### 5. 多轮 Self-Refine

支持配置多轮：

```python
SELF_REFINE_MAX_ITER=1  # 默认 1 轮（Feedback + 1 次 Iterate）
SELF_REFINE_MAX_ITER=2  # Feedback + 最多 2 次 Iterate
```

不建议超过 2 轮。每轮都要检查总超时预算，超时立即返回当前最佳回复。

### 6. 超时控制

统一总超时 **20s**：

```python
deadline = time.time() + 20.0

# Call 1
remaining = deadline - time.time()
replies = react_generate(timeout=remaining)

# Call 2
remaining = deadline - time.time()
feedback = self_refine(timeout=remaining)

# Call 3（如果需要）
remaining = deadline - time.time()
replies = iterate(timeout=remaining)
```

每个 `llm_client.chat()` 调用传入 `timeout=remaining`。

### 7. reasoning_content 回传

DeepSeek V4 thinking + tool call 要求：如果 assistant message 包含 `tool_calls`，后续请求中必须继续回传该 assistant message 的 `reasoning_content` 字段（可为空字符串，但不能缺失）。

**责任划分**：
- `generator.py`：构造 messages 时，为每个 assistant message 显式保留 `reasoning_content`。
- `qwen_client.py`：不主动删除或过滤 messages 中的 `reasoning_content`，原样透传给 DeepSeek API。

`generator.py` 中保存 assistant message 的代码：

```python
assistant_msg = {"role": "assistant"}
if getattr(msg, "content", None):
    assistant_msg["content"] = msg.content
if getattr(msg, "reasoning_content", None):
    assistant_msg["reasoning_content"] = msg.reasoning_content
if getattr(msg, "tool_calls", None):
    assistant_msg["tool_calls"] = [dict(tc) for tc in msg.tool_calls]
messages.append(assistant_msg)
```

`qwen_client.py` 只需确保不清理 `reasoning_content`，现有逻辑已记录 `last_thinking`，无需额外改动接口。

### 8. Session Memory 清理

删除 `src/reply/session_memory.py` 中的死代码：
- `SessionSnapshot.bot_replies`
- `SessionSnapshot.add_reply()`
- `SessionSnapshot.get_recent_replies()`
- `SessionMemory.add_reply()`
- `SessionMemory.get_recent_replies()`

保留工具结果缓存逻辑不变。

同步删除 `src/reply/generator.py` 中对 `session_memory.add_reply()` 的调用。

受影响的测试文件：
- `src/tests/test_session_memory.py`：删除 `get_recent_replies` / `add_reply` 相关测试。
- `tests_integration/test_hermes_integration.py`：整体删除或改为测试 ReAct + Self-Refine。
- `src/tests/test_reply_generator.py`（如存在 Hermes fallback 相关断言）：更新预期。

### 9. Hermes 清理

删除 `complex_llm_client` 相关代码：
- `ReplyGenerator.__init__` 中的 `complex_llm_client` 参数
- `self.complex_llm_client` 属性
- `_hermes_system_prompt()` 方法
- Hermes fallback 逻辑（`use_hermes` 判断和切换）
- `last_hermes_*` debug 字段
- `wechat_bot.py` 中的 `complex_llm_client` 传入

### 10. persona.md 修改

删除 `<rule name="输出格式">` 段，或改为：

```markdown
<rule name="思考与输出">
你可以在心里思考，也可以调用 think 工具记录思考过程。
思考可以充分、深入，但最终回复必须是 JSON：`{"replies": [...]}`
JSON 之外不要输出任何文字。
</rule>
```

具体格式指令在每轮 user prompt 末尾追加，不在 system prompt 里重复。

### Prompt 文件模板

新增三个 prompt 文件：

**`prompts/feedback.md`**

```markdown
请作为质量检查员，评估以上 assistant 回复。

检查维度：
1. 诚实性 / 幻觉：回复中的事实、数字、事件是否有出处？是否编造？
2. 一致性：是否和上下文、工具结果矛盾？是否自我矛盾？
3. 人设符合度：是否像示例用户甲本人？是否保持简洁、幽默、略带傲娇？是否变成 AI 助手腔？
4. 称呼正确性：是否使用对方偏好称呼？（如示例用户申必须叫“示例用户申”）
5. 极简原则：是否啰嗦、重复、废话？
6. 格式正确性：是否是有效 JSON？结构是否为 {"replies": [...]}？
7. 安全性：是否泄露敏感信息？是否有不合适内容？

如果回复质量合格，输出：`{"decision": "pass"}`
如果发现问题，输出：`{"decision": "fail", "issues": ["问题1描述", "问题2描述"]}`

不要输出任何其他文字。
```

**`prompts/iterate.md`**

```markdown
请基于以上反馈改进 assistant 的回复。

要求：
- 修复反馈中列出的所有问题。
- 保持示例用户甲本人语气，不要变成 AI 助手腔。
- 保持简洁，不要啰嗦。
- 输出 JSON：`{"replies": ["改进后的回复1", "改进后的回复2"]}`
- JSON 之外不要输出任何文字。
```

**`prompts/reply_format.txt`**（生成阶段追加）

```
请直接输出 JSON：`{"replies": ["回复1", "回复2"]}`。
JSON 之外不要输出任何文字。
```

### 11. 环境变量开关

```bash
# 总开关
ENABLE_REACT_TOOLS=1          # 启用 think 工具
ENABLE_SELF_REFINE=1          # 启用 Feedback + Iterate
SELF_REFINE_MAX_ITER=1        # Iterate 最大轮数
```

开关组合语义：

| ENABLE_REACT_TOOLS | ENABLE_SELF_REFINE | 行为 |
|---|---|---|
| 0 | 0 | 原有单次推理模式（无 think 工具，无 Feedback） |
| 1 | 0 | ReAct 生成（含 think 工具），生成后不做 Feedback |
| 0 | 1 | **等价于 (1, 1)**：Self-Refine 依赖 ReAct 生成的完整 messages，若 ReAct 关闭则 Self-Refine 自动开启 |
| 1 | 1 | 完整 ReAct + Self-Refine |

实现时统一在 `__init__` 中规范化：

```python
self.enable_react_tools = os.environ.get("ENABLE_REACT_TOOLS", "1").lower() in ("1", "true")
self.enable_self_refine = os.environ.get("ENABLE_SELF_REFINE", "1").lower() in ("1", "true")
if self.enable_self_refine:
    self.enable_react_tools = True
```

## 接口变更

### `src/utils/qwen_client.py`

无需新增公共接口，但需确保以下行为：

1. 响应中的 `reasoning_content` 继续记录到 `self.last_thinking`。
2. 请求透传 `messages` 时，不删除或过滤任何字段（包括 `reasoning_content`）。
3. DeepSeek 官方端点继续自动附加 `extra_body={"thinking": {"type": "enabled"}}`。

当前实现已满足 1 和 3，只需验证 2。

### `ReplyGenerator.__init__`

移除：
```python
complex_llm_client=None,
```

保留：
```python
llm_client=None, memory_engine=None, tool_registry=None,
judge_worker=None, enable_time_awareness=True,
enable_reply_restraint=True, enable_unread_dedup=True,
enable_timestamps=True, enable_mode_detection=None
```

### `ReplyGenerator.generate`

签名不变：
```python
def generate(self, unreplied, all_messages, is_group=False, tick_id=None) -> List[str]:
```

内部实现改为 ReAct + Self-Refine。

### 新增内部方法

```python
def _react_generate(
    self,
    messages: List[Dict],
    tools: List[Dict],
    deadline: float
) -> Tuple[List[str], List[Dict], List[Dict]]:
    """ReAct 生成阶段。
    返回：
      - replies: 解析后的回复列表
      - final_messages: 最终 messages（含 tool 历史）
      - tool_call_summary: 工具调用摘要
    """

def _self_refine(
    self,
    messages: List[Dict],
    replies: List[str],
    deadline: float
) -> Tuple[str, Optional[List[str]], List[Dict], List[str]]:
    """Feedback 阶段。
    返回：
      - decision: "pass" 或 "fail"
      - issues: 当 decision="fail" 时的问题列表；decision="pass" 时为 None
      - updated_messages: 追加 Feedback 后的 messages
      - raw_feedback_text: 原始 Feedback 文本（用于日志/debug）
    """

def _iterate(
    self,
    messages: List[Dict],
    issues: List[str],
    deadline: float
) -> Tuple[List[str], List[Dict], str]:
    """Iterate 阶段。
    返回：
      - replies: 修正后的回复列表
      - updated_messages: 追加 Iterate 后的 messages
      - raw_iterate_text: 原始 Iterate 文本（用于日志/debug）
    """
```

### 复杂场景判定标准

复杂场景指需要深度思考或外部信息补充的场景。判定规则（满足任一即视为复杂）：

1. **Skill 匹配**：命中以下 skill 之一：
   - `value_investing`
   - `answering_questions`
   - `receiving_share`
   - `tuya_smart_home`
   - `3d_print_automation`
2. **消息特征**：
   - 消息长度 > 200 字，且包含问号或“怎么/为什么/如何/建议”等咨询词。
   - 消息明确请求分析、建议、预测。
3. **上下文特征**：
   - 当前话题与前文话题明显跳转，需要整合多轮信息。

简单闲聊（`casual_chat`、`group_banter`、`handling_praise`、`handling_vent`）不强制触发 `think`，但模型仍可自主调用。

## 可观测性

Admin / 开发者需要能看到一个 tick 内的完整多轮轨迹，包括：

1. ReAct 循环中每轮 LLM 的输入/输出（含 `think` 工具调用）。
2. Feedback 的判断结果和原始文本。
3. Iterate 的输入/输出。
4. 最终返回给用户的 replies。

当前已有观测点：

| 字段/机制 | 当前记录内容 | 改造后记录内容 |
|---|---|---|
| `last_llm_messages` | 最后一次 LLM 请求的 messages | 最终 Self-Refine 后的完整 messages |
| `last_raw_response` | 最后一次 LLM 原始响应 | 最终 Iterate（或 ReAct）的原始响应 |
| `last_tool_calls` | 工具调用摘要 | 包含 `think` 工具在内的所有 tool_calls |
| `last_generation_trace` | 每轮请求/响应/工具执行 trace | 追加 Feedback 和 Iterate 的 trace |
| `tick_log`（SQLite） | 存 system prompt、user prompt、raw response、tool results、all_messages | 增加 `self_refine_applied` 标记、`feedback_decision`、`feedback_issues`、`iterate_count` |
| DebugLogger | tick 决策、发送结果 | 增加 ReAct 轮数、Self-Refine 阶段耗时 |

新增字段：

```python
self.last_self_refine_applied: bool = False
self.last_feedback_decision: str = ""  # "pass" / "fail" / ""
self.last_feedback_issues: List[str] = []
self.last_iterate_count: int = 0
self.last_feedback_raw: str = ""  # Feedback 原始输出
self.last_iterate_raw: str = ""  # Iterate 原始输出
```

`tick_log` 表新增字段：
- `self_refine_applied` (INTEGER): 是否启用 Self-Refine
- `feedback_decision` (TEXT): pass/fail/空
- `feedback_issues` (TEXT): JSON 序列化的问题列表
- `iterate_count` (INTEGER): Iterate 调用次数
- `react_round_count` (INTEGER): ReAct 工具调用轮数
- `think_tool_called` (INTEGER): 是否调用过 think 工具

实现时保证：即使 `ENABLE_SELF_REFINE=0`，这些字段也存在，只是值为空/0。

## 实现顺序

按以下顺序分阶段实现，每阶段可独立验证：

### Phase 1: 清理旧代码
1. 删除 `src/reply/generator.py` 中的 Hermes 相关代码。
2. 删除 `src/reply/generator.py` 中的两步推理代码。
3. 删除 `src/reply/session_memory.py` 中的 `bot_replies` 死代码。
4. 删除 `wechat_bot.py` 中的 `complex_llm_client` 传入。
5. 更新/删除受影响的测试文件。
6. 验证：现有单次推理测试全部通过。

### Phase 2: 改造 ReAct 生成
1. 注册 `think` 工具。
2. 在现有 ReAct 循环中处理 `think` 工具（不调用外部服务）。
3. 提高 `max_tokens` 到 10000，增加 `max_tool_calls=10` 限制。
4. 验证现有 `reasoning_content` 保留逻辑继续工作（当前代码第 438-439 行已有）。
5. 在生成阶段 user prompt 末尾追加格式指令。
6. 验证：简单消息正常回复，复杂消息能触发 `think` 工具。

### Phase 3: 接入 Self-Refine
1. 新增 `prompts/feedback.md` 和 `prompts/iterate.md`。
2. 实现 `_self_refine()` 和 `_iterate()`。
3. 实现多轮 Self-Refine 和总超时控制。
4. 验证：Feedback 能识别幻觉/格式错误并修正。

### Phase 4: 真实场景验证
1. 在测试群跑 20~30 条消息。
2. 测量延迟分布（P50/P95/P99）。
3. 人工评估回复质量和人设保持度。

## 测试计划

### 单元测试

1. `think` 工具正确注册和执行。
2. ReAct 循环在 tool_calls 达到上限后正确降级。
3. Feedback pass 时不走 Iterate。
4. Feedback fail 时 Iterate 被调用并返回新回复。
5. reasoning_content 在多轮 tool call 后正确回传。
6. 超时后返回上一阶段可用结果。
7. 环境变量关闭时切回单次推理。

### 集成测试

1. 真实 API 调用：投资类问题触发 think + search_memory。
2. 真实 API 调用：Feedback 检测到幻觉并修正。
3. 20s 总超时在慢网络下生效。
4. 关闭 `ENABLE_SELF_REFINE` 后行为与旧版本一致。

### 真实场景验证

在测试群跑 20~30 条消息，观察：
- 回复质量是否提升
- 延迟是否在 20s 内
- 是否出现人设漂移
- 是否出现 400 / 超时等错误

## 回滚策略

1. 环境变量 `ENABLE_REACT_TOOLS=0` / `ENABLE_SELF_REFINE=0` 可立即切回旧逻辑。
2. 如果线上出问题，修改 `.env` 后重启 bot 即可，无需改代码。
3. Git 保留旧版本分支，必要时 `git checkout` 回滚。

## 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| thinking + tool call 400 错误 | bot 不可用 | 修复 reasoning_content 回传；单测覆盖 |
| Self-Refine 破坏人设 | 回复变无聊 | Feedback prompt 强调保持人设；小样本验证 |
| think 工具滥用 | 延迟飙升 | max_tool_calls=10；超时控制 |
| Feedback 误判 | 好回复被改差 | 默认 1 轮 Iterate；可配置关闭 |
| 成本上涨 | 运营压力 | 统一模型保证缓存命中；可配置关闭 |
| 单元测试大量失败 | 开发阻塞 | Hermes/两步代码清理时同步更新测试 |

## 验收标准

- [ ] `ENABLE_SELF_REFINE=0` 时，行为与重构前一致。
- [ ] `ENABLE_SELF_REFINE=1` 时，投资/复杂问题能触发 `think` 工具。
- [ ] Feedback 能识别并修正明显幻觉/格式错误。
- [ ] 运行 100 条真实/模拟请求 benchmark，P95 总延迟 ≤ 20s，P99 ≤ 25s。
- [ ] benchmark 中 `think` 工具在复杂场景（投资/问答/长文本）触发率 ≥ 70%。
- [ ] 测试群小样本验证回复质量提升，人设未明显漂移。
- [ ] 所有现有测试通过，新增测试覆盖 ReAct + Self-Refine 路径。
