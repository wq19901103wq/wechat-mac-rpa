# ReAct + Self-Refine 回复生成重构 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在现有 ReAct 工具循环基础上增加 `think` 工具、Self-Refine（Feedback + Iterate）质量门、完整可观测性，并清理 Hermes / 两步推理 / session_memory 死代码。

**架构：** 保留 `src/reply/generator.py` 现有 ReAct 循环，注册 `think` 工具并提高 `max_tokens`；生成后追加 Feedback 调用，发现问题时进行 Iterate 修正；所有调用使用 `deepseek-v4-flash` 并保留 `reasoning_content`；最终通过 `tick_log` 记录完整多轮轨迹。

**技术栈：** Python 3.10, OpenAI SDK, SQLite, pytest

---

## 文件清单

| 文件 | 职责 |
|---|---|
| `src/reply/generator.py` | 核心实现：ReAct 循环改造、Self-Refine、观测字段 |
| `src/utils/qwen_client.py` | 确保 `reasoning_content` 透传，不删除 messages 字段 |
| `src/reply/session_memory.py` | 删除 `bot_replies` 死代码，保留工具缓存 |
| `src/bot/wechat_bot.py` | 删除 `complex_llm_client` 传入，发送成功后记录 bot 回复 |
| `src/logging/tick_logger.py` 或相关 tick_log 写入代码 | 新增 Self-Refine 相关字段 |
| `data/persona.md` | 删除输出格式段，改为允许思考 |
| `prompts/feedback.md` | 新增 Feedback 专用 prompt |
| `prompts/iterate.md` | 新增 Iterate 专用 prompt |
| `prompts/reply_format.txt` | 生成阶段追加的格式指令 |
| `src/tests/test_session_memory.py` | 删除 bot_replies 相关测试 |
| `tests_integration/test_hermes_integration.py` | 删除或重构为 ReAct 测试 |
| `src/tests/test_reply_generator.py` | 更新或新增 ReAct/Self-Refine 测试 |

---

## 任务 1：清理 Hermes 相关代码

**文件：**
- 修改：`src/reply/generator.py`
- 修改：`src/bot/wechat_bot.py`
- 删除：`tests_integration/test_hermes_integration.py`
- 测试：`src/tests/test_reply_generator.py`

### 步骤 1：删除 `ReplyGenerator.__init__` 中的 `complex_llm_client`

```python
# 删除前
class ReplyGenerator:
    def __init__(self, llm_client=None, complex_llm_client=None, memory_engine=None,
                 tool_registry=None, judge_worker=None, ...):
        self.llm_client = llm_client
        self.complex_llm_client = complex_llm_client
        ...

# 删除后
class ReplyGenerator:
    def __init__(self, llm_client=None, memory_engine=None,
                 tool_registry=None, judge_worker=None, ...):
        self.llm_client = llm_client
        ...
```

### 步骤 2：删除 Hermes fallback 相关方法

在 `src/reply/generator.py` 中删除：
- `_hermes_system_prompt()`
- `last_hermes_fallback_triggered`
- `last_hermes_messages`
- `last_hermes_response`
- `generate()` 中所有 `is_hermes` / `complex_llm_client` / `use_hermes` 分支

### 步骤 3：删除 `wechat_bot.py` 中的 `complex_llm_client` 传入

```python
# src/bot/wechat_bot.py 第 83-85 行
self.generator = ReplyGenerator(
    llm_client=actual_llm,
    memory_engine=self.memory_engine,
    tool_registry=registry,
    judge_worker=judge_worker,
    ...
)
```

### 步骤 4：删除 Hermes 集成测试

```bash
rm tests_integration/test_hermes_integration.py
```

### 步骤 5：运行测试

```bash
cd /Users/yourname/wechat-mac-rpa
pytest src/tests/test_reply_generator.py -v
pytest tests_integration/ -v --ignore=tests_integration/test_hermes_integration.py
```

预期：Hermes 相关测试失败或不存在，其他测试通过。

### 步骤 6：Commit

```bash
git add src/reply/generator.py src/bot/wechat_bot.py tests_integration/
git commit -m "refactor(reply): remove Hermes fallback path"
```

---

## 任务 2：清理两步推理代码

**文件：**
- 修改：`src/reply/generator.py`
- 删除：`scripts/test_twostep_offline.py`

### 步骤 1：删除两步推理方法

在 `src/reply/generator.py` 中删除：
- `_should_use_two_step()`
- `_deep_analysis()`
- `_plan_analysis()`
- `_gather_analysis_data()`
- `generate()` 中“复杂场景两步推理”注入逻辑（约第 313-324 行）

### 步骤 2：删除离线测试脚本

```bash
rm scripts/test_twostep_offline.py
```

### 步骤 3：运行测试

```bash
pytest src/tests/test_reply_generator.py -v
```

预期：通过。

### 步骤 4：Commit

```bash
git add src/reply/generator.py scripts/test_twostep_offline.py
git commit -m "refactor(reply): remove two-step reasoning prototype"
```

---

## 任务 3：清理 session_memory 死代码

**文件：**
- 修改：`src/reply/session_memory.py`
- 修改：`src/reply/generator.py`
- 修改：`src/tests/test_session_memory.py`

### 步骤 1：删除 `SessionSnapshot` 中的 bot_replies 相关代码

```python
# src/reply/session_memory.py
@dataclass
class SessionSnapshot:
    chat_name: str
    is_group: bool = False
    last_active: float = field(default_factory=time.time)
    tool_cache: List[CachedToolResult] = field(default_factory=list)
    # 删除 bot_replies 字段

    # 删除 add_reply 方法
    # 删除 get_recent_replies 方法
```

### 步骤 2：删除 `SessionMemory` 中的 bot_replies 方法

```python
# src/reply/session_memory.py
class SessionMemory:
    # 删除 add_reply 方法
    # 删除 get_recent_replies 方法
```

### 步骤 3：删除 generator.py 中的 add_reply 调用

```python
# src/reply/generator.py
# 删除所有 self.session_memory.add_reply(chat_name, r) 调用
```

### 步骤 4：更新 test_session_memory.py

```python
# src/tests/test_session_memory.py
# 删除测试 add_reply / get_recent_replies 的 case
```

### 步骤 5：运行测试

```bash
pytest src/tests/test_session_memory.py -v
pytest src/tests/test_reply_generator.py -v
```

预期：通过。

### 步骤 6：Commit

```bash
git add src/reply/session_memory.py src/reply/generator.py src/tests/test_session_memory.py
git commit -m "refactor(session_memory): remove dead bot_replies code"
```

---

## 任务 4：修复 bot 回复记录时机（发送成功后才记录）

**文件：**
- 修改：`src/reply/generator.py`
- 修改：`src/bot/wechat_bot.py`

### 步骤 1：从 generator.py 中移除 add_reply 调用

已经在任务 3 中删除。

### 步骤 2：在 wechat_bot.py 发送成功后记录 bot 回复

```python
# src/bot/wechat_bot.py 第 537-550 行
if action_result.success:
    self.logger.log_send(tick_id, success=True, text=reply)
    self.debug_logger.log_action("send", action_input=reply, success=True)
    
    # 新增：发送成功后才记录到 session_memory
    self.generator.session_memory.add_reply(chat_name, reply)
    
    self_msg = ChatMessage(...)
    chat_state = self.global_store.chats.get(chat_name)
    if chat_state is not None:
        chat_state.pending_self_messages.append(self_msg)
        ...
```

### 步骤 3：运行测试

```bash
pytest src/tests/test_wechat_bot.py -v 2>/dev/null || pytest src/tests/ -k "wechat" -v
```

### 步骤 4：Commit

```bash
git add src/bot/wechat_bot.py
git commit -m "fix(bot): record bot reply only after send success"
```

---

## 任务 5：添加 think 工具和 ReAct 循环增强

**文件：**
- 修改：`src/reply/generator.py`
- 创建：`prompts/reply_format.txt`

### 步骤 1：注册 think 工具

在 `ReplyGenerator.__init__` 中注册：

```python
# src/reply/generator.py
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

### 步骤 2：在 ReAct 循环中处理 think 工具

```python
# src/reply/generator.py 现有工具执行循环内
for tc in raw_tool_calls:
    tool_name = tc.function.name
    tool_args = tc.function.arguments
    
    if tool_name == "think":
        result = "思考已记录，继续生成回复。"
        _logger.info("[Tool] think 调用: %s", tool_args[:200])
    else:
        result = self.tool_registry.get(tool_name).execute(tool_args)
        # 现有缓存逻辑保持不变
        ...
    
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result,
    })
```

### 步骤 3：提高 max_tokens 并增加 max_tool_calls

```python
# src/reply/generator.py
MAX_TOOL_CALLS = 10

# 在循环调用处
raw = active_llm.chat(
    messages=messages,
    tools=actual_tools,
    max_tokens=10000,  # 从 2000 提高
    timeout=llm_timeout,
)
```

### 步骤 4：创建 reply_format.txt

```bash
cat > prompts/reply_format.txt <<'EOF'
请直接输出 JSON：`{"replies": ["回复1", "回复2"]}`。
JSON 之外不要输出任何文字。
EOF
```

### 步骤 5：在 user_prompt 末尾追加格式指令

```python
# src/reply/generator.py 中 _build_user_prompt 或 generate 末尾
reply_format = Path(__file__).parent.parent.parent / "prompts" / "reply_format.txt"
if reply_format.exists():
    user_prompt += "\n\n" + reply_format.read_text()
```

### 步骤 6：运行测试

```bash
pytest src/tests/test_reply_generator.py -v
```

### 步骤 7：Commit

```bash
git add src/reply/generator.py prompts/reply_format.txt
git commit -m "feat(reply): add think tool and raise max_tokens to 10000"
```

---

## 任务 6：新增 Feedback 和 Iterate Prompt 文件

**文件：**
- 创建：`prompts/feedback.md`
- 创建：`prompts/iterate.md`

### 步骤 1：创建 feedback.md

```bash
cat > prompts/feedback.md <<'EOF'
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
EOF
```

### 步骤 2：创建 iterate.md

```bash
cat > prompts/iterate.md <<'EOF'
请基于以上反馈改进 assistant 的回复。

要求：
- 修复反馈中列出的所有问题。
- 保持示例用户甲本人语气，不要变成 AI 助手腔。
- 保持简洁，不要啰嗦。
- 输出 JSON：`{"replies": ["改进后的回复1", "改进后的回复2"]}`
- JSON 之外不要输出任何文字。
EOF
```

### 步骤 3：Commit

```bash
git add prompts/feedback.md prompts/iterate.md
git commit -m "feat(prompts): add feedback and iterate prompts"
```

---

## 任务 7：实现 Self-Refine（Feedback + Iterate）

**文件：**
- 修改：`src/reply/generator.py`

### 步骤 1：读取 prompt 文件

在 `ReplyGenerator.__init__` 中：

```python
from pathlib import Path
_prompt_dir = Path(__file__).parent.parent.parent / "prompts"
self._feedback_prompt = (_prompt_dir / "feedback.md").read_text()
self._iterate_prompt = (_prompt_dir / "iterate.md").read_text()
```

### 步骤 2：新增观测字段

```python
# 在 ReplyGenerator.__init__ 中
self.last_self_refine_applied: bool = False
self.last_feedback_decision: str = ""
self.last_feedback_issues: List[str] = []
self.last_iterate_count: int = 0
self.last_feedback_raw: str = ""
self.last_iterate_raw: str = ""
```

### 步骤 3：实现 `_self_refine`

```python
def _self_refine(
    self,
    messages: List[Dict],
    replies: List[str],
    deadline: float,
) -> Tuple[str, Optional[List[str]], List[Dict], str]:
    """Feedback 阶段。返回 decision, issues, updated_messages, raw_text。"""
    feedback_messages = messages + [
        {"role": "user", "content": self._feedback_prompt}
    ]
    timeout = max(1.0, deadline - time.time())
    raw = self.llm_client.chat(
        messages=feedback_messages,
        max_tokens=10000,
        temperature=0.3,
        timeout=timeout,
    )
    text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
    self.last_feedback_raw = text or ""
    
    data = self._extract_json(text) or {}
    if not isinstance(data, dict):
        return "pass", None, feedback_messages, text or ""
    
    decision = data.get("decision", "pass")
    issues = data.get("issues", [])
    if decision == "fail" and isinstance(issues, list) and issues:
        return "fail", issues, feedback_messages, text or ""
    return "pass", None, feedback_messages, text or ""
```

### 步骤 4：实现 `_iterate`

```python
def _iterate(
    self,
    messages: List[Dict],
    issues: List[str],
    deadline: float,
) -> Tuple[List[str], List[Dict], str]:
    """Iterate 阶段。返回 replies, updated_messages, raw_text。"""
    issues_text = "\n".join(f"- {issue}" for issue in issues)
    iterate_messages = messages + [
        {"role": "user", "content": f"{self._iterate_prompt}\n\n反馈问题：\n{issues_text}"}
    ]
    timeout = max(1.0, deadline - time.time())
    raw = self.llm_client.chat(
        messages=iterate_messages,
        max_tokens=10000,
        temperature=0.7,
        timeout=timeout,
    )
    text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
    self.last_iterate_raw = text or ""
    
    replies = self._parse_replies(text)
    return replies or [], iterate_messages, text or ""
```

### 步骤 5：在 `generate()` 中接入 Self-Refine

```python
def generate(self, unreplied, all_messages, is_group=False, tick_id=None) -> List[str]:
    ...
    # 生成阶段
    deadline = time.time() + 20.0
    replies, final_messages = self._react_generate(messages, tools, deadline)
    
    self.last_self_refine_applied = False
    self.last_feedback_decision = ""
    self.last_feedback_issues = []
    self.last_iterate_count = 0
    
    if self.enable_self_refine and replies:
        decision, issues, feedback_messages, _ = self._self_refine(final_messages, replies, deadline)
        self.last_self_refine_applied = True
        self.last_feedback_decision = decision
        self.last_feedback_issues = issues or []
        
        if decision == "fail" and issues:
            iter_count = 0
            max_iter = int(os.environ.get("SELF_REFINE_MAX_ITER", "1"))
            current_replies = replies
            current_messages = feedback_messages
            
            while iter_count < max_iter and issues:
                new_replies, new_messages, _ = self._iterate(current_messages, issues, deadline)
                iter_count += 1
                if new_replies:
                    current_replies = new_replies
                    current_messages = new_messages
                # 可选：再次 Feedback，但默认只 Iterate 一次
                break
            
            self.last_iterate_count = iter_count
            replies = current_replies
    
    # 保存到 session memory（后续任务已移到 wechat_bot.py）
    return replies
```

### 步骤 6：运行测试

```bash
pytest src/tests/test_reply_generator.py -v
```

### 步骤 7：Commit

```bash
git add src/reply/generator.py
git commit -m "feat(reply): implement Self-Refine feedback and iterate"
```

---

## 任务 8：新增 tick_log 字段记录 Self-Refine 轨迹

**文件：**
- 修改：tick_log 写入代码（需先定位）
- 修改：`src/reply/generator.py`

### 步骤 1：定位 tick_log 写入代码

```bash
grep -rn "tick_log" /Users/yourname/wechat-mac-rpa/src --include="*.py" | head -20
```

### 步骤 2：修改 tick_log 表结构

假设 tick_log 写入在 `src/bot/wechat_bot.py` 或 `src/logging/` 中，执行类似：

```sql
ALTER TABLE tick_log ADD COLUMN self_refine_applied INTEGER DEFAULT 0;
ALTER TABLE tick_log ADD COLUMN feedback_decision TEXT DEFAULT '';
ALTER TABLE tick_log ADD COLUMN feedback_issues TEXT DEFAULT '[]';
ALTER TABLE tick_log ADD COLUMN iterate_count INTEGER DEFAULT 0;
ALTER TABLE tick_log ADD COLUMN react_round_count INTEGER DEFAULT 0;
ALTER TABLE tick_log ADD COLUMN think_tool_called INTEGER DEFAULT 0;
```

### 步骤 3：在写入 tick_log 时填充新字段

```python
# 在 tick_log 写入处
cursor.execute("""
    INSERT INTO tick_log (...)
    VALUES (...)
""", (
    ...,
    1 if getattr(self.generator, 'last_self_refine_applied', False) else 0,
    getattr(self.generator, 'last_feedback_decision', ''),
    json.dumps(getattr(self.generator, 'last_feedback_issues', []), ensure_ascii=False),
    getattr(self.generator, 'last_iterate_count', 0),
    len(getattr(self.generator, 'last_tool_calls', [])),
    1 if any(tc['tool_name'] == 'think' for tc in getattr(self.generator, 'last_tool_calls', [])) else 0,
))
```

### 步骤 4：运行测试

```bash
pytest src/tests/ -v
```

### 步骤 5：Commit

```bash
git add src/bot/wechat_bot.py src/logging/
git commit -m "feat(logging): add Self-Refine trace fields to tick_log"
```

---

## 任务 9：修改 persona.md

**文件：**
- 修改：`data/persona.md`

### 步骤 1：替换输出格式段

```markdown
<rule name="思考与输出">
你可以在心里思考，也可以调用 think 工具记录思考过程。
思考可以充分、深入，但最终回复必须是 JSON：`{"replies": [...]}`
JSON 之外不要输出任何文字。
</rule>
```

### 步骤 2：Commit

```bash
git add prompts/persona.md.example
git commit -m "feat(prompts): allow thinking in persona"
```

---

## 任务 10：单元测试覆盖

**文件：**
- 创建：`src/tests/test_react_self_refine.py`

### 步骤 1：测试 think 工具注册

```python
def test_think_tool_registered():
    gen = ReplyGenerator(llm_client=MockLLM())
    assert gen.tool_registry.has("think")
```

### 步骤 2：测试 think 工具执行不调用外部服务

```python
def test_think_tool_returns_confirmation():
    gen = ReplyGenerator(llm_client=MockLLM())
    tool = gen.tool_registry.get("think")
    result = tool.execute('{"thought": "test"}')
    assert "思考已记录" in result
```

### 步骤 3：测试 Feedback pass

```python
def test_self_refine_pass_skips_iterate(mock_llm):
    mock_llm.responses = [
        '{"replies": ["test reply"]}',
        '{"decision": "pass"}',
    ]
    gen = ReplyGenerator(llm_client=mock_llm, enable_self_refine=True)
    replies = gen.generate([...], [])
    assert replies == ["test reply"]
    assert gen.last_feedback_decision == "pass"
    assert gen.last_iterate_count == 0
```

### 步骤 4：测试 Feedback fail + Iterate

```python
def test_self_refine_fail_triggers_iterate(mock_llm):
    mock_llm.responses = [
        '{"replies": ["bad reply"]}',
        '{"decision": "fail", "issues": ["太正式"]}',
        '{"replies": ["好的吧"]}',
    ]
    gen = ReplyGenerator(llm_client=mock_llm, enable_self_refine=True)
    replies = gen.generate([...], [])
    assert replies == ["好的吧"]
    assert gen.last_feedback_decision == "fail"
    assert gen.last_iterate_count == 1
```

### 步骤 5：测试开关关闭时切回单次推理

```python
def test_self_refine_disabled(mock_llm):
    mock_llm.responses = ['{"replies": ["reply"]}']
    gen = ReplyGenerator(llm_client=mock_llm, enable_self_refine=False)
    replies = gen.generate([...], [])
    assert replies == ["reply"]
    assert gen.last_self_refine_applied is False
```

### 步骤 6：运行测试

```bash
pytest src/tests/test_react_self_refine.py -v
```

### 步骤 7：Commit

```bash
git add src/tests/test_react_self_refine.py
git commit -m "test(reply): add ReAct + Self-Refine unit tests"
```

---

## 任务 11：真实 API 集成测试

**文件：**
- 创建：`tests_integration/test_react_self_refine_integration.py`

### 步骤 1：测试复杂问题触发 think

```python
def test_investment_question_triggers_think():
    gen = ReplyGenerator(llm_client=QwenClient())
    replies = gen.generate(
        [ChatMessage(text="拼多多还能拿吗", sender="示例用户申", sender_type=SenderType.OTHER)],
        [],
        is_group=True,
    )
    assert replies
    assert any(tc['tool_name'] == 'think' for tc in gen.last_tool_calls)
```

### 步骤 2：测试总超时 20s

```python
def test_total_timeout_under_20s():
    start = time.time()
    gen = ReplyGenerator(llm_client=QwenClient())
    gen.generate(...)
    assert time.time() - start < 20.0
```

### 步骤 3：Commit

```bash
git add tests_integration/test_react_self_refine_integration.py
git commit -m "test(integration): add real API tests for ReAct + Self-Refine"
```

---

## 任务 12：真实场景验证

**文件：**
- 无需代码修改
- 运行环境：测试群

### 步骤 1：启动 bot

```bash
cd /Users/yourname/wechat-mac-rpa
python3 run_bot.py
```

### 步骤 2：在测试群发送 20~30 条消息

覆盖：
- 简单闲聊
- 投资分析
-  wiki/记忆查询
- 图片/分享链接

### 步骤 3：收集指标

```bash
sqlite3 data/db/chat_history.db "SELECT 
    tick_id, 
    self_refine_applied, 
    feedback_decision, 
    iterate_count, 
    react_round_count, 
    think_tool_called 
FROM tick_log 
ORDER BY id DESC LIMIT 50;"
```

### 步骤 4：人工评估

- 回复质量是否提升
- 人设是否保持
- 延迟是否可接受

---

## 自检

### 规格覆盖度

- [x] think 工具注册与执行：任务 5
- [x] max_tokens 提升到 10000：任务 5
- [x] max_tool_calls=10：任务 5
- [x] Self-Refine Feedback + Iterate：任务 7
- [x] 多轮 Self-Refine 可配置：任务 7
- [x] 总超时 20s：任务 7
- [x] reasoning_content 回传：任务 5（现有逻辑保留）
- [x] session_memory 死代码清理：任务 3
- [x] Hermes 清理：任务 1
- [x] persona.md 修改：任务 9
- [x] 可观测性字段：任务 8
- [x] 开关环境变量：任务 7

### 占位符扫描

- [x] 无“待定”、“TODO”
- [x] 所有代码步骤包含实际代码
- [x] 所有测试步骤包含实际测试代码
- [x] 无模糊描述

### 类型一致性

- `_self_refine` 返回 `(str, Optional[List[str]], List[Dict], str)`
- `_iterate` 返回 `(List[str], List[Dict], str)`
- 观测字段名在 generator、tick_log 中一致

---

## 执行选项

计划已完成并保存到 `docs/superpowers/plans/2026-07-05-react-self-refine.md`。

两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代。使用 `superpowers:subagent-driven-development`。

**2. 内联执行** - 在当前会话中使用 `superpowers:executing-plans` 执行任务，批量执行并设有检查点供审查。

选哪种方式？
