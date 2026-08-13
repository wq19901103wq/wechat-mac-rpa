# 强制排查流程（必须按步骤执行，禁止跳步）

> 遇到问题 → **先看日志** → **从日志反推** → **再碰代码**
> 
> 禁止行为：凭代码假设推断、凭记忆推断、"我觉得应该..."。

---

## Phase 1: 收集证据（先看日志，绝不假设）

1. **确认 Bot 运行状态**
   ```bash
   ps aux | grep "run_bot" | grep -v grep
   ```

2. **查看最新 runtime 日志**
   ```bash
   ls -lt data/logs/runtime_*.log | head -3
   tail -100 data/logs/runtime_$(date +%Y%m%d).log
   ```

3. **查看最新 debug 文件（含 LLM 调用详情）**
   ```bash
   ls -lt data/debug/tick_*.json | head -5
   ```

4. **提取关键字段**（用脚本，不手动翻）
   ```bash
   python3 -c "
   import json
   with open('data/debug/tick_XXX.json') as f:
       d = json.load(f)
   print('reply_llm_calls:', len(d.get('reply_llm_calls', [])))
   print('reply_tool_calls:', len(d.get('reply_tool_calls', [])))
   print('reply_raw_response:', repr(d.get('reply_raw_response', ''))[:200])
   "
   ```

---

## Phase 2: 定位根因（从日志反推，不碰代码）

### 2.1 工具是否注册？
**检查 `reply_generation_trace` 中 `llm_request` 的 `tools` 字段：**
```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('data/debug/tick_*.json'))[-10:]:
    d = json.load(open(f))
    for entry in d.get('reply_generation_trace', []):
        if entry.get('type') == 'llm_request':
            tools = entry.get('tools', [])
            print(f'{f.split(\"/\")[-1]}: tools={len(tools)} {[t.get(\"function\",{}).get(\"name\") for t in tools]}')
"
```
- **tools=0**：没传 tools → 检查 `force_no_tools` 或 `tool_registry`
- **tools=3 缺 search_memory**：`search_memory` 没注册 → 检查 `memory_engine` 初始化时机
- **tools=4 全有**：注册正确，看模型是否调用

### 2.2 模型是否调用了工具？
**检查 `reply_tool_calls`：**
```bash
python3 -c "
import json, glob
counts = {}
for f in glob.glob('data/debug/tick_*.json'):
    for tc in json.load(open(f)).get('reply_tool_calls', []):
        name = tc.get('tool_name', '?')
        counts[name] = counts.get(name, 0) + 1
for name, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {name}: {n}')
"
```
- **某个工具=0 从未被调用**：模型收到了但不选 → 改 schema 描述或 system prompt
- **某个工具被调用但结果为空**：工具执行逻辑有问题

### 2.3 模型输出了什么？
**检查 `reply_raw_response`：**
- 空回复？→ max_tokens 不够或 content 被截断
- JSON 但内容不对？→ prompt 引导问题
- 思考过程？→ system prompt 规则不清晰

---

## Phase 3: 验证假设（本地模拟运行）

**禁止直接改生产代码后让用户试。必须先本地验证。**

```bash
python3 -c "
import os
os.environ['DASHSCOPE_API_KEY'] = 'dummy'
from src.bot.wechat_bot import WeChatBot
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.tools import get_registry

class MockLLM:
    calls = []
    def chat(self, *args, **kwargs):
        self.calls.append(kwargs)
        return '{\"replies\": [\"test\"]}'

bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280, llm_client=MockLLM(), use_openclaw=False)
registry = get_registry()
print('Tools:', [t.name for t in registry.list_tools()])
"
```

---

## Phase 4: 修改代码（禁止硬编码/正则补丁）

### 红线
- **禁止**用正则/启发式方法修具体 case
- **禁止**为某一个用户/群写特殊逻辑
- **禁止**增大 max_chars 当万能药（先想清楚为什么需要这么大）

### 修改前必须问自己的 3 个问题
1. 这个修改对**所有用户**都适用吗？
2. 如果换一个人/群，还会出问题吗？
3. 这是修根因还是打补丁？

---

## 本次事故的复盘

| 错误 | 正确做法 |
|------|---------|
| 凭 `has_tools=true` 幻觉出"4个tools" | 看 `reply_generation_trace.llm_request.tools` 的实际内容 |
| 凭代码逻辑推断"应该注册了" | 先查日志确认实际运行时注册了几项 |
| 改完代码没验证就汇报 | 本地 MockLLM 验证 registry.list_tools() |
| 用正则预搜修"周远是谁"case | 应该修 `search_memory` 不被调用的根因 |
