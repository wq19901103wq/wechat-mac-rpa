# 架构文档：ReAct + Self-Refine

## 概述

回复生成系统基于 **ReAct（推理+行动）** 模式，加上 **Self-Refine（自检+修正）** 质量门。

```
收到消息 → persona + 上下文 + 工具 → ReAct 工具循环 → Self-Refine → 输出
```

## 数据流

```python
generate()
  ├── _system_prompt(persona.md + 工具列表)    ← 角色人设 + 可用工具
  ├── _build_tools_context()                   ← 工具缓存数据
  ├── _build_user_prompt()                     ← 聊天历史 + 上下文
  │
  ├── ReAct 工具循环 (max_tokens=10000)
  │   ├── think（可选）                        ← 自选深度思考
  │   ├── search_memory（可选）                ← 查记忆
  │   ├── web_search（可选）                   ← 搜网页
  │   └── 草拟回复 {"replies": [...]}
  │
  ├── _self_refine(messages, replies)          ← 质量门
  │   ├── Feedback → decision=pass → 原样输出
  │   └── Feedback → decision=fail → Iterate → 修正版
  │
  └── submit_to_judge + 返回
```

## 三阶段调用

### Call 1：生成（已有）

```python
active_llm.chat(
    messages=[
        {"role": "system", "content": persona.md + 工具描述},
        {"role": "user", "content": user_prompt},
        # ... 工具调用历史（如有）...
        {"role": "assistant", "content": '{"replies": [...]}'}
    ],
    tools=[search_memory, web_search, think, ...],
    max_tokens=10000,
)
```

参数：temperature 默认 0.7，max_tokens=10000

### Call 2：Feedback（新增）

在 Call 1 的 messages 基础上追加一条 user message，不传 tools。

```python
self.llm_client.chat(
    messages=messages + [{"role": "user", "content": "检查以上对话中 assistant 最终回复的质量。..."}],
    temperature=0.7, max_tokens=10000,
)
```

输出格式：
```json
{"decision": "pass"}
{"decision": "fail", "issues": ["具体问题1", "具体问题2"]}
```

评估标准：
1. 诚实性——每条回复内容在对话记录（含工具结果）中有出处吗？
2. 极简——1-3条？没有废话和复述？
3. 格式——有效 JSON？
4. 语气——自然像聊天？

### Call 3：Iterate（可选，仅 Feedback 发现问题时执行）

```python
self.llm_client.chat(
    messages=messages + [{"role": "user", "content": "基于以下反馈改进 assistant 的回复。..."}],
    temperature=0.7, max_tokens=10000,
)
```

输出格式：`{"replies": [...]}`

## 参数统一

所有 LLM 调用统一使用相同的参数：

| 参数 | 值 | 原因 |
|------|-----|------|
| temperature | 0.7（默认） | 与项目默认一致 |
| max_tokens | 10000 | 给 reasoning_content（thinking）留足空间，thinking 和输出共享此预算 |
| timeout | 30s（主生成）/ 600s（Hermes） | 保持现有 timeout 策略 |

## 前缀缓存优化

Feedback 和 Iterate 的 messages 以 Call 1 的完整 messages（system + user + tool 历史）为前缀。相同上下文的后缀追加请求可以命中 DeepSeek 的 prompt 缓存，只需计算新增部分。

## 注册工具

### think 工具

```python
self.tool_registry.register(
    name="think",
    description="在回复前停下来深入思考。用于需要深度推理、权衡多因素、分析意图的场景。"
                "此工具不获取新信息，只记录你的思考过程。在想清楚之前随时可以调用，次数不限。",
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

模型在工具循环中可以自行决定何时调用 think。简单闲聊不调，复杂场景先想再查、查完再想。

## 延迟预算

| 路径 | 额外调用 | 额外延迟 |
|------|----------|---------|
| 正常（~70%）：Feedback → pass | Feedback x1 | +~1-2s |
| 修正（~30%）：Feedback → fail → Iterate | Feedback x1 + Iterate x1 | +~3-5s |

## 相关文件

- `src/reply/generator.py` — 主要实现
- `src/utils/qwen_client.py` — LLM 客户端（含 thinking: enabled）
- `data/persona.md` — 角色私人人设（Git 忽略）
- `prompts/persona_mode_detection.md` — 模式检测版人设

## 历史

- 2026-07-05: 最终定稿。移除两阶段推理，改为 Self-Refine。所有 LLM 调用统一 max_tokens=10000。增加 think 工具。
