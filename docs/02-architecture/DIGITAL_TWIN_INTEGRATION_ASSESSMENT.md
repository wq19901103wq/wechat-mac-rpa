# WeChat Digital Twin → wechat-mac-rpa 集成评估 v2

> 更新：2026-05-22 | 决策：以 digital-twin 为准，合并系统 prompt、benchmark、Judge

---

## 1. 决策总结

| 决策点 | 方案 |
|--------|------|
| 系统 prompt | **以 digital-twin 为准**（"你是林岚本人"），融合 wechat-mac-rpa 的工具调用 |
| 回复格式 | 统一为 wechat-mac-rpa 的 `{"replies": [...]}` |
| 检索方案 | 先用 TF-IDF（54MB），验证后升级 BGE Dense |
| 索引/数据 | 放 `data/` 目录（已 gitignore），不上传 git |
| Benchmark | 合并双方 case，存入 `cases.db` |
| Judge 标准 | **以 digital-twin 的对抗评测标准为准** |
| 子项目位置 | `wechat-mac-rpa/digital-twin/` |

---

## 2. 双项目对比

### 2.1 系统 Prompt 对比

| 维度 | wechat-mac-rpa | digital-twin |
|------|--------------|-------------|
| 身份 | "不爱说话，林岚的小号/分身" | "你是林岚本人" |
| 语气 | casual、傲娇、不用"您" | 极简（10-15字）、短句连发、语气词指纹 |
| 工具 | search_memory, web_search, stock_query, browse_url, get_weather | **无** |
| 规则 | 私聊必回、禁止敷衍、禁止编造、纠正后认错 | 上下文优先、忽略案例事实、参考案例风格 |
| 格式 | `{"replies": ["msg1"]}` | `"msg"` 或 `["msg1", "msg2"]` |
| 检索增强 | 无（纯 prompt + wiki） | TF-IDF/BGE 向量检索 + LLM Rerank |
| few-shot | 无 | 动态注入：检索 3 个相似历史对话案例 |

### 2.2 回复生成流程对比

```
wechat-mac-rpa:
  ChatMessage → build_user_prompt → build_system_prompt → LLM → parse_replies

digital-twin:
  message → enrich_query → vector_search(10) → LLM_rerank(3) 
  → build_prompt(few_shot + wiki) → LLM → parse_reply
```

### 2.3 Benchmark 对比

| | wechat-mac-rpa | digital-twin |
|------|--------------|-------------|
| case 数 | P0:27, P2:25, P4:29, Judge:12 | 50 个对抗 case |
| 类型 | 工具决策、回复质量、搜索召回 | 风格一致性、幻觉、知识边界 |
| Judge 方式 | 7 维度 LLM 评分 | LLM-as-Judge（step4_judge.py） |
| 存储 | `cases.db` SQLite | `adversarial_test_cases_v2.json` |

---

## 3. Prompt 合并方案

### 3.1 合并后 system prompt（草案）

```
你是林岚本人。用户不是在跟AI聊天，是在微信上给林岚发消息。

## 说话风格
- 每条消息很短（10-15字），经常一次发2-3条连发
- 高频语气词：哈、吧、啊、哈哈哈、呢、hhh、哇、呀
- 不用"您"，不客套，不解释，不铺垫
- 风格：casual、略带傲娇、适当幽默反转

## 可用工具
- search_memory(query)：搜索本地长期记忆
- get_current_time：获取当前时间
- get_weather(city, date)：查询天气
- web_search(query)：搜索网页
- browse_url(url)：浏览链接
- stock_query(stock_code)：查询股票

【不调用 search_memory 的场景】
- 纯情绪/打招呼（哈哈哈、晚安、谢谢）→ 不调用工具
- 用户发链接 → browse_url，不是 search_memory
- 用户搜网页 → web_search，不是 search_memory
- 用户陈述句（"我也在XX上班"）→ 不调用工具

## 输出格式
{"replies": ["回复1", "回复2"]}  ← JSON，不是 markdown

## 规则
1. 私聊必须回复
2. 禁止敷衍词：收到、好的、嗯、OK、1
3. 被纠正后先承认错误，不能继续编造
4. 不知道就说不知道，严禁编造数字/日期/人名

## 上下文优先原则
当前对话的上下文是了解当前情况的唯一来源。检索案例来自其他对话，
参考语气和风格，但忽略案例中的具体事实。

## 参考案例（风格参考，不要照搬事实）
{dynamic_few_shot}
```

### 3.2 融合变更点

| 变更 | 说明 |
|------|------|
| 身份改为 "林岚本人" | digital-twin 标准 |
| 删除 "不爱说话"、"小号/分身" | 简化身份 |
| 保留工具定义 | wechat-mac-rpa 需要 |
| 保留不调用规则 | P0 benchmark 驱动 |
| 保留输出格式 JSON | wechat-mac-rpa 格式 |
| 新增语气词指纹 | digital-twin 标准 |
| 新增 {dynamic_few_shot} | 检索增强 |
| 新增上下文优先原则 | digital-twin 标准 |

---

## 4. ReplyGenerator 改造

### 4.1 当前架构

```
ReplyGenerator.generate(unreplied, all_messages, is_group)
  → _system_prompt()      # 固定 prompt
  → _build_user_prompt()   # wiki + history
  → _route_skills()        # skill 路由
  → LLM.chat(messages, tools)
  → _parse_replies()       # JSON 解析
```

### 4.2 改造后架构

```
DigitalTwinReplyGenerator.generate(unreplied, all_messages, is_group)
  → _search_similar()      # 向量检索历史对话（新增）
  → _llm_rerank()          # LLM 重排序 top-3（新增）
  → _system_prompt()       # 数字人 prompt + {dynamic_few_shot}
  → _build_user_prompt()   # wiki + history（保留）
  → _route_skills()        # skill 路由（保留）
  → LLM.chat(messages, tools)  # 保留
  → _parse_replies()       # JSON 解析（兼容数字人格式）
```

### 4.3 回复格式兼容

```python
# digital-twin 输出格式：
"直接回复文本"
["连发1", "连发2"]

# wechat-mac-rpa 格式：
{"replies": ["回复1", "回复2"]}

# 兼容方法：
# 1. 先尝试解析 {"replies": [...]}
# 2. 失败则尝试 ["msg1", "msg2"]
# 3. 失败则当作单条文本 "msg"
```

---

## 5. Benchmark 合并

### 5.1 50 个对抗 case 入库

```sql
-- 新表
CREATE TABLE benchmark_adversarial_cases (
    id TEXT PRIMARY KEY,        -- tc_0022_adv
    query TEXT NOT NULL,        -- 用户消息
    sender TEXT,                -- 发送者
    chat_type TEXT,             -- single/group
    ground_truth TEXT,          -- 期望回复
    context_json TEXT,          -- 对话上下文
    category TEXT,              -- hallucination/style/knowledge_boundary/...
    severity TEXT,              -- easy/medium/hard
    original_query TEXT,        -- 原始 query（leak-free 版本）
    enabled INTEGER DEFAULT 1
);
```

### 5.2 统一的 Judge 标准

**以 digital-twin 为准** — 原因是 digital-twin 的评测标准更贴近"像不像真人"：

| 维度 | old Judge | new Judge (digital-twin aligned) |
|------|-----------|------|
| 幻觉控制 | 编造事实 | 同，但区分"在已知事实上夸张（OK）" vs "凭空编造（NOT OK）" |
| 风格一致性 | 逼格语气 | **新增：语气词指纹命中率、短句连发模式** |
| 记忆召回 | 是否调工具 | 同 |
| 上下文理解 | 是否理解意图 | **新增：是否错误引用了检索案例中的事实** |
| 个性一致性 | 第一人称 | 同 |

### 5.3 合并后 benchmark 全景

| # | Benchmark | cases | 来源 |
|---|-----------|-------|------|
| P0 | Tool Decision | 27 | wechat-mac-rpa |
| P1 | OCR Quality | 33 | wechat-mac-rpa |
| P2 | Reply Quality | 25 | wechat-mac-rpa |
| P4 | Memory Search | 29 | wechat-mac-rpa |
| P5 | Unread Badge | 23 | wechat-mac-rpa |
| Judge | Judge Quality | 12 | 生产 badcase |
| **P6** | **Adversarial** | **50** | **digital-twin** |
| Stability | Reply Stability | 12 | 生产 prompt |

---

## 6. 需要解决的问题

### 6.1 路径依赖

```python
# 当前硬编码 → 改为相对
Path("/Users/yourname/wechat-digital-twin/...")
→ Path(__file__).parent.parent / "data" / "vector_indexes" / "..."

# .env 共用 → 已经 OK
Path("/Users/yourname/wechat-mac-rpa/.env")
→ Path(__file__).parent.parent.parent / ".env"
```

### 6.2 索引文件

| 索引 | 大小 | 用途 | 存放位置 |
|------|------|------|---------|
| `vector_index_messages.pkl` | 54 MB | TF-IDF 消息级索引（V2） | `data/vector_indexes/` |
| `vector_index_dense_messages.pkl` | 1.7 GB | BGE Dense 索引（V4） | `data/vector_indexes/` |
| `vector_index_hybrid.pkl` | 1.8 GB | Hybrid 索引 | `data/vector_indexes/` |

全部在 `data/` 下，已在 `.gitignore` 中。

### 6.3 检索延迟

TF-IDF 模式：+0.5-1s（推荐先用）
BGE Dense 模式：+2-3s（需要 275MB 模型 + 1.7GB 索引加载进内存）

### 6.4 MemoryEngine 重复注入

- digital-twin 的 `_build_prompt()` 注入了 wiki 记忆
- wechat-mac-rpa 的 `_build_user_prompt()` 也注入
- **解决**：集成后只保留 ReplyGenerator 的记忆注入，digital-twin 检索层只提供 `{dynamic_few_shot}` 部分

### 6.5 工具调用 + 短句风格冲突

digital-twin 风格是"短句连发，不解释"，但工具调用需要 LLM 输出 function calling JSON。这个组合可能让 LLM 困惑。

**缓解**：在 few-shot 案例中不要包含工具调用的例子，让 LLM 自然地用短句风格回复，只在需要时调用工具。

### 6.6 索引过期

当前索引是静态快照，新聊天记录不会自动进入。

**解决**：
- 每周从 WeFlow 导出新消息 → 增量更新索引
- 脚本：`python digital-twin/scripts/update_index.py`
- 增量 embedding 追加到现有索引文件

### 6.7 digital-twin 缺少群聊支持

digital-twin 的 `reply_with_context()` 接受 `chat_type` 参数，但 `_build_prompt` 中对群聊的处理较简单（仅加权不同 chat_type 的检索结果）。群聊 @检测、多人对话上下文等需要增强。

### 6.8 回复生成器入口统一

当前有两个入口：
```python
# wechat-mac-rpa: wechat_bot.py
replies = self.generator.generate(to_reply, all_messages, is_group=is_group)

# digital-twin: rpa_bot_dense_message_level.py
reply = bot.reply_with_context(message, history, sender_name, chat_type)
```

参数格式不同。需要统一为 wechat-mac-rpa 的 `ChatMessage` 格式，在 `generate()` 内部调用 digital-twin 的检索逻辑。

---

## 7. 实施计划

### Phase 1: 文件迁移（0.5 天）

- [ ] 复制 `outputs/rpa_integration/` 到 `digital-twin/`
- [ ] 复制 `models/bge-small-zh-v1.5/` 到 `digital-twin/models/`（gitignore）
- [ ] 复制索引文件到 `data/vector_indexes/`
- [ ] 修复所有硬编码路径
- [ ] 验证 `DigitalTwinBot` 可初始化

### Phase 2: Prompt 融合 + ReplyGenerator 改造（1 天）

- [ ] 合并 system prompt（以 digital-twin 为准 + 工具）
- [ ] 新建 `DigitalTwinReplyGenerator`，继承现有 `ReplyGenerator`
- [ ] 集成检索步骤到 `generate()` 流程
- [ ] 兼容 digital-twin 回复格式
- [ ] `_parse_replies()` 兼容三种格式

### Phase 3: Benchmark 合并（0.5 天）

- [ ] 50 个对抗 case 转换为 `benchmark_adversarial_cases` 表格式
- [ ] 迁移脚本入库
- [ ] 更新 `run_daily_benchmark.py` 加载 P6
- [ ] 更新 Judge 维度（新增风格一致性）

### Phase 4: 验证 + 上线（1 天）

- [ ] 用现有 stability benchmark 对比新旧回复质量
- [ ] A/B 测试：old prompt vs new prompt 回复
- [ ] 确认工具调用不退化
- [ ] 灰度上线

---

## 8. 其他注意事项

### 8.1 style_profile.json

digital-twin 有 `style_profile.json`，记录了每个 sender 的平均回复长度。这个数据来自历史聊天分析，集成后可以用于个性化回复长度。

### 8.2 索引构建依赖 WeFlow

当前索引从 WeFlow 导出的聊天记录构建。如果没有 WeFlow 连接，索引无法更新。需要确保 WeFlow pipeline 正常工作。

### 8.3 BGE 模型 ONNX 导出

digital-twin 有 `export_bge_onnx.py` 脚本，可以将 BGE 模型导出为 ONNX 格式（更小、更快、无需 torch）。这对减少依赖有帮助。

### 8.4 检索案例的"事实污染"风险

digital-twin prompt 已强调"忽略案例事实"，但 LLM 不一定遵守。需要在 Judge 中专门检测：回复中是否出现了检索案例中的具体事实（人名、数字、事件）而没有在当前对话中验证过。
