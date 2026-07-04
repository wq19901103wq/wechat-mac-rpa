# Digital Twin 迁移 Spec — 模块级合并方案

> 原则：冲突以 digital-twin 为准 | digital-twin 缺的从 wechat-mac-rpa 补 | 不改 RPA 感知/行动层

---

## 1. 模块对照表

| 模块 | digital-twin | wechat-mac-rpa | 合并方向 |
|------|-------------|--------------|---------|
| **系统 Prompt** | ✅ "你是王芊本人" | "不爱说话/小号/分身" | → DT |
| **回复风格** | ✅ 短句连发、语气词指纹 | casual 傲娇 | → DT（保留傲娇） |
| **检索增强** | ✅ TF-IDF/BGE + Rerank | 无 | → DT |
| **Wiki 记忆** | ✅ 已调用 MemoryEngine | ✅ MemoryEngine | → 合并去重 |
| **工具调用** | ❌ 无 | ✅ 5 tools + loop | ← RPA 补 |
| **Skill 路由** | ❌ 无 | ✅ _route_skills | ← RPA 补 |
| **回复解析** | ⚠️ 简单 `["a","b"]` | ✅ `{"replies": [...]}` | → 合并兼容 |
| **多轮上下文** | ⚠️ `history` param | ✅ `ChatMessage` + `GlobalStore` | ← RPA 补 |
| **群聊支持** | ⚠️ 仅加权检索 | ✅ `is_group` / @检测 | ← RPA 补 |
| **会话缓存** | ❌ 无 | ✅ `SessionMemory` | ← RPA 补 |
| **Judge 评分** | ⚠️ step4_judge.py | ✅ 7 维度 JudgeWorker | → 合并（DT 标准） |
| **Benchmark** | ⚠️ 50 对抗 case | ✅ 6 benchmarks + DB | → 合并 |
| **Badcase 闭环** | ❌ 无 | ✅ JudgeWorker → DB | ← RPA 补 |
| **风格配置** | ✅ style_profile.json | ❌ 无 | → DT |
| **LLM Client** | ⚠️ 直接 openai | ✅ QwenClient 封装 | → 统一 QwenClient |

---

## 2. 新模块结构

```
wechat-mac-rpa/
├── src/                          # 现有 RPA 层（不改）
│   ├── bot/wechat_bot.py
│   ├── perception/
│   ├── action/
│   ├── session/
│   ├── memory/engine.py
│   ├── reply/
│   │   ├── generator.py          # 保留（回退用）
│   │   ├── dt_generator.py       # 新建：数字人 Generator
│   │   ├── policy.py
│   │   └── session_memory.py
│   ├── tools/
│   ├── badcase/
│   │   ├── judge_worker.py
│   │   ├── case_generator.py
│   │   └── case_db.py
│   └── tests/
│
├── digital-twin/                 # 子项目
│   ├── __init__.py
│   ├── bot.py                    # DigitalTwinBot（迁移自 rpa_bot_dense_message_level.py）
│   ├── retrieval.py              # DenseVectorIndex + LLMReranker
│   ├── system_prompt.md          # 数字人 prompt（主）
│   ├── style_profile.json        # 风格配置
│   ├── models/                   # BGE 模型（gitignore）
│   ├── tests/
│   │   ├── test_bot.py
│   │   ├── test_retrieval.py
│   │   └── adversarial_cases.json   # 50 cases
│   ├── scripts/
│   │   ├── build_index.py        # 索引构建
│   │   └── update_index.py       # 增量更新
│   └── requirements.txt
│
├── data/                         # 共享数据（gitignore）
│   ├── vector_indexes/           # 索引文件
│   │   ├── tfidf_messages.pkl
│   │   └── dense_messages.pkl
│   ├── memory/                   # wiki
│   ├── cases.db                  # benchmark DB
│   └── review_drafts/            # badcase drafts
```

---

## 3. 各模块迁移 Spec

### 3.1 system_prompt.md（DT 为主，RPA 补工具）

**来源**: digital-twin `outputs/rpa_integration/system_prompt.md`

**变更**:
```
保留（DT原文）:
  - "你是王芊本人。用户不是在跟AI聊天"
  - 说话极简（10-15字）、短句连发
  - 语气词指纹（哈、吧、啊、哈哈哈、呢）
  - 上下文优先原则（忽略案例事实）
  - {dynamic_few_shot} 占位符

新增（RPA补充）:
  - 可用工具（5个）+ 不调用规则
  - 输出格式 {"replies": [...]}
  - 规则：私聊必回、禁止敷衍、纠正后认错、不编造

删除（DT原文中删除）:
  - "禁止输出 markdown、解释、思考过程" → 移到规则里
```

**文件**: `digital-twin/system_prompt.md`

---

### 3.2 bot.py — DigitalTwinBot（DT 为主，路径修复）

**来源**: digital-twin `outputs/rpa_integration/rpa_bot_dense_message_level.py`

**迁移变更**:

```python
# 路径修复
Path("/Users/yihanwang/wechat-digital-twin/...")  
→ Path(__file__).parent / "models" / "bge-small-zh-v1.5"

Path("/Users/yihanwang/wechat-mac-rpa/.env")
→ Path(__file__).parent.parent / ".env"

# 索引路径
→ Path(__file__).parent.parent / "data" / "vector_indexes" / "tfidf_messages.pkl"

# MemoryEngine 导入修复
from memory.engine import MemoryEngine
→ from src.memory.engine import MemoryEngine
```

**保持不变**:
- `DenseVectorIndex` 类（检索）
- `LLMReranker` 类（重排序）
- `_enrich_query()` 方法
- `_build_prompt()` 方法（wiki + few_shot + style）
- `_init_llm()` 方法

---

### 3.3 src/reply/dt_generator.py — 数字人 ReplyGenerator（新建）

**定位**: 替代现有 `ReplyGenerator.generate()`，加入检索增强

**类结构**:
```python
class DigitalTwinReplyGenerator(ReplyGenerator):
    """数字人回复生成器 — 检索增强 + 工具调用"""
    
    def __init__(self, llm_client, complex_llm_client=None, memory_engine=None):
        super().__init__(llm_client, complex_llm_client, memory_engine)
        # 新增：初始化数字人检索
        self.dt_bot = DigitalTwinBot()  # 来自 digital-twin/bot.py
    
    def generate(self, unreplied, all_messages, is_group, tick_id):
        """重写：检索增强 + 工具调用 + 回复生成"""
        # 1. 从 DigitalTwinBot 获取检索增强的 system prompt
        last_msg = unreplied[-1]
        sender = last_msg.sender if not is_group else last_msg.chat_name
        enhanced_prompt = self.dt_bot._build_prompt(
            message=last_msg.text,
            sender_name=sender,
            chat_type="group" if is_group else "single",
            context=self._format_history(all_messages),
        )
        
        # 2. 替换 system_prompt（保留 RPA 的工具 + 规则）
        system_prompt = enhanced_prompt  # DT prompt 已包含一切
        
        # 3. 构建 user_prompt（保留 RPA 的记忆注入）
        user_prompt = self._build_user_prompt(unreplied, all_messages, is_group)
        
        # 4. LLM 工具调用循环（保留 RPA 的 tool loop）
        return self._run_agent_loop(system_prompt, user_prompt, ...)
```

**关键兼容点**:

回复格式解析需要同时支持三种格式：
```python
def _parse_replies(self, text):
    # 1. {"replies": [...]}  ← RPA 格式
    # 2. ["msg1", "msg2"]    ← DT 连发格式  
    # 3. "msg"               ← DT 单条格式
    # 统一转换为 ["msg1", "msg2"]
```

---

### 3.4 digital-twin/retrieval.py — 检索模块（DT 原样迁移）

**来源**: `rpa_bot_dense_message_level.py` 中提取

**类**:
- `DenseVectorIndex`: BGE embedding + cosine similarity + TF-IDF hybrid
- `LLMReranker`: LLM 重排序 top-10 → top-3

**变更**: 仅路径修复

---

### 3.5 digital-twin/style_profile.json（DT 原样保留）

```json
{
  "sender_personas": {
    "秋水文章": {"avg_reply_length": 25},
    "王芊@ai开发小分队": {"avg_reply_length": 15},
    ...
  }
}
```

用于个性化回复长度提示。

---

### 3.6 src/badcase/judge_worker.py — Judge 维度合并

**变更**: 新增 2 个维度（DT 标准），现有 7 个维度保留

| 维度 | 来源 | 说明 |
|------|------|------|
| 幻觉控制 | RPA | 保留，评分标准已对齐 |
| 记忆召回 | RPA | 保留 |
| 幽默感 | RPA | 保留 |
| 逼格语气 | RPA | → 改为"风格一致性" |
| 个性一致性 | RPA | 保留 |
| 简洁度 | RPA | 保留 |
| 上下文理解 | RPA | 保留 |
| **语气词指纹** | **DT 新增** | 是否使用了"哈、吧、啊、呢"等 |
| **短句连发** | **DT 新增** | 是否 10-15 字、是否连发 2-3 条 |
| **事实污染** | **DT 新增** | 是否错误引用了检索案例的事实 |

**评分标准**（以 DT 为准）:
```markdown
### 8. 语气词指纹（DT 标准）
- 5分: 每条回复带有≥1个语气词（哈、吧、啊、哈哈哈、呢、hhh、哇、呀）
- 3分: 部分回复有语气词，部分像客服
- 1分: 完全无语气词，像 AI 客服

### 9. 短句连发模式（DT 标准）
- 5分: 每条 10-15 字，需要多句时连发 2-3 条
- 3分: 单条偏长（20-30字），未连发
- 1分: 单条超长（>50字），像小作文

### 10. 事实污染检测（DT 标准）
- 5分: 未引用任何检索案例中的具体事实
- 1分: 引用了检索案例中的人名/数字/事件，且当前上下文无法验证
```

---

### 3.7 src/badcase/case_db.py — 新表

```sql
-- 对抗测试 case（来自 DT）
CREATE TABLE benchmark_adversarial_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT UNIQUE NOT NULL,     -- tc_0022_adv
    query TEXT NOT NULL,              -- 用户消息
    sender TEXT,                      -- 发送者
    chat_type TEXT DEFAULT 'single',  -- single | group
    ground_truth TEXT,                -- 期望回复
    context_json TEXT,                -- 对话上下文 [{"role":"user","content":"..."}]
    category TEXT,                    -- hallucination | style | knowledge_boundary | persona
    severity TEXT DEFAULT 'medium',   -- easy | medium | hard
    notes TEXT,
    enabled INTEGER DEFAULT 1
);
```

---

### 3.8 src/tests/test_adversarial_benchmark.py — 新建 P6

**case 来源**: `benchmark_adversarial_cases` 表

**评测方式**: 对每个 case：
1. 调用 `DigitalTwinReplyGenerator.generate()` 生成回复
2. JudgeWorker 评分（10 维度）
3. 对比 ground_truth 计算风格相似度

---

### 3.9 索引构建脚本

**迁移**: `scripts/build_dense_message_index.py` → `digital-twin/scripts/build_index.py`

**修复**:
- 输入路径指向 `data/weflow_exports/`
- 输出路径指向 `data/vector_indexes/`
- 新增 `--incremental` 参数支持增量更新

---

### 3.10 数据去重

| 数据 | 位置 | 去重方案 |
|------|------|---------|
| wiki 记忆 | `data/memory/wiki/` | 唯一保留（DT 已调用 RPA 的 MemoryEngine） |
| 向量索引 | `data/vector_indexes/` | RPA 原无此数据，DT 提供 |
| 风格配置 | `digital-twin/style_profile.json` | DT 提供，RPA 无 |
| BGE 模型 | `digital-twin/models/` | DT 提供，gitignore |
| 对抗 case | `cases.db → benchmark_adversarial_cases` | DT 提供，入库 |
| Badcase draft | `data/review_drafts/` + `cases.db` | RPA 提供，保留 |

---

## 4. 冲突裁决（全部以 DT 为准）

| 冲突点 | RPA 方案 | DT 方案 | 裁决 |
|--------|---------|--------|------|
| 身份 | "不爱说话，小号/分身" | "你是王芊本人" | **DT** |
| 称呼 | 不用"您" | 不用"您" | 一致 |
| 风格 | casual + 傲娇 | 极简 + 短句连发 + 语气词 | **DT**（保留傲娇作为调味） |
| 回复长度 | ≤50字 | 10-15字，常连发 | **DT** |
| 语气词 | 无要求 | 哈、吧、啊、呢 高频 | **DT** |
| 工具输出 | 直接 JSON | 不展示思考过程 | 一致（已在规则中） |
| 记忆注入 | prompt 中 [我的信息][对方信息] | prompt 中 ## 关于你的记忆 | **DT**（格式更自然） |
| 上下文 | 20条历史 + 10分钟窗口 | 视检索案例为风格参考 | **DT**（上下文优先原则） |
| Judge 标准 | 幻觉/召回/幽默/逼格/个性/简洁/上下文 | + 语气词/短句/事实污染 | **DT 优先**（新增3维度） |

---

## 5. 实施顺序

```
Phase 1 ─ 文件迁移 + 路径修复
  ├── 创建 digital-twin/ 目录结构
  ├── 复制代码、模型、索引
  ├── 修复所有硬编码路径
  └── 验证 DigitalTwinBot 初始化

Phase 2 ─ Prompt 融合
  ├── 合并 system_prompt.md
  ├── 新增工具定义 + 不调用规则
  └── 保留 {dynamic_few_shot} 占位符

Phase 3 ─ ReplyGenerator 改造
  ├── 新建 dt_generator.py
  ├── 集成检索步骤
  ├── 兼容回复格式解析
  └── 保留工具调用循环

Phase 4 ─ Judge + Benchmark 合并
  ├── JudgeWorker 新增 3 维度
  ├── 对抗 case 入库
  ├── 新建 P6 benchmark
  └── 更新 daily runner

Phase 5 ─ 验证 + 上线
  ├── A/B 对比新旧回复质量
  ├── 确认工具调用不退化
  └── 灰度切换
```
