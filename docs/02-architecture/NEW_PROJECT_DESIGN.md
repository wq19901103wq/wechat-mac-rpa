# 新项目设计文档 — 合并重构

> 背景：wechat-mac-rpa + wechat-digital-twin → 全新项目，去掉历史包袱

---

## 0. 旧项目的问题

从旧项目迭代中学到的：

| 问题 | 表现 | 新项目怎么避免 |
|------|------|-------------|
| 分层混乱 | RPA 层混入记忆注入逻辑 | 严格分层，每层只做一件事 |
| Prompt 散落 | system prompt 在 generator.py 里硬编码 | 独立 `prompts/` 目录，版本管理 |
| Case 硬编码 | benchmark case 写在 Python 代码里 | 全部入库 `cases.db` |
| 无标准评测 | benchmark 是后来加的，Judge 也是 | 项目第一天就有 benchmark + Judge |
| 路径硬编码 | `/Users/yourname/...` 到处都是 | 全部相对路径 + 配置文件 |
| 两套 identity | "不爱说话/小号" vs "林岚本人" | 统一为 DT 标准 |
| 两套回复格式 | `{"replies":[...]}` vs `["msg","msg"]` | 统一 |
| MemoryEngine 混杂 | 同时管理 wiki + overrides + 别名 | 职责拆分 |
| Judge 标准不统一 | 幻觉评分反复调 | 以 DT 对抗评测标准为准 |

---

## 1. 项目定位

**微信数字人回复系统** — 一套完整的端到端 pipeline：
```
微信消息 → 感知(RPA) → 检索(DT) → 生成(LLM) → 工具调用 → 发送(RPA)
              ↑                        ↑
          benchmark 持续监控 ←── Judge 自动评分
```

不再叫 "RPA"，RPA 只是其中的一个子系统。

---

## 2. 架构分层

```
┌──────────────────────────────────────────────────┐
│  App Layer: run_bot.py, schedule, CLI            │
├──────────────────────────────────────────────────┤
│  Orchestration: BotOrchestrator                  │
│    感知 → 去重 → 检索增强 → 推理 → 行动          │
├────────────┬──────────────┬──────────────────────┤
│  Generate  │   Retrieve   │   Action              │
│  LLM生成  │   向量检索    │   发送/切换           │
│  工具调用  │   LLM重排    │                      │
├────────────┴──────────────┼──────────────────────┤
│  Session: 去重, 状态, 持久化                      │
├───────────────────────────┼──────────────────────┤
│  Perception: 截图, OCR, 布局, API                │
├───────────────────────────┴──────────────────────┤
│  Domain Models: ChatMessage, PerceptionResult    │
└──────────────────────────────────────────────────┘

横向贯穿（独立子系统）:
  Memory   — wiki 记忆, 别名, BM25 搜索
  Tools    — web_search, stock_query, etc
  Judge    — 10 维度自动评分
  Bench    — 指标采集, 趋势, Dashboard
  Storage  — cases.db (case, prompt, metrics)
```

---

## 3. 目录结构

```
wechat-twin/                          # 项目根
├── app/
│   ├── run_bot.py                    # 生产入口
│   ├── schedule.py                   # 定时任务
│   └── config.py                     # 全局配置
│
├── src/
│   ├── models/                       # L1: 领域模型
│   │   ├── message.py                # ChatMessage, SenderType
│   │   ├── perception.py             # PerceptionResult
│   │   └── action.py                 # ActionResult
│   │
│   ├── perception/                   # L2: 感知层
│   │   ├── capture.py                # 窗口截图
│   │   ├── ocr.py                    # Vision OCR
│   │   ├── layout.py                 # 布局解析
│   │   ├── smart_pipeline.py         # 智能感知管道
│   │   └── profile.py                # 布局配置
│   │
│   ├── session/                      # L3: 会话层
│   │   ├── store.py                  # GlobalStore (去重+持久化)
│   │   └── dedup.py                  # 去重算法
│   │
│   ├── generate/                     # L4: 生成层（核心）
│   │   ├── generator.py              # 主生成器（DT prompt + 工具）
│   │   ├── retriever.py              # 向量检索 + LLM Rerank
│   │   ├── router.py                 # 模型路由 (deepseek/hermes)
│   │   ├── tools.py                  # 工具注册+执行
│   │   └── formatter.py              # 回复格式解析/兼容
│   │
│   ├── action/                       # L5: 行动层
│   │   ├── sender.py                 # 消息发送
│   │   ├── switcher.py               # 聊天切换
│   │   └── login.py                  # 登录恢复
│   │
│   ├── memory/                       # 记忆系统
│   │   ├── wiki.py                   # Wiki 读写
│   │   ├── search.py                 # BM25 搜索
│   │   ├── aliases.py                # 别名管理
│   │   └── overrides.py              # 外挂配置
│   │
│   ├── judge/                        # 评测系统（独立子系统）
│   │   ├── worker.py                 # JudgeWorker (10 维度)
│   │   ├── scorer.py                 # 评分标准/维度定义
│   │   └── prompt.py                 # Judge prompt 模板
│   │
│   ├── bench/                        # Benchmark 系统
│   │   ├── runner.py                 # 统一 benchmark runner
│   │   ├── cases.py                  # Case 加载（从 DB）
│   │   ├── metrics.py                # 指标计算
│   │   ├── report.py                 # HTML Dashboard 生成
│   │   └── monitor.py                # 定时监控
│   │
│   ├── storage/                      # 存储层
│   │   ├── db.py                     # SQLite ORM
│   │   ├── models.py                 # DB schema
│   │   └── migrate.py                # 迁移脚本
│   │
│   └── utils/                        # 工具函数
│       ├── llm.py                    # LLM 客户端
│       ├── text.py                   # 文本处理
│       └── chat.py                   # 聊天工具函数
│
├── prompts/                          # Prompt 管理
│   ├── persona.md                    # DT 人格 prompt
│   ├── judge.md                      # Judge prompt
│   └── versions/                     # prompt 版本历史
│       ├── persona_v1.md
│       └── persona_v2.md
│
├── data/                             # 数据目录（gitignore）
│   ├── cases.db                      # 主数据库
│   ├── memory/                       # wiki 文件
│   │   ├── wiki/
│   │   └── overrides/
│   ├── vector_indexes/               # 检索索引
│   ├── screenshots/                  # 截图存档
│   ├── debug/                        # tick 级调试日志
│   └── review_drafts/                # badcase drafts (JSON 备份)
│
├── models/                           # ML 模型（gitignore）
│   └── bge-small-zh-v1.5/
│
├── tests/
│   ├── unit/                         # 单元测试
│   │   ├── test_retriever.py
│   │   ├── test_generator.py
│   │   ├── test_session.py
│   │   └── ...
│   ├── bench/                        # Benchmark 测试
│   │   ├── test_tool_decision.py     # P0
│   │   ├── test_reply_quality.py     # P2
│   │   ├── test_memory_search.py     # P4
│   │   ├── test_unread_badge.py      # P5
│   │   ├── test_adversarial.py       # P6
│   │   ├── test_judge.py             # Judge Quality
│   │   └── test_stability.py         # Reply Stability
│   └── fixtures/                     # 测试 fixture
│
├── scripts/
│   ├── migrate_old_data.py           # 从旧项目迁移数据
│   ├── build_index.py                # 构建检索索引
│   └── run_eval.py                   # 跑全量评测
│
├── .env                              # API keys
├── requirements.txt                  # 基础依赖
├── requirements-ml.txt               # ML 依赖（torch, transformers）
├── AGENTS.md
└── README.md
```

---

## 4. 核心设计决策

### 4.1 Database-First

所有 case、prompt、指标从一开始就在 DB 里。不再有 JSON 文件散落。

```sql
-- 7 张核心表
cases               -- 生产 badcase（从 old project 迁移）
bench_tool_cases    -- P0 工具决策 case
bench_reply_cases   -- P2 回复质量 case  
bench_search_cases  -- P4 记忆搜索 case
bench_adversarial   -- P6 DT 对抗 case
daily_metrics       -- 指标时间序列
prompt_versions     -- prompt 版本历史
```

### 4.2 Prompt 版本管理

`prompts/persona.md` 是当前版本。每次修改 prompt 时：
1. 复制到 `prompts/versions/persona_v{N}.md`
2. 记录到 `prompt_versions` 表（version, date, git_commit, benchmark_scores）
3. 可以在 dashboard 里对比不同版本 prompt 的 benchmark 指标

### 4.3 Benchmark-First

项目第一天就有完整的 benchmark 体系：
- `src/bench/runner.py`: 统一跑所有 benchmark
- `src/bench/cases.py`: 从 DB 加载 case，不从代码读
- `src/bench/report.py`: HTML Dashboard
- `src/bench/monitor.py`: 每 3h 自动监控

新增 case：`INSERT INTO bench_reply_cases (...) VALUES (...)`，不需要改代码。

### 4.4 检索系统设计

```
用户消息 → Embedding(可选) → 向量搜索(top-10) → LLM Rerank(top-3) → Few-shot注入
```

支持三种模式（配置切换）：
- `none`: 无检索（纯 prompt）
- `tfidf`: TF-IDF 检索（轻量，无 ML 依赖）
- `dense`: BGE Dense（需要 torch + 275MB 模型）

默认 `tfidf`，需要更好效果时切 `dense`。

### 4.5 工具系统

```python
class ToolRegistry:
    tools: dict[str, Tool]  # name → Tool
    
    def register(name, description, parameters, func)
    def execute(name, arguments) → str
    def to_openai_schemas() → list[dict]
    def to_prompt_description() → str  # 生成给 LLM 看的工具描述
```

工具定义与 prompt 分离。prompt 中通过 `{tools_description}` 占位符注入。

### 4.6 Judge 10 维度

```
核心维度（决定 is_badcase）:
  1. 幻觉控制 — 编造事实？
  2. 记忆召回 — 该调工具没调？

风格维度（评价质量）:
  3. 风格一致性 — 是否符合林岚说话方式
  4. 语气词指纹 — 哈、吧、啊、呢 命中率
  5. 短句连发 — 是否 10-15 字，连发 2-3 条
  6. 个性一致性 — 第一人称，身份不分裂
  7. 上下文理解 — 是否理解了对话意图

DT 特有维度:
  8. 事实污染 — 是否错误引用了检索案例的事实
  9. 冗余回复 — 是否重复了之前说过的话

感知维度:
 10. OCR/感知错误 — 是否因感知层错误导致回复跑偏
```

---

## 5. 数据迁移计划

从两个旧项目迁移到新项目：

| 数据 | 来源 | 目标 |
|------|------|------|
| wiki 文件 | `wechat-mac-rpa/data/memory/wiki/` | `data/memory/wiki/` |
| 别名/overrides | `wechat-mac-rpa/data/memory/overrides/` | `data/memory/overrides/` |
| 生产 badcase | `wechat-mac-rpa/data/review_drafts/` | `cases` 表 |
| P0/P2/P4 case | `wechat-mac-rpa/data/cases.db` | `bench_*_cases` 表 |
| 向量索引 | `wechat-digital-twin/outputs/cache/` | `data/vector_indexes/` |
| BGE 模型 | `wechat-digital-twin/models/` | `models/bge-small-zh-v1.5/` |
| 对抗 case | `wechat-digital-twin/outputs/evaluation/` | `bench_adversarial` 表 |
| 风格配置 | `wechat-digital-twin/outputs/rpa_integration/style_profile.json` | `data/style_profile.json` |
| 历史 debug log | `wechat-mac-rpa/data/debug/` | 可选导入 |

迁移脚本: `scripts/migrate_old_data.py`

---

## 6. 相比旧项目的简化

| 旧问题 | 新方案 |
|--------|--------|
| `generator.py` 350 行 generate() | 拆成 retriever + router + formatter |
| `wechat_bot.py` 120 行 __init__ | 配置移到 config.py |
| system prompt 在代码里 | `prompts/persona.md` |
| benchmark case 硬编码 | 全部在 DB |
| `judge_worker.py` 含 prompt 模板 | `prompts/judge.md` 独立 |
| 三个不同的 LLM client | 统一 `src/utils/llm.py` |
| `MemoryEngine` 500 行 | 拆成 wiki + search + aliases |
| VisionPipeline vs SmartPipeline 重复 | 统一为 `smart_pipeline.py` |
| 多处硬编码路径 | 全部相对路径 + 配置文件 |

---

## 7. 启动顺序

```
1. 创建项目骨架 (目录 + __init__.py + requirements.txt)
2. 搭建 DB schema + 迁移脚本
3. 搭建 models + utils (领域模型, LLM client, 工具函数)
4. 迁移 memory 模块 (wiki + search + aliases)
5. 迁移 perception 模块 (capture, ocr, layout, smart_pipeline)
6. 迁移 session 模块 (store, dedup)
7. 搭建 generate 模块 (retriever, generator, router, tools, formatter)
8. 整合 prompts/ (persona.md, judge.md)
9. 搭建 judge + bench 模块 (worker, scorer, runner, report, monitor)
10. 迁移 action 模块 (sender, switcher, login)
11. 跑数据迁移脚本
12. 跑全量 benchmark 建立 baseline
13. 写 AGENTS.md + README.md
```
