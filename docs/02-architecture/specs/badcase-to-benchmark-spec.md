# Badcase → Benchmark 自动化闭环 Spec

## 1. 背景

生产环境出现 badcase 时，目前靠人工看日志、记问题、改 prompt。这种 case-by-case 的优化方式存在两个问题：
1. 修复 A 可能引入 B 的 regression
2. 没有可量化的回归验证手段

本系统建立一个从生产 badcase 到 benchmark case 的自动化闭环：检测到 badcase → LLM 判定 → 有把握的自动入库 / 不确定的进审核台 → 人工确认 → 合并到 benchmark → 后续所有修改都必须通过 benchmark 回归。

## 2. 目标

- **自动捕获**：Bot 回复后异步判定是否是 badcase，无需人工盯着日志
- **自动分级**：LLM 输出 confidence，高置信度直接入库，低置信度进审核台
- **人工审核**：提供一个本地 Web 审核台，支持查看上下文、截图、编辑 case、入库/丢弃
- **冻结标准**：入库的 case 成为 benchmark 的一部分，后续 LLM 输出变化不影响 case 定义
- **零侵入**：主循环（`run_bot.py`）只做异步提交，不阻塞、不增加延迟

## 3. 非目标

- 不替代现有的 `ERROR_CASE_GUIDE.md` OCR 错误管理体系（那是感知层的事）
- 不做复杂的权限管理（本地单用户）
- 不做远程部署（审核台只在本地跑）

## 4. 架构

```
┌─────────────────────────────────────────────────────────────┐
│                       生产运行时                             │
│  ┌──────────┐    generate()    ┌──────────────────────┐    │
│  │ ReplyGen │ ───────────────▶ │  JudgeWorker.submit  │    │
│  │ erator   │   (异步,不阻塞)   │   (丢进内存队列)      │    │
│  └──────────┘                  └──────────────────────┘    │
│                                          │                  │
│                                          ▼                  │
│                               ┌──────────────────────┐    │
│                               │  后台单线程消费队列   │    │
│                               │  (调用 LLM Judge)    │    │
│                               └──────────────────────┘    │
│                                          │                  │
│                    ┌─────────────────────┼─────────────────┤
│                    │                     │                 │
│                    ▼                     ▼                 │
│         ┌─────────────────┐  ┌──────────────────────┐    │
│         │  auto_commit    │  │  pending_draft       │    │
│         │  (直接追加到     │  │  (data/review_drafts │    │
│         │   benchmark)    │  │   /pending/)         │    │
│         └─────────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      审核台 (按需启动)                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │  GET /       │      │  GET /draft  │      │ POST /   │ │
│  │  列表页      │      │  /{id}       │      │ commit   │ │
│  │              │      │  详情页      │      │ 入库     │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         │                     │                     │      │
│         ▼                     ▼                     ▼      │
│  ┌────────────────────────────────────────────────────┐   │
│  │              直接读写 JSON / Python 文件              │   │
│  │  pending/  committed/  dismissed/  benchmark 文件   │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 5. 数据流

### 5.1 生产侧数据流

1. `ReplyGenerator.generate()` 生成回复后，若 `bot_should_reply=True`，调用 `JudgeWorker.submit(tick_data)`
2. `submit()` 把 `tick_data` 丢进 `asyncio.Queue`，立即返回（< 1ms）
3. 后台 worker 单线程从队列消费，调用 LLM Judge API
4. Judge 返回结果后：
   - 若满足 `auto_commit` 条件 → 直接追加到 benchmark 文件，LLM 响应缓存到 `fixtures/auto_cases/`，记录到 `committed/` 日志
   - 若不满足 → 写入 `data/review_drafts/pending/{draft_id}.json`

### 5.2 审核台数据流

1. 用户启动 `python scripts/review_server.py`
2. 浏览器打开 `http://localhost:8765`
3. 列表页读取 `pending/` 目录下的所有 draft JSON，按 severity + confidence 排序
4. 用户点击 draft 进入详情页，查看对话上下文、截图、LLM 判定、自动生成的 case 代码
5. 用户操作：
   - **入库**：把 case 代码追加到对应 benchmark 文件，draft 移入 `committed/`
   - **修改后入库**：用户编辑 case 代码后再追加，draft 移入 `committed/`
   - **丢弃**：draft 移入 `dismissed/`，记录丢弃原因

## 6. Judge Prompt 规格

```
你是对话质量审计专家。判断以下 Bot 回复是否有质量问题，需要记录到 benchmark。

## 判定原则
- 重点关注：Bot 是否编造了记忆中没有的具体事实？是否该调工具没调？是否纠正后不生效？
- 区分"角色扮演自嘲"和"真错了"：Bot 说"我瞎编的"如果是在承认错误，是正常行为，不是 badcase
- 区分"玩笑"和"事实陈述"：如果 Bot 明确在调侃，即使内容夸张也不算 badcase

## 输入信息
{conversation_context}  // 最近 5 轮对话
{bot_reply}             // Bot 本轮回复
{full_user_prompt}      // Bot 实际收到的完整 User Prompt（含记忆注入、历史消息、未读消息）
{tool_calls}            // Bot 本轮调用的工具列表
{full_context}          // Bot 实际看到的完整上下文：System Prompt + Tools 定义 + 完整 Messages 列表

## 输出格式（纯 JSON，不要 markdown）
{
  "is_badcase": true | false,
  "badcase_type": "hallucination" | "missing_tool_call" | "correction_not_persistent" | "wrong_fact" | "none",
  "severity": "P0" | "P1" | "P2",
  "confidence": 0.0 ~ 1.0,
  "auto_commit": true | false,
  "reason": "一句话理由，引用 Bot 回复原文作为证据",
  "expected_behavior": "Bot 应该怎么做才对？"
}

## auto_commit 规则
- true：你非常确定这是 badcase，证据明确无歧义，不需要人工审核
- false：情况有模糊性，或涉及隐私判断，建议人工确认
```

### 6.1 auto_commit 判定标准（系统侧硬编码）

Judge 输出 `auto_commit=true` 只是建议，系统侧还要满足以下条件才会真正自动入库：

```python
AUTO_COMMIT_RULES = {
    "min_confidence": 0.90,
    "allowed_types": {"missing_tool_call", "hallucination", "wrong_fact"},
    "forbidden_types": {"correction_not_persistent"},  # 多轮纠正必须人工审
}

def should_auto_commit(judge_result: dict) -> bool:
    return (
        judge_result.get("auto_commit") is True
        and judge_result.get("confidence", 0) >= AUTO_COMMIT_RULES["min_confidence"]
        and judge_result.get("badcase_type") in AUTO_COMMIT_RULES["allowed_types"]
    )
```

## 7. Draft JSON Schema

```json
{
  "draft_id": "tick_8877_20260519_230434",
  "tick_id": 8877,
  "timestamp": "2026-05-19T23:04:34",
  "chat_name": "示例用户甲 @示例交流群",
  "status": "pending",
  
  "judge_result": {
    "is_badcase": true,
    "badcase_type": "missing_tool_call",
    "severity": "P0",
    "confidence": 0.95,
    "auto_commit": true,
    "reason": "用户说'明天'，Bot 直接回复天气数据，tool_calls 为空",
    "expected_behavior": "应调用 get_weather(city='上海', date='明天')"
  },
  
  "conversation": [
    {"role": "user", "sender": "示例用户甲", "text": "有什么适合的运动"},
    {"role": "bot", "text": "大晚上快11点了你问适合的运动？居家平板撑..."},
    {"role": "user", "sender": "示例用户甲", "text": "明天"},
    {"role": "user", "sender": "示例用户甲", "text": "我刚说错了"}
  ],

  "bot_reply": "明天上海多云25℃，不冷不热的，适合去滨江步道骑行...",
  "tool_calls": [],
  "memory_injected": "...",

  "full_system_prompt": "你是一个微信助手，名叫小明...",
  "full_tools_context": "{\"get_weather\": {...}}",
  "full_user_prompt": "[会话]\n时间：...\n[历史消息]...\n[未读消息]...",
  "full_llm_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],

  "assets": {
    "screenshot_path": "data/screenshots/wechat_8877_20260519_230434.png",
    "prompt_md_path": "data/debug/prompts/tick_2026-05-19T23-04-34.148188_8877.md",
    "tick_json_path": "data/debug/tick_2026-05-19T23-04-34.148188_8877.json"
  },
  
  "generated_case": {
    "module": "P0",
    "case_code": "BenchmarkCase(...)"
  },
  
  "review_history": [],
  "committed_at": null,
  "committed_by": null,
  "dismissed_at": null,
  "dismiss_reason": null
}
```

## 8. 审核台 API 设计

### 8.1 后端 API

```python
# FastAPI 路由

@app.get("/")
def list_drafts(
    status: str = "pending",      # pending | committed | dismissed
    severity: Optional[str] = None,  # P0 | P1 | P2
    module: Optional[str] = None,    # P0 | P2 | P3
):
    """返回 draft 列表（轻量，不含完整对话）"""
    return [
        {
            "draft_id": "...",
            "timestamp": "...",
            "severity": "P0",
            "badcase_type": "missing_tool_call",
            "confidence": 0.95,
            "bot_reply_preview": "明天上海多云25℃...",
            "status": "pending"
        }
    ]

@app.get("/draft/{draft_id}")
def get_draft(draft_id: str):
    """返回单个 draft 的完整信息"""
    return {
        "draft_id": "...",
        "conversation": [...],
        "judge_result": {...},
        "generated_case": {"module": "P0", "case_code": "..."},
        "assets": {...}
    }

@app.post("/draft/{draft_id}/commit")
def commit_draft(
    draft_id: str,
    case_code: Optional[str] = None,  # 用户编辑后的代码，不传则使用 generated_case
    notes: Optional[str] = None,
):
    """入库：追加到 benchmark 文件，移入 committed/"""
    return {"success": true, "benchmark_file": "src/tests/test_tool_decision_benchmark.py"}

@app.post("/draft/{draft_id}/dismiss")
def dismiss_draft(draft_id: str, reason: str):
    """丢弃：移入 dismissed/"""
    return {"success": true}
```

### 8.2 前端页面

**列表页 `/`**：
- 顶部：统计卡片（待审 P0/P1/P2 数量、今日自动入库数）
- 左侧：筛选栏（status、severity、module、badcase_type）
- 主区域：draft 卡片列表，每张卡片显示：
  - 时间、chat_name、severity 标签
  - Bot 回复 preview（截断 60 字）
  - confidence 进度条
  - LLM reason 一句话
  - 操作按钮（查看详情）

**详情页 `/draft/{id}`**：
- 上部三栏：
  - 左：对话上下文（用户消息绿色，Bot 回复蓝色，可折叠）
  - 中：截图（点击放大）+ prompt.md 下载链接
  - 右：Judge 判定结果（JSON 格式化展示）
- 中部（full-width）：
  - **Bot 实际看到的完整上下文（Judge 判定依据）**
    - System Prompt
    - Tools 定义
    - 完整 User Prompt（实际发给 LLM 的 user 消息，含 `[会话]`、`[历史消息]`、`[未读消息]`）
    - 完整 Messages 列表（含 tool 返回结果）
- 下部：
  - 可编辑的 case 代码文本框（语法高亮用 CodeMirror 或纯 textarea）
  - 三个大按钮：✅ 直接入库 / ✏️ 修改后入库 / 🗑️ 丢弃
  - 如果点击"修改后入库"，文本框必须被编辑过才能提交

## 9. 与现有 Benchmark 的集成

### 9.1 P0 Tool 决策

新增 case 追加到 `src/tests/test_tool_decision_benchmark.py` 的 `BENCHMARK_CASES` 列表末尾。

自动生成规则：
- `badcase_type=missing_tool_call` → `should_call_memory=False`，同时在注释中标注"应调用 get_weather"
- 未来扩展 P0 时，自动生成的 case 可能包含 `expected_tools: ["get_weather"]`

### 9.2 P2 回复质量

追加到 `src/tests/test_reply_quality_benchmark.py`。

自动生成规则：
- 根据 conversation 构建 `all_messages` 和 `unreplied`
- 根据 badcase_type 选择 rubric
- 自动生成 `forbidden_keywords`（从 Bot 实际回复中提取编造的具体事实词）

### 9.3 Case 命名规范

```
auto_{badcase_type}_{tick_id}
# 例如：
auto_missing_tool_call_8877
auto_hallucination_286
auto_correction_not_persistent_512
```

## 10. 文件结构

```
wechat-mac-rpa/
├── src/
│   └── badcase/
│       ├── __init__.py
│       ├── judge_worker.py         # 异步 LLM Judge + 自动入库
│       ├── case_generator.py       # 根据 draft 生成 benchmark case 代码
│       └── review_server.py        # FastAPI 审核台后端
├── scripts/
│   └── review_server.py            # 启动入口 (python scripts/review_server.py)
├── data/
│   └── review_drafts/
│       ├── pending/                # 待审核
│       ├── committed/              # 已入库
│       └── dismissed/              # 已丢弃
├── src/tests/fixtures/
│   └── auto_cases/                 # 自动入库的 LLM 响应缓存
└── docs/05-specs/
    └── badcase-to-benchmark-spec.md # 本文件
```

## 11. 模块设计

### 11.1 JudgeWorker

```python
class JudgeWorker:
    def __init__(self, model: str = "deepseek-v4-flash"):
        self.client = QwenClient(model=model)
        self.queue = asyncio.Queue()
        self._running = False
    
    def submit(self, tick_data: dict):
        """主循环调用，立即返回"""
        if not self._running:
            self._start()
        asyncio.create_task(self.queue.put(tick_data))
    
    def _start(self):
        """启动后台消费协程"""
        self._running = True
        asyncio.create_task(self._consume_loop())
    
    async def _consume_loop(self):
        while True:
            tick_data = await self.queue.get()
            try:
                result = await self._judge(tick_data)
                if result["is_badcase"]:
                    draft = self._build_draft(tick_data, result)
                    if self._should_auto_commit(result):
                        self._auto_commit(draft)
                    else:
                        self._save_pending(draft)
            except Exception as e:
                logger.error(f"Judge failed: {e}")
            finally:
                self.queue.task_done()
    
    async def _judge(self, tick_data: dict) -> dict:
        """调用 LLM Judge，返回结构化结果"""
        ...
```

### 11.2 CaseGenerator

```python
class CaseGenerator:
    """根据 draft 生成对应 benchmark 模块的 case 代码"""
    
    def generate(self, draft: dict) -> dict:
        module = self._route_module(draft)
        if module == "P0":
            code = self._generate_p0_case(draft)
        elif module == "P2":
            code = self._generate_p2_case(draft)
        else:
            code = self._generate_p3_case(draft)
        return {"module": module, "case_code": code}
    
    def _route_module(self, draft: dict) -> str:
        type_to_module = {
            "missing_tool_call": "P0",
            "hallucination": "P2",
            "wrong_fact": "P2",
            "correction_not_persistent": "P3",
        }
        return type_to_module.get(draft["judge_result"]["badcase_type"], "P2")
```

### 11.3 ReviewServer (FastAPI)

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Badcase Review")

# 前端静态文件
app.mount("/static", StaticFiles(directory="src/badcase/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return _render_list_page()

@app.get("/draft/{draft_id}", response_class=HTMLResponse)
def draft_detail(draft_id: str):
    return _render_detail_page(draft_id)

# API 路由...
```

## 12. 启动与使用流程

```bash
# 1. 日常运行（JudgeWorker 随 bot 自动启动）
python run_bot.py

# 2. 查看自动入库记录
cat data/review_drafts/committed/$(date +%Y-%m-%d).jsonl

# 3. 打开审核台（需要时启动）
python scripts/review_server.py
# 🚀 Review server running at http://localhost:8765

# 4. 一键打开浏览器
open http://localhost:8765
```

## 13. 回归验证

每个自动/人工入库的 case，入库后应立即验证是否能 reproduce badcase：

```bash
# 自动入库时触发
pytest src/tests/test_tool_decision_benchmark.py -v -k "auto_missing_tool_call_8877"
# 预期：FAIL（因为 badcase 还没修）

# 系统性修复后再次运行
pytest src/tests/test_tool_decision_benchmark.py -v
# 预期：PASS（修复成功且无 regression）
```

## 14. 风险与兜底

| 风险 | 兜底方案 |
|------|---------|
| LLM Judge API 失败 | 重试 3 次，失败后把 tick_data 原样写入 `pending/` 的 `unjudged/` 子目录，人工后续补审 |
| 自动入库的 case 有问题 | 审核台支持"撤销入库"，从 benchmark 文件删除对应 case |
| 审核台端口冲突 | 默认 8765，支持 `--port` 参数 |
| 磁盘膨胀 | `pending/` 保留 30 天，`committed/` 和 `dismissed/` 保留 90 天，定期清理 |

## 15. 迭代计划

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| v0.1 | JudgeWorker + auto_commit + 写入 pending | P0 |
| v0.2 | 审核台后端 API + 前端列表/详情页 | P0 |
| v0.3 | CaseGenerator 支持 P0/P2/P3 | P0 |
| v0.4 | 一键入库/丢弃/修改 | P0 |
| v0.5 | 截图预览 + prompt.md 查看 | P1 |
| v0.6 | 撤销入库 + 批量操作 | P1 |
