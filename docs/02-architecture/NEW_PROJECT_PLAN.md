# 新项目计划 — wechat-twin

> 合并 wechat-mac-rpa + wechat-digital-twin，大部份直接搬老项目代码

---

## 1. 搬什么

### 从 wechat-mac-rpa 搬（不改或微改）

| 模块 | 操作 | 说明 |
|------|------|------|
| `src/models/` | 直接搬 | 领域模型，无依赖 |
| `src/capture/` | 直接搬 | 截图模块，无变化 |
| `src/ocr/` | 直接搬 | OCR，修 normalized_x/y 硬编码分辨率 |
| `src/layout/` | 直接搬 | 布局解析 |
| `src/message/` | 直接搬 | 消息提取，修 used_self 去重 |
| `src/perception/` | 直接搬 | SmartPipeline + VisionPipeline，修 is_group 缺失 |
| `src/session/` | 直接搬 | GlobalStore 去重+持久化 |
| `src/action/` | 直接搬 | 消息发送+聊天切换，修 cliclick 硬编码路径 |
| `src/tools/` | 直接搬 | tool_registry + builtin_tools（已修 bs4） |
| `src/memory/` | 直接搬 | MemoryEngine（已修路径+case-insensitive） |
| `src/badcase/` | 直接搬 | JudgeWorker + CaseGenerator + CaseDB |
| `src/utils/` | 直接搬 | chat_utils, text_utils, xml_utils, qwen_client |
| `src/logging/` | 直接搬 | BotLogger, DebugLogger |
| `scripts/` | 直接搬 | monitor_benchmark, run_daily_benchmark, generate_dashboard, migrate_benchmarks_to_db |
| `data/memory/` | 直接搬 | wiki + overrides |
| `tests/` | 直接搬 | 所有 test 文件 + fixtures |
| `.env` | 直接复制 | API keys |

### 从 wechat-digital-twin 搬

| 模块 | 操作 | 说明 |
|------|------|------|
| `outputs/rpa_integration/system_prompt.md` | 搬到 `prompts/persona.md` | DT 人格 prompt |
| `outputs/rpa_integration/style_profile.json` | 搬到 `data/style_profile.json` | 风格配置 |
| `outputs/rpa_integration/rpa_bot_dense_message_level.py` | 搬到 `src/generate/retriever.py` | 检索模块，修路径 |
| `outputs/evaluation/adversarial_test_cases_v2.json` | 搬到 `data/adversarial_cases.json` | 50 对抗 case |
| `models/bge-small-zh-v1.5/` | 搬到 `models/bge-small-zh-v1.5/` | BGE 模型 |
| `outputs/cache/*.pkl` | 搬到 `data/vector_indexes/` | 向量索引 |

---

## 2. 要改的文件（只有这些需要动手）

| 文件 | 改动 |
|------|------|
| `src/reply/generator.py` | **唯一大改**：system prompt 换成 DT 的，加入 retriever，保留工具调用 |
| `src/reply/generator.py:_system_prompt()` | 改为读 `prompts/persona.md` |
| `src/reply/generator.py:generate()` | 加入检索步骤：调 retriever.search() → few-shot 注入 |
| `src/reply/generator.py:_parse_replies()` | 兼容 DT 的 `["msg1","msg2"]` 格式 |
| `src/ocr/vision_ocr.py:normalized_x/y` | 1760/1280 → 用实际图片尺寸 |
| `src/perception/vision_pipeline.py` | 补 `is_group` 字段 |
| `src/action/chat_list_clicker.py` | 硬编码 cliclick → shutil.which |
| `src/memory/engine.py` | 已完成（绝对路径+case-insensitive） |
| `src/badcase/judge_worker.py` | 加 DT 的 3 个维度（语气词、短句、事实污染） |
| `src/badcase/case_db.py` | 加 `bench_adversarial_cases` 表 |
| `src/tools/builtin_tools.py` | 已完成（bs4 替换） |
| 所有文件 | 搜 `/Users/yourname/` 硬编码路径 → 改相对路径 |

---

## 3. 不改的文件

- 整个 perception 链路（capture/ocr/layout/extractor/smart_pipeline）逻辑不变
- 整个 action 链路（sender/switcher/login）逻辑不变
- session 去重逻辑不变
- MemoryEngine 搜索逻辑不变
- JudgeWorker 判定逻辑不变（只加维度，不改现有逻辑）
- benchmark 框架不变
- case_db 结构不变（只加表）
- scripts 脚本不变

---

## 4. 新项目目录

```
wechat-twin/
├── src/                    # 老项目 wechat-mac-rpa/src/ 直接搬
│   ├── models/
│   ├── capture/
│   ├── ocr/
│   ├── layout/
│   ├── message/
│   ├── perception/
│   ├── session/
│   ├── reply/              # ← generator.py 大改
│   ├── action/
│   ├── tools/
│   ├── memory/
│   ├── badcase/
│   ├── utils/
│   ├── logging/
│   └── tests/
├── prompts/                # 新：prompt 目录
│   └── persona.md          # ← 从 DT 搬
├── data/                   # 老项目 data/ + DT 的索引
│   ├── memory/wiki/
│   ├── memory/overrides/
│   ├── vector_indexes/     # ← 从 DT 搬
│   ├── style_profile.json  # ← 从 DT 搬
│   ├── adversarial_cases.json  # ← 从 DT 搬
│   ├── cases.db
│   ├── screenshots/
│   ├── debug/
│   └── review_drafts/
├── models/                 # ← 从 DT 搬 BGE 模型
├── scripts/                # 老项目 scripts/ 直接搬
├── tests/                  # 老项目 tests/ 直接搬（额外 test 目录）
├── .env
├── requirements.txt
├── requirements-ml.txt
├── AGENTS.md
└── README.md
```

---

## 5. generator.py 改动细节

### 5.1 system prompt 来源变更

```python
# 旧：硬编码在 _system_prompt() 方法里
def _system_prompt(self) -> str:
    lines_local = [
        "核心人设与风格",
        "你叫'不爱说话'，是林岚的小号/分身...",
        ...
    ]

# 新：读 prompts/persona.md，注入动态内容
def _system_prompt(self) -> str:
    prompt = (PROJECT_ROOT / "prompts" / "persona.md").read_text()
    prompt = prompt.replace("{tools_description}", self._tools_description())
    prompt = prompt.replace("{dynamic_few_shot}", self._retrieve_few_shot())
    return prompt
```

### 5.2 加入检索步骤

```python
def _retrieve_few_shot(self) -> str:
    """检索相似历史对话作为 few-shot"""
    if not self.retriever:
        return "（无相关历史对话）"
    cases = self.retriever.search(message, sender, chat_type, top_k=3)
    if not cases:
        return "（无相关历史对话）"
    return "\n\n".join(f"--- 案例 {i+1} ---\n{c}" for i, c in enumerate(cases, 1))
```

### 5.3 兼容 DT 回复格式

```python
def _parse_replies(self, text: str) -> List[str]:
    # 先尝试 {"replies": [...]}（现有格式）
    # 再尝试 ["msg1", "msg2"]（DT 格式）
    # 最后当纯文本
```

### 5.4 其他 generate() 逻辑不变

- 工具调用循环不变
- Skill 路由不变
- Hermes fallback 不变
- 空回复重试不变
- Judge 提交不变

---

## 6. 搬的顺序

```
1. 建目录
2. 复制 .env, requirements.txt
3. 复制 src/ 全部（从 wechat-mac-rpa）
4. 复制 scripts/ 全部
5. 复制 tests/ 全部
6. 复制 data/memory/ 全部
7. 从 DT 复制 system_prompt → prompts/persona.md
8. 从 DT 复制 style_profile.json → data/
9. 从 DT 复制 adversarial_cases → data/
10. 从 DT 复制模型 → models/
11. 从 DT 复制索引 → data/vector_indexes/
12. 修路径（全局替换 /Users/yourname/...）
13. 修 generator.py（读 prompt 文件 + 检索 + 格式兼容）
14. 修 vision_ocr.py（分辨率）
15. 修 vision_pipeline.py（is_group）
16. 修 chat_list_clicker.py（cliclick）
17. 修 judge_worker.py（加 DT 维度）
18. 跑 benchmark 验证
```

---

## 7. 不改但要注意的

- `src/reply/policy.py` — 回复决策逻辑不变（群聊/私聊判断）
- `src/bot/wechat_bot.py` — Bot 主循环不变，只 init 时传 retriever
- `src/tools/stock_tools.py` — 股票查询不变
- `src/session/global_store.py` — 去重逻辑不变
- `src/perception/smart_pipeline.py` — 感知管道不变
