# 缺失模块清单

> 本文档记录当前代码中**已实现**和**待实现**的模块状态。
> 最后更新: 2026-05-15

---

## 已实现模块

### P0 — 核心流程

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| OpenClaw LLM 客户端 | `src/llm/openclaw_client.py` | ✅ | Kimi 本地代理连接 |
| Qwen API 客户端 | `src/llm/qwen_client.py` | ✅ | DashScope 多模态 API |
| 全局消息存储 | `src/session/global_store.py` | ✅ | LCS 去重 + 持久化 |
| 会话记忆缓存 | `src/reply/session_memory.py` | ✅ | 跨 tick 工具缓存 |
| 股票查询工具 | `src/tools/stock_tools.py` | ✅ | 已落地 |

### P1 — 工具与增强

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 工具注册表 | `src/tools/tool_registry.py` | ✅ | OpenAI function calling 格式 |
| 内置工具 | `src/tools/builtin_tools.py` | ✅ | 时间/天气/搜索/股票 |
| 聊天列表点击器 | `src/action/chat_list_clicker.py` | ✅ | 坐标计算 + 点击 |
| 登录恢复 | `src/action/login_recovery.py` | ✅ | 自动点击登录按钮 |
| 记忆引擎 | `src/memory/engine.py` | ✅ | LLM Wiki + Overrides |

### P2 — 感知扩展

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| WeFlow 客户端 | `src/perception/weflow_client.py` | ✅ | 微信数据库读取（实验性） |
| WeFlow 管道 | `src/perception/weflow_pipeline.py` | ✅ | 数据库驱动感知（实验性） |

### P3 — 工程化

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Badcase 生成器 | `src/badcase/case_generator.py` | ✅ | 从 tick 数据生成 benchmark case |
| Judge Worker | `src/badcase/judge_worker.py` | ✅ | 异步质量评估 |
| Review Server | `src/badcase/review_server.py` | ✅ | 人工审核 Web 服务 |
| Benchmark - Reply Quality | `src/tests/test_reply_quality_benchmark.py` | ✅ | 24 cases, LLM-as-a-Judge |
| Benchmark - Tool Decision | `src/tests/test_tool_decision_benchmark.py` | ✅ | 27 cases |
| Benchmark - Memory Search | `src/tests/test_memory_search_benchmark.py` | ✅ | 29 cases |
| Benchmark - Chat List Unread | `src/tests/test_chat_list_unread_benchmark.py` | ✅ | 23 cases |
| Benchmark - OCR Quality | `src/benchmarks/ocr_quality.py` | ✅ | 33 个私有真实 cases，代表性/回归分层 |
| Private Benchmark Runner | `src/benchmarks/private_runner.py` | ✅ | 来源留痕、失败归因、机器候选筛选 |

---

## 待实现/优化项

| 优先级 | 项 | 说明 |
|--------|---|------|
| 🟡 P1 | `src/storage/chat_history.py` 分片优化 | 大文件加载慢 |
| 🟡 P1 | WeFlow 管道生产化 | 当前为实验性，`WEFLOW_MODE=ocr` 默认关闭 |
| 🟢 P2 | 多显示器场景支持 | 坐标计算需适配多屏 |

---

## 已删除/废弃

| 模块 | 原因 |
|------|------|
| `src/storage/message_store.py` | 功能合并到 `chat_history.py` 和 `global_store.py` |
| `core/auto_bot_vision_ocr_v*.py` | 由模块化架构完全替代 |
