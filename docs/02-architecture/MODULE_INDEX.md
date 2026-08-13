# 模块索引 (Module Index)

> **本文档描述的是当前已落地的生产架构（Current Production Architecture）。**
>
> AI 开发时的快速导航页。
> 
> 规则：如果你不知道该改哪个文件，先查此表。

---

## 按问题类型索引

### "消息识别错了"
| 现象 | 可能原因 | 修改文件 |
|------|---------|---------|
| 聊天名识别错 | title_y_max / title_x_max_ratio 不准 | `src/layout/profile.py` |
| 输入框内容混入消息 | input_y_min 太松 | `src/layout/profile.py` |
| 时间戳被当成消息 | TIMESTAMP_PATTERNS 不完整 | `src/layout/layout_parser.py` |
| 自己消息被当成对方 | 绿色气泡检测失败 | `src/layout/layout_parser.py` |
| 消息顺序错乱 | 提取时未按 y 排序 | `src/message/extractor.py` |
| 昵称识别错 | nickname 区域边界不对 | `src/layout/profile.py` |

### "回复时机错了"
| 现象 | 可能原因 | 修改文件 |
|------|---------|---------|
| 自己发的话又回复 | 去重/回声检测失效 | `src/session/global_store.py` |
| 对同一句话反复回复 | `merge_tick()` 去重不严格 | `src/session/global_store.py` |
| 群聊没@也回复 | `@检测` 或 `群聊判断` 错误 | `src/reply/policy.py` |
| 回复太频繁 | cooldown 时间太短 | `src/session/global_store.py` |

### "发送内容错了"
| 现象 | 可能原因 | 修改文件 |
|------|---------|---------|
| 回复内容太长/太啰嗦 | 系统提示词 | `src/reply/generator.py` |
| 回复不相关 | LLM prompt 或上下文 | `src/reply/generator.py` |

### "发送动作异常"
| 现象 | 可能原因 | 修改文件 |
|------|---------|---------|
| 发出去是乱码 | 用了 keystroke 输入中文 | `src/action/message_sender.py` |
| 没发出去 | AppleScript 失败 | `src/action/message_sender.py` |
| 切换聊天失败 | 坐标点击未命中或聊天列表未识别 | `src/action/ui_interactor.py` / `src/layout/layout_parser.py` |
| 截图失败 | 找不到微信窗口 | `src/capture/window_capture.py` |

### "排查问题找不到信息"
| 现象 | 可能原因 | 修改文件 |
|------|---------|---------|
| 不知道 Bot 为什么没回复 | execution.jsonl 缺少 decision 日志 | `src/logging/bot_logger.py` |
| 历史记录丢失/找不到 | GlobalStore 分片写入逻辑错误 | `src/session/global_store.py` |
| 单文件过大加载慢 | 未按 chat_name 分片 jsonl | `src/session/global_store.py` |

### "Benchmark 回归失败"
| 现象 | 可能原因 | 修改文件 |
|------|---------|---------|
| Reply Quality 失败 | Prompt 变更导致回复质量下降 | `src/reply/generator.py` |
| Tool Decision 失败 | 新增工具导致过度调用 search_memory | `src/reply/generator.py` prompt |
| OCR Quality 失败 | 感知层 prompt/API 变更 | `src/perception/smart_pipeline.py` |
| Memory Search 失败 | Wiki 别名缺失或 BM25 参数变更 | `src/memory/engine.py` |
| Chat List Unread 失败 | 未读角标识别逻辑变更 | `src/perception/smart_pipeline.py` |

---

## 按文件索引

### `src/models/base.py`
- **定位**: L1 领域模型
- **改什么**: 基础数据结构变更
- **不改什么**: 业务逻辑
- **Spec**: [specs/MODELS_SPEC.md](specs/MODELS_SPEC.md)

### `src/capture/window_capture.py`
- **定位**: L2 截图
- **改什么**: 窗口查找逻辑、截图方式、Retina 适配
- **不改什么**: OCR、消息解析
- **Spec**: [specs/CAPTURE_SPEC.md](specs/CAPTURE_SPEC.md)

### `src/ocr/vision_ocr.py`
- **定位**: L2 OCR
- **改什么**: 改用其他 OCR 引擎（如 EasyOCR）、坐标转换
- **不改什么**: 过滤、布局解析
- **Spec**: [specs/OCR_SPEC.md](specs/OCR_SPEC.md)

### `src/layout/profile.py`
- **定位**: L2 配置
- **改什么**: 所有边界值、颜色阈值
- **不改什么**: 解析逻辑本身

### `src/layout/layout_parser.py`
- **定位**: L3 布局分组
- **改什么**: 区域分组算法、时间戳检测、气泡检测
- **不改什么**: 消息去重、发送逻辑
- **Spec**: [specs/LAYOUT_SPEC.md](specs/LAYOUT_SPEC.md)

### `src/message/extractor.py`
- **定位**: L3 消息提取
- **改什么**: 消息合并规则、昵称匹配、sender_type 判定
- **不改什么**: OCR、截图

### `src/perception/smart_pipeline.py`
- **定位**: L3.5 智能感知管道（主力）
- **改什么**: 本地预判与 API 兜底的切换逻辑、像素差异阈值、多模态 API 调用
- **不改什么**: 去重策略、回复生成
- **Spec**: [specs/PERCEPTION_SPEC.md](specs/PERCEPTION_SPEC.md)

### `src/perception/vision_pipeline.py`
- **定位**: L3.5 纯本地 OCR 管道（备用回退）
- **改什么**: 聚合视觉链路、错误处理、聊天切换预留接口
- **不改什么**: 去重策略、回复生成

### `src/perception/weflow_client.py`
- **定位**: L3.5 WeFlow 客户端
- **改什么**: 微信数据库连接配置
- **不改什么**: 消息解析逻辑
- **注意**: 实验性模块，默认关闭

### `src/perception/weflow_pipeline.py`
- **定位**: L3.5 WeFlow 感知管道
- **改什么**: 数据库驱动消息同步逻辑
- **不改什么**: 回复生成
- **注意**: 实验性模块，默认关闭

### `src/session/global_store.py`
- **定位**: L4 会话/去重/持久化
- **改什么**: `merge_tick()` 去重算法（LCS 序列对齐）、`_match_single()` 匹配逻辑、`_is_fuzzy_duplicate()` 模糊兜底、持久化格式
- **不改什么**: 回复生成
- **注意**: 该文件同时承担会话状态管理和 JSON 持久化职责，所有去重策略集中于此
- **Spec**: [specs/SESSION_SPEC.md](specs/SESSION_SPEC.md)

### `src/reply/policy.py`
- **定位**: L4 决策
- **改什么**: 回复触发条件、@检测、私聊/群聊区分
- **不改什么**: 发送执行
- **Spec**: [specs/REPLY_SPEC.md](specs/REPLY_SPEC.md)（含 generator + policy）

### `src/reply/generator.py`
- **定位**: L4 生成
- **改什么**: Prompt 工程、LLM 调用
- **不改什么**: 去重逻辑
- **注意**: 兜底回复已废弃（返回空列表），不再生成固定话术
- **Spec**: [specs/REPLY_SPEC.md](specs/REPLY_SPEC.md)（含 generator + policy）

### `src/reply/session_memory.py`
- **定位**: L4 会话缓存
- **改什么**: 工具缓存 TTL、缓存 key 生成规则
- **不改什么**: 工具实现本身

### `src/action/message_sender.py`
- **定位**: L4 执行
- **改什么**: 发送方式、剪贴板处理、快捷键
- **不改什么**: 回复内容决策
- **Spec**: [specs/ACTION_SPEC.md](specs/ACTION_SPEC.md)

### `src/action/ui_interactor.py`
- **定位**: L4 坐标/UI 操作
- **改什么**: 聊天列表点击、输入框聚焦、坐标点击逻辑
- **不改什么**: 回复内容决策
- **依赖**: 由 `VisionPipeline` / `LayoutParser` 提供 `ChatListItem` 坐标

### `src/llm/openclaw_client.py`
- **定位**: L4 LLM 客户端
- **改什么**: Kimi 本地代理连接配置
- **不改什么**: 业务逻辑

### `src/utils/qwen_client.py`
- **定位**: L4 LLM 客户端
- **改什么**: DashScope API 调用参数
- **不改什么**: 业务逻辑

### `src/bot/wechat_bot.py`
- **定位**: L5 编排
- **改什么**: 主循环流程、错误处理、多会话管理
- **原则**: 保持薄（thin），只负责调用各层，不包业务逻辑
- **Spec**: [specs/BOT_SPEC.md](specs/BOT_SPEC.md)

### `src/badcase/case_generator.py`
- **定位**: L5 辅助工具
- **改什么**: benchmark case 生成逻辑
- **不改什么**: 生产运行逻辑

### `src/badcase/judge_worker.py`
- **定位**: L5 辅助工具
- **改什么**: Judge LLM 评估逻辑
- **不改什么**: 生产运行逻辑

### `src/badcase/review_server.py`
- **定位**: L5 辅助工具
- **改什么**: 人工审核 Web 服务

### `src/memory/engine.py`
- **定位**: L4 长期记忆
- **改什么**: Wiki 更新 prompt、BM25 搜索参数、别名发现规则、外挂 overrides 格式
- **不改什么**: 消息去重、回复生成
- **Spec**: [specs/MEMORY_SPEC.md](specs/MEMORY_SPEC.md)

### `src/tools/tool_registry.py`
- **定位**: L4 工具注册
- **改什么**: 新增工具、工具参数 schema、内置工具实现
- **不改什么**: LLM 调用逻辑
- **Spec**: [specs/TOOLS_SPEC.md](specs/TOOLS_SPEC.md)

### `src/utils/`
- **定位**: L1-L5 共享工具
- **改什么**: 群聊检测正则、名称归一化规则、XML 解析逻辑、文本压缩策略
- **不改什么**: 业务逻辑
- **核心约束**: 被 2 个及以上模块使用的规则必须放在此处，禁止分散实现
- **Spec**: [specs/UTILS_SPEC.md](specs/UTILS_SPEC.md)

### `src/utils/chat_utils.py`
- **定位**: L1-L5 共享工具
- **改什么**: 群聊检测正则、名称归一化

### `src/utils/text_utils.py`
- **定位**: L1-L5 共享工具
- **改什么**: 文本压缩、格式化

### `src/utils/xml_utils.py`
- **定位**: L1-L5 共享工具
- **改什么**: XML 解析、转义

### `src/utils/debug_logger.py`
- **定位**: L4 调试工具
- **改什么**: 日志字段、序列化格式

### `src/utils/llm_client.py`
- **定位**: L4 LLM 抽象
- **改什么**: 通用 LLM 调用封装

### `src/logging/bot_logger.py`
- **定位**: L4 可观测性
- **改什么**: 日志级别、execution.jsonl 事件类型、埋点位置
- **不改什么**: 业务决策逻辑
- **排查必读: [../03-guides/LOGGING_DESIGN.md](../03-guides/LOGGING_DESIGN.md)`

### `src/session/global_store.py`
- **定位**: L4 持久化 + 会话去重
- **改什么**: 分片策略、查询接口、去重算法、旧版迁移逻辑、截图保留策略
- **不改什么**: 业务决策逻辑
- **排查必读: [../03-guides/LOGGING_DESIGN.md](../03-guides/LOGGING_DESIGN.md)`

> 注：`src/storage/chat_history.py` 在 ARCHITECTURE.md 中标注为"待拆分/尚未创建"，当前持久化职责由 `src/session/global_store.py` 承担。

### Benchmark 测试文件
- `src/benchmarks/ocr_quality.py` — 私有 OCR 质量评估核心（代表性/回归分层）
- `src/benchmarks/private_runner.py` — 私有 benchmark 编排、版本留痕和机器候选筛选
- `scripts/run_private_benchmarks.py` — 缓存优先的统一命令入口与报告
- `src/tests/test_reply_quality_benchmark.py` — 回复质量评估（24 cases）
- `src/tests/test_reply_quality_benchmark_v2.py` — 回复质量多维度评估
- `src/tests/test_reply_stability_benchmark.py` — 回复稳定性一致性
- `src/tests/test_tool_decision_benchmark.py` — 工具决策评估（27 cases）
- `src/tests/test_memory_search_benchmark.py` — 记忆搜索评估（29 cases）
- `src/tests/test_chat_list_unread_benchmark.py` — 未读角标评估（23 cases）
- `src/tests/test_judge_quality_benchmark_v2.py` — 私有固定 GT + 数据库新增标签的 Judge 质量评估（多轮多数投票）

---

## 依赖图

```
models/base.py
    ↑
    ├── capture/window_capture.py
    ├── ocr/vision_ocr.py
    ├── layout/profile.py
    │       ↑
    │   layout/layout_parser.py
    │
    ├── message/extractor.py
    │       ↑
    │   perception/vision_pipeline.py  ← 纯本地 OCR 管道
    │       ↑
    │   perception/smart_pipeline.py  ← 主力：本地预判 + API 兜底
    │       ↑
    │   session/global_store.py
    │   reply/policy.py
    │   llm/openclaw_client.py      ← Kimi 本地代理
    │   utils/qwen_client.py          ← DashScope API
    │       ↑
    │   reply/generator.py
    │   action/message_sender.py
    │   action/ui_interactor.py
    │       ↑
    │   bot/wechat_bot.py
    │
    ├── utils/                      ← L1-L5 共享
    │
    ├── badcase/                    ← L5 辅助，独立不进入生产链
    │
    ├── logging/bot_logger.py
    └── session/global_store.py     ← L4 持久化+去重（storage/chat_history 待拆分）
```

**注意**: 箭头方向表示 "被依赖"。没有循环依赖。

**新增依赖规则**: `logging` 和 `session/global_store` 可被 Bot (L5) 直接依赖，但不可被 L1-L3 依赖。
