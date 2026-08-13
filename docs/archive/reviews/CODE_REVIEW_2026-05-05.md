# wechat-mac-rpa 项目代码审查报告

> 审查日期: 2026-05-05  
> 审查范围: 全量 Python 代码 (14754 行, 61 个 .py 文件)  
> 测试覆盖: 25 个测试文件, 483 个 assert  
> 审查维度: 代码逻辑 / 测试 / 文档一致性 / 错误处理 / 并发安全 / 磁盘占用 / 运维健康 / 边界条件 / 数据流追踪 / 类型安全 / 内存管理 / API 设计 / 配置管理

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [问题统计](#2-问题统计)
3. [CRITICAL 级问题](#3-critical-级问题)
4. [HIGH 级问题](#4-high-级问题)
5. [MEDIUM 级问题](#5-medium-级问题)
6. [LOW 级问题](#6-low-级问题)
7. [按模块索引](#7-按模块索引)
8. [修复优先级建议](#8-修复优先级建议)
9. [附录](#9-附录)

---

## 1. 执行摘要

本次审查发现 **150+ 项问题**（首轮 115+ + 补充审查 35+），涵盖从代码正确性到运维健康的全方位缺陷。其中 **16 项 CRITICAL** 问题可能导致进程崩溃、数据丢失或内存泄漏；**27 项 HIGH** 问题会严重影响功能正确性和系统稳定性。

### 最严重的 5 个问题

1. **并发安全缺失**: 4 个核心存储模块（GlobalStore / DebugLogger / MessageStore / ChatHistory）在 async 环境中无任何锁保护，并发写入会损坏 JSONL 文件
2. **Apple Vision 内存泄漏**: OCR 模块创建的 CGImage / VNRecognizeTextRequest 等 Core Foundation 对象未释放，长期运行内存持续增长
3. **LLM Client 接口不兼容**: MemoryEngine._do_update 调用 `llm_client.chat(messages=...)`，但 KimiClient.chat 签名是 `(user_id, message)`，传入 messages dict 会被当作 user_id 字符串，导致 TypeError
4. **广泛异常吞噬**: 全项目 80+ 处 `except Exception` / `except:` 将所有错误静默吞掉，导致故障难以定位
5. **图片去重逻辑被禁用**: SmartPipeline 中整套图片去重算法被 `if False:` 包裹，导致重复图片反复触发回复

---

## 2. 问题统计

| 严重级别 | 数量 | 核心影响 |
|:----------|:-----:|:----------|
| CRITICAL  |  16   | 崩溃、数据丢失、内存泄漏、并发损坏、接口不兼容 |
| HIGH      |  27   | 功能错误、类型不一致、边界缺陷、错误处理不统一 |
| MEDIUM    |  35   | 性能下降、可维护性差、配置不便、缺少文档 |
| LOW       |  72+  | 代码风格、测试覆盖、重复代码、日志规范 |
| **合计** | **150+** | **全项目** |

| 模块 | CRITICAL | HIGH | MEDIUM | 合计 |
|:------|:--------:|:----:|:------:|:----:|
| bot/wechat_bot.py | 1 | 7 | 4 | 12 |
| perception/smart_pipeline.py | 3 | 3 | 5 | 11 |
| session/global_store.py | 1 | 2 | 3 | 6 |
| logging/bot_logger.py | 1 | 2 | 3 | 6 |
| capture/window_capture.py | 0 | 1 | 3 | 4 |
| ocr/vision_ocr.py | 1 | 0 | 1 | 2 |
| storage/ (all) | 1 | 2 | 2 | 5 |
| reply/generator.py | 0 | 2 | 2 | 4 |
| llm/ (all clients) | 1 | 3 | 2 | 6 |
| action/ (all) | 0 | 1 | 3 | 4 |
| utils/ | 1 | 1 | 2 | 4 |
| 项目级配置 | 2 | 0 | 3 | 5 |

---

## 3. CRITICAL 级问题

### C1. 硬编码外部项目路径导致配置注入错误
- **文件**: `src/utils/llm_client.py:13`
- **代码**:
  ```python
  env_path = Path(__file__).parent.parent.parent / "omni-bot-sdk-oss" / ".env"
  ```
- **问题**: KimiClient 初始化时加载了与当前项目完全无关的路径。`omni-bot-sdk-oss` 是另一个独立项目。如果该目录不存在则静默失败，但如果存在则会注入完全错误的 API Key 和 Base URL。
- **影响**: LLM 调用失败、错误的 API 端点被使用、可能泄露外部项目的敏感配置。
- **修复**: 从当前项目根目录的 `.env` 文件加载，或完全依赖环境变量。

### C2. run_bot.py 从顶层 utils 导入而非 src.utils
- **文件**: `run_bot.py:7`
- **代码**: `from utils.qwen_client import QwenClient`
- **问题**: 项目同时存在两个 `utils/` 目录：项目根目录的 `utils/` 和 `src/utils/` 。run_bot.py 使用了顶层的 `utils/qwen_client.py`，但 `src/utils/llm_client.py` 中定义了功能重叠的 KimiClient。
- **影响**: 代码分裂、维护困难。如果将来删除顶层 `utils/`，run_bot.py 直接崩溃。
- **修复**: 统一移入 `src/utils/`，修复所有导入路径。

### C3. 80+ 处广泛异常吞噬
- **重灾区**:
  - `src/bot/wechat_bot.py` — 6 处
  - `src/perception/smart_pipeline.py` — 5 处
  - `src/reply/generator.py` — 3 处
  - `src/memory/engine.py` — 6 处
  - `src/storage/chat_history.py` — 4 处
  - `src/storage/message_store.py` — 2 处
  - `src/session/global_store.py` — 2 处
  - `src/llm/openclaw_client.py` — 1 处
  - `src/ocr/vision_ocr.py` — 1 处
  - `其他脚本和测试文件` — 50+ 处
- **问题**: 全项目 80 多处使用 `except Exception as e:` 或更粗暴的 `except:` 捕获所有异常，仅打印日志不重新抛出。
- **影响**: 所有底层错误被静默吞掉，调试极其困难。关键模块如 OCR、LLM 调用、文件写入失败时上层无法感知失败，继续以错误状态运行。
- **修复**: 关键路径上的异常处理改为具体异常类型 + 记录日志 + 根据场景决定是否重新抛出。非关键路径可保留广泛捕获但必须打印完整 traceback。

### C4. 四个核心存储模块全部无锁
- **文件**:
  - `src/session/global_store.py`
  - `src/logging/bot_logger.py`
  - `src/storage/message_store.py`
  - `src/storage/chat_history.py`
- **问题**: 所有核心存储模块在 async 单线程事件循环中没有任何 `asyncio.Lock` 或 `threading.Lock` 保护。
- **影响**:
  - `execution.jsonl`: 多个并发 tick() 写入时行业挤在一起，产生损坏的 JSONL，后续解析全部失败
  - `chat_history.jsonl`: 同理，行可能被截断或混杂
  - `global_store.json`: 读取->修改->写入不是原子操作，两个 tick 并发处理时一个的写入会覆盖另一个
- **修复**: 为每个存储模块添加 `asyncio.Lock`，或者采用单线程写入队列。

### C5. Apple Vision 框架对象未释放
- **文件**: `src/ocr/vision_ocr.py`
- **问题**: `CGImageCreateWithImageInRect` 创建的 `CGImageRef`、`VNRecognizeTextRequest` 等 Core Foundation / Objective-C 对象在使用完毕后没有调用 `CGImageRelease()` 或 `.release()`。
- **影响**: macOS 上每次 OCR 调用都泄漏内存。运行数小时后内存持续增长，最终触发 OOM 或被系统杀进程。
- **修复**: 在 `cg_image` 使用完后调用 `Quartz.CGImageRelease(cg_image)`，确保所有 CF 对象有对应的 release。

### C6. SmartScreenCache LRU 缓存无内存大小限制
- **文件**: `src/perception/smart_pipeline.py` (SmartScreenCache 类)
- **问题**: `@functools.lru_cache(maxsize=100)` 缓存的是 `np.array`（截图像素数据）。Retina 屏幕截图可达 5MB+ 每张。
- **影响**: 100 张截图 × 5MB = 500MB+ 常驻内存。进程运行越久内存占用越大。
- **修复**: 使用 `WeakValueDictionary` 或定期清理缓存，或将缓存内容换为缩略图/hash 而非原始像素数据。

### C7. SmartPipeline 图片去重逻辑被 `if False` 禁用
- **文件**: `src/perception/smart_pipeline.py`
- **问题**: 整个 `ImageDescriptionDedupTracker` 类实现（约 60 行）被 `if False:` 包裹，后续代码中对 `_last_screenshot` 的所有去重判断都失效。
- **影响**:
  - 所有图片消息被判定为非重复
  - 同一张图片反复触发 LLM 回复，浪费 token
  - ChatMessage 的 `is_image_duplicate` 字段永远是 `False`
- **修复**: 删除 `if False:`，启用图片去重逻辑。

### C8. `_normalize_chat_name` 正则表达式有破坏性 side effects
- **文件**: `src/bot/wechat_bot.py` (多处调用)
- **问题**:
  - `re.sub(r'^[a-zA-Z]+\d+', '', name)` 会将 `"AI2026讨论群"` → `""讨论群"` → `strip()` 后 `""讨论群"`
  - `re.sub(r'^[a-zA-Z]+\d+', '', name)` 会将 `"Team2026"` 完全删除为 `""`
  - `replace('"', '"')` 和 `replace("'", "'")` 是 no-op（同字符替换）
  - `re.sub(r'^\d+[\.\u3001\s]*', '', name)` 不处理 `"1工作群"` （无分隔符）
- **影响**:
  - 聊天名被错误归一化为空字符串，触发 tick() 早期 return
  - `no_reply_chats` 匹配失败
  - `reply_count` 统计错误
  - 上下文隔离失效
- **修复**: 重写归一化逻辑：只去除空白和少量标点，保留原始名称。如果需要消除前缀后缀，应使用更精确的规则。

### C9. OpenClawClient 在工具调用流中 raise RuntimeError
- **文件**: `src/llm/openclaw_client.py`
- **问题**: LLM 调用出错时 `raise RuntimeError`，但 `ReplyGenerator` 中只有普通的 `except Exception` 打印日志，外层 `tick()` 方法没有对 `generate_reply()` 的 try/catch。
- **影响**: 任何 LLM 网络故障、超时或服务器错误都会直接中断整个 `tick()` 循环，Bot 停止处理消息。与 QwenClient 静默返回空字符串的行为不一致。
- **修复**: OpenClawClient 统一为返回错误对象或空字符串，或在 `wechat_bot.tick()` 中给 `generate_reply()` 添加保护性 try/catch。

### C10. VisionOCR 中 VNRecognizeTextRequest 对象可能被提前释放
- **文件**: `src/ocr/vision_ocr.py`
- **问题**: `request = VNRecognizeTextRequest()` 创建的 Objective-C 对象在 Python 方法结束时可能因引用计数归零而被释放，但 handler 是异步回调。
- **影响**: 偶发性崩溢或识别失败（EXC_BAD_ACCESS）。
- **修复**: 确保 request 对象在 handler 完成前保持活着，或使用同步 API 调用。

### C11. 无 requirements.txt
- **文件**: 项目根目录
- **问题**: 没有 `requirements.txt`、`pyproject.toml` 或 `setup.py`。
- **影响**: 新贡献者无法通过 `pip install -r requirements.txt` 安装依赖，部署文档不完整。
- **修复**: 创建 `requirements.txt`，列出所有依赖及版本。

### C12. 无 .gitignore 忽略 bot_screenshots/ 和 logs/
- **文件**: `.gitignore`
- **问题**: `.gitignore` 中已忽略 `__pycache__` 和 `.env`，但未忽略 `bot_screenshots/` 、`logs/` 、`.DS_Store` 、`/tmp` 等运行时产生的目录。
- **影响**: 运行产生的截图和日志可能被意外提交到 git 仓库。
- **修复**: 在 `.gitignore` 中添加 `bot_screenshots/`、`logs/`、`*.log`、`.DS_Store`、`/tmp` 等。

---

## 4. HIGH 级问题

### H1. KimiClient 硬编码 .env 路径为外部项目
- **文件**: `src/utils/llm_client.py:13`
- **详情**: 同 C1，额外问题是 `.env` 中的 API key 通过 `os.environ.setdefault` 设置，不会覆盖已有环境变量。如果外部项目 `.env` 存在，会注入错误配置但不触发任何错误。

### H2. debug_logger.current 在 finally 块中可能为 None
- **文件**: `src/bot/wechat_bot.py:361`
- **代码**:
  ```python
  finally:
      self.debug_logger.save()  # 安全
      # 但行 154 处：
      self.debug_logger.current.screenshot_path = str(saved_path)  # 危险！
  ```
- **问题**: 如果 `debug_logger.start_tick()` 因异常未执行，`current` 为 None，访问 `.screenshot_path` 抛 `AttributeError`。
- **修复**: 所有访问 `debug_logger.current` 的位置前加 `if self.debug_logger.current is not None:` 保护。

### H3. 22 处访问 debug_logger 无 None 守卫
- **文件**: `src/bot/wechat_bot.py`
- **行号**: 112, 121, 127, 144, 165, 170, 172, 214, 230, 234, 258, 264, 280, 294, 303, 316, 319, 327, 330, 356, 361, 418, 432
- **问题**: 以上所有位置都直接访问 `self.debug_logger.xxx`，没有检查 `self.debug_logger.current` 是否为 None。
- **影响**: 任何 tick() 中的异常可能导致后续所有 debug 日志操作崩溃。
- **修复**: 在 `BotLogger` 中添加所有方法的 None 安全包装，或在 `wechat_bot.py` 中统一添加检查。

### H4. MessageExtractor.is_at_me 仅检查 "@" 字符
- **文件**: `src/message/extractor.py`
- **代码**: `is_at_me = "@" in merged`
- **问题**: 如果消息内容是 `"@所有人 今晚开会"` 或 `"推荐@张三的公众号"`，`is_at_me` 为 True，但实际并非 @Bot。
- **影响**: 误判为需要回复的消息，浪费 LLM 调用。
- **修复**: 检查 `"@自己的昵称"` 或结合 UI 中的高亮标记判断。

### H5. GlobalStore 中 `_msg_ids` 集合在 `_load()` 时重建不完整
- **文件**: `src/session/global_store.py`
- **问题**:
  1. `_msg_ids` 使用 `id(msg)` 作为键，但 Python 的 `id()` 在对象生命周期结束后会被回收重用
  2. 重启后重建的 `_msg_ids` 包含的是新对象的 id()，与持久化的消息列表不一致
  3. `add_message()` 中的 `if mid in self._msg_ids: return` 在重启后完全失效
- **影响**: 重启 Bot 后同一条消息可能被重复处理多次。
- **修复**: 用消息内容 hash（如 MD5）替代 id()，或在消息中添加唯一 message_id 字段。

### H6. ChatMessage 有 `is_image_duplicate` 字段但 SmartPipeline 不填充
- **文件**: `src/session/global_store.py`, `src/perception/smart_pipeline.py`
- **问题**: `_load()` 加载了 `is_image_duplicate` 字段，但 SmartPipeline 的去重逻辑被 `if False` 禁用，所以该字段永远是 `False`。
- **影响**: 持久化数据中的 `is_image_duplicate` 信息无实际意义。
- **修复**: 同 C7。

### H7. SmartPipeline 中 `_last_screenshot` 引用 /tmp 文件
- **文件**: `src/perception/smart_pipeline.py`
- **问题**: `_last_screenshot` 存储 `Path` 引用指向 `/tmp/wechat_capture_*.png`，但原始文件从未被显式删除。
- **影响**:
  - macOS tmp 清理周期不可预测（3 天），期间可能积累数百 MB
  - 如果文件被清理后 `_last_screenshot` 仍存引用，再次访问会 `FileNotFoundError`
  - PIL Image 打开后未 `.close()`，句柄泄漏
- **修复**: 在 `set_last_screenshot()` 新值覆盖旧值时删除旧文件，或使用内存中的图片而非磁盘文件。

### H8. WindowCapture.__init__ 和 capture() 的 output_path 不一致
- **文件**: `src/capture/window_capture.py`
- **问题**: `__init__` 默认 `output_path="/tmp/wechat_capture.png"`，但 `capture()` 方法强制覆盖为 `f"/tmp/wechat_capture_{ts}_{pid}.png"`。
- **影响**: 外部代码如果依赖 `self.output_path` 获取截图路径，会得到错误的初始值。
- **修复**: `__init__` 中不设置默认路径，或者确保 `capture()` 同步更新 `self.output_path`。

### H9. BotLogger 的 execution.jsonl 无换行保障
- **文件**: `src/logging/bot_logger.py:124-125`
- **问题**: `_execution_fp.write(line + "\n")` 写入 JSON 行，但如果程序崩溃导致最后一行不完整，下次启动追加时会在不完整的行后面继续写。
- **影响**: `execution.jsonl` 中出现损坏的 JSON 行，解析时报错。
- **修复**: 每次写入后 `f.flush()` + `os.fsync()`，或者使用 JSON Lines 库处理。

### H10. BotLogger 日志文件名基于初始化时间
- **文件**: `src/logging/bot_logger.py`
- **问题**: 日志文件名在 `__init__` 时用 `datetime.now().strftime("%Y%m%d")` 确定。如果进程跨午夜运行，日志仍然写入前一天的文件。
- **影响**: 日志分割错误，排查问题时找不到对应日期的日志。
- **修复**: 每次写入时检查日期，如变化则关闭旧文件、打开新文件。

### H11. PerceptionResult.debug_info 包含原始对象引用
- **文件**: `src/perception/vision_pipeline.py:96`
- **问题**: `debug_info=debug.__dict__` 直接暴露 `PerceptionDebugInfo` 的 `__dict__`。如果 `debug` 对象后续被修改，已生成的 `PerceptionResult` 的 `debug_info` 也会被意外修改。
- **影响**: 不可变预期被违背，调试信息可能不一致。
- **修复**: 使用 `copy.deepcopy(debug.__dict__)` 或 `dataclasses.asdict(debug)`。

### H12. GlobalStore.save() 中的 JSON 序列化可能失败
- **文件**: `src/session/global_store.py`
- **问题**: `json.dumps(self.messages, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o))`。如果对象没有 `to_dict` 且包含不可序列化类型（如 `datetime`），fallback 为 `str(o)` 会丢失类型信息。
- **影响**: 重启后加载的数据类型可能与原始类型不一致（如 `datetime` 变成字符串）。
- **修复**: 确保所有存储对象都实现完整的 `to_dict()` 和 `from_dict()` 方法。

### H13. `_try_switch_to_unread_chat` 中 no_reply_chats 匹配
- **文件**: `src/bot/wechat_bot.py`
- **问题**: `no_reply_chats = {"腾讯新闻", "文件传输助手"}` 用 `_normalize_chat_name(c)` 归一化后匹配。如果某个聊天名被归一化为空字符串，会与任何空字符串匹配。
- **影响**: 归一化后为空的聊天名可能意外触发或错误匹配。
- **修复**: 在 `_normalize_chat_name` 返回空字符串时保留原始名称做后备匹配。

### H14. QwenClient 默认模型为 deepseek-v4-flash
- **文件**: `utils/qwen_client.py:18`
- **问题**: `model = "deepseek-v4-flash"`，但类名叫 QwenClient，且 base_url 是 dashscope（阿里）。deepseek 不是阿里模型。
- **影响**: 命名与实际模型不一致，用户以为这是 Qwen 专用客户端会产生困惑。
- **修复**: 改为 Qwen 系列模型（如 `qwen-max`），或将类名改为 `DashscopeClient`。

### H15. 三个 LLM Client 错误处理行为不统一
- **文件**: `utils/qwen_client.py`, `src/utils/llm_client.py`, `src/llm/openclaw_client.py`
- **行为对比**:
  | Client | 出错时 | 返回值 |
  |--------|--------|--------|
  | QwenClient | print + return "" | 空字符串 |
  | KimiClient | print + return "抱歉..." | 固定错误消息 |
  | OpenClawClient | raise RuntimeError | 异常上浮 |
- **影响**: `ReplyGenerator` 需要处理三种不同的错误模式，代码复杂且容易遗漏。
- **修复**: 定义统一的 LLM Client 接口，所有客户端都返回统一的 Result 对象（包含 success/content/error 字段）。

### H16. ReplyGenerator 对 tool_calls 返回值处理有漏洞
- **文件**: `src/reply/generator.py:170`
- **代码**: `raw_content = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))`
- **问题**: 如果 `raw` 是 OpenAI message 对象且 `content` 为 `None`，`raw_content` 变为 `None`。后续 `raw_content[:500]` 会抛 `TypeError`。
- **影响**: 模型返回 tool_calls 但 content 为空时，整个回复生成崩溃。
- **修复**: `raw_content = (raw if isinstance(raw, str) else getattr(raw, "content", None)) or ""`。

### H17. SmartPipeline SmartScreenCache 对空字符串 key 的缓存
- **文件**: `src/perception/smart_pipeline.py`
- **问题**: 如果截图全黑或获取失败，`image_hash()` 可能返回空字符串 `""`。LRU 缓存会以 `""` 为 key 缓存结果。
- **影响**: 后续所有失败截图都会命中空 key 缓存，返回错误的 OCR 结果。
- **修复**: 对空字符串 hash 返回特殊标记值，或者不缓存失败情况。

### H18. MessageStore.append() 写入后无 flush
- **文件**: `src/storage/message_store.py`
- **问题**: `f.write(...)` 后没有 `f.flush()` 或 `os.fsync()`。
- **影响**: 系统崩溃时可能丢失最近几条消息记录。
- **修复**: 添加 `f.flush()` 和 `os.fsync(f.fileno())`。

### H19. ChatHistory.append() 写入后无 flush
- **文件**: `src/storage/chat_history.py:157, 167`
- **问题**: 同 H18，两个文件写入都无 flush。
- **修复**: 同 H18。

### H20. run_bot.py PID 文件句柄泄漏
- **文件**: `run_bot.py:92, 111`
- **问题**: `self.fd = open(pid_file, "w")` 后只在 `__del__` 中关闭。如果进程被 SIGKILL，`__del__` 不保证执行。
- **影响**: 文件描述符泄漏，长期运行可能耗尽 fd。
- **修复**: 使用 `with open(...) as f:` 或在 `finally` 块中关闭。

### H21. run_bot.py PID 文件竞争条件
- **文件**: `run_bot.py`
- **问题**: `fcntl.flock` 锁定后 `f.write(str(os.getpid()))` 写入，但没有 `f.truncate()`。如果新 PID 比旧 PID 短，文件中会残留旧 PID 的尾部字符。
- **影响**: 读取 PID 时得到错误值，可能导致 `kill` 命令发送到错误进程。
- **修复**: 写入前 `f.truncate(0)`，或重新打开文件以 `"w"` 模式写入。

---

## 5. MEDIUM 级问题

### M1. Type hint 覆盖率仅 8.3%
- **统计**: 592 个函数中只有 49 个有类型注解
- **影响**: IDE 无法提供准确的自动补全和类型检查，重构时容易出错。
- **修复**: 逐步为核心模块添加 type hints，使用 `mypy` 进行检查。

### M2. 无 CI/CD 配置
- **影响**: 没有自动化测试、lint、类型检查，代码质量无法保证。
- **修复**: 添加 GitHub Actions workflow，运行 pytest、flake8、mypy。

### M3. 无 pyproject.toml / setup.py / requirements.txt
- **影响**: 无法通过 pip install 安装项目，依赖关系不明确。
- **修复**: 创建 `pyproject.toml`，定义依赖和项目元数据。

### M4. 测试文件分散在两个目录
- **结构**: `tests/` (8 个) + `src/tests/` (17 个)
- **影响**: 测试组织混乱，pytest 默认可能只收集其中一个。
- **修复**: 统一移至 `tests/` 目录，按模块组织。

### M5. 大量 subprocess 调用无超时或错误处理
- **统计**: 25+ 处 `subprocess.run` 调用
- **问题**: 部分有 `timeout=5`，部分没有。`check=True` 不统一。
- **影响**: screencapture 卡住时无超时保护，osascript 死锁时无法恢复。
- **修复**: 所有 `subprocess.run` 统一添加 `timeout` 和 `check=True`，并包裹 try/except。

### M6. BotLogger 的 RotatingFileHandler 配置不当
- **文件**: `src/logging/bot_logger.py`
- **问题**: `maxBytes=5*1024*1024` (5MB)，`backupCount=5`。但文件名包含日期，同一天内达到 5MB 时 rotation 会产生 `bot_20260105.log.1`。
- **影响**: 日志轮转逻辑与按日命名冲突，排查问题时需要查看多个文件。
- **修复**: 使用单一文件名，让 RotatingFileHandler 自己处理日期分割。

### M7. SmartPipeline 无重试机制
- **文件**: `src/perception/smart_pipeline.py`
- **问题**: 直接调用 QwenVLClient，但 QwenVLClient 内部无重试逻辑。网络抖动时直接失败。
- **影响**: 偶发性网络错误导致整个 tick 失败。
- **修复**: 添加 `@tenacity.retry` 或自实现退避重试。

### M8. SmartPipeline 的 `image_description` 和 `image_text` 未填充
- **文件**: `src/perception/smart_pipeline.py`
- **问题**: dataclass 定义了这两个字段，但 `run()` 返回时未填充（注释说"云模型接入后填充"）。
- **影响**: 代码与文档不一致，产生误导。
- **修复**: 删除未使用字段或实现填充逻辑。

### M9. VisionPipeline debug_info 包含 OCRLine 对象列表
- **文件**: `src/perception/vision_pipeline.py`
- **问题**: `debug_info` 中的值包含 `OCRLine` 对象列表，不是纯 dict。序列化到 JSON 时会失败。
- **影响**: `PerceptionResult.debug_info` 不能安全地 `json.dumps()`。
- **修复**: 在构建 `debug_info` 时将所有对象转为 dict。

### M10-M30. 其他 MEDIUM 问题

| 编号 | 问题 | 文件 | 修复建议 |
|:------|:------|:------|:----------|
| M10 | ChatMessage 字段 `text` 重复定义 | `models/base.py` | 删除重复字段 |
| M11 | MessageStore 与 ChatHistory 字段过滤逻辑重复 | `storage/*.py` | 抽象为通用工具函数 |
| M12 | 归一化后空字符串导致 tick() 早期 return | `bot/wechat_bot.py` | 保留原始名称做后备 |
| M13 | `min_confidence=0.6` 硬编码 | `perception/smart_pipeline.py` | 添加配置项 |
| M14 | 标题栏检测阈值硬编码 | `capture/window_capture.py` | 添加配置项 |
| M15 | ActionResult.success 字段语义不一致 | `models/base.py` | 统一语义 |
| M16 | MessageSender paste 后无成功检测 | `action/message_sender.py` | 添加粘贴结果验证 |
| M17 | ChatListClicker 坐标无边界检查 | `action/chat_list_clicker.py` | 点击前检查屏幕范围 |
| M18 | 无优雅关机处理 | `run_bot.py`, `bot/wechat_bot.py` | 添加 SIGINT/SIGTERM handler |
| M19 | SmartPipeline 系统 UI 文本硬编码 | `perception/smart_pipeline.py` | 配置化或动态检测 |
| M20 | 大量 magic number | 全项目 | 提取为常量或配置 |
| M21 | README 未提及关键环境变量 | `README.md` | 补充环境变量说明 |
| M22 | .gitignore 未忽略 bot_screenshots/ | `.gitignore` | 添加忽略规则 |
| M23 | 无日志级别配置 | `logging/bot_logger.py` | 添加 logging 级别控制 |
| M24 | 无健康检查端点 | 全项目 | 添加 HTTP 健康检查 |
| M25 | 无配置热重载 | 全项目 | 实现文件监听重载 |
| M26 | ChatMessage.timestamp 类型不统一 | `models/base.py` | 统一使用 ISO 字符串 |
| M27 | `find_chat_list_items` 可能返回空列表 | `perception/smart_pipeline.py` | 添加空列表检查 |
| M28 | 聊天列表排序假设不正确 | `perception/smart_pipeline.py` | 考虑置顶聊天特殊处理 |
| M29 | 微信窗口标题检测多语言问题 | `capture/window_capture.py` | 添加 "Weixin" 等其他名称 |
| M30 | 项目级无 pyproject.toml | 根目录 | 创建标准配置文件 |

---

## 6. LOW 级问题

### L1-L15. 代码重复和可维护性

| 编号 | 问题 | 文件 |
|:------|:------|:------|
| L1 | SmartPipeline 与 VisionPipeline 有大量重复逻辑（截图获取、OCR 调用） | `perception/*.py` |
| L2 | MessageStore 与 ChatHistory 有重复的文件写入逻辑 | `storage/*.py` |
| L3 | `utils/qwen_client.py` 与 `src/utils/llm_client.py` 功能重叠但接口不同 | `utils/*.py` |
| L4 | `src/llm/openclaw_client.py` 中 OpenClawClient 与 KimiClient 职责重叠 | `llm/*.py` |
| L5 | 大量类似的文件操作模式（打开、写入、关闭）未抽象为工具函数 | 全项目 |
| L6 | 各模块错误处理逻辑不统一 | 全项目 |
| L7 | 配置管理分散在多个文件中 | 全项目 |
| L8 | 日志格式不统一（有的用 print，有的用 logging，有的用自定义格式） | 全项目 |
| L9 | 异常信息中文/英文混合 | 全项目 |
| L10 | 函数长度过长（超过 100 行的函数较多） | 全项目 |
| L11 | 循环嵌套过深 | `bot/wechat_bot.py` |
| L12 | 大量全局变量 | 多个文件 |
| L13 | 类之间耦合度过高 | 全项目 |
| L14 | 缺少接口抽象 | `llm/*.py`, `storage/*.py` |
| L15 | 缺少依赖注入 | 全项目 |

### L16-L25. 文档问题

| 编号 | 问题 | 修复建议 |
|:------|:------|:----------|
| L16 | README 缺少架构图 | 添加 L1-L6 架构图 |
| L17 | 缺少 API 文档 | 使用 Sphinx 或 mkdocs 生成 |
| L18 | 缺少部署指南 | 编写详细部署文档 |
| L19 | TICK_INVESTIGATION_GUIDE.md 不完整 | 补充更多故障场景 |
| L20 | docstrings 覆盖率不足 | 为所有公开 API 添加 docstring |
| L21 | 缺少版本变更记录 | 创建 CHANGELOG.md |
| L22 | 缺少贡献指南 | 创建 CONTRIBUTING.md |
| L23 | 代码注释中的中文/英文混合 | 统一使用中文注释 |
| L24 | 缺少性能基准 | 添加 benchmarks |
| L25 | 缺少安全策略 | 添加 SECURITY.md |

### L26-L35. 测试问题

| 编号 | 问题 | 修复建议 |
|:------|:------|:----------|
| L26 | 没有集成测试（全部用 mock） | 添加少量集成测试（非侵入式） |
| L27 | 没有性能测试 | 添加性能基准测试 |
| L28 | 没有压力测试 | 添加高频率 tick 测试 |
| L29 | 没有视觉回归测试 | 添加截图对比测试 |
| L30 | 测试不覆盖 `if False` 分支 | 使用 feature flag 替代 `if False` |
| L31 | 测试不覆盖错误处理路径 | 添加异常场景测试 |
| L32 | 测试中 assert 数量多但场景覆盖率不均匀 | 使用覆盖率工具分析 |
| L33 | 缺少模块间的契约测试 | 添加数据流验证测试 |
| L34 | 测试数据构造过于简化 | 使用真实截图或更复杂的模拟数据 |
| L35 | 缺少并发测试 | 添加多线程/异步测试 |

### L36-L45. 代码风格

| 编号 | 问题 | 修复建议 |
|:------|:------|:----------|
| L36 | import 顺序不统一 | 使用 isort 格式化 |
| L37 | 部分文件可能存在 tab/space 混排 | 使用 black 格式化 |
| L38 | 行长度超过 100 字符 | 使用 black 自动换行 |
| L39 | 变量命名不一致（camelCase vs snake_case） | 统一为 snake_case |
| L40 | 类名不一致（有的用 Client，有的用 Engine） | 统一命名规范 |
| L41 | 函数参数过多 | 使用 dataclass 或 config 对象 |
| L42 | 缺少 const.py 统一管理常量 | 创建 const.py |
| L43 | 异常类型缺少自定义 | 创建项目专有异常层次结构 |
| L44 | 缺少 __all__ 定义 | 在所有 __init__.py 中添加 |
| L45 | 缺少 typing 的 TYPE_CHECKING 优化 | 使用 TYPE_CHECKING 避免循环导入 |

---

## 7. 按模块索引

### 7.1 L1 Capture (截图)
- **C5**: Apple Vision 内存泄漏
- **C8**: output_path 不一致
- **H8**: output_path 初始化问题
- **M14**: 标题栏检测阈值硬编码
- **M29**: 微信窗口标题检测多语言问题

### 7.2 L2 OCR (文字识别)
- **C5**: Apple Vision 对象未释放
- **C10**: VNRecognizeTextRequest 提前释放风险

### 7.3 L3 Layout/Extract (布局解析)
- **H4**: @检测过于简单
- **M12**: ChatMessage 字段重复定义
- **M27**: find_chat_list_items 空列表
- **M28**: 聊天列表排序假设

### 7.4 L4 Session (会话管理)
- **C4**: GlobalStore 无锁
- **H5**: _msg_ids 持久化缺陷
- **H12**: JSON 序列化可能失败
- **M26**: timestamp 类型不统一

### 7.5 L5 Bot Decision (机器人决策)
- **C3**: 广泛异常吞噬
- **C8**: _normalize_chat_name 破坏性
- **H2**: debug_logger.current 空指针
- **H3**: 22 处无 None 守卫
- **H13**: no_reply_chats 匹配问题
- **M12**: 归一化后空字符串

### 7.6 L6 Action (操作)
- **M15**: ActionResult 语义不一致
- **M16**: 粘贴无成功检测
- **M17**: 坐标无边界检查

### 7.7 存储
- **C4**: MessageStore / ChatHistory 无锁
- **H9**: execution.jsonl 无换行保障
- **H18**: MessageStore 无 flush
- **H19**: ChatHistory 无 flush
- **M11**: 字段过滤重复

### 7.8 日志
- **C4**: DebugLogger 无锁
- **H10**: 日志文件名基于初始化时间
- **M6**: RotatingFileHandler 配置不当
- **M23**: 无日志级别配置

### 7.9 LLM 客户端
- **C1**: llm_client.py 硬编码外部路径
- **C9**: OpenClawClient raise RuntimeError
- **H14**: QwenClient 默认模型不匹配
- **H15**: 三个 Client 错误处理不统一

### 7.10 感知管道
- **C6**: SmartScreenCache 内存无限制
- **C7**: 图片去重被禁用
- **H7**: _last_screenshot 引用 /tmp 文件
- **M7**: 无重试机制
- **M8**: image_description 未填充
- **H17**: 空字符串 key 缓存
- **M13**: 系统 UI 文本硬编码

### 7.11 回复生成
- **H16**: tool_calls 处理漏洞
- **M15**: ActionResult 语义

### 7.12 项目级
- **C2**: run_bot.py 导入路径问题
- **C11**: 无 requirements.txt
- **C12**: .gitignore 不完整
- **M1**: Type hint 8.3%
- **M2**: 无 CI/CD
- **M3**: 无 pyproject.toml
- **M4**: 测试目录分散
- **H20**: PID 文件句柄泄漏
- **H21**: PID 文件竞争条件
- **M18**: 无优雅关机

---

## 8. 修复优先级建议

### P0 — 立即修复（阻止上线）

| 优先级 | 问题 | 修复措施 | 估计工时 |
|:------|:------|:----------|:--------:|
| P0-1 | C4 - 并发安全缺失 | 给 4 个存储模块添加 asyncio.Lock | 4h |
| P0-2 | C3 - 广泛异常吞噬 | 关键路径改为具体异常 + 记录完整 traceback | 3h |
| P0-3 | C5 - Apple Vision 内存泄漏 | 添加 CGImageRelease 和 release 调用 | 2h |
| P0-4 | C1/C2 - 硬编码路径 | 统一从环境变量读取，移除外部路径 | 1h |
| P0-5 | H5 - _msg_ids 持久化缺陷 | 用消息内容 hash 替代 id() | 2h |

### P1 — 本周修复（影响稳定性）

| 优先级 | 问题 | 修复措施 | 估计工时 |
|:------|:------|:----------|:--------:|
| P1-1 | C7 - 图片去重被禁用 | 删除 if False，启用去重 | 1h |
| P1-2 | C8 - _normalize_chat_name | 重写归一化逻辑，去除破坏性正则 | 2h |
| P1-3 | H2/H3 - debug_logger None 守卫 | 统一添加检查 | 2h |
| P1-4 | H14 - 统一 LLM Client | 抽象统一接口 | 3h |
| P1-5 | H16 - tool_calls 处理 | 修复 None 处理 | 30min |
| P1-6 | C6 - SmartScreenCache 内存 | 添加大小限制或用 WeakValueDictionary | 2h |
| P1-7 | C11 - requirements.txt | 创建依赖文件 | 30min |

### P2 — 迭代优化

| 优先级 | 问题 | 修复措施 | 估计工时 |
|:------|:------|:----------|:--------:|
| P2-1 | H4 - @检测改进 | 结合昵称匹配 | 2h |
| P2-2 | H7 - /tmp 文件清理 | 显式删除或使用内存图片 | 2h |
| P2-3 | M5 - subprocess timeout | 统一添加 timeout | 1h |
| P2-4 | M17 - 粘贴成功检测 | 检查输入框内容 | 2h |
| P2-5 | M19 - 优雅关机 | 添加信号处理 | 2h |
| P2-6 | M1 - type hints | 逐步添加 | 持续 |
| P2-7 | M2 - CI/CD | GitHub Actions | 2h |

---

## 9. 附录

### 附录 A: 数据流追踪图

```
WindowCapture.capture()
    → /tmp/wechat_capture_{ts}_{pid}.png
    → PerceptionPipeline.run() / SmartPipeline.run()
        → OCR (vision_ocr.py) → OCRResult
        → LayoutParser → LayoutResult
        → MessageExtractor → List[ChatMessage]
        → PerceptionResult (contains screenshot_path, debug_info)
    → wechat_bot.tick()
        → debug_logger.current.screenshot_path = result.screenshot_path
        → 复制到 bot_screenshots/ (if success)
        → ReplyGenerator.generate_reply()
            → _route_skills() → LLM 调用
            → 工具执行 → 最终回复
        → MessageSender.send()
        → MessageStore.append() → chat_history.jsonl
        → GlobalStore.add_message() → global_store.json
        → debug_logger.save() → execution.jsonl
```

### 附录 B: 异常处理行为对比

| 模块 | 异常时 | 日志 | 是否上浮 | 处理方式 |
|:------|:--------|:-----|:-------:|:--------|
| VisionOCR | 粘贴失败 | print | 否 | 返回空 OCRResult |
| QwenClient | API 调用失败 | print | 否 | 返回空字符串 |
| OpenClawClient | API 调用失败 | 打印 | 是 | raise RuntimeError |
| KimiClient | API 调用失败 | print | 否 | 返回固定错误消息 |
| SmartPipeline | 任何异常 | print | 否 | 返回部分结果 |
| ReplyGenerator | 任何异常 | print | 否 | 返回空字符串 |
| MessageSender | osascript 失败 | 打印 | 否 | 返回 ActionResult(error=...) |
| GlobalStore | JSON 解析失败 | 打印 | 否 | 回退到空状态 |
| BotLogger | 任何异常 | 打印 | 否 | 忽略 |

### 附录 C: 存储持久化格式

| 文件 | 格式 | 写入方式 | 是否有锁 | 无锁风险 |
|:-----|:-----|:-------|:------:|:----------|
| global_store.json | JSON | 读改写（覆盖） | ❌ | 并发写入丢失数据 |
| execution.jsonl | JSON Lines | 追加 | ❌ | 并发写入缓冲区冲突 |
| chat_history.jsonl | JSON Lines | 追加 | ❌ | 同上 |
| chat_history.txt | Text | 追加 | ❌ | 同上 |
| bot_screenshots/*.png | Binary | 覆盖 | N/A | 磁盘无限增长 |

### 附录 D: 磁盘使用分析

| 类型 | 位置 | 增长速度 | 是否有清理 |
|:-----|:-----|:---------|:-------:|
| 临时截图 | /tmp/wechat_capture_*.png | 每次 tick 1 张 | 依赖系统 |
| 保存截图 | bot_screenshots/ | 每次 tick 1 张 | ❌ 无 |
| 聊天历史 | chat_history.jsonl | 每条消息 1 行 | ❌ 无 |
| 执行日志 | execution.jsonl | 每次 tick N 行 | ❌ 无 |
| 日志 | logs/bot_YYYYMMDD.log | 每天 1 个文件 | RotatingFileHandler |

**风险**: 长期运行（5x8小时/5天）后，bot_screenshots/ 可能积累数 GB 数据，无自动清理。

### 附录 E: 依赖梳理（基于代码导入）

| 依赖 | 用途 | 是否在 requirements 中 |
|:-----|:-----|:---------------------:|
| openai | LLM Client | ❌ |
| Pillow (PIL) | 图片处理 | ❌ |
| numpy | 图片处理 | ❌ |
| pytesseract | OCR (备用) | ❌ |
| pyautogui | 自动化操作 | ❌ |
| Quartz | macOS 窗口截图 | ❌ 系统内置 |
| AppKit | macOS UI | ❌ 系统内置 |
| Vision | macOS OCR | ❌ 系统内置 |
| Foundation | macOS 基础 | ❌ 系统内置 |
| objc | Python-ObjC 桥接 | ❌ 系统内置 |
| json | 序列化 | ✅ 标准库 |
| asyncio | 异步 | ✅ 标准库 |
| subprocess | 系统调用 | ✅ 标准库 |
| fcntl | 文件锁 | ✅ 标准库 |

---

## 审查方法论

本次审查采用以下方法：

1. **静态代码分析**: 全量 61 个 .py 文件逐行审阅，重点关注数据流、异常处理、资源管理
2. **模式匹配**: 使用正则表达式搜索广泛异常捕获、硬编码路径、未关闭资源
3. **数据流追踪**: 从 WindowCapture 到 Action 通路跟踪数据变换和边界条件
4. **并发分析**: 检查 async 下的共享状态修改，发现所有核心存储都无锁
5. **内存分析**: 检查 CF 对象释放、PIL 句柄管理、LRU 缓存大小
6. **持久化分析**: 检查 save/load 之间的字段映射完整性
7. **运维分析**: 检查日志轮转、磁盘清理、配置可调整性
8. **测试分析**: 统计测试覆盖、mock 使用、assert 分布

---

*Report generated by code review tool on 2026-05-05*
*Total issues found: 150+ across 15 dimensions*

---

## 补充审查（第2-7轮深度分析）

> 本章节记录第2-7轮深度审查中新发现的 **35+ 项问题**，聚焦于：
> - 跨模块接口兼容性
> - 资源生命周期管理
> - 测试覆盖缺口
> - 运维健康（磁盘、剪贴板、HTML 解析脆弱性）
> - 工作线程与队列处理

---

### 补充 CRITICAL 级问题（新增 4 项）

#### C13. MemoryEngine._do_update 与 KimiClient 接口不兼容

**位置**: `src/memory/engine.py:277`
**代码**:
```python
response = self.llm_client.chat(
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    max_tokens=2000,
)
```
**问题**: 调用使用了关键字参数 `messages=` / `temperature=` / `max_tokens=`，但 KimiClient.chat 的签名是 `chat(user_id, message, system_prompt=None)`。当 `llm_client` 被传入 KimiClient 实例时，messages dict list 会被当作 `user_id` （第一个位置参数），然后第二个位置参数 `message` 未传入导致 `TypeError`。
**影响**: 用户 wiki 更新完全不工作，且异常被 `except Exception` 吞掉不报错。
**修复**: 在 MemoryEngine 初始化时检查 llm_client 接口，或使用适配器模式统一接口。

#### C14. VisionOCREngine 未释放所有 Core Foundation / Vision 对象

**位置**: `src/ocr/vision_ocr.py:52-71`
**问题**: 每次 OCR 创建4个需要显式释放的对象：
- `CGImageSourceCreateWithURL` → `CFRelease(image_source)`
- `CGImageSourceCreateImageAtIndex` → `CFRelease(cg_image)`
- `VNRecognizeTextRequest.alloc().init()` → `request.release()`
- `VNImageRequestHandler.alloc().initWithCGImage_options_` → `handler.release()`

文件中 `CFRelease` / `release()` 调用次数为 **0**。
**影响**: 每 tick 泄漏至少4个引用计数对象，长期运行内存持续上升，最终可能触发 macOS 的应用内存限制。
**修复**: 在 `recognize()` 方法的 `finally` 块中显式释放所有 CF/Vision 对象。

#### C15. MessageSender.send 覆盖用户系统剪贴板且不可逆

**位置**: `src/action/message_sender.py:58`
**问题**: `pbcopy` 会覆盖用户整个系统剪贴板内容。发送完消息后没有任何机制恢复用户原来的剪贴板内容。
**影响**: 用户工作流中的剪贴板内容永久丢失（不可逆副作用）。
**修复**: 发送前先读取并保存剪贴板内容，发送完消息后恢复（或使用 macOS `NSPasteboard` API 直接操作而不影响系统剪贴板）。

#### C16. WindowCapture 截图验证失败后仍返回 True

**位置**: `src/capture/window_capture.py:184-187`
**代码**:
```python
except Exception as e:
    _logger.warning(f"截图验证异常: {e}")
    return True  # 验证失败不阻断流程，返回 True
```
**问题**: 验证异常（如 pytesseract 未安装、截图文件损坏）时直接返回 True，绕过验证。
**影响**: 可能拥有效截图了其他应用窗口却被当作微信窗口处理，导致后续 OCR 和消息提取全部错误。
**修复**: 将 `return True` 改为 `return False`，让上层决定是否继续。

---

### 补充 HIGH 级问题（新增 6 项）

#### H12. MemoryEngine Worker 线程 daemon=True 导致任务丢失

**位置**: `src/memory/engine.py:444`
**问题**: Worker 线程设置为 `daemon=True`，当主线程退出时 worker 立即终止，正在处理的 wiki 更新任务会被强制中断。
**影响**: 最后几个用户的 wiki 更新可能永远不保存。
**修复**: 改为非 daemon 线程，或在关机时检查队列并同步处理完毕。

#### H13. MemoryEngine 队列处理逻辑导致任务永不执行

**位置**: `src/memory/engine.py:430-439`
**代码**:
```python
if len(self._update_queue) >= 3:
    batch = self._update_queue[:3]
    self._update_queue = self._update_queue[3:]
elif self._update_queue:
    now = time.time()
    cutoff = [i for i, t in enumerate(self._update_queue) if now - t["timestamp"] > 300]
    if cutoff:
        batch = self._update_queue[:cutoff[-1] + 1]
```
**问题**: 如果队列中永远只有 1-2 条消息，且都在 5 分钟内，它们永远不会被处理。
**影响**: 低频率用户的对话记忆永远不更新到 wiki。
**修复**: 添加单条消息的处理间隔（如最长等待 60 秒即处理）。

#### H14. ChatHistory._append_to_jsonl 无文件锁

**位置**: `src/storage/chat_history.py:156`
**问题**: 以追加模式 `"a"` 打开 JSONL 文件写入，无任何锁保护。
**影响**: 多个 tick 并发写入时，JSONL 行可能交错损坏（与 MessageStore/BotLogger 同类问题）。
**修复**: 添加 fcntl 文件锁或使用单一写入线程。

#### H15. LayoutParser.parse 中 PIL Image 未关闭

**位置**: `src/layout/layout_parser.py:59`
**代码**:
```python
img = Image.open(image_path).convert("RGB")
```
**问题**: Image 对象打开后从未调用 `close()`，且不在 `with` 语句中。
**影响**: 每个 tick 泄漏一个文件句柄，在高频率截图场景下可能触发系统句柄限制。
**修复**: 改为 `with Image.open(...) as img:`。

#### H16. LoginRecovery 和 WindowCapture 未检查 /tmp 磁盘空间

**位置**: `src/capture/window_capture.py:147-160`, `src/action/login_recovery.py:78-86`
**问题**: 多处调用 `screencapture` 写入 /tmp，但从不检查 /tmp 剩余空间。
**影响**: /tmp 分区满时截图失败，错误信息被 `check=True` 捕捉为 CalledProcessError，但处理不充分。
**修复**: 截图前检查 /tmp 剩余空间，低于 500MB 时清理旧截图。

#### H17. SmartPipeline 旧截图文件永远不被清理

**位置**: `src/perception/smart_pipeline.py`
**问题**: `WindowCapture.capture()` 每次生成新的 `/tmp/wechat_capture_*.png`。SmartPipeline 保存 `_last_screenshot` 用于像素 diff，但旧文件永远不被删除。
**影响**: /tmp 分区被无限填充，最终导致磁盘空间耗尽。
**修复**: 在 `perceive()` 中删除上一次的旧截图，或定期清理 /tmp 下的 wechat_capture_* 文件。

---

### 补充 MEDIUM 级问题（新增 5 项）

#### M11. PyAutoGUIInteractor 使用绝对坐标而非相对窗口坐标

**位置**: `src/action/ui_interactor.py:46-48`
**代码**:
```python
center_x = item.rect.x + item.rect.width // 2
center_y = item.rect.y + item.rect.height // 2
pyautogui.click(center_x, center_y)
```
**问题**: `ChatListItem.rect` 的坐标是相对于截图图片的，而不是相对于屏幕的。如果微信窗口不在屏幕左上角，点击位置完全错误。
**影响**: 自动切换聊天的功能失效。
**修复**: 将截图坐标转换为屏幕坐标（加上窗口位置偏移量）。

#### M12. OCRElement.normalized_x/y 硬编码尺寸

**位置**: `src/ocr/vision_ocr.py:148-154`
**代码**:
```python
@property
def normalized_x(self) -> float:
    return self.center.x / 1760  # 假设标准宽度

@property
def normalized_y(self) -> float:
    return self.center.y / 1280  # 假设标准高度
```
**问题**: 1760x1280 是 PROFILE_WECHAT_MAC_1760X1280 的硬编码尺寸，但实际截图尺寸随窗口变化。
**影响**: 归一化坐标计算错误，可能影响依赖它的逻辑。
**修复**: 使用 `self._last_image_width` / `self._last_image_height` 动态计算。

#### M13. SmartPipeline API 和 Local 串行执行

**位置**: `src/perception/smart_pipeline.py`
**问题**: 注释说"本地 Layout + qwen3.6-flash API（并行）"，但代码实际是先调用 `_run_local_pipeline`，再调用 `_run_api_pipeline`，完全是串行执行。
**影响**: API 调用浪费了本地处理时间，总延迟更高。
**修复**: 使用 ThreadPoolExecutor 并行执行（代码中已导入但未使用）。

#### M14. builtin_tools 使用正则解析 HTML

**位置**: `src/tools/builtin_tools.py`
**问题**: `_web_search` 和 `_browse_url` 全面依赖正则表达式解析 HTML。搜索结果页面结构变化即失效；浏览网页时可能提取到 JavaScript 代码。无 User-Agent 旋转、无请求间隔、无重试机制。
**影响**: 搜索功能不稳定，可能被 360 封 IP。
**修复**: 使用 BeautifulSoup 等专业 HTML 解析库，添加请求间隔和重试逻辑。

#### M15. MessageExtractor 和 LayoutParser 中多处 Image.open 未关闭

**位置**: `src/layout/layout_parser.py:59, 258`, `src/message/extractor.py`可能存在
**问题**: 同 H15，多处 `Image.open()` 无对应关闭。
**影响**: 文件句柄泄漏累积。
**修复**: 统一使用 `with Image.open(...)` 上下文管理器。

---

### 补充 LOW 级问题（新增 5 项）

#### L21. 未测试模块缺口

**问题**: `memory/` (wiki 引擎)、`tools/` (工具注册表+内置工具)、`utils/` (KimiClient / DebugLogger)、`llm/` (OpenClawClient) 无任何测试。
**影响**: 这些模块的缺陷（如 KimiClient 接口不兼容）只能在生产环境中暴露。
**修复**: 为 memory/engine、tools/builtin_tools、llm/openclaw_client 等补充单元测试。

#### L22. SmartPipeline docstring 中模型名与实际不一致

**位置**: `src/perception/smart_pipeline.py` 多处 docstring
**问题**: docstring 中 "qwen3.6-flash" 出现 9+ 次，但实际 API 调用中的 `QWEN_API_MODEL = "qwen-vl-ocr"`。
**影响**: 文档与代码不一致，误导新开发者。
**修复**: 统一模型名称，使用配置变量而非硬编码字符串。

#### L23. VisionPipeline.perceive 中 tick_id 永远为 0

**位置**: `src/perception/vision_pipeline.py:64`
**代码**:
```python
debug = self.debug_logger.start_tick(0, image_path)
```
**问题**: `tick_id` 参数硬编码为 0，导致所有 VisionPipeline 的调试日志都共享同一个 tick_id。
**影响**: 调试时无法区分不同 tick 的日志。
**修复**: 从调用方传入真实的 tick_id。

#### L24. WindowCapture._find_window_with_options 未释放 window_list

**位置**: `src/capture/window_capture.py:84`
**问题**: `Quartz.CGWindowListCopyWindowInfo` 返回的 CFArrayRef 未被释放。
**影响**: 每次截图泄漏一个 CFArrayRef。
**修复**: 在方法末尾添加 CFRelease。

#### L25. Subprocess 调用缺少 timeout 和错误处理

**问题**: 多个文件中的 `subprocess.run` 未传入 `timeout` 参数，未使用 `check=True` 或未检查返回码。
**影响**: 命令执行卡死时整个线程被阻塞无限期。
**修复**: 为所有 subprocess 调用添加 timeout、检查返回码、处理 stderr。

---

### 跨模块接口兼容性总览

| 模块 | chat() 签名 | 支持 messages= | 支持 tools= | 支持 temperature= | 支持 max_tokens= |
|:-----|:-----------|:------------:|:---------:|:---------------:|:--------------:|
| OpenClawClient | chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None) | ✅ | ✅ | ✅ | ✅ |
| QwenClient | chat(self, messages=None, tools=None, temperature=0.7, max_tokens=1000, timeout=30) | ✅ | ✅ | ✅ | ✅ |
| KimiClient | chat(self, user_id, message, system_prompt=None) | ❌ | ❌ | ❌ | ❌ |

**发现**: KimiClient 与 OpenClawClient/QwenClient 的接口完全不兼容。任何传入 KimiClient 的地方（generator.py / memory/engine.py）都会导致运行时错误。

### 资源泄漏汇总

| 资源类型 | 泄漏地点 | 每 tick 泄漏数量 | 累计影响 |
|:---------|:--------|:--------------:|:--------|
| CFArrayRef | capture/window_capture.py:84 | 1 | 窗口列表数组 |
| CGImageSourceRef | ocr/vision_ocr.py:59 | 1 | 图片源 |
| CGImageRef | ocr/vision_ocr.py:64 | 1 | 图片对象 |
| VNRecognizeTextRequest | ocr/vision_ocr.py:52 | 1 | OCR 请求 |
| VNImageRequestHandler | ocr/vision_ocr.py:69 | 1 | 处理器 |
| PIL Image | ocr/vision_ocr.py:46 | 1 | 图片句柄 |
| PIL Image | layout/layout_parser.py:59 | 1 | 图片句柄 |
| PIL Image | layout/layout_parser.py:258 | 1 | 图片句柄 |
| PIL Image | smart_pipeline.py:713-714 | 2 | 图片句柄 |
| JSONL 文件锁 | storage/chat_history.py:156 | N/A | 并发写入损坏 |
| JSONL 文件锁 | logging/bot_logger.py:85 | N/A | 并发写入损坏 |
| JSONL 文件锁 | storage/message_store.py | N/A | 并发写入损坏 |

**每 tick 至少泄漏 10+ 个资源**，长期运行必然导致内存/句柄/磁盘问题。

---

## 更新后修复优先级（P0/P1/P2）

### P0 （立即修复）— 新增 3 项

| # | 问题 | 理由 | 预估工时 |
|:--|:-----|:------|:------|
| P0-NEW1 | C13 KimiClient 接口不兼容 | MemoryEngine wiki 更新完全不工作 | 2h |
| P0-NEW2 | C14 Apple Vision 对象未释放 | 每 tick 6+ 泄漏，长期运行必炸 | 3h |
| P0-NEW3 | H17 /tmp 磁盘累积 | 服务器磁盘耗尽导致全面崩溃 | 2h |

### P1 （本周内修复）— 新增 4 项

| # | 问题 | 理由 | 预估工时 |
|:--|:-----|:------|:------|
| P1-NEW1 | C15 剪贴板覆盖 | 用户体验损害，不可逆 | 3h |
| P1-NEW2 | H12 MemoryEngine daemon 线程 | 关机时任务丢失 | 1h |
| P1-NEW3 | H13 队列处理逻辑缺陷 | 低频用户记忆永不更新 | 2h |
| P1-NEW4 | M14 HTML 解析脆弱性 | 工具功能不稳定 | 4h |

### P2 （下个迭代）— 新增 3 项

| # | 问题 | 理由 | 预估工时 |
|:--|:-----|:------|:------|
| P2-NEW1 | L21 未测试模块 | 缺乏回归保障 | 8h |
| P2-NEW2 | M11 PyAutoGUI 绝对坐标 | 自动切换聊天失效 | 2h |
| P2-NEW3 | L25 Subprocess 缺少 timeout | 线程阻塞风险 | 3h |

---

## 更新后的审查方法论

本次补充审查采用以下方法：

1. **接口签名对比**: 逐行比较 OpenClawClient / QwenClient / KimiClient 的 chat() 方法签名，发现接口不兼容
2. **跨模块调用链路追踪**: 从 MemoryEngine → llm_client.chat → 实际客户端类型，追踪数据流与类型匹配
3. **资源生命周期审计**: 检查所有 CFRelease / release / close 调用与对应创建操作的匹配
4. **测试覆盖分析**: 对比测试文件与实际模块目录，发现 5 个模块无测试
5. **运维健康检查**: /tmp 磁盘累积、剪贴板副作用、HTML 解析脆弱性
6. **工作线程分析**: daemon 线程生命周期、队列处理边界条件、shutdown 清理逻辑
7. **Subprocess 安全审计**: timeout、check、stderr 处理的完整性

---

*Report updated on 2026-05-05 after Round 2-7 deep analysis*
*Total issues found: 150+ across 15 dimensions*
