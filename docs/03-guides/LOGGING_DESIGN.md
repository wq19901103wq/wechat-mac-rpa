# 日志与历史记录设计方案

> 本文档是 `ARCHITECTURE.md` 的补充，专门描述**运行时日志**和**聊天记录持久化**的设计。
> 
> 目标：让任何 AI Agent 在排查问题时，能精准定位到"哪一步、哪个模块、因为什么"出了问题。

---

## 一、设计目标

1. **运行可观测**：每个 tick 的完整决策链路都有结构化记录
2. **故障可回溯**：出问题后能 30 秒内定位到根因模块
3. **存储可扩展**：聊天记录按会话分片，避免单文件过大
4. **AI 友好**：日志格式同时支持人类阅读（emoji + 分级）和机器解析（JSON Lines）

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer (Bot)                                    │
│  - 每个 tick 调用 Logger 记录阶段事件                        │
│  - 每条消息写入 ChatHistory 持久化                           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                             ▼
┌──────────────────────┐              ┌──────────────────────────┐
│  BotLogger           │              │  ChatHistory             │
│  运行时日志系统       │              │  结构化聊天记录存储       │
├──────────────────────┤              ├──────────────────────────┤
│ runtime_YYYYMMDD.log │              │ history/{chat}.jsonl     │
│   → 人类可读，分级    │              │   → 按聊天分片，JSONL    │
│ execution.jsonl      │              │ chat_history.txt         │
│   → 机器可读，逐行JSON│              │   → 人类可读文本日志      │
└──────────────────────┘              └──────────────────────────┘
```

---

## 三、BotLogger 设计（运行时日志）

### 3.1 输出文件

| 文件 | 格式 | 用途 | 保留策略 |
|------|------|------|---------|
| `logs/runtime_YYYYMMDD.log` | 纯文本 | 人类排查报错、堆栈、异常 | 5MB 轮转，保留 3 个备份 |
| `logs/execution.jsonl` | JSON Lines | 机器分析决策链路、grep 过滤 | 无限追加，定期归档 |

### 3.2 日志级别策略

- **DEBUG**: OCR 元素详情、Layout 分组详情、坐标计算中间值
- **INFO**: Tick 开始/结束、截图成功、发现新消息、发送成功
- **WARNING**: 截图失败、OCR 为空、Layout 异常兜底、AppleScript 错误
- **ERROR**: 模块异常、LLM 调用失败、发送失败、未知异常
- **CRITICAL**: 数据目录不可写、权限问题

### 3.3 Execution Log（决策流水）事件类型

`execution.jsonl` 中每行一个 JSON，必须包含 `ts` 和 `event` 字段。

#### `tick_start`
```json
{"ts":"2026-04-15T09:12:34.123456","event":"tick_start","tick_id":42,"interval":5.0}
```

#### `capture`
```json
{"ts":"2026-04-15T09:12:34.456789","event":"capture","tick_id":42,"success":true,"window":{"x":100,"y":100,"width":1280,"height":832}}
```

#### `ocr`
```json
{"ts":"2026-04-15T09:12:35.012345","event":"ocr","tick_id":42,"element_count":23,"duration_ms":456.7,"sample_texts":["群名","昵称","消息内容"]}
```

#### `layout`
```json
{"ts":"2026-04-15T09:12:35.123456","event":"layout","tick_id":42,"chat_name":"测试群","title_elem_count":1,"input_elem_count":2,"timestamp_elem_count":3,"self_bubble_count":2,"message_candidate_count":8}
```

#### `messages`
```json
{"ts":"2026-04-15T09:12:35.234567","event":"messages","tick_id":42,"total_messages":6,"new_messages":1,"message_details":[{"text":"在吗","sender":"小王","sender_type":"other","is_at_me":false}]}
```

#### `decision`
```json
{"ts":"2026-04-15T09:12:36.345678","event":"decision","tick_id":42,"should_reply":true,"reason":"私聊新消息","latest_text":"在吗","reply_text":"在的，有什么可以帮你的？"}
```

#### `send`
```json
{"ts":"2026-04-15T09:12:37.456789","event":"send","tick_id":42,"success":true,"text_length":12,"text_preview":"在的，有什么可以帮你的？","error":null}
```

> **接口与 JSON 映射说明**：`log_send()` 的接口只接收原始 `text: str`，内部会自动计算 `text_length` 和 `text_preview`（前 200 字）后写入 `execution.jsonl`。调用方不需要也不应该自行拆分。

#### `exception`
```json
{"ts":"2026-04-15T09:12:38.567890","event":"exception","tick_id":42,"phase":"ocr","exception_type":"VisionError","exception_msg":"...","traceback":"..."}
```

#### `smart_pipeline_skip`
```json
{"ts":"2026-04-15T09:12:34.567890","event":"smart_pipeline_skip","tick_id":42,"reason":"pixel_diff","diff_ratio":0.0012,"threshold":0.005,"skip_count":86,"api_count":7}
```
触发条件：像素差异 < 阈值，跳过 API 调用。

#### `smart_pipeline_api_call`
```json
{"ts":"2026-05-03T09:12:35.567890","event":"smart_pipeline_api_call","tick_id":42,"model":"qwen3.6-flash","latency_ms":3250,"success":true,"messages_count":3,"chat_list_count":5}
```
触发条件：像素差异 >= 阈值或本地预判为空，调用 qwen3.x-flash 多模态 API 兜底。

#### `smart_pipeline_fallback`
```json
{"ts":"2026-04-15T09:12:37.567890","event":"smart_pipeline_fallback","tick_id":42,"reason":"api_empty_response","fallback_count":1,"local_messages_count":2}
```
触发条件：API 返回空或失败，回退到本地消息提取。

### 3.4 BotLogger 接口

```python
class BotLogger:
    def __init__(self, logs_dir: str = None, max_bytes: int = 5*1024*1024, backup_count: int = 3)
    
    # 通用日志（同时输出到 console 和 runtime log）
    def debug(self, msg: str)
    def info(self, msg: str)
    def warning(self, msg: str)
    def error(self, msg: str, exc_info: bool = False)
    def critical(self, msg: str)
    
    # 结构化执行流水（写入 execution.jsonl）
    def log_tick_start(self, tick_id: int, interval: float)
    def log_capture(self, tick_id: int, success: bool, window_info: dict = None, error: str = None)
    def log_ocr(self, tick_id: int, element_count: int, duration_ms: float, sample_texts: List[str])
    def log_layout(self, tick_id: int, chat_name: str, title_elem_count: int, input_elem_count: int, timestamp_elem_count: int, self_bubble_count: int, message_candidate_count: int)
    def log_messages(self, tick_id: int, total_messages: int, new_messages: int, message_details: List[dict])
    def log_decision(self, tick_id: int, should_reply: bool, reason: str, latest_text: str, reply_text: str = None, extra: dict = None)
    def log_send(self, tick_id: int, success: bool, text: str, error: str = None)
    def log_exception(self, tick_id: int, phase: str, exc: Exception)
    def log_stats(self, tick_id: int, stats: dict)
```

### 3.5 在 Bot 主循环中的埋点位置

```python
# 伪代码（当前模块化架构埋点风格）
def tick(self):
    self.tick_id += 1
    tick_id = self.tick_id
    logger.log_tick_start(tick_id, self.interval)
    
    try:
        # 1. Perception (Capture + OCR + Layout + Extract)
        result = self.vision_pipeline.perceive()
        if not result.success:
            logger.log_capture(tick_id, False, error=result.error)
            return
        logger.log_capture(tick_id, True, window_info=result.window_info)
        logger.log_ocr(tick_id, result.element_count, result.ocr_duration_ms, result.sample_texts)
        logger.log_layout(tick_id, result.chat_name, result.title_elem_count,
                          result.input_elem_count, result.timestamp_elem_count,
                          result.self_bubble_count, result.message_candidate_count)
        
        # 2. Session (去重)
        new_messages = self.chat_session.filter_new(result.messages)
        logger.log_messages(tick_id, len(result.messages), len(new_messages),
                           [m.to_dict() for m in new_messages])
        
        # 3. Decision (Policy + Generator)
        if new_messages:
            latest = new_messages[-1]
            should, reason = self.reply_policy.should_reply(latest, self.chat_session)
            
            if should:
                reply = self.reply_generator.generate(latest, self.chat_session)
                logger.log_decision(tick_id, True, reason, latest.text, reply)
                
                # 4. Action
                ok = self.message_sender.send(reply)
                logger.log_send(tick_id, ok, reply, error=None if ok else "applescript_failed")
            else:
                logger.log_decision(tick_id, False, reason, latest.text)
    except Exception as e:
        logger.log_exception(tick_id, "main_loop", e)
    
    time.sleep(self.interval)
```

---

## 四、ChatHistory 设计（聊天记录持久化）

### 4.1 存储目录结构

```
~/wechat-mac-rpa/data/
├── history/
│   ├── 测试群.jsonl               # 按聊天名称分片
│   ├── 小王.jsonl
│   ├── _unknown.jsonl
│   └── 测试群_export.json          # 导出文件（按需生成）
├── logs/
│   ├── runtime_20260415.log
│   ├── execution.jsonl
│   └── chat_history.txt            # 人类可读汇总
└── screenshots/
    └── wechat_20260415_091234_123.png
```

### 4.2 为什么用 JSON Lines？

- **追加友好**：不需要读整个文件再重写
- **grep 友好**：`grep "测试群" history/测试群.jsonl | tail -10`
- **流式读取**：可以逐行解析，不占用大量内存
- **容错性强**：即使某一行损坏，不影响其他行

### 4.3 HistoryRecord 数据模型

```python
@dataclass
class HistoryRecord:
    text: str
    sender: str
    sender_type: str        # self | other | system
    chat_name: str
    is_at_me: bool = False
    timestamp: str = ""     # ISO 格式
    message_hash: str = ""  # md5(chat_name:sender:text:bubble_y)
    confidence: float = 0.0
    bubble_y: int = 0       # 用于位置关联和回声检测
    source: str = "ocr"     # ocr | manual | api
    tick_id: int = 0
    screenshot_path: str = ""
```

**设计要点**：
- `message_hash` 包含 `bubble_y`，避免同一文字在不同位置被去重
- `tick_id` 关联到 execution.jsonl，出问题时可双向追溯
- `screenshot_path` 保留证据，方便人工复核
  - **注意**：Bot 保存截图到 `data/screenshots/` 后会更新此路径为保存后的真实路径
  - 若仍为 `/tmp/wechat_capture_*.png`，说明保存截图时抛异常，需检查 `data/screenshots/` 目录权限

### 4.4 ChatHistory 接口

```python
class ChatHistory:
    def __init__(self, storage_dir: str = None)
    
    # 写入
    def append_messages(self, chat_name: str, messages: List[dict], tick_id: int = 0, screenshot_path: str = "") -> List[HistoryRecord]
    
    # 读取
    def get_messages(self, chat_name: str = None, since: datetime = None, until: datetime = None, limit: int = 500) -> List[HistoryRecord]
    def get_recent_chats(self, hours: float = 24.0, limit: int = 100) -> Dict[str, List[HistoryRecord]]
    def get_last_message(self, chat_name: str) -> Optional[HistoryRecord]
    
    # 统计与导出
    def get_stats(self) -> dict
    def export_chat(self, chat_name: str, output_path: str = None) -> str
    
    # 兼容
    def _migrate_legacy_history()  # 自动迁移旧版 message_history.json
```

### 4.5 旧版兼容性

旧版 `logs/message_history.json` 是一个巨大的聚合 JSON 数组。启动时：
1. 读取旧版文件
2. 按 `chat_name` 拆分到对应的 `{chat_name}.jsonl`
3. 将旧版文件重命名为 `message_history.json.bak.YYYYMMDD`
4. 后续只读写 `.jsonl` 文件

### 4.6 查询示例

```bash
# 查看最近 10 条消息
tail -n 10 ~/wechat-mac-rpa/data/history/测试群.jsonl | jq .

# 查看最近 1 小时的消息
python3 -c "
from src.storage.chat_history import ChatHistory
from datetime import datetime, timedelta
h = ChatHistory()
msgs = h.get_messages('测试群', since=datetime.now()-timedelta(hours=1))
for m in msgs:
    print(m.timestamp, m.sender, m.text)
"
```

---

## 五、排查问题速查表

| 问题现象 | 查看文件 | 关键词 / 命令 |
|---------|---------|--------------|
| Bot 不回复某条消息 | `execution.jsonl` | `grep '"event":"decision"' execution.jsonl \| jq .` |
| 发了乱码/循环发送 | `execution.jsonl` | `grep '"event":"send"' execution.jsonl \| tail -20` |
| OCR 识别不到文字 | `runtime_YYYYMMDD.log` | 搜索 `OCR` 或 `element_count=0` |
| 聊天名识别错误 | `execution.jsonl` | `grep '"event":"layout"' execution.jsonl \| jq '{chat_name,title_elem_count}'` |
| 输入框内容混入消息 | `execution.jsonl` | 检查 `layout` 事件中的 `input_elem_count` 和 `message_candidate_count` |
| 崩溃/异常 | `runtime_YYYYMMDD.log` | 搜索 `ERROR` 或 `CRITICAL`，看堆栈 |
| 想看完整聊天上下文 | `history/{chat}.jsonl` | `cat history/测试群.jsonl \| jq '{timestamp,sender,text}'` |

---

## 六、与现有 MessageStore 的关系

**现状**：`src/storage/message_store.py` 和 `src/storage/chat_history.py` 并存。`MessageStore` 是较早的实现，`ChatHistory` 是按本设计文档的 JSON Lines 分片方案。

**迁移策略**：
1. 新代码优先使用 `ChatHistory` + `BotLogger`
2. `MessageStore` 保留作为兼容层，内部可委托给 `ChatHistory`
3. `src/bot/wechat_bot.py` 已使用 `ChatHistory`，`MessageStore` 仅作历史兼容

---

## 七、待实现清单

- [x] `BotLogger` 设计文档与接口定义
- [x] `ChatHistory` 设计文档与接口定义
- [x] `HistoryRecord` 数据模型
- [x] 将 `BotLogger` 集成到 `src/bot/wechat_bot.py`（重构时）
- [x] 将 `ChatHistory` 集成到 `src/bot/wechat_bot.py` 主循环（在 tick 结束后持久化消息，而非集成到 `ChatSession`）
- [x] 编写 `test_logging.py` 和 `test_chat_history.py`
- [ ] 增加日志清理/归档策略（如 execution.jsonl 超过 100MB 时压缩归档）

