# 代码审查：核心模块稳定性与正确性

> 审查日期：2026-06-05
> 审查范围：8 个核心模块（~5300 行源码）
> 审查方法：逐文件精读 + 中文 Code Review 规范分级标注
> 关联文档：[PROJECT_AUDIT_20260504](./PROJECT_AUDIT_20260504.md)

---

## 总体评价

整体架构思路清晰，五层感知→推理→行动→记忆→飞轮的设计完整且有层次。SmartPipeline 的像素预判优化（92.6% tick 跳过 API）、消息发送的三重安全校验、LCS 去重思路、别名自动发现等都是亮点。本次审查聚焦**长期运行稳定性**和**边界条件正确性**，发现 5 项必须修复问题、6 项建议修改、4 项参考建议。

---

## [必须修复] 5 项

### 1. `_extract_json` 括号深度计数不处理字符串内括号

**文件**：`src/reply/generator.py` 第 524-557 行

**问题**：原实现用括号深度计数 `depth++`/`depth--` 寻找 JSON 边界，但未跟踪是否在字符串内部。当 LLM 返回的 JSON 值中包含 `{` 或 `}` 字符时（如 `{"replies": ["这是{测试}消息"]}`），`depth` 会在字符串内的 `}` 处归零，导致 JSON 被截断解析失败。

**影响**：LLM 回复偶尔包含花括号字符时，`_parse_replies` 返回空列表，bot 无回复但也不报错，静默丢失回复。

**修复**（已实施）：

```python
# 修复前：手动括号计数，不区分字符串内外
depth = 0
for i in range(start, len(text)):
    ch = text[i]
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            return json.loads(text[start:i + 1])

# 修复后：使用 json.JSONDecoder.raw_decode() 精确解析
decoder = json.JSONDecoder()
obj, _ = decoder.raw_decode(text, start)
return obj
```

`raw_decode()` 内部完整实现了 JSON 字符串转义解析，能正确处理字符串内的花括号、嵌套 JSON 等。

**验证**：语法检查通过。

---

### 2. LCS 算法 O(m×n) 无上限保护

**文件**：`src/session/global_store.py` 第 186-219 行

**问题**：`_lcs_match` 的 DP 表大小为 `len(history) × len(tick)`，且 `_match_single` 内部调用 `SequenceMatcher.ratio()` 也是 O(L²)。整体复杂度 O(m × n × L²)。大群聊（历史 200 条，tick 50 条）下每个 tick 耗时可能到秒级，影响 bot 响应延迟。

**影响**：活跃群聊中 bot tick 延迟升高，严重时可能超时。

**修复**（已实施）：

```python
# 新增：限制参与比对的 history 范围
_MAX_HISTORY_FOR_LCS = 80
if m > _MAX_HISTORY_FOR_LCS:
    _logger.info(f"[LCS] history={m} 超过上限 {_MAX_HISTORY_FOR_LCS}，截取最近部分")
    history = history[-_MAX_HISTORY_FOR_LCS:]
    m = _MAX_HISTORY_FOR_LCS
```

80 条足以覆盖一次 tick 的消息范围（通常 < 30 条），同时将最坏情况从 200×50=10000 格降到 80×50=4000 格。

**验证**：语法检查通过。

---

### 3. tick_log SQLite 连接未用 context manager

**文件**：`src/bot/wechat_bot.py` 第 342-389 行

**问题**：原代码在 `try` 块内 `conn.commit(); conn.close()`，如果 `execute` 抛异常，`conn.close()` 不会执行，连接泄漏。SQLite 长期运行会积累 `OperationalError: database is locked` 或文件句柄耗尽。

**影响**：长期运行后 tick_log 写入失败频率上升，badcase 分析数据缺失。

**修复**（已实施）：

```python
# 修复前
try:
    conn = get_db()._get_conn()
    conn.execute(...)
    conn.commit(); conn.close()  # 异常时不会执行
except Exception as e:
    self.logger.warning("tick_log 写入失败: %s", e)

# 修复后
conn = None
try:
    conn = get_db()._get_conn()
    conn.execute(...)
    conn.commit()
except Exception as e:
    self.logger.warning("tick_log 写入失败: %s", e)
finally:
    if conn:
        try:
            conn.close()
        except Exception:
            pass
```

**验证**：语法检查通过。

---

### 4. 剪贴板保存/恢复竞态

**文件**：`src/action/message_sender.py` 第 284-298、442-450 行

**问题**：`send()` 在开始时 `pbpaste` 保存原始剪贴板，结束时 `pbcopy` 恢复。如果两个 `send()` 调用间隔很短（bot 连续回复两条消息），第二次 `send()` 保存的 `original_clipboard` 是第一次 `send()` 写入的内容（`pbcopy(text)` 的结果），用户原始剪贴板内容永久丢失。

**影响**：用户剪贴板中复制的链接、代码片段等内容被静默替换为 bot 回复文本。

**修复**（已实施）：

```python
# 新增：类级别发送锁
self._send_lock = threading.Lock()

# send() 改为持锁后委托
def send(self, text, chat_name=""):
    with self._send_lock:
        return self._send_impl(text, chat_name)

def _send_impl(self, text, chat_name=""):
    # 原有 send 逻辑不变
```

整个 save_clipboard → send → restore_clipboard 流程串行化，确保剪贴板操作不被打断。

**验证**：语法检查通过。

---

### 5. Memory Worker `_do_update` 异常后任务丢失

**文件**：`src/memory/engine.py` 第 1022-1024 行

**问题**：Worker 循环 `for task in batch: self._do_update(task)` 没有 try/except。`_do_update` 内部 `_try_generate_wiki` 最多重试 3 次后抛异常，但如果出现未预期的异常（如 `KeyError`、网络断开、JSON 解析失败），整个 batch 剩余任务全部跳过。且这些任务已从队列中移除，永久丢失。

**影响**：wiki 更新静默失败，用户画像信息过时，且无日志可追溯。

**修复**（已实施）：

```python
# 修复前
for task in batch:
    self._do_update(task)
    time.sleep(1)

# 修复后
for task in batch:
    try:
        self._do_update(task)
    except Exception as e:
        _logger.error(f"Worker 处理任务失败: {e}, task_type={task.get('type')}, "
                      f"user={task.get('user_name') or task.get('group_name')}")
    time.sleep(1)
```

单条任务异常不影响 batch 中剩余任务，且 error 日志包含任务信息便于排查。

**验证**：语法检查通过。

---

## [建议修改] 6 项

### 6. SmartPipeline 稳定模式无退出机制

**文件**：`src/perception/smart_pipeline.py`

**问题**：`_consecutive_low_diff` 计数器只递增（hash 一致时）或归零（diff 高于阈值时）。一旦进入稳定模式（`_consecutive_low_diff >= 3`），阈值降低到 `0.0005`。后续即使界面有变化，只要 diff < 0.0005 就仍然跳过。长时间无消息时计数器只增不减，稳定模式永远无法退出。

**建议**：增加超时退出机制：

```python
self._stable_mode_entered_at = None

# 进入稳定模式时记录时间
if not self._stable_mode and self._consecutive_low_diff >= 3:
    self._stable_mode = True
    self._stable_mode_entered_at = time.time()
    self._pixel_diff_threshold *= 0.5

# 检查超时（5 分钟）
if self._stable_mode and self._stable_mode_entered_at:
    if time.time() - self._stable_mode_entered_at > 300:
        self._stable_mode = False
        self._pixel_diff_threshold = self._original_threshold
        self._consecutive_low_diff = 0
        _logger.info("[Pipeline] 稳定模式超时退出，恢复原始阈值")
```

---

### 7. 多线程共享状态无锁保护

**文件**：多个模块

**问题清单**：

| 变量 | 文件 | 线程 | 风险 |
|------|------|------|------|
| `last_raw_response` | generator.py | tick_log 读，ReAct 循环写 | 数据不一致 |
| `last_generation_trace` | generator.py | tick_log 读，ReAct 循环写 | 列表并发读写 |
| `_last_switch_time` | wechat_bot.py | 主线程读写 | 未来引入定时器时竞态 |
| `_update_queue` | engine.py | enqueue 和 worker 可能跨线程 | `List` 非线程安全 |

**建议**：
- `last_generation_trace` 改为每 tick 传参而非类属性
- `_update_queue` 用 `queue.Queue` 替换 `List`
- 至少给 `last_*` 调试字段加 `threading.Lock`

---

### 8. 大量硬编码参数

**文件**：多个模块

**问题**：20+ 处硬编码数字散落在各模块中，调参需改源码重启。

| 参数 | 当前值 | 文件 |
|------|--------|------|
| 像素差异阈值 | 0.001 | smart_pipeline.py |
| 稳定模式因子 | 0.5 | smart_pipeline.py |
| Fuzzy matching 阈值 | 0.80 | global_store.py |
| 消息最大条数 | 200 | global_store.py |
| ReAct 最大轮数 | 10 | generator.py |
| 工具超时 | 25s/60s/600s | generator.py |
| Worker 批量大小 | 3 | engine.py |
| Worker 间隔 | 5s | engine.py |
| LCS history 上限 | 80 | global_store.py（本次新增） |

**建议**：统一到 `config.yaml` 或 `pydantic.BaseSettings`，支持环境变量覆盖：

```yaml
pipeline:
  pixel_diff_threshold: 0.001
  stable_mode_factor: 0.5
  stable_mode_timeout: 300

dedup:
  fuzzy_threshold: 0.80
  lcs_max_history: 80
  max_messages: 200
```

---

### 9. WeFlow 模式切换后无回退

**文件**：`src/bot/wechat_bot.py`

**问题**：初始化时注入 WeFlow 历史后切换到 OCR 模式。如果 OCR 持续失败（窗口被遮挡、截屏异常），没有回退到 WeFlow 的逻辑。

**建议**：增加 OCR 连续失败计数器，超过阈值自动回退 WeFlow 模式：

```python
self._ocr_fail_count = 0
self._OCR_FAIL_THRESHOLD = 5

# tick 中 OCR 失败时
self._ocr_fail_count += 1
if self._ocr_fail_count >= self._OCR_FAIL_THRESHOLD:
    self.logger.warning("OCR 连续失败 %d 次，回退 WeFlow 模式", self._ocr_fail_count)
    self.mode = "weflow"

# OCR 成功时重置
self._ocr_fail_count = 0
```

---

### 10. 别名过滤逻辑重复

**文件**：`src/memory/engine.py` 第 591-618 行 vs 第 630-668 行

**问题**：`_extract_aliases_from_user_wiki` 和 `_extract_aliases_from_group_wiki` 有几乎相同的过滤逻辑（`invalid_keywords`、长度检查、标点检查），但各自硬编码 `invalid_keywords` 列表，容易只改一处忘改另一处。

**建议**：提取公共方法：

```python
_INVALID_ALIAS_KEYWORDS = ["说", "提到", "认为", "和", "与", "让", "叫", "是", "在", "觉得", "告诉", "问", "回答", "表示", "介绍", "@"]
_INVALID_ALIAS_PUNCTS = set('。，；！？.,;!?')
_MAX_ALIAS_LENGTH = 30

def _validate_alias(self, alias: str, main_name: str, existing_mains: set) -> bool:
    """校验别名是否有效：非空、非主名、非他人主名、长度合规、无描述性关键词、无标点。"""
    if not alias or alias == main_name:
        return False
    if alias in existing_mains and alias != main_name:
        return False
    if len(alias) > _MAX_ALIAS_LENGTH:
        return False
    if any(kw in alias for kw in _INVALID_ALIAS_KEYWORDS):
        return False
    if any(c in alias for c in _INVALID_ALIAS_PUNCTS):
        return False
    return True
```

---

### 11. Worker 队列无容量限制

**文件**：`src/memory/engine.py`

**问题**：`_update_queue` 是纯 `List`，无最大长度。如果 LLM 持续 429/超时，队列无限增长（每 5s 取 3 条，但 enqueue 可能更快），内存持续上涨。

**建议**：

```python
_MAX_QUEUE_SIZE = 100

def enqueue_update(self, task):
    with self._queue_lock:
        if len(self._update_queue) >= self._MAX_QUEUE_SIZE:
            # 丢弃最旧的任务
            self._update_queue.pop(0)
            _logger.warning("Worker 队列已满，丢弃最旧任务")
        self._update_queue.append(task)
```

---

## [仅供参考] 4 项

### 12. 单文件职责过重

| 文件 | 行数 | 建议拆分 |
|------|------|---------|
| generator.py | 981 | → `skill_router.py` + `prompt_builder.py` + `reply_parser.py` |
| engine.py | 1078 | → `wiki_manager.py` + `alias_discovery.py` + `bm25_search.py` |
| smart_pipeline.py | 942 | → `pixel_diff.py` + `ocr_merge.py` + `pipeline.py` |

### 13. async/sync 混用

项目混用 `asyncio` 和 `threading`，部分 `async def` 实际同步调用。长期建议统一异步模型，推荐全 `threading`（RPA 场景偏 I/O 密集，async 优势不大且增加复杂度）。

### 14. 缺少 trace_id

多轮 ReAct 循环和异步流程的日志难以串联追踪。建议在 tick 级别注入 `trace_id`，贯穿 `wechat_bot → generator → memory engine` 的所有日志输出。

### 15. 缺少集成测试

30+ 测试文件偏单元测试。RPA 场景建议增加 mock 截图 → SmartPipeline → GlobalStore → WeChatBot 的集成测试链路，覆盖端到端的消息去重和回复生成流程。

---

## 修复状态跟踪

| # | 优先级 | 问题 | 状态 | 修改文件 |
|---|--------|------|------|---------|
| 1 | 必须修复 | `_extract_json` 字符串内括号误判 | ✅ 已修复 | generator.py |
| 2 | 必须修复 | LCS O(m×n) 无上限 | ✅ 已修复 | global_store.py |
| 3 | 必须修复 | SQLite 连接泄漏 | ✅ 已修复 | wechat_bot.py |
| 4 | 必须修复 | 剪贴板竞态 | ✅ 已修复 | message_sender.py |
| 5 | 必须修复 | Worker 任务丢失 | ✅ 已修复 | engine.py |
| 6 | 建议修改 | 稳定模式无退出 | ⬜ 待实施 | smart_pipeline.py |
| 7 | 建议修改 | 线程安全 | ⬜ 待实施 | 多文件 |
| 8 | 建议修改 | 硬编码参数 | ⬜ 待实施 | 多文件 |
| 9 | 建议修改 | WeFlow 无回退 | ⬜ 待实施 | wechat_bot.py |
| 10 | 建议修改 | 别名过滤重复 | ⬜ 待实施 | engine.py |
| 11 | 建议修改 | Worker 队列无上限 | ⬜ 待实施 | engine.py |
| 12 | 仅供参考 | 单文件过大 | ⬜ 低优先 | - |
| 13 | 仅供参考 | async/sync 混用 | ⬜ 低优先 | - |
| 14 | 仅供参考 | 缺 trace_id | ⬜ 低优先 | - |
| 15 | 仅供参考 | 缺集成测试 | ⬜ 低优先 | - |

---

## 值得学习的地方

- **像素预判优化**：92.6% 的 tick 跳过 API 调用，思路实用且效果显著
- **消息发送安全链**：frontmost 校验 + 异常内容熔断 + 剪贴板清理，三层防护意识到位
- **别名自动发现**：从 wiki 内容中提取别名并做严格过滤，巧妙的记忆增强手段
- **数据飞轮设计**：tick_log → badcase 分析 → Judge 质量评分，闭环思路完整
- **WeFlow/OCR 双模式**：初始化用 WeFlow 注入历史，运行切 OCR 降低延迟，模式切换设计优雅

---

## 与 PROJECT_AUDIT_20260504 的关系

本次审查与 5 月 4 日的全面审计互补：

| 维度 | 5月4日审计 | 本次审查 |
|------|-----------|---------|
| 范围 | 全项目 88 项 | 核心模块 15 项 |
| 焦点 | 文档一致性、测试状态、安全、运维 | 长期运行稳定性、边界条件正确性 |
| 深度 | 13 遍交叉审查，覆盖广 | 逐文件精读，问题深挖到根因和修复方案 |
| 修复 | 部分提供方向 | 5 项已实施修复，6 项提供具体代码建议 |

建议将本次 5 项已修复问题同步到 [FIX_LOG.md](./FIX_LOG.md)。
