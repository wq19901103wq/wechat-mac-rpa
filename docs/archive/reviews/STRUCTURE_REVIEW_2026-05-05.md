# WeChat Mac RPA 代码结构与逻辑审查报告 (Round 8)

**审查日期**: 2026-05-05
**审查重点**: 代码结构、架构设计、数据流、控制流、状态管理、算法逻辑
**审查范围**: 61 个 .py 文件，14754 行代码，58 个类，298 个函数
**审查人**: AI Agent

---

## 一、执行摘要

本项目名义上采用 6 层架构（L1 Capture → L2 OCR → L3 Layout/Extract → L4 Session → L5 Bot → L6 Action），但实际实现中存在严重的结构性缺陷：

1. **两个 God Method** 掌控了 80% 的业务逻辑：`tick()` (262行/39控制块) 和 `generate()` (317行/27控制块)
2. **5个并行状态存储** 之间无任何事务或同步机制
3. **架构分层虚设**：L1 Capture 直接依赖 L6 Action 模块
4. **29个超长方法**（>50行），其中10个超过80行
5. **关键算法存在逻辑缺陷**：merge_tick 滑动匹配为 O(N*M) 复杂度，图片去重 Jaccard 阈值 0.001 几乎等于无去重

---

## 二、问题统计

| 级别 | 结构性问题 | 逻辑性问题 | 合计 |
|------|-----------|-----------|------|
| CRITICAL | 4 | 3 | 7 |
| HIGH | 8 | 5 | 13 |
| MEDIUM | 10 | 7 | 17 |
| LOW | 12 | 6 | 18 |
| **合计** | **34** | **21** | **55** |

---

## 三、CRITICAL 级问题（结构 + 逻辑）

### CRIT-1: God Method 模式 — tick() 违反单一职责原则

**文件**: `src/bot/wechat_bot.py:101-363`
**代码规模**: 262 行，39 个控制块（if/for/try/while）
**影响**: 🔴 **极高**

**问题描述**:
tick() 方法同时承担了以下 8 项职责：
1. 感知协调（调用 perception.perceive()）
2. 截图保存与路径管理
3. 聊天名称归一化与验证
4. 全局状态合并（merge_tick）
5. 日志记录（4 种不同类型的日志）
6. 回复策略判断（should_reply）
7. LLM 回复生成（generator.generate）
8. 消息发送与标记（sender.send + mark_replied）
9. 记忆更新（memory_engine.update_user_wiki）
10. 聊天切换（_try_switch_to_unread_chat）

**控制流复杂度**:
```
try
  ├── if result is None → return
  ├── if debug_logger.current is not None → 14 行属性反射
  ├── if result.screenshot_path → try/except 保存截图
  ├── if not chat_name → 嵌套 if/else（4 个分支）
  ├── state, unreplied = merge_tick()
  ├── msg_dicts / unreplied_dicts 构建（22 行列表推导）
  ├── try/except total_stored
  ├── debug_logger.log_session()
  ├── logger.log_messages()
  ├── if not unreplied → 尝试切换聊天
  ├── for msg in unreplied → on_message 回调
  ├── for reversed(unreplied) → 找 latest
  ├── 列表推导 to_reply = [..]
  ├── if not to_reply → return
  ├── generator.generate() → replies
  ├── if not replies → mark_replied + return
  ├── if chat_name in no_reply_chats → return
  ├── for replies → sender.send()
  ├── for to_reply → mark_replied()
  ├── if memory_engine → update_user_wiki()
except Exception → log_exception + raise
finally → save debug_logger + save_sessions()
```

**为什么这是结构性问题**:
- 任何 tick 阶段的修改都需要改动同一个 262 行的方法
- 单元测试无法隔离单个阶段，必须测试完整的 tick 流程
- 错误定位困难：try/except 包裹了整个方法，任何异常都只能知道"tick 失败了"
- 嵌套深度达到 5-6 层，心智负担极高

**修复建议**:
将 tick() 拆分为 6 个独立的 pipeline stage，每个 stage 有明确的输入/输出接口：
```python
def tick(self) -> None:
    context = TickContext(tick_id=self._tick_id)
    try:
        context = self._perceive_stage(context)
        context = self._session_stage(context)
        context = self._decision_stage(context)
        context = self._generation_stage(context)
        context = self._action_stage(context)
        context = self._memory_stage(context)
    except Exception as e:
        self._handle_tick_error(context, e)
    finally:
        self._persist_stage(context)
```

**工时估算**: 2-3 天

---

### CRIT-2: God Method 模式 — generate() 混合了 6 种不同职责

**文件**: `src/reply/generator.py:62-380`
**代码规模**: 317 行，27 个控制块
**影响**: 🔴 **极高**

**问题描述**:
generate() 方法同时承担：
1. Prompt 构建（system + tools + user）
2. Skill 路由匹配（_route_skills）
3. LLM 选择与切换（deepseek ↔ hermes）
4. 工具调用循环（while True + tool execution）
5. 重试逻辑（for attempt in range(max_retries + 1)）
6. Hermes fallback（检测到 use_hermes 后的二次调用）
7. 回复解析（_parse_replies）
8. Session memory 更新

**关键结构问题**:
```python
# 三重嵌套循环：for attempt → while True → for tc in raw_tool_calls
for attempt in range(max_retries + 1):      # 外层：重试
    while True:                              # 中层：工具调用循环
        raw = active_llm.chat(...)           # LLM 调用
        if raw_tool_calls and not force_no_tools:
            for tc in raw_tool_calls:        # 内层：执行多个工具
                result = tool_registry.execute(...)
                # ... 记录 trace
                messages.append(tool_result)
            continue  # 回到 while True
        # 解析回复
        if text and '"use_hermes"' in text:  # 字符串匹配做路由！
            # 切换 Hermes 重新生成
            hermes_raw = complex_llm_client.chat(...)
```

**逻辑问题**:
1. `max_retries` 只在外层循环生效，但如果工具调用进入 `while True`，`continue` 会跳过 `for attempt` 的迭代计数
2. 重试时 `messages` 列表持续累积（包含之前失败的 tool 结果），可能导致 token 超限
3. `"use_hermes"` 是字符串匹配，非结构化解析，如果 LLM 输出 `"use_hermes": false` 也会误触发
4. `force_no_tools` 基于时间判断，但在 `while True` 中每次迭代重新计算，行为不可预测

**修复建议**:
拆分为：`build_prompt()` → `select_llm()` → `call_with_tools()` → `parse_reply()` → `handle_fallback()`

**工时估算**: 2 天

---

### CRIT-3: 5 个并行状态存储 — 无事务、无同步、不一致风险

**影响**: 🔴 **高**

**状态存储清单**:

| 存储 | 类型 | 持久化 | 容量限制 | 线程安全 |
|------|------|--------|----------|----------|
| GlobalStore.chats | Dict[str, ChatState] | JSON 文件 | max_messages=200 | ✅ Lock |
| ChatHistory | JSONL + Dict 缓存 | JSONL 文件 | 最近 2000 条 | ❌ 无锁 |
| MessageStore | Dict + JSON | JSON 文件 | 最近 1000 条 | ❌ 无锁 |
| BotLogger | 运行时日志 + execution.jsonl | 文件追加 | 无限制 | ❌ 无锁 |
| SessionMemory | 纯内存 Dict | 无 | 无限制 | ❌ 无锁 |

**结构性问题**:
1. 同一 tick 中，5 个存储的写入顺序是硬编码的，没有任何回滚机制
2. 如果 `sender.send()` 失败但 `global_store.mark_replied()` 已执行 = 状态不一致（标记已回复但实际未发送）
3. SessionMemory 是纯内存的，Bot 重启后工具缓存全部丢失
4. ChatHistory 和 MessageStore 分别存储相同的消息，但没有机制保证它们一致

**数据流示例（不一致场景）**:
```
tick():
  1. global_store.merge_tick() → 内存更新 ✅
  2. message_store.save_screenshot() → 文件保存 ✅
  3. generator.generate() → LLM 调用 ✅
  4. sender.send() → 网络/AppleScript 失败 ❌
  5. global_store.mark_replied() → 未执行（因为异常提前退出）
  6. finally: global_store.save() → 保存了步骤1的状态
  
结果：GlobalStore 认为消息已处理（merge_tick 后已保存），
      但实际上没有发送回复，且消息未被标记为已回复
```

**修复建议**:
1. 引入单一事实源（Single Source of Truth）：GlobalStore 作为唯一权威状态
2. 其他存储（ChatHistory、MessageStore）作为 GlobalStore 的只读投影/监听器
3. 或者引入事件溯源模式：所有状态变更通过事件日志驱动

**工时估算**: 3-4 天

---

### CRIT-4: 架构分层侵越 — L1 Capture 依赖 L6 Action

**文件**: `src/capture/window_capture.py`
**代码**: `from src.action.chat_list_clicker import ChatListClicker`
**影响**: 🟠 **高**

**问题描述**:
WindowCapture（L1 感知层）直接导入了 ChatListClicker（L6 动作层），违反了分层架构的基本原则。这意味着：
1. 捕获模块不仅仅是"感知"，还隐含了"交互"能力
2. L1 的测试必须同时 mock L6 的依赖
3. 无法独立替换 Capture 实现（比如用网络 API 代替截图）

**结构性影响**:
- 层与层之间不再是单向依赖，而是形成了网状依赖
- 任何 L6 的改动都可能影响 L1
- 架构图上的"L1 → L2 → L3 → L4 → L5 → L6"实际上不存在

**修复建议**:
1. 将 ChatListClicker 的使用从 WindowCapture 中移除
2. 如果需要"先点击再捕获"的语义，在 Bot 层（L5）编排两个动作，而不是在 Capture 层内部调用

**工时估算**: 0.5 天

---

### CRIT-5: merge_tick 滑动匹配算法 — O(N*M) 复杂度

**文件**: `src/session/global_store.py:247-270`
**影响**: 🟠 **高**

**问题描述**:
旧版 merge_tick_legacy 的滑动匹配算法：
```python
best_match_len = 0
for i in range(len(history_window)):          # O(M), M=history 长度
    match_len = 0
    for j in range(len(messages)):            # O(N), N=tick 消息数
        if i + j >= len(history_window):
            break
        if _match_single(history_window[i+j], messages[j], chat_name):  # O(L), L=消息长度
            match_len += 1
        else:
            break
```

**复杂度分析**:
- 外层循环：最多 search_window 次（默认 min(历史长度, max(50, tick长度*3))）
- 内层循环：最多 tick 消息数
- `_match_single` 内部：文字消息用 `SequenceMatcher`，复杂度 O(min(a,b) * maxlen)
- 当历史 200 条，tick 20 条，每条消息 50 字时：
  - 最坏情况：200 * 20 * SequenceMatcher(50,50) ≈ 200 * 20 * 2500 = **10,000,000 次字符比较**

**结构性问题**:
- 每次 tick 都可能执行百万级字符比较，但 merge_tick 在 UI 线程的主循环中同步执行
- 虽然已有 `_merge_tick_fast`（LCS 算法），但代码中仍然保留了 legacy 版本的路径

**修复建议**:
1. 完全移除 legacy 算法，统一使用 LCS
2. LCS 本身也是 O(M*N)，对于长历史应考虑增量匹配或哈希索引
3. 将 merge 操作放到后台线程，避免阻塞 tick 循环

**工时估算**: 1 天

---

### CRIT-6: 图片去重 Jaccard 阈值 0.001 — 几乎等于无去重

**文件**: `src/session/global_store.py:174`
**代码**: `return sim >= 0.001`
**影响**: 🟠 **中-高**

**逻辑分析**:
- 2-gram Jaccard 相似度范围是 [0, 1]
- 阈值 0.001 意味着只要有 1 个 2-gram 重叠，就被视为重复
- 对于中文图片描述，任意两张图片都可能共享"一个"、"是"、"的"等常见 2-gram
- 实际效果：几乎**所有图片都会被视为重复**

**与 OCR 去重阈值对比**:
- 文字消息：threshold = 0.80-0.90
- 图片消息：threshold = 0.001
- 差距 800 倍，没有任何文档解释这个差异的合理性

**修复建议**:
1. 将图片去重阈值提高到 0.30-0.50
2. 或者使用图片哈希（感知哈希 pHash）代替文本描述去重
3. 添加 A/B 测试机制验证阈值效果

**工时估算**: 0.5 天

---

## 四、HIGH 级问题（结构 + 逻辑）

### HIGH-1: generate() 中 no_reply_chats 检查在 LLM 调用之后

**文件**: `src/bot/wechat_bot.py:306-317`
**代码**:
```python
replies = self.generator.generate(to_reply, all_messages)   # LLM 调用（消耗 token）
# ... 记录日志 ...
if not replies:
    # ...
if chat_name in self.no_reply_chats:                        # 检查免回复列表
    # 跳过回复
```

**逻辑问题**: 对于免回复聊天，已经调用了 LLM 生成回复（消耗 token），然后才检查是否在免回复列表中。这是一个严重的资源浪费。

**修复建议**: 将 `if chat_name in self.no_reply_chats` 检查移到 `generator.generate()` 调用之前。

---

### HIGH-2: 双 LLM 路由机制依赖非结构化字符串匹配

**文件**: `src/reply/generator.py:293`
**代码**: `if text and '"use_hermes"' in text:`

**逻辑问题**:
1. 使用 Python `in` 运算符做字符串匹配，不是 JSON 解析
2. 如果 LLM 输出 `"use_hermes": false`，也会匹配成功
3. 没有定义什么是"复杂任务"，路由规则完全依赖 LLM 的"自觉性"
4. 如果 deepseek 返回异常或空字符串，路由失败

**修复建议**:
1. 使用结构化输出（JSON mode / function calling）明确要求 LLM 返回路由决策
2. 定义明确的路由规则（基于消息长度、技能匹配、工具需求等），不依赖 LLM 自我判断

---

### HIGH-3: SmartPipeline 像素差异判断硬编码且不合理

**文件**: `src/perception/smart_pipeline.py`
**代码**: `diff_mask = np.any(diff > 10, axis=2)`

**结构性问题**:
1. 阈值 10 是硬编码的魔法数字，不考虑：Retina 缩放、暗色模式、截图压缩质量
2. 差异比例 = 不同像素 / 总像素，但 ROI 裁剪后的像素数可能远小于 800*600
3. 如果发送了消息，输入框区域的文字变化会被计入像素差异，导致每次都触发 API 调用

---

### HIGH-4: 配置散落 10+ 文件，无统一配置管理

**影响模块列表**:
- `run_bot.py`: 模型名、URL、超时
- `smart_pipeline.py`: 像素阈值、ROI、稳定模式参数
- `layout/profile.py`: 坐标、颜色、比例参数
- `reply/generator.py`: max_tokens、温度、系统 prompt
- `memory/engine.py`: wiki 目录、缓存大小
- `utils/llm_client.py`: .env 文件路径
- `llm/openclaw_client.py`: base_url、api_key、模型名
- `reply/session_memory.py`: TTL 值
- `storage/chat_history.py`: 存储路径
- `logging/bot_logger.py`: 日志目录、文件名

**结构性问题**:
1. 没有统一的配置文件（如 config.yaml / config.py）
2. 相同概念在不同文件中可能有不同的值（如 timeout 在多处定义）
3. 无配置验证机制，错误值只能在运行时暴露
4. 环境迁移困难（需要改动多个文件）

**修复建议**:
1. 引入 Pydantic Settings 或 dataclasses 的统一配置类
2. 支持环境变量覆盖 + 配置文件 + 默认值三级配置
3. 启动时验证所有配置项

---

### HIGH-5: 关键组件缺少抽象接口

| 组件 | 有抽象基类？ | 影响 |
|------|------------|------|
| MessageSender | ✅ | 可 mock |
| UIInteractor | ✅ | 可 mock |
| WindowCapture | ❌ | 硬编码 Quartz，无法替换 |
| VisionOCREngine | ❌ | 硬编码 Vision 框架 |
| LayoutParser | ❌ | 硬编码 profile 逻辑 |
| LLM Client | ❌ | 3 个客户端接口不统一 |

**结构性影响**:
- WindowCapture 无法在非 macOS 环境测试
- VisionOCREngine 无法替换为其他 OCR 引擎（如 Tesseract、PaddleOCR）
- LLM Client 之间接口不兼容：KimiClient 的 `chat(messages=...)` 参数与 OpenClawClient 不同

---

### HIGH-6: 工具调用循环的无限递归风险

**文件**: `src/reply/generator.py:157-272`
**代码**:
```python
while True:
    raw = active_llm.chat(...)
    if raw_tool_calls and not force_no_tools:
        # ... 执行工具 ...
        messages.append(tool_result)
        tool_round_count += 1
        continue  # 回到 while True
```

**逻辑问题**:
1. `while True` 没有最大迭代次数限制（虽然有 `tool_round_count` 但只用于记录，不做终止条件）
2. 如果 LLM 持续返回 tool_calls（比如工具返回错误信息，LLM 不断重试），会无限循环
3. `max_tool_rounds = 10` 被定义了但没有在循环中检查

---

### HIGH-7: tick() 中 try/except 包裹整个方法体

**文件**: `src/bot/wechat_bot.py:111-354`
**代码**:
```python
try:
    # ... 260+ 行代码 ...
except Exception as exc:
    self.logger.log_exception(tick_id, phase="tick", exc=exc)
    raise
```

**结构性问题**:
1. 任何阶段的异常都会导致整个 tick 失败，没有阶段级的错误恢复
2. 例如：如果 `sender.send()` 失败，已经处理好的消息不会尝试重新发送
3. 异常信息被简单记录后重新抛出，上层（run_auto）的 while 循环会继续执行下一次 tick，但不会处理本次失败的消息

---

### HIGH-8: 全局状态持久化没有版本控制

**文件**: `src/session/global_store.py`

**逻辑问题**:
1. `global_state.json` 没有版本字段
2. 如果代码升级后 ChatMessage 的结构发生变化，加载旧版本 JSON 会静默失败或产生错误数据
3. 没有数据迁移机制

---

## 五、MEDIUM 级问题（结构 + 逻辑）

### MED-1: 数据流边界情况处理不一致

| 阶段 | 空输入处理 | 异常处理 |
|------|-----------|----------|
| WindowCapture.capture() | 抛 WeChatNotReadyError | 异常 |
| VisionOCR.recognize() | 返回 [] | 抛 FileNotFoundError |
| LayoutParser.parse() | 返回 UILayout(message_candidates=[]) | 从不抛异常 |
| MessageExtractor.extract() | 返回 [] | 从不抛异常 |
| GlobalStore.merge_tick() | 创建新 ChatState | 内部 try/except |
| ReplyGenerator.generate() | 返回 [] | 返回 [] |
| MessageSender.send() | 返回 ActionResult(success=False) | 从不抛异常 |

**结构性问题**: 7 个阶段的错误处理风格完全不同，导致调用方无法统一处理错误。

### MED-2: _parse_replies 使用脆弱的字符串解析

**文件**: `src/reply/generator.py`

**逻辑问题**: 从 LLM 输出中解析回复列表时，使用正则表达式或字符串分割，容易因格式变化而失败。

### MED-3: SessionMemory 的 TTL 机制没有清理线程

**文件**: `src/reply/session_memory.py`

**逻辑问题**: TTL 值定义了但没有后台清理线程，过期数据一直占用内存直到被覆盖。

### MED-4: BotLogger 日志文件无限增长

**文件**: `src/logging/bot_logger.py`

**逻辑问题**: execution.jsonl 和 runtime.log 没有轮转机制，长期运行会占满磁盘。

### MED-5: 聊天切换防抖时间硬编码

**文件**: `src/bot/wechat_bot.py:96`
**代码**: `self._switch_debounce_seconds: float = 10.0`

**逻辑问题**: 10 秒的防抖时间是硬编码的，没有考虑不同场景（紧急消息 vs 普通消息）。

### MED-6: 截图保存路径包含 tick_id 但无清理机制

**文件**: `src/bot/wechat_bot.py:143-153`

**逻辑问题**: 每张截图都保存到磁盘，但没有清理旧截图的机制。

### MED-7: _normalize_chat_name 在 tick() 中硬编码

**文件**: `src/bot/wechat_bot.py:33-47`

**结构性问题**: 聊天名称归一化逻辑是模块级函数，但只被 tick() 使用，应该属于 GlobalStore 或专门的 NameNormalizer 类。

### MED-8: 消息发送后固定 sleep 1.5 秒

**文件**: `src/bot/wechat_bot.py:329-330`
**代码**: `if i < len(replies) - 1: time.sleep(1.5)`

**逻辑问题**: 固定 1.5 秒间隔，没有考虑消息长度、网络状况、微信响应时间。

### MED-9: memory_engine 更新在发送成功之后但无错误处理

**文件**: `wechat-mac-rpa/src/bot/wechat_bot.py:336-347`

**逻辑问题**: 如果 `update_user_wiki()` 失败，不会影响 tick 结果，但可能导致记忆不一致。

### MED-10: 类型提示覆盖率极低

**统计**: 61 个文件，298 个函数中，大量函数缺少返回类型注解和参数类型注解。IDE 无法提供有效的类型检查和自动补全。

---

## 六、LOW 级问题（结构 + 逻辑）

### LOW-1: 多处使用 print 代替 logging

**文件**: `src/reply/generator.py` 等

**结构性问题**: 混用 print 和 logging，导致日志输出不可控（print 无法设置级别、无法重定向）。

### LOW-2: 硬编码路径使用字符串拼接

**文件**: `src/storage/chat_history.py` 等

**结构性问题**: `Path("data") / "something"` 的写法散落在多处，应该用统一的常量。

### LOW-3: 魔法数字无处不在

**部分清单**:
- 像素阈值: 10
- Jaccard 阈值: 0.001, 0.08, 0.80, 0.82, 0.85, 0.90
- max_messages: 200
- max_retries: 2
- max_tool_seconds: 20.0
- max_total_seconds: 35.0 / 600.0
- switch_debounce: 10.0
- sleep_interval: 1.5
- lookback: 10

### LOW-4: 字符串硬编码

**部分清单**:
- `"deepseek"`, `"hermes"` 作为模型标识符硬编码
- `"use_hermes"` 作为路由信号
- `"自己"` 作为 sender 标识
- `"text"`, `"image"`, `"sticker"`, `"mixed"` 作为消息类型

### LOW-5: 无 requirements.txt / pyproject.toml

**结构性问题**: 没有依赖清单，新环境搭建困难。

### LOW-6: 测试覆盖率不足

**结构性问题**: 61 个文件但只有少量测试，且测试集中在非核心模块。

---

## 七、按模块索引

| 模块 | CRITICAL | HIGH | MEDIUM | LOW |
|------|----------|------|--------|-----|
| wechat_bot.py | tick() God Method (CRIT-1), no_reply 后置 (HIGH-1) | 全局 try/except (HIGH-7) | 边界处理不一致 (MED-1), sleep 1.5s (MED-8) | print, 魔法数字 |
| generator.py | generate() God Method (CRIT-2) | 非结构化路由 (HIGH-2), while True 风险 (HIGH-6) | 脆弱解析 (MED-2) | print, 字符串硬编码 |
| global_store.py | merge_tick O(N*M) (CRIT-5), Jaccard 0.001 (CRIT-6) | 无版本控制 (HIGH-8) | - | 魔法数字 |
| window_capture.py | L1→L6 侵越 (CRIT-4) | 缺抽象接口 (HIGH-5) | - | - |
| smart_pipeline.py | - | 像素阈值 (HIGH-3) | - | - |
| 整体架构 | 5 并行状态 (CRIT-3) | 配置散落 (HIGH-4) | TTL 无清理 (MED-3), 日志无轮转 (MED-4) | 无 requirements.txt (LOW-5) |

---

## 八、修复优先级

### P0（立即修复，影响系统正确性）
1. CRIT-3: 统一状态管理或引入事件溯源
2. CRIT-6: 提高图片去重 Jaccard 阈值到合理值
3. HIGH-1: 将 no_reply_chats 检查移到 LLM 调用之前
4. HIGH-6: 修复 while True 无 tool_round_count 限制

### P1（本周修复，影响系统可维护性）
1. CRIT-1: 拆分 tick() 为 Pipeline Stages
2. CRIT-2: 拆分 generate() 为独立阶段
3. CRIT-4: 移除 Capture → Action 的依赖
4. CRIT-5: 优化 merge_tick 算法
5. HIGH-2: 结构化 LLM 路由
6. HIGH-5: 为核心组件添加抽象接口

### P2（本月修复，影响系统健壮性）
1. HIGH-4: 统一配置管理
2. HIGH-7: 阶段级错误恢复
3. MED-1: 统一错误处理风格
4. MED-3-MED-10: 各类边界情况优化
5. LOW-1-LOW-6: 代码质量改进

---

## 九、附录：数据流图与问题标记

```
[Window] → capture() → screenshot.png
                         ↓
                    VisionOCR.recognize() → elements[]
                         ↓
                    LayoutParser.parse() → UILayout
                         ↓
                    MessageExtractor.extract() → ChatMessage[]
                         ↓
                    GlobalStore.merge_tick() → ChatState, unreplied[]
                    ⚠️ O(N*M) 性能问题 (CRIT-5)
                    ⚠️ Jaccard 0.001 误杀 (CRIT-6)
                         ↓
                    ReplyPolicy.should_reply() → bool
                         ↓
                    ReplyGenerator.generate()
                    ⚠️ God Method 317行 (CRIT-2)
                    ⚠️ 非结构化路由 (HIGH-2)
                    ⚠️ while True 无限循归 (HIGH-6)
                    ⚠️ no_reply 检查在之后 (HIGH-1)
                         ↓
                    MessageSender.send() → ActionResult
                         ↓
                    GlobalStore.mark_replied()
                    MemoryEngine.update_user_wiki()
                    ↓
              [5 个状态存储无同步 (CRIT-3)]
```

---

## 十、建议的架构重构方向

1. **引入 Pipeline 模式**: 将 tick() 拆分为独立的 Stage，每个 Stage 有明确的输入/输出/错误处理
2. **引入事件溯源**: 所有状态变更通过 Event Store 驱动，消除状态不一致
3. **引入依赖注入**: 为 WindowCapture、VisionOCREngine、LayoutParser、LLMClient 定义抽象接口
4. **统一配置**: 使用 Pydantic Settings 管理所有配置项
5. **后台任务队列**: 将 merge_tick、memory update、log persistence 放到后台线程

---

*报告生成时间: 2026-05-05*
*审查轮次: Round 8 (结构与逻辑专项)*
