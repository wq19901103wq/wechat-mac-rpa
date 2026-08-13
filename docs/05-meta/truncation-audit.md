# 系统截断逻辑审计与改进建议

> 创建时间：2026-05-29
> 状态：已修复 3 处，其余待改进

---

## 一、已修复项

| # | 位置 | 问题 | 修复方式 | 状态 |
|---|------|------|---------|------|
| 1 | `global_store.py:404-409` | `max_messages=200` 存储层粗暴裁剪老消息 | 删除裁剪逻辑，历史消息完整保留 | ✅ 已修复 |
| 2 | `message_sender.py:404` | fallback keystroke 路径硬编码 `text[:60]`，长消息被截断发送 | 改为 `text`，完整发送 | ✅ 已修复 |
| 3 | `weflow_pipeline.py:35,56,65` | `history_limit=100` 死代码，赋了值但从未使用 | 删除相关代码 | ✅ 已修复 |

---

## 二、现存截断逻辑全景清单

### 2.1 Prompt 构建层（显示层截断，合理但可优化）

#### #4 历史消息窗口（`generator.py:912-939`）

```python
recent_20 = list(all_messages[-20:]) if len(all_messages) > 20 else list(all_messages)
recent_10min = [m for m in all_messages if _msg_ts(m) >= cutoff_ts]
candidate = [m for m in all_messages if id(m) in union_ids]
recent = list(candidate[-max_history:]) if len(candidate) > max_history else list(candidate)  # max_history=80
```

**上下文**：`_build_user_prompt` 构建 prompt 时从历史消息中选。
**当前行为**：取"最近20条"与"10分钟内"并集，最终上限 80 条。
**问题**：
- 80 条上限是硬编码，没有根据实际 token 数动态调整
- 截断时没有打点（没有日志说明"因为超过80条截断了"）
- 如果 80 条里有很多长消息，prompt 仍然可能超长

**改进建议**：
1. 引入**基于 token 数的动态预算**：给历史消息分配固定 token 预算（如 4000 tokens），按时间倒序累加，超预算时停止
2. 截断时打 INFO 级别日志：`[Prompt] 历史消息共 120 条，按 4000 token 预算截取 67 条`
3. 优先保证"未读消息"的 token 不被历史消息挤占

---

#### #5 记忆读取截断（`generator.py:868-888`）

```python
self_memory = self.memory_engine.get_user_memory("林岚", max_chars=4000)
memory_text = self.memory_engine.get_user_memory(clean_sender, max_chars=6000)
group_text = self.memory_engine.get_group_memory(chat_name, max_chars=6000)
```

**上下文**：从 `memory_engine` 读取 wiki 记忆时。
**当前行为**：按字符数硬截断 wiki 内容。
**问题**：
- 截断时没有日志打点
- `_compress_wiki` 虽然尽量在段落边界截断，但仍然是"一刀切"
- 不同角色的记忆预算不一致（自己 4000，别人 6000），没有文档说明为什么

**改进建议**：
1. 截断时打日志：`[Memory] 林岚 wiki 原始 15000 字符，截断到 4000 字符`
2. 引入**重要性分层**：wiki 中人工标注的 facts 优先保留，自动推断的次要信息可截断
3. 考虑用 LLM 对超长 wiki 做摘要，而不是直接截断

---

### 2.2 LLM 调用层（合理，但可补充监控）

#### #6 单条消息 content 截断（`generator.py:514-515`）

```python
if "content" in cm and isinstance(cm["content"], str) and len(cm["content"]) > 10000:
    cm["content"] = cm["content"][:10000] + "\n\n... [truncated, see markdown for full content]"
```

**上下文**：发送给 LLM API 前。
**当前行为**：单条 content > 10000 字符时截断。
**问题**：
- 有截断标记，但没有日志打点
- 10000 字符阈值是硬编码，没有根据模型上下文窗口调整

**改进建议**：
1. 截断时打 WARNING 日志：`[LLM] message content 超长截断: 23000 -> 10000 字符`
2. 阈值应与模型上下文窗口联动（如 claude-3.5 可用 200K，不需要卡 10000）

---

#### #7-9 max_tokens 限制

```python
# generator.py
max_tokens=2000   # 主回复
max_tokens=256    # skill router
# openclaw_client.py
max_tokens=1024   # 默认
# smart_pipeline.py
max_tokens=4096   # qwen API
```

**上下文**：各模块调用 LLM API 时的输出长度限制。
**评估**：合理。这些是业务层面的输出控制，但应统一配置到 `config.py` 或环境变量，不要散落在各处硬编码。

---

### 2.3 工具层（合理，但可优化体验）

#### #10 网页抓取截断（`tools/builtin_tools.py:199-201`）

```python
max_len = 12000
result = text[:max_len]
if len(text) > max_len:
    result += "\n（...内容已截断，原始长度: {} 字符）".format(len(text))
```

**上下文**：`fetch_webpage` 工具返回网页正文。
**当前行为**：正文截断到 12000 字符。
**问题**：
- 有截断标记，但没有日志打点
- 12000 字符对现代网页来说可能仍然太长（塞进 prompt 占大量 token）
- 直接截断可能导致关键信息丢失（比如文章后半部分才有结论）

**改进建议**：
1. 截断时打日志：`[Tool] fetch_webpage 截断: 原始 45000 字符 -> 12000 字符`
2. 引入**智能摘要**：用轻量模型对超长网页做摘要，返回摘要而非原文截断
3. 或支持分页：返回前 N 字 + "还有 X 字，需要深入阅读请说'继续'"

---

#### #11 搜索结果 snippet 截断（`tools/builtin_tools.py:89`）

```python
snippet = snippet[:200] + "..."
```

**上下文**：`web_search` 工具处理搜索结果。
**当前行为**：单条 snippet 截断到 200 字符。
**评估**：合理。搜索 snippet 本身就是摘要，200 字符足够判断相关性。

---

### 2.4 记忆引擎层（合理但缺乏可观测性）

#### #12 wiki 压缩截断（`memory/engine.py:379-393`）

```python
truncated = wiki[:max_chars]
last_break = max(truncated.rfind("\n## "), truncated.rfind("\n- "), truncated.rfind("\n\n"))
if last_break > max_chars * 0.5:
    truncated = truncated[:last_break]
return truncated.strip() + "\n（…记忆已截断）"
```

**上下文**：读取 wiki 记忆时按 `max_chars` 截断。
**当前行为**：尽量在段落边界截断，加标记。
**问题**：
- 截断时没有日志打点
- 标记是返回给 LLM 看的，但系统日志里没有记录

**改进建议**：
1. 每次截断都记录日志：`[Memory] wiki 截断: 15000 -> 4000 字符，在段落边界截断`
2. 考虑在 wiki 文件元数据中标注优先级，让核心 facts 不被截断

---

#### #13 BM25 搜索结果截断（`memory/engine.py:780-974`）

```python
# 先加本人的（完整保留）
for snippet in primary_snippets:
    if len(truncated) + len(snippet) + 1 > max_chars:
        if not truncated:
            truncated = snippet[:max_chars] + "\n（…内容截断）"
        break
    truncated = truncated + "\n" + snippet if truncated else snippet

# 再加其他人的（超长的截断）
for snippet in other_snippets:
    if len(truncated) + len(snippet) + 1 > max_chars:
        truncated += "\n（…更多结果省略）"
        break
    ...
```

**上下文**：`search_keyword` 返回 BM25 搜索结果。
**当前行为**：优先保留与查询人相关的结果，其他人结果超长时省略。
**问题**：
- 有省略标记，但没有日志打点
- 如果第一条本人 snippet 就超过预算，直接硬截断到 `max_chars`

**改进建议**：
1. 截断时记录日志，说明保留了几条本人、几条其他人
2. 对单条超长 snippet 也做智能摘要，而不是直接硬截断

---

#### #14 记忆更新输入超长截断（`memory/engine.py:456-461`）

```python
conv_lines = conv.strip().split("\n")
truncated = "\n".join(conv_lines[len(conv_lines) // 2:])
current_prompt = header + marker + truncated
_logger.warning(f"输入超长，截断 conversation 后重试 ({attempt}/2)")
```

**上下文**：wiki 更新时 LLM 输入超长，截断 conversation 后重试。
**当前行为**：直接砍掉前半段对话历史。
**问题**：
- 有 WARNING 日志（这是少数打了点的）
- 但截断策略粗暴：直接砍一半，可能丢掉关键上下文

**改进建议**：
1. 不要砍一半，而是保留**开头（背景）+ 结尾（最新对话）**，砍中间
2. 或用轻量模型先对 conversation 做摘要，用摘要替换原文

---

#### #15 debug prompt 日志截断（`memory/engine.py:353-358`）

```python
max_len = 80000
if len(prompt) > max_len:
    truncated = f"{prompt[:40000]}\n\n... [中间部分截断，共 {len(prompt)} 字符] ...\n\n{prompt[-40000:]}"
```

**上下文**：保存 prompt debug 文件到磁盘。
**当前行为**：超过 80000 字符时保留头尾各 40000。
**评估**：合理。这只是 debug 日志文件，不影响运行。

---

#### #16 搜索结果数量限制（`memory/engine.py:895-923`）

```python
non_primary[:10]
scored[:10]
```

**上下文**：BM25 排序后取最相关结果。
**评估**：合理。取前 10 个相关结果是常见做法。

---

#### #17 wiki 更新队列批处理（`memory/engine.py:984-990`）

```python
batch = self._update_queue[:3]
```

**上下文**：异步 wiki 更新任务批处理。
**评估**：合理。每批 3 条是并发控制，防止一次处理太多。

---

### 2.5 通用工具层

#### #18 `_truncate_text` / `_compress_text`（`utils/text_utils.py:5-19`）

```python
def _truncate_text(text: str, max_len: int, suffix: str = "\n\n... [truncated]") -> str:
    if not text or len(text) <= max_len:
        return text
    return text[:max_len] + suffix

def _compress_text(text: str, max_chars: int) -> str:
    head_len = int(max_chars * 0.4)
    tail_len = int(max_chars * 0.6)
    return text[:head_len] + "\n...（中间省略）...\n" + text[-tail_len:]
```

**上下文**：通用文本处理工具函数。
**评估**：合理。但调用方应在截断时自行打日志。

---

### 2.6 Debug / Trace 日志层（全部不影响核心逻辑）

#### #19-20 `generator.py` 和 `message_sender.py` 多处预览截断

```python
raw_content[:500]
raw_content[:2000]
tool_args[:100]
result[:1000]
hermes_text[:100]
user_text[:30]
err[:200]
text[:80]
```

**上下文**：trace/debug 日志记录。
**评估**：合理。这些只是日志预览，不影响业务逻辑。

---

### 2.7 数据库层

#### #21 `case_db.py` 多处字段截断

```python
text[:5000]
sp[:30000]
up[:30000]
tc[:10000]
comment[:1000]
str(args)[:2000]
str(content)[:3000]
reason[:120]
_raw[:4000]
```

**上下文**：SQLite 数据库存储。
**评估**：合理。数据库字段长度保护是必要的。但建议：
1. 在 schema 中明确定义 TEXT 字段，SQLite 对 TEXT 没有硬长度限制，这些截断是防御性的
2. 如果某条数据被截断了，应记录 WARNING 日志

---

### 2.8 数据清理层

#### #22 按时间 cutoff 清理旧数据（`case_db.py:367-476`）

```python
cutoff = (datetime.now() - timedelta(days=days)).isoformat()
```

**上下文**：数据库维护、统计查询。
**评估**：合理。按时间清理/统计是正常运维需求。

---

#### #23 临时截图清理（`window_capture.py:213-216`）

```python
cutoff = time.time() - 3600
if os.path.getmtime(old) < cutoff:
    os.remove(old)
```

**上下文**：截图保存后清理 1 小时前的临时文件。
**评估**：合理。磁盘空间管理。

---

### 2.9 WeFlow 层

#### #24 WeFlow 增量同步分页（`weflow_pipeline.py:329-330`）

```python
limit = self.tick_limit  # 20
max_rounds = 5
```

**上下文**：WeFlow 增量同步时分页拉取消息。
**评估**：合理。API 分页控制，最多拉 100 条新消息。

---

### 2.10 窗口 / 回复控制层

#### #25 窗口尺寸检查（`smart_pipeline.py:235-236`）

```python
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
```

**上下文**：截图前检查窗口尺寸。
**评估**：合理。窗口太小会导致 OCR 失败。

---

#### #26 回复条数限制（`generator.py:562`）

```python
return [str(r).strip() for r in replies if str(r).strip() not in ("收到", "好的", "嗯", "OK", "1")][:3]
```

**上下文**：解析 LLM 输出后最多取 3 条回复。
**评估**：合理。控制一次发送的条数。

---

## 三、系统性问题总结

### 3.1 "没有打点"——截断缺乏可观测性

大部分截断逻辑**没有日志记录**，导致：
- 出了问题无法追溯（比如用户问"为什么这条历史消息没进 prompt"，查不到日志）
- 无法统计截断频率（哪些模块经常触发截断？是不是阈值设太低了？）

**应统一要求**：所有截断操作必须打日志，格式建议：
```
[Truncation] <模块> <描述>: 原始 <原始大小> -> 截断后 <截断后大小>, 原因 <原因>
```

### 3.2 "直接截断太粗暴"——缺乏智能替代方案

当前几乎所有截断都是**硬截断**（`text[:max_len]`），更好的替代方案：

| 场景 | 当前做法 | 更好的做法 |
|------|---------|-----------|
| 超长网页正文 | 硬截断到 12000 字 | **关键字上下文截取**：用用户问题里的关键词在网页中定位，提取关键词周围的上下文 |
| 超长 wiki 记忆 | 硬截断到 4000/6000 字 | **关键字上下文截取**：用查询关键词在 wiki 中定位相关章节，保留上下文 |
| 超长 conversation | 直接砍一半 | 保留开头+结尾，中间用摘要替代 |
| 历史消息窗口 | 硬取最近 80 条 | 按 token 预算动态调整，优先保证未读消息 |
| 单条 message content | 硬截断到 10000 字 | 按模型上下文窗口动态调整 |

#### 已实现的工具：`extract_context_around_keywords`

已在 `src/utils/text_utils.py` 中实现，核心逻辑：

```python
from src.utils.text_utils import extract_context_around_keywords

# 示例：用户问"许安的年收入是多少"
keywords = ["许安", "年收入", "工资", "薪水"]
result = extract_context_around_keywords(
    text=long_wiki_text,      # 原始长文本
    keywords=keywords,        # 从用户问题提取的关键词
    max_chars=6000,           # 总字符预算
    context_radius=500,       # 每个匹配点前后保留 500 字符
)
```

**策略**：
1. 在文本中搜索所有关键词出现的位置（忽略大小写、去除标点）
2. 每个匹配点提取前后 `context_radius` 字符的上下文窗口
3. 合并重叠/相邻的窗口
4. 按匹配密度（窗口内关键词出现次数）排序
5. 在 `max_chars` 预算内保留最相关的窗口
6. 窗口之间用 `...（省略 X 字符）...` 连接，标明跳过了多少内容
7. **无匹配时 fallback**：如果关键词在文本中完全没出现，fallback 到 `_compress_text`（保留头尾），不会盲目从头截断

**适用场景**：
- `fetch_webpage`：用用户问题提取关键词，在网页正文中定位相关段落
- `_compress_wiki`：用查询关键词在 wiki 中定位相关章节
- `search_keyword` 结果：围绕 BM25 匹配的关键词保留完整上下文
- `_truncate_messages`：如果消息内容是非结构化长文本，可用关键词定位关键部分

**不适用场景**：
- 历史消息窗口（时间序列，按时间取更合理）
- 日志预览截断（只是日志）
- 数据库字段保护（防御性截断）

### 3.3 阈值散落各处——缺乏统一配置

`max_tokens=2000/256/1024/4096`、`max_chars=4000/6000/12000/2000` 等阈值散落在各个文件的硬编码中，没有统一配置中心。建议：
- 创建 `src/config/limits.py` 统一存放所有截断阈值
- 支持环境变量覆盖

---

## 四、待办清单

- [x] 1. 删除 `global_store.py` 的 `max_messages=200` 裁剪逻辑
- [x] 2. 修复 `message_sender.py` 的 `text[:60]` bug
- [x] 3. 删除 `weflow_pipeline.py` 的 `history_limit` 死代码
- [ ] 4. 为所有截断逻辑补充日志打点
- [ ] 5. 历史消息窗口从"条数限制"改为"token 预算限制"
- [ ] 6. 超长网页正文引入摘要替代硬截断
- [ ] 7. 超长 conversation 截断改为"保留头尾+摘要中间"
- [ ] 8. 创建 `src/config/limits.py` 统一阈值配置
- [ ] 9. wiki 记忆截断引入重要性分层
