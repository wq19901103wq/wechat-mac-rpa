# Performance Optimization Spec

> 最后更新: 2026-05-15
> 状态: **设计文档**，部分方案已落地，部分待实施

---

## 1. 现状分析（数据驱动）

基于 2026-05-15 生产日志分析：

### 1.1 无消息 tick 基准耗时
```
_run_local_only(): 2300-3000ms (平均 ~2600ms)
  └─ capture():     ~500-800ms  (screencapture)
  └─ ocr.recognize(): ~1000-1500ms (macOS Vision)
  └─ layout.parse():  ~500-800ms (气泡检测+列表解析)
```
**问题**: 即使截图完全相同（MD5 hash 一致），仍然执行完整 OCR + Layout。

### 1.2 有消息 tick 额外耗时
```
_run_with_api() 额外增加:
  └─ API 请求 (qwen3.6-flash): 2800-4300ms

generator.generate():
  └─ _load_skill_one_liners(): 每次读取所有 SKILL.md（磁盘 I/O）
  └─ _build_user_prompt():
      └─ get_user_memory("示例用户甲"): 文件 I/O + 压缩
      └─ get_user_memory(sender): 文件 I/O + 压缩
      └─ get_group_memory():      文件 I/O + 压缩
      └─ search_related_mentions(): 扫描所有别名 + grep 所有 wiki
  └─ _route_skills(): 额外轻量 LLM 调用 (~1000-3000ms)
  └─ active_llm.chat(): 主 LLM 调用 deepseek (~3000-10000ms)

sender.send():
  └─ AppleScript delays + Python sleeps: ~2000-4000ms
```

### 1.3 一个完整"消息→回复"链路的耗时估算
```
感知:   ~5500ms (capture + OCR + API)
生成:   ~7500ms (skill I/O + memory I/O + route LLM + main LLM)
发送:   ~3000ms (AppleScript)
轮询:   ~2500ms (平均等待，5s 间隔均匀分布)
总计:   ~18500ms (约 18-20 秒)
```

### 1.4 每 tick 同步 I/O 开销
- `debug_logger.save()`: JSON 序列化 + 文件写入（即使无消息也要写 ~6KB JSON）
- `global_store.save()`: 索引文件写入（即使 `_dirty` 为空）
- `global_store.save_screenshot()`: `shutil.copy2` 复制截图到 `data/screenshots/`

---

## 2. 瓶颈分解与优先级

| 优先级 | 瓶颈 | 当前耗时 | 目标耗时 | 投入产出比 | 落地状态 |
|--------|------|---------|---------|-----------|---------|
| 🔴 P0 | hash 相同仍做 OCR + Layout | ~2600ms | ~100ms | 最高 | ❌ **未落地** |
| 🔴 P0 | `_load_skill_one_liners()` 无缓存 | ~50-200ms | ~1ms | 高 | ❌ **未落地** |
| 🔴 P0 | `_route_skills()` 额外 LLM 调用 | ~1000-3000ms | ~10ms | 高 | ❌ **未落地** |
| 🟡 P1 | memory_engine 文件 I/O 无缓存 | ~100-300ms | ~10ms | 中高 | ❌ **未落地** |
| 🟡 P1 | AppleScript delay 过于保守 | ~2000-4000ms | ~1000-1500ms | 中 | ❌ **未落地** |
| 🟡 P1 | debug_logger 同步写盘 | ~50-200ms | ~0ms (异步) | 中 | ❌ **未落地** |
| 🟢 P2 | global_store 保存降频 | ~50-100ms | ~0ms (降频) | 低 | ❌ **未落地** |
| 🟢 P2 | LCS 去重算法 O(m×n) | ~10-50ms | ~5ms | 低 | ✅ 窗口已限制为 50，暂不优化 |

---

## 3. 优化方案（设计阶段）

> ⚠️ 以下方案均为设计文档，**尚未在代码中实现**。
> 如需实施，请参照具体代码位置修改。

### 3.1 P0: hash 相同时直接返回缓存结果

**现状**：
```python
if curr_hash == self._last_hash:
    skip_api = True
    return self._run_local_only(image_path, ...)  # 仍做 OCR！
```

**优化**：
```python
if curr_hash == self._last_hash and self._last_result is not None:
    self.skip_count += 1
    self._consecutive_low_diff += 1
    return self._last_result  # 直接返回缓存，零感知耗时
```

**实施位置**: `src/perception/smart_pipeline.py`

---

### 3.2 P0: Skill 元数据缓存

**现状**：`_system_prompt()` 每次调用 `_load_skill_one_liners()`，扫描 `skills/` 目录。

**优化**：在 `ReplyGenerator.__init__` 中增加 `_skill_manifest_cache`，用目录 mtime 判断刷新。

**实施位置**: `src/reply/generator.py`

---

### 3.3 P0: `_route_skills()` 本地关键词匹配替代 LLM

**优化方案 A**：每个 `SKILL.md` 头部增加 `## 触发关键词` 段落，本地匹配优先。

**实施位置**: `src/reply/generator.py` + `skills/*.md`

---

### 3.4 P1: MemoryEngine 读缓存

**优化**：`MemoryEngine._load_wiki()` 增加 `(content, mtime)` 缓存。

**实施位置**: `src/memory/engine.py`

---

### 3.5 P1-P2: 其他优化

见原始设计文档归档（`docs/05-meta/reply_latency_optimization.md` 包含更详细的 AppleScript delay 调优方案）。

---

## 4. Profiling 点索引（已落地）

以下 profiling 点已在代码中植入，日志格式统一为 `[Perf][<模块>] <阶段>=<耗时>ms`。

### 4.1 感知层 `src/perception/smart_pipeline.py`

| 日志关键词 | 计时范围 | 说明 |
|-----------|---------|------|
| `[SmartPipeline] 截图成功: capture=...ms` | `capture.capture()` 总耗时 | 含窗口查找 + screencapture + 验证 |
| `[SmartPipeline] 本地处理完成: ocr=...ms layout=...ms` | `_run_local_only()` 内部 | OCR 和 Layout 分别计时 |
| `[SmartPipeline] API请求成功: latency=...ms` | `_run_api_pipeline()` 网络耗时 | 纯 API 网络往返 |

### 4.2 生成层 `src/reply/generator.py`

| 日志关键词 | 计时范围 | 说明 |
|-----------|---------|------|
| `[Perf][Generate] total=...ms sp=...ms tc=...ms up=...ms route=...ms llm=...ms parse=...ms` | `generate()` 内部子阶段 | system_prompt / tools_context / user_prompt / route_skills / LLM 调用 / parse_replies |
| `[Perf][Memory] self=...ms other=...ms group=...ms mentions=...ms` | `_build_user_prompt()` 内部 memory 调用 | Bot wiki / 对方 wiki / 群 wiki / 相关人搜索 |

### 4.3 发送层 `src/action/message_sender.py`

| 日志关键词 | 计时范围 | 说明 |
|-----------|---------|------|
| `[Perf][Sender] total=...ms read_clipboard=...ms activate=...ms pbcopy=...ms focus=...ms paste=...ms verify=...ms return=...ms` | `send()` 内部子阶段 | 每个 subprocess / AppleScript / sleep 阶段 |

### 4.4 提取分析命令

```bash
# 提取最近一轮的各阶段耗时
cd /Users/yourname/wechat-mac-rpa
grep -E "\[Perf\]|\[SmartPipeline\] (截图成功|本地处理完成|API请求成功|完成)" \
  data/logs/runtime_$(date +%Y%m%d).log | tail -50
```
