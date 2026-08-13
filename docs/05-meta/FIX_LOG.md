
## 2026-06-05 - 修复 CODE_REVIEW_20260605 5 项必须修复问题

> 关联文档: [CODE_REVIEW_20260605](./CODE_REVIEW_20260605.md)

### 1. `_extract_json` 字符串内括号误判 (generator.py)

- **问题**: 括号深度计数不区分字符串内外，LLM 回复含 `{}` 字符时 JSON 被截断解析失败
- **修复**: 改用 `json.JSONDecoder.raw_decode()` 精确解析，正确处理字符串内花括号

### 2. LCS O(m×n) 无上限保护 (global_store.py)

- **问题**: 大群聊（200+ 条历史）下 LCS DP 表过大，每个 tick 耗时可能到秒级
- **修复**: 新增 `_MAX_HISTORY_FOR_LCS = 80`，超过时截取最近部分

### 3. tick_log SQLite 连接泄漏 (wechat_bot.py)

- **问题**: `conn.close()` 在 try 块内，异常时连接不释放
- **修复**: 改为 `conn = None` + `finally: conn.close()`，确保连接始终释放

### 4. 剪贴板保存/恢复竞态 (message_sender.py)

- **问题**: 并发 send 调用时第二次保存的 original_clipboard 是第一次写入的内容，用户原始剪贴板丢失
- **修复**: 新增 `self._send_lock = threading.Lock()`，send() 持锁后委托 `_send_impl()`

### 5. Memory Worker 任务丢失 (engine.py)

- **问题**: `_do_update(task)` 无 try/except，单条任务异常导致 batch 剩余任务全部跳过且永久丢失
- **修复**: 加 try/except，单条失败不影响后续，error 日志含任务信息

### 验证结果
- 5 个文件 Python 语法检查全部通过 ✓

## 2026-04-13 - 修复 error_20260413_001 (聊天名称识别错误)

### 问题
- 聊天名称 "示例用户甲" 被错误识别为 "®v QS."
- 原因: 标题栏识别范围 `TITLE_Y_MAX = 60` 太宽泛，包含窗口控制按钮区域和右侧图标

### 修复方案
1. **收紧标题栏 Y 范围**: `TITLE_Y_MAX` 60 → 50
2. **添加 X 范围过滤**: 新增 `TITLE_X_MAX_RATIO = 0.70`，排除右侧图标区域（搜索、电话按钮等）
3. **特殊字符过滤**: 排除包含 ®、©、™、QS 等明显非昵称字符的元素

### 代码变更
```python
# src/parser/wechat_parser.py
TITLE_Y_MAX = 50           # 收紧
TITLE_X_MAX_RATIO = 0.70   # 新增

# _parse_chat_area 方法中
title_x_max = self.image_width * self.TITLE_X_MAX_RATIO
title_elems = [e for e in elements if e.y < self.TITLE_Y_MAX and e.x < title_x_max]

# 过滤特殊字符
filtered = [e for e in title_elems if not any(c in e.text for c in ['®', '©', '™', 'QS'])]
```

### 验证结果
- error_20260413_001.png: 识别正确 "示例用户甲" ✓
- 全量测试: 8/8 通过 ✓（private_w1han.png 已移除，避免真实私聊隐私泄露）

### 状态
- 错误样本状态: fixed
- 修复时间: 2026-04-13 06:40
