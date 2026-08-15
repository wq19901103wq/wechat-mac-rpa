# AI Quickstart: 微信 Mac RPA

> 如果你是一名 AI Agent / 开发者，第一次接触这个项目，请从这里开始。

---

## 30 秒看懂项目

这是一个 **macOS 微信自动化回复机器人**。

- **输入**: 定时截图微信窗口 → OCR 识别文字
- **处理**: 解析聊天布局 → 提取最新消息 → 判断是否需要回复
- **输出**: 调用 LLM 生成回复 → 用 AppleScript 粘贴发送

**当前稳定版本**: 模块化架构版（`src/bot/wechat_bot.py`）
**架构文档**: [`../02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md)

---

## 5 分钟上手

### 1. 读三个文件

按顺序读：
1. [`../02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md) — 系统架构总览
2. [`../04-troubleshooting/LESSONS_LEARNED.md`](../04-troubleshooting/LESSONS_LEARNED.md) — 踩坑记录（避免重复踩坑）
3. 你负责的模块接口（见下方"模块速查"）

### 2. 修改代码前必读

- **所有布局边界值**都在 `LayoutProfile` 里，不要硬编码
- **去重逻辑**在 `ChatSession` 里，不要在 Bot 里写 `last_reply_content in text`
- **发送消息**只能用 `pbcopy + Command+V`，**严禁**用 `keystroke "a"`

### 3. 排查问题看日志

```bash
# 查看人类可读的运行日志
tail -f ~/wechat-mac-rpa/data/logs/runtime_$(date +%Y%m%d).log

# 查看结构化决策流水（机器解析）
tail -f ~/wechat-mac-rpa/data/logs/execution.jsonl | jq .

# 查看某个聊天的历史
jq . ~/wechat-mac-rpa/data/history/测试群.jsonl
```

### 4. 测试

```bash
cd /path/to/wechat-mac-rpa
python -m pytest src/tests -v
```

---

## 模块速查表

| 如果你要改... | 修改文件 |
|--------------|---------|
| 截图逻辑 | `src/capture/window_capture.py` ✅ |
| OCR 识别 | `src/ocr/vision_ocr.py` ✅ |
| 布局解析 | `src/layout/layout_parser.py` ✅ |
| 消息提取 | `src/message/extractor.py` ✅ |
| 感知管道（智能预判/API兜底切换） | `src/perception/smart_pipeline.py` ✅ |
| 纯本地 OCR 管道 | `src/perception/vision_pipeline.py` ✅ |
| 布局配置（边界值、阈值） | `src/layout/profile.py` ✅ |
| 会话/去重 | `src/session/chat_session.py` ✅ |
| 回复策略 | `src/reply/policy.py` ✅ |
| 回复生成 | `src/reply/generator.py` ✅ |
| 发送动作 | `src/action/message_sender.py` ✅ |
| 主循环 | `src/bot/wechat_bot.py` ✅ |
| 运行日志 | `src/logging/bot_logger.py` ✅ |
| 聊天记录 | `src/storage/chat_history.py` ✅ |
| 测试 | `tests/` 目录 ✅ |

---

## 数据流速记

```
截图 → OCR → Parser解析 → 存储去重 → 回复生成/决策 → 
Action发送 → 记录状态
```

---

## 关键禁忌（会直接导致 bug）

❌ **不要用 `keystroke "a" using command down`**  
原因：中文输入法下会产生拼音碎片（如 `laayaua5aapangaaaaa~`）  
✅ 正确做法：`pbcopy` + `keystroke "v" using command down`

❌ **不要把边界值写死在不同模块里**  
原因：`input_y_min` 一旦改漏就会让输入框内容混入消息  
✅ 正确做法：全部引用 `LayoutProfile`

❌ **不要用简单字符串包含判断去重**  
原因：无法区分"同一位置的消息残留"和"不同的人发同样的话"  
✅ 正确做法：用 `MessageIdentity`（chat_name + sender + hash + y坐标）去重，并且回声检测以**时间窗口**（如 10 秒内）为首要条件，y 坐标仅作辅助

---

## FAQ

**Q: 为什么旧版本是 monolithic 的？**  
A: 历史原因，为了快速迭代。核心逻辑验证完成后，已按 [`../02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md) 完成模块化拆分。旧版本 `core/auto_bot_vision_ocr_v2/v3/v4.py` 已删除，当前唯一入口是 `src/bot/wechat_bot.py`。

**Q: 改了边界值后怎么验证？**  
A: `tests/fixtures/errors/` 下有 23 个回归测试用例，要求：
- 聊天名识别准确率 ≥ 95%
- 发送者类型识别准确率 ≥ 90%

**Q: 出了问题怎么排查？**
A: 三步定位法：
1. `execution.jsonl` 搜 `"event":"decision"` 看是否决策跳过
2. `execution.jsonl` 搜 `"event":"layout"` 看聊天名/气泡数是否正常
3. `runtime_YYYYMMDD.log` 搜 `ERROR` 看异常堆栈

**Q: 新功能加在哪里？**  
A: 先看它属于哪一层：
- 改识别逻辑 → `layout/` 或 `message/`
- 改回复策略 → `reply/`
- 改发送方式 → `action/`
- 改主循环流程 → `bot/`
