# OCR 错误案例管理指南

## 概述

当 OCR 识别出现错误时，自动保存错误案例以便后续修复和回归测试。

---

## 🚀 自动添加错误案例

### 方式1: 代码中自动捕获

在模块化 OCR 代码中，当检测到识别错误时调用（注意：`auto_add_error_case.py` 当前依赖已删除的 V4 代码，需适配到 `src/` 新架构后使用）：

```python
from tests.auto_add_error_case import auto_add_error_case

# 当检测到发送者识别错误时
if actual_sender != expected_sender:
    auto_add_error_case(
        image_path="/tmp/wechat_ocr.png",
        expected_result={
            "chat_name": "周末兴趣群",
            "messages": [
                {
                    "sender": "wanglc",  # 正确的发送者
                    "sender_type": "other",
                    "text": "是不是忙着切号呢",
                    "is_at_me": False
                }
            ]
        },
        error_type="WRONG_SENDER",
        error_details=f"发送者被识别为 {actual_sender}，应为 wanglc"
    )
```

### 方式2: 命令行手动添加

```bash
# 添加已知错误的截图
python3 tests/auto_add_error_case.py /path/to/error_screenshot.png

# 查看所有错误案例
python3 tests/auto_add_error_case.py --list

# 标记错误为已修复
python3 tests/auto_add_error_case.py --fix error_20260412_001

# 提升为正式测试用例
python3 tests/auto_add_error_case.py --promote error_20260412_001
```

---

## 📁 错误案例存储

### 目录结构

```
tests/fixtures/errors/
├── error_20260412_001.png      # 错误截图
├── error_20260412_001.json     # 错误信息
├── error_20260412_002.png
└── error_20260412_002.json
```

### JSON 格式

```json
{
  "error_name": "error_20260412_001",
  "error_type": "WRONG_SENDER",
  "error_details": "发送者被识别为'对方'，应为'wanglc'",
  "created_at": "2026-04-12T10:30:00",
  "image_path": "tests/fixtures/errors/error_20260412_001.png",
  "status": "pending",
  "expected": {
    "chat_name": "周末兴趣群",
    "messages": [
      {
        "sender": "wanglc",
        "sender_type": "other",
        "text": "是不是忙着切号呢",
        "is_at_me": false
      }
    ]
  },
  "notes": "需要修正昵称识别逻辑"
}
```

---

## 🔄 错误处理流程

```
识别错误发生
     ↓
自动保存到 errors/
     ↓
修正 OCR 代码
     ↓
运行测试 (包含错误案例)
     ↓
测试通过 → 标记为 fixed
     ↓
验证稳定 → 提升为正式测试用例
```

---

## 🧪 测试错误案例

运行测试时会自动包含错误案例:

```bash
# 运行所有测试（包含错误案例）
./tests/run_tests.sh

# 输出示例:
# 📁 正式测试用例: 5个
# ...
# 🐛 错误案例回归测试 (2个)
# 🔍 测试: error_20260412_001
#    ✅ 错误已修复！
# 🔍 测试: error_20260412_002
#    ❌ 错误仍未修复 (1个问题)
```

---

## 📝 错误类型定义

| 错误类型 | 说明 | 示例 |
|---------|------|------|
| WRONG_SENDER | 发送者识别错误 | wanglc → 对方 |
| WRONG_COUNT | 消息数量不匹配 | 预期3条，识别2条 |
| LOW_SIMILARITY | 内容相似度低 | 相似度 < 90% |
| MISSED_MESSAGE | 遗漏消息 | 某条消息未识别 |
| WRONG_DIRECTION | 方向判断错误 | 自己消息识别为对方 |
| CHAT_LIST_ERROR | 聊天列表错误 | 昵称识别错误 |

---

## 💡 最佳实践

1. **及时记录**: 发现错误立即记录，避免遗漏
2. **详细描述**: 错误详情要写清楚，便于后续修复
3. **修正预期**: 添加错误案例后，立即修正 JSON 中的预期结果
4. **回归测试**: 修复代码后，运行测试验证
5. **及时归档**: 错误修复稳定后，提升为正式测试用例

---

## 📊 错误统计

```bash
# 查看错误统计
python3 tests/auto_add_error_case.py --list

# 输出示例:
# 📋 待修复错误案例 (3个):
#
# error_20260412_001:
#   类型: WRONG_SENDER
#   详情: 发送者 wanglc 被识别为 对方
#   时间: 2026-04-12T10:30:00
#
# error_20260412_002:
#   类型: LOW_SIMILARITY
#   ...
```
