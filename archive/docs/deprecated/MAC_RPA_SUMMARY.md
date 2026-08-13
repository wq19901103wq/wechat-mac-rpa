# Mac 微信 RPA 调研总结

> **当前项目状态更新**：本项目已全面采用 **Vision OCR 视觉识别方案（L1-L5 模块化架构）**（无需关闭 SIP、无需数据库解密）。下文中的"数据库解密方案"仅作为技术调研背景保留，实际实现请参考 `src/bot/wechat_bot.py` 和 `ARCHITECTURE.md`。

## 🔍 调研结果

### 现有开源项目

| 项目 | 技术方案 | 功能 | 维护状态 |
|------|----------|------|----------|
| **wemac** | AppleScript + Python | 群聊机器人 | ⭐ 活跃 |
| **chatlog-bot** | 数据库解密 + HTTP | 消息读取 | ⭐ 活跃 |
| **wechat-db-decrypt-macos** | 数据库解密 | 读取消息 | ⭐ 活跃 |
| **WeChat-MCP** | Accessibility API | 读/发消息 | ⭐ 活跃 |
| **WeChatAutoHelper** | Java + AppleScript | 消息发送 | 可用 |

### Mac RPA 技术路径

```
┌─────────────────────────────────────────────────────┐
│                    Mac 微信 RPA                      │
├─────────────────────────────────────────────────────┤
│  消息接收              │  消息发送                   │
├──────────────────────┼─────────────────────────────┤
│  1. 数据库解密        │  1. AppleScript            │
│     (SQLite)         │     (系统级 UI 操作)        │
│                      │                             │
│  2. Accessibility    │  2. Accessibility API      │
│     (AXUIElement)    │     (AXUIElement)          │
│                      │                             │
│  3. 屏幕截图+OCR     │  3. pyautogui 模拟按键     │
│                      │                             │
└──────────────────────┴─────────────────────────────┘
```

## 🎯 调研方案（供技术参考，非当前项目推荐）

> 以下 A/B/C 方案是技术调研阶段整理的参考架构，不代表当前项目实践。当前项目已全面采用 **Vision OCR 方案**，详见 `core/auto_bot_vision_ocr_v3.py` 和 `ARCHITECTURE.md`。

### 方案 A：纯 Accessibility API

**适用场景**：手动打开聊天窗口，自动回复

**优点**：
- ✅ 无需关闭 SIP
- ✅ 无需数据库解密
- ✅ 实现简单

**缺点**：
- ❌ 无法自动读取消息（需手动输入）
- ❌ 需要保持微信窗口打开

**参考实现**：
```bash
cd wechat-mac-rpa/examples
python3 simple_mac_bot.py
```

---

### 方案 B：数据库解密 + Accessibility 发送

> ⚠️ 此方案需要关闭 SIP，已被本项目废弃，仅作为技术调研背景保留。

**适用场景**：全自动消息接收和回复

**架构**：
```
微信数据库 → 解密 → HTTP API → 业务逻辑 → Accessibility → 微信发送
```

**需要**：
1. 关闭 SIP（系统完整性保护）
2. 数据库解密工具
3. 文件监听（FSEvents）检测新消息

**开源参考**：
- https://github.com/rockswang/chatlog-bot
- https://github.com/x5iu/wemac

---

### 方案 C：WeChaty + Mac Puppet（不推荐）

**状态**：PadLocal 跑路，免费 puppet 基本不可用

---

## 📊 对比表

| 功能 | Windows RPA | Mac RPA | Web 协议 |
|------|-------------|---------|----------|
| 消息接收 | ✅ 数据库轮询 | ⚠️ 需解密 | ❌ 已失效 |
| 消息发送 | ✅ UI 自动化 | ✅ Accessibility | ❌ 已失效 |
| 稳定性 | 中 | 高 | 低 |
| 封号风险 | 中 | 低 | 高 |
| 部署难度 | 中 | 低（无需 VM）| - |
| 多开支持 | 难 | 中等 | - |

## 🚀 当前项目状态

```
wechat-mac-rpa/
├── ✅ src/bot/wechat_bot.py         # Vision OCR 全自动机器人（当前唯一版本）
├── ✅ tests/test_real_scene_extraction.py  # 真实场景回归测试
├── ✅ examples/simple_mac_bot.py           # 简易版（可用）
├── ✅ ARCHITECTURE.md                      # 模块化架构文档
└── ❌ 数据库解密部分（已废弃，不再维护）
```

## 💡 建议

### 如果你：

**只是想快速体验**
→ 用当前的 `simple_mac_bot.py`，手动输入消息，AI 自动回复到微信

**需要全自动机器人**
→ 使用 `src/bot/wechat_bot.py`（当前唯一维护版本）
> ⚠️ 数据库解密方案（方案 B）已被本项目废弃，请勿关闭 SIP。

**有 Windows 电脑**
→ 回退到之前的 Windows 视觉 RPA 方案，功能更完整

## 🔗 参考链接

- **wemac**: https://github.com/x5iu/wemac
- **WeChat-MCP**: https://github.com/abhayforsure/ai-wechat-api
- **调研报告**: https://yage.ai/share/wechat-uia-platform-feasibility-survey-20260327.html

## ⚠️ 风险提示

1. **关闭 SIP** 会降低系统安全性，请谨慎操作
2. **Accessibility API** 需要授权：系统设置 → 隐私与安全 → 辅助功能
3. **微信风控**：避免高频发送，建议间隔 1-3 秒

---

**Mac 方案优势**：
- 无需 Windows 虚拟机/云服务器
- 数据库格式比 Windows 更稳定
- Accessibility API 官方支持

**劣势**：
- ~~消息读取需要关闭 SIP~~（已废弃，当前 Vision OCR 方案无需此操作）
- 社区资源比 Windows 少
