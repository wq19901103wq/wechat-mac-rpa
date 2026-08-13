# 微信 RPA 解决方案汇总

## 当前状态
- ✅ 主方案: **Vision OCR 视觉识别（新架构 L1-L5）**
- ✅ 微信运行且已登录
- ✅ OCR 识别正常
- ✅ 自动登录恢复正常
- ✅ LLM 连接正常
- ✅ 消息发送正常

---

## 🚀 当前方案: 模块化 Vision OCR（唯一维护版本）

代码位于 `src/`，按 L1-L5 分层：
- **L1 Domain**: `models/base.py`
- **L2 Capture/OCR**: `capture/window_capture.py`, `ocr/vision_ocr.py`
- **L3 Layout/Extract**: `layout/layout_parser.py`, `layout/profile.py`, `message/extractor.py`
- **L3.5 Pipeline**: `perception/smart_pipeline.py`（主力：本地预判 + API 兜底）, `perception/vision_pipeline.py`（备用回退）
- **L4 Session/Reply/Action**: `session/chat_session.py`, `reply/generator.py`, `action/message_sender.py`, `action/login_recovery.py`
- **L5 Bot**: `bot/wechat_bot.py`

### 运行方式

```bash
cd ~/wechat-mac-rpa
python3 run_bot.py
```

### 核心特性
- 绿色气泡检测识别自己消息
- 昵称区域检测识别对方消息
- 多行消息自动合并
- 自动登录恢复（检测并点击登录按钮）
- 真实场景回归测试覆盖

---

## 历史版本（已删除）

旧 monolithic 版本 `auto_bot_vision_ocr_v2.py/v3.py/v4.py` 已删除，由新架构完全替代。

---

## 🚀 方案 D: Accessibility API 版

**优点**: 界面控制精确  
**缺点**: 需要辅助功能权限

### 使用方法

```bash
cd ~/wechat-mac-rpa
~~已删除~~ 原: ./run_auto_accessibility.sh
```

### 授予权限
```
系统设置 → 隐私与安全 → 辅助功能 → 添加终端程序
```

---

## 🚀 方案 E: 简化版机器人

**优点**: 配置最简单、立即可用  
**缺点**: 需要手动输入消息

### 使用方法

```bash
cd ~/wechat-mac-rpa
~~已删除~~ 原: python3 run_simple.py
```

然后输入格式：`聊天名称|消息内容`

示例：
```
文件传输助手|你好
文件传输助手|讲个Python装饰器的用法
家庭群|大家晚上好
```

---

## 📊 方案对比

| 功能 | 新架构模块化 | Accessibility | 简化版 |
|------|-------------|---------------|--------|
| 全自动监听 | ✅ | ✅ | ❌ |
| 多对话管理 | ✅ | ✅ | ❌ |
| 发言人识别 | ✅ 精确 | ✅ | - |
| 无需 SIP | ✅ | ✅ | ✅ |
| 无需 db_key | ✅ | ✅ | ✅ |
| 需要窗口可见 | ✅ | ✅ | - |
| 需要辅助功能 | ❌ | ✅ | ❌ |
| 自动登录恢复 | ✅ | - | - |
| TDD 回归测试 | ✅ | - | - |

---

## 🔧 实用工具

### 查看 OCR 识别历史
```bash
python3 scripts/view_ocr_history.py
```

### 导出识别日志
```bash
python3 scripts/view_ocr_history.py export
```

### 查看原始日志
```bash
python3 scripts/view_ocr_history.py raw 100
```

### 布局分析器（调试）
```bash
~~已删除~~ 原: python3 core/wechat_layout_analyzer.py
```

---

## 📁 相关文件

### 当前架构（唯一维护版本）
| 文件 | 说明 |
|------|------|
| `src/bot/wechat_bot.py` | ⭐ L5 主循环编排 |
| `src/perception/smart_pipeline.py` | L3.5 智能感知管道（主力：本地预判 + API 兜底） |
| `src/perception/vision_pipeline.py` | L3.5 纯本地 OCR 管道（备用回退） |
| `src/capture/window_capture.py` | 窗口捕获（含登录恢复） |
| `src/ocr/vision_ocr.py` | Vision OCR 引擎 |
| `src/layout/layout_parser.py` | UI 布局分组 |
| `src/message/extractor.py` | 消息提取 |
| `src/session/chat_session.py` | 会话与去重 |
| `src/action/login_recovery.py` | 登录恢复处理 |
| `tests/test_real_scene_extraction.py` | 真实场景回归测试 |
| `docs/02-architecture/ARCHITECTURE.md` | 架构设计文档 |

### 历史版本（已删除）
| 文件 | 状态 |
|------|------|
| `core/auto_bot_vision_ocr_v4.py` | 已删除 |
| `core/auto_bot_vision_ocr_v3.py` | 已删除 |
| `core/auto_bot_vision_ocr_v2.py` | 已删除 |

---

## ⚠️ 注意事项

1. **微信窗口需要可见**
   - OCR 需要截取微信窗口
   - 窗口不能被其他窗口完全遮挡

2. **避免高频发送**
   - 建议间隔 3-5 秒
   - 防止被微信限制

3. **OCR 精度**
   - 小字体昵称可能识别不准
   - 复杂背景可能影响识别

4. **系统消息过滤**
   - 自动过滤 "[图片]"、"[视频]" 等
   - 过滤时间戳、撤回消息提示

---

## ❌ 废弃方案

### 数据库解密方案（不再使用）
```
原方案: 解密微信 SQLite 数据库读取消息
状态: 已废弃
原因: 需要关闭 SIP + 获取 db_key，过于复杂且有安全风险
替代: Vision OCR 视觉识别
```

---


