# Mac 微信全自动 RPA 机器人

数据库解密 + FSEvents 监听 + Kimi 大模型

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Mac 微信全自动机器人                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐    解密     ┌──────────────┐        │
│   │ 微信加密数据库 │  ───────→  │  明文 SQLite  │        │
│   │  (msg_*.db)  │            │              │        │
│   └──────────────┘            └──────────────┘        │
│          ↑                            ↓                │
│    FSEvents                        SQL 查询             │
│    文件监听                    读取最新消息              │
│          │                            ↓                │
│          └────────────────→   ┌──────────────┐        │
│                               │   Kimi LLM   │        │
│                               │   生成回复    │        │
│                               └──────────────┘        │
│                                      ↓                 │
│                               ┌──────────────┐        │
│                               │ Accessibility│        │
│                               │  发送消息     │        │
│                               └──────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🔧 前置条件

### 1. 关闭 SIP（系统完整性保护）

```bash
# 重启进入恢复模式
cmd + R

# 在终端执行
csrutil disable
reboot
```

### 2. 获取微信数据库密钥

使用开源工具：
```bash
# 方案1: wechat-dump (推荐)
git clone https://github.com/0xHJK/wechat-dump.git
cd wechat-dump
python3 decrypt.py

# 方案2: 使用 dbkey 工具
# 从 GitHub 搜索 wechat-db-key-mac
```

### 3. 授权 Accessibility

系统设置 → 隐私与安全 → 辅助功能 → 添加终端/Python

## 🚀 安装

```bash
pip install pyobjc pyautogui pyperclip openai watchdog
```

## 📁 项目结构

```
wechat-mac-rpa/
├── core/
│   ├── db_decrypt.py      # 数据库解密
│   ├── db_watcher.py      # FSEvents 监听
│   ├── message_reader.py  # 消息读取
│   └── bot_engine.py      # 机器人引擎
├── utils/
│   ├── accessibility.py   # Accessibility API
│   └── llm_client.py      # Kimi 客户端
├── config/
│   └── config.yaml        # 配置文件
└── run.py                 # 启动入口
```

## ⚠️ 安全提示

- 关闭 SIP 会降低系统安全性
- 仅用于个人学习和研究
- 遵守微信使用规范
