# 手动获取微信 db_key 的完整指南

> ⚠️ **重要警告：数据库解密方案已废弃**
> 
> 当前项目已全面迁移至 **Vision OCR 视觉识别方案**，无需关闭 SIP、无需获取 db_key。
> 保留本文档仅供历史参考，请勿按本文档操作获取 db_key 或配置数据库解密。

## 方法 1: 使用微信小助手（最可靠）

由于自动安装失败，请按以下步骤手动安装：

### 步骤 1: 下载微信小助手

1. 打开浏览器访问：
   ```
   https://github.com/MustangYM/WeChatExtension-ForMac/releases
   ```

2. 下载最新版本的 `WeChatExtension.zip`

### 步骤 2: 手动安装

```bash
# 1. 解压下载的文件
unzip ~/Downloads/WeChatExtension.zip -d /tmp/

# 2. 进入目录
cd /tmp/WeChatExtension-ForMac-master/WeChatExtension/Rely

# 3. 运行安装脚本
./Install.sh
```

或者手动安装：
```bash
# 1. 备份微信
sudo cp /Applications/WeChat.app/Contents/MacOS/WeChat /Applications/WeChat.app/Contents/MacOS/WeChat.backup

# 2. 复制 framework
sudo cp -R /tmp/WeChatExtension-ForMac-master/WeChatExtension/Rely/Plugin/WeChatExtension/WeChatExtension.framework \
          /Applications/WeChat.app/Contents/MacOS/
```

### 步骤 3: 获取 db_key

1. 完全退出微信（Cmd+Q）
2. 重新打开微信
3. 在菜单栏找到「微信小助手」→「设置」
4. 查看 db_key（32位十六进制字符串）

### 步骤 4: 配置到本项目

```bash
# 编辑配置文件
nano ~/wechat-mac-rpa/config/config.yaml

# 修改 db_key
wechat:
  db_key: "你的32位密钥"
```

---

## 方法 2: 使用 Windows 虚拟机/双系统

Mac 版微信获取 db_key 比较困难，Windows 版本更容易：

1. 在 Windows 上安装微信
2. 登录微信
3. 使用 Windows 版的微信数据库工具获取 db_key
4. db_key 在 Mac 和 Windows 上是相同的！

推荐的 Windows 工具：
- [wechat-dump](https://github.com/0xHJK/wechat-dump)
- [wxBackup](https://github.com/TransparentLC/wxBackup)

---

## 方法 3: 使用微信内置开发者模式

某些版本的微信有隐藏的开发者模式：

```bash
# 尝试开启开发者模式
# 在微信运行时，在终端执行：
defaults write com.tencent.xinWeChat WebKitDeveloperExtras -bool true
```

然后查看微信日志文件是否有密钥信息。

---

## 方法 4: 使用数据库文件的时间戳特征

某些版本的微信使用固定的密钥模式：

```python
# 尝试的常见密钥列表
common_keys = [
    "00000000000000000000000000000000",
    "1234567890abcdef1234567890abcdef",
    "deadbeefdeadbeefdeadbeefdeadbeef",
    # 等等...
]
```

---

## 方法 5: 使用云备份恢复

1. 微信 → 设置 → 通用 → 聊天记录备份与迁移
2. 备份到另一台设备
3. 某些备份工具会显示 db_key

---

## 临时解决方案

如果以上方法都不可行，您可以：

### 1. 先使用简化版机器人
```bash
cd ~/wechat-mac-rpa
python3 run_simple.py
```

### 2. 等待微信小助手更新
关注 https://github.com/MustangYM/WeChatExtension-ForMac

### 3. 使用其他平台
- 使用 iPad 微信 + 快捷指令
- 使用 Windows 微信 + 自动化工具
- 使用微信网页版（已停止服务）

---

## 需要帮助？

如果以上方法都无法解决问题，建议：

1. **先用简化版**（方案 A）满足基本需求
2. **创建 Issue** 在 GitHub 上寻求帮助
3. **等待更新**，微信数据库解密工具在不断更新

---

## 补充：数据库文件位置

```
~/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/
├── 2.0b4.0.9/                           # 版本号
│   └── [32位哈希]/                      # 用户ID
│       └── Message/
│           ├── msg_0.db                  # 主消息数据库
│           ├── msg_1.db                  # 其他数据库
│           └── ...
```

数据库文件是 SQLCipher 加密的，需要 db_key 才能解密。
