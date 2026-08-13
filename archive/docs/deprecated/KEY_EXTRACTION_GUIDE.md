# 微信数据库密钥获取指南

> ⚠️ **重要警告：数据库解密方案已废弃**
> 
> 当前项目已全面迁移至 **Vision OCR 视觉识别方案**，无需关闭 SIP、无需获取 db_key。
> 保留本文档仅供历史参考，请勿按本文档操作获取 db_key 或配置数据库解密。

## 当前状态
- ✅ SIP 已关闭
- ✅ 微信正在运行且已登录
- ✅ 找到数据库目录
- ❌ strings 命令未找到密钥（可能是微信版本更新导致）

## 可选方案

### 方案1: 使用第三方工具（推荐）

#### 方法 A: 使用微信小助手
1. 安装微信小助手插件：https://github.com/MustangYM/WeChatExtension-ForMac
2. 安装后在微信菜单中找到「设置」→「小助手」
3. 查看 db_key

#### 方法 B: 使用 pywxdump
```bash
pip3 install pywxdump
wxdump info
```

#### 方法 C: 使用 wechat-dump-rs
```bash
cargo install wechat-dump-rs
wechat-dump-rs
```

### 方案2: 使用简化版机器人（立即可用）

如果无法获取 db_key，可以使用简化版机器人：

```bash
cd ~/wechat-mac-rpa
python3 run_simple.py
```

**功能**: 手动输入消息 → AI生成回复 → 自动发送到微信

**使用方法**:
```
文件传输助手|你好
文件传输助手|讲个Python装饰器
```

### 方案3: 使用 Accessibility API 方式

不需要 db_key，直接读取微信界面消息（需要授权辅助功能）：

```bash
# 1. 授权辅助功能
系统设置 → 隐私与安全 → 辅助功能 → 添加终端

# 2. 运行
python3 examples/simple_mac_bot.py
```

## 手动获取 db_key 的步骤

如果以上方法都不行，可以尝试以下高级方法：

### 方法1: 使用 Frida 注入
```bash
pip3 install frida-tools

# 创建脚本 find_key.js
# ...

frida -p $(pgrep WeChat) -l find_key.js
```

### 方法2: 使用 lldb 调试
```bash
# 附加到微信
lldb -p $(pgrep WeChat)

# 在 lldb 中设置断点并查找密钥
# ...
```

### 方法3: 分析内存转储
```bash
# 生成核心转储
gcore -o /tmp/wechat_core $(pgrep WeChat)

# 搜索密钥
strings /tmp/wechat_core | grep -E "[0-9a-f]{64}" | head -5
```

## 建议

对于您的情况，我建议：

1. **立即使用**: 简化版机器人（不需要 db_key）
2. **稍后尝试**: 安装微信小助手插件获取 db_key
3. **最终目标**: 配置全自动机器人

## 联系支持

如果遇到问题，可以：
1. 查看详细文档: `AUTO_BOT_GUIDE.md`
2. 尝试配置向导: `python3 setup_auto.py`
3. 使用简化版: `python3 run_simple.py`
