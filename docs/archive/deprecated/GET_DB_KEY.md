# 获取微信数据库密钥 (db_key)

> ⚠️ **重要警告：数据库解密方案已废弃**
> 
> 当前项目已全面迁移至 **Vision OCR 视觉识别方案**，无需关闭 SIP、无需获取 db_key。
> 保留本文档仅供历史参考，请勿按本文档操作获取 db_key 或配置数据库解密。

## 方法 1: 使用 wechat-dump (推荐)

```bash
# 1. 安装依赖
brew install sqlcipher openssl

# 2. 下载工具
git clone https://github.com/0xHJK/wechat-dump.git
cd wechat-dump

# 3. 运行解密工具
python3 decrypt.py

# 4. 按提示输入微信数据目录
# 通常: ~/Library/Containers/com.tencent.xinWeChat/Data/...

# 5. 工具会显示 db_key，复制保存
```

## 方法 2: 从内存中提取 (高级)

```bash
# 使用 lldb 附加到微信进程
lldb -p $(pgrep WeChat)

# 在 lldb 中执行命令查找密钥
# 需要了解 SQLCipher 密钥存储位置
```

## 方法 3: 使用第三方工具

搜索以下工具：
- "wechat db key mac"
- "chatlog-bot"
- "wechat-db-decrypt-macos"

## 配置 db_key

获取到 db_key 后，编辑配置文件：

```bash
# 编辑配置
nano config/config.yaml

# 修改 db_key 字段
db_key: "1234567890abcdef..."

# 或设置环境变量
export WECHAT_DB_KEY="1234567890abcdef..."
```

## 验证

```bash
# 测试解密
sqlcipher ~/Library/Containers/.../msg_0.db
> PRAGMA key = "x'YOUR_DB_KEY'";
> .tables
# 如果能显示表名，说明密钥正确
```

## 常见问题

### Q: 提示 "file is not a database"
- db_key 不正确
- 数据库文件已损坏

### Q: 找不到微信目录
```bash
# 手动查找
find ~/Library/Containers/com.tencent.xinWeChat -name "msg_*.db"
```

### Q: SIP 已关闭但仍无法访问
```bash
# 检查文件权限
ls -la ~/Library/Containers/com.tencent.xinWeChat/Data/

# 可能需要修改权限
chmod -R 755 ~/Library/Containers/com.tencent.xinWeChat/Data/
```
