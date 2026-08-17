# Windows 开发计划：本地数据读取 + RPA（目标机迁移）

> 版本: v1.1
> 日期: 2026-08-17
> 目标: 开发/运行目标机器从 macOS 迁移到 Windows，利用 Windows 可自由读取微信进程内存的能力，实现可靠的本地数据读取（联系人/会话/消息），并作为 RPA 的结构化数据源

> **v1.1 更新（2026-08-17，Phase 0 已在 Windows 真机验证通过）**
> - 提 key 工具链变更：原引用的 `scan_keys.py`（扫 `x'<hex>'`）在微信 4.1+ 已失效（4.1+ 不再缓存明文 key 字符串），改用 `wcdb-key-tool`（TANGandXue，MIT）只读扫描 `com.Tencent.WCDB.Config.Cipher` 对象；实测 23/23 个库 1.3 秒取 key、HMAC 全部通过，无需注入/管理员。
> - 本机数据目录实测为 `D:\Users\gxh\xwechat_files\gqingfeng_0c9c\db_storage`（非默认 `Documents\xwechat_files`）。
> - 消息表映射规则已确认：`Msg_<md5(talker)>`；发送者通过 `Name2Id.rowid = real_sender_id` 解析。

---

## 1. 背景与决策

### 1.1 为什么放弃 macOS 本地库解密

| 方案 | 结论 |
|------|------|
| macOS 内存提 key 解密（wechat-export-macos / wechat-cli / wechat-dump-rs / wxkey） | 需 `sudo codesign --force --deep --sign -` 重签微信（改代码身份、破坏登录态、更新即还原）或关闭 SIP，属于"绕过技术保护措施"，**有封号风险**，用户明确拒绝 |
| WeFlow（hicccc77/WeFlow） | 同为"提取内存密钥 + 解密"路线，已被 DMCA 下架，README 自述"不再提取密钥 / 不再解密数据库"，已弃坑 |
| macOS AX 辅助功能读界面 | 微信 4.x 主窗口自绘，AX 树几乎为空（`entire contents` 仅 4 个元素、无列表项），读不出联系人 |
| 本地明文（MMKV / config） | 实测无联系人明文，联系人只存在于加密的 `contact.db` |

**补充（v1.1）**：`wcdb-key-tool` 的 macOS 路线（LLDB 断点 `CCKeyDerivationPBKDF` 抓 passphrase + PBKDF2 派生）同样**必须**先 ad-hoc 重签去掉 Hardened Runtime 才能 `task_for_pid`，且需 sudo + 首次退出登录重登触发断点——与上表 macOS 方案同等风险，用户同样拒绝。

**结论**：macOS 上本地库解密路线整体封死；安全路线只剩纯视觉 UI 自动化（不动进程、不改签名）。

### 1.2 为什么 Windows 可行

- Windows 没有 Hardened Runtime 限制，`ReadProcessMemory`（pymem/ctypes）同用户权限即可读微信进程内存，**无需重签、无需关闭任何保护**，仅需微信已登录
- 微信 4.x 在 Windows 上同样是 WCDB/SQLCipher 4，密钥格式与解密算法与 macOS 完全一致
- 工具生态成熟，且多为被动读内存（不注入、不挂钩），封号风险远低于 macOS 重签

**决策**：目标机器切换为 **Windows**。数据读取走"进程内存提 key + SQLCipher 解密"路线；UI 操作继续走视觉/自动化。

---

## 2. 调查结论（证据与原理）

### 2.1 微信 4.x 本地库加密原理

- 每个 `.db`（contact/session/message/...）是独立 SQLCipher 4 加密库
- 解密参数：AES-256-CBC、HMAC-SHA512、page_size=4096、reserve=80（IV 16 + HMAC 64）
- HMAC 密钥派生：`PBKDF2-HMAC-SHA512(enc_key, salt XOR 0x3a, iterations=2, dklen=32)`
- 通过 DB 文件头 16 字节盐值匹配内存中的 key + Page-1 HMAC 校验，确认 key 正确

### 2.2 联系人表结构

`contact.db` → `contact` 表：

```sql
SELECT username, alias, remark, nick_name, local_type FROM contact;
```

- `username`：wxid（如 `wxid_xxx`），群聊为 `xxx@chatroom`
- `nick_name` / `remark` / `alias`：昵称 / 备注 / 别名
- `local_type = 0` 表示普通好友

### 2.3 微信 4.1+ 密钥存储变化（关键）

- 微信 **4.1+ 不再在进程内存里缓存明文 `x'<64hex key><32hex salt>'` 字符串**，基于该模式的老工具（`scan_keys.py` / wechat-dump-rs / PyWxDump / chatlog / WeChatMsg）全部失效；其中多个已被 Tencent DMCA 下架或作者删库
- 4.1+ 的 key 材料以 **XOR 编码形式**存在于 WCDB 的 `com.Tencent.WCDB.Config.Cipher` 对象中
- 方案：**`wcdb-key-tool`（TANGandXue，MIT）** 只读扫描该对象 → XOR 解码候选 `x'<key><salt>'` → 对每个 .db 第一页做 HMAC 校验，通过才采纳。Windows 单文件实现 `wcdb_key_tool_windows.py`，无第三方依赖（走系统 CNG），无需管理员

### 2.4 参考工具与源码

| 工具 | 平台 | 说明 | 状态 |
|------|------|------|------|
| TANGandXue/wcdb-key-tool | Win/mac/Linux | **4.1+ 唯一可用路线**：Windows 只读 Config.Cipher 扫描 + HMAC；macOS/Linux 走调试器断点 | ✅ 已 vendor 至 `third_party/wcdb-key-tool/` |
| ZedeX/weixin-decrypte-script | Windows | `scan_keys.py`（pymem 提 key）+ `decrypt_db.py` + `api_server.py` | ⚠️ 仅 4.0.x 可用；4.1+ 扫不到 key |
| ydotdog/wechat-export-macos | macOS/通用 | `decrypt_db.py` 跨平台可复用，`config.py` 已内置 Windows 路径检测 | ⚠️ 同上，依赖 `x'<hex>'` 扫描 |
| 328336690/wechat-decrypt | Windows | 内存提 key + 解密 + 实时消息监听 + MCP Server | ⚠️ 需确认 4.1+ 兼容性 |
| 0xlane/wechat-dump-rs | Win/mac | Rust，内存暴力搜索 | ❌ 已被 DMCA 下架 |
| xaoyaoo/PyWxDump、sjzar/chatlog | Windows | 老牌工具 | ❌ 作者删库（2026-08 已仅剩 README） |

---

## 3. Windows 目标环境（本机实测）

- Windows 10/11 x64，微信 **4.1.12.55**（已登录，进程名 `Weixin.exe`，主进程 + 多个子进程）
- Python 3.10+，依赖：`pycryptodome zstandard`（提 key 走 vendor 脚本，无额外依赖）
- 权限：**无需管理员**（同用户 `ReadProcessMemory` 即可；实测非管理员成功）
- 数据目录（本机实测）：`D:\Users\gxh\xwechat_files\gqingfeng_0c9c\db_storage\`
  - `contact/contact.db` — 联系人（实测 4496 条）
  - `session/session.db` — 会话（实测 403 条）
  - `message/message_*.db` — 聊天记录（按会话分表 `Msg_<md5(talker)>`，跨 `message_0/1/2` 分片）
  - `message/message_resource.db` — 会话/发送者索引
- 注意：默认路径是 `Documents\xwechat_files`，但本机微信文件管理被改到 D 盘；探测逻辑需覆盖非默认路径

---

## 4. 开发方案（分阶段）

### Phase 0：环境与最小验证（Windows 机器，✅ 2026-08-17 已完成）

```bash
pip install pycryptodome zstandard
# 1) 提 key + 解密（只读，微信已登录即可，无需管理员）
python third_party/wcdb-key-tool/wcdb_key_tool_windows.py extract --db-dir "D:\Users\gxh\xwechat_files\gqingfeng_0c9c\db_storage" --output all_keys.json
python third_party/wcdb-key-tool/wcdb_key_tool_windows.py decrypt --db-dir "D:\Users\gxh\xwechat_files\gqingfeng_0c9c\db_storage" --keys all_keys.json --output decrypted
# 3) 读联系人
python -c "import sqlite3; ..."   # 见 Phase 1 封装
```

验收：✅ 23/23 库取 key + 解密 HMAC 全通过；✅ 联系人 4496 条（wxid / 昵称 / 备注字段齐全）；✅ 会话 403 条；✅ 消息明文可读。

### Phase 1：数据读取层（Windows 原生模块，✅ 已实现）

- 实现于 `src/platform/windows/`，核心模块：
  - `config.py` — 自动探测 `db_storage`（环境变量 `WECHAT_DATA_DIR` 覆盖）+ 缓存目录
  - `key_provider.py` — 提 key（subprocess 调 vendor 脚本）+ 缓存到 `data/wechat/`；微信不在/失败时回退缓存
  - `decryptor.py` — 解密到 `data/wechat/decrypted/`，按 key 指纹避免重复解密
  - `message_codec.py` — 消息内容解码（ZSTD / 明文、`Name2Id.rowid → real_sender_id` 发送者解析）
  - `wechat_data.py` — 稳定接口：`list_contacts()` / `get_sessions()` / `get_messages(talker, limit, offset)`
- 参考 `api_server.py` 可暴露 REST API（`/api/v1/contact`、`/api/v1/chatlog` 等，后续 Phase 按需）

### Phase 2：UI 自动化层（Windows 移植）

- 替换 macOS 依赖：`MacOSSystemAutomation`（Quartz/CGEvent）→ `pywinauto` / `pywin32` / `uiautomation`
- 窗口截图：`mss` / Pillow `ImageGrab`（只截微信窗口，不截全屏）
- 感知 → 推理 → 行动 → 记忆架构保持不变，OCR + LLM 驱动逻辑复用

### Phase 3：集成与多账号

- 数据源切换：视觉 OCR / 本地库双通道，环境变量切换（对齐现有 `WEFLOW_MODE=ocr/weflow/hybrid` 设计）
- 多账号：不同 `xwechat_files/<wxid>` 目录对应不同登录账号，提 key 时按进程匹配
- 兜底：本地库 API 异常时自动 fallback 到 OCR，Bot 不中断

---

## 5. 风险与注意事项

- **封号风险**：被动读内存风险低但非零；严禁注入、挂钩、修改微信文件或数据
- **微信更新**：可能改变内存布局/进程名（`Weixin.exe`）或 `Config.Cipher` 对象结构，提 key 逻辑需随版本适配（vendor 脚本可整体升级）
- **key 变化**：重新登录/账号切换后 key 会变，需自动重新提 key（已内置）
- **权限**：提 key 不需要管理员（实测），但微信必须已登录
- **数据安全**：解密 key 与聊天记录为敏感数据，不得提交到 Git（解密输出在 `data/`，已 `.gitignore`）
- **进程名**：当前按 `Weixin.exe` 匹配，如微信进程名不同需调整

---

## 6. 下一步

1. ✅ 装微信 4.x 并登录，确认 `db_storage` 存在（本机：`D:\Users\gxh\xwechat_files\gqingfeng_0c9c\db_storage`）
2. ✅ 装 Python 依赖，跑 Phase 0 提 key + 解密，验证能读出联系人
3. ✅ Phase 1 封装数据读取层（`src/platform/windows/`）
4. ⏳ Phase 2：UI 自动化层 Windows 移植（pywinauto/pywin32/mss）
5. ⏳ Phase 3：集成与多账号（本地库 ↔ OCR 双通道，`WEFLOW_MODE` 对齐）
