# wcdb-key-tool

微信数据库（WCDB / SQLCipher4）密钥提取工具，覆盖 Linux / macOS / Windows 三个平台。

Extract WeChat (WCDB/SQLCipher4) database encryption keys — Linux, macOS and Windows.

本仓库只做一件事：把从微信自己的进程里，合法地取出**自己账号**数据库密钥这件事的
思路和代码公开出来，不做采集、不做批量、不接触任何服务器通信。

## 背景 / Background

微信 4.1+ 版本不再在进程内存中缓存明文数据库密钥（raw key），而是只保留一个
passphrase，真正的加密密钥需要再做一轮很慢的密码学运算才能算出来。这导致所有
基于内存模式扫描（`x'<hex>'`）的老工具全部失效。

WeChat 4.1+ no longer caches the raw database encryption key in process memory —
only a passphrase remains, from which the real key must be derived. This breaks
every existing tool that relies on `x'<hex>'` memory pattern scanning.

## 兼容性 / Compatibility

| 平台 | 微信 4.0.x（老版本） | 微信 4.1+（新版本） |
|------|---|---|
| Linux | ✅ 内存扫描 | ✅ GDB 断点 + PBKDF2 派生（已验证） |
| macOS | ✅ 内存扫描 | ✅ LLDB 断点 + PBKDF2 派生（已验证） |
| Windows | ✅ 内存扫描 | ✅ 运行时 `Config.Cipher` 只读扫描 + PBKDF2（已验证） |

- Linux: `wcdb_key_tool.py`
- macOS: `wcdb_key_tool_macos.py`
- Windows: `wcdb_key_tool_windows.py`

## 核心原理 / How It Works

三个平台最终都是同一套密码学逻辑，只是"怎么从微信进程里把原料弄出来"这一步不同：

1. **拿到密钥原料**（三条路线，按微信版本二选一）
   - 老版本：微信自己把 `raw key + salt` 拼成十六进制字符串，明文缓存在进程内存里，
     直接内存扫描就能拿到。
   - 新版本：内存里只剩 passphrase，拿不到直接能用的密钥，得走第 2 步。
   - 新版本的 passphrase 只在微信**登录时**做一次计算，之后就常驻内存不会重复计算，
     所以要抓这一刻，得用调试器在这次计算发生的地方下断点，逼用户退出登录再重新
     登录来触发一次新的计算，从函数参数寄存器里把 passphrase 读出来。
     - Linux：微信 Linux 二进制没有暴露可用的系统符号，只能用 ELF 静态分析（在
       `.rodata` 里找 WCDB 特征字符串，顺着交叉引用找到处理密钥的函数入口），
       再用 GDB 在这个地址上断点。（`elf_analyzer.py` + `gdb_capture.py`，已在
       真机验证）
     - macOS：微信直接调用了苹果系统自带的 `CommonCrypto` 库做密钥派生，这是一个
       公开的系统符号，不需要逆向微信自己的二进制，直接用 LLDB 断在
       `CCKeyDerivationPBKDF` 上就行。（`lldb_capture.py`，已在真机验证 18/18）
   - Windows：主路径是只读运行时扫描。微信 4.1+ 会把可用的
     `Config.Cipher` 相关对象留在进程内存里，脚本先定位这个对象，再解码候选
     `x'<64hex key><32hex salt>'` 片段并用 HMAC 校验；老版本继续走内存扫描。
     `capture-experimental` 仍保留为研究性备用路线，不是主路径。

2. **PBKDF2 派生每个库的真实密钥**：拿到 passphrase 后，对每个数据库文件，用它
   自己的 16 字节 salt 做 `PBKDF2-HMAC-SHA512`（256,000 轮迭代），算出这个库专属
   的 32 字节 AES-256 密钥。

3. **HMAC 校验**：派生出密钥后不能直接信，要用密钥对数据库第一页做
   `HMAC-SHA512` 校验，跟数据库自己存的 HMAC 对上了，才说明这把密钥真的对。
   这一步是三个平台通用的"防止抓错"的安全网，也是判断运行时扫描或备用方案
   有没有真的成功的唯一标准——命中了不代表读到的就是对的东西，HMAC 校验通过
   才算数。

4. **AES-256-CBC 逐页解密**：还原出标准 SQLite 文件。

## 安装 / Install

单文件即可运行，无需 pip install 任何第三方库（三个平台都是直接调用系统自带的
加密库：Linux 走 OpenSSL、macOS 走 CommonCrypto、Windows 走 CNG）。

```bash
# Linux
sudo apt install gdb

# macOS
xcode-select --install   # 提供 lldb

# Windows
# 老版本内存扫描 / 新版 runtime 扫描都不需要额外依赖
# capture-experimental 只在你要测试研究性路线时需要 cdb.exe
```

## 使用 / Usage

```bash
# Linux
sudo python3 wcdb_key_tool.py extract --decrypt

# macOS（首次需要先对微信重签名去掉 Hardened Runtime，见脚本内 Prerequisite）
sudo codesign --force --deep --sign - /Applications/WeChat.app
sudo python3 wcdb_key_tool_macos.py extract --decrypt

# Windows（优先：新版 runtime 扫描；老版本会自动回退到内存扫描）
python3 wcdb_key_tool_windows.py extract --decrypt

# Windows（研究性备用路线，非主路径）
python3 wcdb_key_tool_windows.py capture-experimental
```

首次提取都需要在微信里**退出登录再重新登录**一次（这是为了触发密钥的重新计算，
断点才有机会命中）；抓到的 passphrase 会缓存下来，之后就不用重复这一步了。

## Windows 新版：runtime Config.Cipher 扫描

Windows 4.1+ 的主路径已经改成只读运行时扫描，不再依赖实验性断点猜测。脚本先在
进程里定位 `com.Tencent.WCDB.Config.Cipher` 相关对象，解码出候选
`x'<key><salt>'` 片段，再用数据库 HMAC 验证；若扫不到，再自动回退到老版本
内存扫描。

`capture-experimental` 仍保留为研究性备用命令，默认不影响 `extract`。

## 安全说明 / Security Notes

- 本工具只用于提取**用户自己设备上自己账号**的数据库密钥
- 调试器只在密钥计算的一瞬间附加、读一次寄存器/内存就立即 detach，不修改微信
  的任何行为，不接触网络协议
- 不会触发封号

## FAQ

**Q: 会不会封号？**
A: 不会。工具只在密钥计算瞬间读一次内存/寄存器值，整个过程很短，不修改任何程序
行为，不接触微信服务器通信。

**Q: passphrase 存在哪里？**
A: 存储在 `~/.wcdb-key-tool/wechat-passphrase.json`（权限 600），仅当前用户可读，
三个平台通用这一个路径。

**Q: 为什么 Linux/macOS 需要 sudo？**
A: GDB / LLDB 需要 `ptrace`（macOS 上是 `task_for_pid`）权限来附加到其他进程的
内存空间。Linux 也可以用 `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
临时放开权限（重启后恢复），不用每次都 sudo。

**Q: 微信更新后还能用吗？**
A: Linux 大概率可以——ELF 静态分析通过字符串交叉引用定位函数，只要微信继续使用
WCDB 的 `com.Tencent.WCDB.Config.Cipher` 字符串，就能自动适配。macOS 断的是苹果
系统函数，跟微信自己的版本无关，理论上更稳定，但微信升级后可能需要重新对 App
执行一次重签名（Hardened Runtime 被系统还原后 `task_for_pid` 会失败）。

## 致谢 / Credits

- [kkocdko](https://kkocdko.site/post/202510212134) — Linux 上 GDB 断点法的原始思路
- [wxchat-export](https://github.com/lopleec/wxchat-export) — Linux 上 ELF 静态分析方法
- [ylytdeng/wechat-decrypt](https://github.com/ylytdeng/wechat-decrypt) — 内存扫描基础代码
- [TANGandXUE](https://github.com/TANGandXUE) — PBKDF2 派生方法 + macOS/Windows 移植 + 完整集成

## License

MIT — see [LICENSE](LICENSE)
