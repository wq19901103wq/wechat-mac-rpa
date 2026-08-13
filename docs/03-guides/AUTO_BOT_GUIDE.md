# Mac 微信全自动 RPA 完整指南

> ⚠️ **本文档描述的数据库解密方案已废弃**
>
> 当前项目已全面迁移至 **Vision OCR 视觉识别方案**，无需关闭 SIP、无需获取 db_key。
>
> **当前唯一可用入口**：
> ```bash
> cd ~/wechat-mac-rpa
> python3 run_bot.py
> ```
>
> **相关文档**：
> - [AI 快速上手](../01-quickstart/AI_QUICKSTART.md)
> - [架构设计](../02-architecture/ARCHITECTURE.md)
> - [解决方案汇总](../04-troubleshooting/SOLUTIONS.md)
>
> 以下正文属于历史归档，操作步骤均已废弃，请勿执行。

---

## 历史归档（已废弃，请勿执行）

原方案基于 SQLCipher 解密微信数据库，需要关闭 SIP、获取 db_key。

由于安全风险高、操作复杂，该方案已被 Vision OCR 视觉识别方案完全替代。

如需了解旧方案的技术细节，请查阅 `docs/archive/deprecated/GET_DB_KEY.md`、`docs/archive/deprecated/KEY_EXTRACTION_GUIDE.md`。

---

**废弃日期**：2026-04-15
**替代方案**：Vision OCR 视觉识别（`src/bot/wechat_bot.py`）
