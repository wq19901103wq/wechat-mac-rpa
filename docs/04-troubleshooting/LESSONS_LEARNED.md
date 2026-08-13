# 微信 Mac RPA 项目经验教训

> 记录开发过程中的关键问题和修复方案，避免重复踩坑

---

## 一、OCR 识别与解析

### 1.1 标题栏识别范围必须精确

**问题**: `title_y_max` 太宽泛，把窗口控制按钮（®、(S.）识别成聊天名称，导致 "示例用户甲" → "®v QS."

**修复**:
```python
title_y_max = 95          # 覆盖 y=90 的标题，排除 y≥100 的消息区
title_x_max_ratio = 0.95  # 排除右侧图标区域
```

**原则**: 标题栏识别宁窄勿宽，必须排除窗口装饰元素。修复后添加回归测试 `test_regression_title_y_max_extracts_chat_name` 确保 y=90 的标题能被捕获。

### 1.2 输入框和消息区可以用 y 坐标精确分割

**问题**: `input_y_min` 设置得太宽松，把输入框内容误识别为消息。

**修正认知**:
- ✅ **y 坐标完全可以区分输入框**。微信 Mac 版输入框固定在底部区域
- ❌ 之前说"不能用 y 坐标"是错的——不是不能，而是阈值设错了

**修复**:
```python
input_y_min = 1040  # 输入框顶部边界（按 LayoutProfile 配置）
```

**同时注意**:
- y 坐标过滤解决**输入框残留**问题
- **已发送的消息**（如循环产生的"aaaa"）在消息区（y < 1160），要靠**去重机制**解决
- 这是两个不同的问题，不能混为一谈

### 1.3 时间戳过滤必须严格

**问题**: OCR 把时间戳（"00:04", "昨天 23:31", "星期六"）识别为消息，打乱消息序列。

**修复**:
```python
TIMESTAMP_PATTERNS = [
    r'^\d{2}:\d{2}$',
    r'^昨天\s*\d{1,2}:\d{2}$',
    r'^星期[一二三四五六日]$',
]
```

---

## 二、消息发送

### 2.1 中文输入法下 "Command+A" 会产生乱码

**问题**: V4 发送消息前用 `keystroke "a" using command down` 全选，在中文输入法下 "a" 被输入成拼音，产生 "laayaua5aapangaaaaa~" 等乱码。

**根因**: `keystroke` 在中文输入法下会先触发输入法，而不是快捷键。

**修复**: 去掉全选，直接像 V2 一样用 `pbcopy + Command+V` 粘贴：
```python
subprocess.run(['pbcopy'], input=text.encode('utf-8'), timeout=2)
# 然后 AppleScript: keystroke "v" using command down
```

**原则**: 避免在中文输入法环境下用 `keystroke` 输入任何字母字符。

### 2.2 没有去重机制会导致循环发送

**问题**: 机器人发送消息后，下一轮 OCR 识别到刚发的消息，再次触发回复，形成死循环。

**修复**:
```python
# 发送后记录内容与时间、估计Y坐标
self.sent_messages.append(SentMessage(text=text, sent_at=time.time(), approx_y=approx_y))

# 下轮识别时回声检测：时间窗口优先（10s内），Y坐标辅助
def _is_echo(self, identity, sent):
    text_match = sent.text in msg.text or msg.text in sent.text
    time_match = (time.time() - sent.sent_at) < 10.0
    y_match = abs(sent.approx_y - identity.approx_y) < 80
    return text_match and time_match and y_match
```

**原则**: 任何自动回复系统必须有**内容+位置+时间**的多维去重，不能仅靠字符串包含或时间戳。聊天滚动时 Y 坐标会变化，因此时间窗口是回声检测的首要条件。

### 2.3 冷却期是必要的，但不能替代去重

**问题**: 仅靠 30 秒冷却期无法阻止循环——30 秒后仍会识别到自己的消息。

**原则**: 冷却期和去重是双重保护，缺一不可。

---

## 三、代码与架构

### 3.1 不要重复造轮子，已有 V2 直接用

**问题**: V4 重新实现了发送逻辑，加了不必要的全选操作，引入了 V2 没有的问题。

**教训**:
- V2 的 `pbcopy + Command+V` 方案已经验证稳定
- 重构时不要轻易改动底层稳定模块
- 如果 V2 能用，优先复用而不是重写

### 3.2 修复要治本，不要堆补丁

**反模式**:
- ❌ "过滤特殊字符 ®"
- ❌ "跳过超长乱码消息"
- ❌ "添加置信度阈值"

**正解**:
- ✅ 找到乱码产生的根因（输入法 + keystroke）
- ✅ 从根因上修复（改用 pbcopy）

### 3.3 模块化不等于没有依赖

**问题**: V4 的 parser 和 storage 各自独立，但去重逻辑跨了多个模块，导致循环发送。

**原则**: 业务逻辑（如循环检测）必须放在 bot 的 orchestration 层，而不是拆到各个模块里。

---

## 四、测试

### 4.1 测试必须验证准确性，不是"能跑通"

**错误标准**:
- ❌ "解析成功，返回了 6 条消息"
- ❌ "没有抛异常"

**正确标准**:
- ✅ 聊天名称准确率 >= 95%
- ✅ 发送者类型识别率 >= 90%
- ✅ 时间戳过滤率 = 100%
- ✅ 输入框残留过滤率达标

### 4.2 错误截图必须立即归档为测试用例

**问题**: 早期没有系统保存错误截图，导致修复后无法回归验证。

**修复**: 建立 `tests/fixtures/errors/` 目录，每个错误包含：
- `error_XXX.png`: 原始截图
- `error_XXX.json`: 期望结果 + 问题描述

### 4.3 测试用例必须包含预期数据

**问题**: 批量导入 20 张截图时，很多用例缺少 `expected` 数据，只能验证"不报错"。

**原则**: 每个测试用例必须有完整的期望输出，包括聊天名称、消息列表、发送者类型。

---

## 五、修复纪律（2026-05-16 新增）

### 5.1 不要把设计特性当 bug 修

**问题**: 本地路径 `_run_local_only` 返回 `messages=[]` 是刻意设计——只做变化检测，消息提取走 API。但我看到"messages=0条"就当成 bug，强行添加了本地消息提取。

**后果**: 本地 OCR 的粗糙结果（聊天列表日期 "05/06"、未读数字 "1000"、图标文字 "i0i0"）全部混进了 session 历史和 LLM prompt，导致回复混乱。

**原则**:
- 看到"异常"代码，先假设是设计特性，查注释/spec/git history 确认
- 不确定就问用户，不要猜

### 5.2 一修一改，禁止顺手多改

**问题**: 用户确认的方案是"扩大 message_region + sender 重试从头来"，但我顺便多写了本地消息提取逻辑，没有汇报。

**后果**: 引入了一个未经评审的架构级改动（改变本地路径的输出契约）。

**原则**:
- 一个提交只解决一个问题
- 涉及契约变更必须汇报并获得确认

### 5.3 修复后必须验证副作用

**问题**: 改完后只验证了"能回复了"，没检查 prompt 质量。

**后果**: 垃圾内容在 prompt 里待了很久才被发现。

**原则**:
- 感知层改动 → 必须抽查 1 个 tick 的 prompt，确认上文干净
- 发送层改动 → 必须检查 `[Sender] verify` 日志，确认没发到错误窗口
- 生成层改动 → 必须检查 prompt 是否泄露隐私或混入无关记忆

### 5.4 修复协议已建立

完整流程见 [FIX_PROTOCOL.md](FIX_PROTOCOL.md)。

---

## 六、快速参考

### 启动自动模式
```bash
cd ~/wechat-mac-rpa
python3 -m src.bot.wechat_bot
```

### 关键文件
| 文件 | 说明 |
|------|------|
| `src/bot/wechat_bot.py` | L5 主循环编排（唯一入口） |
| `src/perception/smart_pipeline.py` | L3.5 智能感知管道（主力：本地预判 + qwen3.6-flash API 兜底） |
| `src/perception/vision_pipeline.py` | L3.5 纯本地 OCR 管道（备用回退） |
| `src/layout/layout_parser.py` + `src/message/extractor.py` | L3 布局解析与消息提取 |
| `src/reply/policy.py` + `src/reply/generator.py` | L4 回复策略与生成 |
| `tests/` 目录 | 各模块独立测试 |
| `docs/02-architecture/ARCHITECTURE.md` | 架构设计文档 |

### 发送消息的正确方式
```python
subprocess.run(['pbcopy'], input=text.encode('utf-8'), timeout=2)
# AppleScript: keystroke "v" using command down + return
```

---

## 七、Prompt Engineering 经验（2026-05-17 新增）

### 7.1 严禁 case-by-case 修 prompt

**反模式**: 看到一个误判截图，立刻在 prompt 里加一条特殊规则。

**后果**: 修好了 A，破坏了 B、C、D——因为没有回归测试验证。

**正解**:
- ✅ 先收集一批测试 case（覆盖各种头像类型、有/无未读、单聊/群聊）
- ✅ 跑 benchmark，看整体指标，找到系统性根因
- ✅ 从根因出发写通用规则，再跑 benchmark 验证

### 7.2 Prompt 工程必须有 benchmark

**原则**: 没有 benchmark 的 prompt 修改都是盲改。

**本次建立的 benchmark**:
```
src/tests/fixtures/unread_badge/
  case_001_tencent_news_tp/    # 服务号，有未读（true positive）
  case_003_wangqian_group_fp/  # 群聊拼贴头像，无未读（false positive）
  ...
  # 23 个 case，覆盖 single / single_icon / group_mosaic 三种头像类型
```

**Benchmark 指标**:
- Precision（精确率）：有未读识别中多少是真的有
- Recall（召回率）：真的有未读的有多少被识别出来
- 不能只盯一个 case，要看整体分布

### 7.3 语气词对模型行为影响巨大

**本次迭代记录**:

| 版本 | 未读角标规则措辞 | Precision | 失败 case |
|------|-----------------|-----------|-----------|
| v0 | 无特殊规则 | ~50% | 大量 |
| v1 | "白色/黑色数字"（含黑色） | ~50% | 大量 |
| v2 | 删除"黑色" | ~50% | 大量 |
| v3 | + "红色纯色圆形...位于头像边界之外" | 63.64% | 4 |
| v4 | + "【严禁】左侧边栏总未读数" | 77.78% | 2 |
| v5 | "【排除】..."→"【严禁】..." | 87.50% | 1 |
| v6 | "【严禁 - 违反则识别失败】...会导致整个识别结果作废" | **100%** | **0** |

**教训**: 同样的语义，措辞越严厉、后果越具体，模型遵守度越高。"【排除】"≈建议，"【严禁】"≈命令，"【严禁 - 违反则XX】"≈强约束。

### 7.4 真实 API 评测必须做，但用缓存避免重复调用

**策略**:
- 开发阶段：每次改 prompt 后跑一次 `--run-api`，结果缓存到 `api_result.json`
- 回归阶段：跑缓存版本，秒级完成
- 集成测试默认 skip，手动触发时跑真实 API

**本次代码**:
```bash
# 真实 API（约 1-2 分钟，消耗额度）
python3 src/tests/test_chat_list_unread_benchmark.py --run-api

# 缓存回归（秒级）
python3 -m pytest src/tests/test_chat_list_unread_benchmark.py -v
```

### 7.5 不要擅自添加未经用户确认的内容

**反模式**: 用户只说"删除黑色"，我顺手加了"头像外右上角""群聊拼贴头像内部不是未读角标"等自己脑补的内容。

**后果**: 用户指出"我没有这么说，这么说也不是事实"——prompt 里的每一条描述都必须是事实，不能是猜测。

**原则**: prompt 修改必须经过用户确认，哪怕只是一句话。prompt 是产品契约，不是个人随笔。

---

**更新时间**: 2026-05-17
**状态**: Prompt Engineering 流程已建立（benchmark + 缓存 + 回归测试）；未读角标识别 Precision 100%（23/23 case 通过）
