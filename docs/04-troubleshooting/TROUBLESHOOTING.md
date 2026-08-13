# Tick 异常深度 Troubleshooting

> 当你已经确定**某个具体 tick 异常**（action 不符合预期、消息丢失、昵称误判等），按本文档系统化诊断。
>
> 如果你只是想快速扫描最近 100 个 tick 的整体健康状况，先看 `TICK_INVESTIGATION_GUIDE.md`。

---

## 诊断流程图

```
发现异常 tick
    │
    ▼
┌─────────────────────┐
│ 1. 加载 tick JSON   │  ← python3 -c "d=json.load(open('...'))"
│    确认字段完整性   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 2. 对照症状速查表   │  ← 本文档 §2
│    定位问题类别     │
└─────────────────────┘
    │
    ├──→ 匹配到已知症状 → 按 §3 深度验证 → 确认根因 → 修复/记录
    │
    └──→ 未匹配 → 进入 §4 全面数据提取 → 人工分析 → 更新速查表
```

---

## 1. 加载与初步检查

### 1.1 确认 tick 文件完整

```python
import json
d = json.load(open('data/debug/tick_<timestamp>_<id>.json'))

required_fields = [
    'ocr_elements', 'layout_title_elements', 'layout_message_candidates',
    'layout_self_bubbles', 'extraction_clusters', 'extraction_messages',
    'bot_chat_name', 'bot_should_reply', 'bot_switch_reason', 'action',
    'screenshot_path'
]
missing = [f for f in required_fields if f not in d]
assert not missing, f"字段缺失: {missing}"
```

### 1.2 一键输出关键指标

```python
print(f"ocr: {len(d['ocr_elements'])} | "
      f"title: {len(d['layout_title_elements'])} | "
      f"candidates: {len(d['layout_message_candidates'])} | "
      f"self_bubbles: {len(d['layout_self_bubbles'])} | "
      f"clusters: {len(d['extraction_clusters'])} | "
      f"messages: {len(d['extraction_messages'])} | "
      f"action: {d['action']!r} | "
      f"screenshot: {'data/screenshots' in d.get('screenshot_path','')}")
```

**指标对照表：**

| ocr | title | candidates | bubbles | clusters | messages | screenshot | 含义 |
|-----|-------|-----------|---------|----------|----------|------------|------|
| 0 | 0 | 0 | 0 | >0 | 0 | /tmp/... | OCR 失败，clusters 是旧数据 |
| >0 | 0 | N | 0 | M | 0 | /tmp/... | title_y_max 过小 或 无标题元素 |
| >0 | >0 | N | K | M | 0 | /tmp/... | 消息提取逻辑失败 |
| >0 | >0 | N | K | M | >0 | data/screenshots/... | 基本正常，检查内容是否正确 |

---

## 2. 症状速查表

### 症状 A：chat_name 为空

**识别**：`bot_chat_name == ''` 且 `layout_title_elements == []`

**可能根因**：
1. `title_y_max` 配置过小（见 §3.1）
2. OCR 未识别到标题区域（截图中窗口未激活）
3. 标题被 `_is_garbage` 过滤（含 ®/©/™/QS/① 或时间戳模式）

**快速验证**：
```python
for e in d['ocr_elements']:
    if 80 <= e['center']['y'] <= 140 and e['center']['x'] > 400:
        print(f'"{e["text"]}" y={e["center"]["y"]} x={e["center"]["x"]}')
# 如果有输出但 title_elements 为空 → title_y_max 过小
# 如果无输出 → OCR 未识别到标题
```

---

### 症状 B：messages 数量明显少于预期

**识别**：`extraction_messages` 数量 < `extraction_clusters` 数量，或 messages 为空但 clusters 有数据

**可能根因**：
1. `_is_noise_candidate` 过滤过严（confidence/area 阈值）
2. `_is_avatar_noise` 过滤了昵称区域元素
3. `used_self` 消耗了所有 candidates（self_bubble 检测误报）
4. 旧代码 bug：cluster[0] 被无条件当昵称后 msg_elems 为空
5. **`input_y_min` 绝对坐标失效**：窗口尺寸变化后，消息区底部内容被误判为输入框（见 §3.4）

**快速验证**：
```python
# 检查 clusters 中是否有 assigned=True 但 messages 未增加的情况
for c in d['extraction_clusters']:
    if c['nickname_assigned'] and len(c['texts']) == 1:
        print(f"BUG: cluster {c['texts']} assigned=True 但只有一个元素")
```

---

### 症状 C：昵称误判（sender 不对）

**识别**：`extraction_messages` 中 sender 是对方消息内容的一部分（如 `sender='S' text='OK'`）

**可能根因**：
1. 旧代码：cluster[0] 在昵称区域内且 `len(cluster)>1` 时无条件当昵称
2. 新代码：`has_outside` 判断错误（cluster 中所有元素都在昵称区域内却被当昵称+消息）

**快速验证**：
```python
for m in d['extraction_messages']:
    if len(m['sender']) <= 2 and m['sender'] != '自己':
        print(f"疑似昵称误判: sender={m['sender']!r} text={m['text']!r}")
```

---

### 症状 D：action='none' 但预期应有 switch

**识别**：`action='none'`、`switch_reason='无未读项'`、但 `unread` 中有非空值

**可能根因**：
1. 唯一未读来自免回复聊天（腾讯新闻/文件传输助手）
2. `chat_list_items` 为空（左侧聊天列表解析失败）
3. 未读数对应昵称与当前聊天相同（已打开该聊天）

**快速验证**：
```python
unread = d['layout_chat_list_unread']
nicknames = d['layout_chat_list_nicknames']
for i, (u, n) in enumerate(zip(unread, nicknames)):
    if u:
        print(f"[{i}] {n!r}: unread={u!r}")
# 如果只有 "腾讯新闻" 或 "文件传输助手" → 预期行为（免回复过滤）
# 如果有其他昵称 → 检查 chat_list_items 是否为空
```

---

### 症状 E：action='none' 但预期应有 send

**识别**：`bot_should_reply=False`、但 `extraction_messages` 中有新的 OTHER 消息

**可能根因**：
1. Policy 过滤（群聊缺少 @、冷却期内、sender_type 不是 OTHER）
2. `new_messages_count=0`（session.filter_new 去重过于激进）
3. `chat_name` 为空导致 tick 提前返回

**快速验证**：
```python
print(f"new_messages={d['bot_new_messages_count']}")
print(f"should_reply={d['bot_should_reply']}")
for m in d['extraction_messages']:
    print(f"  sender_type={m['sender_type']} sender={m['sender']!r} text={m['text']!r}")
# 检查是否有 sender_type='other' 的消息
```

---

### 症状 F：OCR 为空但 clusters 有旧数据

**识别**：`ocr_elements=[]`、`candidates=[]`、但 `clusters>0`

**根因**：截图失败（WeChat 未就绪/最小化/需扫码），但 extractor 使用了上一 tick 的缓存数据

**处理**：非代码 bug，检查 WeChat 窗口状态

---

### 症状 G：screenshot_path 指向 /tmp 而非 data/screenshots

**识别**：`screenshot_path` 包含 `/tmp/wechat_capture` 而不是 `data/screenshots/`

**根因**：
1. **旧代码**：WindowCapture 使用固定 `/tmp/wechat_capture.png`，每次覆盖旧文件；Bot 保存后未回写路径
2. **已修复后仍出现**：Bot 保存截图时抛异常，路径未更新

**影响**：
- 无法根据 tick JSON 直接找到对应截图
- `/tmp` 下旧截图已被覆盖，彻底丢失

**快速验证**：
```python
sp = d.get('screenshot_path', '')
if '/tmp/wechat_capture' in sp:
    print(f"旧路径未更新: {sp}")
    # 尝试根据 tick 时间戳在 data/screenshots/ 中查找
    import glob, os
    tick_ts = Path(path).stem.split('_')[1]  # tick_2026-04-19T09-05-32.xxx
    candidates = glob.glob(f"data/screenshots/*{tick_ts.replace('-','').replace(':','')}*.png")
    print(f"可能匹配的截图: {candidates}")
elif 'data/screenshots' in sp and os.path.exists(sp):
    print(f"✅ 路径正确且存在: {sp}")
else:
    print(f"⚠️ 路径异常或文件不存在: {sp}")
```

### 症状 H：微信已打开但截图失败、隐藏或提示禁止捕获

**识别**：微信已登录且窗口未最小化，但 Bot 持续报告“未能获取微信窗口画面”，或截图时微信窗口消失、系统提示禁止捕获。

**排查**：
1. 确认“屏幕录制”权限授予的是**实际启动 Bot 的应用**，而不只是常用终端。
2. 退出由 Codex、Claude Code 等自动化开发环境启动的进程，改用已授权的普通 Terminal/iTerm 或 LaunchAgent 启动。
3. 在同一启动宿主中运行 `screencapture -x /tmp/wechat-test.png`。
4. 如果系统截图也失败，先排查 macOS 权限、启动宿主和微信运行环境，不要修改项目截图逻辑。

提交问题时请附上 macOS 版本、微信版本、启动命令和实际启动宿主。此症状也可能与微信风控有关，但仅凭截图失败无法确认。

---

## 3. 深度验证方法

### 3.1 验证 title_y_max

```python
# 找出所有可能的标题元素（右侧 + y<150）
candidates = [e for e in d['ocr_elements']
              if e['center']['x'] > 400 and e['center']['y'] < 150]
for e in sorted(candidates, key=lambda x: x['center']['y']):
    print(f'"{e["text"]}" y={e["center"]["y"]}')

# 当前配置值
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
print(f"当前 title_y_max = {PROFILE_WECHAT_MAC_1760X1280.title_y_max}")
# 如果最大标题 y > title_y_max → 需要增大配置
```

### 3.2 验证消息提取完整链路

```python
# 从 OCR → Layout → Extraction 逐步验证
print("=== OCR 右侧元素 ===")
for e in d['ocr_elements']:
    if e['center']['x'] >= 480:
        print(f'"{e["text"]}" x={e["center"]["x"]} y={e["center"]["y"]}')

print("\n=== Layout candidates ===")
for c in d['layout_message_candidates']:
    print(f'"{c["text"]}" x={c["cx"]} y={c["cy"]}')

print("\n=== Self bubbles ===")
for b in d['layout_self_bubbles']:
    print(f"x={b['x']}-{b['x']+b['w']} y={b['y']}-{b['y']+b['h']}")

print("\n=== Clusters → Messages 映射 ===")
for i, c in enumerate(d['extraction_clusters']):
    print(f"[{i}] texts={c['texts']} assigned={c['nickname_assigned']}")
    # 找出对应的消息
    msgs = [m for m in d['extraction_messages']
            if m['text'] == ' '.join(c['texts'][1:] if c['nickname_assigned'] else c['texts'])]
    if msgs:
        print(f"    → message: sender={msgs[0]['sender']!r} text={msgs[0]['text']!r}")
    else:
        print(f"    → 无对应消息 (BUG)")
```

### 3.4 验证 input_y_min（窗口尺寸适配）

```python
# 检查消息区底部是否有内容被误判为输入框
print("=== input_elements（被过滤为输入框的内容）===")
for e in d['layout_input_elements']:
    print(f'"{e["text"]}" y={e["y"]}')

print("\n=== message_candidates（消息候选）===")
for c in d['layout_message_candidates']:
    print(f'"{c["text"]}" y={c["cy"]}')

# 检查：截图高度 vs Profile 窗口高度
from pathlib import Path
from PIL import Image
img = Image.open(Path(d['screenshot_path']))
print(f"\n截图高度: {img.height}")
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
print(f"Profile 窗口高度: {PROFILE_WECHAT_MAC_1760X1280.window_height}")
print(f"Profile input_y_min: {PROFILE_WECHAT_MAC_1760X1280.input_y_min}")
print(f"按高度比例应调整至: {int(PROFILE_WECHAT_MAC_1760X1280.input_y_min * img.height / PROFILE_WECHAT_MAC_1760X1280.window_height)}")

# 若截图高度显著大于 Profile 窗口高度，且 input_elements 中包含明显是消息的内容
# → input_y_min 绝对坐标失效，需要改为动态计算
```

### 3.3 验证 Bot 决策逻辑

```python
print("=== Bot 决策链 ===")
print(f"chat_name={d['bot_chat_name']!r}")
print(f"new_messages={d['bot_new_messages_count']}")
print(f"should_reply={d['bot_should_reply']}")
print(f"switch_reason={d['bot_switch_reason']!r}")
print(f"switch_target={d['bot_switch_target']!r}")
print(f"action={d['action']!r}")
print(f"action_error={d['action_error']!r}")

# 检查 switch 过滤链
unread = d['layout_chat_list_unread']
nicknames = d['layout_chat_list_nicknames']
print(f"\n=== 聊天列表 ===")
for u, n in zip(unread, nicknames):
    flag = "✅ 有效未读" if u and n not in {"腾讯新闻", "文件传输助手"} else "❌ 免回复/无未读"
    print(f"  {n!r}: unread={u!r} {flag}")
```

---

## 4. 未知异常：全面数据提取

如果速查表未匹配，运行以下脚本提取 tick 的全部关键数据：

```python
import json
from pathlib import Path

def analyze_tick(path: str):
    d = json.load(open(path))
    print(f"=== {Path(path).name} ===")

    # Layer 0: OCR
    print(f"\n[OCR] elements={len(d['ocr_elements'])}")
    left = [e for e in d['ocr_elements'] if e['center']['x'] < 400]
    right = [e for e in d['ocr_elements'] if e['center']['x'] >= 400]
    print(f"  左侧: {len(left)} 右侧: {len(right)}")

    # Layer 1: Layout
    print(f"\n[Layout]")
    print(f"  title: {len(d['layout_title_elements'])} elements")
    for e in d['layout_title_elements']:
        print(f"    \"{e['text']}\" y={e['y']}")
    print(f"  candidates: {len(d['layout_message_candidates'])}")
    print(f"  self_bubbles: {len(d['layout_self_bubbles'])}")
    print(f"  chat_list: {len(d['layout_chat_list_groups'])} groups")
    print(f"    nicknames={d['layout_chat_list_nicknames']}")
    print(f"    unread={d['layout_chat_list_unread']}")

    # Layer 2: Extraction
    print(f"\n[Extraction]")
    print(f"  clusters: {len(d['extraction_clusters'])}")
    for c in d['extraction_clusters']:
        print(f"    texts={c['texts']} top_x={c['top_x']} in_nick={c['in_nick_range']} assigned={c['nickname_assigned']}")
    print(f"  messages: {len(d['extraction_messages'])}")
    for m in d['extraction_messages']:
        print(f"    sender={m['sender']!r} type={m['sender_type']} text={m['text']!r}")

    # Layer 3: Bot
    print(f"\n[Bot]")
    print(f"  chat_name={d['bot_chat_name']!r}")
    print(f"  new_messages={d['bot_new_messages_count']}")
    print(f"  should_reply={d['bot_should_reply']}")
    print(f"  switch_reason={d['bot_switch_reason']!r}")
    print(f"  switch_target={d['bot_switch_target']!r}")
    print(f"  action={d['action']!r}")
    print(f"  screenshot_path={d.get('screenshot_path','')!r}")

analyze_tick("data/debug/tick_<timestamp>_<id>.json")
```

---

## 5. 根因确认后

1. **如果是配置问题**（如 title_y_max 过小）→ 修改 `profile.py` → 添加回归测试 → 提交
2. **如果是代码 bug**（如 cluster[0] 误判）→ 修复代码 → 添加单元测试 → 提交
3. **如果是预期行为**（如腾讯新闻免回复过滤）→ 更新本文档速查表，减少同类误报
4. **如果是未知问题** → 保存 fixture 截图到 `tests/fixtures/` → 添加回归测试 → 更新速查表
