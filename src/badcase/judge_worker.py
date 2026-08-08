#!/usr/bin/env python3
"""JudgeWorker - 异步 badcase 判定，支持查证反思"""

import json
import logging
import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.memory.engine import MemoryEngine
from src.tools.builtin_tools import _web_search

from .case_db import get_db
from .case_generator import CaseGenerator

_logger = logging.getLogger("src.badcase.judge_worker")

PROJECT_ROOT = Path(__file__).parent.parent.parent

_JUDGE_PROMPT_TEMPLATE = """【你只允许输出纯 JSON，不要有任何分析、思考、解释或自然语言。如果违反，判定无效。】

你是 Bot QA 审计员。逐项检查，每项独立判断。

## 时间锚点
当前时间: {current_time}。消息时间戳相对于此时间。

## Bot 本轮回复
（注意：Bot 的回复可能包含多条独立短消息，之间用" | "分隔，每条消息是单独发送的）
{bot_reply}

## Bot 调用的工具及返回结果（事实来源，截断已放宽到 {tool_max} 字）
{tool_calls}

## Bot 看到的完整上下文（最近 {msg_max} 条消息，截断 {ctx_max} 字/条）
{full_messages}

## Bot 的系统提示词（行为规范）
{system_prompt}

## Bot 看到的用户提示词（含 wiki、历史消息、未读消息等完整上下文）
{user_prompt}

## 9 项独立检查

### 检查1：幻觉
Bot 说了什么具体事实（人名/数字/事件/毕业院校/比分），在【工具返回】、【上下文】或【系统提示词中的wiki】中能找到依据吗？
- **系统提示词中的 wiki 信息也是有效依据**，不能只看工具返回
- 依据充足（工具返回/上下文/wiki中有明确记录）→ false
- 工具返回被截断，但 wiki 或上下文中有明确记录 → false
- search_memory 返回被截断，但系统提示词的 wiki 中有该信息 → false
- 完全没有依据（工具/wiki/上下文均无）→ true，detail 写"无依据事实：XXX"
- **Bot 自己说"记岔了"不能豁免，仍然要判 issue**

### 检查2：时间误判
消息有时间标签吗？Bot 忽略了时间吗？Bot 说的"今天/现在"真的是今天/现在吗？
- 昨晚发的"好困" → Bot 说"通宵了" → true
- 凌晨/周末 → Bot 说"今天跌了X"（股市不开盘） → 调用 verify_tool 查当前时间
- 几小时前的消息 → Bot 当现在发的 → true
- 没有时间标签或正确理解 → false

### 检查3：过度回复 / 漏回复
用户最后一条需要回复吗？
- OK/好的/收到/嗯/纯表情/纯图片 → Bot 还回复了 → true（过度）
- Bot 在历史里已经回复过**这条完全相同的消息** → 又回了一遍 → true（过度，完全重复扣20-30分）
- **用户发了新问题（不是已回复过的旧消息）→ Bot 空回复 → true（漏回复）**
- **用户反问（带问号/质疑语调）→ Bot 不回复 → true（漏回复）**
- 部分重复或轻微过度 → issue=true，detail说明程度（部分重复扣5-10分）
- 正常需要回且回了 → false
- 不需要回且没回 → false

### 检查4：信息不完整
工具返回有截断提示吗？Bot 把不完整信息当完整结论了？
- 工具返回提示"已截断"，且系统提示词的 wiki / 上下文中**没有**该信息的明确记录 → Bot 断言了具体结论 → true
- 工具返回截断，但 wiki 或上下文中有明确记录 → false（Bot 有其它可靠来源）
- 工具返回完整 → false

### 检查5：答非所问
用户问什么话题？Bot 答什么话题？一致吗？
- 用户问A，Bot 答无关的B → true（如用户问茅台，Bot 答拼多多）
- 用户纠正/质疑 Bot → Bot 没有调用工具查证而是随口回应 → true
- 回答切题 → false

### 检查6：无信息增量 / 重复上文（不触发 badcase，只影响简洁度评分）
Bot 的回复相比上文和工具返回，带来了新的信息吗？
- **啰嗦的定义：没有带来信息增量，只是重复上文已经说过的话**
- **三句话表达同一个意思** → true（如"哈哈哈我是真人啊""就是本g神本人""不是不爱说话那个bot"——三句都在说"我是真人不是bot"）
- **上文已有同类表达，Bot 又重复一遍** → true（如上文已经有人祝生日快乐，Bot 又发一遍生日祝福；上文已经安慰过，Bot 又用不同措辞重复安慰）
- **确认类回复堆叠** → true（如"这是《后来》吧""刘若英那个经典""栀子花白花瓣..."——前两句都在确认歌名，只有第三句有增量；更好的方式是直接说"《后来》，栀子花白花瓣落在我蓝色百褶裙上"）
- Bot 把用户的问题重复一遍再回答 → true（如用户问"天气怎么样"，Bot 答"天气怎么样？今天挺热的"）
- Bot 反复使用无意义套话（"氛围很好""感觉不错"）→ true
- Bot 把上文已知信息复述一遍，没有新增内容 → true
- **Bot 回复以复述上文已知信息为开头或主体，再附加少量新内容 → true**（如用户说"富比家存款2000万还有3套房"，Bot 回复"富比家2000万存款加3套房，我要是她的话也是吃喝不愁了"——前半句完全复述用户刚说的话，无信息增量；更好的方式是直接说"我要是她的话也是吃喝不愁了"）
- 回复有实质新信息、新观点或新情感，且**没有复述上文已知信息** → false

### 检查7：格式/风格问题（bad_style）
- Bot 把多条内容用换行符 `\\n` 合并成一条长消息 → true（微信聊天不会这样换行，应该分多条发）
- 其他明显格式差异 → true
- 轻微风格问题 → false

### 检查8：工具调用
- 该调用工具时没调用 → true
- 不该调用时乱调用 → true
- 工具调用正确 → false

### 检查9：自相矛盾
Bot 本轮回复与上下文中的自己矛盾吗？
- 前面说自己做过X，后面否认 → true
- 前面夸自己，后面说自己傻 → 自嘲不能推翻前面的自我评价，但实质性矛盾 → true
- 无矛盾 → false

### 检查10：事实错误（非幻觉，有依据但理解错了）
Bot 引用了工具返回或上下文中的信息，但理解/推断错误？
- 有依据但结论错误 → true（如数字算错、关系搞错）
- 完全无依据 → 归入检查1（幻觉），不重复计
- 理解正确 → false

## 8 维度评分细则（满分100）

权重分配：
- 幻觉控制 25%
- 上下文理解 15%
- 回复必要性 15%
- 简洁度 10%
- 个性一致性 10%
- 时间推理 5%
- 信息准确性 5%
- 亮点加分项 15%

### 维度1：幻觉控制（25分）
四级评分流程：
1. **第一步**：识别 Bot 的所有事实断言（人名、数字、事件、时间、属性等）
2. **第二步**：逐一在【工具返回】和【上下文】中寻找原始依据
3. **第三步**：区分"可直接查证的事实"与"不可查证的推断"
   - 可直接查证的事实（如具体比分、股价、人名）必须有明确依据
   - 不可查证的推断（如"看起来很累"、"应该没问题"）需判断是否合理
4. **第四步**：评估查证义务
   - Bot 有工具可用但未调用查证 → 扣分
   - 不确定语气（"可能"、"好像"）不能替代查证义务

**评分标准**：
- 100分：所有事实断言均有明确依据，无幻觉
- 75分：少量轻微推断，不影响核心结论
- 50分：有不确定表述但未核实，或轻微幻觉
- 25分：明显幻觉，但非核心事实
- 0分：核心事实 hallucination

**豁免规则**：
- 亲昵称呼（"亲爱的"、"宝子"）不视为幻觉
- 修辞表达（"美呆了"、"绝绝子"）不视为幻觉
- 主观感受表达不视为幻觉

### 维度2：上下文理解（15分）
- 是否正确理解用户意图和对话脉络
- 是否正确处理多轮对话的指代和省略
- 是否正确区分不同说话人（群聊场景）

### 维度3：回复必要性（15分）
- 是否该回则回、不该回则不回
- 是否避免过度回复和漏回复

### 维度4：简洁度（10分）
- **核心标准：是否有信息增量**。啰嗦的定义是"没有带来信息增量，只是重复上文已经说过的话"
- Bot 重复用户问题、复述上文已知信息、反复使用无意义套话 → 低分（0~40）
- **Bot 回复包含复述上文已知信息（即使后面有新观点）→ 中低分（40~70），复述比例越高分越低**（如用户刚说的事实 Bot 又重复一遍作为回复主体，再附加少量评论）
- 回复有新的信息、观点或情感表达，且**没有复述上文已知信息** → 高分（80~100）
- 简洁不等于短，长但有增量也可以高分；短但全是废话也可以低分

### 维度5：个性一致性（10分）
- 是否符合 Bot 设定的人设
- 前后言论是否一致（实质性矛盾扣分，自嘲豁免）

### 维度6：时间推理（5分）
- 是否正确理解消息的时间属性
- "今天"、"现在"等时间词是否准确

### 维度7：信息准确性（5分）
- 有依据的事实是否理解正确
- 数字、逻辑关系是否准确

### 维度8：亮点加分项（15分）
**基础分 30 分**（所有回复默认有 30 分基础分，再根据实际情况加减）

加分项：
- 精准引用上下文细节 +5~10
- 幽默或高情商回复 +5~10
- 主动提供有价值的额外信息 +5~10
- 处理复杂/模糊问题的能力 +5~10

扣分项：
- 回复平庸无特色 -5~10
- 机械感明显 -5~10
- 错失展示个性/才华/文艺/格局的机会 -5~15（如用户明显给了展示才华的台阶——发诗句邀对诗、夸你写得好、聊文学/创作等——Bot 只说"我试试""好好发挥""早呀"等废话，没有真的展示才华）

## 评分计算

```
overall_score_raw = round(
    幻觉控制 × 0.25 +
    上下文理解 × 0.15 +
    回复必要性 × 0.15 +
    简洁度 × 0.10 +
    个性一致性 × 0.10 +
    时间推理 × 0.05 +
    信息准确性 × 0.05 +
    亮点加分项 × 0.15
)

checks_issue_count = 统计 checks 中 issue=true 的项数
overall_score = max(0, overall_score_raw - checks_issue_count × 12)
```

## 输出（纯 JSON）

```json
{{
  "checks": {{
    "幻觉": {{"issue": true/false, "detail": "证据"}},
    "时间误判": {{"issue": true/false, "detail": "证据"}},
    "过度回复": {{"issue": true/false, "detail": "证据"}},
    "信息不完整": {{"issue": true/false, "detail": "证据"}},
    "答非所问": {{"issue": true/false, "detail": "证据"}},
    "无信息增量": {{"issue": true/false, "detail": "证据"}},
    "格式问题": {{"issue": true/false, "detail": "证据"}},
    "工具调用": {{"issue": true/false, "detail": "证据"}},
    "自相矛盾": {{"issue": true/false, "detail": "证据"}},
    "事实错误": {{"issue": true/false, "detail": "证据"}}
  }},
  "dimensions": {{
    "幻觉控制": {{"score": 0~100, "comment": "评价"}},
    "上下文理解": {{"score": 0~100, "comment": "评价"}},
    "回复必要性": {{"score": 0~100, "comment": "评价"}},
    "简洁度": {{"score": 0~100, "comment": "评价"}},
    "个性一致性": {{"score": 0~100, "comment": "评价"}},
    "时间推理": {{"score": 0~100, "comment": "评价"}},
    "信息准确性": {{"score": 0~100, "comment": "评价"}},
    "亮点加分项": {{"score": 0~100, "comment": "评价"}}
  }},
  "overall_score": 0~100,
  "is_badcase": true/false,
  "badcase_type": "hallucination/time_misread/over_reply/info_incomplete/wrong_topic/bad_format/missing_tool_call/contradiction/wrong_fact/none",
  "confidence": 0.0~1.0,
  "reason": "出问题的检查项 + 证据",
  "expected_behavior": "应该怎么做",
  "verify_tool": {{"name": "search_memory", "query": "搜索词"}} 或 {{"name": "web_search", "query": "搜索词"}} 或 {{"name": "get_current_time", "query": ""}} 或 null
}}
```

## 判定规则

- `is_badcase = (任一 checks 中 issue == true) OR (overall_score < 60)`
- 幻觉控制 ≤ 30 分 → 必为 badcase
- 上下文理解 ≤ 30 分 → 必为 badcase

## 查证机制（verify_tool）
如果你怀疑 Bot 的某个回复是幻觉或编造，但工具返回结果似乎被截断（信息不足），**必须**设置 verify_tool 让系统查证。不要因为"不确定"就判 false。
- 当 Bot 说了具体的人名/数字/事件但你看不到依据 → 调用 search_memory 验证
- 当 Bot 回复"不知道"但你觉得搜索可能找到 → 调用 search_memory 验证
- 不要假设截断的信息里有依据，去验证它
- 当 Bot 说"今天未开盘"但你不知道是否真的未开盘 → 调用 get_current_time 或 web_search 验证
- 当 Bot 说了人名/数字但你从截断中看不到 → 调用 search_memory 验证
- verify_tool 设为 null 只有在你100%确定时才允许
"""


def _strip_output_format_section(prompt: str) -> str:
    """删除 prompt 中"### N. 输出格式"到下一个"### "之间的内容。

    用字符串操作替代正则（Rule 3.4）。prompt 来自已存储的生产 Bot system
    prompt，judge 作为消费者无法回到生成阶段加开关，属合理的消费端裁剪。
    """
    marker = "### "
    search_from = 0
    while True:
        idx = prompt.find(marker, search_from)
        if idx == -1:
            break
        after_marker = prompt[idx + len(marker):idx + len(marker) + 30]
        # 严格匹配 "数字. 输出格式"（数字后必须紧跟 ". 输出格式"）
        if after_marker and after_marker[0].isdigit() and after_marker[1:].startswith(". 输出格式"):
            # 找到输出格式 section：删除从 idx 到下一个 ### 或结尾
            next_section = prompt.find(marker, idx + len(marker))
            if next_section == -1:
                prompt = prompt[:idx]
            else:
                prompt = prompt[:idx] + prompt[next_section:]
            search_from = idx  # idx 位置现在是下一个 section 的 marker
        else:
            search_from = idx + len(marker)
    return prompt


def _empty_judge_result(reason: str = "") -> dict:
    return {
        "is_badcase": False, "badcase_type": "none", "severity": "P2",
        "confidence": 0.0, "auto_commit": False, "overall_score": 0,
        "dimensions": {}, "reason": reason, "expected_behavior": "",
        "_verify_tool": None,
    }


def _get_qwen_client():
    from src.utils.qwen_client import QwenClient
    return QwenClient(model="deepseek-v4-flash")


class JudgeWorker:
    JUDGE_MAX_CONTEXT = 99999     # 不截断消息内容
    JUDGE_MAX_TOOL_RESULT = 99999 # 不截断工具结果
    JUDGE_MAX_MESSAGES = 99       # 全量消息

    def __init__(self, model: str = "deepseek-v4-flash", use_fewshot: bool = True):
        self.client = _get_qwen_client()
        self.queue: queue.Queue = queue.Queue()
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self.case_generator = CaseGenerator()
        self.use_fewshot = use_fewshot
        self._pending_dir = PROJECT_ROOT / "data" / "review_drafts" / "pending"
        self._committed_dir = PROJECT_ROOT / "data" / "review_drafts" / "committed"
        self._dismissed_dir = PROJECT_ROOT / "data" / "review_drafts" / "dismissed"
        for d in [self._pending_dir, self._committed_dir, self._dismissed_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._start_consumer()

    def _start_consumer(self):
        if self._running:
            return
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._consumer_thread.start()

    def submit(self, tick_data: dict):
        self.queue.put(tick_data)

    def _consume_loop(self):
        while self._running:
            try:
                tick_data = self.queue.get(timeout=1)
                self._process_one(tick_data)
            except queue.Empty:
                continue
            except Exception as e:
                _logger.error("[Judge] consume error: %s", e)

    def _process_one(self, tick_data: dict):
        tick_id = tick_data.get("tick_id", 0)
        _logger.info("[Judge] processing tick %s", tick_id)
        judge_result = self._judge(tick_data)

        # 更新 tick_log
        try:
            import json as _json
            conn = get_db()._get_conn()
            conn.execute("""UPDATE tick_log SET judge_score=?, judge_is_badcase=?, judge_badcase_type=?,
                judge_dimensions_json=?, judge_reason=?, judge_raw_response=? WHERE tick_id=?""", (
                judge_result.get('overall_score', 0),
                1 if judge_result.get('is_badcase') else 0,
                judge_result.get('badcase_type', ''),
                _json.dumps(judge_result.get('dimensions', {}), ensure_ascii=False),
                judge_result.get('reason', ''),
                judge_result.get('_raw_response', '') or '',
                tick_id,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            _logger.warning("judge db write failed: %s", e)

        if not judge_result.get("is_badcase"):
            return
        draft = self._build_draft(tick_data, judge_result)
        generated = self.case_generator.generate(draft)
        draft["generated_case"] = generated
        if self._should_auto_commit(judge_result):
            self._auto_commit(draft)
        else:
            self._save_pending(draft)

    def _judge(self, tick_data: dict) -> dict:
        prompt = self._build_judge_prompt(tick_data)
        raw = self.client.chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=20000, timeout=60, response_format={"type": "json_object"})
        result = self._parse_judge_response(raw)
        result["_raw_response"] = raw

        # 查证反思（支持 search_memory / web_search / get_current_time）
        verify = result.get("_verify_tool")
        if isinstance(verify, dict) and verify.get("name") and verify.get("query") is not None:
            tool_name = verify["name"]
            tool_query = (verify.get("query") or "").replace("'", "\\'")
            _logger.info("[Judge] 查证: %s(%s)", tool_name, tool_query)
            try:
                if tool_name == "search_memory":
                    m = MemoryEngine()
                    tool_result = str(m.search_keyword(tool_query))[:5000]
                elif tool_name == "web_search":
                    tool_result = str(_web_search(query=tool_query))[:3000]
                elif tool_name == "get_current_time":
                    from datetime import datetime
                    tool_result = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
                else:
                    tool_result = f"未知工具: {tool_name}"

                verify_prompt = f"""查证结果（{tool_name}("{tool_query}")）：
{tool_result}

基于查证结果重新判断。如果结果支持 Bot 的回复 → is_badcase=false。输出格式同上。"""
                raw2 = self.client.chat(
                    messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}, {"role": "user", "content": verify_prompt}],
                    temperature=0.3, max_tokens=20000, timeout=60, response_format={"type": "json_object"},
                )
                result = self._parse_judge_response(raw2)
                result["_raw_response"] = raw2
            except Exception as e:
                _logger.warning("[Judge] 查证失败: %s", e)
        return result

    def _build_judge_prompt(self, tick_data: dict) -> str:
        sp = tick_data.get("full_system_prompt", "")
        llm_messages = tick_data.get("full_llm_messages", [])
        bot_reply = tick_data.get("bot_reply_text", "") or tick_data.get("reply_text", "")

        # 时间锚点：优先 tick 发生时间，fallback 当前
        tick_ts = tick_data.get("created_at", "")
        if tick_ts:
            try:
                now = datetime.fromisoformat(tick_ts)
            except Exception as e:
                _logger.warning("parse tick timestamp failed: %s", e)
                now = datetime.now()
        else:
            now = datetime.now()
        current_time = now.isoformat()

        tc = tick_data.get("tool_calls", [])
        if not tc:
            trace = tick_data.get("reply_generation_trace", [])
            for t in trace:
                if t.get("type") == "llm_response":
                    tc2 = t.get("tool_calls", [])
                    if tc2:
                        tc = tc2
                        break

        tool_results_raw = tick_data.get("tool_results_json", "")
        if tool_results_raw:
            try:
                tc_with_results = json.loads(tool_results_raw)
                for x in tc_with_results:
                    if isinstance(x.get("result"), str) and len(x["result"]) > self.JUDGE_MAX_TOOL_RESULT:
                        x["result"] = x["result"][:self.JUDGE_MAX_TOOL_RESULT] + "..."
            except Exception:
                tc_with_results = tc
        else:
            tc_with_results = tc
        tool_calls_text = json.dumps(tc_with_results, ensure_ascii=False, indent=2)

        truncated = []
        for m in (llm_messages or [])[-self.JUDGE_MAX_MESSAGES:]:
            cm = dict(m)
            if isinstance(cm.get("content"), str) and len(cm["content"]) > self.JUDGE_MAX_CONTEXT:
                cm["content"] = cm["content"][:self.JUDGE_MAX_CONTEXT] + "..."
            if "tool_calls" in cm and cm["tool_calls"]:
                cm["tool_calls"] = [{"id": x.get("id"), "name": x.get("function", {}).get("name", "?")} for x in cm["tool_calls"]]
            truncated.append(cm)
        msgs_text = json.dumps(truncated, ensure_ascii=False, indent=2) if truncated else "(无)"

        # 过滤 system prompt 输出格式 section（删除"### N. 输出格式"到下一个"### "之间内容）。
        # sp 来自 tick_log 存储的生产 Bot system prompt（已生成），judge 作为消费者无法
        # 回到生成阶段加开关，只能用字符串处理消费端裁剪（非 Rule 1.6 的事后打补丁）。
        sp = _strip_output_format_section(sp)

        # 注入人工标注的 few-shot（从 tick_log 读取最近的人工标注 case）
        human_shots = self._load_human_fewshot() if self.use_fewshot else ""

        return _JUDGE_PROMPT_TEMPLATE.format(
            current_time=current_time,
            tool_max=self.JUDGE_MAX_TOOL_RESULT, msg_max=self.JUDGE_MAX_MESSAGES, ctx_max=self.JUDGE_MAX_CONTEXT,
            system_prompt=sp, user_prompt=tick_data.get("full_user_prompt", ""), full_messages=msgs_text, tool_calls=tool_calls_text, bot_reply=bot_reply,
            human_fewshot=human_shots,
        )

    def _load_human_fewshot(self) -> str:
        """加载人工标注的 few-shot 示例。"""
        try:
            conn = get_db()._get_conn()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT tick_id, chat_name, replies_sent_json, human_notes, human_badcase_type "
                "FROM tick_log WHERE human_is_badcase=1 ORDER BY id DESC LIMIT 5"
            ).fetchall()
            conn.close()
            if not rows:
                return ""
            shots = []
            for r in rows:
                d = dict(r)
                notes = (d.get("human_notes") or "").strip()
                btype = (d.get("human_badcase_type") or "").strip()
                if not notes:
                    continue
                # 提取判定逻辑而非列举案例
                logic = notes[:120]
                shots.append(f"人类判定为 {btype}：{logic}")
            if not shots:
                return ""
            return "人类 QA 判定过以下 badcase，学习他们的判断思路：\n" + "\n".join(f"- {s}" for s in shots)
        except Exception:
            return ""

    def _parse_judge_response(self, raw: str) -> dict:
        if not raw:
            return _empty_judge_result("空返回")
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s:e + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _empty_judge_result("JSON 解析失败")

        checks = data.get("checks", {})
        dims_raw = data.get("dimensions", {})

        # 统计 checks issue 数量（仅核心 checks 触发强制 is_badcase）
        CORE_CHECKS = {"幻觉", "事实错误", "时间误判", "自相矛盾", "工具调用", "答非所问"}
        checks_issue_count = sum(
            1 for name, c in checks.items()
            if isinstance(c, dict) and c.get("issue") is True and name in CORE_CHECKS
        )

        # 处理 dimensions
        if dims_raw:
            dims = {}
            for name in ["幻觉控制","上下文理解","回复必要性","简洁度","个性一致性","时间推理","信息准确性","亮点加分项"]:
                d = dims_raw.get(name, {})
                dims[name] = {"score": int(d.get("score", 50)), "comment": d.get("comment", "")}

            # 强制降分：checks 中幻觉/事实错误/时间误判 issue=true 且幻觉控制>50，强制降到40
            if checks_issue_count > 0:
                if (checks.get("幻觉", {}).get("issue") is True or
                    checks.get("事实错误", {}).get("issue") is True or
                    checks.get("时间误判", {}).get("issue") is True):
                    if dims["幻觉控制"]["score"] > 50:
                        dims["幻觉控制"]["score"] = 40

            # 计算 overall_score_raw
            overall_score_raw = round(
                dims["幻觉控制"]["score"] * 0.25 +
                dims["上下文理解"]["score"] * 0.15 +
                dims["回复必要性"]["score"] * 0.15 +
                dims["简洁度"]["score"] * 0.10 +
                dims["个性一致性"]["score"] * 0.10 +
                dims["时间推理"]["score"] * 0.05 +
                dims["信息准确性"]["score"] * 0.05 +
                dims["亮点加分项"]["score"] * 0.15
            )
            # checks 惩罚
            overall_score = max(0, overall_score_raw - checks_issue_count * 12)
        elif checks:
            # fallback: 从 checks 推导 dimensions
            dims = {}
            # 幻觉/事实错误/时间误判/自相矛盾 都映射到幻觉控制=25
            hallucination_issue = (
                checks.get("幻觉", {}).get("issue") is True or
                checks.get("事实错误", {}).get("issue") is True or
                checks.get("时间误判", {}).get("issue") is True or
                checks.get("自相矛盾", {}).get("issue") is True
            )
            dims["幻觉控制"] = {"score": 25 if hallucination_issue else 75, "comment": ""}
            dims["上下文理解"] = {"score": 25 if checks.get("答非所问", {}).get("issue") is True else 75, "comment": ""}
            dims["回复必要性"] = {"score": 25 if checks.get("过度回复", {}).get("issue") is True else 75, "comment": ""}
            dims["简洁度"] = {"score": 25 if checks.get("格式问题", {}).get("issue") is True else 75, "comment": ""}
            dims["个性一致性"] = {"score": 25 if checks.get("自相矛盾", {}).get("issue") is True else 75, "comment": ""}
            dims["时间推理"] = {"score": 25 if checks.get("时间误判", {}).get("issue") is True else 75, "comment": ""}
            dims["信息准确性"] = {"score": 25 if (checks.get("信息不完整", {}).get("issue") is True or checks.get("事实错误", {}).get("issue") is True) else 75, "comment": ""}
            dims["亮点加分项"] = {"score": 30, "comment": ""}

            overall_score_raw = round(
                dims["幻觉控制"]["score"] * 0.25 +
                dims["上下文理解"]["score"] * 0.15 +
                dims["回复必要性"]["score"] * 0.15 +
                dims["简洁度"]["score"] * 0.10 +
                dims["个性一致性"]["score"] * 0.10 +
                dims["时间推理"]["score"] * 0.05 +
                dims["信息准确性"]["score"] * 0.05 +
                dims["亮点加分项"]["score"] * 0.15
            )
            overall_score = max(0, overall_score_raw - checks_issue_count * 12)
        else:
            # issues fallback（兼容旧格式）
            dims = {}
            for name in ["幻觉控制", "上下文理解", "回复必要性", "简洁度",
                         "个性一致性", "时间推理", "信息准确性", "亮点加分项"]:
                dims[name] = {"score": 50, "comment": ""}
            overall_score = 50
            checks_issue_count = 0

        # is_badcase 判定
        # 任何 checks issue=true 即触发，或 overall<60，或幻觉控制≤30，或上下文理解≤30
        is_badcase = (
            checks_issue_count > 0 or
            overall_score < 60 or
            dims.get("幻觉控制", {}).get("score", 100) <= 30 or
            dims.get("上下文理解", {}).get("score", 100) <= 30
        )

        return {
            "is_badcase": is_badcase,
            "badcase_type": data.get("badcase_type", "none"),
            "severity": data.get("severity", "P2"),
            "confidence": float(data.get("confidence", 0.0)),
            "auto_commit": bool(data.get("auto_commit")),
            "overall_score": overall_score,
            "dimensions": dims,
            "checks": checks,
            "reason": data.get("reason", ""),
            "expected_behavior": data.get("expected_behavior", ""),
            "_verify_tool": data.get("verify_tool"),
        }

    def _build_draft(self, tick_data: dict, judge_result: dict) -> dict:
        return {
            "draft_id": f"tick_{tick_data.get('tick_id',0)}_{datetime.now().isoformat()}",
            "tick_id": tick_data.get("tick_id", 0),
            "timestamp": datetime.now().isoformat(),
            "chat_name": tick_data.get("chat_name", ""),
            "status": "pending",
            "judge_result": judge_result,
            "conversation": tick_data.get("session_input_messages", []),
            "bot_reply": tick_data.get("bot_reply_text", ""),
            "tool_calls": tick_data.get("tool_calls", []),
            "full_system_prompt": tick_data.get("full_system_prompt", ""),
            "full_user_prompt": tick_data.get("full_user_prompt", ""),
            "full_tools_context": tick_data.get("full_tools_context", ""),
            "full_llm_messages": tick_data.get("full_llm_messages", []),
        }

    def _should_auto_commit(self, judge_result: dict) -> bool:
        if not judge_result.get("is_badcase"):
            return False
        confidence = judge_result.get("confidence", 0)
        is_p0 = judge_result.get("severity") == "P0"
        auto = judge_result.get("auto_commit", False)
        allowed = {"missing_tool_call","redundant_tool_call","hallucination","wrong_fact","time_misread","over_reply","info_incomplete"}
        bt = judge_result.get("badcase_type","")
        return auto and confidence >= 0.9 and (is_p0 or bt in allowed)

    def _auto_commit(self, draft: dict):
        draft["status"] = "committed"
        draft["committed_at"] = datetime.now().isoformat()
        path = self._committed_dir / f"{draft['draft_id']}.json"
        path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_pending(self, draft: dict):
        path = self._pending_dir / f"{draft['draft_id']}.json"
        path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
