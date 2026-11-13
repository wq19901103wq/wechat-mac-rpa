#!/usr/bin/env python3
"""Load the fixed group-chat bad cases for the unified review page."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data/reports/private/situation_c0_group_self_context_freshness_boundary_fixed40_v1"
OUTPUT = ROOT / "data/reports/private/group_c0_badcase_taxonomy_34/index.html"

CATEGORIES = {
    "没继续本人刚建立的信息": {
        "human_5f517d0ac445": "真人会顺着现场补出人物姓名；C0重复“牛”并另造人物关系梗，没有沿用本人正在补充的信息。",
        "human_b14a4156ba70": "真人除了回答“浴场”，还延续了本人刚建立的价格和熟人调侃立场；C0只做字面回答。",
        "human_b7d90205be58": "真人延续本人正在试水、正式面试不会这么快的状态；C0改写成赶在元旦前的通用计划。",
        "human_f4337241908b": "真人顺着群友对本人的称呼，用第三人称继续自我撇清；C0只做通用否认。",
        "human_b062002557e5": "真人继续追问本人刚提出的灰度假设；C0只接“卧槽”的情绪，没有继续这个未完问题。",
        "human_0d4bb126f8cb": "真人直接给出本人掌握的级别数字；C0退回旧信息“坑满了”，没有继续回答新的级别问题。",
        "human_a67930029579": "真人补充本人换班的真实阻碍和即时情绪；C0只是把“还没决定”重新说完整。",
        "human_679e4041a693": "真人继续本人此前反复自嘲亏损、没本金的立场；C0换成了一个新造的扣税回嘴。",
        "human_b2d2eafbb9c0": "真人以两条短消息延续本人当时的即时反应；C0把两个动作合成一句，丢掉了本人现场的发送方式。",
        "human_ef1eb0f0cff7": "真人接住对方关于在家待产的具体回忆继续调侃；C0只重复“想休假”这个表层意思。",
        "human_52f5b21d587c": "真人直接回答有没有钱；C0把对方刚说的“值班、没钱、惨”重新组合，没有带入本人掌握的回答。",
        "human_eb7da1c841f8": "真人沿用群里正在形成的“难受”短句接龙；C0把刚出现的数字重新组织成解释。",
        "human_8d12e044ac41": "真人继续此前一直在追问的外卖供应情况；C0只围绕“饿晕了”做通用文字调侃。",
    },
    "没回答当前明确问题或请求": {
        "human_f6ee62cc14ed": "对方问是否接这笔操作，真人直接答应并说明时间；C0要求先看看又转去卖房梗，没有给出是否接的答案。",
        "human_5f0f4d34babe": "对方在问要不要把美元换掉，真人给出另一种可执行去向；C0给的是泛化折中建议，没有形成本人式明确选择。",
    },
    "回复太通用": {
        "human_c1e61037ab83": "C0的“让银行听你的”适用于任何类似场面，没有继续本人刚说的“预期太高”这一具体站位。",
        "human_e963fc5e4000": "C0只是常见的反问式回嘴；真人的反向自谦更贴合本人当时的关系站位。",
        "human_f4e67e3c04ac": "C0的“吃下安利”是任何人都能说的接话；真人继续了本人正在使用的短反问调侃。",
        "human_fdd62ca44def": "C0把邀约补成标准的“组一局”，还加了不必要条件；真人只是现场直接接受。",
        "human_42cb750bf3fc": "C0在“还没”后追加无法核实的时间估计；真人只给当下确定、直接的答案。",
    },
    "解释或组织过于工整": {
        "human_7cc57ec54014": "C0补出完整行情解释再加自嘲，像整理后的回答；真人是多条粗短的即时交易状态。",
        "human_ec997e96dd9c": "C0把“卡没下来”和“蔚来”拼成完整机巧句；真人继续说自己的买卖经历并分条自嘲。",
        "human_74065d04c49c": "C0给出整齐的三步验证方案；真人说的是现场实际如何合并实验结果。",
        "human_f655dfbf6b61": "C0一次补全基金身份、行情、估值和操作建议；真人只给当前立场下的短替代建议。",
    },
    "回错聊天对象、线程或本人立场": {
        "human_546022814a09": "最新话题是冰箱污染，C0直接给处置建议；真人回到当时仍在进行的餐具支线，说明缺少媒体时线程可能判断错。",
        "human_f342b7baf16e": "C0跨到人物所在地和互相捧哏，偏离正在讨论的买菜业务竞争。",
        "human_7dcd89af4878": "C0只总结周末核酸；真人回扣本人早上请假看病和徐汇这一尚未接完的线程。",
        "human_28edc542e109": "C0同时接两个话题并替第三个人安排请客；真人是在回应三人当下的关系站位。",
        "human_a51cb38bcc23": "C0继续旧的司机玩笑，没有回应最新的“看来有购车计划”这一判断。",
        "human_54b2afba1e43": "本人刚明确说试过、麻烦且浪费时间，C0却突然改成立即购买，回错了本人的当前立场。",
    },
    "与上下文直接矛盾": {
        "human_e08f8580f627": "本人刚说明0.7的车位比很紧张、可能得买，C0却说“够停就行”，与可见上下文直接相反。",
        "human_4409736e4fec": "当前时间已经是00:54，C0回答“11点下班”在这个语境中无法成立；真人回答仍未下班且双十一到1点。",
    },
    "缺失媒体，无法可靠归因": {
        "human_fd95a5f360e5": "关键内容是未识别图片；仅靠文字无法判断真人为何提杭州，也无法判断C0问“啥图”是否更合理。",
        "human_1217b75184c8": "“这里会变红”依赖未识别链接或图片；缺少媒体时无法可靠判断具体信息来自哪里。",
    },
}

CATEGORY_DESCRIPTIONS = {
    "没继续本人刚建立的信息": "真人会继续自己刚说过的事实、状态、判断或表达方式；C0没有把这条线接下去。",
    "没回答当前明确问题或请求": "对方已经提出可回答的问题或请求，但C0没有给出实际答案。",
    "回复太通用": "回复放进很多相似聊天都成立，没有体现当前人物、关系或现场。",
    "解释或组织过于工整": "C0把现场整理成完整解释、标准建议或精巧句式，少了真人即时接话的粗糙感。",
    "回错聊天对象、线程或本人立场": "C0接到了旧话题、错误的人，或说出了与本人当前立场不一致的话。",
    "与上下文直接矛盾": "C0回复与页面中可见的聊天事实直接冲突；按当前口径，这一类属于硬伤。",
    "缺失媒体，无法可靠归因": "关键图片、链接或表情内容不可见，现有文字不足以判断真正原因。",
    "数据泄漏，不能用于判断": "生成回复时召回的参考案例包含了本题真人答案，因此这条只能展示，不能计入效果或问题分类。",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def section(prompt: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", prompt, flags=re.S)
    return match.group(1).strip() if match else "源数据中没有这一段"


def compact_xml_messages(raw: str) -> str:
    messages = re.findall(r"<message\b[^>]*>(.*?)</message>", raw, flags=re.S)
    if not messages:
        return raw
    return "\n\n".join(html.unescape(message.strip()) for message in messages)


def chronological_timeline(prompt: str) -> str:
    """Merge other/self messages by their original index, then append unread messages."""
    earlier = []
    for tag in ("history", "consumed_self_replies"):
        raw = section(prompt, tag)
        for index, message in re.findall(
            r'<message\b[^>]*index="(\d+)"[^>]*>(.*?)</message>', raw, flags=re.S
        ):
            earlier.append((int(index), html.unescape(message.strip())))
    earlier.sort(key=lambda item: item[0])

    unread = []
    for index, message in re.findall(
        r'<message\b[^>]*index="(\d+)"[^>]*>(.*?)</message>',
        section(prompt, "unread"),
        flags=re.S,
    ):
        unread.append((int(index), html.unescape(message.strip())))
    unread.sort(key=lambda item: item[0])

    timeline = [message for _, message in earlier]
    timeline.extend(f"【本次需要回复】{message}" for _, message in unread)
    return "\n\n".join(timeline) or "源数据中没有可展示的消息"


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", "", html.unescape(str(value or ""))).strip()


def find_content_leaks(human_reply: list[str] | str, few_shot_content: str) -> list[str]:
    """Return example IDs containing every message of the target human reply."""
    replies = [human_reply] if isinstance(human_reply, str) else human_reply
    target = [_normalized_text(message) for message in replies if _normalized_text(message)]
    if not target:
        return []
    matches = []
    for example_id, body in re.findall(
        r'<example\s+id="([^"]+)"[^>]*>(.*?)</example>',
        few_shot_content or "",
        flags=re.S,
    ):
        messages = [
            _normalized_text(message)
            for message in re.findall(r"<message\b[^>]*>(.*?)</message>", body, flags=re.S)
        ]
        if all(message in messages for message in target):
            matches.append(example_id)
    return matches


def scan_result_directory(report_dir: Path) -> dict:
    """Scan recorded A/B runs without regenerating any reply."""
    cases = []
    for path in sorted(report_dir.glob("results*.json")):
        cases.extend(load_json(path).get("cases", []))
    seen = set()
    leaks = []
    for case in cases:
        case_id = case.get("case_id")
        if case_id in seen:
            continue
        seen.add(case_id)
        for condition, condition_data in case.get("conditions", {}).items():
            for run in condition_data.get("runs", []):
                matched_ids = find_content_leaks(
                    case.get("human_reply") or [],
                    str(run.get("few_shot_content") or ""),
                )
                if matched_ids:
                    leaks.append({
                        "case_id": case_id,
                        "condition": condition,
                        "repetition": run.get("repetition"),
                        "few_shot_ids": matched_ids,
                        "human_reply": case.get("human_reply") or [],
                    })
    return {"case_count": len(seen), "leaks": leaks}


def load_badcase_review() -> dict:
    cases: dict[str, dict] = {}
    prompts: dict[str, tuple[str, dict]] = {}
    for path in sorted(REPORT_DIR.glob("results.shard-*-of-4.json")):
        for case in load_json(path).get("cases", []):
            cases[case["case_id"]] = case
    for path in sorted(REPORT_DIR.glob("generation_checkpoint.shard-*-of-4.json")):
        data = load_json(path)
        for key, entry in data.get("entries", {}).items():
            if not key.endswith(":baseline") or entry.get("status") not in {"scorable", "complete"}:
                continue
            result = entry["result"]
            prompts[key.removesuffix(":baseline")] = (
                result.get("trace", {}).get("user_prompt", ""),
                result.get("input_metadata", {}),
            )

    labels: dict[str, tuple[str, str]] = {}
    for category, mapping in CATEGORIES.items():
        for case_id, why in mapping.items():
            if case_id in labels:
                raise ValueError(f"duplicate classification: {case_id}")
            labels[case_id] = (category, why)

    rows = []
    for index, case_id in enumerate(sorted(cases), start=1):
        case = cases[case_id]
        baseline = case["conditions"]["baseline"]["runs"][0]
        candidate = case["conditions"]["candidate"]["runs"][0]
        baseline_leaks = find_content_leaks(case["human_reply"], baseline.get("few_shot_content", ""))
        candidate_leaks = find_content_leaks(case["human_reply"], candidate.get("few_shot_content", ""))
        leak_conditions = []
        if baseline_leaks:
            leak_conditions.append("原版")
        if candidate_leaks:
            leak_conditions.append("候选版")
        valid = not leak_conditions
        category, why = labels.get(case_id, (
            "Judge没有识别出原版C0",
            "Judge在原版对照中没有正确找出真人回复；这道题不用于归纳原版C0为什么像AI。",
        ))
        prompt, metadata = prompts.get(case_id, ("", {}))
        rows.append({
            "index": index,
            "id": case_id,
            "category": category,
            "category_description": CATEGORY_DESCRIPTIONS.get(category, "这道题不属于原版C0被识别出的坏案例。"),
            "why": why,
            "chat_name": metadata.get("chat_name", "源数据未提供群名"),
            "history": compact_xml_messages(section(prompt, "history")),
            "unread": compact_xml_messages(section(prompt, "unread")),
            "self_replies": compact_xml_messages(section(prompt, "consumed_self_replies")),
            "timeline": chronological_timeline(prompt),
            "human_reply": case["human_reply"],
            "baseline_reply": baseline.get("reply") or [],
            "candidate_reply": candidate.get("reply") or [],
            "baseline_judge": baseline.get("judge") or {},
            "candidate_judge": candidate.get("judge") or {},
            "hard_error": valid and category == "与上下文直接矛盾",
            "valid": valid,
            "leak_conditions": leak_conditions,
            "leak_few_shot_ids": sorted(set(baseline_leaks + candidate_leaks)),
        })

    invalid_ids = {row["id"] for row in rows if not row["valid"]}
    valid_rows = [row for row in rows if row["valid"]]
    return {
        "rows": rows,
        "summary": {
            "original_cases": len(rows),
            "invalid_cases": len(invalid_ids),
            "valid_cases": len(valid_rows),
            "baseline_identified": sum(bool(row["baseline_judge"].get("correct")) for row in valid_rows),
            "candidate_identified": sum(bool(row["candidate_judge"].get("correct")) for row in valid_rows),
        },
    }


def build() -> None:
    review = load_badcase_review()
    cases = {
        row["id"]: row
        for row in review["rows"]
        if (row["valid"] and row["baseline_judge"].get("correct")) or not row["valid"]
    }
    labels = {
        case_id: (
            (row["category"], row["why"])
            if row["valid"]
            else (
                "数据泄漏，不能用于判断",
                "原版和候选版召回的参考案例包含了本题真人答案；该题保留供核对，但不计入33条有效案例。",
            )
        )
        for case_id, row in cases.items()
    }

    rows = []
    for index, case_id in enumerate(labels, start=1):
        category, why = labels[case_id]
        item = cases[case_id]
        judge = item["baseline_judge"]
        rows.append(
            {
                "index": index,
                "id": case_id,
                "category": category,
                "category_description": CATEGORY_DESCRIPTIONS[category],
                "why": why,
                "chat_name": item["chat_name"],
                "history": item["history"],
                "unread": item["unread"],
                "self_replies": item["self_replies"],
                "timeline": item["timeline"],
                "human_reply": "\n".join(item["human_reply"]),
                "c0_reply": "\n".join(item["baseline_reply"]),
                "confidence": judge["confidence"],
                "judge_reason": judge["reason"],
                "judge_uncertainty": judge.get("uncertainty", "未记录"),
                "hard_error": item["hard_error"],
                "valid": item["valid"],
                "leak_conditions": item["leak_conditions"],
                "leak_few_shot_ids": item["leak_few_shot_ids"],
            }
        )

    counts = {
        category: sum(row["category"] == category for row in rows)
        for category in CATEGORY_DESCRIPTIONS
    }
    total = len(rows)
    valid_total = sum(row["valid"] for row in rows)
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    count_buttons = "".join(
        f'<button class="filter" data-category="{html.escape(category)}">{html.escape(category)} <b>{count}</b></button>'
        for category, count in counts.items()
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C0 群聊案例复盘（{total}条展示，{valid_total}条有效）</title>
<style>
:root{{--bg:#f3f5f7;--card:#fff;--ink:#182026;--muted:#65727c;--line:#dce2e7;--blue:#1769e0;--red:#b42318;--green:#157347}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}
main{{max-width:920px;margin:auto;padding:16px}} h1{{font-size:24px;margin:4px 0 8px}} .note,.summary{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}}
.note p{{margin:6px 0}} .filters{{display:flex;gap:8px;overflow:auto;padding:4px 0 10px;position:sticky;top:0;background:var(--bg);z-index:3}}
button{{white-space:nowrap;border:1px solid #bfd0e7;background:#fff;border-radius:999px;padding:9px 12px;color:#234}} button.active{{background:var(--blue);border-color:var(--blue);color:#fff}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:12px 0;overflow:hidden}} .case-head{{padding:14px}} .tag{{display:inline-block;background:#eaf2ff;color:#1659b7;border-radius:6px;padding:2px 7px;font-size:12px}}
.case h2{{font-size:18px;margin:8px 0 3px}} .meta{{color:var(--muted);font-size:13px}} .why{{background:#fff8e6;border-left:4px solid #e7a000;padding:10px 12px;margin:10px 0 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 14px 14px}} .reply{{border-radius:10px;padding:11px;white-space:pre-wrap}} .human{{background:#eaf8ef}} .ai{{background:#fff0f0}}
.reply strong{{display:block;margin-bottom:5px}} details{{border-top:1px solid var(--line)}} summary{{cursor:pointer;padding:12px 14px;font-weight:600}} .detail{{padding:0 14px 14px}} pre{{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;border-radius:9px;padding:10px;margin:7px 0;font:13px/1.55 ui-monospace,SFMono-Regular,monospace}}
.judge{{background:#f1f6ff;border-radius:9px;padding:11px}} .hard{{color:var(--red);font-weight:700}} .ok{{color:var(--green);font-weight:600}} .hidden{{display:none}}
@media(max-width:620px){{main{{padding:10px}} .pair{{grid-template-columns:1fr}} h1{{font-size:21px}} .filters{{margin:0 -10px;padding-left:10px}}}}
</style></head><body><main>
<h1>C0 群聊案例复盘：{total}条展示，{valid_total}条有效</h1>
<section class="note"><p><b>这是什么：</b>页面展示34条案例。其中33条是 Judge 认出真人回复、因此也识别出 C0 回复是 AI 的有效案例；另1条存在 few-shot 数据泄漏，只保留供你核对，不计入任何结果。</p><p><b>这不是什么：</b>不是240题正式测试结果，也不是新一轮实验。</p><p><b>分类怎么来的：</b>根据每条已有的 Judge 原始理由，结合完整上下文、真人回复和C0回复复盘归类。分类用于找改进方向，不是真实世界的绝对标签。</p><p><b>“本人此前说过的话”：</b>生成时提供给C0理解上下文、但禁止原样复用的本人消息；单独展开是为了明确谁说了什么。</p></section>
<section class="summary"><b>数量：</b>{valid_total}条有效案例 + {total-valid_total}条泄漏案例 = {total}条展示。点击下面分类可筛选。</section>
<div class="filters"><button class="filter active" data-category="all">全部 <b>{total}</b></button>{count_buttons}</div>
<div id="cases"></div>
<script id="data" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), root=document.getElementById('cases');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
root.innerHTML=data.map(x=>`<article class="case" data-category="${{esc(x.category)}}"><div class="case-head"><span class="tag">${{esc(x.category)}}</span><h2>${{x.index}}. ${{esc(x.chat_name)}}</h2><div class="meta">Case ID：${{esc(x.id)}} · ${{x.valid?`Judge置信度：${{Math.round(x.confidence*100)}}%`:'无有效 Judge 结论'}}</div><div class="why"><b>为什么归到这类：</b>${{esc(x.why)}}<br><span class="meta">分类含义：${{esc(x.category_description)}}</span></div></div><div class="pair"><div class="reply human"><strong>真人原回复</strong>${{esc(x.human_reply)}}</div><div class="reply ai"><strong>当前 C0 回复</strong>${{esc(x.c0_reply)}}</div></div><details><summary>查看按原始顺序合并的完整聊天</summary><div class="detail"><pre>${{esc(x.timeline)}}</pre></div></details><details><summary>${{x.valid?'查看 Judge 的判断和原始理由':'查看数据泄漏详情'}}</summary><div class="detail judge">${{x.valid?`<p><b>判断：</b>Judge认出了真人回复，因此C0在本题被识别为AI。</p><p><b>原始理由：</b>${{esc(x.judge_reason)}}</p><p><b>不确定因素：</b>${{esc(x.judge_uncertainty)}}</p><p class="${{x.hard_error?'hard':'ok'}}"><b>是否按当前口径算硬伤：</b>${{x.hard_error?'是——与可见上下文直接矛盾':'否'}}</p>`:`<p class="hard"><b>无效原因：</b>${{esc(x.leak_conditions.join('、'))}}召回的参考案例包含本题真人答案。</p><p><b>泄漏来源 Case ID：</b>${{esc(x.leak_few_shot_ids.join('、'))}}</p><p>本题不计入 Judge 识别率、改善/退步数量或问题分类。</p>`}}</div></details></article>`).join('');
document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active'));b.classList.add('active');const c=b.dataset.category;document.querySelectorAll('.case').forEach(x=>x.classList.toggle('hidden',c!=='all'&&x.dataset.category!==c));}});
</script></main></body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(rows)} cases; counts={counts}")


if __name__ == "__main__":
    build()
