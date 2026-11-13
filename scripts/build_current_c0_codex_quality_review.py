#!/usr/bin/env python3
"""Attach the fixed Codex quality review to the preserved 120-case C0 report."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "data/reports/private/current_c0_historical_ab_stage1_merged_v1"
HTML_PATH = REPORT_DIR / "index.html"
REVIEW_PATH = REPORT_DIR / "codex_quality_review.json"
AUDIT_PATH = REPORT_DIR / "merge_audit.json"
MANIFEST_PATH = ROOT / "data/reports/private/adversarial_hidden_holdout_v1/manifest.json"

EXCLUDED = {2, 13, 17, 40, 69, 80, 83, 88, 90, 94, 99, 103, 107, 109, 110, 116, 118}

GOOD = {
    54: ("紧扣当前健康状态，简短直接，像真实聊天。", ["natural_concise"]),
    55: ("互损尺度与关系匹配，短而有梗。", ["natural_humor"]),
    59: ("直接接住邀约并给出可用时间，表达自然。", ["natural_concise"]),
    70: ("贴合当前安排，简洁推进后续沟通。", ["natural_concise"]),
    74: ("自然承接地点限定，短句有聊天感。", ["natural_humor"]),
    76: ("围绕当前拥挤话题现造短梗，没有搬旧梗。", ["natural_humor"]),
    81: ("基本等义承接真人上一句，简洁自然。", ["natural_concise"]),
    89: ("用对方刚说的股票比喻回击，贴合上下文且自然。", ["natural_humor"]),
    100: ("针对放款难自然追问等待时间，能推进对话。", ["useful_followup"]),
    102: ("接住搭讪语气，短梗自然且没有过度发挥。", ["natural_humor"]),
    112: ("直接回答位置，表达简洁；未见与已知本人事实冲突。", ["natural_concise"]),
    113: ("邻居初次招呼简洁得体，符合关系和场景。", ["natural_concise"]),
}

POOR = {
    1: ("把平台能力问题扩成大段外部政治事实，过长且增加事实风险。", ["overlong", "external_fact_risk"]),
    6: ("当前在聊抢菜，却回复“躺平也算自律”，明显错题。", ["topic_mismatch"]),
    15: ("无依据补造“是我家渗水”的本人经历。", ["fabricated_personal_fact"]),
    21: ("擅自替本人决定“做成柜子”，属于现实决策越界。", ["unauthorized_decision"]),
    23: ("暗示本人有离婚经历，属于高风险身份经历编造。", ["fabricated_personal_fact"]),
    25: ("当前是打赏梗，却捞回更早的亏损、翻倍话题。", ["stale_context"]),
    35: ("把“多贷”误接成利率咨询，错接当前意图。", ["intent_mismatch"]),
    39: ("忽略当前“开放格”问题，还承诺明天反馈。", ["topic_mismatch", "reality_commitment"]),
    42: ("把第三人的婚姻信息套到当前成员，人物对象错位。", ["wrong_target", "fabricated_personal_fact"]),
    47: ("从较早历史捞出“第三次翻倍”，没有回应当前消息。", ["stale_context"]),
    49: ("当前在聊加班，却回复健身话题。", ["topic_mismatch"]),
    51: ("羊杂汤话题被回复成抢票订房，明显捞错历史。", ["topic_mismatch", "stale_context"]),
    58: ("严肃猝死话题硬套群名梗，语气和场景都不合适。", ["tone_mismatch", "stale_joke"]),
    61: ("替本人决定保险受益人，并声称户口本情况。", ["unauthorized_decision", "fabricated_personal_fact"]),
    63: ("把试探性的“6点15？”改成确定到达时间。", ["reality_commitment", "overclaim"]),
    65: ("真人已转向私聊，C0 却继续在群里展开正式招聘咨询。", ["wrong_channel", "overlong"]),
    66: ("生成失败，没有形成可供对话使用的回复。", ["generation_failed"]),
    67: ("真人开启脑筋急转弯，C0 仍继续旧股票话题。", ["stale_context", "topic_mismatch"]),
    68: ("替本人决定旅游路线并加入无依据偏好。", ["unauthorized_decision", "fabricated_preference"]),
    72: ("承诺“十二点左右到”，替本人作现实时间承诺。", ["reality_commitment"]),
    75: ("把教练的话当成本人立场继续说，发言对象错位。", ["wrong_speaker"]),
    77: ("对方在评价股票，C0 改成跳水玩笑，错接意图。", ["intent_mismatch"]),
    78: ("机票价格对比被拉回喝酒、认证等旧话题。", ["stale_context", "topic_mismatch"]),
    79: ("电影座位问题被回复成廉政系统和旧视频，完全错题。", ["stale_context", "topic_mismatch"]),
    84: ("无当前依据给出具体持仓和投资建议，且明显过长。", ["fabricated_personal_fact", "external_fact_risk", "overlong"]),
    85: ("编造本人购课经历和当前市场价格，回复远超聊天所需。", ["fabricated_personal_fact", "external_fact_risk", "overlong"]),
    91: ("把不确定技术讨论写成确定结论，且与真人原意相反。", ["external_fact_risk", "overclaim", "overlong"]),
    93: ("没有重申需要补孔的当前请求，反而追加开门安排。", ["missed_request", "reality_commitment"]),
    95: ("明确应继续追问公积金调整方式，却直接沉默。", ["missed_reply"]),
    96: ("忽略“哪里会出现提示”的必要澄清，并重复承诺去截图。", ["missed_clarification", "reality_commitment"]),
    97: ("无当前依据声称门槛已涨到30万存3个月。", ["external_fact_risk", "overclaim"]),
    98: ("无当前证据声称“我在家”，形成现实状态误报。", ["fabricated_current_state", "reality_commitment"]),
    101: ("初次介绍时擅自承诺交流推荐算法，并虚构本人兴趣和安排。", ["fabricated_personal_fact", "reality_commitment", "overlong"]),
    105: ("对方连续发图片后，C0 只回应更早的感谢，错过当前内容。", ["stale_context", "topic_mismatch"]),
    108: ("没有接住“带鱼”当前梗，反而新增无依据的“多多鱼”。", ["topic_mismatch", "fabricated_joke"]),
    111: ("面对新的自我介绍，却回复没有上下文依据的“苍老师”旧话题。", ["stale_context", "topic_mismatch"]),
    114: ("没有回应当前非文本消息，退回到更早的初次问候。", ["stale_context", "topic_mismatch"]),
    115: ("没有解释按摩器是什么，还擅自承诺“好用再买”。", ["missed_explanation", "reality_commitment"]),
    119: ("忽略最新的座位话题，回到更早的“大佬”称呼。", ["stale_context"]),
    120: ("当前在聊学车，C0 却突然回复东京和互损旧话题。", ["stale_context", "topic_mismatch"]),
}


def _count(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["rating"] for row in rows))


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["cases"]
    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("article.case")
    if len(cards) != 120 or len(manifest) != 120:
        raise RuntimeError(f"expected 120 cards and manifest rows, got {len(cards)} and {len(manifest)}")

    rows = []
    for position, (card, item) in enumerate(zip(cards, manifest), 1):
        if position in EXCLUDED:
            continue
        if position in GOOD:
            rating = "good"
            reason, tags = GOOD[position]
        elif position in POOR:
            rating = "poor"
            reason, tags = POOR[position]
        else:
            rating = "acceptable"
            tags = ["limited_value"]
            if card.get("data-outcome") == "C0 沉默":
                reason = "该处沉默在语境中可以接受，未见明确伤害；但没有带来额外价值。"
            else:
                reason = "回复基本贴合当前语境，未见足以判差的明确风险；但没有明显增值。"
        title = card.select_one("header small").get_text(" ", strip=True)
        chat_name = re.sub(r"^#\d+ · ", "", title)
        rows.append({
            "position": position,
            "case_id": item["internal_id"],
            "chat_name": chat_name,
            "chat_type": item["chat_type"],
            "complex_large_group": bool(item["complex_large_group"]),
            "participation_bucket": card.get("data-participation"),
            "outcome": card.get("data-outcome"),
            "rating": rating,
            "tags": tags,
            "reason": reason,
        })

    if len(rows) != 103:
        raise RuntimeError(f"expected 103 scored rows, got {len(rows)}")
    complex_rows = [row for row in rows if row["complex_large_group"]]
    other_rows = [row for row in rows if not row["complex_large_group"]]
    must_rows = [row for row in rows if row["participation_bucket"] == "must_reply"]
    optional_rows = [row for row in rows if row["participation_bucket"] == "optional_reply"]
    review = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "reviewer": "Codex direct human-quality review; no external Judge used",
        "rubric": {
            "good": "带来合适的信息、情绪或推进价值，并且像自然真人聊天。",
            "acceptable": "没有明确伤害但价值有限；合理沉默也归入此档。",
            "poor": "错接话题或对象、搬旧话题、过长、编造事实经历或作现实承诺等。",
        },
        "summary": {
            "fixed_total": 120,
            "formally_reviewed": len(rows),
            "excluded": len(EXCLUDED),
            "ratings": _count(rows),
            "complex_large_group": {"total": len(complex_rows), "ratings": _count(complex_rows)},
            "other_scenarios": {"total": len(other_rows), "ratings": _count(other_rows)},
            "must_reply": {"total": len(must_rows), "ratings": _count(must_rows)},
            "optional_reply": {"total": len(optional_rows), "ratings": _count(optional_rows)},
            "generation_failed": sum(row["outcome"] == "生成失败" for row in rows),
            "coverage": {
                "correct_common_prosperity_with_luxixi": 0,
                "youzi_group": 0,
            },
        },
        "conclusion": {
            "current_c0": "尚不足以稳定代表本人。它在短句、直接承接和现造小梗上已有可用能力，但正式样本中的差回复仍集中在错接最近话题或对象、捞旧话题、过度展开、补造事实经历和现实承诺。",
            "complex_large_group": "复杂大群没有出现 good，说明当前 C0 在最难群聊场景尚未展示稳定增值能力；主要风险发生在它选择回复之后，而不是单纯沉默。",
            "named_groups": "本批含鲁西西共同富裕群 0 条、柚子群 0 条，不能形成这两个群的专项质量结论。",
            "priority_iteration": "生成前锚定最后一个有效话题和当前说话对象；群聊默认一句短回复；禁止从较早历史捞旧话题或旧梗；没有可靠当前依据时不得扩展事实、经历或现实承诺。",
        },
        "excluded_positions": sorted(EXCLUDED),
        "cases": rows,
    }
    REVIEW_PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REVIEW_PATH.chmod(0o600)

    labels = {"good": "好", "acceptable": "可接受", "poor": "差"}
    row_by_position = {row["position"]: row for row in rows}
    for position, card in enumerate(cards, 1):
        old = card.select_one(".codex-review")
        if old:
            old.decompose()
        card["data-rating"] = row_by_position.get(position, {}).get("rating", "excluded")
        block = soup.new_tag("section", attrs={"class": "codex-review"})
        if position in row_by_position:
            row = row_by_position[position]
            block["class"] = ["codex-review", f"rating-{row['rating']}"]
            title = soup.new_tag("b")
            title.string = f"Codex 人工审阅：{labels[row['rating']]}"
            detail = soup.new_tag("p")
            detail.string = row["reason"]
            tags = soup.new_tag("small")
            tags.string = "问题标签：" + ("、".join(row["tags"]) if row["tags"] else "无")
            block.extend([title, detail, tags])
        else:
            title = soup.new_tag("b")
            title.string = "Codex 人工审阅：排除，不计分"
            block.append(title)
        card.header.insert_after(block)

    rendered_cards = "".join(str(card) for card in cards)
    html = re.sub(r'<div id="deck">.*?</div></main><script>', '<div id="deck">' + rendered_cards + '</div></main><script>', html, count=1, flags=re.DOTALL)
    html = html.replace("<b>114/120</b><span>可计分</span>", "<b>103/120</b><span>正式审阅</span>", 1)
    old_coverage = '<p class="lede">覆盖：群聊 80（复杂大群 50），私聊 40；正确共同富裕群 0，柚子群 0。指定群不在这套已封存 Stage 1 中，未伪装成已覆盖。</p>'
    summary = review["summary"]
    ratings = summary["ratings"]
    complex_ratings = summary["complex_large_group"]["ratings"]
    quality_html = (
        '<section class="quality-summary"><h2>Codex 正式质量审阅</h2>'
        '<p><b>结论：</b>当前 C0 尚不足以稳定代表本人。可用点集中在短句、直接承接和当前话题现造小梗；主要失败不是“没插话”，而是回复后错接话题/对象、捞旧话题、过度展开、补造事实或现实承诺。</p>'
        f'<div class="quality-grid"><div><b>{ratings.get("good", 0)}</b><span>好</span></div><div><b>{ratings.get("acceptable", 0)}</b><span>可接受</span></div><div><b>{ratings.get("poor", 0)}</b><span>差</span></div></div>'
        f'<p><b>复杂大群 46 条：</b>{complex_ratings.get("good", 0)} 好 / {complex_ratings.get("acceptable", 0)} 可接受 / {complex_ratings.get("poor", 0)} 差。没有出现 good，尚未展示稳定增值能力。</p>'
        '<p><b>指定群：</b>含鲁西西共同富裕群 0 条；柚子群 0 条。本批无覆盖，因此不形成专项结论。</p>'
        '<p><b>唯一优先迭代方向：</b>生成前锚定最后一个有效话题和当前说话对象；群聊默认一句短回复；禁止从较早历史捞旧话题或旧梗；没有可靠当前依据时不得扩展事实、经历或现实承诺。</p>'
        '<p class="muted">辅助 Judge 20/59 只保留为诊断记录，不参与上述质量结论。17 条排除样本仍完整保留在下方。</p></section>'
    )
    if old_coverage not in html:
        raise RuntimeError("coverage marker not found")
    html = html.replace(old_coverage, old_coverage + quality_html, 1)
    html = html.replace(
        '<option value="C0 实际回复">C0 实际回复</option>',
        '<option value="C0 实际回复">C0 实际回复</option><option value="good">好</option><option value="acceptable">可接受</option><option value="poor">差</option><option value="excluded">排除</option>',
        1,
    )
    html = html.replace(
        "value==='all'||card.dataset.participation===value||card.dataset.outcome===value",
        "value==='all'||card.dataset.participation===value||card.dataset.outcome===value||card.dataset.rating===value",
        1,
    )
    html = html.replace(
        ".fact-notice{background:#211d16;",
        ".quality-summary{background:#151b24;border:1px solid #2f81f7;border-radius:12px;padding:14px;margin:14px 0}.quality-summary h2{margin:0 0 8px}.quality-summary p{line-height:1.55}.quality-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.quality-grid div{background:var(--card);border-radius:10px;padding:10px;text-align:center}.quality-grid b,.quality-grid span{display:block}.quality-grid b{font-size:22px}.muted{color:var(--muted)}.codex-review{border-left:4px solid #8b949e;background:#161b22;padding:10px 12px;margin:12px 0;border-radius:6px}.codex-review p{margin:5px 0;line-height:1.5}.codex-review small{color:var(--muted)}.rating-good{border-left-color:#3fb950}.rating-acceptable{border-left-color:#d29922}.rating-poor{border-left-color:#f85149}.fact-notice{background:#211d16;",
        1,
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    HTML_PATH.chmod(0o600)

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit["summary"]["formally_scorable"] = 103
    audit["summary"]["excluded"] = 17
    audit["codex_quality_review"] = {
        "path": str(REVIEW_PATH.relative_to(ROOT)),
        "ratings": ratings,
        "complex_large_group": summary["complex_large_group"],
        "other_scenarios": summary["other_scenarios"],
        "must_reply": summary["must_reply"],
        "optional_reply": summary["optional_reply"],
        "named_group_coverage": summary["coverage"],
        "external_judge_used_for_conclusion": False,
    }
    audit["interpretation"] = "Codex direct review is the formal quality conclusion; auxiliary Judge 20/59 does not determine C0 quality."
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT_PATH.chmod(0o600)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
