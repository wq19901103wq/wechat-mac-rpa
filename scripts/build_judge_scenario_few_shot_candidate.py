#!/usr/bin/env python3
"""Build the Judge candidate that uses labeled, development-only scenario examples."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import (
    _json_sha256,
    assert_reference_isolated,
    reference_leakage_details,
)

PRIVATE = ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
DEV_BLIND = PRIVATE / "dev_blind.json"
DEV_ANSWERS = PRIVATE / "dev_answers.json"
BASE_PROFILE = PRIVATE / "judge_profile_v2_content_anchored_candidate.json"
BASE_REFERENCE = PRIVATE / "judge_reference.json"
OUT_PROFILE = PRIVATE / "judge_profile_v2_scenario_few_shot_candidate_v6.json"
OUT_REFERENCE = PRIVATE / "judge_reference_scenario_few_shot_candidate_v6.json"
FORMAL_RESULTS = ROOT / "data/reports/private/situation_c0_latest_turn_form_formal240_ab_v4"


RATIONALES = {
    "case_f6608dc748a32ffcb8acb783": {
        "reason": "真人只确认已经接到客服电话；另一项擅自补出‘今晚7点上门’，更完整但没有上下文依据。完整和具体不能压过事实证据。",
        "boundary": "具体信息若已在上下文或可靠参考事实中出现，则不因具体而扣分。",
    },
    "case_20315f2c67f7a10a3a7e5daa": {
        "reason": "对方刚说电梯完全没用，再解释‘一楼电梯就是摆设’只重复显然信息；真人转而追问三楼并补充采光观察。",
        "boundary": "对方明确没听懂，或解释带来决策所需新信息时，不执行重复解释扣分。",
    },
    "case_2180eda4e3aa7ee2ad1171bd": {
        "reason": "真人补充‘没去、失联48小时、今天回我’等自身掌握的新情况；不能因为 Judge 事先不知道这些私人事实就判成编造。",
        "boundary": "未知事实本身也不是正证据；若与上下文或可靠事实直接冲突，仍应作为反证。",
    },
    "case_5e9dd8c470b40c81d37d52a9": {
        "reason": "‘羡慕大佬’在这段关系和连续聊天中确实是真人说法；固定句式既不能自动判 AI，也不能仅凭命中句式自动判真人。",
        "boundary": "必须结合人物站位、当前语境和稳定表达习惯，不能机械匹配关键词。",
    },
    "case_23a4cb239fa90897e56bdc09": {
        "reason": "真人直接说永久合同实际没多大用；另一项是更工整、更像成品的机智话，但‘更巧妙’不是本人证据。",
        "boundary": "若机智表达能由本人稳定习惯或当前具体互动支持，才可作为弱证据。",
    },
    "case_2fd0078f3a38a10e8a8ba498": {
        "reason": "真人用两句短话确认并压缩了对方的判断，表达自己的认同和收束；这具有立场与互动作用，不是解释人人已懂的信息。",
        "boundary": "只有纯粹换词重说、且没有确认、立场、情绪、总结、收束或关系作用时，才按无价值复述扣分。",
    },
    "case_fbc031b5c4e6a97d376dc264": {
        "reason": "对方吹捧本人时，真人延续自己刚才的观点，用泛化表达淡化吹捧；这种自谦式改述有关系与立场作用，不能机械视为复述。",
        "boundary": "改述本身不是正证据；仍需结合说话者刚才的立场和人物关系，不能只因简短或自然就判真人。",
    },
}


DECISION_PRIORITY = [
    "P0 可判性：若上下文含未识别图片、文件、菜单或表情，导致候选中的具体信息可能来自媒体，不得因纯文本未出现该信息就判为编造；决定性证据写‘无’，仍需二选一时 confidence 不得高于 0.55。此类题原则上应在抽样阶段排除。",
    "P1 可靠事实：先检查与完整上下文和可靠参考事实的直接冲突。Judge 不知道的私人或现场事实记为未知，不得当作编造或反证。",
    "P2 强负证据：对人人已懂的词义、数字大小或显然因果进行没有任何新语用作用的多余解释，必须扣分。简短确认、压缩总结、延续本人刚才立场、自谦、表达情绪或收束话题具有新语用作用，不因语义相近自动扣分。脱离现场硬贴老梗、机械套话才是反证；现场自然接梗或群内短时形成的共同话语不自动扣分。P2 优先于‘紧扣最后一句’。",
    "P3 本人特异证据：先看最新现场、人物关系和群内共同话语，再看反向站位、被吹捧时假装不是本人、浑水摸鱼、私人知识和稳定句式。跨线程承接只有比当前现场承接更符合本人关系与习惯时才是证据；复杂、精巧或补全实体名本身不是正证据。",
    "P4 弱表面证据：自然、完整、礼貌、简短、俏皮、贴题只能在 P1-P3 无差异时作弱平局信号，不能单独决定答案。",
    "P5 置信度：没有可核对的决定性证据、两项都合理或主要依赖推断时，decisive_evidence 写‘无’，confidence 不得高于 0.65；不是普遍降置信度。",
]


DECISION_GATES = [
    {
        "name": "obvious_restatement_veto",
        "rule": (
            "逐项检查候选是否在解释双方显然已懂的词义、数字大小或因果，且没有增加确认、立场、情绪、总结、"
            "话题收束或关系作用。只有同时满足这两个条件才记为强反证。简短赞同、压缩总结、延续本人刚才观点、"
            "自谦式改述不命中此门；但也不得仅凭自然或简短把它们当作真人正证据。"
        ),
    },
    {
        "name": "missing_media_guard",
        "rule": (
            "若上下文出现未识别图片、文件、菜单、表情或空白媒体消息，候选中的具体事实可能来自缺失媒体。"
            "不得因纯文本中找不到依据就否定该候选，也不得把该具体事实当真人正证据；决定性证据写‘无’，"
            "confidence 不得高于 0.55。"
        ),
    },
    {
        "name": "quality_blindfold",
        "rule": (
            "先假设两项回复的礼貌、完整、帮助程度和文采全部不可见，再比较谁更像这个真人会实际发出。"
            "若去掉这些质量信号后没有人物关系、线程选择、私人知识或稳定习惯等证据，必须判为难分，"
            "不得把‘回复得更好’转换为‘更像真人’。"
        ),
    },
    {
        "name": "weak_identity_evidence_gate",
        "rule": (
            "简短、直接、口语化、礼貌程度，以及是否更自然承接最后一句，都只是通用弱信号，"
            "不得写成本人特异证据。若两项都自然，且区分主要依赖这些信号，decisive_evidence 必须写‘无’，"
            "confidence 不得高于 0.65；不得虚构人物习惯、关系或私人知识来提高置信度。"
        ),
    },
    {
        "name": "live_group_context_gate",
        "rule": (
            "活跃熟人群先判断当前现场正在形成什么共同话语、调侃对象和关系站位。"
            "针对最新数字、收益或行为的粗短拆台，以及短时间复用群友刚说过的话，都可能是自然真人互动；"
            "不得仅因措辞粗、旧或重复就扣分。只有与现场脱节、无人物关系支撑的硬贴老梗或机械复用才记反证。"
        ),
    },
    {
        "name": "clever_completion_guard",
        "rule": (
            "不得因候选跨线程更巧、补全了股票名等实体、把多个旧话题拼得更完整，就提高真人概率。"
            "若另一候选自然承接最新现场，而该候选过度工整地拼接旧线程、补全上下文未要求的信息或给出科普式完整回答，"
            "应把这种工整视为潜在模型信号；但只有结合具体上下文才能扣分。"
        ),
    },
]


NEW_PATTERN_CARDS = [
    {
        "id": "live_group_banter",
        "relationships": ["group"],
        "independent_source": "abstract_development_rule_no_holdout_answer",
        "positive_pattern": "活跃熟人群里，真人会对最新数字、收益或行为直接粗短拆台，人物关系可能比措辞完整度更有辨识力。",
        "similar_negative": "模型也能生成短吐槽；只有它确实站在当前现场和关系里时才是正证据。",
        "boundary": "不得把粗短、冒犯或不完整本身当作真人标签。",
    },
    {
        "id": "local_shared_phrase",
        "relationships": ["group"],
        "independent_source": "abstract_development_rule_no_holdout_answer",
        "positive_pattern": "群友刚形成的共同说法被多人短时复用，可能是现场参与而非机械复读。",
        "similar_negative": "脱离当前现场的模板句、陈旧梗或无关系支撑的复用仍是模型风险。",
        "boundary": "区分现场刚出现的共同话语与从外部生硬搬入的老梗。",
    },
    {
        "id": "clever_cross_thread_not_identity",
        "relationships": ["group", "private"],
        "independent_source": "abstract_development_rule_no_holdout_answer",
        "positive_pattern": "跨线程只有符合本人稳定关系、知识和回应习惯时才构成真人证据。",
        "similar_negative": "跨线程更复杂、更巧，或主动补全实体名，并不代表更像真人。",
        "boundary": "当前现场的朴素承接与旧线程的精巧承接冲突时，先看关系和习惯，不按复杂度选。",
    },
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _formal_cases() -> list[dict]:
    cases = []
    for path in sorted(FORMAL_RESULTS.glob("results.shard-*-of-*.json")):
        for row in _load(path).get("cases") or []:
            baseline = ((row.get("conditions") or {}).get("baseline") or {}).get("runs") or []
            candidate = ((row.get("conditions") or {}).get("candidate") or {}).get("runs") or []
            cases.append({
                "case_id": row["case_id"],
                "relationship": row.get("relationship"),
                "context": row.get("context") or [],
                "human_reply": row.get("human_reply") or [],
                "option_A": (baseline[0].get("reply") if baseline else []) or [],
                "option_B": (candidate[0].get("reply") if candidate else []) or [],
            })
    if len(cases) != 240 or len({row["case_id"] for row in cases}) != 240:
        raise RuntimeError(f"expected 240 unique formal cases, got {len(cases)}")
    return cases


def _remove_contaminated_reference_rows(reference: dict, cases: list[dict]) -> list[dict]:
    removed = []
    while True:
        leaks = reference_leakage_details(reference, cases)
        if not leaks:
            return removed
        indexes: dict[str, set[int]] = {"examples": set(), "scenario_few_shots": set()}
        for leak in leaks:
            match = re.match(r"reference\.(examples|scenario_few_shots)\[(\d+)\]", leak["reference_path"])
            if not match:
                raise RuntimeError(f"cannot remove contamination outside reference examples: {leak}")
            indexes[match.group(1)].add(int(match.group(2)))
        for section, section_indexes in indexes.items():
            rows = reference.get(section) or []
            for index in sorted(section_indexes, reverse=True):
                removed.append({"section": section, "index": index, "id": rows[index].get("id")})
                del rows[index]


def main() -> int:
    blind = _load(DEV_BLIND)
    answers = _load(DEV_ANSWERS)
    by_id = {row["case_id"]: row for row in blind["cases"]}
    answer_by_id = {row["case_id"]: row["human_option"] for row in answers["cases"]}
    if set(RATIONALES) - set(by_id) or set(RATIONALES) - set(answer_by_id):
        raise RuntimeError("scenario few-shot source is incomplete")

    scenario_few_shots = []
    for case_id, notes in RATIONALES.items():
        case = by_id[case_id]
        scenario_few_shots.append({
            "id": case_id,
            "source_split": "independent_dev_only",
            "relationships": [case["relationship"]],
            "context": case["context_original"],
            "option_A": case["option_A"],
            "option_B": case["option_B"],
            "human_option": answer_by_id[case_id],
            "reason": notes["reason"],
            "boundary": notes["boundary"],
        })

    profile = _load(BASE_PROFILE)
    profile.update({
        "name": "judge_v2_scenario_few_shot_candidate_v6",
        "hypothesis": "a narrower restatement veto preserves penalties for redundant explanation without rejecting pragmatic confirmation or self-effacing reformulation",
        "decision_priority": DECISION_PRIORITY,
        "instructions": (
            "判断匿名回复 A、B 中哪个更可能是该聊天中真人本人实际发出的历史回复，而不是挑选质量更高的回复。"
            "必须严格按 reference_pack.decision_priority 的 P0 到 P5 顺序判断；低优先级理由不得推翻高优先级证据。"
            "还必须逐项执行 reference_pack.decision_gates；只有同时满足 obvious_restatement_veto 的两个条件才可记强反证；"
            "命中 weak_identity_evidence_gate 时不得把通用风格信号改写成本人特异证据。"
            "执行 quality_blindfold 时，先忽略礼貌、完整、帮助程度、文采和推进对话能力，再判断真人概率。"
            "scenario_few_shots 是独立开发样本，只示范判断方法和边界，不得按关键词照抄结论。"
            "先逐项写出双方是否重复显然信息、剥离质量信号后的本人特异证据和反证，再写决定性证据与不确定性；"
            "最后逐字复制被选回复并映射到 A/B。"
            "若难分仍须二选一，但必须按 P5 限制置信度。"
        ),
    })
    profile["pattern_cards"] = list(profile.get("pattern_cards", [])) + NEW_PATTERN_CARDS
    reference = _load(BASE_REFERENCE)
    reference["decision_priority"] = DECISION_PRIORITY
    reference["decision_gates"] = DECISION_GATES
    reference["scenario_few_shots"] = scenario_few_shots
    reference["scenario_few_shot_policy"] = {
        "source": "dev_blind.json + dev_answers.json",
        "acceptance_case_answers_used": False,
        "selection": "seven development cases covering group specificity, truly redundant explanation, pragmatic confirmation, self-effacing reformulation, unknown facts, phrase context, and polished-reply bias; no validation answers are used",
    }
    formal_cases = _formal_cases()
    removed = _remove_contaminated_reference_rows(reference, formal_cases)
    reference["scenario_few_shot_policy"]["contamination_filter"] = {
        "test_cases": 240,
        "removed_reference_rows": removed,
        "rule": "remove every top-level example containing a formal human or generated answer, or an embedded formal context-plus-answer sequence",
    }
    reference.pop("reference_sha256", None)
    reference["reference_sha256"] = _json_sha256(reference)
    assert_reference_isolated(reference, formal_cases)
    OUT_PROFILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_REFERENCE.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "profile": str(OUT_PROFILE),
        "reference": str(OUT_REFERENCE),
        "examples": len(reference.get("examples") or []),
        "scenario_few_shots": len(reference.get("scenario_few_shots") or []),
        "removed": removed,
        "reference_sha256": reference["reference_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
