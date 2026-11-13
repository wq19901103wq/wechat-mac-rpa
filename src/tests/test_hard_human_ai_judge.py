from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_hard_human_ai_judge import (
    build_imitation_prompt,
    parse_replies,
    select_hard_holdouts,
    style_references,
)
from scripts.evaluate_human_ai_judge import _candidate_identity, _case_id


def _candidate(reply, timestamp, chat_id="same"):
    return Candidate(
        relation="group",
        chat_id=chat_id,
        context=["上下文" + str(timestamp)],
        replies=[reply],
        score=1,
        timestamp=timestamp,
    )


def test_style_references_exclude_all_targets_and_duplicate_target_reply():
    target = _candidate("真人答案", 100)
    other_target = _candidate("另一个答案", 90)
    duplicate = _candidate("真人答案", 80)
    safe = [_candidate("参考" + str(i), i) for i in range(6)]
    targets = {_candidate_identity(target), _candidate_identity(other_target)}
    refs = style_references(target, [target, other_target, duplicate, *safe], targets)

    assert len(refs) == 6
    assert all(_candidate_identity(row) not in targets for row in refs)
    assert all("真人答案" not in row.replies for row in refs)


def test_imitation_prompt_never_contains_target_reply():
    target = _candidate("保密答案", 100)
    prompt = build_imitation_prompt(target, [_candidate("参考语气", 1)])
    assert "保密答案" not in str(prompt)
    assert "参考语气" in str(prompt)


def test_parse_replies_accepts_wrapped_json_and_rejects_plain_text():
    assert parse_replies('说明\n{"replies":["一句", "二句"]}\n完') == ["一句", "二句"]
    assert parse_replies("普通回复") == []


def test_hard_holdouts_exclude_previously_seen_cases():
    rows = [_candidate("参考" + str(i), i) for i in range(1, 8)]
    seen = rows[-1]
    selected = select_hard_holdouts(
        rows,
        excluded=set(),
        per_relation=1,
        minimum_prior_references=1,
        excluded_case_ids={_case_id(seen)},
        minimum_context_messages=1,
    )
    assert seen not in selected
