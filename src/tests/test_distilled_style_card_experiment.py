from scripts.replay_distilled_style_card_ab import (
    S4_CASE_IDS,
    S4_STYLE_CARD,
    _install_style_card,
    _load_human_evidence,
    _select_cases,
)


class _DummyGenerator:
    def _system_prompt(self, marker: str) -> str:
        return f"base:{marker}"


def test_style_card_is_appended_without_replacing_current_prompt():
    generator = _DummyGenerator()
    _install_style_card(generator)
    prompt = generator._system_prompt("x")
    assert prompt.startswith("base:x")
    assert prompt.endswith(S4_STYLE_CARD)


def test_style_card_contains_hard_boundaries_and_no_verified_raw_lines(tmp_path):
    evidence = tmp_path / "human.jsonl"
    evidence.write_text(
        '{"context":["我就当是夸奖啦"],"reply":["不 是诋毁"],"timestamp":1,'
        '"semantic_profile":{"response_move":"premise_reversal","tone":"dry_counter"}}\n',
        encoding="utf-8",
    )
    rows, summary = _load_human_evidence(evidence)
    assert rows[0]["reply"][0] not in S4_STYLE_CARD
    assert summary["row_count"] == 1
    assert "Do not import people, numbers, objects, events, transactions" in S4_STYLE_CARD
    assert "Do not make believable promises" in S4_STYLE_CARD
    assert "Do not use humor in serious distress" in S4_STYLE_CARD


def test_fixed_case_selection_is_ordered_and_complete():
    cases = [{"case_id": case_id} for case_id in reversed(S4_CASE_IDS)]
    assert [case["case_id"] for case in _select_cases(cases)] == list(S4_CASE_IDS)


def test_fixed_case_selection_rejects_missing_case():
    try:
        _select_cases([{"case_id": S4_CASE_IDS[0]}])
    except ValueError as exc:
        assert "fixed S4 cases missing" in str(exc)
    else:
        raise AssertionError("missing fixed cases must fail")
