from scripts.freeze_judge_dataset_splits import QUOTAS, freeze_splits


def _case(index: int, category: str) -> dict:
    case = {
        "internal_id": f"case-{index}",
        "event_cluster": f"event-{index}",
        "chat_type": "group",
        "complex_large_group": False,
        "specified_group": None,
    }
    if category == "specified_common":
        case["specified_group"] = "correct_common_prosperity_with_luxixi"
    elif category == "specified_youzi":
        case["specified_group"] = "youzi_group_2"
    elif category == "private":
        case["chat_type"] = "private"
    elif category == "other_group_complex":
        case["complex_large_group"] = True
    return case


def test_freeze_splits_are_fixed_disjoint_and_balanced() -> None:
    category_totals = {
        category: sum(split[category] for split in QUOTAS.values())
        for category in next(iter(QUOTAS.values()))
    }
    cases = []
    index = 0
    for category, count in category_totals.items():
        for _ in range(count):
            cases.append(_case(index, category))
            index += 1

    first = freeze_splits({"cases": cases}, seed="fixed")
    second = freeze_splits({"cases": list(reversed(cases))}, seed="fixed")

    assert first == second
    assert first["development"]["count"] == 180
    assert {name: split["count"] for name, split in first["splits"].items()} == {
        "fixed_test": 240,
    }
    assert {
        name: split["category_counts"] for name, split in first["splits"].items()
    } == QUOTAS

    ids = [
        case_id
        for split in first["splits"].values()
        for case_id in split["case_ids"]
    ]
    events = [
        event
        for split in first["splits"].values()
        for event in split["event_clusters"]
    ]
    assert len(ids) == len(set(ids)) == 240
    assert len(events) == len(set(events)) == 240
