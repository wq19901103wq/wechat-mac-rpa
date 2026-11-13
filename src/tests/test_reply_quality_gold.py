from src.benchmarks.reply_quality_gold import build_human_gold_packet


def _case(case_id, action, is_group):
    return {
        "case_id": case_id,
        "is_group": is_group,
        "pre_context": [{"text": f"context-{case_id}"}],
        "target_messages": [{"text": f"target-{case_id}"}],
        "historical_system_evidence": {
            "action": action,
            "replies": [f"reply-{case_id}"] if action == "replied" else [],
        },
    }


def test_gold_sampling_covers_action_and_chat_strata_without_labels():
    packet = {
        "source_fingerprint": "source",
        "cases": [
            _case("a", "replied", True),
            _case("b", "replied", False),
            _case("c", "filtered_before_generation", True),
            _case("d", "filtered_before_generation", False),
            _case("e", "generated_silence_or_send_failed", False),
        ],
    }
    manifest, private = build_human_gold_packet(packet, 4)
    assert manifest["selected_cases"] == 4
    assert set(manifest["selected_strata"]) == {
        "reply_group",
        "reply_private",
        "silence_group",
        "silence_private",
    }
    assert manifest["excluded_ambiguous_actions"] == {
        "generated_silence_or_send_failed": 1
    }
    assert all(case["human_review"]["status"] == "pending" for case in private["cases"])
    assert all(case["human_review"]["reason"] is None for case in private["cases"])


def test_gold_sampling_is_deterministic():
    cases = [_case(str(index), "replied", bool(index % 2)) for index in range(8)]
    first = build_human_gold_packet({"cases": cases}, 3)[1]
    second = build_human_gold_packet({"cases": list(reversed(cases))}, 3)[1]
    assert [case["case_id"] for case in first["cases"]] == [
        case["case_id"] for case in second["cases"]
    ]
