from scripts.audit_judge_split_exposure import audit_exposure


def test_audit_exposure_rejects_any_prior_case_id() -> None:
    ledger = {
        "splits": {
            "validation": {"case_ids": ["a", "b"]},
            "test_a": {"case_ids": ["c"]},
        }
    }

    result = audit_exposure(ledger, {"b", "old"})

    assert result["status"] == "contaminated"
    assert result["splits"]["validation"] == {
        "count": 2,
        "previously_exposed": 1,
        "unexposed": 1,
        "case_id_previously_exposed": 1,
        "event_cluster_previously_exposed": 0,
        "eligible_as_unseen": False,
    }
    assert result["splits"]["test_a"]["eligible_as_unseen"] is True


def test_audit_exposure_rejects_prior_event_with_new_case_id() -> None:
    ledger = {
        "splits": {
            "validation": {
                "case_ids": ["new-case"],
                "event_clusters": ["seen-event"],
            },
            "test_a": {
                "case_ids": ["another-case"],
                "event_clusters": ["new-event"],
            },
        }
    }

    result = audit_exposure(ledger, set(), {"seen-event"})

    assert result["status"] == "contaminated"
    assert result["splits"]["validation"]["case_id_previously_exposed"] == 0
    assert result["splits"]["validation"]["event_cluster_previously_exposed"] == 1
    assert result["splits"]["validation"]["eligible_as_unseen"] is False
    assert result["splits"]["test_a"]["eligible_as_unseen"] is True
