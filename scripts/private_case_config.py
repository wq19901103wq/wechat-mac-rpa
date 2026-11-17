"""Load local-only identities and frozen case selections for historical scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PRIVATE_CASE_CONFIG = (
    Path(__file__).resolve().parent.parent / "data/private_benchmarks/case_config.json"
)


def load_private_case_config(*required_keys: str) -> dict[str, Any]:
    try:
        config = json.loads(PRIVATE_CASE_CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(
            "Missing local data/private_benchmarks/case_config.json; "
            "restore the private config or prepare it from config/private_case_config.json.example"
        ) from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("Cannot read local private case config as JSON") from None
    if not isinstance(config, dict):
        raise RuntimeError("Private case config must be a JSON object")
    for key in required_keys:
        value = config.get(key)
        expected_type = (
            dict if key == "user_confirmation"
            else list if key in {
                "pending_confirmation_server_ids", "user_confirmed_human_server_ids",
                "selected_anchors", "excluded_exposed_anchors", "case_ids",
            }
            else str
        )
        if not isinstance(value, expected_type) or (isinstance(value, str) and not value.strip()):
            raise RuntimeError(f"Missing or invalid private case config field: {key}")
    return config
