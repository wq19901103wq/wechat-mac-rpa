import json

import pytest

from src.reply.c0_config import current_c0_baseline_id, load_current_c0
from src.reply.c0_factory import create_current_c0_generator


class _Client:
    pass


def test_current_c0_factory_pins_adopted_execution_conditions(monkeypatch):
    monkeypatch.setenv("ENABLE_REACT_TOOLS", "1")
    monkeypatch.setenv("ENABLE_SELF_REFINE", "1")

    generator = create_current_c0_generator(
        private_client=_Client(),
        group_client=_Client(),
        enable_react_tools=True,
        enable_self_refine=True,
    )

    assert generator.enable_react_tools is False
    assert generator.enable_self_refine is False
    assert generator.default_disabled_skills == frozenset({"handling_praise"})
    assert "casual_chat" not in generator.default_disabled_skills


def test_current_c0_rejects_unknown_disabled_skill(tmp_path):
    config = load_current_c0().copy()
    config.pop("config_path", None)
    config.pop("few_shot_path", None)
    config["skills"] = {"disabled": ["not-a-real-skill"]}
    path = tmp_path / "invalid-c0.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unknown"):
        load_current_c0(path)


def test_current_c0_baseline_id_includes_execution_conditions():
    config = load_current_c0()
    baseline_id = current_c0_baseline_id()
    assert config["execution"] == {
        "enable_react_tools": False,
        "enable_self_refine": False,
    }
    assert baseline_id == "c0-" + __import__("hashlib").sha256(
        __import__("json").dumps(
            {
                key: value
                for key, value in config.items()
                if key not in {"config_path", "few_shot_path"}
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
