import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import private_case_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("contents", [None, "secret-invalid-json", "[]", "{}", '{"self_sender": ""}', '{"self_sender": ["secret-value"]}'])
def test_invalid_private_config_fails_without_exposing_contents(tmp_path, monkeypatch, contents):
    path = tmp_path / "case_config.json"
    monkeypatch.setattr(private_case_config, "PRIVATE_CASE_CONFIG", path)
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError) as error:
        private_case_config.load_private_case_config("self_sender")

    assert "secret" not in str(error.value)
    if contents is None or contents == "secret-invalid-json":
        assert error.value.__suppress_context__


def test_private_collection_fields_require_their_expected_types(tmp_path, monkeypatch):
    path = tmp_path / "case_config.json"
    monkeypatch.setattr(private_case_config, "PRIVATE_CASE_CONFIG", path)
    for field, value in (("case_ids", "secret-value"), ("user_confirmation", [])):
        path.write_text(json.dumps({field: value}), encoding="utf-8")
        with pytest.raises(RuntimeError, match=field):
            private_case_config.load_private_case_config(field)


def test_historical_modules_and_help_work_without_private_config(tmp_path):
    modules = (
        "build_adversarial_stage2_holdout", "build_youzi_identity_review",
        "seal_youzi_stage2_quota", "evaluate_current_c0_historical_ab",
        "discover_youzi_judge_candidates", "discover_fixed_judge_candidates",
    )
    script = """
import importlib
import sys
from pathlib import Path
from scripts import private_case_config
private_case_config.PRIVATE_CASE_CONFIG = Path(sys.argv[1])
for name in sys.argv[2:]:
    module = importlib.import_module('scripts.' + name)
    if name in ('build_youzi_identity_review', 'evaluate_current_c0_historical_ab'):
        original_args = sys.argv
        sys.argv = [name, '--help']
        try:
            module.main()
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('help did not exit')
        finally:
            sys.argv = original_args
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "missing.json"), *modules],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script_name", ["build_youzi_identity_review.py", "seal_youzi_stage2_quota.py"])
def test_formerly_standalone_scripts_resolve_shared_config_without_running_main(script_name):
    result = subprocess.run(
        [sys.executable, "-c", """
import runpy
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.argv[1]).parent)
runpy.run_path(sys.argv[1], run_name='privacy_import_check')
""", str(ROOT / "scripts" / script_name)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_synthetic_identity_and_target_group_are_isolated(
    tmp_path, monkeypatch, synthetic_private_case_config,
):
    from scripts import build_youzi_identity_review as review
    from scripts import seal_youzi_stage2_quota as seal

    self_sender = synthetic_private_case_config["self_sender"]
    assert review.is_user_confirmed_human({"senderUsername": self_sender, "isSend": True})
    assert not review.is_user_confirmed_human({"senderUsername": "other", "isSend": True})
    assert not review.is_user_confirmed_human({"senderUsername": self_sender, "isSend": False})

    source = tmp_path / "youzi_group_2_page5k_example.json"
    payload = {"talker": "another-group", "messages": []}
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(seal, "SOURCE_DIR", tmp_path)
    monkeypatch.setattr(seal, "ROOT", tmp_path)
    for load in (lambda: review.load_messages(tmp_path), seal.load_source):
        with pytest.raises(RuntimeError, match="unexpected talker"):
            load()
    payload["talker"] = synthetic_private_case_config["target_chat"]
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert review.load_messages(tmp_path) == []
    assert seal.load_source()[0] == []


def test_historical_sources_only_load_private_config_when_needed(
    tmp_path, monkeypatch, synthetic_private_case_config,
):
    historical = importlib.import_module("scripts.evaluate_current_c0_historical_ab")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "session": {"displayName": "示例会话", "wxid": "example-session"},
        "messages": [],
    }), encoding="utf-8")
    case = {"source_path": str(source)}
    assert historical._source({**case, "specified_group": "correct_common_prosperity_with_luxixi"}) == (
        [], synthetic_private_case_config["common_chat_name"], "example-session",
    )
    monkeypatch.setattr(historical, "load_youzi_source", lambda: ([{"synthetic": True}], []))
    assert historical._source({"source_path": "adversarial_stage2_youzi_quota_v1/manifest.json"}) == (
        [{"synthetic": True}], synthetic_private_case_config["target_chat_name"],
        synthetic_private_case_config["target_chat"],
    )
    monkeypatch.setattr(private_case_config, "PRIVATE_CASE_CONFIG", tmp_path / "missing.json")
    assert historical._source(case) == ([], "示例会话", "example-session")


def test_all_cases_bypasses_private_case_selection(tmp_path, monkeypatch):
    historical = importlib.import_module("scripts.evaluate_current_c0_historical_ab")
    monkeypatch.setattr(private_case_config, "PRIVATE_CASE_CONFIG", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["historical", "--all-cases"])
    monkeypatch.setattr(historical, "_load_env", lambda: None)

    class CasesLoaded(Exception):
        pass

    def load_cases(path, case_ids):
        assert case_ids is None
        raise CasesLoaded

    monkeypatch.setattr(historical, "_load_cases", load_cases)
    with pytest.raises(CasesLoaded):
        historical.main()


def test_existing_local_config_has_private_permissions():
    private_path = private_case_config.PRIVATE_CASE_CONFIG
    if not private_path.exists():
        pytest.skip("Local private config is not required in a fresh checkout")
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600


def test_private_config_is_ignored_and_example_is_shareable():
    private_path = private_case_config.PRIVATE_CASE_CONFIG
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", os.fspath(private_path)], cwd=ROOT,
    )
    assert result.returncode == 0
    example_path = ROOT / "config/private_case_config.json.example"
    result = subprocess.run(["git", "check-ignore", "--quiet", os.fspath(example_path)], cwd=ROOT)
    assert result.returncode == 1
    assert isinstance(json.loads(example_path.read_text(encoding="utf-8")), dict)
