"""测试会话级配置：隔离生产日志目录，防止测试输出污染 data/logs。"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# 必须在首个 get_logger() 调用前设置，使 BotLogger 默认目录指向临时目录
os.environ.setdefault(
    "WECHAT_BOT_LOGS_DIR",
    tempfile.mkdtemp(prefix="wechat-bot-test-logs-"),
)


@pytest.fixture
def synthetic_private_case_config(monkeypatch, tmp_path):
    from scripts import private_case_config

    example = Path(__file__).resolve().parents[2] / "config/private_case_config.json.example"
    config = json.loads(example.read_text(encoding="utf-8"))
    private_path = tmp_path / "case_config.json"
    private_path.write_text(json.dumps(config), encoding="utf-8")
    private_path.chmod(0o600)
    monkeypatch.setattr(private_case_config, "PRIVATE_CASE_CONFIG", private_path)
    return config


def pytest_addoption(parser):
    parser.addoption("--run-api", action="store_true", default=False, help="Run with real API calls")
    parser.addoption("--n-runs", type=int, default=3, help="Number of runs per case")
