"""测试会话级配置：隔离生产日志目录，防止测试输出污染 data/logs。"""

import os
import tempfile

# 必须在首个 get_logger() 调用前设置，使 BotLogger 默认目录指向临时目录
os.environ.setdefault(
    "WECHAT_BOT_LOGS_DIR",
    tempfile.mkdtemp(prefix="wechat-bot-test-logs-"),
)


def pytest_addoption(parser):
    parser.addoption("--run-api", action="store_true", default=False, help="Run with real API calls")
    parser.addoption("--n-runs", type=int, default=3, help="Number of runs per case")
