#!/usr/bin/env python3
"""wechat-twin Bot — 合并重构版本"""
import logging
import sys, os, fcntl, signal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 加载 .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.bot.wechat_bot import WeChatBot
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.perception.smart_pipeline import SmartPerceptionPipeline
from src.utils.qwen_client import QwenClient

_logger = logging.getLogger(__name__)


class SingleInstanceLock:
    def __init__(self, pid_file=""):
        if not pid_file:
            pid_file = str(Path(__file__).parent / "bot.pid")
        self.pid_file = pid_file

    def __enter__(self):
        old_pid = ""
        try:
            with open(self.pid_file) as f: old_pid = f.read().strip()
        except Exception as e:
            _logger.warning("read pid file failed: %s", e)
        try:
            self.fd = open(self.pid_file, "r+")
        except FileNotFoundError:
            self.fd = open(self.pid_file, "w+")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            self.fd.close()
            print(f"❌ Bot 已在运行 (PID {old_pid or 'unknown'})")
            sys.exit(1)
        self.fd.seek(0); self.fd.truncate()
        self.fd.write(str(os.getpid())); self.fd.flush()
        return self

    def __exit__(self, *args):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            try: os.remove(self.pid_file)
            except Exception as e:
                _logger.warning("remove pid file failed: %s", e)


def main():
    with SingleInstanceLock():
        print("=" * 60)
        print("🤖 wechat-twin Bot")
        print("=" * 60)
        print("配置:")
        model_name = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        print(f"  • LLM: {model_name} ({base_url})")
        print("  • Prompt: DT style (data/persona.md)")
        print("  • 检索: 待启用")
        interval = 5.0
        print(f"  • 轮询: 每 {interval:.0f} 秒")

        llm = QwenClient()
        perception = SmartPerceptionPipeline(PROFILE_WECHAT_MAC_1760X1280)
        bot = WeChatBot(
            profile=PROFILE_WECHAT_MAC_1760X1280,
            llm_client=llm,
            perception=perception,
        )

        def _request_stop(signum, _frame):
            print(f"\n👋 收到信号 {signum}，将在当前 tick 结束后停止...")
            bot.running = False

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)

        try:
            bot.run_auto(interval=interval)
        except KeyboardInterrupt:
            bot.running = False
        finally:
            print("\n👋 正在保存状态并停止后台任务...")
            bot.save_sessions()
            if hasattr(bot, 'memory_engine') and bot.memory_engine:
                bot.memory_engine.shutdown()
            bot.running = False


if __name__ == "__main__":
    main()
