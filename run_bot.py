#!/usr/bin/env python3
"""wechat-twin Bot — 合并重构版本"""
import logging
import sys, os, fcntl, signal
from pathlib import Path

project_python = Path(__file__).absolute().parent / ".venv" / "bin" / "python"
if (__name__ == "__main__" and project_python.is_file()
        and Path(sys.prefix).absolute() != project_python.parent.parent):
    print(f"使用项目 Python: {project_python}", flush=True)
    os.execv(str(project_python), [str(project_python), "-u", str(Path(__file__).absolute()), *sys.argv[1:]])

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
from src.logging.bot_logger import get_logger
from src.reply.digital_human import load_digital_human_config

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
        digital_human = load_digital_human_config()
        print(f"  • 私聊/群聊: digital-human production ({digital_human['instance']})")
        print(f"  • 仓库: {digital_human['repository']}")
        print(f"  • 请求超时: {digital_human['timeout']}s；失败不回退旧生成器")
        print("  • 消息来源: 微信窗口截图 / OCR")
        interval = 5.0
        print(f"  • 轮询: 每 {interval:.0f} 秒")

        try:
            bot = WeChatBot(
                profile=PROFILE_WECHAT_MAC_1760X1280,
            )
        except Exception:
            get_logger().error("Bot 启动失败", exc_info=True)
            raise

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
            try:
                bot.generator.close()
            except Exception as e:
                _logger.warning("关闭 digital-human 失败: %s", e)
            bot.save_sessions()
            if hasattr(bot, 'memory_engine') and bot.memory_engine:
                bot.memory_engine.shutdown()
            bot.running = False


if __name__ == "__main__":
    main()
