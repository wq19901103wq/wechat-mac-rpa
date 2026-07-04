"""截图持久化与清理工具。

原位于 src/session/global_store.py，拆分出来避免 GlobalStore 职责过重。
"""

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_logger = logging.getLogger("src.utils.screenshot_store")

# 截图保留策略：超过此秒数的旧截图将被清理（默认 1 天）
SCREENSHOT_MAX_AGE_SECONDS = 86400
# 清理检查间隔：每这么多秒执行一次清理，避免每次保存都遍历目录
_SCREENSHOT_CLEANUP_INTERVAL = 600


class ScreenshotStore:
    """管理截图保存和过期清理。"""

    def __init__(
        self,
        screenshots_dir: Path,
        max_age_seconds: int = SCREENSHOT_MAX_AGE_SECONDS,
        cleanup_interval: int = _SCREENSHOT_CLEANUP_INTERVAL,
    ):
        self._screenshots_dir = screenshots_dir
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._max_age_seconds = max_age_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = 0.0

    def save(self, image_path: str, session_id: Optional[str] = None) -> str:
        """保存截图到目标目录并触发过期清理。"""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"wechat_{session_id}_{timestamp}.png"
        filepath = self._screenshots_dir / filename
        shutil.copy2(image_path, filepath)
        self.cleanup_old_screenshots()
        return str(filepath)

    def cleanup_old_screenshots(self) -> None:
        """清理超过保留期的旧截图。"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        cutoff = now - self._max_age_seconds
        try:
            removed = 0
            for entry in self._screenshots_dir.iterdir():
                if not entry.is_file() or entry.suffix != ".png":
                    continue
                try:
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                        removed += 1
                except OSError as e:
                    _logger.debug("删除旧截图失败 %s: %s", entry.name, e)
            if removed:
                _logger.info(
                    "[ScreenshotStore] 清理旧截图 %d 张（保留期 %d 秒）",
                    removed,
                    self._max_age_seconds,
                )
        except OSError as e:
            _logger.debug("[ScreenshotStore] 清理截图目录失败: %s", e)


def get_default_screenshot_store(project_root: Optional[Path] = None) -> ScreenshotStore:
    """返回默认截图存储（data/screenshots/）。"""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    return ScreenshotStore(project_root / "data" / "screenshots")
