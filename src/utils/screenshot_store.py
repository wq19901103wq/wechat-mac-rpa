"""RPA screenshot path compatibility."""
from pathlib import Path
from wechat_reader.utils.screenshot_store import ScreenshotStore, SCREENSHOT_MAX_AGE_SECONDS

def get_default_screenshot_store(project_root=None):
    return ScreenshotStore((Path(project_root) if project_root else Path(__file__).parent.parent.parent) / "data/screenshots")
