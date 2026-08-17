"""Windows 微信数据目录与缓存目录探测。

微信文件管理可改到任意盘符/目录，默认路径（Documents\\xwechat_files）不一定存在，
因此提供自动探测 + 环境变量覆盖（WECHAT_DATA_DIR / WECHAT_CACHE_DIR）。
"""

import logging
import os
from pathlib import Path

_logger = logging.getLogger("src.platform.windows.config")

ENV_DATA_DIR = "WECHAT_DATA_DIR"
ENV_CACHE_DIR = "WECHAT_CACHE_DIR"
DEFAULT_CACHE_DIR = Path("data") / "wechat"


def repo_root() -> Path:
    """仓库根目录（src/platform/windows/ 上溯三级）。"""
    return Path(__file__).resolve().parents[3]


# vendor 的 wcdb-key-tool（MIT）Windows 只读密钥提取/解密脚本
VENDOR_KEY_TOOL = repo_root() / "third_party" / "wcdb-key-tool" / "wcdb_key_tool_windows.py"


def get_db_storage_dir() -> Path:
    """返回微信 db_storage 目录（加密 .db 所在目录）。

    优先使用环境变量 WECHAT_DATA_DIR；未设置或指向无效时自动探测。
    """
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path
        _logger.warning("%s 指向的目录不存在: %s", ENV_DATA_DIR, path)
    return auto_detect_db_storage_dir()


def auto_detect_db_storage_dir() -> Path:
    """在常见位置搜索 xwechat_files/<账号>/db_storage，返回最近活跃的账号。"""
    candidates: list[Path] = []
    seen: set[str] = set()
    home = Path.home()
    roots = [
        home / "Documents" / "xwechat_files",
        home / "xwechat_files",
    ]
    for drive in "DEFGHIJKLMNOPQRSTUVWXYZ":
        roots.append(Path(f"{drive}:\\") / "Users" / home.name / "xwechat_files")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for account in root.iterdir():
                db_storage = account / "db_storage"
                if not db_storage.is_dir():
                    continue
                key = os.path.normcase(str(db_storage))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(db_storage)
        except OSError:
            continue
    if not candidates:
        raise RuntimeError(
            f"未找到微信数据目录 db_storage；请设置环境变量 {ENV_DATA_DIR} 指向它"
        )
    candidates.sort(key=_account_activity_mtime, reverse=True)
    chosen = candidates[0]
    _logger.info("自动探测到微信数据目录: %s", chosen)
    return chosen


def _account_activity_mtime(db_storage: Path) -> float:
    """以 message 目录 mtime 近似账号活跃度（best-effort）。"""
    message_dir = db_storage / "message"
    target = message_dir if message_dir.is_dir() else db_storage
    try:
        return target.stat().st_mtime
    except OSError:
        return 0.0


def get_cache_dir() -> Path:
    """返回解密缓存根目录（默认 data/wechat，可用 WECHAT_CACHE_DIR 覆盖）。"""
    env = os.environ.get(ENV_CACHE_DIR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR
