"""把微信加密 db_storage 解密到缓存目录（gitignored 的 data/wechat/decrypted）。"""

import logging
import subprocess  # nosec B404
import sys
from pathlib import Path

from src.platform.windows.config import VENDOR_KEY_TOOL, get_cache_dir

_logger = logging.getLogger("src.platform.windows.decryptor")

DECRYPTED_DIRNAME = "decrypted"
_FINGERPRINT_FILENAME = ".keys_fingerprint"


def decrypted_dir(cache_dir: Path | None = None) -> Path:
    """解密结果目录（默认 <cache_dir>/decrypted）。"""
    return (cache_dir or get_cache_dir()) / DECRYPTED_DIRNAME


def ensure_decrypted(
    db_storage: Path,
    keys_path: Path,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """确保 db_storage 已解密到缓存目录，返回解密目录。

    密钥文件未变化且已有解密结果时直接复用，避免重复全量解密。
    """
    cache_dir = cache_dir or get_cache_dir()
    out = decrypted_dir(cache_dir)
    fingerprint = cache_dir / _FINGERPRINT_FILENAME
    keys_mtime = str(keys_path.stat().st_mtime)
    if (
        not force
        and out.is_dir()
        and fingerprint.exists()
        and fingerprint.read_text(encoding="utf-8") == keys_mtime
    ):
        _logger.info("解密结果已缓存且密钥未变，跳过解密: %s", out)
        return out

    cache_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(VENDOR_KEY_TOOL), "decrypt",
        "--db-dir", str(db_storage), "--keys", str(keys_path), "--output", str(out),
    ]
    _logger.info("解密微信数据库 -> %s", out)
    try:
        _run_vendor(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"解密失败: {exc}") from exc
    fingerprint.write_text(keys_mtime, encoding="utf-8")
    return out


def _run_vendor(cmd: list[str]) -> None:
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)  # nosec B603
