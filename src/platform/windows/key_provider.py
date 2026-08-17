"""微信数据库密钥提取（只读，无注入）。

复用 third_party/wcdb-key-tool（MIT）的 Windows 只读 Config.Cipher 扫描：
定位进程内 com.Tencent.WCDB.Config.Cipher 对象 -> XOR 解码候选 key ->
对每个 .db 第一页做 HMAC-SHA512 校验。微信 4.1+ 不再缓存明文 x'<hex>'，
这是当前唯一不注入、不重签的提 key 路线。
"""

import logging
import subprocess  # nosec B404
import sys
from pathlib import Path

from src.platform.windows.config import ENV_CACHE_DIR, VENDOR_KEY_TOOL, get_cache_dir

_logger = logging.getLogger("src.platform.windows.key_provider")

KEYS_FILENAME = "keys.json"


def keys_cache_path(cache_dir: Path | None = None) -> Path:
    """密钥缓存文件路径（默认 <cache_dir>/keys.json）。"""
    return (cache_dir or get_cache_dir()) / KEYS_FILENAME


def extract_keys(db_storage: Path, cache_dir: Path | None = None) -> Path:
    """提取密钥并缓存；微信不可用时回退旧缓存。

    Returns:
        密钥文件路径（<cache_dir>/keys.json）。
    """
    cache_dir = cache_dir or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = keys_cache_path(cache_dir)
    cmd = [
        sys.executable, str(VENDOR_KEY_TOOL), "extract",
        "--db-dir", str(db_storage), "--output", str(out),
    ]
    _logger.info("提取微信数据库密钥: %s", db_storage)
    try:
        _run_vendor(cmd)
    except subprocess.CalledProcessError as exc:
        _logger.warning("提 key 失败（%s），尝试回退旧缓存 %s", exc, out)
        if out.exists():
            return out
        raise RuntimeError(
            f"提 key 失败且无可用缓存：请确认微信已登录并运行（{exc}）"
        ) from exc
    if not out.exists():
        raise RuntimeError(f"提 key 未产出密钥文件: {out}")
    _logger.info("密钥已缓存: %s（可在 %s 覆盖缓存目录）", out, ENV_CACHE_DIR)
    return out


def _run_vendor(cmd: list[str]) -> None:
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)  # nosec B603
