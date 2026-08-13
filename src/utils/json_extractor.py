"""从 LLM 回复中提取 JSON 对象的通用工具。"""

import json
from typing import Any, Dict, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 回复中提取 JSON 对象。支持 markdown 代码块和裸 JSON。

    使用 json.JSONDecoder.raw_decode() 精确解析，避免手动括号计数
    在字符串内遇到 } 时误判 JSON 边界的问题。

    策略：先原样解析，失败再标准化 Unicode 引号重试。避免无条件
    替换破坏值内合法含中文引号的 JSON（如 {"replies": ["他说"好的""]}）。
    """
    text = text.strip()
    if not text:
        return None
    # 第一步：原样解析
    result = _try_decode(text)
    if result is not None:
        return result
    # 第二步：标准化 Unicode 引号后重试（LLM 有时把字符串分隔符写成中文引号）
    text = _normalize_quotes(text)
    return _try_decode(text)


def _normalize_quotes(text: str) -> str:
    """将各种 Unicode 引号替换为 ASCII 引号。"""
    replacements = {
        '“': '"',  # "
        '”': '"',  # "
        '「': '"',  # 「
        '」': '"',  # 」
        '『': '"',  # 『
        '』': '"',  # 』
        '＂': '"',  # ＂
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def _try_decode(text: str) -> Optional[Dict[str, Any]]:
    """尝试从文本中解析 JSON 对象。支持 markdown 代码块。"""
    # 去掉 markdown 代码块
    if "```" in text:
        parts = text.split("```", 2)
        if len(parts) >= 3:
            code_content = parts[1]
            if code_content.lstrip().startswith("json"):
                code_content = code_content.lstrip()[4:].lstrip()
            text = code_content
    # 找到第一个 { 的位置
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
        return obj
    except json.JSONDecodeError:
        # 尝试从每个 { 位置解析（LLM 有时在 JSON 前加了多余文字）
        for idx in range(start, len(text)):
            if text[idx] == '{':
                try:
                    obj, _ = decoder.raw_decode(text, idx)
                    return obj
                except json.JSONDecodeError:
                    continue
        return None
