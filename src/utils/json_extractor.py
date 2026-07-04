"""从 LLM 回复中提取 JSON 对象的通用工具。"""

import json
from typing import Optional, Dict, Any


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 回复中提取 JSON 对象。支持 markdown 代码块和裸 JSON。

    使用 json.JSONDecoder.raw_decode() 精确解析，避免手动括号计数
    在字符串内遇到 } 时误判 JSON 边界的问题。
    """
    text = text.strip()
    if not text:
        return None
    # 去掉 markdown 代码块
    if "```" in text:
        parts = text.split("```", 2)
        if len(parts) >= 3:
            code_content = parts[1]
            # 去掉可能的 "json" 语言标记
            if code_content.lstrip().startswith("json"):
                code_content = code_content.lstrip()[4:].lstrip()
            text = code_content
    # 找到第一个 { 的位置，尝试 raw_decode
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
        return obj
    except json.JSONDecodeError:
        # 如果 raw_decode 失败，尝试清理常见问题后重试
        # 例如 LLM 在 JSON 前后加了多余文字
        for idx in range(start, len(text)):
            if text[idx] == '{':
                try:
                    obj, _ = decoder.raw_decode(text, idx)
                    return obj
                except json.JSONDecodeError:
                    continue
        return None
