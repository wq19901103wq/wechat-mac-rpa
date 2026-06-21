#!/usr/bin/env python3
"""文本处理通用工具函数。"""

import re


def _truncate_text(text: str, max_len: int, suffix: str = "\n\n... [truncated]") -> str:
    """截断文本到指定长度，保留尾部提示。"""
    if not text or len(text) <= max_len:
        return text
    return text[:max_len] + suffix


def _compress_text(text: str, max_chars: int) -> str:
    """压缩长文本：优先保留开头和结尾，中间用省略号连接。"""
    if not text or len(text) <= max_chars:
        return text
    # 保留头 40% + 尾 60%
    head_len = int(max_chars * 0.4)
    tail_len = int(max_chars * 0.6)
    return text[:head_len] + "\n...（中间省略）...\n" + text[-tail_len:]


def extract_context_around_keywords(
    text: str,
    keywords: list[str],
    max_chars: int,
    context_radius: int = 500,
    fallback_to_compress: bool = True,
) -> str:
    """根据关键词在文本中提取相关上下文，替代粗暴的从头截断。

    策略：
    1. 在文本中搜索所有关键词出现的位置（忽略大小写、去除常见标点）
    2. 每个匹配位置提取前后 context_radius 字符的上下文窗口
    3. 合并重叠/相邻的窗口
    4. 按匹配关键词数量排序，在 max_chars 预算内保留最相关的窗口
    5. 窗口之间用省略标记连接，标明跳过了多少字符

    如果没有匹配到任何关键词，fallback 到 _compress_text（保留头尾）。

    Args:
        text: 原始长文本
        keywords: 关键词列表（如用户问题里的名词、查询词）
        max_chars: 总字符预算上限
        context_radius: 每个匹配点前后保留的字符数
        fallback_to_compress: 无匹配时是否 fallback 到头尾压缩

    Returns:
        截取后的上下文文本
    """
    if not text or len(text) <= max_chars:
        return text

    if not keywords:
        return _compress_text(text, max_chars) if fallback_to_compress else text[:max_chars]

    # 清理关键词：去空白、转小写、去除常见标点
    clean_kws = []
    for kw in keywords:
        ck = kw.strip().lower()
        ck = re.sub(r"[，。！？、；：\"'（）【】\[\]{}]", '', ck)
        if ck and len(ck) >= 2:  # 忽略单字和空串
            clean_kws.append(ck)

    if not clean_kws:
        return _compress_text(text, max_chars) if fallback_to_compress else text[:max_chars]

    # 在文本中搜索每个关键词的出现位置
    text_lower = text.lower()
    matches = []  # [(start, end, matched_kw), ...]
    for kw in clean_kws:
        for m in re.finditer(re.escape(kw), text_lower):
            matches.append((m.start(), m.end(), kw))

    if not matches:
        return _compress_text(text, max_chars) if fallback_to_compress else text[:max_chars]

    # 按位置排序
    matches.sort(key=lambda x: x[0])

    # 构建上下文窗口并合并重叠/相邻的窗口
    windows = []
    for start, end, kw in matches:
        w_start = max(0, start - context_radius)
        w_end = min(len(text), end + context_radius)
        windows.append((w_start, w_end))

    # 合并重叠或相邻（间隔 < context_radius // 2）的窗口
    merged: list[list[int]] = []
    for w_start, w_end in windows:
        if not merged:
            merged.append([w_start, w_end])
            continue
        last = merged[-1]
        if w_start <= last[1] + context_radius // 2:
            last[1] = max(last[1], w_end)
        else:
            merged.append([w_start, w_end])

    # 计算每个窗口的匹配密度（关键词出现次数），用于排序
    def window_score(window):
        w_start, w_end = window
        count = sum(1 for s, e, _ in matches if s >= w_start and e <= w_end)
        return count

    merged.sort(key=window_score, reverse=True)

    # 在 max_chars 预算内取窗口，优先匹配密度高的
    result_parts: list[str] = []
    budget = max_chars
    last_end = 0

    # 先取第一个窗口（通常最重要），完整保留
    for window in merged:
        w_start, w_end = window
        segment = text[w_start:w_end]
        # 如果窗口在开头，不需要前省略号；否则加
        prefix = ""
        if w_start > 0 and not result_parts:
            prefix = f"...（前略 {w_start} 字符）...\n\n"
        elif w_start > 0 and result_parts:
            prefix = f"\n\n...（省略 {w_start - last_end} 字符）...\n\n"

        needed = len(prefix) + len(segment)
        if needed > budget:
            # 预算不够，尝试截断当前窗口的后半部分
            remaining = budget - len(prefix)
            if remaining > 100:
                segment = segment[:remaining] + "\n（…当前段落截断）"
                result_parts.append(prefix + segment)
            break

        result_parts.append(prefix + segment)
        budget -= needed
        last_end = w_end

        if budget <= 0:
            break

    if not result_parts:
        # 极端情况：一个窗口都塞不下，fallback
        return _compress_text(text, max_chars) if fallback_to_compress else text[:max_chars]

    result = "".join(result_parts)
    # 如果最后一个窗口没覆盖到结尾，加尾部省略
    last_window_end = merged[len(result_parts) - 1][1] if result_parts else 0
    if last_window_end < len(text):
        omitted = len(text) - last_window_end
        result += f"\n\n...（后略 {omitted} 字符）..."

    return result
