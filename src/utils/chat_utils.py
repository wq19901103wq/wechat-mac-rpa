#!/usr/bin/env python3
"""聊天相关的通用工具函数：群聊判断、名称归一化等。

所有涉及 chat_name 的处理逻辑统一放在这里，禁止各模块自己写正则。
"""




def _safe_filename(name: str) -> str:
    """把聊天名转成安全的文件名（替换非法字符，限制长度）。"""
    invalid = '<>:"/\\|?*'
    for c in invalid:
        name = name.replace(c, '_')
    # 限制长度，保留尾部用于可读性
    if len(name) > 180:
        name = name[:180]
    return name


def _is_group_chat_name(chat_name: str) -> bool:
    """判断聊天名称是否为群聊（以群人数结尾，如 '示例交流群（128）' 或 'xxx (5)'）。"""
    if not chat_name:
        return False
    # 找到最后一个 '(' 或 '（'，检查后面是否是数字 + 对应闭合括号
    for open_ch, close_ch in (('(', ')'), ('（', '）')):
        idx = chat_name.rfind(open_ch)
        if idx != -1 and chat_name.endswith(close_ch):
            mid = chat_name[idx + 1:-1]
            if mid.isdigit():
                return True
    return False


def _normalize_chat_name(name: str) -> str:
    """对聊天名称进行 Unicode 归一化，防止 OCR 差异导致 session 分裂。

    群聊名通常以群人数结尾（如 '示例交流群（128）'），
    去掉后缀得到稳定的群聊标识（用于 session key）。
    """
    if not name:
        return ""
    name = name.replace("(", "（").replace(")", "）")
    name = name.replace("—", "—").replace("–", "—")
    name = name.replace(" ", "").replace("\u00a0", "").replace("\t", "")
    # 去掉开头的序号前缀（如 "1. 群名"、"1、群名"、"1 群名"）
    i = 0
    while i < len(name) and (name[i].isdigit() or name[i] in '.、 \t'):
        i += 1
    name = name[i:]
    # 去掉群人数后缀（如 '示例交流群（128）' → '示例交流群'）
    idx = name.rfind('（')
    if idx != -1 and name.endswith('）') and name[idx + 1:-1].isdigit():
        name = name[:idx]
    return name.strip()


def _extract_session_key(chat_name: str) -> str:
    """从原始 chat_name 提取 session key（用于 GlobalStore 索引）。

n    等价于 _normalize_chat_name，但语义更明确。
    """
    return _normalize_chat_name(chat_name)
