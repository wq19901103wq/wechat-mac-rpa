#!/usr/bin/env python3
"""股票查询工具 - 调用腾讯财经 API 获取实时行情"""

import ssl
import urllib.request
from typing import Any, Dict


def _fetch_stock(codes: str) -> Dict[str, Any]:
    """调用腾讯财经 API 获取股票数据"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"https://qt.gtimg.cn/q={codes}"
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=10) as resp:
            raw = resp.read().decode("gb2312", errors="ignore")
    except Exception as e:
        return {"error": f"请求失败: {e}"}

    results = {}
    for line in raw.strip().split(";"):
        line = line.strip()
        if not line:
            continue
        # 格式: v_sh600519="1~贵州茅台~600519~..."
        if "=" not in line:
            continue
        prefix, data = line.split("=", 1)
        code = prefix.replace("v_", "").replace("sh", "").replace("sz", "").replace("hk", "")
        data = data.strip('"')
        parts = data.split("~")
        if len(parts) < 10:
            continue

        # 统一解析核心字段
        is_hk = prefix.startswith("v_hk")
        is_us = prefix.startswith("v_us")

        result = {
            "代码": code,
            "名称": parts[1] if len(parts) > 1 else "",
            "当前价": parts[3] if len(parts) > 3 else "",
            "昨收": parts[4] if len(parts) > 4 else "",
            "今开": parts[5] if len(parts) > 5 else "",
            "涨跌额": parts[31] if len(parts) > 31 else "",
            "涨跌幅(%)": parts[32] if len(parts) > 32 else "",
            "成交量": parts[36] if len(parts) > 36 else "",
            "成交额": parts[37] if len(parts) > 37 else "",
            "换手率(%)": parts[38] if len(parts) > 38 else "",
            "市盈率(动)": parts[39] if len(parts) > 39 else "",
            "总市值(亿)": parts[44] if len(parts) > 44 else "",
            "流通市值(亿)": parts[45] if len(parts) > 45 else "",
        }

        # 港股特有字段
        if is_hk and len(parts) > 49:
            result["52周最高"] = parts[48] if len(parts) > 48 else ""
            result["52周最低"] = parts[49] if len(parts) > 49 else ""
            result["英文名"] = parts[46] if len(parts) > 46 else ""
            # 港股时间格式不同
            if len(parts) > 30:
                result["时间"] = parts[30]

        # A 股特有字段
        if not is_hk and not is_us:
            if len(parts) > 34:
                result["最高价"] = parts[42] if len(parts) > 42 else ""
                result["最低价"] = parts[34] if len(parts) > 34 else ""
            if len(parts) > 30:
                result["时间"] = parts[30]

        results[code] = result

    return results


def stock_query(stock_code: str = "") -> str:
    """
    查询股票实时行情。支持 A股(sh600519/sz000001)、港股(hk00700)、美股(AAPL)。
    多个代码用逗号分隔。
    """
    if not stock_code:
        return "请提供股票代码，如 sh600519、hk00700、AAPL"

    # 标准化代码
    codes = []
    for c in stock_code.replace("，", ",").split(","):
        c = c.strip().upper()
        if not c:
            continue
        # 自动补前缀
        if c.isdigit():
            if len(c) == 6:
                if c.startswith(("6", "5", "9")):
                    codes.append(f"sh{c}")
                else:
                    codes.append(f"sz{c}")
            elif len(c) == 5:
                codes.append(f"hk{c}")
        elif c.startswith("HK") and c[2:].isdigit():
            codes.append(c.lower())
        elif c.startswith(("SH", "SZ")) and c[2:].isdigit():
            codes.append(c.lower())
        else:
            # 美股或其他（腾讯财经需要 us 前缀）
            codes.append(f"us{c}")

    if not codes:
        return "无法识别股票代码格式"

    data = _fetch_stock(",".join(codes))
    if "error" in data:
        return data["error"]

    lines = []
    for code, info in data.items():
        lines.append(f"【{info['名称']} ({code})】")
        for k, v in info.items():
            if k != "名称" and k != "代码" and v:
                lines.append(f"  {k}: {v}")
        lines.append("")

    return "\n".join(lines) if lines else "未获取到数据"
