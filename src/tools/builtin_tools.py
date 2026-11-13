"""内置工具 - 时间、天气、搜索"""


from datetime import datetime
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/digital-human"))
from digital_human.tools import web as _web_tools

from .print_3d_tools import register_print3d_tools
from .stock_tools import stock_query
from .tool_registry import get_registry
from .tuya_tools import register_tuya_tools


_DIRECT_PROXIES = {"http": "", "https": ""}


def _get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S %A")


def _get_weather(city: str = "", date: str = "今天") -> str:
    """获取指定城市的天气"""
    if not city:
        return "请提供城市名称"
    try:
        # 使用 wttr.in 免费天气 API
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        resp = requests.get(url, timeout=10, proxies=_DIRECT_PROXIES)
        data = resp.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        desc = current["lang_zh"][0]["value"] if "lang_zh" in current else current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]
        return f"{city} {date}：{desc}，{temp}℃，湿度{humidity}%，风速{wind}km/h"
    except Exception as e:
        return f"获取{city}天气失败: {e}"


def _web_search(query: str = "", *, structured=False, fetch=None):
    return _web_tools.web_search(query, structured=structured, fetch=fetch or requests.get)


def _browse_url(url: str = "", *, structured=False, fetch=None):
    return _web_tools.browse_url(url, structured=structured, fetch=fetch or requests.get)


def _search_in_page(url: str = "", keyword: str = "") -> str:
    return _web_tools.search_in_page(url, keyword)


def register_builtin_tools(registry=None):
    """注册所有内置工具"""
    if registry is None:
        registry = get_registry()

    registry.register(
        name="get_current_time",
        description="获取当前日期和时间",
        parameters={
            "type": "object",
            "properties": {},
        },
        func=_get_current_time,
    )

    registry.register(
        name="get_weather",
        description="获取指定城市的天气信息。日期可选，默认为今天。",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'上海'、'北京'、'东京'",
                },
                "date": {
                    "type": "string",
                    "description": "日期，如'今天'、'明天'、'后天'，默认为今天",
                },
            },
            "required": ["city"],
        },
        func=_get_weather,
    )


    functions = dict(web_search=_web_search, browse_url=_browse_url, search_in_page=_search_in_page)
    for schema in _web_tools.WEB_SCHEMAS:
        definition = schema['function']
        registry.register(**definition, func=functions[definition['name']])

    registry.register(
        name="stock_query",
        description="查询股票实时行情。支持A股（sh600519/sz000001）、港股（hk00700）、美股（AAPL）。多个代码用逗号分隔。",
        parameters={
            "type": "object",
            "properties": {
                "stock_code": {
                    "type": "string",
                    "description": "股票代码，如 sh600519、sz000001、hk00700、AAPL。多个用逗号分隔。",
                },
            },
            "required": ["stock_code"],
        },
        func=stock_query,
    )

    # 注册 Tuya 智能家居工具
    register_tuya_tools(registry)

    # 注册 3D 打印工具
    register_print3d_tools(registry)
