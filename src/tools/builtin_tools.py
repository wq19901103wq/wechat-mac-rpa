"""内置工具 - 时间、天气、搜索"""


from datetime import datetime
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .print_3d_tools import register_print3d_tools
from .stock_tools import stock_query
from .tool_registry import get_registry
from .tuya_tools import register_tuya_tools


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
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        desc = current["lang_zh"][0]["value"] if "lang_zh" in current else current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]
        return f"{city} {date}：{desc}，{temp}℃，湿度{humidity}%，风速{wind}km/h"
    except Exception as e:
        return f"获取{city}天气失败: {e}"


def _web_search(query: str = "") -> str:
    """网页搜索（360 搜索，中文查询效果最佳）"""
    if not query:
        return "请提供搜索关键词"
    try:
        from bs4 import BeautifulSoup

        url = "https://www.so.com/s"
        params = {"q": query}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for li in soup.find_all("li", class_="res-list"):
            # 标题和链接
            title = ""
            link = ""
            h3 = li.find("h3")
            if h3:
                a = h3.find("a")
                if a:
                    title = a.get_text(strip=True)
                    link = a.get("href", "")
                    # 优先从 data-mdurl 拿真实 URL（360 跳转链接是加密的）
                    mdurl = a.get("data-mdurl", "")
                    if mdurl and mdurl.startswith(("http://", "https://")):
                        link = mdurl
                    elif link.startswith("https://www.so.com/link?"):
                        parsed = urlparse(link)
                        qs = parse_qs(parsed.query)
                        if "url" in qs:
                            link = unquote(qs["url"][0])

            if not title or len(title) <= 3 or "360" in title.lower():
                continue

            # 摘要
            snippet = ""
            desc = li.find("p", class_="res-desc")
            if desc:
                snippet = desc.get_text(strip=True)
                snippet = ' '.join(snippet.split())
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."

            line = title
            if link:
                line += f"\n   链接：{link}"
            if snippet:
                line += f"\n   摘要：{snippet}"
            results.append(line)

            if len(results) >= 20:
                break

        if results:
            return f"搜索结果（{query}）：\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
        return f"未找到关于'{query}'的搜索结果"
    except Exception as e:
        return f"搜索失败: {e}"


# 页面缓存：URL -> 页面全文（用于 search_in_page 工具）
_PAGE_CACHE: dict = {}

MAX_CACHE_SIZE = 20

def _add_to_cache(url: str, text: str):
    """缓存页面全文，大小限制。"""
    _PAGE_CACHE[url] = text
    if len(_PAGE_CACHE) > MAX_CACHE_SIZE:
        oldest = next(iter(_PAGE_CACHE))
        del _PAGE_CACHE[oldest]


def _search_keyword_in_text(text: str, keyword: str, results: list, is_fuzzy: bool = False, word: str = "") -> list:
    """在 text 中查找 keyword（忽略大小写），将前后 200 字上下文加入 results。
    使用 str.find 替代正则，避免 re 模块依赖。
    返回传入的 results 列表（已追加匹配结果）。
    """
    lower_text = text.lower()
    lower_kw = keyword.lower()
    pos = 0
    while True:
        idx = lower_text.find(lower_kw, pos)
        if idx == -1:
            break
        start = max(0, idx - 200)
        end = min(len(text), idx + len(keyword) + 200)
        ctx = text[start:end]
        if start > 0:
            ctx = "..." + ctx
        if end < len(text):
            ctx = ctx + "..."
        label = f"[关键词'{word}' 位置 {idx}]" if is_fuzzy else f"[位置 {idx}]"
        results.append(f"{label}: {ctx}")
        if len(results) >= 5:
            return results
        pos = idx + len(keyword)
    return results


def _browse_url(url: str = "") -> str:
    """打开指定链接，提取网页正文内容。同时缓存全文供 search_in_page 使用。"""
    if not url:
        return "请提供要浏览的链接"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. 提取 <title>
        title = soup.title.get_text(strip=True) if soup.title else ""

        # 2. 提取正文：按优先级尝试
        body = ""
        parsed_url = urlparse(url)
        if parsed_url.hostname == "mp.weixin.qq.com":
            # 微信公众号文章
            content_div = soup.find("div", id="js_content")
            if content_div:
                body = str(content_div)

        if not body:
            for selector in ["article", "main", '[role="main"]', 'div.content']:
                tag = soup.select_one(selector)
                if tag and len(tag.get_text(strip=True)) > 200:
                    body = str(tag)
                    break

        if not body and soup.body:
            body = str(soup.body)

        # 3. 用 BeautifulSoup 清理标签和脚本/style
        clean_soup = BeautifulSoup(body, "html.parser") if body else BeautifulSoup("", "html.parser")
        for tag in clean_soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = clean_soup.get_text(separator="\n")
        text = '\n'.join(line for line in text.splitlines() if line.strip())
        text = ' '.join(text.split())

        # 4. 缓存全文（供 search_in_page 后续搜索），截断返回
        _add_to_cache(url, text)

        max_len = 12000
        result = text[:max_len]
        if len(text) > max_len:
            result += f"\n\n[全文 {len(text)} 字，已截断。用 search_in_page(url, keyword) 搜索页面内特定内容]"

        preview = f"标题：{title}\n" if title else ""
        preview += f"链接：{url}\n"
        preview += f"正文：{result}"
        return preview
    except Exception as e:
        return f"浏览链接失败: {e}"


def _search_in_page(url: str = "", keyword: str = "") -> str:
    """在已缓存的页面中搜索关键词，返回上下文。"""
    if not url or not keyword:
        return "请提供 url 和 keyword 参数"
    if url not in _PAGE_CACHE:
        return f"页面未缓存。请先用 browse_url 打开 {url}"
    text = _PAGE_CACHE[url]
    # 查找关键词位置，返回前后各 200 字的上下文
    results: list[str] = []
    results = _search_keyword_in_text(text, keyword, results, is_fuzzy=False)
    if not results:
        # 模糊搜索：按空格拆词
        words = keyword.split()
        for w in words:
            results = _search_keyword_in_text(text, w, results, is_fuzzy=True, word=w)
            if results:
                break
    if not results:
        return f"在页面中未找到 '{keyword}'。可尝试其他关键词，或查看缓存页面有 {len(text)} 字。"
    return f"在 {url} 中搜索 '{keyword}'（全文 {len(text)} 字）：\n" + "\n\n".join(results)


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

    registry.register(
        name="browse_url",
        description="打开指定链接，提取网页正文内容。用户分享链接或提到 URL 时使用。",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要浏览的链接地址，如 https://example.com/article",
                },
            },
            "required": ["url"],
        },
        func=_browse_url,
    )

    registry.register(
        name="search_in_page",
        description="在已浏览的网页中搜索关键词，返回上下文（前后各200字）。browse_url 后发现信息被截断时使用。",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "已浏览过的链接地址",
                },
                "keyword": {
                    "type": "string",
                    "description": "要搜索的关键词，如'泰州队'、'积分榜'",
                },
            },
            "required": ["url", "keyword"],
        },
        func=_search_in_page,
    )

    registry.register(
        name="web_search",
        description="在网页上搜索实时信息。遇到任何你不完全了解的产品、车型、技术、新闻、八卦、陌生名词时，必须调用此工具搜索后再评论，禁止基于已有知识猜测具体配置/价格/细节/事实。用于查询公众人物、公司、产品、技术、新闻八卦、陌生名词等本地记忆里没有的内容。不要用于查询身边的亲友同事同学。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
            },
            "required": ["query"],
        },
        func=_web_search,
    )

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
