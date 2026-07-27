#!/usr/bin/env python3
"""Tools 模块单元测试 —— ToolRegistry + builtin_tools + stock_tools"""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.builtin_tools import _browse_url, _get_current_time, _get_weather, _web_search, register_builtin_tools
from src.tools.stock_tools import _fetch_stock, stock_query
from src.tools.tool_registry import Tool, ToolRegistry, get_registry


class TestTool:
    def test_creation(self):
        def dummy_func(x: int) -> int:
            return x * 2

        tool = Tool("double", "Double a number", {"type": "object", "properties": {"x": {"type": "integer"}}}, dummy_func)
        assert tool.name == "double"
        assert tool.description == "Double a number"

    def test_to_openai_schema(self):
        def dummy_func():
            pass

        tool = Tool("test", "desc", {"type": "object"}, dummy_func)
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test"

    def test_execute_success(self):
        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool("add", "Add two numbers", {"type": "object"}, add)
        result = tool.execute('{"a": 1, "b": 2}')
        assert result == "3"

    def test_execute_empty_args(self):
        def noop() -> str:
            return "ok"

        tool = Tool("noop", "No op", {"type": "object"}, noop)
        result = tool.execute("")
        assert result == "ok"

    def test_execute_error(self):
        def fail():
            raise ValueError("boom")

        tool = Tool("fail", "Always fails", {"type": "object"}, fail)
        result = tool.execute("")
        assert "工具执行出错" in result
        assert "boom" in result


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register("foo", "foo tool", {"type": "object"}, lambda: "bar")
        assert reg.has("foo")
        tool = reg.get("foo")
        assert tool.name == "foo"

    def test_get_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("missing")

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register("a", "A", {"type": "object"}, lambda: 1)
        reg.register("b", "B", {"type": "object"}, lambda: 2)
        assert len(reg.list_tools()) == 2

    def test_to_openai_schemas(self):
        reg = ToolRegistry()
        reg.register("x", "X", {"type": "object"}, lambda: 1)
        schemas = reg.to_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "x"

    def test_global_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2


class TestBuiltinTools:
    def test_get_current_time_format(self):
        result = _get_current_time()
        assert "年" in result
        assert "月" in result
        assert "日" in result

    @patch("src.tools.builtin_tools.requests.get")
    def test_get_weather_success(self, mock_get):
        mock_get.return_value.json.return_value = {
            "current_condition": [{
                "temp_C": "25",
                "lang_zh": [{"value": "晴"}],
                "humidity": "60",
                "windspeedKmph": "10",
            }]
        }
        result = _get_weather("上海")
        assert "上海" in result
        assert "25℃" in result

    def test_get_weather_empty_city(self):
        result = _get_weather("")
        assert "请提供城市名称" in result

    @patch("src.tools.builtin_tools.requests.get")
    def test_web_search_success(self, mock_get):
        mock_get.return_value.text = '''
        <li class="res-list">
            <h3><a href="https://example.com" data-mdurl="https://example.com">Example Title</a></h3>
            <p class="res-desc">This is a snippet.</p>
        </li>
        '''
        result = _web_search("test")
        assert "Example Title" in result
        assert "snippet" in result
        assert mock_get.call_args.kwargs["proxies"] == {"http": "", "https": ""}

    def test_web_search_empty_query(self):
        result = _web_search("")
        assert "请提供搜索关键词" in result

    @patch("src.tools.builtin_tools.requests.get")
    def test_browse_url_success(self, mock_get):
        mock_get.return_value.text = "<html><title>Test Page</title><body>Hello World</body></html>"
        mock_get.return_value.apparent_encoding = "utf-8"
        result = _browse_url("https://example.com")
        assert "Test Page" in result
        assert "Hello World" in result
        assert mock_get.call_args.kwargs["proxies"] == {"http": "", "https": ""}

    def test_browse_url_empty(self):
        result = _browse_url("")
        assert "请提供要浏览的链接" in result

    def test_browse_url_adds_protocol(self):
        with patch("src.tools.builtin_tools.requests.get") as mock_get:
            mock_get.return_value.text = "<html><title>T</title><body>B</body></html>"
            mock_get.return_value.apparent_encoding = "utf-8"
            _browse_url("example.com")
            assert mock_get.call_args[0][0].startswith("https://")

    def test_register_builtin_tools(self):
        register_builtin_tools()
        reg2 = get_registry()
        assert reg2.has("get_current_time")
        assert reg2.has("get_weather")
        assert reg2.has("web_search")
        assert reg2.has("browse_url")
        assert reg2.has("stock_query")


class TestStockTools:
    def test_fetch_stock_parses_response(self):
        raw = 'v_sh600519="1~贵州茅台~600519~1800.00~1750.00~1760.00~~~~~~~~~10.00~0.50~10000~1000000~0.50~30~50~2024-01-15 15:00:00"'
        with patch("src.tools.stock_tools.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw.encode("gb2312")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            result = _fetch_stock("sh600519")
            assert "600519" in result
            assert "贵州茅台" in result["600519"]["名称"]

    def test_stock_query_empty_code(self):
        result = stock_query("")
        assert "请提供股票代码" in result

    def test_stock_query_auto_prefix_sh(self):
        raw = 'v_sh600519="1~贵州茅台~600519~1800~1750~1760~~~~~~~~~10~0.5~10000~1000000~0.5~30~50~2024-01-15 15:00:00"'
        with patch("src.tools.stock_tools.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw.encode("gb2312")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            result = stock_query("600519")
            assert "贵州茅台" in result

    def test_stock_query_auto_prefix_sz(self):
        raw = 'v_sz000001="1~平安银行~000001~10.00~9.80~9.90~~~~~~~~~0.10~1.00~10000~100000~0.10~5~10~2024-01-15 15:00:00"'
        with patch("src.tools.stock_tools.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw.encode("gb2312")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            result = stock_query("000001")
            assert "平安银行" in result

    def test_stock_query_auto_prefix_hk(self):
        raw = 'v_hk00700="1~腾讯控股~00700~400.00~390.00~395.00~~~~~~~~~5.00~1.20~10000~4000000~0.50~20~25~2024-01-15 16:00:00~Tencent~480~360~2024-01-15 16:00:00"'
        with patch("src.tools.stock_tools.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw.encode("gb2312")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            result = stock_query("00700")
            assert "腾讯控股" in result

    def test_stock_query_us(self):
        raw = 'v_usAAPL="1~苹果~AAPL~180.00~175.00~176.00~~~~~~~~~5.00~2.80~10000~1800000~0.50~25~30~2024-01-15 16:00:00"'
        with patch("src.tools.stock_tools.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw.encode("gb2312")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            result = stock_query("AAPL")
            assert "苹果" in result

    def test_stock_query_multiple(self):
        raw = 'v_sh600519="1~茅台~600519~1800~1750~1760~~~~~~~~~10~0.5~10000~1000000~0.5~30~50~2024-01-15 15:00:00";v_sz000001="1~平安~000001~10~9.8~9.9~~~~~~~~~0.1~1~10000~100000~0.1~5~10~2024-01-15 15:00:00"'
        with patch("src.tools.stock_tools.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw.encode("gb2312")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            result = stock_query("sh600519,sz000001")
            assert "茅台" in result
            assert "平安" in result
