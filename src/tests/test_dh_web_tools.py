"""RPA consumes DH tools without replacing either application's src package."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.tools import builtin_tools
from src.tools.tool_registry import ToolRegistry


def test_rpa_reuses_dh_schemas_and_namespace():
    original_src = sys.modules['src']
    registry = ToolRegistry()
    builtin_tools.register_builtin_tools(registry)
    shared = builtin_tools._web_tools
    assert Path(shared.__file__).resolve() == (
        Path(__file__).resolve().parents[2] / 'third_party/digital-human/src/tools/web.py')
    assert sys.modules['src'] is original_src
    assert shared.__name__ == 'digital_human.tools.web'
    for schema in shared.WEB_SCHEMAS:
        assert registry.get(schema['function']['name']).to_openai_schema() == schema


def test_search_preserves_rpa_parser_and_transport():
    html = ('<li class="res-list"><h3><a href="https://www.so.com/link?x=y" '
            'data-mdurl="https://example.com/roster">Club roster</a></h3>'
            '<p class="res-desc">Verified roster</p></li>')
    fetch = Mock(return_value=SimpleNamespace(text=html))
    result = builtin_tools._web_search('club roster', structured=True, fetch=fetch)
    assert result['results'] == [dict(title='Club roster', url='https://example.com/roster',
                                      snippet='Verified roster')]
    assert fetch.call_args.args == ('https://www.so.com/s',)
    assert fetch.call_args.kwargs['params'] == {'q': 'club roster'}


def test_browse_and_page_search_share_the_dh_cache():
    url = 'https://mp.weixin.qq.com/s'
    html = '<body><div id="js_content"><p>Verified roster</p></div><p>Noise</p></body>'
    response = SimpleNamespace(url=url, text=html, apparent_encoding='utf-8')
    result = builtin_tools._browse_url(url, structured=True, fetch=lambda *a, **k: response)
    assert result['text'] == 'Verified roster'
    assert 'Verified roster' in builtin_tools._search_in_page(url, 'roster')


@pytest.mark.parametrize('method,arg', [('_web_search', 'club roster'),
                                      ('_browse_url', 'https://example.com')])
def test_transport_failure_is_not_returned_as_facts(method, arg):
    result = getattr(builtin_tools, method)(arg, structured=True,
                                            fetch=Mock(side_effect=TimeoutError('offline')))
    assert result == dict(error='TimeoutError')
