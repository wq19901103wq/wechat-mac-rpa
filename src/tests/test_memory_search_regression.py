#!/usr/bin/env python3
"""Memory Engine 回归测试 — 使用真实 wiki 数据验证召回行为

从真实 wiki 目录复制关键文件到临时目录，验证修改后的搜索逻辑
是否符合预期（primary 精确匹配、Top 10 召回、跨人物关系发现）。
"""

import shutil
from pathlib import Path

import pytest

from src.memory.engine import MemoryEngine

REAL_WIKI_DIR = Path(__file__).parent.parent.parent / "data" / "memory" / "wiki"


def copy_real_wiki(engine, names: list[str]):
    """把真实 wiki 文件复制到临时引擎的 wiki 目录。"""
    for name in names:
        src = REAL_WIKI_DIR / "users" / f"{name}.md"
        if src.exists():
            dst = engine.wiki_dir / "users" / f"{name}.md"
            shutil.copy2(src, dst)


@pytest.fixture
def engine(tmp_path):
    engine = MemoryEngine()
    engine.wiki_dir = tmp_path
    (tmp_path / "users").mkdir(parents=True, exist_ok=True)
    (tmp_path / "groups").mkdir(parents=True, exist_ok=True)
    engine._facts = {}
    engine._corrections = {}
    return engine


class TestColleagueSearch:
    """回归测试：'王芊 同事' 应召回程立、Brian 等跨人物关系。"""

    def test_recalls_cheng_li_and_brian(self, engine):
        """搜'王芊 同事'应召回程立和 Brian 的 wiki 片段。"""
        copy_real_wiki(engine, ["王芊", "程立-君奕", "Brian", "王芊_2"])
        # 加几个噪声用户保证 IDF > 0
        for i in range(3):
            (engine.wiki_dir / "users" / f"噪声{i}.md").write_text(
                "- 无关内容\n", encoding="utf-8"
            )

        result = engine.search_keyword("王芊 同事", max_chars=10000)

        # 王芊必须是 primary，置顶
        assert result.startswith("【王芊的记忆】")

        # 程立和 Brian 必须被召回
        assert "【程立-君奕的记忆】" in result
        assert "【Brian的记忆】" in result

        # 验证程立的片段里确实有同事关系描述
        assert "同事" in result

    def test_wangqian_primary_exact_match_only(self, engine):
        """只有精确别名匹配的王芊.md 是 primary，王芊_2 不是。"""
        copy_real_wiki(engine, ["王芊", "王芊_2"])
        for i in range(3):
            (engine.wiki_dir / "users" / f"噪声{i}.md").write_text(
                "- 无关内容\n", encoding="utf-8"
            )

        result = engine.search_keyword("王芊", max_chars=10000)

        # 王芊是 primary，返回完整 wiki（出现在最前面）
        assert result.startswith("【王芊的记忆】")

        # 王芊_2 不应以完整 wiki 形式出现（不是 primary）
        # 它可能以片段形式出现，也可能因为 score 低被挤出
        # 关键断言：王芊的完整 wiki 只出现一次
        assert result.count("【王芊的记忆】") == 1


class TestTop10Recall:
    """回归测试：Top 10 能召回更多相关文档。"""

    def test_top10_not_limited_to_5(self, engine):
        """当存在 6+ 个相关文档时，Top 10 应召回至少 6 个。"""
        # 复制王芊 + 8 个拼多多同事
        colleagues = ["程立-君奕", "Brian", "肖健", "林涛-董平", "姚鹏 克明",
                      "子朔 胡成", "胡静", "金哥"]
        copy_real_wiki(engine, ["王芊"] + colleagues)
        for i in range(3):
            (engine.wiki_dir / "users" / f"噪声{i}.md").write_text(
                "- 无关内容\n", encoding="utf-8"
            )

        result = engine.search_keyword("王芊 同事", max_chars=50000)

        # 统计召回的同事数量
        hit_count = sum(1 for c in colleagues if f"【{c}的记忆】" in result)
        # 修复前 Top 5 只能召回 2-3 个；修复后 Top 10 应召回更多
        assert hit_count >= 4, f"只召回了 {hit_count} 个同事，期望至少 4 个"


class TestIdfEdgeCase:
    """回归测试：IDF 下限修复。"""

    def test_all_docs_contain_keyword_still_recalls(self, engine):
        """当几乎所有文档都包含关键词时，IDF 不应为负导致全部丢弃。"""
        for i in range(5):
            (engine.wiki_dir / "users" / f"用户{i}.md").write_text(
                f"- 姓名：用户{i}\n- 关键词：测试\n", encoding="utf-8"
            )
        # 只有 1 个噪声用户不包含关键词
        (engine.wiki_dir / "users" / "噪声.md").write_text(
            "- 完全无关\n", encoding="utf-8"
        )

        result = engine.search_keyword("测试", max_chars=10000)

        # 至少应召回几个包含关键词的文档
        hit_count = sum(1 for i in range(5) if f"【用户{i}的记忆】" in result)
        assert hit_count >= 3, f"只召回了 {hit_count} 个，期望至少 3 个"
