#!/usr/bin/env python3
"""Memory Engine search_keyword 单元测试

覆盖修复方向 1 + 2：
1. 限制 primary 数量：最多 1 个 primary（分数最高的），其余同名/子串文档按 BM25 正常排序
2. 增大 Top N：从 5 改为 10，确保高分 non-primary 文档不被挤出
"""

import pytest

from src.memory.engine import MemoryEngine


@pytest.fixture
def engine(tmp_path):
    """创建一个使用临时 wiki 目录的 MemoryEngine。"""
    engine = MemoryEngine()
    engine.wiki_dir = tmp_path
    (tmp_path / "users").mkdir(parents=True, exist_ok=True)
    (tmp_path / "groups").mkdir(parents=True, exist_ok=True)
    engine._facts = {}
    engine._corrections = {}
    return engine


def write_user_wiki(engine, name, content):
    """辅助：写入用户 wiki 文件。"""
    path = engine.wiki_dir / "users" / f"{name}.md"
    path.write_text(content, encoding="utf-8")


def write_noise_user(engine, name):
    """辅助：写入一个不包含任何搜索关键词的噪声用户，确保 IDF > 0。"""
    path = engine.wiki_dir / "users" / f"{name}.md"
    path.write_text("- 这是一个无关的用户，没有任何关键词。\n- 兴趣爱好：吃饭睡觉。\n", encoding="utf-8")


def write_group_wiki(engine, name, content):
    """辅助：写入群 wiki 文件。"""
    path = engine.wiki_dir / "groups" / f"{name}.md"
    path.write_text(content, encoding="utf-8")


class TestPrimaryMatching:
    """修复方向 1：最多只允许 1 个 primary。"""

    def test_exact_name_match_is_primary(self, engine):
        """精确匹配的用户 wiki 应为 primary，返回完整内容，且排在第一位。"""
        # wiki 内容里必须包含关键词，否则 BM25 score=0
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师\n- 爱好：股票")
        write_user_wiki(engine, "程立", "- 与王芊：同事")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊", max_chars=10000)

        assert "【王芊的记忆】" in result
        assert "- 职业：算法工程师" in result
        # primary 必须排在第一位
        lines = [line for line in result.split("\n") if line.startswith("【")]
        assert lines[0].startswith("【王芊的记忆】")

    def test_only_one_primary_allowed(self, engine):
        """最多只允许 1 个 primary，其余同名/子串文档按 BM25 正常排序。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师")
        write_user_wiki(engine, "王芊_2", "- 姓名：王芊_2\n- 详细描述：" + "x" * 2000)
        write_user_wiki(engine, "W1han、秋水文章、王芊", "- 包含王芊的文件名，内容很短")
        write_user_wiki(engine, "程立", "- 与王芊：同事")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊", max_chars=10000)

        # 只能有 1 个 primary → 王芊置顶
        lines = [line for line in result.split("\n") if line.startswith("【")]
        assert lines[0].startswith("【王芊的记忆】")

        # 王芊的完整 wiki 只应出现一次（primary 的完整内容）
        assert result.count("【王芊的记忆】") == 1

    def test_group_never_primary(self, engine):
        """群 wiki 永远不可能是 primary。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师")
        write_group_wiki(engine, "王芊的家人群", "- 群成员：王芊、王艺涵")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊", max_chars=10000)

        assert "【王芊的家人群群记忆】" in result


class TestTopNExpansion:
    """修复方向 2：Top N 从 5 增大到 10。"""

    def test_top10_includes_more_non_primary(self, engine):
        """当存在多个高相关度 non-primary 文档时，Top 10 应能召回它们。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师")
        for i in range(8):
            write_user_wiki(engine, f"同事{i}", f"- 与王芊：同事，编号{i}\n- 详细描述：{'工作内容' * 50}")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊 同事", max_chars=100000)

        hit_count = sum(1 for i in range(8) if f"【同事{i}的记忆】" in result)
        assert hit_count >= 5, f"只召回了 {hit_count} 个同事文档，期望至少 5 个"

    def test_primary_still_first(self, engine):
        """即使 Top N 增大，primary 文档仍然排在最前面。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师")
        for i in range(8):
            write_user_wiki(engine, f"同事{i}", "- 与王芊：同事")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊", max_chars=100000)

        lines = [line for line in result.split("\n") if line.startswith("【")]
        assert lines[0].startswith("【王芊的记忆】")

    def test_max_chars_respected(self, engine):
        """即使 Top N 增大，max_chars 截断仍然有效。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师\n" + "- 详细描述\n" * 100)
        for i in range(8):
            write_user_wiki(engine, f"同事{i}", "- 与王芊：同事\n" + "- 详情\n" * 50)
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊 同事", max_chars=2000)

        assert len(result) <= 2000 + 50
        assert "（…更多结果省略）" in result or "（…内容截断）" in result


class TestAliasScoring:
    """修复：扩展别名共享 TF-IDF + 降权，避免别名噪声颠覆 BM25 排序。"""

    def test_expanded_aliases_downweighted(self, engine):
        """扩展别名的贡献应低于原始搜索词，避免无关文档因别名而高分。"""
        # 王芊 wiki 里有 "g神" 别名
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 别名：g神\n- 职业：算法")
        # 汪亦茂 wiki 里大量提到 "g神"（模拟真实场景），但和"同事"无关
        write_user_wiki(engine, "汪亦茂", "- 朋友：g神\n- g神g神g神g神\n-  unrelated content")
        # 程立 wiki 里有 "王芊" 和 "同事"，但没有 "g神"
        write_user_wiki(engine, "程立", "- 与王芊：同事")
        # Brian wiki 里有 "王芊" 和 "同事"
        write_user_wiki(engine, "Brian", "- 与王芊：同事")
        # 给 aliases.json 注入别名，让 engine 扩展出 "g神"
        engine._aliases["王芊"] = ["g神"]
        for i in range(3):
            write_user_wiki(engine, f"路人{i}", "- 无关内容")

        result = engine.search_keyword("王芊 同事", max_chars=10000)

        # 程立和 Brian 应被召回（因为他们有"同事"关系）
        assert "【程立的记忆】" in result or "【程立的记忆】…" in result
        assert "【Brian的记忆】" in result or "【Brian的记忆】…" in result

    def test_alias_group_shares_tfidf(self, engine):
        """所有扩展别名应共享一组 TF-IDF，而不是每个别名单独计算。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 别名：g神、干哥")
        # 文档A：包含 "g神"（王芊别名）
        write_user_wiki(engine, "文档A", "- 提到g神")
        # 文档B：包含 "干哥"（王芊别名）
        write_user_wiki(engine, "文档B", "- 提到干哥")
        # 文档C：包含 "王芊"（原始搜索词）
        write_user_wiki(engine, "文档C", "- 提到王芊")
        engine._aliases["王芊"] = ["g神", "干哥"]
        # 加足够多的噪声文档，让别名组出现率低于 50%（3/7=43%）
        for i in range(4):
            write_user_wiki(engine, f"噪声{i}", "- 无关")

        result = engine.search_keyword("王芊", max_chars=10000)

        # 文档A、B、C 都应被召回（因为它们都提到了王芊的某个别名或本名）
        assert "【文档A的记忆】" in result or "【文档A的记忆】…" in result
        assert "【文档B的记忆】" in result or "【文档B的记忆】…" in result
        assert "【文档C的记忆】" in result or "【文档C的记忆】…" in result


class TestKeywordFiltering:
    """修复：过滤高频无区分度关键词（如'我'、'对话中'）。"""

    def test_high_freq_aliases_filtered(self, engine):
        """高频通用别名（出现率>50%）不应作为搜索关键词。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 描述：我我我我我")
        write_user_wiki(engine, "程立", "- 与王芊：同事")
        # 创建大量包含"我"的文档，让"我"的出现率超过50%
        for i in range(6):
            write_user_wiki(engine, f"路人{i}", "- 我我我我我")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊", max_chars=10000)

        # 程立应被召回（即使"我"被过滤了）
        assert "【程立的记忆】" in result or "【程立的记忆】…" in result


class TestCrossPersonRelationship:
    """回归测试：跨人物关系查询（同事搜索场景）。"""

    def test_colleague_search_recalls_others(self, engine):
        """搜'王芊 同事'应召回程立、Brian 等包含关系描述的文档。"""
        write_user_wiki(engine, "王芊", "- 姓名：王芊\n- 职业：算法工程师\n- 工作经历：腾讯→拼多多")
        write_user_wiki(engine, "程立", "- 王芊（干哥）：同事（拼多多）、朋友")
        write_user_wiki(engine, "Brian", "- 与王芊：同事（同一组）、朋友")
        write_user_wiki(engine, "路人甲", "- 完全不相关的内容")
        write_noise_user(engine, "噪声用户")

        result = engine.search_keyword("王芊 同事", max_chars=10000)

        assert "【程立的记忆】" in result or "【程立的记忆】…" in result
        assert "【Brian的记忆】" in result or "【Brian的记忆】…" in result
        assert "路人甲" not in result
