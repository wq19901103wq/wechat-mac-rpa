#!/usr/bin/env python3
"""Memory Engine search_keyword 单元测试

覆盖修复方向 1 + 2：
1. 限制 primary 数量：最多 1 个 primary（分数最高的），其余同名/子串文档按 BM25 正常排序
2. 增大 Top N：从 5 改为 10，确保高分 non-primary 文档不被挤出
"""

import json
import pytest

from src.memory.engine import MemoryEngine


@pytest.fixture
def engine(tmp_path):
    """创建一个使用临时 wiki 目录的 MemoryEngine。

    隔离 aliases（默认空）：避免加载真实 data/memory/overrides/aliases.json
    导致测试用名被 _resolve_alias 改写（如'程立'被规范成'程立-君奕'，
    使断言'【程立的记忆】'对不上实际'【程立-君奕的记忆】'）。需要别名的
    测试自行注入 engine._aliases[...]。
    """
    engine = MemoryEngine()
    engine.wiki_dir = tmp_path
    (tmp_path / "users").mkdir(parents=True, exist_ok=True)
    (tmp_path / "groups").mkdir(parents=True, exist_ok=True)
    engine._facts = {}
    engine._corrections = {}
    engine._aliases = {}
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


class TestAliasExtractionSanitization:
    """别名提取的拆分 + 脏数据过滤（修复'王总'召回失败）。"""

    def test_user_wiki_alias_string_split(self, engine):
        """## 别名 段落里的 '老王、王总' 必须拆成两条，不能整串入库。"""
        wiki = (
            "# 王旭东\n\n## 别名\n- 老王、王总\n\n## 基本信息\n- 职业：测试\n"
        )
        aliases = engine._extract_aliases_from_user_wiki(wiki, "王旭东")
        assert aliases == ["老王", "王总"]

    def test_group_wiki_alias_string_split(self, engine):
        """群成员画像 'Paul、坤蜀黍' 也按顿号拆分。"""
        wiki = (
            "# 群\n\n## 群成员画像\n"
            "### **坤蜀黍 Paul（Paul、坤蜀黍）**\n- 说明\n"
        )
        result = engine._extract_aliases_from_group_wiki(wiki)
        assert result.get("坤蜀黍 Paul") == ["Paul", "坤蜀黍"]

    def test_role_blacklist_rejected(self, engine):
        """角色词 / 系统占位符不能当别名。"""
        wiki = (
            "# 王芊\n\n## 别名\n- Bot\n- 对话中\n- 匿名\n- 群主\n- 记录者\n"
            "- 真别名\n\n## 基本信息\n- x\n"
        )
        aliases = engine._extract_aliases_from_user_wiki(wiki, "王芊")
        assert aliases == ["真别名"]

    def test_room_number_and_sentence_rejected(self, engine):
        """房号、整句描述不能当别名。"""
        wiki = (
            "# 小丁\n\n## 别名\n- 4-1-2503\n- 6幢5号501\n"
            "- 被群友称为哥\n- 丁总\n\n## 基本信息\n- x\n"
        )
        aliases = engine._extract_aliases_from_user_wiki(wiki, "小丁")
        assert aliases == ["丁总"]

    def test_search_recalls_via_split_alias(self, engine, tmp_path):
        """端到端：aliases.json 存了整串 '老王、王总'，搜 '王总' 仍能召回本人。"""
        # 重定向 overrides + 清空别名表，隔离真实数据避免 resolve 走偏
        engine.overrides_dir = tmp_path / "overrides"
        engine.overrides_dir.mkdir(parents=True, exist_ok=True)
        engine._aliases = {}
        write_user_wiki(engine, "王旭东", "- 姓名：王旭东\n- 爱好：干脆面")
        write_noise_user(engine, "噪声用户")
        # 模拟历史脏数据：整串未拆分，经 merge 清洗后应拆成两条
        engine._merge_aliases("王旭东", ["老王、王总"])
        assert engine._aliases["王旭东"] == ["老王", "王总"]
        result = engine.search_keyword("王总", max_chars=10000)
        assert "【王旭东的记忆】" in result

    def test_sanitize_wiki_aliases_removes_cross_person_alias(self, engine):
        """清洗 wiki 别名时应剔除已属于其他用户的别名。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "# 薛定谔的王芊\n\n## 别名\n- 别名：陈大发、芊总\n"

        cleaned = engine._sanitize_wiki_aliases(wiki, "薛定谔的王芊")

        assert "芊总" not in cleaned
        assert "陈大发" in cleaned

    def test_sanitize_wiki_aliases_resolves_filename_alias(self, engine):
        """wiki 文件名是别名时，应归一化到主名再判断归属。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "# 陈大发\n\n## 别名\n- 别名：芊总\n"

        cleaned = engine._sanitize_wiki_aliases(wiki, "陈大发")

        assert "芊总" not in cleaned

    def test_sanitize_wiki_aliases_keeps_group_wiki_aliases(self, engine):
        """群 wiki 不应被误清洗掉成员的合法别名。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "# 共同富裕群\n\n## 群成员画像\n**王芊（芊总）**\n**薛定谔的王芊（陈大发）**\n"

        cleaned = engine._sanitize_wiki_aliases(wiki, "共同富裕群")

        # 群名不在 _aliases 中，不应做跨人归属校验
        assert "芊总" in cleaned
        assert "陈大发" in cleaned

    def test_sanitize_wiki_aliases_preserves_alias_with_source_annotation(self, engine):
        """带 （来源：...） 注解的有效别名不应被误清洗。"""
        engine._aliases = {"王芊": ["芊总"]}
        wiki = (
            "# 王芊\n\n## 别名\n"
            "- 别名：Co总（来源：2025-01-01 群聊，时鹏、李雪怡、bot 均如此称呼）、扛把子\n"
        )

        cleaned = engine._sanitize_wiki_aliases(wiki, "王芊")

        assert "Co总" in cleaned
        assert "扛把子" in cleaned


class TestWikiFactSanitizer:
    """落盘前事实清洗 _sanitize_wiki_facts（防止 Bot 来源被写成确定事实）。"""

    def test_marks_bot_source_line_as_unverified(self, engine):
        """来源含 Bot 且不含用户明确陈述的行应被追加 [待验证]。"""
        wiki = (
            "# 用户\n\n## 基本信息\n"
            "- 职业：外企员工（来源：2022-02-11 Bot说，Esther未否认）\n"
            "- 爱好：跑步\n"
        )

        cleaned = engine._sanitize_wiki_facts(wiki)

        assert "职业：外企员工（来源：2022-02-11 Bot说，Esther未否认） [待验证]" in cleaned
        assert "爱好：跑步" in cleaned

    def test_marks_bot_confirmation_as_unverified(self, engine):
        wiki = "# 用户\n\n## 基本信息\n- 职业：外企员工（来源：Bot确认）\n"

        cleaned = engine._sanitize_wiki_facts(wiki)

        assert "Bot确认） [待验证]" in cleaned

    def test_keeps_user_source_line_unchanged(self, engine):
        """用户自述或明确确认的事实不应被误标。"""
        wiki = (
            "# 用户\n\n## 基本信息\n"
            "- 职业：算法工程师（来源：2026-06-13 王芊自述）\n"
            "- 出生日期：1990-11-03（用户确认）\n"
        )

        cleaned = engine._sanitize_wiki_facts(wiki)

        assert "[待验证]" not in cleaned

    def test_idempotent_for_already_unverified(self, engine):
        """已是 [待验证] 的行不应被重复追加标记。"""
        wiki = "# 用户\n\n## 基本信息\n- 职业：外企员工（来源：Bot推测）[待验证]\n"

        cleaned = engine._sanitize_wiki_facts(wiki)

        assert cleaned.count("[待验证]") == 1


class TestWikiLimitsEnforcement:
    """代码级长度护栏 _enforce_wiki_limits（NFR-2，P0-B）。"""

    def _engine(self):
        e = MemoryEngine.__new__(MemoryEngine)
        return e

    def test_short_wiki_unchanged(self):
        e = self._engine()
        assert e._enforce_wiki_limits("短 wiki", 4000) == "短 wiki"

    def test_bloated_wiki_truncated_under_limit(self):
        e = self._engine()
        wiki = "# 用户\n\n## 基本信息\n- 姓名：测试\n- 职业：工程师\n"
        wiki += "## 近期动态\n" + "\n".join(f"- 2026-06-20 第{i}条" + "X" * 60 for i in range(150))
        wiki += "\n## 与其他人的关系\n- 张三：同事\n"
        out = e._enforce_wiki_limits(wiki, 4000)
        assert len(out) <= 4000
        # 身份/关系保留
        assert "## 基本信息" in out
        assert "## 与其他人的关系" in out
        # 原始被压缩
        assert len(out) < len(wiki)

    def test_volatile_keeps_newest_drops_oldest(self):
        e = self._engine()
        wiki = "# 用户\n## 近期动态\n" + "\n".join(f"- 第{i}条" + "Y" * 80 for i in range(100))
        wiki += "\n## 基本信息\n- x\n"
        out = e._enforce_wiki_limits(wiki, 4000)
        # 最新的保留，最老的丢弃
        assert "第99条" in out
        assert "第0条" not in out

    def test_save_wiki_applies_limits(self, engine, tmp_path):
        """_save_wiki 应自动套用长度护栏。"""
        path = tmp_path / "users" / "x.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        bloated = "# x\n## 近期动态\n" + "\n".join(f"- {i}" + "Z" * 200 for i in range(100))
        engine._save_wiki(path, bloated)
        saved = path.read_text(encoding="utf-8")
        assert len(saved) <= 4000


class TestNormalizeChatName:
    """群名归一化（FR-13）。"""

    def test_strip_emoji_and_fold_space(self):
        from src.memory.engine import normalize_chat_name
        assert normalize_chat_name("3D 打印技术 交流群") == "3D 打印技术 交流群"
        # emoji 剥离
        assert normalize_chat_name("🎀苏苏🎀") == "苏苏"
        assert normalize_chat_name("Nico. 🤍") == "Nico."
        # 空格折叠
        assert normalize_chat_name("築岛  周一至周五") == "築岛 周一至周五"

    def test_empty(self):
        from src.memory.engine import normalize_chat_name
        assert normalize_chat_name("") == ""


class TestLintMemory:
    """Lint 健康检查（FR-7）。"""

    def test_lint_detects_bloated_and_duplicates(self, engine, tmp_path):
        engine._aliases = {}
        # 膨胀 wiki
        write_user_wiki(engine, "大用户", "x" * 5000)
        # 正常 wiki
        write_user_wiki(engine, "小用户", "x" * 100)
        # 重复（归一化后同名）
        write_user_wiki(engine, "🎀苏苏🎀", "内容")
        write_user_wiki(engine, "苏苏", "内容")

        report = engine.lint_memory(max_wiki_chars=4000)
        bloated_names = [b["path"] for b in report["bloated"]]
        assert any("大用户" in p for p in bloated_names)
        assert not any("小用户" in p for p in bloated_names)
        # 苏苏 / 🎀苏苏🎀 归一化后都是 "苏苏"
        dup_found = any("苏苏" in d.get("normalized", "") for d in report["duplicates"])
        assert dup_found

    def test_lint_detects_alias_conflicts(self, engine, tmp_path):
        engine._aliases = {"王芊": ["g神"], "张波": ["g神"]}
        report = engine.lint_memory()
        conflicts = [c for c in report["conflicts"] if c["alias"] == "g神"]
        assert len(conflicts) == 1
        assert set(conflicts[0]["mains"]) == {"王芊", "张波"}

    def test_ad_group_rejected_on_update(self, engine, tmp_path):
        """广告群名不应入队生成 wiki（FR-14）。"""
        engine.overrides_dir = tmp_path / "overrides"
        engine.overrides_dir.mkdir(parents=True, exist_ok=True)
        engine._aliases = {}
        # 给一个假 llm_client 触发入队路径
        engine.llm_client = object()
        engine.update_group_wiki("玲珑小番茄6.99一斤茅台路百果园", "x", [], [])
        assert engine._update_queue == []


def test_shutdown_drains_pending_updates(tmp_path):
    """退出时即使队列不足批量阈值，也必须处理完已有任务。"""
    engine = MemoryEngine(llm_client=object())
    engine.wiki_dir = tmp_path
    processed = []
    engine._do_update = processed.append

    engine.update_user_wiki("测试用户", "测试会话", [], [])
    engine.shutdown()

    assert len(processed) == 1
    assert processed[0]["user_name"] == "测试用户"
    assert engine._update_queue == []


class TestLLMRerank:
    """LLM rerank：BM25 候选用 LLM 按语义相关性重排 + 降级。"""

    def test_fallback_when_no_llm_client(self, engine):
        """llm_client 为 None 时应原样返回 scored（降级不阻断）。"""
        engine._aliases = {}
        scored = [("a", "内容a", False, 1.0, False), ("b", "内容b", False, 0.5, False)]
        out = engine._llm_rerank(scored, "查询", ["查询"])
        assert out == scored  # 原顺序不变

    def test_rerank_reorders_by_llm_order(self, engine):
        """LLM 返回 [2,1] 时，候选应按此顺序重排。"""
        engine._aliases = {}

        class _FakeLLM:
            def chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None):
                return "[2, 1]"

        engine.llm_client = _FakeLLM()
        scored = [
            ("a", "内容a", False, 1.0, False),
            ("b", "内容b", False, 0.5, False),
        ]
        out = engine._llm_rerank(scored, "查询", ["查询"])
        assert [x[0] for x in out] == ["b", "a"]  # LLM 顺序

    def test_rerank_keeps_unlisted_at_tail(self, engine):
        """LLM 未列出的候选应保留在后面（按原 BM25 顺序）。"""
        engine._aliases = {}

        class _FakeLLM:
            def chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None):
                return "[3]"  # 只列了第3个

        engine.llm_client = _FakeLLM()
        scored = [
            ("a", "内容a", False, 1.0, False),
            ("b", "内容b", False, 0.8, False),
            ("c", "内容c", False, 0.5, False),
        ]
        out = engine._llm_rerank(scored, "查询", ["查询"])
        assert out[0][0] == "c"  # LLM 指定第3个排第一
        # 其余按原顺序跟在后面
        assert [x[0] for x in out[1:]] == ["a", "b"]

    def test_rerank_fallback_on_parse_failure(self, engine):
        """LLM 返回无法解析的内容时应回退 BM25。"""
        engine._aliases = {}

        class _FakeLLM:
            def chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None):
                return "我觉得没法排序"

        engine.llm_client = _FakeLLM()
        scored = [("a", "内容a", False, 1.0, False), ("b", "内容b", False, 0.5, False)]
        out = engine._llm_rerank(scored, "查询", ["查询"])
        assert out == scored  # 回退原顺序

    def test_rerank_fallback_on_exception(self, engine):
        """LLM 调用抛异常时应回退 BM25。"""
        engine._aliases = {}

        class _BoomLLM:
            def chat(self, **kw):
                raise RuntimeError("LLM 挂了")

        engine.llm_client = _BoomLLM()
        scored = [("a", "内容a", False, 1.0, False), ("b", "内容b", False, 0.5, False)]
        out = engine._llm_rerank(scored, "查询", ["查询"])
        assert out == scored  # 回退原顺序

    def test_rerank_relation_disambiguation(self, engine):
        """关系消歧场景：搜'王芊妈妈'，LLM 应把母亲(秋水文章)排到岳母(枫叶)前面。

        模拟 BM25 把岳母排前（词面'妈妈'匹配'艺涵妈妈'），LLM 理解语义后翻转。
        """
        engine._aliases = {}
        # 模拟 wiki 内容
        write_user_wiki(engine, "秋水文章", "- 母亲：王秋文（王芊的母亲）")
        write_user_wiki(engine, "枫叶 艺涵妈妈", "- 王艺涵的妈妈（岳母，非王芊母亲）")
        write_noise_user(engine, "噪声用户")

        # LLM 理解"王芊妈妈=母亲"语义，把秋水文章（编号2）排前
        class _SmartLLM:
            def chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None):
                # 候选顺序：枫叶(1, BM25高分) 秋水文章(2)，LLM 翻转
                return "[2, 1]"

        engine.llm_client = _SmartLLM()
        result = engine.search_keyword("王芊 妈妈", max_chars=5000)
        # 秋水文章应排在枫叶前面
        idx_qiushui = result.find("【秋水文章的记忆】")
        idx_fengye = result.find("【枫叶 艺涵妈妈的记忆】")
        assert idx_qiushui != -1, "秋水文章应被召回"
        assert idx_qiushui < idx_fengye, "母亲(秋水文章)应排在岳母(枫叶)前面"


class TestAliasConflictGuard:
    """跨人别名冲突防护：一个昵称只能归属一个人。"""

    def test_reject_alias_already_owned_by_another_user(self, engine, tmp_path):
        """当某个别名已经被分配给另一个用户时，禁止再分配给新用户。"""
        engine.overrides_dir = tmp_path / "overrides"
        engine.overrides_dir.mkdir(parents=True, exist_ok=True)
        engine._aliases = {"王芊": ["芊小微"], "薛定谔的王芊": []}

        engine._do_merge_aliases("薛定谔的王芊", ["芊小微"])

        # 芊小微 仍只属于 王芊
        assert engine._aliases["王芊"] == ["芊小微"]
        assert "芊小微" not in engine._aliases["薛定谔的王芊"]

    def test_allow_alias_for_same_user(self, engine, tmp_path):
        """同一用户重复合并同一别名不应重复添加。"""
        engine.overrides_dir = tmp_path / "overrides"
        engine.overrides_dir.mkdir(parents=True, exist_ok=True)
        engine._aliases = {"王芊": ["芊小微"]}

        engine._do_merge_aliases("王芊", ["芊小微", "韭菜芊"])

        assert engine._aliases["王芊"] == ["芊小微", "韭菜芊"]

    def test_persisted_aliases_json_no_conflict(self, engine, tmp_path):
        """冲突别名不应写入 aliases.json。"""
        engine.overrides_dir = tmp_path / "overrides"
        engine.overrides_dir.mkdir(parents=True, exist_ok=True)
        aliases_path = engine.overrides_dir / "aliases.json"
        aliases_path.write_text(
            '{"users": {"王芊": {"aliases": ["芊小微"], "notes": ""}, '
            '"薛定谔的王芊": {"aliases": [], "notes": ""}}}',
            encoding="utf-8",
        )
        engine._aliases = {
            "王芊": ["芊小微"],
            "薛定谔的王芊": [],
        }

        engine._do_merge_aliases("薛定谔的王芊", ["芊小微", "大发"])

        data = json.loads(aliases_path.read_text(encoding="utf-8"))
        assert "芊小微" not in data["users"]["薛定谔的王芊"]["aliases"]
        assert "大发" in data["users"]["薛定谔的王芊"]["aliases"]


class TestAliasExtractionOwnership:
    """别名提取阶段就应拒绝跨人冲突，不把错误发现传给 merge。"""

    def test_user_wiki_skips_alias_owned_by_other(self, engine):
        """用户 wiki 提取到已属于别人的别名时应跳过。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "## 别名\n- 别名：芊总、陈大发\n"

        aliases = engine._extract_aliases_from_user_wiki(wiki, "薛定谔的王芊")

        assert "芊总" not in aliases
        assert "陈大发" in aliases  # 属于自己，允许再次出现

    def test_user_wiki_keeps_alias_owned_by_same_user(self, engine):
        """用户 wiki 提取到属于当前用户自己的别名时应保留（去重）。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "## 别名\n- 别名：芊总、韭菜芊\n"

        aliases = engine._extract_aliases_from_user_wiki(wiki, "王芊")

        assert "芊总" in aliases  # 属于自己，允许再次出现
        assert "韭菜芊" in aliases

    def test_group_wiki_skips_alias_owned_by_other(self, engine):
        """群 wiki 提取到已属于别人的别名时应跳过。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "## 群成员画像\n**王芊（芊总/韭菜芊）**\n**薛定谔的王芊（芊总/陈大发/大发）**\n"

        result = engine._extract_aliases_from_group_wiki(wiki)

        assert "芊总" not in result.get("薛定谔的王芊", [])
        assert "陈大发" in result.get("薛定谔的王芊", [])
        assert "大发" in result.get("薛定谔的王芊", [])
        assert "芊总" in result.get("王芊", [])

    def test_group_wiki_resolves_main_name(self, engine):
        """群 wiki 用别名做主名时应归一化到主名再判断归属。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "## 群成员画像\n**陈大发（芊总）**\n"

        result = engine._extract_aliases_from_group_wiki(wiki)

        # 应归一化为 薛定谔的王芊，且芊总已属于王芊，应被跳过
        assert "芊总" not in result.get("薛定谔的王芊", [])

    def test_group_wiki_keeps_alias_for_same_owner(self, engine):
        """群 wiki 提取到属于该成员自己的别名时应保留。"""
        engine._aliases = {
            "王芊": ["芊总"],
            "薛定谔的王芊": ["陈大发"],
        }
        wiki = "## 群成员画像\n**王芊（芊总/韭菜芊）**\n"

        result = engine._extract_aliases_from_group_wiki(wiki)

        assert "芊总" in result.get("王芊", [])
        assert "韭菜芊" in result.get("王芊", [])
