#!/usr/bin/env python3
"""Regression tests for memory evidence hygiene & fact-check evidence policy.

Covers, with **synthetic entity names** (never real-name query routing):

1. Authoritative entity-correction precedence in get_user_memory / search / update context.
2. Bot/self-role messages excluded from the wiki evidence transcript.
3. Assistant-history lines removed from self-refinement fact-check evidence.
4. Untrusted ``[待验证]`` derived lines excluded from runtime retrieval.
5. Generic fact-check issue policy: ``unknown`` blocks like ``contradicted``,
   ``nonfactual`` never blocks, and every ``entailed`` claim is forwarded to the
   independent verifier (no directional keyword gate).
6. Contradictory facts are not merged — corrections stay a distinct authoritative block.

These are bounded unit tests; no LLM/API is called.
"""

import json

import pytest

from src.memory.evidence import (
    format_evidence_conversation,
    is_self_message,
    strip_unverified_lines,
)
from src.memory.engine import MemoryEngine
from src.memory.wiki_prompts import BATCH_UPDATE_GROUP_PROMPT, RUNTIME_UPDATE_GROUP_PROMPT
from src.models.base import ChatMessage, SenderType
from src.reply.evidence_utils import strip_assistant_history_lines
from src.reply.generator import _collect_fact_issues


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path):
    """MemoryEngine isolated to a temp wiki dir with empty overrides."""
    eng = MemoryEngine()
    eng.wiki_dir = tmp_path
    (tmp_path / "users").mkdir(parents=True, exist_ok=True)
    (tmp_path / "groups").mkdir(parents=True, exist_ok=True)
    eng._aliases = {}
    eng._facts = {}
    eng._corrections = {}
    eng._entity_corrections = {}
    yield eng
    eng.shutdown()


def _user_msg(sender, text, sender_type=SenderType.OTHER, account=""):
    return ChatMessage(
        text=text, sender=sender, sender_type=sender_type,
        chat_name="测试群", account=account, create_time=1700000000,
    )


def _write_user_wiki(eng, name, content):
    path = eng.wiki_dir / "users" / f"{name}.md"
    path.write_text(content, encoding="utf-8")


def _write_noise_user(eng, name="噪声用户"):
    _write_user_wiki(eng, name, "- 与本次测试无关的内容：吃饭睡觉。\n")


# ── 1. correction precedence ─────────────────────────────────────────────

class TestCorrectionPrecedence:
    def test_get_user_memory_correction_first(self, engine):
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不在 ZetaBank 工作"]}
        _write_user_wiki(
            engine, "SyntheticPerson",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 当前公司：某银行\n",
        )
        _write_noise_user(engine)

        result = engine.get_user_memory("SyntheticPerson", max_chars=10000)

        # 权威纠正必须出现在 wiki 本体之前（最高优先级）
        assert result.index("权威纠正") < result.index("当前公司")
        assert "SyntheticPerson 不在 ZetaBank 工作" in result

    def test_get_user_memory_correction_survives_truncation(self, engine):
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不在 ZetaBank 工作"]}
        _write_user_wiki(
            engine, "SyntheticPerson",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 简介：" + "很长的内容" * 200 + "\n",
        )
        _write_noise_user(engine)

        result = engine.get_user_memory("SyntheticPerson", max_chars=200)

        # 截断后纠正仍在（它在最前面，从尾部截断不会丢）
        assert "SyntheticPerson 不在 ZetaBank 工作" in result

    def test_search_keyword_presents_correction_for_primary(self, engine):
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不在 ZetaBank 工作"]}
        _write_user_wiki(
            engine, "SyntheticPerson",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 当前公司：某银行\n",
        )
        _write_noise_user(engine)

        result = engine.search_keyword("SyntheticPerson", max_chars=10000)

        assert "【SyntheticPerson的记忆】" in result
        assert "SyntheticPerson 不在 ZetaBank 工作" in result
        # 纠正出现在返回文本靠前位置
        assert result.index("SyntheticPerson 不在 ZetaBank 工作") < result.index("当前公司")

    def test_update_identity_context_includes_correction(self, engine):
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不是用户中学同学"]}
        ctx = engine._build_identity_context(["SyntheticPerson"])
        assert "SyntheticPerson 不是用户中学同学" in ctx
        assert "权威纠正" in ctx

    def test_entity_corrections_loaded_generically(self, engine, tmp_path):
        # 用临时 overrides 目录验证数据驱动加载（代码不含实体名）
        engine.overrides_dir = tmp_path / "overrides"
        engine.overrides_dir.mkdir(parents=True, exist_ok=True)
        (engine.overrides_dir / "corrections.json").write_text(
            json.dumps({"entities": {"GenericEntity": {"corrections": ["generic rule"]}}},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        engine._load_overrides()
        assert engine._entity_corrections["GenericEntity"] == ["generic rule"]

    def test_search_surfaces_correction_for_suffixed_entity_doc(self, engine):
        # wiki 文件名带群后缀（如 SyntheticPerson@devgroup），纠正仍应随该文档出现
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不在 ZetaBank 工作"]}
        _write_user_wiki(
            engine, "SyntheticPerson@devgroup",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 当前公司：某银行\n",
        )
        _write_noise_user(engine)

        result = engine.search_keyword("SyntheticPerson", max_chars=10000)

        assert "SyntheticPerson 不在 ZetaBank 工作" in result

    def test_base_name_matching_avoids_false_positive(self, engine):
        # "SyntheticPerson" 的纠正不应误配给 "SyntheticPersonX"（前缀相同但非分隔符）
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不在 ZetaBank 工作"]}
        assert engine._entity_corrections_for_doc("SyntheticPerson@devgroup") == [
            "SyntheticPerson 不在 ZetaBank 工作"
        ]
        assert engine._entity_corrections_for_doc("SyntheticPersonX") == []

    def test_search_keyword_returns_correction_without_wiki_file(self, engine):
        # 实体有权威纠正但没有 users/<entity>.md：search_keyword 也应返回该纠正
        # （correction-only 用户文档；不重复已存在 wiki 的实体）
        engine._entity_corrections = {
            "SyntheticNoWikiPerson": ["SyntheticNoWikiPerson 不在 ZetaBank 工作"]
        }
        _write_noise_user(engine)

        result = engine.search_keyword("SyntheticNoWikiPerson", max_chars=10000)

        assert "SyntheticNoWikiPerson 不在 ZetaBank 工作" in result
        assert "【SyntheticNoWikiPerson的记忆】" in result


# ── 2. Bot/self-role exclusion from wiki evidence ────────────────────────

class TestBotRoleExclusion:
    def test_is_self_message_role_based(self):
        self_msg = _user_msg("我", "Bot说的话", sender_type=SenderType.SELF)
        other_msg = _user_msg("Alice", "Alice说的话")
        assert is_self_message(self_msg) is True
        assert is_self_message(other_msg) is False

    def test_format_evidence_conversation_excludes_self_and_bot_replies(self):
        msgs = [
            _user_msg("Alice", "我在某银行工作"),
            _user_msg("我", "我猜你在云顶银行工作", sender_type=SenderType.SELF),
            _user_msg("Bob", "他在银行上班"),
        ]
        out = format_evidence_conversation(msgs, bot_replies=["Bot补充回复"])

        assert "我在某银行工作" in out
        assert "他在银行上班" in out
        # Bot/self 内容必须被排除
        assert "我猜你在云顶银行工作" not in out
        assert "Bot补充回复" not in out

    def test_memory_engine_format_conversation_excludes_self(self, engine):
        msgs = [
            _user_msg("Alice", "Alice事实"),
            _user_msg("我", "Bot自述", sender_type=SenderType.SELF),
        ]
        out = engine._format_conversation(msgs, bot_replies=["Bot回复"])
        assert "Alice事实" in out
        assert "Bot自述" not in out
        assert "Bot回复" not in out

    @pytest.mark.parametrize(
        "prompt", [RUNTIME_UPDATE_GROUP_PROMPT, BATCH_UPDATE_GROUP_PROMPT]
    )
    def test_group_wiki_prompt_rejects_bot_originated_memes(self, prompt):
        assert "群内梗、文化和常用语只能从真人成员的明确发言归纳" in prompt
        assert "无法归因到真人成员的明确发言，必须删除" in prompt


# ── 3. assistant-history evidence exclusion (self-refine) ────────────────

class TestAssistantHistoryExclusion:
    def test_strip_assistant_history_lines(self):
        evidence = (
            "<session>\n聊天：测试群\n</session>\n\n"
            "<history>\nAlice：你好\n我（09:01）：我是Bot历史回复\n我：另一条Bot回复\nBob：在吗\n</history>\n\n"
            "<tool_results>\n我：工具结果不是对话行\n</tool_results>"
        )
        cleaned = strip_assistant_history_lines(evidence)

        # <history> 内的 我： 行被剔除
        assert "我是Bot历史回复" not in cleaned
        assert "另一条Bot回复" not in cleaned
        # 其他角色的历史行保留
        assert "Alice：你好" in cleaned
        assert "Bob：在吗" in cleaned
        # 非 history 区块（tool_results / session）保留
        assert "<tool_results>" in cleaned
        assert "工具结果不是对话行" in cleaned
        assert "<session>" in cleaned

    def test_no_history_block_unchanged(self):
        ev = "<context>\n<other_info>一些记忆</other_info>\n</context>"
        assert strip_assistant_history_lines(ev) == ev

    def test_unread_and_context_preserved(self):
        evidence = (
            "<context>\n<group_info>群记忆内容</group_info>\n</context>\n\n"
            "<unread>\n1. Alice：新消息\n</unread>"
        )
        cleaned = strip_assistant_history_lines(evidence)
        assert "群记忆内容" in cleaned
        assert "Alice：新消息" in cleaned

    def test_structured_self_message_is_removed_from_history(self):
        evidence = (
            '<history>\n<message role="other">Alice：你好</message>\n'
            '<message role="self">我：Bot历史回复</message>\n</history>\n'
            '<unread><message role="other">Alice：在吗</message></unread>'
        )

        cleaned = strip_assistant_history_lines(evidence)

        assert "Bot历史回复" not in cleaned
        assert "Alice：你好" in cleaned
        assert "Alice：在吗" in cleaned


# ── 4. untrusted [待验证] line exclusion from runtime retrieval ──────────

class TestUntrustedLineExclusion:
    def test_strip_unverified_lines_removes_every_marked_line(self):
        wiki = (
            "## 基本信息\n"
            "- 姓名：Alice\n"
            "- 公司：某银行 [待验证]\n"
            "1. 学校：某大学 [待验证]\n"
            "> 城市：甲城 [待验证]\n"
            "职业：工程师 [待验证]\n"
            "## 近期动态\n"
            "- 2026-06-01 稳定内容\n"
        )
        cleaned = strip_unverified_lines(wiki)
        assert "公司：某银行 [待验证]" not in cleaned
        assert "学校：某大学 [待验证]" not in cleaned
        assert "城市：甲城 [待验证]" not in cleaned
        assert "职业：工程师 [待验证]" not in cleaned
        assert "- 姓名：Alice" in cleaned
        assert "2026-06-01 稳定内容" in cleaned

    def test_get_user_memory_excludes_unverified_lines(self, engine):
        _write_user_wiki(
            engine, "SyntheticPerson",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 公司：某银行 [待验证]\n- 城市：甲城\n",
        )
        _write_noise_user(engine)
        result = engine.get_user_memory("SyntheticPerson", max_chars=10000)
        assert "某银行 [待验证]" not in result
        assert "城市：甲城" in result

    def test_search_keyword_excludes_unverified_lines(self, engine):
        _write_user_wiki(
            engine, "SyntheticPerson",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 公司：某银行 [待验证]\n- 城市：甲城\n",
        )
        _write_noise_user(engine)
        result = engine.search_keyword("SyntheticPerson", max_chars=10000)
        assert "某银行 [待验证]" not in result
        assert "城市：甲城" in result


# ── 5. generic fact-check issue policy (no regex semantic parser) ────────

class TestGenericFactIssuePolicy:
    """通用证据策略：unknown 与 contradicted 均阻塞；nonfactual 永不阻塞；
    entailed 全部进入独立复核（不做方向性关键词过滤）。"""

    def test_contradicted_blocks(self):
        claims = [
            {"claim": "某商品在甲城更便宜", "verdict": "contradicted", "reason": "证据相反"},
        ]
        issues, verify = _collect_fact_issues(claims)
        assert issues == ["事实矛盾：某商品在甲城更便宜；证据相反"]
        assert verify == []

    def test_unknown_blocks_like_contradicted(self):
        claims = [
            {"claim": "某人在乙城某机构工作", "verdict": "unknown", "reason": "无直接证据"},
        ]
        issues, verify = _collect_fact_issues(claims)
        assert issues == ["事实无依据：某人在乙城某机构工作；无直接证据"]
        assert verify == []

    def test_unknown_without_reason_still_blocks(self):
        claims = [{"claim": "某人是用户中学同学", "verdict": "unknown"}]
        issues, verify = _collect_fact_issues(claims)
        assert issues == ["事实无依据：某人是用户中学同学"]
        assert verify == []

    def test_nonfactual_is_not_blocking(self):
        claims = [{"claim": "某人会飞", "verdict": "nonfactual", "reason": "荒诞夸张"}]
        issues, verify = _collect_fact_issues(claims)
        assert issues == []
        assert verify == []

    def test_every_entailed_claim_goes_to_verifier(self):
        # 所有 entailed 命题都进入独立复核候选，不做方向性关键词过滤。
        claims = [
            {"claim": "某人在甲城某机构工作", "verdict": "entailed", "reason": "直接说明"},
            {"claim": "某人是用户老朋友", "verdict": "entailed", "reason": "直接说明"},
        ]
        issues, verify = _collect_fact_issues(claims)
        assert issues == []
        assert verify == ["某人在甲城某机构工作", "某人是用户老朋友"]

    def test_mixed_claims(self):
        claims = [
            {"claim": "A", "verdict": "contradicted", "reason": "r1"},
            {"claim": "B", "verdict": "unknown", "reason": "r2"},
            {"claim": "C", "verdict": "nonfactual", "reason": "r3"},
            {"claim": "D", "verdict": "entailed", "reason": "r4"},
        ]
        issues, verify = _collect_fact_issues(claims)
        assert issues == ["事实矛盾：A；r1", "事实无依据：B；r2"]
        assert verify == ["D"]

    def test_malformed_claims_ignored(self):
        claims = [{"claim": "无verdict"}, "not-a-dict", None, 123]
        issues, verify = _collect_fact_issues(claims)
        assert issues == []
        assert verify == []

    def test_empty_and_non_list_inputs(self):
        assert _collect_fact_issues([]) == ([], [])
        assert _collect_fact_issues(None) == ([], [])
        assert _collect_fact_issues("not-a-list") == ([], [])


# ── 6. no merging of contradictory facts ─────────────────────────────────

class TestNoMergingOfContradictoryFacts:
    def test_correction_is_distinct_authoritative_block(self, engine):
        engine._entity_corrections = {"SyntheticPerson": ["SyntheticPerson 不在 ZetaBank 工作"]}
        # wiki 里残留旧矛盾说法
        _write_user_wiki(
            engine, "SyntheticPerson",
            "## 基本信息\n- 姓名：SyntheticPerson\n- 当前公司：ZetaBank\n",
        )
        _write_noise_user(engine)

        result = engine.get_user_memory("SyntheticPerson", max_chars=10000)

        # 纠正与 wiki 是分开的两个区块：纠正在前、wiki 旧说法在后（不合并、不改写）
        assert "权威纠正" in result
        assert result.index("SyntheticPerson 不在 ZetaBank 工作") < result.index("ZetaBank")
        # 旧说法保留但不占主导（由上层用纠正做约束）
        assert "当前公司：ZetaBank" in result
