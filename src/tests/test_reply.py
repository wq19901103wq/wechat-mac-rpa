#!/usr/bin/env python3
"""Tests for L4 Reply modules."""

import json
import time
import xml.etree.ElementTree as ET
import pytest

from src.models.base import ChatMessage, SenderType
from src.reply.generator import ReplyGenerator
from src.reply.policy import ReplyPolicy
from src.session.global_store import ChatState
from src.tools import get_registry, register_builtin_tools

# 测试用全局工具注册表
_TEST_TOOL_REGISTRY = get_registry()
register_builtin_tools(_TEST_TOOL_REGISTRY)


class TestReplyPolicy:
    @pytest.mark.parametrize('message_type,text', [('image', ''), ('image', '图片上的文字'), ('mixed', '')])
    def test_images_are_passed_to_model_for_reply_decision(self, message_type, text):
        msg = ChatMessage(text, 'friend', SenderType.OTHER, 'example',
                          message_type=message_type, image_description='灰色的加载失败图标',
                          image_text='图片上的文字')
        assert ReplyPolicy().should_reply(msg, ChatState('c1', 'example')) is True

    def test_self_message_returns_false(self):
        policy = ReplyPolicy()
        session = ChatState(chat_id="c1", chat_name="Friend")
        msg = ChatMessage(
            text="hello",
            sender="me",
            sender_type=SenderType.SELF,
            chat_name="Friend",
        )
        assert policy.should_reply(msg, session) is False

    def test_system_message_returns_false(self):
        policy = ReplyPolicy()
        session = ChatState(chat_id="c1", chat_name="Friend")
        msg = ChatMessage(
            text="system alert",
            sender="system",
            sender_type=SenderType.SYSTEM,
            chat_name="Friend",
        )
        assert policy.should_reply(msg, session) is False

    def test_group_chat_with_at_returns_true(self):
        policy = ReplyPolicy()
        from src.session.global_store import ChatState
        session = ChatState(chat_id="c1", chat_name="Group (3)")
        msg = ChatMessage(
            text="@me hello",
            sender="friend",
            sender_type=SenderType.OTHER,
            chat_name="Group (3)",
            is_at_me=True,
        )
        assert policy.should_reply(msg, session) is True

    def test_normal_private_chat_returns_true(self):
        policy = ReplyPolicy()
        from src.session.global_store import ChatState
        session = ChatState(chat_id="c1", chat_name="Alice")
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        assert policy.should_reply(msg, session) is True


class TestReplyGenerator:
    def test_skill_content_override_precedes_disk_and_allows_empty_body(self):
        overridden = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            skill_content_overrides={"group_banter": "frozen old body"},
        )
        empty = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            skill_content_overrides={"group_banter": ""},
        )

        assert overridden._load_skill_content("group_banter") == "frozen old body"
        assert empty._load_skill_content("group_banter") == ""

    def test_latest_turn_form_sorting_is_private_only_by_default(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)

        assert gen.persona_few_shot_retriever.prefer_latest_turn_form is True
        assert gen.group_persona_few_shot_retriever.prefer_latest_turn_form is False

        disabled = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            prefer_latest_turn_form=False,
        )
        assert disabled.persona_few_shot_retriever.prefer_latest_turn_form is False
        assert disabled.group_persona_few_shot_retriever.prefer_latest_turn_form is False

        enabled_for_both = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            group_prefer_latest_turn_form=True,
        )
        assert enabled_for_both.persona_few_shot_retriever.prefer_latest_turn_form is True
        assert enabled_for_both.group_persona_few_shot_retriever.prefer_latest_turn_form is True

    def test_information_gap_response_move_is_enabled_for_both_retrievers_by_default(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)

        assert gen.persona_few_shot_retriever.enable_information_gap_response_move is True
        assert gen.group_persona_few_shot_retriever.enable_information_gap_response_move is True

        disabled = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_information_gap_response_move=False,
        )
        assert disabled.persona_few_shot_retriever.enable_information_gap_response_move is False
        assert disabled.group_persona_few_shot_retriever.enable_information_gap_response_move is False

    def test_prompt_is_one_xml_request_without_unread_duplication(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        history = ChatMessage(
            text="旧消息",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="A&B",
        )
        unread = ChatMessage(
            text="新消息 <确认>",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="A&B",
        )

        prompt = gen._build_user_prompt([unread], [history, unread])
        root = ET.fromstring(prompt)

        assert root.tag == "request"
        assert [message.text for message in root.findall("./history/message")] == ["Alice：旧消息"]
        assert [message.text for message in root.findall("./unread/message")] == ["Alice：新消息 <确认>"]

    def test_self_history_is_separated_and_forbidden_after_unread(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        other = ChatMessage(
            text="真人旧消息",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        self_reply = ChatMessage(
            text="Bot 已用旧梗",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = ChatMessage(
            text="真人新消息",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        prompt = gen._build_user_prompt([unread], [other, self_reply, unread], is_group=True)
        root = ET.fromstring(prompt)

        assert [message.text for message in root.findall("./history/message")] == ["Alice：真人旧消息"]
        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.attrib["reuse"] == "forbidden"
        assert [message.text for message in consumed.findall("./message")] == ["我：Bot 已用旧梗"]
        guard = root.find("./reply_guard")
        assert guard is not None
        assert guard.attrib == {"priority": "final", "enforcement": "hard"}
        assert "不得 callback" in "".join(guard.itertext())
        assert "不能成为回复的主语、宾语、修饰对象、背景或隐含前提" in "".join(guard.itertext())
        assert "没有独立新内容" in "".join(guard.itertext())
        assert "明确提问、请求解释或继续对话不适用此条" in "".join(guard.itertext())
        assert [child.tag for child in root][-3:] == ["unread", "consumed_self_replies", "reply_guard"]

    def test_reply_guard_exists_without_self_history(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        unread = ChatMessage(
            text="真人新消息",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))

        assert [child.tag for child in root][-1] == "reply_guard"
        assert "必须按 identity_provocation 处理" in "".join(root.find("./reply_guard").itertext())

    def test_group_unified_history_keeps_timeline_and_self_reuse_guard(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_unified_history=True,
        )
        history = [
            ChatMessage(
                text="先问一下",
                sender="Alice",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
            ChatMessage(
                text="我已经回过",
                sender="我",
                sender_type=SenderType.SELF,
                chat_name="测试群",
            ),
            ChatMessage(
                text="接着回应你",
                sender="Bob",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]
        unread = ChatMessage(
            text="现在继续说",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(
            gen._build_user_prompt([unread], [*history, unread], is_group=True)
        )
        messages = root.findall("./history/message")

        assert [message.attrib["index"] for message in messages] == ["1", "2", "3"]
        assert [message.attrib["role"] for message in messages] == ["other", "self", "other"]
        assert [message.text for message in messages] == [
            "Alice：先问一下",
            "我：我已经回过",
            "Bob：接着回应你",
        ]
        assert root.find("./consumed_self_replies") is None
        guard_text = "".join(root.find("./reply_guard").itertext())
        assert "只用于按原时间顺序理解谁在接谁的话" in guard_text
        assert "不得照抄本人旧句、复用旧梗" in guard_text

    def test_group_unified_history_does_not_change_private_history(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_unified_history=True,
        )
        self_reply = ChatMessage(
            text="我已经回过",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="Alice",
        )
        unread = ChatMessage(
            text="继续说",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )

        root = ET.fromstring(
            gen._build_user_prompt([unread], [self_reply, unread], is_group=False)
        )

        assert root.find("./history/message") is None
        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.find("./message").attrib["role"] == "self"

    def test_group_unified_relevant_self_selects_from_one_timeline(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_unified_relevant_self=True,
        )
        history = [
            ChatMessage(
                text="今晚我六点下课",
                sender="我",
                sender_type=SenderType.SELF,
                chat_name="测试群",
            ),
            ChatMessage(
                text="那你七点再来",
                sender="Alice",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]
        unread = ChatMessage(
            text="还是六点直接过来吧",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(
            gen._build_user_prompt([unread], [*history, unread], is_group=True)
        )

        assert [message.attrib["role"] for message in root.findall("./history/message")] == [
            "self",
            "other",
        ]
        assert root.find("./consumed_self_replies") is None
        assert root.find("./current_stance_evidence") is None
        assert root.find("./group_unified_relevant_self") is None
        prompt = gen._with_group_unified_relevant_self_schema(gen._system_prompt(is_group=True))
        assert 'name="directly_answered_self_quote"' in prompt
        assert "从 history 中 role=self" in prompt

        request = gen._append_request_tail_sections(
            ET.tostring(root, encoding="unicode"),
            [
                '<group_unified_relevant_self priority="final" enforcement="hard">'
                "test"
                "</group_unified_relevant_self>"
            ],
        )
        request_root = ET.fromstring(request)
        assert request_root.find("./group_unified_relevant_self") is not None

    def test_active_thread_candidate_uses_self_history_as_context_without_reusing_it(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_active_thread_continuation=True,
        )
        self_reply = ChatMessage(
            text="我外婆做饭还行，我妈就不太行",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = ChatMessage(
            text="那你家现在谁做饭",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [self_reply, unread], is_group=True))
        consumed = root.find("./consumed_self_replies")
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert consumed is not None
        assert consumed.attrib["reuse"] == "context_only"
        assert "优先顺着同一条线继续说" in guard_text
        assert "不得照抄旧句、重复旧梗、虚构新细节" in guard_text
        assert "新的明确问题或请求仍优先" in guard_text
        assert "已用素材不能成为回复的主语" not in guard_text

    def test_response_action_candidate_limits_presence_reply_to_known_facts(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_response_action_grounding=True,
        )
        unread = ChatMessage(
            text="人呢",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert "把叫人本身当作最新问题先接住" in guard_text
        assert "第一项只用“在啊”或“咋了”" in guard_text
        assert "第二项再单独反问“你人呢”或“你在哪”" in guard_text
        assert "不要把两项合成一句" in guard_text
        assert "不能改答前一条" in guard_text
        assert "整组 replies 只完成这一个作用" in guard_text
        assert "不得追加回应前一条、建议、解释或调侃" in guard_text
        assert "不得补充所在地、正在做的事、到达状态或时间承诺" in guard_text

    def test_response_action_candidate_targets_latest_presence_call(self):
        unread = [
            ChatMessage(
                text="打了疫苗全身不舒服",
                sender="Bob",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
            ChatMessage(
                text="人呢",
                sender="Bob",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]

        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_response_action_grounding=True,
        )
        baseline_root = ET.fromstring(baseline._build_user_prompt(unread, unread, is_group=True))
        candidate_root = ET.fromstring(candidate._build_user_prompt(unread, unread, is_group=True))

        assert baseline_root.find("./response_target_candidate") is None
        target = candidate_root.find("./response_target_candidate")
        assert target is not None
        assert target.attrib["index"] == "2"
        assert target.attrib["kind"] == "presence_call"

    def test_latest_request_first_handles_general_request_before_banter(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text="你们啥时候去？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert root.find("./response_target_candidate").attrib["kind"] == "explicit_request"
        target_text = "".join(root.find("./response_target_candidate").itertext())
        assert "回复只处理这个诉求" in guard_text
        assert "目标原文：你们啥时候去？" in target_text
        assert "这只是待核对的候选，不是强制目标" in target_text
        assert "回答、拒绝、澄清或追问后立即结束" in guard_text
        assert "不得继续旧话题、调侃、建议或总结" in guard_text
        assert "不要机械追最后一句" in guard_text
        assert "不要抢答明确问别人的话" in guard_text
        assert "本轮要处理的诉求原句是什么" in guard_text
        assert "支持答案的当前原句、可信人设、可靠记忆或工具结果是什么" in guard_text
        assert "不要输出这份核对" in guard_text
        assert "不得替自己回答是或否、给数字或补细节" in guard_text
        assert "只简短说不确定，或问一个确实必要的问题" in guard_text
        assert "未来邀请没有本人安排的直接证据" in guard_text
        assert "不得承诺“回头”“稍后”“我去”“我会”" in guard_text

    def test_latest_request_first_targets_unresolved_need_not_last_question_mark(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = [
            ChatMessage(
                text="做饭或者陪我去医院",
                sender="Alice",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
            ChatMessage(
                text="芊儿要下班了吗？",
                sender="Bob",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]

        root = ET.fromstring(gen._build_user_prompt(unread, unread, is_group=True))
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert "找出最近仍未处理且确实指向你的诉求" in guard_text
        assert "不要机械追最后一句" in guard_text
        assert "不要抢答明确问别人的话" in guard_text

    def test_latest_request_first_forbids_summary_style_social_replies(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text="晚上去水裹过夜，第二天六点火车",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert "不得继续旧话题、调侃、建议或总结" in guard_text
        assert "未来邀请没有本人安排的直接证据" in guard_text

    @pytest.mark.parametrize(
        "question",
        [
            "你下班了吗？",
            "你到家了吗？",
            "帮我问问能不能换班",
            "公积金有什么政策优惠？",
        ],
    )
    def test_latest_request_first_does_not_invent_answers_to_private_fact_questions(
        self, question
    ):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text=question,
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))
        target = root.find("./response_target_candidate")
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert target is not None
        assert target.attrib["kind"] == "explicit_request"
        assert "不得替自己回答是或否、给数字或补细节" in guard_text
        assert "只简短说不确定，或问一个确实必要的问题" in guard_text
        assert "需要未来执行的请求" in guard_text
        assert "不得承诺“回头”“稍后”“我去”“我会”" in guard_text
        assert "不得继续旧话题、调侃、建议或总结" in guard_text

    @pytest.mark.parametrize(
        "message",
        [
            "你们出发了么",
            "你到家了么",
            "明天去杭州踏个青，有人一起不",
            "PDD不给员工发物资吗",
            "贷款利率有什么优惠",
            "我发给你麻麻了",
        ],
    )
    def test_latest_request_first_requires_evidence_without_skipping_the_request(
        self, message
    ):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text=message,
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert "回复只处理这个诉求" in guard_text
        assert "不得替自己回答是或否、给数字或补细节" in guard_text
        assert "只简短说不确定，或问一个确实必要的问题" in guard_text

    def test_latest_request_first_preserves_addressee_check_in_groups(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text="所以大佬为什么要出来吃火鸡@AmourDeMai",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert "结合说话人、称呼、@ 对象和 history" in guard_text
        assert "不要抢答明确问别人的话" in guard_text

    def test_latest_request_first_is_isolated_from_current_baseline(self):
        baseline = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_response_action_grounding=True,
        )
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_response_action_grounding=True,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text="好的，就这一个对吧？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        baseline_guard = "".join(
            ET.fromstring(baseline._build_user_prompt([unread], [unread], is_group=True))
            .find("./reply_guard")
            .itertext()
        )
        candidate_guard = "".join(
            ET.fromstring(candidate._build_user_prompt([unread], [unread], is_group=True))
            .find("./reply_guard")
            .itertext()
        )

        assert "回复只处理这个诉求" not in baseline_guard
        assert "回复只处理这个诉求" in candidate_guard

    def test_latest_request_postcheck_rewrites_only_when_checker_flags_it(self):
        class CheckClient:
            def chat(self, **kwargs):
                return (
                    '{"decision":"rewrite","reason":"漏回最新问题",'
                    '"replies":["我算过了"]}'
                )

        gen = ReplyGenerator(
            llm_client=CheckClient(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        prompt = ('<request><response_target_candidate source="unread">'
                  '你算过积分吗</response_target_candidate></request>')

        replies, messages = gen._check_latest_request_reply(
            prompt,
            ["我们没用"],
            [{"role": "user", "content": prompt}],
            float("inf"),
        )

        assert replies == ["我算过了"]
        assert messages[-1]["role"] == "assistant"
        assert gen.last_latest_request_grounding["attempted"] is True
        assert gen.last_latest_request_grounding["status"] == "rewritten"
        assert gen.last_latest_request_grounding["reply_before"] == ["我们没用"]
        assert gen.last_latest_request_grounding["reply_after"] == ["我算过了"]

    def test_latest_request_postcheck_keeps_passed_reply_byte_for_byte(self):
        class CheckClient:
            def chat(self, **kwargs):
                return (
                    '{"decision":"pass","reason":"已直接回答",'
                    '"replies":["被检查器改过的文字"]}'
                )

        gen = ReplyGenerator(
            llm_client=CheckClient(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        prompt = ('<request><response_target_candidate source="unread">'
                  '你算过积分吗</response_target_candidate></request>')

        replies, _ = gen._check_latest_request_reply(
            prompt,
            ["原回复"],
            [{"role": "user", "content": prompt}],
            float("inf"),
        )

        assert replies == ["原回复"]
        assert gen.last_latest_request_grounding["status"] == "passed"

    def test_latest_request_postcheck_is_skipped_without_request_candidate(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise AssertionError("baseline-like input must not call the checker")

        gen = ReplyGenerator(
            llm_client=FailingClient(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )

        replies, messages = gen._check_latest_request_reply(
            "<request />",
            ["原回复"],
            [{"role": "user", "content": "<request />"}],
            float("inf"),
        )

        assert replies == ["原回复"]
        assert len(messages) == 1
        assert gen.last_latest_request_grounding["attempted"] is False
        assert gen.last_latest_request_grounding["status"] == "no_request_candidate"

    def test_latest_request_candidate_includes_direct_send_request(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text="户型图，精装修现场照片发一下，他帮你去问问",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )

        prompt = gen._build_user_prompt([unread], [unread], is_group=False)

        assert "<response_target_candidate " in prompt
        assert "户型图，精装修现场照片发一下" in prompt

    def test_reply_guard_covers_whole_batch_and_group_restraint(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        unread = ChatMessage(
            text="[表情包]",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            message_type="emoji",
        )

        prompt = gen._build_user_prompt([unread], [unread], is_group=True)
        guard_text = "".join(ET.fromstring(prompt).find("./reply_guard").itertext())

        assert "扫描全部 unread" in guard_text
        assert "明确 @ 其他人" in guard_text
        assert "连续图片、表情、视频或链接" in guard_text
        assert "追问你先前说过的话" in guard_text
        assert "按固定顺序决策" in guard_text
        assert "最近且尚未解决的" in guard_text
        assert "不能覆盖它" in guard_text
        assert "问号对应的原句" in guard_text
        assert "未发生、正在发生、马上发生、已经发生" in guard_text
        assert "不要生造外号" in guard_text
        assert "不得猜测属相" in guard_text
        assert "单次经历只能证明那一次发生过" in guard_text
        assert "不能据此断言当前普遍允许" in guard_text
        assert "只有历史个例时不得直接回答“可以”" in guard_text
        assert "先给短而直接的正常回应" in guard_text

    def test_explicit_question_before_media_is_marked_as_response_target(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        question = ChatMessage(
            text="你为什么觉得这个价格合理？",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        image = ChatMessage(
            text="[图片]",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            message_type="image",
        )

        root = ET.fromstring(
            gen._build_user_prompt([question, image], [question, image], is_group=True)
        )
        target = root.find("./response_target_candidate")

        assert target is not None
        assert target.attrib == {
            "source": "unread",
            "index": "1",
            "kind": "explicit_request",
            "requires_addressee_check": "true",
        }
        assert "后续图片、表情、链接或闲聊不得把它盖掉" in (target.text or "")
        assert "response_target_candidate" in "".join(root.find("./reply_guard").itertext())

    def test_plain_statement_does_not_create_response_target(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        unread = ChatMessage(
            text="今天市场波动挺大",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))

        assert root.find("./response_target_candidate") is None

    def test_private_reply_guard_does_not_force_group_one_sentence_shape(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        unread = ChatMessage(
            text="帮我分别回答这三个问题",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="Bob",
        )

        prompt = gen._build_user_prompt([unread], [unread], is_group=False)
        guard_text = "".join(ET.fromstring(prompt).find("./reply_guard").itertext())

        assert "群聊优先短而克制" not in guard_text

    def test_reply_guard_allows_answering_direct_followup_to_consumed_self_reply(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        self_reply = ChatMessage(
            text="你要不设下规则",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = ChatMessage(
            text="啥规则",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        prompt = gen._build_user_prompt([unread], [self_reply, unread], is_group=True)
        guard_text = "".join(ET.fromstring(prompt).find("./reply_guard").itertext())

        assert "必须用它理解追问并直接回答" in guard_text
        assert "明确提问、请求解释或继续对话不适用此条" in guard_text


    def test_system_prompt_does_not_embed_skill_catalog(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)

        prompt = gen._system_prompt()

        assert ET.fromstring(prompt).tag == "instructions"
        assert "<available_skills>" not in prompt
        assert "<active_skill" not in prompt

    def test_appended_sections_stay_before_final_reply_guard(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        self_reply = ChatMessage(
            text="Bot 已用旧梗",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = ChatMessage(
            text="真人新消息",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        prompt = gen._build_user_prompt([unread], [self_reply, unread], is_group=True)

        prompt = gen._append_request_sections(
            prompt,
            [
                '<active_skill name="group_banter"><skill /></active_skill>',
                '<style_examples source="verified_human"><example /></style_examples>',
            ],
        )
        root = ET.fromstring(prompt)

        assert [child.tag for child in root][-3:] == [
            "active_skill",
            "style_examples",
            "reply_guard",
        ]

    def test_latest_request_contract_is_final_request_section(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text="你们现在有模型用 online learning 的吗？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        prompt = gen._build_user_prompt([unread], [unread], is_group=True)
        prompt = gen._append_request_sections(
            prompt,
            [
                '<active_skill name="group_banter"><skill /></active_skill>',
                '<style_examples source="verified_human"><example /></style_examples>',
            ],
        )
        prompt = gen._append_request_tail_sections(
            prompt,
            ['<latest_request_contract priority="final" enforcement="hard" />'],
        )
        root = ET.fromstring(prompt)

        assert [child.tag for child in root][-4:] == [
            "active_skill",
            "style_examples",
            "reply_guard",
            "latest_request_contract",
        ]

    def test_baseline_has_no_latest_request_contract(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        unread = ChatMessage(
            text="你还有多少？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="Bob",
        )

        prompt = gen._build_user_prompt([unread], [unread], is_group=False)

        assert ET.fromstring(prompt).find("./latest_request_contract") is None

    @pytest.mark.parametrize("reply_restraint, expected", [(False, True), (True, False)])
    def test_offline_force_reply_contract_follows_reply_restraint(
        self, monkeypatch, reply_restraint, expected
    ):
        gen = ReplyGenerator(llm_client=object(), tool_registry=_TEST_TOOL_REGISTRY)
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["收到"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        unread = ChatMessage(
            text="群里随口聊一句",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        gen.generate(
            [unread], [unread], is_group=True, enable_reply_restraint=reply_restraint
        )
        root = ET.fromstring(gen.last_user_prompt)

        assert (root.find("./offline_force_reply_contract") is not None) is expected

    @pytest.mark.parametrize("enabled", [False, True])
    def test_no_restatement_action_contract_is_candidate_only(self, monkeypatch, enabled):
        gen = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_no_restatement_action=enabled,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["问下来源"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        unread = ChatMessage(
            text="我叔叔说回流员工会升职加薪",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        gen.generate([unread], [unread], is_group=True)
        root = ET.fromstring(gen.last_user_prompt)

        assert (root.find("./no_restatement_action") is not None) is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_plain_response_before_styling_contract_is_candidate_only(self, monkeypatch, enabled):
        gen = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_plain_response_before_styling=enabled,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["这也太小道了"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        unread = ChatMessage(
            text="又听说他们要裁员了",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        gen.generate([unread], [unread], is_group=True, enable_reply_restraint=False)
        root = ET.fromstring(gen.last_user_prompt)

        assert (root.find("./plain_response_before_styling") is not None) is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_current_stance_continuity_contract_is_candidate_only(self, monkeypatch, enabled):
        gen = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=enabled,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["还得催下"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        messages = [
            ChatMessage(
                sender="我",
                text="晚上还要开会，得催下",
                sender_type=SenderType.SELF,
                chat_name="群",
            ),
            ChatMessage(
                sender="小王",
                text="晚点就晚点吧",
                sender_type=SenderType.OTHER,
                chat_name="群",
            ),
        ]

        gen.generate([messages[-1]], messages, is_group=True, enable_reply_restraint=False)

        root = ET.fromstring(gen.last_user_prompt)
        assert (root.find("./current_stance_continuity") is not None) is enabled
        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.attrib["reuse"] == ("context_only" if enabled else "forbidden")
        evidence = root.find("./current_stance_evidence")
        assert (evidence is not None) is enabled
        assert ('"current_stance":"none"' in gen.last_system_prompt) is enabled
        assert ('<field name="current_stance">' in gen.last_system_prompt) is enabled
        assert ('先写“拒绝改变安排：”' in gen.last_system_prompt) is enabled
        assert ('逐字引用最新且仍同时成立的 1 至 3 条' in gen.last_system_prompt) is enabled
        assert ('多条之间用“ || ”分隔' in gen.last_system_prompt) is enabled
        assert ('必须同时满足 current_stance 中所有' in gen.last_system_prompt) is enabled
        assert ('调侃、预测、假设' in gen.last_system_prompt) is enabled
        continuity = root.find("./current_stance_continuity")
        continuity_text = "".join(continuity.itertext()) if continuity is not None else ""
        assert ('直接要求、建议或提出方案改变' in continuity_text) is enabled
        assert ('必须填 none' in continuity_text) is enabled
        assert ('写出每一条 reply 后' in continuity_text) is enabled
        assert ('必须丢弃并重写' in continuity_text) is enabled
        assert ('较早安排失效' in continuity_text) is enabled
        assert ('不得把不同阶段或互相冲突的安排拼在一起' in continuity_text) is enabled
        assert ('生成每条 reply 后必须逐条核对' in gen.last_system_prompt) is enabled
        assert ('增加无依据事实' in gen.last_system_prompt) is enabled
        assert ('本人后说的话若更新了人物、行动或先后关系' in gen.last_system_prompt) is enabled

    def test_current_stance_evidence_contains_only_last_eight_self_texts(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        history = [
            ChatMessage(
                sender="我",
                text=f"本人文本{i}",
                sender_type=SenderType.SELF,
                chat_name="群",
            )
            for i in range(10)
        ]
        history.extend([
            ChatMessage(
                sender="我",
                text="图片占位",
                sender_type=SenderType.SELF,
                message_type="image",
                chat_name="群",
            ),
            ChatMessage(
                sender="小王",
                text="晚点就晚点吧",
                sender_type=SenderType.OTHER,
                chat_name="群",
            ),
        ])

        root = ET.fromstring(
            gen._build_user_prompt([history[-1]], history, is_group=True)
        )
        evidence = root.find("./current_stance_evidence")

        assert evidence is not None
        assert evidence.attrib == {
            "source": "recent_self_text",
            "limit": "8",
            "reuse": "context_only",
        }
        assert [message.text for message in evidence.findall("./message")] == [
            f"我：本人文本{i}" for i in range(2, 10)
        ]
        assert "直接要求、建议或提出方案改变" in "".join(evidence.itertext())
        assert "按从旧到新的顺序排列" in "".join(evidence.itertext())
        assert "前面的安排失效" in "".join(evidence.itertext())

    @pytest.mark.parametrize(
        ("enabled", "is_group", "expected"),
        [
            (False, True, False),
            (True, True, True),
            (True, False, False),
        ],
    )
    def test_group_relevant_self_evidence_is_group_candidate_only(
        self, enabled, is_group, expected
    ):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_relevant_self_evidence=enabled,
        )
        messages = [
            ChatMessage(
                sender="我",
                text="晚上还要上班，得快点",
                sender_type=SenderType.SELF,
                chat_name="测试会话",
            ),
            ChatMessage(
                sender="小王",
                text="晚点就晚点吧",
                sender_type=SenderType.OTHER,
                chat_name="测试会话",
            ),
        ]

        root = ET.fromstring(
            gen._build_user_prompt([messages[-1]], messages, is_group=is_group)
        )

        contract = root.find("./group_relevant_self_evidence")
        assert (contract is not None) is expected
        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.attrib["reuse"] == ("context_only" if expected else "forbidden")
        evidence = root.find("./current_stance_evidence")
        assert (evidence is not None) is expected
        if expected:
            text = "".join(contract.itertext()) + "".join(evidence.itertext())
            assert "明确否定、纠正、回答、挑战、辩解或反驳" in text
            assert "忽略后会改变本人的立场" in text
            assert "仅话题相同、用词相同、时间接近" in text
            assert "directly_answered_self_quote" in text
            assert "字段为 none 时" in text
            assert "不得照抄原句" in text
            assert "不得猜测缺失图片内容" in text

    def test_group_relevant_self_evidence_does_not_change_private_prompt(self):
        messages = [
            ChatMessage(
                sender="我",
                text="晚上还要上班，得快点",
                sender_type=SenderType.SELF,
                chat_name="小王",
            ),
            ChatMessage(
                sender="小王",
                text="晚点就晚点吧",
                sender_type=SenderType.OTHER,
                chat_name="小王",
            ),
        ]
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_relevant_self_evidence=True,
        )

        assert candidate._build_user_prompt(
            [messages[-1]], messages, is_group=False
        ) == baseline._build_user_prompt([messages[-1]], messages, is_group=False)

    def test_group_self_context_freshness_boundary_changes_only_the_conflicting_rules(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        original = baseline._system_prompt()

        candidate = baseline._with_group_self_context_freshness_boundary(original)

        assert '"directly_answered_self_quote":"none"' in candidate
        assert '<field name="directly_answered_self_quote">' in candidate
        assert "逐字复制 1 条本人原话" in candidate
        assert "短感叹、图片、表情或没有明确指向的反应都必须填 none" in candidate
        assert "decision.directly_answered_self_quote 为 none 时，本规则完整适用" in candidate
        assert "仅允许把该原句表达的当前事实、状态、计划、限制或判断作为背景" in candidate
        assert "带回原句之外的旧人物或旧事件" in candidate
        assert "role=self 历史已经用作笑点的任何素材" in candidate
        assert original == baseline._system_prompt()

    def test_group_self_context_freshness_boundary_is_group_only(self, monkeypatch):
        messages = [
            ChatMessage(
                sender="我",
                text="晚上还要上班，得快点",
                sender_type=SenderType.SELF,
                chat_name="测试会话",
            ),
            ChatMessage(
                sender="小王",
                text="你晚上晚点下班就好了",
                sender_type=SenderType.OTHER,
                chat_name="测试会话",
            ),
        ]

        def build(is_group):
            gen = ReplyGenerator(
                llm_client=object(),
                tool_registry=_TEST_TOOL_REGISTRY,
                enable_group_relevant_self_evidence=True,
                enable_group_self_context_freshness_boundary=True,
            )
            gen.enable_self_refine = False
            monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
            monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
            monkeypatch.setattr(
                gen,
                "_react_generate",
                lambda prompt_messages, tools, deadline, chat_name: (["知道了"], prompt_messages),
            )
            monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
            gen.generate([messages[-1]], messages, is_group=is_group)
            return gen

        group = build(True)
        private = build(False)

        assert '"directly_answered_self_quote":"none"' in group.last_system_prompt
        assert '<field name="directly_answered_self_quote">' in group.last_system_prompt
        assert '"directly_answered_self_quote":"none"' not in private.last_system_prompt
        assert '<field name="directly_answered_self_quote">' not in private.last_system_prompt
        assert "role=self 历史已经用作笑点的任何素材" in private.last_system_prompt
        group_root = ET.fromstring(group.last_user_prompt)
        private_root = ET.fromstring(private.last_user_prompt)
        assert group_root.find("./group_relevant_self_evidence") is not None
        assert private_root.find("./group_relevant_self_evidence") is None
        group_rules = "".join(group_root.find("./group_relevant_self_evidence").itertext())
        assert "字段为 none 时" in group_rules
        assert "不得带回旧人物、旧事件、旧笑点、数字梗、比喻或虚构关系" in group_rules

    def test_group_adjacent_self_response_changes_only_the_short_response_boundary(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        original = baseline._with_group_self_context_freshness_boundary(
            baseline._system_prompt()
        )

        candidate = baseline._with_group_adjacent_self_response(original)

        assert "消息顺序上紧跟本人原话" in candidate
        assert "具体接话关系" in candidate
        assert "普通感叹" not in candidate
        assert "没有具体接话关系的反应仍必须填 none" in candidate
        assert original == baseline._with_group_self_context_freshness_boundary(
            baseline._system_prompt()
        )

    def test_group_adjacent_self_response_is_group_only_and_shared_by_generation_and_review(
        self, monkeypatch
    ):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["review"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "pass",
                    "answered_self_quote": "1是户数啊",
                    "reason": "",
                    "replies": ["0.7紧张的"],
                }, ensure_ascii=False)

        messages = [
            ChatMessage(
                sender="我",
                text="1是户数啊",
                sender_type=SenderType.SELF,
                chat_name="测试会话",
            ),
            ChatMessage(
                sender="林涛",
                text="我靠",
                sender_type=SenderType.OTHER,
                chat_name="测试会话",
            ),
        ]
        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_relevant_self_evidence=True,
            enable_group_self_context_freshness_boundary=True,
            enable_group_adjacent_self_response=True,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda prompt_messages, tools, deadline, chat_name: (["0.7紧张的"], prompt_messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)

        gen.generate([messages[-1]], messages, is_group=True)

        user_rules = "".join(
            ET.fromstring(gen.last_user_prompt)
            .find("./group_relevant_self_evidence")
            .itertext()
        )
        assert "消息顺序上紧跟本人原话" in gen.last_system_prompt
        assert "具体在回应该原话" in user_rules
        assert "普通感叹、仅同话题或仅时间相邻不得触发" in user_rules
        assert "紧邻本人消息后的短感叹" in captured["review"]
        assert "普通感叹仍填 none" in captured["review"]

        private = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_relevant_self_evidence=True,
            enable_group_self_context_freshness_boundary=True,
            enable_group_adjacent_self_response=True,
        )
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        assert private._build_user_prompt(
            [messages[-1]], messages, is_group=False
        ) == baseline._build_user_prompt([messages[-1]], messages, is_group=False)


    @pytest.mark.parametrize(
        ("enabled", "is_group", "expected"),
        [
            (False, True, False),
            (True, True, True),
            (True, False, False),
        ],
    )
    def test_group_speaker_target_attribution_is_group_candidate_only(
        self, monkeypatch, enabled, is_group, expected
    ):
        gen = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_speaker_target_attribution=enabled,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["还在催"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        messages = [
            ChatMessage(
                sender="我",
                text="这个今天得确认下来",
                sender_type=SenderType.SELF,
                chat_name="测试会话",
            ),
            ChatMessage(
                sender="小王",
                text="现在确认了吗",
                sender_type=SenderType.OTHER,
                chat_name="测试会话",
            ),
        ]

        gen.generate(
            [messages[-1]], messages, is_group=is_group, enable_reply_restraint=False
        )
        root = ET.fromstring(gen.last_user_prompt)

        assert (root.find("./group_speaker_target_attribution") is not None) is expected
        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.attrib["reuse"] == ("context_only" if expected else "forbidden")
        evidence = root.find("./current_stance_evidence")
        assert (evidence is not None) is expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("@[联系人] parker是我们公司平台的负责大佬", True),
            ("@王苊 parker是我们公司平台的负责大佬", True),
            ("@小王 这是我们老板", True),
            ("@小王 负责平台这块", True),
            ("@韭菜三号", False),
            ("饿晕了 / @韭菜三号", False),
            ("@小王 你先给我当三年司机练练手", False),
        ],
    )
    def test_explicit_third_party_guard_only_detects_named_introductions(
        self, text, expected
    ):
        message = ChatMessage(
            sender="介绍人",
            text=text,
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert ReplyGenerator._has_explicit_third_party_introduction([message]) is expected

    def test_explicit_third_party_guard_is_candidate_group_only(self):
        message = ChatMessage(
            sender="介绍人",
            text="@[联系人] parker是我们公司平台的负责大佬",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_explicit_third_party_guard=True,
        )

        baseline_group = ET.fromstring(
            baseline._build_user_prompt([message], [message], is_group=True)
        )
        candidate_group = ET.fromstring(
            candidate._build_user_prompt([message], [message], is_group=True)
        )
        candidate_private = ET.fromstring(
            candidate._build_user_prompt([message], [message], is_group=False)
        )

        assert baseline_group.find("./group_explicit_third_party_guard") is None
        assert candidate_group.find("./group_explicit_third_party_guard") is not None
        assert candidate_private.find("./group_explicit_third_party_guard") is None

    def test_explicit_third_party_guard_keeps_nontrigger_prompt_identical_to_baseline(self):
        message = ChatMessage(
            sender="小王",
            text="饿晕了 / @韭菜三号",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_explicit_third_party_guard=True,
        )

        assert candidate._build_user_prompt(
            [message], [message], is_group=True
        ) == baseline._build_user_prompt([message], [message], is_group=True)

    def test_group_target_action_contract_keeps_private_and_nontrigger_prompts_identical(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_target_action_contract=True,
        )
        message = ChatMessage(
            sender="小王",
            text="我们一起吃饭了",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [message], [message], is_group=False
        ) == baseline._build_user_prompt([message], [message], is_group=False)
        assert candidate._build_user_prompt(
            [message], [message], is_group=True
        ) == baseline._build_user_prompt([message], [message], is_group=True)

    def test_group_public_question_followup_only_changes_public_concrete_question(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_public_question_specific_followup=True,
        )
        question = ChatMessage(
            sender="小王",
            text="这是什么虫子？",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        nontrigger = ChatMessage(
            sender="小李",
            text="大家接龙报名？",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [question], [question], is_group=False
        ) == baseline._build_user_prompt([question], [question], is_group=False)
        assert candidate._build_user_prompt(
            [nontrigger], [nontrigger], is_group=True
        ) == baseline._build_user_prompt([nontrigger], [nontrigger], is_group=True)

        root = ET.fromstring(candidate._build_user_prompt(
            [question], [question], is_group=True
        ))
        contract = root.find("./group_public_question_specific_followup")
        assert contract is not None
        assert contract.attrib["enforcement"] == "guidance"
        assert contract.findtext("./speaker") == "小王"
        assert contract.findtext("./question") == "这是什么虫子？"
        rules = "\n".join(node.text or "" for node in contract.findall("./rule"))
        assert "不得编造" in rules
        assert "代替他人确认" in rules
        assert "不强迫正面作答" in rules

    @pytest.mark.parametrize(
        ("text", "is_at_me"),
        [
            ("@小王 这是什么虫子？", False),
            ("这是什么虫子？", True),
            ("你到哪了？", False),
            ("面包群收款发了么？", False),
            ("地址发我一下？", False),
            ("大家接龙报名？", False),
            ("咋了？", False),
        ],
    )
    def test_group_public_question_followup_excludes_nonpublic_or_risky_questions(
        self, text, is_at_me
    ):
        message = ChatMessage(
            sender="小王",
            text=text,
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            is_at_me=is_at_me,
        )

        assert ReplyGenerator._group_public_question_specific_followup_contract(
            [message], [message]
        ) is None

    def test_group_instant_self_correction_contract_uses_adjacent_self_text(self):
        previous = ChatMessage(
            sender="我",
            text="房价看板块 不看接到",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="不是这个同济融创玫瑰公馆吗？",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        contract = ReplyGenerator._group_instant_self_correction_contract(
            [incoming], [previous, incoming]
        )

        assert contract == {
            "previous_self_text": "房价看板块 不看接到",
            "intervening_text": "不是这个同济融创玫瑰公馆吗？",
        }

    @pytest.mark.parametrize(
        "unreplied,all_messages",
        [
            (
                lambda previous, incoming, older: [incoming, older],
                lambda previous, incoming, older: [previous, incoming, older],
            ),
            (
                lambda previous, incoming, older: [incoming],
                lambda previous, incoming, older: [previous, older, incoming],
            ),
        ],
    )
    def test_group_instant_self_correction_contract_rejects_non_immediate_shapes(
        self, unreplied, all_messages
    ):
        previous = ChatMessage(
            sender="我",
            text="安家跟弱智你不觉着么",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="我以前没看过",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        older = ChatMessage(
            sender="小李",
            text="旧消息",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert ReplyGenerator._group_instant_self_correction_contract(
            unreplied(previous, incoming, older),
            all_messages(previous, incoming, older),
        ) is None

    def test_group_instant_self_correction_contract_rejects_non_text(self):
        previous = ChatMessage(
            sender="我",
            text="上一句",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="[图片]",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            message_type="image",
        )

        assert ReplyGenerator._group_instant_self_correction_contract(
            [incoming], [previous, incoming]
        ) is None

    def test_group_instant_self_correction_changes_only_triggering_group_prompt(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_instant_self_correction=True,
        )
        previous = ChatMessage(
            sender="我",
            text="安家跟弱智你不觉着么",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="我以前没看过",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        )
        assert candidate._build_user_prompt(
            [incoming], [incoming], is_group=True
        ) == baseline._build_user_prompt([incoming], [incoming], is_group=True)

        root = ET.fromstring(candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=True
        ))
        contract = root.find("./group_instant_self_correction")
        assert contract is not None
        assert contract.get("enforcement") == "hard"
        assert contract.findtext("./previous_self_text") == "安家跟弱智你不觉着么"
        assert contract.findtext("./intervening_text") == "我以前没看过"
        assert "比较关系" in "".join(contract.itertext())

    @pytest.mark.parametrize("unread_count", [1, 2])
    def test_group_visible_self_stance_continuity_contract_accepts_strict_tail(
        self, unread_count
    ):
        previous = ChatMessage(
            sender="我",
            text="没事 一起还了 主要利息太高",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小王",
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for text in ("5还利息高？", "比银行便宜吧？")[:unread_count]
        ]

        contract = ReplyGenerator._group_visible_self_stance_continuity_contract(
            unread, [previous, *unread]
        )

        assert contract is not None
        assert contract["previous_self_message"] is previous
        assert contract["unread_messages"] == unread

    @pytest.mark.parametrize(
        "build_messages",
        [
            lambda previous, unread: (
                [
                    unread[0],
                    unread[1],
                    ChatMessage(
                        sender="小王",
                        text="第三条",
                        sender_type=SenderType.OTHER,
                        chat_name="测试群",
                    ),
                ],
                [
                    previous,
                    unread[0],
                    unread[1],
                    ChatMessage(
                        sender="小王",
                        text="第三条",
                        sender_type=SenderType.OTHER,
                        chat_name="测试群",
                    ),
                ],
            ),
            lambda previous, unread: (
                unread,
                [previous, unread[0], ChatMessage(
                    sender="小李",
                    text="插话",
                    sender_type=SenderType.OTHER,
                    chat_name="测试群",
                ), unread[1]],
            ),
        ],
    )
    def test_group_visible_self_stance_continuity_contract_rejects_non_strict_tail(
        self, build_messages
    ):
        previous = ChatMessage(
            sender="我",
            text="得先还上",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小王",
                text="我跌破29了",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
            ChatMessage(
                sender="小王",
                text="我完了",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]
        to_reply, all_messages = build_messages(previous, unread)

        assert ReplyGenerator._group_visible_self_stance_continuity_contract(
            to_reply, all_messages
        ) is None

    def test_group_visible_self_stance_continuity_contract_rejects_media_and_multiple_speakers(self):
        previous = ChatMessage(
            sender="我",
            text="可怕",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        media = ChatMessage(
            sender="小王",
            text="[图片]",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            message_type="image",
        )
        first = ChatMessage(
            sender="小王",
            text="金山银山都跌完了",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        second = ChatMessage(
            sender="小李",
            text="确实",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert ReplyGenerator._group_visible_self_stance_continuity_contract(
            [media], [previous, media]
        ) is None
        assert ReplyGenerator._group_visible_self_stance_continuity_contract(
            [first, second], [previous, first, second]
        ) is None

    def test_group_visible_self_stance_continuity_changes_only_triggering_group_prompt(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_visible_self_stance_continuity=True,
        )
        previous = ChatMessage(
            sender="我",
            text="solo还能有位置么",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="好的，那我不找了，就1吧",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        )
        assert candidate._build_user_prompt(
            [incoming], [incoming], is_group=True
        ) == baseline._build_user_prompt([incoming], [incoming], is_group=True)

    def test_group_visible_self_stance_continuity_opens_only_latest_self_message(self):
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_visible_self_stance_continuity=True,
        )
        older_self = ChatMessage(
            sender="我",
            text="很早以前的旧梗",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        previous = ChatMessage(
            sender="我",
            text="得先还上",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小王",
                text="我跌破29了",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
            ChatMessage(
                sender="小王",
                text="我完了",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]

        root = ET.fromstring(candidate._build_user_prompt(
            unread, [older_self, previous, *unread], is_group=True
        ))
        contract = root.find("./group_visible_self_stance_continuity")
        consumed = root.find("./consumed_self_replies")
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert contract is not None
        assert [
            (message.attrib["role"], message.text)
            for message in contract.findall("./local_timeline/message")
        ] == [
            ("self", "我：得先还上"),
            ("other", "小王：我跌破29了"),
            ("other", "小王：我完了"),
        ]
        assert consumed is not None
        assert consumed.attrib["reuse"] == "forbidden"
        assert [message.text for message in consumed.findall("./message")] == [
            "我：很早以前的旧梗"
        ]
        contract_text = "".join(contract.itertext())
        assert "不得照抄本人原句" in contract_text
        assert "复用旧梗" in contract_text
        assert "不可见的事实" in contract_text
        assert "其他本人历史仍为禁止素材" in guard_text

    def test_group_current_event_self_role_continuity_accepts_multi_speaker_event(self):
        previous = ChatMessage(
            sender="我",
            text="我争取 但是8点下班",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        first = ChatMessage(
            sender="小王",
            text="那就九点开始吧",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小李",
                text="可以搞晚点",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
            ChatMessage(
                sender="小周",
                text="等你下班过来",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]

        contract = ReplyGenerator._group_current_event_self_role_continuity_contract(
            unread, [previous, first, *unread]
        )

        assert contract is not None
        assert contract["previous_self_message"] is previous
        assert contract["event_messages"] == [first, *unread]

    def test_group_current_event_self_role_continuity_skips_later_self_reaction(self):
        relevant = ChatMessage(
            sender="我",
            text="鲜奶冻都吃吧",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        first = ChatMessage(
            sender="小王",
            text="我吃鲜奶冻",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        reaction = ChatMessage(
            sender="我",
            text="[表情]",
            sender_type=SenderType.SELF,
            chat_name="测试群",
            message_type="emoji",
        )
        incoming = ChatMessage(
            sender="小李",
            text="秀秀吃不吃？",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        contract = ReplyGenerator._group_current_event_self_role_continuity_contract(
            [incoming], [relevant, first, reaction, incoming]
        )

        assert contract is not None
        assert contract["previous_self_message"] is relevant
        assert contract["event_messages"] == [first, reaction, incoming]

    @pytest.mark.parametrize(
        ("previous_text", "other_texts"),
        [
            ("难受", ["难受"]),
            ("这么厉害", ["太厉害了"]),
            ("18年的时候 来了", ["所以来了"]),
            ("昨天我去了一趟", ["昨天真冷"]),
            ("你在家不是有个T9可以帮你面么", ["不一样啊", "t9直接给我过", "你准备元旦前面？", "我还打算先帮你去美团探探呢"]),
            ("我也想买点", ["最近波动挺大", "@韭菜三号 能买点吗"]),
        ],
    )
    def test_group_current_event_self_role_continuity_rejects_weak_overlap(
        self, previous_text, other_texts
    ):
        previous = ChatMessage(
            sender="我",
            text=previous_text,
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = [
            ChatMessage(
                sender="小王",
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for text in other_texts
        ]

        assert ReplyGenerator._group_current_event_self_role_continuity_contract(
            [incoming[-1]], [previous, *incoming]
        ) is None

    def test_group_current_event_self_role_continuity_accepts_quoted_self_text(self):
        older = ChatMessage(
            sender="我",
            text="销售会要你身份证",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        quoted = ChatMessage(
            sender="我",
            text="我是把身份证给销售 销售去的[引用 小王：销售会帮忙吗]",
            sender_type=SenderType.SELF,
            chat_name="测试群",
            message_type="other",
        )
        incoming = ChatMessage(
            sender="小李",
            text="所以身份证是销售拿去办的",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        contract = ReplyGenerator._group_current_event_self_role_continuity_contract(
            [incoming], [older, quoted, incoming]
        )

        assert contract is not None
        assert contract["previous_self_message"] is quoted

    @pytest.mark.parametrize(
        "messages",
        [
            [
                ChatMessage(sender="我", text="外卖到了吗", sender_type=SenderType.SELF, chat_name="测试群"),
                ChatMessage(sender="小王", text="到了", sender_type=SenderType.OTHER, chat_name="测试群"),
                ChatMessage(sender="小李", text="人脸识别明天开会", sender_type=SenderType.OTHER, chat_name="测试群"),
                ChatMessage(sender="小周", text="算法要上线了", sender_type=SenderType.OTHER, chat_name="测试群"),
            ],
            [
                ChatMessage(sender="我", text="我刚下班", sender_type=SenderType.SELF, chat_name="测试群"),
                ChatMessage(sender="小王", text="快来", sender_type=SenderType.OTHER, chat_name="测试群"),
                ChatMessage(sender="小李", text="你到哪了？", sender_type=SenderType.OTHER, chat_name="测试群"),
            ],
        ],
    )
    def test_group_current_event_self_role_continuity_rejects_unsafe_tail(self, messages):
        assert ReplyGenerator._group_current_event_self_role_continuity_contract(
            [messages[-1]], messages
        ) is None

    def test_group_current_event_self_role_continuity_changes_only_triggering_group_prompt(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_current_event_self_role_continuity=True,
        )
        previous = ChatMessage(
            sender="我",
            text="推荐算法这块怎么做",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="大佬",
            text="推荐算法要先分召回和排序",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        )
        assert candidate._build_user_prompt(
            [incoming], [incoming], is_group=True
        ) == baseline._build_user_prompt([incoming], [incoming], is_group=True)

        root = ET.fromstring(candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=True
        ))
        contract = root.find("./group_current_event_self_role_continuity")
        consumed = root.find("./consumed_self_replies")
        assert contract is not None
        assert contract.findtext("./local_timeline/message") == "我：推荐算法这块怎么做"
        assert consumed is None
        contract_text = "".join(contract.itertext())
        assert "回复谁" in contract_text
        assert "接话动作" in contract_text
        assert "不可见的实时" in contract_text

    def test_group_explicit_self_stance_action_keeps_local_timeline_order(self):
        first_self = ChatMessage(
            sender="我",
            text="反正我感觉不搞个3年5年的没啥用",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        first_other = ChatMessage(
            sender="阿绿",
            text="医用面膜试过没有",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        second_self = ChatMessage(
            sender="我",
            text="没用过",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        third_self = ChatMessage(
            sender="我",
            text="我怎么感觉你在做广告",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="阿绿",
            text="肯定你买又不会打钱给我",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        messages = [first_self, first_other, second_self, third_self, incoming]

        contract = ReplyGenerator._group_explicit_self_stance_action_contract(
            [incoming], messages
        )

        assert contract is not None
        assert contract["local_timeline"] == messages
        assert contract["opened_self_messages"] == [
            first_self,
            second_self,
            third_self,
        ]

    def test_group_explicit_self_stance_action_opens_only_local_self_messages(self):
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_explicit_self_stance_action=True,
        )
        older_self = ChatMessage(
            sender="我",
            text="很早以前的旧梗",
            sender_type=SenderType.SELF,
            chat_name="测试群",
            reply_time=100,
        )
        first_self = ChatMessage(
            sender="我",
            text="感觉现在不想买",
            sender_type=SenderType.SELF,
            chat_name="测试群",
            reply_time=4000,
        )
        other = ChatMessage(
            sender="小王",
            text="电车后面补贴要取消了",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            create_time=4001,
        )
        second_self = ChatMessage(
            sender="我",
            text="主要感觉没有用车需求",
            sender_type=SenderType.SELF,
            chat_name="测试群",
            reply_time=4002,
        )
        incoming = ChatMessage(
            sender="小李",
            text="看来有购车计划啊",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            create_time=4003,
        )

        root = ET.fromstring(candidate._build_user_prompt(
            [incoming],
            [older_self, first_self, other, second_self, incoming],
            is_group=True,
            enable_timestamps=False,
        ))
        contract = root.find("./group_explicit_self_stance_action")
        consumed = root.find("./consumed_self_replies")

        assert contract is not None
        assert [
            (message.attrib["role"], message.text)
            for message in contract.findall("./local_timeline/message")
        ] == [
            ("self", "我：感觉现在不想买"),
            ("other", "小王：电车后面补贴要取消了"),
            ("self", "我：主要感觉没有用车需求"),
            ("other", "小李：看来有购车计划啊"),
        ]
        assert consumed is not None
        assert [message.text for message in consumed.findall("./message")] == [
            "我：很早以前的旧梗"
        ]
        contract_text = "".join(contract.itertext())
        guard_text = "".join(root.find("./reply_guard").itertext())
        assert "reply_target" in contract_text
        assert "interaction_action" in contract_text
        assert "self_stance" in contract_text
        assert "new_evidence" in contract_text
        assert "不强迫正面作答" in contract_text
        assert "不得编造" in contract_text
        assert "没有新证据" in contract_text
        assert "已开放的 role=self" in guard_text

    def test_group_explicit_self_stance_action_changes_only_eligible_group_prompt(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_explicit_self_stance_action=True,
        )
        previous = ChatMessage(
            sender="我",
            text="感觉太麻烦了",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="阿绿",
            text="用起来还挺方便的",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        directed_elsewhere = ChatMessage(
            sender="阿绿",
            text="@小李 你怎么看",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        )
        assert candidate._build_user_prompt(
            [incoming], [incoming], is_group=True
        ) == baseline._build_user_prompt([incoming], [incoming], is_group=True)
        assert candidate._build_user_prompt(
            [directed_elsewhere], [previous, directed_elsewhere], is_group=True
        ) == baseline._build_user_prompt(
            [directed_elsewhere], [previous, directed_elsewhere], is_group=True
        )

    def test_group_explicit_self_stance_action_adds_final_pending_judgment_priority(
        self, monkeypatch
    ):
        candidate = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_explicit_self_stance_action=True,
            enable_latest_request_first=True,
        )
        candidate.enable_self_refine = False
        monkeypatch.setattr(candidate, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(candidate, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            candidate,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["0.7也挺紧张"], messages),
        )
        monkeypatch.setattr(candidate, "_submit_to_judge", lambda *args, **kwargs: None)
        previous = ChatMessage(
            sender="我",
            text="太紧张了还是得买的",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        evidence = ChatMessage(
            sender="林涛",
            text="1比0.7",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="林涛",
            text="我靠",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        candidate.generate(
            [incoming], [previous, evidence, incoming], is_group=True
        )
        root = ET.fromstring(candidate.last_user_prompt)
        child_tags = [child.tag for child in root]
        priority = root.find("./group_pending_self_judgment_priority")

        assert priority is not None
        priority_text = "".join(priority.itertext())
        assert "尚未完成的具体问题或判断" in priority_text
        assert "惊叹、表情、重复" in priority_text
        assert "独立的新问题、请求、选择或动作" in priority_text
        assert "不得强迫正面作答" in priority_text
        assert child_tags.index("group_pending_self_judgment_priority") > child_tags.index(
            "latest_request_contract"
        )

    @pytest.mark.parametrize(
        ("previous_text", "unread_texts", "expected_action"),
        [
            (
                "没事 一起还了 主要利息太高",
                ["5还利息高？", "比银行便宜吧？"],
                "respond_to_stance_challenge",
            ),
            (
                "solo还能有位置么",
                ["好的，那我不找了，就1吧"],
                "accept_or_reject_group_choice",
            ),
            (
                "solo还能有位置么",
                ["好的，那我不找了，就1吧", "1我吃过的，蛮好吃的"],
                "accept_or_reject_group_choice",
            ),
        ],
    )
    def test_group_pending_self_action_detects_only_supported_actions(
        self, previous_text, unread_texts, expected_action
    ):
        previous = ChatMessage(
            sender="我",
            text=previous_text,
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小王",
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for text in unread_texts
        ]

        contract = ReplyGenerator._group_pending_self_action_contract(
            unread, [previous, *unread]
        )

        assert contract is not None
        assert contract["previous_self_message"] is previous
        assert contract["pending_action"] == expected_action

    @pytest.mark.parametrize(
        ("previous_text", "unread_texts"),
        [
            ("可怕", ["金山银山都跌完了"]),
            ("得先还上", ["我跌破29了", "我完了"]),
        ],
    )
    def test_group_pending_self_action_rejects_adjacent_but_unresolved_tails(
        self, previous_text, unread_texts
    ):
        previous = ChatMessage(
            sender="我",
            text=previous_text,
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小王",
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for text in unread_texts
        ]

        assert ReplyGenerator._group_pending_self_action_contract(
            unread, [previous, *unread]
        ) is None

    def test_group_pending_self_action_changes_only_triggering_group_prompt(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_pending_self_action=True,
        )
        previous = ChatMessage(
            sender="我",
            text="可怕",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="金山银山都跌完了",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        )
        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=True
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=True
        )

    def test_group_pending_self_action_opens_only_the_adjacent_self_message(self):
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_pending_self_action=True,
        )
        older_self = ChatMessage(
            sender="我",
            text="很早以前的旧梗",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        previous = ChatMessage(
            sender="我",
            text="没事 一起还了 主要利息太高",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = [
            ChatMessage(
                sender="小王",
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for text in ("5还利息高？", "比银行便宜吧？")
        ]

        root = ET.fromstring(candidate._build_user_prompt(
            unread, [older_self, previous, *unread], is_group=True
        ))
        contract = root.find("./group_pending_self_action")
        consumed = root.find("./consumed_self_replies")
        guard_text = "".join(root.find("./reply_guard").itertext())

        assert contract is not None
        assert contract.findtext("./pending_action") == "respond_to_stance_challenge"
        assert [
            (message.attrib["role"], message.text)
            for message in contract.findall("./local_timeline/message")
        ] == [
            ("self", "我：没事 一起还了 主要利息太高"),
            ("other", "小王：5还利息高？"),
            ("other", "小王：比银行便宜吧？"),
        ]
        assert consumed is not None
        assert [message.text for message in consumed.findall("./message")] == [
            "我：很早以前的旧梗"
        ]
        contract_text = "".join(contract.itertext())
        assert "不要只因最后一条" in contract_text
        assert "不强迫正面作答" in contract_text
        assert "不得从态度、偏好或一句抱怨推导" in contract_text
        assert "其他本人历史仍为禁止素材" in guard_text

    @pytest.mark.parametrize(
        ("messages", "unread_index", "expected_type"),
        [
            (
                [
                    ("小王", "事到如今最对不起的就是西西"),
                    ("小李", "事到如今最对不起的就是西西"),
                ],
                1,
                "exact_repeat",
            ),
            (
                [("小王", "只有我亏钱"), ("小李", "还有我")],
                1,
                "shared_stance",
            ),
            (
                [("小王", "你们定"), ("小李", "我都行")],
                1,
                "shared_deferral",
            ),
        ],
    )
    def test_group_pattern_participation_detects_supported_patterns(
        self, messages, unread_index, expected_type
    ):
        timeline = [
            ChatMessage(
                sender=sender,
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for sender, text in messages
        ]

        contract = ReplyGenerator._group_pattern_participation_contract(
            [timeline[unread_index]], timeline
        )

        assert contract is not None
        assert contract["pattern_type"] == expected_type
        assert contract["interaction_need"] == "participate_in_group_pattern_content"

    @pytest.mark.parametrize(
        "timeline",
        [
            [("小王", "事到如今最对不起的就是西西"), ("小王", "事到如今最对不起的就是西西")],
            [("小王", "好的"), ("小李", "好的")],
            [("小王", "[旺柴]"), ("小李", "[旺柴]")],
            [("小王", "[强][强]"), ("小李", "[强][强]")],
            [("小王", "我买了茅台"), ("小李", "我买了宁德")],
            [("小王", "明天几点集合"), ("小李", "几点集合")],
        ],
    )
    def test_group_pattern_participation_rejects_unsupported_patterns(self, timeline):
        messages = [
            ChatMessage(
                sender=sender,
                text=text,
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            )
            for sender, text in timeline
        ]

        assert ReplyGenerator._group_pattern_participation_contract(
            [messages[-1]], messages
        ) is None

    def test_group_pattern_participation_does_not_use_self_status_as_pattern_evidence(self):
        previous = ChatMessage(
            sender="我",
            text="我六点下班",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小王",
            text="还有我",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert ReplyGenerator._group_pattern_participation_contract(
            [incoming], [previous, incoming]
        ) is None

    def test_group_pattern_participation_does_not_skip_self_turn_for_plus_one(self):
        earlier = ChatMessage(
            sender="小王",
            text="那我就放心了",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        self_turn = ChatMessage(
            sender="我",
            text="迟到一会",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小李",
            text="+1",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert ReplyGenerator._group_pattern_participation_contract(
            [incoming], [earlier, self_turn, incoming]
        ) is None

    def test_group_pattern_participation_changes_only_triggering_group_prompt(self):
        baseline = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_pattern_participation=True,
        )
        previous = ChatMessage(
            sender="小王",
            text="只有我亏钱",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        incoming = ChatMessage(
            sender="小李",
            text="还有我",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        unrelated = ChatMessage(
            sender="小李",
            text="半导体今天又跌了",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        ) == baseline._build_user_prompt(
            [incoming], [previous, incoming], is_group=False
        )
        assert candidate._build_user_prompt(
            [unrelated], [previous, unrelated], is_group=True
        ) == baseline._build_user_prompt(
            [unrelated], [previous, unrelated], is_group=True
        )

        root = ET.fromstring(candidate._build_user_prompt(
            [incoming], [previous, incoming], is_group=True
        ))
        contract = root.find("./group_pattern_participation")
        assert contract is not None
        assert contract.findtext("./pattern_type") == "shared_stance"
        assert contract.findtext("./interaction_need") == "participate_in_group_pattern_content"
        assert "禁止输出“队形”“复读”“接龙”" in "".join(contract.itertext())
        assert "不强制复读" in "".join(contract.itertext())
        assert "不得改变本人原有立场" in "".join(contract.itertext())

    @pytest.mark.parametrize(
        ("text", "is_at_me", "expected_target", "expected_action"),
        [
            ("@我 现在方便吗", True, "self", "answer"),
            ("@[联系人] 明天去哪接你们", False, "self", "give_concrete_time_or_place"),
            ("@[联系人] 来王者", False, "self", "coordinate_participation"),
            ("@小王 你把地址发一下", False, "named_other", "react_only"),
            ("大家按这个格式接龙", False, "group", "perform_group_operation"),
            ("你们几点到", False, "group", "give_concrete_time_or_place"),
        ],
    )
    def test_group_target_action_contract_extracts_explicit_target_and_action(
        self, text, is_at_me, expected_target, expected_action
    ):
        message = ChatMessage(
            sender="小王",
            text=text,
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            is_at_me=is_at_me,
        )
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_target_action_contract=True,
        )

        root = ET.fromstring(gen._build_user_prompt([message], [message], is_group=True))
        contract = root.find("./group_target_action_contract")

        assert contract is not None
        assert contract.attrib["enforcement"] == "guidance"
        assert contract.findtext("./addressed_to") == expected_target
        assert contract.findtext("./interaction_need") == expected_action
        assert contract.findtext("./answer_authority") == (
            "named_person_only"
            if expected_target == "named_other"
            else "independent_support_required"
        )
        assert contract.find("./required_action") is None

    def test_group_target_action_contract_tracks_second_person_thread_to_self(self):
        messages = [
            ChatMessage(
                sender="我",
                text="我下去了",
                sender_type=SenderType.SELF,
                chat_name="测试群",
            ),
            ChatMessage(
                sender="肖健",
                text="你下去了吗",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_target_action_contract=True,
        )

        root = ET.fromstring(
            gen._build_user_prompt([messages[-1]], messages, is_group=True)
        )
        contract = root.find("./group_target_action_contract")

        assert contract is not None
        assert contract.findtext("./addressed_to") == "self"
        assert contract.findtext("./interaction_need") == "report_own_status"

        rules = "\n".join(node.text or "" for node in contract.findall("./rule"))
        assert "不要求正面作答" in rules
        assert "反问核实前提" in rules
        assert "两者都不是答案证据" in rules
        assert "目标消息中的提问、假设和要求本身不算依据" in rules
        assert "较早本人发言已经直接说明同一状态或计划时，可以自然沿用" in rules

    @pytest.mark.parametrize(
        "text",
        [
            "你明天不上班吗",
            "所以今天跌的话是继续买嘛",
            "你们都是一口价 不再另算加班工资嘛",
        ],
    )
    def test_group_target_action_contract_does_not_treat_question_as_answer_evidence(
        self, text
    ):
        message = ChatMessage(
            sender="小王",
            text=text,
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_target_action_contract=True,
        )

        root = ET.fromstring(gen._build_user_prompt([message], [message], is_group=True))
        contract = root.find("./group_target_action_contract")

        assert contract is not None
        assert contract.findtext("./answer_authority") == "independent_support_required"
        rules = "\n".join(node.text or "" for node in contract.findall("./rule"))
        assert "不得替本人、他人、群体或组织直接确认或否认" in rules

    def test_group_target_action_contract_allows_direct_prior_self_evidence(self):
        messages = [
            ChatMessage(
                sender="我",
                text="我明天请假，不上班",
                sender_type=SenderType.SELF,
                chat_name="测试群",
            ),
            ChatMessage(
                sender="小王",
                text="你明天不上班吗",
                sender_type=SenderType.OTHER,
                chat_name="测试群",
            ),
        ]
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_target_action_contract=True,
            enable_group_unified_history=True,
        )

        root = ET.fromstring(
            gen._build_user_prompt([messages[-1]], messages, is_group=True)
        )
        history = "".join(root.find("./history").itertext())
        rules = "\n".join(
            node.text or ""
            for node in root.findall("./group_target_action_contract/rule")
        )

        assert "我明天请假，不上班" in history
        assert "较早本人发言已经直接说明同一状态或计划时，可以自然沿用" in rules

    @pytest.mark.parametrize(
        ("enabled", "is_group", "has_contract", "expected_status"),
        [
            (False, True, True, "candidate_disabled"),
            (True, False, True, "private_chat"),
            (True, True, False, "contract_absent"),
        ],
    )
    def test_group_answer_authority_postcheck_only_runs_for_triggering_group_contract(
        self, enabled, is_group, has_contract, expected_status
    ):
        class Client:
            def chat(self, **kwargs):
                raise AssertionError("不应调用检查器")

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=enabled,
        )
        messages = [{
            "role": "user",
            "content": (
                "<group_target_action_contract></group_target_action_contract>"
                if has_contract else "<unread>普通消息</unread>"
            ),
        }]

        replies, returned_messages = gen._check_group_answer_authority_reply(
            ["原回复"], messages, float("inf"), is_group=is_group
        )

        assert replies == ["原回复"]
        assert returned_messages == messages
        assert gen.last_group_answer_authority_review["attempted"] is False
        assert gen.last_group_answer_authority_review["status"] == expected_status

    def test_group_answer_authority_postcheck_keeps_passed_reply_byte_for_byte(self):
        captured = {}
        original = ["那不然呢"]

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "pass",
                    "reason": "这是顺势反问，不是无依据确认",
                    "replies": ["模型返回的这句不得替换原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        messages = [{
            "role": "user",
            "content": "<group_target_action_contract></group_target_action_contract>",
        }]

        replies, _ = gen._check_group_answer_authority_reply(
            original, messages, float("inf"), is_group=True
        )

        assert replies == original
        assert gen.last_group_answer_authority_review["status"] == "passed"
        assert "目标消息中的问题、假设、要求或建议不是答案证据" in captured["prompt"]
        assert "泛化态度" in captured["prompt"]
        assert "反问、澄清、接梗" in captured["prompt"]

    @pytest.mark.parametrize(
        ("original", "rewritten"),
        [
            ("要啊 备着不亏", "你是问我会不会备着？"),
            ("能啊，这还用问", "你说的能是指哪件事"),
            ("拿得住就入", "你想问现在这个点位？"),
        ],
    )
    def test_group_answer_authority_postcheck_can_remove_unsupported_conclusion(
        self, original, rewritten
    ):
        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "decision": "rewrite",
                    "reason": "目标问题本身不能证明这个具体结论",
                    "replies": [rewritten],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        messages = [{
            "role": "user",
            "content": "<group_target_action_contract></group_target_action_contract>",
        }]

        replies, _ = gen._check_group_answer_authority_reply(
            [original], messages, float("inf"), is_group=True
        )

        assert replies == [rewritten]
        assert gen.last_group_answer_authority_review["status"] == "rewritten"

    @pytest.mark.parametrize("raw", ["not-json", '{"decision":"rewrite","replies":[]}'])
    def test_group_answer_authority_postcheck_preserves_reply_on_invalid_result(self, raw):
        class Client:
            def chat(self, **kwargs):
                return raw

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        messages = [{
            "role": "user",
            "content": "<group_target_action_contract></group_target_action_contract>",
        }]

        replies, _ = gen._check_group_answer_authority_reply(
            ["原回复"], messages, float("inf"), is_group=True
        )

        assert replies == ["原回复"]
        assert gen.last_group_answer_authority_review["status"] == "invalid_response"

    def test_group_answer_authority_postcheck_preserves_reply_without_time_budget(self):
        class Client:
            def chat(self, **kwargs):
                raise AssertionError("不应调用检查器")

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        messages = [{
            "role": "user",
            "content": "<group_target_action_contract></group_target_action_contract>",
        }]

        replies, _ = gen._check_group_answer_authority_reply(
            ["原回复"], messages, 0.0, is_group=True
        )

        assert replies == ["原回复"]
        assert gen.last_group_answer_authority_review["status"] == "insufficient_time_budget"

    def test_current_stance_continuity_keeps_private_stance_evidence(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        history = [
            ChatMessage(
                sender="我",
                text="这个今天得确认下来",
                sender_type=SenderType.SELF,
                chat_name="小王",
            ),
            ChatMessage(
                sender="小王",
                text="现在确认了吗",
                sender_type=SenderType.OTHER,
                chat_name="小王",
            ),
        ]

        root = ET.fromstring(
            gen._build_user_prompt([history[-1]], history, is_group=False)
        )

        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.attrib["reuse"] == "context_only"
        assert root.find("./current_stance_evidence") is not None
        assert root.find("./group_speaker_target_attribution") is None

    def test_current_stance_postcheck_rewrites_reversed_group_order(self):
        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "decision": "rewrite",
                    "reason": "把本人先到写成了本人要赶到",
                    "replies": ["你们到了先点，我6点得去晚修"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": "我6点钟晚修 || 我晚上还要上班"},
                "replies": ["我尽量赶"],
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            ["我尽量赶"], messages, float("inf"), is_group=True
        )

        assert replies == ["你们到了先点，我6点得去晚修"]
        assert gen.last_current_stance_grounding["status"] == "rewritten"
        assert gen.last_current_stance_grounding["generated_current_stance"] == "我6点钟晚修 || 我晚上还要上班"
        assert gen.last_current_stance_grounding["current_stance"] == "none"

    def test_current_stance_postcheck_uses_other_participant_to_update_group_order(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "rewrite",
                    "current_assignment": "我们要一个半小时到那边 || 看来是你先到了 || 没事 你到了先点",
                    "current_attitude": "卧槽 快点啊 我晚上还要上班",
                    "reason": "其他参与者后续已经明确本人先到并由本人先点",
                    "replies": ["那我到了先点，你们快点"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {
                    "current_stance": "你们可以到了先点 || 卧槽 快点啊 我晚上还要上班"
                },
                "replies": ["我尽量快点"],
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            ["我尽量快点"], messages, float("inf"), is_group=True
        )

        assert replies == ["那我到了先点，你们快点"]
        assert "current_assignment" in captured["prompt"]
        assert "证据可以来自本人或其他参与者" in captured["prompt"]
        assert "current_attitude" in captured["prompt"]
        assert "只能以本人原话为证据" in captured["prompt"]
        assert "必须同时保留另一项仍成立的状态" in captured["prompt"]

    def test_current_stance_postcheck_rewrites_attitude_reversal(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "rewrite",
                    "current_assignment": "看来是你先到了 || 没事 你到了先点",
                    "current_attitude": "卧槽 快点啊 我晚上还要上班",
                    "reason": "本人刚在催对方快点，当前回复却让对方慢慢来",
                    "replies": ["那我先点，你们快点"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": "卧槽 快点啊 我晚上还要上班"},
                "replies": ["行，我先点，你们慢慢来"],
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            ["行，我先点，你们慢慢来"], messages, float("inf"), is_group=True
        )

        assert replies == ["那我先点，你们快点"]
        assert "本人刚在催对方快点时，不得回复让对方慢慢来" in captured["prompt"]
        assert gen.last_current_stance_grounding["current_assignment"] == "看来是你先到了 || 没事 你到了先点"
        assert gen.last_current_stance_grounding["current_attitude"] == "卧槽 快点啊 我晚上还要上班"

    def test_current_stance_postcheck_keeps_passed_reply_byte_for_byte(self):
        original = ["快点啊 你们到了先点"]

        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "decision": "pass",
                    "reason": "",
                    "replies": ["这段返回值不应替换原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": "我6点钟晚修"},
                "replies": original,
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            original, messages, float("inf"), is_group=True
        )

        assert replies == original
        assert gen.last_current_stance_grounding["status"] == "passed"

    @pytest.mark.parametrize(
        ("enabled", "is_group", "stance", "expected_status"),
        [
            (False, True, "我6点钟晚修", "candidate_disabled"),
            (True, False, "我6点钟晚修", "private_chat"),
        ],
    )
    def test_current_stance_postcheck_does_not_call_outside_exact_condition(
        self, enabled, is_group, stance, expected_status
    ):
        class Client:
            def chat(self, **kwargs):
                raise AssertionError("不应调用检查器")

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=enabled,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": stance},
                "replies": ["原回复"],
            }, ensure_ascii=False),
        }]

        replies, returned_messages = gen._check_current_stance_reply(
            ["原回复"], messages, float("inf"), is_group=is_group
        )

        assert replies == ["原回复"]
        assert returned_messages == messages
        assert gen.last_current_stance_grounding["attempted"] is False
        assert gen.last_current_stance_grounding["status"] == expected_status

    def test_current_stance_postcheck_independently_checks_when_generation_says_none(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "rewrite",
                    "current_stance": "卧槽 快点啊 我晚上还要上班",
                    "reason": "较新的本人消息已经改成催对方快点，当前回复却重拾旧安排",
                    "replies": ["你们快点，我晚上还要上班"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": "none"},
                "replies": ["我6点晚修，你们到了先点，我晚点过去"],
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            ["我6点晚修，你们到了先点，我晚点过去"],
            messages,
            float("inf"),
            is_group=True,
        )

        assert replies == ["你们快点，我晚上还要上班"]
        assert gen.last_current_stance_grounding["status"] == "rewritten"
        assert gen.last_current_stance_grounding["generated_current_stance"] == "none"
        assert gen.last_current_stance_grounding["current_assignment"] == "none"
        assert gen.last_current_stance_grounding["current_attitude"] == "none"
        assert gen.last_current_stance_grounding["raw_response"]
        assert "不得相信第一次生成填写的 decision.current_stance" in captured["prompt"]

    def test_current_stance_postcheck_passes_unchanged_when_exact_condition_is_absent(self):
        original = ["原回复 一个字也不能变"]

        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "decision": " pass ",
                    "current_stance": "none",
                    "reason": "",
                    "replies": ["模型返回的这句不得替换原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": "none"},
                "replies": original,
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            original, messages, float("inf"), is_group=True
        )

        assert replies == original
        assert gen.last_current_stance_grounding["status"] == "passed"

    def test_group_self_context_postcheck_independently_checks_answered_self_quote(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "rewrite",
                    "answered_self_quote": "我怎么感觉你在做广告",
                    "reason": "当前回复把对方被怀疑做广告的身份认领成了本人",
                    "replies": ["没打钱还这么卖力"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_self_context_freshness_boundary=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"directly_answered_self_quote": "none"},
                "replies": ["被你发现了，我这就把广告费打过去"],
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            ["被你发现了，我这就把广告费打过去"],
            messages,
            float("inf"),
            is_group=True,
        )

        assert replies == ["没打钱还这么卖力"]
        assert gen.last_current_stance_grounding["status"] == "rewritten"
        assert gen.last_current_stance_grounding["answered_self_quote"] == "我怎么感觉你在做广告"
        assert "不得相信第一次生成填写的 decision.directly_answered_self_quote" in captured["prompt"]
        assert "阶段1只判断接话关系" in captured["prompt"]
        assert "必须完全忽略 current_replies" in captured["prompt"]
        assert "不得根据候选回复是否冲突来反推接话关系" in captured["prompt"]
        assert "只要上述接话关系成立" in captured["prompt"]
        assert "阶段2才读取 current_replies" in captured["prompt"]
        assert "只有忽略该本人原话会改变本人在当前回复中" not in captured["prompt"]
        assert "不要求 unread 复述本人原话" in captured["prompt"]
        assert "辩解或反驳" in captured["prompt"]
        assert "同一发言人的紧邻一段消息" in captured["prompt"]
        assert "两处证据缺一不可" in captured["prompt"]
        assert "短感叹、图片、表情" in captured["prompt"]
        assert "不得把对方的身份或立场认领为本人" in captured["prompt"]

    def test_group_self_context_postcheck_keeps_reply_when_no_self_quote_is_answered(self):
        original = ["又咋了"]

        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "decision": "pass",
                    "answered_self_quote": "none",
                    "reason": "",
                    "replies": ["不应替换原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_self_context_freshness_boundary=True,
        )

        replies, _ = gen._check_current_stance_reply(
            original,
            [{"role": "assistant", "content": '{"replies":["又咋了"]}'}],
            float("inf"),
            is_group=True,
        )

        assert replies == original
        assert gen.last_current_stance_grounding["status"] == "passed"
        assert gen.last_current_stance_grounding["answered_self_quote"] == "none"

    def test_group_self_context_postcheck_uses_only_focused_evidence(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["messages"] = [dict(message) for message in kwargs["messages"]]
                return json.dumps({
                    "decision": "pass",
                    "answered_self_quote": "none",
                    "reason": "",
                    "replies": ["原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_self_context_freshness_boundary=True,
        )
        history = "\n".join(
            f'<message index="{index}" role="other" type="text">消息{index}</message>'
            for index in range(1, 15)
        )
        prompt = (
            "<request>\n"
            "<context><group_memory>不应发送给复核模型</group_memory></context>\n"
            f"<history>\n{history}\n</history>\n"
            '<unread><message index="1" role="other" type="text">卧槽</message></unread>\n'
            '<consumed_self_replies reuse="forbidden">'
            '<message index="1" role="self" type="text">这个点还在仓库里</message>'
            '</consumed_self_replies>\n'
            '<current_stance_evidence source="recent_self_text" limit="8">'
            '<message index="1" role="self" type="text">我怎么感觉你在做广告</message>'
            '</current_stance_evidence>\n'
            "<style_examples>不应发送给复核模型</style_examples>\n"
            "</request>"
        )

        replies, _ = gen._check_current_stance_reply(
            ["原回复"],
            [
                {"role": "system", "content": "完整大段 system prompt"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": '{"replies":["原回复"]}'},
            ],
            float("inf"),
            is_group=True,
        )

        assert replies == ["原回复"]
        assert len(captured["messages"]) == 2
        review_input = captured["messages"][-1]["content"]
        assert "卧槽" in review_input
        assert "这个点还在仓库里" in review_input
        assert "我怎么感觉你在做广告" in review_input
        assert '<reply_chain order="old_to_new">' in review_input
        assert "消息1<" not in review_input
        assert "消息14<" not in review_input
        assert "完整大段 system prompt" not in review_input
        assert "不应发送给复核模型" not in review_input

    def test_group_self_context_postcheck_restores_self_and_other_message_order(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "pass",
                    "answered_self_quote": "none",
                    "reason": "",
                    "replies": ["原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_self_context_freshness_boundary=True,
        )
        prompt = (
            "<request>\n"
            '<history><message index="2" role="other" type="text">蛋蛋毛：拼多多时间，中午12点</message>'
            '<message index="3" role="other" type="text">富比：你说你又不差钱</message>'
            '<message index="4" role="other" type="text">富比：跑去多多折腾啥</message></history>\n'
            '<unread><message index="1" role="other" type="text">富比：不对，我错了</message></unread>\n'
            '<consumed_self_replies reuse="forbidden">'
            '<message index="1" role="self" type="text">我：主要是穷</message>'
            '</consumed_self_replies>\n'
            '<current_stance_evidence source="recent_self_text" limit="8">'
            '<message index="1" role="self" type="text">我：主要是穷</message>'
            '</current_stance_evidence>\n'
            "</request>"
        )

        gen._check_current_stance_reply(
            ["原回复"],
            [{"role": "user", "content": prompt}],
            float("inf"),
            is_group=True,
        )

        review_input = captured["prompt"]
        assert '<reply_chain order="old_to_new">' in review_input
        assert '<message_before_same_speaker_segment>' in review_input
        assert '<same_speaker_previous_segment speaker="富比">' in review_input
        assert review_input.index("蛋蛋毛：拼多多时间，中午12点") < review_input.index("富比：你说你又不差钱")
        assert review_input.index("富比：你说你又不差钱") < review_input.index("富比：跑去多多折腾啥")
        assert review_input.index("富比：跑去多多折腾啥") < review_input.index("富比：不对，我错了")
        assert "<consumed_self_replies" not in review_input
        assert gen.last_current_stance_grounding["answered_self_quote"] == "none"

    def test_group_self_context_postcheck_keeps_self_turn_before_direct_response(self):
        captured = {}

        class Client:
            def chat(self, **kwargs):
                captured["prompt"] = kwargs["messages"][-1]["content"]
                return json.dumps({
                    "decision": "pass",
                    "answered_self_quote": "我怎么感觉你在做广告",
                    "reason": "",
                    "replies": ["原回复"],
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_self_context_freshness_boundary=True,
        )
        prompt = (
            "<request>\n"
            '<history><message index="2" role="other" type="other">阿绿：[旺柴]</message>'
            '<message index="3" role="other" type="text">阿绿：我有用过啊啊</message></history>\n'
            '<unread><message index="1" role="other" type="text">阿绿：肯定你买又不会打钱给我</message></unread>\n'
            '<consumed_self_replies reuse="forbidden">'
            '<message index="1" role="self" type="text">我：我怎么感觉你在做广告</message>'
            '</consumed_self_replies>\n'
            '<current_stance_evidence source="recent_self_text" limit="8">'
            '<message index="1" role="self" type="text">我：我怎么感觉你在做广告</message>'
            '</current_stance_evidence>\n'
            "</request>"
        )

        gen._check_current_stance_reply(
            ["原回复"],
            [{"role": "user", "content": prompt}],
            float("inf"),
            is_group=True,
        )

        review_input = captured["prompt"]
        assert review_input.index("我：我怎么感觉你在做广告") < review_input.index("阿绿：[旺柴]")
        assert review_input.index("阿绿：[旺柴]") < review_input.index("阿绿：我有用过啊啊")
        assert review_input.index("阿绿：我有用过啊啊") < review_input.index("阿绿：肯定你买又不会打钱给我")
        assert gen.last_current_stance_grounding["answered_self_quote"] == "我怎么感觉你在做广告"

    def test_current_stance_postcheck_skips_without_time_budget(self):
        class Client:
            def chat(self, **kwargs):
                raise AssertionError("不应调用检查器")

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"current_stance": "我6点钟晚修"},
                "replies": ["原回复"],
            }, ensure_ascii=False),
        }]

        replies, _ = gen._check_current_stance_reply(
            ["原回复"], messages, 0.0, is_group=True
        )

        assert replies == ["原回复"]
        assert gen.last_current_stance_grounding["status"] == "insufficient_time_budget"

    def test_latest_request_candidate_overrides_evidence_schema_without_changing_baseline(self):
        schema = (
            '<output_schema><format>{"decision":{"evidence":"not_applicable"}}</format>'
            '<field name="evidence">只能是 supported、unsupported、not_applicable。</field>'
            '</output_schema>'
        )

        candidate_schema = ReplyGenerator._with_latest_request_evidence_schema(schema)

        assert '"evidence":"missing"' in candidate_schema
        assert "来源标签和原句" in candidate_schema
        assert "不得填写 supported、unsupported 或 not_applicable" in candidate_schema
        assert '"evidence":"not_applicable"' in schema

    def test_latest_request_candidate_overrides_json_escaped_evidence_schema(self):
        schema = (
            r'"<output_schema><format>{\"decision\":{\"evidence\":\"not_applicable\"}}</format>'
            r'<field name=\"evidence\">只能是 supported、unsupported、not_applicable。</field>'
            r'</output_schema>"'
        )

        candidate_schema = ReplyGenerator._with_latest_request_evidence_schema(schema)

        assert r'\"evidence\":\"missing\"' in candidate_schema
        assert r'<field name=\"evidence\">只能填写 missing' in candidate_schema
        assert "不得填写 supported、unsupported 或 not_applicable" in candidate_schema

    def test_latest_request_candidate_finds_inline_colloquial_quantity_question(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        messages = [
            ChatMessage(
                text="怎么办",
                sender="Bob",
                sender_type=SenderType.OTHER,
                chat_name="Bob",
            ),
            ChatMessage(
                text="你还有多少",
                sender="Bob",
                sender_type=SenderType.OTHER,
                chat_name="Bob",
            ),
        ]

        root = ET.fromstring(gen._build_user_prompt(messages, messages, is_group=False))
        target = root.find("./response_target_candidate")

        assert target is not None
        assert target.attrib["index"] == "2"
        assert "你还有多少" in (target.text or "")

    @pytest.mark.parametrize(
        "statement",
        [
            "我没什么想法",
            "不管多少都行",
            "这多少有点离谱",
        ],
    )
    def test_latest_request_candidate_does_not_treat_quantity_statements_as_questions(
        self, statement
    ):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        unread = ChatMessage(
            text=statement,
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="Bob",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=False))

        assert root.find("./response_target_candidate") is None

    def test_candidate_generation_checks_that_latest_request_was_answered(self, monkeypatch):
        gen = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["这个我不确定"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        checked = []
        monkeypatch.setattr(
            gen,
            "_check_latest_request_reply",
            lambda prompt, replies, messages, deadline: (
                checked.append((prompt, list(replies))) or (replies, messages)
            ),
        )
        unread = ChatMessage(
            text="雅克那边有多少人啊？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        replies = gen.generate([unread], [unread], is_group=True)
        root = ET.fromstring(gen.last_user_prompt)

        assert replies == ["这个我不确定"]
        assert root[-1].tag == "latest_request_contract"
        contract_text = "".join(root[-1].itertext())
        assert "最终回复必须真正处理它" in contract_text
        assert "回答、拒绝、澄清或问一个必要问题后立即结束" in contract_text
        assert "硬伤只指回复与当前上下文或可信 Wiki" in contract_text
        assert len(checked) == 1
        assert checked[0][1] == ["这个我不确定"]
        assert gen.last_latest_request_grounding["status"] == "one_pass_contract"

    @pytest.mark.parametrize(
        "text",
        [
            "啥叫减按？是1%还是0.5%",
            "到底按100元或者200元算？",
        ],
    )
    def test_unresolved_fact_choice_guard_detects_numeric_fact_options(self, text):
        assert ReplyGenerator._has_numeric_fact_choice(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "是不是按1%算？",
            "这个比例是多少？",
            "你想喝咖啡还是茶？",
            "买1个还是2个？",
        ],
    )
    def test_unresolved_fact_choice_guard_ignores_non_choices_and_preferences(self, text):
        assert ReplyGenerator._has_numeric_fact_choice(text) is False

    @pytest.mark.parametrize("enabled", [False, True])
    def test_unresolved_fact_choice_guard_is_injected_only_when_enabled_and_matched(
        self, monkeypatch, enabled
    ):
        gen = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_unresolved_fact_choice_guard=enabled,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["我不确定"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        unread = ChatMessage(
            text="啥叫减按？是1%还是0.5%",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        gen.generate([unread], [unread], is_group=True)
        root = ET.fromstring(gen.last_user_prompt)

        assert (root.find("./unresolved_fact_choice_guard") is not None) is enabled

    def test_unresolved_fact_choice_guard_does_not_change_unmatched_prompt(self, monkeypatch):
        baseline = ReplyGenerator(llm_client=object(), tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_unresolved_fact_choice_guard=True,
        )
        unread = ChatMessage(
            text="这个比例是多少？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        for gen in (baseline, candidate):
            gen.enable_self_refine = False
            monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
            monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
            monkeypatch.setattr(
                gen,
                "_react_generate",
                lambda messages, tools, deadline, chat_name: (["我不确定"], messages),
            )
            monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
            gen.generate([unread], [unread], is_group=True)

        assert candidate.last_user_prompt == baseline.last_user_prompt

    @pytest.mark.parametrize(
        "text",
        [
            "啥叫减按？是1%还是0.5%",
            "item2item里freq(a,b)用1还是2？",
            "填2还是3？",
        ],
    )
    def test_unresolved_fact_choice_guard_is_the_only_prompt_delta_for_human_cases(
        self, monkeypatch, text
    ):
        baseline = ReplyGenerator(llm_client=object(), tool_registry=_TEST_TOOL_REGISTRY)
        candidate = ReplyGenerator(
            llm_client=object(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_unresolved_fact_choice_guard=True,
        )
        unread = ChatMessage(
            text=text,
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        for gen in (baseline, candidate):
            gen.enable_self_refine = False
            monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
            monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
            monkeypatch.setattr(
                gen,
                "_react_generate",
                lambda messages, tools, deadline, chat_name: (["我不确定"], messages),
            )
            monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
            gen.generate([unread], [unread], is_group=True)

        baseline_root = ET.fromstring(baseline.last_user_prompt)
        candidate_root = ET.fromstring(candidate.last_user_prompt)
        guard = candidate_root.find("./unresolved_fact_choice_guard")
        assert guard is not None
        candidate_root.remove(guard)

        for root in (baseline_root, candidate_root):
            for node in root.iter():
                if node.text is not None and not node.text.strip():
                    node.text = None
                if node.tail is not None and not node.tail.strip():
                    node.tail = None
        assert ET.tostring(candidate_root) == ET.tostring(baseline_root)

    @pytest.mark.parametrize(
        "reply",
        ["我都没敢看了", "不清楚，没细问"],
    )
    def test_latest_request_evidence_guard_constrains_missing_reply(self, reply):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"evidence": "missing"},
                "replies": [reply],
            }, ensure_ascii=False),
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>你还有多少</response_target_candidate>",
            [reply],
            messages,
        )

        assert guarded == ["我也说不准"]
        assert gen.last_latest_request_grounding["status"] == "missing_reply_constrained"

    def test_latest_request_target_uses_only_labelled_original_text(self):
        target = ReplyGenerator._latest_request_target_text(
            "<response_target_candidate>目标原文：你还有多少。"
            "这只是待核对的候选。先检查它之后的 unread。"
            "</response_target_candidate>"
        )

        assert target == "你还有多少"

    def test_latest_request_target_does_not_guess_from_last_unread_message(self):
        target = ReplyGenerator._latest_request_target_text(
            '<request><unread><message index="1" role="other" type="text">'
            '小周：前一句</message><message index="2" role="other" type="text">'
            '小周：你叔叔是不是在华为很牛逼</message></unread></request>'
        )

        assert target == ""

    @pytest.mark.parametrize(
        ("target", "reply"),
        [
        ("我猜到了", "确实"),
        ("还有什么时候滚床单", "不告诉你"),
        ("还请你吃饭？", "请不起，你标准太高"),
        ("或者组合在一起效果不好", "得拆开看每个策略的贡献"),
        ("肯定你买又不会打钱给我", "广告费都省了"),
    ],
    )
    def test_latest_request_evidence_guard_keeps_normal_chat(self, target, reply):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        prompt = (
            '<request><unread><message index="1" role="other" type="text">'
            f'小周：{target}</message></unread></request>'
        )
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"evidence": "missing"},
                "replies": [reply],
            }, ensure_ascii=False),
        }]

        guarded = gen._apply_latest_request_evidence_guard(prompt, [reply], messages)

        assert guarded == [reply]
        assert gen.last_latest_request_grounding["status"] == "missing_not_grounded_request"

    def test_latest_request_evidence_guard_does_not_constrain_unlabelled_unread(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        prompt = (
            '<request><unread><message index="1" role="other" type="text">'
            '小周：你叔叔是不是在华为很牛逼</message></unread></request>'
        )
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["是啊"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(prompt, ["是啊"], messages)

        assert guarded == ["是啊"]
        assert gen.last_latest_request_grounding["status"] == "missing_not_grounded_request"

    def test_latest_request_evidence_guard_does_not_rewrite_unlabelled_invitation(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        prompt = (
            '<request><unread><message index="1" role="other" type="text">'
            '小周：你要跟我一起吃饭么</message></unread></request>'
        )
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["可以啊"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(prompt, ["可以啊"], messages)

        assert guarded == ["可以啊"]

    def test_latest_request_evidence_guard_keeps_necessary_question(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["晚上几点"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>晚上一起吃饭</response_target_candidate>",
            ["晚上几点"],
            messages,
        )

        assert guarded == ["晚上几点"]
        assert gen.last_latest_request_grounding["status"] == "missing_kept_necessary_question"

    def test_latest_request_evidence_guard_does_not_mistake_future_action_for_question(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["好，我回头拍几张发你"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>目标原文：户型图和照片发一下。</response_target_candidate>",
            ["好，我回头拍几张发你"],
            messages,
        )

        assert guarded == ["这会儿发不了"]

    def test_latest_request_evidence_guard_keeps_clean_uncertainty(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["这个我还真不确定"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>现在有 online learning 吗</response_target_candidate>",
            ["这个我还真不确定"],
            messages,
        )

        assert guarded == ["这个我还真不确定"]
        assert gen.last_latest_request_grounding["status"] == "missing_kept_uncertainty"

    def test_latest_request_evidence_guard_removes_invented_reason_from_uncertainty(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["这个我真不清楚，会上没细说"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>目标原文：雅克那边有多少人啊。</response_target_candidate>",
            ["这个我真不清楚，会上没细说"],
            messages,
        )

        assert guarded == ["这我真不清楚"]

    def test_latest_request_evidence_guard_removes_future_check_from_uncertainty(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["这个我得问下组里，不确定"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>现在有 online learning 吗</response_target_candidate>",
            ["这个我得问下组里，不确定"],
            messages,
        )

        assert guarded == ["这我真不清楚"]

    def test_latest_request_evidence_guard_declines_unsupported_action_request(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["好呀，回头发你"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>户型图发一下</response_target_candidate>",
            ["好呀，回头发你"],
            messages,
        )

        assert guarded == ["这会儿发不了"]

    def test_latest_request_evidence_guard_rejects_request_as_its_own_evidence(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"evidence": "unread: 发下户型图和照片"},
                "replies": ["好呀，回头拍下发你"],
            }, ensure_ascii=False),
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>发下户型图和照片</response_target_candidate>",
            ["好呀，回头拍下发你"],
            messages,
        )

        assert guarded == ["这会儿发不了"]

    def test_latest_request_evidence_guard_asks_for_missing_invitation_detail(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["可以啊"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>目标原文：晚上一起吃饭？。"
            "这只是待核对的候选。先检查说话人。"
            "</response_target_candidate>",
            ["可以啊"],
            messages,
        )

        assert guarded == ["在哪边"]

    def test_latest_request_evidence_guard_does_not_misread_helper_text_as_action(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": '{"decision":{"evidence":"missing"},"replies":["没细看"]}',
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>目标原文：你还有多少。"
            "这只是待核对的候选。先检查说话人。"
            "</response_target_candidate>",
            ["没细看"],
            messages,
        )

        assert guarded == ["我也说不准"]

    def test_latest_request_evidence_guard_keeps_independently_supported_reply(self):
        gen = ReplyGenerator(
            llm_client=None,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
        )
        gen.last_latest_request_grounding = {}
        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "decision": {"evidence": "history: 我晚上七点有空"},
                "replies": ["七点可以"],
            }, ensure_ascii=False),
        }]

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>晚上几点吃饭</response_target_candidate>",
            ["七点可以"],
            messages,
        )

        assert guarded == ["七点可以"]
        assert gen.last_latest_request_grounding["status"] == "evidence_guard_passed"

    def test_latest_request_evidence_guard_does_not_change_baseline(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)

        guarded = gen._apply_latest_request_evidence_guard(
            "<response_target_candidate>你还有多少</response_target_candidate>",
            ["没多少了"],
            [{"role": "assistant", "content": '{"decision":{"evidence":"missing"}}'}],
        )

        assert guarded == ["没多少了"]

    def test_group_feature_plan_does_not_run_for_private_chat(self, monkeypatch):
        class AuxiliaryMustNotRun:
            def chat(self, **_kwargs):
                raise AssertionError("private generation must not call group auxiliary analysis")

        gen = ReplyGenerator(
            llm_client=AuxiliaryMustNotRun(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_feature_plan_before_generation=True,
            enable_self_refine=False,
        )
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["收到"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        message = ChatMessage(
            text="今晚去吗？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="Bob",
        )

        assert gen.generate([message], [message], is_group=False) == ["收到"]
        assert "<group_generation_plan" not in gen.last_user_prompt
        assert gen.last_generation_trace == []

    def test_group_feature_plan_injects_only_validated_contract(self, monkeypatch):
        class AuxiliaryClient:
            def __init__(self):
                self.responses = iter([
                    json.dumps({
                        "reply_target": "specific_other",
                        "interaction_action": "answer",
                        "event_stage": "active",
                        "self_context_relevance": False,
                        "pattern_participation": False,
                        "evidence": ["Bob：今晚去吗？"],
                        "prohibited_assumptions": ["不能假设自己会去"],
                    }, ensure_ascii=False),
                    json.dumps({
                        "reply_target": "specific_other",
                        "reply_action": "answer",
                        "allowed_facts": ["Bob：今晚去吗？"],
                        "forbidden_assumptions": ["不能承诺时间"],
                        "tone": "简短自然",
                        "plan": "直接回应是否参加，不补充未知安排。",
                    }, ensure_ascii=False),
                ])

            def chat(self, **_kwargs):
                return next(self.responses)

        class FinalGroupClient:
            def chat(self, **_kwargs):
                raise AssertionError("auxiliary analysis must not use the group generation client")

        auxiliary_client = AuxiliaryClient()
        final_group_client = FinalGroupClient()
        gen = ReplyGenerator(
            llm_client=object(),
            group_llm_client=final_group_client,
            group_feature_plan_llm_client=auxiliary_client,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_feature_plan_before_generation=True,
            enable_self_refine=False,
            generation_timeout_seconds=40,
        )
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (
                (["俺也去"], messages)
                if gen.llm_client is final_group_client
                else (_ for _ in ()).throw(AssertionError("final generation must retain the group client"))
            ),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        message = ChatMessage(
            text="今晚去吗？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert gen.generate([message], [message], is_group=True) == ["俺也去"]
        root = ET.fromstring(gen.last_user_prompt)
        plan = root.find("./group_generation_plan")
        assert plan is not None
        assert plan.findtext("./reply_target") == "specific_other"
        assert plan.findtext("./reply_action") == "answer"
        assert [item.text for item in plan.findall("./allowed_facts/item")] == ["Bob：今晚去吗？"]
        assert [(entry["type"], entry["status"]) for entry in gen.last_generation_trace] == [
            ("group_feature_extraction", "accepted"),
            ("group_generation_plan", "accepted"),
        ]

    def test_group_feature_plan_evidence_excludes_style_examples(self):
        prompt = (
            "<history><message>Bob：今晚去吗？</message></history>"
            "<unread><message>Bob：现在回复我</message></unread>"
            "<style_examples><example>示例里的旧事实</example></style_examples>"
        )
        source = ReplyGenerator._current_chat_evidence_source(prompt)

        accepted, reason = ReplyGenerator._validated_prompt_quotes(
            ["Bob：今晚去吗？"], source, field="evidence"
        )
        rejected, rejected_reason = ReplyGenerator._validated_prompt_quotes(
            ["示例里的旧事实"], source, field="evidence"
        )

        assert accepted == ["Bob：今晚去吗？"]
        assert reason == ""
        assert rejected is None
        assert rejected_reason == "evidence_not_in_source_prompt"

    def test_group_feature_plan_auxiliary_timeout_preserves_final_budget(self):
        class TimeoutRecordingClient:
            def __init__(self):
                self.timeouts = []
                self.responses = iter([
                    json.dumps({
                        "reply_target": "specific_other",
                        "interaction_action": "answer",
                        "event_stage": "active",
                        "self_context_relevance": False,
                        "pattern_participation": False,
                        "evidence": ["Bob：今晚去吗？"],
                        "prohibited_assumptions": [],
                    }, ensure_ascii=False),
                    json.dumps({
                        "reply_target": "specific_other",
                        "reply_action": "answer",
                        "allowed_facts": ["Bob：今晚去吗？"],
                        "forbidden_assumptions": [],
                        "tone": "简短自然",
                        "plan": "直接回答。",
                    }, ensure_ascii=False),
                ])

            def chat(self, **kwargs):
                self.timeouts.append(kwargs["timeout"])
                return next(self.responses)

        client = TimeoutRecordingClient()
        gen = ReplyGenerator(
            llm_client=client,
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_feature_plan_before_generation=True,
            enable_self_refine=False,
            generation_timeout_seconds=360,
        )
        user_prompt = "<request><history><message>Bob：今晚去吗？</message></history><unread /></request>"

        plan = gen._run_group_feature_plan_before_generation(
            user_prompt, overall_deadline=time.time() + 360.0
        )

        assert "<group_generation_plan" in plan
        assert client.timeouts == [90.0, 90.0]

    @pytest.mark.parametrize(
        "feature_response,plan_response,expected_type,expected_reason",
        [
            (
                {
                    "reply_target": "specific_other",
                    "interaction_action": "answer",
                    "event_stage": "active",
                    "self_context_relevance": False,
                    "pattern_participation": False,
                    "evidence": ["编造的上下文"],
                    "prohibited_assumptions": [],
                },
                None,
                "group_feature_extraction",
                "evidence_not_in_source_prompt",
            ),
            (
                {
                    "reply_target": "specific_other",
                    "interaction_action": "answer",
                    "event_stage": "active",
                    "self_context_relevance": False,
                    "pattern_participation": False,
                    "evidence": ["Bob：今晚去吗？"],
                    "prohibited_assumptions": [],
                },
                {
                    "reply_target": "group",
                    "reply_action": "answer",
                    "allowed_facts": ["Bob：今晚去吗？"],
                    "forbidden_assumptions": [],
                    "tone": "简短自然",
                    "plan": "答一句。",
                },
                "group_generation_plan",
                "plan_reply_target_mismatch",
            ),
        ],
    )
    def test_group_feature_plan_falls_back_on_untrusted_auxiliary_output(
        self, monkeypatch, feature_response, plan_response, expected_type, expected_reason
    ):
        class AuxiliaryClient:
            def __init__(self):
                responses = [json.dumps(feature_response, ensure_ascii=False)]
                if plan_response is not None:
                    responses.append(json.dumps(plan_response, ensure_ascii=False))
                self.responses = iter(responses)

            def chat(self, **_kwargs):
                return next(self.responses)

        gen = ReplyGenerator(
            llm_client=AuxiliaryClient(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_feature_plan_before_generation=True,
            enable_self_refine=False,
            generation_timeout_seconds=40,
        )
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["我再看看"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        message = ChatMessage(
            text="今晚去吗？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        assert gen.generate([message], [message], is_group=True) == ["我再看看"]
        assert "<group_generation_plan" not in gen.last_user_prompt
        failed = gen.last_generation_trace[-1]
        assert failed["type"] == expected_type
        assert failed["status"] == "fallback"
        assert failed["fallback_reason"] == expected_reason

    def test_returns_non_empty_string(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        reply = gen.generate([msg], [msg])
        assert isinstance(reply, list)
        assert len(reply) >= 0

    def test_handles_none_llm_client_gracefully(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        reply = gen.generate([msg], [msg])
        assert isinstance(reply, list)

    def test_fallback_when_llm_fails(self):
        class FailingLLM:
            def chat(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        gen = ReplyGenerator(llm_client=FailingLLM(), tool_registry=_TEST_TOOL_REGISTRY)
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        reply = gen.generate([msg], [msg])
        assert reply == []
        assert gen.last_generation_failed is True


class TestCombinedReplyReview:
    """多个后置审查同时激活时合并为一次 LLM 调用（分层复用结论）。"""

    def test_combined_review_merges_latest_request_and_authority_into_one_call(self):
        calls = []

        class Client:
            def chat(self, **kwargs):
                calls.append([dict(message) for message in kwargs["messages"]])
                return json.dumps({
                    "results": {
                        "latest_request": {"decision": "rewrite", "reason": "漏回了最新诉求",
                                           "replies": ["改后的回复"]},
                        "authority": {"decision": "pass", "reason": "", "replies": ["不应采用"]},
                    }
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        user_prompt = "<response_target_candidate>目标原文：你们啥时候去？</response_target_candidate>"
        messages = [{"role": "user",
                     "content": "<unread>你们啥时候去？</unread><group_target_action_contract></group_target_action_contract>"}]

        result = gen._run_combined_reply_review(
            user_prompt, ["原回复"], messages, float("inf"), float("inf"), is_group=True)

        assert result is not None
        replies, review_messages = result
        assert len(calls) == 1  # 两个审查只发起一次 LLM 调用
        prompt = calls[0][-1]["content"]
        assert "<combined_reply_review>" in prompt
        assert "<latest_request_reply_check>" in prompt
        assert "<group_answer_authority_check>" in prompt
        assert "results.latest_request" in prompt
        assert "results.authority" in prompt
        # 应用顺序：latest_request 的 rewrite 生效，authority 的 pass 不替换回复
        assert replies == ["改后的回复"]
        assert review_messages[-1]["role"] == "assistant"
        assert gen.last_latest_request_grounding["status"] == "rewritten"
        assert gen.last_latest_request_grounding["reply_after"] == ["改后的回复"]
        assert gen.last_group_answer_authority_review["status"] == "passed"
        # 未激活的 stance 仍由原方法写入观测 state（gate 直接返回，无额外调用）
        assert gen.last_current_stance_grounding["status"] == "candidate_disabled"
        assert len(calls) == 1

    def test_combined_review_applies_rewrites_in_fixed_order(self):
        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "results": {
                        "latest_request": {"decision": "rewrite", "reason": "", "replies": ["中间回复"]},
                        "stance": {"decision": "rewrite", "current_assignment": "none",
                                   "current_attitude": "none", "reason": "写反态度",
                                   "replies": ["最终回复"]},
                    }
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_current_stance_continuity=True,
        )
        user_prompt = "<response_target_candidate>目标原文：几点到？</response_target_candidate>"
        messages = [{"role": "user", "content": "<unread>几点到？</unread>"}]

        result = gen._run_combined_reply_review(
            user_prompt, ["原回复"], messages, float("inf"), float("inf"), is_group=True)

        assert result is not None
        replies, _ = result
        assert replies == ["最终回复"]
        assert gen.last_latest_request_grounding["status"] == "rewritten"
        assert gen.last_current_stance_grounding["status"] == "rewritten"
        assert gen.last_current_stance_grounding["current_assignment"] == "none"
        assert gen.last_group_answer_authority_review["status"] == "candidate_disabled"

    def test_combined_review_invalid_sub_result_keeps_original_reply(self):
        class Client:
            def chat(self, **kwargs):
                return json.dumps({
                    "results": {
                        "latest_request": {"decision": "rewrite", "reason": "", "replies": ["改后"]},
                        "authority": "pass",
                    }
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        user_prompt = "<response_target_candidate>目标原文：去吗？</response_target_candidate>"
        messages = [{"role": "user",
                     "content": "<unread>去吗？</unread><group_target_action_contract></group_target_action_contract>"}]

        result = gen._run_combined_reply_review(
            user_prompt, ["原回复"], messages, float("inf"), float("inf"), is_group=True)

        assert result is not None
        replies, _ = result
        assert replies == ["改后"]
        assert gen.last_latest_request_grounding["status"] == "rewritten"
        assert gen.last_group_answer_authority_review["status"] == "invalid_response"

    def test_freshness_boundary_stance_degrades_to_sequential_path(self):
        class Client:
            def chat(self, **kwargs):
                return json.dumps({"decision": "pass", "reason": "", "replies": []},
                                  ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_self_context_freshness_boundary=True,
            enable_group_answer_authority_rewrite=True,
        )
        user_prompt = "<response_target_candidate>目标原文：去吗？</response_target_candidate>"
        messages = [{"role": "user",
                     "content": "<unread>去吗？</unread><group_target_action_contract></group_target_action_contract>"}]

        result = gen._run_combined_reply_review(
            user_prompt, ["原回复"], messages, float("inf"), float("inf"), is_group=True)

        assert result is None  # freshness 形态不支持组合，降级为顺序调用

    def test_single_active_review_returns_none(self):
        class Client:
            def chat(self, **kwargs):
                return json.dumps({"decision": "pass", "reason": "", "replies": []},
                                  ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_group_answer_authority_rewrite=True,
        )
        user_prompt = "没有任何诉求候选"
        messages = [{"role": "user",
                     "content": "<unread>去吗？</unread><group_target_action_contract></group_target_action_contract>"}]

        result = gen._run_combined_reply_review(
            user_prompt, ["原回复"], messages, float("inf"), float("inf"), is_group=True)

        assert result is None  # 只有 authority 激活，走原路径


class TestCombinedReplyReviewWiring:
    """_generate 主链路接线：两个审查同时激活时走组合路径（一次 LLM 调用）。"""

    def test_generate_uses_single_combined_review_call_when_two_checks_active(self, monkeypatch):
        calls = []

        class Client:
            def chat(self, **kwargs):
                calls.append(1)
                return json.dumps({
                    "results": {
                        "latest_request": {"decision": "rewrite", "reason": "漏回最新诉求",
                                           "replies": ["改后的回复"]},
                        "stance": {"decision": "pass", "current_assignment": "none",
                                   "current_attitude": "none", "reason": "", "replies": []},
                    }
                }, ensure_ascii=False)

        gen = ReplyGenerator(
            llm_client=Client(),
            tool_registry=_TEST_TOOL_REGISTRY,
            enable_latest_request_first=True,
            enable_current_stance_continuity=True,
        )
        gen.enable_self_refine = False
        monkeypatch.setattr(gen, "_route_skills", lambda *args, **kwargs: [])
        monkeypatch.setattr(gen, "_load_skill_content", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            gen,
            "_react_generate",
            lambda messages, tools, deadline, chat_name: (["原回复"], messages),
        )
        monkeypatch.setattr(gen, "_submit_to_judge", lambda *args, **kwargs: None)
        # 组合路径下这两个独立检查不应被调用（authority 未激活，仍会调用写 state）
        monkeypatch.setattr(
            gen, "_check_latest_request_reply",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("应走组合路径")),
        )
        monkeypatch.setattr(
            gen, "_check_current_stance_reply",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("应走组合路径")),
        )
        unread = ChatMessage(
            text="你们啥时候去？",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        replies = gen.generate([unread], [unread], is_group=True)

        assert len(calls) == 1  # 路由/生成被 mock，只剩组合审查这一次真实 LLM 调用
        assert replies == ["改后的回复"]
        assert gen.last_latest_request_grounding["status"] == "rewritten"
        assert gen.last_current_stance_grounding["status"] == "passed"
