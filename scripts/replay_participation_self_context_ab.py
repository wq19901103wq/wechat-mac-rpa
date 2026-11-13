#!/usr/bin/env python3
"""A/B separating recent self context from forbidden reply-material reuse."""

from __future__ import annotations

import replay_participation_recent_thread_ab as base


OLD_SECTION = r'''            sections.append(
                '<consumed_self_replies reuse="forbidden">\n'
                + "\n".join(consumed_self_items)
                + "\n</consumed_self_replies>"
            )'''
NEW_SECTION = r'''            sections.append(
                '<recent_self_messages participation_context="allowed" reply_material_reuse="forbidden">\n'
                + "\n".join(consumed_self_items)
                + "\n</recent_self_messages>"
            )'''

OLD_RULES = '''                "<rule>consumed_self_replies 是你已经发过的话，通常只用于排除素材；若 unread 明确追问其中某句话，必须用它理解追问并直接回答，但仍不得复用其中的包袱。</rule>",
                "<rule>候选回复只要依赖其中已用过的事实、数字、人物标签、虚构关系、处境、反差或包袱，就必须丢弃。</rule>",
                "<rule>不得围绕已用素材新增动作、状态或场景；已用素材不能成为回复的主语、宾语、修饰对象、背景或隐含前提。</rule>",
                "<rule>即使 unread 再次引用这些内容，也不得 callback；宁可字面回应、换全新角度或不回复。</rule>",
                "<rule>若 unread 只是复述或抱怨已用素材且没有独立新内容，replies 必须为空；明确提问、请求解释或继续对话不适用此条。</rule>",'''
NEW_RULES = '''                "<rule>recent_self_messages 同时用于判断是否参与和排除重复素材：若 unread 在最近几分钟内直接回答、补充或延续其中正在进行的当前话题，应视为对话仍在继续并参与。</rule>",
                "<rule>生成回复时不得照搬 recent_self_messages 中已经说过的措辞、完整判断、人物标签、虚构关系、反差或包袱；但可以用其中事实理解 unread 并保持当前话题连贯。</rule>",
                "<rule>不得围绕已用包袱生造新动作、状态或场景；允许针对 unread 的独立新增内容作简短直接回应。</rule>",
                "<rule>不得在 unread 没有新增内容时 callback；unread 直接延续当前话题时，应回应新增内容而不是复读旧话。</rule>",
                "<rule>若 unread 只是复述或抱怨已用素材且没有独立新内容，replies 必须为空；明确提问、请求解释或继续对话不适用此条。</rule>",'''


def _candidate_source(c0_source: str) -> str:
    if c0_source.count(OLD_SECTION) != 1 or c0_source.count(OLD_RULES) != 1:
        raise ValueError("self-context candidate anchors did not match exactly once")
    return c0_source.replace(OLD_SECTION, NEW_SECTION, 1).replace(OLD_RULES, NEW_RULES, 1)


base.CANDIDATE_LABEL = "C2"
base.CANDIDATE_RULE = "Separate recent self participation context from forbidden reply-material reuse."
base._candidate_source = _candidate_source


if __name__ == "__main__":
    base.main()
