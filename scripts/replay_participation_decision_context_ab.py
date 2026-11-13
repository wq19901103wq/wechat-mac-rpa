#!/usr/bin/env python3
"""A/B allowing recent self messages for participation decision only."""

from __future__ import annotations

import replay_participation_recent_thread_ab as base


OLD_RULE = (
    '                "<rule>consumed_self_replies 是你已经发过的话，通常只用于排除素材；若 unread 明确追问其中某句话，'
    '必须用它理解追问并直接回答，但仍不得复用其中的包袱。</rule>",'
)
NEW_RULE = (
    '                "<rule>consumed_self_replies 只可用于第一步二元参与判断：若 unread 在最近几分钟内直接回答、补充或延续'
    '你刚才发起或参与的当前话题，应判断为需要参与；这项许可不延伸到回复内容。生成时它仍只是排除素材，不得复用其中的'
    '事实、措辞、判断、人物标签、关系、处境或包袱。</rule>",'
)


def _candidate_source(c0_source: str) -> str:
    if c0_source.count(OLD_RULE) != 1:
        raise ValueError("decision-context candidate anchor did not match exactly once")
    return c0_source.replace(OLD_RULE, NEW_RULE, 1)


base.CANDIDATE_LABEL = "C3"
base.CANDIDATE_RULE = "Allow consumed self messages for participation decision only, never reply material."
base._candidate_source = _candidate_source


if __name__ == "__main__":
    base.main()
