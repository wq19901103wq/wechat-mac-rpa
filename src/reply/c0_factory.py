"""The single construction entrypoint for the adopted C0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm.volcengine_anthropic_client import (
    DEFAULT_VOLCENGINE_CONFIG,
    VolcengineAnthropicClient,
    load_volcengine_credentials,
)
from src.reply.c0_config import load_current_c0
from src.reply.generator import ReplyGenerator


def create_current_c0_clients(
    *,
    config: dict[str, Any] | None = None,
    volcengine_config: Path = DEFAULT_VOLCENGINE_CONFIG,
    private_client: Any = None,
    group_client: Any = None,
) -> tuple[Any, Any]:
    config = config or load_current_c0()
    if private_client is not None and group_client is not None:
        return private_client, group_client
    api_key, base_url = load_volcengine_credentials(volcengine_config)
    private_client = private_client or VolcengineAnthropicClient(api_key, base_url, config["private"]["model"])
    group_client = group_client or VolcengineAnthropicClient(api_key, base_url, config["group"]["model"])
    return private_client, group_client


def create_current_c0_generator(
    *,
    memory_engine: Any = None,
    tool_registry: Any = None,
    judge_worker: Any = None,
    private_client: Any = None,
    group_client: Any = None,
    config: dict[str, Any] | None = None,
    volcengine_config: Path = DEFAULT_VOLCENGINE_CONFIG,
    **overrides: Any,
) -> ReplyGenerator:
    config = config or load_current_c0()
    private_client, group_client = create_current_c0_clients(
        config=config,
        volcengine_config=volcengine_config,
        private_client=private_client,
        group_client=group_client,
    )
    features = config["features"]
    few_shot = config["few_shot"]
    execution = config["execution"]
    arguments = {
        "llm_client": private_client,
        "group_llm_client": group_client,
        "memory_engine": memory_engine,
        "tool_registry": tool_registry,
        "judge_worker": judge_worker,
        "generation_timeout_seconds": config["private"]["timeout_seconds"],
        "group_generation_timeout_seconds": config["group"]["timeout_seconds"],
        "enable_persona_few_shots": few_shot["enabled"],
        "persona_few_shot_path": config["few_shot_path"],
        "group_persona_few_shot_path": config["few_shot_path"],
        "persona_few_shot_count": few_shot["count"],
        "persona_few_shot_max_chars": few_shot["max_chars"],
        "prefer_latest_turn_form": features["prefer_latest_turn_form"],
        "group_prefer_latest_turn_form": features["group_prefer_latest_turn_form"],
        "enable_precise_situation_matching": features["enable_precise_situation_matching"],
        "enable_information_gap_response_move": features["enable_information_gap_response_move"],
        "enable_response_action_grounding": features["enable_response_action_grounding"],
        "disabled_skills": config["skills"]["disabled"],
    }
    arguments.update(overrides)
    arguments["enable_react_tools"] = execution["enable_react_tools"]
    arguments["enable_self_refine"] = execution["enable_self_refine"]
    return ReplyGenerator(**arguments)
