#!/usr/bin/env python3
"""Run an isolated C0 vs strict same-chat persona-few-shot style A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reply.few_shot import PersonaFewShotRetriever, _chat_id, _terms  # noqa: E402

PRIVATE_MODE = 0o600
PROTECTED_C0_SHA256 = "625b47803fff55a2e18914c5e734fcb2f01ae7c16e5180fdebdc817badd2656b"
CUTOFF = 1774972800.0
S2_GUARDRAILS = """
<candidate_experiment_guardrails>
现实事实只有在当前聊天原文或可靠记忆/工具证据明确支持时才能断言；不确定就坦率说不知道、自然岔开或不回复。明显荒诞的夸张和双方都能识别的玩笑不属于事实断言。
不得承诺会让对方合理相信本人将在线下执行的行动，例如转账、发红包、赴约、到场后联系；也不得声称这些行动已经发生。可以用不造成真实期待的明显玩笑回应。
</candidate_experiment_guardrails>
""".strip()
_COMMON_UNITS = {
    "一个", "一下", "不是", "不会", "什么", "他们", "你们", "可以", "已经", "怎么",
    "我们", "没有", "现在", "知道", "自己", "这个", "那个", "还是", "就是", "可能", "还有",
}


def _load_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


class StrictSameChatRetriever(PersonaFewShotRetriever):
    """Experimental retriever: never transfers style examples across chats."""

    def retrieve(
        self,
        query: str,
        chat_name: str,
        is_group: bool,
        limit: int = 3,
        relationship: str | None = None,
        chat_id: str | None = None,
        interaction_context: str = "",
    ) -> list[dict[str, Any]]:
        del relationship, interaction_context
        current_chat_id = chat_id or (_chat_id(chat_name) if chat_name else "")
        query_terms = Counter(_terms(query))
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in self._load():
            if row.get("chat_id") != current_chat_id:
                continue
            if is_group != (row.get("relationship") == "group"):
                continue
            sample_text = " ".join(row["context"])
            sample_terms = Counter(_terms(sample_text))
            overlap = sum(min(count, sample_terms.get(term, 0)) for term, count in query_terms.items())
            lexical = overlap / max(1, sum(query_terms.values()))
            length_similarity = 1.0 / (1.0 + abs(len(query) - len(sample_text)) / 20.0)
            scored.append((-(lexical * 8.0 + length_similarity), str(row["id"]), row))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [row for _, _, row in scored[: max(0, limit)]]


def _semantic_units(text: str) -> set[str]:
    normalized = "".join(char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")
    units = {
        normalized[index:index + 2]
        for index in range(max(0, len(normalized) - 1))
        if normalized[index:index + 2] not in _COMMON_UNITS
    }
    units.update(part.lower() for part in text.split() if len(part) >= 2 and part.lower() not in _COMMON_UNITS)
    return units


class RelevantSameChatRetriever(PersonaFewShotRetriever):
    """S2 retriever: same chat is necessary, but irrelevant examples are omitted."""

    def retrieve(
        self,
        query: str,
        chat_name: str,
        is_group: bool,
        limit: int = 2,
        relationship: str | None = None,
        chat_id: str | None = None,
        interaction_context: str = "",
    ) -> list[dict[str, Any]]:
        del relationship, interaction_context
        current_chat_id = chat_id or (_chat_id(chat_name) if chat_name else "")
        query_units = _semantic_units(query)
        scored: list[tuple[float, int, str, dict[str, Any]]] = []
        for row in self._load():
            if row.get("chat_id") != current_chat_id:
                continue
            if is_group != (row.get("relationship") == "group"):
                continue
            sample_units = _semantic_units(" ".join(row["context"]))
            overlap = len(query_units & sample_units)
            dice = (2.0 * overlap) / max(1, len(query_units) + len(sample_units))
            if overlap < 2 or dice < 0.06:
                continue
            scored.append((-dice, -overlap, str(row["id"]), row))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))

        selected: list[dict[str, Any]] = []
        if scored and limit > 0:
            selected.append({**scored[0][3], "retrieval_slot": "topic"})

        relation_candidates = [
            item for item in scored[1:]
            if item[3].get("humor_type") != "none" or item[3].get("response_mode") in {"playful", "sincere"}
        ]
        if relation_candidates and limit > 1:
            selected.append({**relation_candidates[0][3], "retrieval_slot": "relationship_tone"})
        return selected[: max(0, limit)]


def _install_s2_guardrails(generator: Any) -> None:
    original_system_prompt = generator._system_prompt

    def guarded_system_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        return original_system_prompt(*args, **kwargs) + "\n\n" + S2_GUARDRAILS

    generator._system_prompt = MethodType(guarded_system_prompt, generator)


def _serialize_message(message: Any) -> dict[str, Any]:
    return {
        "sender": message.sender,
        "sender_type": str(message.sender_type.value),
        "text": message.text,
        "timestamp": message.timestamp,
        "message_type": message.message_type,
    }


def _run_one(case: dict[str, Any], version: str, view_path: Path) -> dict[str, Any]:
    from src.reply.generator import ReplyGenerator
    from src.tools.tool_registry import ToolRegistry
    from src.utils.qwen_client import QwenClient

    started = time.monotonic()
    result: dict[str, Any] = {"status": "failed", "action": None, "replies": []}
    try:
        client = QwenClient()
        generator = ReplyGenerator(
            llm_client=client,
            memory_engine=None,
            tool_registry=ToolRegistry(),
            judge_worker=None,
        )
        if version in {"S1", "S2"}:
            generator.persona_few_shot_retriever = (
                StrictSameChatRetriever(view_path)
                if version == "S1"
                else RelevantSameChatRetriever(view_path)
            )
            generator.enable_persona_few_shots = True
            generator.persona_few_shot_allow_unreviewed = False
            generator.persona_few_shot_ready = generator.persona_few_shot_retriever.is_approved()
            generator.persona_few_shot_count = 3 if version == "S1" else 2
        if version == "S2":
            _install_s2_guardrails(generator)
        replies = generator.generate(
            case["unreplied"],
            case["all_messages"],
            is_group=case["is_group"],
            tick_id=case["source_tick_id"],
            evaluation_time=case["evaluation_time"],
        )
        generation_failed = bool(generator.last_generation_failed)
        result.update({
            "status": "completed",
            "action": "generation_failed" if generation_failed else ("reply" if replies else "silence"),
            "replies": replies,
            "generation_failed": generation_failed,
            "model": client.model,
            "few_shot_ids": list(generator.last_few_shot_ids),
            "loaded_skills": list(generator.last_loaded_skills),
            "llm_call_count": len(generator.last_llm_calls),
            "tool_call_count": len(generator.last_tool_calls),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="隔离回放 C0 与严格同聊天风格 few-shot")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate", choices=("S1", "S2"), default="S1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/cases.db")
    parser.add_argument("--scenarios", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json")
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json")
    parser.add_argument("--view", type=Path, default=PROJECT_ROOT / "data/reports/private/style_same_chat_v1/persona_examples.jsonl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.output is None:
        filename = "same_chat_style_ab_replay.json" if args.candidate == "S1" else "same_chat_relevant_style_ab_replay.json"
        args.output = PROJECT_ROOT / "data/reports/private" / filename
    _load_env()
    os.environ["ENABLE_SELF_REFINE"] = "0"
    os.environ["ENABLE_FACT_CHECK"] = "1"
    os.environ["ENABLE_PERSONA_FEW_SHOTS"] = "1"
    os.environ["PERSONA_FEW_SHOT_COUNT"] = "3"

    from src.benchmarks.reply_system_replay import load_replay_cases

    generator_path = PROJECT_ROOT / "src/reply/generator.py"
    c0_sha = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    if c0_sha != PROTECTED_C0_SHA256:
        raise ValueError("current generator no longer matches protected C0")
    view_report = json.loads(args.view.with_name("report.json").read_text(encoding="utf-8"))
    if view_report.get("review_status") != "approved" or view_report.get("verified_count") != 973:
        raise ValueError("same-chat view is not the expected verified dataset")
    if hashlib.sha256(args.view.read_bytes()).hexdigest() != view_report.get("examples_sha256"):
        raise ValueError("same-chat view hash mismatch")
    source_fingerprint, cases = load_replay_cases(args.db, args.scenarios, args.selection)
    if args.limit is not None:
        cases = cases[: args.limit]
    if any(case["evaluation_time"].timestamp() < CUTOFF for case in cases):
        raise ValueError("evaluation cases must occur after all candidate style examples")

    validation = {
        "case_count": len(cases),
        "runs": args.runs,
        "source_fingerprint": source_fingerprint,
        "c0_generator_sha256": c0_sha,
        "c0_persona_path": os.environ.get("PERSONA_FEW_SHOT_PATH"),
        "candidate_view": str(args.view.relative_to(PROJECT_ROOT)),
        "candidate_verified_rows": view_report["verified_count"],
        "candidate_version": args.candidate,
        "candidate_retrieval": (
            "strict chat_id equality, maximum 3 examples"
            if args.candidate == "S1"
            else "strict chat_id equality plus lexical relevance threshold; zero allowed; maximum one topic and one labeled relationship-tone example"
        ),
        "candidate_guardrails": args.candidate == "S2",
        "only_intended_difference": (
            "positive style example source and same-chat retrieval"
            if args.candidate == "S1"
            else "relevance-gated same-chat examples plus explicit unsupported-fact and believable-action-commitment boundaries"
        ),
        "bot_constructed": False,
        "action_tools_registered": False,
        "memory_enabled": False,
        "judge_worker_enabled": False,
        "self_refine": False,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    examples_by_id = {
        row["id"]: row
        for row in (json.loads(line) for line in args.view.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "same_chat_style_ab_replay" if args.candidate == "S1" else "same_chat_relevant_style_ab_replay",
        "scope": "expression style screening; participation differences are reported but not the main score",
        "status": "running",
        "validation": validation,
        "cases": [
            {
                "case_id": case["case_id"],
                "is_group": case["is_group"],
                "evaluation_time": case["evaluation_time"].isoformat(),
                "context_original": [_serialize_message(message) for message in case["all_messages"]],
                "target_original": [_serialize_message(message) for message in case["unreplied"]],
                "runs": {"C0": [], args.candidate: []},
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {item["case_id"]: item for item in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for case in cases:
            for version in ("C0", args.candidate):
                result = _run_one(case, version, args.view)
                result["run"] = run_number
                if version == args.candidate:
                    result["human_style_examples"] = [examples_by_id[row_id] for row_id in result.get("few_shot_ids", [])]
                report_by_id[case["case_id"]]["runs"][version].append(result)
                _write_private(args.output, report)
    report["status"] = "completed"
    report["completed_runs"] = args.runs * len(cases) * 2
    report["expected_runs"] = report["completed_runs"]
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
