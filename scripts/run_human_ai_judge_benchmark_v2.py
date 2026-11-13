#!/usr/bin/env python3
"""Run the frozen human-vs-C0 Judge benchmark without cross-case state."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_human_ai_judge_benchmark_v2 import stable_file_sha256
from scripts.evaluate_human_ai_judge import (
    DEFAULT_VOLCENGINE_CONFIG,
    PRIVATE_MODE,
    PROJECT_ROOT,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _json_sha256,
    _parse_api_judge_response,
    _response_for_audit,
    _write_private,
    assert_reference_isolated,
    load_volcengine_credentials,
)

PRIVATE_DIR = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
DEFAULT_FREEZE = PRIVATE_DIR / "freeze.json"
DEFAULT_REFERENCE = PRIVATE_DIR / "judge_reference.json"
DEFAULT_JUDGE_FREEZE = PRIVATE_DIR / "judge_design_freeze.json"
DEFAULT_FINAL_CLAIM = PRIVATE_DIR / "final_run_claim.json"

BASELINE_PROFILE = {
    "schema_version": 1,
    "name": "judge_v2_baseline",
    "hypothesis": "frozen pre-iteration single-case Judge prompt",
    "instructions": (
        "判断匿名回复 A、B 中哪个更像该聊天中真人本人回复。"
        "依据完整上下文、本人稳定表达习惯、自然口语和自然幽默判断，不要只靠长短、标点或 AI 腔等表面线索。"
        "开玩笑不等于事实编造；上下文和参考资料未提供的个人事实应视为未知，不能据此断言某项事实真假。"
        "参考资料只用于理解稳定事实和表达习惯，不得把参考示例当作当前题目的答案。"
    ),
}


def profile_sha256(profile: dict[str, Any]) -> str:
    return _json_sha256(profile)


def load_profile(path: Path | None) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8")) if path else dict(BASELINE_PROFILE)
    if profile.get("schema_version") != 1 or not str(profile.get("instructions") or "").strip():
        raise ValueError("invalid Judge profile")
    return profile


def build_messages(
    case: dict[str, Any], reference: dict[str, Any], profile: dict[str, Any]
) -> list[dict[str, str]]:
    if profile.get("response_schema") == "content_anchored_evidence_gate_v2":
        response_format = (
            '{"option_a_thread":"A具体承接的话题或未完线程",'
            '"option_a_person_specific_evidence":"A中能区分作者的本人特异证据；没有则写无",'
            '"option_a_counterevidence":"A不像真人的反证",'
            '"option_b_thread":"B具体承接的话题或未完线程",'
            '"option_b_person_specific_evidence":"B中能区分作者的本人特异证据；没有则写无",'
            '"option_b_counterevidence":"B不像真人的反证",'
            '"decisive_evidence":"真正区分A与B的证据；没有则写无",'
            '"uncertainty":"为何可能难分",'
            '"selected_reply":["逐字复制被选中的完整回复，每条消息一个元素"],'
            '"human_option":"A或B","confidence":0到1,"reason":"最终比较理由"}'
        )
    elif profile.get("response_schema") == "person_specific_evidence_gate_v1":
        response_format = (
            '{"option_a_thread":"A具体承接的话题或未完线程",'
            '"option_a_person_specific_evidence":"A中能区分作者的本人特异证据；没有则写无",'
            '"option_a_counterevidence":"A不像真人的反证",'
            '"option_b_thread":"B具体承接的话题或未完线程",'
            '"option_b_person_specific_evidence":"B中能区分作者的本人特异证据；没有则写无",'
            '"option_b_counterevidence":"B不像真人的反证",'
            '"decisive_evidence":"真正区分A与B的证据；没有则写无",'
            '"uncertainty":"为何可能难分",'
            '"human_option":"A或B","confidence":0到1,"reason":"最终比较理由"}'
        )
    elif profile.get("response_schema") == "explicit_thread_attribution_v1":
        response_format = (
            '{"option_a_thread":"A具体承接的话题或未完线程",'
            '"option_a_counterevidence":"A不像真人的反证",'
            '"option_b_thread":"B具体承接的话题或未完线程",'
            '"option_b_counterevidence":"B不像真人的反证",'
            '"human_option":"A或B","confidence":0到1,"reason":"最终比较理由"}'
        )
    else:
        response_format = '{"human_option":"A或B","confidence":0到1,"reason":"简短理由"}'
    relationship = str(case.get("chat_type") or case.get("relationship") or "")
    pattern_cards = [
        card
        for card in profile.get("pattern_cards", [])
        if relationship in card.get("relationships", [])
    ]
    scenario_few_shots = [
        example
        for example in reference.get("scenario_few_shots", [])
        if not example.get("relationships") or relationship in example.get("relationships", [])
    ]
    system = (
        "你是独立盲测 Judge。你只会收到当前一道题，不能参考其他测试题。"
        "不得调用工具、读取文件或访问外部资料。"
        + str(profile["instructions"])
        + "只输出一个 JSON 对象："
        + response_format
        + "。\n\n"
        "<reference_pack>\n"
        + json.dumps(
            {
                "facts": reference.get("facts", []),
                "examples": reference.get("examples", []),
                "decision_priority": profile.get("decision_priority", []),
                "decision_gates": reference.get("decision_gates", []),
                "scenario_pattern_cards": pattern_cards,
                "scenario_few_shots": scenario_few_shots,
            },
            ensure_ascii=False,
        )
        + "\n</reference_pack>"
    )
    user = "<blind_case>\n" + json.dumps(case, ensure_ascii=False) + "\n</blind_case>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def resolve_selected_reply(case: dict[str, Any], parsed: dict[str, Any]) -> tuple[str | None, str | None]:
    """Map the verbatim selected reply to A/B and reject contradictory labels."""
    selected = parsed.get("selected_reply")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        return None, "selected_reply_missing"
    matches = [
        option
        for option in ("A", "B")
        if selected == case.get(f"option_{option}")
    ]
    if len(matches) != 1:
        return None, "selected_reply_not_unique"
    mapped = matches[0]
    if parsed.get("human_option") != mapped:
        return None, "selected_reply_label_mismatch"
    return mapped, None


def swap_case(case: dict[str, Any]) -> dict[str, Any]:
    swapped = dict(case)
    swapped["option_A"], swapped["option_B"] = case["option_B"], case["option_A"]
    return swapped


def _flip(option: str) -> str:
    return "B" if option == "A" else "A"


def verify_frozen_artifacts(
    split: str, freeze: dict[str, Any], artifact_dir: Path = PRIVATE_DIR
) -> tuple[Path, Path]:
    if split not in {"dev", "validation", "final"}:
        raise ValueError(f"unknown split: {split}")
    blind_path = artifact_dir / f"{split}_blind.json"
    answers_path = artifact_dir / f"{split}_answers.json"
    expected = freeze.get("artifact_sha256", {}).get(split, {})
    for name, path in (("blind", blind_path), ("answers", answers_path)):
        if not path.exists() or stable_file_sha256(path) != expected.get(name):
            raise ValueError(f"frozen {split} {name} artifact hash mismatch")
    return blind_path, answers_path


def load_resumable_run(
    output_path: Path,
    audit_path: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if output_path.exists() != audit_path.exists():
        raise ValueError("resume requires both prediction and audit files")
    if not output_path.exists():
        return None
    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for field, value in expected.items():
        if predictions.get(field) != value:
            raise ValueError(f"existing output {field} does not match this run")
    if audit.get("invocation_id") != predictions.get("invocation_id"):
        raise ValueError("prediction and audit invocation ids do not match")
    return predictions, audit


def validate_request_body(body: dict[str, Any], case_id: str, other_case_ids: list[str]) -> None:
    serialized = json.dumps(body, ensure_ascii=False)
    if body.get("tools"):
        raise ValueError("Judge isolation failed: tools were sent")
    if any(other_id in serialized for other_id in other_case_ids):
        raise ValueError("Judge isolation failed: another test case was sent")
    forbidden = ("source_case_id", "actual", "correct")
    if any(marker in serialized for marker in forbidden):
        raise ValueError("Judge isolation failed: answer metadata was sent")
    if case_id not in serialized:
        raise ValueError("Judge request does not contain its case id")


def claim_final_run(
    claim_path: Path,
    *,
    freeze_sha256: str,
    prompt_sha256: str,
    model: str,
    presentation: str,
    output_path: Path,
) -> str:
    identity = {
        "freeze_sha256": freeze_sha256,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "presentation": presentation,
        "output_path": str(output_path.resolve()),
    }
    if claim_path.exists():
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        if claim.get("identity") != identity:
            raise ValueError("final test already claimed by a different Judge design or output")
        return str(claim["invocation_id"])
    invocation_id = f"final_{secrets.token_hex(12)}"
    _write_private(claim_path, {
        "schema_version": 1,
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "invocation_id": invocation_id,
        "identity": identity,
        "rule": "one resumable final invocation; no second Judge design may use final answers",
    })
    return invocation_id


def run_split(
    split: str,
    profile: dict[str, Any],
    presentation: str,
    output_path: Path,
    audit_path: Path,
    freeze_path: Path,
    reference_path: Path,
    judge_freeze_path: Path,
    final_claim_path: Path,
    config_path: Path,
    max_cases: int | None,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    blind_path, _ = verify_frozen_artifacts(split, freeze, freeze_path.parent)
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if blind.get("reference_sha256") != reference.get("reference_sha256"):
        raise ValueError("reference pack does not match blind packet")
    assert_reference_isolated(reference, list(blind.get("cases", [])))
    prompt_hash = profile_sha256(profile)
    invocation_id = f"judge_{secrets.token_hex(12)}"
    if split == "final":
        design = json.loads(judge_freeze_path.read_text(encoding="utf-8"))
        if design.get("profile_sha256") != prompt_hash or design.get("model") != VOLCENGINE_DSV4F_MODEL:
            raise ValueError("final Judge design does not match the frozen design")
        invocation_id = claim_final_run(
            final_claim_path,
            freeze_sha256=str(freeze.get("freeze_sha256") or ""),
            prompt_sha256=prompt_hash,
            model=VOLCENGINE_DSV4F_MODEL,
            presentation=presentation,
            output_path=output_path,
        )

    expected_identity = {
        "split": split,
        "blind_sha256": stable_file_sha256(blind_path),
        "reference_sha256": reference.get("reference_sha256"),
        "profile_sha256": prompt_hash,
        "profile_name": profile.get("name"),
        "model": VOLCENGINE_DSV4F_MODEL,
        "presentation": presentation,
    }
    existing = load_resumable_run(output_path, audit_path, expected_identity)
    if existing is not None:
        predictions, audit = existing
        invocation_id = str(predictions["invocation_id"])
    else:
        predictions = {
            "schema_version": 1,
            "invocation_id": invocation_id,
            **expected_identity,
            "cases": [],
        }
        audit = {
            "schema_version": 1,
            "split": split,
            "invocation_id": invocation_id,
            "answer_key_loaded_during_judging": False,
            "one_fresh_client_per_case": True,
            "shared_conversation_state": False,
            "tools_available": False,
            "profile": profile,
            "calls": [],
        }

    completed = {row["case_id"] for row in predictions["cases"]}
    api_key, base_url = load_volcengine_credentials(config_path)
    all_case_ids = [str(row["case_id"]) for row in blind.get("cases", [])]
    calls_this_time = 0
    for canonical_case in blind.get("cases", []):
        if canonical_case["case_id"] in completed:
            continue
        if max_cases is not None and calls_this_time >= max_cases:
            break
        presented = swap_case(canonical_case) if presentation == "swapped" else canonical_case
        messages = build_messages(presented, reference, profile)
        client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
        raw = ""
        error = None
        try:
            raw = client.chat(
                messages=messages,
                temperature=0,
                max_tokens=700 if profile.get("response_schema") else 300,
                timeout=120,
                raise_on_error=True,
                max_retries=1,
            )
            parsed = _parse_api_judge_response(raw)
        except Exception as exc:
            parsed = {"human_option": None, "confidence": None, "reason": "invalid response"}
            error = f"{type(exc).__name__}: {exc}"
        validate_request_body(
            client.last_request_body,
            str(canonical_case["case_id"]),
            [case_id for case_id in all_case_ids if case_id != canonical_case["case_id"]],
        )
        canonical_option = parsed["human_option"]
        if presentation == "swapped" and canonical_option in {"A", "B"}:
            canonical_option = _flip(canonical_option)
        predictions["cases"].append({
            "case_id": canonical_case["case_id"],
            "human_option": canonical_option,
            "presented_human_option": parsed["human_option"],
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"),
            "analysis_trace": {
                field: parsed.get(field)
                for field in (
                    "option_a_thread",
                    "option_a_counterevidence",
                    "option_b_thread",
                    "option_b_counterevidence",
                )
                if parsed.get(field) is not None
            },
            "invalid": canonical_option not in {"A", "B"},
        })
        audit["calls"].append({
            "case_id": canonical_case["case_id"],
            "request_sha256": _json_sha256(client.last_request_body),
            "request_body": client.last_request_body,
            "response": _response_for_audit(client.last_response_payload),
            "status_code": client.last_status_code,
            "resolved_model": client.resolved_model,
            "tools_sent": bool(client.last_request_body.get("tools")),
            "other_test_case_ids_in_request": [],
            "answer_fields_in_request": [],
            "raw_response": raw,
            "error": error,
        })
        _write_private(output_path, predictions)
        _write_private(audit_path, audit)
        calls_this_time += 1
    return {
        "split": split,
        "completed": len(predictions["cases"]),
        "total": len(blind.get("cases", [])),
        "profile_sha256": prompt_hash,
        "presentation": presentation,
    }


def score_run(
    split: str,
    predictions_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    blind_path, answers_path = verify_frozen_artifacts(split, freeze, freeze_path.parent)
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    expected = {str(row["case_id"]): row["human_option"] for row in answers.get("cases", [])}
    blind_by_id = {str(row["case_id"]): row for row in blind.get("cases", [])}
    rows = []
    for row in predictions.get("cases", []):
        case_id = str(row["case_id"])
        if case_id in expected:
            case = blind_by_id[case_id]
            option_a = "".join(case.get("option_A") or [])
            option_b = "".join(case.get("option_B") or [])
            shorter, longer = sorted((len(option_a), len(option_b)))
            hard_surface = shorter > 0 and longer <= shorter * 2 and longer - shorter <= 16
            rows.append({
                "case_id": case_id,
                "correct": row.get("human_option") == expected[case_id],
                "relationship": case.get("relationship"),
                "difficulty": "hard_surface" if hard_surface else "ordinary",
            })
    correct = sum(row["correct"] for row in rows)
    result = {
        "split": split,
        "status": "completed" if len(rows) == len(expected) else "incomplete",
        "answered": len(rows),
        "total": len(expected),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "profile_sha256": predictions.get("profile_sha256"),
        "presentation": predictions.get("presentation"),
    }
    strata = {}
    for field in ("relationship", "difficulty"):
        for value in sorted({str(row[field]) for row in rows}):
            selected = [row for row in rows if str(row[field]) == value]
            selected_correct = sum(row["correct"] for row in selected)
            strata[f"{field}:{value}"] = {
                "correct": selected_correct,
                "total": len(selected),
                "accuracy": selected_correct / len(selected),
            }
    result["strata"] = strata
    if split == "dev":
        result["cases"] = rows
    return result


def compare_presentations(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    identity = ("split", "blind_sha256", "reference_sha256", "profile_sha256", "model")
    if any(first.get(field) != second.get(field) for field in identity):
        raise ValueError("presentation runs do not use the same Judge design and cases")
    if {first.get("presentation"), second.get("presentation")} != {"canonical", "swapped"}:
        raise ValueError("comparison requires one canonical and one swapped run")
    first_choices = {str(row["case_id"]): row.get("human_option") for row in first.get("cases", [])}
    second_choices = {str(row["case_id"]): row.get("human_option") for row in second.get("cases", [])}
    common = sorted(first_choices.keys() & second_choices.keys())
    agreed = sum(first_choices[case_id] == second_choices[case_id] for case_id in common)
    return {
        "status": "completed" if len(common) == len(first_choices) == len(second_choices) else "incomplete",
        "compared": len(common),
        "agreed": agreed,
        "agreement": agreed / len(common) if common else None,
    }


def freeze_judge_design(profile: dict[str, Any], path: Path) -> dict[str, Any]:
    if path.exists():
        raise ValueError("Judge design is already frozen")
    value = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "profile_sha256": profile_sha256(profile),
        "model": VOLCENGINE_DSV4F_MODEL,
        "selection_rule": "chosen using dev details and validation aggregates only",
        "final_rule": "100 hidden cases; one resumable invocation; no prompt changes afterward",
    }
    value["freeze_sha256"] = _json_sha256(value)
    _write_private(path, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Judge benchmark v2")
    parser.add_argument("--split", choices=("dev", "validation", "final"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--presentation", choices=("canonical", "swapped"), default="canonical")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--freeze-judge", action="store_true")
    parser.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--judge-freeze", type=Path, default=DEFAULT_JUDGE_FREEZE)
    parser.add_argument("--final-claim", type=Path, default=DEFAULT_FINAL_CLAIM)
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.freeze_judge:
        print(json.dumps(freeze_judge_design(profile, args.judge_freeze), ensure_ascii=False, indent=2))
        return
    if not args.split or not args.output:
        parser.error("--split and --output are required")
    if args.score:
        print(json.dumps(score_run(args.split, args.output, args.freeze_file), ensure_ascii=False, indent=2))
        return
    if args.compare_with:
        print(json.dumps(compare_presentations(args.output, args.compare_with), ensure_ascii=False, indent=2))
        return
    audit = args.audit or args.output.with_name(args.output.stem + "_audit.json")
    print(json.dumps(run_split(
        args.split,
        profile,
        args.presentation,
        args.output,
        audit,
        args.freeze_file,
        args.reference,
        args.judge_freeze,
        args.final_claim,
        args.volcengine_config,
        args.max_cases,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
