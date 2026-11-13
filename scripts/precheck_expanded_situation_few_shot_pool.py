#!/usr/bin/env python3
"""Compare old and expanded few-shot retrieval on the saved 80-case development set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import _export_payloads, rebuild_generation_inputs  # noqa: E402
from scripts.run_situation_c0_ab import (  # noqa: E402
    DEFAULT_CANDIDATE_FEW_SHOT,
    DEFAULT_DEVELOPMENT,
    DEFAULT_EXPORTS,
    DEFAULT_FEW_SHOT,
    DEFAULT_FIXED_TEST,
    DEFAULT_NAMED_CONTEXT_FEW_SHOT,
    _candidate,
)
from src.reply.few_shot import PersonaFewShotRetriever  # noqa: E402


DEFAULT_BASELINE_RESULTS = ROOT / "data/reports/private/situation_c0_no_restatement_action_ab_v1/results.json"
DEFAULT_BASELINE_CHECKPOINT = ROOT / "data/reports/private/situation_c0_no_restatement_action_ab_v1/generation_checkpoint.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/situation_c0_expanded_pool_precheck_v1/results.json"


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["id"])] = row
    return result


def _pair(row: dict[str, Any]) -> str:
    context = "\n".join(str(value).strip() for value in row.get("context") or [] if str(value).strip())
    reply = "\n".join(str(value).strip() for value in row.get("reply") or [] if str(value).strip())
    return f"{context}\n<REPLY>\n{reply}"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _retrieve(
    retriever: PersonaFewShotRetriever,
    *,
    query: str,
    chat_name: str,
    chat_id: str,
    is_group: bool,
    case_id: str,
    skills: list[str],
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows = retriever.retrieve(
        query=query,
        chat_name=chat_name,
        chat_id=chat_id,
        is_group=is_group,
        limit=3,
        relationship="group" if is_group else "acquaintance",
        interaction_context=" ".join(skills),
        exclude_ids={case_id},
        use_situation=True,
    )
    return rows, round((time.perf_counter() - started) * 1000, 3)


def run(args: argparse.Namespace) -> None:
    baseline_results = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.baseline_checkpoint.read_text(encoding="utf-8"))
    development = json.loads(args.development.read_text(encoding="utf-8"))
    fixed = json.loads(args.fixed_test.read_text(encoding="utf-8"))
    development_by_id = {str(row["id"]): row for row in development["cases"]}
    exports = _export_payloads(args.exports)
    old_by_id = _load_jsonl(args.baseline_pool)
    new_by_id = _load_jsonl(args.candidate_pool)
    added_ids = set(new_by_id) - set(old_by_id)
    fixed_ids = {str(row.get("internal_id") or row.get("id") or "") for row in fixed["cases"]}
    fixed_pairs = {
        _pair({"context": row.get("context") or row.get("context_original") or [], "reply": row.get("human_reply") or row.get("human_reply_original") or []})
        for row in fixed["cases"]
    }

    baseline = PersonaFewShotRetriever(
        args.baseline_pool,
        render_path=args.named_context_pool,
        enable_precise_situation=True,
    )
    candidate = PersonaFewShotRetriever(
        args.candidate_pool,
        render_path=args.named_context_pool,
        enable_precise_situation=True,
    )
    if not baseline.is_approved() or not candidate.is_approved():
        raise ValueError("both retrieval pools must be approved")

    cases = []
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    added_recall_count = 0
    holdout_leaks = []
    for result_case in baseline_results["cases"]:
        case_id = str(result_case["case_id"])
        row = development_by_id[case_id]
        source = _candidate(row)
        payload = exports[source.chat_id]
        to_reply, all_messages, metadata = rebuild_generation_inputs(source, payload, compact_verified_turn=True)
        last_msg = to_reply[-1]
        route_text = last_msg.text or last_msg.image_description or ""
        prior = list(all_messages[:-len(to_reply)]) + list(to_reply[:-1])
        context_msgs = [(message.sender, message.text) for message in prior if message.text][-3:]
        query = "\n".join([text for _, text in context_msgs] + [route_text])
        saved = checkpoint["entries"].get(f"{case_id}:baseline", {})
        skills = list(((saved.get("result") or {}).get("trace") or {}).get("loaded_skills") or [])
        if not skills:
            skills = ["group_banter" if source.relation == "group" else "casual_chat"]
        old_rows, old_ms = _retrieve(
            baseline,
            query=query,
            chat_name=metadata["chat_name"],
            chat_id=source.chat_id,
            is_group=source.relation == "group",
            case_id=case_id,
            skills=skills,
        )
        new_rows, new_ms = _retrieve(
            candidate,
            query=query,
            chat_name=metadata["chat_name"],
            chat_id=source.chat_id,
            is_group=source.relation == "group",
            case_id=case_id,
            skills=skills,
        )
        baseline_times.append(old_ms)
        candidate_times.append(new_ms)
        old_ids = [str(item["id"]) for item in old_rows]
        new_ids = [str(item["id"]) for item in new_rows]
        recalled_added = [row_id for row_id in new_ids if row_id in added_ids]
        added_recall_count += len(recalled_added)
        leaks = [
            row_id for row_id in new_ids
            if row_id in fixed_ids or _pair(new_by_id[row_id]) in fixed_pairs
        ]
        if leaks:
            holdout_leaks.append({"case_id": case_id, "few_shot_ids": leaks})
        cases.append({
            "case_id": case_id,
            "relationship": result_case["relationship"],
            "context": result_case["context"],
            "human_reply": result_case["human_reply"],
            "query": query,
            "skills": skills,
            "changed": old_ids != new_ids,
            "baseline": {"ids": old_ids, "rows": old_rows, "retrieval_ms": old_ms},
            "candidate": {"ids": new_ids, "rows": new_rows, "retrieval_ms": new_ms, "added_ids": recalled_added},
        })

    changed = [row for row in cases if row["changed"]]
    summary = {
        "case_count": len(cases),
        "changed_case_count": len(changed),
        "unchanged_case_count": len(cases) - len(changed),
        "added_pool_ids": len(added_ids),
        "added_id_recall_occurrences": added_recall_count,
        "candidate_holdout_leak_count": sum(len(row["few_shot_ids"]) for row in holdout_leaks),
        "retrieval_ms": {
            "baseline": {
                "mean": round(statistics.mean(baseline_times), 3),
                "p50": round(statistics.median(baseline_times), 3),
                "p95": round(_percentile(baseline_times, 0.95), 3),
            },
            "candidate": {
                "mean": round(statistics.mean(candidate_times), 3),
                "p50": round(statistics.median(candidate_times), 3),
                "p95": round(_percentile(candidate_times, 0.95), 3),
            },
        },
    }
    output = {"schema_version": 1, "feature": "expanded few-shot retrieval pool only", "summary": summary, "holdout_leaks": holdout_leaks, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--fixed-test", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--baseline-pool", type=Path, default=DEFAULT_FEW_SHOT)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_FEW_SHOT)
    parser.add_argument("--named-context-pool", type=Path, default=DEFAULT_NAMED_CONTEXT_FEW_SHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
