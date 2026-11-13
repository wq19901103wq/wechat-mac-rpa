#!/usr/bin/env python3
"""Merge completed formal-240 shards and build the fixed mobile report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from build_clean_human_c0_iteration_report import build_formal_report
from run_clean_human_c0_baseline import PROFILE, REFERENCE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/reports/private/situation_c0_latest_turn_form_formal240_ab_v4"
DEFAULT_OUTPUT = ROOT / "data/reports/private/clean_human_c0_iteration_v1"
EXPECTED_JUDGE_PROFILE_SHA256 = "28a36ce0edaf9603f1422048bb53008de1117cd492c93243364505e8a2c5dfff"
EXPECTED_JUDGE_REFERENCE_SHA256 = "4a2a327626b384130e02f738d9556487a494b41f4a721bb250555806dff256a4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def response_time(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(nearest_rank(values, 0.95), 3),
    }


def exact_sign_test(improved: int, regressed: int) -> float:
    n = improved + regressed
    tail = sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize(rows: list[dict]) -> dict:
    baseline_correct = sum(row["baseline"]["judge"]["correct"] for row in rows)
    candidate_correct = sum(row["candidate"]["judge"]["correct"] for row in rows)
    improved = sum(
        row["baseline"]["judge"]["correct"] and not row["candidate"]["judge"]["correct"]
        for row in rows
    )
    regressed = sum(
        not row["baseline"]["judge"]["correct"] and row["candidate"]["judge"]["correct"]
        for row in rows
    )
    return {
        "paired": len(rows),
        "baseline_judge_correct": baseline_correct,
        "candidate_judge_correct": candidate_correct,
        "baseline_judge_discrimination_accuracy": baseline_correct / len(rows),
        "candidate_judge_discrimination_accuracy": candidate_correct / len(rows),
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(rows) - improved - regressed,
        "net_improvement": improved - regressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    shard_paths = sorted(args.input_dir.glob("results.shard-*-of-4.json"))
    assert len(shard_paths) == 4, f"expected 4 result shards, found {len(shard_paths)}"
    shards = [load(path) for path in shard_paths]
    rows = [row for shard in shards for row in shard.get("cases", [])]
    manifest = load(args.input_dir / "manifest.json")
    manifest_ids = list(manifest["case_ids"])

    assert len(rows) == 240, f"expected 240 cases, found {len(rows)}"
    assert len({row["case_id"] for row in rows}) == 240, "case IDs are not unique"
    assert {row["case_id"] for row in rows} == set(manifest_ids), "result and manifest case IDs differ"
    assert sum(row["relationship"] == "group" for row in rows) == 200
    assert sum(row["relationship"] == "private" for row in rows) == 40
    assert manifest["per_case_few_shot_overlap_excluded"] is True
    assert "private chats add latest-turn-form ranking" in manifest["conditions"]["candidate"]
    assert "group chats reuse the baseline output" in manifest["conditions"]["candidate"]
    libraries = manifest["conditions"]["retrieval_libraries"]
    assert libraries["baseline"] == libraries["candidate"], "A/B retrieval libraries differ"
    assert libraries["baseline"]["rows"] == 80530

    converted = []
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for source in rows:
        conditions = source.get("conditions") or {}
        assert set(conditions) == {"baseline", "candidate"}
        output = {
            "case_id": source["case_id"],
            "chat_type": source["relationship"],
            "context": source.get("context") or [],
            "human_reply": source.get("human_reply") or "",
        }
        for condition in ("baseline", "candidate"):
            runs = conditions[condition].get("runs") or []
            assert len(runs) == 1, f"{source['case_id']} {condition}: expected one run"
            run = runs[0]
            assert run.get("status") == "scorable", f"{source['case_id']} {condition}: not scorable"
            assert run.get("reply"), f"{source['case_id']} {condition}: empty reply"
            assert run.get("judge"), f"{source['case_id']} {condition}: missing Judge result"
            assert run.get("error") is None, f"{source['case_id']} {condition}: generation error"
            assert isinstance(run["judge"].get("correct"), bool)
            output[condition] = {"c0_reply": run["reply"], "judge": run["judge"]}
            (baseline_times if condition == "baseline" else candidate_times).append(
                float(run["duration_seconds"])
            )
        if source["relationship"] == "group":
            assert (
                output["baseline"]["c0_reply"] == output["candidate"]["c0_reply"]
            ), f"{source['case_id']}: group candidate differs from baseline"
            assert (
                output["baseline"]["judge"] == output["candidate"]["judge"]
            ), f"{source['case_id']}: group Judge result differs from baseline"
        converted.append(output)

    converted.sort(key=lambda row: manifest_ids.index(row["case_id"]))
    overall = summarize(converted)
    group = summarize([row for row in converted if row["chat_type"] == "group"])
    private = summarize([row for row in converted if row["chat_type"] == "private"])

    assert group["baseline_judge_correct"] == group["candidate_judge_correct"]
    assert group["improved"] == 0 and group["regressed"] == 0 and group["unchanged"] == 200
    assert sha256(PROFILE) == EXPECTED_JUDGE_PROFILE_SHA256
    assert sha256(REFERENCE) == EXPECTED_JUDGE_REFERENCE_SHA256

    hard_failure_review = [
        {
            "case_id": "03ff311b5114cdc82df4",
            "change": "improved",
            "hard_failure": False,
            "reason": "两版都是对调薪话题的调侃；候选新增的比喻没有与上下文或可靠 Wiki 直接矛盾。",
        },
        {
            "case_id": "941586b0d2a997f36d02",
            "change": "improved",
            "hard_failure": False,
            "reason": "候选询问对方所在银行并估计等待时间；上下文和可靠 Wiki 没有相反事实。",
        },
        {
            "case_id": "0100e87196fbd0d2b3c4",
            "change": "improved",
            "hard_failure": False,
            "reason": "候选询问床垫加床架套装价格；可见上下文和可靠 Wiki 没有提供套装构成真值。",
        },
        {
            "case_id": "6fabbe2b455527885f14",
            "change": "improved",
            "hard_failure": False,
            "reason": "候选表示下次换到光线好的地方；缺失媒体使现场动作不可核实，没有直接矛盾。",
        },
        {
            "case_id": "5ebf20c87ef04e6ed035",
            "change": "regressed",
            "hard_failure": False,
            "reason": "候选称自己没有驾照属于无依据新增信息；上下文只确认不会开车，未形成直接矛盾。",
        },
        {
            "case_id": "d825c6ff678ccbaec5a9",
            "change": "regressed",
            "hard_failure": False,
            "reason": "候选提出到了直接敲门属于新增安排；上下文和可靠 Wiki 没有相反安排。",
        },
        {
            "case_id": "d4b63595d5da2a257473",
            "change": "regressed",
            "hard_failure": False,
            "reason": "候选只回答周六，较旧版少了下午；信息减少不构成与上下文或可靠 Wiki 的直接矛盾。",
        },
    ]

    summary = {
        "status": "complete",
        **overall,
        "segments": {
            "group": {"name": "群聊", **group},
            "private": {"name": "私聊", **private},
        },
        "response_time_seconds": {
            "baseline": response_time(baseline_times),
            "candidate": response_time(candidate_times),
        },
        "paired_exact_sign_test_p": exact_sign_test(overall["improved"], overall["regressed"]),
        "judge": {
            "version": "v6",
            "profile": str(PROFILE.relative_to(ROOT)),
            "profile_sha256": sha256(PROFILE),
            "reference": str(REFERENCE.relative_to(ROOT)),
            "reference_sha256": sha256(REFERENCE),
            "formal_240_reference_overlap": 0,
        },
        "adopted": True,
        "decision": "采用：私聊启用最新一句形式排序；群聊继续使用旧策略",
        "attributable_new_hard_failures": 0,
        "hard_failure_review": hard_failure_review,
        "hard_failure_status": (
            "已复核私聊 7 条 Judge 判断翻转案例：与完整上下文或可靠 Wiki 直接矛盾的疑似案例 0 条，"
            "确认硬伤 0 条，需重复归因复验 0 条，可归因新增硬伤 0 条"
        ),
    }
    payload = {"schema_version": 1, "manifest": manifest, "summary": summary, "cases": converted}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_formal_report(payload, output_dir=args.output_dir)
    (args.output_dir / "index.html").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
