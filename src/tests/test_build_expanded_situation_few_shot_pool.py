import json

from scripts import build_expanded_situation_few_shot_pool as pool_builder
from src.reply.few_shot import PersonaFewShotRetriever


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_excludes_shared_source_id_and_exact_legacy_content(monkeypatch, tmp_path):
    monkeypatch.setattr(pool_builder, "ROOT", tmp_path)
    base = tmp_path / "base.jsonl"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "out" / "persona_examples.jsonl"
    _write_jsonl(
        base,
        [
            {
                "id": "derived-a",
                "source_message_id": "source-1",
                "context": ["不同内容"],
                "reply": ["也要按来源排除"],
                "source_provenance": "explicit_human_marker",
            },
            {"id": "derived-b", "context": ["旧数据"], "reply": ["完全一致"], "source_provenance": "explicit_human_marker"},
            {"id": "same-derived-id", "context": ["安全"], "reply": ["保留"], "source_provenance": "explicit_human_marker"},
        ],
    )
    fixed.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "internal_id": "other-derived-a",
                        "source_message_id": "source-1",
                        "context": ["固定题内容可以不同"],
                        "reply": ["来源相同仍排除"],
                    },
                    {
                        "internal_id": "other-derived-b",
                        "context": ["旧数据"],
                        "reply": ["完全一致"],
                    },
                    {
                        "internal_id": "same-derived-id",
                        "context": ["另一条"],
                        "reply": ["不能只看派生 ID"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = pool_builder.build(base, fixed, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["id"] for row in rows] == ["same-derived-id"]
    assert report["counts"]["fixed_source_id_excluded"] == 1
    assert report["counts"]["fixed_pair_excluded_unique_ids"] == 1
    assert report["leakage_exclusion"]["derived_ids_not_compared"] == ["id", "internal_id"]
    assert report["counts"]["output"] <= report["counts"]["base_input"]
    assert report["base_sha256"]
    assert report["examples_sha256"] == report["output_sha256"]
    assert report["output_sha256"]
    assert PersonaFewShotRetriever(output).is_approved()
