import json
from pathlib import Path

from scripts.generate_star_history import generate_chart, generate_svg


def test_generate_svg_from_paginated_github_response(tmp_path: Path):
    stargazers = tmp_path / "stargazers.json"
    repository = tmp_path / "repository.json"
    stargazers.write_text(
        json.dumps([[{"starred_at": "2026-05-04T08:23:05Z"}, {"starred_at": "2026-05-05T09:00:00Z"}]]),
        encoding="utf-8",
    )
    repository.write_text(
        json.dumps({
            "created_at": "2026-05-02T08:32:15Z",
            "full_name": "owner/repo",
            "stargazers_count": 2,
        }),
        encoding="utf-8",
    )

    svg = generate_svg(stargazers, repository)

    assert svg.startswith("<svg")
    assert "owner/repo" in svg
    assert "2 stars" in svg
    assert "2026-05-02" in svg


def test_generate_chart_uses_aggregate_history_when_timestamps_are_restricted(tmp_path: Path):
    stargazers = tmp_path / "stargazers.json"
    repository = tmp_path / "repository.json"
    history = tmp_path / "history.json"
    stargazers.write_text("[]", encoding="utf-8")
    repository.write_text(json.dumps({
        "created_at": "2026-05-02T08:32:15Z",
        "full_name": "owner/repo",
        "stargazers_count": 3,
    }), encoding="utf-8")
    history.write_text(json.dumps({
        "points": [
            {"date": "2026-05-02", "stars": 0},
            {"date": "2026-05-05", "stars": 2},
        ],
    }), encoding="utf-8")

    svg, points = generate_chart(stargazers, repository, history)

    assert "3 stars" in svg
    assert points[-1][1] == 3
