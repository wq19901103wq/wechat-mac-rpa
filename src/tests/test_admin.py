import json

import scripts.admin as admin
from scripts.admin import _expand_persona_few_shots


def test_expand_persona_few_shots_from_ids(tmp_path, monkeypatch):
    examples = tmp_path / "persona_examples.jsonl"
    examples.write_text(json.dumps({
        "id": "style_one",
        "context": ["在吗"],
        "reply": ["咋啦"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("PERSONA_FEW_SHOT_PATH", str(examples))

    result = _expand_persona_few_shots("【persona few-shot 正文已省略；ids=style_one】")

    assert "对方：在吗" in result
    assert "本人：咋啦" in result


def test_expand_persona_few_shots_uses_dotenv_path(tmp_path, monkeypatch):
    examples = tmp_path / "persona_examples.jsonl"
    examples.write_text(json.dumps({
        "id": "style_v4",
        "context": ["旧日志"],
        "reply": ["读当前库"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        f"PERSONA_FEW_SHOT_PATH={examples}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PERSONA_FEW_SHOT_PATH", raising=False)
    monkeypatch.setattr(admin, "WORK_DIR", str(tmp_path))

    result = _expand_persona_few_shots("【persona few-shot 正文已省略；ids=style_v4】")

    assert "对方：旧日志" in result
    assert "本人：读当前库" in result
