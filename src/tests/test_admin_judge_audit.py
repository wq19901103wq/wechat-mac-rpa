import json

import scripts.admin as admin


def _write(path, name, value):
    (path / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_embed_benchmark_removes_only_duplicate_heading(tmp_path):
    report = tmp_path / "report.html"
    report.write_text(
        "<html><head><style>.judge-page{max-width:920px}</style></head>"
        "<body><div class='judge-page'><div><h1>重复标题</h1>"
        "<p id='content'>正文</p></div></div><script>window.ready=true</script></body></html>",
        encoding="utf-8",
    )

    page = admin._embed_benchmark(report, "后台标题", "/benchmark/judge/group")

    assert "class='judge-page'" in page
    assert "id='content'" in page
    assert "window.ready=true" in page
    assert "重复标题" not in page


def test_youzi_identity_review_route_embeds_report(monkeypatch, tmp_path):
    report = tmp_path / "report.html"
    report.write_text(
        "<html><head><style>.case{display:block}</style></head>"
        "<body><main><h1>重复标题</h1><section class='case'>"
        "真实昵称鲁西西 · 身份无法确认 · 完整上下文</section></main></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(admin, "YOUZI_IDENTITY_REVIEW", report)

    page = admin.benchmark_youzi_identity_review().body.decode("utf-8")

    assert "wechat-twin" in page
    assert "真实昵称鲁西西" in page
    assert "身份无法确认" in page
    assert "完整上下文" in page
    assert "重复标题" not in page


def test_strict_historical_judge_route_embeds_report(monkeypatch, tmp_path):
    report = tmp_path / "index.html"
    report.write_text(
        "<html><head><style>.summary{display:grid}</style></head>"
        "<body><main><h1>重复标题</h1><section class='verdict'>"
        "只能辅助诊断</section><article>完整上下文</article></main></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(admin, "STRICT_HISTORICAL_JUDGE_REVIEW", report)

    page = admin.benchmark_strict_historical_judge().body.decode("utf-8")

    assert "wechat-twin" in page
    assert "只能辅助诊断" in page
    assert "完整上下文" in page
    assert "重复标题" not in page


def test_iteration_ledger_separates_real_iterations_from_preparation():
    page = admin.benchmark_judge_iteration_ledger().body.decode("utf-8")

    for expected in ("Bot 真实能力迭代", "Judge 真实能力迭代", "2 轮", "4 轮", "83/100", "不计入能力轮次"):
        assert expected in page
    assert "/benchmark/judge/strict-historical" in page


def test_strict_historical_judge_case_route_shows_complete_call(monkeypatch, tmp_path):
    report = tmp_path / "index.html"
    report.write_text("<html></html>", encoding="utf-8")
    _write(tmp_path, "generation_results.json", {"cases": [{
        "case_id": "case_1", "chat_name": "共同富裕群", "generation": {
            "trace": {"user_prompt": "C0 完整提示词", "tool_calls": []}
        }, "validity": {"formally_scorable": True},
    }]})
    _write(tmp_path, "judge_results.json", {"cases": [{
        "case_id": "case_1", "selected_option": "B", "correct": False,
        "confidence": 0.9, "request_messages": [{"role": "user", "content": "Judge 完整提示词"}],
        "raw_response": "Judge 原始响应", "tool_events": [],
        "subprocess_audit": {"ephemeral": True, "tool_call_count": 0},
    }]})
    monkeypatch.setattr(admin, "STRICT_HISTORICAL_JUDGE_REVIEW", report)

    page = admin.benchmark_strict_historical_judge_case("case_1").body.decode("utf-8")

    for expected in ("共同富裕群", "Judge 完整提示词", "Judge 原始响应", "C0 完整提示词", "误判"):
        assert expected in page


def test_strict_historical_judge_case_route_supports_fair_result_names(monkeypatch, tmp_path):
    report = tmp_path / "fair_index.html"
    report.write_text("<html></html>", encoding="utf-8")
    _write(tmp_path, "fair_generation_results.json", {"cases": [{
        "case_id": "source_case", "internal_id": "fair_case", "chat_name": "柚子群2",
        "generation": {"trace": {"user_prompt": "公平 C0 提示词"}},
        "validity": {"formally_scorable": True},
    }]})
    _write(tmp_path, "fair_judge_results.json", {"cases": [{
        "case_id": "fair_case", "selected_option": "A", "correct": True,
        "confidence": 0.7, "request_messages": [{"role": "user", "content": "公平 Judge 提示词"}],
        "raw_response": "公平 Judge 响应", "tool_events": [], "subprocess_audit": {"returncode": 0},
    }]})
    monkeypatch.setattr(admin, "STRICT_HISTORICAL_JUDGE_REVIEW", report)

    page = admin.benchmark_strict_historical_judge_case("fair_case").body.decode("utf-8")

    for expected in ("柚子群2", "公平 C0 提示词", "公平 Judge 提示词", "公平 Judge 响应"):
        assert expected in page


def test_judge_audit_renders_complete_evidence_without_raw_json(monkeypatch, tmp_path):
    monkeypatch.setattr(admin, "JUDGE_PRIVATE_DIR", tmp_path)
    blind = {
        "batch_id": "batch_test",
        "instruction": "匿名判断 <真人>",
        "cases": [{
            "case_id": "public_case",
            "relationship": "group",
            "context_original": ["<history>完整上下文</history>", "<unread>问题</unread>"],
            "option_A": ["我老家泰州的，淮扬菜"],
            "option_B": ["真人原话"],
        }],
    }
    _write(tmp_path, "human_ai_judge_blind.json", blind)
    _write(tmp_path, "human_ai_judge_answers.json", {
        "blind_sha256": "blind_hash",
        "cases": [{"case_id": "public_case", "source_case_id": "source_case", "human_option": "B"}],
    })
    _write(tmp_path, "human_ai_judge_predictions.json", {
        "cases": [{"case_id": "public_case", "human_option": "B", "confidence": 0.78, "reason": "略显刻意"}],
    })
    _write(tmp_path, "human_ai_judge_result.json", {"correct": 1, "total": 1})
    _write(tmp_path, "human_ai_judge_replay.json", {"cases": [{
        "case_id": "source_case",
        "c0_trace": {
            "duration_seconds": 1.2,
            "input_metadata": {"history_count": 3},
            "generation_trace": {
                "system_prompt": "系统提示词",
                "user_prompt": "完整生产输入；记忆：籍贯：江苏泰州",
                "few_shot_content": "同聊天示例",
                "tools_context": "工具上下文",
                "loaded_skills": ["group_banter"],
                "skill_content": "技能原文",
                "tool_calls": [],
                "llm_calls": [{"response_preview": "C0 原始输出"}],
            },
        },
    }]})
    (tmp_path / "human_ai_judge_events.jsonl").write_text(
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 12}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    body = admin._judge_audit_html()

    for expected in (
        "根据执行时版本代码与 blind 文件重建",
        "完整上下文",
        "我老家泰州的，淮扬菜",
        "真人原话",
        "略显刻意",
        "记忆：籍贯：江苏泰州",
        "完整生产输入",
        "同聊天示例",
        "工具上下文",
        "技能原文",
        "C0 原始输出",
        "input_tokens",
        "泰州事实核查",
    ):
        assert expected in body
    assert "[{" not in body
    assert "is_badcase" not in body


def test_judge_audit_prefers_invocation_capture(monkeypatch, tmp_path):
    monkeypatch.setattr(admin, "JUDGE_PRIVATE_DIR", tmp_path)
    _write(tmp_path, "human_ai_judge_blind.json", {"batch_id": "batch", "cases": []})
    for name in ("answers", "predictions", "result", "replay"):
        _write(tmp_path, f"human_ai_judge_{name}.json", {"cases": []})
    _write(tmp_path, "human_ai_judge_audit.json", {
        "prompt": "captured prompt",
        "prompt_sha256": "captured hash",
        "argv": ["codex", "exec"],
        "events": [{"type": "turn.completed"}],
        "tool_call_count": 0,
    })

    body = admin._judge_audit_html()

    assert "调用时原始捕获" in body
    assert "captured prompt" in body
    assert "captured hash" in body


def test_judge_audit_renders_revised_balanced_run(monkeypatch, tmp_path):
    monkeypatch.setattr(admin, "JUDGE_PRIVATE_DIR", tmp_path)
    case = {
        "case_id": "public_revised",
        "relationship": "private",
        "context_original": [
            '<history><message index="1" role="other" type="text">对方：前一句</message>'
            '<message index="2" role="self" type="text">我：上一句</message></history>',
            '<unread><message index="1" role="other" type="text">对方：最后一句</message></unread>',
        ],
        "option_A": ["C0 回复"],
        "option_B": ["真人回复"],
    }
    _write(tmp_path, "human_ai_judge_revised_blind.json", {
        "batch_id": "revised",
        "reference_sha256": "reference hash",
        "cases": [case],
    })
    _write(tmp_path, "human_ai_judge_revised_answers.json", {
        "blind_sha256": "blind hash",
        "cases": [{"case_id": "public_revised", "source_case_id": "source_revised", "human_option": "B"}],
    })
    _write(tmp_path, "human_ai_judge_revised_predictions.json", {
        "blind_sha256": "blind hash",
        "cases": [{"case_id": "public_revised", "human_option": "A", "confidence": 0.7, "reason": "误判理由"}],
    })
    _write(tmp_path, "human_ai_judge_revised_result.json", {
        "correct": 0,
        "total": 1,
        "surface_baselines": {"rules": {"fewer_punctuation": {"correct": 1, "decided": 1, "accuracy": 1.0}}},
    })
    _write(tmp_path, "human_ai_judge_revised_replay.json", {
        "generator_label": "protected C0",
        "cases": [{"case_id": "source_revised", "c0_trace": {"generation_trace": {"user_prompt": "C0 完整输入"}}}],
    })
    _write(tmp_path, "human_ai_judge_revised_reference.json", {
        "reference_sha256": "reference hash",
        "human_before": "2026-04-01",
        "facts": [{"fact": "本人老家是江苏泰州"}],
        "isolation": {"whole_test_chats_excluded": True},
        "examples": [{"relationship": "private", "context": ["参考上下文"], "human_reply": ["参考真人回复"]}],
    })
    _write(tmp_path, "human_ai_judge_revised_audit.json", {
        "prompt": "完整 Judge prompt",
        "prompt_sha256": "prompt hash",
        "blind_sha256": "blind hash",
        "reference_sha256": "reference hash",
        "argv": ["codex", "exec", "--ephemeral"],
        "events": [{"type": "turn.completed", "usage": {"input_tokens": 99}}],
        "stdout": "原始 stdout",
        "stderr": "原始 stderr",
        "last_message": "原始 last message",
        "tool_call_count": 0,
        "event_parse_errors": [],
    })

    body = admin._judge_audit_html()

    for expected in (
        "查看 Judge 原始提示词",
        "30 道盲测题",
        "最近的聊天",
        "对方最后一句",
        "对方：前一句",
        "我：上一句",
        "对方：最后一句",
        "回复 A",
        "回复 B",
        "Judge 选择",
        "完整 Judge prompt",
        "原始 stdout",
        "原始 stderr",
        "C0 回复",
        "真人回复",
        "误判理由",
        "C0 完整输入",
        "参考真人回复",
        "本次观察到 0 次 Judge 工具调用",
    ):
        assert expected in body
    assert "旧 6 题批次（已作废）" in body
    assert "[{" not in body


def test_judge_audit_prefers_independent_api_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(admin, "JUDGE_PRIVATE_DIR", tmp_path)
    case = {
        "case_id": "api_case",
        "context_original": [
            '<conversation_history><message role="self" type="text">我：前一句</message></conversation_history>',
            '<messages_to_reply><message role="other" type="text">对方：问题</message></messages_to_reply>',
        ],
        "option_A": ["C0 回复"],
        "option_B": ["真人回复"],
    }
    _write(tmp_path, "human_ai_judge_revised_blind.json", {"cases": [case]})
    _write(tmp_path, "human_ai_judge_revised_answers.json", {
        "cases": [{"case_id": "api_case", "human_option": "B"}],
    })
    _write(tmp_path, "human_ai_judge_api_independent_predictions.json", {
        "cases": [{"case_id": "api_case", "human_option": "A", "reason": "API 误判理由"}],
    })
    _write(tmp_path, "human_ai_judge_api_independent_result.json", {"correct": 0, "total": 1})
    _write(tmp_path, "human_ai_judge_api_independent_audit.json", {
        "isolation": {
            "codex_used": False, "one_api_request_per_case": True, "new_client_per_case": True,
            "shared_conversation_state": False, "answer_key_loaded_during_judging": False,
            "tools_available": False,
        },
        "calls": [{
            "case_id": "api_case", "recorded_at_utc": "2026-08-17T00:00:00Z",
            "transport": "direct_https_api", "conversation_state": "none",
            "endpoint": "https://example.test/v1/messages",
            "headers": {"x-api-key": "[REDACTED]"},
            "request_body": {"model": "deepseek-v4-flash", "system": "独立 system prompt", "messages": [{
                "role": "user", "content": [{"type": "text", "text": "仅当前题 user prompt"}],
            }]},
            "request_sha256": "request hash", "other_test_case_ids_in_request": [],
            "answer_fields_in_request": [], "tools_sent": False, "status_code": 200,
            "elapsed_ms": 1234, "resolved_model": "deepseek-v4-flash-resolved",
            "response": {"content": [{"type": "text", "text": "response text"}]},
            "parsed_prediction": {"human_option": "A", "reason": "API 误判理由"},
        }],
    })

    body = admin._judge_audit_html()

    for expected in (
        "这轮 Judge 未通过", "输入含有 C0 约束线索", "没有使用 Codex", "最近的聊天",
        "对方最后一句", "C0 回复", "真人回复", "API 误判理由", "独立 system prompt",
        "仅当前题 user prompt", "&#91;REDACTED]", "deepseek-v4-flash-resolved", "response text",
        "旧 27/30 批次（已作废）",
    ):
        assert expected in body
    assert "[{" not in body


def test_judge_audit_prefers_boundary_candidate_with_full_context(monkeypatch, tmp_path):
    monkeypatch.setattr(admin, "JUDGE_PRIVATE_DIR", tmp_path)
    base = tmp_path / "human_ai_judge_benchmark_v2"
    base.mkdir()
    case = {
        "case_id": "case_boundary",
        "relationship": "private",
        "context_original": [
            '<conversation_history><message index="1" role="other">第一句</message>'
            '<message index="2" role="self">第二句</message>'
            '<message index="3" role="other">第三句</message></conversation_history>',
            '<messages_to_reply><message index="4" role="other">最后问题</message></messages_to_reply>',
        ],
        "option_A": ["真人原回复"],
        "option_B": ["C0 原回复一", "C0 原回复二"],
    }
    _write(base, "final_blind.json", {"cases": [case]})
    _write(base, "final_answers.json", {"cases": [{"case_id": "case_boundary", "human_option": "A"}]})
    for name, choice in (
        ("final_v2_codex_cli_canonical_comparison.json", "B"),
        ("final_v3_codex_cli_canonical_comparison.json", "A"),
    ):
        _write(base, name, {"cases": [{"case_id": "case_boundary", "human_option": choice}]})
    _write(base, "final_v2_boundary_candidate_codex_cli_canonical_comparison.json", {
        "cases": [{
            "case_id": "case_boundary", "human_option": "A", "confidence": 0.61,
            "reason": "两项都能承接，只有一项较弱的本人证据",
        }],
    })
    _write(base, "final_v2_boundary_candidate_codex_cli_canonical_comparison_audit.json", {
        "provider": "codex-cli", "model": "codex", "presentation": "canonical",
        "profile": {"instructions": "候选 Judge 完整规则"},
        "calls": [{
            "case_id": "case_boundary", "recorded_at_utc": "2026-08-18T00:00:00Z",
            "request_messages": [
                {"role": "system", "content": "完整 system prompt"},
                {"role": "user", "content": "完整单题 user prompt"},
            ],
            "request_sha256": "unique request hash", "answer_fields_in_request": [],
            "other_test_case_ids_in_request": [], "tools_configured": True,
            "tool_events": [], "elapsed_ms": 1234,
            "raw_response": '{"human_option":"A"}', "error": None,
            "response": {
                "argv": ["codex", "exec", "--ephemeral"], "returncode": 0,
                "stdout": "原始 stdout", "stderr": "原始 stderr",
                "events": [{"type": "turn.completed"}], "event_parse_errors": [],
                "prompt_transport": "single input",
            },
        }],
    })

    body = admin._judge_audit_html()

    for expected in (
        "边界候选 Judge", "完整聊天上下文（3 条）", "第一句", "第二句", "第三句",
        "最后问题", "真人原回复", "C0 原回复一", "C0 原回复二", "真人原回复：A",
        "候选 Judge 选择：<b>A</b>", "V2 选 B", "V3 选 A", "完整 system prompt",
        "完整单题 user prompt", "unique request hash", "原始 stdout", "原始 stderr",
        "候选 Judge 完整规则", "旧 30 题 API 批次（已作废）",
    ):
        assert expected in body
    assert "[{" not in body
    assert "is_badcase" not in body


def test_judge_audit_puts_all_v2_misclassifications_first(monkeypatch, tmp_path):
    monkeypatch.setattr(admin, "JUDGE_PRIVATE_DIR", tmp_path)
    base = tmp_path / "human_ai_judge_benchmark_v2"
    base.mkdir()
    cases = []
    answers = []
    predictions = []
    calls = []
    for index in range(14):
        case_id = f"wrong_{index:02d}"
        cases.append({
            "case_id": case_id,
            "context_original": [
                f'<conversation_history><message index="1" role="other">上下文 {index}</message></conversation_history>',
                f'<messages_to_reply><message index="2" role="other">目标消息 {index}</message></messages_to_reply>',
            ],
            "option_A": [f"真人回复 {index}"],
            "option_B": [f"对照回复 {index}"],
        })
        answers.append({"case_id": case_id, "human_option": "A"})
        predictions.append({
            "case_id": case_id, "human_option": "B",
            "confidence": 0.99 - index / 100, "reason": f"V2 理由 {index}",
        })
        calls.append({
            "case_id": case_id,
            "request_messages": [
                {"role": "system", "content": f"system prompt {index}"},
                {"role": "user", "content": f"user prompt {index}"},
            ],
            "request_sha256": f"hash-{index}", "tool_events": [],
            "response": {"stdout": f"stdout {index}"},
        })
    _write(base, "final_blind.json", {"cases": cases})
    _write(base, "final_answers.json", {"cases": answers})
    _write(base, "final_v2_codex_cli_canonical_comparison.json", {"cases": predictions})
    _write(base, "final_v2_codex_cli_canonical_comparison_audit.json", {"calls": calls})
    _write(base, "final_v2_boundary_candidate_codex_cli_canonical_comparison.json", {
        "cases": predictions,
    })
    _write(base, "final_v2_boundary_candidate_codex_cli_canonical_comparison_audit.json", {
        "calls": calls,
    })

    body = admin._judge_audit_html()

    assert body.count('data-v2-misclassification="true"') == 14
    assert body.count("重点误判") == 4
    assert "4 个高置信误判" in body
    assert "其余 10 个误判" in body
    assert "上下文 0" in body
    assert "目标消息 0" in body
    assert "真人回复 0" in body
    assert "对照回复 0" in body
    assert "真人原回复：A" in body
    assert "V2 选择：<b>B</b>" in body
    assert "V2 原始理由：</b>V2 理由 0" in body
    assert "system prompt 0" in body
    assert "user prompt 0" in body
    assert "还不足以稳定、单独决定回复版本胜负" in body
    assert body.index("Codex V2 的 14 个误判") < body.index("展开 100 题完整总览")
