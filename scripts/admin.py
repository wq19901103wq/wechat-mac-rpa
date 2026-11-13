#!/usr/bin/env python3
"""wechat-twin Admin — 统一开发者后台"""
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import hashlib
import html
import json as _json
import subprocess  # nosec B404
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("scripts.admin")


from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from src.badcase.case_db import get_db

WORK_DIR = str(Path(__file__).parent.parent)
KIMI_BIN = os.environ.get("KIMI_BIN", str(Path.home() / ".local" / "bin" / "kimi"))

app = FastAPI(title="wechat-twin Admin")

_PERSONA_FEW_SHOT_MARKER = re.compile(r"【persona few-shot 正文已省略；ids=([^】]+)】")


def _expand_persona_few_shots(text: str) -> str:
    if not _PERSONA_FEW_SHOT_MARKER.search(text):
        return text
    configured_path = os.environ.get("PERSONA_FEW_SHOT_PATH", "")
    if not configured_path:
        env_path = Path(WORK_DIR) / ".env"
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("PERSONA_FEW_SHOT_PATH="):
                    configured_path = line.split("=", 1)[1].strip()
                    break
        except OSError:
            pass
    examples_path = Path(
        configured_path
        or Path(WORK_DIR) / "data" / "few_shot" / "persona_examples.jsonl"
    )
    if not examples_path.is_absolute():
        examples_path = Path(WORK_DIR) / examples_path
    try:
        rows = {}
        for line in examples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                rows[str(row.get("id", ""))] = row
    except (OSError, _json.JSONDecodeError) as exc:
        _logger.warning("读取 persona few-shot 示例失败: %s", exc)
        return text

    def _replacement(match: re.Match) -> str:
        ids = [item for item in match.group(1).split(",") if item]
        parts = [f"【persona few-shot 正文（从当前示例库展开；ids={','.join(ids)}）】"]
        for row_id in ids:
            row = rows.get(row_id)
            if row is None:
                parts.append(f"示例 {row_id}：当前示例库中不存在")
                continue
            parts.append(f"示例 {row_id}：")
            parts.extend(f"对方：{item}" for item in row.get("context", []))
            parts.extend(f"本人：{item}" for item in row.get("reply", []))
        return "\n".join(parts)

    return _PERSONA_FEW_SHOT_MARKER.sub(_replacement, text)

# 截图列表页缓存（模块级，首次扫描后复用）
_screenshot_cache: Optional[List[Dict[str, Any]]] = None

HEADER = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>wechat-twin Admin</title>
<style>:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff}
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);padding:0;display:flex;min-height:100vh}
nav{width:200px;background:var(--card);border-right:1px solid var(--border);padding:16px 0;flex-shrink:0}
nav a{display:block;padding:8px 20px;color:var(--text);text-decoration:none;font-size:13px;transition:background .15s}
nav a:hover{background:rgba(255,255,255,.05)}nav a.active{color:var(--blue);background:rgba(88,166,255,.1)}
main{flex:1;padding:24px;overflow:auto}h1{font-size:20px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}th{color:var(--muted);font-weight:600;font-size:11px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:12px}
.metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin:12px 0}
.metric{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center}
.metric .value{font-size:28px;font-weight:700}.metric .label{font-size:11px;color:var(--muted);margin-top:4px}
/* Lightbox */
#lightbox{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.9);z-index:9999;cursor:zoom-out;align-items:center;justify-content:center}
#lightbox img{max-width:95vw;max-height:95vh;object-fit:contain}
#lightbox.show{display:flex}
/* diff styles - full scrollable side-by-side */
.diff-container{display:flex;gap:0;margin-top:4px;max-height:65vh;overflow:auto;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--border);border-radius:6px;background:var(--card)}
.diff-container::-webkit-scrollbar{width:8px;height:8px}
.diff-container::-webkit-scrollbar-track{background:transparent}
.diff-container::-webkit-scrollbar-thumb{background:rgba(255,255,255,.15);border-radius:4px}
.diff-container::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.25)}
.diff-col{flex:1;min-width:0;display:flex;flex-direction:column;overflow:hidden}
.diff-col+.diff-col{border-left:1px solid var(--border)}
.diff-header{position:sticky;top:0;z-index:10;display:flex;align-items:center;padding:6px 10px;font-size:12px;font-weight:600;font-family:-apple-system,BlinkMacSystemFont,sans-serif;border-bottom:1px solid var(--border);background:var(--card)}
.diff-header.control{color:var(--red);border-left:3px solid var(--red)}
.diff-header.exp{color:var(--green);border-left:3px solid var(--green)}
.diff-body{overflow-y:auto;max-height:calc(65vh - 32px)}
.diff-body::-webkit-scrollbar{width:6px}
.diff-body::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:3px}
.ln{color:var(--muted);text-align:right;padding:4px 6px;width:44px;font-size:10px;user-select:none;background:#1c2128;border-right:1px solid var(--border);vertical-align:top;white-space:nowrap}
.eq{color:#adbac7;padding:4px 6px;background:transparent;vertical-align:top;line-height:1.5}
.del{background:rgba(248,81,73,.08);color:#ff9d9d;padding:4px 6px;border-left:3px solid #f85149;vertical-align:top;line-height:1.5}
.add{background:rgba(63,185,80,.08);color:#7ee787;padding:4px 6px;border-left:3px solid #3fb950;vertical-align:top;line-height:1.5}
.empty{background:rgba(255,255,255,.02);color:var(--muted);padding:4px 6px;vertical-align:top;line-height:1.5}
.skip{color:var(--muted);text-align:center;padding:8px;background:rgba(255,255,255,.02);font-style:italic;font-size:10px}
@media(max-width:700px){
body{display:block}
body>nav{display:none}
body>main{width:100%;padding:10px;overflow:visible}
}
</style></head><body>
<div id="lightbox" onclick="this.classList.remove('show')"><img id="lightbox-img" src=""></div>
<nav>
<div style="padding:12px 20px;font-weight:700;font-size:15px;margin-bottom:8px">wechat-twin</div>
"""

FOOTER = """</main>
<script>
document.addEventListener('dblclick',function(e){
  var img=e.target.closest('img');
  if(!img || img.id==='lightbox-img')return;
  var lb=document.getElementById('lightbox');
  document.getElementById('lightbox-img').src=img.src;
  lb.classList.add('show');
});
</script>
</body></html>"""

def _page(title: str, content: str, active: str = "") -> str:
    nav = HEADER
    for href, label in [("/", "📊 Dashboard"), ("/ticks", "🔍 Tick"), ("/gt", "🏷️ GT"), ("/review", "🧑‍⚖️ 审核"), ("/screenshots", "📸 截图OCR"), ("/benchmark/judge", "📊 Judge"), ("/benchmark/judge/iteration-ledger", "📒 迭代账单"), ("/benchmark/judge/strict-historical", "✅ Judge验收"), ("/benchmark/judge/group", "💬 群聊Judge"), ("/benchmark/judge/group/youzi-identity", "👤 柚子身份"), ("/benchmark/reply", "🤖 回复"), ("/benchmark/reply/current-c0", "🆚 C0×真人"), ("/experiments", "🧪 实验"), ("/code-audit", "🐛 审计"), ("/wiki-review", "🧠 Wiki审核")]:
        cls = ' class="active"' if href == active else ""
        nav += f'<a href="{href}"{cls}>{label}</a>'
    nav += "</nav><main>"
    return nav + f"<h1>{title}</h1>" + content + FOOTER


@app.get("/", response_class=HTMLResponse)
def dashboard():
    db = get_db()
    today = __import__('datetime').datetime.now().strftime("%Y-%m-%d")
    conn = db._get_conn()
    total = conn.execute("SELECT COUNT(*) FROM tick_log WHERE date(created_at)=?", (today,)).fetchone()[0]
    replied = conn.execute("SELECT COUNT(*) FROM tick_log WHERE date(created_at)=? AND should_reply=1", (today,)).fetchone()[0]
    avg_score = conn.execute("SELECT ROUND(AVG(judge_score),1) FROM tick_log WHERE date(created_at)=? AND judge_score>0", (today,)).fetchone()[0] or 0
    skipped = conn.execute("SELECT COUNT(*) FROM tick_log WHERE date(created_at)=? AND skip_reason IS NOT NULL", (today,)).fetchone()[0]
    conn.close()

    content = f"""
    <div class="metrics">
      <div class="metric"><div class="value" style="color:var(--blue)">{total}</div><div class="label">今日 Tick</div></div>
      <div class="metric"><div class="value" style="color:var(--green)">{replied}</div><div class="label">回复数</div></div>
      <div class="metric"><div class="value" style="color:var(--purple)">{avg_score}</div><div class="label">平均 Judge 分</div></div>
      <div class="metric"><div class="value" style="color:var(--muted)">{skipped}</div><div class="label">跳过数</div></div>
      <div class="metric"><div class="value" style="color:var(--yellow)">{round(skipped*100/max(total,1))}%</div><div class="label">跳过率</div></div>
    </div>
    <p style="color:var(--muted);font-size:13px">数据来自 tick_log 表 · 刷新页面更新</p>"""
    return HTMLResponse(_page("Dashboard", content, "/"))


@app.get("/ticks", response_class=HTMLResponse)
def tick_list(page: int = Query(1), filter: str = Query("all")):
    db = get_db()
    conn = db._get_conn()
    offset = (page - 1) * 20
    if filter == "skipped":
        rows = conn.execute(
            "SELECT id, session_id, tick_id, chat_name, messages_count, new_messages_count, should_reply, send_success, skip_reason, judge_score, human_is_badcase, human_badcase_type, replies_sent_json, raw_response, duration_ms, created_at FROM tick_log WHERE skip_reason IS NOT NULL ORDER BY created_at DESC, tick_id DESC LIMIT ? OFFSET ?",
            (20, offset),
        ).fetchall()
    elif filter == "replied":
        rows = conn.execute(
            "SELECT id, session_id, tick_id, chat_name, messages_count, new_messages_count, should_reply, send_success, skip_reason, judge_score, human_is_badcase, human_badcase_type, replies_sent_json, raw_response, duration_ms, created_at FROM tick_log WHERE should_reply=1 ORDER BY created_at DESC, tick_id DESC LIMIT ? OFFSET ?",
            (20, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, session_id, tick_id, chat_name, messages_count, new_messages_count, should_reply, send_success, skip_reason, judge_score, human_is_badcase, human_badcase_type, replies_sent_json, raw_response, duration_ms, created_at FROM tick_log ORDER BY created_at DESC, tick_id DESC LIMIT ? OFFSET ?",
            (20, offset),
        ).fetchall()
    conn.close()

    rows_html = ""
    for r in rows:
        rp_raw = r["raw_response"] or ""
        if r["skip_reason"]:
            status = "⏭️跳过"
        elif r["send_success"]:
            status = "✅已发送"
        elif r["replies_sent_json"] and r["replies_sent_json"] not in ("[]", ""):
            status = "⚠️未发送（有草稿）"
        elif r["should_reply"]:
            if rp_raw.startswith("[空回复"):
                status = "⚠️模型空回复"
            elif '"replies"' in rp_raw:
                status = "⏭️决定不回复"
            else:
                status = "⚠️未生成回复"
        else:
            status = "⏭️无消息"
        llm_score = f'{r["judge_score"]:.0f}' if r['judge_score'] else "-"
        human = ""
        if r['human_is_badcase'] == 1:
            human = f'❌{r["human_badcase_type"] or "badcase"}'
        elif r['human_is_badcase'] == 0:
            human = "✅OK"
        reply_preview = ""
        rp = r['replies_sent_json']
        if rp and rp != '[]':
            try:
                import json as _j2
                arr = _j2.loads(rp)
                reply_preview = " | ".join(str(x) for x in (arr if isinstance(arr, list) else []))
            except Exception as e:
                _logger.debug("解析 replies_sent_json 失败: %s", e)
        rows_html += f"""<tr>
          <td><a href="/ticks/{r['id']}" style="color:var(--blue)">{html.escape(r['session_id'] or '')}:#{r['tick_id']}</a></td>
          <td>{html.escape(r['chat_name'] or '-')}</td><td>{r['new_messages_count'] or r['messages_count'] or 0}条</td>
          <td>{html.escape(status)}</td><td style="font-size:11px">{html.escape(reply_preview)}</td><td>{llm_score}</td><td>{html.escape(human)}</td><td>{r['duration_ms'] or 0}ms</td>
          <td style="font-size:11px;color:var(--muted)">{html.escape(r['created_at'] if r['created_at'] else '')}</td></tr>"""

    conn = db._get_conn()
    if filter == "skipped":
        total = conn.execute("SELECT COUNT(*) FROM tick_log WHERE skip_reason IS NOT NULL").fetchone()[0]
    elif filter == "replied":
        total = conn.execute("SELECT COUNT(*) FROM tick_log WHERE should_reply=1").fetchone()[0]
    else:
        total = conn.execute("SELECT COUNT(*) FROM tick_log").fetchone()[0]
    conn.close()
    total_pages = (total + 19) // 20
    content = f"""<p style="margin-bottom:12px"><a href="?filter=all">全部({total})</a> | <a href="?filter=replied">已回复</a> | <a href="?filter=skipped">跳过</a> | <span style="color:var(--muted);font-size:12px">每页20条</span></p>
    <table><tr><th>Tick</th><th>聊天</th><th>消息</th><th>状态</th><th>回复</th><th>LLM</th><th>👤</th><th>耗时</th><th>时间</th></tr>{rows_html}</table>
    <div style="margin-top:12px;font-size:13px">
      <a href="?page={page-1}&filter={html.escape(filter)}" style="color:var(--blue);margin-right:12px" {'hidden' if page<=1 else ''}>← 上一页</a>
      第 {page} 页 / 共 {total_pages} 页
      <a href="?page={page+1}&filter={html.escape(filter)}" style="color:var(--blue);margin-left:12px">下一页 →</a>
    </div>"""
    return HTMLResponse(_page("Tick 查看", content, "/ticks"))


@app.get("/ticks/{id}", response_class=HTMLResponse)
def tick_detail(id: int):
    db = get_db()
    conn = db._get_conn()
    r = conn.execute("SELECT * FROM tick_log WHERE id=?", (id,)).fetchone()
    conn.close()
    if not r:
        return HTMLResponse("<h1>Tick not found</h1>")
    d = dict(r)
    replies = d.get("replies_sent_json", "[]") or "[]"
    # 格式化 Bot 回复
    try:
        import json as _j4
        reply_list = _j4.loads(replies) if replies else []
        replies_display = "<br>".join(f'<span style="background:rgba(88,166,255,.15);padding:2px 8px;border-radius:4px;margin:2px;display:inline-block;font-size:13px">{html.escape(str(r))}</span>' for r in reply_list) if reply_list else html.escape(replies)
    except:
        replies_display = replies
    sp = d.get("system_prompt") or ""
    up = d.get("user_prompt") or ""
    raw = d.get("raw_response") or ""
    tools = d.get("tool_calls_json") or "[]"
    tool_results = d.get("tool_results_json") or "[]"

    # 原有信息卡片
    ms = d.get("duration_ms",0) or 0
    raw_response = d.get("raw_response") or ""
    if d.get("skip_reason"):
        status = "⏭️跳过"
    elif d.get("send_success"):
        status = "✅已发送"
    elif d.get("replies_sent_json") and d.get("replies_sent_json") not in ("[]", ""):
        status = "⚠️未发送（有草稿）"
    elif d.get("should_reply"):
        if raw_response.startswith("[空回复"):
            status = "⚠️模型空回复"
        elif '"replies"' in raw_response:
            status = "⏭️决定不回复"
        else:
            status = "⚠️未生成回复"
    else:
        status = "⏭️无消息"
    content = f"""
    <div class="card"><b>{html.escape(d.get("session_id",""))}:#{d["tick_id"]}</b> — {html.escape(d.get("created_at",""))}</div>
    <div class="card"><b>聊天:</b> {html.escape(d.get("chat_name","?"))} {"(群)" if d.get("is_group") else "(私)"} | <b>状态:</b> {html.escape(status)} | <b>耗时:</b> {ms}ms</div>
    <div class="card"><b>消息:</b> 总{d.get("messages_count",0)}条 新{d.get("new_messages_count",0)}条 | <b>发送:</b> {"OK" if d.get("send_success") else "N/A"}</div>
    """
    # Bot 回复卡片（最终输出），按实际流程放在 Self-Refine 之后
    raw_response = d.get("raw_response") or ""
    if (not replies_display or replies_display == "[]") and raw_response:
        bot_reply_html = f'<div class="card" style="border-left:3px solid var(--blue)"><b>🤖 Bot 回复:</b><br>{replies_display}<div style="margin-top:8px;color:var(--muted);font-size:12px">原始输出摘要：<pre style="white-space:pre-wrap;font-size:10px">{html.escape(raw_response[:500])}</pre></div></div>'
    else:
        bot_reply_html = f'<div class="card" style="border-left:3px solid var(--blue)"><b>🤖 Bot 回复:</b><br>{replies_display}</div>'

    # 完整 LLM 消息流（按 message 顺序：system → user → assistant → tool → assistant(feedback) → ...）
    llm_messages_json = d.get("llm_messages_json") or "[]"
    llm_flow_html = ""
    try:
        msgs = _json.loads(llm_messages_json) if llm_messages_json else []
        if msgs:
            flow_items = []
            for idx, m in enumerate(msgs, 1):
                role = m.get("role", "?")
                msg_content = m.get("content") or ""
                reasoning = m.get("reasoning_content") or ""
                tool_calls = m.get("tool_calls") or []
                tool_call_id = m.get("tool_call_id") or ""
                if role == "system":
                    title = f"#{idx} system"
                    body = f'<pre style="font-size:10px;white-space:pre-wrap">{html.escape(msg_content)}</pre>'
                    color = "var(--blue)"
                elif role == "user":
                    title = f"#{idx} user"
                    msg_content = _expand_persona_few_shots(msg_content)
                    body = f'<pre style="font-size:10px;white-space:pre-wrap">{html.escape(msg_content)}</pre>'
                    color = "var(--green)"
                elif role == "assistant":
                    title = f"#{idx} assistant"
                    parts = []
                    if reasoning:
                        parts.append(f'<div style="color:var(--muted);font-size:11px;margin-bottom:4px">[思考过程]</div><pre style="font-size:10px;white-space:pre-wrap;background:rgba(0,0,0,.15);padding:6px;border-radius:4px">{html.escape(reasoning)}</pre>')
                    if msg_content:
                        parts.append(f'<div style="color:var(--muted);font-size:11px;margin:4px 0">[回复]</div><pre style="font-size:10px;white-space:pre-wrap">{html.escape(msg_content)}</pre>')
                    if tool_calls:
                        tc_html = ""
                        for tc in tool_calls:
                            fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                            tname = fn.get("name") or tc.get("tool_name") or "?"
                            targs = fn.get("arguments") or tc.get("arguments") or ""
                            tc_html += f'<div style="margin:4px 0;padding:6px;background:rgba(255,255,255,.05);border-left:2px solid var(--yellow);border-radius:3px"><b style="color:var(--yellow);font-size:11px">{html.escape(tname)}</b> <span style="color:var(--muted);font-size:10px">{html.escape(str(targs)[:200])}</span></div>'
                        parts.append(f'<div style="color:var(--muted);font-size:11px;margin:4px 0">[工具调用]</div>{tc_html}')
                    if not parts:
                        parts.append('<span style="color:var(--muted);font-size:12px">(空：模型未返回 content / reasoning / tool_calls)</span>')
                    body = "\n".join(parts)
                    color = "var(--purple)"
                elif role == "tool":
                    title = f"#{idx} tool ({html.escape(tool_call_id)})"
                    body = f'<pre style="font-size:10px;white-space:pre-wrap">{html.escape(msg_content)}</pre>'
                    color = "var(--yellow)"
                else:
                    title = f"#{idx} {html.escape(role)}"
                    body = f'<pre style="font-size:10px;white-space:pre-wrap">{html.escape(str(m)[:2000])}</pre>'
                    color = "var(--muted)"
                flow_items.append(f'<div style="margin:8px 0;padding:10px;border-left:3px solid {color};background:rgba(255,255,255,.03);border-radius:4px"><div style="font-size:12px;font-weight:bold;margin-bottom:4px">{title}</div>{body}</div>')
            llm_flow_html = f'<div class="card" style="border-left:3px solid var(--purple)"><b>🧬 LLM 完整消息流（按 message 顺序）</b>{"".join(flow_items)}</div>'
    except Exception as e:
        _logger.warning("渲染 LLM 消息流失败: %s", e)
    content += llm_flow_html

    # Bot 最终回复（作为消息流之后的摘要）
    content += bot_reply_html

    # === 新增：Judge 评分 ===
    judge_dims = ""
    if d.get("judge_dimensions_json"):
        try:
            import json as _j
            dims = _j.loads(d["judge_dimensions_json"])
            for name, dd in dims.items():
                s = int(dd.get("score", 0))
                filled = min(s // 10, 10)
                bar = "█"*filled + "░"*(10-filled)
                judge_dims += f'<div style="margin:2px 0;font-size:11px">{bar} {html.escape(name)}: {html.escape(str(dd.get("score","?")))}/100 — {html.escape(dd.get("comment",""))}</div>'
        except Exception as e:
            _logger.debug("解析 judge_dimensions_json 失败: %s", e)
    js = d.get("judge_score")
    js_str = f"{js:.0f}" if js is not None else "?"
    content += f"""
    <div class="card" style="border-left:3px solid orange">
      <b>LLM Judge:</b> {js_str}/100 | is_badcase: {'是' if d.get('judge_is_badcase') else '否'} | {html.escape(d.get("judge_badcase_type") or "?")}<br>{judge_dims}
    </div>
    """

    # === 新增：人工点评 ===
    human = d.get("human_is_badcase")
    ht = d.get("human_badcase_type", "")
    hn = d.get("human_notes", "")
    checked1 = "checked" if human == 1 else ""
    checked0 = "checked" if human == 0 else ""
    types = [("","--类型--"),("hallucination","幻觉"),("persona_break","人设分裂"),("wrong_fact","事实错误"),("bad_style","风格问题"),("contradiction","前后矛盾"),("other","其他")]
    sel_opts = "".join(f'<option value="{v}" {"selected" if ht==v else ""}>{l}</option>' for v,l in types)
    content += f"""
    <div class="card" style="border-left:3px solid green">
      <b>人工点评:</b>
      <form id="gt-form" style="margin-top:8px">
        <label style="display:block;margin:8px 0"><input type="radio" name="is_badcase" value="1" {checked1}> badcase <input type="radio" name="is_badcase" value="0" {checked0} style="margin-left:12px"> 正常</label>
        <select name="badcase_type" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:4px 8px;border-radius:4px;margin:4px 0">{sel_opts}</select>
        <textarea name="notes" rows="2" placeholder="点评..." style="width:100%;background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:8px;border-radius:4px;font-size:13px;margin:4px 0">{html.escape(hn or "")}</textarea>
        <button type="submit" style="background:#58a6ff;color:#fff;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;font-size:13px">保存</button> <span id="save-status" style="font-size:12px;color:#3fb950"></span>
      </form>
    </div>
    <script>document.getElementById("gt-form").addEventListener("submit",async function(e){{e.preventDefault();var f=e.target;var r=await fetch("/api/gt/{id}",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{is_badcase:f.is_badcase.value==="1",badcase_type:f.badcase_type.value,notes:f.notes.value}})}});document.getElementById("save-status").textContent=(await r.json()).success?"OK":"FAIL";}});</script>
    """
    # lgtm[py/reflected-xss] content 已用 html.escape 转义，待迁移模板引擎
    return HTMLResponse(_page(f"Tick {html.escape(d.get('session_id',''))}:#{d['tick_id']}", content, "/ticks"))


@app.get("/gt", response_class=HTMLResponse)
def gt_list():
    db = get_db()
    conn = db._get_conn()
    # Show ticks where Judge might be wrong: high score but human disagrees, or low score but human says OK
    rows = conn.execute("""SELECT id, session_id, tick_id, chat_name, judge_score, judge_is_badcase, human_is_badcase, human_badcase_type, raw_response
        FROM tick_log WHERE judge_score > 0 AND (human_is_badcase IS NULL OR human_is_badcase != judge_is_badcase)
        ORDER BY created_at DESC, tick_id DESC LIMIT 50""").fetchall()
    conn.close()
    rows_html = ""
    for r in rows:
        j = "✅正常" if r["judge_is_badcase"] == 0 else "❌badcase"
        h = "—" if r["human_is_badcase"] is None else ("✅正常" if r["human_is_badcase"] == 0 else f"❌{r['human_badcase_type']}")
        cls = "" if r["human_is_badcase"] is None else ("style='color:var(--yellow)'" if r["human_is_badcase"] != r["judge_is_badcase"] else "")
        rows_html += f"""<tr {cls}><td><a href="/ticks/{r['id']}" style="color:var(--blue)">{r['session_id'] or ''}:#{r['tick_id']}</a></td>
          <td>{r['chat_name']}</td><td>{r['judge_score']:.0f}</td><td>{j}</td><td>{h}</td>
          <td style="font-size:11px;color:var(--muted)">{(r['raw_response'] or '')}</td></tr>"""

    content = f"""<p style="color:var(--muted);font-size:13px;margin-bottom:12px">标注 Judge 判定可能错误的 tick。点击 tick 进入详情页，底部可设置 GT。</p>
    <table><tr><th>Tick</th><th>聊天</th><th>Judge分</th><th>Judge判</th><th>人工判</th><th>回复</th></tr>{rows_html}</table>"""
    return HTMLResponse(_page("GT 标注", content, "/gt"))


@app.get("/review", response_class=HTMLResponse)
def review_list():
    db = get_db()
    conn = db._get_conn()
    rows = conn.execute("SELECT id, draft_id, chat_name, status, badcase_type, severity, confidence, overall_score, judge_reason FROM cases ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    rows_html = ""
    for r in rows:
        rows_html += f"""<tr><td><a href="/review/{r['draft_id']}" style="color:var(--blue)">{r['draft_id']}</a></td>
          <td>{r['chat_name']}</td><td>{r['status']}</td><td>{r['badcase_type']}</td>
          <td>{r['confidence']:.0%}</td><td>{r['overall_score']:.0f}</td></tr>"""

    return HTMLResponse(_page("Badcase 审核", f"""<table><tr><th>Draft</th><th>聊天</th><th>状态</th><th>类型</th><th>置信</th><th>分</th></tr>{rows_html}</table>""", "/review"))


@app.post("/api/gt/{id}")
async def save_gt(id: int, request: Request):
    body = await request.json()
    db = get_db()
    conn = db._get_conn()
    conn.execute("""UPDATE tick_log SET
        human_is_badcase=?, human_badcase_type=?, human_notes=?,
        human_labeled_at=datetime('now','localtime')
        WHERE id=?""",
        (1 if body.get("is_badcase") else 0,
         body.get("badcase_type", ""),
         body.get("notes", ""),
         id))
    conn.commit()
    conn.close()
    return JSONResponse({"success": True})


# ── 截图 + OCR 查看 ──

DEBUG_DIR = Path(__file__).parent.parent / "data" / "debug"
SCREENSHOTS_DIR = Path(__file__).parent.parent / "data" / "screenshots"

def _safe_path(base: Path, rel: str) -> Path | None:
    """只允许 base 目录下的纯文件名，防止目录遍历。"""
    import re
    if not rel or not re.fullmatch(r"[a-zA-Z0-9_.\-]+", rel):
        return None
    # lgtm[py/path-injection] rel 已用正则限制为纯文件名
    target = base / rel
    if target.is_file():
        return target
    return None


@app.get("/api/screenshot-image/{filename:path}")
def serve_screenshot(filename: str):
    """返回截图图片文件"""
    # 1. data/screenshots/（只允许该目录下的文件）
    path = _safe_path(SCREENSHOTS_DIR, filename)
    # lgtm[py/path-injection] path 来自 _safe_path，已验证为允许目录下的纯文件名
    if path and path.exists():
        return FileResponse(str(path), media_type="image/png")
    # 2. 系统临时目录
    tmp_path = _safe_path(Path(tempfile.gettempdir()), filename)
    # lgtm[py/path-injection] tmp_path 来自 _safe_path，已验证为临时目录下的纯文件名
    if tmp_path and tmp_path.exists():
        return FileResponse(str(tmp_path), media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/screenshots", response_class=HTMLResponse)
def screenshots_list(page: int = Query(1, ge=1), limit: int = Query(20, ge=5, le=100),
                     filter: str = Query("all"), refresh: bool = Query(False)):
    """截图 + OCR 列表页 — 从 debug JSON 目录查。首次加载扫描全部元数据并缓存，后续按真实总数分页。"""
    import time as _time
    _t0 = _time.time()
    # 扫描并解析所有 debug JSON 的元数据，缓存到模块级变量，重启前不重复扫
    global _screenshot_cache
    if refresh:
        _screenshot_cache = None
    if _screenshot_cache is None:
        files = [f for f in os.listdir(str(DEBUG_DIR)) if f.startswith("tick_") and f.endswith(".json")]
        files.sort(reverse=True)
        items = []
        for fname in files:
            try:
                f = DEBUG_DIR / fname
                dbg = _json.loads(f.read_text(encoding="utf-8"))
                has_api = bool(dbg.get("api_prompt"))
                tid = int(fname.rsplit("_", 1)[-1].split(".")[0])
                sp = dbg.get("screenshot_path", "") or dbg.get("perception_screenshot_path", "") or ""
                fpath = Path(sp) if sp and Path(sp).exists() else None
                chat_name = dbg.get("perception_chat_name", "") or dbg.get("bot_chat_name", "") or "-"
                ocr_count = len(dbg.get("ocr_elements", []))
                layout_msgs = len(dbg.get("extraction_messages", []))
                chat_items = len(dbg.get("layout_chat_list_nicknames", []))
                msg_new = dbg.get("bot_new_messages_count", 0)
                ocr_summary = f"OCR:{ocr_count} msg:{layout_msgs} chat:{chat_items}"
                if msg_new:
                    ocr_summary += f" 新:{msg_new}"
                items.append({
                    "tid": tid,
                    "sp": sp,
                    "chat_name": chat_name,
                    "ocr_summary": ocr_summary,
                    "has_api": has_api,
                    "fpath_exists": fpath is not None,
                })
            except Exception as e:
                _logger.debug("读取截图 debug 文件 %s 失败: %s", fname, e)
                continue
        _screenshot_cache = items

    if filter == "api":
        filtered = [it for it in _screenshot_cache if it["has_api"]]
    elif filter == "skip":
        filtered = [it for it in _screenshot_cache if not it["has_api"]]
    else:
        filtered = _screenshot_cache

    total = len(filtered)
    offset = (page - 1) * limit
    page_items = filtered[offset:offset + limit]

    rows_html = ""
    for it in page_items:
        tid = it["tid"]
        sp = it["sp"]
        chat_name = it["chat_name"]
        ocr_summary = it["ocr_summary"]
        has_api = it["has_api"]
        fpath_exists = it["fpath_exists"]
        fname = Path(sp).name if sp else ""
        img_tag = '<img src="/api/screenshot-image/' + fname + '" style="max-width:240px;max-height:160px;border-radius:4px;border:1px solid var(--border)" loading="lazy" onerror="this.style.display=\'none\'">' if fname and fpath_exists else '<span style="color:var(--muted)">—</span>'
        api_info = '<span style="color:var(--green)">API</span>' if has_api else '<span style="color:var(--yellow)">跳过</span>'

        rows_html += "<tr>"
        rows_html += '<td><a href="/screenshots/' + str(tid) + '" style="color:var(--blue)">#' + str(tid) + '</a></td>'
        rows_html += "<td>-</td>"
        rows_html += "<td>" + html.escape(chat_name) + "</td>"
        rows_html += "<td>" + img_tag + "</td>"
        rows_html += '<td style="font-size:12px">' + html.escape(ocr_summary) + '</td>'
        rows_html += "<td>" + api_info + "</td>"
        rows_html += "</tr>"

    safe_filter = html.escape(filter)
    total_pages = (total + limit - 1) // limit
    prev_hidden = 'hidden' if page <= 1 else ''
    next_hidden = 'hidden' if page >= total_pages else ''
    content = f"""
    <p style="color:var(--muted);font-size:13px;margin-bottom:12px">截图、OCR 识别结果、多模态 API — {total} 条
    <a href="?" style="color:var(--blue);margin-left:8px">{'<b>[全部]</b>' if filter=='all' else '[全部]'}</a>
    <a href="?filter=api" style="color:var(--blue);margin-left:4px">{'<b>[API]</b>' if filter=='api' else '[API]'}</a>
    <a href="?filter=skip" style="color:var(--blue);margin-left:4px">{'<b>[跳过]</b>' if filter=='skip' else '[跳过]'}</a>
    <a href="?refresh=1&filter={safe_filter}" style="color:var(--yellow);margin-left:12px;font-size:12px">🔄 刷新缓存</a>
    </p>
    <table>
      <tr><th>Tick</th><th>时间</th><th>聊天</th><th>截图</th><th>OCR/Layout</th><th>API</th></tr>
      {rows_html}
    </table>
    <div style="margin-top:12px;font-size:13px">
      <a href="?page={page-1}&limit={limit}&filter={safe_filter}" style="color:var(--blue);margin-right:12px" {prev_hidden}>上一页</a>
      第 {page} 页 / 共 {total_pages} 页
      <a href="?page={page+1}&limit={limit}&filter={safe_filter}" style="color:var(--blue);margin-left:12px" {next_hidden}>下一页</a>
    </div>"""
    _logger.info("[admin] screenshots_list page=%s limit=%s filter=%s total=%s cost=%.2fs", page, limit, filter, total, _time.time() - _t0)
    return HTMLResponse(_page("截图 & OCR 查看", content, "/screenshots"))


@app.get("/screenshots/{id}", response_class=HTMLResponse)
def screenshot_detail(id: int):
    """单个 tick 的截图 + OCR + Layout + API 详细视图（数据来自 debug JSON）"""
    # 查询 tick_log 获取 tick_id 和 session_id
    db = get_db()
    conn = db._get_conn()
    try:
        row = conn.execute(
            "SELECT tick_id, session_id, chat_name, screenshot_path FROM tick_log WHERE tick_id = ?",
            (id,)
        ).fetchone()
    finally:
        conn.close()

    tick_id = id
    session_id = ""
    db_chat_name = ""
    db_screenshot = ""

    if row:
        tick_id = row["tick_id"]
        session_id = row["session_id"] or ""
        db_chat_name = row["chat_name"] or ""
        db_screenshot = row["screenshot_path"] or ""

    # 找 debug JSON（无论数据库有无记录，都尝试从 debug JSON 读取）
    debug_files = sorted(DEBUG_DIR.glob(f"tick_*_{tick_id}.json"))
    if not debug_files:
        return HTMLResponse("<h1>Debug data not found</h1><p>没有找到 tick #%s 的 debug 数据</p>" % html.escape(str(tick_id)))

    dbg = {}
    sp = db_screenshot
    raw_chat = db_chat_name or str(tick_id)
    ts = ""
    try:
        dbg = _json.loads(debug_files[-1].read_text(encoding="utf-8"))
        if not sp:
            sp = dbg.get("screenshot_path", "") or dbg.get("perception_screenshot_path", "") or ""
        if not db_chat_name:
            raw_chat = dbg.get("perception_chat_name", "") or dbg.get("bot_chat_name", "") or ""
        ts = dbg.get("timestamp", "")
    except Exception as e:
        _logger.debug("读取 debug JSON 失败: %s", e)

    ocr_html = "<span style='color:var(--muted)'>无 OCR 数据</span>"
    layout_html = "<span style='color:var(--muted)'>无 Layout 数据</span>"
    api_prompt_html = "<span style='color:var(--muted)'>无 API 数据</span>"
    api_response_html = "<span style='color:var(--muted)'>无 API 响应</span>"
    api_thinking_html = "<span style='color:var(--muted)'>无 Thinking</span>"

    if dbg:
        try:
            # OCR elements
            ocr_elems = dbg.get("ocr_elements", [])
            if ocr_elems:
                ocr_rows = ""
                for elem in ocr_elems:
                    text = elem.get("text", "")
                    bbox = elem.get("bbox", [])
                    conf = elem.get("confidence", 0)
                    ocr_rows += f"<tr><td style='font-size:10px;color:var(--muted)'>{html.escape(str(bbox))}</td><td>{html.escape(text)}</td><td>{conf:.0%}</td></tr>"
                ocr_html = f"<table><tr><th>BBox</th><th>Text</th><th>置信度</th></tr>{ocr_rows}</table>"
            else:
                ocr_html = "<span style='color:var(--muted)'>OCR elements 为空（可能为本地跳过路径）</span>"

            # Layout groups
            layout_parts = []
            for key, label in [("layout_left_elements", "左侧聊天列表"), ("layout_right_elements", "右侧消息区"),
                               ("layout_title_elements", "标题栏"), ("layout_input_elements", "输入框"),
                               ("layout_chat_list_groups", "聊天分组"), ("layout_chat_list_nicknames", "昵称列表"),
                               ("layout_chat_list_unread", "未读标记"), ("layout_message_candidates", "消息候选")]:
                val = dbg.get(key, [])
                if val:
                    if isinstance(val, list):
                        items = "<br>".join(html.escape(str(v)) for v in val)
                        layout_parts.append(f"<div class='card'><b>{html.escape(label)}</b> ({len(val)}):<br><span style='font-size:11px'>{items}</span></div>")
                    else:
                        layout_parts.append(f"<div class='card'><b>{html.escape(label)}</b>: {html.escape(str(val))}</div>")
            if layout_parts:
                layout_html = "".join(layout_parts)

            # API prompt & response & thinking
            api_prompt = dbg.get("api_prompt", "")
            api_response = dbg.get("api_response", "")
            api_thinking = dbg.get("api_thinking", "")
            if api_prompt:
                api_prompt_html = f"<pre style='font-size:10px;max-height:300px;overflow:auto;white-space:pre-wrap;background:rgba(0,0,0,.2);padding:8px;border-radius:4px'>{html.escape(api_prompt)}</pre>"
            if api_response:
                api_response_html = f"<pre style='font-size:10px;max-height:300px;overflow:auto;white-space:pre-wrap;background:rgba(0,0,0,.2);padding:8px;border-radius:4px'>{html.escape(api_response)}</pre>"
            if api_thinking:
                api_thinking_html = f"<pre style='font-size:10px;max-height:300px;overflow:auto;white-space:pre-wrap;background:rgba(0,0,0,.2);padding:8px;border-radius:4px'>{html.escape(api_thinking)}</pre>"

            # 提取消息
            msgs = dbg.get("extraction_messages", [])
            if msgs:
                msgs_html = ""
                for m in msgs:
                    msgs_html += f"<tr><td>{html.escape(m.get('sender',''))}</td><td>{html.escape(m.get('text',''))}</td><td>{html.escape(m.get('chat_name',''))}</td></tr>"
                layout_parts.append(f"<div class='card'><b>提取的消息</b> ({len(msgs)}):<table><tr><th>发送者</th><th>文本</th><th>聊天</th></tr>{msgs_html}</table></div>")
        except Exception as e:
            _logger.debug("渲染截图详情失败: %s", e)

    fname = Path(sp).name if sp else ""
    img_html = f'<img src="/api/screenshot-image/{html.escape(fname)}" style="max-width:100%;border-radius:4px;border:1px solid var(--border)" onerror="this.style.display=\'none\'">' if fname else '<span style="color:var(--muted)">无截图</span>'

    display_id = f'{session_id}:#{tick_id}' if session_id else f'#{tick_id}'
    tick_link = f'<a href="/ticks/{id}" style="color:var(--blue);font-size:13px">→ 查看 Tick 详情</a>'
    content = f"""
    <div class="card"><b>{html.escape(display_id)}</b> — {html.escape(ts)} | {html.escape(raw_chat)} | {tick_link}<br><span style="font-size:11px;color:var(--muted)">{html.escape(sp)}</span></div>

    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <div style="flex:1;min-width:300px">
        <div class="card"><b>📸 截图</b> <span style="font-size:11px;color:var(--muted)">双击看大图</span><br>{img_html}</div>
      </div>
      <div style="flex:2;min-width:400px">
        <div class="card" style="border-left:3px solid var(--blue)"><b>🤖 多模态 API Prompt</b> ({len(dbg.get('api_prompt',''))} 字)<br>{api_prompt_html}</div>
        <div class="card" style="border-left:3px solid var(--green)"><b>🤖 多模态 API Response</b><br>{api_response_html}</div>
        <div class="card" style="border-left:3px solid var(--yellow)"><b>💭 Thinking</b> ({len(dbg.get('api_thinking',''))} 字)<br>{api_thinking_html}</div>
      </div>
    </div>

    <details style="margin-top:12px"><summary style="cursor:pointer;color:var(--muted)">📋 OCR 识别结果</summary><div class="card">{ocr_html}</div></details>
    <details style="margin-top:8px"><summary style="cursor:pointer;color:var(--muted)">📐 Layout 分组</summary><div class="card">{layout_html}</div></details>
    <p style="margin-top:12px"><a href="/screenshots" style="color:var(--blue)">返回列表</a></p>
    """
    # lgtm[py/reflected-xss] content 已用 html.escape 转义，待迁移模板引擎
    return HTMLResponse(_page(f"截图 {html.escape(display_id)}", content, "/screenshots"))


# ── Benchmark Dashboard ──

BENCHMARK_JUDGE = Path(__file__).parent.parent / "data" / "reports" / "benchmark_judge.html"
BENCHMARK_REPLY = Path(__file__).parent.parent / "data" / "reports" / "benchmark_reply.html"
GROUP_CHAT_JUDGE_REVIEW = Path(__file__).parent.parent / "data" / "reports" / "private" / "c0_guardrail_ab_v2" / "index.html"
STRICT_HISTORICAL_JUDGE_REVIEW = Path(__file__).parent.parent / "data" / "reports" / "private" / "strict_historical_judge_acceptance_v5_group_balanced" / "index.html"
YOUZI_IDENTITY_REVIEW = Path(__file__).parent.parent / "data" / "reports" / "private" / "youzi_identity_review_v1" / "report.html"
CURRENT_C0_HISTORICAL_AB = Path(__file__).parent.parent / "data" / "reports" / "private" / "clean_human_c0_iteration_v1" / "index.html"
JUDGE_PRIVATE_DIR = Path(__file__).parent.parent / "data" / "reports" / "private"


def _audit_escape(value: Any) -> str:
    """Escape private evidence while keeping JSON arrays from appearing raw in HTML."""
    return html.escape(str(value)).replace("[", "&#91;").replace("{", "&#123;")


def _audit_pre(value: Any, max_height: str = "28rem") -> str:
    return (
        f'<pre style="white-space:pre-wrap;overflow:auto;max-height:{max_height};'
        'overflow-wrap:anywhere;background:rgba(0,0,0,.2);padding:10px;border-radius:6px;'
        f'font-size:11px;line-height:1.5">{_audit_escape(value)}</pre>'
    )


def _load_private_json(name: str) -> dict[str, Any]:
    path = JUDGE_PRIVATE_DIR / name
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as exc:
        _logger.warning("读取 Judge 审计文件 %s 失败: %s", path, exc)
        return {}


def _load_private_jsonl(name: str) -> list[dict[str, Any]]:
    path = JUDGE_PRIVATE_DIR / name
    try:
        return [_json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, _json.JSONDecodeError) as exc:
        _logger.warning("读取 Judge 事件文件 %s 失败: %s", path, exc)
        return []


def _audit_named_values(value: Any) -> str:
    if isinstance(value, dict):
        return "".join(
            f'<div style="margin:8px 0"><b>{_audit_escape(key)}</b>{_audit_named_values(item)}</div>'
            for key, item in value.items()
        )
    if isinstance(value, list):
        if not value:
            return '<div style="color:var(--muted)">（空）</div>'
        return "".join(
            f'<details style="margin:6px 0"><summary>第 {index} 项</summary>{_audit_named_values(item)}</details>'
            for index, item in enumerate(value, start=1)
        )
    return _audit_pre("" if value is None else value)


def _audit_messages(context: list[Any], sections: tuple[str, ...]) -> list[dict[str, Any]]:
    """Turn the stored XML-like chat evidence into readable, escaped chat rows."""
    messages = []
    for block in context:
        text = str(block)
        if not text.startswith(sections):
            continue
        for match in re.finditer(r'<message\s+([^>]*)>(.*?)</message>', text, re.DOTALL):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
            body = html.unescape(match.group(2)).strip()
            if not body:
                continue
            try:
                index = int(attrs.get("index", "0"))
            except ValueError:
                index = 0
            messages.append({"index": index, "role": attrs.get("role", "other"), "body": body})
    return sorted(messages, key=lambda item: item["index"])


def _audit_chat_rows(messages: list[dict[str, Any]]) -> str:
    return "".join(
        f'<div class="judge-chat-row {"self" if message["role"] == "self" else "other"}">'
        f'{_audit_escape(message["body"])}</div>'
        for message in messages
    ) or '<p style="color:var(--muted)">没有可显示的消息</p>'


def _audit_reply_rows(value: Any) -> str:
    replies = value if isinstance(value, list) else [value]
    return "".join(
        f'<div class="judge-reply-line">{_audit_escape(reply)}</div>'
        for reply in replies if str(reply).strip()
    ) or '<div class="judge-reply-line">（空回复）</div>'


def _historical_judge_audit_html() -> str:
    blind = _load_private_json("human_ai_judge_blind.json")
    answers = _load_private_json("human_ai_judge_answers.json")
    predictions = _load_private_json("human_ai_judge_predictions.json")
    result = _load_private_json("human_ai_judge_result.json")
    replay = _load_private_json("human_ai_judge_replay.json")
    audit = _load_private_json("human_ai_judge_audit.json")
    events = audit.get("events") or _load_private_jsonl("human_ai_judge_events.jsonl")
    if not blind:
        return '<div class="card">暂无 Judge 盲测审计数据。</div>'

    if audit:
        evidence_status = "调用时原始捕获"
        prompt = audit.get("prompt", "")
        argv = audit.get("argv") or []
        prompt_hash = audit.get("prompt_sha256", "")
        tool_call_count = audit.get("tool_call_count", 0)
        evidence_warning = "本批提示词、命令和事件均在 Judge 调用时留存。"
    else:
        from scripts.evaluate_human_ai_judge import build_codex_judge_prompt
        import hashlib

        prompt = build_codex_judge_prompt(blind)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        argv = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "read-only", "--cd", "<temporary-directory>",
            "--skip-git-repo-check", "--json", "--output-last-message",
            "<temporary-last-message-path>", "-",
        ]
        tool_call_count = 0
        evidence_status = "根据执行时版本代码与 blind 文件重建"
        evidence_warning = (
            "本批次没有调用时 audit 文件：blind、答案、预测和事件是原始落盘证据；"
            "Judge prompt 与临时路径参数根据执行时版本代码重建，并非独立原始捕获。"
        )

    answer_by_public = {row.get("case_id"): row for row in answers.get("cases", [])}
    prediction_by_public = {row.get("case_id"): row for row in predictions.get("cases", [])}
    replay_by_source = {row.get("case_id"): row for row in replay.get("cases", [])}
    shorter_hits = 0
    shorter_total = 0
    for case in blind.get("cases", []):
        actual = answer_by_public.get(case.get("case_id"), {}).get("human_option")
        lengths = {
            option: len("".join(str(item) for item in case.get(f"option_{option}", [])))
            for option in ("A", "B")
        }
        if actual in {"A", "B"} and lengths["A"] != lengths["B"]:
            shorter_total += 1
            shorter_hits += min(lengths, key=lengths.get) == actual
    event_cards = []
    for index, event in enumerate(events, start=1):
        event_cards.append(
            f'<details><summary>事件 {index} · {_audit_escape(event.get("type", "unknown"))}</summary>'
            f'{_audit_named_values(event)}</details>'
        )

    case_cards = []
    for index, case in enumerate(blind.get("cases", []), start=1):
        public_id = case.get("case_id")
        answer = answer_by_public.get(public_id, {})
        prediction = prediction_by_public.get(public_id, {})
        source_id = answer.get("source_case_id")
        source = replay_by_source.get(source_id, {})
        actual = answer.get("human_option", "?")
        predicted = prediction.get("human_option", "?")
        correct = actual == predicted
        trace = source.get("c0_trace") or {}
        generation = trace.get("generation_trace") or {}
        context = "\n\n".join(str(item) for item in case.get("context_original", []))
        option_a = "\n".join(str(item) for item in case.get("option_A", []))
        option_b = "\n".join(str(item) for item in case.get("option_B", []))
        fact_note = ""
        if "我老家泰州" in option_a + option_b:
            fact_note = (
                '<div class="card" style="border-left:3px solid var(--yellow)">'
                '<b>泰州事实核查（已纠正）</b><p style="margin-top:6px">“老家泰州”有 C0 当轮 user prompt '
                '注入的记忆“籍贯：江苏泰州”支持，并经用户本人确认，不是编造。Judge 原始理由也没有判它编造，'
                '而是认为菜名罗列和邀约略显刻意；该风格判断是否合理仍需人工审阅。</p></div>'
            )
        trace_sections = []
        for label, key in (
            ("输入元数据", "input_metadata"),
            ("System prompt", "system_prompt"),
            ("User prompt（完整生产输入）", "user_prompt"),
            ("同聊天 few-shot", "few_shot_content"),
            ("工具上下文", "tools_context"),
            ("加载的 skills", "loaded_skills"),
            ("Skill 内容", "skill_content"),
            ("C0 工具调用", "tool_calls"),
            ("C0 LLM 调用", "llm_calls"),
        ):
            value = trace.get(key) if key == "input_metadata" else generation.get(key)
            trace_sections.append(
                f'<details><summary>{label}</summary>{_audit_named_values(value)}</details>'
            )
        case_cards.append(f'''
        <details class="card"{' open' if index == 1 else ''}>
          <summary style="cursor:pointer"><b>案例 {index}</b> · Judge 选 {predicted} · 真人是 {actual} · {'命中' if correct else '误判'}</summary>
          <p style="color:var(--muted);margin:8px 0">盲测 ID：{_audit_escape(public_id)} · 来源 ID：{_audit_escape(source_id)} · 关系：{_audit_escape(case.get('relationship', ''))}</p>
          <h3>Judge 看到的完整聊天上下文</h3>{_audit_pre(context, "45rem")}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px">
            <div class="card"><b>匿名选项 A</b>{_audit_pre(option_a)}</div>
            <div class="card"><b>匿名选项 B</b>{_audit_pre(option_b)}</div>
          </div>
          <div class="card"><b>Judge 原始判断</b><p style="margin-top:6px">选择 {predicted} · 置信度 {_audit_escape(prediction.get('confidence', ''))}</p>{_audit_pre(prediction.get('reason', ''))}</div>
          <div class="card"><b>答案揭晓</b><p style="margin-top:6px">真人：选项 {actual}；C0：选项 {'B' if actual == 'A' else 'A'}；结果：{'命中' if correct else '误判'}</p></div>
          {fact_note}
          <details><summary><b>C0 完整生成 trace</b> · 耗时 {_audit_escape(trace.get('duration_seconds', ''))} 秒</summary>{''.join(trace_sections)}</details>
        </details>''')

    argv_html = "".join(f"<li><code>{_audit_escape(arg)}</code></li>" for arg in argv)
    return f'''
    <div class="card" style="border-left:4px solid var(--red)">
      <b>结论状态：已作废</b>
      <p style="margin-top:6px">历史结果虽为 {result.get('correct', 0)}/{result.get('total', 0)}，但不能证明 Judge 能力，也不能用于评价或替换 Bot。</p>
    </div>
    <div class="metrics">
      <div class="metric"><div class="value">{result.get('correct', 0)}/{result.get('total', 0)}</div><div class="label">历史命中（已作废）</div></div>
      <div class="metric"><div class="value">{len(events)}</div><div class="label">Judge 事件</div></div>
      <div class="metric"><div class="value">{tool_call_count}</div><div class="label">Judge 工具调用</div></div>
    </div>
    <div class="card"><b>证据状态</b><p style="margin-top:6px">{_audit_escape(evidence_status)}</p><p style="color:var(--yellow);margin-top:6px">{_audit_escape(evidence_warning)}</p><p style="margin-top:6px">批次：{_audit_escape(blind.get('batch_id', ''))}<br>Blind SHA256：{_audit_escape(answers.get('blind_sha256', ''))}<br>Prompt SHA256：{_audit_escape(prompt_hash)}</p></div>
    <div class="card" style="border-left:4px solid var(--yellow)"><b>当前流程复核</b>
      <p style="margin-top:6px">答案没有出现在 Judge prompt；匿名选项使用加密随机数随机放入 A/B；Judge 在独立 ephemeral 进程和临时目录运行。原始事件记录到的工具调用为 {tool_call_count} 次。</p>
      <p style="margin-top:6px;color:var(--yellow)">仍不能据此认定流程无泄漏：本批缺少调用时完整 audit，且样本只有 {len(blind.get('cases', []))} 题。仅用“选择更短回复”的规则就能命中 {shorter_hits}/{shorter_total} 个非等长案例，说明题目存在明显表面线索，历史高分不足以证明 Judge 理解本人风格。</p>
    </div>
    <details class="card"><summary><b>Judge instruction</b></summary>{_audit_pre(blind.get('instruction', ''))}</details>
    <details class="card"><summary><b>Judge 完整提示词</b></summary>{_audit_pre(prompt, "60rem")}</details>
    <details class="card"><summary><b>Judge 隔离调用参数</b></summary><ol style="padding-left:24px;margin-top:8px">{argv_html}</ol></details>
    <details class="card"><summary><b>Judge 原始事件与 token 用量</b></summary>{''.join(event_cards) or '<p>无事件记录</p>'}</details>
    <h2 style="margin:20px 0 12px">逐题完整审计</h2>
    {''.join(case_cards)}
    '''


def _revised_judge_audit_html() -> str:
    blind = _load_private_json("human_ai_judge_revised_blind.json")
    answers = _load_private_json("human_ai_judge_revised_answers.json")
    predictions = _load_private_json("human_ai_judge_revised_predictions.json")
    result = _load_private_json("human_ai_judge_revised_result.json")
    replay = _load_private_json("human_ai_judge_revised_replay.json")
    reference = _load_private_json("human_ai_judge_revised_reference.json")
    audit = _load_private_json("human_ai_judge_revised_audit.json")
    if not blind or not audit:
        return '<div class="card">新版平衡盲测的调用证据尚不完整。</div>'

    events = audit.get("events") or _load_private_jsonl("human_ai_judge_revised_events.jsonl")
    answer_by_public = {row.get("case_id"): row for row in answers.get("cases", [])}
    prediction_by_public = {row.get("case_id"): row for row in predictions.get("cases", [])}
    replay_by_source = {row.get("case_id"): row for row in replay.get("cases", [])}
    answer_counts = {option: 0 for option in ("A", "B")}
    for answer in answer_by_public.values():
        option = answer.get("human_option")
        if option in answer_counts:
            answer_counts[option] += 1

    source_id_hits = sum(
        bool(source_id and source_id in audit.get("prompt", ""))
        for source_id in replay_by_source
    )
    hash_checks = {
        "Blind hash 与答案文件一致": audit.get("blind_sha256") == answers.get("blind_sha256"),
        "Blind hash 与预测文件一致": audit.get("blind_sha256") == predictions.get("blind_sha256"),
        "Reference hash 三处一致": bool(audit.get("reference_sha256"))
        and audit.get("reference_sha256") == blind.get("reference_sha256") == reference.get("reference_sha256"),
        "A/B 答案位置平衡": answer_counts == {"A": 15, "B": 15},
        "来源 case ID 未进入提示词": source_id_hits == 0,
        "本次观察到 Judge 工具调用为 0": audit.get("tool_call_count") == 0,
        "事件解析无错误": not audit.get("event_parse_errors"),
    }
    check_html = "".join(
        f'<li style="color:{"var(--green)" if passed else "var(--red)"}">'
        f'{"通过" if passed else "失败"} · {_audit_escape(label)}</li>'
        for label, passed in hash_checks.items()
    )

    baselines = result.get("surface_baselines", {})
    rules = baselines.get("rules", {})
    baseline_html = "".join(
        f'<tr><td>{_audit_escape(name)}</td><td>{value.get("correct", 0)}/{value.get("decided", 0)}</td>'
        f'<td>{value.get("accuracy", 0):.1%}</td></tr>'
        for name, value in rules.items()
    )
    reference_html = "".join(
        f'<details><summary>真人参考 {index} · {_audit_escape(example.get("relationship", ""))}</summary>'
        f'<b>上下文</b>{_audit_named_values(example.get("context"))}'
        f'<b>真人回复</b>{_audit_named_values(example.get("human_reply"))}</details>'
        for index, example in enumerate(reference.get("examples", []), start=1)
    )
    event_html = "".join(
        f'<details><summary>事件 {index} · {_audit_escape(event.get("type", "unknown"))}</summary>'
        f'{_audit_named_values(event)}</details>'
        for index, event in enumerate(events, start=1)
    )
    argv_html = "".join(f'<li><code>{_audit_escape(arg)}</code></li>' for arg in audit.get("argv", []))

    case_cards = []
    for index, case in enumerate(blind.get("cases", []), start=1):
        public_id = case.get("case_id")
        answer = answer_by_public.get(public_id, {})
        prediction = prediction_by_public.get(public_id, {})
        source = replay_by_source.get(answer.get("source_case_id"), {})
        actual = answer.get("human_option", "?")
        predicted = prediction.get("human_option", "?")
        correct = actual == predicted
        trace = source.get("c0_trace") or {}
        generation = trace.get("generation_trace") or {}
        trace_sections = []
        for label, value in (
            ("输入元数据", trace.get("input_metadata")),
            ("System prompt", generation.get("system_prompt")),
            ("User prompt（C0 完整生产输入）", generation.get("user_prompt")),
            ("同聊天 few-shot", generation.get("few_shot_content")),
            ("工具上下文", generation.get("tools_context")),
            ("加载的 skills", generation.get("loaded_skills")),
            ("Skill 内容", generation.get("skill_content")),
            ("C0 工具调用", generation.get("tool_calls")),
            ("C0 LLM 调用", generation.get("llm_calls")),
        ):
            trace_sections.append(f'<details><summary>{label}</summary>{_audit_named_values(value)}</details>')
        context_original = case.get("context_original", [])
        prior_messages = _audit_messages(
            context_original, ("<conversation_history>", "<history>", "<consumed_self_replies")
        )
        target_messages = _audit_messages(context_original, ("<messages_to_reply>", "<unread>"))
        recent_messages = prior_messages[-8:]
        result_class = "wrong" if not correct else "correct"
        case_cards.append(f'''
        <section class="card judge-case {result_class}">
          <div class="judge-case-title"><b>第 {index} 题</b><span class="judge-badge {result_class}">{'判断正确' if correct else 'Judge 误判'}</span></div>
          <div class="judge-context">
            <div class="judge-section-label">最近的聊天</div>
            {_audit_chat_rows(recent_messages)}
            <div class="judge-section-label target">对方最后一句</div>
            {_audit_chat_rows(target_messages)}
          </div>
          <div class="judge-options">
            <div class="judge-option"><b>回复 A</b>{_audit_reply_rows(case.get('option_A', []))}</div>
            <div class="judge-option"><b>回复 B</b>{_audit_reply_rows(case.get('option_B', []))}</div>
          </div>
          <div class="judge-verdict">
            Judge 选择 <b>{predicted}</b>，真人实际是 <b>{actual}</b>
            <span>{_audit_escape(prediction.get('reason', ''))}</span>
          </div>
          <details class="judge-raw"><summary>查看本题原始审计</summary>
            <h3>完整原始上下文</h3>{_audit_named_values(context_original)}
            <h3>Judge 原始判断</h3>{_audit_named_values(prediction)}
            <h3>C0 完整生成 trace · {_audit_escape(trace.get('duration_seconds', ''))} 秒</h3>{''.join(trace_sections)}
            <p class="judge-ids">盲测 ID：{_audit_escape(public_id)} · 来源 ID：{_audit_escape(answer.get('source_case_id', ''))}</p>
          </details>
        </section>''')

    return f'''
    <style>
      .judge-summary{{display:flex;align-items:center;gap:14px;border-left:4px solid var(--yellow)}}
      .judge-score{{font-size:30px;font-weight:750;white-space:nowrap}}
      .judge-summary p{{color:var(--muted);font-size:13px;line-height:1.5}}
      .judge-case{{padding:14px;border-left:4px solid var(--green)}}
      .judge-case.wrong{{border-left-color:var(--red)}}
      .judge-case-title{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}}
      .judge-badge{{font-size:11px;color:var(--green);background:rgba(63,185,80,.12);padding:4px 8px;border-radius:999px}}
      .judge-badge.wrong{{color:var(--red);background:rgba(248,81,73,.12)}}
      .judge-context{{background:rgba(0,0,0,.16);border-radius:8px;padding:10px}}
      .judge-section-label{{color:var(--muted);font-size:11px;margin:2px 0 8px}}
      .judge-section-label.target{{color:var(--yellow);margin-top:12px}}
      .judge-chat-row{{width:fit-content;max-width:88%;padding:7px 10px;border-radius:10px;margin:5px 0;font-size:13px;line-height:1.45;background:#222a35}}
      .judge-chat-row.self{{margin-left:auto;background:#173b28}}
      .judge-options{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}}
      .judge-option{{border:1px solid var(--border);border-radius:8px;padding:10px;font-size:12px}}
      .judge-reply-line{{margin-top:7px;padding:8px 10px;background:rgba(88,166,255,.08);border-radius:8px;font-size:14px;line-height:1.4}}
      .judge-verdict{{margin-top:10px;font-size:13px}}
      .judge-verdict span{{display:block;color:var(--muted);margin-top:5px;line-height:1.45}}
      .judge-raw{{margin-top:12px;border-top:1px solid var(--border);padding-top:10px}}
      .judge-raw summary,.judge-tech summary{{cursor:pointer;color:var(--blue);font-size:13px}}
      .judge-raw h3{{font-size:13px;margin:14px 0 6px}}
      .judge-ids{{color:var(--muted);font-size:11px;margin-top:12px}}
      @media(max-width:700px){{
        main{{padding:12px}}nav{{display:none}}h1{{font-size:18px}}
        .judge-options{{grid-template-columns:1fr}}
        .judge-chat-row{{max-width:96%}}
        .judge-summary{{align-items:flex-start}}
      }}
    </style>
    <div class="card judge-summary"><div class="judge-score">{result.get('correct', 0)}/{result.get('total', 0)}</div>
      <p>Judge 从真人回复和 C0 回复中辨认哪一个更像本人。这个分数只验证 Judge，不代表 Bot 回复质量。</p>
    </div>
    <details class="card judge-tech"><summary><b>查看 Judge 原始提示词</b></summary>{_audit_pre(audit.get('prompt', ''), '70rem')}</details>
    <details class="card judge-tech"><summary><b>查看评测说明和完整技术审计</b></summary>
      <h3>泄漏与能力复核</h3><ul style="padding-left:24px;margin:8px 0">{check_html}</ul>
      <p>本次观察到 {audit.get('tool_call_count', 0)} 次 Judge 工具调用；A/B 真人答案位置为 A {answer_counts['A']} / B {answer_counts['B']}。</p>
      <h3 style="margin-top:14px">表面规则基线</h3><table><tr><th>规则</th><th>命中/可判</th><th>准确率</th></tr>{baseline_html}</table>
      <details><summary>隔离调用参数</summary><ol style="padding-left:24px;margin-top:8px">{argv_html}</ol></details>
      <details><summary>原始事件</summary>{event_html or '<p>无事件</p>'}</details>
      <details><summary>stdout</summary>{_audit_pre(audit.get('stdout', ''), '50rem')}</details>
      <details><summary>stderr</summary>{_audit_pre(audit.get('stderr', ''), '20rem')}</details>
      <details><summary>last message</summary>{_audit_pre(audit.get('last_message', ''), '50rem')}</details>
      <details><summary>冻结事实、隔离信息与 32 条真人参考</summary>
        <h3>事实</h3>{_audit_named_values(reference.get('facts'))}<h3>隔离信息</h3>{_audit_named_values(reference.get('isolation'))}
        <p>真人回复截止：{_audit_escape(reference.get('human_before', ''))} · Reference SHA256：{_audit_escape(reference.get('reference_sha256', ''))}</p>{reference_html}
      </details>
      <details><summary>C0 生成批次与生产差异</summary>{_audit_named_values({
        'generator_label': replay.get('generator_label'),
        'backend': replay.get('backend'),
        'validation': replay.get('validation'),
        'generation_outcomes': replay.get('generation_outcomes'),
        'c0_sha256_after': replay.get('c0_sha256_after'),
      })}</details>
    </details>
    <h2 style="margin:20px 0 12px">30 道盲测题</h2>{''.join(case_cards)}
    '''


def _api_independent_judge_audit_html() -> str:
    blind = _load_private_json("human_ai_judge_revised_blind.json")
    answers = _load_private_json("human_ai_judge_revised_answers.json")
    predictions = _load_private_json("human_ai_judge_api_independent_predictions.json")
    result = _load_private_json("human_ai_judge_api_independent_result.json")
    audit = _load_private_json("human_ai_judge_api_independent_audit.json")
    if not all((blind, answers, predictions, result, audit)):
        return '<div class="card">逐题独立 API Judge 的审计数据尚不完整。</div>'

    answer_by_id = {row.get("case_id"): row for row in answers.get("cases", [])}
    prediction_by_id = {row.get("case_id"): row for row in predictions.get("cases", [])}
    call_by_id = {row.get("case_id"): row for row in audit.get("calls", [])}
    calls = audit.get("calls", [])
    isolation = audit.get("isolation", {})
    checks = {
        "没有使用 Codex，也无法翻阅 Codex Thread 历史": isolation.get("codex_used") is False,
        "每道题单独一次 API 请求": isolation.get("one_api_request_per_case") is True and len(calls) == len(blind.get("cases", [])),
        "每题新建 Client，不共享对话状态": isolation.get("new_client_per_case") is True and isolation.get("shared_conversation_state") is False,
        "Judge 调用结束前没有加载答案文件": isolation.get("answer_key_loaded_during_judging") is False,
        "请求中没有其他测试题": not any(call.get("other_test_case_ids_in_request") for call in calls),
        "请求中没有答案或来源身份字段": not any(call.get("answer_fields_in_request") for call in calls),
        "没有向模型提供工具": isolation.get("tools_available") is False and not any(call.get("tools_sent") for call in calls),
        "30 次请求全部成功": bool(calls) and all(call.get("status_code") == 200 for call in calls),
    }
    check_html = "".join(
        f'<li class="judge-check {"pass" if passed else "fail"}">{"通过" if passed else "失败"} · {_audit_escape(label)}</li>'
        for label, passed in checks.items()
    )

    case_cards = []
    for index, case in enumerate(blind.get("cases", []), start=1):
        case_id = case.get("case_id")
        answer = answer_by_id.get(case_id, {})
        prediction = prediction_by_id.get(case_id, {})
        call = call_by_id.get(case_id, {})
        actual = answer.get("human_option", "?")
        predicted = prediction.get("human_option", "?")
        correct = actual == predicted
        result_class = "correct" if correct else "wrong"
        context_original = case.get("context_original", [])
        prior_messages = _audit_messages(
            context_original, ("<conversation_history>", "<history>", "<consumed_self_replies")
        )
        target_messages = _audit_messages(context_original, ("<messages_to_reply>", "<unread>"))
        request_body = call.get("request_body") or {}
        request_messages = request_body.get("messages") or []
        user_prompt = request_messages[0].get("content") if request_messages else ""
        if isinstance(user_prompt, list):
            user_prompt = "\n".join(str(block.get("text", "")) for block in user_prompt if isinstance(block, dict))
        case_cards.append(f'''
        <section class="card judge-case {result_class}">
          <div class="judge-case-title"><b>第 {index} 题</b><span class="judge-badge {result_class}">{'判断正确' if correct else 'Judge 误判'}</span></div>
          <div class="judge-context">
            <div class="judge-section-label">最近的聊天</div>{_audit_chat_rows(prior_messages[-8:])}
            <div class="judge-section-label target">对方最后一句</div>{_audit_chat_rows(target_messages)}
          </div>
          <div class="judge-options">
            <div class="judge-option"><b>回复 A</b>{_audit_reply_rows(case.get('option_A', []))}</div>
            <div class="judge-option"><b>回复 B</b>{_audit_reply_rows(case.get('option_B', []))}</div>
          </div>
          <div class="judge-verdict">Judge 选择 <b>{predicted}</b>，真人实际是 <b>{actual}</b>
            <span>{_audit_escape(prediction.get('reason', ''))}</span>
          </div>
          <details class="judge-raw"><summary>查看本题原始提示词与 API 调用</summary>
            <h3>完整原始聊天上下文</h3>{_audit_named_values(context_original)}
            <h3>Judge system prompt</h3>{_audit_pre(request_body.get('system', ''), '55rem')}
            <h3>Judge user prompt（仅本题）</h3>{_audit_pre(user_prompt, '55rem')}
            <h3>请求过程</h3>{_audit_named_values({
                '请求时间（UTC）': call.get('recorded_at_utc'),
                '传输方式': call.get('transport'),
                '会话状态': call.get('conversation_state'),
                'Endpoint': call.get('endpoint'),
                'Headers（密钥已脱敏）': call.get('headers'),
                '请求模型': request_body.get('model'),
                '请求 SHA256': call.get('request_sha256'),
                '其他测试题 ID': call.get('other_test_case_ids_in_request'),
                '答案字段': call.get('answer_fields_in_request'),
                '工具': call.get('tools_sent'),
                'HTTP 状态': call.get('status_code'),
                '耗时毫秒': call.get('elapsed_ms'),
                '实际模型': call.get('resolved_model'),
            })}
            <h3>模型原始响应（thinking 未保存）</h3>{_audit_named_values(call.get('response'))}
            <h3>解析后的原始判断</h3>{_audit_named_values(call.get('parsed_prediction'))}
          </details>
        </section>''')

    model_names = sorted({str(call.get("resolved_model")) for call in calls if call.get("resolved_model")})
    return f'''
    <style>
      .judge-summary{{display:flex;align-items:center;gap:14px;border-left:4px solid var(--red)}}
      .judge-score{{font-size:30px;font-weight:750;white-space:nowrap}}
      .judge-summary p{{color:var(--muted);font-size:13px;line-height:1.5}}
      .judge-case{{padding:14px;border-left:4px solid var(--green)}}.judge-case.wrong{{border-left-color:var(--red)}}
      .judge-case-title{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}}
      .judge-badge{{font-size:11px;color:var(--green);background:rgba(63,185,80,.12);padding:4px 8px;border-radius:999px}}
      .judge-badge.wrong{{color:var(--red);background:rgba(248,81,73,.12)}}
      .judge-context{{background:rgba(0,0,0,.16);border-radius:8px;padding:10px}}
      .judge-section-label{{color:var(--muted);font-size:11px;margin:2px 0 8px}}.judge-section-label.target{{color:var(--yellow);margin-top:12px}}
      .judge-chat-row{{width:fit-content;max-width:88%;padding:7px 10px;border-radius:10px;margin:5px 0;font-size:13px;line-height:1.45;background:#222a35}}
      .judge-chat-row.self{{margin-left:auto;background:#173b28}}
      .judge-options{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}}
      .judge-option{{border:1px solid var(--border);border-radius:8px;padding:10px;font-size:12px}}
      .judge-reply-line{{margin-top:7px;padding:8px 10px;background:rgba(88,166,255,.08);border-radius:8px;font-size:14px;line-height:1.4}}
      .judge-verdict{{margin-top:10px;font-size:13px}}.judge-verdict span{{display:block;color:var(--muted);margin-top:5px;line-height:1.45}}
      .judge-raw{{margin-top:12px;border-top:1px solid var(--border);padding-top:10px}}.judge-raw summary,.judge-tech summary{{cursor:pointer;color:var(--blue);font-size:13px}}
      .judge-raw h3{{font-size:13px;margin:14px 0 6px}}.judge-check{{margin:6px 0;color:var(--green)}}.judge-check.fail{{color:var(--red)}}
      @media(max-width:700px){{main{{padding:12px}}nav{{display:none}}h1{{font-size:18px}}.judge-options{{grid-template-columns:1fr}}.judge-chat-row{{max-width:96%}}.judge-summary{{align-items:flex-start}}}}
    </style>
    <div class="card judge-summary"><div class="judge-score">{result.get('correct', 0)}/{result.get('total', 0)}</div>
      <p><b>这轮 Judge 未通过，且输入含有 C0 约束线索，结果已作废。</b><br>旧 27/30 同样不可信；页面保留旧记录仅供审计，不用于评价 Judge 或 Bot。</p>
    </div>
    <details class="card judge-tech" open><summary><b>查看隔离检查</b></summary>
      <ul style="padding-left:24px;margin:10px 0">{check_html}</ul>
      <p style="color:var(--muted)">调用 {len(calls)} 次 · 模型 {_audit_escape(', '.join(model_names))} · API key 全部脱敏 · 核分在全部请求完成后单独进行。</p>
    </details>
    <h2 style="margin:20px 0 12px">30 道逐题独立盲测</h2>{''.join(case_cards)}
    '''


def _boundary_candidate_judge_audit_html() -> str:
    base = "human_ai_judge_benchmark_v2"
    blind = _load_private_json(f"{base}/final_blind.json")
    answers = _load_private_json(f"{base}/final_answers.json")
    v2 = _load_private_json(f"{base}/final_v2_codex_cli_canonical_comparison.json")
    v3 = _load_private_json(f"{base}/final_v3_codex_cli_canonical_comparison.json")
    candidate = _load_private_json(
        f"{base}/final_v2_boundary_candidate_codex_cli_canonical_comparison.json"
    )
    audit = _load_private_json(
        f"{base}/final_v2_boundary_candidate_codex_cli_canonical_comparison_audit.json"
    )
    if not blind or not answers or not candidate or not audit:
        return ""

    answer_by_id = {row.get("case_id"): row for row in answers.get("cases", [])}
    candidate_by_id = {row.get("case_id"): row for row in candidate.get("cases", [])}
    v2_by_id = {row.get("case_id"): row for row in v2.get("cases", [])}
    v3_by_id = {row.get("case_id"): row for row in v3.get("cases", [])}
    call_by_id = {row.get("case_id"): row for row in audit.get("calls", [])}

    def score(rows: dict[str, dict[str, Any]]) -> tuple[int, int]:
        total = len(blind.get("cases", []))
        correct = sum(
            rows.get(case.get("case_id"), {}).get("human_option")
            == answer_by_id.get(case.get("case_id"), {}).get("human_option")
            for case in blind.get("cases", [])
        )
        return correct, total

    candidate_score = score(candidate_by_id)
    v2_score = score(v2_by_id)
    v3_score = score(v3_by_id)
    calls = audit.get("calls", [])
    isolation = audit.get("isolation") or {}
    checks = {
        "100 道题分别调用独立进程": len(calls) == 100
        and len({call.get("request_sha256") for call in calls}) == 100,
        "请求中没有答案字段": not any(call.get("answer_fields_in_request") for call in calls),
        "请求中没有其他测试题 ID": not any(call.get("other_test_case_ids_in_request") for call in calls),
        "实际没有工具调用": not any(call.get("tool_events") for call in calls),
        "全部调用成功": bool(calls) and not any(call.get("error") for call in calls),
    }
    check_html = "".join(
        f'<li class="judge-check {"pass" if passed else "fail"}">'
        f'{"通过" if passed else "失败"} · {_audit_escape(label)}</li>'
        for label, passed in checks.items()
    )

    case_cards = []
    for index, case in enumerate(blind.get("cases", []), start=1):
        case_id = case.get("case_id")
        actual = answer_by_id.get(case_id, {}).get("human_option", "?")
        prediction = candidate_by_id.get(case_id, {})
        predicted = prediction.get("human_option", "?")
        correct = actual == predicted
        result_class = "correct" if correct else "wrong"
        prior_messages = _audit_messages(
            case.get("context_original", []),
            ("<conversation_history>", "<history>", "<consumed_self_replies"),
        )
        target_messages = _audit_messages(
            case.get("context_original", []), ("<messages_to_reply>", "<unread>")
        )
        call = call_by_id.get(case_id, {})
        request_messages = call.get("request_messages") or []
        system_prompt = next(
            (row.get("content", "") for row in request_messages if row.get("role") == "system"), ""
        )
        user_prompt = next(
            (row.get("content", "") for row in request_messages if row.get("role") == "user"), ""
        )
        response = call.get("response") or {}
        old_v2 = v2_by_id.get(case_id, {})
        old_v3 = v3_by_id.get(case_id, {})
        case_cards.append(f'''
        <section class="card judge-case {result_class}">
          <div class="judge-case-title"><b>第 {index} 题</b><span class="judge-badge {result_class}">{'判断正确' if correct else 'Judge 误判'}</span></div>
          <div class="judge-context">
            <div class="judge-section-label">完整聊天上下文（{len(prior_messages)} 条）</div>{_audit_chat_rows(prior_messages)}
            <div class="judge-section-label target">需要回复的消息</div>{_audit_chat_rows(target_messages)}
          </div>
          <div class="judge-options">
            <div class="judge-option {'human' if actual == 'A' else ''}"><b>回复 A</b>{_audit_reply_rows(case.get('option_A', []))}</div>
            <div class="judge-option {'human' if actual == 'B' else ''}"><b>回复 B</b>{_audit_reply_rows(case.get('option_B', []))}</div>
          </div>
          <div class="judge-verdict"><b>真人原回复：{actual}</b> · 候选 Judge 选择：<b>{predicted}</b> · 置信度 {_audit_escape(prediction.get('confidence', ''))}
            <span>{_audit_escape(prediction.get('reason', ''))}</span>
            <span>对照：V2 选 {_audit_escape(old_v2.get('human_option', '?'))}；V3 选 {_audit_escape(old_v3.get('human_option', '?'))}</span>
          </div>
          <details class="judge-raw"><summary>查看本题原始提示词与完整调用记录</summary>
            <h3>完整原始上下文</h3>{_audit_named_values(case.get('context_original', []))}
            <h3>Judge system prompt</h3>{_audit_pre(system_prompt, '60rem')}
            <h3>Judge user prompt（仅本题）</h3>{_audit_pre(user_prompt, '60rem')}
            <h3>调用信息</h3>{_audit_named_values({
                'case_id': case_id,
                '请求时间（UTC）': call.get('recorded_at_utc'),
                '请求 SHA256': call.get('request_sha256'),
                '耗时毫秒': call.get('elapsed_ms'),
                '答案字段': call.get('answer_fields_in_request'),
                '其他测试题 ID': call.get('other_test_case_ids_in_request'),
                '工具配置': call.get('tools_configured'),
                '工具调用事件': call.get('tool_events'),
                '命令参数': response.get('argv'),
                '返回码': response.get('returncode'),
                'prompt 传输方式': response.get('prompt_transport'),
            })}
            <h3>模型原始回复</h3>{_audit_pre(call.get('raw_response', ''))}
            <h3>stdout</h3>{_audit_pre(response.get('stdout', ''), '40rem')}
            <h3>stderr</h3>{_audit_pre(response.get('stderr', ''), '20rem')}
            <h3>原始事件</h3>{_audit_named_values(response.get('events', []))}
            <h3>事件解析错误</h3>{_audit_named_values(response.get('event_parse_errors', []))}
            <h3>调用错误</h3>{_audit_named_values(call.get('error'))}
          </details>
        </section>''')

    return f'''
    <style>
      .judge-summary{{display:flex;align-items:center;gap:14px;border-left:4px solid var(--yellow)}}
      .judge-score{{font-size:30px;font-weight:750;white-space:nowrap}}.judge-summary p{{color:var(--muted);font-size:13px;line-height:1.5}}
      .judge-case{{padding:14px;border-left:4px solid var(--green)}}.judge-case.wrong{{border-left-color:var(--red)}}
      .judge-case-title{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}}
      .judge-badge{{font-size:11px;color:var(--green);background:rgba(63,185,80,.12);padding:4px 8px;border-radius:999px}}.judge-badge.wrong{{color:var(--red);background:rgba(248,81,73,.12)}}
      .judge-context{{background:rgba(0,0,0,.16);border-radius:8px;padding:10px;max-height:34rem;overflow:auto}}
      .judge-section-label{{position:sticky;top:0;color:var(--muted);font-size:11px;margin:2px 0 8px;background:#161b22;padding:4px}}.judge-section-label.target{{color:var(--yellow);margin-top:12px}}
      .judge-chat-row{{width:fit-content;max-width:88%;padding:7px 10px;border-radius:10px;margin:5px 0;font-size:13px;line-height:1.45;background:#222a35}}.judge-chat-row.self{{margin-left:auto;background:#173b28}}
      .judge-options{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}}.judge-option{{border:1px solid var(--border);border-radius:8px;padding:10px;font-size:12px}}.judge-option.human{{border-color:var(--green)}}
      .judge-reply-line{{margin-top:7px;padding:8px 10px;background:rgba(88,166,255,.08);border-radius:8px;font-size:14px;line-height:1.4}}
      .judge-verdict{{margin-top:10px;font-size:13px}}.judge-verdict span{{display:block;color:var(--muted);margin-top:5px;line-height:1.45}}
      .judge-raw{{margin-top:12px;border-top:1px solid var(--border);padding-top:10px}}.judge-raw summary,.judge-tech summary{{cursor:pointer;color:var(--blue);font-size:13px}}.judge-raw h3{{font-size:13px;margin:14px 0 6px}}
      .judge-check{{margin:6px 0;color:var(--green)}}.judge-check.fail{{color:var(--red)}}
      @media(max-width:700px){{main{{padding:12px}}nav{{display:none}}h1{{font-size:18px}}.judge-options{{grid-template-columns:1fr}}.judge-chat-row{{max-width:96%}}.judge-summary{{align-items:flex-start;flex-direction:column}}}}
    </style>
    <div class="card judge-summary"><div class="judge-score">{candidate_score[0]}/{candidate_score[1]}</div>
      <p><b>边界候选 Judge：开发集结果，不是隐藏终测。</b><br>V2 为 {v2_score[0]}/{v2_score[1]}，V3 为 {v3_score[0]}/{v3_score[1]}。候选版只用于检查误判边界；当前仍保留 V2 作为临时诊断基线，不让 Judge 直接决定 Bot 版本胜负。</p>
    </div>
    <details class="card judge-tech"><summary><b>查看隔离检查与批次说明</b></summary>
      <ul style="padding-left:24px;margin:10px 0">{check_html}</ul>
      <p style="color:var(--muted)">Provider：{_audit_escape(audit.get('provider', ''))} · Model：{_audit_escape(audit.get('model', ''))} · Presentation：{_audit_escape(audit.get('presentation', ''))} · Sandbox：{_audit_escape(isolation.get('sandbox', 'read-only'))}</p>
      <h3>候选 Judge 配置</h3>{_audit_named_values(audit.get('profile', {}))}
    </details>
    <h2 style="margin:20px 0 12px">100 道题：完整上下文与原始回复</h2>{''.join(case_cards)}
    '''


def _v2_misclassification_review_html() -> str:
    base = "human_ai_judge_benchmark_v2"
    blind = _load_private_json(f"{base}/final_blind.json")
    answers = _load_private_json(f"{base}/final_answers.json")
    v2 = _load_private_json(f"{base}/final_v2_codex_cli_canonical_comparison.json")
    audit = _load_private_json(
        f"{base}/final_v2_codex_cli_canonical_comparison_audit.json"
    )
    if not blind or not answers or not v2 or not audit:
        return ""

    answer_by_id = {row.get("case_id"): row for row in answers.get("cases", [])}
    prediction_by_id = {row.get("case_id"): row for row in v2.get("cases", [])}
    call_by_id = {row.get("case_id"): row for row in audit.get("calls", [])}
    review_notes = {
        "case_08ae4c9b560b9e29dc35a3cd": (
            "跳回前文",
            "V2 只看最后一句“谢谢大佬”，偏爱局部最顺的“客气啥”；真人却跳回刚才的婚讯继续表达惊讶。它低估了真人延迟回应前文的情况。",
        ),
        "case_83d891ed03dbeb39a562d0b6": (
            "偏爱理想推进",
            "V2 认为“翻翻个人邮箱”更完整、更会办事，但真人只是简短说“我知道”。它把更理想的任务推进误当成了本人表达。",
        ),
        "case_ffbc847254438160d97f9b10": (
            "排斥真实细节",
            "V2 把包含泰州、唐山和春节安排的真人原话当成凭空添加，转而选择更安全客气的回复。这里说明它会把真实但上下文未充分铺垫的个人事实误判为 AI 编造。",
        ),
        "case_af4a91fdd9a39be7b84f5400": (
            "偏爱礼貌顺序",
            "V2 认为应先礼貌听介绍，但真人直接问“还有合适的坑不”。它偏爱更得体的对话顺序，漏掉本人在求职话题里的直接推进。",
        ),
        "case_1f5752ba2f143df80adcbadb": (
            "不接受犹豫",
            "真人一边回忆年终奖、一边重复确认补偿，表达有摇摆；V2 选择了更凝练的事后总结，忽略了真人常见的边想边说。",
        ),
        "case_11257fdcc943a222edf607e8": (
            "不接受自我修正",
            "真人先说了一句不够顺的话，再补充“没看月考评”；V2 更喜欢一次说清楚的版本，因而把真实的自我修正当成了不自然。",
        ),
        "case_0849a4bae944c28fbf55cc3c": (
            "偏爱完整确认",
            "真人只回答“是的”，V2 却偏爱把床脚和床垫步骤一起确认的完整回复。它把信息更充分误当成了更像本人。",
        ),
        "case_99123e0718527b0c6a3a9bd2": (
            "排斥重复确认",
            "真人继续确认烤箱项目是否额外收费，即使前一句已经说“没有加”；V2 偏爱直接追问金额，低估了真人重复确认口径的习惯。",
        ),
        "case_fc4aea76813bde66b8afa198": (
            "偏爱丰富回应",
            "真人只说“啊 这么快”，V2 选择了同时表达惊讶并追问去向的版本。回复更丰富并不等于更像本人。",
        ),
        "case_4e06c121dbc8a2fb724c1979": (
            "漏掉临时换话题",
            "V2 认为“辛苦”最适合承接“好了”，但真人立刻切到“图搜又崩了”的新故障。它过度依赖最后一句的局部闭环。",
        ),
        "case_81fc4a27de20e8568f66ccf9": (
            "忽略追问惯性",
            "对方问“你呢”，真人没有先介绍自己，而是继续追问对方在哪里当老师；V2 偏爱完整回答反问，忽略了本人沿自己问题继续问的方式。",
        ),
        "case_2c785834effca374ffb2761e": (
            "偏爱直接下结论",
            "真人先确认月日均是否包含理财，V2 却偏爱根据已有资产直接判断“已经超了”。它低估了本人先核对统计口径的习惯。",
        ),
        "case_01217e2aed559032c384df1f": (
            "漏掉话题前推",
            "V2 选择紧贴“老师资源”的熟人社会回应，真人却把教育话题推进到以后高考。它把紧密承接看得过重。",
        ),
        "case_ce29722235cfe95b9f7cba90": (
            "弱证据二选一",
            "质保和最终价格都能自然承接报价；真人选择问最终价格。V2 没有足够本人证据区分，却仍给出了方向性判断，属于本来就不该强判的边界题。",
        ),
    }

    wrong_cases = []
    for case in blind.get("cases", []):
        case_id = case.get("case_id")
        actual = answer_by_id.get(case_id, {}).get("human_option")
        prediction = prediction_by_id.get(case_id, {})
        if actual and prediction.get("human_option") != actual:
            wrong_cases.append((float(prediction.get("confidence") or 0), case, actual, prediction))
    wrong_cases.sort(key=lambda row: row[0], reverse=True)

    cards = []
    for index, (confidence, case, actual, prediction) in enumerate(wrong_cases, start=1):
        case_id = case.get("case_id")
        predicted = prediction.get("human_option", "?")
        prior_messages = _audit_messages(
            case.get("context_original", []),
            ("<conversation_history>", "<history>", "<consumed_self_replies"),
        )
        target_messages = _audit_messages(
            case.get("context_original", []), ("<messages_to_reply>", "<unread>")
        )
        call = call_by_id.get(case_id, {})
        request_messages = call.get("request_messages") or []
        system_prompt = next(
            (row.get("content", "") for row in request_messages if row.get("role") == "system"), ""
        )
        user_prompt = next(
            (row.get("content", "") for row in request_messages if row.get("role") == "user"), ""
        )
        response = call.get("response") or {}
        boundary, explanation = review_notes.get(
            case_id, ("待复核边界", "这道误判尚未形成稳定归因。")
        )
        emphasis = index <= 4
        cards.append(f'''
        <section class="card judge-case wrong v2-misclassification {'priority' if emphasis else ''}" data-v2-misclassification="true">
          <div class="judge-case-title"><b>{'重点误判' if emphasis else '误判'} {index} · {_audit_escape(boundary)}</b><span class="judge-badge wrong">置信度 {confidence:.2f}</span></div>
          <div class="judge-context">
            <div class="judge-section-label">完整聊天上下文（{len(prior_messages)} 条）</div>{_audit_chat_rows(prior_messages)}
            <div class="judge-section-label target">需要回复的消息</div>{_audit_chat_rows(target_messages)}
          </div>
          <div class="judge-options">
            <div class="judge-option {'human' if actual == 'A' else ''}"><b>回复 A{'（真人）' if actual == 'A' else ''}</b>{_audit_reply_rows(case.get('option_A', []))}</div>
            <div class="judge-option {'human' if actual == 'B' else ''}"><b>回复 B{'（真人）' if actual == 'B' else ''}</b>{_audit_reply_rows(case.get('option_B', []))}</div>
          </div>
          <div class="judge-verdict"><b>真人原回复：{actual}</b> · V2 选择：<b>{_audit_escape(predicted)}</b> · 置信度 {confidence:.2f}
            <span><b>V2 原始理由：</b>{_audit_escape(prediction.get('reason', ''))}</span>
            <span class="review-explanation"><b>本次审阅：</b>{_audit_escape(explanation)}</span>
          </div>
          <details class="judge-raw"><summary>查看本题原始提示词与完整调用记录</summary>
            <h3>完整原始上下文</h3>{_audit_named_values(case.get('context_original', []))}
            <h3>Judge system prompt</h3>{_audit_pre(system_prompt, '60rem')}
            <h3>Judge user prompt（仅本题）</h3>{_audit_pre(user_prompt, '60rem')}
            <h3>调用信息</h3>{_audit_named_values({
                'case_id': case_id,
                '请求时间（UTC）': call.get('recorded_at_utc'),
                '请求 SHA256': call.get('request_sha256'),
                '耗时毫秒': call.get('elapsed_ms'),
                '答案字段': call.get('answer_fields_in_request'),
                '其他测试题 ID': call.get('other_test_case_ids_in_request'),
                '工具配置': call.get('tools_configured'),
                '工具调用事件': call.get('tool_events'),
                '命令参数': response.get('argv'),
                '返回码': response.get('returncode'),
                'prompt 传输方式': response.get('prompt_transport'),
            })}
            <h3>模型原始回复</h3>{_audit_pre(call.get('raw_response', ''))}
            <h3>stdout</h3>{_audit_pre(response.get('stdout', ''), '40rem')}
            <h3>stderr</h3>{_audit_pre(response.get('stderr', ''), '20rem')}
            <h3>原始事件</h3>{_audit_named_values(response.get('events', []))}
            <h3>事件解析错误</h3>{_audit_named_values(response.get('event_parse_errors', []))}
            <h3>调用错误</h3>{_audit_named_values(call.get('error'))}
          </details>
        </section>''')

    return f'''
    <style>
      .v2-review-summary{{border-left:4px solid var(--red)}}
      .v2-review-summary p{{color:var(--muted);font-size:13px;line-height:1.6}}
      .v2-review-summary strong{{color:var(--text)}}
      .v2-misclassification.priority{{border-width:2px;border-color:var(--red);border-left-width:5px}}
      .review-explanation{{color:var(--text)!important;background:rgba(248,81,73,.08);padding:9px 10px;border-radius:8px}}
    </style>
    <div class="card v2-review-summary">
      <h2 style="margin:0 0 8px">Codex V2 的 14 个误判</h2>
      <p><strong>结论：V2 可以继续做诊断辅助，但还不足以稳定、单独决定回复版本胜负。</strong><br>
      这 100 题是复用题，不是正式隐藏终测；V2 得分 86/100，仍有 14 个误判。错误集中在偏爱更完整、更理想的回复，以及漏掉真人跳回前文、重复确认、自我修正和临时换话题。前三个 0.98 和一个 0.97 的高置信误判说明它还存在明显过度自信。</p>
    </div>
    <h2 style="margin:20px 0 12px">4 个高置信误判</h2>{''.join(cards[:4])}
    <h2 style="margin:20px 0 12px">其余 10 个误判</h2>{''.join(cards[4:])}
    '''


def _judge_audit_html() -> str:
    v2_review = _v2_misclassification_review_html()
    boundary_candidate = _boundary_candidate_judge_audit_html()
    historical = _historical_judge_audit_html()
    if boundary_candidate:
        return v2_review + (
            '<details class="card" style="margin-top:24px"><summary><b>展开 100 题完整总览与候选 Judge 对照</b></summary>'
            + boundary_candidate + '</details>'
            '<details class="card" style="margin-top:24px"><summary><b>旧 30 题 API 批次（已作废）</b></summary>'
            + _api_independent_judge_audit_html() + '</details>'
            '<details class="card"><summary><b>旧 27/30 批次（已作废）</b></summary>'
            + _revised_judge_audit_html() + '</details>'
            '<details class="card"><summary><b>更早的 6 题批次（已作废）</b></summary>'
            + historical + '</details>'
        )
    if _load_private_json("human_ai_judge_api_independent_audit.json"):
        return _api_independent_judge_audit_html() + (
            '<details class="card" style="margin-top:24px"><summary><b>旧 27/30 批次（已作废）</b></summary>'
            + _revised_judge_audit_html() + '</details>'
            '<details class="card"><summary><b>更早的 6 题批次（已作废）</b></summary>'
            + historical + '</details>'
        )
    if not _load_private_json("human_ai_judge_revised_blind.json"):
        return historical
    return _revised_judge_audit_html() + (
        '<details class="card" style="margin-top:24px"><summary><b>旧 6 题批次（已作废）</b></summary>'
        + historical + '</details>'
    )

def _embed_benchmark(html_path: Path, title: str, active: str) -> str:
    """提取独立 HTML 的 body + style 内容，嵌入 admin 框架。"""
    if not html_path.exists():
        return _page(title, "<p style='color:var(--muted)'>No data. Run scripts/generate_benchmark_dashboard.py</p>", active)
    raw = html_path.read_text(encoding="utf-8")
    # 提取 <style>...</style>
    style_start = raw.find("<style>")
    style_end = raw.find("</style>")
    style = raw[style_start:style_end + 8] if style_start >= 0 and style_end > style_start else ""
    # 提取 <body>...</body>
    body_start = raw.find("<body>")
    body_end = raw.find("</body>")
    if body_start >= 0 and body_end > body_start:
        body = raw[body_start + 6:body_end]
        # 去掉独立报告的重复标题，但保留标题前后的布局容器。
        body = re.sub(r"<h1\b[^>]*>.*?</h1>", "", body, count=1, flags=re.DOTALL | re.IGNORECASE)
        return _page(title, style + body, active)
    return _page(title, raw, active)


def _judge_bot_iteration_ledger_html() -> str:
    return '''
    <style>
    .ledger-summary{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
    .ledger-summary .card b{display:block;font-size:25px;color:var(--blue)}
    .ledger-list{display:grid;gap:10px}.ledger-row{border-left:4px solid var(--border)}
    .ledger-row.failed{border-left-color:var(--red)}.ledger-row.partial{border-left-color:var(--yellow)}
    .ledger-row.invalid{border-left-color:var(--muted)}.ledger-tag{display:inline-block;padding:2px 7px;border-radius:999px;background:rgba(139,148,158,.15);font-size:12px;margin-left:5px}
    .ledger-note{border-left:4px solid var(--yellow)}
    @media(max-width:700px){.ledger-summary{grid-template-columns:1fr}.ledger-row p{margin-top:7px}}
    </style>
    <div class="ledger-summary">
      <div class="card"><b>5 轮</b>Bot 真实能力迭代<br><span style="color:var(--muted)">第 3 轮形成当前第二轮 C0 基线；随后两轮候选均未采用</span></div>
      <div class="card"><b>4 轮</b>Judge 真实能力迭代<br><span style="color:var(--muted)">暂无正式通过版；Codex V2 仅作辅助诊断</span></div>
    </div>
    <div class="card ledger-note"><b>怎么算一轮</b><p>只有回复生成方法或 Judge 判断方法发生实质变化，并完成对照验证，才算能力迭代。数据准备、补题、重跑、流程审计、修复上下文、公平封存和页面改版都不计轮次。</p></div>
    <h2>Bot 迭代</h2>
    <div class="ledger-list">
      <div class="card ledger-row invalid"><b>C0 · 原代码基线</b><span class="ledger-tag">不算迭代</span><p>用于所有候选的对照。当前仍保留，未被替换。</p></div>
      <div class="card ledger-row failed"><b>Bot 第 1 轮 · S2</b><span class="ledger-tag">整体变差</span><p>参与判断和规则堆叠只有局部收益，同时增加生硬、误读和编造风险。拒绝替换 C0。</p></div>
      <div class="card ledger-row failed"><b>Bot 第 2 轮 · S4</b><span class="ledger-tag">未证明提升</span><p>尝试为场景匹配并注入真人原话；话题匹配和结果完整性不足，路线结束。拒绝替换 C0。</p></div>
      <div class="card ledger-row partial"><b>Bot 第 3 轮 · C0 Guardrail V2</b><span class="ledger-tag">9 胜 / 3 负 / 8 平</span><p>20 条完整上下文人工匿名 A/B：群聊 6/1/7，私聊 3/2/1。减少旧梗复用、显然信息重复解释、机制编造，并改善当前话题承接与身份纠正；但可信历史明确为“属狗”时仍答“属马”，所以暂不整体替换 C0。</p><p><b>下一轮唯一优先项：</b>让可信私人事实真正约束生成。旧支线回捞、亲密私聊过度展开、语义扩大已记录为后续候选，不在同一轮混改。</p><p><b>方向发现方法：</b>冻结 Judge 只负责提名 → 人工查看完整上下文做匿名 A/B → 将胜负理由按稳定错因聚类 → 只保留跨案例复现或有可靠历史冲突的方向 → 每轮只改一个问题并用固定测试集复验。难分案例不强行总结。</p><p><a href="/benchmark/judge/group">查看第二轮全部案例 →</a></p></div>
      <div class="card ledger-row failed"><b>Bot 第 4 轮 · C0 Guardrail V3</b><span class="ledger-tag">3 胜 / 7 负 / 10 平</span><p>只增加抽象双关与字面解释约束；目标问题没有修好，又带来旧支线回捞等回归。候选不采用，回到第二轮 C0 基线。</p></div>
      <div class="card ledger-row failed"><b>Bot 第 5 轮 · C0 Guardrail V4</b><span class="ledger-tag">5 胜 / 5 负 / 10 平</span><p>只增加上下文证据门控。部分案例更短、少猜机制与用途，但总体没有净胜；可信私人事实仍答错，无依据的平台行为断言仍出现。候选不采用，继续保留第二轮 C0 基线。</p><p><b>发现方法：</b>Judge 只提名 → 人工完整上下文复核 → 将机制幻觉、动机猜测、主动解释等稳定错因聚类 → 单变量验证；失败方向与发现过程同样保留。</p></div>
    </div>
    <h2>Judge 迭代</h2>
    <div class="ledger-list">
      <div class="card ledger-row invalid"><b>Judge 第 1 轮 · 批量 Codex 30 题</b><span class="ledger-tag">27/30 · 作废</span><p>30 题合批，可能跨题学习风格线索，并存在会话、工具和既有讨论污染风险。成绩不可信。</p></div>
      <div class="card ledger-row failed"><b>Judge 第 2 轮 · 每题独立模型 API</b><span class="ledger-tag">15/30 · 失败</span><p>切断跨题共享后只有随机线附近，证明原 27/30 不能代表真实能力。</p></div>
      <div class="card ledger-row partial"><b>Judge 第 3 轮 · Codex V2</b><span class="ledger-tag">86/100 · 仅诊断</span><p>能识别多数题，但复用了旧 100 题，并有 14 个误判、4 个高置信误判。作为临时辅助工具，不是正式隐藏终测通过版。</p></div>
      <div class="card ledger-row failed"><b>Judge 第 4 轮 · 群聊提示词候选</b><span class="ledger-tag">局部升、整体降</span><p>旧群聊 13 题从 12/13 到 13/13，但旧整体 100 题从 89/100 降到 88/100；修正 4 题同时新增 5 个误判。拒绝升级。</p></div>
    </div>
    <h2>最新公平验收（不算第 5 轮）</h2>
    <div class="card ledger-row failed"><b>冻结 Codex V2 · 全新公平 100 题：83/100</b><p>100 题全部有效；C0 与 Judge 上下文范围一致；每题独立进程；答案字段 0、工具调用 0；真人位置 A/B 为 50/50，Judge 选择 A/B 为 49/51。共 17 个误判，其中 13 个是置信度不低于 0.8 的高置信误判，因此未通过独立验收。</p><p><a href="/benchmark/judge/strict-historical">查看 100 题完整上下文、真人/C0 回复、选择与理由 →</a></p></div>
    <h2>明确不计入能力轮次</h2>
    <div class="card"><p>历史聊天导出、真人/Bot 身份核对、共同富裕群纠错、柚子群日期调整、样本扩充、120/240 分组、HTML 手机版调整、独立域名、Codex 工具隔离、上下文对齐、18 题公平替换、超时重跑和事实核查记录补齐。</p></div>
    <div class="card ledger-note"><b>当前结论</b><p>Judge 已按当前门槛冻结，C0 正在迭代，Bot 不启动。第四轮证据门控候选没有净收益，继续保留第二轮 C0 基线；唯一固定测试集同步补齐，但不阻塞 C0。</p></div>
    '''

@app.get("/benchmark/judge", response_class=HTMLResponse)
def benchmark_judge():
    return HTMLResponse(_page("Judge Quality Benchmark", _judge_audit_html(), "/benchmark/judge"))


@app.get("/benchmark/judge/iteration-ledger", response_class=HTMLResponse)
def benchmark_judge_iteration_ledger():
    return HTMLResponse(_page("Judge / Bot 真实迭代账单", _judge_bot_iteration_ledger_html(), "/benchmark/judge/iteration-ledger"))

@app.get("/benchmark/judge/group", response_class=HTMLResponse)
def benchmark_group_chat_judge():
    return HTMLResponse(_embed_benchmark(GROUP_CHAT_JUDGE_REVIEW, "群聊 Judge 审计", "/benchmark/judge/group"))

@app.get("/benchmark/judge/strict-historical", response_class=HTMLResponse)
def benchmark_strict_historical_judge():
    return HTMLResponse(_embed_benchmark(STRICT_HISTORICAL_JUDGE_REVIEW, "Judge 独立验收", "/benchmark/judge/strict-historical"))

@app.get("/benchmark/judge/strict-historical/case/{case_id}", response_class=HTMLResponse)
def benchmark_strict_historical_judge_case(case_id: str):
    report_dir = STRICT_HISTORICAL_JUDGE_REVIEW.parent
    generation_path = report_dir / ("fair_generation_results.json" if (report_dir / "fair_generation_results.json").exists() else "generation_results.json")
    judge_path = report_dir / ("fair_judge_results.json" if (report_dir / "fair_judge_results.json").exists() else "judge_results.json")
    generation = _json.loads(generation_path.read_text(encoding="utf-8"))
    judged = _json.loads(judge_path.read_text(encoding="utf-8"))
    case = next((row for row in generation.get("cases", []) if str(row.get("internal_id") or row.get("case_id")) == case_id), None)
    judge = next((row for row in judged.get("cases", []) if row.get("case_id") == case_id), None)
    if case is None or judge is None:
        raise HTTPException(status_code=404, detail="Judge case not found")
    judge_audit = {
        "request_messages": judge.get("request_messages"),
        "raw_response": judge.get("raw_response"),
        "tool_events": judge.get("tool_events"),
        "subprocess_audit": judge.get("subprocess_audit"),
    }
    c0_audit = {"generation": case.get("generation"), "validity": case.get("validity")}
    content = (
        f'<div class="card"><b>{html.escape(str(case.get("chat_name") or case_id))}</b>'
        f'<p>Judge：{html.escape(str(judge.get("selected_option")))} · '
        f'{"正确" if judge.get("correct") else "误判"} · 置信度 '
        f'{html.escape(str(judge.get("confidence")))}</p></div>'
        '<div class="card"><h2>Judge 原始提示词与完整调用</h2>'
        f'{_audit_pre(_json.dumps(judge_audit, ensure_ascii=False, indent=2), "70vh")}</div>'
        '<div class="card"><h2>C0 生成提示词、工具与排除审计</h2>'
        f'{_audit_pre(_json.dumps(c0_audit, ensure_ascii=False, indent=2), "70vh")}</div>'
    )
    return HTMLResponse(_page("单题完整调用审计", content, "/benchmark/judge/strict-historical"))

@app.get("/benchmark/judge/group/youzi-identity", response_class=HTMLResponse)
def benchmark_youzi_identity_review():
    return HTMLResponse(_embed_benchmark(YOUZI_IDENTITY_REVIEW, "柚子群身份人工确认", "/benchmark/judge/group/youzi-identity"))

@app.get("/benchmark/reply", response_class=HTMLResponse)
def benchmark_reply():
    return HTMLResponse(_embed_benchmark(BENCHMARK_REPLY, "Bot 回复质量 Benchmark", "/benchmark/reply"))

@app.get("/benchmark/reply/current-c0", response_class=HTMLResponse)
def benchmark_current_c0_historical_ab():
    return HTMLResponse(_embed_benchmark(CURRENT_C0_HISTORICAL_AB, "C0 固定 240 题正式 A/B", "/benchmark/reply/current-c0"))

@app.post("/api/refresh-benchmark")
def refresh_benchmark():
    script = str(Path(__file__).parent.parent / "scripts" / "generate_benchmark_dashboard.py")
    try:
        subprocess.run(["python3", script], timeout=60, capture_output=True)  # nosec
        return JSONResponse({"success": True})
    except Exception as e:
        _logger.warning("刷新 benchmark 失败: %s", e)
        return JSONResponse({"success": False, "error": "刷新失败，请查看服务端日志"})


# ── 实验 A/B 对比 ──

def _reply_ab_plan_html() -> str:
    """首轮回复质量 A/B 的待确认方案；此页不执行实验。"""
    return """
    <section class="card" style="border-color:var(--blue)">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <h2 style="font-size:18px">首轮回复质量 A/B 方案</h2>
        <span style="color:var(--yellow);border:1px solid var(--yellow);border-radius:999px;padding:2px 8px;font-size:12px">待确认</span>
      </div>
      <p style="margin-top:8px;line-height:1.7">目标是让完整回复更像本人、更自然风趣，不再把“该不该插话”作为本轮主变量。</p>
      <p style="color:var(--muted);font-size:13px;line-height:1.7">当前 C0 继续冻结且可恢复；候选版从同一 C0 快照独立产生，不在当前版本里切换开关或覆盖它。</p>
    </section>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">
      <section class="card">
        <h3 style="font-size:15px;color:var(--blue)">A · Control</h3>
        <p style="margin-top:8px;line-height:1.7">完整冻结当前 C0：当前模型、参数、生产上下文、工具、Wiki/RAG、Self-Refine、回复决策与 few-shot 机制均不变。</p>
      </section>
      <section class="card">
        <h3 style="font-size:15px;color:var(--green)">B · Treatment（唯一变量）</h3>
        <p style="margin-top:8px;line-height:1.7">优先只使用<strong>同聊天、2026 年 4 月前、可信的真人回复</strong>作为风格 few-shot；数量不足时宁可少放或不放，不跨聊天硬补。</p>
        <p style="margin-top:8px;color:var(--muted);font-size:13px;line-height:1.7">示例只用于学节奏、表达动作和临场反转；不迁移具体笑点、事实、数字、人物标签或关系。</p>
      </section>
    </div>

    <section class="card">
      <h3 style="font-size:15px">已发现的生产差异（实验前必须核对）</h3>
      <ul style="margin:10px 0 0 20px;line-height:1.8">
        <li>当前检索不是“只用同聊天”：同 chat 目前只加 1.5 分，不是硬过滤。</li>
        <li><code>ENABLE_PERSONA_FEW_SHOTS</code> 的实际生产开关状态需在回放前确认。</li>
        <li>离线回放必须与生产的模型、采样参数、上下文截取、工具与 Wiki/RAG 一致。</li>
      </ul>
      <p style="margin-top:10px;color:var(--yellow);line-height:1.7">任一项无法对齐，就停在数据检查阶段：先汇报差异，等你确认后才继续。</p>
    </section>

    <section class="card">
      <h3 style="font-size:15px">怎么比</h3>
      <ol style="margin:10px 0 0 20px;line-height:1.8">
        <li>固定同一批真实聊天场景，保留生产实际可见的完整上下文。样本建议<strong>群聊 70% / 私聊 30%</strong>，以群聊为主、私聊作回归保护。</li>
        <li>A/B 使用同一模型、参数、工具、Wiki/RAG 和回放次数；每个场景至少回放 3 次。</li>
        <li>左右匿名随机展示，人工逐例选“A 更好 / B 更好 / 相当”。</li>
        <li>结果页必须展示：完整上下文、目标消息、同场景真人原话（如果确实存在）、A/B 原始回复、few-shot 是否命中及来源 ID、人工理由和 Judge 诊断。</li>
      </ol>
      <div style="margin-top:12px;padding:12px;border:1px solid var(--border);border-radius:8px">
        <b>群聊样本必须分层覆盖</b>
        <p style="margin-top:6px;color:var(--muted);line-height:1.7">被点名或明确提问、跨消息承接前文、多人连续发言与对象识别、可回可不回、熟人玩笑互损、事实问答。各类单独报告胜/负/相当，避免总分掩盖某类群聊的明显退步。</p>
      </div>
    </section>

    <section class="card">
      <h3 style="font-size:15px">怎么判断质量</h3>
      <div class="metrics">
        <div class="metric"><div class="label">主要目标</div><div style="margin-top:6px">像本人，不显 AI</div></div>
        <div class="metric"><div class="label">上下文</div><div style="margin-top:6px">接住具体语境和关系</div></div>
        <div class="metric"><div class="label">幽默</div><div style="margin-top:6px">自然、新鲜、像本人</div></div>
        <div class="metric"><div class="label">价值</div><div style="margin-top:6px">有信息、情绪或推进作用</div></div>
      </div>
      <p style="line-height:1.7">不把“更长、更礼貌、更完整”当成更好；粗话、互损和攻击性需结合真实关系与语境判断。可回可不回的场景不做过度优化。</p>
    </section>

    <section class="card" style="border-color:var(--red)">
      <h3 style="font-size:15px;color:var(--red)">硬门槛</h3>
      <ul style="margin:10px 0 0 20px;line-height:1.8">
        <li>新增事实幻觉：直接淘汰；明确的玩笑不机械算幻觉。</li>
        <li>新增真实行动承诺（如真会转账、发红包）：直接淘汰。</li>
        <li>旧梗、重复、明显失败率或延迟恶化：不采纳。</li>
      </ul>
    </section>

    <section class="card">
      <h3 style="font-size:15px">Judge V2 的位置</h3>
      <p style="margin-top:8px;line-height:1.7">Codex V2 只辅助标出可能的 AI 腔、过度理想化、事实与承诺风险。它不能单独决定 A/B 胜负，也不以自身分数代替完整聊天体验的人工判断。</p>
    </section>

    <section class="card" style="border-color:var(--yellow)">
      <h3 style="font-size:15px">放行条件</h3>
      <p style="margin-top:8px;line-height:1.7">B 在群聊子集上的胜数必须明确多于负数，私聊不得明显退步；严重问题不增加，且真人感与幽默的改善不能靠更频繁插话或更长回复换取。结果不稳定就不采纳。</p>
      <p style="margin-top:8px;color:var(--yellow)">本页只是方案；你确认后才会创建候选版和开始离线回放。</p>
    </section>
    """


ADVERSARIAL_PROTOCOL = (
    Path(__file__).parent.parent
    / "docs"
    / "03-guides"
    / "JUDGE_BOT_ADVERSARIAL_PROTOCOL_V1.json"
)
ADVERSARIAL_SAMPLE_PLAN = (
    Path(__file__).parent.parent
    / "docs"
    / "03-guides"
    / "JUDGE_BOT_ADVERSARIAL_SAMPLE_PLAN_V1.json"
)
GROUP_JUDGE_SAMPLE_MANIFEST = (
    Path(__file__).parent.parent
    / "data"
    / "reports"
    / "private"
    / "group_chat_judge_review"
    / "manifest.json"
)
JUDGE_DEV_REPLAY = (
    Path(__file__).parent.parent
    / "data"
    / "reports"
    / "private"
    / "human_ai_judge_benchmark_v2"
    / "dev_replay.json"
)
ADVERSARIAL_HIDDEN_PUBLIC_SUMMARY = (
    Path(__file__).parent.parent
    / "data"
    / "reports"
    / "private"
    / "adversarial_hidden_holdout_v1"
    / "public_summary.json"
)
ADVERSARIAL_YOUZI_PUBLIC_SUMMARY = (
    Path(__file__).parent.parent
    / "data"
    / "reports"
    / "private"
    / "adversarial_stage2_youzi_quota_v1"
    / "public_summary.json"
)
ADVERSARIAL_STAGE2_PUBLIC_SUMMARY = (
    Path(__file__).parent.parent
    / "data"
    / "reports"
    / "private"
    / "adversarial_stage2_holdout_v1"
    / "public_summary.json"
)
REPLY_PRODUCTION_ALIGNMENT = (
    Path(__file__).parent.parent
    / "docs"
    / "03-guides"
    / "REPLY_PRODUCTION_ALIGNMENT_V1.json"
)


def _reply_production_alignment_html() -> str:
    """Render the read-only production-alignment audit without private cases."""
    try:
        audit = _json.loads(REPLY_PRODUCTION_ALIGNMENT.read_bytes())
    except (OSError, _json.JSONDecodeError) as exc:
        _logger.warning("读取回复生产一致性审计失败: %s", exc)
        return '<section class="card">生产一致性审计暂时无法读取。</section>'

    dataset = audit["stage_2_dataset"]
    execution = audit["execution"]
    baseline = audit["baseline"]
    contamination = dataset["contamination"]
    missing_fields = "、".join(dataset["missing_production_fields"])
    differences = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in audit["known_legacy_replay_differences"]
    )
    requirements = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in audit["production_alignment_requirements"]
    )
    proposal = audit["first_small_step_proposal"]
    owner_decision = audit["owner_decision"]
    replay = audit["single_step_replay_review"]
    preview = replay["synthetic_preview"]
    verification = replay["verification"]
    captured_evidence = "".join(
        f"<li>{html.escape(item)}</li>" for item in replay["captured_evidence"]
    )
    replay_differences = "".join(
        f"<li>{html.escape(item)}</li>" for item in replay["remaining_differences"]
    )
    preview_context = "".join(
        f'<div class="align-message">{html.escape(message)}</div>'
        for message in preview["context"]
    )
    return f"""
    <style>
      .align-hero{{border-color:var(--red);background:rgba(248,81,73,.08)}}
      .align-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
      .align-stat{{border:1px solid var(--border);border-radius:8px;padding:12px}}
      .align-stat strong{{display:block;font-size:24px;color:var(--blue)}}
      .align-stat span{{font-size:12px;color:var(--muted);line-height:1.55}}
      .align-list{{margin:8px 0 0 20px;line-height:1.75}}
      .align-code{{font-size:12px;color:var(--muted);overflow-wrap:anywhere;line-height:1.65}}
      .align-message{{border-left:3px solid var(--border);padding:8px 10px;margin-top:8px;line-height:1.6}}
      .align-reply{{border:1px solid var(--blue);border-radius:8px;padding:12px;margin-top:10px;line-height:1.6}}
      @media(max-width:700px){{.align-grid{{grid-template-columns:1fr}}.align-list{{margin-left:18px}}}}
    </style>

    <section class="card align-hero">
      <h2 style="font-size:20px">回复评测生产一致性审计</h2>
      <p style="margin-top:8px;line-height:1.75"><b>结论：当前不能进入正式 A/B。</b>现有 240 条可做风格、Judge 和流程诊断，但没有一条具备生产回放所需字段。</p>
      <p class="align-code" style="margin-top:8px">C0：{html.escape(baseline['tag'])} · {html.escape(baseline['commit'])}<br>Generator SHA256：{html.escape(baseline['generator_sha256'])}</p>
    </section>

    <section class="card">
      <h3>现有样本能做什么</h3>
      <div class="align-grid" style="margin-top:10px">
        <div class="align-stat"><strong>{dataset['cases']}</strong><span>验证集 / 流程诊断集<br>不是全新隐藏终验集</span></div>
        <div class="align-stat"><strong>{dataset['context_messages']}</strong><span>上下文消息<br>仅有 speaker / content</span></div>
        <div class="align-stat"><strong>{dataset['fully_production_comparable_cases']}/{dataset['cases']}</strong><span>严格生产可比<br>因此禁止正式 Bot A/B</span></div>
        <div class="align-stat"><strong>已读取</strong><span>分类任务曾看到真人答案<br>不得再称 fresh hidden</span></div>
      </div>
      <p class="align-code" style="margin-top:10px">缺失生产字段：{html.escape(missing_fields)}</p>
    </section>

    <section class="card" style="border-color:var(--yellow)">
      <h3>重复任务污染证据</h3>
      <p style="margin-top:8px;line-height:1.75">分层整理任务把每题的完整上下文和 <b>human_reply</b> 一起交给了 {html.escape(contamination['curator_model'])}。虽然逐题独立、无工具、无跨题共享会话，但答案已经被迭代步骤读取。</p>
      <p style="margin-top:8px;color:var(--yellow)">处理：保留全部历史证据；240 条降级为验证与流程诊断，不删除、不包装成绩。</p>
    </section>

    <section class="card">
      <h3>旧离线 replay 与生产的差异</h3>
      <ul class="align-list">{differences}</ul>
      <p style="margin-top:10px;color:var(--yellow);line-height:1.7">任何差异都必须先展示给你确认；未经确认，不执行 replay。</p>
    </section>

    <section class="card">
      <h3>以后可比 replay 的最低条件</h3>
      <ul class="align-list">{requirements}</ul>
      <p style="margin-top:10px;line-height:1.7">链路固定为 <b>Policy → Store → Generator</b>；保持生产中的“同聊天加权” few-shot，不擅自改成严格同聊天。</p>
      <p style="margin-top:10px;color:var(--green);line-height:1.7"><b>已确认：严格重建。</b>另建生产一致的全新样本与回放链；现有 240 条永久保留为诊断集。完成一致性验收前不执行 replay，不进入正式 A/B。</p>
    </section>

    <section class="card" style="border-color:var(--blue)">
      <h3>单步回放基础 · 已验证</h3>
      <p style="margin-top:8px;line-height:1.75">归档格式：<b>{html.escape(replay['schema'])}</b>。默认关闭，只有显式设置归档目录才启用；当前没有真实生产样本。</p>
      <p style="margin-top:8px;line-height:1.75"><b>停止点：</b>{html.escape(replay['cutoff'])}。</p>
      <p style="margin-top:8px;line-height:1.75"><b>首轮合格范围：</b>{html.escape(replay['eligible_subset'])}。</p>
      <ul class="align-list">{captured_evidence}</ul>
    </section>

    <section class="card">
      <h3>最小假回放 · 仅用于看格式</h3>
      <p class="align-code">{html.escape(preview['label'])}</p>
      {preview_context}
      <div class="align-reply"><b>生成回复</b><br>{html.escape(preview['reply'])}</div>
      <div class="align-grid" style="margin-top:10px">
        <div class="align-stat"><strong>{preview['model_calls']}</strong><span>独立模型调用记录（假客户端）</span></div>
        <div class="align-stat"><strong>{preview['tool_execution_count']}</strong><span>工具执行</span></div>
        <div class="align-stat"><strong>{'是' if preview['eligible_for_formal_replay'] else '否'}</strong><span>符合首轮回放字段要求</span></div>
        <div class="align-stat"><strong>发送前</strong><span>截断于 {html.escape(preview['stopped_before'])}</span></div>
      </div>
    </section>

    <section class="card" style="border-color:var(--yellow)">
      <h3>仍然存在的生产差异</h3>
      <ul class="align-list">{replay_differences}</ul>
      <p style="margin-top:10px;color:var(--yellow);line-height:1.7">因此这一步只证明归档证据链可用，不证明评测可靠，也不允许进入正式 A/B。</p>
    </section>

    <section class="card">
      <h3>本次离线验证</h3>
      <div class="align-grid" style="margin-top:10px">
        <div class="align-stat"><strong>{verification['tests']}/{verification['tests']}</strong><span>测试通过</span></div>
        <div class="align-stat"><strong>{verification['fresh_qualified_samples']}</strong><span>全新合格生产样本</span></div>
        <div class="align-stat"><strong>{verification['real_model_calls']}</strong><span>真实模型调用</span></div>
        <div class="align-stat"><strong>{'是' if verification['bot_started'] else '否'}</strong><span>Bot 启动</span></div>
      </div>
    </section>

    <section class="card" style="border-color:var(--blue)">
      <h3>流程可靠后唯一建议的小步</h3>
      <p style="margin-top:8px;line-height:1.75">从完整真人聊天提炼少量“上下文触发的表达/幽默动作”：承接前文、轻微反转、短促自嘲、熟人互损尺度。只迁移抽象动作，不迁移具体旧梗、事实、数字、人物关系或现实行动。</p>
      <p style="margin-top:8px;color:var(--yellow)">这只是待确认方向，不是已批准实验；事实编造、现实承诺、错对象或旧梗复用任一退步即淘汰。</p>
    </section>

    <section class="card">
      <h3>本轮公开状态</h3>
      <ul class="align-list">
        <li>模型调用：{execution['model_calls']}。</li>
        <li>Bot 启动：{'是' if execution['bot_started'] else '否'}。</li>
        <li>C0 修改：{'是' if execution['c0_modified'] else '否'}。</li>
        <li>正式 A/B：{'允许' if execution['formal_ab_allowed'] else '禁止'}。</li>
        <li>生产对齐选择：{'严格重建（已确认）' if owner_decision['decision'] == 'strict_rebuild' else html.escape(owner_decision['decision'])}。</li>
      </ul>
    </section>
    """


def _adv_messages(messages: List[str], empty_text: str = "无") -> str:
    if not messages:
        return f'<div class="adv-empty">{html.escape(empty_text)}</div>'
    return "".join(
        f'<div class="adv-message">{html.escape(str(message))}</div>'
        for message in messages
    )


def _adv_prompt_chat_context(case: Dict[str, Any]) -> List[str]:
    """Extract every history/unread message actually supplied to C0."""
    prompt = (
        case.get("c0_trace", {})
        .get("generation_trace", {})
        .get("user_prompt", "")
    )
    messages: List[str] = []
    for section_name in ("history", "unread"):
        section = re.search(
            rf"<{section_name}>(.*?)</{section_name}>", prompt, re.DOTALL
        )
        if not section:
            continue
        for raw_message in re.findall(
            r"<message\b[^>]*>(.*?)</message>", section.group(1), re.DOTALL
        ):
            plain = re.sub(r"<[^>]+>", "", raw_message)
            messages.append(html.unescape(plain).strip())
    return [message for message in messages if message]


def _adv_case_details(
    *,
    case_id: str,
    title: str,
    badges: List[str],
    context: List[str],
    human_reply: Optional[List[str]] = None,
    c0_reply: Optional[List[str]] = None,
    note: str = "",
) -> str:
    badge_html = "".join(
        f'<span class="adv-badge">{html.escape(badge)}</span>' for badge in badges
    )
    reply_html = ""
    if human_reply is not None:
        reply_html += (
            '<div class="adv-answer"><b>真人原回复</b>'
            f'{_adv_messages(human_reply, "真人选择沉默")}</div>'
        )
    if c0_reply is not None:
        reply_html += (
            '<div class="adv-answer"><b>C0 原回复</b>'
            f'{_adv_messages(c0_reply, "C0 选择沉默")}</div>'
        )
    note_html = f'<p class="adv-note">{html.escape(note)}</p>' if note else ""
    return f"""
      <details class="adv-case">
        <summary><span>{html.escape(title)}</span><small>{html.escape(case_id)}</small></summary>
        <div class="adv-badges">{badge_html}</div>
        {note_html}
        <div class="adv-answer"><b>完整聊天上下文</b>{_adv_messages(context)}</div>
        {reply_html}
      </details>
    """


def _adversarial_sample_inventory_html(sample_plan: Dict[str, Any]) -> str:
    try:
        summary = _json.loads(ADVERSARIAL_HIDDEN_PUBLIC_SUMMARY.read_bytes())
    except (OSError, _json.JSONDecodeError) as exc:
        _logger.warning("读取 hidden 聚合摘要失败: %s", exc)
        return '<section class="card">hidden 聚合摘要暂时无法读取。</section>'

    coverage = summary["coverage"]
    checks = summary["checks"]
    sealed_at = html.escape(str(summary["sealed_at"]))
    manifest_hash = html.escape(str(summary["manifest_sha256"]))
    return f"""
    <section class="card">
      <h3>第一阶段首筛集已封存（只展示覆盖，不公开题目）</h3>
      <p class="adv-kv" style="margin-top:6px">这 120 条只用于筛掉明显无效方向，不承担最终质量结论。内容、内部 ID、上下文、回复和答案均不在本页展示，也不得进入 Prompt、RAG、Wiki、few-shot、逐题讨论或调参。</p>
      <div class="adv-inventory">
        <div class="adv-stock"><strong>{coverage['total']}</strong><span>首筛封存总数<br>不是最终终验</span></div>
        <div class="adv-stock"><strong>{coverage['group_chat']}</strong><span>群聊<br>占首筛集 2/3</span></div>
        <div class="adv-stock"><strong>{coverage['private_chat']}</strong><span>私聊<br>独立检查退步</span></div>
        <div class="adv-stock"><strong>{coverage['complex_large_group']}</strong><span>复杂大群<br>单独报告</span></div>
        <div class="adv-stock"><strong>{coverage['other_group_chat']}</strong><span>其他群聊<br>补足场景广度</span></div>
        <div class="adv-stock"><strong>{coverage['distinct_chats']}</strong><span>不同聊天来源<br>降低单群偏差</span></div>
        <div class="adv-stock"><strong>{coverage['distinct_event_clusters']}</strong><span>独立事件簇<br>每簇最多一题</span></div>
        <div class="adv-stock"><strong>0</strong><span>首筛指定群<br>封存后不回填</span></div>
      </div>
      <p class="adv-kv" style="margin-top:10px">封存时间：{sealed_at}<br>私有 manifest SHA256：{manifest_hash}</p>
    </section>

    <section class="card">
      <h3>为什么首筛集的指定群仍是 0</h3>
      <ul class="adv-list">
        <li><b>含鲁西西的正确共同富裕群：</b>新导出已找到截止日前可靠容量，但第一阶段早已封存，不能事后把新材料回填进去；新材料只进入第二阶段候选终筛。</li>
        <li><b>柚子群：</b>第一阶段封存时仍按原截止线，因此不能回填；用户后来只对第二阶段柚子群放宽日期，新材料仅作为第二阶段候选容量。</li>
        <li><b>另一个同名共同富裕群：</b>继续完全排除，不进入覆盖数量或结论。</li>
      </ul>
    </section>

    <section class="card">
      <h3>封存检查</h3>
      <ul class="adv-list">
        <li>真人身份：{'通过' if checks['verified_human_identity'] else '未通过'}（本人发送且早于自动化截止时间）。</li>
        <li>完整上下文可复现：{'通过' if checks['complete_context_reproducible'] else '未通过'}。</li>
        <li>事件簇隔离：{'通过' if checks['event_cluster_isolation'] else '未通过'}。</li>
        <li>已知曝光原文精确指纹排除：{'通过' if checks['known_exposure_exact_match_excluded'] else '未通过'}。</li>
        <li>模型调用：{checks['model_calls']}。</li>
      </ul>
      <p class="adv-warning">方案已确认，但尚未进入实验。首筛结果一旦用于迭代，这 120 条就降级为回归集，不能再称为最终 hidden。词法指纹排除不能冒充完整语义去重。</p>
    </section>
    """


def _adversarial_plan_html() -> str:
    """Render the frozen, non-executing Judge–Bot experiment protocol."""
    try:
        raw = ADVERSARIAL_PROTOCOL.read_bytes()
        protocol = _json.loads(raw)
        protocol_hash = hashlib.sha256(raw).hexdigest()
        sample_plan = _json.loads(ADVERSARIAL_SAMPLE_PLAN.read_bytes())
        youzi_summary = _json.loads(ADVERSARIAL_YOUZI_PUBLIC_SUMMARY.read_bytes())
        stage_2_summary = _json.loads(ADVERSARIAL_STAGE2_PUBLIC_SUMMARY.read_bytes())
    except (OSError, _json.JSONDecodeError) as exc:
        _logger.warning("读取 Judge–Bot 实验协议失败: %s", exc)
        return '<section class="card">实验协议暂时无法读取。</section>'

    c0_hash = protocol["baselines"]["bot"]["generator_sha256"]
    stages = sample_plan["staged_holdout_plan"]
    stage_1 = stages["stage_1_screening"]
    stage_2 = stages["stage_2_final_validation"]
    stage_2_strata = sample_plan["stage_2_required_strata"]
    capacity = sample_plan["stage_2_capacity_audit"]
    available = capacity["available"]
    gaps = capacity["gaps"]
    youzi_coverage = youzi_summary["coverage"]
    youzi_manifest_hash = html.escape(str(youzi_summary["manifest_sha256"]))
    stage_2_coverage = stage_2_summary["coverage"]
    stage_2_manifest_hash = html.escape(str(stage_2_summary["manifest_sha256"]))
    return f"""
    <style>
      .adv-hero{{border:1px solid var(--blue);background:linear-gradient(135deg,rgba(88,166,255,.12),rgba(63,185,80,.04))}}
      .adv-status{{display:inline-flex;border:1px solid var(--yellow);color:var(--yellow);border-radius:999px;padding:3px 9px;font-size:12px}}
      .adv-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
      .adv-flow{{display:flex;gap:8px;align-items:stretch;overflow:auto;padding-bottom:4px}}
      .adv-step{{min-width:150px;flex:1;border:1px solid var(--border);border-radius:8px;padding:12px;background:rgba(255,255,255,.02)}}
      .adv-arrow{{align-self:center;color:var(--muted)}}
      .adv-list{{margin:8px 0 0 20px;line-height:1.75}}
      .adv-kv{{font-size:12px;color:var(--muted);overflow-wrap:anywhere;line-height:1.65}}
      .adv-split{{border-left:3px solid var(--blue)}}
      .adv-inventory{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px}}
      .adv-stock{{display:flex;gap:10px;align-items:center;border:1px solid var(--border);border-radius:8px;padding:10px}}
      .adv-stock strong{{font-size:22px;color:var(--blue)}}
      .adv-stock span{{font-size:12px;line-height:1.45;color:var(--muted)}}
      .adv-stock.adv-missing{{border-color:var(--red)}}
      .adv-stock.adv-missing strong{{color:var(--red)}}
      .adv-warning{{margin-top:10px;padding:10px;border-radius:7px;background:rgba(248,81,73,.10);color:#ff9d9d;line-height:1.65}}
      .adv-case{{border:1px solid var(--border);border-radius:8px;margin:8px 0;overflow:hidden}}
      .adv-case summary{{display:flex;justify-content:space-between;gap:8px;padding:11px;cursor:pointer;line-height:1.45}}
      .adv-case summary small{{color:var(--muted);overflow-wrap:anywhere;text-align:right}}
      .adv-badges{{display:flex;gap:5px;flex-wrap:wrap;padding:0 11px 8px}}
      .adv-badge{{font-size:11px;padding:2px 7px;border:1px solid var(--border);border-radius:999px;color:var(--muted)}}
      .adv-note{{margin:0 11px 8px;color:var(--yellow);font-size:12px;line-height:1.6}}
      .adv-answer{{margin:0 11px 11px;padding:10px;border-radius:7px;background:rgba(255,255,255,.025)}}
      .adv-answer>b{{display:block;margin-bottom:7px;font-size:12px}}
      .adv-message{{padding:6px 8px;margin-top:5px;border-left:2px solid var(--blue);background:rgba(88,166,255,.06);font-size:13px;line-height:1.55;overflow-wrap:anywhere;white-space:pre-wrap}}
      .adv-empty{{color:var(--muted);font-size:12px}}
      @media(max-width:700px){{
        .adv-grid{{grid-template-columns:1fr}}
        .adv-flow{{display:block;overflow:visible}}
        .adv-step{{min-width:0;margin-bottom:8px}}
        .adv-arrow{{display:block;text-align:center;transform:rotate(90deg);height:20px}}
        .adv-list{{margin-left:18px}}
        .adv-inventory{{grid-template-columns:1fr 1fr}}
        .adv-case summary{{display:block}}
        .adv-case summary small{{display:block;margin-top:4px;text-align:left}}
      }}
    </style>

    <section class="card adv-hero">
      <span class="adv-status">Stage 2 已降级为流程诊断集 · 禁止正式 A/B</span>
      <h2 style="font-size:20px;margin-top:10px">Judge–Bot 对抗式迭代蓝图</h2>
      <p style="margin-top:8px;line-height:1.75">使用现成 LLM，交替冻结 Judge 和 Bot。目标是让回复更像本人、更自然风趣；“骗过 Judge”只用于制造难例，不是质量结论。</p>
      <div class="adv-kv" style="margin-top:10px">协议 {html.escape(protocol['protocol_id'])} · 状态 {html.escape(protocol['status'])}<br>协议 SHA256：{protocol_hash}<br>C0 SHA256：{c0_hash}</div>
    </section>

    <section class="card">
      <h3>先分清三件事</h3>
      <div class="adv-grid" style="margin-top:10px">
        <div class="adv-step"><b>风格 Judge</b><p class="adv-kv">只做匿名真人/Bot 辨认；不擅自判定“泰州”等事实真假。</p></div>
        <div class="adv-step"><b>事实与边界检查</b><p class="adv-kv">单独核对事实、对象、现实承诺、旧梗和重复；有需要才使用获准证据或工具。</p></div>
        <div class="adv-step"><b>Bot</b><p class="adv-kv">改善完整聊天体验；不能靠复制原话、真人偶然错误或单一 Judge 偏好晋级。</p></div>
        <div class="adv-step"><b>真人审阅</b><p class="adv-kv">最终裁决完整聊天中的真人感、幽默、价值与边界。</p></div>
      </div>
    </section>

    <section class="card">
      <h3>四组数据，聊天线程不能拆</h3>
      <p class="adv-kv" style="margin-top:6px">同一聊天线程、连续时间段、同一事件及语义近重复整体进入一组，不能逐消息随机切分。</p>
      <div class="adv-grid" style="margin-top:10px">
        <div class="adv-step adv-split"><b>D-Jtrain</b><p class="adv-kv">只给 Judge 迭代使用。</p></div>
        <div class="adv-step adv-split"><b>D-Btrain</b><p class="adv-kv">只给 Bot 迭代使用。</p></div>
        <div class="adv-step adv-split"><b>D-dev</b><p class="adv-kv">重复做晋级和回归检查。</p></div>
        <div class="adv-step adv-split"><b>D-validation</b><p class="adv-kv">现有样本已被整理任务读取，只能用于验证与流程诊断。</p></div>
      </div>
      <p style="margin-top:10px;line-height:1.7"><b>群聊优先：</b>hidden 至少 60% 群聊；包含鲁西西的正确“共同富裕群”和柚子群优先核验，但不为满足指定群配额而把已曝光或身份不明内容塞入 hidden；另一个同名共同富裕群排除。</p>
      <p style="margin-top:8px;line-height:1.7"><b>复杂大群必须单独抽样：</b>覆盖短时间多人连续发言、回复对象容易混淆、跨消息承接、话题快速切换及多个子话题并行；这类题单独报告，不能让简单群聊的成绩掩盖退步。</p>
      <p class="adv-kv" style="margin-top:6px">下方只展示验证集的抽样规则、聚合覆盖、状态与哈希，不展示任何题目内容。</p>
    </section>

    {_adversarial_sample_inventory_html(sample_plan)}

    <section class="card">
      <h3>两阶段样本方案</h3>
      <div class="adv-grid" style="margin-top:10px">
        <div class="adv-step"><b>第一阶段 · 120 条首筛</b><p class="adv-kv">已封存：{stage_1['group_chat']} 条群聊、{stage_1['private_chat']} 条私聊，其中 {stage_1['complex_large_group']} 条复杂大群。只用于淘汰明显无效方向，不形成最终质量结论。</p></div>
        <div class="adv-step"><b>第二阶段 · {stage_2_coverage['total']} 条验证与流程诊断</b><p class="adv-kv">已构建：{stage_2_coverage['group_chat']} 条群聊、{stage_2_coverage['private_chat']} 条私聊，其中 {stage_2_coverage['complex_large_group']} 条复杂大群；已被分层整理任务读取，不再是全新隐藏终验。</p></div>
      </div>
      <ul class="adv-list">
        <li>第二阶段至少来自 {stage_2['distinct_chat_min']} 个不同聊天；普通聊天最多 {stage_2['max_cases_per_chat']} 题。</li>
        <li><b>指定群例外：</b>正确共同富裕群至少 {stage_2['correct_common_prosperity_with_luxixi_min']} 个、柚子群至少 {stage_2['youzi_group_min']} 个独立事件簇，不受普通聊天 3 题上限约束。</li>
        <li>指定群每个事件仍必须同时通过真人身份、对应群的截止时间、目标群身份、完整上下文、未曝光和语义近重复检查。</li>
        <li><b>日期例外：</b>仅柚子群最多放宽至 2026-07-31 23:59:59；实际封存数据只用到 2026-07-08，无需追加 7 月 10–31 日。其他群仍按 2026-04-01 前，除日期外的标准全部不变。</li>
        <li>指定群数量已补齐，但字段完整性仍不满足生产回放；不能用数量覆盖掩盖生产差异。</li>
        <li>第二阶段不得复用第一阶段的题目、事件簇、聊天线程或语义近重复。</li>
      </ul>
    </section>

    <section class="card" style="border-color:var(--green)">
      <h3 style="color:var(--green)">验证与流程诊断集：覆盖已构建</h3>
      <p class="adv-kv" style="margin-top:6px">以下是排除第一阶段事件及答案近重复后实际选中的聚合覆盖。普通聊天最多 3 题，两个指定群按独立事件簇例外计数；本页不公开任何隐藏题、聊天 ID、事件 ID 或答案。</p>
      <div class="adv-inventory">
        <div class="adv-stock"><strong>{stage_2_coverage['total']}</strong><span>诊断集总数<br>覆盖目标 {stage_2['total_target']} · 已满足</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['distinct_chats']}</strong><span>不同聊天<br>目标 {stage_2['distinct_chat_min']} · 已满足</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['group_chat']}</strong><span>群聊<br>目标 {stage_2['group_chat_target']} · 已满足</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['private_chat']}</strong><span>私聊<br>目标 {stage_2['private_chat_target']} · 已满足</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['complex_large_group']}</strong><span>复杂大群<br>目标 {stage_2['complex_large_group_target']} · 已超过 2</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['correct_common_prosperity_with_luxixi']}</strong><span>正确共同富裕群<br>目标 {stage_2['correct_common_prosperity_with_luxixi_min']} · 已满足</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['youzi_group']}</strong><span>柚子群<br>目标 {stage_2['youzi_group_min']} · 已满足</span></div>
        <div class="adv-stock"><strong>{stage_2_coverage['distinct_event_clusters']}</strong><span>独立事件<br>每事件最多一题</span></div>
      </div>
      <p class="adv-kv" style="margin-top:8px"><b>正确共同富裕群：</b>最终选入 20 个通过身份、截止时间、完整上下文、事件隔离、答案曝光和答案近重复检查的独立事件。</p>
      <p class="adv-kv" style="margin-top:8px"><b>柚子群：</b>已从扩展只读导出中封存 20 个不同事件，每个事件最多一题；真人身份、完整前置上下文、已知原文曝光和语义近重复均已检查。导出实际截至 2026-07-08，入选事件更早，因此无需追加 7 月 10–31 日数据。</p>
      <p class="adv-kv" style="margin-top:8px">柚子群私有 manifest SHA256：{youzi_manifest_hash}<br>公开聚合核对：{youzi_coverage['selected_events']}/{youzi_coverage['target_events']} · 缺口 {youzi_coverage['quota_gap']} · 入选事件最晚 {html.escape(str(youzi_coverage['latest_selected_date']))}</p>
      <p class="adv-kv" style="margin-top:8px">Stage 2 私有 manifest SHA256：{stage_2_manifest_hash}<br>扫描已知曝光文件：{stage_2_summary['exposure_files_scanned']} 个 · 模型调用：{stage_2_summary['checks']['model_calls']}</p>
      <p class="adv-warning"><b>仍未进入实验：</b>样本数量、身份和事件独立性检查已通过，但上下文仅有 speaker/content，严格生产可比为 0/240；且分层整理任务已读取真人答案。它只能做验证和流程诊断，不能支撑正式 A/B。语义近重复筛选是确定性的保守词法规则，不冒充人工审阅。</p>
    </section>

    <section class="card">
      <h3>复杂场景最低覆盖（第二阶段，可交叉计数）</h3>
      <div class="adv-inventory">
        <div class="adv-stock"><strong>{stage_2_strata['multi_speaker_target_resolution_min']}</strong><span>多人并行<br>回复对象识别</span></div>
        <div class="adv-stock"><strong>{stage_2_strata['topic_switch_or_parallel_threads_min']}</strong><span>话题切换<br>多个子话题</span></div>
        <div class="adv-stock"><strong>{stage_2_strata['cross_message_reference_min']}</strong><span>跨消息承接<br>追问与指代</span></div>
        <div class="adv-stock"><strong>{stage_2_strata['fact_grounding_min']}</strong><span>事实依据<br>避免幻觉</span></div>
        <div class="adv-stock"><strong>{stage_2_strata['humor_or_emotional_response_min']}</strong><span>幽默或情绪<br>本人表达</span></div>
        <div class="adv-stock"><strong>{stage_2_strata['real_world_commitment_boundary_min']}</strong><span>现实承诺<br>玩笑边界</span></div>
      </div>
      <p class="adv-kv" style="margin-top:8px">由独立整理者在封存前标注；封存后只公开聚合数量，不公开题目或答案。</p>
    </section>

    <section class="card">
      <h3>身份怎样计分</h3>
      <ul class="adv-list">
        <li><b>已确认真人：</b>可作为 2026 年 4 月前真人标准。</li>
        <li><b>已确认 Bot 实发：</b>必须有 <code>send_success=1</code> 的完整发送证据链。</li>
        <li><b>身份未知：</b>只展示、只诊断，绝不冒充真人答案。</li>
      </ul>
    </section>

    <section class="card">
      <h3>双方怎样轮流提升</h3>
      <div class="adv-flow" style="margin-top:10px">
        <div class="adv-step"><b>① 冻结 C0</b><p class="adv-kv">只改善 J0 的辨认边界。</p></div><span class="adv-arrow">→</span>
        <div class="adv-step"><b>② 冻结 J0</b><p class="adv-kv">B1 只验证一个小假设。</p></div><span class="adv-arrow">→</span>
        <div class="adv-step"><b>③ 收难例</b><p class="adv-kv">B1 难例只能进入下一轮 J-train。</p></div><span class="adv-arrow">→</span>
        <div class="adv-step"><b>④ 冻结 J1</b><p class="adv-kv">再验证一个 B2 假设。</p></div>
      </div>
      <p style="margin-top:10px;color:var(--yellow);line-height:1.7">同一轮不得同时调整 Judge 和 Bot；C0 永远不被候选覆盖。</p>
    </section>

    <section class="card">
      <h3>Judge 调用隔离</h3>
      <ul class="adv-list">
        <li>每道题一次全新的独立模型 API 调用；只含一题。</li>
        <li>完整聊天上下文 + 两份匿名回复，A/B 位置平衡并交换复测。</li>
        <li>看不到答案、身份、模型名、生成轨迹、其他题、Codex 历史或文件系统。</li>
        <li>风格 Judge 不开放工具；事实核查是另一条可审计链，避免把未知事实直接判成编造。</li>
        <li>保存原始提示词、输入、输出、模型参数与工具调用数。</li>
      </ul>
    </section>

    <section class="card">
      <h3>怎样防止双方刷题</h3>
      <ul class="adv-list">
        <li>未来真正的 hidden 不进入 Prompt、Wiki、RAG、few-shot 或逐题错误分析。</li>
        <li>未来终测只反馈聚合结果，不反馈逐题答案。</li>
        <li>保留冻结旧 Judge、当前 Judge 和真人抽查，不能只骗一个 Judge。</li>
        <li>Bot 混淆率提高但真人体验没提高：淘汰。</li>
        <li>Judge 靠长短、标点、错字或俏皮程度提高：不升级。</li>
        <li>同一假设连续两轮失败：停止该方向。</li>
      </ul>
    </section>

    <section class="card">
      <h3>晋级门槛</h3>
      <div class="adv-grid" style="margin-top:10px">
        <div class="adv-step"><b>Judge</b><p class="adv-kv">第二阶段比冻结 J0 至少高 5 个百分点；总体 ≥75%；各主要分层 ≥65%；交换位置不退步；无明显表面线索依赖。</p></div>
        <div class="adv-step"><b>Bot</b><p class="adv-kv">真人匿名 A/B：胜数必须多于负数，双侧 sign test p&lt;0.05，净胜率至少提升 10 个百分点；复杂大群净胜为正且不退步；私聊无实质退步。</p></div>
      </div>
      <p class="adv-kv" style="margin-top:8px">新增事实幻觉 = 0，新增现实行动承诺 = 0；旧梗、跨轮重复和生成失败不得恶化；相同回放条件下 P95 延迟不超过 C0 10%，且必须满足生产 SLO。不能靠更长回复或更频繁插话换提升。Judge 混淆率不能单独证明 Bot 质量。</p>
    </section>

    <section class="card" style="border-color:var(--red)">
      <h3 style="color:var(--red)">执行前必须核对的生产差异</h3>
      <p style="margin-top:8px;line-height:1.7">GlobalStore/未读状态、历史时点 Wiki、上下文与时间戳、模型参数、工具证据、few-shot/RAG、Self-Refine、事实检查、参与决策和拆消息发送链。</p>
      <p style="margin-top:8px;color:var(--yellow);line-height:1.7">任何一项无法对齐，必须先把 diff 发给你确认；未经确认不运行。</p>
    </section>

    <section class="card" style="border-color:var(--yellow)">
      <h3>当前封存状态</h3>
      <ul class="adv-list">
        <li>协议规则：已写入 v1 文件，已确认，尚未执行。</li>
        <li>已曝光材料：只用于训练、诊断或压力场景，不计入 hidden。</li>
        <li>第一阶段首筛集：120 条已私有封存，抽样规则与覆盖已确认；本页不公开逐题内容。</li>
        <li>第二阶段验证与流程诊断集：240 条已构建；严格生产可比 0/240，且已被分层整理任务读取，不再视为全新隐藏终验。</li>
        <li>模型调用：0。</li>
        <li>C0 修改：无。</li>
        <li>Bot 启动：无。</li>
      </ul>
    </section>
    """

@app.get("/experiments", response_class=HTMLResponse)
def experiments_list():
    db = get_db()
    conn = db._get_conn()
    exps = conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    conn.close()

    plan = (
        '<section class="card" style="border-color:var(--blue)">'
        '<b>新方案：Judge–Bot 对抗式迭代蓝图</b>'
        '<p style="margin-top:6px;color:var(--muted);line-height:1.7">现有 240 条已降级为验证与流程诊断集；正式 A/B 仍禁止。</p>'
        '<a href="/experiments/reply-alignment" style="display:inline-block;margin-top:8px;color:var(--yellow)">先看生产一致性审计 →</a><br>'
        '<a href="/experiments/adversarial-plan" style="display:inline-block;margin-top:8px;color:var(--blue)">手机查看完整方案 →</a>'
        '</section>'
        + _reply_ab_plan_html()
    )
    if not exps:
        content = plan + "<p style='color:var(--muted)'>暂无已执行实验。</p>"
        return HTMLResponse(_page("A/B 实验", content, "/experiments"))

    rows = ""
    for e in exps:
        icon = "✅" if e["is_improvement"] else "—"
        bc_diff = (e["control_badcase_rate"] or 0) - (e["exp_badcase_rate"] or 0)
        score_diff = (e["exp_avg_score"] or 0) - (e["control_avg_score"] or 0)
        rows += f"""<tr>
          <td><a href="/experiments/{e['id']}" style="color:var(--blue)">{e['name']}</a></td>
          <td>{e['description'] or ''}</td>
          <td>{e['n_samples']}</td>
          <td>{(e['control_badcase_rate'] or 0)*100:.0f}% → {(e['exp_badcase_rate'] or 0)*100:.0f}%</td>
          <td>{e['control_avg_score']:.1f} → {e['exp_avg_score']:.1f}</td>
          <td>{icon} {e['summary'] or ''}</td>
          <td style="font-size:11px;color:var(--muted)">{(e['created_at'] or '')}</td>
        </tr>"""

    content = plan + f"""<h2 style="font-size:16px;margin:24px 0 8px">已执行实验</h2><table>
    <tr><th>实验</th><th>描述</th><th>N</th><th>Badcase 率</th><th>均分</th><th>结论</th><th>时间</th></tr>
    {rows}</table>"""
    return HTMLResponse(_page("A/B 实验", content, "/experiments"))


@app.get("/experiments/adversarial-plan", response_class=HTMLResponse)
def adversarial_plan():
    return HTMLResponse(_page("Judge–Bot 对抗实验蓝图", _adversarial_plan_html(), "/experiments"))


@app.get("/experiments/reply-alignment", response_class=HTMLResponse)
def reply_production_alignment():
    return HTMLResponse(_page("回复评测生产一致性审计", _reply_production_alignment_html(), "/experiments"))


@app.get("/experiments/{exp_id}", response_class=HTMLResponse)
def experiment_detail(exp_id: int):
    import json
    db = get_db()
    conn = db._get_conn()

    exp = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
    if not exp:
        conn.close()
        return HTMLResponse("<h1>Experiment not found</h1>")

    dims = json.loads(exp["dimension_diffs_json"] or "{}")

    # Config params diff — 从 run_experiment.py 动态导入，计算与 CONTROL 的差异
    exp_name = exp['name'] or ''
    param_html = ""
    try:
        import sys
        from pathlib import Path
        exp_script = Path(__file__).parent / "run_experiment.py"
        if str(exp_script.parent.parent) not in sys.path:
            sys.path.insert(0, str(exp_script.parent.parent))
        from scripts.run_experiment import BOT_EXPERIMENTS, CONTROL

        exp_cfg = BOT_EXPERIMENTS.get(exp_name)
        if exp_cfg:
            # 功能开关对比
            features = [
                ("时间感知", "enable_time_awareness"),
                ("时间戳注入", "enable_timestamps"),
                ("回复克制", "enable_reply_restraint"),
                ("未读去重", "enable_unread_dedup"),
                ("search_in_page", "enable_search_in_page"),
            ]
            diffs = []
            for label, attr in features:
                c_val = getattr(CONTROL, attr, True)
                e_val = getattr(exp_cfg, attr, c_val)
                if c_val != e_val:
                    diffs.append(f'{label}: {"✅→❌ 关闭" if e_val else "❌→✅ 开启"}')
            # 截断长度对比
            if CONTROL.browse_truncate != exp_cfg.browse_truncate:
                diffs.append(f'浏览截断: {CONTROL.browse_truncate}→{exp_cfg.browse_truncate}')
            if CONTROL.tool_result_truncate != exp_cfg.tool_result_truncate:
                diffs.append(f'工具截断: {CONTROL.tool_result_truncate}→{exp_cfg.tool_result_truncate}')

            param_html = '<div class="card"><b>⚙️ 实验参数（CONTROL 线上配置 → 实验组差异）：</b><br>'
            if diffs:
                param_html += '<div style="margin-top:4px">' + ' · '.join(f'<span style="margin:4px 8px;font-size:12px">{d}</span>' for d in diffs) + '</div>'
            else:
                param_html += '<span style="font-size:12px;color:var(--muted)">无差异（与 CONTROL 配置相同）</span>'
            param_html += '</div>'
    except Exception as e:
        import logging
        logging.getLogger("admin").warning(f"实验配置 diff 计算失败: {e}")
        param_html = '<!-- config diff error: 详见服务端日志 -->'

    # 实验策略详细说明
    exp_desc = exp['description'] or ''
    content = f"""<div class="card" style="border-left:3px solid var(--blue);margin-bottom:16px">
  <h2>🧪 实验: {exp['name']}</h2>
  <div style="font-size:13px;line-height:1.7;color:var(--text);margin:8px 0">{exp_desc}</div>
  <div style="font-size:11px;color:var(--muted);margin-top:8px">N={exp['n_samples']} · 对照组=CONTROL(线上当前配置) · 实验组={exp_name} · 固定Judge=v4-pro</div>
</div>""" + param_html

    # Summary
    bc_diff = (exp['control_badcase_rate'] or 0) - (exp['exp_badcase_rate'] or 0)
    score_diff = (exp['exp_avg_score'] or 0) - (exp['control_avg_score'] or 0)
    content += f"""
    <div class="card"><b>📊 结果</b></div>
    <div class="metrics">
      <div class="metric"><div class="val">{(exp['control_badcase_rate'] or 0)*100:.0f}% → {(exp['exp_badcase_rate'] or 0)*100:.0f}%</div><div class="lbl">Badcase 率（{bc_diff:+.0%}）</div></div>
      <div class="metric"><div class="val">{exp['control_avg_score']:.1f} → {exp['exp_avg_score']:.1f}</div><div class="lbl">均分（{score_diff:+.1f}）</div></div>
      <div class="metric"><div class="val">{'✅ 提升' if exp['is_improvement'] else '— 无差异'}</div><div class="lbl">{exp['summary'] or ''}</div></div>
    </div>"""

    # Dimension diffs
    if dims:
        content += '<div class="card"><b>各维度差异（实验 - 基线）：</b><br>'
        for dim, diff in sorted(dims.items(), key=lambda x: -x[1]):
            color = "var(--green)" if diff > 0.1 else ("var(--red)" if diff < -0.1 else "var(--muted)")
            bar_w = min(abs(diff) * 50, 200)
            bar_color = "var(--green)" if diff > 0 else "var(--red)"
            content += f'<div style="display:flex;align-items:center;margin:4px 0"><span style="width:100px;font-size:12px">{dim}</span><span style="color:{color};font-weight:600;width:40px">{diff:+.1f}</span><div style="width:200px;background:rgba(255,255,255,.05);border-radius:3px;height:12px"><div style="width:{bar_w}px;height:12px;background:{bar_color};border-radius:3px"></div></div></div>'
        content += '</div>'

    # Per-tick: side-by-side comparison of actual bot replies
    results = conn.execute("""
        SELECT c.tick_id,
               MAX(CASE WHEN c.config_name='control' THEN c.judge_score END) as c_score,
               MAX(CASE WHEN c.config_name='control' THEN c.judge_is_badcase END) as c_bc,
               MAX(CASE WHEN c.config_name='control' THEN c.bot_reply END) as c_reply,
               MAX(CASE WHEN c.config_name='control' THEN c.judge_dimensions_json END) as c_dims,
               MAX(CASE WHEN c.config_name='control' THEN c.judge_reason END) as c_reason,
               MAX(CASE WHEN c.config_name='control' THEN c.system_prompt END) as c_sp,
               MAX(CASE WHEN c.config_name='control' THEN c.user_prompt END) as c_up,
               MAX(CASE WHEN c.config_name!='control' THEN c.judge_score END) as e_score,
               MAX(CASE WHEN c.config_name!='control' THEN c.judge_is_badcase END) as e_bc,
               MAX(CASE WHEN c.config_name!='control' THEN c.bot_reply END) as e_reply,
               MAX(CASE WHEN c.config_name!='control' THEN c.judge_dimensions_json END) as e_dims,
               MAX(CASE WHEN c.config_name!='control' THEN c.judge_reason END) as e_reason,
               MAX(CASE WHEN c.config_name!='control' THEN c.system_prompt END) as e_sp,
               MAX(CASE WHEN c.config_name!='control' THEN c.user_prompt END) as e_up
        FROM experiment_results c
        WHERE c.experiment_id=?
        GROUP BY c.tick_id ORDER BY c.tick_id
    """, (exp_id,)).fetchall()
    conn.close()


    content += '<h2>📋 逐 Tick 对比</h2>'
    # 获取上下文数据
    conn2 = db._get_conn()
    tick_ids = [r['tick_id'] for r in results]
    ctx_data = {}
    if tick_ids:
        placeholders = ','.join('?' * len(tick_ids))
        ctx_rows = conn2.execute(
            "SELECT tick_id, system_prompt, user_prompt, tool_calls_json FROM tick_log "
            "WHERE tick_id IN (" + placeholders + ")",  # nosec B608
            tick_ids,
        ).fetchall()
        for cr in ctx_rows:
            d = dict(cr)
            ctx_data[d['tick_id']] = d
    conn2.close()

    for r in results:
        c_s = r["c_score"] or 0; e_s = r["e_score"] or 0
        diff = e_s - c_s
        cls = "judge-match" if diff >= 0 else "judge-mismatch"
        c_icon = "❌" if r["c_bc"] else "✅"
        e_icon = "❌" if r["e_bc"] else "✅"
        arrow = "↑" if diff > 1 else ("↓" if diff < -1 else "→")

        ctx = ctx_data.get(r['tick_id'], {})
        sp = ctx.get('system_prompt') or ''
        up = ctx.get('user_prompt') or ''
        tc = (ctx.get('tool_calls_json') or '[]')

        # Parse dimension scores
        c_dims_json = r["c_dims"] or "{}"
        e_dims_json = r["e_dims"] or "{}"
        try: c_dims = json.loads(c_dims_json)
        except: c_dims = {}
        try: e_dims = json.loads(e_dims_json)
        except: e_dims = {}
        dim_comparison = ""
        for dim_name in ["幻觉控制", "上下文理解", "回复必要性", "简洁度", "个性一致性", "时间推理", "信息准确性", "亮点加分项"]:
            cv = c_dims.get(dim_name, {}).get("score", 0)
            ev = e_dims.get(dim_name, {}).get("score", 0)
            d = ev - cv
            color = "var(--green)" if d > 0 else ("var(--red)" if d < 0 else "var(--muted)")
            dim_comparison += f'<span style="margin:2px 6px;font-size:10px">{dim_name}: {cv}→{ev} <b style="color:{color}">{d:+d}</b></span>'

        # 提示词 Diff — 对比完整 prompt（system + user）
        rd = dict(r)
        c_full = (rd.get('c_sp') or '') + '\n---\n' + (rd.get('c_up') or '')
        e_full = (rd.get('e_sp') or '') + '\n---\n' + (rd.get('e_up') or '')
        prompt_diff = ""
        if c_full.strip() and e_full.strip():
            import difflib
            cl = c_full.splitlines(); el = e_full.splitlines()
            sm = difflib.SequenceMatcher(None, cl, el)
            left_rows = []; right_rows = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    # 显示少量上下文，大量相同行折叠
                    lines = cl[i1:i2]
                    if len(lines) <= 5:
                        for ln in lines:
                            left_rows.append(f'<tr><td class="ln">{i1+1}</td><td class="eq">{ln}</td></tr>')
                            right_rows.append(f'<tr><td class="ln">{j1+1}</td><td class="eq">{ln}</td></tr>')
                            i1+=1; j1+=1
                    else:
                        skip_n = len(lines) - 4
                        for ln in lines:
                            left_rows.append(f'<tr><td class="ln">{i1+1}</td><td class="eq">{ln}</td></tr>')
                            right_rows.append(f'<tr><td class="ln">{j1+1}</td><td class="eq">{ln}</td></tr>')
                            i1+=1; j1+=1
                        left_rows.append(f'<tr><td colspan="2" class="skip">··· {skip_n} 行相同 ···</td></tr>')
                        right_rows.append(f'<tr><td colspan="2" class="skip">··· {skip_n} 行相同 ···</td></tr>')
                        for ln in lines[-2:]:
                            left_rows.append(f'<tr><td class="ln">{i1+len(lines)-2}</td><td class="eq">{ln}</td></tr>')
                            right_rows.append(f'<tr><td class="ln">{j1+len(lines)-2}</td><td class="eq">{ln}</td></tr>')
                            i1+=1; j1+=1
                elif tag == 'delete':
                    for ln in cl[i1:i2]:
                        left_rows.append(f'<tr><td class="ln">{i1+1}</td><td class="del">{ln}</td></tr>')
                        right_rows.append(f'<tr><td class="ln"></td><td class="empty"></td></tr>')
                        i1+=1
                elif tag == 'insert':
                    for ln in el[j1:j2]:
                        left_rows.append(f'<tr><td class="ln"></td><td class="empty"></td></tr>')
                        right_rows.append(f'<tr><td class="ln">{j1+1}</td><td class="add">{ln}</td></tr>')
                        j1+=1
                elif tag == 'replace':
                    for ln in cl[i1:i2]:
                        left_rows.append(f'<tr><td class="ln">{i1+1}</td><td class="del">{ln}</td></tr>')
                        right_rows.append(f'<tr><td class="ln"></td><td class="empty"></td></tr>')
                        i1+=1
                    for ln in el[j1:j2]:
                        left_rows.append(f'<tr><td class="ln"></td><td class="empty"></td></tr>')
                        right_rows.append(f'<tr><td class="ln">{j1+1}</td><td class="add">{ln}</td></tr>')
                        j1+=1

            # 统计删除/新增/修改行数
            del_count = sum(i2-i1 for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag=='delete')
            add_count = sum(j2-j1 for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag=='insert')
            mod_count = sum(i2-i1 for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag=='replace')
            change_count = del_count + add_count + mod_count
            if change_count > 0:
                diff_title = f'提示词 Diff — 删除 {del_count} 行 / 新增 {add_count} 行 / 修改 {mod_count} 行'
            else:
                diff_title = '提示词 Diff — 无变化'

            # diff summary bar
            summary_bar = ""
            if change_count > 0:
                total_lines = max(len(cl), len(el))
                del_w = max(1, int(del_count/total_lines*200)) if total_lines else 0
                add_w = max(1, int(add_count/total_lines*200)) if total_lines else 0
                mod_w = max(1, int(mod_count/total_lines*200)) if total_lines else 0
                summary_bar = f'<div style="display:flex;height:4px;border-radius:2px;margin:4px 0;overflow:hidden;max-width:200px"><div style="width:{del_w}px;height:4px;background:#f85149"></div><div style="width:{mod_w}px;height:4px;background:#d29922"></div><div style="width:{add_w}px;height:4px;background:#3fb950"></div></div>'

            prompt_diff = '<div style="margin:8px 0 4px;font-size:11px;color:var(--muted)">📝 ' + diff_title + '</div>'
            prompt_diff += '<div style="margin:4px 0">' + summary_bar + '</div>'
            prompt_diff += '<div class="diff-container">'
            prompt_diff += '<div class="diff-col"><div class="diff-header control">线上配置 (CONTROL)</div><div class="diff-body"><table style="width:100%;border-collapse:collapse">' + ''.join(left_rows) + '</table></div></div>'
            prompt_diff += '<div class="diff-col"><div class="diff-header exp">' + exp_name + ' (实验组)</div><div class="diff-body"><table style="width:100%;border-collapse:collapse">' + ''.join(right_rows) + '</table></div></div>'
            prompt_diff += '</div>'

        content += f"""<div class="card {cls}">
  <h3><a href="/ticks/{r['tick_id']}" style="color:var(--blue)">#{r['tick_id']}</a> {arrow} {diff:+.0f}分</h3>
  {prompt_diff}
  <table><tr>
    <th style="width:50%">基线 {c_icon} {c_s:.0f}分</th>
    <th style="width:50%">实验组 {e_icon} {e_s:.0f}分</th>
  </tr><tr>
    <td style="font-size:12px;white-space:pre-wrap">{r['c_reply'] or ''}</td>
    <td style="font-size:12px;white-space:pre-wrap">{r['e_reply'] or ''}</td>
  </tr></table>
  <div style="margin:4px 0;color:var(--muted);font-size:10px">{dim_comparison}</div>
  <div style="font-size:11px;color:var(--muted);margin:4px 0">基线理由: {r['c_reason'] or ''}</div>
  <div style="font-size:11px;color:var(--muted);margin:4px 0">实验理由: {r['e_reason'] or ''}</div>
  <details style="margin-top:4px"><summary style="cursor:pointer;font-size:11px;color:var(--blue)">上下文（System: {len(sp)}字 User: {len(up)}字）</summary>
    <div style="font-size:10px;max-height:400px;overflow:auto;white-space:pre-wrap;background:rgba(0,0,0,.2);padding:8px;border-radius:3px;margin-top:4px">{up}</div>
  </details>
</div>"""

    return HTMLResponse(_page(f"实验: {exp['name']}", content, "/experiments"))


# ── 代码审计 ──

def _ensure_code_audit_migration():
    """迁移旧版 code_audit 表：checked INTEGER -> status TEXT"""
    try:
        db = get_db()
        conn = db._get_conn()
        cols = [c[1] for c in conn.execute("PRAGMA table_info(code_audit)").fetchall()]
        if "checked" in cols and "status" not in cols:
            conn.execute("ALTER TABLE code_audit ADD COLUMN status TEXT DEFAULT 'pending'")
            conn.execute("UPDATE code_audit SET status = CASE WHEN checked = 1 THEN 'fixed' ELSE 'pending' END")
            conn.commit()
            print("[CodeAudit] migrated checked -> status")
        conn.close()
    except Exception as e:
        print(f"[CodeAudit] migration check: {e}")


CODE_AUDIT_ISSUES = [
    {
        "key": "api-timestamp-missing",
        "severity": "P0",
        "title": "API 路径时间戳系统性缺失",
        "file": "src/perception/smart_pipeline.py",
        "lines": "48-142, 893-904",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/perception/smart_pipeline.py#L893",
        "problem": "System Prompt 的 messages 格式只有 sender/text/type，未要求 API 返回时间戳；解析代码固定用 strptime('%Y-%m-%d %H:%M:%S')，但截图中的时间格式是'昨天 21:58'、'11:34'等，完全不匹配。",
        "impact": "所有 API 路径消息 create_time 100% fallback 到 int(time.time())。同 tick 内多条消息时间戳完全相同，导致历史窗口'最近10分钟'cutoff失效、already_handled去重误判、LLM时间推理维度失效。Tick 409已证实此症状。",
        "fix": "1) System Prompt 增加 timestamp 字段要求；2) 解析逻辑支持'昨天 HH:MM'、'HH:MM'、'YYYY-MM-DD HH:MM'多种格式；3) 使用相对时间转换（昨天=今天日期-1天+HH:MM）。"
    },
    {
        "key": "layout-timestamp-bug",
        "severity": "P0",
        "title": "layout_parser 聊天列表时间戳检测逻辑完全错误",
        "file": "src/layout/layout_parser.py",
        "lines": "354",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/layout/layout_parser.py#L354",
        "problem": "e.text in TIMESTAMP_PATTERNS 永远为False（列表元素是正则串）；e.text[1]==':' 对'11:34'判断第二个字符'1'；e.text[2:].isdigit() 对'11:34'得到':34'.isdigit()。三个条件全部永远为False。",
        "impact": "ChatListItem.timestamp 永远为空。当前无下游直接消费此字段，但这是一个彻底失效的功能——未来任何人基于时间戳做排序/判断都会失败。",
        "fix": "改为正则匹配：any(re.match(p, e.text) for p in TIMESTAMP_PATTERNS)"
    },
    {
        "key": "judge-weight-mismatch",
        "severity": "P0",
        "title": "judge_worker 维度权重表与 Prompt 模板不一致",
        "file": "src/badcase/judge_worker.py",
        "lines": "602-611, 215",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/badcase/judge_worker.py#L602",
        "problem": "Prompt中回复必要性20%/简洁度15%，代码中15%/10%；代码多出一个'工具调用正确性'10%维度，Prompt中完全没有。",
        "impact": "Judge LLM按Prompt打分，代码用另一套权重算总分。'工具调用正确性'缺失时fallback到50分，每个case被系统性扣5分，borderline case可能错误判为badcase。",
        "fix": "统一权重表：代码DIM_WEIGHTS与Prompt模板完全一致，或将'工具调用正确性'合并到'信息准确性'中。"
    },
    {
        "key": "timestamp-extract-inconsistent",
        "severity": "P1",
        "title": "_format_message_line 与 _msg_ts 时间戳提取逻辑不一致",
        "file": "src/reply/generator.py",
        "lines": "772-784, 916-921",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/reply/generator.py#L772",
        "problem": "_format_message_line对SELF消息不优先用reply_time；_msg_ts对SELF消息优先reply_time。两者fallback路径也不同（timestamp解析 vs time.time()）。",
        "impact": "Bot自己发的消息在prompt中不显示时间标签（因为create_time为空），但会被正确纳入历史窗口。显示与选择逻辑不一致，未来维护者容易困惑。",
        "fix": "统一两个函数的时间戳提取优先级：SELF→reply_time→create_time→timestamp解析→time.time()；OTHER→create_time→timestamp解析→time.time()"
    },
    {
        "key": "bot-self-msg-no-create-time",
        "severity": "P1",
        "title": "Bot 自身消息不设置 create_time",
        "file": "src/bot/wechat_bot.py",
        "lines": "416-420",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/bot/wechat_bot.py#L416",
        "problem": "发送成功后创建ChatMessage时只设置了reply_time，没有设置create_time。",
        "impact": "结合P1-2，Bot消息在prompt中永远不显示时间标签。LLM无法判断Bot消息的发送时间，只能依赖消息顺序推断。",
        "fix": "ChatMessage(..., create_time=int(time.time()), reply_time=time.time())"
    },
    {
        "key": "already-handled-mislabel",
        "severity": "P2",
        "title": "already_handled 可能错误标记连续消息",
        "file": "src/reply/generator.py",
        "lines": "962-975",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/reply/generator.py#L962",
        "problem": "如果用户连续发3条消息，Bot只回复了第3条，第1/2条也会因为'reply_time > ts'被标记为'⚠️(可跳过)'。",
        "impact": "这是提示性标记不强制跳过，但可能误导LLM跳过需要单独回复的消息。属于设计缺陷。",
        "fix": "更精确匹配：检查Bot回复的上一条消息是否与当前未读消息内容对应，而非仅比较时间。"
    },
]


@app.post("/api/code-audit/{key}")
async def save_code_audit(key: str, request: Request):
    body = await request.json()
    db = get_db()
    conn = db._get_conn()
    status = body.get("status", "pending")
    if status not in ("pending", "todo", "rethink", "fixed", "wontfix", "deferred"):
        status = "pending"
    conn.execute("""INSERT INTO code_audit (issue_key, status, notes, updated_at)
        VALUES (?, ?, ?, datetime('now','localtime'))
        ON CONFLICT(issue_key) DO UPDATE SET
        status=excluded.status, notes=excluded.notes, updated_at=excluded.updated_at""",
        (key, status, body.get("notes", "")))
    conn.commit()
    conn.close()
    return JSONResponse({"success": True})


@app.get("/code-audit", response_class=HTMLResponse)
def code_audit_page():
    db = get_db()
    conn = db._get_conn()
    _ensure_code_audit_migration()
    rows = conn.execute("SELECT issue_key, status, notes, ai_proposal FROM code_audit").fetchall()
    conn.close()
    state_map = {r["issue_key"]: {"status": r["status"] or "pending", "notes": r["notes"] or "", "ai_proposal": r["ai_proposal"] or ""} for r in rows}

    severity_color = {"P0": "#f85149", "P1": "#d29922", "P2": "#58a6ff"}
    status_label = {"pending": "⏳ 待处理", "todo": "🔧 需要修复", "rethink": "💡 需AI重新思考", "fixed": "✅ 已修复", "wontfix": "🚫 不需要修复", "deferred": "⏸️ 搁置"}
    status_color = {"pending": "var(--muted)", "todo": "var(--blue)", "rethink": "var(--yellow)", "fixed": "var(--green)", "wontfix": "var(--muted)", "deferred": "var(--muted)"}

    issues_json = _json.dumps(CODE_AUDIT_ISSUES, ensure_ascii=False)
    state_json = _json.dumps({r["issue_key"]: {"status": r["status"] or "pending", "notes": r["notes"] or "", "ai_proposal": r["ai_proposal"] or ""} for r in rows}, ensure_ascii=False)

    content = f"""
    <div class="audit-layout">
      <div class="audit-sidebar">
        <div class="audit-sidebar-header">🐛 代码审计 ({len(CODE_AUDIT_ISSUES)}条)</div>
        <div id="issue-list"></div>
      </div>
      <div class="audit-main">
        <div id="issue-detail">
          <div style="color:var(--muted);text-align:center;padding:60px 20px">👈 点击左侧 issue 开始审计</div>
        </div>
      </div>
    </div>
    <script>
    const issues = {issues_json};
    const stateMap = {state_json};
    const statusLabel = {{
      pending: "⏳ 待处理", ai_analyzing: "🤖 AI 分析中", rethink: "✅ AI 分析完成", failed: "❌ 分析失败",
      todo: "🔧 需要修复", fixed: "✅ 已修复", wontfix: "🚫 不需要修复", deferred: "⏸️ 搁置"
    }};
    const statusColor = {{
      pending: "var(--muted)", ai_analyzing: "var(--yellow)", rethink: "var(--green)", failed: "var(--red)",
      todo: "var(--blue)", fixed: "var(--green)", wontfix: "var(--muted)", deferred: "var(--muted)"
    }};
    const severityColor = {{P0: "#f85149", P1: "#d29922", P2: "#58a6ff"}};
    let currentKey = null;
    let currentRounds = [];

    function renderIssueList() {{
      const list = document.getElementById('issue-list');
      list.innerHTML = issues.map((issue, idx) => {{
        const st = stateMap[issue.key] || {{}};
        const status = st.status || 'pending';
        const active = issue.key === currentKey ? 'active' : '';
        return `<div class="issue-item ${{active}}" data-key="${{issue.key}}" onclick="selectIssue('${{issue.key}}')">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
            <span class="issue-sev" style="background:${{severityColor[issue.severity]}}">${{issue.severity}}</span>
            <span class="issue-status" style="color:${{statusColor[status]}}">${{statusLabel[status]}}</span>
          </div>
          <div class="issue-title">${{issue.title}}</div>
        </div>`;
      }}).join('');
    }}

    async function selectIssue(key) {{
      currentKey = key;
      renderIssueList();
      const issue = issues.find(i => i.key === key);
      const resp = await fetch('/api/code-audit/' + key);
      const data = await resp.json();
      currentRounds = data.rounds || [];
      stateMap[key] = {{status: data.status, notes: data.notes, ai_proposal: data.ai_proposal}};
      renderIssueList(); // 数据已更新，刷新左侧列表状态

      const detail = document.getElementById('issue-detail');
      detail.innerHTML = `
        <div class="issue-header">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span class="issue-sev" style="background:${{severityColor[issue.severity]}};font-size:13px;padding:3px 10px">${{issue.severity}}</span>
            <h2 style="margin:0;font-size:18px">${{issue.title}}</h2>
          </div>
          <div style="font-size:12px;color:var(--muted)">
            📍 <a href="${{issue.github_url}}" target="_blank" style="color:var(--blue)">${{issue.file}}:${{issue.lines}}</a>
          </div>
        </div>
        <div class="issue-info">
          <div class="info-section">
            <div class="info-label" style="color:var(--red)">❌ 问题</div>
            <div class="info-text">${{issue.problem}}</div>
          </div>
          <div class="info-section">
            <div class="info-label" style="color:var(--yellow)">⚠️ 影响</div>
            <div class="info-text">${{issue.impact}}</div>
          </div>
          <div class="info-section">
            <div class="info-label" style="color:var(--green)">💡 修复建议</div>
            <div class="info-text">${{issue.fix}}</div>
          </div>
        </div>
        <div id="rounds-area" class="rounds-area">
          ${{renderRounds()}}
        </div>
        <div class="input-area">
          <div class="status-bar">
            <span>当前状态: <strong style="color:${{statusColor[data.status]}}">${{statusLabel[data.status]}}</strong></span>
            <span id="save-status"></span>
          </div>
          <textarea id="notes-input" rows="3" placeholder="写出你的要求/点评给 AI...">${{data.notes || ''}}</textarea>
          <div class="input-actions">
            <button class="btn-analyze" id="btn-analyze" onclick="analyzeIssue()">🤖 请求 AI 分析</button>

            <button class="btn-save" onclick="saveStatus()">💾 保存状态</button>
          </div>
          <div id="analyze-status" class="status-msg"></div>
        </div>
      `;
    }}

    function renderRounds() {{
      if (!currentRounds.length) {{
        return '<div style="color:var(--muted);text-align:center;padding:40px 20px;font-size:13px">💬 还没有分析记录。在下方输入要求，点击"请求 AI 分析"开始第一轮。</div>';
      }}
      const st = stateMap[currentKey] || {{}};
      const currentStatus = st.status || 'pending';
      return currentRounds.map((r, idx) => {{
        const isLatest = idx === 0;
        const isAnalyzing = !r.proposal || r.proposal === '';
        const isFailed = r.proposal && r.proposal.indexOf('分析失败:') === 0;
        let aiLabel, aiContent;
        if (isAnalyzing) {{
          aiLabel = '🤖 AI 正在分析...';
          aiContent = '<span style="color:var(--yellow)">⏳ 分析中，请稍候（约1-3分钟）...</span>';
        }} else if (isFailed) {{
          aiLabel = '❌ 分析失败';
          aiContent = '<span style="color:var(--red)">' + escapeHtml(r.proposal) + '</span>';
        }} else {{
          aiLabel = '🤖 AI 方案';
          aiContent = markdownToHtml(r.proposal);
        }}
        const canAct = isLatest && currentStatus !== 'todo' && currentStatus !== 'fixed';
        const actions = canAct ? `
          <div style="margin-top:10px;display:flex;gap:8px;">
            <button class="btn-execute" onclick="executeRound(${{r.round}})">✅ 执行此方案</button>
            <button class="btn-reject" onclick="rejectRound(${{r.round}})">❌ 驳回</button>
          </div>
        ` : '';
        return `
        <div class="round-block">
          <div class="round-header">Round ${{r.round}} · ${{r.created_at || ''}}</div>
          <div class="user-bubble">
            <div class="bubble-label">👤 用户要求</div>
            <div class="bubble-content">${{escapeHtml(r.notes || '(无点评)')}}</div>
          </div>
          <div class="ai-bubble">
            <div class="bubble-label">${{aiLabel}}</div>
            <div class="bubble-content markdown-body">${{aiContent}}</div>
          </div>
          ${{actions}}
        </div>
        `;
      }}).join('');
    }}

    async function analyzeIssue() {{
      if (!currentKey) return;
      const notes = document.getElementById('notes-input').value.trim();
      if (!notes) {{
        alert('请先输入具体要求');
        return;
      }}
      const btn = document.getElementById('btn-analyze');
      const status = document.getElementById('analyze-status');
      btn.disabled = true;
      status.textContent = '⏳ AI 分析中（约 1-3 分钟）...';
      status.className = 'status-msg loading';

      // 1. 立即清空输入框，让用户要求显示在对话区
      document.getElementById('notes-input').value = '';

      // 2. 立即在本地添加一个"分析中"的 round，给用户即时反馈
      const nextRound = currentRounds.length > 0
        ? Math.max(...currentRounds.map(r => r.round)) + 1
        : 1;
      const nowStr = new Date().toLocaleString('zh-CN');
      const tempRound = {{
        round: nextRound,
        notes: notes,
        proposal: '',
        created_at: nowStr
      }};
      currentRounds.unshift(tempRound);
      renderRounds();

      try {{
        const resp = await fetch('/api/code-audit/' + currentKey + '/analyze', {{
          method: 'POST', headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{issue: issues.find(i => i.key === currentKey), notes: notes}})
        }});
        const data = await resp.json();
        if (data.success) {{
          status.textContent = '✓ 分析完成 (' + data.reply.length + ' 字符)';
          status.className = 'status-msg success';
          // 刷新 rounds 和左侧列表（会从 DB 加载完整数据）
          await selectIssue(currentKey);
          renderIssueList();
          // 显示确认按钮
          const confirmBtn = document.getElementById('btn-confirm');
          if (confirmBtn) confirmBtn.style.display = 'inline-block';
        }} else {{
          status.textContent = '✗ ' + (data.error || '失败');
          status.className = 'status-msg error';
          // 从后端重新加载状态（后端已设为 failed），刷新左侧列表
          await selectIssue(currentKey);
          renderIssueList();
          // 恢复 notes 到输入框让用户重试
          document.getElementById('notes-input').value = notes;
        }}
      }} catch(err) {{
        status.textContent = '✗ 网络错误: ' + err.message;
        status.className = 'status-msg error';
        // 从后端重新加载状态，刷新左侧列表
        await selectIssue(currentKey);
        renderIssueList();
        document.getElementById('notes-input').value = notes;
      }} finally {{
        btn.disabled = false;
      }}
    }}

    async function executeRound(roundNum) {{
      if (!currentKey) return;
      if (!confirm('确认执行 Round ' + roundNum + ' 的方案？状态将变为「需要修复」。')) return;
      const resp = await fetch('/api/code-audit/' + currentKey + '/execute', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{round: roundNum}})
      }});
      const data = await resp.json();
      if (data.success) {{
        await selectIssue(currentKey);
        renderIssueList();
      }}
    }}

    async function rejectRound(roundNum) {{
      if (!currentKey) return;
      if (!confirm('驳回 Round ' + roundNum + ' 的方案？状态将回到「待处理」，你可以写新要求重新分析。')) return;
      const resp = await fetch('/api/code-audit/' + currentKey + '/reject', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{round: roundNum}})
      }});
      const data = await resp.json();
      if (data.success) {{
        await selectIssue(currentKey);
        renderIssueList();
      }}
    }}

    async function saveStatus() {{
      if (!currentKey) return;
      const notes = document.getElementById('notes-input').value;
      const resp = await fetch('/api/code-audit/' + currentKey, {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{status: 'pending', notes: notes}})
      }});
      const data = await resp.json();
      const statusEl = document.getElementById('save-status');
      if (data.success) {{
        statusEl.textContent = '✓ 已保存';
        statusEl.style.color = 'var(--green)';
        setTimeout(() => statusEl.textContent = '', 2000);
      }}
    }}

    function escapeHtml(text) {{
      if (!text) return '';
      return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    function markdownToHtml(text) {{
      if (!text) return '';
      // 保护代码块
      let html = text;
      const codeBlocks = [];
      html = html.replace(/```([\\s\\S]*?)```/g, function(match, code) {{
        codeBlocks.push(escapeHtml(code));
        return '\\x00CODE' + (codeBlocks.length - 1) + '\\x00';
      }});
      // 行内代码
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      // 标题
      html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
      html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
      html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
      html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
      // 粗体
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      // 表格
      html = html.replace(/((?:^\|.*\|[\\s\\n]*)+)/gm, function(match) {{
        const rows = match.trim().split('\\n').filter(function(r) {{ return r.trim(); }});
        if (rows.length < 2) return match;
        let tbl = '<table style="border-collapse:collapse;margin:8px 0;font-size:12px">';
        rows.forEach(function(row) {{
          if (row.replace(/[\|\-\\s]/g, '') === '') return;
          const cells = row.split('|').filter(function(c) {{ return c !== ''; }}).map(function(c) {{
            return '<td style="border:1px solid var(--border);padding:4px 8px">' + c.trim() + '</td>';
          }});
          tbl += '<tr>' + cells.join('') + '</tr>';
        }});
        tbl += '</table>';
        return tbl;
      }});
      // 分隔线
      html = html.replace(/^---+$/gim, '<hr style="border:0;border-top:1px solid var(--border);margin:12px 0">');
      // 列表
      html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
      html = html.replace(/(<li>.*<\/li>\\s*)+/g, function(match) {{
        return '<ul style="margin-left:16px;margin-bottom:8px">' + match + '</ul>';
      }});
      // 段落处理（代码块还是占位符，不会被破坏）
      html = html.split('\\n').map(function(line) {{
        line = line.trim();
        if (!line) return '';
        if (line.match(/^<[h|p|u|o|t|d|l]/)) return line;
        if (line.indexOf('\\x00CODE') !== -1) return line;
        return '<p style="margin:4px 0">' + line + '</p>';
      }}).join('');
      // 恢复代码块（diff 类型做红绿高亮）
      html = html.replace(/\\x00CODE(\d+)\\x00/g, function(match, idx) {{
        let code = codeBlocks[idx];
        // 去掉末尾多余的换行，避免 <pre> 最后一行是空行
        code = code.replace(/\\n$/, '');
        if (code.indexOf('diff\\n') === 0) {{
          const lines = code.split('\\n').map(function(line) {{
            if (line.indexOf('- ') === 0) {{
              return '<span style="color:#f85149;background:rgba(248,81,73,0.08);padding:1px 4px;border-radius:2px">' + line + '</span>';
            }} else if (line.indexOf('+ ') === 0) {{
              return '<span style="color:#3fb950;background:rgba(63,185,80,0.08);padding:1px 4px;border-radius:2px">' + line + '</span>';
            }}
            return line;
          }});
          code = lines.join('\\n');
        }}
        return '<pre style="background:#0d1117;padding:10px;border-radius:6px;overflow-x:auto;border:1px solid var(--border);margin:8px 0"><code>' + code + '</code></pre>';
      }});
      return html;
    }}

    renderIssueList();
    if (issues.length) selectIssue(issues[0].key);

    // 切回 tab 时自动刷新当前 issue（防止后台分析完成后状态未更新）
    document.addEventListener('visibilitychange', function() {{
      if (!document.hidden && currentKey) {{
        selectIssue(currentKey);
      }}
    }});
    </script>
    <style>
    .audit-layout {{display:flex;height:calc(100vh - 48px);gap:0}}
    .audit-sidebar {{width:280px;flex-shrink:0;border-right:1px solid var(--border);overflow-y:auto;background:var(--card)}}
    .audit-sidebar-header {{padding:12px 16px;font-size:13px;font-weight:600;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--card);z-index:10}}
    .issue-item {{padding:10px 16px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .15s}}
    .issue-item:hover {{background:rgba(255,255,255,.03)}}
    .issue-item.active {{background:rgba(88,166,255,.08);border-left:3px solid var(--blue)}}
    .issue-sev {{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff}}
    .issue-status {{font-size:11px}}
    .issue-title {{font-size:12px;margin-top:4px;line-height:1.4}}
    .audit-main {{flex:1;overflow-y:auto;padding:20px}}
    .issue-header {{margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)}}
    .issue-info {{display:grid;gap:12px;margin-bottom:20px}}
    .info-section {{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}}
    .info-label {{font-size:12px;font-weight:600;margin-bottom:6px}}
    .info-text {{font-size:12px;color:var(--muted);line-height:1.6}}
    .rounds-area {{margin-bottom:20px}}
    .round-block {{margin-bottom:16px}}
    .round-header {{font-size:11px;color:var(--muted);margin-bottom:8px;padding-left:8px}}
    .user-bubble {{background:rgba(88,166,255,.06);border:1px solid rgba(88,166,255,.15);border-radius:8px;padding:12px;margin-bottom:8px}}
    .ai-bubble {{background:rgba(63,185,80,.06);border:1px solid rgba(63,185,80,.15);border-radius:8px;padding:12px}}
    .bubble-label {{font-size:11px;font-weight:600;margin-bottom:6px}}
    .user-bubble .bubble-label {{color:var(--blue)}}
    .ai-bubble .bubble-label {{color:var(--green)}}
    .bubble-content {{font-size:13px;line-height:1.7;white-space:pre-wrap}}
    .bubble-content h1,.bubble-content h2,.bubble-content h3 {{margin:8px 0 4px;font-size:14px}}
    .bubble-content pre {{background:#0d1117;padding:10px;border-radius:6px;overflow-x:auto;border:1px solid var(--border);margin:8px 0}}
    .bubble-content code {{background:#21262d;padding:1px 4px;border-radius:3px;font-family:monospace;font-size:12px}}
    .bubble-content pre code {{background:transparent;padding:0}}
    .bubble-content li {{margin-left:16px;margin-bottom:2px}}
    .input-area {{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;position:sticky;bottom:0}}
    .status-bar {{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-size:12px;color:var(--muted)}}
    .input-area textarea {{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px;font-size:13px;resize:vertical;margin-bottom:8px}}
    .input-actions {{display:flex;gap:8px;align-items:center}}
    .btn-analyze {{background:var(--blue);color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px}}
    .btn-analyze:disabled {{opacity:.5;cursor:not-allowed}}
    .btn-confirm {{background:var(--green);color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px}}
    .btn-save {{background:#30363d;color:var(--text);border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px}}
    .status-msg {{font-size:12px;margin-top:6px}}
    .status-msg.loading {{color:var(--yellow)}}
    .status-msg.success {{color:var(--green)}}
    .status-msg.error {{color:var(--red)}}
    </style>
    """
    return HTMLResponse(_page("代码审计", content, "/code-audit"))


# ============ 对话式代码审计 API ============

async def _analyze_with_kimi(issue: dict, notes: str, timeout: int = 300) -> dict:
    import logging
    logger = logging.getLogger("admin")
    prompt = f"""你是一个代码审计专家。请分析以下代码问题并给出修复方案。

## 问题信息
- 标题: {issue['title']}
- 文件: {issue['file']}
- 行号: {issue['lines']}
- 级别: {issue['severity']}

## 问题描述
{issue['problem']}

## 影响
{issue['impact']}

## 用户反馈/要求
{notes if notes else '请分析问题根因并给出具体修复方案。'}

## 要求
1. 请先读取相关源码文件进行分析
2. 给出具体的修复方案（包含代码 diff）
3. 不要直接修改任何文件，只给出方案
4. 用中文回复
"""
    cmd = [KIMI_BIN, "--quiet", "--yolo", "-p", prompt, "-w", WORK_DIR]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=WORK_DIR,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        reply = stdout.decode('utf-8', errors='replace')
        if proc.returncode != 0:
            err = stderr.decode('utf-8', errors='replace').strip() or f"Kimi 退出码 {proc.returncode}"
            logger.warning(f"Kimi 分析失败: {err}")
            return {"success": False, "error": "Kimi 分析失败，请查看服务端日志"}
        return {"success": True, "reply": reply}
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception as e:
            logger.debug("终止 Kimi 子进程失败: %s", e)
        return {"success": False, "error": f"分析超时（>{timeout}秒）"}
    except Exception as e:
        logger.warning(f"Kimi 分析异常: {e}")
        return {"success": False, "error": "分析异常，请查看服务端日志"}


@app.get("/api/code-audit/{key}")
def api_get_code_audit(key: str):
    db = get_db()
    conn = db._get_conn()
    _ensure_code_audit_migration()
    row = conn.execute("SELECT issue_key, status, notes, ai_proposal FROM code_audit WHERE issue_key=?", (key,)).fetchone()
    rounds = conn.execute(
        "SELECT round_num, user_notes, ai_proposal, created_at FROM code_audit_round WHERE issue_key=? ORDER BY round_num DESC",
        (key,)
    ).fetchall()
    conn.close()
    return JSONResponse({
        "issue_key": key,
        "status": row["status"] if row else "pending",
        "notes": row["notes"] if row else "",
        "ai_proposal": row["ai_proposal"] if row else "",
        "rounds": [{"round": r["round_num"], "notes": r["user_notes"], "proposal": r["ai_proposal"], "created_at": r["created_at"]} for r in rounds],
    })


@app.post("/api/code-audit/{key}/analyze")
async def api_analyze_code_audit(key: str, request: Request):
    body = await request.json()
    notes = body.get("notes", "")
    issue = next((i for i in CODE_AUDIT_ISSUES if i["key"] == key), None)
    if not issue:
        return JSONResponse({"success": False, "error": "Issue not found"})

    db = get_db()
    conn = db._get_conn()

    # 1. 先保存 notes，状态设为 ai_analyzing
    conn.execute("""
        INSERT INTO code_audit (issue_key, severity, status, notes, updated_at)
        VALUES (?, ?, 'ai_analyzing', ?, datetime('now','localtime'))
        ON CONFLICT(issue_key) DO UPDATE SET
        status='ai_analyzing', notes=excluded.notes, updated_at=excluded.updated_at
    """, (key, issue["severity"], notes))

    # 2. 立即创建 round 记录（ai_proposal 先为空），让用户切回 tab 能看到自己的要求
    max_round = conn.execute("SELECT MAX(round_num) FROM code_audit_round WHERE issue_key=?", (key,)).fetchone()[0] or 0
    new_round = max_round + 1
    conn.execute(
        "INSERT INTO code_audit_round (issue_key, round_num, user_notes, ai_proposal) VALUES (?, ?, ?, ?)",
        (key, new_round, notes, ""),
    )
    # 3. 清空 code_audit.notes，因为要求已经归档到 round 中，输入框应该空着准备下一轮
    conn.execute(
        "UPDATE code_audit SET notes='' WHERE issue_key=?",
        (key,)
    )
    conn.commit()
    conn.close()

    # 3. 开始分析
    result = await _analyze_with_kimi(issue, notes, timeout=60000)

    # 4. 分析完成，更新 round 记录
    db = get_db()
    conn = db._get_conn()
    if result["success"]:
        conn.execute(
            "UPDATE code_audit_round SET ai_proposal=? WHERE issue_key=? AND round_num=?",
            (result["reply"], key, new_round),
        )
        conn.execute("""
            INSERT INTO code_audit (issue_key, severity, status, notes, ai_proposal, updated_at)
            VALUES (?, ?, 'rethink', ?, ?, datetime('now','localtime'))
            ON CONFLICT(issue_key) DO UPDATE SET
            status='rethink', notes=excluded.notes, ai_proposal=excluded.ai_proposal, updated_at=excluded.updated_at
        """, (key, issue["severity"], notes, result["reply"]))
    else:
        # 分析失败，round 记录错误信息，状态改为 failed
        err_msg = "分析失败: " + result.get("error", "未知错误")
        conn.execute(
            "UPDATE code_audit_round SET ai_proposal=? WHERE issue_key=? AND round_num=?",
            (err_msg, key, new_round),
        )
        conn.execute(
            "UPDATE code_audit SET status='failed', ai_proposal=? WHERE issue_key=?",
            (err_msg, key),
        )
    conn.commit()
    conn.close()
    return JSONResponse({"success": result["success"], "reply": result.get("reply", ""), "error": result.get("error", ""), "round": new_round})


@app.post("/api/code-audit/{key}/execute")
async def api_execute_code_audit(key: str, request: Request):
    db = get_db()
    conn = db._get_conn()
    conn.execute(
        "UPDATE code_audit SET status='todo', updated_at=datetime('now','localtime') WHERE issue_key=?",
        (key,),
    )
    conn.commit()
    conn.close()
    return JSONResponse({"success": True})


@app.post("/api/code-audit/{key}/reject")
async def api_reject_code_audit(key: str, request: Request):
    db = get_db()
    conn = db._get_conn()
    conn.execute(
        "UPDATE code_audit SET status='pending', updated_at=datetime('now','localtime') WHERE issue_key=?",
        (key,),
    )
    conn.commit()
    conn.close()
    return JSONResponse({"success": True})


# ── Wiki 事实审核台（消费 review.json / decisions.json）──
WIKI_AUDIT_DIR = Path(__file__).parent.parent / "data" / "memory" / "wiki_audit"
WIKI_USERS_DIR = Path(__file__).parent.parent / "data" / "memory" / "wiki" / "users"
WIKI_GROUPS_DIR = Path(__file__).parent.parent / "data" / "memory" / "wiki" / "groups"


def _load_review_items() -> list:
    p = WIKI_AUDIT_DIR / "review.json"
    if not p.exists():
        return []
    return _json.loads(p.read_text(encoding="utf-8"))


def _load_decisions() -> dict:
    """返回 {id: decision}。"""
    p = WIKI_AUDIT_DIR / "decisions.json"
    if not p.exists():
        return {}
    return {d["id"]: d for d in _json.loads(p.read_text(encoding="utf-8"))}


def _wiki_line_context(name: str, is_group: bool, needle: str, span: int = 2) -> str:
    """返回 needle 所在行 ±span 的上下文（带行号），帮助定位。"""
    d = WIKI_GROUPS_DIR if is_group else WIKI_USERS_DIR
    path = d / f"{name}.md"
    if not path.exists():
        return "(wiki 文件不存在)"
    lines = path.read_text(encoding="utf-8").split("\n")
    idx = -1
    for i, ln in enumerate(lines):
        if needle and needle[:30] in ln:
            idx = i
            break
    if idx < 0:
        return "(行未找到，可能已被清洗)"
    lo, hi = max(0, idx - span), min(len(lines), idx + span + 1)
    out = []
    for i in range(lo, hi):
        mark = "▶" if i == idx else " "
        out.append(f"{mark} {i+1:3d} | {html.escape(lines[i])}")
    return "\n".join(out)


@app.get("/wiki-review", response_class=HTMLResponse)
def wiki_review_list():
    items = _load_review_items()
    decisions = _load_decisions()
    # 统计
    n_total = len(items)
    n_decided = len(decisions)
    n_pending = n_total - n_decided
    # 按 wiki 分组计数
    by_wiki: dict = {}
    for it in items:
        by_wiki.setdefault(it["wiki"], 0)
        by_wiki[it["wiki"]] += 1
    wiki_chips = " ".join(f'<span class="wchip" data-w="{html.escape(w)}">{html.escape(w)} ({c})</span>' for w, c in sorted(by_wiki.items()))

    rows = ""
    for it in items:
        dec = decisions.get(it["id"])
        status = '<span class="st st-done">已确认</span>' if dec else '<span class="st st-pending">待确认</span>'
        reason = html.escape(it.get("reason", ""))
        fact = html.escape(it.get("fact", "") or it.get("line", "") or it.get("wiki_excerpt", ""))
        action_badge = ""
        if dec:
            a = dec.get("action")
            label = {"delete": "🗑删除", "fix": "✏️修正", "mark": "❓标待验证", "skip": "⏭️跳过"}.get(a, a)
            action_badge = f'<span class="badge badge-{a}">{label}</span>'
            if a == "fix":
                action_badge += f'<span style="color:var(--green);font-size:11px"> → {html.escape(dec.get("new_value","")[:40])}</span>'
        rows += f"""<tr class="rrow" data-w="{html.escape(it['wiki'])}">
          <td>{html.escape(it['id'])}</td>
          <td>{html.escape(it['wiki'])}{'(群)' if it.get('is_group') else ''}</td>
          <td>{reason}</td>
          <td style="font-size:12px">{fact[:80]}</td>
          <td>{status}</td>
          <td>{action_badge}</td>
          <td><a href="/wiki-review/{it['id']}" style="color:var(--blue)">处理 →</a></td>
        </tr>"""

    content = f"""
    <div class="metrics">
      <div class="metric"><div class="value">{n_total}</div><div class="label">总条目</div></div>
      <div class="metric"><div class="value" style="color:var(--yellow)">{n_pending}</div><div class="label">待确认</div></div>
      <div class="metric"><div class="value" style="color:var(--green)">{n_decided}</div><div class="label">已确认</div></div>
    </div>
    <div class="card"><b>按 wiki：</b> {wiki_chips}</div>
    <div class="card" style="border-left:3px solid var(--blue)">
      <b>写回 wiki：</b> 确认完毕后，在服务器执行
      <code style="background:#1c2128;padding:2px 6px;border-radius:4px">python3 scripts/clean_wiki_errors.py --apply-decisions</code>
      （带备份，仅写已确认非 skip 项）
    </div>
    <div style="margin:8px 0">
      <label style="font-size:12px;color:var(--muted)"><input type="checkbox" id="only-pending" checked> 只看待确认</label>
    </div>
    <table id="rtable"><tr><th>ID</th><th>Wiki</th><th>类型</th><th>内容</th><th>状态</th><th>决策</th><th></th></tr>{rows}</table>
    <script>
    document.querySelectorAll('.wchip').forEach(c=>c.addEventListener('click',function(){{document.querySelectorAll('.rrow').forEach(r=>{{r.style.display=(!this.dataset.w||r.dataset.w===this.dataset.w)?'':'none'}})}}));
    document.getElementById('only-pending').addEventListener('change',function(){{document.querySelectorAll('.rrow').forEach(r=>{{const done=r.querySelector('.st-done');r.style.display=(this.checked&&done)?'none':''}})}});
    </script>
    <style>.wchip{{display:inline-block;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:3px 10px;font-size:11px;margin:2px;cursor:pointer}}.wchip:hover{{border-color:var(--blue)}}.st{{font-size:11px;padding:2px 6px;border-radius:4px}}.st-pending{{background:rgba(210,153,34,.15);color:var(--yellow)}}.st-done{{background:rgba(63,185,80,.15);color:var(--green)}}.badge{{font-size:11px;padding:2px 6px;border-radius:4px;margin-right:4px}}.badge-delete{{background:rgba(248,81,73,.15);color:var(--red)}}.badge-fix{{background:rgba(88,166,255,.15);color:var(--blue)}}.badge-mark{{background:rgba(210,153,34,.15);color:var(--yellow)}}.badge-skip{{background:rgba(139,148,158,.15);color:var(--muted)}}</style>
    """
    return HTMLResponse(_page("Wiki 事实审核", content, "/wiki-review"))


@app.get("/wiki-review/{item_id}", response_class=HTMLResponse)
def wiki_review_detail(item_id: str):
    items = _load_review_items()
    it = next((x for x in items if x["id"] == item_id), None)
    if not it:
        return HTMLResponse(_page("未找到", "<p>条目不存在</p>", "/wiki-review"), status_code=404)
    decisions = _load_decisions()
    dec = decisions.get(item_id)
    needle = it.get("line") or it.get("wiki_excerpt") or it.get("fact", "")
    ctx = _wiki_line_context(it["wiki"], it.get("is_group", False), needle)
    # 当前决策
    cur_action = dec.get("action") if dec else ""
    cur_value = dec.get("new_value", "") if dec else ""

    evidence = ""
    if it.get("evidence"):
        hal = it.get("hallucinated_evidence", False)
        hal_badge = ' <span style="color:var(--red);font-size:11px">⚠️ 证据疑似LLM编造（不在检索片段中），已降级 unverified</span>' if hal else ""
        border = "var(--red)" if hal else "var(--muted)"
        evidence = f'<div class="card" style="border-left:3px solid {border}"><b>audit 证据：</b>{hal_badge}<pre style="font-size:12px;white-space:pre-wrap">{html.escape(it["evidence"])}</pre></div>'

    # 原始检索片段：让用户逐字核对 LLM 引用的证据是否真实存在
    raw_ev_block = ""
    if it.get("raw_evidence"):
        raw_ev_block = f'<div class="card" style="border-left:3px solid var(--green)"><b>原始检索片段（逐字核对用，LLM 引用的证据必须在此段内）：</b><pre style="font-size:10px;white-space:pre-wrap;max-height:300px;overflow:auto">{html.escape(it["raw_evidence"])}</pre></div>'

    content = f"""
    <div class="card"><b>{html.escape(it['id'])}</b> · <b>{html.escape(it['wiki'])}</b>{'(群)' if it.get('is_group') else ''}
      · <span style="color:var(--yellow)">{html.escape(it.get('reason',''))}</span></div>
    <div class="card" style="border-left:3px solid var(--blue)">
      <b>涉及事实：</b><br><span style="font-size:13px">{html.escape(it.get('fact','') or '')}</span>
      {"<br><br><b>wiki 原文片段：</b><br><code style='font-size:12px'>"+html.escape(it.get('wiki_excerpt','') or '')+"</code>" if it.get('wiki_excerpt') else ''}
      {"<br><br><b>整行：</b><br><code style='font-size:12px'>"+html.escape(it.get('line','') or '')+"</code>" if it.get('line') else ''}
    </div>
    {evidence}
    {raw_ev_block}
    <div class="card" style="border-left:3px solid var(--green)">
      <b>wiki 上下文（▶ 标记目标行）：</b>
      <pre style="font-size:11px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace">{ctx}</pre>
    </div>
    <div class="card" id="decision-panel">
      <b>你的确认：</b>
      <div style="margin:10px 0;display:flex;gap:8px;flex-wrap:wrap">
        <button class="dbtn" data-action="delete" style="background:rgba(248,81,73,.15);color:var(--red);border:1px solid var(--red)">🗑 删除整行</button>
        <button class="dbtn" data-action="fix" style="background:rgba(88,166,255,.15);color:var(--blue);border:1px solid var(--blue)">✏️ 修正为</button>
        <input id="new-value" placeholder="修正后的正确内容（如：高中同学）" value="{html.escape(cur_value)}"
          style="flex:1;min-width:200px;background:#1c2128;border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px;font-size:13px">
        <button class="dbtn" data-action="mark" style="background:rgba(210,153,34,.15);color:var(--yellow);border:1px solid var(--yellow)">❓ 标[待验证]</button>
        <button class="dbtn" data-action="skip" style="background:rgba(139,148,158,.15);color:var(--muted);border:1px solid var(--muted)">⏭ 确认无误跳过</button>
      </div>
      <div id="msg" style="font-size:12px;margin-top:6px"></div>
      <div style="margin-top:8px"><a href="/wiki-review" style="color:var(--blue);font-size:13px">← 返回列表</a></div>
    </div>
    <script>
    const id={_json.dumps(item_id)};
    const curAction={_json.dumps(cur_action)};
    const allIds={_json.dumps([x["id"] for x in items])};
    const decidedIds={_json.dumps(list(decisions.keys()))};
    function nextPendingId(){{const i=allIds.indexOf(id);for(let k=i+1;k<allIds.length;k++){{if(!decidedIds.includes(allIds[k]))return allIds[k]}}for(let k=0;k<i;k++){{if(!decidedIds.includes(allIds[k]))return allIds[k]}}return null}}
    function highlight(){{document.querySelectorAll('.dbtn').forEach(b=>{{b.style.outline=b.dataset.action===curAction?'2px solid #fff':'none'}})}}
    highlight();
    document.querySelectorAll('.dbtn').forEach(b=>b.addEventListener('click',async()=>{{
      const action=b.dataset.action;
      const nv=document.getElementById('new-value').value.trim();
      if(action==='fix'&&!nv){{document.getElementById('msg').innerHTML='<span style="color:var(--red)">请填写修正值</span>';return}}
      document.getElementById('msg').innerHTML='<span style="color:var(--muted)">提交中...</span>';
      const r=await fetch('/api/wiki-review/'+id,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:_json.stringify({{action,new_value:nv}})}});
      const j=await r.json();
      if(j.success){{
        decidedIds.push(id);
        const nxt=nextPendingId();
        const link=nxt?`<a href="/wiki-review/${{nxt}}" style="color:var(--blue)">下一条 →</a>`:'<a href="/wiki-review" style="color:var(--green)">全部确认完毕 →</a>';
        document.getElementById('msg').innerHTML=`<span style="color:var(--green)">✅ 已保存（${{action}}）。</span> ${{link}}`;
        highlight();
      }}else{{document.getElementById('msg').innerHTML='<span style="color:var(--red)">失败</span>'}}
    }}));
    </script>
    """
    return HTMLResponse(_page("Wiki 审核 · " + it["wiki"], content, "/wiki-review"))


@app.post("/api/wiki-review/{item_id}")
async def wiki_review_save(item_id: str, request: Request):
    body = await request.json()
    action = body.get("action")
    if action not in ("delete", "fix", "mark", "skip"):
        return JSONResponse({"success": False, "error": "非法 action"}, status_code=400)
    items = _load_review_items()
    it = next((x for x in items if x["id"] == item_id), None)
    if not it:
        return JSONResponse({"success": False, "error": "条目不存在"}, status_code=404)
    dec = {
        "id": item_id,
        "wiki": it["wiki"],
        "is_group": it.get("is_group", False),
        "line": it.get("line") or it.get("wiki_excerpt") or it.get("fact", ""),
        "action": action,
        "new_value": body.get("new_value", ""),
        "decided_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 合并到 decisions.json（去重）
    p = WIKI_AUDIT_DIR / "decisions.json"
    existing = []
    if p.exists():
        existing = _json.loads(p.read_text(encoding="utf-8"))
    existing = [d for d in existing if d["id"] != item_id]
    existing.append(dec)
    p.write_text(_json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse({"success": True})


if __name__ == "__main__":
    import uvicorn
    print("wechat-twin Admin → http://localhost:8765")
    uvicorn.run(app, host="127.0.0.1", port=8766)
