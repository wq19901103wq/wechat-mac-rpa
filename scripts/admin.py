#!/usr/bin/env python3
"""wechat-twin Admin — 统一开发者后台"""
import logging
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import html
import json as _json
import subprocess  # nosec B404
from typing import Any, Dict, Optional

_logger = logging.getLogger("scripts.admin")


from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from src.badcase.case_db import get_db

WORK_DIR = str(Path(__file__).parent.parent)
KIMI_BIN = "/Users/yihanwang/.local/bin/kimi"

app = FastAPI(title="wechat-twin Admin")

# 截图列表页缓存（模块级，首次扫描后复用）
_screenshot_cache: Optional[Dict[str, Any]] = None

HEADER = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>wechat-twin Admin</title>
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
</style></head><body>
<div id="lightbox" onclick="this.classList.remove('show')"><img id="lightbox-img" src=""></div>
<nav>
<div style="padding:12px 20px;font-weight:700;font-size:15px;margin-bottom:8px">wechat-twin</div>
<a href="/">📊 Dashboard</a>
<a href="/ticks">🔍 Tick 查看</a>
<a href="/gt">🏷️ GT 标注</a>
<a href="/review">🧑‍⚖️ 审核</a>
<a href="/screenshots">📸 截图OCR</a>
<a href="/benchmark/judge">📊 Judge质量</a>
<a href="/benchmark/reply">🤖 回复质量</a>
<a href="/experiments">🧪 实验</a>
<a href="/code-audit">🐛 代码审计</a>
</nav><main>"""

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
    for href, label in [("/", "📊 Dashboard"), ("/ticks", "🔍 Tick"), ("/gt", "🏷️ GT"), ("/review", "🧑‍⚖️ 审核"), ("/screenshots", "📸 截图OCR"), ("/benchmark/judge", "📊 Judge"), ("/benchmark/reply", "🤖 回复"), ("/experiments", "🧪 实验"), ("/code-audit", "🐛 审计"), ("/wiki-review", "🧠 Wiki审核")]:
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
            "SELECT id, session_id, tick_id, chat_name, messages_count, new_messages_count, should_reply, skip_reason, judge_score, human_is_badcase, human_badcase_type, replies_sent_json, duration_ms, created_at FROM tick_log WHERE skip_reason IS NOT NULL ORDER BY created_at DESC, tick_id DESC LIMIT ? OFFSET ?",
            (20, offset),
        ).fetchall()
    elif filter == "replied":
        rows = conn.execute(
            "SELECT id, session_id, tick_id, chat_name, messages_count, new_messages_count, should_reply, skip_reason, judge_score, human_is_badcase, human_badcase_type, replies_sent_json, duration_ms, created_at FROM tick_log WHERE should_reply=1 ORDER BY created_at DESC, tick_id DESC LIMIT ? OFFSET ?",
            (20, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, session_id, tick_id, chat_name, messages_count, new_messages_count, should_reply, skip_reason, judge_score, human_is_badcase, human_badcase_type, replies_sent_json, duration_ms, created_at FROM tick_log ORDER BY created_at DESC, tick_id DESC LIMIT ? OFFSET ?",
            (20, offset),
        ).fetchall()
    conn.close()

    rows_html = ""
    for r in rows:
        status = "⏭️跳过" if r["skip_reason"] else "✅回复" if r["should_reply"] else "⏭️无消息"
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
    status = d.get("skip_reason") or ("已回复" if d.get("should_reply") else "无消息")
    content = f"""
    <div class="card"><b>{html.escape(d.get("session_id",""))}:#{d["tick_id"]}</b> — {html.escape(d.get("created_at",""))}</div>
    <div class="card"><b>聊天:</b> {html.escape(d.get("chat_name","?"))} {"(群)" if d.get("is_group") else "(私)"} | <b>状态:</b> {html.escape(status)} | <b>耗时:</b> {ms}ms</div>
    <div class="card"><b>消息:</b> 总{d.get("messages_count",0)}条 新{d.get("new_messages_count",0)}条 | <b>发送:</b> {"OK" if d.get("send_success") else "N/A"}</div>
    <div class="card"><b>Bot 回复:</b><br>{replies_display}</div>
    <div class="card" style="border-left:3px solid var(--blue)"><b>📝 System Prompt ({len(sp)}字)</b><pre style="font-size:10px;white-space:pre-wrap">{html.escape(sp)}</pre></div>
    <div class="card" style="border-left:3px solid var(--green)"><b>📝 User Prompt ({len(up)}字)</b><pre style="font-size:10px;white-space:pre-wrap">{html.escape(up)}</pre></div>
    <div class="card" style="border-left:3px solid var(--muted)"><b>📝 Raw Response</b><pre style="font-size:10px;white-space:pre-wrap">{html.escape(raw)}</pre></div>
    """
    # 工具调用 + 结果（合并 tool_calls_json 和 tool_results_json）
    try:
        import json as _j3
        tc_list = _j3.loads(tools) if tools else []
        tr_list = _j3.loads(tool_results) if tool_results else []
        # 合并：优先用 tool_results_json 的完整结果，没有的话 fallback 到 result_preview
        all_tools = []
        seen = set()
        for tr in tr_list:
            name = tr.get("tool", "?")
            seen.add(name)
            all_tools.append({"tool_name": name, "arguments": tr.get("args", ""), "result_preview": tr.get("result", "")})
        for t in tc_list:
            name = t.get('tool_name', '?')
            if name not in seen:
                all_tools.append(t)  # 保持原始 result_preview（500字）
        if all_tools:
            tools_html = ""
            for t in all_tools:
                tname = t.get('tool_name', '?')
                targs = t.get('arguments', '') or ''
                tresult = t.get('result_preview', '') or ''
                # Parse args if JSON string
                try:
                    args_obj = _j3.loads(targs) if isinstance(targs, str) else targs
                    targs = ' '.join(f'{k}={v}' for k,v in (args_obj.items() if isinstance(args_obj, dict) else []))
                except Exception as e:
                    _logger.debug("解析工具参数失败: %s", e)
                tools_html += f"""<div style="margin:8px 0;padding:10px;background:rgba(255,255,255,.03);border-left:3px solid var(--yellow);border-radius:4px">
                  <div style="font-size:12px;margin-bottom:4px"><b style="color:var(--yellow)">{html.escape(tname)}</b> <span style="color:var(--muted);font-size:10px">{html.escape(targs)}</span></div>
                  <pre style="font-size:11px;max-height:250px;overflow:auto;white-space:pre-wrap;background:rgba(0,0,0,.3);padding:8px;border-radius:4px;margin:0">{html.escape(tresult)}</pre>
                </div>"""
            content += f"""<div class="card" style="border-left:3px solid var(--yellow)"><b>🔧 工具调用 & 结果 ({len(all_tools)}项)</b>{tools_html}</div>"""
    except Exception as e:
        _logger.warning("渲染工具调用失败: %s", e)
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
    """截图 + OCR 列表页 — 从 debug JSON 目录查。首次加载慢(~20s)，后续走浏览器缓存。"""
    import time as _time
    _t0 = _time.time()
    # 用 os.listdir 扫一遍，缓存到模块级变量，重启前不重复扫
    global _screenshot_cache
    cache_key = "files"
    if refresh:
        _screenshot_cache = {}
    if "_screenshot_cache" not in globals() or _screenshot_cache is None:
        _screenshot_cache = {}
    if cache_key not in _screenshot_cache:
        files = [f for f in os.listdir(str(DEBUG_DIR)) if f.startswith("tick_") and f.endswith(".json")]
        files.sort(reverse=True)
        _screenshot_cache[cache_key] = files
    all_files = _screenshot_cache[cache_key]
    MAX_SCAN = 500
    scan_files = all_files[:MAX_SCAN * 2]  # 多扫一些，filter 可能过滤掉很多

    all_debug = []
    for fname in scan_files:
        try:
            f = DEBUG_DIR / fname
            dbg = _json.loads(f.read_text(encoding="utf-8"))
            has_api = bool(dbg.get("api_prompt"))
            if filter == "api" and not has_api:
                continue
            if filter == "skip" and has_api:
                continue
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
            all_debug.append((tid, sp, chat_name, ocr_summary, has_api, fpath))
            if len(all_debug) >= MAX_SCAN:
                break
        except Exception as e:
            _logger.debug("读取截图 debug 文件 %s 失败: %s", fname, e)
            continue

    total = len(all_debug)
    offset = (page - 1) * limit
    page_items = all_debug[offset:offset + limit]

    rows_html = ""
    for tid, sp, chat_name, ocr_summary, has_api, fpath in page_items:
        fname = Path(sp).name if sp else ""
        img_tag = '<img src="/api/screenshot-image/' + fname + '" style="max-width:240px;max-height:160px;border-radius:4px;border:1px solid var(--border)" loading="lazy" onerror="this.style.display=\'none\'">' if fname and Path(sp).exists() else '<span style="color:var(--muted)">—</span>'
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
      <a href="?page={page-1}&limit={limit}&filter={safe_filter}" style="color:var(--blue);margin-right:12px" {'hidden' if page<=1 else ''}>上一页</a>
      第 {page} 页 / 共 {(total+limit-1)//limit} 页
      <a href="?page={page+1}&limit={limit}&filter={safe_filter}" style="color:var(--blue);margin-left:12px">下一页</a>
    </div>"""
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
                for e in ocr_elems:
                    text = e.get("text", "")
                    bbox = e.get("bbox", [])
                    conf = e.get("confidence", 0)
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
        # 去掉 <h1>
        h1_end = body.find("</h1>")
        if h1_end > 0:
            body = body[h1_end + 5:]
        return _page(title, style + body, active)
    return _page(title, raw, active)

@app.get("/benchmark/judge", response_class=HTMLResponse)
def benchmark_judge():
    return HTMLResponse(_embed_benchmark(BENCHMARK_JUDGE, "Judge Quality Benchmark", "/benchmark/judge"))

@app.get("/benchmark/reply", response_class=HTMLResponse)
def benchmark_reply():
    return HTMLResponse(_embed_benchmark(BENCHMARK_REPLY, "Bot 回复质量 Benchmark", "/benchmark/reply"))

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

@app.get("/experiments", response_class=HTMLResponse)
def experiments_list():
    db = get_db()
    conn = db._get_conn()
    exps = conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    conn.close()

    if not exps:
        content = "<p style='color:var(--muted)'>暂无实验。运行 <code>python3 scripts/run_experiment.py --exp &lt;name&gt; --all-labeled</code></p>"
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

    content = f"""<table>
    <tr><th>实验</th><th>描述</th><th>N</th><th>Badcase 率</th><th>均分</th><th>结论</th><th>时间</th></tr>
    {rows}</table>"""
    return HTMLResponse(_page("A/B 实验", content, "/experiments"))


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
        "key": "weflow-mode-check",
        "severity": "P1",
        "title": "WeFlow 模式判断 hasattr 永远为 True",
        "file": "src/session/global_store.py",
        "lines": "390",
        "github_url": "https://github.com/wq19901103wq/wechat-mac-rpa/blob/main/src/session/global_store.py#L390",
        "problem": "hasattr(messages[0], 'local_id') 对任何 ChatMessage 永远为True（dataclass定义了该字段）。",
        "impact": "当前_weflow_mode='ocr'，不会触发此分支。但如果未来启用WeFlow持续模式，OCR消息会错误进入_merge_tick_weflow。",
        "fix": "messages[0].local_id is not None"
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
