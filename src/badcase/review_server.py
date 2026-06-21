#!/usr/bin/env python3
"""
Badcase 审核台 - FastAPI + 原生 HTML/JS

启动：
    python scripts/review_server.py
    # 或
    uvicorn src.badcase.review_server:app --host 0.0.0.0 --port 8765 --reload

访问：http://localhost:8765
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
PENDING_DIR = PROJECT_ROOT / "data" / "review_drafts" / "pending"
COMMITTED_DIR = PROJECT_ROOT / "data" / "review_drafts" / "committed"
DISMISSED_DIR = PROJECT_ROOT / "data" / "review_drafts" / "dismissed"

for d in (PENDING_DIR, COMMITTED_DIR, DISMISSED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> Optional[str]:
    """只允许纯文件名（不含路径分隔符），防止目录遍历。"""
    if not name:
        return None
    stripped = name.replace("/", "").replace("\\", "").replace("..", "")
    if not stripped or stripped in (".", ".."):
        return None
    return stripped


def _safe_project_path(rel: str) -> Optional[Path]:
    """只允许项目根目录下的相对路径（白名单字符），防止目录遍历。"""
    import re
    if not rel or rel.startswith("/") or not re.fullmatch(r"[a-zA-Z0-9_./\-]+", rel):
        return None
    try:
        # lgtm[py/path-injection] rel 已限制为白名单字符且不含 ..
        target = (PROJECT_ROOT / rel).resolve()
        # lgtm[py/path-injection] 已验证 target 在项目根目录内且为文件
        root = PROJECT_ROOT.resolve()
        if target.is_file() and (target == root or root in target.parents):
            return target
    except (ValueError, OSError):
        pass
    return None


app = FastAPI(title="Badcase Review")

# ------------------------------------------------------------------
# 前端 HTML 模板
# ------------------------------------------------------------------

_LIST_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Badcase 审核台</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:20px}
.header{text-align:center;margin-bottom:24px}
.header h1{font-size:24px;margin-bottom:6px}
.stats{display:flex;gap:12px;justify-content:center;margin-bottom:20px;flex-wrap:wrap}
.stat{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 20px;min-width:100px;text-align:center}
.stat .num{font-size:22px;font-weight:700;color:var(--blue)}
.stat .label{font-size:12px;color:var(--muted)}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;justify-content:center}
.filters select,.filters input{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:14px}
.drafts{display:grid;gap:10px;max-width:900px;margin:0 auto}
.draft{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;cursor:pointer;transition:transform .15s,border-color .15s}
.draft:hover{transform:translateY(-1px);border-color:var(--blue)}
.draft-header{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.draft-id{font-size:13px;color:var(--muted);font-family:monospace}
.sev{font-size:11px;padding:2px 8px;border-radius:12px;font-weight:600}
.sev.P0{background:rgba(248,81,73,.15);color:var(--red)}
.sev.P1{background:rgba(211,153,34,.15);color:var(--yellow)}
.sev.P2{background:rgba(88,166,255,.15);color:var(--blue)}
.type{font-size:12px;color:var(--muted);background:rgba(255,255,255,.05);padding:2px 8px;border-radius:4px}
.preview{color:var(--text);font-size:14px;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.reason{font-size:12px;color:var(--muted);margin-bottom:4px}
.meta{display:flex;gap:12px;font-size:12px;color:var(--muted)}
.conf-bar{height:4px;background:var(--border);border-radius:2px;margin-top:8px;overflow:hidden}
.conf-inner{height:100%;border-radius:2px}
.empty{text-align:center;color:var(--muted);padding:60px 20px}
</style>
</head>
<body>
<div class="header"><h1>🤖 Badcase 审核台</h1></div>
<div class="stats">
  <div class="stat"><div class="num" id="stat-pending">-</div><div class="label">待审</div></div>
  <div class="stat"><div class="num" id="stat-p0">-</div><div class="label">P0</div></div>
  <div class="stat"><div class="num" id="stat-p1">-</div><div class="label">P1</div></div>
  <div class="stat"><div class="num" id="stat-p2">-</div><div class="label">P2</div></div>
</div>
<div class="filters">
  <select id="filter-status" onchange="render()">
    <option value="pending">待审</option>
    <option value="committed">已入库</option>
    <option value="dismissed">已丢弃</option>
  </select>
  <select id="filter-severity" onchange="render()">
    <option value="">全部 severity</option>
    <option value="P0">P0</option>
    <option value="P1">P1</option>
    <option value="P2">P2</option>
  </select>
  <select id="filter-module" onchange="render()">
    <option value="">全部 module</option>
    <option value="P0">P0 Tool</option>
    <option value="P2">P2 Reply</option>
    <option value="P3">P3 Multi-turn</option>
  </select>
  <input type="text" id="filter-search" placeholder="搜索..." oninput="render()">
</div>
<div class="drafts" id="drafts-container">
  <div class="empty">加载中...</div>
</div>

<script>
let allDrafts = [];
async function load() {
  const res = await fetch('/api/drafts');
  allDrafts = await res.json();
  render();
}
function render() {
  const status = document.getElementById('filter-status').value;
  const sev = document.getElementById('filter-severity').value;
  const mod = document.getElementById('filter-module').value;
  const search = document.getElementById('filter-search').value.toLowerCase();

  // 统计
  const pending = allDrafts.filter(d => d.status === 'pending');
  document.getElementById('stat-pending').textContent = pending.length;
  document.getElementById('stat-p0').textContent = pending.filter(d => d.severity === 'P0').length;
  document.getElementById('stat-p1').textContent = pending.filter(d => d.severity === 'P1').length;
  document.getElementById('stat-p2').textContent = pending.filter(d => d.severity === 'P2').length;

  let filtered = allDrafts.filter(d => d.status === status);
  if (sev) filtered = filtered.filter(d => d.severity === sev);
  if (mod) filtered = filtered.filter(d => d.module === mod);
  if (search) filtered = filtered.filter(d =>
    (d.bot_reply_preview || '').toLowerCase().includes(search) ||
    (d.reason || '').toLowerCase().includes(search)
  );

  const container = document.getElementById('drafts-container');
  if (filtered.length === 0) {
    container.innerHTML = '<div class="empty">暂无数据</div>';
    return;
  }
  container.innerHTML = filtered.map(d => `
    <div class="draft" onclick="location.href='/draft/${d.draft_id}'">
      <div class="draft-header">
        <span class="draft-id">${d.draft_id}</span>
        <span class="sev ${d.severity}">${d.severity}</span>
        <span class="type">${d.badcase_type}</span>
        <span class="type">${d.module || 'P2'}</span>
      </div>
      <div class="preview">${escapeHtml(d.bot_reply_preview || '')}</div>
      <div class="reason">${escapeHtml(d.reason || '')}</div>
      <div class="meta">
        <span>置信度: ${(d.confidence * 100).toFixed(0)}%</span>
        <span>${d.timestamp?.slice(0, 16) || ''}</span>
      </div>
      <div class="conf-bar"><div class="conf-inner" style="width:${d.confidence * 100}%;background:${d.confidence >= 0.9 ? 'var(--green)' : d.confidence >= 0.7 ? 'var(--yellow)' : 'var(--red)'}"></div></div>
    </div>
  `).join('');
}
function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
load();
</script>
</body>
</html>
"""

_DETAIL_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>审核详情 - {draft_id}</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#bc8cff;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:20px}
.container{max-width:1100px;margin:0 auto}
.header{display:flex;align-items:center;gap:12px;margin-bottom:20px}
.header h1{font-size:20px}
.back{color:var(--blue);text-decoration:none;font-size:14px}
.back:hover{text-decoration:underline}
.sev{font-size:12px;padding:2px 10px;border-radius:12px;font-weight:600}
.sev.P0{background:rgba(248,81,73,.15);color:var(--red)}
.sev.P1{background:rgba(211,153,34,.15);color:var(--yellow)}
.sev.P2{background:rgba(88,166,255,.15);color:var(--blue)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
.panel h2{font-size:14px;color:var(--muted);margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
.msg{margin-bottom:10px;padding:10px 12px;border-radius:8px;font-size:13px}
.msg.user{background:rgba(88,166,255,.08);border-left:3px solid var(--blue)}
.msg.bot{background:rgba(188,140,255,.08);border-left:3px solid var(--purple)}
.msg .sender{font-size:11px;color:var(--muted);margin-bottom:4px}
.msg .text{white-space:pre-wrap;word-break:break-word}
.json-view{background:rgba(0,0,0,.2);border-radius:6px;padding:12px;font-family:monospace;font-size:12px;white-space:pre-wrap;overflow:auto;max-height:300px}
.full-width{grid-column:1 / -1}
.textarea{width:100%;min-height:200px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:12px;font-family:monospace;font-size:13px;resize:vertical}
.actions{display:flex;gap:12px;margin-top:20px;justify-content:center}
.btn{border:none;border-radius:8px;padding:10px 24px;font-size:14px;cursor:pointer;font-weight:600;transition:opacity .15s}
.btn:hover{opacity:.85}
.btn-green{background:var(--green);color:#fff}
.btn-blue{background:var(--blue);color:#fff}
.btn-red{background:var(--red);color:#fff}
.screenshot{max-width:100%;border-radius:8px;border:1px solid var(--border);cursor:pointer}
.note{font-size:12px;color:var(--muted);margin-top:8px;text-align:center}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <a href="/" class="back">← 返回列表</a>
    <h1>审核详情</h1>
    <span class="sev {severity}">{severity}</span>
    <span style="font-size:13px;color:var(--muted)">{draft_id}</span>
  </div>

  <div class="grid">
    <div class="panel">
      <h2>📥 对话上下文</h2>
      <div id="conversation">{conversation_html}</div>
    </div>
    <div class="panel">
      <h2>🧑‍⚖️ LLM 判定结果</h2>
      <div class="json-view">{judge_json}</div>
    </div>
    <div class="panel">
      <h2>📤 Bot 实际回复</h2>
      <div class="msg bot"><div class="text">{bot_reply}</div></div>
      <h2 style="margin-top:16px">🖼️ 截图</h2>
      {screenshot_html}
    </div>
    <div class="panel">
      <h2>💾 资产链接</h2>
      <div style="font-size:13px;line-height:2">
        {assets_links}
      </div>
    </div>
    <div class="panel full-width">
      <h2>🔍 Bot 实际看到的完整上下文（Judge 判定依据）</h2>
      <h3 style="margin-top:12px;font-size:13px;color:var(--muted)">System Prompt</h3>
      <div class="json-view">{system_prompt_html}</div>
      <h3 style="margin-top:12px;font-size:13px;color:var(--muted)">Tools 定义</h3>
      <div class="json-view">{tools_context_html}</div>
      <h3 style="margin-top:12px;font-size:13px;color:var(--muted)">完整 User Prompt（实际发给 LLM 的 user 消息）</h3>
      <div class="json-view">{user_prompt_html}</div>
      <h3 style="margin-top:12px;font-size:13px;color:var(--muted)">完整 Messages 列表（含 tool 返回结果）</h3>
      <div class="json-view">{llm_messages_html}</div>
    </div>
    <div class="panel full-width">
      <h2>📝 自动生成的 Case 代码（可编辑）</h2>
      <textarea class="textarea" id="case-code">{case_code}</textarea>
    </div>
  </div>

  <div class="actions">
    <button class="btn btn-green" onclick="doCommit()">✅ 直接入库</button>
    <button class="btn btn-blue" onclick="doCommitEdit()">✏️ 修改后入库</button>
    <button class="btn btn-red" onclick="doDismiss()">🗑️ 丢弃</button>
  </div>
  <div class="note">入库后 case 将追加到对应的 benchmark 文件中，不可自动回滚</div>
</div>

<script>
async function doCommit() {
  const code = document.getElementById('case-code').value;
  const res = await fetch('/api/draft/{draft_id}/commit', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({case_code: code})
  });
  const data = await res.json();
  if (data.success) { alert('已入库: ' + data.benchmark_file); location.href='/'; }
  else { alert('失败: ' + (data.error || 'unknown')); }
}
async function doCommitEdit() {
  const code = document.getElementById('case-code').value;
  const res = await fetch('/api/draft/{draft_id}/commit', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({case_code: code, edited: true})
  });
  const data = await res.json();
  if (data.success) { alert('已修改入库: ' + data.benchmark_file); location.href='/'; }
  else { alert('失败: ' + (data.error || 'unknown')); }
}
async function doDismiss() {
  const reason = prompt('丢弃原因（可选）:');
  if (reason === null) return;
  const res = await fetch('/api/draft/{draft_id}/dismiss', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({reason})
  });
  const data = await res.json();
  if (data.success) { location.href='/'; }
  else { alert('失败: ' + (data.error || 'unknown')); }
}
</script>
</body>
</html>
"""

# ------------------------------------------------------------------
# API 路由
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return _LIST_PAGE_HTML


@app.get("/draft/{draft_id}", response_class=HTMLResponse)
def draft_detail_page(draft_id: str):
    draft = _load_draft_anywhere(draft_id)
    if not draft:
        return HTMLResponse("<h1>Draft not found</h1><a href='/'>返回</a>", status_code=404)

    draft_id = draft.get("draft_id") or draft.get("tick_id", "unknown")
    judge = draft.get("judge_result", {})
    severity = judge.get("severity", "P2")

    # Bot 回复：兼容多种结构
    bot_reply = draft.get("bot_reply", "")
    if not bot_reply:
        reply_data = draft.get("reply", {})
        if isinstance(reply_data, dict):
            replies = reply_data.get("replies", [])
            if replies:
                bot_reply = replies[0] if isinstance(replies, list) else str(replies)
        elif isinstance(reply_data, list) and reply_data:
            bot_reply = reply_data[0]

    # 对话 HTML：兼容 conversation / context.session_input_messages
    conversation_html = ""
    conversation = draft.get("conversation", [])
    if not conversation:
        ctx = draft.get("context", {})
        msgs = ctx.get("session_input_messages", [])
        conversation = []
        for m in msgs:
            role = m.get("role", "user")
            conversation.append({
                "role": role,
                "sender": m.get("name", "用户" if role == "user" else "Bot"),
                "text": m.get("content", ""),
            })
    for turn in conversation:
        role_cls = "bot" if turn.get("role") == "bot" else "user"
        sender = turn.get("sender", "")
        text = _escape_html(turn.get("text", ""))
        conversation_html += f'<div class="msg {role_cls}"><div class="sender">{_escape_html(sender)}</div><div class="text">{text}</div></div>'

    # Judge JSON
    judge_json = _escape_html(json.dumps(judge, ensure_ascii=False, indent=2))

    # 截图
    assets = draft.get("assets", {})
    screenshot_path = assets.get("screenshot_path", "")
    if not screenshot_path:
        screenshot_path = draft.get("screenshot_path", "")
    if screenshot_path and (PROJECT_ROOT / screenshot_path).exists():
        screenshot_html = f'<img class="screenshot" src="/{screenshot_path}" onclick="window.open(this.src)">'
    else:
        screenshot_html = '<div style="color:var(--muted);font-size:12px">截图不可用</div>'

    # 资产链接
    links = []
    for key, path in assets.items():
        if path and (PROJECT_ROOT / path).exists():
            links.append(f'<div><a href="/{path}" style="color:var(--blue)" target="_blank">{_escape_html(key)}</a></div>')
    # 顶层 screenshot_path 也作为资产展示
    sp = draft.get("screenshot_path", "")
    if sp and sp not in assets.values() and (PROJECT_ROOT / sp).exists():
        links.append(f'<div><a href="/{sp}" style="color:var(--blue)" target="_blank">screenshot</a></div>')
    assets_links = "\n".join(links) if links else '<div style="color:var(--muted)">无</div>'

    # Case 代码
    generated = draft.get("generated_case", {})
    case_code = generated.get("case_code", "")
    if not case_code:
        case_code = draft.get("case_code", "")

    # 用 replace 避免 CSS 变量 {--bg} 等被 format 误解析
    html = _DETAIL_PAGE_HTML
    html = html.replace("{draft_id}", draft_id)
    html = html.replace("{severity}", severity)
    html = html.replace("{conversation_html}", conversation_html)
    html = html.replace("{judge_json}", judge_json)
    html = html.replace("{bot_reply}", _escape_html(bot_reply))
    html = html.replace("{screenshot_html}", screenshot_html)
    html = html.replace("{assets_links}", assets_links)
    # Bot 实际看到的完整上下文
    system_prompt = draft.get("full_system_prompt", "")
    tools_context = draft.get("full_tools_context", "")
    llm_messages = draft.get("full_llm_messages", [])

    if system_prompt:
        system_prompt_html = _escape_html(system_prompt[:3000])
    else:
        system_prompt_html = '<div style="color:var(--muted);font-size:12px">未记录</div>'

    if tools_context:
        tools_context_html = _escape_html(tools_context[:3000])
    else:
        tools_context_html = '<div style="color:var(--muted);font-size:12px">未记录</div>'

    user_prompt = draft.get("full_user_prompt", "")
    if user_prompt:
        user_prompt_html = _escape_html(user_prompt[:5000])
    else:
        user_prompt_html = '<div style="color:var(--muted);font-size:12px">未记录</div>'

    if llm_messages:
        truncated = []
        for m in llm_messages[-10:]:
            cm = dict(m)
            if "content" in cm and isinstance(cm["content"], str) and len(cm["content"]) > 500:
                cm["content"] = cm["content"][:500] + "... [truncated]"
            if "tool_calls" in cm and cm["tool_calls"]:
                cm["tool_calls"] = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name")} for tc in cm["tool_calls"]]
            truncated.append(cm)
        llm_messages_html = _escape_html(json.dumps(truncated, ensure_ascii=False, indent=2))
    else:
        llm_messages_html = '<div style="color:var(--muted);font-size:12px">未记录</div>'

    html = html.replace("{system_prompt_html}", system_prompt_html)
    html = html.replace("{tools_context_html}", tools_context_html)
    html = html.replace("{user_prompt_html}", user_prompt_html)
    html = html.replace("{llm_messages_html}", llm_messages_html)
    html = html.replace("{case_code}", _escape_html(case_code))
    return HTMLResponse(html)


@app.get("/api/drafts")
def api_list_drafts(
    status: str = Query("all"),
    severity: Optional[str] = Query(None),
    module: Optional[str] = Query(None),
):
    drafts = []
    for d in (PENDING_DIR, COMMITTED_DIR, DISMISSED_DIR):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                judge = data.get("judge_result", {})
                gen = data.get("generated_case", {})
                item = {
                    "draft_id": data.get("draft_id", f.stem),
                    "status": data.get("status", "pending"),
                    "severity": judge.get("severity", "P2"),
                    "badcase_type": judge.get("badcase_type", "unknown"),
                    "module": gen.get("module", "P2"),
                    "confidence": judge.get("confidence", 0),
                    "reason": judge.get("reason", ""),
                    "bot_reply_preview": (data.get("bot_reply", "") or "")[:60],
                    "timestamp": data.get("timestamp", ""),
                }
                if status != "all" and item["status"] != status:
                    continue
                if severity and item["severity"] != severity:
                    continue
                if module and item["module"] != module:
                    continue
                drafts.append(item)
            except Exception:
                continue
    return drafts


@app.get("/api/draft/{draft_id}")
def api_get_draft(draft_id: str):
    draft = _load_draft_anywhere(draft_id)
    if not draft:
        return JSONResponse({"error": "not found"}, status_code=404)
    return draft


@app.post("/api/draft/{draft_id}/commit")
def api_commit_draft(draft_id: str, body: Dict[str, Any] = Body(default={})):
    draft = _load_draft_anywhere(draft_id)
    if not draft:
        return JSONResponse({"success": False, "error": "draft not found"})

    case_code = body.get("case_code", "")
    if not case_code:
        # 使用自动生成的
        case_code = draft.get("generated_case", {}).get("case_code", "")

    module = draft.get("generated_case", {}).get("module", "P2")
    benchmark_file = _get_benchmark_file(module)

    try:
        _append_case_to_benchmark(benchmark_file, case_code)
    except Exception as e:
        import logging
        logging.getLogger("review_server").warning(f"提交 benchmark 失败: {e}")
        return JSONResponse({"success": False, "error": "提交 benchmark 失败，请查看服务端日志"})

    # 移动 draft
    src = _find_draft_path(draft_id)
    if src:
        draft["status"] = "committed"
        draft["committed_at"] = datetime.now().isoformat()
        draft["committed_by"] = "manual"
        if "review_history" not in draft:
            draft["review_history"] = []
        draft["review_history"].append({
            "action": "commit",
            "time": datetime.now().isoformat(),
            "edited": body.get("edited", False),
        })
        dst = COMMITTED_DIR / f"{draft_id}.json"
        dst.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        src.unlink()

    return JSONResponse({
        "success": True,
        "benchmark_file": str(benchmark_file.relative_to(PROJECT_ROOT)),
    })


@app.post("/api/draft/{draft_id}/dismiss")
def api_dismiss_draft(draft_id: str, body: Dict[str, Any] = Body(default={})):
    draft = _load_draft_anywhere(draft_id)
    if not draft:
        return JSONResponse({"success": False, "error": "draft not found"})

    src = _find_draft_path(draft_id)
    if src:
        draft["status"] = "dismissed"
        draft["dismissed_at"] = datetime.now().isoformat()
        draft["dismiss_reason"] = body.get("reason", "")
        if "review_history" not in draft:
            draft["review_history"] = []
        draft["review_history"].append({
            "action": "dismiss",
            "time": datetime.now().isoformat(),
            "reason": body.get("reason", ""),
        })
        dst = DISMISSED_DIR / f"{draft_id}.json"
        dst.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        src.unlink()

    return JSONResponse({"success": True})


# 静态文件服务（截图、prompt.md 等）
@app.get("/{path:path}")
def serve_static(path: str):
    file_path = _safe_project_path(path)
    # lgtm[py/path-injection] file_path 来自 _safe_project_path，已验证在项目根目录内
    if file_path and file_path.exists() and file_path.is_file():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    return RedirectResponse("/")


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _load_draft_anywhere(draft_id: str) -> Optional[Dict]:
    safe_id = _safe_filename(draft_id)
    if not safe_id:
        return None
    for d in (PENDING_DIR, COMMITTED_DIR, DISMISSED_DIR):
        p = d / f"{safe_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _find_draft_path(draft_id: str) -> Optional[Path]:
    safe_id = _safe_filename(draft_id)
    if not safe_id:
        return None
    for d in (PENDING_DIR, COMMITTED_DIR, DISMISSED_DIR):
        p = d / f"{safe_id}.json"
        if p.exists():
            return p
    return None


def _get_benchmark_file(module: str) -> Path:
    mapping = {
        "P0": PROJECT_ROOT / "src" / "tests" / "test_tool_decision_benchmark.py",
        "P2": PROJECT_ROOT / "src" / "tests" / "test_reply_quality_benchmark.py",
        "P3": PROJECT_ROOT / "src" / "tests" / "test_reply_quality_benchmark.py",
    }
    return mapping.get(module, mapping["P2"])


def _append_case_to_benchmark(benchmark_file: Path, case_code: str):
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
    content = benchmark_file.read_text(encoding="utf-8")
    marker = "# -------------------------------------------------------------------------\n# Auto-generated cases\n# -------------------------------------------------------------------------"
    if marker not in content:
        content = content.rstrip() + f"\n\n{marker}\n{case_code}\n"
    else:
        parts = content.split(marker, 1)
        content = parts[0] + marker + "\n" + case_code + "\n" + parts[1]
    benchmark_file.write_text(content, encoding="utf-8")


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
