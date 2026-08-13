#!/usr/bin/env python3
"""生成 Judge + Bot 回复质量 Benchmark HTML Dashboard。Judge 数据以 DB 实时查询为主。"""

import json, sys, os, sqlite3
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cases.db"
OUTPUT_JUDGE = PROJECT_ROOT / "benchmark_judge.html"
OUTPUT_REPLY = PROJECT_ROOT / "benchmark_reply.html"

DIM_NAMES = ["幻觉控制", "时间推理", "回复必要性", "信息准确性", "上下文理解"]
DIM_COLORS = ["#f85149", "#f778ba", "#ff7849", "#58a6ff", "#3fb950"]

CSS = """<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff}
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);padding:24px;max-width:1400px;margin:0 auto}
h1{font-size:22px;margin-bottom:4px}h2{font-size:16px;margin:24px 0 12px}
.muted{color:var(--muted);font-size:12px;margin-bottom:20px}
.metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;margin-bottom:20px}
.metric{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;text-align:center}
.metric .val{font-size:28px;font-weight:700}.metric .lbl{font-size:11px;color:var(--muted);margin-top:2px}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:8px}
.card h3{font-size:14px;margin-bottom:8px}.card .meta{font-size:11px;color:var(--muted);margin-bottom:6px}
.bar-bg{background:rgba(255,255,255,.05);border-radius:4px;height:20px;margin:2px 0;overflow:hidden;position:relative}
.bar-fg{height:100%;border-radius:4px}.bar-label{position:absolute;left:8px;top:2px;font-size:11px;color:var(--text);z-index:1}
table{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:10px}
.judge-match{border-left:3px solid var(--green)}.judge-mismatch{border-left:3px solid var(--red)}
.tag{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;margin:1px}
.tag-bad{background:rgba(248,81,73,.2);color:var(--red)}.tag-ok{background:rgba(63,185,80,.2);color:var(--green)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.filters a{color:var(--text);text-decoration:none;font-size:12px;padding:4px 12px;border:1px solid var(--border);border-radius:4px;cursor:pointer}
.filters a:hover,.filters a.sel{background:rgba(88,166,255,.15);border-color:var(--blue);color:var(--blue)}
</style>"""


# =============================================================================
# Judge Quality（实时 DB 查询，人类标注变更立即生效）
# =============================================================================

def build_judge_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # 所有人工标注的 case
    labeled = conn.execute("SELECT * FROM tick_log WHERE human_is_badcase IS NOT NULL ORDER BY id").fetchall()

    tp = conn.execute("SELECT COUNT(*) FROM tick_log WHERE human_is_badcase=1 AND judge_is_badcase=1").fetchone()[0]
    fp = conn.execute("SELECT COUNT(*) FROM tick_log WHERE human_is_badcase=0 AND judge_is_badcase=1").fetchone()[0]
    fn = conn.execute("SELECT COUNT(*) FROM tick_log WHERE human_is_badcase=1 AND judge_is_badcase=0").fetchone()[0]
    tn = conn.execute("SELECT COUNT(*) FROM tick_log WHERE human_is_badcase=0 AND judge_is_badcase=0").fetchone()[0]
    hb = tp + fn
    total_labeled = tp + fp + fn + tn
    acc = (tp + tn) / total_labeled if total_labeled > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / hb if hb > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # 维度平均分对比
    dim_scores_ok = {d: [] for d in DIM_NAMES}
    dim_scores_bad = {d: [] for d in DIM_NAMES}
    for r in labeled:
        d = dict(r)
        dims_json = d.get("judge_dimensions_json", "{}") or "{}"
        try: dims = json.loads(dims_json)
        except Exception as e:
            _logger.warning("load judge dimensions failed: %s", e)
            dims = {}
        target = dim_scores_bad if d["human_is_badcase"] else dim_scores_ok
        for dim_name in DIM_NAMES:
            s = dims.get(dim_name, {}).get("score", 0)
            if s: target[dim_name].append(s)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>Judge Benchmark</title>{CSS}</head><body>
<h1>📊 Judge Quality（实时数据）</h1>
<div class="muted">{total_labeled} 个人工标注 · 标注后刷新立即生效 · {now}</div>

<div class="metrics">
  <div class="metric"><div class="val" style="color:{'var(--green)' if acc > 0.5 else 'var(--red)'}">{acc:.0%}</div><div class="lbl">Accuracy</div></div>
  <div class="metric"><div class="val">{prec:.0%}</div><div class="lbl">Precision</div></div>
  <div class="metric"><div class="val">{rec:.0%}</div><div class="lbl">Recall</div></div>
  <div class="metric"><div class="val">{f1:.3f}</div><div class="lbl">F1</div></div>
</div>
<div style="font-size:13px;color:var(--muted);margin-bottom:20px">
  TP={tp} FP={fp} TN={tn} FN={fn} · Judge正确={tp+tn}/{total_labeled}
</div>

<h2>📐 各维度 Judge 评分对比</h2>
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;margin-bottom:20px">"""
    for i, dim_name in enumerate(DIM_NAMES):
        ok_avg = round(sum(dim_scores_ok[dim_name]) / len(dim_scores_ok[dim_name]), 1) if dim_scores_ok[dim_name] else 0
        bad_avg = round(sum(dim_scores_bad[dim_name]) / len(dim_scores_bad[dim_name]), 1) if dim_scores_bad[dim_name] else 0
        html += f"""<div class="card">
  <div style="font-size:13px;margin-bottom:4px">{dim_name}</div>
  <div style="font-size:11px;color:var(--green);margin:2px 0">✅ OK case 均分: {ok_avg}</div>
  <div style="font-size:11px;color:var(--red);margin:2px 0">❌ BAD case 均分: {bad_avg}</div>
  <div class="bar-bg" style="margin-top:4px"><div class="bar-fg" style="width:{bad_avg/5*100}%;background:var(--red)"></div><span class="bar-label" style="font-size:10px">bad: {bad_avg}/5</span></div>
</div>"""
    html += '</div>'

    # 逐 case
    html += '<h2>📋 逐 Case 对比</h2><div class="filters"><a class="sel" onclick="f(event,\'all\')">全部</a> <a onclick="f(event,\'match\')">Judge正确</a> <a onclick="f(event,\'mismatch\')">Judge错误</a></div>'
    html += '<div class="grid2" id="grid">'

    for r in labeled:
        d = dict(r)
        match = d["judge_is_badcase"] == d["human_is_badcase"]
        icon = "✅" if match else "❌"
        cls = "judge-match" if match else "judge-mismatch"
        filter_class = "match" if match else "mismatch"
        j_bc = "BAD" if d["judge_is_badcase"] else "OK"
        h_bc = "BAD" if d["human_is_badcase"] else "OK"
        j_score = d.get("judge_score", 0) or 0
        j_type = d.get("judge_badcase_type", "") or "none"
        h_type = d.get("human_badcase_type", "") or "none"
        sid = d.get("session_id", "")[:12] or ""

        # 维度 bars
        dims_json = d.get("judge_dimensions_json", "{}") or "{}"
        try: dims = json.loads(dims_json)
        except Exception as e:
            _logger.warning("load judge dimensions failed: %s", e)
            dims = {}
        dim_bars = ""
        for i, dim_name in enumerate(DIM_NAMES):
            s = dims.get(dim_name, {}).get("score", 0)
            dim_bars += f'<div class="bar-bg"><div class="bar-fg" style="width:{s/5*100}%;background:{DIM_COLORS[i]}"></div><span class="bar-label" style="font-size:9px">{dim_name}: {s}</span></div>'

        replies = d.get("replies_sent_json", "[]") or "[]"
        try: rlist = json.loads(replies); reply = " | ".join(rlist) if isinstance(rlist, list) else replies
        except Exception as e:
            _logger.warning("load replies failed: %s", e)
            reply = replies

        html += f"""<div class="card {cls} c {filter_class}">
  <h3>{icon} <a href="/ticks/{d['id']}" style="color:var(--blue)">{sid}:#{d['tick_id']}</a> <span style="font-size:10px;color:var(--muted)">{d.get('chat_name','')}</span></h3>
  <div class="meta">
    <span class="tag tag-bad">Human: {h_bc} {h_type}</span>
    <span class="tag tag-bad">Judge: {j_bc} {j_type}</span>
    Score: {j_score:.0f}
  </div>
  <div style="font-size:12px;margin:4px 0">回复: {reply[:100]}</div>
  <div class="meta">{d.get('human_notes','')[:120]}</div>
  <div class="meta" style="color:var(--yellow)">{(d.get('judge_reason') or '')[:120]}</div>
  {dim_bars}
</div>"""

    html += '</div>'
    html += f"""<div class="muted" style="margin-top:24px">Generated at {now}</div>
<script>
function f(e,t){{document.querySelectorAll('.filters a').forEach(a=>a.classList.remove('sel'));e.target.classList.add('sel');document.querySelectorAll('.c').forEach(c=>c.style.display=t==='all'||c.classList.contains(t)?'':'none')}}
</script></body></html>"""
    conn.close()
    return html


# =============================================================================
# Bot Reply Quality（所有已回复 tick + 筛选）
# =============================================================================

def load_all_reply_cases() -> list:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT *, (human_is_badcase IS NOT NULL) as labeled FROM tick_log WHERE should_reply=1 ORDER BY created_at DESC, tick_id DESC"
    ).fetchall()
    conn.close()
    cases = []
    for r in rows:
        d = dict(r)
        replies = d.get("replies_sent_json", "[]") or "[]"
        try: rlist = json.loads(replies); bot_reply = " | ".join(rlist) if isinstance(rlist, list) else replies
        except Exception as e:
            _logger.warning("load bot replies failed: %s", e)
            bot_reply = replies
        tc = d.get("tool_calls_json", "[]") or "[]"
        try: tool_calls = json.loads(tc)
        except Exception as e:
            _logger.warning("load tool calls failed: %s", e)
            tool_calls = []
        cases.append({
            "id": d["id"], "session_id": d.get("session_id", ""), "tick_id": d.get("tick_id", 0),
            "chat_name": d.get("chat_name", ""), "bot_reply": bot_reply,
            "human_is_badcase": d.get("human_is_badcase"), "human_badcase_type": d.get("human_badcase_type", ""),
            "human_notes": d.get("human_notes", ""), "judge_is_badcase": bool(d.get("judge_is_badcase", 0)),
            "judge_score": d.get("judge_score", 0) or 0, "judge_reason": d.get("judge_reason", "") or "",
            "judge_badcase_type": d.get("judge_badcase_type", "") or "",
            "labeled": bool(d["labeled"]), "tool_calls": tool_calls, "created_at": d.get("created_at", ""),
        })
    return cases


def build_reply_html() -> str:
    all_cases = load_all_reply_cases()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not all_cases: return "<h1>No reply data yet.</h1>"

    total = len(all_cases)
    labeled = [c for c in all_cases if c["labeled"]]
    labeled_bad = [c for c in labeled if c["human_is_badcase"]]
    judge_bad = [c for c in all_cases if c["judge_is_badcase"]]
    unlabeled = total - len(labeled)

    type_counts = {}
    for c in all_cases:
        t = (c.get("human_badcase_type") or c.get("judge_badcase_type") or "").strip()
        if t and t != "none": type_counts[t] = type_counts.get(t, 0) + 1

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>Bot Reply</title>{CSS}</head><body>
<h1>🤖 Bot 回复质量</h1>
<div class="muted">{total} 个已回复 tick · 人工标注 {len(labeled)} 个（{len(labeled_bad)} badcase）· Judge 判定 {len(judge_bad)} 个 badcase · {unlabeled} 个待标注 · {now}</div>
<div class="metrics">
  <div class="metric"><div class="val" style="color:{'var(--red)' if len(labeled_bad) > 0 else 'var(--green)'}">{len(labeled_bad)}/{len(labeled)}</div><div class="lbl">人工 Badcase 率</div></div>
  <div class="metric"><div class="val">{len(judge_bad)}/{total}</div><div class="lbl">Judge Badcase 率</div></div>
  <div class="metric"><div class="val">{total}</div><div class="lbl">总回复数</div></div>
</div>
<div class="filters">
  <a class="sel" onclick="filterCards('all',this)">全部({total})</a>
  <a onclick="filterCards('labeled',this)">已标注({len(labeled)})</a>
  <a onclick="filterCards('bad',this)">人工badcase({len(labeled_bad)})</a>
  <a onclick="filterCards('judge_bad',this)">Judge badcase({len(judge_bad)})</a>
  <a onclick="filterCards('unlabeled',this)">待标注({unlabeled})</a>
</div>"""
    # Type distribution
    if type_counts:
        html += '<h2>📊 Badcase 类型分布</h2><div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px"><div class="card" style="flex:1;min-width:280px">'
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = n / max(len(labeled_bad) + len(judge_bad), 1) * 100
            html += f'<div class="bar-bg"><div class="bar-fg" style="width:{min(pct,100)}%;background:var(--red)"></div><span class="bar-label">{t}: {n}</span></div>'
        html += '</div></div>'

    html += '<h2>📋 逐 Case</h2><div class="grid2" id="case-grid">'
    for c in all_cases:
        sid = c["session_id"][:12] if c["session_id"] else ""
        name = f"{sid}:#{c['tick_id']}" if sid else f"#{c['tick_id']}"
        if c["labeled"] and c["human_is_badcase"]:
            icon, cls = "❌", "judge-mismatch"
            tag = f'<span class="tag tag-bad">人工: {c.get("human_badcase_type","badcase")}</span>'
        elif c["labeled"] and not c["human_is_badcase"]:
            icon, cls = "✅", "judge-match"; tag = '<span class="tag tag-ok">人工: OK</span>'
        elif c["judge_is_badcase"]:
            icon, cls = "🔍", "judge-mismatch"
            tag = f'<span class="tag tag-bad">Judge: {c.get("judge_badcase_type","badcase")}</span>'
        else:
            icon, cls = "—", "judge-match"; tag = '<span class="tag tag-ok">未标注</span>'
        labels = "labeled" if c["labeled"] else "unlabeled"
        if c["labeled"] and c["human_is_badcase"]: labels += " bad"
        if c["judge_is_badcase"]: labels += " judge_bad"
        if c["labeled"] and not c["human_is_badcase"]: labels += " ok"

        html += f'<div class="card {cls} case-item" data-labels="{labels}">'
        html += f'<h3>{icon} {name} <span style="font-size:10px;color:var(--muted)">{c.get("created_at","")[:16]}</span></h3>'
        html += f'<div class="meta">{c["chat_name"]} | {tag} | Judge={c["judge_score"]:.0f} {"BAD" if c["judge_is_badcase"] else "OK"}</div>'
        html += f'<div style="font-size:12px;margin:4px 0">回复: {c["bot_reply"][:120]}</div>'
        if c.get("human_notes"): html += f'<div class="meta">备注: {c["human_notes"][:120]}</div>'
        if c.get("judge_reason"): html += f'<div class="meta" style="color:var(--yellow)">Judge: {c["judge_reason"][:100]}</div>'
        tools = ", ".join(t.get("tool_name", "?") for t in c["tool_calls"][:2])
        if tools: html += f'<div class="meta">🔧 {tools}</div>'
        html += '</div>'
    html += '</div>'

    html += f"""<div class="muted" style="margin-top:24px">Generated at {now}</div>
<script>
function filterCards(type,el){{document.querySelectorAll('.filters a').forEach(a=>a.classList.remove('sel'));el.classList.add('sel');document.querySelectorAll('.case-item').forEach(card=>{{var l=card.getAttribute('data-labels');if(type==='all')card.style.display='';else if(type==='labeled')card.style.display=l.includes('labeled')?'':'none';else if(type==='unlabeled')card.style.display=!l.includes('labeled')?'':'none';else card.style.display=l.includes(type)?'':'none'}})}}
</script></body></html>"""
    return html


def main():
    for html, path, label in [
        (build_judge_html(), OUTPUT_JUDGE, "Judge Quality"),
        (build_reply_html(), OUTPUT_REPLY, "Bot Reply Quality"),
    ]:
        path.write_text(html, encoding="utf-8")
        print(f"{label}: {path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
