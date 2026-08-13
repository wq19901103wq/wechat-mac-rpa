"""
ACP 代码审计 Demo v2 - 使用 kimi --print 直接调用，更稳定
端口: 8767
"""
import asyncio
import json
import os

import sys
import logging
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

_logger = logging.getLogger(__name__)


WORK_DIR = str(Path(__file__).resolve().parents[3])
KIMI_BIN = os.environ.get("KIMI_BIN", "kimi")


# ============ Analyze with kimi --print ============

async def analyze_with_kimi(issue: dict, notes: str, timeout: int = 300) -> dict:
    """使用 kimi --print 直接调用，非交互式"""
    
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

    cmd = [
        KIMI_BIN,
        "--quiet",
        "--yolo",
        "-p", prompt,
        "-w", WORK_DIR,
    ]
    
    print(f"[Analyze] Running: {' '.join(cmd[:6])}...")
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WORK_DIR,
        )
        
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        
        reply = stdout.decode('utf-8', errors='replace')
        stderr_text = stderr.decode('utf-8', errors='replace')
        
        print(f"[Analyze] Done, stdout={len(reply)} chars, stderr={len(stderr_text)} chars, returncode={proc.returncode}")
        
        # 过滤掉stderr中的日志行（只保留 kimi 的回复）
        if stderr_text:
            print(f"[Analyze] stderr preview: {stderr_text[:500]}")
        
        return {
            "success": True,
            "reply": reply,
            "stderr": stderr_text,
            "returncode": proc.returncode,
        }
        
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception as e:
            _logger.warning("proc.kill failed: %s", e)
        return {"success": False, "error": f"分析超时（>{timeout}秒）"}
    except Exception as e:
        print(f"[Analyze] Error: {e}")
        return {"success": False, "error": "分析异常，请查看服务端日志"}


# ============ FastAPI App ============

app = FastAPI(title="ACP Audit Demo v2")

AUDIT_ISSUES = [
    {
        "key": "api_timestamp_missing",
        "severity": "P0",
        "title": "API 时间戳系统性缺失",
        "file": "src/perception/smart_pipeline.py",
        "lines": "893-904",
        "problem": "System Prompt 未要求 API 返回时间戳，解析固定用 %Y-%m-%d %H:%M:%S，与截图中'昨天 21:58'格式不匹配",
        "impact": "所有 API 路径消息 create_time 100% fallback 到当前时间",
        "notes": "",
        "ai_proposal": "",
    },
    {
        "key": "layout_timestamp_bug",
        "severity": "P0",
        "title": "布局引擎时间戳 100% fallback",
        "file": "src/perception/layout_analyzer.py",
        "lines": "待确认",
        "problem": "布局分析提取时间戳时未处理多种格式",
        "impact": "布局路径消息时间戳全部 fallback",
        "notes": "",
        "ai_proposal": "",
    },
    {
        "key": "judge_weight_mismatch",
        "severity": "P0",
        "title": "消息判断与过滤逻辑权重不匹配",
        "file": "src/bot/message_router.py",
        "lines": "待确认",
        "problem": "judge 逻辑权重与实际过滤条件不一致",
        "impact": "误判消息类型，导致路由错误",
        "notes": "",
        "ai_proposal": "",
    },
    {
        "key": "weflow_mode_check",
        "severity": "P1",
        "title": "weflow_mode 检查应在解析后",
        "file": "src/bot/message_router.py",
        "lines": "待确认",
        "problem": "weflow_mode 检查位置过早",
        "impact": "某些消息被错误跳过",
        "notes": "",
        "ai_proposal": "",
    },
    {
        "key": "timestamp_extract_inconsistent",
        "severity": "P1",
        "title": "时间戳提取与解析逻辑不一致",
        "file": "src/perception/smart_pipeline.py",
        "lines": "待确认",
        "problem": "多处时间戳提取逻辑不统一",
        "impact": "维护困难，容易出错",
        "notes": "",
        "ai_proposal": "",
    },
    {
        "key": "bot_self_msg_no_create_time",
        "severity": "P1",
        "title": "机器人消息无 create_time",
        "file": "src/bot/message_router.py",
        "lines": "待确认",
        "problem": "机器人自己发的消息没有设置 create_time",
        "impact": "消息时间戳为空",
        "notes": "",
        "ai_proposal": "",
    },
    {
        "key": "already_handled_mislabel",
        "severity": "P2",
        "title": "已处理消息误判为 error",
        "file": "src/reply/generator.py",
        "lines": "965-974",
        "problem": "already_handled 消息被错误标记为 error",
        "impact": "日志混乱，统计不准",
        "notes": "",
        "ai_proposal": "",
    },
]


@app.get("/", response_class=HTMLResponse)
async def index():
    cards_html = ""
    for issue in AUDIT_ISSUES:
        severity_color = {"P0": "#f85149", "P1": "#d29922", "P2": "#58a6ff"}[issue["severity"]]
        cards_html += f'''
        <div class="card" id="card-{issue['key']}">
            <div class="card-header">
                <span class="severity" style="background:{severity_color}">{issue["severity"]}</span>
                <span class="title">{issue["title"]}</span>
            </div>
            <div class="card-body">
                <p><strong>文件:</strong> {issue["file"]}:{issue["lines"]}</p>
                <p><strong>问题:</strong> {issue["problem"]}</p>
                <p><strong>影响:</strong> {issue["impact"]}</p>
                <div class="input-group">
                    <label>点评 / 给 AI 的要求:</label>
                    <textarea id="notes-{issue['key']}" rows="3" placeholder="写出你的分析要求..."></textarea>
                </div>
                <div class="actions">
                    <button class="btn-analyze" onclick="analyzeIssue('{issue['key']}')">
                        🤖 请求 AI 分析
                    </button>
                    <span class="status" id="status-{issue['key']}"></span>
                </div>
                <div class="result" id="result-{issue['key']}" style="display:none">
                    <div class="result-header">
                        <span>AI 分析结果</span>
                        <button class="btn-copy" onclick="copyResult('{issue['key']}')">复制</button>
                    </div>
                    <div class="result-content" id="result-content-{issue['key']}"></div>
                </div>
            </div>
        </div>
        '''

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ACP 代码审计 Demo</title>
<style>
    :root {{
        --bg: #0d1117;
        --card-bg: #161b22;
        --border: #30363d;
        --text: #c9d1d9;
        --text-secondary: #8b949e;
        --green: #3fb950;
        --blue: #58a6ff;
        --red: #f85149;
        --yellow: #d29922;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--bg);
        color: var(--text);
        margin: 0;
        padding: 20px;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 30px;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
    }}
    .card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 20px;
        overflow: hidden;
    }}
    .card-header {{
        padding: 16px 20px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .severity {{
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }}
    .title {{
        font-weight: 600;
        font-size: 16px;
    }}
    .card-body {{
        padding: 16px 20px;
    }}
    .card-body p {{
        margin: 8px 0;
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.5;
    }}
    .input-group {{
        margin-top: 16px;
    }}
    .input-group label {{
        display: block;
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }}
    textarea {{
        width: 100%;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        padding: 10px;
        font-size: 14px;
        resize: vertical;
        box-sizing: border-box;
    }}
    .actions {{
        margin-top: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .btn-analyze {{
        background: var(--blue);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
    }}
    .btn-analyze:hover {{ opacity: 0.9; }}
    .btn-analyze:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
    }}
    .status {{
        font-size: 13px;
        color: var(--text-secondary);
    }}
    .result {{
        margin-top: 16px;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }}
    .result-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        border-bottom: 1px solid var(--border);
        font-size: 13px;
        color: var(--text-secondary);
    }}
    .btn-copy {{
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        padding: 2px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
    }}
    .result-content {{
        padding: 16px;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }}
    .result-content code {{
        background: #21262d;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'SF Mono', monospace;
        font-size: 13px;
    }}
    .result-content pre {{
        background: #0d1117;
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
        border: 1px solid var(--border);
    }}
    .result-content pre code {{
        background: transparent;
        padding: 0;
    }}
    .loading {{
        color: var(--yellow);
    }}
    .success {{
        color: var(--green);
    }}
    .error {{
        color: var(--red);
    }}
</style>
</head>
<body>
<div class="container">
    <h1>🤖 ACP 代码审计 Demo</h1>
    {cards_html}
</div>

<script>
const issues = {json.dumps(AUDIT_ISSUES, ensure_ascii=False)};

function getIssue(key) {{
    return issues.find(i => i.key === key);
}}

async function analyzeIssue(key) {{
    const issue = getIssue(key);
    const notes = document.getElementById('notes-' + key).value;
    const btn = document.querySelector('#card-' + key + ' .btn-analyze');
    const status = document.getElementById('status-' + key);
    const result = document.getElementById('result-' + key);
    const content = document.getElementById('result-content-' + key);
    
    btn.disabled = true;
    status.textContent = '⏳ 分析中（约 1-3 分钟）...';
    status.className = 'status loading';
    result.style.display = 'none';
    
    try {{
        const resp = await fetch('/api/analyze', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{issue: issue, notes: notes}})
        }});
        const data = await resp.json();
        
        if (data.success) {{
            status.textContent = '✓ 分析完成 (' + data.reply.length + ' 字符)';
            status.className = 'status success';
            content.textContent = data.reply;
            result.style.display = 'block';
        }} else {{
            status.textContent = '✗ ' + (data.error || '失败');
            status.className = 'status error';
        }}
    }} catch (err) {{
        status.textContent = '✗ 网络错误: ' + err.message;
        status.className = 'status error';
    }} finally {{
        btn.disabled = false;
    }}
}}

function copyResult(key) {{
    const text = document.getElementById('result-content-' + key).textContent;
    navigator.clipboard.writeText(text).then(() => {{
        alert('已复制到剪贴板');
    }});
}}
</script>
</body>
</html>
"""


@app.post("/api/analyze")
async def api_analyze(request: Request):
    body = await request.json()
    issue = body.get("issue", {})
    notes = body.get("notes", "")
    result = await analyze_with_kimi(issue, notes, timeout=300)
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8767)
