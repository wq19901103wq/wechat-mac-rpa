"""
ACP 代码审计 Demo
独立运行: python scripts/acp_audit_demo.py
端口: 8767
"""
import asyncio
import json
import sys

from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, '/Users/yourname/.local/share/uv/tools/kimi-cli/lib/python3.13/site-packages')

import acp
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import uvicorn


WORK_DIR = "/Users/yourname/wechat-mac-rpa"
KIMI_PYTHON = "/Users/yourname/.local/share/uv/tools/kimi-cli/bin/python"


# ============ ACP Client ============

class AuditAcpClient(acp.Client):
    """收集 ACP agent 的回复、思考、工具调用"""

    def __init__(self):
        self.messages: list[str] = []
        self.thoughts: list[str] = []
        self.tool_calls: list[dict] = []
        self.done_event = asyncio.Event()

    def reset(self):
        self.messages.clear()
        self.thoughts.clear()
        self.tool_calls.clear()
        self.done_event.clear()

    def on_connect(self, conn):
        pass

    def session_update(self, session_id, update, **kwargs):
        if isinstance(update, acp.schema.AgentMessageChunk):
            text = update.content.text if update.content else ""
            self.messages.append(text)
        elif isinstance(update, acp.schema.AgentThoughtChunk):
            text = update.content.text if update.content else ""
            self.thoughts.append(text)
        elif isinstance(update, acp.schema.ToolCallStart):
            self.tool_calls.append({
                "title": update.title,
                "status": update.status,
                "tool_call_id": update.tool_call_id,
            })
        elif isinstance(update, acp.schema.ToolCallProgress):
            # 更新已有 tool call 状态
            for tc in self.tool_calls:
                if tc.get("tool_call_id") == update.tool_call_id:
                    tc["status"] = update.status
                    break
        elif hasattr(update, 'session_update') and update.session_update == "usage_update":
            self.done_event.set()

    def request_permission(self, options, session_id, tool_call, **kwargs):
        return acp.schema.RequestPermissionResponse(approved=True)

    def get_full_reply(self) -> str:
        # Kimi Code 在工具调用模式下，分析结果可能在 thoughts 里
        # 合并 messages 和 thoughts，优先用 messages（正式回复）
        reply = "".join(self.messages)
        thoughts = "".join(self.thoughts)
        if reply and thoughts:
            return f"【分析过程】\n{thoughts}\n\n【最终结论】\n{reply}"
        if reply:
            return reply
        return thoughts

    def get_thought_text(self) -> str:
        return "".join(self.thoughts)


# ============ ACP Manager ============

class AcpManager:
    """管理 kimi acp 子进程和 session"""

    def __init__(self):
        self.client = AuditAcpClient()
        self.conn: Optional[acp.client.ClientSideConnection] = None
        self.process = None
        self.session_id: Optional[str] = None
        self._connected = False

    async def connect(self):
        """启动 kimi acp 并建立连接"""
        if self._connected:
            return

        print("[ACP] Starting kimi acp process...")
        ctx = acp.spawn_agent_process(
            self.client,
            "kimi", "acp",
            cwd=WORK_DIR,
        )
        self._ctx = ctx
        self._gen = ctx.__aenter__()
        self.conn, self.process = await self._gen
        self._connected = True
        print(f"[ACP] Process started, pid={self.process.pid}")

        # Initialize
        init_resp = await self.conn.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_info=acp.schema.Implementation(name="audit-demo", version="0.1"),
        )
        print(f"[ACP] Initialized, protocol={init_resp.protocol_version}")

    async def new_session(self):
        """创建新 session"""
        await self.connect()
        resp = await self.conn.new_session(cwd=WORK_DIR)
        self.session_id = resp.session_id
        print(f"[ACP] Session created: {self.session_id}")
        return self.session_id

    async def analyze(self, issue: dict, notes: str, timeout: int = 120) -> dict:
        """
        分析单个 issue。
        返回: {"reply": str, "thoughts": str, "tool_calls": list, "success": bool, "error": str}
        """
        await self.connect()
        if not self.session_id:
            await self.new_session()

        self.client.reset()

        # 构造 prompt
        prompt = self._build_prompt(issue, notes)
        print(f"[ACP] Sending prompt ({len(prompt)} chars)...")

        try:
            resp = await asyncio.wait_for(
                self.conn.prompt(
                    [acp.text_block(prompt)],
                    session_id=self.session_id,
                ),
                timeout=timeout,
            )
            print(f"[ACP] PromptResponse: stop_reason={resp.stop_reason}")

            # prompt() 返回时所有 session_update 应该已处理完毕
            # 但额外给事件循环一个机会处理任何 pending 的回调
            await asyncio.sleep(0.5)

            return {
                "success": True,
                "reply": self.client.get_full_reply(),
                "thoughts": self.client.get_thought_text(),
                "tool_calls": self.client.tool_calls,
                "stop_reason": resp.stop_reason,
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "分析超时", "reply": self.client.get_full_reply()}
        except Exception as e:
            import traceback
            print(f"[ACP] Error: {e}")
            traceback.print_exc()
            return {"success": False, "error": "分析异常，请查看服务端日志", "reply": self.client.get_full_reply()}

    def _build_prompt(self, issue: dict, notes: str) -> str:
        return f"""你是一个代码审计专家。请分析以下代码问题并给出修复方案。

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

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
        if hasattr(self, '_ctx'):
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
        self._connected = False
        print("[ACP] Closed")


# ============ FastAPI App ============

app = FastAPI(title="ACP Audit Demo")
acp_manager = AcpManager()

# 7 条发现（简化版）
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
                    <div class="result-tabs">
                        <button class="tab-btn active" onclick="showTab('{issue['key']}', 'reply')">AI 方案</button>
                        <button class="tab-btn" onclick="showTab('{issue['key']}', 'thoughts')">思考过程</button>
                        <button class="tab-btn" onclick="showTab('{issue['key']}', 'tools')">工具调用</button>
                    </div>
                    <div class="tab-content" id="tab-reply-{issue['key']}"></div>
                    <div class="tab-content" id="tab-thoughts-{issue['key']}" style="display:none"></div>
                    <div class="tab-content" id="tab-tools-{issue['key']}" style="display:none"></div>
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
        padding: 16px;
    }}
    .result-tabs {{
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
    }}
    .tab-btn {{
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        padding: 4px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 13px;
    }}
    .tab-btn.active {{
        background: var(--blue);
        color: white;
        border-color: var(--blue);
    }}
    .tab-content {{
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
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
    .tool-item {{
        padding: 8px;
        background: var(--card-bg);
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 13px;
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
    
    btn.disabled = true;
    status.textContent = '分析中...';
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
            status.textContent = '✓ 分析完成';
            status.className = 'status success';
            
            document.getElementById('tab-reply-' + key).innerHTML = markdownToHtml(data.reply);
            document.getElementById('tab-thoughts-' + key).textContent = data.thoughts || '(无思考过程)';
            
            const toolsEl = document.getElementById('tab-tools-' + key);
            if (data.tool_calls && data.tool_calls.length > 0) {{
                toolsEl.innerHTML = data.tool_calls.map(tc => 
                    `<div class="tool-item">🛠️ ${{tc.title}} <span style="color:var(--green)">${{tc.status}}</span></div>`
                ).join('');
            }} else {{
                toolsEl.textContent = '(无工具调用)';
            }}
            
            result.style.display = 'block';
            showTab(key, 'reply');
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

function showTab(key, tab) {{
    const card = document.getElementById('card-' + key);
    card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    card.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    
    event.target.classList.add('active');
    document.getElementById('tab-' + tab + '-' + key).style.display = 'block';
}}

function markdownToHtml(text) {{
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code style="background:#30363d;padding:2px 4px;border-radius:3px;">$1</code>')
        .replace(/```([\s\S]*?)```/g, '<pre style="background:#0d1117;padding:12px;border-radius:6px;overflow-x:auto;"><code>$1</code></pre>')
        .replace(/^- (.*$)/gim, '<li>$1</li>')
        .replace(/\\n/g, '<br>');
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

    print(f"[API] Analyze request: {issue.get('key')}")
    result = await acp_manager.analyze(issue, notes, timeout=120)
    print(f"[API] Analyze done: success={result['success']}, reply_len={len(result.get('reply', ''))}")
    return JSONResponse(result)


@app.on_event("shutdown")
async def shutdown():
    await acp_manager.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8767)
