"""Local-only Human Gold review UI for reply quality."""

from __future__ import annotations

import html
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = Path(
    os.environ.get(
        "REPLY_QUALITY_GOLD_PACKET",
        PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
)
GRADES = {"good", "acceptable", "bad", "critical"}
ACTIONS = {"reply", "silence"}
_WRITE_LOCK = threading.Lock()


def _load_packet(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_packet(path: Path, packet: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def save_human_review(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action")
    grade = payload.get("experience_grade")
    reason = str(payload.get("reason") or "").strip()
    if action not in ACTIONS:
        raise ValueError("请选择应该回复还是沉默")
    if grade not in GRADES:
        raise ValueError("请选择四档体验等级")
    if not reason:
        raise ValueError("请填写判断理由")

    with _WRITE_LOCK:
        packet = _load_packet(path)
        case = next(
            (item for item in packet.get("cases", []) if item["case_id"] == payload.get("case_id")),
            None,
        )
        if case is None:
            raise KeyError("case 不存在")
        case["human_review"] = {
            "status": "completed",
            "action": action,
            "experience_grade": grade,
            "reason": reason,
            "relationship_and_context_fit": payload.get("relationship_and_context_fit"),
            "information_emotion_or_momentum_value": payload.get(
                "information_emotion_or_momentum_value"
            ),
            "stale_joke_or_repetition": payload.get("stale_joke_or_repetition"),
            "fact_identity_or_boundary_risk": payload.get(
                "fact_identity_or_boundary_risk"
            ),
        }
        _save_packet(path, packet)
        completed = sum(
            item.get("human_review", {}).get("status") == "completed"
            for item in packet["cases"]
        )
    return {"saved": True, "completed": completed, "total": len(packet["cases"])}


def _messages(title: str, messages: list[dict[str, Any]]) -> str:
    rows = []
    for message in messages:
        role = "真人" if message.get("role") == "human" else "Bot"
        text = html.escape(str(message.get("text") or ""))
        rows.append(f'<div class="message"><span>{role}</span><p>{text}</p></div>')
    return f'<section class="card"><h2>{title}</h2>{"".join(rows) or "<p>无</p>"}</section>'


def _option(value: str, label: str, current: Any) -> str:
    selected = " selected" if current == value else ""
    return f'<option value="{value}"{selected}>{label}</option>'


def render_review_page(packet: dict[str, Any], index: int) -> str:
    cases = packet.get("cases", [])
    if not cases:
        return "<!doctype html><meta charset='utf-8'><p>没有待评审 case</p>"
    index = max(0, min(index, len(cases) - 1))
    case = cases[index]
    review = case.get("human_review") or {}
    completed = sum(
        item.get("human_review", {}).get("status") == "completed" for item in cases
    )
    candidate = case["candidate"]
    candidate_text = "（候选选择沉默）"
    if candidate["action"] == "reply":
        candidate_text = "<br>".join(html.escape(str(text)) for text in candidate["replies"])
    reason = html.escape(str(review.get("reason") or ""), quote=True)
    previous_index = max(0, index - 1)
    next_index = min(len(cases) - 1, index + 1)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>回复质量 Human Gold</title>
<style>
:root{{--bg:#f5f1e8;--card:#fffdf8;--ink:#24231f;--muted:#706b61;--line:#d9d1c2;--accent:#176b5b;--danger:#a53d32}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:920px;margin:0 auto;padding:28px 18px 80px}} header{{display:flex;justify-content:space-between;gap:16px;align-items:end;margin-bottom:20px}}
h1{{font-size:28px;margin:0}} h2{{font-size:16px;margin:0 0 12px;color:var(--muted)}} .progress{{font-variant-numeric:tabular-nums;color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:12px 0;box-shadow:0 4px 18px #4b44350b}}
.message{{display:grid;grid-template-columns:56px 1fr;gap:10px;padding:8px 0;border-bottom:1px solid #eee8dc}} .message:last-child{{border:0}}
.message span{{font-size:13px;color:var(--muted)}} .message p{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}}
.candidate{{font-size:18px;white-space:pre-wrap}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} label{{display:block;font-weight:650;margin:10px 0 5px}}
select,textarea{{width:100%;border:1px solid var(--line);border-radius:9px;background:white;color:var(--ink);padding:10px;font:inherit}} textarea{{min-height:100px;resize:vertical}}
.actions{{display:flex;justify-content:space-between;gap:10px;margin-top:18px}} button,.link{{border:0;border-radius:9px;padding:10px 16px;text-decoration:none;font-weight:700;cursor:pointer}}
button{{background:var(--accent);color:white}} .link{{background:#e8e1d5;color:var(--ink)}} #status{{min-height:24px;color:var(--danger);margin-top:10px}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}} header{{align-items:start;flex-direction:column}}}}
</style></head><body><main>
<header><div><h1>回复质量 Human Gold</h1><div class="progress">已完成 {completed}/{len(cases)}</div></div><div>第 {index + 1}/{len(cases)} 条</div></header>
{_messages("之前的聊天", case["pre_context"])}
{_messages("本轮新消息", case["target_messages"])}
<section class="card"><h2>待评价的历史候选（不是正确答案）</h2><div class="candidate">{candidate_text}</div></section>
<section class="card"><h2>你的判断</h2>
<div class="grid"><div><label for="action">此刻应该</label><select id="action"><option value="">请选择</option>{_option("reply", "回复", review.get("action"))}{_option("silence", "沉默", review.get("action"))}</select></div>
<div><label for="grade">候选体验等级</label><select id="grade"><option value="">请选择</option>{_option("good", "好：自然且有明确价值", review.get("experience_grade"))}{_option("acceptable", "可接受：没明显问题但价值有限", review.get("experience_grade"))}{_option("bad", "差：不自然、没接住或重复", review.get("experience_grade"))}{_option("critical", "严重：事实、身份或边界错误", review.get("experience_grade"))}</select></div></div>
<label for="reason">为什么</label><textarea id="reason" placeholder="结合关系、上下文和真人体验说明">{reason}</textarea>
<div class="grid"><div><label for="fit">关系与语境</label><select id="fit"><option value="">未判断</option>{_option("fits", "符合", review.get("relationship_and_context_fit"))}{_option("does_not_fit", "不符合", review.get("relationship_and_context_fit"))}{_option("unclear", "不确定", review.get("relationship_and_context_fit"))}</select></div>
<div><label for="value">信息/情绪/推进价值</label><select id="value"><option value="">未判断</option>{_option("clear", "明确", review.get("information_emotion_or_momentum_value"))}{_option("limited", "有限", review.get("information_emotion_or_momentum_value"))}{_option("none", "没有", review.get("information_emotion_or_momentum_value"))}</select></div>
<div><label for="repetition">旧梗或重复</label><select id="repetition"><option value="">未判断</option>{_option("none", "没有", review.get("stale_joke_or_repetition"))}{_option("stale", "旧梗", review.get("stale_joke_or_repetition"))}{_option("repeated", "重复", review.get("stale_joke_or_repetition"))}{_option("unclear", "不确定", review.get("stale_joke_or_repetition"))}</select></div>
<div><label for="risk">事实/身份/边界风险</label><select id="risk"><option value="">未判断</option>{_option("none", "没有", review.get("fact_identity_or_boundary_risk"))}{_option("fact", "事实", review.get("fact_identity_or_boundary_risk"))}{_option("identity", "身份/对象", review.get("fact_identity_or_boundary_risk"))}{_option("boundary", "边界", review.get("fact_identity_or_boundary_risk"))}{_option("unclear", "不确定", review.get("fact_identity_or_boundary_risk"))}</select></div></div>
<div id="status"></div><div class="actions"><a class="link" href="/?index={previous_index}">上一条</a><button id="save">保存并下一条</button><a class="link" href="/?index={next_index}">跳过</a></div>
</section></main><script>
document.getElementById('save').onclick=async()=>{{
 const field=id=>document.getElementById(id).value;
 const payload={{case_id:{json.dumps(case['case_id'])},action:field('action'),experience_grade:field('grade'),reason:field('reason'),relationship_and_context_fit:field('fit')||null,information_emotion_or_momentum_value:field('value')||null,stale_joke_or_repetition:field('repetition')||null,fact_identity_or_boundary_risk:field('risk')||null}};
 const response=await fetch('/api/review',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}}); const result=await response.json();
 if(!response.ok){{document.getElementById('status').textContent=result.error||'保存失败';return}} window.location='/?index={next_index}';
}};
</script></body></html>"""


def create_app(
    packet_path: Path = DEFAULT_PACKET, access_token: str | None = None
) -> FastAPI:
    app = FastAPI(title="Reply Quality Human Gold")
    configured_token = access_token or os.environ.get("REPLY_QUALITY_REVIEW_TOKEN")

    def authorized(request: Request) -> bool:
        if not configured_token:
            return True
        supplied = request.cookies.get("reply_quality_review_token", "")
        return secrets.compare_digest(supplied, configured_token)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        index: int = Query(0, ge=0),
        token: str | None = Query(None),
    ):
        if configured_token and token and secrets.compare_digest(token, configured_token):
            response = RedirectResponse(url=f"/?index={index}", status_code=303)
            response.set_cookie(
                "reply_quality_review_token",
                configured_token,
                httponly=True,
                samesite="strict",
            )
            return response
        if not authorized(request):
            return HTMLResponse("访问口令无效", status_code=403)
        return HTMLResponse(render_review_page(_load_packet(packet_path), index))

    @app.post("/api/review")
    def review(request: Request, payload: dict[str, Any] = Body(...)):
        if not authorized(request):
            return JSONResponse({"error": "访问口令无效"}, status_code=403)
        try:
            return JSONResponse(save_human_review(packet_path, payload))
        except (ValueError, KeyError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)

    return app


app = create_app()
