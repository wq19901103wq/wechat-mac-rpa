#!/usr/bin/env python3
"""
百炼 Qwen-VL-OCR 精度验证脚本

用法:
    export DASHSCOPE_API_KEY=sk-xxxxx
    python3 scripts/benchmark_qwen_vl_ocr.py --mode legacy_errors
    python3 scripts/benchmark_qwen_vl_ocr.py --mode regression
    python3 scripts/benchmark_qwen_vl_ocr.py --mode single --image tests/fixtures/regression_chat_list_pollution_20260421.png
    python3 scripts/benchmark_qwen_vl_ocr.py --mode sample --count 10

输出:
    - results/qwen_vl_ocr_{timestamp}/report.md   汇总报告
    - results/qwen_vl_ocr_{timestamp}/details.json 详细结果
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 自动加载 .env 文件中的环境变量
def _load_env():
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

from src.ocr.vision_ocr import VisionOCREngine
from src.layout.layout_parser import LayoutParser
from src.message.extractor import MessageExtractor
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: 需要安装 openai 库: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Prompt 设计：要求模型输出结构化 JSON，与现有测试预期格式兼容
# ---------------------------------------------------------------------------

PROMPT_V1 = """你是一位专精 UI 截图文字识别的 OCR 引擎。请仔细识别这张微信 Mac 版截图中的文字信息，并输出为 JSON。

截图包含以下区域：
1. 左侧聊天列表：每个条目包含头像、昵称、最后一条消息预览时间、未读角标数字（红色圆形背景）
2. 中间上方标题栏：当前聊天名称
3. 中间消息区域：消息按对话顺序从上到下排列

请严格按以下 JSON 格式输出（不要加 markdown 代码块，直接输出纯 JSON）：

{
  "chat_name": "当前聊天名称，如果没有则空字符串",
  "chat_list": [
    {"nickname": "昵称1", "unread_count": "未读数量，没有则为空字符串"},
    {"nickname": "昵称2", "unread_count": "3"}
  ],
  "messages": [
    {"sender": "自己", "text": "消息内容"},
    {"sender": "对方", "text": "消息内容"}
  ]
}

【关键识别规则 - 必须严格遵守】

1. 未读角标（unread_count）：
   - 必须是红色圆形背景中的白色/黑色数字，位于头像右上角
   - 预览消息右侧的时间戳（如"09:31"、"昨天"、"00:57"）是消息时间，不是未读角标，unread_count 必须设为空字符串""
   - 如果没有红色圆形数字，unread_count 为空字符串""

2. 聊天列表（chat_list）：
   - 左侧每个条目提取昵称，忽略头像区域的所有数字（除非是红色圆形未读角标）
   - 当前高亮选中的聊天条目也必须包含
   - 预览消息文字不要放入 nickname

3. 消息 sender 判断（这是最容易出错的地方，请仔细看气泡颜色）：
   - 绿色背景的气泡 = "自己" 发送的消息
   - 白色或浅灰色背景的气泡 = "对方" 发送的消息
   - 如果看不清颜色：右侧对齐的气泡是自己，左侧对齐的气泡是对方
   - 时间戳（如"昨天 21:58"、"11:34"、"00:22"）不是消息，不要输出

4. 消息（messages）：
   - 只包含实际对话内容，排除所有时间戳
   - 按截图中从上到下顺序排列

5. 输入框过滤（重要）：
   - 截图最底部是输入框区域（有表情😊、文件📎、截图✂️、语音🎤按钮）
   - 输入框中的文字是未发送的草稿，不是已发送的消息，必须排除
   - 不要输出输入框中的任何内容

6. 只输出 JSON，不要任何解释文字。
"""

PROMPT_V2 = """你正在分析一张微信 Mac 客户端的截图。请按以下步骤思考，最终输出 JSON。

【第一步：理解整体布局】
截图从左到右分为三个纵向区域：
- 最左侧窄条：聊天列表（显示头像+昵称+预览+未读角标）
- 中间上方：标题栏（当前聊天名称）
- 中间下方：消息对话区域

【第二步：提取聊天列表】
每个列表条目从左到右的元素顺序是：头像 → 昵称 → 消息预览 → 时间戳。

未读角标的精确定义（最容易误判，请特别注意）：
- ✅ 是未读角标：红色圆形小徽章，直径很小，内有 1-2 位白色数字，严格位于头像的右上角外侧，与头像边缘有微小间距
- ❌ 不是未读角标：头像内部的数字（如"10"、"100"、"1000"这类群成员数）、预览消息右侧的时间（如"09:31"、"昨天"、"00:57"）、昵称中的数字
- 绝大多数条目的未读角标为空字符串""

昵称提取规则：
- 只提取条目中的用户/群名称，忽略预览消息文字
- 当前高亮选中的条目也必须包含
- 忽略头像内部的所有数字和图形

【第三步：提取消息并判断 sender】
微信 Mac 版消息布局铁律：
- 自己的消息：显示在屏幕右侧，背景为浅绿色（#95EC69），左侧显示对方头像的地方没有头像
- 对方的消息：显示在屏幕左侧，背景为白色或浅灰色，左侧有圆形头像
- 群聊中，对方消息上方会显示发送者昵称（如 "wanglc"），必须读取并作为 sender 输出
- 私聊中，对方消息上方没有昵称标签，sender 统一输出 "对方"
- 时间戳（灰色小字，如"昨天 21:58"、"11:34"）穿插在消息之间，不是消息内容，必须排除
- 如果无法分辨颜色，优先以水平位置为准：右侧是自己，左侧是对方

sender 判断规则（按优先级）：
1. 绿色背景气泡 → sender = "自己"
2. 灰色/白色背景气泡 + 群聊（chat_name 中有群成员数如 "(5)"）→ sender = 消息上方显示的实际昵称（如 "wanglc"）
3. 灰色/白色背景气泡 + 私聊 → sender = "对方"

【输出格式】
严格按以下 JSON 输出，不要 markdown 代码块，不要解释：

{
  "chat_name": "标题栏中的当前聊天名称",
  "chat_list": [
    {"nickname": "昵称", "unread_count": ""},
    {"nickname": "昵称", "unread_count": "3"}
  ],
  "messages": [
    {"sender": "自己", "text": "消息内容"},
    {"sender": "wanglc", "text": "群聊中对方的消息，sender 为实际昵称"},
    {"sender": "对方", "text": "私聊中对方的消息"}
  ]
}
"""

PROMPT = PROMPT_V2

# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    image_path: str
    chat_name: str
    chat_list: List[Dict[str, str]]
    messages: List[Dict[str, str]]
    latency_ms: float
    error: Optional[str] = None
    raw_response: str = ""


class QwenVLOCRClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-vl-ocr", enable_thinking: bool = True):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "请设置 DASHSCOPE_API_KEY 环境变量，或在阿里云百炼控制台获取 API Key"
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model = model
        self.extra_body = {}
        # qwen3.5-flash 默认开启 thinking，可通过 extra_body 关闭
        if not enable_thinking and "qwen3.5" in model:
            self.extra_body["enable_thinking"] = False

    def recognize(self, image_path: str) -> OCRResult:
        b64 = self._image_to_base64(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]

        start = time.time()
        try:
            kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            response = self.client.chat.completions.create(**kwargs)
            latency_ms = (time.time() - start) * 1000
            raw = response.choices[0].message.content or ""
            parsed = self._extract_json(raw)
            return OCRResult(
                image_path=image_path,
                chat_name=parsed.get("chat_name", ""),
                chat_list=parsed.get("chat_list", []),
                messages=parsed.get("messages", []),
                latency_ms=latency_ms,
                raw_response=raw,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return OCRResult(
                image_path=image_path,
                chat_name="",
                chat_list=[],
                messages=[],
                latency_ms=latency_ms,
                error=str(e),
                raw_response="",
            )

    @staticmethod
    def _image_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        # 去掉 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)


# ---------------------------------------------------------------------------
# 本地 OCR Pipeline（对比基线）
# ---------------------------------------------------------------------------

class LocalOCRBaseline:
    def __init__(self):
        profile = PROFILE_WECHAT_MAC_1760X1280
        self.ocr = VisionOCREngine()
        self.layout = LayoutParser(profile)
        self.extractor = MessageExtractor(profile)

    def recognize(self, image_path: str) -> OCRResult:
        start = time.time()
        try:
            elements = self.ocr.recognize(image_path)
            layout = self.layout.parse(elements, image_path)
            messages = self.extractor.extract(layout)
            latency_ms = (time.time() - start) * 1000

            chat_list = [
                {"nickname": item.nickname, "unread_count": item.unread_count or ""}
                for item in layout.chat_list_items
            ]
            msgs = [
                {"sender": "自己" if m.sender_type.value == "self" else "对方", "text": m.text}
                for m in messages
            ]
            return OCRResult(
                image_path=image_path,
                chat_name=layout.chat_name or "",
                chat_list=chat_list,
                messages=msgs,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return OCRResult(
                image_path=image_path,
                chat_name="",
                chat_list=[],
                messages=[],
                latency_ms=latency_ms,
                error=str(e),
            )


# ---------------------------------------------------------------------------
# 对比评分
# ---------------------------------------------------------------------------

@dataclass
class ComparisonScore:
    chat_name_match: bool
    chat_list_nickname_iou: float
    chat_list_unread_accuracy: float
    messages_text_iou: float
    overall_score: float


def _normalize_chat_list(chat_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """规范化 chat_list，统一字段。"""
    result = []
    for item in chat_list:
        nick = item.get("nickname", "")
        unread = str(item.get("unread_count", ""))
        if nick:
            result.append({"nickname": nick, "unread_count": unread})
    return result


def _normalize_messages(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """规范化 messages。"""
    return [
        {"sender": m.get("sender", ""), "text": m.get("text", "")}
        for m in msgs
        if m.get("text", "")
    ]


def score_vs_expected(qwen: OCRResult, expected: dict) -> ComparisonScore:
    """将 Qwen-VL-OCR 结果与 .json 预期对比。"""
    exp_name = expected.get("chat_name", "")
    qwen_name = qwen.chat_name or ""
    name_match = exp_name == qwen_name

    exp_list = _normalize_chat_list(expected.get("chat_list", []))
    qwen_list = _normalize_chat_list(qwen.chat_list)

    exp_nicks = {i["nickname"] for i in exp_list}
    qwen_nicks = {i["nickname"] for i in qwen_list}
    union_nicks = exp_nicks | qwen_nicks
    inter_nicks = exp_nicks & qwen_nicks
    nick_iou = len(inter_nicks) / len(union_nicks) if union_nicks else 1.0

    # 未读角标：只在昵称交集上比较
    unread_correct = 0
    unread_total = 0
    exp_unread = {i["nickname"]: i["unread_count"] for i in exp_list}
    qwen_unread = {i["nickname"]: i["unread_count"] for i in qwen_list}
    for nick in inter_nicks:
        unread_total += 1
        if exp_unread.get(nick) == qwen_unread.get(nick):
            unread_correct += 1
    unread_acc = unread_correct / unread_total if unread_total else 1.0

    # 消息文本：忽略 sender，只看 text 集合的 IoU
    exp_msgs = [m["text"] for m in _normalize_messages(expected.get("messages", []))]
    qwen_msgs = [m["text"] for m in _normalize_messages(qwen.messages)]
    exp_set = set(exp_msgs)
    qwen_set = set(qwen_msgs)
    union_msgs = exp_set | qwen_set
    inter_msgs = exp_set & qwen_set
    msg_iou = len(inter_msgs) / len(union_msgs) if union_msgs else 1.0

    overall = (nick_iou + unread_acc + msg_iou) / 3
    if not name_match:
        overall *= 0.9  # 聊天名错一个小惩罚

    return ComparisonScore(
        chat_name_match=name_match,
        chat_list_nickname_iou=nick_iou,
        chat_list_unread_accuracy=unread_acc,
        messages_text_iou=msg_iou,
        overall_score=overall,
    )


def score_vs_local(qwen: OCRResult, local: OCRResult) -> ComparisonScore:
    """将 Qwen-VL-OCR 与本地 OCR 对比（本地作为参考基线）。"""
    dummy_expected = {
        "chat_name": local.chat_name,
        "chat_list": local.chat_list,
        "messages": local.messages,
    }
    return score_vs_expected(qwen, dummy_expected)


# ---------------------------------------------------------------------------
# 批量运行 & 报告
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    image_path: str
    has_expected: bool
    qwen: OCRResult
    local: OCRResult
    vs_expected: Optional[ComparisonScore] = None
    vs_local: Optional[ComparisonScore] = None


def run_benchmark(image_paths: List[str], expected_map: Dict[str, dict], model: str = "qwen-vl-ocr", enable_thinking: bool = True) -> List[BenchmarkRun]:
    qwen_client = QwenVLOCRClient(model=model, enable_thinking=enable_thinking)
    local_client = LocalOCRBaseline()
    results = []

    for idx, path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] {Path(path).name} ...", end=" ", flush=True)

        qwen_result = qwen_client.recognize(path)
        local_result = local_client.recognize(path)

        expected = expected_map.get(path)
        vs_expected = score_vs_expected(qwen_result, expected) if expected else None
        vs_local = score_vs_local(qwen_result, local_result)

        run = BenchmarkRun(
            image_path=path,
            has_expected=expected is not None,
            qwen=qwen_result,
            local=local_result,
            vs_expected=vs_expected,
            vs_local=vs_local,
        )
        results.append(run)

        status = "OK" if not qwen_result.error else f"ERR:{qwen_result.error[:30]}"
        score_str = f"expected={vs_expected.overall_score:.2f}" if vs_expected else f"local={vs_local.overall_score:.2f}"
        print(f"{status} | Qwen:{qwen_result.latency_ms:.0f}ms Local:{local_result.latency_ms:.0f}ms | {score_str}")

        # 防止触发百炼限流，简单 sleep
        time.sleep(0.5)

    return results


def generate_report(runs: List[BenchmarkRun], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) JSON 详细结果
    details = []
    for r in runs:
        details.append({
            "image": r.image_path,
            "has_expected": r.has_expected,
            "qwen": {
                "chat_name": r.qwen.chat_name,
                "chat_list": r.qwen.chat_list,
                "messages": r.qwen.messages,
                "latency_ms": round(r.qwen.latency_ms, 1),
                "error": r.qwen.error,
            },
            "local": {
                "chat_name": r.local.chat_name,
                "chat_list": r.local.chat_list,
                "messages": r.local.messages,
                "latency_ms": round(r.local.latency_ms, 1),
                "error": r.local.error,
            },
            "vs_expected": {
                "chat_name_match": r.vs_expected.chat_name_match,
                "chat_list_nickname_iou": round(r.vs_expected.chat_list_nickname_iou, 3),
                "chat_list_unread_accuracy": round(r.vs_expected.chat_list_unread_accuracy, 3),
                "messages_text_iou": round(r.vs_expected.messages_text_iou, 3),
                "overall_score": round(r.vs_expected.overall_score, 3),
            } if r.vs_expected else None,
            "vs_local": {
                "chat_name_match": r.vs_local.chat_name_match,
                "chat_list_nickname_iou": round(r.vs_local.chat_list_nickname_iou, 3),
                "chat_list_unread_accuracy": round(r.vs_local.chat_list_unread_accuracy, 3),
                "messages_text_iou": round(r.vs_local.messages_text_iou, 3),
                "overall_score": round(r.vs_local.overall_score, 3),
            } if r.vs_local else None,
        })

    json_path = output_dir / f"details_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    # 2) Markdown 报告
    total = len(runs)
    errors = sum(1 for r in runs if r.qwen.error)
    expected_runs = [r for r in runs if r.vs_expected]
    local_runs = [r for r in runs if r.vs_local]

    avg_qwen_latency = sum(r.qwen.latency_ms for r in runs) / total if total else 0
    avg_local_latency = sum(r.local.latency_ms for r in runs) / total if total else 0

    avg_expected_score = sum(r.vs_expected.overall_score for r in expected_runs) / len(expected_runs) if expected_runs else 0
    avg_local_score = sum(r.vs_local.overall_score for r in local_runs) / len(local_runs) if local_runs else 0

    md_lines = [
        "# Qwen-VL-OCR 精度验证报告",
        f"",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本数: {total}",
        f"- API 错误数: {errors}",
        f"- 模型: qwen-vl-ocr (阿里云百炼)",
        f"",
        "## 总体评分",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 平均延迟 (Qwen) | {avg_qwen_latency:.0f} ms |",
        f"| 平均延迟 (本地 OCR) | {avg_local_latency:.0f} ms |",
        f"| vs 预期均分 | {avg_expected_score:.3f} |" if expected_runs else "",
        f"| vs 本地 OCR 均分 | {avg_local_score:.3f} |" if local_runs else "",
        f"",
        "## 逐样本详情",
        "",
        "| # | 截图 | Qwen延迟 | 本地延迟 | vs预期 | vs本地 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]

    for idx, r in enumerate(runs, 1):
        name = Path(r.image_path).name
        vs_exp = f"{r.vs_expected.overall_score:.2f}" if r.vs_expected else "-"
        vs_loc = f"{r.vs_local.overall_score:.2f}" if r.vs_local else "-"
        status = "❌ ERR" if r.qwen.error else "✅ OK"
        md_lines.append(
            f"| {idx} | {name} | {r.qwen.latency_ms:.0f}ms | {r.local.latency_ms:.0f}ms | {vs_exp} | {vs_loc} | {status} |"
        )

    md_lines.extend([
        "",
        "## 重点 case 分析",
        "",
    ])

    # 找出得分最低的 3 个
    scored = [(r, r.vs_expected.overall_score if r.vs_expected else r.vs_local.overall_score) for r in runs]
    scored.sort(key=lambda x: x[1])
    for r, score in scored[:3]:
        md_lines.extend([
            f"### {Path(r.image_path).name} (score={score:.3f})",
            "",
            "**Qwen-VL-OCR 输出:**",
            "",
            "```json",
            json.dumps({
                "chat_name": r.qwen.chat_name,
                "chat_list": r.qwen.chat_list,
                "messages": r.qwen.messages,
            }, ensure_ascii=False, indent=2),
            "```",
            "",
            "**本地 OCR 输出:**",
            "",
            "```json",
            json.dumps({
                "chat_name": r.local.chat_name,
                "chat_list": r.local.chat_list,
                "messages": r.local.messages,
            }, ensure_ascii=False, indent=2),
            "```",
            "",
        ])

    md_path = output_dir / f"report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n📊 报告已生成:")
    print(f"   - JSON: {json_path}")
    print(f"   - Markdown: {md_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def load_expected_json(image_path: str) -> Optional[dict]:
    json_path = image_path.replace(".png", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_legacy_error_images() -> List[str]:
    base = PROJECT_ROOT / "tests" / "fixtures" / "legacy" / "errors"
    pngs = sorted(base.glob("*.png"))
    return [str(p) for p in pngs]


def get_regression_images() -> List[str]:
    base = PROJECT_ROOT / "tests" / "fixtures"
    pngs = sorted(base.glob("regression_*.png"))
    return [str(p) for p in pngs]


def get_historical_sample(count: int = 10) -> List[str]:
    base = PROJECT_ROOT / "tests" / "fixtures" / "historical_screenshots"
    pngs = list(base.glob("*.png"))
    random.shuffle(pngs)
    return [str(p) for p in pngs[:count]]


def main():
    parser = argparse.ArgumentParser(description="百炼 Qwen-VL-OCR 精度验证")
    parser.add_argument("--mode", choices=["legacy_errors", "regression", "single", "sample", "all"],
                        default="regression", help="运行模式")
    parser.add_argument("--image", type=str, help="单图模式时的图片路径")
    parser.add_argument("--count", type=int, default=10, help="sample 模式采样数量")
    parser.add_argument("--model", type=str, default="qwen-vl-ocr",
                        choices=["qwen-vl-ocr", "qwen-vl-plus", "qwen-vl-plus-latest", "qwen3-vl-plus", "qwen3-vl-flash", "qwen3.5-flash"],
                        help="百炼模型名称")
    parser.add_argument("--output-dir", type=str, default="results/qwen_vl_ocr",
                        help="输出目录")
    parser.add_argument("--no-thinking", action="store_true",
                        help="关闭思考模式（仅对 qwen3.5-flash 有效）")
    args = parser.parse_args()

    # 检查 API Key
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: 请先设置环境变量 DASHSCOPE_API_KEY")
        print("       获取方式: 阿里云百炼控制台 -> API Key 管理")
        sys.exit(1)

    # 收集图片
    if args.mode == "legacy_errors":
        images = get_legacy_error_images()
    elif args.mode == "regression":
        images = get_regression_images()
    elif args.mode == "single":
        if not args.image:
            print("ERROR: --mode single 需要 --image 参数")
            sys.exit(1)
        images = [args.image]
    elif args.mode == "sample":
        images = get_historical_sample(args.count)
    elif args.mode == "all":
        images = get_legacy_error_images() + get_regression_images()
    else:
        images = []

    if not images:
        print("WARNING: 没有找到图片")
        sys.exit(1)

    print(f"即将验证 {len(images)} 张截图...")
    print(f"模型: {args.model} @ 阿里云百炼")
    print()

    # 加载预期
    expected_map = {}
    for img in images:
        exp = load_expected_json(img)
        if exp:
            expected_map[img] = exp

    # 运行
    results = run_benchmark(images, expected_map, model=args.model, enable_thinking=not args.no_thinking)

    # 生成报告
    output_dir = PROJECT_ROOT / args.output_dir / args.model
    generate_report(results, output_dir)


if __name__ == "__main__":
    main()
