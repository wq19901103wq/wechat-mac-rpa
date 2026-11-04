#!/usr/bin/env python3
"""Generate text drafts from the shared DH checkout; never enter the bot loop."""

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reply.digital_human import DigitalHumanDraftClient, load_digital_human_config, messages_input


def screenshot_input():
    from dotenv import dotenv_values
    from src.capture.window_capture import CaptureValidationError, WeChatNotReadyError, WindowNotFoundError
    from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
    from src.perception.smart_pipeline import SmartPerceptionPipeline

    pipeline = SmartPerceptionPipeline(PROFILE_WECHAT_MAC_1760X1280,
        api_key=dotenv_values(ROOT / '.env').get('DASHSCOPE_API_KEY'))
    try:
        result = pipeline.perceive()
    except (CaptureValidationError, WeChatNotReadyError, WindowNotFoundError) as exc:
        raise RuntimeError(str(exc)) from exc
    if result is None or not result.messages:
        raise RuntimeError('截图未识别到聊天消息')
    return messages_input(result.messages, is_group=result.is_group)


def main():
    parser = argparse.ArgumentParser(description="digital-human 只生成、不发送文本入口")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--instance")
    parser.add_argument("--timeout", type=float)
    for key in ("instances_root", "settings_file", "env_file"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="人工新写文本，DH 按接收时间标记；不是历史消息")
    inputs.add_argument("--input-file", type=Path, help="结构化原始消息或截图观察输入 JSON")
    inputs.add_argument("--screenshot", action="store_true", help="识别当前微信截图并生成草稿，不进入 Bot 循环")
    inputs.add_argument("--stream", action="store_true", help="从标准输入逐行读取输入 JSON，复用同一 DH 函数运行时")
    parser.add_argument("--chat-type", choices=("private", "group"))
    parser.add_argument("--chat-name")
    parser.add_argument("--source-chat-id")
    parser.add_argument("--sender", default="other")
    parser.add_argument("--save-input", type=Path, help="保存单次草稿的结构化输入，供本地复现")
    args = parser.parse_args()
    try:
        config = load_digital_human_config() if (ROOT / 'config/digital_human.json').is_file() else {}
        args.repository = args.repository or config.get('repository', ROOT.parent / 'digital-human')
        args.instance = args.instance or config.get('instance')
        args.timeout = args.timeout if args.timeout is not None else config.get('timeout', 900)
        for key in ('instances_root', 'settings_file', 'env_file'):
            if getattr(args, key) is None:
                setattr(args, key, config.get(key))
        if not args.instance:
            parser.error('需要 --instance 或 config/digital_human.json 本机配置')
        if args.text is not None:
            if not all((args.chat_type, args.chat_name, args.source_chat_id)):
                parser.error("--text 需要 --chat-type、--chat-name 和 --source-chat-id")
            online_input = {"kind": "composed", "chat_type": args.chat_type, "chat_name": args.chat_name,
                            "source_chat_id": args.source_chat_id, "text": args.text, "sender": args.sender}
        elif args.input_file is not None:
            online_input = json.loads(args.input_file.read_text(encoding="utf-8"))
        elif args.screenshot:
            with redirect_stdout(sys.stderr):
                online_input = screenshot_input()
        if args.save_input is not None:
            if args.stream:
                parser.error('--save-input 仅支持单次输入')
            args.save_input.write_text(json.dumps(online_input, ensure_ascii=False, indent=2), encoding='utf-8')
        with DigitalHumanDraftClient(args.repository, args.instance, timeout=args.timeout,
                instances_root=args.instances_root, settings_file=args.settings_file, env_file=args.env_file) as client:
            print(f"DH 已就绪；初始化 {client.startup_ms / 1000:.2f} 秒", file=sys.stderr, flush=True)
            if args.stream:
                for line in sys.stdin:
                    print(json.dumps(client.draft(json.loads(line)), ensure_ascii=False), flush=True)
            else:
                print(json.dumps(client.draft(online_input), ensure_ascii=False), flush=True)
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"草稿失败（未发送）：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
