#!/usr/bin/env python3
"""
Badcase 审核台启动入口

用法:
    python scripts/review_server.py
    python scripts/review_server.py --port 8765

然后浏览器打开 http://localhost:8765
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Badcase Review Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="监听端口 (默认 8765)")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    print(f"🚀 Badcase Review Server")
    print(f"   URL: http://localhost:{args.port}")
    print(f"   按 Ctrl+C 停止")

    uvicorn.run(
        "src.badcase.review_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
