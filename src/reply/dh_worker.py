"""Run DH under its own src package; stdout is reserved for the bridge."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import json
import logging
import os
from pathlib import Path
import sys
import traceback

from dotenv import load_dotenv

import dh_profile
from exposure_guard import ContextPrefetch, capture_generation, review_replies


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--instances-root", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--settings-file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    for key in tuple(os.environ):
        if key.startswith("DH_"):
            del os.environ[key]
    load_dotenv(args.env_file, override=True)
    os.environ.update(DH_INSTANCE=args.instance, DH_INSTANCES_ROOT=args.instances_root,
                      DH_ENV_FILE=args.env_file)
    if args.settings_file is not None:
        os.environ["DH_SETTINGS_FILE"] = args.settings_file
    sys.path.insert(0, str(args.repository.resolve()))
    runtime = None
    pool = ThreadPoolExecutor(max_workers=1)
    pending = active = None
    try:
        with redirect_stdout(sys.stderr):
            from src.generator.draft import DraftRuntime
            runtime = DraftRuntime(args.instance)
            try:
                dh_profile.install(namespace="src")
            except (ModuleNotFoundError, AttributeError):
                logging.getLogger(__name__).warning("DH 计时模块与当前源码不兼容，跳过埋点", exc_info=True)
        emit(dict(type="ready", protocol=1, binding=runtime.binding,
                  source=str(Path(sys.modules[DraftRuntime.__module__].__file__).resolve()),
                  client_identity={key: client.cache_identity() for key, client in runtime.generator._llm.items()
                                   if hasattr(client, "cache_identity")},
                  models={key: client.model_name for key, client in runtime.generator._llm.items()}))
        while line := sys.stdin.buffer.readline(16 * 1024 * 1024 + 1):
            request = None
            try:
                if len(line) > 16 * 1024 * 1024 or not line.endswith(b"\n"):
                    raise ValueError("DH worker 输入超过协议限制")
                request = json.loads(line)
                with redirect_stdout(sys.stderr):
                    if request["operation"] == "draft":
                        pending = None
                        review = request.get("review")
                        if review is not None and (active is None or active.future is None or active.future.done()):
                            pending = active = ContextPrefetch(pool, runtime.generator._llm[review["client"]],
                                request["request"]["input"], review["target_indices"])
                        dh_profile.begin(request["request"]["request_id"])
                        try:
                            with capture_generation(runtime.generator._llm,
                                    on_messages=pending.start if pending is not None else None) as messages:
                                result = runtime.draft(request["request"])
                            result["generation_messages"] = messages
                        finally:
                            dh_profile.finish()
                    elif request["operation"] == "chat":
                        result = runtime.generator._llm[request["client"]].chat(
                            request["messages"], **request["kwargs"])
                    elif request["operation"] == "review":
                        prefetch, pending = pending, None
                        result = review_replies(runtime.generator._llm[request["client"]],
                                                **request["review"], prefetch=prefetch)
                    else:
                        raise ValueError("未知 DH worker 操作")
                emit(dict(id=request["id"], result=result))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit(dict(type="error", id=request.get("id") if isinstance(request, dict) else None,
                          error=type(exc).__name__))
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit(dict(type="error", error=type(exc).__name__))
        return 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if runtime is not None:
            with redirect_stdout(sys.stderr):
                runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
