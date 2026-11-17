"""Run DH under its own src package; stdout is reserved for the bridge."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import json
import logging
import os
from pathlib import Path
import sys
import traceback

from dotenv import load_dotenv

import dh_profile
from exposure_guard import ContextPrefetch, capture_generation, review_replies
from sent_ledger import SentReplyLedger


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


@contextmanager
def request_trace(directory, runtime, request, *, kind="draft"):
    if directory is None:
        yield None
        return
    from src.infrastructure.tracing import CaseTrace
    trace = CaseTrace(Path(directory), {"case_id": request["request_id"]},
        dict(kind=kind, data_ref=runtime.binding["data"],
             candidate_ref=runtime.binding["generator"], cache={"enabled": False}))
    try:
        with trace.operation("generation" if kind == "draft" else kind, kind, 0, request):
            yield trace.ref
    except BaseException:
        trace.finish("failed")
        raise
    else:
        trace.finish("complete")


def review_client(runtime, request, prefetch, config_key="review_model_config"):
    config = request.get(config_key)
    if config is None:
        return runtime.generator._llm[request["client"]]
    if not request.get("trace_dir") or request["review"].get("parallel", True) or prefetch is not None:
        raise ValueError("模型配置覆盖仅支持独立的串行审查追踪")
    if config_key == "context_model_config" and (request.get("review_model_config") is not None
            or request["review"].get("direct_context")):
        raise ValueError("上下文模型覆盖不能与整次模型覆盖或直接原文审查混用")
    from src.infrastructure.config import load_settings
    from src.llm import ChatClient
    return ChatClient(load_settings(), config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    runtime = None
    ledger = None
    pool = ThreadPoolExecutor(max_workers=1)
    pending = active = None
    try:
        line = sys.stdin.buffer.readline(16 * 1024 * 1024 + 1)
        if len(line) > 16 * 1024 * 1024 or not line.endswith(b"\n"):
            raise ValueError("DH worker 初始化输入超过协议限制或不完整")
        initialization = json.loads(line)
        if (not isinstance(initialization, dict) or initialization.get("operation") != "initialize"
                or initialization.get("protocol") != 1 or not isinstance(initialization.get("config"), dict)):
            raise ValueError("DH worker 初始化协议不符")
        config = initialization["config"]
        fields = {"instance", "instances_root", "env_file", "settings_file", "generator", "data",
                  "sent_replies_db", "sent_replies_history_db", "source_chat_ids"}
        if set(config) != fields or any(not isinstance(config[key], str) or not config[key].strip()
                for key in ("instance", "instances_root", "env_file")):
            raise ValueError("DH worker 初始化配置不符")
        vars(args).update(config)
        for key in ("sent_replies_db", "sent_replies_history_db"):
            if getattr(args, key) is not None:
                setattr(args, key, Path(getattr(args, key)))
        for key in tuple(os.environ):
            if key.startswith("DH_"):
                del os.environ[key]
        load_dotenv(args.env_file, override=True)
        os.environ.update(DH_INSTANCE=args.instance, DH_INSTANCES_ROOT=args.instances_root,
                          DH_ENV_FILE=args.env_file)
        if args.settings_file is not None:
            os.environ["DH_SETTINGS_FILE"] = args.settings_file
        sys.path.insert(0, str(args.repository.resolve()))
        with redirect_stdout(sys.stderr):
            from src.generator.draft import DraftRuntime
            options = {}
            if args.generator is not None:
                options['generator_ref'] = args.generator
            if args.data is not None:
                options['data_ref'] = args.data
            if args.sent_replies_db is not None:
                ledger = SentReplyLedger(args.sent_replies_db, args.source_chat_ids,
                    history_database=args.sent_replies_history_db)
                options['sent_replies'] = ledger.read()
            runtime = DraftRuntime(args.instance, **options)
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
                    # 消费掉本行剩余字节直到换行符，重新对齐协议流，
                    # 避免后续 readline 读到半行垃圾而连锁失败
                    while True:
                        chunk = sys.stdin.buffer.readline(16 * 1024 * 1024 + 1)
                        if not chunk or chunk.endswith(b"\n"):
                            break
                    raise ValueError("DH worker 输入超过协议限制")
                request = json.loads(line)
                with redirect_stdout(sys.stderr):
                    if request["operation"] == "draft":
                        if ledger is not None:
                            evidence = ledger.read()
                            if evidence:
                                runtime.update_sent_replies(evidence)
                        pending = None
                        review = request.get("review")
                        if review is not None and (active is None or active.future is None or active.future.done()):
                            pending = active = ContextPrefetch(pool, runtime.generator._llm[review["client"]],
                                request["request"]["input"], review["target_indices"])
                        dh_profile.begin(request["request"]["request_id"])
                        try:
                            with request_trace(request.get("trace_dir"), runtime, request["request"]) as trace_ref:
                                with capture_generation(runtime.generator._llm,
                                        on_messages=pending.start if pending is not None else None) as messages:
                                    result = runtime.draft(request["request"])
                            if trace_ref is not None:
                                result["trace_ref"] = trace_ref
                            result["generation_messages"] = messages
                        except Exception:
                            # draft 失败：取消预取并清空引用，避免陈旧 future 占住单线程池
                            if pending is not None and pending.future is not None:
                                pending.future.cancel()
                            pending = active = None
                            raise
                        finally:
                            dh_profile.finish()
                    elif request["operation"] == "chat":
                        result = runtime.generator._llm[request["client"]].chat(
                            request["messages"], **request["kwargs"])
                    elif request["operation"] == "review":
                        prefetch, pending = pending, None
                        client = review_client(runtime, request, prefetch)
                        context_options = {}
                        if request.get("context_model_config") is not None:
                            context_options["context_client"] = review_client(
                                runtime, request, prefetch, "context_model_config")
                        with request_trace(request.get("trace_dir"), runtime,
                                dict(request_id=request.get("trace_case_id", request["id"]),
                                     review=request["review"]), kind="review") as trace_ref:
                            result = review_replies(client, **request["review"], prefetch=prefetch, **context_options)
                        if trace_ref is not None:
                            result["trace_ref"] = trace_ref
                        if request.get("review_model_config") is not None:
                            result["client_identity"] = client.cache_identity()
                        if context_options:
                            result["context_review"]["client_identity"] = context_options["context_client"].cache_identity()
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
