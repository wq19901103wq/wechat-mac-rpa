"""Persistent stdio bridge to the configured DH checkout."""

import json
import logging
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
from threading import Lock, Thread
import uuid


_logger = logging.getLogger(__name__)
_LIMIT = 16 * 1024 * 1024


class _RemoteClient:
    def __init__(self, runtime, name, model):
        self.runtime, self.name, self.model_name = runtime, name, model

    def chat(self, messages, **kwargs):
        return self.runtime.call("chat", client=self.name, messages=messages, kwargs=kwargs)


# 审查/闲聊类操作比 draft 更快，给更短的超时，避免下游挂起拖住整个主循环 10 分钟
REVIEW_TIMEOUT_SECONDS = 300


class WorkerRuntime:
    def __init__(self, repository, instance, *, timeout, instances_root, settings_file=None, env_file=None,
                 generator_ref=None, data_ref=None, sent_replies_db=None, source_chat_ids=None,
                 sent_replies_history_db=None):
        self.timeout = timeout
        self._lock = Lock()
        self._responses = queue.Queue()
        self._process = None
        python = repository / ".venv/bin/python"
        command = [str(python) if python.is_file() else sys.executable,
                   "-u", str(Path(__file__).with_name("dh_worker.py")),
                   "--repository", str(repository)]
        config = dict(instance=instance, instances_root=str(instances_root), env_file=str(env_file),
            settings_file=str(settings_file) if settings_file is not None else None,
            generator=generator_ref, data=data_ref,
            sent_replies_db=str(sent_replies_db) if sent_replies_db is not None else None,
            sent_replies_history_db=str(sent_replies_history_db) if sent_replies_history_db is not None else None,
            source_chat_ids=source_chat_ids)
        try:
            self._process = subprocess.Popen(command, cwd=repository, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            Thread(target=self._read, args=(self._process.stdout,), daemon=True).start()
            Thread(target=self._logs, args=(self._process.stderr,), daemon=True).start()
            self._send(dict(operation="initialize", protocol=1, config=config))
            ready = self._receive()
            if ready.get("type") != "ready" or ready.get("protocol") != 1:
                raise RuntimeError("DH worker 初始化协议不符")
            self.binding = ready["binding"]
            self.source = ready["source"]
            self.client_identity = ready["client_identity"]
            self.clients = {key: _RemoteClient(self, key, model)
                for key, model in ready["models"].items()}
            _logger.info("DH worker ready pid=%s source=%s data=%s sent_replies=%s",
                         self._process.pid, self.source, self.binding.get("data"), self.binding.get("sent_replies"))
        except BaseException:
            self.close()
            raise

    def _read(self, stream):
        try:
            with stream:
                while line := stream.readline(_LIMIT + 1):
                    if len(line) > _LIMIT or not line.endswith(b"\n"):
                        raise ValueError("DH worker 输出超过协议限制")
                    self._responses.put(json.loads(line))
            self._responses.put(RuntimeError("DH worker 已退出"))
        except Exception as exc:
            self._responses.put(exc)

    def _logs(self, stream):
        with stream:
            for line in stream:
                _logger.info("DH worker: %s", line.decode("utf-8", errors="replace").rstrip())

    def _receive(self, timeout=None):
        try:
            result = self._responses.get(timeout=self.timeout if timeout is None else timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("DH worker 请求超时") from exc
        if isinstance(result, Exception):
            self.close()
            raise result
        if not isinstance(result, dict):
            self.close()
            raise ValueError("DH worker 返回无效协议对象")
        return result

    def _send(self, payload):
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(line) > _LIMIT:
            raise ValueError("DH worker 输入超过协议限制")
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def call(self, operation, *, timeout=None, **payload):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("DH worker 已关闭")
            request_id = uuid.uuid4().hex
            self._send(dict(id=request_id, operation=operation, **payload))
            wait = self.timeout if timeout is None else timeout
            deadline = time.monotonic() + wait
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise TimeoutError("DH worker 请求超时")
                response = self._receive(remaining)
                if response.get("id") != request_id:
                    if response.get("type") == "error" and response.get("id") is None:
                        # 协议流层面的外来错误（如上游坏行重同步后的告警）：
                        # 不销毁 worker，继续等待本请求的响应
                        continue
                    self.close()
                    raise ValueError("DH worker 响应请求不符")
                if response.get("type") == "error":
                    raise RuntimeError("DH worker 失败：" + str(response.get("error")))
                return response["result"]

    def draft(self, request, *, review=None):
        return self.call("draft", request=request, review=review)

    def review(self, client, **review):
        return self.call("review", timeout=min(self.timeout, REVIEW_TIMEOUT_SECONDS),
                         client=client, review=review)

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
