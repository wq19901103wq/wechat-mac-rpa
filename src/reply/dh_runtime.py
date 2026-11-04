"""Persistent stdio bridge to the configured DH checkout."""

import json
import logging
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
from threading import Lock, Thread
from types import SimpleNamespace
import uuid


_logger = logging.getLogger(__name__)
_LIMIT = 16 * 1024 * 1024


class _RemoteClient:
    def __init__(self, runtime, name, model):
        self.runtime, self.name, self.model_name = runtime, name, model

    def chat(self, messages, **kwargs):
        return self.runtime.call("chat", client=self.name, messages=messages, kwargs=kwargs)


class WorkerRuntime:
    def __init__(self, repository, instance, *, timeout, instances_root, settings_file=None, env_file=None):
        self.timeout = timeout
        self._lock = Lock()
        self._responses = queue.Queue()
        self._process = None
        python = repository / ".venv/bin/python"
        command = [str(python) if python.is_file() else sys.executable,
                   "-u", str(Path(__file__).with_name("dh_worker.py")),
                   "--repository", str(repository), "--instance", instance,
                   "--instances-root", str(instances_root), "--env-file", str(env_file)]
        if settings_file is not None:
            command.extend(["--settings-file", str(settings_file)])
        try:
            self._process = subprocess.Popen(command, cwd=repository, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            Thread(target=self._read, args=(self._process.stdout,), daemon=True).start()
            Thread(target=self._logs, args=(self._process.stderr,), daemon=True).start()
            ready = self._receive()
            if ready.get("type") != "ready" or ready.get("protocol") != 1:
                raise RuntimeError("DH worker 初始化协议不符")
            self.binding = ready["binding"]
            self.source = ready["source"]
            self.client_identity = ready["client_identity"]
            self.generator = SimpleNamespace(_llm={key: _RemoteClient(self, key, model)
                for key, model in ready["models"].items()})
            _logger.info("DH worker ready pid=%s source=%s", self._process.pid, self.source)
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

    def _receive(self):
        try:
            result = self._responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("DH worker 请求超时") from exc
        if isinstance(result, Exception):
            self.close()
            raise result
        if not isinstance(result, dict):
            self.close()
            raise ValueError("DH worker 返回无效协议对象")
        if result.get("type") == "error":
            raise RuntimeError("DH worker 失败：" + str(result.get("error")))
        return result

    def call(self, operation, **payload):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("DH worker 已关闭")
            request_id = uuid.uuid4().hex
            line = json.dumps(dict(id=request_id, operation=operation, **payload),
                              ensure_ascii=False, allow_nan=False).encode() + b"\n"
            if len(line) > _LIMIT:
                raise ValueError("DH worker 输入超过协议限制")
            self._process.stdin.write(line)
            self._process.stdin.flush()
            response = self._receive()
            if response.get("id") != request_id:
                self.close()
                raise ValueError("DH worker 响应请求不符")
            return response["result"]

    def draft(self, request, *, review=None):
        return self.call("draft", request=request, review=review)

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
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
