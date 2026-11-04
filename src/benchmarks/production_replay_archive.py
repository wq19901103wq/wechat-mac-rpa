"""Append-only evidence for production-aligned, single-step reply replay."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "production-single-step-v2"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    return repr(value)


def _public_url(value: Any) -> str:
    """Keep endpoint identity while dropping credentials, query and fragment."""
    if not value:
        return ""
    try:
        parts = urlsplit(str(value))
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except Exception:
        return ""


def describe_llm_client(client: Any) -> Dict[str, Any]:
    if isinstance(client, RecordingLLMClient):
        client = client.wrapped
    nested = getattr(client, "client", None)
    return {
        "class": f"{type(client).__module__}.{type(client).__name__}",
        "model": getattr(client, "model", ""),
        "max_tokens": getattr(client, "max_tokens", None),
        "base_url": _public_url(getattr(client, "base_url", "") or getattr(nested, "base_url", "")),
    }


def snapshot_session_memory(generator: Any) -> Dict[str, Any]:
    memory = getattr(generator, "session_memory", None)
    return {
        "class": f"{type(memory).__module__}.{type(memory).__name__}" if memory is not None else "",
        "sessions": _jsonable(getattr(memory, "_sessions", {})),
    }


def generator_runtime_config(generator: Any) -> Dict[str, Any]:
    fields = (
        "enable_time_awareness",
        "enable_reply_restraint",
        "enable_unread_dedup",
        "enable_timestamps",
        "enable_react_tools",
        "enable_self_refine",
        "enable_fact_check",
        "enable_persona_few_shots",
        "persona_few_shot_count",
        "persona_few_shot_max_chars",
        "persona_few_shot_allow_unreviewed",
    )
    return {name: _jsonable(getattr(generator, name, None)) for name in fields}


def _serialize_llm_response(response: Any) -> Any:
    if isinstance(response, str):
        return {"kind": "text", "content": response}
    return {
        "kind": "message",
        "content": getattr(response, "content", ""),
        "reasoning_content": getattr(response, "reasoning_content", ""),
        "tool_calls": _jsonable(getattr(response, "tool_calls", None)),
        "raw": _jsonable(response),
    }


class RecordingLLMClient:
    """Transparent client proxy that records exact per-call inputs and outputs."""

    def __init__(self, wrapped: Any, archive: "ProductionReplayArchive", role: str):
        self.wrapped = wrapped
        self.archive = archive
        self.role = role

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        started = time.time()
        request = {"args": _jsonable(args), "kwargs": _jsonable(kwargs)}
        try:
            response = self.wrapped.chat(*args, **kwargs)
        except Exception as exc:
            self.archive.record_llm_call(
                self.role,
                describe_llm_client(self.wrapped),
                request,
                error=f"{type(exc).__name__}: {exc}",
                started_at=started,
            )
            raise
        self.archive.record_llm_call(
            self.role,
            describe_llm_client(self.wrapped),
            request,
            response=_serialize_llm_response(response),
            started_at=started,
        )
        return response


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False)
    temp.replace(path)


def snapshot_global_store(store: Any) -> Dict[str, Any]:
    chats = {}
    for name, state in sorted(getattr(store, "chats", {}).items()):
        chats[name] = {
            "chat_id": getattr(state, "chat_id", ""),
            "chat_name": getattr(state, "chat_name", name),
            "is_group": bool(getattr(state, "is_group", False)),
            "messages": _jsonable(getattr(state, "messages", [])),
            "pending_self_messages": _jsonable(getattr(state, "pending_self_messages", [])),
        }
    return {
        "max_messages": getattr(store, "max_messages", None),
        "chats": chats,
    }


class ProductionReplayArchive:
    """Capture immutable inputs and a pre-action decision point for each real tick."""

    def __init__(self, root: Union[Path, str], session_id: str, config: Dict[str, Any]):
        self.session_dir = Path(root) / session_id
        self.ticks_dir = self.session_dir / "ticks"
        self.ticks_dir.mkdir(parents=True, exist_ok=False)
        self._active: Dict[int, Path] = {}
        self._current_tick_id: Optional[int] = None
        self._llm_call_counts: Dict[int, int] = {}
        self._pending_llm_calls: Dict[int, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        _atomic_json(
            self.session_dir / "session.json",
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "replay_contract": {
                    "mode": "single_step",
                    "cutoff": "before_any_external_side_effect_or_state_advance",
                    "candidate_branch_continuation": False,
                    "all_tool_execution_blocked_in_replay": True,
                },
            },
        )

    def capture_session_start(self, store: Any, source_files: Optional[List[Path]] = None) -> None:
        state_path = self.session_dir / "session_start_state.json.gz"
        _atomic_gzip_json(state_path, snapshot_global_store(store))
        sources = []
        for source in source_files or []:
            if source.is_file():
                digest = _sha256(source)
                archive_name = f"{digest[:16]}_{source.name}"
                target = self.session_dir / "dependencies" / archive_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                sources.append({
                    "path": str(source),
                    "sha256": digest,
                    "size": source.stat().st_size,
                    "snapshot": str(target.relative_to(self.session_dir)),
                })
        _atomic_json(
            self.session_dir / "session_start_evidence.json",
            {"state_sha256": _sha256(state_path), "source_files": sources},
        )

    def begin_tick(self, tick_id: int, store: Any) -> None:
        tick_dir = self.ticks_dir / str(tick_id)
        tick_dir.mkdir(parents=False, exist_ok=False)
        state_path = tick_dir / "state_before.json.gz"
        _atomic_gzip_json(state_path, snapshot_global_store(store))
        _atomic_json(
            tick_dir / "tick.json",
            {
                "schema_version": SCHEMA_VERSION,
                "tick_id": tick_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "state_before_sha256": _sha256(state_path),
                "decision_point_captured": False,
                "eligible_for_formal_replay": False,
            },
        )
        self._active[tick_id] = tick_dir
        self._current_tick_id = tick_id
        self._llm_call_counts[tick_id] = 0
        self._pending_llm_calls[tick_id] = []

    def install_generator(self, generator: Any) -> None:
        generator.llm_client = self.wrap_llm_client(generator.llm_client, "reply_pipeline")
        generator._production_replay_archive = self

    def wrap_llm_client(self, client: Any, role: str) -> Any:
        if client is None or isinstance(client, RecordingLLMClient):
            return client
        return RecordingLLMClient(client, self, role)

    def record_llm_call(
        self,
        role: str,
        client: Dict[str, Any],
        request: Dict[str, Any],
        response: Any = None,
        error: str = "",
        started_at: float = 0.0,
    ) -> None:
        tick_id = self._current_tick_id
        if tick_id is None or tick_id not in self._active:
            return
        with self._lock:
            sequence = self._llm_call_counts.get(tick_id, 0) + 1
            self._llm_call_counts[tick_id] = sequence
            self._pending_llm_calls.setdefault(tick_id, []).append({
                "sequence": sequence,
                "role": role,
                "client": client,
                "started_at_unix": started_at,
                "finished_at_unix": time.time(),
                "request": request,
                "response": response,
                "error": error,
            })

    def _flush_llm_calls(self, tick_id: int) -> None:
        with self._lock:
            calls = list(self._pending_llm_calls.get(tick_id, []))
        for call in calls:
            _atomic_gzip_json(
                self._active[tick_id] / "llm_calls" / f"{call['sequence']:03d}.json.gz",
                call,
            )
        with self._lock:
            self._pending_llm_calls.pop(tick_id, None)

    def capture_generator_pre_state(self, tick_id: int, generator: Any) -> None:
        registry = getattr(generator, "tool_registry", None)
        _atomic_gzip_json(
            self._active[tick_id] / "generator_before.json.gz",
            {
                "runtime_config": generator_runtime_config(generator),
                "llm_client": describe_llm_client(getattr(generator, "llm_client", None)),
                "fact_check_client": describe_llm_client(getattr(generator, "_fact_check_client", None))
                if getattr(generator, "_fact_check_client", None) is not None else None,
                "session_memory": snapshot_session_memory(generator),
                "tool_schemas": _jsonable(registry.to_openai_schemas()) if registry is not None else [],
            },
        )

    def capture_perception(self, tick_id: int, result: Any) -> None:
        tick_dir = self._active[tick_id]
        screenshot = Path(getattr(result, "screenshot_path", "") or "")
        screenshot_record: Dict[str, Any] = {"available": False}
        if screenshot.is_file():
            target = tick_dir / f"screenshot{screenshot.suffix or '.bin'}"
            shutil.copy2(screenshot, target)
            screenshot_record = {
                "available": True,
                "file": target.name,
                "sha256": _sha256(target),
            }
        _atomic_json(
            tick_dir / "perception.json",
            {"result": result, "screenshot": screenshot_record},
        )

    def capture_decision_point(
        self,
        tick_id: int,
        debug_info: Any,
        replies: List[str],
        generator: Any = None,
    ) -> None:
        tick_dir = self._active[tick_id]
        self._flush_llm_calls(tick_id)
        trace = getattr(debug_info, "reply_generation_trace", []) or []
        tool_calls = [entry for entry in trace if entry.get("type") == "tool_execution"]
        exact_generator_state = None
        if generator is not None:
            exact_generator_state = {
                "system_prompt": getattr(generator, "last_system_prompt", ""),
                "user_prompt": getattr(generator, "last_user_prompt", ""),
                "raw_response": getattr(generator, "last_raw_response", ""),
                "llm_messages": getattr(generator, "last_llm_messages", []),
                "few_shot_ids": getattr(generator, "last_few_shot_ids", []),
                "few_shot_content": getattr(generator, "last_few_shot_content", ""),
                "loaded_skills": getattr(generator, "last_loaded_skills", []),
                "skill_injected_content": getattr(generator, "last_skill_injected_content", ""),
                "tools_context": getattr(generator, "last_tools_context", ""),
                "feedback_raw": getattr(generator, "last_feedback_raw", ""),
                "iterate_raw": getattr(generator, "last_iterate_raw", ""),
                "session_memory_after": snapshot_session_memory(generator),
            }
        _atomic_gzip_json(
            tick_dir / "decision_point.json.gz",
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "cutoff_phase": "before_send_mark_replied_pending_memory_or_switch",
                "replies": replies,
                "debug_info": debug_info,
                "exact_generator_state": exact_generator_state,
                "tool_execution_count": len(tool_calls),
            },
        )
        tick_meta = json.loads((tick_dir / "tick.json").read_text(encoding="utf-8"))
        tick_meta["decision_point_captured"] = True
        tick_meta["llm_call_count"] = self._llm_call_counts.get(tick_id, 0)
        has_exact_capture = generator is None or tick_meta["llm_call_count"] > 0
        tick_meta["eligible_for_formal_replay"] = len(tool_calls) == 0 and has_exact_capture
        if tool_calls:
            tick_meta["ineligible_reason"] = "generation_executed_tools"
        elif not has_exact_capture:
            tick_meta["ineligible_reason"] = "missing_exact_llm_call_capture"
        else:
            tick_meta["ineligible_reason"] = ""
        _atomic_json(tick_dir / "tick.json", tick_meta)

    def finish_tick(self, tick_id: int, debug_info: Any) -> None:
        tick_dir = self._active.get(tick_id)
        if tick_dir is None:
            return
        if self._pending_llm_calls.get(tick_id):
            self._flush_llm_calls(tick_id)
        else:
            self._pending_llm_calls.pop(tick_id, None)
        self._active.pop(tick_id, None)
        if self._current_tick_id == tick_id:
            self._current_tick_id = None
        _atomic_json(tick_dir / "production_observation.json", debug_info)
        tick_meta = json.loads((tick_dir / "tick.json").read_text(encoding="utf-8"))
        tick_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        tick_meta["production_observation_sha256"] = _sha256(tick_dir / "production_observation.json")
        _atomic_json(tick_dir / "tick.json", tick_meta)
        record = {
            "tick_id": tick_id,
            "tick_file": str((tick_dir / "tick.json").relative_to(self.session_dir)),
            "tick_sha256": _sha256(tick_dir / "tick.json"),
        }
        with (self.session_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class DecisionPointCutoff(BaseException):
    """Fail-closed replay control flow that broad ``Exception`` handlers cannot swallow."""


class ReplaySideEffectSink:
    def __init__(self, output_path: Union[Path, str]):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def block(self, kind: str, payload: Dict[str, Any]) -> None:
        record = {"kind": kind, "payload": _jsonable(payload)}
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        raise DecisionPointCutoff(f"single-step replay stopped before {kind}")

    def guard_callable(self, kind: str, name: str, func: Callable) -> Callable:
        def guarded(**kwargs):
            self.block(kind, {"name": name, "arguments": kwargs})
        return guarded


class ReplaySender:
    def __init__(self, sink: ReplaySideEffectSink, wrapped: Any = None):
        self.sink = sink
        self.wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        if self.wrapped is None:
            raise AttributeError(name)
        return getattr(self.wrapped, name)

    def send(self, text: str, chat_name: str = "") -> None:
        self.sink.block("send_text", {"text": text, "chat_name": chat_name})

    def send_file(self, path: str, chat_name: str = "") -> None:
        self.sink.block("send_file", {"path": path, "chat_name": chat_name})


def guarded_tool_registry(registry: Any, sink: ReplaySideEffectSink) -> Any:
    from src.tools.tool_registry import ToolRegistry

    guarded = ToolRegistry()
    for tool in registry.list_tools():
        guarded.register(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            func=sink.guard_callable("tool_execution", tool.name, tool.func),
        )
    return guarded
