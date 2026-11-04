"""digital-human 草稿生成的逐阶段计时埋点。

与 data/dh-local-preparation-profile-20261102 的离线 profile 同一套标签口径，
在生产 draft() 上启用：monkeypatch DH 内部关键函数，按轮累计 wall/cpu 时间，
每轮 draft 结束输出一条 JSON 汇总日志（logger 名 dh_profile）。
不改 DH 行为，只读计时；install() 幂等。
"""

import importlib
import json
import logging
import os
import sys
import time
from threading import Lock

_logger = logging.getLogger(__name__)
_lock = Lock()
_metrics: dict[str, dict] = {}
_started = 0.0
_request_id = ""
_installed = False
_namespace = "digital_human"


def _module(name):
    return importlib.import_module(_namespace + "." + name)


def _measure(owner, name, label):
    original = getattr(owner, name)

    def call(*args, **kwargs):
        started, cpu = time.perf_counter(), time.process_time()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed, used = time.perf_counter() - started, time.process_time() - cpu
            with _lock:
                item = _metrics.setdefault(label, {"calls": 0, "seconds": 0.0,
                                                   "cpu_seconds": 0.0, "max_seconds": 0.0})
                item["calls"] += 1
                item["seconds"] += elapsed
                item["cpu_seconds"] += used
                item["max_seconds"] = max(item["max_seconds"], elapsed)

    setattr(owner, name, call)


def install(namespace="digital_human"):
    """在 DH runtime 加载后调用一次；重复调用无副作用。

    真实 runtime 未加载（如测试用替身）时跳过埋点。
    """
    global _installed, _namespace
    if _installed:
        return
    _namespace = namespace
    if namespace + ".generator.generator" not in sys.modules:
        _logger.info("dh_profile skipped: DH generator not loaded")
        _installed = True
        return
    generator = _module("generator.generator")
    selection = _module("generator.fewshot.learned_selection")
    session = _module("generator.fewshot.session")
    online = _module("data.online")
    sources = _module("data.history.sources")
    seals = _module("iteration.materials.seals")
    rank = _module("generator.fewshot.rank")
    wiki = _module("generator.context.wiki_tools")
    for owner, name, label in (
        (generator.ReplyGenerator, "build_prompt", "build_prompt"),
        (generator.ReplyGenerator, "_style_block", "style_block"),
        (generator.ReplyGenerator, "_check_sources", "generator_source_checks"),
        (generator.ReplyGenerator, "_check_request_sources", "request_source_checks"),
        (generator.ReplyGenerator, "_generate", "model_generate"),
        (selection, "recall", "recall"),
        (selection.LearnedSelector, "select", "learned_selection"),
        (session.RecallSession, "retrieve", "retrieve"),
        (session.RecallSession, "_candidate_rows", "candidate_rows"),
        (rank, "score_and_select", "score_and_select"),
        (online.OnlineInputs, "excluded_ids", "excluded_ids"),
        (seals.MaterialSeal, "check", "material_seal_check"),
        (sources.HistorySources, "check", "history_check"),
        (wiki.Background, "message", "wiki_message"),
        (wiki.Background, "generate", "wiki_generate"),
    ):
        if hasattr(owner, name):
            _measure(owner, name, label)
    _installed = True
    _logger.info("dh_profile installed")


def begin(request_id: str):
    global _started, _request_id, _metrics
    with _lock:
        _metrics = {}
        _request_id = request_id
        _started = time.perf_counter()


def finish(extra: dict | None = None):
    with _lock:
        summary = {"request_id": _request_id,
                   "total_seconds": round(time.perf_counter() - _started, 3),
                   "loadavg": [round(v, 1) for v in os.getloadavg()],
                   "stages": {label: {**item, "seconds": round(item["seconds"], 3),
                                      "cpu_seconds": round(item["cpu_seconds"], 3),
                                      "max_seconds": round(item["max_seconds"], 3)}
                              for label, item in sorted(_metrics.items())}}
    if extra:
        summary.update(extra)
    _logger.info("DH_PROFILE %s", json.dumps(summary, ensure_ascii=False))
