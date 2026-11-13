"""Bounded draft-only replay of recorded DH prompts with selected examples.

This diagnoses rendering only; it does not rerun retrieval or adopt a generator.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def shorten_examples(text, keep):
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    cursor, pieces, counts = 0, [], []
    while True:
        start = text.find('<example id="', cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        end = text.index("</example>", start) + len("</example>")
        block = text[start:end]
        example = ET.fromstring(block)
        context = example.find("context")
        if context is None or any(child.tag != "message" for child in context):
            raise ValueError("unexpected example context")
        counts.append({"id": example.attrib["id"], "before": len(context),
                       "after": min(keep, len(context))})
        if len(context) > keep:
            left = block.index("<context>") + len("<context>")
            right = block.index("</context>", left)
            content = block[left:right]
            for _ in range(len(context) - keep):
                content = content[content.index("</message>") + len("</message>"):]
            block = block[:left] + content + block[right:]
        pieces.extend((text[cursor:start], block))
        cursor = end
    return "".join(pieces), counts


def replay_body(trace):
    operations = [op for op in trace.get("operations", []) if op["kind"] == "generate"]
    events = [event for event in trace.get("events", []) if event["kind"] == "llm"]
    events += [event for op in operations for event in op["events"] if event["kind"] == "llm"]
    if len(events) != 1 or events[0]["status"] != "ok":
        raise ValueError("requires one successful model request, without tool follow-ups")
    return deepcopy(events[0]["request"])


def replace_examples(request, rendered):
    candidate = deepcopy(request)
    root = ET.fromstring(rendered)
    blocks = root.findall("example")
    if len(blocks) > 3:
        raise ValueError("at most three selected examples")
    for block in blocks:
        block.tail = None
    replacement = "\n".join(ET.tostring(block, encoding="unicode") for block in blocks)
    changed = 0
    for message in candidate["body"]["messages"]:
        text = message["content"]
        if '<example id="' not in text:
            continue
        left = text.index('<example id="')
        right = text.rindex("</example>") + len("</example>")
        if len(ET.fromstring("<root>" + text[left:right] + "</root>")) != 3:
            raise ValueError("requires exactly three original examples")
        message["content"] = text[:left] + replacement + text[right:]
        changed += 1
    if changed != 1:
        raise ValueError("requires one example-bearing message")
    return candidate


def select_excerpts(pool, selections):
    wanted = {item["id"] for item in selections}
    sources = {}
    with pool.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in wanted:
                sources[row["id"]] = (row, hashlib.sha256(line.encode()).hexdigest())
            if len(sources) == len(wanted):
                break
    rows, evidence = [], []
    for selection in selections:
        source, line_hash = sources[selection["id"]]
        if source["content_sha256"] != selection["content_sha256"]:
            raise ValueError("selected source changed")
        indexes = [selection["context_indices"], selection["reply_indices"]]
        for indices, values in zip(indexes, [source["context_messages"], source["reply"]]):
            if (not indices or any(type(i) is not int or i < 0 or i >= len(values) for i in indices)
                    or indices != sorted(set(indices))):
                raise ValueError("invalid excerpt indices")
            if indices != list(range(len(values))):
                raise ValueError("selected examples must retain complete source context and replies")
        if source["relationship"] != "private":
            raise ValueError("sender aliases require a private conversation")
        context = [{"sender": "本人" if source["context_messages"][i]["is_self"] else "对方",
                    "is_self": source["context_messages"][i]["is_self"],
                    "text": source["context_messages"][i]["text"]} for i in indexes[0]]
        reply = [source["reply"][i] for i in indexes[1]]
        rows.append({"id": "manual-excerpt-" + source["id"],
                     "context_messages": context, "reply": reply})
        evidence.append({**selection, "source_line_sha256": line_hash,
                         "source_provenance": source["source_provenance"],
                         "source_order_verified": source["source_span"]["order_verified"],
                         "derivation": "complete_example_with_sender_aliases",
                         "context_messages": context, "reply": reply})
    return rows, evidence


def worker(payload):
    repository = Path(payload["repository"])
    sys.path.insert(0, str(repository))
    from src.infrastructure.config import load_settings
    from src.iteration import versions
    from src.generator.generator import ReplyGenerator
    from src.llm import ChatClient
    if payload.get("render_only"):
        from src.generator.fewshot.session import RecallSession
        from src.generator.context.identity import NamedIdentity
        rows = payload["rows"]
        rendered, ids = RecallSession.render(rows, max_chars=5000)
        if ids != [row["id"] for row in rows]:
            raise ValueError("selected examples exceed rendering budget")
        if not rows:
            return {"rendered": "<style_examples/>"}
        rendered = NamedIdentity("王芊").examples(rendered, rows, ids)
        root = ET.fromstring(rendered)
        for example in root.findall("example"):
            example.set("derivation", "complete_example_with_sender_aliases")
        return {"rendered": ET.tostring(root, encoding="unicode")}
    versions.switch_instance(payload["instance"])
    settings = load_settings()
    settings["llm"]["max_retries"] = 0
    request = payload["request"]
    body = request["body"]
    config = {**payload["client_config"], **{key: body[key] for key in
              ("model", "temperature", "max_tokens", "thinking", "output_config") if key in body},
              "protocol": request["protocol"], "timeout_seconds": payload["timeout"]}
    messages = body["messages"]
    if "system" in body:
        messages = [{"role": "system", "content": body["system"]}, *messages]
    started = time.monotonic()
    raw = ChatClient(settings, config).chat(messages, json_mode="response_format" in body)
    elapsed = round(time.monotonic() - started, 3)
    try:
        parsed = ReplyGenerator._validate(ReplyGenerator._parse(raw), False)
    except Exception as exc:
        return {"status": "parse_error", "raw": raw, "error": type(exc).__name__, "chat_elapsed_seconds": elapsed}
    if parsed is None:
        return {"status": "invalid_output", "raw": raw, "chat_elapsed_seconds": elapsed}
    return {"status": "ok", "replies": parsed["replies"], "raw": raw, "chat_elapsed_seconds": elapsed}


def run_selected(args):
    if args.arm != "candidate" or not 0 < args.timeout <= 20 or not 0 < args.total_seconds <= 60:
        raise ValueError("selection replay requires candidate only, <=20s each and <=60s total")
    plan = json.loads(args.selection.read_text())
    if not 1 <= len(plan["cases"]) <= 2:
        raise ValueError("at most two selected-example candidates")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(args.selection.read_text())
    deadline = time.monotonic() + args.total_seconds
    for index, case in enumerate(plan["cases"]):
        trace_path = Path(case["trace"])
        trace = json.loads(trace_path.read_text())
        request = replay_body(trace)
        binding = trace["result"]["binding"]
        config_path = args.repository / "instances" / binding["instance"] / "generators" / binding["generator"] / "config.json"
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != binding["config_sha256"]:
            raise ValueError("baseline generator config changed")
        pool = args.repository / "instances" / binding["instance"] / "data" / binding["data"] / "fewshot_pool.jsonl"
        rows, excerpts = select_excerpts(pool, case["examples"])
        base_payload = {"repository": str(args.repository), "instance": binding["instance"]}
        command = [str(args.repository / ".venv/bin/python"), str(Path(__file__).resolve()), "--worker"]
        prepared = subprocess.run(command, input=json.dumps({**base_payload, "render_only": True, "rows": rows}),
                                  text=True, capture_output=True, timeout=min(10, max(.01, deadline - time.monotonic())), cwd=args.repository)
        candidate = replace_examples(request, json.loads(prepared.stdout)["rendered"])
        event = next(e for e in trace["events"] if e["kind"] == "llm")
        evidence = {"trace": str(trace_path), "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    "case_id": trace["case_id"], "binding": binding, "excerpts": excerpts,
                    "diagnostic_only": True, "adopted": False, "sent": False,
                    "limits": "Complete source examples, manually selected; source ordering is not certified. Frozen history, prompt and model. No retrieval or judge. Historical baseline has uncontrolled sampling/cache timing.",
                    "requests": {"baseline": request, "candidate": candidate},
                    "results": {"baseline": {"reused": True, "status": trace["status"],
                        "raw": event["response"]["text"], "replies": trace["result"]["replies"],
                        "elapsed_seconds": event["elapsed_ms"] / 1000}}}
        output = args.output / f"{index:02}.json"
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        limit = min(args.timeout, deadline - time.monotonic())
        tick = time.monotonic()
        if limit <= 0:
            result = {"status": "budget_exhausted"}
        else:
            payload = {**base_payload, "request": candidate, "client_config": config["llm"]["private"], "timeout": limit}
            try:
                process = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True,
                                         timeout=limit, cwd=args.repository)
                result = (json.loads(process.stdout) if process.returncode == 0
                          else {"status": "error", "error": "worker_failed"})
            except subprocess.TimeoutExpired:
                result = {"status": "timeout"}
        result["elapsed_seconds"] = round(time.monotonic() - tick, 3)
        evidence["results"]["candidate"] = result
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        print(json.dumps({"case_id": trace["case_id"], **result}, ensure_ascii=False), flush=True)


def run(args):
    if (not all(math.isfinite(n) and n > 0 for n in (args.timeout, args.total_seconds))
            or len(args.trace) > 3):
        raise ValueError("positive finite deadlines and at most three traces required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.total_seconds
    for index, path in enumerate(args.trace):
        trace = json.loads(path.read_text())
        request = replay_body(trace)
        candidate = deepcopy(request)
        counts = []
        for message in candidate["body"]["messages"]:
            message["content"], selected = shorten_examples(message["content"], args.keep_context)
            counts.extend(selected)
        if len(counts) != 3 or request == candidate:
            raise ValueError("requires exactly three examples and a changed context")
        binding = trace["operations"][0]["input"]["binding"]
        config_path = (args.repository / "instances" / binding["instance"] / "generators"
                       / binding["generator"] / "config.json")
        config = json.loads(config_path.read_text())
        client_config = config["llm"][trace["case"]["request"]["input"]["chat_type"]]
        evidence = {"trace": str(path), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "case_id": trace["case_id"], "binding": binding, "contexts": counts,
                    "diagnostic_only": True, "adopted": False, "sent": False,
                    "limits": "Frozen prompt replay; no retrieval, tools, judge or production adoption.",
                    "requests": {"baseline": request, "candidate": candidate}, "results": {}}
        for arm, body in evidence["requests"].items():
            if args.arm != "both" and arm != args.arm:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = {"status": "budget_exhausted"}
            else:
                limit = min(args.timeout, remaining)
                payload = {"repository": str(args.repository), "request": body,
                           "client_config": client_config, "instance": binding["instance"], "timeout": limit}
                tick = time.monotonic()
                try:
                    process = subprocess.run([str(args.repository / ".venv/bin/python"),
                        str(Path(__file__).resolve()), "--worker"], input=json.dumps(payload),
                        text=True, capture_output=True, timeout=limit, cwd=args.repository)
                    result = (json.loads(process.stdout) if process.returncode == 0
                              else {"status": "error", "error": "worker_failed"})
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout"}
                result["elapsed_seconds"] = round(time.monotonic() - tick, 3)
            evidence["results"][arm] = result
            (args.output / f"{index:02}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
            print(json.dumps({"case_id": trace["case_id"], "arm": arm, **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        try:
            print(json.dumps(worker(json.load(sys.stdin)), ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"status": "error", "error": type(exc).__name__, "detail": str(exc)}))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--repository", type=Path, required=True)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--trace", type=Path, action="append")
        source.add_argument("--selection", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--keep-context", type=int, default=3)
        parser.add_argument("--arm", choices=("both", "baseline", "candidate"), default="both")
        parser.add_argument("--timeout", type=float, default=20)
        parser.add_argument("--total-seconds", type=float, default=85)
        args = parser.parse_args()
        if args.selection:
            run_selected(args)
        else:
            run(args)
