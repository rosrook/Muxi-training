#!/usr/bin/env python3
"""
Outer-layer queue client for multi-instance vLLM (teacher layout).

- temperature=0 always
- Fan-out ONLY across free instances (ports), NOT via vLLM continuous batch
- Each worker pulls next job when free (one in-flight per instance)
- Per-request timeout + limited retries for long-CoT failures
- After max retries: record failure, continue

Example:
  python3 eval_queue_client.py \\
    --base-url-template http://127.0.0.1:{port}/v1 \\
    --ports 8000-8007 \\
    --model qwen35-4b-nonfg-sft \\
    --input prompts.jsonl \\
    --output results.jsonl \\
    --timeout 300 --max-retries 2

prompts.jsonl line:
  {"id":"x1","messages":[{"role":"user","content":"..."}]}
optional: "max_tokens": 2048
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def parse_ports(spec: str) -> list[int]:
    # "8000-8007" or "8000,8001,8002"
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class Job:
    raw: dict[str, Any]
    attempt: int = 0
    last_error: str | None = None


@dataclass
class Stats:
    ok: int = 0
    fail: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def inc_ok(self) -> None:
        with self.lock:
            self.ok += 1

    def inc_fail(self) -> None:
        with self.lock:
            self.fail += 1


def worker(
    worker_id: int,
    base_url: str,
    model: str,
    jobs: "queue.Queue[Job | None]",
    out_path: Path,
    write_lock: threading.Lock,
    stats: Stats,
    timeout: float,
    max_retries: int,
    default_max_tokens: int,
) -> None:
    while True:
        job = jobs.get()
        if job is None:
            jobs.task_done()
            break
        jid = job.raw.get("id", f"row-{worker_id}")
        messages = job.raw.get("messages")
        max_tokens = int(job.raw.get("max_tokens", default_max_tokens))
        if not isinstance(messages, list):
            rec = {
                "id": jid,
                "ok": False,
                "error": "missing messages",
                "worker": worker_id,
                "base_url": base_url,
            }
            with write_lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.inc_fail()
            jobs.task_done()
            continue

        try:
            t0 = time.time()
            resp = chat_completion(
                base_url,
                model,
                messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            elapsed = time.time() - t0
            content = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            rec = {
                "id": jid,
                "ok": True,
                "content": content,
                "usage": resp.get("usage"),
                "elapsed_s": round(elapsed, 3),
                "attempt": job.attempt + 1,
                "worker": worker_id,
                "base_url": base_url,
                "raw_request": {"messages": messages, "max_tokens": max_tokens},
            }
            with write_lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.inc_ok()
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            job.attempt += 1
            job.last_error = err
            # timeout / connection: retry limited times
            if job.attempt <= max_retries:
                jobs.put(job)
            else:
                rec = {
                    "id": jid,
                    "ok": False,
                    "error": err,
                    "attempts": job.attempt,
                    "worker": worker_id,
                    "base_url": base_url,
                    "traceback": traceback.format_exc()[-2000:],
                }
                with write_lock:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.inc_fail()
        finally:
            jobs.task_done()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url-template", default="http://127.0.0.1:{port}/v1")
    ap.add_argument("--ports", default="8000-8007", help="8000-8007 or comma list")
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request HTTP timeout seconds")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    ports = parse_ports(args.ports)
    jobs: queue.Queue[Job | None] = queue.Queue()
    n_in = 0
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.put(Job(raw=json.loads(line)))
            n_in += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # truncate output for a clean run
    args.output.write_text("", encoding="utf-8")

    write_lock = threading.Lock()
    stats = Stats()
    threads: list[threading.Thread] = []
    for i, port in enumerate(ports):
        base = args.base_url_template.format(port=port)
        t = threading.Thread(
            target=worker,
            args=(
                i,
                base,
                args.model,
                jobs,
                args.output,
                write_lock,
                stats,
                args.timeout,
                args.max_retries,
                args.max_tokens,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # wait work complete then poison pills
    jobs.join()
    for _ in ports:
        jobs.put(None)
    for t in threads:
        t.join()

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "n_in": n_in,
                "ok": stats.ok,
                "fail": stats.fail,
                "workers": len(ports),
                "ports": ports,
                "temperature": 0,
                "timeout": args.timeout,
                "max_retries": args.max_retries,
            },
            indent=2,
        )
    )
    return 0 if stats.fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
