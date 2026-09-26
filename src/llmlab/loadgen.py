"""Closed-loop streaming client: exactly N requests in flight (book §load method).

Workers run send → stream → record → repeat. Staggered starts prevent
lock-step; a warmup + window/min-completed/max window rule defines the
measurement window; in-flight requests drain (timeout if too slow).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import psutil

from llmlab.metrics import request_timing
from llmlab.schemas import ClientConfig, RequestRecord
from llmlab.workload import Request

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    requests: list[RequestRecord] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    window_start_ns: int = 0
    window_end_ns: int = 0
    loop_lag_p99_ms: float = 0.0
    client_cpu_max: float = 0.0


async def _read_sse(
    resp: aiohttp.ClientResponse,
    run_id: str,
    request_id: str,
    worker_id: int,
    t_send_ns: int,
    timeout_s: float,
) -> tuple[list[int], list[dict[str, Any]], int, int, str | None]:
    """Read SSE stream until [DONE]; return chunk times, chunk rows, n_prompt, n_out, error."""
    chunk_times: list[int] = []
    rows: list[dict[str, Any]] = []
    n_prompt = 0
    n_out = 0
    error: str | None = None
    seq = 0
    buf = b""
    try:
        async for data, _ in resp.content.iter_chunks():
            if data is None:
                break
            buf += data
            while b"\n\n" in buf or b"\n" in buf:
                # Split on newlines; SSE events are data: lines.
                if b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                else:
                    event, buf = buf.split(b"\n", 1)
                for raw in event.split(b"\n"):
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[len(b"data:") :].strip()
                    if payload == b"[DONE]":
                        return chunk_times, rows, n_prompt, n_out, error
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        error = "bad-sse-json"
                        continue
                    if isinstance(obj, dict) and "usage" in obj:
                        usage = obj["usage"] or {}
                        n_prompt = int(usage.get("prompt_tokens", n_prompt))
                        n_out = int(usage.get("completion_tokens", n_out))
                        continue
                    choices = obj.get("choices", []) if isinstance(obj, dict) else []
                    if choices:
                        now = time.perf_counter_ns()
                        chunk_times.append(now)
                        text = choices[0].get("text", "") or ""
                        rows.append(
                            {
                                "request_id": request_id,
                                "seq": seq,
                                "t_ns": now - t_send_ns,
                                "text_len": len(text),
                            }
                        )
                        seq += 1
    except TimeoutError:
        error = "timeout"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return chunk_times, rows, n_prompt, n_out, error


async def _worker(
    worker_id: int,
    requests: Iterator[Request],
    base_url: str,
    model_name: str,
    cfg: ClientConfig,
    run_id: str,
    session: aiohttp.ClientSession,
    out_records: list[RequestRecord],
    out_chunks: list[dict[str, Any]],
    state: dict[str, Any],
    stagger_delay: float,
    lock: asyncio.Lock,
) -> None:
    await asyncio.sleep(stagger_delay)
    while True:
        async with lock:
            if state["window_closed"]:
                return
            try:
                req = next(requests)
            except StopIteration:
                state["window_closed"] = True
                return
        t_send = time.perf_counter_ns()
        status = "ok"
        err: str | None = None
        chunk_times: list[int] = []
        rows: list[dict[str, Any]] = []
        n_prompt = 0
        n_out = 0
        try:
            timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
            async with session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": list(req.prompt_token_ids),
                    "max_tokens": req.max_tokens,
                    "min_tokens": req.max_tokens,
                    "ignore_eos": True,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    status = "error"
                    err = f"http-{resp.status}"
                else:
                    chunk_times, rows, n_prompt, n_out, err = await _read_sse(
                        resp,
                        run_id,
                        req.request_id,
                        worker_id,
                        t_send,
                        cfg.request_timeout_s,
                    )
                    if err is not None and not chunk_times:
                        status = "error"
        except TimeoutError:
            status = "timeout"
            err = "timeout"
        except Exception as e:
            status = "error"
            err = f"{type(e).__name__}: {e}"
        t_last = chunk_times[-1] if chunk_times else None
        in_window = state["window_start_ns"] <= t_send <= state["window_end_ns"]
        ttft: float | None
        e2e: float | None
        tpot: float | None
        if chunk_times and n_out:
            timing = request_timing(t_send, chunk_times, n_out)
            ttft = timing.ttft_s
            e2e = timing.e2e_s
            tpot = timing.tpot_s
            multi = timing.multi_token or (n_out > len(chunk_times))
        else:
            ttft = e2e = tpot = None
            multi = False
        rec = RequestRecord(
            run_id=run_id,
            request_id=req.request_id,
            worker_id=worker_id,
            t_send_ns=t_send,
            t_first_ns=chunk_times[0] if chunk_times else None,
            t_last_ns=t_last,
            n_chunks=len(chunk_times),
            n_prompt_tokens=n_prompt,
            n_output_tokens=n_out,
            status=status,
            error=err,
            in_window=in_window,
            ttft_s=ttft,
            e2e_s=e2e,
            tpot_s=tpot,
            multi_token_chunks=multi,
        )
        async with lock:
            out_records.append(rec)
            out_chunks.extend(rows)
            if in_window and status == "ok":
                state["completed_in_window"] += 1
            # Check window close: both window_min_s and min_completed, or max.
            now = time.perf_counter_ns()
            elapsed = (now - state["window_start_ns"]) / 1e9
            if not state["window_closed"] and (
                elapsed >= cfg.window_max_s
                or (
                    elapsed >= cfg.window_min_s
                    and state["completed_in_window"] >= cfg.min_completed
                )
            ):
                state["window_closed"] = True
                state["window_end_ns"] = now


async def _lag_monitor(stop: asyncio.Event, lags_ms: list[float]) -> None:
    while not stop.is_set():
        t0 = time.perf_counter_ns()
        await asyncio.sleep(0.01)
        lags_ms.append((time.perf_counter_ns() - t0) / 1e6 - 10.0)


def _cpu_sampler(stop: threading.Event, out: list[float]) -> None:
    """Sample this process's CPU as a fraction of total machine capacity."""
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # initialize baseline
    try:
        ncpu = psutil.cpu_count(logical=True) or 1
    except Exception:
        ncpu = 1
    while not stop.is_set():
        stop.wait(0.2)
        # Process percent is relative to one core; normalize to machine (0..1).
        out.append(min(proc.cpu_percent(interval=None) / 100.0 / ncpu, 1.0))


async def run_closed_loop(
    requests: Iterator[Request],
    base_url: str,
    model_name: str,
    concurrency: int,
    cfg: ClientConfig,
    run_id: str,
    lifetime_estimate_s: float | None = None,
) -> LoadResult:
    """Run a closed-loop sweep point: N workers, send→stream→repeat.

    Formula: throughput λ = N/L; stagger spreads starts over stagger_s
    (auto = pre-run lifetime estimate) to avoid lock-step.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    # Warmup gate: window opens after warmup_s.
    stagger = cfg.stagger_s
    if stagger == "auto":
        stagger = lifetime_estimate_s or 5.0
    # Pre-window: send warmup traffic? The runner sends explicit warm-ups;
    # here warmup_s delays the window start while workers already run.
    result = LoadResult()
    out_records: list[RequestRecord] = []
    out_chunks: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    t_start = time.perf_counter_ns()
    window_start = t_start + int(cfg.warmup_s * 1e9)
    state: dict[str, Any] = {
        "window_closed": False,
        "window_start_ns": window_start,
        "window_end_ns": window_start,
        "completed_in_window": 0,
    }
    lags_ms: list[float] = []
    cpu_vals: list[float] = []
    stop_lag = asyncio.Event()
    stop_cpu = threading.Event()
    lag_task = asyncio.create_task(_lag_monitor(stop_lag, lags_ms))
    cpu_thread = threading.Thread(target=_cpu_sampler, args=(stop_cpu, cpu_vals), daemon=True)
    cpu_thread.start()
    # Stagger delays spread over stagger_s.
    import random

    rng = random.Random(0)
    staggers = [rng.uniform(0, float(stagger)) for _ in range(concurrency)]
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    _worker(
                        w,
                        requests,
                        base_url,
                        model_name,
                        cfg,
                        run_id,
                        session,
                        out_records,
                        out_chunks,
                        state,
                        staggers[w],
                        lock,
                    )
                )
                for w in range(concurrency)
            ]
            # Workers run through warmup (pre-window) into the window, then
            # stop sending once it closes; drain in-flight with a timeout.
            try:
                budget = cfg.warmup_s + cfg.window_max_s + cfg.drain_timeout_s + 60.0
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=budget)
            except TimeoutError:
                for t in tasks:
                    t.cancel()
                # Mark unfinished worker iterations as timeouts is handled
                # per-request; here we just stop waiting.
            if state["window_end_ns"] == window_start:
                state["window_end_ns"] = time.perf_counter_ns()
    finally:
        stop_lag.set()
        await asyncio.sleep(0)
        lag_task.cancel()
        stop_cpu.set()
        cpu_thread.join(timeout=2.0)
    # Mark timeouts: requests still in flight at drain end are already
    # recorded; anything sent but unfinished past drain gets timeout status
    # via the per-request timeout above.
    result.requests = out_records
    result.chunks = out_chunks
    result.window_start_ns = window_start
    result.window_end_ns = int(state["window_end_ns"])
    if lags_ms:
        import numpy as np

        result.loop_lag_p99_ms = float(np.percentile(np.asarray(lags_ms), 99))
    result.client_cpu_max = max(cpu_vals) if cpu_vals else 0.0
    # Fix in_window flags now that the final window is known.
    for rec in result.requests:
        rec.in_window = result.window_start_ns <= rec.t_send_ns <= result.window_end_ns
    return result
