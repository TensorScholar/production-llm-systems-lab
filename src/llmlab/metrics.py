"""Canonical metric definitions shared by real and mock runs (book Ch. 4.1).

Definitions (also in docs/methodology/metrics.md):
- TTFT = first choice chunk − send.
- E2E = last chunk − send.
- TPOT = (E2E − TTFT)/(n_out − 1), n_out from reported usage; undefined if n_out < 2.
- ITL = gaps between consecutive chunks after the first, pooled event-level.
- Request throughput = in-window completions ÷ window length.
- Output-token throughput: client (Σ n_out) and server (generation-token counter delta).
- KV occupancy = time-mean of KV-usage gauge × C.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestTiming:
    ttft_s: float
    e2e_s: float
    tpot_s: float | None
    itl_s: npt.NDArray[np.float64]
    multi_token: bool


def request_timing(
    t_send_ns: int, chunk_t_ns: Sequence[int], n_output_tokens: int
) -> RequestTiming:
    """Per-request timing from send time, chunk times, and usage count."""
    if len(chunk_t_ns) == 0:
        raise ValueError("no chunks recorded")
    ttft = (chunk_t_ns[0] - t_send_ns) / 1e9
    e2e = (chunk_t_ns[-1] - t_send_ns) / 1e9
    itls = (
        np.diff(np.asarray(chunk_t_ns, dtype=np.float64)) / 1e9
        if len(chunk_t_ns) > 1
        else np.zeros(0)
    )
    tpot = (e2e - ttft) / (n_output_tokens - 1) if n_output_tokens >= 2 else None
    # multi_token heuristic: more output tokens than chunk events implies some
    # chunk carried multiple tokens.
    multi = n_output_tokens > len(chunk_t_ns)
    return RequestTiming(ttft_s=ttft, e2e_s=e2e, tpot_s=tpot, itl_s=itls, multi_token=multi)


@dataclass(frozen=True)
class Gated:
    value: float | None
    n: int
    reason: str | None


def gated_percentile(values: npt.NDArray[np.float64], q: float, min_tail: int) -> Gated:
    """Percentile gated on tail samples: n·(1−q/100) ≥ min_tail.

    Formula: gate = n·(1 − q/100) ≥ min_tail; else null with reason.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return Gated(value=None, n=0, reason="no finite samples")
    tail = n * (1.0 - q / 100.0)
    if tail < min_tail:
        return Gated(value=None, n=n, reason=f"only {tail:.1f} tail samples < min_tail {min_tail}")
    return Gated(value=float(np.percentile(arr, q)), n=n, reason=None)


def bootstrap_ci(
    values: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat`. Book §measurement method."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("no finite samples")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = stat(sample)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _col_mean(df: pd.DataFrame, name: str) -> float | None:
    if df.empty or name not in df.columns:
        return None
    vals = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def summarize_run(
    req: pd.DataFrame,
    server: pd.DataFrame,
    gpu: pd.DataFrame | None,
    window: tuple[int, int],
    capacity_tokens: int,
    min_tail: int = 20,
) -> dict[str, Any]:
    """Summarize one run into the canonical derived row.

    Formula set: throughput = completions/window; occupancy = mean(KV gauge)·C.
    """
    w0, w1 = window
    window_s = (w1 - w0) / 1e9 if w1 > w0 else float("nan")
    in_w = req[req["in_window"].astype(bool)] if not req.empty else req
    ok = in_w[in_w["status"] == "ok"] if not in_w.empty else in_w
    n_ok = len(ok)
    req_per_s = (n_ok / window_s) if window_s and window_s > 0 else None
    out_tok_client = float(ok["n_output_tokens"].sum()) if n_ok else 0.0
    out_tok_per_s = (out_tok_client / window_s) if window_s and window_s > 0 else None
    ttfts = ok["ttft_s"].to_numpy(dtype=np.float64) if n_ok else np.zeros(0)
    tpots = (
        pd.to_numeric(ok["tpot_s"], errors="coerce").to_numpy(dtype=np.float64)
        if n_ok
        else np.zeros(0)
    )
    e2es = ok["e2e_s"].to_numpy(dtype=np.float64) if n_ok else np.zeros(0)
    out: dict[str, Any] = {
        "n_completed": n_ok,
        "n_total": len(in_w),
        "window_s": window_s,
        "req_per_s": req_per_s,
        "out_tok_per_s_client": out_tok_per_s,
    }
    for label, arr in (("ttft", ttfts), ("tpot", tpots), ("e2e", e2es)):
        for q in (50, 90, 95, 99):
            g = gated_percentile(arr, float(q), min_tail)
            out[f"{label}_p{q}"] = g.value
            if g.reason is not None:
                out[f"{label}_p{q}_reason"] = g.reason
    # Server counters over the window.
    srv = server
    if not srv.empty and "t_ns" in srv.columns:
        srv = srv[(srv["t_ns"] >= w0) & (srv["t_ns"] <= w1)]
    kv_mean = _col_mean(
        srv[srv["name"] == "kv_usage"] if not srv.empty and "name" in srv.columns else srv,
        "value",
    )
    out["kv_occupancy_tokens"] = (kv_mean * capacity_tokens) if kv_mean is not None else None
    out["kv_usage_mean"] = kv_mean
    out["running_mean"] = _col_mean(
        srv[srv["name"] == "running"] if not srv.empty and "name" in srv.columns else srv,
        "value",
    )
    out["waiting_mean"] = _col_mean(
        srv[srv["name"] == "waiting"] if not srv.empty and "name" in srv.columns else srv,
        "value",
    )
    if not srv.empty and "name" in srv.columns:
        pre = srv[srv["name"] == "preemptions"]["value"]
        out["preemptions_delta"] = float(pre.max() - pre.min()) if len(pre) else None
        gen = srv[srv["name"] == "generation_tokens"]["value"]
        gen_delta = float(gen.max() - gen.min()) if len(gen) else None
        out["out_tok_per_s_server"] = (
            gen_delta / window_s if gen_delta is not None and window_s else None
        )
    else:
        out["preemptions_delta"] = None
        out["out_tok_per_s_server"] = None
    if gpu is not None and not gpu.empty:
        out["gpu_mem_used_mean"] = _col_mean(gpu, "mem_used_bytes")
        out["gpu_mem_used_max"] = (
            float(pd.to_numeric(gpu["mem_used_bytes"], errors="coerce").max())
            if "mem_used_bytes" in gpu.columns
            else None
        )
        out["gpu_sm_clock_min"] = (
            float(pd.to_numeric(gpu["sm_clock_mhz"], errors="coerce").min())
            if "sm_clock_mhz" in gpu.columns
            else None
        )
        out["gpu_power_mean"] = _col_mean(gpu, "power_w")
        out["gpu_temp_max"] = (
            float(pd.to_numeric(gpu["temp_c"], errors="coerce").max())
            if "temp_c" in gpu.columns
            else None
        )
    else:
        out["gpu_mem_used_mean"] = None
        out["gpu_mem_used_max"] = None
        out["gpu_sm_clock_min"] = None
        out["gpu_power_mean"] = None
        out["gpu_temp_max"] = None
    return out


def harness_accuracy(
    records: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    prompt_ids: dict[str, list[int]],
    emitted: dict[str, list[int]],
    output_tokens: int,
) -> dict[str, Any]:
    """B0 harness accuracy vs mock ground truth (same clock, in-process).

    For each request, match client chunk receipts to server emission times
    via the prompt hash key. TTFT error = |first receipt − first emission|;
    ITL error = per-gap |(client gap) − (server gap)| pooled across requests.
    Transit time is excluded by construction: both stamps use the same
    perf_counter clock, so the residual is pure harness timestamping error.
    Formula: p99(|err|) ≤ 1 ms (claim B0-01).
    """
    from llmlab.testing.mock_server import prompt_key

    ttft_errs: list[float] = []
    itl_errs: list[float] = []
    n_matched = 0
    chunks_by_req: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        chunks_by_req.setdefault(c["request_id"], []).append(c)
    send_by_req = {r["request_id"]: r["t_send_ns"] for r in records}
    for req_id, prompt in prompt_ids.items():
        key = prompt_key(prompt, output_tokens)
        truth = emitted.get(key)
        mine = sorted(chunks_by_req.get(req_id, []), key=lambda c: c["seq"])
        if not truth or not mine or req_id not in send_by_req:
            continue
        t_send = send_by_req[req_id]
        client_abs = [t_send + c["t_ns"] for c in mine]
        # First-chunk (TTFT) error; truncate to common length.
        m = min(len(truth), len(client_abs))
        if m == 0:
            continue
        n_matched += 1
        ttft_errs.append(abs(client_abs[0] - truth[0]) / 1e9)
        for i in range(1, m):
            itl_errs.append(
                abs((client_abs[i] - client_abs[i - 1]) - (truth[i] - truth[i - 1])) / 1e9
            )

    def _pct(xs: list[float], q: float) -> float | None:
        return float(np.percentile(np.asarray(xs), q)) if xs else None

    return {
        "n_matched": n_matched,
        "n_ttft": len(ttft_errs),
        "n_itl": len(itl_errs),
        "ttft_median_err_s": _pct(ttft_errs, 50),
        "itl_median_err_s": _pct(itl_errs, 50),
        "ttft_p90_err_s": _pct(ttft_errs, 90),
        "itl_p90_err_s": _pct(itl_errs, 90),
        "ttft_p99_err_s": _pct(ttft_errs, 99),
        "itl_p99_err_s": _pct(itl_errs, 99),
        "ttft_max_err_s": max(ttft_errs) if ttft_errs else None,
        "itl_max_err_s": max(itl_errs) if itl_errs else None,
    }
