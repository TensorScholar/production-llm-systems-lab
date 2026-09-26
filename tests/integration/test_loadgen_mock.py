"""Integration: load generator vs in-process mock (B0 thresholds)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from llmlab.loadgen import run_closed_loop
from llmlab.schemas import ClientConfig, WorkloadSpec
from llmlab.testing.mock_server import MockConfig, create_app
from llmlab.workload import request_stream, safe_token_ids


def _spec() -> WorkloadSpec:
    return WorkloadSpec(kind="fixed_shape", prompt_tokens=32, output_tokens=16, seed=7)


def _safe() -> np.ndarray:
    return safe_token_ids({"added_tokens": []}, 1000)


def _cfg(**kw: object) -> ClientConfig:  # type: ignore[no-untyped-def]
    base: dict[str, object] = {
        "warmup_s": 0.2,
        "stagger_s": 0.2,
        "window_min_s": 2.0,
        "window_max_s": 6.0,
        "min_completed": 20,
        "request_timeout_s": 30.0,
        "drain_timeout_s": 30.0,
    }
    base.update(kw)
    return ClientConfig(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_b0_accuracy_exact_match_n1() -> None:
    """Exact B0 check at N=1: client receipts vs server emission truth (same clock).

    Environment note: the B0-01 claim criterion (p99 <= 1 ms) is scoped to the
    reference setup and recorded in results/derived/B0 + the claims ledger.
    This Tier-0 gate verifies the exact-matching mechanism and timestamping
    fidelity (median sub-ms, p99 bounded by machine scheduler granularity —
    ~1-2 ms on macOS CI runners). Stable gates: median <= 1 ms, p90 <= 2 ms;
    p99 <= 10 ms guards against catastrophic tails. A broken harness
    (dropped/merged chunks, wrong window accounting) fails by 10-100x margin.
    """
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    from llmlab.metrics import harness_accuracy
    from llmlab.testing import mock_server as ms

    mock = MockConfig(capacity_tokens=4096, ttft_ms=20.0, itl_ms=5.0, seed=0)
    app = create_app(mock)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        base = str(client.make_url("/"))[:-1]
        spec = _spec()
        safe = _safe()
        from llmlab.workload import fixed_shape_request

        prompts = {
            f"req-{i:08d}": list(fixed_shape_request(spec, safe, i).prompt_token_ids)
            for i in range(200)
        }
        res = await run_closed_loop(
            request_stream(spec, safe), base, "mock-model", 1, _cfg(), "B0-exact"
        )
        ok_ids = {r.request_id for r in res.requests if r.in_window and r.status == "ok"}
        assert len(ok_ids) >= 20
        records = [
            {"request_id": r.request_id, "t_send_ns": r.t_send_ns}
            for r in res.requests
            if r.request_id in ok_ids
        ]
        chunks = [c for c in res.chunks if c["request_id"] in ok_ids]
        acc = harness_accuracy(records, chunks, prompts, app[ms._STATE_KEY].emitted, 16)
        assert acc["n_matched"] >= 20
        assert acc["ttft_median_err_s"] is not None and acc["ttft_median_err_s"] <= 0.001
        assert acc["itl_median_err_s"] is not None and acc["itl_median_err_s"] <= 0.001
        assert acc["ttft_p90_err_s"] is not None and acc["ttft_p90_err_s"] <= 0.002
        assert acc["itl_p90_err_s"] is not None and acc["itl_p90_err_s"] <= 0.002
        assert acc["ttft_p99_err_s"] is not None and acc["ttft_p99_err_s"] <= 0.010
        assert acc["itl_p99_err_s"] is not None and acc["itl_p99_err_s"] <= 0.010
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_window_accounting_and_errors() -> None:
    mock = MockConfig(capacity_tokens=4096, ttft_ms=2.0, itl_ms=1.0, error_rate=0.2, seed=1)
    client = TestClient(TestServer(create_app(mock)))
    await client.start_server()
    try:
        base = str(client.make_url("/"))[:-1]
        res = await run_closed_loop(
            request_stream(_spec(), _safe()), base, "mock-model", 4, _cfg(), "B0-err"
        )
        errs = [r for r in res.requests if r.status == "error"]
        assert len(errs) > 0
        assert all(r.error for r in errs)
        # Window flags consistent with final window.
        for r in res.requests:
            assert r.in_window == (res.window_start_ns <= r.t_send_ns <= res.window_end_ns)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_multi_token_flagged() -> None:
    mock = MockConfig(capacity_tokens=4096, ttft_ms=2.0, itl_ms=1.0, multi_token_every=1)
    client = TestClient(TestServer(create_app(mock)))
    await client.start_server()
    try:
        base = str(client.make_url("/"))[:-1]
        res = await run_closed_loop(
            request_stream(_spec(), _safe()), base, "mock-model", 2, _cfg(), "B0-multi"
        )
        ok = [r for r in res.requests if r.in_window and r.status == "ok"]
        assert ok and all(r.multi_token_chunks for r in ok)
    finally:
        await client.close()
