"""Unit tests for llmlab.metrics edge conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llmlab import metrics


def test_request_timing_basic() -> None:
    rt = metrics.request_timing(0, [100_000_000, 200_000_000, 300_000_000], 3)
    assert rt.ttft_s == pytest.approx(0.1)
    assert rt.e2e_s == pytest.approx(0.3)
    assert rt.tpot_s == pytest.approx(0.1)
    assert rt.itl_s.tolist() == pytest.approx([0.1, 0.1])


def test_one_output_token_tpot_none() -> None:
    rt = metrics.request_timing(0, [50_000_000], 1)
    assert rt.tpot_s is None
    assert rt.itl_s.size == 0


def test_zero_output_token_tpot_none() -> None:
    rt = metrics.request_timing(0, [50_000_000], 0)
    assert rt.tpot_s is None


def test_no_chunks_raises() -> None:
    with pytest.raises(ValueError):
        metrics.request_timing(0, [], 5)


def test_multi_token_flag() -> None:
    rt = metrics.request_timing(0, [100_000_000, 200_000_000], 10)
    assert rt.multi_token is True
    rt2 = metrics.request_timing(0, [100_000_000, 200_000_000], 2)
    assert rt2.multi_token is False


def test_gated_percentile_gate() -> None:
    vals = np.arange(10, dtype=float)
    g = metrics.gated_percentile(vals, 99, min_tail=20)
    assert g.value is None
    assert g.reason is not None
    g2 = metrics.gated_percentile(np.arange(3000, dtype=float), 99, min_tail=20)
    assert g2.value is not None


def test_gated_empty() -> None:
    g = metrics.gated_percentile(np.array([]), 50, min_tail=20)
    assert g.value is None


def test_bootstrap_coverage() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 1.0, size=500)
    lo, hi = metrics.bootstrap_ci(data, lambda a: float(np.mean(a)), n_boot=200, seed=1)
    assert lo <= 10.0 <= hi
    assert (hi - lo) < 0.3  # within ±3% nominal check scale


def test_summarize_run_basic() -> None:
    req = pd.DataFrame(
        [
            {
                "in_window": True,
                "status": "ok",
                "ttft_s": 0.5,
                "e2e_s": 2.5,
                "tpot_s": 0.02,
                "n_output_tokens": 101,
            }
            for _ in range(30)
        ]
    )
    server = pd.DataFrame(
        [
            {"t_ns": 0, "name": "kv_usage", "value": 0.5},
            {"t_ns": 1, "name": "running", "value": 4.0},
            {"t_ns": 1, "name": "waiting", "value": 0.0},
            {"t_ns": 1, "name": "preemptions", "value": 0.0},
            {"t_ns": 0, "name": "generation_tokens", "value": 0.0},
            {"t_ns": 1, "name": "generation_tokens", "value": 3000.0},
        ]
    )
    out = metrics.summarize_run(req, server, None, (0, 60_000_000_000), 80000, min_tail=1)
    assert out["n_completed"] == 30
    assert out["req_per_s"] == pytest.approx(0.5)
    assert out["kv_occupancy_tokens"] == pytest.approx(40000.0)
    assert out["ttft_p50"] == pytest.approx(0.5)
