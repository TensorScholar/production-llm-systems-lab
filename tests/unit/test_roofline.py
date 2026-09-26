"""Unit tests for llmlab.calc.roofline."""

from __future__ import annotations

import itertools

import pytest

from llmlab.calc import roofline
from llmlab.models import ModelSpec

QWEN = ModelSpec(
    name="qwen2.5-7b-instruct",
    n_layers=28,
    n_q_heads=28,
    n_kv_heads=4,
    head_dim=128,
    hidden_size=3584,
    vocab_size=152064,
    tie_word_embeddings=True,
    attention="gqa",
    param_count=7_616_620_544,
)


def test_decode_weight_bytes_qwen() -> None:
    # Hand calculation: 7616620544*2 - 0 (tied) = 15233241088 B ≈ 14.19 GB.
    w = roofline.decode_weight_bytes(QWEN, 2)
    assert w == 7_616_620_544 * 2
    assert w / 1e9 == pytest.approx(15.233, rel=0.01)


def test_decode_weight_bytes_untied() -> None:
    spec = ModelSpec(
        name="u",
        n_layers=32,
        n_q_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        vocab_size=128256,
        tie_word_embeddings=False,
        attention="gqa",
        param_count=8_000_000_000,
    )
    w = roofline.decode_weight_bytes(spec, 2)
    assert w == 8_000_000_000 * 2 - 128256 * 4096 * 2


def test_tpot_floor_qwen_600gbs() -> None:
    w = roofline.decode_weight_bytes(QWEN, 2)
    m_kv = 2 * 28 * 4 * 128 * 2
    floor = roofline.tpot_floor_s(w, 2567.5 * m_kv, 6.0e11)
    # (15233241088 + 147215360) / 6e11 ≈ 25.6 ms (blueprint quotes ≈23.6 ms
    # from an illustrative 14.15 GB weight figure; this test uses the exact
    # hand calculation above).
    assert floor == pytest.approx(0.02563, rel=0.02)


def test_n1_tpot_is_tdec() -> None:
    cal = roofline.calibrate(0.030, 0.5, 14_000_000_000, 2567.5, 57344)
    p = roofline.predict_closed_loop(1, cal, 14_000_000_000, 2567.5, 57344, 2048, 1024, 2048)
    assert p.tpot_s == pytest.approx(0.030, rel=1e-6)


def test_tpot_monotone_in_n() -> None:
    cal = roofline.Calibration(bw_eff_Bps=5.0e11, t_prefill_s=0.4)
    tpots = [
        roofline.predict_closed_loop(n, cal, 14_000_000_000, 2567.5, 57344, 2048, 1024, 2048).tpot_s
        for n in (1, 2, 4, 8, 16)
    ]
    assert all(b >= a for a, b in itertools.pairwise(tpots))


def test_solver_converges_and_throughput() -> None:
    cal = roofline.Calibration(bw_eff_Bps=5.0e11, t_prefill_s=0.4)
    p = roofline.predict_closed_loop(4, cal, 14_000_000_000, 2567.5, 57344, 2048, 1024, 2048)
    assert p.req_per_s == pytest.approx(4 / p.lifetime_s, rel=1e-6)
    assert p.out_tok_per_s == pytest.approx(p.req_per_s * 1024, rel=1e-9)


def test_errors() -> None:
    with pytest.raises(ValueError):
        roofline.tpot_floor_s(1, 1.0, 0.0)
    with pytest.raises(ValueError):
        roofline.ttft_floor_s(1, 1, 0.0)
    assert roofline.ttft_floor_s(7_000_000_000, 2048, 1.25e14) == pytest.approx(
        2 * 7_000_000_000 * 2048 / 1.25e14
    )
