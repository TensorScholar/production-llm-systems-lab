"""Unit tests for llmlab.calc.kv (incl. Chapter 4 golden values)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llmlab.calc import kv
from llmlab.models import ModelSpec

GOLDEN = yaml.safe_load(
    (Path(__file__).parent.parent / "golden" / "ch4_examples.yaml").read_text(encoding="utf-8")
)

LLAMA_SHAPE = ModelSpec(
    name="llama3-8b-shape",
    n_layers=32,
    n_q_heads=32,
    n_kv_heads=8,
    head_dim=128,
    hidden_size=4096,
    vocab_size=128256,
    tie_word_embeddings=False,
    attention="gqa",
)


def test_m_kv_ch4() -> None:
    assert kv.kv_bytes_per_token(LLAMA_SHAPE, 2) == GOLDEN["m_kv_bytes_per_token"]


def test_bytes_per_sequence() -> None:
    m_kv: int = GOLDEN["m_kv_bytes_per_token"]
    assert 4096 * m_kv == GOLDEN["bytes_per_sequence_T4096_MiB"] * 1024**2


def test_batch_gib() -> None:
    m_kv: int = GOLDEN["m_kv_bytes_per_token"]
    for row in GOLDEN["batch_gib"]:
        gib = row["tokens"] * m_kv / 1024**3
        assert gib == pytest.approx(row["gib"], abs=0.001)


def test_free_slots() -> None:
    fs = GOLDEN["free_slots"]
    assert fs["total"] - fs["used"] == fs["free"] == 56_128


def test_free_slots_after_reserve() -> None:
    fs = GOLDEN["free_slots"]
    m_kv: int = GOLDEN["m_kv_bytes_per_token"]
    reserve_tokens = (3 * 1024**3) // m_kv
    assert fs["free"] - reserve_tokens == GOLDEN["free_slots_after_3gib_reserve"] == 31_552


def test_exercise_values() -> None:
    ex = GOLDEN["exercise"]
    assert 16384 * ex["m_kv_bytes_per_token"] == pytest.approx(ex["gib_for_16384_tokens"] * 1024**3)


def test_slack_values() -> None:
    for t_str, expected in GOLDEN["slack_q16"].items():
        assert kv.slack_tokens(int(t_str), 16) == expected


def test_g_decode_ch4() -> None:
    # Blueprint quotes 2567.5; the exact rational is 2626560/1023 ≈ 2567.507.
    # See docs/methodology/kv-accounting.md for the derivation note.
    assert kv.g_decode(2048, 1024, 16) == pytest.approx(
        GOLDEN["g_decode_P2048_O1024_q16"], abs=GOLDEN["g_decode_tolerance"]
    )
    assert kv.g_decode(2048, 1024, 16) == pytest.approx(2626560 / 1023, rel=1e-12)


def test_g_max() -> None:
    assert kv.g_max(2048, 1024, 16) == 3072


def test_pool_capacity_round_trip() -> None:
    m_kv: int = GOLDEN["m_kv_bytes_per_token"]
    cap = kv.pool_capacity(24 * 1024**3, m_kv, 16)
    assert cap.tokens == 196_608
    assert kv.pool_bytes(cap.num_blocks, 16, m_kv) <= 24 * 1024**3


def test_blocks_and_errors() -> None:
    assert kv.blocks_for_tokens(0, 16) == 0
    assert kv.blocks_for_tokens(17, 16) == 2
    with pytest.raises(ValueError):
        kv.blocks_for_tokens(-1, 16)
    with pytest.raises(ValueError):
        kv.slack_tokens(5, 0)
    with pytest.raises(ValueError):
        kv.max_concurrency_ratio(100, 0)


def test_saturation_interval_ordering() -> None:
    lo, hi = kv.predicted_saturation_interval(80000, 2567.5, 3072)
    assert lo < hi
    assert kv.predicted_occupancy(4, 2567.5, 80000) == pytest.approx(4 * 2567.5)
    assert kv.predicted_occupancy(1000, 2567.5, 80000) == 80000
