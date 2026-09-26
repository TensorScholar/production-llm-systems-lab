"""Property tests for llmlab.calc.kv."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from llmlab.calc import kv
from llmlab.models import ModelSpec


def _spec(layers: int, kv_heads: int, head_dim: int) -> ModelSpec:
    return ModelSpec(
        name="prop",
        n_layers=layers,
        n_q_heads=max(kv_heads, 1),
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=max(kv_heads, 1) * head_dim,
        vocab_size=32000,
        tie_word_embeddings=False,
        attention="gqa",
    )


@given(st.integers(1, 64), st.integers(1, 16), st.integers(8, 256))
def test_kv_bytes_monotone(layers: int, kv_heads: int, head_dim: int) -> None:
    base = kv.kv_bytes_per_token(_spec(layers, kv_heads, head_dim), 2)
    assert kv.kv_bytes_per_token(_spec(layers + 1, kv_heads, head_dim), 2) > base
    assert kv.kv_bytes_per_token(_spec(layers, kv_heads + 1, head_dim), 2) > base
    assert kv.kv_bytes_per_token(_spec(layers, kv_heads, head_dim + 1), 2) > base


@given(st.integers(0, 10000), st.integers(1, 64))
def test_slack_bounds(tokens: int, q: int) -> None:
    assert 0 <= kv.slack_tokens(tokens, q) <= q - 1
    assert (tokens + kv.slack_tokens(tokens, q)) % q == 0


@given(st.integers(0, 4096), st.integers(2, 512), st.integers(1, 64))
def test_g_decode_range(prompt: int, output: int, q: int) -> None:
    g = kv.g_decode(prompt, output, q)
    assert prompt <= g <= kv.g_max(prompt, output, q)
