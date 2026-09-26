"""Unit tests for llmlab.models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmlab.models import UnsupportedArchitecture, spec_from_hf_config

FIX = Path(__file__).parent.parent / "fixtures" / "hf_configs"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_qwen2_5_7b_gqa() -> None:
    spec = spec_from_hf_config(_load("qwen2.5-7b.json"), name="qwen2.5-7b-instruct")
    assert (spec.n_layers, spec.n_kv_heads, spec.head_dim) == (28, 4, 128)
    assert spec.attention == "gqa"


def test_qwen2_sliding_false_accepted() -> None:
    # sliding_window present but use_sliding_window=false => full attention.
    spec = spec_from_hf_config(_load("qwen2.5-7b.json"), name="q")
    assert spec.attention == "gqa"


def test_llama3_8b_shape() -> None:
    spec = spec_from_hf_config(_load("llama3-8b-shape.json"), name="llama3-8b")
    assert (spec.n_layers, spec.n_kv_heads, spec.head_dim) == (32, 8, 128)
    assert spec.attention == "gqa"


def test_mha_detected() -> None:
    spec = spec_from_hf_config(_load("mha-example.json"), name="mha")
    assert spec.attention == "mha"
    assert spec.n_kv_heads == spec.n_q_heads


def test_mla_rejected() -> None:
    with pytest.raises(UnsupportedArchitecture, match="MLA"):
        spec_from_hf_config(_load("deepseek-mla.json"), name="mla")


def test_falcon_rejected() -> None:
    with pytest.raises(UnsupportedArchitecture):
        spec_from_hf_config(_load("falcon-mqa.json"), name="falcon")


def test_active_sliding_rejected() -> None:
    with pytest.raises(UnsupportedArchitecture, match=r"[Ss]lid"):
        spec_from_hf_config(_load("qwen2-sliding-disabled.json"), name="sliding")
