"""Unit tests for vLLM startup-log parsing and CLI-arg building."""

from __future__ import annotations

import pytest

from llmlab.engines.vllm import (
    ParseError,
    build_cli_args,
    parse_startup_log,
)
from llmlab.schemas import ModelRef

MODEL = ModelRef(id="q", hf_repo="Qwen/Qwen2.5-7B-Instruct", revision="a" * 40)

LOG = """INFO: vLLM version: 0.6.3
INFO: GPU KV cache size: 80,000 tokens
INFO: Maximum concurrency for 3,072 tokens per request: 26.04x
INFO: Model loading took 14.20 GiB
"""


def test_parse_mock_style_log() -> None:
    rep = parse_startup_log(LOG, "vllm-startup-v1")
    assert rep.kv_cache_tokens == 80000
    assert rep.max_concurrency_tokens == 3072
    assert rep.max_concurrency == pytest.approx(26.04)
    assert rep.model_load_bytes == pytest.approx(14.20 * 1024**3)


def test_parse_missing_fields_none() -> None:
    rep = parse_startup_log("nothing here", "vllm-startup-v1")
    assert rep.kv_cache_tokens is None
    assert rep.max_concurrency is None


def test_unknown_parser() -> None:
    with pytest.raises(ParseError):
        parse_startup_log(LOG, "nope-v9")


def test_build_cli_args() -> None:
    args = build_cli_args(
        MODEL,
        {"seed": 0, "enable_prefix_caching": False, "max_num_seqs": 256, "block_size": 16},
    )
    assert "--seed" in args and "0" in args
    assert "--no-enable-prefix-caching" in args
    assert "--max-num-seqs" in args
    assert "--revision" in args and "a" * 40 in args


def test_build_cli_args_true_flag() -> None:
    args = build_cli_args(MODEL, {"ignore_eos": True})
    assert "--ignore-eos" in args
