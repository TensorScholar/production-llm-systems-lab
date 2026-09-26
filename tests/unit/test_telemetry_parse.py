"""Unit tests for llmlab.telemetry parsing (mock fixtures; T13 adds real ones)."""

from __future__ import annotations

import sys
import types

import pytest

from llmlab.schemas import MetricsMap
from llmlab.telemetry import (
    NvmlSampler,
    parse_cache_config_info,
    parse_metrics_text,
)

MOCK_METRICS = """# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.42
# HELP vllm:num_requests_running Running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 8.0
# HELP vllm:num_requests_waiting Waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 2.0
# HELP vllm:num_preemptions_total Preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 0.0
# HELP vllm:prompt_tokens_total Prompt tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 2048.0
# HELP vllm:generation_tokens_total Generation tokens
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 1024.0
# HELP vllm:cache_config_info Cache config
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",num_gpu_blocks="8192"} 1.0
"""

MMAP = MetricsMap(
    kv_usage=["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    running=["vllm:num_requests_running"],
    waiting=["vllm:num_requests_waiting"],
    preemptions=["vllm:num_preemptions_total", "vllm:num_preemptions"],
    prompt_tokens=["vllm:prompt_tokens_total"],
    generation_tokens=["vllm:generation_tokens_total"],
    cache_config_info=["vllm:cache_config_info"],
    essential=["kv_usage", "running", "waiting"],
)


def test_parse_mock_metrics() -> None:
    parsed = parse_metrics_text(MOCK_METRICS, MMAP)
    assert parsed["kv_usage"] == pytest.approx(0.42)
    assert parsed["running"] == pytest.approx(8.0)
    assert parsed["waiting"] == pytest.approx(2.0)
    assert parsed["prompt_tokens"] == pytest.approx(2048.0)


def test_first_match_wins() -> None:
    text = MOCK_METRICS + "vllm:gpu_cache_usage_perc 0.99\n"
    parsed = parse_metrics_text(text, MMAP)
    assert parsed["kv_usage"] == pytest.approx(0.42)


def test_cache_config_labels() -> None:
    labels = parse_cache_config_info(MOCK_METRICS)
    assert labels.get("block_size") == "16"


def test_nvml_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: None  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: "h"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetName = lambda h: "Fake GPU"  # type: ignore[attr-defined]
    mem = types.SimpleNamespace(used=1_000, total=24_000)
    fake.nvmlDeviceGetMemoryInfo = lambda h: mem  # type: ignore[attr-defined]
    fake.nvmlSystemGetDriverVersion = lambda: "000.00"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = NvmlSampler()
    assert sampler.available() is True
    info = sampler.device_info()
    assert info["name"] == "Fake GPU"
