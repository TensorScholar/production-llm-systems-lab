"""Unit tests for the mock server: admission, metrics, usage, faults."""

from __future__ import annotations

import asyncio

import pytest

from llmlab.testing.mock_server import MockConfig, create_app, startup_lines


def test_startup_lines_match_parser_regexes() -> None:
    import re

    text = startup_lines(MockConfig(capacity_tokens=4096), 32, 16)
    assert re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    assert re.search(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", text)
    assert re.search(r"Model loading took\s*([\d.]+)\s*(GiB|GB)", text)


async def _start_app(cfg: MockConfig) -> tuple:  # type: ignore[no-untyped-def]
    from aiohttp.test_utils import TestClient, TestServer

    client = TestClient(TestServer(create_app(cfg)))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_health_models_metrics() -> None:
    client = await _start_app(MockConfig(capacity_tokens=512))
    try:
        async with client.get("/health") as r:
            assert r.status == 200
        async with client.get("/v1/models") as r:
            data = await r.json()
            assert data["data"][0]["id"] == "mock-model"
        async with client.get("/metrics") as r:
            text = await r.text()
            assert "vllm:kv_cache_usage_perc 0.0" in text
            assert "vllm:num_requests_running 0" in text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_completion_usage_chunk() -> None:
    client = await _start_app(MockConfig(capacity_tokens=512, ttft_ms=1, itl_ms=1))
    try:
        payload = {"model": "mock-model", "prompt": list(range(32)), "max_tokens": 8}
        async with client.post("/v1/completions", json=payload) as r:
            assert r.status == 200
            text = await r.text()
            assert "[DONE]" in text
            assert '"completion_tokens": 8' in text
            assert '"prompt_tokens": 32' in text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_error_injection() -> None:
    client = await _start_app(MockConfig(capacity_tokens=512, ttft_ms=1, itl_ms=1, error_rate=1.0))
    try:
        async with client.post(
            "/v1/completions", json={"model": "m", "prompt": [1, 2], "max_tokens": 4}
        ) as r:
            assert r.status == 500
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_fifo_capacity_bound() -> None:
    # C=48, P+O=16 -> at most 3 concurrent; 6 parallel requests must all succeed.
    client = await _start_app(MockConfig(capacity_tokens=48, ttft_ms=2, itl_ms=1))
    try:

        async def one() -> int:
            async with client.post(
                "/v1/completions", json={"model": "m", "prompt": list(range(8)), "max_tokens": 8}
            ) as r:
                assert r.status == 200
                return r.status

        results = await asyncio.gather(*[one() for _ in range(6)])
        assert all(s == 200 for s in results)
        async with client.get("/metrics") as r:
            text = await r.text()
            assert "vllm:num_requests_running 0" in text
    finally:
        await client.close()
