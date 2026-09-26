"""Engine lifecycle via mock: start -> ready -> report -> stop + timeout path."""

from __future__ import annotations

import pytest

from llmlab.engines.vllm import EngineStartupError, VLLMServer
from llmlab.schemas import EngineConfig, ModelRef

MODEL = ModelRef(id="mock", hf_repo="mock-model", revision="mock")
BASE = EngineConfig(
    name="mock",
    version="mock-0.1",
    command=["python", "-m", "llmlab.testing.mock_server"],
    port=18091,
    startup_timeout_s=60,
    base_args={},
)


def test_lifecycle_mock(tmp_path) -> None:  # type: ignore[no-untyped-def]
    log = tmp_path / "engine.log"
    server = VLLMServer(BASE, MODEL, {"mock": {"capacity_tokens": 4096}}, log)
    server.start()
    try:
        server.wait_ready(timeout_s=60)
        report = server.startup_report()
        assert report.kv_cache_tokens == 4096
        assert report.max_concurrency_tokens == 48  # mock default shape 32+16
        assert server.version() == "mock-0.1"
        assert log.exists()
    finally:
        server.stop()


def test_startup_timeout_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = BASE.model_copy(update={"port": 19999, "startup_timeout_s": 3})
    server = VLLMServer(bad, MODEL, {"mock": {"capacity_tokens": 64}}, tmp_path / "x.log")
    # Point at a port where nothing listens by never starting: call wait_ready
    # directly without start (connection refused until timeout).
    with pytest.raises(EngineStartupError):
        server.wait_ready(timeout_s=3)


def test_context_manager(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with VLLMServer(BASE, MODEL, {}, tmp_path / "e.log") as server:
        server.wait_ready(timeout_s=60)
        assert server.startup_report().kv_cache_tokens == 4096
