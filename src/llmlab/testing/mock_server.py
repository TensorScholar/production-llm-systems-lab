"""Deterministic fake vLLM with known ground truth (MOCK evidence only).

Behaves like vLLM from the outside: startup lines, /health, /v1/models,
streaming /v1/completions (SSE with a final usage chunk), and /metrics
using the vLLM metric names. Admits FIFO reserving P+O tokens up front, so
ground truth is known exactly: g = P+O, N_sat = floor(C/(P+O)).

Never write mock output under results/ — the runner enforces this.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class MockConfig:
    capacity_tokens: int = 4096
    ttft_ms: float = 20.0
    itl_ms: float = 5.0
    multi_token_every: int = 0
    error_rate: float = 0.0
    stall_ms: float = 0.0
    stall_every: int = 0
    port: int = 8000
    seed: int = 0
    model_name: str = "mock-model"


@dataclass
class _State:
    cfg: MockConfig
    used: int = 0
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    ticket: int = 0
    served: int = 0
    running: int = 0
    waiting: int = 0
    preemptions: int = 0
    prompt_total: int = 0
    gen_total: int = 0
    emitted: dict[str, list[int]] = field(default_factory=dict)
    rng_seed: int = 0

    def __post_init__(self) -> None:
        import random

        self.rng = random.Random(self.rng_seed)


def prompt_key(prompt: list[int], max_tokens: int) -> str:
    """Ground-truth key: hash of (prompt ids, max_tokens) for B0 matching."""
    blob = json.dumps({"p": prompt, "o": max_tokens}, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def startup_lines(cfg: MockConfig, prompt_tokens: int = 32, output_tokens: int = 16) -> str:
    """vLLM-format startup lines exercised by the real parser."""
    per_req = prompt_tokens + output_tokens
    max_conc = cfg.capacity_tokens / per_req if per_req else 0.0
    return (
        f"GPU KV cache size: {cfg.capacity_tokens:,} tokens\n"
        f"Maximum concurrency for {per_req:,} tokens per request: {max_conc:.2f}x\n"
        "Model loading took 0.50 GiB\n"
    )


async def _health(_req: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _models(req: web.Request) -> web.Response:
    state: _State = req.app[_STATE_KEY]
    return web.json_response(
        {"object": "list", "data": [{"id": state.cfg.model_name, "object": "model"}]}
    )


async def _version(_req: web.Request) -> web.Response:
    return web.json_response({"version": "mock-0.1"})


async def _debug_emitted(req: web.Request) -> web.Response:
    """MOCK-only introspection for B0 ground truth (never on a real engine)."""
    state: _State = req.app[_STATE_KEY]
    return web.json_response(
        {k: v for k, v in state.emitted.items() if not k.startswith("arrival:")},
    )


async def _metrics(req: web.Request) -> web.Response:
    state: _State = req.app[_STATE_KEY]
    cfg = state.cfg
    kv_usage = state.used / cfg.capacity_tokens if cfg.capacity_tokens else 0.0
    lines = [
        "# HELP vllm:kv_cache_usage_perc KV cache usage",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc {kv_usage}",
        "# HELP vllm:num_requests_running Running",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running {state.running}",
        "# HELP vllm:num_requests_waiting Waiting",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting {state.waiting}",
        "# HELP vllm:num_preemptions_total Preemptions",
        "# TYPE vllm:num_preemptions_total counter",
        f"vllm:num_preemptions_total {state.preemptions}",
        "# HELP vllm:prompt_tokens_total Prompt tokens",
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total {state.prompt_total}",
        "# HELP vllm:generation_tokens_total Generation tokens",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total {state.gen_total}",
        "# HELP vllm:cache_config_info Cache config",
        "# TYPE vllm:cache_config_info gauge",
        'vllm:cache_config_info{block_size="16"} 1',
    ]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def _completions(req: web.Request) -> web.StreamResponse:
    state: _State = req.app[_STATE_KEY]
    cfg = state.cfg
    try:
        body: dict[str, Any] = await req.json()
    except Exception:
        return web.json_response({"error": "bad json"}, status=400)
    prompt = body.get("prompt", [])
    n_prompt = len(prompt) if isinstance(prompt, list) else 32
    n_out = int(body.get("max_tokens", 16) or 16)
    need = n_prompt + n_out
    t_arrival_ns = time.perf_counter_ns()
    if isinstance(prompt, list):
        try:
            state.emitted["arrival:" + prompt_key([int(x) for x in prompt], n_out)] = [t_arrival_ns]
        except (TypeError, ValueError):
            pass

    # Fault injection: fail fast before admission.
    state.served += 1
    my_index = state.served
    if cfg.error_rate > 0 and state.rng.random() < cfg.error_rate:
        return web.json_response({"error": "injected fault"}, status=500)

    # FIFO admission reserving P+O up front.
    async with state.cond:
        my_ticket = state.ticket
        state.ticket += 1
        waiters: collections.deque[int] = req.app[_WAITERS_KEY]
        waiters.append(my_ticket)
        state.waiting = len(waiters) - 0
        while waiters[0] != my_ticket or state.used + need > cfg.capacity_tokens:
            state.waiting = len(waiters)
            await state.cond.wait()
        waiters.popleft()
        state.used += need
        state.running += 1
        state.waiting = len(waiters)
        state.cond.notify_all()

    request_id = f"mock-{my_index:08d}"
    emit_times: list[int] = []
    try:
        resp = web.StreamResponse(status=200, reason="OK")
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(req)
        stall = (
            cfg.stall_ms / 1000.0 if cfg.stall_every and my_index % cfg.stall_every == 0 else 0.0
        )
        await asyncio.sleep(cfg.ttft_ms / 1000.0 + stall)
        multi = cfg.multi_token_every and my_index % cfg.multi_token_every == 0
        # Emit O tokens across chunks; multi mode merges two tokens into one chunk.
        remaining = n_out
        chunk_idx = 0
        first = True
        while remaining > 0:
            if first:
                await asyncio.sleep(0)
                first = False
            else:
                await asyncio.sleep(cfg.itl_ms / 1000.0)
            take = 2 if (multi and remaining > 1 and chunk_idx == 0) else 1
            remaining -= take
            chunk_idx += 1
            emit_times.append(time.perf_counter_ns())
            payload = {
                "id": request_id,
                "choices": [{"index": 0, "text": "t" * take, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(payload)}\n\n".encode())
        state.prompt_total += n_prompt
        state.gen_total += n_out
        usage = {
            "id": request_id,
            "choices": [],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_out,
                "total_tokens": n_prompt + n_out,
            },
        }
        await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        state.emitted[request_id] = emit_times
        if isinstance(prompt, list):
            try:
                state.emitted[prompt_key([int(x) for x in prompt], n_out)] = emit_times
            except (TypeError, ValueError):
                pass
        await resp.write_eof()
        return resp
    finally:
        async with state.cond:
            state.used -= need
            state.running -= 1
            state.cond.notify_all()


def create_app(cfg: MockConfig) -> web.Application:
    """Create the mock vLLM aiohttp application (in-process testing)."""
    app = web.Application()
    app[_STATE_KEY] = _State(cfg=cfg, rng_seed=cfg.seed)
    app[_WAITERS_KEY] = collections.deque()
    app.router.add_get("/health", _health)
    app.router.add_get("/v1/models", _models)
    app.router.add_get("/version", _version)
    app.router.add_get("/metrics", _metrics)
    app.router.add_get("/debug/emitted", _debug_emitted)
    app.router.add_post("/v1/completions", _completions)
    return app


try:
    _STATE_KEY = web.AppKey("llmlab_state", _State)
    _WAITERS_KEY: Any = web.AppKey("llmlab_waiters", collections.deque)
except AttributeError:  # aiohttp without AppKey
    _STATE_KEY = "state"  # type: ignore[assignment]
    _WAITERS_KEY = "waiters"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mock vLLM server (MOCK evidence only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capacity-tokens", type=int, default=4096, dest="capacity_tokens")
    parser.add_argument("--ttft-ms", type=float, default=20.0)
    parser.add_argument("--itl-ms", type=float, default=5.0)
    parser.add_argument("--multi-token-every", type=int, default=0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--stall-ms", type=float, default=0.0)
    parser.add_argument("--stall-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = MockConfig(
        capacity_tokens=args.capacity_tokens,
        ttft_ms=args.ttft_ms,
        itl_ms=args.itl_ms,
        multi_token_every=args.multi_token_every,
        error_rate=args.error_rate,
        stall_ms=args.stall_ms,
        stall_every=args.stall_every,
        port=args.port,
        seed=args.seed,
    )
    print(startup_lines(cfg), flush=True)
    web.run_app(create_app(cfg), port=args.port, print=None)


if __name__ == "__main__":
    main()
