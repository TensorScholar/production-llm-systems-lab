"""KV-cache accounting (book Ch. 3.4, 4.4, 4.4.1).

All functions are pure; integer-exact where the maths allows. All
arithmetic uses explicit units (bytes as int).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from llmlab.models import ModelSpec

logger = logging.getLogger(__name__)


def kv_bytes_per_token(spec: ModelSpec, kv_dtype_bytes: int) -> int:
    """KV bytes per token: m_kv = 2·L·n_kv·d_h·b. Book §Ch.3.4."""
    return 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * kv_dtype_bytes


def blocks_for_tokens(tokens: int, block_size: int) -> int:
    """Number of paged blocks for T tokens: ceil(T/q). Book §Ch.4.4.1."""
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    return math.ceil(tokens / block_size) if tokens else 0


def slack_tokens(tokens: int, block_size: int) -> int:
    """Wasted slots in the last block: (q − T mod q) mod q. Book §Ch.4.4.1."""
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    return (block_size - tokens % block_size) % block_size


@dataclass(frozen=True)
class PoolCapacity:
    num_blocks: int
    tokens: int


def pool_capacity(pool_bytes: int, m_kv: int, block_size: int) -> PoolCapacity:
    """Usable KV pool capacity from pool bytes. Book §Ch.4.4.

    Formula: num_blocks = floor(pool_bytes / (q·m_kv)); tokens = num_blocks·q.
    """
    if pool_bytes < 0 or m_kv <= 0 or block_size <= 0:
        raise ValueError("pool_bytes >= 0, m_kv > 0, block_size > 0 required")
    num_blocks = pool_bytes // (block_size * m_kv)
    return PoolCapacity(num_blocks=num_blocks, tokens=num_blocks * block_size)


def pool_bytes(num_blocks: int, block_size: int, m_kv: int) -> int:
    """Pool bytes from block count: B_pool = num_blocks·q·m_kv. Book §Ch.4.4."""
    if num_blocks < 0 or block_size <= 0 or m_kv <= 0:
        raise ValueError("num_blocks >= 0, block_size > 0, m_kv > 0 required")
    return num_blocks * block_size * m_kv


def max_concurrency_ratio(capacity_tokens: int, tokens_per_request: int) -> float:
    """Fluid (non-paged) max concurrency: C / T_req. Book §Ch.4.4."""
    if tokens_per_request <= 0:
        raise ValueError("tokens_per_request must be > 0")
    return capacity_tokens / tokens_per_request


def max_concurrency_blocks(num_blocks: int, tokens_per_request: int, block_size: int) -> int:
    """Paged max concurrency: floor(num_blocks / ceil(T_req/q)). Book §Ch.4.4.1."""
    if tokens_per_request <= 0:
        raise ValueError("tokens_per_request must be > 0")
    per_req = blocks_for_tokens(tokens_per_request, block_size)
    if per_req == 0:
        raise ValueError("tokens_per_request must be > 0 blocks")
    return num_blocks // per_req


def g_decode(prompt: int, output: int, block_size: int) -> float:
    """Mean decode-lifetime KV footprint (tokens).

    Formula: g_bar = (1/(O−1)) · Σ_{k=1}^{O−1} q·ceil((P+k)/q),
    summed exactly, not approximated. Book §Ch.4.4.1.
    """
    if prompt < 0 or output < 1 or block_size <= 0:
        raise ValueError("prompt >= 0, output >= 1, block_size > 0 required")
    if output == 1:
        return float(block_size * math.ceil(prompt / block_size)) if prompt else 0.0
    total = sum(block_size * math.ceil((prompt + k) / block_size) for k in range(1, output))
    return total / (output - 1)


def g_max(prompt: int, output: int, block_size: int) -> int:
    """Peak KV footprint (tokens): q·ceil((P+O)/q). Book §Ch.4.4.1."""
    if prompt < 0 or output < 0 or block_size <= 0:
        raise ValueError("prompt >= 0, output >= 0, block_size > 0 required")
    return block_size * math.ceil((prompt + output) / block_size)


def g_semi_empirical(
    prompt: int, output: int, block_size: int, ttft_s: float, tpot_s: float
) -> float:
    """Time-weighted mean footprint using measured TTFT/TPOT.

    Formula: [TTFT·(q·ceil(P/q)/2) + (O−1)·TPOT·g_decode]
             / [TTFT + (O−1)·TPOT]. Book §Ch.4.4.1.
    """
    if ttft_s < 0 or tpot_s <= 0:
        raise ValueError("ttft_s >= 0 and tpot_s > 0 required")
    g_dec = g_decode(prompt, output, block_size)
    prefill_tokens = block_size * math.ceil(prompt / block_size) if prompt else 0
    decode_time = (output - 1) * tpot_s
    total_time = ttft_s + decode_time
    if total_time <= 0:
        raise ValueError("total time must be > 0")
    return (ttft_s * (prefill_tokens / 2) + decode_time * g_dec) / total_time


def predicted_occupancy(n: int, g: float, capacity_tokens: int) -> float:
    """Predicted mean KV occupancy (tokens): min(N·g, C). Book §Ch.4.4."""
    if n < 0 or g < 0 or capacity_tokens < 0:
        raise ValueError("n, g, capacity_tokens must be >= 0")
    return min(n * g, float(capacity_tokens))


def predicted_saturation_interval(
    capacity_tokens: int, g_mean: float, g_peak: int
) -> tuple[float, float]:
    """Predicted saturation interval (C/g_max, C/g_bar). Book §Ch.4.4.

    Below C/g_max the requests fit even if every one is at full length;
    above C/g_bar average demand exceeds capacity.
    """
    if capacity_tokens <= 0 or g_mean <= 0 or g_peak <= 0:
        raise ValueError("capacity_tokens, g_mean, g_peak must be > 0")
    return (capacity_tokens / g_peak, capacity_tokens / g_mean)
