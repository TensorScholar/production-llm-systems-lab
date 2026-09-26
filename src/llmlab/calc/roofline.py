"""Bounds and the calibrated closed-loop serving model (book Ch. 3.2–3.3, 4.1.1, 4.3).

Valid only below saturation. Assumptions are stated in
docs/methodology/kv-accounting.md: prefill cost independent of chunking,
decode tokens riding in mixed batches add no time, no queueing, uniform
request shape.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from llmlab.models import ModelSpec

logger = logging.getLogger(__name__)

_MAX_ITERS = 200
_TOL = 1e-9


def decode_weight_bytes(spec: ModelSpec, weight_dtype_bytes: int) -> int:
    """Decode-streamed weight bytes.

    Formula: W_dec = param_count·b − (vocab·hidden·b if not tied): the
    input-embedding table is gathered, not streamed. Book §Ch.3.2.
    """
    if spec.param_count is None:
        raise ValueError("param_count required for decode_weight_bytes")
    if weight_dtype_bytes <= 0:
        raise ValueError("weight_dtype_bytes must be > 0")
    total = spec.param_count * weight_dtype_bytes
    if not spec.tie_word_embeddings:
        total -= spec.vocab_size * spec.hidden_size * weight_dtype_bytes
    return total


def tpot_floor_s(decode_weight_bytes: int, kv_bytes_read: float, peak_bw_Bps: float) -> float:
    """Bandwidth floor on TPOT: (W_dec + KV_read) / peak_BW. Book §Ch.3.3."""
    if peak_bw_Bps <= 0:
        raise ValueError("peak_bw_Bps must be > 0")
    if decode_weight_bytes < 0 or kv_bytes_read < 0:
        raise ValueError("byte quantities must be >= 0")
    return (decode_weight_bytes + kv_bytes_read) / peak_bw_Bps


def ttft_floor_s(param_count: int, prompt_tokens: int, peak_flops: float) -> float:
    """Compute floor on TTFT: 2·N·P / peak FLOPS. Book §Ch.3.2."""
    if peak_flops <= 0:
        raise ValueError("peak_flops must be > 0")
    if param_count < 0 or prompt_tokens < 0:
        raise ValueError("param_count and prompt_tokens must be >= 0")
    return 2.0 * param_count * prompt_tokens / peak_flops


@dataclass(frozen=True)
class Calibration:
    bw_eff_Bps: float
    t_prefill_s: float


def calibrate(tpot_1: float, ttft_1: float, w_dec: int, g: float, m_kv: int) -> Calibration:
    """Calibrate effective bandwidth and prefill time from N=1 runs.

    Formula: BW_eff = (W_dec + g·m_kv)/TPOT(1); t_pf = TTFT(1) − TPOT(1).
    Book §Ch.4.3.
    """
    if tpot_1 <= 0 or ttft_1 < tpot_1 or w_dec < 0 or g < 0 or m_kv <= 0:
        raise ValueError("need tpot_1 > 0, ttft_1 >= tpot_1, non-negative sizes")
    bw_eff = (w_dec + g * m_kv) / tpot_1
    return Calibration(bw_eff_Bps=bw_eff, t_prefill_s=ttft_1 - tpot_1)


@dataclass(frozen=True)
class ServingPrediction:
    n: int
    tpot_s: float
    ttft_s: float
    lifetime_s: float
    req_per_s: float
    out_tok_per_s: float
    prefill_fraction: float


def predict_closed_loop(
    n: int,
    cal: Calibration,
    w_dec: int,
    g: float,
    m_kv: int,
    prompt: int,
    output: int,
    token_budget: int,
) -> ServingPrediction:
    """Closed-loop serving prediction for concurrency N. Book §Ch.4.3.

    Equations (solved for λ by bisection; λ depends on TPOT which depends
    on λ):
      t_dec(N) = (W_dec + N·g·m_kv) / BW_eff
      n_c(N) = ceil(P / (B_tok − (N−1)))
      λ_o = λ·(N−1)/N (rate of other requests' prefills)
      f_o = λ_o·t_pf
      S = (1 − f_o)/t_dec(N) + λ_o·n_c(N) (tok/s for one decoding request)
      TPOT = 1/S; TTFT = t_pf + t_dec(N); L = TTFT + (O−1)·TPOT; λ = N/L.
    """
    if n < 1 or prompt < 1 or output < 1 or token_budget < 1:
        raise ValueError("n, prompt, output, token_budget must be >= 1")
    if cal.bw_eff_Bps <= 0 or cal.t_prefill_s < 0:
        raise ValueError("calibration must have positive bandwidth")
    t_dec = (w_dec + n * g * m_kv) / cal.bw_eff_Bps
    denom = token_budget - (n - 1)
    if denom <= 0:
        raise ValueError("token budget too small for concurrency N")
    n_c = math.ceil(prompt / denom)
    t_pf = cal.t_prefill_s

    # Fixed-point: λ = N / L(λ). Bisect λ in (0, N / t_pf_floor].
    lo = 1e-12
    hi = n / max(t_pf + t_dec, 1e-12) * 2.0 + 1.0

    def lifetime(lam: float) -> float:
        lam_o = lam * (n - 1) / n
        f_o = lam_o * t_pf
        if f_o >= 1.0:
            return math.inf
        s = (1.0 - f_o) / t_dec + lam_o * n_c
        if s <= 0:
            return math.inf
        tpot = 1.0 / s
        return t_pf + t_dec + (output - 1) * tpot

    def residual(lam: float) -> float:
        L = lifetime(lam)
        if not math.isfinite(L) or L <= 0:
            return lam  # push downward
        return lam - n / L

    # Bisection on residual (monotone increasing in lam).
    r_hi = residual(hi)
    it = 0
    while r_hi <= 0 and it < _MAX_ITERS:
        hi *= 2.0
        r_hi = residual(hi)
        it += 1
    for _ in range(_MAX_ITERS):
        mid = 0.5 * (lo + hi)
        r_mid = residual(mid)
        if abs(r_mid) <= _TOL * max(mid, 1e-12):
            lo = hi = mid
            break
        if r_mid > 0:
            hi = mid
        else:
            lo = mid
    else:
        raise RuntimeError("closed-loop solver did not converge")
    lam = 0.5 * (lo + hi)
    L = lifetime(lam)
    if not math.isfinite(L):
        raise RuntimeError("closed-loop solver did not converge")
    lam_o = lam * (n - 1) / n
    f_o = lam_o * t_pf
    s = (1.0 - f_o) / t_dec + lam_o * n_c
    tpot = 1.0 / s
    ttft = t_pf + t_dec
    prefill_frac = (lam_o * t_pf) if lam > 0 else 0.0
    return ServingPrediction(
        n=n,
        tpot_s=tpot,
        ttft_s=ttft,
        lifetime_s=L,
        req_per_s=lam,
        out_tok_per_s=lam * output,
        prefill_fraction=min(max(prefill_frac, 0.0), 1.0),
    )
