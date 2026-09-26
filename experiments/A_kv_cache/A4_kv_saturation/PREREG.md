# PREREG — A4 KV saturation under closed-loop concurrency (Tier 2, reference GPU)

**Status:** preregistered before any A3/A4 GPU run. Do not edit after runs begin;
record deviations in `README.md`.

## Analytical predictions (from `llmlab predict`, committed as predictions_pre.json)

- `m_kv` = 57,344 B/token (56 KiB) for Qwen2.5-7B-Instruct BF16.
- Mean decode footprint `ḡ` = 2,567.5 tokens; peak `g_max` = 3,072 tokens
  (P=2048, O=1024, block 16).
- TPOT floor ≈ 23.6 ms at 600 GB/s peak (illustrative; recomputed from pinned values).
- Illustrative saturation band: if A3 measures C ≈ 80K tokens, the knee is
  predicted at N ≈ 26–31. Real band comes from A3 via the two-stage procedure.

## Hypotheses

- **H-A4-1 (claim A4-01):** below saturation, mean KV occupancy = N·ḡ
  (±10% analytic ḡ; ±5% semi-empirical).
- **H-A4-2 (claim A4-02):** queueing (waiting ≥ 0.5) begins inside
  [C/g_max, C/ḡ] (± one grid step); the reduced pool's knee scales with its
  capacity ratio (±15%).
- **H-A4-3 (claim A4-03):** TPOT(1) ≥ (W_dec + ḡ·m_kv)/peak bandwidth.
- **H-A4-4 (claim A4-04):** N=1-calibrated mixed-batch model predicts TPOT and
  output throughput for unsaturated N ≥ 2 (median residual ≤ 20%).
- **H-A4-5 (claim A4-05):** above saturation, throughput gains ≤ 5% per grid
  step and TTFT ≈ TTFT_unsat + waiting/λ (±20%).

## Grid

Grids are dense around the predicted saturation band (see manifest).
3 repetitions per (config, N); one fresh engine instance per (config, repetition);
N order shuffled within each instance (seed 7).

## Exclusion rules

- Error rate > 1%, usage-token mismatch, missing essential metrics, client
  health breach, throttle, non-idle start, or instability (±10% halves) each
  invalidate the run; invalid runs are kept with reasons.
- `completed_in_window < min_completed` is informational only.

## Decision rule (pre-committed)

Cap = largest tested N where every valid repetition had waiting < 0.5, zero
preemptions, and KV usage ≤ 0.90 headroom. Binding constraint and levers per
the analysis module rules.
