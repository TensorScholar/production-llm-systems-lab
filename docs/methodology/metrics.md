# Metrics methodology

Canonical definitions shared by real and mock runs (`src/llmlab/metrics.py`).

## Per-request timing

- **TTFT** = first choice chunk − send (seconds).
- **E2E latency** = last chunk − send.
- **TPOT** = (E2E − TTFT)/(n_out − 1), with n_out from the reported usage
  chunk; undefined (null) when n_out < 2 (zero/one-token outputs).
- **ITL** = gaps between consecutive chunks after the first, pooled across
  requests for event-level statistics. Per-request TPOT must not be blurred
  with globally pooled inter-token events: TPOT is a per-request statistic;
  pooled ITLs are an event-level statistic. They coincide (TPOT = mean ITL)
  only when every chunk holds exactly one token.
- Chunks with empty text still count as timing events; the usage-only chunk
  supplies token counts, not timing.
- Malformed timestamps, incomplete streams, rejected and failed requests are
  recorded with status error/timeout and excluded from latency statistics but
  counted in error rates. NaN/Infinity is never emitted as a reported result.

## Window rules

Warmup lasts `warmup_s`. The window closes once both `window_min_s` and
`min_completed` are reached, or at `window_max_s`. A request is in-window if
sent inside it. After close, workers stop sending and in-flight requests
drain; requests unfinished after `drain_timeout_s` are recorded as timeout.

## Rates

- Request throughput = in-window completions ÷ window length.
- Output-token throughput, two ways: client (Σ n_out ÷ window) and server
  (generation-token counter delta ÷ window).

## Occupancy and telemetry

- KV occupancy = time-mean of the KV-usage gauge × C.
- Also: mean running/waiting, preemption-counter delta, GPU memory used
  (mean/max — the engine's reservation, NOT occupancy), SM clock minimum,
  mean power, max temperature.
- NVML utilization and memory are not occupancy: utilization measures SM
  activity, memory measures reservation including weights and overhead.

## Percentile gates

p50/p90/p95/p99 are reported only if at least `min_tail_samples` samples lie
beyond them (n·(1−q/100) ≥ min_tail); otherwise null with a stated reason,
shown as n/a — never silently dropped.

## Checks

Error rate ≤ 1%, usage match, essential metrics, client health
(loop-lag p99 ≤ 5 ms, process CPU ≤ 80%), no throttle, idle start,
stability (±10% halves, ≥ 40 completions) invalidate; min_completed is
informational.
