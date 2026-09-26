# B0 — harness accuracy

Method: in-process mock server at N ∈ {1, 8, 32, 64} (Tier 0, no GPU).
Client chunk receipts are matched exactly to server emission timestamps via
prompt-hash keys on the same `perf_counter` clock, so transit time is
excluded by construction; the residual is pure harness timestamping error.

- Claim B0-01: p99 |TTFT error| and p99 |ITL error| ≤ 1 ms.
- Table: `results/derived/B0/harness_accuracy.csv` (MOCK evidence; mock output
  never enters `results/` — the committed table is derived evidence with
  provenance, produced by the B0 generator).

Environment note: timestamping tails depend on machine timer/scheduler
granularity. Reference criterion (1 ms) is evaluated on the reference setup;
local measurements are recorded honestly in the table below.

## Results

MOCK evidence from `results/derived/B0/harness_accuracy.csv`
(machine: MacBook Air M2-class, uvloop, in-process mock, P=32/O=16,
ttft 50 ms / itl 20 ms; generator: `generate_b0.py`):

| N | matched | TTFT median | TTFT p99 | ITL median | ITL p99 | loop-lag p99 |
|---|---------|-------------|----------|------------|---------|--------------|
| 1 | 55 | 0.21 ms | 0.46 ms | 0.07 ms | 1.92 ms | 2.73 ms |
| 8 | 437 | 0.15 ms | 3.45 ms | 0.14 ms | 4.82 ms | 3.50 ms |
| 32 | 1760 | 0.15 ms | 2.65 ms | 0.22 ms | 5.88 ms | 4.02 ms |
| 64 | 3389 | 0.20 ms | 17.7 ms | 0.29 ms | 13.2 ms | 11.6 ms |

Timestamping medians are 0.1–0.3 ms at all N (harness sound); p99 tails of
2–18 ms come from machine scheduler/timer granularity on this non-reference
machine (run-to-run variance confirms: an earlier run gave N=1 p99 8.6 ms).
Claim B0-01 (p99 ≤ 1 ms, scoped to the reference setup) is therefore
**inconclusive** here (`in_scope_reference=False`); it must be re-measured on
the reference setup before T23 can mark it supported/refuted.
