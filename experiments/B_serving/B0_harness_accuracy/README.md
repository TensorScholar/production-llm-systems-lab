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

<!-- Filled by the B0 generator; DATA REQUIRED until then. -->
