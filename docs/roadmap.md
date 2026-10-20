# Roadmap

Long-term architecture (experiments A–D, simulator, capacity planning) with the v0.1 slice highlighted.

## v0.1 (in progress) — the controlled vertical slice

- **A1** calculator: KV accounting as tested code. ✅ specified (T05/T07)
- **A3** engine reconciliation: analytical capacity vs vLLM startup-reported capacity. (needs GPU, T13/T22)
- **A4** KV saturation as concurrency rises (closed-loop). (needs GPU, T22)
- **B0** load-generator / harness timing accuracy (mock ground truth, Tier-0). ✅ specified (T11)

## Post-v0.1 (roadmap only — detailed specification unavailable)

The recovered sources preserve architecture-level intent but **not** complete experimental contracts for:

- **B1–B4** serving experiments (hypotheses, grids, thresholds missing).
- **C-series** scheduling-simulator calibration/validation criteria (missing).
- **D-series** capacity-planning protocols (missing).
- Second engine compatibility contract (missing).

Do not invent these protocols from names alone; they require an approved design
before implementation.
Status of each item: **DETAILED POST-v0.1 SPECIFICATION NOT AVAILABLE**.
