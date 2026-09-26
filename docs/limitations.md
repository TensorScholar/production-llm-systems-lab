# Limitations

- Single GPU (24 GB class) and single model (Qwen2.5-7B-Instruct); no claim
  generalizes to other hardware, models, precisions, or token budgets.
- Fixed request shape (2048-token prompts, 1024-token outputs); mixed lengths
  and prefix sharing are out of scope for v0.1 (they change ḡ).
- Closed-loop (fixed concurrency) only; open-loop arrival processes are v0.2.
- No output-quality measurement; no latency-target claim; no multi-replica
  behaviour; no security guarantees.
- vLLM version-specific details (flags, metric names, log forms) are pinned
  in T13; behaviour may differ across versions.
- The 45 ms TPOT discussion is an analytical pre-screen, not a measurement.
- The prefix-cache/preemption question is retained as an open question for
  B3 (post-v0.1, source-blocked), not a measured fact.
