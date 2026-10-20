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
- Automatic Prefix Caching (APC) is off in the v0.1 experiments: the runner
  enforces the prefix-caching-off flag, and each run starts at a different offset
  so prompts are not reused.
- The earlier book statement that a cached-but-rarely-reused prefix can directly
  force active requests into preemption is suspected to be wrong for vLLM. In the
  APC design, cached blocks that are no longer referenced by an active sequence
  belong to the free, evictable pool and are reclaimed LRU-style, whereas blocks
  referenced by an active sequence are pinned and are not evicted by the APC
  cache policy.
- Under memory pressure, active sequences are relieved by preemption (recompute
  or swap), not by APC evicting their own blocks. The actual interaction was to
  be tested by experiment B3, which is post-v0.1, so it remains a suspicion
  rather than a measured fact.
