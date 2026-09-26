# PREREG — A3 engine reconciliation (Tier 2, reference GPU)

**Status:** preregistered before any A3/A4 GPU run. Do not edit after runs begin;
record deviations in `README.md`.

## Hypotheses

- **H-A3-1 (claim A3-01):** vLLM's startup-reported KV capacity satisfies
  `kv_cache_tokens == num_gpu_blocks × block_size` exactly, and the reported
  max concurrency equals `C / max_model_len` within ±0.5%.
- **H-A3-2 (claim A3-02):** non-KV overhead
  `overhead = util × NVML_total − weight_bytes − pool_bytes`, fitted on the
  `pool-full` config, predicts held-out pool sizes within ±5%.

## Method

Startup-only runs: `util ∈ {0.80, 0.90} × max_model_len ∈ {4096, 8192}`,
2 repetitions, one fresh engine instance per (config, repetition).
Prefix caching off; `max_num_seqs ≥ 2 × max(N)`; explicit token budget and
block size 16. Model `Qwen/Qwen2.5-7B-Instruct` at the pinned revision (T13).

## Pass criteria

- A3-01: exact block identity; concurrency ratio ±0.5%.
- A3-02: held-out pool prediction ±5%.

## Two-stage procedure (authorized in advance)

1. Run A3 first.
2. Re-run `llmlab predict` to fill capacity-dependent A4 predictions from
   `results/derived/A3/reconciliation.csv`. This is the ONLY authorized
   post-PREREG change to predictions, and it uses no A4 data.
3. Run A4.

## Exclusion rules

- Runs failing essential-metrics or idle checks are invalid and kept with reasons.
- Log/`cache_config_info` disagreement is a failed check, not a guess.
