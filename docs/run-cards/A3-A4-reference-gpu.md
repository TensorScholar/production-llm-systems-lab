# Run card — A3/A4 on the reference GPU (Tier 2)

Expected budget: ~1 GPU-hour (T13) + 0.5 (A3) + 7–8 (A4) ≈ 10 GPU-hours.

## 1. Provision

- One 24 GB CUDA GPU (A10G-class); record NVML device info.
- CUDA-compatible host; Docker or a locked venv for the engine.

## 2. Engine environment (T13)

```bash
# Start vLLM with base args; confirm token-ID prompts, min_tokens,
# ignore_eos, stream_options usage, prefix-caching-off flag, /version,
# startup log lines, cache_config_info, metric names.
# Save: startup log, idle + busy /metrics text under tests/fixtures/vllm/<version>/
# Pin: configs/models revision, configs/engines/vllm.yaml version+digest,
#      configs/engines/vllm-requirements.lock, vllm-metrics.yaml names.
```

## 3. Pre-download

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --revision <pinned>
```

## 4. Predict (preregistration)

```bash
python -m llmlab.cli predict experiments/A_kv_cache/A4_kv_saturation/manifest.yaml
git add results/derived/A4/predictions_pre.json experiments/*/PREREG.md claims/ledger.yaml
git commit -m "prereg: freeze A3/A4 predictions and criteria"
```

## 5. Run A3, re-predict, run A4

```bash
python -m llmlab.cli run experiments/A_kv_cache/A3_engine_reconciliation/manifest.yaml
python -m llmlab.cli predict experiments/A_kv_cache/A4_kv_saturation/manifest.yaml  # stage 2 only
python -m llmlab.cli run experiments/A_kv_cache/A4_kv_saturation/manifest.yaml
```

## 6. Validate and commit

```bash
python -m llmlab.cli validate --results results/
python -m llmlab.cli analyze experiments/A_kv_cache/A4_kv_saturation/manifest.yaml
python -m llmlab.cli report experiments/A_kv_cache/A4_kv_saturation/manifest.yaml
git add results docs claims && git commit -m "evidence: A3/A4 reference-GPU runs"
```

Dry-run on mock first: `make demo-mock`.
