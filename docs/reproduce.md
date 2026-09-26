# Reproduce

## Tier 0 (any machine, no GPU)

```bash
pip install -e ".[dev]"
pytest -q
make a1 && diff results/derived/A1/kv_table.csv tests/golden/a1_kv_table.csv
make demo-mock
python -m llmlab.cli validate --results results/
```

Tier-0 outputs reproduce exactly (analytical identities) or qualitatively
(mock pipeline recovers the mock's known N_sat = floor(C/(P+O))).

## Tier 2 (reference GPU, ~10 GPU-hours)

Follow `docs/run-cards/A3-A4-reference-gpu.md`. Requires a CUDA GPU,
pinned versions (T13), and model access. Results reproduce qualitatively
(saturation knee inside the predicted band), not bit-identically.
