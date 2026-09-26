# Production LLM Systems Lab — production-relevant measurement of LLM serving systems (not operation of production systems).

> **Scope.** This lab performs **production-relevant measurement** of LLM serving on bounded, named test setups (one GPU, one model, fixed workloads). It is **not** evidence of operating commercial production systems, GPU fleets, multi-replica serving, or hyperscale infrastructure.

**Status:** pre-release (v0.1 in development). No GPU measurements have been taken yet; all quantitative tables in this repo are analytical identities or mock-harness outputs unless explicitly labelled MEASURED.

## What v0.1 shows

- **Question:** how many long-context requests can one 24 GB GPU serve at once, and what limits that number?
- **Method:** published mechanism → tested analytical model → explicit prediction → measurement → residual → engineering decision.
- **Slice:** KV-cache capacity (A1 calculator, A3 engine reconciliation, A4 saturation sweep) plus load-generator accuracy (B0).

## What v0.1 does NOT demonstrate

- Commercial production operation or fleet-scale experience.
- Multi-replica reliability, universal serving speedups, or universal engine ranking.
- Output-quality conclusions (quality is not measured) or security guarantees.
- Generalization to untested hardware, models, or workloads.

## Quickstart

```bash
pip install -e ".[dev]"
make lint && make type && pytest -q   # Tier-0 checks
make a1                                # analytical KV table vs golden
make demo-mock                         # end-to-end mock pipeline (MOCK labels)
python -m llmlab.cli validate --results results/
```

Tier-2 (reference GPU) runs follow `docs/run-cards/A3-A4-reference-gpu.md` and require a CUDA GPU; they do not run in CI or on machines without one.

## Repository map

- `src/llmlab/` — library (`calc`, `models`, `workload`, `loadgen`, `metrics`, `telemetry`, `engines`, `runner`, `analysis`, `report`)
- `configs/` — model / hardware / engine configs
- `experiments/` — manifests + preregistrations (A1, A3, A4, B0)
- `results/` — immutable runs + derived evidence
- `claims/` — claims ledger (see `CLAIMS.md`)
- `docs/` — methodology, stories, run cards, ADRs, roadmap

## Citation

See `CITATION.cff`.
