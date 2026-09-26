# PRODUCTION LLM SYSTEMS LAB — CANONICAL IMPLEMENTATION BLUEPRINT

**Repository root:** `production-llm-systems-lab/`  
**Python package:** `llmlab`  
**Canonical executable specification:** Claude v0.1 implementation specification, final, 2026-09-25  
**Long-term architecture status:** recovered at architecture/roadmap level; the surviving source is partially text-corrupted and interrupted during Experiment A1  
**Purpose of this file:** provide one safe source of truth for OpenCode/other implementation agents without inventing missing Claude requirements

---

## 0. Control statement

This file consolidates two recovered Claude artifacts:

1. a **long-term `llm-serving-lab` companion-lab design**, and
2. the **complete `Production LLM Systems Lab: v0.1 Implementation Specification`**.

The v0.1 implementation specification is the newer, complete, executable source and therefore controls implementation details whenever it conflicts with the earlier long-term design. In particular, the final v0.1 spec uses:

- repository root `production-llm-systems-lab/`;
- Python package `llmlab`;
- vLLM as the only implemented engine in v0.1;
- fixed-concurrency / closed-loop load generation for the v0.1 KV-capacity question;
- tasks T00–T24 as the implementation and release sequence.

The earlier design used the historical names `llm-serving-lab` / `servelab` and described a broader v1 architecture. Those names are retained here only as provenance; **do not rename the v0.1 package away from `llmlab` merely to match the historical design**.

### Source integrity

Recovered source SHA-256:

- long-term design source: `a20314760c203b64dce5c8e3e52edd027a88308a3d20e8854270400d86a443be`
- v0.1 implementation specification source: `4ca61a447d4d159f093d457ce5c25a06143fa9eb7cf8ec46d6a6ed41a5d415ba`

The long-term design source is visibly damaged by pasted-text loss in places and ends during the A1 experiment specification. This canonicalization therefore **does not manufacture B1–B4, C-series, or D-series protocols that are absent from the surviving source**. Where the source supports only architecture-level intent, this file says so explicitly.

---

# Part I — Recovered Long-Term Architecture and Non-Negotiable Design Rules

## 1. Mission

The lab exists to demonstrate the book's engineering method on real serving systems:

> published mechanism → tested analytical model → explicit prediction → measurement on a named setup → residual analysis → engineering decision

The repository is evidence of disciplined reasoning and measurement on bounded test setups. It is **not** evidence that the author operates commercial production systems, GPU fleets, multi-replica serving, or hyperscale infrastructure.

The word **Production** must be scoped in the README's first sentence as **production-relevant measurement**, not a claim of operating production systems.

## 2. Evidence discipline

The recovered design establishes these governing rules:

1. **Predict, then measure.** Predictions and hypotheses are committed before GPU runs when the experiment requires preregistration.
2. **Publish the residual.** Disagreement between analytical prediction and observation is evidence, not something to hide.
3. **Own the measurement, not the engine.** The project builds the load generator, metrics, telemetry, provenance, analysis and reporting; serving engines are treated as external systems.
4. **One evidence chain.** Manifests lead to immutable runs, then derived data, then figures/tables/docs. Derived evidence is reproducible from raw evidence and is not edited by hand.
5. **Evidence labels are mandatory.** A result must be distinguished as analytical/modelled, simulated/mock, measured, or externally cited. Mock output must never masquerade as measurement.
6. **Claims are scoped.** Claims are limited to the engine, model, GPU, workload and configuration actually tested.
7. **Refuted hypotheses remain visible.** A failed prediction is retained in the claims/evidence record.
8. **The lab tests the book.** Book equations/examples become executable checks; contradictions discovered by measurement go into an errata path rather than being concealed.
9. **Do not infer production-scale authority.** The design explicitly excludes production-fleet experience, universal engine rankings, universal speedups, untested hardware/model generalization, output-quality conclusions when quality is not measured, and security guarantees.

## 3. Long-term repository architecture recovered from Claude

The surviving long-term design describes these major subsystems:

- **models** — normalize model configuration into a typed `ModelSpec`; refuse unsupported architectures rather than guessing.
- **calc** — implement book equations as code, including KV accounting, roofline/traffic models and occupancy-related analytical calculations.
- **workload** — deterministic/versioned workload materialization with explicit token lengths, arrival structure, shared-prefix structure where applicable, and stable hashes.
- **loadgen** — replay requests and record request/chunk timing; the broader design anticipated multiple drivers/modes, while v0.1 intentionally implements only closed-loop fixed concurrency.
- **telemetry** — time-aligned engine metrics, GPU/NVML signals and host telemetry.
- **engines** — reproducible engine start/stop and parsing of engine self-reports; engines remain external rather than imported into the library environment.
- **metrics** — canonical metric definitions shared by real and mock/simulated runs.
- **sim** — a later scheduling/memory simulator whose outputs are intended to use the same result shape as real runs.
- **capacity** — a later capacity-planning layer using measured curves/calibrated models and declared targets/prices.
- **runner** — experiment manifests, parameter grids, repetitions, resume logic, provenance and execution control.
- **report** — generated figures/tables with evidence labels and provenance.
- **testing.mock_server** — deterministic fake streaming server with scripted timing/failure behavior for harness validation.
- **claims** — a ledger linking conclusions to evidence, assumptions, experiment IDs and status; an errata path records contradictions with book text.

The earlier design also named these experiment families:

- `A_kv_cache/`
- `B_serving/`
- `C_scheduling_sim/`
- `D_capacity/`

and referred at architecture level to later experiments including B1–B4, C3 and D4. **Their complete protocols do not survive in the recovered long-term source.** Do not invent them from the names alone.

## 4. Long-term data/evidence topology

The recovered architecture establishes the following one-way evidence flow:

```text
configuration + workload + manifest
            ↓
       preregistration
            ↓
      experiment/run
            ↓
       immutable raw data
            ↓
 canonical metrics / analysis
            ↓
       derived evidence
            ↓
 figures / tables / story
            ↓
 claims ledger / engineering decision
```

The broader design expects:

```text
results/runs/<run_id>/
results/derived/<experiment>/
results/index.parquet
results/archive_manifest.json
claims/ledger.yaml
claims/errata.yaml
```

and documentation for methodology, reproducibility, run cards, ADRs, limitations and book mapping.

The exact v0.1 tree in Part II supersedes this broader layout wherever the two differ.

## 5. Long-term scope boundaries recovered from Claude

The earlier design explicitly kept the following outside the first serving-lab scope or outside v1:

- writing a new inference engine or replacing vLLM;
- a commercial serving platform;
- dashboards as a product surface;
- multi-node or split-prefill/decode serving;
- quantization and speculative decoding in this serving-lab release (reserved for a later Chapter 5 lab);
- generic LLM quality evaluation inside this repository;
- claims of universal engine superiority or reproduction of headline paper multipliers;
- production-fleet or multi-replica operational experience claims;
- security guarantees;
- automatic cost claims unless prices are explicitly supplied and scoped.

The long-term design also imposed bounded ambition: one main engine, one secondary engine, at most three models and four workload types in v1. Because the detailed post-v0.1 protocols are missing, these are roadmap constraints, **not permission to invent the missing experiment definitions**.

## 6. Two book corrections discovered during Claude's design work

These are not measured results; they are design-review findings that the lab is intended to verify or constrain:

1. **Prefix-cache/preemption claim:** the earlier chapter statement that a cached but rarely reused prefix can directly force active requests into preemption is suspected to be wrong for vLLM. The recovered design notes that unused cached blocks are in the free/evictable pool and that B3 was intended to test the actual behavior. Do not patch the book from assumption alone; verify against the pinned vLLM behavior before finalizing the correction.
2. **45 ms TPOT hardware feasibility:** the earlier design observes that a 7–8B FP16/BF16-class model may have a memory-bandwidth lower bound that rules out a low-bandwidth GPU for a 45 ms TPOT target. Treat this as an analytical pre-screen that must be recomputed from the pinned model/hardware values, not as a universal hardware verdict.

---

# Part II — Complete Claude v0.1 Implementation Specification

**Authority:** This section is the complete recovered final v0.1 implementation specification. It is preserved as the executable implementation contract. Do not reduce it to an MVP, silently skip later tasks, or substitute a new architecture.

Production LLM Systems Lab: v0.1 Implementation Specification

Release: v0.1.0 · Status: implementation spec, final · Date: 2026-09-25
Package: llmlab (installed from source; not published to PyPI in v0.1)
Effort: about 4–5 focused weeks, plus about 10 GPU-hours

1. Architecture summary

1.1 The vertical slice

Question: How many long-context requests can one GPU serve at once, and what limits that number?

Stage: Theory
What v0.1 does: Uses the KV-cache formula and paged block accounting to compute
each request's average KV footprint over its lifetime (ḡ). From that, it
derives a predicted concurrency range where the server starts queueing, a
lower bound on TPOT set by memory bandwidth, and a model of how prefill work
slows decoding.
────────────────────────────────────────
Stage: Implementation
What v0.1 does: The formulas become tested code. vLLM runs on one 24 GB GPU. A
small, tested load generator keeps exactly N requests in flight at all times,
 starting a new one as soon as one finishes.
────────────────────────────────────────
Stage: Measurement
What v0.1 does: TTFT, ITL, TPOT, end-to-end latency, throughput, GPU memory,
and vLLM's own KV-usage, queue and preemption counters.
────────────────────────────────────────
Stage: Comparison
What v0.1 does: Predictions are written and committed before the runs. They're
compared with observations under pass/fail criteria set in advance, and each
gap is explained.
────────────────────────────────────────
Stage: Engineering decision
What v0.1 does: Sets the admission cap for each replica, identifies which
resource is the bottleneck, and names which changes would raise capacity (and
 which wouldn't).

It covers four experiments from the long-term plan:
- A1: the KV calculator.
- A3: checking the calculator against the KV capacity vLLM reports at startup.
- A4: KV saturation as concurrency rises.
- B0: accuracy of the load generator itself.

1.2 Data flow

configs/{models,hardware,engines}          experiments/A_kv_cache/<exp>/{manifest.yaml, PREREG.md}
        │                                                   │
models.py ─► calc/kv.py, calc/roofline.py ─► llmlab predict ─► results/derived/<exp>/predictions_pre.json
        │                                                         (committed before any GPU run)
engines/vllm.py ─► startup report ─► A3 reconciliation ─► KV capacity C
        │
workload.py ─► loadgen.py (closed-loop, N in flight) ◄──► vLLM /v1/completions (SSE)
                    │                 telemetry.py ◄── vLLM /metrics, NVML
                    ▼                        ▼
          results/runs/<exp>/<run_id>/   (immutable; one directory per run)
                    │
       metrics.py ─► analysis/kv_capacity.py ─► derived CSVs, claims_eval.json, decision.json
                    │
                report.py ─► docs/figures/*.svg ─► docs/stories/01-kv-capacity.md

1.3 What v0.1 builds from the long-term architecture

┌──────────────────────┬──────────────────────────────────────────────────┐
│ Long-term component  │                   v0.1 status                    │
├──────────────────────┼──────────────────────────────────────────────────┤
│ models, calc (KV,    │ Implemented. MHA/GQA/MQA only; anything else is  │
│ roofline)            │ rejected with UnsupportedArchitecture.           │
├──────────────────────┼──────────────────────────────────────────────────┤
│                      │ Minimal. Fixed-shape requests with token-ID      │
│ workload             │ prompts. Mixed workloads, prefix structures and  │
│                      │ trace replay come later.                         │
├──────────────────────┼──────────────────────────────────────────────────┤
│ loadgen              │ Minimal. Closed-loop (fixed concurrency) only.   │
│                      │ Open-loop (fixed arrival rate) is v0.2.          │
├──────────────────────┼──────────────────────────────────────────────────┤
│ metrics, telemetry,  │                                                  │
│ provenance, runner,  │ Implemented for this slice.                      │
│ report               │                                                  │
├──────────────────────┼──────────────────────────────────────────────────┤
│ engines              │ vLLM only. No shared engine interface until a    │
│                      │ second engine exists.                            │
├──────────────────────┼──────────────────────────────────────────────────┤
│ analysis             │ One module for this story: kv_capacity.          │
├──────────────────────┼──────────────────────────────────────────────────┤
│ sim, capacity,       │ Later (roadmap). No placeholder packages or      │
│ llama.cpp engine,    │ empty directories are created now.               │
│ B1–B4, C, D          │                                                  │
└──────────────────────┴──────────────────────────────────────────────────┘

1.4 Design rules

1. Rule of two. No abstract base class, plugin system or registry until a second implementation exists. A module becomes a package when it passes about 400 lines or gets a second implementation.
2. Predictions come first. predictions_pre.json and PREREG.md are committed before the first GPU run, and llmlab validate checks the git timestamps.
3. One set of metric definitions. metrics.py evaluates real runs and mock runs alike.
4. Every figure carries its evidence label. Figures are built only from derived data and have provenance files. Output from the fake server is labelled MOCK and is blocked from results/.
5. Units are explicit. Bytes are int, time is int nanoseconds internally, and GiB-vs-GB labels are applied only when displaying.

1.5 Reference setup

┌──────────┬──────────────────────────────────────────────────────────────┐
│   Item   │                         v0.1 choice                          │
├──────────┼──────────────────────────────────────────────────────────────┤
│ GPU      │ One 24 GB, roughly 600 GB/s GPU (A10G-class). The actual     │
│          │ device is recorded from NVML.                                │
├──────────┼──────────────────────────────────────────────────────────────┤
│ Model    │ Qwen/Qwen2.5-7B-Instruct, BF16, pinned by commit hash in T13 │
│          │  (Apache-2.0, ungated)                                       │
├──────────┼──────────────────────────────────────────────────────────────┤
│ Engine   │ vLLM, exact version pinned in T13. Runs in its own           │
│          │ environment (Docker digest or locked venv).                  │
├──────────┼──────────────────────────────────────────────────────────────┤
│ Workload │ Fixed shape: 2,048-token prompt, 1,024-token output, prefix  │
│          │ caching off                                                  │
├──────────┼──────────────────────────────────────────────────────────────┤
│          │ Rough pre-run estimates say KV capacity runs out first on    │
│ Why this │ this GPU while prefill compute is still around 25–30% busy,  │
│  shape   │ so the saturation point should be clearly caused by memory.  │
│          │ Workloads where prefill compute is the bottleneck are v0.2.  │
├──────────┼──────────────────────────────────────────────────────────────┤
│ GPU      │ About 1 GPU-hour for T13, 0.5 for A3, 7–8 for A4, so about   │
│ budget   │ 10 in total                                                  │
└──────────┴──────────────────────────────────────────────────────────────┘

2. Repository tree (v0.1)

production-llm-systems-lab/
├── README.md
├── CLAIMS.md                         # generated from claims/ledger.yaml
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md                   # incl. how to contribute independent reproduction runs
├── LICENSE                           # Apache-2.0 (code)
├── LICENSE-DATA                      # CC-BY-4.0 (results, figures, docs)
├── Makefile
├── pyproject.toml
├── uv.lock
├── .gitignore                        # .cache/, build/, results/**/chunks.parquet
├── .github/workflows/
│   ├── ci.yml
│   └── results-integrity.yml
├── configs/
│   ├── models/qwen2.5-7b-instruct.yaml
│   ├── hardware/a10g-24gb.yaml
│   └── engines/
│       ├── vllm.yaml
│       ├── vllm-metrics.yaml         # logical name → candidate metric names (version-keyed)
│       └── vllm-requirements.lock    # pip freeze of the engine environment (from T13)
├── experiments/
│   ├── A_kv_cache/
│   │   ├── A1_calculator/{manifest.yaml, README.md}
│   │   ├── A3_engine_reconciliation/{manifest.yaml, PREREG.md, README.md}
│   │   └── A4_kv_saturation/{manifest.yaml, PREREG.md, README.md}
│   └── B_serving/
│       └── B0_harness_accuracy/{manifest.yaml, README.md}
├── src/llmlab/
│   ├── __init__.py                   # __version__
│   ├── cli.py
│   ├── units.py
│   ├── schemas.py
│   ├── models.py
│   ├── calc/
│   │   ├── __init__.py
│   │   ├── kv.py
│   │   └── roofline.py
│   ├── workload.py
│   ├── loadgen.py
│   ├── metrics.py
│   ├── telemetry.py
│   ├── engines/
│   │   ├── __init__.py
│   │   └── vllm.py
│   ├── provenance.py
│   ├── runner.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── kv_capacity.py
│   ├── report.py
│   └── testing/
│       ├── __init__.py
│       └── mock_server.py
├── claims/ledger.yaml
├── results/
│   ├── runs/<exp>/<run_id>/          # committed except chunks.parquet
│   ├── engines/<instance_id>/        # engine.log, startup_report.json, version.json
│   ├── derived/<exp>/                # CSV/JSON + *.provenance.json
│   └── archive_manifest.json         # SHA-256 + DOI of archived raw chunks
├── docs/
│   ├── stories/01-kv-capacity.md     # narrative + generated blocks
│   ├── figures/<exp>/*.svg           # + *.provenance.json
│   ├── methodology/metrics.md
│   ├── methodology/kv-accounting.md
│   ├── reproduce.md
│   ├── run-cards/A3-A4-reference-gpu.md
│   ├── limitations.md
│   ├── roadmap.md                    # full long-term architecture; v0.1 marked
│   └── adr/
│       ├── 0001-engine-instance-per-repetition.md
│       ├── 0002-closed-loop-for-capacity.md
│       ├── 0003-storage-split.md
│       ├── 0004-rule-of-two.md
│       └── 0005-token-id-prompts-forced-lengths.md
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── hf_configs/{qwen2.5-7b.json, llama3-8b-shape.json, mha-example.json,
    │   │               falcon-mqa.json, deepseek-mla.json, qwen2-sliding-disabled.json}
    │   ├── tokenizer_added_tokens.json
    │   └── vllm/<version>/{startup.log, metrics_idle.txt, metrics_busy.txt, version.json}
    ├── unit/{test_units.py, test_models.py, test_kv.py, test_roofline.py,
    │         test_workload.py, test_metrics.py, test_telemetry_parse.py,
    │         test_vllm_parse.py, test_provenance.py, test_schemas.py}
    ├── property/{test_kv_properties.py, test_metric_identities.py}
    ├── integration/{test_loadgen_mock.py, test_runner_mock_e2e.py,
    │                test_analysis_ground_truth.py, test_report_golden.py}
    └── golden/{a1_kv_table.csv, ch4_examples.yaml}

3. Detailed implementation specification

3.1 Conventions

- Units: bytes as int; time as int nanoseconds from time.perf_counter_ns() (monotonic). Wall-clock UTC appears only in metadata. Anything shown to a person carries an explicit unit (GiB or GB).
- IDs:
  - Run: {exp}-{config}-r{rep}-n{N}-{YYYYMMDDTHHMMZ}-{4 hex}
  - Engine instance: {exp}-{config}-r{rep}-{UTC}-{4 hex}
- Errors: library code raises typed exceptions (UnsupportedArchitecture, EngineStartupError, ParseError, ManifestError). The CLI maps them to exit codes. Nothing fails silently: a missing essential metric makes the run invalid with a stated reason.
- Logging: logging only; no print in the library. The CLI prints summaries.
- Formula docstrings: every function implementing a formula states the formula and the book section it comes from.

3.2 Configuration and experiment formats

All YAML is validated by pydantic models in schemas.py. Unknown keys are errors (extra="forbid").

# configs/models/qwen2.5-7b-instruct.yaml
id: qwen2.5-7b-instruct
hf_repo: Qwen/Qwen2.5-7B-Instruct
revision: "<40-hex commit, pinned in T13>"
license: Apache-2.0
serving_dtype: bfloat16          # float16 on GPUs without BF16
kv_cache_dtype_bytes: 2

# configs/hardware/a10g-24gb.yaml
id: a10g-24gb
name: NVIDIA A10G
peak_mem_bw_Bps: 6.0e11          # datasheet ceiling; author verifies against the source
peak_dense_fp16_flops: <datasheet value>
sources: ["<datasheet URL>"]
notes: "Datasheet ceilings, not achievable rates. Usable memory comes from NVML at run time."

# configs/engines/vllm.yaml
name: vllm                       # "mock" is reserved for testing.mock_server
version: "<exact, pinned in T13>"
command: ["vllm", "serve"]       # or a docker run prefix ending in an image@sha256 digest
host: 127.0.0.1
port: 8000
startup_timeout_s: 1200
env: {}
base_args:                       # ⚠ flag names verified in T13
  seed: 0
  enable_prefix_caching: false
  max_num_seqs: 256              # must be ≥ 2 × the largest N, so it never limits concurrency
  max_num_batched_tokens: 2048   # set explicitly; defaults change between versions
  block_size: 16
metrics_map: configs/engines/vllm-metrics.yaml
log_parser: vllm-startup-v1

# configs/engines/vllm-metrics.yaml   (first matching name wins; ⚠ names verified in T13)
kv_usage:          [vllm:kv_cache_usage_perc, vllm:gpu_cache_usage_perc]
running:           [vllm:num_requests_running]
waiting:           [vllm:num_requests_waiting]
preemptions:       [vllm:num_preemptions_total, vllm:num_preemptions]
prompt_tokens:     [vllm:prompt_tokens_total]
generation_tokens: [vllm:generation_tokens_total]
cache_config_info: [vllm:cache_config_info]
essential: [kv_usage, running, waiting]

# experiments/A_kv_cache/A4_kv_saturation/manifest.yaml
experiment_id: A4
title: "KV saturation under closed-loop concurrency"
prereg: PREREG.md
model: configs/models/qwen2.5-7b-instruct.yaml
hardware: configs/hardware/a10g-24gb.yaml
engine: configs/engines/vllm.yaml
workload: {kind: fixed_shape, prompt_tokens: 2048, output_tokens: 1024, seed: 20260925}
configs:                                   # grids finalized from `llmlab predict` in T21
  - name: pool-full
    engine_args: {gpu_memory_utilization: 0.90, max_model_len: 4096}
    concurrency: [1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 48]
  - name: pool-reduced
    engine_args: {gpu_memory_utilization: 0.80, max_model_len: 4096}
    concurrency: [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
repetitions: 3                             # one fresh engine instance per (config, repetition)
order: {randomize: true, seed: 7}          # N order shuffled within each engine instance
client: {warmup_s: 60, stagger_s: auto, window_min_s: 180, window_max_s: 480,
         min_completed: 150, request_timeout_s: 900, drain_timeout_s: 900}
idle_wait: {kv_usage_max: 0.005, timeout_s: 180}
telemetry: {metrics_interval_s: 1.0, nvml_interval_s: 0.2}
gates: {min_tail_samples: 20}
checks: {max_error_rate: 0.01, stability_tolerance: 0.10,
         loop_lag_p99_ms: 5.0, client_cpu_max: 0.80}
decision: {kv_headroom: 0.90}

Experiment format: each experiment is a directory holding manifest.yaml (machine-readable), PREREG.md (hypotheses, predictions, criteria, grid, exclusion rules; committed before runs), and README.md (method, deviations, links to results and story). A3 uses the same manifest schema with concurrency: [], which means startup-only runs.

3.3 Components

units.py: unit discipline

- Responsibility: byte and time constants, parsing, and labelled formatting.
- Interface:

KiB, MiB, GiB = 1024, 1024**2, 1024**3
KB, MB, GB = 10**3, 10**6, 10**9
def parse_bytes(text: str) -> int: ...          # "24GiB", "24 GB", "57344"
def fmt_bytes(n: int, system: Literal["binary", "decimal"] = "binary", digits: int = 2) -> str: ...
def ns_to_s(ns: int) -> float: ...

- Validation: round-trip tests. A regression test reproduces the v8 case study: free slots = 196,608 − 140,480 = 56,128. It also shows how the wrong figure of about 53,000 arises (6.85e9 / 128e3), which is exactly the mix-up fmt_bytes prevents.

schemas.py: typed configs and records

- Responsibility: pydantic v2 models for every file the lab reads or writes, with JSON Schema export (llmlab schema --out docs/schemas/).
- Models:
  - Configs: ModelRef, HardwareProfile, EngineConfig, MetricsMap, WorkloadSpec, ClientConfig, SweepConfig, ExperimentManifest
  - Engine and run records: StartupReport, EngineInstance, RequestRecord, ChunkRecord, CheckResult, RunMetadata
  - Claims and outputs: Claim, ClaimEvaluation, Decision
- Key fields:
  - StartupReport: kv_cache_tokens, num_gpu_blocks, block_size, max_concurrency, max_concurrency_tokens, model_load_bytes, raw_lines, parser_version (all int | None except the last two).
  - RequestRecord: run_id, request_id, worker_id, t_send_ns, t_first_ns, t_last_ns, n_chunks, n_prompt_tokens, n_output_tokens, status ∈ {ok, error, timeout}, error, in_window, ttft_s, e2e_s, tpot_s, itl_p50_s, itl_p95_s, itl_max_s, multi_token_chunks.
  - RunMetadata: identity, manifest_sha256, prereg_commit, code (commit, dirty, llmlab_version), environment, engine (version, full resolved args, instance_id), GPU (name, driver, CUDA driver, total and baseline memory), model (ref, weight_bytes, m_kv_bytes), workload (spec, safe_ids_sha256), timing (UTC start/end, window ns), counts, checks: dict[str, CheckResult], valid, invalid_reasons, evidence ∈ {MEASURED, MOCK}, tier.
- Validation: every model round-trips YAML/JSON; unknown keys are rejected; golden schema files are exported and diffed in CI.

models.py: architecture facts

- Responsibility: turn a Hugging Face config.json plus checkpoint metadata into a ModelSpec, and refuse architectures v0.1 doesn't model.
- Interface:

@dataclass(frozen=True)
class ModelSpec:
    name: str; n_layers: int; n_q_heads: int; n_kv_heads: int; head_dim: int
    hidden_size: int; vocab_size: int; tie_word_embeddings: bool
    attention: Literal["mha", "gqa", "mqa"]
    param_count: int | None; weight_bytes: int | None

class UnsupportedArchitecture(Exception): reason: str

def spec_from_hf_config(config: Mapping[str, Any], *, name: str) -> ModelSpec: ...
def load_model_spec(ref: ModelRef, cache_dir: Path) -> ModelSpec: ...   # network: pinned revision

- Details:
  - Reads the standard keys: num_hidden_layers, num_attention_heads, num_key_value_heads (defaults to the number of attention heads), head_dim (defaults to hidden size ÷ heads), hidden_size, vocab_size, tie_word_embeddings.
  - Raises UnsupportedArchitecture for: MLA (kv_lora_rank), active sliding windows (use_sliding_window: true, or layer_types containing sliding or linear attention), SSM/hybrid model_types, nested text_config, and non-standard key sets (for example Falcon's multi_query).
  - Gotcha, tested: Qwen2 configs contain sliding_window with use_sliding_window: false. That is full attention and must not be rejected.
  - Parameter count and weight bytes come from checkpoint safetensors metadata via huggingface_hub (get_safetensors_metadata ⚠). The fallback is model.safetensors.index.json → metadata.total_size.
- Validation: fixture tests. Qwen2.5-7B must give 28 layers, 4 KV heads and head dimension 128 (GQA). The Llama-3-8B-shaped config must give 32/8/128. MHA must be detected, and the MLA, Falcon and active-sliding configs must be rejected.

calc/kv.py: KV accounting (Ch. 3.4, 4.4, 4.4.1)

- Interface:

def kv_bytes_per_token(spec: ModelSpec, kv_dtype_bytes: int) -> int: ...     # 2·L·n_kv·d_h·b
def blocks_for_tokens(tokens: int, block_size: int) -> int: ...              # ceil(T/q)
def slack_tokens(tokens: int, block_size: int) -> int: ...                   # (q − T mod q) mod q
@dataclass(frozen=True)
class PoolCapacity: num_blocks: int; tokens: int
def pool_capacity(pool_bytes: int, m_kv: int, block_size: int) -> PoolCapacity: ...
def pool_bytes(num_blocks: int, block_size: int, m_kv: int) -> int: ...
def max_concurrency_ratio(capacity_tokens: int, tokens_per_request: int) -> float: ...
def max_concurrency_blocks(num_blocks: int, tokens_per_request: int, block_size: int) -> int: ...
def g_decode(prompt: int, output: int, block_size: int) -> float: ...
    # (1/(O−1)) Σ_{k=1}^{O−1} q·ceil((P+k)/q), summed exactly, not approximated
def g_max(prompt: int, output: int, block_size: int) -> int: ...            # q·ceil((P+O)/q)
def g_semi_empirical(prompt: int, output: int, block_size: int,
                     ttft_s: float, tpot_s: float) -> float: ...
    # [TTFT·(q·ceil(P/q)/2) + (O−1)·TPOT·g_decode] / [TTFT + (O−1)·TPOT]
def predicted_occupancy(n: int, g: float, capacity_tokens: int) -> float: ...  # min(N·g, C)
def predicted_saturation_interval(capacity_tokens: int, g_mean: float, g_peak: int
                                  ) -> tuple[float, float]: ...                 # (C/g_max, C/ḡ)

- Details: all pure and integer-exact where the maths allows. The docstrings explain why the saturation interval has these bounds:
  - Below C/g_max, the requests fit even if every one is at full length.
  - Above C/ḡ, average demand is larger than capacity.
- Validation:
  - Golden tests with all Chapter 4 examples (tests/golden/ch4_examples.yaml):
    - 131,072 B/token
    - 512 MiB per sequence
    - 7.3828 / 9.7656 / 17.1484 GiB
    - 56,128 free slots, and 31,552 after reserving 3 GiB
    - exercise values: 81,920 B/token and 1.25 GiB
    - slack values for T = 1, 16, 17, 31, 32, 33
  - g_decode(2048, 1024, 16) = 2,567.5 exactly.
  - Property tests: bytes grow monotonically with layers, KV heads and head dimension; 0 ≤ slack ≤ q−1; g_decode lies between P and g_max.

calc/roofline.py: bounds and the calibrated serving model (Ch. 3.2–3.3, 4.1.1, 4.3)

- Interface:

def decode_weight_bytes(spec: ModelSpec, weight_dtype_bytes: int) -> int: ...
    # param_count·b − (vocab·hidden·b if not tied): the input-embedding table is gathered, not streamed
def tpot_floor_s(decode_weight_bytes: int, kv_bytes_read: float, peak_bw_Bps: float) -> float: ...
def ttft_floor_s(param_count: int, prompt_tokens: int, peak_flops: float) -> float: ...  # 2·N·P / peak
@dataclass(frozen=True)
class Calibration: bw_eff_Bps: float; t_prefill_s: float     # from N=1 runs only
def calibrate(tpot_1: float, ttft_1: float, w_dec: int, g: float, m_kv: int) -> Calibration: ...
@dataclass(frozen=True)
class ServingPrediction: n: int; tpot_s: float; ttft_s: float; lifetime_s: float
                         req_per_s: float; out_tok_per_s: float; prefill_fraction: float
def predict_closed_loop(n: int, cal: Calibration, w_dec: int, g: float, m_kv: int,
                        prompt: int, output: int, token_budget: int) -> ServingPrediction: ...

- Model, documented with its assumptions in kv-accounting.md and valid only below saturation. For each N, the equations are:

  - t_dec(N) = (W_dec + N·ḡ·m_kv) / BW_eff
  - n_c(N) = ⌈P / (B_tok − (N−1))⌉
  - λ_o = λ·(N−1)/N (the rate of other requests' prefills)
  - f_o = λ_o·t_pf
  - S = (1 − f_o)/t_dec(N) + λ_o·n_c(N) (tokens per second for one decoding request)
  - TPOT = 1/S
  - TTFT = t_pf + t_dec(N)
  - L = TTFT + (O−1)·TPOT
  - λ = N/L

  These are circular (λ depends on TPOT, which depends on λ), so the code solves for λ by bisection. Sanity checks: at N = 1, TPOT = t_dec(1) exactly. Output throughput is λ·O.
- Assumptions (stated):
  - Prefill cost doesn't depend on how the prompt is chunked.
  - Decode tokens riding along in a mixed batch add no time.
  - No queueing, and all requests have the same shape.
- Validation:
  - Worked value: decode_weight_bytes for Qwen2.5-7B ≈ 14.15 GB, so the floor at 600 GB/s ≈ 23.6 ms.
  - Limits: f_o = 0 at N = 1; TPOT never decreases as N grows; the solver converges to within 1e-9 relative change or raises.
  - Tests against hand-worked examples.

workload.py: deterministic fixed-shape requests

- Interface:

@dataclass(frozen=True)
class Request: request_id: str; prompt_token_ids: tuple[int, ...]; max_tokens: int
def safe_token_ids(tokenizer_json: Mapping[str, Any], vocab_size: int) -> np.ndarray: ...
def fixed_shape_request(spec: WorkloadSpec, safe_ids: np.ndarray, index: int) -> Request: ...
def request_stream(spec: WorkloadSpec, safe_ids: np.ndarray, start: int = 0) -> Iterator[Request]: ...
def safe_ids_digest(safe_ids: np.ndarray) -> str: ...

- Details:
  - Each prompt is drawn from np.random.default_rng([seed, index]), so request i is reproducible without storing any files.
  - Special and added tokens (from tokenizer.json → added_tokens) are excluded, as are IDs at or above the model vocabulary size.
  - Each run starts at a different start offset, so no two runs reuse the same prompts (prefix caching is off in any case).
- Validation: same seed and index give the same tokens; no special tokens appear; length is exactly P. In integration runs, server-reported prompt_tokens == P for every successful request.

testing/mock_server.py: a fake vLLM with known ground truth

- Responsibility: a fake server that behaves like vLLM from the outside, so the whole pipeline can run with no GPU and recover known answers.
- Interface: python -m llmlab.testing.mock_server --port --capacity-tokens C --ttft-ms --itl-ms [--multi-token-every K] [--error-rate p] [--stall-ms s --stall-every K]. It also provides create_app(cfg) -> aiohttp.web.Application for tests that run the server in the same process.
- Behaviour:
  - Prints vLLM-format startup lines (GPU KV cache size: C tokens, Maximum concurrency for ...), so the real parser is exercised.
  - Serves /health, /v1/models and streaming /v1/completions (with a final usage chunk).
  - Admits requests first-in-first-out, reserving P+O tokens up front, so ground truth is known exactly: g = P+O and N_sat = ⌊C/(P+O)⌋.
  - Serves /metrics using the vLLM metric names (KV usage, running, waiting, token counters).
  - When running in-process, it records its own emission timestamps for B0.
- Validation: its own unit tests check admission order, usage chunks, metrics values and injected faults.

loadgen.py: closed-loop streaming client

- Interface:

@dataclass
class LoadResult:
    requests: list[RequestRecord]; chunks: list[ChunkRecord]
    window_start_ns: int; window_end_ns: int
    loop_lag_p99_ms: float; client_cpu_max: float
async def run_closed_loop(requests: Iterator[Request], base_url: str, model_name: str,
                          concurrency: int, cfg: ClientConfig, run_id: str) -> LoadResult: ...

- Details:
  - Payload to POST /v1/completions (the fields marked ⚠ are vLLM extensions):
    - prompt: token-ID list
    - max_tokens and min_tokens ⚠: both O
    - ignore_eos ⚠: true
    - temperature: 0
    - stream: true
    - stream_options: {"include_usage": true}
  - Workers: N async workers each run send → stream → record → repeat.
  - Staggered start: workers begin at random offsets spread over stagger_s (when set to auto, the pre-run lifetime estimate). This prevents lock-step: identically shaped requests started together would finish together and swing KV occupancy between empty and full.
  - Parsing: SSE data: lines are read until [DONE]. Every chunk that has choices is a timing event, including ones with empty text; the usage-only chunk supplies token counts.
  - Measurement window: warmup lasts warmup_s. The window closes once both window_min_s and min_completed are reached, or at window_max_s, whichever comes first. A request is in the window if it was sent inside it. After the window closes, workers stop sending and in-flight requests drain; any that don't finish in time are recorded as timeout, so slow requests aren't silently dropped.
  - Client health: a lag monitor (a 10 ms sleep loop) and psutil CPU sampling feed the client-health checks.
- Validation (B0): against the in-process mock server at N ∈ {1, 8, 32, 64}:
  - p99 |TTFT error| and p99 |ITL error| ≤ 1 ms
  - window accounting exact
  - errors and timeouts recorded
  - multi-token chunks flagged

metrics.py: canonical definitions

- Interface:

@dataclass(frozen=True)
class RequestTiming: ttft_s: float; e2e_s: float; tpot_s: float | None
                     itl_s: np.ndarray; multi_token: bool
def request_timing(t_send_ns: int, chunk_t_ns: Sequence[int], n_output_tokens: int) -> RequestTiming: ...
@dataclass(frozen=True)
class Gated: value: float | None; n: int; reason: str | None
def gated_percentile(values: np.ndarray, q: float, min_tail: int) -> Gated: ...  # n·(1−q/100) ≥ min_tail
def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float],
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]: ...
def summarize_run(req: pd.DataFrame, server: pd.DataFrame, gpu: pd.DataFrame | None,
                  window: tuple[int, int], capacity_tokens: int) -> dict[str, Any]: ...

- Definitions (the same text appears in docs/methodology/metrics.md):
  - TTFT = first choice chunk − send.
  - E2E = last chunk − send.
  - TPOT = (E2E − TTFT)/(n_out − 1), where n_out comes from reported usage; undefined if n_out < 2.
  - ITL = gaps between consecutive chunks after the first, pooled across requests for event-level statistics.
  - Request throughput = in-window completions ÷ window length.
  - Output-token throughput, reported two ways: from the client (Σ n_out) and from the server (change in the generation-token counter over the window).
  - KV occupancy = time-mean of the KV-usage gauge × C.
  - Also reported: mean running and waiting counts; change in the preemption counter; GPU memory used (mean and max; this is the engine's reservation, not occupancy), SM clock minimum, mean power, maximum temperature.
  - Percentile gate: p50, p90, p95 and p99 are each reported only if at least min_tail samples lie beyond them; otherwise the value is null with a stated reason.
- Validation:
  - Unit tests on synthetic streams covering 1-token outputs, multi-token chunks, errors and timeouts.
  - Property test: TPOT equals the mean of that request's ITLs when every chunk holds one token (the Chapter 4.1 identity).
  - Bootstrap coverage check on simulated normal data (within ±3% of nominal).

telemetry.py: server metrics and GPU sampling

- Interface:

class PrometheusScraper:
    def __init__(self, url: str, interval_s: float, metrics_map: MetricsMap): ...
    async def run(self, stop: asyncio.Event) -> pd.DataFrame: ...   # t_ns, logical_name, value
    async def snapshot_raw(self) -> str: ...
def parse_metrics_text(text: str, metrics_map: MetricsMap) -> dict[str, float]: ...
def parse_cache_config_info(text: str) -> dict[str, str]: ...        # labels, if exposed ⚠
class NvmlSampler:
    def __init__(self, device_index: int, interval_s: float): ...
    def device_info(self) -> dict[str, Any]: ...
    def run_in_thread(self, stop: threading.Event) -> pd.DataFrame: ...

- Details:
  - Parses with prometheus_client.parser. Raw /metrics text is saved at the start and end of every window.
  - A missing metric that isn't essential produces a failed check; a missing essential one makes the run invalid.
  - The NVML sampler records memory used/total, GPU utilization, SM and memory clocks, power, temperature and throttle reasons. If nvidia-ml-py is missing, GPU telemetry is marked unavailable, and GPU-memory claims can't be evaluated from that run.
- Validation: parser tests on the real captured fixtures from T13 (plus the mock server's output); the NVML sampler is tested against a fake NVML module.

engines/vllm.py: engine lifecycle and self-reports

- Interface:

def build_cli_args(model: ModelRef, args: Mapping[str, Any]) -> list[str]: ...
    # snake_case → --kebab-case; True → --flag; False → --no-flag ⚠; adds --revision
def parse_startup_log(text: str, parser_version: str) -> StartupReport: ...
class VLLMServer:
    def __init__(self, engine: EngineConfig, model: ModelRef, args: Mapping[str, Any],
                 log_path: Path): ...
    def start(self) -> None: ...
    def wait_ready(self, timeout_s: float) -> None: ...     # /health 200, then /v1/models lists model
    def startup_report(self) -> StartupReport: ...          # log + cache_config_info (cross-checked)
    def version(self) -> str: ...                           # /version ⚠, falls back to log line
    def stop(self, timeout_s: float = 60) -> None: ...      # SIGINT → SIGTERM → SIGKILL
    def __enter__(self) -> "VLLMServer": ...; def __exit__(self, *exc: object) -> None: ...

- Details:
  - Logs are written to results/engines/<instance_id>/engine.log.
  - Regexes, keyed by parser_version and checked against fixtures (⚠ exact text verified in T13):
    - GPU KV cache size:\s*([\d,]+)\s*tokens
    - Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x
    - Model loading took\s*([\d.]+)\s*(GiB|GB)
  - If log values and cache_config_info disagree, that is a failed check, not a guess.
  - The runner enforces the required arguments: prefix caching off, max_num_seqs ≥ 2 × the largest N, explicit token budget and block size.
  - This class also launches the mock server when engine.name == "mock"; no separate abstraction exists.
- Validation: fixture parsing tests. A lifecycle test launches the mock via command: [python, -m, llmlab.testing.mock_server] and exercises start → ready → report → stop, including the startup-timeout path.

provenance.py

- Interface:
  - capture_code() -> CodeInfo (commit and dirty flag, via git)
  - capture_env() -> EnvInfo (Python version, platform, CPU, pip freeze file with its SHA-256)
  - new_run_id(...), new_instance_id(...)
  - sha256_file(path), sha256_json(obj)
  - prereg_commit(path) -> tuple[str, datetime] (the last commit touching the PREREG file)
- Validation: tests in a temporary git repo covering the dirty flag, stable hashes and the prereg lookup.

runner.py: experiment execution

- Interface:

def run_experiment(manifest_path: Path, results_dir: Path, *, configs: list[str] | None = None,
                   reps: list[int] | None = None, allow_dirty: bool = False,
                   dry_run: bool = False) -> list[Path]: ...

- Algorithm for each config × repetition:
  a. Start a fresh engine instance, wait for it to be ready, and save its startup report and version.
  b. Send 8 warm-up requests at concurrency 4 and discard them.
  c. For each N in shuffled order (seeded by order.seed + rep):
     - Wait until the engine is idle (KV usage ≤ kv_usage_max, running = waiting = 0), or mark the point as a failed check.
     - Start the scrapers.
     - Call run_closed_loop (skipped in startup-only mode).
     - Stop the scrapers.
     - Run the checks and write the run directory.
  d. Stop the engine.
- Resume: a run is skipped if a valid run directory with the same (manifest hash, config, repetition, N) already exists.
- Refusals: the runner refuses a dirty working tree unless --allow-dirty is passed (and records that it was), and refuses engine.name == "mock" when writing under results/ (mock output goes to build/).
- Checks (all recorded; each marked as invalidating or informational):

| Check                                                                                       | Invalidates the run?   |
|---------------------------------------------------------------------------------------------|------------------------|
| Error rate ≤ 1%                                                                             | Yes                    |
| Usage tokens match P and O for every successful request                                     | Yes                    |
| Essential metrics present                                                                   | Yes                    |
| Client health (loop lag p99 ≤ 5 ms, CPU ≤ 80%)                                              | Yes                    |
| No hardware or thermal throttle reasons                                                     | Yes                    |
| Engine was idle before start                                                                | Yes                    |
| Throughput in the first and second halves of the window within ±10% (when ≥ 40 completions) | Yes                    |
| completed_in_window ≥ min_completed                                                         | No, informational only |

- Validation: a mock end-to-end test (N ∈ {1, 2, 4, 8}, P = 32, O = 16, short windows) produces schema-valid run directories, correct resume behaviour, and a run marked invalid when errors are injected.

analysis/kv_capacity.py: prediction vs observation, then decision

- Interface:

def load_runs(results_dir: Path, experiment_id: str, valid_only: bool = True) -> pd.DataFrame: ...
def reconcile_startup(reports: pd.DataFrame, spec: ModelSpec, m_kv: int,
                      gpu_total_bytes: int) -> pd.DataFrame: ...
def fit_overhead(recon: pd.DataFrame, train_config: str) -> OverheadModel: ...  # predicts held-out pools
def summarize_points(runs: pd.DataFrame) -> pd.DataFrame: ...   # per config×N: median/min/max over reps
def occupancy_comparison(points: pd.DataFrame, capacity: dict[str, int],
                         g: float, g_semi: pd.Series) -> pd.DataFrame: ...
def detect_saturation(points: pd.DataFrame) -> pd.DataFrame: ...
    # per config and indicator: interval [last N without, first N with]
def serving_model_check(points: pd.DataFrame, ...) -> pd.DataFrame: ...  # calibrate on N=1; predict N≥2
def littles_law_check(points: pd.DataFrame) -> pd.DataFrame: ...        # TTFT ≈ TTFT_unsat + waiting/λ
def evaluate_claims(ledger: list[Claim], tables: dict[str, pd.DataFrame]) -> list[ClaimEvaluation]: ...
def derive_decision(points: pd.DataFrame, sat: pd.DataFrame, model_tbl: pd.DataFrame,
                    headroom: float) -> Decision: ...

- Details:
  - A3's non-KV overhead is util × NVML total − weight bytes − pool bytes. It is fitted on one config and used to predict the pool for the others.
  - Saturation indicators: waiting ≥ 0.5 (the primary one), running < 0.95·N, preemptions > 0, KV usage ≥ 0.95.
  - Each claim ID has exactly one evaluator function, registered in a plain dict.
  - Decision rules:
    - The cap is the largest tested N where every valid repetition had waiting < 0.5, zero preemptions, and KV usage ≤ kv_headroom.
    - The binding constraint is kv_capacity if KV usage ≥ 0.9 at the saturation point and the model's prefill fraction there is < 0.5. It is prefill_compute if the prefill fraction ≥ 0.8 before KV usage reaches 0.9. Otherwise it is mixed.
    - The levers are chosen from the binding constraint.
    - The predicted cap is ⌊C·h/ḡ⌋ (h = headroom), reported next to the measured cap.
- Validation: runs against mock results, where the truth is known (N_sat = ⌊C/(P+O)⌋, g = P+O). The recovered saturation interval must contain the true value, occupancy error must be ≤ 2%, and the Little's-law check must pass. Synthetic tables cover every decision branch.

report.py: figures and story

- Interface:
  - make_figures(derived_dir, fig_dir, meta) -> list[Path]
  - render_story(template: Path, derived_dir: Path, out: Path) -> None, which replaces the blocks between <!-- BEGIN:GENERATED name --> and <!-- END:GENERATED --> and leaves hand-written prose untouched.
- Figures (SVG, 6.5 in wide, readable in grayscale through markers and line styles, no title inside the image; caption and evidence label live in the markdown):

| Figure             | Content                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------|
| a3_reconciliation  | Predicted vs reported KV capacity; overhead bars                                                         |
| a4_occupancy       | KV tokens vs N, with the predicted N·ḡ line, capacity line and predicted saturation band                 |
| a4_throughput_ttft | Output tokens per second vs N; TTFT p50 vs N (log scale) with the Little's-law prediction above the knee |
| a4_tpot            | TPOT vs N, with the bandwidth floor, the decode-only KV-read model and the calibrated mixed-batch model  |
| a4_queue           | Running and waiting requests vs N                                                                        |
| a4_timeseries      | One saturated run: KV usage, running, waiting and preemptions over time                                  |

- Provenance: each figure has a *.provenance.json with the script version, SHA-256 of each input file, run IDs, claim IDs, and an evidence label such as MEASURED · vLLM x.y.z · A10G · Qwen2.5-7B-Instruct@abc1234 · 2048/1024 · 3 reps.
- Validation: a golden test on figure data CSVs and generated markdown blocks from a fixed mock dataset (not a pixel comparison).

cli.py: llmlab commands (argparse, no extra dependencies)

┌──────────────────────────────┬──────────────────────────────────────────┐
│           Command            │                  Effect                  │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab kv --model <yaml>     │                                          │
│ [--block-size 16]            │ Prints the KV table.                     │
│ [--kv-dtype-bytes 2] [--pool │                                          │
│  4.5GiB] [--context 3072]    │                                          │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab a1 <manifest>         │ Writes results/derived/A1/kv_table.csv.  │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │ Writes predictions_pre.json              │
│ llmlab predict <manifest>    │ (capacity-dependent parts use A3 derived │
│                              │  data when present).                     │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab run <manifest>        │                                          │
│ [--configs …] [--reps …]     │ Executes the runs.                       │
│ [--allow-dirty] [--dry-run]  │                                          │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab analyze <manifest>    │ Writes derived tables, claims_eval.json  │
│                              │ and decision.json.                       │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab report <manifest>     │ Writes figures and the story.            │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab validate [--results   │ Integrity checks (§3.7).                 │
│ results/]                    │                                          │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab archive --out         │ Bundles raw chunks into a tarball and    │
│ build/archive/               │ updates archive_manifest.json.           │
├──────────────────────────────┼──────────────────────────────────────────┤
│ llmlab schema --out          │ Exports the JSON Schemas.                │
│ docs/schemas/                │                                          │
└──────────────────────────────┴──────────────────────────────────────────┘

3.4 Experiments, predictions and claims

Experiment: A1 calculator
Tier: 0
What runs: llmlab a1 over registered models
Output: derived/A1/kv_table.csv, golden in CI
────────────────────────────────────────
Experiment: B0 harness accuracy
Tier: 0
What runs: In-process mock at N ∈ {1, 8, 32, 64}
Output: derived/B0/harness_accuracy.csv + README table
────────────────────────────────────────
Experiment: A3 reconciliation
Tier: 2
What runs: Startup-only: util ∈ {0.80, 0.90} × max_model_len ∈ {4096, 8192}, 2
repetitions
Output: derived/A3/reconciliation.csv, overhead_model.json
────────────────────────────────────────
Experiment: A4 KV saturation
Tier: 2
What runs: Two pool sizes × 12 N values × 3 repetitions (closed-loop)
Output: derived/A4/* (§3.5)

Illustrative magnitudes only, from assumptions; real predictions are generated by llmlab predict:
- m_kv = 57,344 B (56 KiB)
- ḡ = 2,567.5 tokens, g_max = 3,072 tokens
- If A3 measures C ≈ 80K tokens, the predicted saturation band is N ≈ 26–31.
- TPOT floor ≈ 23.6 ms at 600 GB/s.

Claims (claims/ledger.yaml). Criteria are frozen in PREREG.md before any GPU run:

ID: A1-01
Type: Identity
Statement (scoped to the pinned setup): The calculator reproduces every Ch. 4
example and exercise, including 56,128 free slots
Pass criterion: Exact
────────────────────────────────────────
ID: A1-02
Type: Identity
Statement (scoped to the pinned setup): m_kv(Qwen2.5-7B-Instruct, BF16) =
57,344 B/token
Pass criterion: Exact, from the pinned config
────────────────────────────────────────
ID: A3-01
Type: Implementation
Statement (scoped to the pinned setup): Reported KV tokens = blocks × block
size; the reported max concurrency = C / max_model_len
Pass criterion: Exact / ±0.5%
────────────────────────────────────────
ID: A3-02
Type: Model
Statement (scoped to the pinned setup): Overhead measured on one config
predicts the pool size of held-out configs
Pass criterion: ±5%
────────────────────────────────────────
ID: A4-01
Type: Model
Statement (scoped to the pinned setup): Below saturation, mean KV occupancy =
N·ḡ
Pass criterion: ±10% (analytic ḡ), ±5% (semi-empirical)
────────────────────────────────────────
ID: A4-02
Type: Model
Statement (scoped to the pinned setup): Engine queueing begins inside [C/g_max,
C/ḡ] (± one grid step); the reduced pool's knee scales with its capacity
ratio
Pass criterion: Interval overlap; ratio ±15%
────────────────────────────────────────
ID: A4-03
Type: Bound
Statement (scoped to the pinned setup): TPOT(1) ≥ (W_dec + ḡ·m_kv) / peak
bandwidth; achieved bandwidth fraction reported
Pass criterion: Bound holds, otherwise it's a measurement error
────────────────────────────────────────
ID: A4-04
Type: Calibrated model
Statement (scoped to the pinned setup): Calibrated on N = 1 only, the
mixed-batch model predicts TPOT and output throughput for unsaturated N ≥ 2
Pass criterion: Median absolute residual ≤ 20%
────────────────────────────────────────
ID: A4-05
Type: Model
Statement (scoped to the pinned setup): Above saturation, throughput gains ≤ 5%
per grid step and TTFT ≈ TTFT_unsat + waiting/λ
Pass criterion: ≤ 5%; ±20%
────────────────────────────────────────
ID: B0-01
Type: Harness
Statement (scoped to the pinned setup): Timing error vs mock ground truth at N
≤ 64
Pass criterion: p99 ≤ 1 ms

Decision output (decision.json, filled into the story): "For {model} on {GPU} with {engine}, serving {P}-token prompts and {O}-token outputs with prefix caching off, cap each replica at {cap} concurrent requests. Beyond {N_sat}, more concurrency raised TTFT by about {slope} s per request while throughput grew ≤ {gain}%. The bottleneck is {binding} (prefill used {f}% of GPU time at the cap), so the levers are {levers}. Before any load was applied, the model predicted the knee at {pred}; the measured knee was {obs}."
- What would change this decision: prefix sharing (which lowers ḡ), mixed request lengths, a different token budget, GPU or precision.
- What it doesn't support: meeting latency targets, output quality, behaviour across multiple replicas.

3.5 Output formats

Per run: results/runs/<exp>/<run_id>/

┌────────────────────────┬───────────────────┬───────────┬───────────────┐
│          File          │      Format       │  In git?  │   Contents    │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│                        │                   │           │ Identity,     │
│ run.json               │ JSON              │ Yes       │ provenance,   │
│                        │ (RunMetadata)     │           │ checks,       │
│                        │                   │           │ validity      │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│ requests.parquet       │ Parquet           │ Yes       │ One row per   │
│                        │ (RequestRecord)   │           │ request       │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│                        │ Parquet:          │ No,       │               │
│ chunks.parquet         │ request_id, seq,  │ archived  │ Raw event     │
│                        │ t_ns (relative),  │ per       │ timing        │
│                        │ text_len          │ release   │               │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│ server_metrics.parquet │ t_ns, name, value │ Yes       │ 1 Hz samples  │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│ gpu.parquet            │ NVML samples      │ Yes       │ 5 Hz samples  │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│ metrics_start.txt,     │ Prometheus text   │ Yes       │ Raw audit     │
│ metrics_end.txt        │                   │           │ snapshots     │
├────────────────────────┼───────────────────┼───────────┼───────────────┤
│                        │ {name: {passed,   │           │ Validity      │
│ checks.json            │ value, threshold, │ Yes       │ details       │
│                        │  invalidating}}   │           │               │
└────────────────────────┴───────────────────┴───────────┴───────────────┘

Per engine instance: results/engines/<instance_id>/: engine.log, startup_report.json, version.json, resolved_args.json.

Derived data: results/derived/

┌───────────┬──────────────────────────────────────────────────────────────┐
│ Directory │                            Files                             │
├───────────┼──────────────────────────────────────────────────────────────┤
│ A1/       │ kv_table.csv                                                 │
├───────────┼──────────────────────────────────────────────────────────────┤
│ A3/       │ reconciliation.csv, overhead_model.json                      │
├───────────┼──────────────────────────────────────────────────────────────┤
│           │ predictions_pre.json, points.csv, summary.csv,               │
│ A4/       │ occupancy.csv, saturation.csv, serving_model.csv,            │
│           │ littles_law.csv, claims_eval.json, decision.json             │
├───────────┼──────────────────────────────────────────────────────────────┤
│ B0/       │ harness_accuracy.csv                                         │
└───────────┴──────────────────────────────────────────────────────────────┘

Every derived file has a *.provenance.json listing its inputs and their SHA-256.

Story: docs/stories/01-kv-capacity.md has nine sections: question, theory, predictions (with commit and timestamp), implementation, measurement (figures), comparison (claims table), engineering decision, limitations and threats to validity, reproduce.

3.6 Documentation files

File: README.md
Purpose: Readable in 3 minutes. First sentence scopes the name. Then: what v0.1
shows and doesn't, the occupancy figure, five findings with claim IDs,
quickstart (Tier 0 tests, make demo-mock, Tier 2 run card), repo map,
citation.
────────────────────────────────────────
File: docs/methodology/metrics.md
Purpose: Metric definitions, window rules, percentile gates, checks, telemetry
pitfalls (NVML utilization and memory are not occupancy).
────────────────────────────────────────
File: docs/methodology/kv-accounting.md
Purpose: Formulas, derivation of ḡ, saturation band, bounds, the calibrated
mixed-batch model, every assumption and when it breaks.
────────────────────────────────────────
File: docs/reproduce.md
Purpose: Commands per tier; which results should reproduce qualitatively vs
exactly.
────────────────────────────────────────
File: docs/run-cards/A3-A4-reference-gpu.md
Purpose: Step by step: provision, engine environment, pre-download, predict,
run, validate, commit. Includes expected GPU-hours.
────────────────────────────────────────
File: docs/limitations.md
Purpose: Single GPU, single model, fixed request shape, closed-loop only, no
quality checks, no multi-replica behaviour.
────────────────────────────────────────
File: docs/roadmap.md
Purpose: Full long-term architecture (A–D, simulator, capacity planning, later
labs), with v0.1 highlighted.
────────────────────────────────────────
File: docs/adr/0001–0005
Purpose: Architecture decisions: engine instance per repetition; closed-loop
load for capacity questions; git vs archive storage split; rule of two;
token-ID prompts and forced lengths, including their realism cost.
────────────────────────────────────────
File: experiments/**/PREREG.md, README.md
Purpose: Per experiment (§3.2).
────────────────────────────────────────
File: CONTRIBUTING.md, CHANGELOG.md, CITATION.cff, CLAIMS.md
Purpose: Standard files; CLAIMS.md is generated.

3.7 Testing requirements and CI

┌─────────────────┬───────────────────────────────────────────────────────┐
│      Layer      │                   Required coverage                   │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │ Every public function in units, models, calc,         │
│ Unit            │ workload, metrics, telemetry parsers, vLLM parsers,   │
│                 │ provenance, schemas                                   │
├─────────────────┼───────────────────────────────────────────────────────┤
│ Property        │ KV growth and bounds, slack bounds, the g_decode      │
│ (Hypothesis)    │ range, the TPOT = mean-ITL identity, serving-model    │
│                 │ limits                                                │
├─────────────────┼───────────────────────────────────────────────────────┤
│ Golden          │ Chapter 4 examples, the A1 table, report data and     │
│                 │ blocks from the fixed mock dataset                    │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │ Load generator vs mock (B0 thresholds); runner end to │
│ Integration     │  end on mock; analysis recovers mock ground truth;    │
│                 │ engine lifecycle via mock                             │
├─────────────────┼───────────────────────────────────────────────────────┤
│ Coverage        │ ≥ 90% on calc, metrics, analysis; ≥ 80% overall.      │
│                 │ GPU-only paths are excluded with a stated reason.     │
├─────────────────┼───────────────────────────────────────────────────────┤
│ Types           │ mypy --strict on units, schemas, models, calc,        │
│                 │ metrics, analysis; standard mode elsewhere            │
└─────────────────┴───────────────────────────────────────────────────────┘

CI (ci.yml, every push and PR, target under 10 minutes): ruff check, ruff format --check, mypy, pytest (unit, property, golden, integration), make a1 compared against golden, a fast make demo-mock, and llmlab validate.

results-integrity.yml (PRs touching results/, claims/ or docs/): runs llmlab validate, which checks:
- every run.json against the schema;
- that there are no mock runs under results/;
- that each run links to an existing engine-instance directory;
- that derived files match the SHA-256 of their inputs;
- that every figure has provenance and a label;
- that every claim has an evaluator, and that its status matches claims_eval.json or is proposed;
- that the PREREG commit is earlier than the first run's start;
- that the story's generated blocks are up to date.

3.8 Dependencies

┌────────────┬────────────────────────────────────────────────────────────┐
│   Group    │ Packages (lower bounds in pyproject.toml; exact versions   │
│            │                        in uv.lock)                         │
├────────────┼────────────────────────────────────────────────────────────┤
│            │ pydantic ≥2.6, pyyaml, numpy, pandas ≥2.2, pyarrow,        │
│ Runtime    │ aiohttp, matplotlib, huggingface_hub, prometheus_client,   │
│            │ psutil                                                     │
├────────────┼────────────────────────────────────────────────────────────┤
│ Optional   │ nvidia-ml-py                                               │
│ gpu        │                                                            │
├────────────┼────────────────────────────────────────────────────────────┤
│ Optional   │ uvloop                                                     │
│ fast       │                                                            │
├────────────┼────────────────────────────────────────────────────────────┤
│ Dev        │ pytest, pytest-asyncio, hypothesis, ruff, mypy,            │
│            │ types-PyYAML                                               │
└────────────┴────────────────────────────────────────────────────────────┘

- Python: ≥ 3.11, with 3.12 as the reference version.
- vLLM is never a dependency of llmlab. It runs in its own environment, pinned by configs/engines/vllm-requirements.lock or an image digest.

4. Cursor task list

Rules for Cursor (paste these with every task):
1. Implement only what the task specifies; ask before adding dependencies, abstractions or files not listed.
2. Bytes are int, time is int ns, and display units are always labelled.
3. Library code has no side effects outside runner, engines, loadgen and telemetry, and uses logging rather than print.
4. Every formula has a docstring giving the formula and its book section.
5. Tests come with the code in the same task; ruff, mypy and pytest must pass.
6. Never write mock output under results/. Never edit derived files by hand.
7. One task per PR with a conventional commit message. Don't squash history, because the commit history is part of the evidence.

---

T00: Repository skeleton (first commit)
- Objective: a CI-verified empty package and the governance documents.
- Files: §5.
- Details:
  - pyproject with the dependency groups and tool settings (ruff, mypy, pytest).
  - Makefile targets setup lint type test a1 demo-mock validate, with unimplemented ones printing "not implemented".
  - __version__ = "0.1.0.dev0".
- Acceptance: make setup lint type test passes locally and in CI; the README states scope and non-claims.
- Dependencies: none.

T01: units.py
- Objective: unit constants, parsing and formatting.
- Files: src/llmlab/units.py, tests/unit/test_units.py.
- Details: as in §3.3, including the 56,128 vs ~53,000 regression test.
- Acceptance: all tests pass; mypy --strict is clean.
- Dependencies: T00.

T02: schemas.py and config loading
- Objective: typed configs and records.
- Files: src/llmlab/schemas.py, tests/unit/test_schemas.py, docs/schemas/ (exported).
- Details:
  - All models in §3.3 with extra="forbid".
  - A load_yaml(path, Model) helper.
  - ExperimentManifest resolves relative config paths.
  - llmlab schema export.
- Acceptance: round-trip tests pass; unknown keys are rejected; exported schemas are committed and diffed in CI.
- Dependencies: T01.

T03: Config files
- Objective: model, hardware and engine configs, plus the metrics map.
- Files: configs/** (§2).
- Details:
  - Values that T13 will pin are left as "PIN-IN-T13", and validation rejects that placeholder unless --allow-unpinned is passed.
  - Hardware peaks cite their sources. The author checks the datasheet values.
- Acceptance: all configs load through the T02 schemas; placeholders are detected.
- Dependencies: T02.

T04: models.py
- Objective: ModelSpec, config parsing and checkpoint metadata.
- Files: src/llmlab/models.py, tests/unit/test_models.py, tests/fixtures/hf_configs/*.
- Details: as in §3.3. The network is used only by load_model_spec; tests use fixtures.
- Acceptance: Qwen2.5-7B → GQA 28/4/128; Llama-3-8B shape → 32/8/128; MHA detected; MLA, Falcon and active-sliding configs raise with specific reasons; the Qwen2 use_sliding_window: false case is accepted.
- Dependencies: T02.

T05: calc/kv.py
- Objective: KV accounting.
- Files: src/llmlab/calc/kv.py, tests/unit/test_kv.py, tests/property/test_kv_properties.py, tests/golden/ch4_examples.yaml.
- Details: as in §3.3; exact sums, no approximations.
- Acceptance: every Chapter 4 golden value matches; g_decode(2048, 1024, 16) = 2567.5; property tests pass; coverage ≥ 95%.
- Dependencies: T01, T04.

T06: calc/roofline.py
- Objective: bounds, calibration and the closed-loop serving model.
- Files: src/llmlab/calc/roofline.py, tests/unit/test_roofline.py.
- Details: equations as in §3.3; bisection solver with a tolerance of 1e-9 and a maximum of 200 iterations, raising if it doesn't converge.
- Acceptance: at N = 1, TPOT = t_dec(1); TPOT never falls as N grows; hand-worked examples within 1e-6; Qwen2.5-7B decode weight bytes checked against a hand calculation.
- Dependencies: T05.

T07: A1 command and golden table
- Objective: llmlab kv and llmlab a1.
- Files: src/llmlab/cli.py (these subcommands), experiments/A_kv_cache/A1_calculator/*, tests/golden/a1_kv_table.csv, Makefile a1.
- Details: the table covers every registered model plus the Llama-3-8B-shaped reference config.
- Acceptance: make a1 output is byte-identical to golden in CI; the CLI prints labelled units.
- Dependencies: T05, T06.

T08: workload.py
- Objective: deterministic fixed-shape requests.
- Files: src/llmlab/workload.py, tests/unit/test_workload.py, tests/fixtures/tokenizer_added_tokens.json.
- Details: as in §3.3.
- Acceptance: same seed and index give identical tokens; no added tokens; length exactly P; the digest is stable.
- Dependencies: T02.

T09: metrics.py
- Objective: canonical metric definitions, gates and run summaries.
- Files: src/llmlab/metrics.py, tests/unit/test_metrics.py, tests/property/test_metric_identities.py, docs/methodology/metrics.md (first version).
- Details: as in §3.3.
- Acceptance: edge-case tests pass; the TPOT = mean-ITL property holds; gates return null with reasons; bootstrap coverage within ±3%.
- Dependencies: T02.

T10: testing/mock_server.py
- Objective: a fake vLLM with known ground truth.
- Files: src/llmlab/testing/mock_server.py, tests/unit/test_mock_server.py.
- Details: as in §3.3; startup lines match the regex formats in §3.3; runnable both as a subprocess and in-process.
- Acceptance: FIFO admission against capacity is correct; /metrics values match internal state; the usage chunk is correct; fault injection works.
- Dependencies: T02.

T11: loadgen.py and B0
- Objective: the closed-loop client and its accuracy report.
- Files: src/llmlab/loadgen.py, tests/integration/test_loadgen_mock.py, experiments/B_serving/B0_harness_accuracy/*, results/derived/B0/harness_accuracy.csv.
- Details: as in §3.3, including staggered starts, window rules and drain.
- Acceptance: B0-01 met on the development machine (p99 error ≤ 1 ms at N ≤ 64); window accounting and timeouts verified; multi-token chunks flagged.
- Dependencies: T08, T09, T10.

T12: telemetry.py
- Objective: Prometheus scraper and NVML sampler.
- Files: src/llmlab/telemetry.py, tests/unit/test_telemetry_parse.py.
- Details: as in §3.3. Start with mock-server fixtures; real vLLM fixtures are added after T13.
- Acceptance: parsing works on mock fixtures now and on vLLM fixtures once T13 lands; missing essential metrics are reported; the fake-NVML test passes.
- Dependencies: T02, T10 (and T13 for the real fixtures).

T13: GPU compatibility check (MANUAL, about 1 GPU-hour)
- Objective: pin versions and capture real fixtures.
- Files:
  - configs/models/*.yaml (revision)
  - configs/engines/vllm.yaml (version, command, digest)
  - configs/engines/vllm-requirements.lock
  - configs/engines/vllm-metrics.yaml (confirmed names)
  - tests/fixtures/vllm/<version>/*
  - docs/adr/ notes
- Details: on the reference GPU, start vLLM with the base arguments and confirm:
  - token-ID prompts, min_tokens, ignore_eos, stream_options usage
  - the prefix-caching-off flag and /version
  - the startup log lines and cache_config_info
  - the metric names

  Save the idle and busy /metrics text and the startup log.
- Acceptance: every ⚠ item is resolved or replaced in the spec; the placeholders are gone; a compatibility note is committed.
- Dependencies: T03. Can run in parallel with T04–T12.

T14: engines/vllm.py
- Objective: engine lifecycle and self-reports.
- Files: src/llmlab/engines/vllm.py, tests/unit/test_vllm_parse.py, lifecycle test.
- Details: as in §3.3; the parser version comes from config.
- Acceptance: real fixtures parse; the mock-launched lifecycle passes; the timeout path raises EngineStartupError; the argument builder is tested for booleans and kebab-case.
- Dependencies: T10, T13.

T15: provenance.py
- Objective: code and environment capture, IDs, hashing.
- Files: src/llmlab/provenance.py, tests/unit/test_provenance.py.
- Acceptance: temporary-repo tests for the dirty flag and prereg lookup; stable hashes.
- Dependencies: T02.

T16: runner.py
- Objective: experiment execution.
- Files: src/llmlab/runner.py, cli.py (run, predict), tests/integration/test_runner_mock_e2e.py.
- Details: the algorithm, checks, resume and refusals in §3.3. predict writes predictions_pre.json.
- Acceptance:
  - The mock end-to-end run produces schema-valid run directories.
  - Resume skips completed runs.
  - Injected errors produce an invalid run with reasons.
  - Writing a mock run under results/ is refused.
  - A dirty tree is refused without the flag.
- Dependencies: T08, T11, T12, T14, T15.

T17: analysis/kv_capacity.py
- Objective: prediction vs observation, claims evaluation and the decision.
- Files: src/llmlab/analysis/kv_capacity.py, cli.py (analyze), tests/integration/test_analysis_ground_truth.py, claims/ledger.yaml (draft entries).
- Details: as in §3.3; one evaluator per claim ID.
- Acceptance: on mock data the true N_sat lies inside the detected interval, occupancy error ≤ 2%, and the Little's-law check passes; every decision branch is covered; the ledger has no claim without an evaluator.
- Dependencies: T05, T06, T09, T16.

T18: report.py and the mock demo
- Objective: figures, provenance files, story rendering and make demo-mock.
- Files: src/llmlab/report.py, cli.py (report), docs/stories/01-kv-capacity.md (template), tests/integration/test_report_golden.py, Makefile demo-mock.
- Details: the six figures in §3.3; the demo writes to build/demo/ with MOCK labels.
- Acceptance: golden figure-data and markdown blocks match; make demo-mock runs end to end in under 3 minutes in CI.
- Dependencies: T17.

T19: validate and archive, plus the integrity workflow
- Objective: integrity checks and raw-data archiving.
- Files: cli.py (validate, archive), .github/workflows/results-integrity.yml.
- Details: every check in §3.7.
- Acceptance: each check has a failing test fixture; validation passes on a clean repo.
- Dependencies: T16, T17, T18.

T20: Documentation
- Objective: methodology, reproduction, run card, limitations, architecture decisions, roadmap, README.
- Files: §3.6.
- Details: the run card lists exact commands and expected GPU-hours; kv-accounting.md derives every formula used.
- Acceptance: link check passes; a colleague can follow the run card without asking questions (dry run on mock).
- Dependencies: T07 onward (can run in parallel).

T21: Preregistration (MANUAL, author)
- Objective: freeze predictions and criteria before touching the GPU.
- Files: experiments/A_kv_cache/{A3,A4}/PREREG.md, claims/ledger.yaml, results/derived/A4/predictions_pre.json, final concurrency grids in manifest.yaml.
- Details: run llmlab predict. Make the grids dense around the predicted saturation band. Commit, and don't change it afterwards; any later deviation goes in the experiment's README.
- Acceptance: llmlab validate confirms the PREREG commit comes before every A3/A4 run.
- Dependencies: T07, T13, T17.

T22: Reference GPU runs (MANUAL, about 8–9 GPU-hours)
- Objective: run A3, then A4, following the run card.
- Files: results/runs/**, results/engines/**.
- Details:
  - Run A3 first and re-run llmlab predict for capacity-dependent predictions. That's allowed only because PREREG defines this exact two-stage procedure in advance.
  - Then run A4 with 3 repetitions per config.
- Acceptance: at least 2 valid repetitions per (config, N); invalid runs kept with reasons; validation passes.
- Dependencies: T16, T19, T20, T21.

T23: Analysis, story and decision (author)
- Objective: turn the results into the published story.
- Files: results/derived/**, docs/figures/**, docs/stories/01-kv-capacity.md (hand-written narrative), claims/ledger.yaml (statuses), CLAIMS.md.
- Details: run llmlab analyze and llmlab report. The author writes the theory and decision prose around the generated blocks. Refuted claims stay in the ledger. Contradictions with the book draft are recorded for the book's errata.
- Acceptance: every claim has a status; the decision section is complete with falsifiers and "not supported" items; validation passes.
- Dependencies: T22.

T24: Release v0.1.0
- Objective: a public, citable release.
- Files: CHANGELOG.md, CITATION.cff (DOI), results/archive_manifest.json, version bump, tag.
- Details: llmlab archive; upload the raw chunks to Zenodo; record the DOI and SHA-256.
- Acceptance: every item in §6 is checked.
- Dependencies: T23.

Critical path: T00 → T01 → T02 → T04 → T05 → T06 → T07, then T08/T09/T10 → T11 → T16 → T17 → T18 → T19 → T21 → T22 → T23 → T24. T12–T15 and T20 run in parallel. T13 is the only GPU task before T22.

5. First commit scope (T00)

README.md            mission; first sentence scopes "Production"; status: pre-release; non-claims; roadmap link
LICENSE              Apache-2.0
LICENSE-DATA         CC-BY-4.0
CITATION.cff         draft (no DOI yet)
CONTRIBUTING.md      workflow, rules for Cursor/AI-assisted changes, one task per PR
CHANGELOG.md         "Unreleased"
pyproject.toml       metadata, dependency groups, ruff/mypy/pytest config
uv.lock
Makefile             setup, lint, type, test (+ stubs for a1, demo-mock, validate)
.gitignore           .cache/, build/, results/**/chunks.parquet, .venv/
.github/workflows/ci.yml
src/llmlab/__init__.py          __version__ = "0.1.0.dev0"
tests/test_smoke.py             imports llmlab, checks version string
docs/roadmap.md                 long-term architecture with v0.1 slice highlighted
docs/adr/0004-rule-of-two.md

Commit message: chore: repository skeleton, licenses, CI, roadmap (v0.1 scope).

Excluded from the first commit: code that produces numbers, configs with guessed values, empty placeholder packages, results.

6. Validation checklist (v0.1 release gate)

Code and tests
- [ ] CI green: ruff, mypy (strict where specified), pytest; coverage ≥ 90% on calc/metrics/analysis and ≥ 80% overall
- [ ] make a1 output byte-identical to golden; all Chapter 4 golden values pass, including 56,128
- [ ] B0-01 met; harness accuracy table committed
- [ ] make demo-mock recovers the mock's known saturation point, labelled MOCK

Preregistration and runs
- [ ] PREREG.md and predictions_pre.json committed before the first A3/A4 run (checked by llmlab validate)
- [ ] vLLM version and image digest, model revision and engine lockfile pinned; no placeholders left
- [ ] Prefix caching confirmed off; max_num_seqs confirmed not to limit concurrency; token budget and block size explicit
- [ ] At least 2 valid repetitions per (config, N); invalid runs kept with reasons
- [ ] No mock runs under results/; raw chunks archived with SHA-256 and DOI

Analysis and honesty
- [ ] Every claim evaluated and marked supported, refuted or inconclusive; refuted claims kept
- [ ] Every figure built from derived data, with provenance and an evidence label in its caption
- [ ] Percentiles that fail the sample gate shown as n/a with a reason, not dropped
- [ ] Decision section states the cap, bottleneck, levers, predicted vs measured knee, falsifiers and what it doesn't support
- [ ] Limitations document covers: single GPU, single model, fixed shape, closed-loop only, no quality checks, no multi-replica behaviour

Documentation and release
- [ ] README readable in 3 minutes; first sentence scopes the name; no unscoped numbers
- [ ] Run card reproduces A3/A4 (dry run on mock verified)
- [ ] Story renders with no stale generated blocks
- [ ] CHANGELOG, CITATION (DOI), tag v0.1.0
- [ ] Book integration noted: A4 results feed the rewrite of Ch. 4.4/4.8, and any contradictions go into the book's errata ledger


That's the full v0.1 spec. Nothing in it has been built or run yet. Before a real vLLM run, the items marked ⚠ need checking in task T13, a one-hour check on a GPU that pins the versions. The magnitudes I quote, such as the saturation point at about 26–31 concurrent requests, are illustrative. The real predictions come from llmlab predict once A3 has measured the KV-cache capacity.

Two decisions in the spec depart from my earlier design:

- The load keeps a fixed number of requests in flight, instead of sending them at a fixed arrival rate. The question here is about how much KV cache concurrent requests fill, and that depends on how many are in flight. So concurrency is the variable to control. Fixed-rate testing moves to v0.2. The spec records this choice as a design decision.
- The workload is 2,048-token prompts with 1,024-token outputs, not the 3,072/256 I first considered. My rough numbers for the shorter-output shape showed prompt processing would use most of the GPU at about the same concurrency where the KV cache fills up. The saturation point would then have two causes at once, which muddies the story. With longer outputs, the rough numbers say the KV cache runs out first while prompt processing uses only about a quarter of the GPU. So the decision the experiment supports is clean: the concurrency cap for this GPU and model is set by memory, not compute.

---

# Part III — Post-v0.1 Continuation Contract

## 1. What completion means under the recovered sources

The **fully specified executable milestone** in the surviving Claude material is **v0.1.0 through T24**.

Therefore an implementation agent must:

1. execute T00–T24 in dependency order;
2. complete every non-external task;
3. perform T13/T21/T22/T23/T24 when the required GPU/account/manual author actions are actually available;
4. preserve `DATA REQUIRED` / unresolved states rather than inventing measurements;
5. pass the v0.1 validation checklist before calling v0.1 complete.

After v0.1, the surviving Claude material provides a **roadmap architecture**, not a complete experiment contract. An agent must not fabricate missing protocols, acceptance thresholds, workload distributions, claims, or release gates for B1–B4/C/D merely to claim that the whole long-term project is finished.

## 2. Allowed post-v0.1 work without a recovered detailed spec

After v0.1 is complete, an agent may continue only with work that is directly implied by the recovered architecture and does not invent scientific outcomes or experiment criteria, for example:

- preserve/extend the documented long-term roadmap;
- keep interfaces and result schemas compatible with future simulator/capacity components;
- improve documentation, CI, provenance validation and contribution ergonomics without changing experimental meaning;
- record explicit TODO/source-blocked items for the missing B/C/D protocols;
- prepare non-executing structural hooks only when the v0.1 rule-of-two and no-placeholder rules permit them.

Do **not** create empty placeholder packages simply to make the roadmap appear implemented. The v0.1 spec explicitly forbids placeholder packages/directories for later `sim`, `capacity`, secondary engine and B1–B4/C/D work.

## 3. Forbidden post-v0.1 invention

Without another authoritative Claude/source document, an agent must not invent:

- B1, B2, B3 or B4 hypotheses, grids, thresholds or pass/fail rules beyond what is explicitly present in surviving files;
- C-series simulator calibration/validation criteria;
- D-series capacity-planning experiment protocols;
- a second engine's compatibility contract;
- production-scale/multi-replica conclusions;
- benchmark numbers, speedups, costs or latency results;
- acceptance thresholds that were not preregistered or source-defined.

If such work is requested, classify it as **SOURCE BLOCKED — DETAILED POST-v0.1 SPECIFICATION NOT RECOVERED** unless a new authoritative design is supplied and approved.

## 4. OpenCode execution rule

When this file is placed in the repository root, OpenCode should treat it as the implementation authority with this precedence:

```text
Part II exact v0.1 implementation specification
        ↓
Part I recovered long-term architecture / design invariants
        ↓
Part III continuation and anti-invention rules
        ↓
existing repository state, if it does not contradict the above
```

OpenCode should not stop after planning. It should implement, test, fix and continue through the complete T00–T24 path whenever dependencies are available.

A failure in tests/lint/type checks is work to fix, not a stop condition.

A missing GPU, credential, model access approval, external quota, Zenodo authorization or required manual author sign-off is an **external blocker** only for the task that genuinely needs it. All independent work must continue.

## 5. Measurement integrity

Never fabricate or infer observed values from the analytical examples in the spec.

Maintain the distinctions:

- **IDENTITY / exact calculation**
- **ANALYTICAL MODEL / bound / prediction**
- **MOCK / harness result**
- **SIMULATED** (future)
- **MEASURED**
- **EXTERNALLY CITED**
- **DATA REQUIRED / INCONCLUSIVE / REFUTED**

A result can be labelled **MEASURED** only when the corresponding run actually executed and its provenance record is present.

## 6. Git and provenance rules

The recovered design treats commit history as part of the evidence chain. Therefore:

- do not squash away preregistration chronology;
- do not rewrite published history;
- do not force-push;
- keep preregistration commits prior to the first corresponding run;
- retain invalid runs with reasons when the spec requires it;
- do not hand-edit derived evidence;
- keep raw-data hashes and archive references;
- never commit credentials or account secrets.

## 7. Canonical stop condition for an autonomous implementation agent

The agent may stop only when one of these is true:

### A. v0.1 COMPLETE

T00–T24 and the full v0.1 release checklist pass, including real GPU/release tasks when the needed external capabilities were available.

### B. EXTERNAL BLOCKER REMAINS

Every task not dependent on the blocker is complete, and the remaining item genuinely requires something outside the agent's control. The final report must identify:

- blocked task ID;
- exact missing capability;
- work already completed around it;
- exact user/external action needed;
- what must be rerun after unblocking.

### C. POST-v0.1 SOURCE BLOCKER

v0.1 is complete, but execution of the broader B/C/D roadmap would require scientific protocols not present in the recovered Claude sources. This is not permission to redesign them silently.

## 8. Required final implementation report

When OpenCode finishes, its final report must include:

1. repository branch / HEAD / git status;
2. T00–T24 status, one by one;
3. test / lint / type / CI results;
4. experiment status for A1, A3, A4 and B0;
5. exact classification of every result as analytical, mock or measured;
6. claims ledger statuses, including refuted/inconclusive claims;
7. provenance/archive status;
8. book errata discovered by the lab;
9. external blockers, if any;
10. remaining work — `NONE` for v0.1 unless an explicitly documented external blocker exists;
11. post-v0.1 roadmap items marked **SOURCE BLOCKED** rather than invented.

---

# Canonicality note

This file intentionally distinguishes **what Claude actually specified completely** from **what Claude only designed at roadmap level**. The objective is not to make the recovered plan look more complete than it is. The objective is to give an implementation agent the strongest possible source of truth while preserving the project's central scientific rule:

> predict first, measure honestly, publish the residual, and never upgrade missing evidence into a result.
