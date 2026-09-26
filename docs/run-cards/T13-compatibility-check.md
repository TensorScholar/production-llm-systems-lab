# T13 — GPU compatibility check (MANUAL, ~1 GPU-hour, EXTERNAL BLOCKER)

**Status:** BLOCKED — no compatible NVIDIA GPU is available to this agent.
All non-version-specific work is complete; the items below are the exact
prerequisites to execute after unblocking.

## What T13 must do (on the reference GPU)

1. Start vLLM with the base arguments (`configs/engines/vllm.yaml`) and confirm:
   - token-ID prompts, `min_tokens`, `ignore_eos`, `stream_options` usage;
   - the prefix-caching-off flag and `/version`;
   - the startup log lines (`GPU KV cache size`, `Maximum concurrency`,
     `Model loading took`) and `cache_config_info`;
   - the Prometheus metric names.
2. Save captured fixtures under `tests/fixtures/vllm/<version>/`:
   `startup.log`, `metrics_idle.txt`, `metrics_busy.txt`, `version.json`.
3. Pin and commit:
   - `configs/models/*.yaml` revision (40-hex);
   - `configs/engines/vllm.yaml` version, command, digest;
   - `configs/engines/vllm-requirements.lock` (pip freeze of engine env);
   - `configs/engines/vllm-metrics.yaml` confirmed names (replace candidates).
4. Resolve every `⚠` item in the blueprint or replace it in-spec; remove all
   `PIN-IN-T13` placeholders; commit a compatibility note.

## Unverified until T13 (do NOT treat as verified)

- vLLM flag names (`base_args` keys and `--no-` negation form);
- Prometheus metric names in `vllm-metrics.yaml`;
- startup log line formats (parser `vllm-startup-v1` regexes);
- `/version` endpoint behaviour and log fallback;
- `min_tokens`/`ignore_eos`/`stream_options` extension behaviour;
- `get_safetensors_metadata` checkpoint-size path in `models.py`.

## Fixture layout (to be filled by T13)

```text
tests/fixtures/vllm/<version>/
├── startup.log
├── metrics_idle.txt
├── metrics_busy.txt
└── version.json
```
