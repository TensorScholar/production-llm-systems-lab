"""Typed configs and records (pydantic v2, extra="forbid").

Every file the lab reads or writes is validated here, with JSON Schema
export via ``llmlab schema``. Book §3.3 / blueprint §3.2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Strict = ConfigDict(extra="forbid")


class ModelRef(BaseModel):
    model_config = Strict

    id: str
    hf_repo: str
    revision: str
    license: str = "Apache-2.0"
    serving_dtype: str = "bfloat16"
    kv_cache_dtype_bytes: int = 2


class HardwareProfile(BaseModel):
    model_config = Strict

    id: str
    name: str
    peak_mem_bw_Bps: float
    peak_dense_fp16_flops: float
    sources: list[str] = Field(default_factory=list)
    notes: str = ""


class EngineConfig(BaseModel):
    model_config = Strict

    name: str = "vllm"
    version: str
    command: list[str] = Field(default_factory=lambda: ["vllm", "serve"])
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout_s: float = 1200.0
    env: dict[str, str] = Field(default_factory=dict)
    base_args: dict[str, Any] = Field(default_factory=dict)
    metrics_map: str = "configs/engines/vllm-metrics.yaml"
    log_parser: str = "vllm-startup-v1"


class MetricsMap(BaseModel):
    model_config = Strict

    kv_usage: list[str] = Field(default_factory=list)
    running: list[str] = Field(default_factory=list)
    waiting: list[str] = Field(default_factory=list)
    preemptions: list[str] = Field(default_factory=list)
    prompt_tokens: list[str] = Field(default_factory=list)
    generation_tokens: list[str] = Field(default_factory=list)
    cache_config_info: list[str] = Field(default_factory=list)
    essential: list[str] = Field(default_factory=lambda: ["kv_usage", "running", "waiting"])


class WorkloadSpec(BaseModel):
    model_config = Strict

    kind: Literal["fixed_shape"] = "fixed_shape"
    prompt_tokens: int = 2048
    output_tokens: int = 1024
    seed: int = 20260925


class ClientConfig(BaseModel):
    model_config = Strict

    warmup_s: float = 60.0
    stagger_s: float | Literal["auto"] = "auto"
    window_min_s: float = 180.0
    window_max_s: float = 480.0
    min_completed: int = 150
    request_timeout_s: float = 900.0
    drain_timeout_s: float = 900.0


class IdleWait(BaseModel):
    model_config = Strict

    kv_usage_max: float = 0.005
    timeout_s: float = 180.0


class TelemetryConfig(BaseModel):
    model_config = Strict

    metrics_interval_s: float = 1.0
    nvml_interval_s: float = 0.2


class Gates(BaseModel):
    model_config = Strict

    min_tail_samples: int = 20


class Checks(BaseModel):
    model_config = Strict

    max_error_rate: float = 0.01
    stability_tolerance: float = 0.10
    loop_lag_p99_ms: float = 5.0
    client_cpu_max: float = 0.80


class DecisionConfig(BaseModel):
    model_config = Strict

    kv_headroom: float = 0.90


class OrderConfig(BaseModel):
    model_config = Strict

    randomize: bool = True
    seed: int = 7


class SweepConfig(BaseModel):
    model_config = Strict

    name: str
    engine_args: dict[str, Any] = Field(default_factory=dict)
    concurrency: list[int] = Field(default_factory=list)


class ExperimentManifest(BaseModel):
    model_config = Strict

    experiment_id: str
    title: str = ""
    prereg: str | None = None
    model: str
    hardware: str
    engine: str
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    configs: list[SweepConfig] = Field(default_factory=list)
    repetitions: int = 3
    order: OrderConfig = Field(default_factory=OrderConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    idle_wait: IdleWait = Field(default_factory=IdleWait)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    gates: Gates = Field(default_factory=Gates)
    checks: Checks = Field(default_factory=Checks)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)

    def resolve(self, base: Path) -> ExperimentManifest:
        """Return a copy with relative config paths resolved against base dir."""
        data = self.model_dump()
        for key in ("model", "hardware", "engine"):
            p = data[key]
            if isinstance(p, str) and not Path(p).is_absolute():
                data[key] = str((base / p).resolve())
        return ExperimentManifest(**data)


class StartupReport(BaseModel):
    model_config = Strict

    kv_cache_tokens: int | None = None
    num_gpu_blocks: int | None = None
    block_size: int | None = None
    max_concurrency: float | None = None
    max_concurrency_tokens: int | None = None
    model_load_bytes: int | None = None
    raw_lines: list[str] = Field(default_factory=list)
    parser_version: str = "vllm-startup-v1"


class EngineInstance(BaseModel):
    model_config = Strict

    instance_id: str
    experiment_id: str
    config_name: str
    repetition: int
    started_utc: str = ""
    engine_name: str = "vllm"
    engine_version: str = ""
    resolved_args: dict[str, Any] = Field(default_factory=dict)


class RequestRecord(BaseModel):
    model_config = Strict

    run_id: str
    request_id: str
    worker_id: int = 0
    t_send_ns: int
    t_first_ns: int | None = None
    t_last_ns: int | None = None
    n_chunks: int = 0
    n_prompt_tokens: int = 0
    n_output_tokens: int = 0
    status: Literal["ok", "error", "timeout"] = "ok"
    error: str | None = None
    in_window: bool = True
    ttft_s: float | None = None
    e2e_s: float | None = None
    tpot_s: float | None = None
    itl_p50_s: float | None = None
    itl_p95_s: float | None = None
    itl_max_s: float | None = None
    multi_token_chunks: bool = False


class ChunkRecord(BaseModel):
    model_config = Strict

    request_id: str
    seq: int
    t_ns: int
    text_len: int = 0


class CheckResult(BaseModel):
    model_config = Strict

    passed: bool
    value: float | bool | str | None = None
    threshold: float | bool | str | None = None
    invalidating: bool = True
    reason: str | None = None


class CodeInfo(BaseModel):
    model_config = Strict

    commit: str = "unknown"
    dirty: bool = False
    llmlab_version: str = "0.1.0.dev0"


class EnvInfo(BaseModel):
    model_config = Strict

    python_version: str = ""
    platform: str = ""
    cpu: str = ""
    pip_freeze_sha256: str = ""


class RunMetadata(BaseModel):
    model_config = Strict

    run_id: str
    experiment_id: str
    manifest_sha256: str = ""
    prereg_commit: str = ""
    prereg_time: str = ""
    code: CodeInfo = Field(default_factory=CodeInfo)
    environment: EnvInfo = Field(default_factory=EnvInfo)
    engine_name: str = "vllm"
    engine_version: str = ""
    engine_args: dict[str, Any] = Field(default_factory=dict)
    instance_id: str = ""
    gpu_name: str = ""
    gpu_total_bytes: int | None = None
    model_id: str = ""
    model_revision: str = ""
    weight_bytes: int | None = None
    m_kv_bytes: int | None = None
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    workload_hash: str = ""
    concurrency: int = 0
    config_name: str = ""
    repetition: int = 0
    started_utc: str = ""
    ended_utc: str = ""
    window_start_ns: int = 0
    window_end_ns: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    checks: dict[str, CheckResult] = Field(default_factory=dict)
    valid: bool = True
    invalid_reasons: list[str] = Field(default_factory=list)
    evidence: Literal["MEASURED", "MOCK"] = "MEASURED"
    tier: int = 2


class Claim(BaseModel):
    model_config = Strict

    id: str
    type: str
    statement: str
    pass_criterion: str = ""
    status: Literal["proposed", "supported", "refuted", "inconclusive", "data_required"] = (
        "data_required"
    )
    evidence: str = ""
    assumptions: str = ""
    scope: str = ""


class ClaimEvaluation(BaseModel):
    model_config = Strict

    claim_id: str
    status: str = "inconclusive"
    observed: str = ""
    expected: str = ""
    residual: str = ""
    reason: str = ""


class Decision(BaseModel):
    model_config = Strict

    cap: int | None = None
    binding: str = "unknown"
    levers: list[str] = Field(default_factory=list)
    predicted_knee: str = ""
    measured_knee: str = ""
    narrative: str = ""


def load_yaml(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load a YAML file into a pydantic model (unknown keys rejected)."""
    from pydantic import TypeAdapter

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    adapter = TypeAdapter(model)
    return adapter.validate_python(raw)


def sha256_json(obj: Any) -> str:
    """Stable SHA-256 of a JSON-serializable object (sorted keys)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
