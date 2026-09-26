"""Runner end-to-end on mock: schema-valid dirs, resume, invalid, refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llmlab.runner import ManifestError, run_experiment
from llmlab.schemas import RunMetadata

REPO = Path(__file__).parent.parent.parent


def _manifest(tmp: Path, name: str, **over: object) -> Path:  # type: ignore[no-untyped-def]
    man: dict[str, object] = {
        "experiment_id": "B0",
        "title": "runner e2e",
        "model": str(REPO / "configs/models/qwen2.5-7b-instruct.yaml"),
        "hardware": str(REPO / "configs/hardware/a10g-24gb.yaml"),
        "engine": str(REPO / "configs/engines/mock.yaml"),
        "workload": {"kind": "fixed_shape", "prompt_tokens": 32, "output_tokens": 16, "seed": 7},
        "configs": [
            {
                "name": "mock",
                "engine_args": {"mock": {"capacity_tokens": 2048, "ttft_ms": 5.0, "itl_ms": 2.0}},
                "concurrency": [1, 2, 4],
            }
        ],
        "repetitions": 1,
        "order": {"randomize": False, "seed": 7},
        "client": {
            "warmup_s": 0.1,
            "stagger_s": 0.1,
            "window_min_s": 2.0,
            "window_max_s": 8.0,
            "min_completed": 5,
            "request_timeout_s": 30.0,
            "drain_timeout_s": 30.0,
        },
        "idle_wait": {"kv_usage_max": 0.05, "timeout_s": 30},
        "telemetry": {"metrics_interval_s": 0.5, "nvml_interval_s": 0.2},
        "gates": {"min_tail_samples": 1},
        "checks": {
            "max_error_rate": 0.5,
            "stability_tolerance": 0.5,
            "loop_lag_p99_ms": 50.0,
            "client_cpu_max": 1.0,
        },
        "decision": {"kv_headroom": 0.9},
    }
    man.update(over)
    p = tmp / name
    p.write_text(yaml.safe_dump(man), encoding="utf-8")
    return p


def test_runner_mock_e2e(tmp_path: Path) -> None:
    man = _manifest(tmp_path, "m.yaml")
    out = tmp_path / "build"
    produced = run_experiment(man, out, allow_dirty=True)
    assert len(produced) == 3
    for d in produced:
        meta = RunMetadata(**yaml.safe_load((d / "run.json").read_text(encoding="utf-8")))
        assert meta.evidence == "MOCK"
        assert (d / "requests.parquet").exists()
        assert (d / "chunks.parquet").exists()
        assert (d / "server_metrics.parquet").exists()
        assert (d / "checks.json").exists()
    # Resume: second run skips all.
    again = run_experiment(man, out, allow_dirty=True)
    assert [p.name for p in again] == [p.name for p in produced]


def test_runner_refuses_mock_under_results(tmp_path: Path) -> None:
    man = _manifest(tmp_path, "m.yaml")
    with pytest.raises(ManifestError, match="mock"):
        run_experiment(man, tmp_path / "results", allow_dirty=True)


def test_runner_invalid_on_errors(tmp_path: Path) -> None:
    man = _manifest(tmp_path, "bad.yaml")
    raw = yaml.safe_load(man.read_text(encoding="utf-8"))
    raw["configs"][0]["engine_args"]["mock"]["error_rate"] = 1.0
    man.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "build-bad"
    produced = run_experiment(man, out, allow_dirty=True)
    assert produced
    for d in produced:
        meta = RunMetadata(**yaml.safe_load((d / "run.json").read_text(encoding="utf-8")))
        assert meta.valid is False
        assert meta.invalid_reasons
