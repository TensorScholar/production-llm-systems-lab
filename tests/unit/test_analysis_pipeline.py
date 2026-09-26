"""Analysis over real runner output + remaining evaluator branches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from llmlab.analysis import kv_capacity as a
from llmlab.models import ModelSpec
from llmlab.schemas import Claim


def test_pipeline_over_runner_output(tmp_path: Path) -> None:
    from llmlab.runner import run_experiment

    repo = Path(__file__).parent.parent.parent
    man = {
        "experiment_id": "B0",
        "title": "t",
        "model": str(repo / "configs/models/qwen2.5-7b-instruct.yaml"),
        "hardware": str(repo / "configs/hardware/a10g-24gb.yaml"),
        "engine": str(repo / "configs/engines/mock.yaml"),
        "workload": {"kind": "fixed_shape", "prompt_tokens": 8, "output_tokens": 4, "seed": 7},
        "configs": [
            {"name": "m", "engine_args": {"mock": {"capacity_tokens": 512}}, "concurrency": [1, 2]}
        ],
        "repetitions": 1,
        "order": {"randomize": False, "seed": 1},
        "client": {
            "warmup_s": 0.1,
            "stagger_s": 0.1,
            "window_min_s": 2.0,
            "window_max_s": 8.0,
            "min_completed": 5,
            "request_timeout_s": 30.0,
            "drain_timeout_s": 30.0,
        },
        "idle_wait": {"kv_usage_max": 0.5, "timeout_s": 30},
        "telemetry": {"metrics_interval_s": 0.5, "nvml_interval_s": 0.2},
        "gates": {"min_tail_samples": 1},
        "checks": {
            "max_error_rate": 0.5,
            "stability_tolerance": 0.9,
            "loop_lag_p99_ms": 500.0,
            "client_cpu_max": 1.0,
        },
        "decision": {"kv_headroom": 0.99},
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(man), encoding="utf-8")
    out = tmp_path / "build"
    run_experiment(p, out, allow_dirty=True)
    runs = a.load_runs(out, "B0", valid_only=False)
    assert not runs.empty
    assert set(runs["N"]) == {1, 2}
    points = a.summarize_points(runs)
    assert set(points["N"]) == {1, 2}
    occ = a.occupancy_comparison(points, {"m": 512}, 12.0, 12.0)
    assert not occ.empty
    sat = a.detect_saturation(points)
    assert set(sat["indicator"]) == {"waiting", "running", "preemptions", "kv_usage"}
    ll = a.littles_law_check(points)
    assert not ll.empty
    sm = a.serving_model_check(points, _spec(), 57344, 15_233_241_088, 8, 4, 64)
    assert not sm.empty
    d = a.derive_decision(points, sat, pd.DataFrame(), 0.99)
    assert d.cap in (1, 2)


def _spec() -> ModelSpec:
    return ModelSpec(
        name="q",
        n_layers=28,
        n_q_heads=28,
        n_kv_heads=4,
        head_dim=128,
        hidden_size=3584,
        vocab_size=152064,
        tie_word_embeddings=False,
        attention="gqa",
        param_count=8_000_000_000,
    )


def test_no_saturation_branch() -> None:
    pts = pd.DataFrame(
        [
            {
                "config": "c",
                "N": 1,
                "waiting_mean_median": 0.0,
                "running_mean_median": 1.0,
                "preemptions_delta_median": 0.0,
                "kv_usage_mean_median": 0.1,
            }
        ]
    )
    sat = a.detect_saturation(pts)
    assert (sat["N_hi"].isna()).all()
    tables = {"saturation": sat, "scalar": {"band": (5.0, 9.0), "grid": [1, 2, 4]}}
    ev = a.evaluate_claims([Claim(id="A4-02", type="M", statement="s")], tables)
    assert ev[0].status == "refuted"


def test_evaluator_exception_path() -> None:
    class Boom:
        pass

    ev = a.evaluate_claims(
        [Claim(id="A4-01", type="M", statement="s")],
        {"occupancy": Boom()},  # type: ignore[dict-item]
    )
    assert ev[0].status == "inconclusive"


def test_a4_01_inconclusive_no_unsat() -> None:
    occ = pd.DataFrame(
        [
            {
                "config": "c",
                "N": 8,
                "observed": 500.0,
                "predicted": 400.0,
                "predicted_semi": 400.0,
                "rel_err": 0.25,
                "rel_err_semi": 0.25,
            }
        ]
    )
    ev = a.evaluate_claims([Claim(id="A4-01", type="M", statement="s")], {"occupancy": occ})
    assert ev[0].status == "inconclusive"


def test_b0_supported_in_scope() -> None:
    b0 = pd.DataFrame(
        [{"ttft_p99_err_s": 0.0005, "itl_p99_err_s": 0.0008, "in_scope_reference": True}]
    )
    ev = a.evaluate_claims([Claim(id="B0-01", type="H", statement="s")], {"b0": b0})
    assert ev[0].status == "supported"
    b1 = pd.DataFrame(
        [{"ttft_p99_err_s": 0.005, "itl_p99_err_s": 0.0008, "in_scope_reference": True}]
    )
    ev2 = a.evaluate_claims([Claim(id="B0-01", type="H", statement="s")], {"b0": b1})
    assert ev2[0].status == "refuted"


def test_write_predictions_param_count_paths(tmp_path: Path) -> None:
    import yaml as _y

    man = {
        "experiment_id": "AX",
        "workload": {"prompt_tokens": 100, "output_tokens": 16},
        "configs": [{"name": "c1", "concurrency": [1]}],
    }
    mp = tmp_path / "m.yaml"
    mp.write_text(_y.safe_dump(man), encoding="utf-8")
    # Untied spec with param_count: exercises the embedding subtraction.
    a.write_predictions(mp, _spec(), tmp_path / "p1.json", None)
    # No param_count at all: weight unknown.
    nospec = ModelSpec(
        name="n",
        n_layers=2,
        n_q_heads=2,
        n_kv_heads=2,
        head_dim=8,
        hidden_size=16,
        vocab_size=32,
        tie_word_embeddings=True,
        attention="gqa",
    )
    a.write_predictions(mp, nospec, tmp_path / "p2.json", None)
    import json as _j

    assert _j.loads((tmp_path / "p2.json").read_text())["model"]["decode_weight_bytes"] == 0
