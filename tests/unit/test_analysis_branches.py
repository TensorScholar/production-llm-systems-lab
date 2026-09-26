"""Analysis branch coverage: evaluators, reconcile, overhead, serving, littles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llmlab.analysis import kv_capacity as a
from llmlab.models import ModelSpec
from llmlab.schemas import Claim

SPEC = ModelSpec(
    name="q",
    n_layers=28,
    n_q_heads=28,
    n_kv_heads=4,
    head_dim=128,
    hidden_size=3584,
    vocab_size=152064,
    tie_word_embeddings=True,
    attention="gqa",
    param_count=7_616_620_544,
)


def _claim(cid: str) -> Claim:
    return Claim(id=cid, type="T", statement="s")


def test_a3_evaluators() -> None:
    recon = pd.DataFrame(
        [
            {
                "config": "a",
                "num_gpu_blocks": 5000,
                "block_size": 16,
                "kv_cache_tokens": 80000,
                "gpu_util": 0.9,
                "weight_bytes": 15_000_000_000,
                "reported_pool_tokens": 80000,
            },
            {
                "config": "b",
                "num_gpu_blocks": 4375,
                "block_size": 16,
                "kv_cache_tokens": 70000,
                "gpu_util": 0.8,
                "weight_bytes": 15_000_000_000,
                "reported_pool_tokens": 70000,
            },
        ]
    )
    out = a.reconcile_startup(recon, SPEC, 57344, 24 * 1024**3)
    assert "overhead_bytes" in out.columns
    assert abs(float(out["pool_match"].iloc[0]) - 1.0) < 1e-9
    model = a.fit_overhead(out, "a")
    assert model.train_config == "a"
    with pytest.raises(ValueError):
        a.fit_overhead(out, "nope")
    ev = a.evaluate_claims([_claim("A3-01")], {"recon": out})
    assert ev[0].status == "supported"
    bad = out.copy()
    bad.loc[0, "kv_cache_tokens"] = 70000
    bad = a.reconcile_startup(
        pd.DataFrame(
            [
                {
                    "config": "a",
                    "num_gpu_blocks": 5000,
                    "block_size": 16,
                    "kv_cache_tokens": 70000,
                    "gpu_util": 0.9,
                    "weight_bytes": 15_000_000_000,
                    "reported_pool_tokens": 70000,
                }
            ]
        ),
        SPEC,
        57344,
        24 * 1024**3,
    )
    ev2 = a.evaluate_claims([_claim("A3-01")], {"recon": bad})
    assert ev2[0].status == "refuted"


def test_a3_02_heldout() -> None:
    held = pd.DataFrame([{"config": "b", "rel_err": 0.02}])
    ev = a.evaluate_claims([_claim("A3-02")], {"overhead_heldout": held})
    assert ev[0].status == "supported"
    held2 = pd.DataFrame([{"config": "b", "rel_err": 0.20}])
    assert a.evaluate_claims([_claim("A3-02")], {"overhead_heldout": held2})[0].status == "refuted"


def test_a4_evaluators_supported_and_refuted() -> None:
    occ = pd.DataFrame(
        [
            {
                "config": "c",
                "N": 4,
                "observed": 10000.0,
                "predicted": 10270.0,
                "predicted_semi": 10050.0,
                "rel_err": -0.026,
                "rel_err_semi": -0.005,
            },
        ]
    )
    assert a.evaluate_claims([_claim("A4-01")], {"occupancy": occ})[0].status == "supported"
    occ_bad = occ.copy()
    occ_bad["rel_err"] = 0.5
    occ_bad["rel_err_semi"] = 0.5
    assert a.evaluate_claims([_claim("A4-01")], {"occupancy": occ_bad})[0].status == "refuted"

    sat = pd.DataFrame([{"config": "c", "indicator": "waiting", "N_lo": 24, "N_hi": 28}])
    tables = {
        "saturation": sat,
        "scalar": {"band": (20.0, 30.0), "grid": [16, 20, 24, 28, 32]},
    }
    assert a.evaluate_claims([_claim("A4-02")], tables)[0].status == "supported"
    tables2 = {
        "saturation": sat,
        "scalar": {"band": (40.0, 50.0), "grid": [16, 20, 24, 28, 32]},
    }
    assert a.evaluate_claims([_claim("A4-02")], tables2)[0].status == "refuted"

    sc = {"tpot1": 0.03, "tpot_floor": 0.025}
    assert a.evaluate_claims([_claim("A4-03")], {"scalar": sc})[0].status == "supported"
    sc2 = {"tpot1": 0.01, "tpot_floor": 0.025}
    assert a.evaluate_claims([_claim("A4-03")], {"scalar": sc2})[0].status == "refuted"

    sm = pd.DataFrame([{"tpot_rel_err": 0.1, "tput_rel_err": 0.1}])
    assert a.evaluate_claims([_claim("A4-04")], {"serving_model": sm})[0].status == "supported"
    sm2 = pd.DataFrame([{"tpot_rel_err": 0.5, "tput_rel_err": 0.1}])
    assert a.evaluate_claims([_claim("A4-04")], {"serving_model": sm2})[0].status == "refuted"

    pts = pd.DataFrame([{"config": "c", "N": 4}])
    ll = pd.DataFrame([{"config": "c", "N": 4, "rel_err": 0.1}])
    assert (
        a.evaluate_claims([_claim("A4-05")], {"littles": ll, "points": pts})[0].status
        == "supported"
    )
    ll2 = pd.DataFrame([{"config": "c", "N": 4, "rel_err": 0.9}])
    assert (
        a.evaluate_claims([_claim("A4-05")], {"littles": ll2, "points": pts})[0].status == "refuted"
    )


def test_serving_model_and_littles_tables() -> None:
    pts = pd.DataFrame(
        [
            {
                "config": "c",
                "N": 1,
                "tpot_p50_median": 0.03,
                "ttft_p50_median": 0.5,
                "req_per_s_median": 2.0,
                "out_tok_per_s_client_median": 2000.0,
                "waiting_mean_median": 0.0,
            },
            {
                "config": "c",
                "N": 2,
                "tpot_p50_median": 0.032,
                "ttft_p50_median": 0.52,
                "req_per_s_median": 3.8,
                "out_tok_per_s_client_median": 3800.0,
                "waiting_mean_median": 0.0,
            },
        ]
    )
    sm = a.serving_model_check(pts, SPEC, 57344, 15_233_241_088, 2048, 1024, 2048)
    assert not sm.empty and "tpot_rel_err" in sm.columns
    ll = a.littles_law_check(pts)
    assert not ll.empty and "rel_err" in ll.columns
    assert a.summarize_points(pd.DataFrame()).empty
    assert a.serving_model_check(
        pts[pts["N"] == 2], SPEC, 57344, 15_233_241_088, 2048, 1024, 2048
    ).empty  # no N=1 base


def test_write_predictions_paths(tmp_path: Path) -> None:
    import yaml

    man = {
        "experiment_id": "AX",
        "workload": {"prompt_tokens": 100, "output_tokens": 16},
        "configs": [{"name": "c1", "concurrency": [1, 2]}],
    }
    mp = tmp_path / "manifest.yaml"
    mp.write_text(yaml.safe_dump(man), encoding="utf-8")
    out = tmp_path / "pred.json"
    a.write_predictions(mp, SPEC, out, {"c1": 4800})
    pred = json.loads(out.read_text(encoding="utf-8"))
    assert pred["configs"]["c1"]["capacity_source"] == "A3-derived"
    assert pred["configs"]["c1"]["saturation_band"][0] < pred["configs"]["c1"]["saturation_band"][1]
    out2 = tmp_path / "pred2.json"
    a.write_predictions(mp, SPEC, out2, None)
    assert json.loads(out2.read_text(encoding="utf-8"))["configs"]["c1"]["capacity_tokens"] is None


def test_load_runs_empty(tmp_path: Path) -> None:
    assert a.load_runs(tmp_path, "ZZZ").empty
