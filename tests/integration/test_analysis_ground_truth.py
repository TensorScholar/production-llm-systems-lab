"""Analysis recovers mock ground truth; decision branches; ledger coverage."""

from __future__ import annotations

import pandas as pd
import yaml

from llmlab.analysis import kv_capacity as a
from llmlab.schemas import Claim

C = 4096
P, OUT = 32, 16
N_SAT = C // (P + OUT)  # 85


def _points() -> pd.DataFrame:
    rows = []
    for n in [1, 2, 4, 8, 16, 32, 64, 85, 86, 96, 128]:
        sat = n > N_SAT
        rows.append(
            {
                "config": "mock",
                "N": n,
                "kv_occupancy_tokens_median": min(n * (P + OUT), C),
                "waiting_mean_median": 5.0 if sat else 0.0,
                "running_mean_median": (C // (P + OUT)) if sat else float(n),
                "preemptions_delta_median": 1.0 if sat else 0.0,
                "kv_usage_mean_median": min(n * (P + OUT) / C, 1.0),
                "ttft_p50_median": 0.05 if not sat else 0.05 + 5.0 / max(n / 0.5, 1),
                "tpot_p50_median": 0.02,
                "req_per_s_median": n / 0.5,
                "out_tok_per_s_client_median": n / 0.5 * OUT,
            }
        )
    return pd.DataFrame(rows)


def test_saturation_interval_contains_truth() -> None:
    sat = a.detect_saturation(_points())
    prim = sat[sat["indicator"] == "waiting"].iloc[0]
    assert prim["N_lo"] == 85 and prim["N_hi"] == 86
    assert prim["N_lo"] <= N_SAT <= prim["N_hi"]


def test_occupancy_error_within_2pct() -> None:
    occ = a.occupancy_comparison(_points(), {"mock": C}, float(P + OUT), float(P + OUT))
    unsat = occ[occ["observed"] < C]
    assert not unsat.empty
    assert float(unsat["rel_err"].abs().max()) <= 0.02


def test_littles_law_passes() -> None:
    ll = a.littles_law_check(_points())
    assert not ll.empty
    unsat = ll[ll["N"] <= 85]
    assert float(unsat["rel_err"].median()) <= 0.20


def test_ledger_every_claim_has_evaluator() -> None:
    from pathlib import Path

    ledger = [
        Claim(**c)
        for c in yaml.safe_load(
            (Path(__file__).parent.parent.parent / "claims/ledger.yaml").read_text(encoding="utf-8")
        )
    ]
    missing = [c.id for c in ledger if c.id not in a.EVALUATORS]
    assert missing == []
    assert len(ledger) == 10


def test_evaluate_claims_data_required_without_data() -> None:
    from pathlib import Path

    ledger = [
        Claim(**c)
        for c in yaml.safe_load(
            (Path(__file__).parent.parent.parent / "claims/ledger.yaml").read_text(encoding="utf-8")
        )
    ]
    evals = a.evaluate_claims(ledger, {"points": pd.DataFrame()})
    by_id = {e.claim_id: e.status for e in evals}
    assert by_id["A4-02"] == "data_required"
    assert by_id["B0-01"] == "data_required"


def test_decision_branches() -> None:
    pts = _points()
    sat = a.detect_saturation(pts)
    d = a.derive_decision(pts, sat, pd.DataFrame(), 0.90)
    # N=85 is queue-free but at KV usage 0.996 > 0.90 headroom, so the cap is 64.
    assert d.cap == 64
    assert d.binding in ("unknown", "kv_capacity", "prefill_compute", "mixed")
    assert d.levers
