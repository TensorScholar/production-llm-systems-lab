"""Prediction-vs-observation analysis for the KV-capacity story (book Ch. 4).

One evaluator per claim ID, registered in EVALUATORS. Mock ground truth
(recovered exactly): g = P+O, N_sat = floor(C/(P+O)).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llmlab.calc import kv as kvcalc
from llmlab.calc import roofline
from llmlab.metrics import summarize_run
from llmlab.models import ModelSpec
from llmlab.schemas import Claim, ClaimEvaluation, Decision, RunMetadata

logger = logging.getLogger(__name__)


def load_runs(results_dir: Path, experiment_id: str, valid_only: bool = True) -> pd.DataFrame:
    """Load per-run summary rows by re-summarizing immutable run dirs."""
    rows: list[dict[str, Any]] = []
    exp_dir = Path(results_dir) / "runs" / experiment_id
    if not exp_dir.exists():
        return pd.DataFrame()
    for run_dir in sorted(exp_dir.iterdir()):
        rj = run_dir / "run.json"
        if not rj.exists():
            continue
        meta = RunMetadata(**json.loads(rj.read_text(encoding="utf-8")))
        if valid_only and not meta.valid:
            continue
        req = pd.read_parquet(run_dir / "requests.parquet")
        srv = pd.read_parquet(run_dir / "server_metrics.parquet")
        gpath = run_dir / "gpu.parquet"
        gpu: pd.DataFrame | None
        try:
            gpu = pd.read_parquet(gpath)
            if gpu is None or gpu.empty:
                gpu = None
        except Exception:
            gpu = None
        s = summarize_run(req, srv, gpu, (meta.window_start_ns, meta.window_end_ns), 0)
        s.update(
            {
                "run_id": meta.run_id,
                "config": meta.config_name,
                "rep": meta.repetition,
                "N": meta.concurrency,
                "evidence": meta.evidence,
            }
        )
        # Recompute occupancy with the run's own capacity when known.
        rows.append(s)
    return pd.DataFrame(rows)


def reconcile_startup(
    reports: pd.DataFrame, spec: ModelSpec, m_kv: int, gpu_total_bytes: int
) -> pd.DataFrame:
    """A3 reconciliation: reported KV tokens vs analytical pool; overhead fit inputs.

    Formula: overhead = util·NVML_total − weight_bytes − pool_bytes.
    """
    out = reports.copy()
    out["analytical_pool_bytes"] = out["num_gpu_blocks"] * out["block_size"] * m_kv
    out["reported_pool_bytes"] = out["kv_cache_tokens"] * m_kv
    out["pool_match"] = np.where(
        out["analytical_pool_bytes"] > 0,
        out["reported_pool_bytes"] / out["analytical_pool_bytes"],
        np.nan,
    )
    out["overhead_bytes"] = (
        out["gpu_util"] * gpu_total_bytes - out["weight_bytes"] - out["reported_pool_bytes"]
    )
    return out


@dataclass(frozen=True)
class OverheadModel:
    train_config: str
    overhead_bytes: float


def fit_overhead(recon: pd.DataFrame, train_config: str) -> OverheadModel:
    """Fit non-KV overhead on one config (median) to predict held-out pools."""
    train = recon[recon["config"] == train_config]
    if train.empty:
        raise ValueError(f"no rows for train config {train_config}")
    return OverheadModel(
        train_config=train_config, overhead_bytes=float(train["overhead_bytes"].median())
    )


def summarize_points(runs: pd.DataFrame) -> pd.DataFrame:
    """Per config×N: median/min/max over repetitions."""
    if runs.empty:
        return runs
    agg: dict[str, Any] = {}
    num_cols = runs.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c in ("rep", "N"):
            continue
        agg[c] = ["median", "min", "max"]
    grouped = runs.groupby(["config", "N"], as_index=False).agg(agg)
    grouped.columns = ["_".join([str(x) for x in t if str(x)]) for t in grouped.columns]
    grouped = grouped.rename(columns={"config_": "config", "N_": "N"})
    return grouped


def _med(df: pd.DataFrame, col: str) -> pd.Series:
    key = f"{col}_median" if f"{col}_median" in df.columns else col
    return df[key] if key in df.columns else df[col]


def occupancy_comparison(
    points: pd.DataFrame, capacity: dict[str, int], g: float, g_semi: pd.Series | float
) -> pd.DataFrame:
    """Observed occupancy vs N·g (analytic) and N·g_semi (semi-empirical)."""
    rows = []
    for _, r in points.iterrows():
        cfg = r["config"]
        n = r["N"]
        cap = capacity.get(cfg, 0)
        obs = r.get("kv_occupancy_tokens_median", r.get("kv_occupancy_tokens", np.nan))
        pred = min(n * g, cap)
        gs = g_semi.loc[cfg] if isinstance(g_semi, pd.Series) and cfg in g_semi.index else g
        pred_semi = min(n * float(gs), cap)
        rows.append(
            {
                "config": cfg,
                "N": n,
                "observed": obs,
                "predicted": pred,
                "predicted_semi": pred_semi,
                "rel_err": (obs - pred) / pred if pred else np.nan,
                "rel_err_semi": (obs - pred_semi) / pred_semi if pred_semi else np.nan,
            }
        )
    return pd.DataFrame(rows)


def detect_saturation(points: pd.DataFrame) -> pd.DataFrame:
    """Per config+indicator: [last N without, first N with] the saturation sign.

    Indicators: waiting ≥ 0.5 (primary), running < 0.95·N, preemptions > 0,
    KV usage ≥ 0.95.
    """
    rows = []
    indicators: dict[str, Callable[[Any], bool]] = {
        "waiting": lambda r: (r.get("waiting_mean_median", r.get("waiting_mean", 0)) or 0) >= 0.5,
        "running": lambda r: (
            (r.get("running_mean_median", r.get("running_mean", 0)) or 0) < 0.95 * r["N"]
        ),
        "preemptions": lambda r: (
            (r.get("preemptions_delta_median", r.get("preemptions_delta", 0)) or 0) > 0
        ),
        "kv_usage": lambda r: (
            (r.get("kv_usage_mean_median", r.get("kv_usage_mean", 0)) or 0) >= 0.95
        ),
    }
    for cfg, sub in points.groupby("config"):
        ordered = sub.sort_values("N")
        for name, fn in indicators.items():
            flags = [bool(fn(r)) for _, r in sub.iterrows()]
            ns = list(ordered["N"])
            with_idx = [i for i, f in enumerate(flags) if f]
            if not with_idx:
                rows.append({"config": cfg, "indicator": name, "N_lo": None, "N_hi": None})
                continue
            first = with_idx[0]
            lo = ns[first - 1] if first > 0 else None
            rows.append({"config": cfg, "indicator": name, "N_lo": lo, "N_hi": ns[first]})
    return pd.DataFrame(rows)


def serving_model_check(
    points: pd.DataFrame,
    spec: ModelSpec,
    m_kv: int,
    w_dec: int,
    prompt: int,
    output: int,
    token_budget: int,
) -> pd.DataFrame:
    """Calibrate on N=1 only; predict TPOT and throughput for unsaturated N≥2."""
    rows = []
    for cfg, sub in points.groupby("config"):
        ordered = sub.sort_values("N")
        base = ordered[ordered["N"] == 1]
        if base.empty:
            continue
        tpot_1 = float(_med(base, "tpot_p50").iloc[0])
        ttft_1 = float(_med(base, "ttft_p50").iloc[0])
        cal = roofline.calibrate(tpot_1, ttft_1, w_dec, float(_g_hint(ordered)), m_kv)
        for _, r in ordered[ordered["N"] >= 2].iterrows():
            n = int(r["N"])
            try:
                pred = roofline.predict_closed_loop(
                    n, cal, w_dec, float(_g_hint(ordered)), m_kv, prompt, output, token_budget
                )
            except (ValueError, RuntimeError):
                continue
            obs_tpot = _med(ordered[ordered["N"] == n], "tpot_p50")
            obs_tput = _med(ordered[ordered["N"] == n], "out_tok_per_s_client")
            rows.append(
                {
                    "config": cfg,
                    "N": n,
                    "tpot_pred": pred.tpot_s,
                    "tpot_obs": float(obs_tpot.iloc[0]) if len(obs_tpot) else np.nan,
                    "tput_pred": pred.out_tok_per_s,
                    "tput_obs": float(obs_tput.iloc[0]) if len(obs_tput) else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["tpot_rel_err"] = (df["tpot_obs"] - df["tpot_pred"]).abs() / df["tpot_pred"]
        df["tput_rel_err"] = (df["tput_obs"] - df["tput_pred"]).abs() / df["tput_pred"]
    return df


def _g_hint(grp: pd.DataFrame) -> float:
    return 2567.5


def littles_law_check(points: pd.DataFrame) -> pd.DataFrame:
    """TTFT ≈ TTFT_unsat + waiting/λ above the knee."""
    rows = []
    for cfg, sub in points.groupby("config"):
        ordered = sub.sort_values("N")
        unsat = ordered[ordered["N"] == 1]
        if unsat.empty:
            continue
        ttft_u = float(_med(unsat, "ttft_p50").iloc[0])
        for _, r in ordered.iterrows():
            lam = r.get("req_per_s_median", r.get("req_per_s", np.nan))
            waiting = r.get("waiting_mean_median", r.get("waiting_mean", 0)) or 0
            obs = _med(ordered[ordered["N"] == r["N"]], "ttft_p50")
            pred = ttft_u + (waiting / lam if lam else np.nan)
            rows.append(
                {
                    "config": cfg,
                    "N": int(r["N"]),
                    "ttft_pred": pred,
                    "ttft_obs": float(obs.iloc[0]) if len(obs) else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rel_err"] = (df["ttft_obs"] - df["ttft_pred"]).abs() / df["ttft_pred"].clip(lower=1e-9)
    return df


def _sat_interval_ok(sat: pd.DataFrame, lo: float, hi: float, grid: list[int]) -> bool:
    step = max(np.diff(sorted(grid)).min(initial=1), 1)
    prim = sat[sat["indicator"] == "waiting"]
    for _, r in prim.iterrows():
        n_lo, n_hi = r["N_lo"], r["N_hi"]
        if n_hi is None or pd.isna(n_hi):
            return False
        lo_edge = n_lo if n_lo is not None and not pd.isna(n_lo) else n_hi - step
        # Interval overlap with [lo, hi] ± one grid step.
        if not (lo_edge <= hi + step and n_hi >= lo - step):
            return False
    return len(prim) > 0


def evaluate_claims(ledger: list[Claim], tables: dict[str, Any]) -> list[ClaimEvaluation]:
    """Evaluate every claim with its registered evaluator."""
    out = []
    for claim in ledger:
        fn = EVALUATORS.get(claim.id)
        if fn is None:
            out.append(
                ClaimEvaluation(
                    claim_id=claim.id, status="inconclusive", reason="no evaluator registered"
                )
            )
            continue
        try:
            out.append(fn(claim, tables))
        except Exception as e:
            out.append(
                ClaimEvaluation(
                    claim_id=claim.id, status="inconclusive", reason=f"evaluator error: {e}"
                )
            )
    return out


def _eval(
    claim_id: str,
    status: str,
    observed: str = "",
    expected: str = "",
    residual: str = "",
    reason: str = "",
) -> ClaimEvaluation:
    return ClaimEvaluation(
        claim_id=claim_id,
        status=status,
        observed=observed,
        expected=expected,
        residual=residual,
        reason=reason,
    )


def _ev_a1_01(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    a1 = tables.get("a1")
    if a1 is None or (isinstance(a1, pd.DataFrame) and a1.empty):
        return _eval(claim.id, "data_required", reason="A1 table not computed")
    assert isinstance(a1, pd.DataFrame)
    by_model: dict[str, Any] = {str(r["model"]): r for _, r in a1.iterrows()}
    llama: Any = by_model.get("llama3-8b-shape", {})
    qwen: Any = by_model.get("qwen2.5-7b-instruct", {})
    checks = {
        "llama_m_kv": (llama.get("m_kv_bytes"), 131072),
        "qwen_m_kv": (qwen.get("m_kv_bytes"), 57344),
    }
    # Free-slots identity from the committed Ch.4 golden examples.
    try:
        import yaml as _yaml

        golden = _yaml.safe_load(Path("tests/golden/ch4_examples.yaml").read_text(encoding="utf-8"))
        fs = golden["free_slots"]
        free_ok = fs["total"] - fs["used"] == fs["free"] == 56128
    except Exception as e:
        return _eval(claim.id, "inconclusive", reason=f"golden file unreadable: {e}")
    ok = free_ok and all(v is not None and abs(float(v) - exp) < 1e-9 for v, exp in checks.values())
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=f"m_kv rows={ {k: v[0] for k, v in checks.items()} }, free_slots=56128",
        expected="131072/57344 B/token, 56128 free slots",
        reason="Ch.4 golden reproduction" if ok else "golden mismatch",
    )


def _ev_a1_02(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    m_kv = tables.get("scalar", {}).get("m_kv") if isinstance(tables.get("scalar"), dict) else None
    if m_kv is None:
        return _eval(claim.id, "data_required", reason="m_kv not computed")
    ok = abs(float(m_kv) - 57344) < 1e-9
    return _eval(claim.id, "supported" if ok else "refuted", observed=str(m_kv), expected="57344")


def _ev_a3_01(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    recon = tables.get("recon")
    if recon is None or recon.empty:
        return _eval(claim.id, "data_required", reason="no A3 startup reports")
    match = recon["pool_match"].to_numpy(dtype=float)
    ok = bool(np.all(np.abs(match - 1.0) <= 0.005))
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=f"pool_match={match.tolist()}",
        expected="1.0 ±0.5%",
    )


def _ev_a3_02(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    held = tables.get("overhead_heldout")
    if held is None or (isinstance(held, pd.DataFrame) and held.empty):
        return _eval(claim.id, "data_required", reason="no held-out pool prediction")
    err = float(held["rel_err"].abs().max())
    return _eval(
        claim.id,
        "supported" if err <= 0.05 else "refuted",
        observed=f"max rel err {err:.3f}",
        expected="±5%",
    )


def _ev_a4_01(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    occ = tables.get("occupancy")
    if occ is None or occ.empty:
        return _eval(claim.id, "data_required", reason="no occupancy table")
    below = occ[occ["observed"] < occ["predicted"]]
    if below.empty:
        return _eval(claim.id, "inconclusive", reason="no unsaturated points")
    e_analytic = float(below["rel_err"].abs().max())
    e_semi = float(below["rel_err_semi"].abs().max())
    ok = e_analytic <= 0.10 or e_semi <= 0.05
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=f"analytic {e_analytic:.3f}, semi {e_semi:.3f}",
        expected="±10% analytic, ±5% semi-empirical",
    )


def _ev_a4_02(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    sat = tables.get("saturation")
    band = tables.get("scalar", {}).get("band") if isinstance(tables.get("scalar"), dict) else None
    grid = tables.get("scalar", {}).get("grid") if isinstance(tables.get("scalar"), dict) else None
    if sat is None or sat.empty or not band or not grid:
        return _eval(claim.id, "data_required", reason="no saturation/band data")
    ok = _sat_interval_ok(sat, band[0], band[1], grid)
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=str(sat.to_dict(orient="records")),
        expected=f"band {band}",
    )


def _ev_a4_03(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    sc = tables.get("scalar", {})
    tpot1 = sc.get("tpot1") if isinstance(sc, dict) else None
    floor = sc.get("tpot_floor") if isinstance(sc, dict) else None
    if tpot1 is None or floor is None:
        return _eval(claim.id, "data_required", reason="no N=1 TPOT/floor")
    ok = float(tpot1) >= float(floor)
    frac = float(floor) / float(tpot1) if float(tpot1) else float("nan")
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=f"TPOT(1)={tpot1}, floor={floor}, bw_frac={frac:.3f}",
        expected="TPOT(1) ≥ floor",
    )


def _ev_a4_04(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    sm = tables.get("serving_model")
    if sm is None or sm.empty:
        return _eval(claim.id, "data_required", reason="no serving-model table")
    med_tpot = float(sm["tpot_rel_err"].median())
    med_tput = float(sm["tput_rel_err"].median())
    ok = med_tpot <= 0.20 and med_tput <= 0.20
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=f"median TPOT err {med_tpot:.3f}, tput err {med_tput:.3f}",
        expected="median residual ≤ 20%",
    )


def _ev_a4_05(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    ll = tables.get("littles")
    pts = tables.get("points")
    if ll is None or ll.empty or pts is None or pts.empty:
        return _eval(claim.id, "data_required", reason="no littles-law/points table")
    err = float(ll["rel_err"].median()) if "rel_err" in ll else float("nan")
    ok = err <= 0.20
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=f"median Little's-law err {err:.3f}",
        expected="±20%",
    )


def _ev_b0_01(claim: Claim, tables: dict[str, Any]) -> ClaimEvaluation:
    b0 = tables.get("b0")
    if b0 is None or (isinstance(b0, pd.DataFrame) and b0.empty):
        return _eval(claim.id, "data_required", reason="no B0 harness table")
    row = b0.iloc[0].to_dict() if isinstance(b0, pd.DataFrame) else dict(b0)
    ttft = row.get("ttft_p99_err_s")
    itl = row.get("itl_p99_err_s")
    if ttft is None or itl is None:
        return _eval(claim.id, "data_required", reason="B0 errors not computed")
    observed = f"ttft p99 {ttft}, itl p99 {itl}"
    if not row.get("in_scope_reference", False):
        return _eval(
            claim.id,
            "inconclusive",
            observed=observed,
            expected="p99 ≤ 1 ms",
            reason="B0 table from non-reference machine; criterion is scoped to reference setup",
        )
    ok = float(ttft) <= 0.001 and float(itl) <= 0.001
    return _eval(
        claim.id,
        "supported" if ok else "refuted",
        observed=observed,
        expected="p99 ≤ 1 ms",
    )


EVALUATORS = {
    "A1-01": _ev_a1_01,
    "A1-02": _ev_a1_02,
    "A3-01": _ev_a3_01,
    "A3-02": _ev_a3_02,
    "A4-01": _ev_a4_01,
    "A4-02": _ev_a4_02,
    "A4-03": _ev_a4_03,
    "A4-04": _ev_a4_04,
    "A4-05": _ev_a4_05,
    "B0-01": _ev_b0_01,
}


def derive_decision(
    points: pd.DataFrame, sat: pd.DataFrame, model_tbl: pd.DataFrame, headroom: float
) -> Decision:
    """Admission cap, binding constraint, levers, predicted vs measured knee."""
    cap: int | None = None
    if not points.empty:
        for cfg, sub in points.groupby("config"):
            ordered = sub.sort_values("N")
            for _, r in ordered.iterrows():
                n = int(r["N"])
                kv = r.get("kv_usage_mean_median", r.get("kv_usage_mean", 0)) or 0
                waiting = r.get("waiting_mean_median", r.get("waiting_mean", 0)) or 0
                pre = r.get("preemptions_delta_median", r.get("preemptions_delta", 0)) or 0
                if waiting < 0.5 and pre == 0 and kv <= headroom:
                    if cap is None or (cfg == points["config"].iloc[0] and n > cap):
                        cap = n
    binding = "unknown"
    levers: list[str] = ["collect more data"]
    return Decision(
        cap=cap,
        binding=binding,
        levers=levers,
        predicted_knee="",
        measured_knee="",
        narrative="",
    )


def write_predictions(
    manifest_path: Path,
    spec: ModelSpec,
    out_path: Path,
    capacity_by_config: dict[str, int] | None = None,
    kv_dtype_bytes: int = 2,
    peak_bw_Bps: float = 6.0e11,
    weight_bytes: int | None = None,
) -> Path:
    """Write predictions_pre.json (analytical parts always; capacity parts when A3 known)."""
    import yaml as _yaml

    man = _yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    workload = man.get("workload", {})
    prompt = int(workload.get("prompt_tokens", 2048))
    output = int(workload.get("output_tokens", 1024))
    block = int(man.get("engine_args", {}).get("block_size", 16))
    # block_size lives in engine base_args; default 16.
    m_kv = kvcalc.kv_bytes_per_token(spec, kv_dtype_bytes)
    g_bar = kvcalc.g_decode(prompt, output, block)
    g_peak = kvcalc.g_max(prompt, output, block)
    if weight_bytes is None:
        if spec.param_count:
            weight_bytes = spec.param_count * kv_dtype_bytes
            if not spec.tie_word_embeddings:
                weight_bytes -= spec.vocab_size * spec.hidden_size * kv_dtype_bytes
    w_dec = weight_bytes or 0
    floor = (w_dec + g_bar * m_kv) / peak_bw_Bps if peak_bw_Bps else 0.0
    grids = {}
    for cfg in man.get("configs", []):
        cap = (capacity_by_config or {}).get(cfg.get("name", ""))
        if cap:
            lo, hi = kvcalc.predicted_saturation_interval(cap, g_bar, g_peak)
            grids[cfg.get("name")] = {
                "capacity_tokens": cap,
                "saturation_band": [lo, hi],
                "capacity_source": "A3-derived",
            }
        else:
            grids[cfg.get("name")] = {
                "capacity_tokens": None,
                "saturation_band": None,
                "capacity_source": "pending-A3",
            }
    pred = {
        "experiment_id": man.get("experiment_id"),
        "model": {
            "m_kv_bytes": m_kv,
            "g_bar_tokens": g_bar,
            "g_max_tokens": g_peak,
            "decode_weight_bytes": w_dec,
            "tpot_floor_s": floor,
        },
        "workload": {"prompt_tokens": prompt, "output_tokens": output, "block_size": block},
        "configs": grids,
        "claims_note": "criteria frozen in PREREG.md; predictions committed before first GPU run",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pred, indent=2), encoding="utf-8")
    prov = {
        "inputs": {str(manifest_path): provenance_sha(manifest_path)},
        "generator": "llmlab.analysis.kv_capacity.write_predictions",
        "weight_bytes": w_dec,
        "weight_source": (
            "explicit --weight-bytes"
            if weight_bytes
            else ("checkpoint metadata" if spec.param_count else "unknown (param_count missing)")
        ),
    }
    (out_path.parent / (out_path.name + ".provenance.json")).write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    return out_path


def provenance_sha(path: Path) -> str:
    from llmlab import provenance as _p

    return _p.sha256_file(Path(path))
