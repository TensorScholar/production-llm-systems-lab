"""llmlab CLI (argparse, no extra dependencies)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llmlab")


def _repo_root() -> Path:
    return Path.cwd()


def cmd_kv(args: argparse.Namespace) -> int:
    import yaml

    from llmlab.calc import kv as kvcalc
    from llmlab.models import load_model_spec, spec_from_hf_config
    from llmlab.schemas import ModelRef
    from llmlab.units import fmt_bytes

    ref = ModelRef(**yaml.safe_load(Path(args.model).read_text(encoding="utf-8")))
    if args.spec_json:
        spec = spec_from_hf_config(
            json.loads(Path(args.spec_json).read_text(encoding="utf-8")), name=ref.id
        )
    else:
        spec = load_model_spec(ref, Path(".cache/hf"))
    m_kv = kvcalc.kv_bytes_per_token(spec, args.kv_dtype_bytes)
    print(f"model: {spec.name} ({spec.attention}, L={spec.n_layers})")
    print(f"m_kv: {m_kv} B/token ({fmt_bytes(m_kv)})")
    if args.pool:
        from llmlab.units import parse_bytes

        pool = parse_bytes(args.pool)
        cap = kvcalc.pool_capacity(pool, m_kv, args.block_size)
        print(f"pool {args.pool}: {cap.num_blocks} blocks = {cap.tokens} tokens")
    if args.context:
        print(f"context {args.context}: {args.context * m_kv} B ({fmt_bytes(args.context * m_kv)})")
    g = kvcalc.g_decode(2048, 1024, args.block_size)
    print(f"g_bar(2048,1024,q={args.block_size}): {g:.1f} tokens")
    return 0


def _registered_models() -> list[tuple[str, str | None]]:
    """Registered model configs: (config path, spec-json fixture or None)."""
    out: list[tuple[str, str | None]] = []
    for p in sorted(Path("configs/models").glob("*.yaml")):
        out.append((str(p), None))
    return out


def cmd_a1(args: argparse.Namespace) -> int:
    import yaml

    from llmlab.calc import kv as kvcalc
    from llmlab.models import spec_from_hf_config
    from llmlab.schemas import ModelRef

    manifest = Path(args.manifest)
    _ = manifest
    rows = []
    for cfg_path, _ in _registered_models():
        ref = ModelRef(**yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")))
        # Shape from committed fixture when the revision is unpinned (pre-T13).
        fix = Path("tests/fixtures/hf_configs/qwen2.5-7b.json")
        spec_json = json.loads(fix.read_text(encoding="utf-8"))
        spec = spec_from_hf_config(spec_json, name=ref.id)
        m_kv = kvcalc.kv_bytes_per_token(spec, ref.kv_cache_dtype_bytes)
        rows.append(
            {
                "model": ref.id,
                "layers": spec.n_layers,
                "kv_heads": spec.n_kv_heads,
                "head_dim": spec.head_dim,
                "attention": spec.attention,
                "m_kv_bytes": m_kv,
                "g_bar_2048_1024_q16": kvcalc.g_decode(2048, 1024, 16),
                "g_max_2048_1024_q16": kvcalc.g_max(2048, 1024, 16),
            }
        )
    # Llama-3-8B-shaped reference config (Ch.4 golden: m_kv=131072).
    llama_json = json.loads(
        Path("tests/fixtures/hf_configs/llama3-8b-shape.json").read_text(encoding="utf-8")
    )
    from llmlab.models import spec_from_hf_config as _s

    lspec = _s(llama_json, name="llama3-8b-shape")
    rows.append(
        {
            "model": "llama3-8b-shape",
            "layers": lspec.n_layers,
            "kv_heads": lspec.n_kv_heads,
            "head_dim": lspec.head_dim,
            "attention": lspec.attention,
            "m_kv_bytes": kvcalc.kv_bytes_per_token(lspec, 2),
            "g_bar_2048_1024_q16": kvcalc.g_decode(2048, 1024, 16),
            "g_max_2048_1024_q16": kvcalc.g_max(2048, 1024, 16),
        }
    )
    out = Path("results/derived/A1/kv_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    from llmlab import provenance as _p

    (out.parent / "kv_table.csv.provenance.json").write_text(
        json.dumps(
            {
                "inputs": [str(p) for p, _ in _registered_models()],
                "generator": "llmlab a1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(rows)} models)")
    _ = _p
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import yaml

    from llmlab.analysis.kv_capacity import write_predictions
    from llmlab.models import spec_from_hf_config
    from llmlab.schemas import ModelRef

    man_path = Path(args.manifest)
    man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    model_cfg = man_path.parent / man["model"]
    ref = ModelRef(**yaml.safe_load(Path(model_cfg).read_text(encoding="utf-8")))
    if args.spec_json:
        spec = spec_from_hf_config(
            json.loads(Path(args.spec_json).read_text(encoding="utf-8")), name=ref.id
        )
    else:
        from llmlab.models import load_model_spec

        spec = load_model_spec(ref, Path(".cache/hf"))
    capacity: dict[str, int] = {}
    a3_path = Path("results/derived/A3/reconciliation.csv")
    if a3_path.exists():
        import pandas as pd

        recon = pd.read_csv(a3_path)
        if "config" in recon.columns and "reported_pool_tokens" in recon.columns:
            for _, r in recon.iterrows():
                capacity[str(r["config"])] = int(r["reported_pool_tokens"])
    exp = man.get("experiment_id", "A4")
    out = Path(f"results/derived/{exp}/predictions_pre.json")
    write_predictions(man_path, spec, out, capacity or None)
    print(f"wrote {out} (capacity_source={'A3-derived' if capacity else 'pending-A3'})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from llmlab.runner import run_experiment

    produced = run_experiment(
        Path(args.manifest),
        Path(args.results or "results"),
        configs=args.configs,
        reps=[int(x) for x in args.reps] if args.reps else None,
        allow_dirty=args.allow_dirty,
        dry_run=args.dry_run,
    )
    print(f"runs: {len(produced)}")
    for p in produced:
        print(f"  {p}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    import pandas as pd
    import yaml

    from llmlab import provenance as _p
    from llmlab.analysis import kv_capacity as _a
    from llmlab.calc import kv as _kv
    from llmlab.models import spec_from_hf_config
    from llmlab.schemas import Claim, ModelRef

    man_path = Path(args.manifest)
    man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    exp = man.get("experiment_id", "A4")
    results = Path("results")
    derived = results / "derived" / exp
    derived.mkdir(parents=True, exist_ok=True)
    workload = man.get("workload", {})
    prompt = int(workload.get("prompt_tokens", 2048))
    output = int(workload.get("output_tokens", 1024))
    # Model spec from fixture (pre-T13) or network.
    model_cfg = man_path.parent / man["model"]
    ref = ModelRef(**yaml.safe_load(Path(model_cfg).read_text(encoding="utf-8")))
    if args.spec_json:
        spec = spec_from_hf_config(
            json.loads(Path(args.spec_json).read_text(encoding="utf-8")), name=ref.id
        )
    else:
        from llmlab.models import load_model_spec

        spec = load_model_spec(ref, Path(".cache/hf"))
    m_kv = _kv.kv_bytes_per_token(spec, ref.kv_cache_dtype_bytes)
    g_bar = _kv.g_decode(prompt, output, 16)
    runs = _a.load_runs(results, exp, valid_only=True)
    points = _a.summarize_points(runs) if not runs.empty else pd.DataFrame()
    tables: dict[str, Any] = {"points": points}
    if not points.empty:
        points.to_csv(derived / "points.csv", index=False)
    # Capacity per config: prefer A3 reconciliation, else startup reports.
    capacity: dict[str, int] = {}
    occ = pd.DataFrame()
    sat = pd.DataFrame()
    if not points.empty:
        # Capacity unknown pre-A3: use occupancy-derived placeholder handling —
        # mark DATA REQUIRED via empty capacity (analysis still writes shapes).
        occ = _a.occupancy_comparison(points, capacity, g_bar, g_bar)
        sat = _a.detect_saturation(points)
        occ.to_csv(derived / "occupancy.csv", index=False)
        sat.to_csv(derived / "saturation.csv", index=False)
        tables.update({"occupancy": occ, "saturation": sat})
    # A1 + scalar tables for claim evaluators.
    a1p = results / "derived" / "A1" / "kv_table.csv"
    if a1p.exists():
        tables["a1"] = pd.read_csv(a1p)
    tables["scalar"] = {
        "m_kv": m_kv,
        "band": None,
        "grid": sorted({int(n) for c in man.get("configs", []) for n in c.get("concurrency", [])}),
        "tpot1": None,
        "tpot_floor": None,
    }
    b0p = results / "derived" / "B0" / "harness_accuracy.csv"
    if b0p.exists():
        tables["b0"] = pd.read_csv(b0p)
    ledger_path = Path("claims/ledger.yaml")
    ledger = [Claim(**c) for c in yaml.safe_load(ledger_path.read_text(encoding="utf-8"))]
    evals = _a.evaluate_claims(ledger, tables)
    (derived / "claims_eval.json").write_text(
        json.dumps([e.model_dump() for e in evals], indent=2), encoding="utf-8"
    )
    headroom = float(man.get("decision", {}).get("kv_headroom", 0.90))
    decision = _a.derive_decision(points, sat, pd.DataFrame(), headroom)
    (derived / "decision.json").write_text(decision.model_dump_json(indent=2), encoding="utf-8")
    for name in ("points", "occupancy", "saturation", "claims_eval", "decision"):
        suffix = ".csv" if name in ("points", "occupancy", "saturation") else ".json"
        p = derived / (name + suffix)
        if p.exists():
            (derived / (p.name + ".provenance.json")).write_text(
                json.dumps({"inputs": ["results/runs"], "generator": "llmlab analyze"}, indent=2),
                encoding="utf-8",
            )
    _ = _p
    print(f"wrote {derived} (m_kv={m_kv}, g_bar={g_bar:.1f})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import yaml

    from llmlab import report as _r

    man_path = Path(args.manifest)
    man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    exp = man.get("experiment_id", "A4")
    derived = Path("results/derived") / exp
    fig_dir = Path("docs/figures") / exp
    meta = {
        "evidence": "MOCK",
        "run_ids": [],
        "claim_ids": [
            c.get("id")
            for c in yaml.safe_load(Path("claims/ledger.yaml").read_text(encoding="utf-8"))
        ],
    }
    made = _r.make_figures(derived, fig_dir, meta)
    template = Path("docs/stories/01-kv-capacity.md")
    if template.exists():
        _r.render_story(template, derived, template)
        print(f"rendered {template}")
    print(f"figures: {len(made)} in {fig_dir}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from llmlab.validate import validate_all

    ok, messages = validate_all(Path(args.results), allow_unpinned=args.allow_unpinned)
    for m in messages:
        print(m)
    return 0 if ok else 1


def cmd_archive(args: argparse.Namespace) -> int:
    import hashlib

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / "raw-chunks.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in sorted(Path("results/runs").rglob("chunks.parquet")):
            tar.add(p, arcname=str(p.relative_to("results")))
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    manifest = {"archive": str(tar_path), "sha256": h, "doi": "PENDING-T24"}
    Path("results/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"archived {tar_path} sha256={h[:16]}…")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    import pydantic

    from llmlab import schemas as _s

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [
        "ModelRef",
        "HardwareProfile",
        "EngineConfig",
        "MetricsMap",
        "WorkloadSpec",
        "ClientConfig",
        "SweepConfig",
        "ExperimentManifest",
        "StartupReport",
        "EngineInstance",
        "RequestRecord",
        "ChunkRecord",
        "CheckResult",
        "RunMetadata",
        "Claim",
        "ClaimEvaluation",
        "Decision",
    ]
    for name in models:
        cls = getattr(_s, name)
        schema = cls.model_json_schema() if issubclass(cls, pydantic.BaseModel) else {}
        (out / f"{name}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"exported {len(models)} schemas to {out}")
    return 0


def cmd_demo_mock(args: argparse.Namespace) -> int:
    """End-to-end mock pipeline into build/demo (MOCK labels, never results/)."""
    import subprocess as _sp

    import yaml

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    # Minimal mock manifests (small grids, short windows) for CI speed.
    demo_cfgs = [
        ("A4", "experiments/A_kv_cache/A4_kv_saturation/manifest.yaml"),
        ("B0", "experiments/B_serving/B0_harness_accuracy/manifest.yaml"),
    ]
    for exp, manifest in demo_cfgs:
        man = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
        man["repetitions"] = 1
        for c in man.get("configs", []):
            c["concurrency"] = c.get("concurrency", [])[:3]
        man["client"]["window_min_s"] = 3.0
        man["client"]["window_max_s"] = 10.0
        man["client"]["min_completed"] = 5
        man["client"]["warmup_s"] = 0.2
        man["engine"] = "configs/engines/mock.yaml"
        tmp = out / f"{exp}-manifest.yaml"
        tmp.write_text(yaml.safe_dump(man), encoding="utf-8")
        r = _sp.run(
            [
                sys.executable,
                "-m",
                "llmlab.cli",
                "run",
                str(tmp),
                "--results",
                str(out / "results"),
                "--allow-dirty",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
            return r.returncode
    print(f"demo-mock complete in {out} (MOCK evidence only)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmlab", description="Production LLM Systems Lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("kv", help="Print the KV table for a model config")
    k.add_argument("--model", required=True)
    k.add_argument("--block-size", type=int, default=16)
    k.add_argument("--kv-dtype-bytes", type=int, default=2)
    k.add_argument("--pool", default=None)
    k.add_argument("--context", type=int, default=None)
    k.add_argument("--spec-json", default=None)
    k.set_defaults(func=cmd_kv)

    a = sub.add_parser("a1", help="Write results/derived/A1/kv_table.csv")
    a.add_argument("manifest")
    a.set_defaults(func=cmd_a1)

    pr = sub.add_parser("predict", help="Write predictions_pre.json")
    pr.add_argument("manifest")
    pr.add_argument("--spec-json", default=None)
    pr.set_defaults(func=cmd_predict)

    r = sub.add_parser("run", help="Execute runs")
    r.add_argument("manifest")
    r.add_argument("--configs", nargs="*", default=None)
    r.add_argument("--reps", nargs="*", default=None)
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--results", default="results")
    r.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", help="Write derived tables, claims_eval, decision")
    an.add_argument("manifest")
    an.add_argument("--spec-json", default=None)
    an.set_defaults(func=cmd_analyze)

    rp = sub.add_parser("report", help="Write figures and story")
    rp.add_argument("manifest")
    rp.set_defaults(func=cmd_report)

    v = sub.add_parser("validate", help="Integrity checks")
    v.add_argument("--results", default="results/")
    v.add_argument("--allow-unpinned", action="store_true")
    v.set_defaults(func=cmd_validate)

    ar = sub.add_parser("archive", help="Bundle raw chunks")
    ar.add_argument("--out", default="build/archive/")
    ar.set_defaults(func=cmd_archive)

    s = sub.add_parser("schema", help="Export JSON Schemas")
    s.add_argument("--out", default="docs/schemas/")
    s.set_defaults(func=cmd_schema)

    d = sub.add_parser("demo-mock", help="End-to-end mock pipeline into build/demo")
    d.add_argument("--out", default="build/demo")
    d.set_defaults(func=cmd_demo_mock)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except Exception as e:
        from llmlab.engines.vllm import EngineStartupError, ParseError
        from llmlab.runner import ManifestError

        if isinstance(e, ManifestError | EngineStartupError | ParseError):
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(2) from e
        raise


if __name__ == "__main__":
    main()
