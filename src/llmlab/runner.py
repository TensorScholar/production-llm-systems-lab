"""Experiment execution: manifests → immutable runs with provenance and checks."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from llmlab import provenance
from llmlab.engines.vllm import VLLMServer
from llmlab.loadgen import run_closed_loop
from llmlab.metrics import summarize_run
from llmlab.schemas import (
    CheckResult,
    ClientConfig,
    EngineConfig,
    ExperimentManifest,
    HardwareProfile,
    MetricsMap,
    ModelRef,
    RequestRecord,
    RunMetadata,
    load_yaml,
)
from llmlab.telemetry import NvmlSampler, PrometheusScraper
from llmlab.workload import request_stream, safe_ids_digest, safe_token_ids

logger = logging.getLogger(__name__)


class ManifestError(Exception):
    """Manifest or configuration error."""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_manifest(manifest_path: Path) -> tuple[ExperimentManifest, Path]:
    raw = yaml.safe_load(_read_text(manifest_path))
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest is not a mapping: {manifest_path}")
    try:
        man = ExperimentManifest(**raw)
    except Exception as e:
        raise ManifestError(f"invalid manifest {manifest_path}: {e}") from e
    return man, manifest_path.parent


def _load_ref(path: Path, model: Any) -> Any:
    return load_yaml(path, model)


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_placeholders(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, str):
        if "PIN-IN-T13" in obj:
            found.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += _check_placeholders(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _check_placeholders(v, f"{path}[{i}]")
    return found


def _wait_idle(
    server: VLLMServer, metrics_map: MetricsMap, kv_max: float, timeout_s: float
) -> tuple[bool, str]:
    import requests as _rq

    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            text = _rq.get(f"{server.base_url}/metrics", timeout=10).text
            from llmlab.telemetry import parse_metrics_text

            parsed = parse_metrics_text(text, metrics_map)
            kv = parsed.get("kv_usage", 1.0)
            running = parsed.get("running", 1.0)
            waiting = parsed.get("waiting", 0.0)
            if kv <= kv_max and running == 0 and waiting == 0:
                return True, ""
            last = f"kv={kv} running={running} waiting={waiting}"
        except Exception as e:
            last = str(e)
        time.sleep(2.0)
    return False, last


def _run_checks(
    *,
    records: list[RequestRecord],
    summary: dict[str, Any],
    metrics_present: dict[str, bool],
    essential: list[str],
    loop_lag_p99_ms: float,
    client_cpu_max: float,
    throttle_reasons: int | None,
    idle_ok: bool,
    idle_detail: str,
    cfg: ClientConfig,
    checks_cfg: Any,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, CheckResult]:
    in_w = [r for r in records if r.in_window]
    ok = [r for r in in_w if r.status == "ok"]
    err_rate = 1.0 - (len(ok) / len(in_w)) if in_w else 1.0
    checks: dict[str, CheckResult] = {}
    checks["error_rate"] = CheckResult(
        passed=err_rate <= checks_cfg.max_error_rate,
        value=err_rate,
        threshold=checks_cfg.max_error_rate,
        invalidating=True,
        reason=None if err_rate <= checks_cfg.max_error_rate else f"error rate {err_rate:.3f}",
    )
    mism = [
        r.request_id
        for r in ok
        if r.n_prompt_tokens != prompt_tokens or r.n_output_tokens != output_tokens
    ]
    checks["usage_tokens"] = CheckResult(
        passed=not mism,
        value=len(mism),
        threshold=0,
        invalidating=True,
        reason=None if not mism else f"{len(mism)} usage mismatches, e.g. {mism[:3]}",
    )
    missing = [k for k in essential if not metrics_present.get(k, False)]
    checks["essential_metrics"] = CheckResult(
        passed=not missing,
        value=len(missing),
        threshold=0,
        invalidating=True,
        reason=None if not missing else f"missing: {missing}",
    )
    lag_ok = loop_lag_p99_ms <= checks_cfg.loop_lag_p99_ms
    cpu_ok = client_cpu_max <= checks_cfg.client_cpu_max
    checks["client_health"] = CheckResult(
        passed=bool(lag_ok and cpu_ok),
        value=loop_lag_p99_ms,
        threshold=checks_cfg.loop_lag_p99_ms,
        invalidating=True,
        reason=None
        if (lag_ok and cpu_ok)
        else f"lag_p99={loop_lag_p99_ms:.2f}ms cpu={client_cpu_max:.2f}",
    )
    if throttle_reasons is None:
        checks["no_throttle"] = CheckResult(
            passed=True,
            value=None,
            threshold=0,
            invalidating=False,
            reason="gpu telemetry unavailable — not evaluated",
        )
    else:
        checks["no_throttle"] = CheckResult(
            passed=throttle_reasons == 0,
            value=throttle_reasons,
            threshold=0,
            invalidating=True,
            reason=None if throttle_reasons == 0 else f"throttle reasons={throttle_reasons}",
        )
    checks["idle_before_start"] = CheckResult(
        passed=idle_ok,
        value=idle_ok,
        threshold=True,
        invalidating=True,
        reason=None if idle_ok else idle_detail,
    )
    # Stability: throughput in halves within ±10% when ≥40 completions.
    completed = list(ok)
    if len(completed) >= 40:
        import numpy as np

        sends = np.array(sorted(r.t_send_ns for r in completed))
        mid = sends[len(sends) // 2]
        w0 = min(r.t_send_ns for r in completed)
        w1 = max(r.t_send_ns for r in completed)
        h1 = float(((sends <= mid).sum()) / max((mid - w0) / 1e9, 1e-9))
        h2 = float(((sends > mid).sum()) / max((w1 - mid) / 1e9, 1e-9))
        denom = max(h1, h2, 1e-9)
        stable = abs(h1 - h2) / denom <= checks_cfg.stability_tolerance
        checks["stability"] = CheckResult(
            passed=stable,
            value=abs(h1 - h2) / denom,
            threshold=checks_cfg.stability_tolerance,
            invalidating=True,
            reason=None if stable else f"halves {h1:.3f} vs {h2:.3f}/s",
        )
    else:
        checks["stability"] = CheckResult(
            passed=True,
            value=len(completed),
            threshold=40,
            invalidating=False,
            reason="fewer than 40 completions — not evaluated",
        )
    checks["min_completed"] = CheckResult(
        passed=len(ok) >= cfg.min_completed,
        value=len(ok),
        threshold=cfg.min_completed,
        invalidating=False,
        reason=None if len(ok) >= cfg.min_completed else "below min_completed",
    )
    return checks


def _write_run_dir(
    run_dir: Path,
    meta: RunMetadata,
    records: list[RequestRecord],
    chunks: list[dict[str, Any]],
    server_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    metrics_start: str,
    metrics_end: str,
    checks: dict[str, CheckResult],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    req_rows = [r.model_dump() for r in records]
    pd.DataFrame(req_rows).to_parquet(run_dir / "requests.parquet", index=False)
    chunk_rows = [
        {
            "request_id": c["request_id"],
            "seq": c["seq"],
            "t_ns": c["t_ns"],
            "text_len": c["text_len"],
        }
        for c in chunks
    ]
    pd.DataFrame(chunk_rows, columns=["request_id", "seq", "t_ns", "text_len"]).to_parquet(
        run_dir / "chunks.parquet", index=False
    )
    if server_df.empty:
        pd.DataFrame(columns=["t_ns", "name", "value"]).to_parquet(
            run_dir / "server_metrics.parquet", index=False
        )
    else:
        server_df.to_parquet(run_dir / "server_metrics.parquet", index=False)
    if gpu_df.empty:
        pd.DataFrame().to_parquet(run_dir / "gpu.parquet", index=False)
    else:
        gpu_df.to_parquet(run_dir / "gpu.parquet", index=False)
    (run_dir / "metrics_start.txt").write_text(metrics_start, encoding="utf-8")
    (run_dir / "metrics_end.txt").write_text(metrics_end, encoding="utf-8")
    (run_dir / "checks.json").write_text(
        json.dumps({k: v.model_dump() for k, v in checks.items()}, indent=2), encoding="utf-8"
    )


def _existing_valid_run(
    exp_dir: Path, manifest_sha: str, config: str, rep: int, n: int
) -> Path | None:
    if not exp_dir.exists():
        return None
    for run_dir in sorted(exp_dir.iterdir()):
        rj = run_dir / "run.json"
        if not rj.exists():
            continue
        try:
            meta = RunMetadata(**json.loads(rj.read_text(encoding="utf-8")))
        except Exception:
            continue
        if (
            meta.manifest_sha256 == manifest_sha
            and meta.config_name == config
            and meta.repetition == rep
            and meta.concurrency == n
            and meta.valid
        ):
            return run_dir
    return None


def run_experiment(
    manifest_path: Path,
    results_dir: Path,
    *,
    configs: list[str] | None = None,
    reps: list[int] | None = None,
    allow_dirty: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Execute an experiment manifest. Returns run directories created/resumed."""
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    man, exp_dir = load_manifest(manifest_path)
    manifest_sha = provenance.sha256_file(manifest_path)

    def _resolve(p: str) -> Path:
        cand = Path(p)
        if cand.is_absolute():
            return cand
        for base in (exp_dir, Path.cwd()):
            if (base / p).exists():
                return base / p
        return exp_dir / p

    model_path = _resolve(man.model)
    hw_path = _resolve(man.hardware)
    eng_path = _resolve(man.engine)
    try:
        model_ref = _load_ref(model_path, ModelRef)
        hw = _load_ref(hw_path, HardwareProfile)
        engine = _load_ref(eng_path, EngineConfig)
    except Exception as e:
        raise ManifestError(f"config load failed: {e}") from e
    placeholders = _check_placeholders(
        {"model": model_ref.model_dump(), "engine": engine.model_dump()}
    )
    is_mock = engine.name == "mock"
    if placeholders and not is_mock:
        raise ManifestError(f"unpinned placeholders remain: {placeholders}")
    if is_mock and results_dir.name == "results" and "build" not in str(results_dir):
        raise ManifestError("mock output under results/ is refused (use build/)")
    code = provenance.capture_code()
    if code.dirty and not allow_dirty:
        raise ManifestError("dirty working tree — pass --allow-dirty to record it")
    env = provenance.capture_env()
    # Resolve metrics map path (relative to experiment dir, else repo root).
    mmap_raw = Path(engine.metrics_map)
    if mmap_raw.is_absolute():
        mmap_path = mmap_raw
    elif (exp_dir / engine.metrics_map).exists():
        mmap_path = exp_dir / engine.metrics_map
    else:
        mmap_path = Path.cwd() / engine.metrics_map
    metrics_map = MetricsMap(**yaml.safe_load(mmap_path.read_text(encoding="utf-8")))
    # Safe token IDs: real tokenizer at pinned revision, or synthetic for mock.
    if is_mock:
        safe_ids = safe_token_ids({"added_tokens": []}, 1000)
        vocab_note = "synthetic-mock-vocab-1000"
    else:
        from huggingface_hub import hf_hub_download as _dl

        tok_path = _dl(
            repo_id=model_ref.hf_repo, filename="tokenizer.json", revision=model_ref.revision
        )
        tok_json = json.loads(Path(tok_path).read_text(encoding="utf-8"))
        try:
            from llmlab.models import load_model_spec as _lms

            vocab_size = _lms(model_ref, Path(".cache/hf")).vocab_size
        except Exception:
            vocab_size = tok_json.get("model", {}).get("vocab_size", 32000)
        safe_ids = safe_token_ids(tok_json, int(vocab_size))
        vocab_note = tok_path
    _ = vocab_note
    digest = safe_ids_digest(safe_ids)
    rng = random.Random(man.order.seed)
    when = provenance.utc_stamp()
    produced: list[Path] = []
    sweep = [c for c in man.configs if configs is None or c.name in configs]
    rep_list = list(range(man.repetitions)) if reps is None else reps
    startup_only = all(len(c.concurrency) == 0 for c in man.configs)
    for sweep_cfg in sweep:
        for rep in rep_list:
            instance_id = provenance.new_instance_id(
                man.experiment_id, sweep_cfg.name, rep, when, rng
            )
            eng_dir = results_dir / "engines" / instance_id
            eng_dir.mkdir(parents=True, exist_ok=True)
            resolved_args = dict(engine.base_args)
            resolved_args.update(sweep_cfg.engine_args)
            (eng_dir / "resolved_args.json").write_text(
                json.dumps(resolved_args, indent=2, sort_keys=True), encoding="utf-8"
            )
            # Concurrency order (shuffled, seeded).
            order = list(sweep_cfg.concurrency)
            if man.order.randomize:
                r = random.Random(man.order.seed + rep)
                r.shuffle(order)
            if dry_run:
                logger.info("dry-run %s rep=%d Ns=%s", sweep_cfg.name, rep, order)
                continue
            server = VLLMServer(engine, model_ref, resolved_args, eng_dir / "engine.log", mmap_path)
            server.start()
            try:
                server.wait_ready()
                report = server.startup_report()
                (eng_dir / "startup_report.json").write_text(
                    report.model_dump_json(indent=2), encoding="utf-8"
                )
                (eng_dir / "version.json").write_text(
                    json.dumps({"version": server.version()}, indent=2), encoding="utf-8"
                )
                capacity = report.kv_cache_tokens or 0
                # Warmup (skipped in startup-only mode).
                if not startup_only:
                    _warmup(server, model_ref, man, safe_ids)
                # Startup-only mode (empty concurrency): one N=0 point per
                # (config, rep) holding the startup report and checks.
                for n in order or [0]:
                    run_id = provenance.new_run_id(
                        man.experiment_id, sweep_cfg.name, rep, n, when, rng
                    )
                    out_dir = results_dir / "runs" / man.experiment_id / run_id
                    existing = _existing_valid_run(
                        results_dir / "runs" / man.experiment_id,
                        manifest_sha,
                        sweep_cfg.name,
                        rep,
                        n,
                    )
                    if existing is not None:
                        logger.info("resume-skip %s (have %s)", run_id, existing.name)
                        produced.append(existing)
                        continue
                    pr_sha, pr_time = provenance.prereg_commit(
                        exp_dir / man.prereg if man.prereg else manifest_path
                    )
                    meta = RunMetadata(
                        run_id=run_id,
                        experiment_id=man.experiment_id,
                        manifest_sha256=manifest_sha,
                        prereg_commit=pr_sha,
                        prereg_time=pr_time,
                        code=code,
                        environment=env,
                        engine_name=engine.name,
                        engine_version=server.version(),
                        engine_args=resolved_args,
                        instance_id=instance_id,
                        gpu_name=hw.name,
                        model_id=model_ref.id,
                        model_revision=model_ref.revision,
                        workload=man.workload,
                        workload_hash=digest,
                        concurrency=n,
                        config_name=sweep_cfg.name,
                        repetition=rep,
                        started_utc=_utc_now(),
                        evidence="MOCK" if is_mock else "MEASURED",
                        tier=0 if is_mock else 2,
                    )
                    records: list[RequestRecord] = []
                    chunks: list[dict[str, Any]] = []
                    server_df = pd.DataFrame(columns=["t_ns", "name", "value"])
                    gpu_df = pd.DataFrame()
                    m_start = m_end = ""
                    checks: dict[str, CheckResult] = {}
                    if startup_only:
                        try:
                            scraper = PrometheusScraper(
                                f"{server.base_url}/metrics",
                                man.telemetry.metrics_interval_s,
                                metrics_map,
                            )
                            m_start = asyncio.run(scraper.snapshot_raw())
                            m_end = m_start
                        except Exception as e:
                            m_end = f"snapshot failed: {e}"
                        checks = {
                            "startup_only": CheckResult(
                                passed=report.kv_cache_tokens is not None,
                                invalidating=True,
                                reason=None if report.kv_cache_tokens else "no kv capacity parsed",
                            )
                        }
                    else:
                        idle_ok, idle_detail = _wait_idle(
                            server,
                            metrics_map,
                            man.idle_wait.kv_usage_max,
                            man.idle_wait.timeout_s,
                        )
                        res, m_start, m_end, server_df, gpu_df = _run_load_point(
                            server=server,
                            metrics_map=metrics_map,
                            man=man,
                            model_name=(model_ref.hf_repo if not is_mock else "mock-model"),
                            safe_ids=safe_ids,
                            n=n,
                            run_id=run_id,
                            rep=rep,
                        )
                        records = res.requests
                        chunks = res.chunks
                        from llmlab.telemetry import parse_metrics_text as _pmt

                        try:
                            if m_start and not m_start.startswith("snapshot failed"):
                                present_start = set(_pmt(m_start, metrics_map))
                            else:
                                present_start = set()
                        except Exception:
                            present_start = set()
                        if not server_df.empty and "name" in server_df.columns:
                            names = set(server_df["name"].unique().tolist())
                        else:
                            names = set()
                        metrics_present = {
                            k: (k in present_start or k in names) for k in metrics_map.essential
                        }
                        throttle = (
                            int(gpu_df["throttle_reasons"].max())
                            if not gpu_df.empty and "throttle_reasons" in gpu_df.columns
                            else None
                        )
                        summary = summarize_run(
                            pd.DataFrame([r.model_dump() for r in records])
                            if records
                            else pd.DataFrame(),
                            server_df,
                            gpu_df if not gpu_df.empty else None,
                            (res.window_start_ns, res.window_end_ns),
                            capacity,
                            man.gates.min_tail_samples,
                        )
                        _ = summary
                        checks = _run_checks(
                            records=records,
                            summary=summary,
                            metrics_present=metrics_present,
                            essential=metrics_map.essential,
                            loop_lag_p99_ms=res.loop_lag_p99_ms,
                            client_cpu_max=res.client_cpu_max,
                            throttle_reasons=throttle,
                            idle_ok=idle_ok,
                            idle_detail=idle_detail,
                            cfg=man.client,
                            checks_cfg=man.checks,
                            prompt_tokens=man.workload.prompt_tokens,
                            output_tokens=man.workload.output_tokens,
                        )
                    invalid = [k for k, v in checks.items() if v.invalidating and not v.passed]
                    meta.valid = not invalid
                    meta.invalid_reasons = [f"{k}: {checks[k].reason}" for k in invalid]
                    if startup_only:
                        meta.window_start_ns = 0
                        meta.window_end_ns = 0
                    else:
                        meta.window_start_ns = res.window_start_ns
                        meta.window_end_ns = res.window_end_ns
                    meta.counts = {
                        "requests": len(records),
                        "ok_in_window": len(
                            [r for r in records if r.in_window and r.status == "ok"]
                        ),
                    }
                    meta.checks = checks
                    meta.ended_utc = _utc_now()
                    _write_run_dir(
                        out_dir,
                        meta,
                        records,
                        chunks,
                        server_df,
                        gpu_df,
                        m_start,
                        m_end,
                        checks,
                    )
                    produced.append(out_dir)
            finally:
                server.stop()
    return produced


def _run_load_point(
    *,
    server: VLLMServer,
    metrics_map: MetricsMap,
    man: ExperimentManifest,
    model_name: str,
    safe_ids: Any,
    n: int,
    run_id: str,
    rep: int,
) -> tuple[Any, str, str, pd.DataFrame, pd.DataFrame]:
    """Execute one (config, rep, N) load point with scrapers; no loop closures."""
    stop_ev = asyncio.Event()
    scraper = PrometheusScraper(
        f"{server.base_url}/metrics", man.telemetry.metrics_interval_s, metrics_map
    )
    nvml = NvmlSampler(interval_s=man.telemetry.nvml_interval_s)
    gpu_stop = threading.Event()
    gpu_rows: list[pd.DataFrame] = []
    gt = threading.Thread(target=lambda: gpu_rows.append(nvml.run_in_thread(gpu_stop)), daemon=True)

    async def _drive() -> tuple[Any, str, str, pd.DataFrame]:
        task = asyncio.create_task(scraper.run(stop_ev))
        try:
            m_s = await scraper.snapshot_raw()
        except Exception as e:
            m_s = f"snapshot failed: {e}"
        stream = request_stream(man.workload, safe_ids, start=rep * 100_000 + n * 997)
        res = await run_closed_loop(stream, server.base_url, model_name, n, man.client, run_id)
        try:
            m_e = await scraper.snapshot_raw()
        except Exception as e:
            m_e = f"snapshot failed: {e}"
        stop_ev.set()
        sdf = await task
        return res, m_s, m_e, sdf

    gt.start()
    try:
        res, m_start, m_end, server_df = asyncio.run(_drive())
    finally:
        gpu_stop.set()
        gt.join(timeout=10)
        gpu_df = gpu_rows[0] if gpu_rows else pd.DataFrame()
    return res, m_start, m_end, server_df, gpu_df


def _warmup(
    server: VLLMServer, model_ref: ModelRef, man: ExperimentManifest, safe_ids: Any
) -> None:
    async def _go() -> None:
        stream = request_stream(man.workload, safe_ids, start=999_000_000)
        cfg = ClientConfig(
            warmup_s=0.0,
            stagger_s=0.0,
            window_min_s=0.1,
            window_max_s=5.0,
            min_completed=8,
            request_timeout_s=man.client.request_timeout_s,
            drain_timeout_s=man.client.drain_timeout_s,
        )
        model_name = model_ref.hf_repo if server.engine.name != "mock" else "mock-model"
        await run_closed_loop(stream, server.base_url, model_name, 4, cfg, "warmup")

    try:
        asyncio.run(_go())
    except Exception as e:
        logger.warning("warmup failed (continuing): %s", e)
