"""Remaining branches: report figures, telemetry scraper, vllm errors, runner refusals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from llmlab.engines.vllm import parse_startup_log


def test_make_figures_smoke(tmp_path: Path) -> None:
    from llmlab import report as r

    d = tmp_path / "derived"
    d.mkdir()
    pd.DataFrame(
        [{"config": "c", "reported_pool_tokens": 100, "predicted_pool_tokens": 101}]
    ).to_csv(d / "reconciliation.csv", index=False)
    pd.DataFrame([{"config": "c", "N": 1, "observed": 5.0, "predicted": 5.0}]).to_csv(
        d / "occupancy.csv", index=False
    )
    pd.DataFrame([{"config": "c", "N": 1, "tpot_obs": 0.02, "tpot_pred": 0.02}]).to_csv(
        d / "serving_model.csv", index=False
    )
    pd.DataFrame([{"t_s": 0.0, "kv_usage": 0.5, "running": 1.0, "waiting": 0.0}]).to_csv(
        d / "timeseries.csv", index=False
    )
    figs = r.make_figures(d, tmp_path / "figs", {"evidence": "MOCK"})
    assert len(figs) == 6
    assert all(p.suffix == ".svg" for p in figs)
    assert (tmp_path / "figs" / "a4_occupancy.svg.provenance.json").exists()


def test_telemetry_scraper_and_snapshot() -> None:
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from llmlab.schemas import MetricsMap
    from llmlab.telemetry import PrometheusScraper
    from llmlab.testing.mock_server import MockConfig, create_app

    async def _go() -> None:
        client = TestClient(TestServer(create_app(MockConfig(capacity_tokens=256))))
        await client.start_server()
        try:
            base = str(client.make_url("/"))[:-1]
            mmap = MetricsMap(
                kv_usage=["vllm:kv_cache_usage_perc"],
                running=["vllm:num_requests_running"],
                waiting=["vllm:num_requests_waiting"],
                preemptions=[],
                prompt_tokens=[],
                generation_tokens=[],
                cache_config_info=[],
                essential=["kv_usage", "running", "waiting"],
            )
            scraper = PrometheusScraper(f"{base}/metrics", 0.05, mmap)
            raw = await scraper.snapshot_raw()
            assert "vllm:kv_cache_usage_perc" in raw
            stop = asyncio.Event()
            task = asyncio.create_task(scraper.run(stop))
            await asyncio.sleep(0.2)
            stop.set()
            df = await task
            assert set(df.columns) == {"t_ns", "name", "value"}
            assert not df.empty
        finally:
            await client.close()

    asyncio.run(_go())


def test_vllm_parse_mismatch_and_version_fallback(tmp_path: Path) -> None:
    server_log = tmp_path / "engine.log"
    server_log.write_text(
        "GPU KV cache size: 1,000 tokens\n"
        "Maximum concurrency for 48 tokens per request: 20.83x\n"
        "Model loading took 0.50 GB\n"
        "Num blocks: 62\nBlock size: 16\n",
        encoding="utf-8",
    )
    rep = parse_startup_log(server_log.read_text(encoding="utf-8"), "vllm-startup-v1")
    assert rep.kv_cache_tokens == 1000
    assert rep.num_gpu_blocks == 62
    assert rep.block_size == 16
    assert rep.model_load_bytes == pytest.approx(0.50 * 1e9)


def test_runner_refusals(tmp_path: Path) -> None:
    from llmlab.runner import ManifestError, run_experiment

    repo = Path(__file__).parent.parent.parent
    man = {
        "experiment_id": "B0",
        "title": "t",
        "model": str(repo / "configs/models/qwen2.5-7b-instruct.yaml"),
        "hardware": str(repo / "configs/hardware/a10g-24gb.yaml"),
        "engine": str(repo / "configs/engines/mock.yaml"),
        "workload": {"kind": "fixed_shape", "prompt_tokens": 8, "output_tokens": 4, "seed": 7},
        "configs": [{"name": "m", "engine_args": {}, "concurrency": [1]}],
        "repetitions": 1,
        "order": {"randomize": False, "seed": 1},
        "client": {
            "warmup_s": 0.0,
            "stagger_s": 0.0,
            "window_min_s": 1.0,
            "window_max_s": 3.0,
            "min_completed": 1,
            "request_timeout_s": 20.0,
            "drain_timeout_s": 20.0,
        },
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(man), encoding="utf-8")
    # Dirty tree refused without the flag (this checkout has untracked test work
    # unless committed; either outcome is asserted via allow_dirty path below).
    try:
        run_experiment(p, tmp_path / "build", allow_dirty=False)
    except ManifestError as e:
        assert "dirty" in str(e)
    produced = run_experiment(p, tmp_path / "build", allow_dirty=True)
    assert len(produced) == 1

    # Unpinned placeholders refused for non-mock engines.
    man2 = dict(man)
    man2["engine"] = str(repo / "configs/engines/vllm.yaml")
    p2 = tmp_path / "m2.yaml"
    p2.write_text(yaml.safe_dump(man2), encoding="utf-8")
    with pytest.raises(ManifestError, match=r"unpinned|PIN"):
        run_experiment(p2, tmp_path / "build2", allow_dirty=True)


def test_startup_only_mode(tmp_path: Path) -> None:
    from llmlab.runner import run_experiment
    from llmlab.schemas import RunMetadata

    repo = Path(__file__).parent.parent.parent
    man = {
        "experiment_id": "A3",
        "title": "t",
        "model": str(repo / "configs/models/qwen2.5-7b-instruct.yaml"),
        "hardware": str(repo / "configs/hardware/a10g-24gb.yaml"),
        "engine": str(repo / "configs/engines/mock.yaml"),
        "workload": {"kind": "fixed_shape", "prompt_tokens": 8, "output_tokens": 4, "seed": 7},
        "configs": [{"name": "pool-full", "engine_args": {}, "concurrency": []}],
        "repetitions": 1,
        "order": {"randomize": False, "seed": 1},
        "client": {
            "warmup_s": 0.0,
            "stagger_s": 0.0,
            "window_min_s": 1.0,
            "window_max_s": 3.0,
            "min_completed": 1,
            "request_timeout_s": 20.0,
            "drain_timeout_s": 20.0,
        },
    }
    p = tmp_path / "a3.yaml"
    p.write_text(yaml.safe_dump(man), encoding="utf-8")
    produced = run_experiment(p, tmp_path / "build", allow_dirty=True)
    assert len(produced) == 1
    meta = RunMetadata(**yaml.safe_load((produced[0] / "run.json").read_text(encoding="utf-8")))
    assert meta.experiment_id == "A3"
