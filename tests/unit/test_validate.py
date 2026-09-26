"""Validate checks: passing repo + failing fixtures per check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llmlab.schemas import Claim
from llmlab.validate import validate_all


def _minimal_results(tmp: Path) -> Path:
    (tmp / "runs").mkdir(parents=True, exist_ok=True)
    (tmp / "derived").mkdir(parents=True, exist_ok=True)
    (tmp / "engines").mkdir(parents=True, exist_ok=True)
    return tmp


def test_validate_clean_repo_ok(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    ok, messages = validate_all(Path("results"))
    assert ok, messages


def test_rejects_mock_run(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]

    res = _minimal_results(tmp_path)
    run_dir = res / "runs" / "B0" / "r1"
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": "r1",
        "experiment_id": "B0",
        "manifest_sha256": "x",
        "code": {"commit": "c", "dirty": False, "llmlab_version": "0.1.0.dev0"},
        "environment": {
            "python_version": "3.12",
            "platform": "p",
            "cpu": "c",
            "pip_freeze_sha256": "h",
        },
        "engine_name": "mock",
        "workload": {"kind": "fixed_shape", "prompt_tokens": 32, "output_tokens": 16, "seed": 7},
        "concurrency": 1,
        "evidence": "MOCK",
        "tier": 0,
    }
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    ok, messages = validate_all(res)
    assert not ok
    assert any("mock runs" in m for m in messages)


def test_bad_run_json_schema(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    res = _minimal_results(tmp_path)
    run_dir = res / "runs" / "A4" / "bad"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": 123}), encoding="utf-8")
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    ok, messages = validate_all(res)
    assert not ok
    assert any("schema" in m for m in messages)


def test_missing_engine_link(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    res = _minimal_results(tmp_path)
    run_dir = res / "runs" / "A4" / "r2"
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": "r2",
        "experiment_id": "A4",
        "manifest_sha256": "x",
        "code": {"commit": "c", "dirty": False, "llmlab_version": "0.1.0.dev0"},
        "environment": {
            "python_version": "3.12",
            "platform": "p",
            "cpu": "c",
            "pip_freeze_sha256": "h",
        },
        "engine_name": "vllm",
        "instance_id": "ghost-instance",
        "workload": {
            "kind": "fixed_shape",
            "prompt_tokens": 2048,
            "output_tokens": 1024,
            "seed": 1,
        },
        "concurrency": 8,
        "evidence": "MEASURED",
        "tier": 2,
    }
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    ok, messages = validate_all(res)
    assert not ok
    assert any("engine instance" in m for m in messages)


def test_provenance_mismatch_detected(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    res = _minimal_results(tmp_path)
    dd = res / "derived" / "A1"
    dd.mkdir(parents=True)
    (dd / "kv_table.csv").write_text("model\nq\n", encoding="utf-8")
    (dd / "kv_table.csv.provenance.json").write_text(
        json.dumps({"inputs": {"some/input.csv": "deadbeef"}}, indent=2), encoding="utf-8"
    )
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    ok, messages = validate_all(res)
    assert not ok
    assert any("missing" in m or "mismatch" in m for m in messages)


def test_claim_without_evaluator_detected(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from llmlab.analysis.kv_capacity import evaluate_claims

    claims = [Claim(id="ZZ-99", type="X", statement="s")]
    evals = evaluate_claims(claims, {})
    assert evals[0].status == "inconclusive"
    assert "no evaluator" in evals[0].reason


def test_a1_and_b0_evaluators(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from llmlab.analysis.kv_capacity import evaluate_claims

    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    a1 = pd.read_csv("results/derived/A1/kv_table.csv")
    b0 = pd.read_csv("results/derived/B0/harness_accuracy.csv")
    ledger = [
        Claim(id="A1-01", type="Identity", statement="s"),
        Claim(id="A1-02", type="Identity", statement="s"),
        Claim(id="B0-01", type="Harness", statement="s"),
    ]
    by_id = {
        e.claim_id: e.status
        for e in evaluate_claims(ledger, {"a1": a1, "b0": b0, "scalar": {"m_kv": 57344}})
    }
    assert by_id["A1-01"] == "supported"
    assert by_id["A1-02"] == "supported"
    assert by_id["B0-01"] == "inconclusive"
