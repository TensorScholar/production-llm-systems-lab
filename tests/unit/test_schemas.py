"""Unit tests for llmlab.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmlab.schemas import (
    ClientConfig,
    ExperimentManifest,
    ModelRef,
    RequestRecord,
    RunMetadata,
    StartupReport,
    load_yaml,
)


def test_model_ref_round_trip() -> None:
    m = ModelRef(
        id="qwen2.5-7b-instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        revision="a" * 40,
    )
    assert ModelRef(**m.model_dump()) == m


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelRef(
            id="x",
            hf_repo="y",
            revision="z",
            bogus_key="nope",  # type: ignore[call-arg]
        )


def test_startup_report_nullable() -> None:
    r = StartupReport(parser_version="vllm-startup-v1")
    assert r.kv_cache_tokens is None


def test_request_record_statuses() -> None:
    for status in ("ok", "error", "timeout"):
        rec = RequestRecord(
            run_id="r",
            request_id="q",
            t_send_ns=0,
            status=status,  # type: ignore[arg-type]
        )
        assert rec.status == status


def test_run_metadata_defaults() -> None:
    meta = RunMetadata(run_id="A4-x", experiment_id="A4")
    assert meta.valid is True
    assert meta.evidence == "MEASURED"


def test_load_yaml_and_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "m.yaml"
    p.write_text(
        "experiment_id: A4\nmodel: configs/models/q.yaml\n"
        "hardware: configs/hardware/h.yaml\nengine: configs/engines/v.yaml\n"
    )
    man = load_yaml(p, ExperimentManifest)
    assert isinstance(man, ExperimentManifest)
    resolved = man.resolve(tmp_path)
    assert "q.yaml" in resolved.model


def test_client_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ClientConfig(warmup_s=1.0, nope=2)  # type: ignore[call-arg]
