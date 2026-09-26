"""Error paths and remaining branches: kv, roofline, models, telemetry, engines."""

from __future__ import annotations

import pytest

from llmlab.calc import kv, roofline
from llmlab.models import ModelSpec, UnsupportedArchitecture, load_model_spec
from llmlab.schemas import ModelRef

SPEC = ModelSpec(
    name="t",
    n_layers=4,
    n_q_heads=8,
    n_kv_heads=2,
    head_dim=64,
    hidden_size=512,
    vocab_size=1000,
    tie_word_embeddings=False,
    attention="gqa",
    param_count=1_000_000,
)


def test_kv_error_paths() -> None:
    with pytest.raises(ValueError):
        kv.pool_capacity(-1, 10, 16)
    with pytest.raises(ValueError):
        kv.pool_bytes(-1, 16, 10)
    with pytest.raises(ValueError):
        kv.max_concurrency_blocks(10, 0, 16)
    with pytest.raises(ValueError):
        kv.g_decode(10, 0, 16)
    with pytest.raises(ValueError):
        kv.g_max(10, 10, 0)
    with pytest.raises(ValueError):
        kv.g_semi_empirical(10, 10, 16, -1.0, 0.01)
    with pytest.raises(ValueError):
        kv.g_semi_empirical(10, 1, 16, 0.0, 0.01)
    with pytest.raises(ValueError):
        kv.predicted_occupancy(-1, 1.0, 10)
    with pytest.raises(ValueError):
        kv.predicted_saturation_interval(0, 1.0, 2)
    assert kv.blocks_for_tokens(0, 16) == 0
    assert kv.g_decode(0, 1, 16) == 0.0
    assert kv.g_decode(16, 1, 16) == 16.0


def test_kv_blocks_ratio() -> None:
    assert kv.max_concurrency_blocks(100, 48, 16) == 100 // 3
    assert kv.max_concurrency_ratio(100, 48) == pytest.approx(100 / 48)


def test_g_semi_empirical_value() -> None:
    g = kv.g_semi_empirical(2048, 1024, 16, 0.5, 0.02)
    assert 1000 < g < 3072


def test_roofline_error_paths() -> None:
    no_params = ModelSpec(
        name="n",
        n_layers=1,
        n_q_heads=1,
        n_kv_heads=1,
        head_dim=8,
        hidden_size=8,
        vocab_size=8,
        tie_word_embeddings=True,
        attention="mha",
    )
    with pytest.raises(ValueError):
        roofline.decode_weight_bytes(no_params, 2)
    with pytest.raises(ValueError):
        roofline.decode_weight_bytes(SPEC, 0)
    with pytest.raises(ValueError):
        roofline.tpot_floor_s(-1, 0.0, 1.0)
    with pytest.raises(ValueError):
        roofline.ttft_floor_s(-1, 1, 1.0)
    with pytest.raises(ValueError):
        roofline.calibrate(0.0, 0.1, 1, 1.0, 8)
    with pytest.raises(ValueError):
        roofline.calibrate(0.1, 0.05, 1, 1.0, 8)
    with pytest.raises(ValueError):
        roofline.predict_closed_loop(0, roofline.Calibration(1.0, 0.1), 1, 1.0, 8, 8, 8, 64)
    with pytest.raises(ValueError):
        roofline.predict_closed_loop(
            100, roofline.Calibration(1.0, 0.1), 1, 1.0, 8, 8, 8, 4
        )  # budget too small


def test_load_model_spec_unpinned() -> None:
    ref = ModelRef(id="x", hf_repo="Qwen/Qwen2.5-7B-Instruct", revision="PIN-IN-T13")
    with pytest.raises(UnsupportedArchitecture, match="not pinned"):
        load_model_spec(ref, __import__("pathlib").Path(".cache"))


def test_load_model_spec_mocked_network(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json
    import sys
    import types

    cfg = {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "tie_word_embeddings": False,
        "model_type": "llama",
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kw: str(cfg_path)  # type: ignore[attr-defined]
    meta = types.SimpleNamespace(parameter_count=1_000_000, shards={})
    fake_hub.get_safetensors_metadata = lambda **kw: meta  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    ref = ModelRef(id="x", hf_repo="Q/X", revision="b" * 40)
    spec = load_model_spec(ref, tmp_path)
    assert spec.n_layers == 4 and spec.param_count == 1_000_000


def test_load_model_spec_index_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json
    import sys
    import types

    cfg = {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "vocab_size": 1000,
        "tie_word_embeddings": True,
        "model_type": "llama",
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    idx_path = tmp_path / "index.json"
    idx_path.write_text(json.dumps({"metadata": {"total_size": 2000000}}), encoding="utf-8")

    def _dl(**kw: object) -> str:  # type: ignore[no-untyped-def]
        assert isinstance(kw.get("filename"), str)
        if "index" in kw["filename"]:
            return str(idx_path)
        return str(cfg_path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = _dl  # type: ignore[attr-defined]

    def _boom(**kw: object) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("no metadata")

    fake_hub.get_safetensors_metadata = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    ref = ModelRef(id="x", hf_repo="Q/X", revision="b" * 40)
    spec = load_model_spec(ref, tmp_path)
    assert spec.attention == "mha"
    assert spec.weight_bytes == 2000000
