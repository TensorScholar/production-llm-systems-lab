"""Unit tests for llmlab.workload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llmlab import workload
from llmlab.schemas import WorkloadSpec

SPEC = WorkloadSpec(kind="fixed_shape", prompt_tokens=32, output_tokens=16, seed=7)


def _safe_ids() -> np.ndarray:
    tok = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "tokenizer_added_tokens.json").read_text(
            encoding="utf-8"
        )
    )
    return workload.safe_token_ids(tok, 1000)


def test_deterministic() -> None:
    safe = _safe_ids()
    a = workload.fixed_shape_request(SPEC, safe, 3)
    b = workload.fixed_shape_request(SPEC, safe, 3)
    assert a == b
    c = workload.fixed_shape_request(SPEC, safe, 4)
    assert a.prompt_token_ids != c.prompt_token_ids


def test_length_and_no_special() -> None:
    safe = _safe_ids()
    req = workload.fixed_shape_request(SPEC, safe, 0)
    assert len(req.prompt_token_ids) == 32
    assert req.max_tokens == 16
    assert 151643 not in req.prompt_token_ids
    assert all(0 <= t < 1000 for t in req.prompt_token_ids)


def test_digest_stable() -> None:
    safe = _safe_ids()
    assert workload.safe_ids_digest(safe) == workload.safe_ids_digest(safe.copy())
    assert len(workload.safe_ids_digest(safe)) == 64


def test_stream_offsets() -> None:
    safe = _safe_ids()
    gen = workload.request_stream(SPEC, safe, start=10)
    first = next(gen)
    assert first.request_id == "req-00000010"
    assert first == workload.fixed_shape_request(SPEC, safe, 10)
