"""Deterministic fixed-shape requests (book workload method).

Each prompt is drawn from np.random.default_rng([seed, index]) so request i
is reproducible without storing files. Special/added tokens are excluded.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from llmlab.schemas import WorkloadSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int


def safe_token_ids(tokenizer_json: Mapping[str, Any], vocab_size: int) -> npt.NDArray[np.int64]:
    """IDs safe for prompt generation: [0, vocab) minus added/special tokens.

    Formula: safe = all_ids − added_ids − {ids ≥ vocab}. Workload method.
    """
    added = set()
    for tok in tokenizer_json.get("added_tokens", []):
        if isinstance(tok, dict) and "id" in tok:
            added.add(int(tok["id"]))
    # Some tokenizer.json files mark special tokens via "special" models; also
    # exclude ids listed under "added_tokens_decoder".
    decoder = tokenizer_json.get("added_tokens_decoder", {})
    if isinstance(decoder, dict):
        for k in decoder:
            try:
                added.add(int(k))
            except ValueError:
                continue
    ids = [i for i in range(vocab_size) if i not in added]
    if not ids:
        raise ValueError("no safe token ids available")
    return np.array(ids, dtype=np.int64)


def fixed_shape_request(spec: WorkloadSpec, safe_ids: npt.NDArray[np.int64], index: int) -> Request:
    """Build request `index`: P token IDs sampled deterministically.

    Formula: rng = default_rng([seed, index]); ids = choice(safe, P).
    """
    rng = np.random.default_rng([spec.seed, index])
    chosen = rng.choice(safe_ids, size=spec.prompt_tokens, replace=True)
    return Request(
        request_id=f"req-{index:08d}",
        prompt_token_ids=tuple(int(v) for v in chosen),
        max_tokens=spec.output_tokens,
    )


def request_stream(
    spec: WorkloadSpec, safe_ids: npt.NDArray[np.int64], start: int = 0
) -> Iterator[Request]:
    """Infinite deterministic stream of requests from `start`."""
    index = start
    while True:
        yield fixed_shape_request(spec, safe_ids, index)
        index += 1


def safe_ids_digest(safe_ids: npt.NDArray[np.int64]) -> str:
    """Stable SHA-256 of the safe-ID vocabulary (provenance)."""
    return hashlib.sha256(np.ascontiguousarray(safe_ids).tobytes()).hexdigest()
