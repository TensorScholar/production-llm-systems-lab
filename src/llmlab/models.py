"""Architecture facts: Hugging Face config.json -> typed ModelSpec.

Refuses architectures v0.1 doesn't model rather than guessing.
Book Ch. 3–4 model facts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llmlab.schemas import ModelRef

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    tie_word_embeddings: bool
    attention: Literal["mha", "gqa", "mqa"]
    param_count: int | None = None
    weight_bytes: int | None = None


class UnsupportedArchitecture(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def spec_from_hf_config(config: Mapping[str, Any], *, name: str) -> ModelSpec:
    """Build a ModelSpec from a Hugging Face config.json mapping.

    Formula: attention kind from head counts; head_dim defaults to
    hidden_size // n_q_heads. Book §Ch.3 model facts.
    """
    cfg = dict(config)
    # Unwrap nested text_config (v0.1 refuses to guess).
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        raise UnsupportedArchitecture("nested text_config: unwrap explicitly (unsupported in v0.1)")
    model_type = str(cfg.get("model_type", "")).lower()
    if model_type in {"mamba", "mamba2", "rwkv", "jamba", "mixture", "hybrid"} or (
        "ssm" in model_type or "hybrid" in model_type
    ):
        raise UnsupportedArchitecture(f"SSM/hybrid model_type unsupported: {model_type!r}")
    if "kv_lora_rank" in cfg:
        raise UnsupportedArchitecture("MLA (kv_lora_rank) unsupported in v0.1")
    # Sliding window: reject only ACTIVE sliding windows.
    use_sliding = cfg.get("use_sliding_window", None)
    layer_types = cfg.get("layer_types", None)
    sliding_window = cfg.get("sliding_window", None)
    if use_sliding is True:
        raise UnsupportedArchitecture("active sliding window (use_sliding_window=true) unsupported")
    if isinstance(layer_types, list) and any(
        str(t).lower() in {"sliding_attention", "sliding", "linear"} for t in layer_types
    ):
        raise UnsupportedArchitecture(f"sliding/linear layer_types unsupported: {layer_types!r}")
    # Falcon multi_query flag style.
    if cfg.get("multi_query", False) is True:
        # Falcon uses n_head_kv=1 style; require standard keys instead.
        if "num_key_value_heads" not in cfg and "n_head_kv" not in cfg:
            raise UnsupportedArchitecture("Falcon multi_query without standard KV-head keys")
    if "num_key_value_heads" not in cfg and "n_head_kv" in cfg:
        cfg["num_key_value_heads"] = cfg["n_head_kv"]
    # Required standard keys.
    try:
        n_layers: int = int(cfg["num_hidden_layers"])
        n_q: int = int(cfg["num_attention_heads"])
        hidden: int = int(cfg["hidden_size"])
        vocab: int = int(cfg["vocab_size"])
    except KeyError as e:
        raise UnsupportedArchitecture(f"non-standard key set, missing {e}") from e
    _ = sliding_window  # documented: sliding_window with use_sliding_window=false is full attention
    n_kv = int(cfg.get("num_key_value_heads", n_q))
    head_dim = int(cfg.get("head_dim", hidden // n_q))
    tie = bool(cfg.get("tie_word_embeddings", False))
    if n_kv == n_q:
        attn: Literal["mha", "gqa", "mqa"] = "mha"
    elif n_kv == 1:
        attn = "mqa"
    else:
        attn = "gqa"
    return ModelSpec(
        name=name,
        n_layers=n_layers,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=vocab,
        tie_word_embeddings=tie,
        attention=attn,
    )


def load_model_spec(ref: ModelRef, cache_dir: Path) -> ModelSpec:
    """Load a ModelSpec for a pinned ModelRef (network: pinned revision).

    Reads config.json via huggingface_hub at the pinned revision, then
    checkpoint size via safetensors metadata. Raises on "PIN-IN-T13".
    """
    if ref.revision == "PIN-IN-T13":
        raise UnsupportedArchitecture("model revision not pinned (T13 required)")
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    cfg_path = hf_hub_download(
        repo_id=ref.hf_repo, filename="config.json", revision=ref.revision, cache_dir=str(cache_dir)
    )
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    spec = spec_from_hf_config(cfg, name=ref.id)
    param_count: int | None = None
    weight_bytes: int | None = None
    try:
        meta = get_safetensors_metadata(repo_id=ref.hf_repo, revision=ref.revision)
        # Sum shard sizes when available.
        total = 0
        for shard in getattr(meta, "shards", {}).values() if hasattr(meta, "shards") else []:
            total += int(getattr(shard, "size", 0) or 0)
        param_count = getattr(meta, "parameter_count", None)
        if total:
            weight_bytes = total
    except Exception as e:
        logger.warning("safetensors metadata failed (%s); trying index fallback", e)
        try:
            idx_path = hf_hub_download(
                repo_id=ref.hf_repo,
                filename="model.safetensors.index.json",
                revision=ref.revision,
                cache_dir=str(cache_dir),
            )
            idx = json.loads(Path(idx_path).read_text(encoding="utf-8"))
            total_size = idx.get("metadata", {}).get("total_size")
            if total_size:
                weight_bytes = int(total_size)
        except Exception as e2:
            logger.warning("index fallback failed: %s", e2)
    return ModelSpec(
        name=spec.name,
        n_layers=spec.n_layers,
        n_q_heads=spec.n_q_heads,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        hidden_size=spec.hidden_size,
        vocab_size=spec.vocab_size,
        tie_word_embeddings=spec.tie_word_embeddings,
        attention=spec.attention,
        param_count=param_count,
        weight_bytes=weight_bytes,
    )
