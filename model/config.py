"""
model/config.py

Loads and validates the model architecture config from configs/model.yaml
into a typed dataclass. Every other model/ file imports ModelConfig from
here rather than reading YAML directly.

Usage:
    from model.config import ModelConfig

    cfg = ModelConfig.from_yaml("configs/model.yaml")
    print(cfg.n_layers, cfg.d_model)
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch
import yaml

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclasses.dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_heads_q: int
    n_heads_kv: int
    head_dim: int
    d_ffn: int
    vocab_size: int
    context_length: int
    dropout: float
    dtype: torch.dtype

    tokenizer_path: str
    pad_token: str
    eof_token: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load and validate config from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        dtype_str = raw.pop("dtype")
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(
                f"unknown dtype '{dtype_str}' in {path} — expected one of {list(_DTYPE_MAP)}"
            )

        cfg = cls(dtype=_DTYPE_MAP[dtype_str], **raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Cross-check derived relationships between fields."""
        if self.n_heads_q * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads_q ({self.n_heads_q}) * head_dim ({self.head_dim}) "
                f"= {self.n_heads_q * self.head_dim}, expected d_model ({self.d_model})"
            )

        if self.n_heads_q % self.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({self.n_heads_q}) must be an exact multiple of "
                f"n_heads_kv ({self.n_heads_kv}) for GQA grouping"
            )

        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

        if not Path(self.tokenizer_path).exists():
            logging.getLogger(__name__).warning(
                "tokenizer_path '%s' does not exist relative to cwd — "
                "make sure to run from the repo root or pass an absolute path",
                self.tokenizer_path,
            )

    @property
    def gqa_group_size(self) -> int:
        """Number of query heads sharing each KV head."""
        return self.n_heads_q // self.n_heads_kv

    @property
    def kv_dim(self) -> int:
        """Total dimension of the K or V projection."""
        return self.n_heads_kv * self.head_dim
