"""Minimal decoder-only Transformer whose attention backend is a registry name.

Architecture (pre-LayerNorm GPT-style):

    tokens -> Embedding -> [ x + Attn(LN(x)) ; x + MLP(LN(x)) ] * n_layers -> LN -> Linear -> logits

* Positional information: rotary position embedding (RoPE, Su et al. 2021)
  applied to q and k inside attention (default), sinusoidal absolute
  embeddings added to token embeddings, or none. RoPE is applied *before*
  the attention backend, so every backend sees identical q/k/v.
* The backend only receives [B, H, L, d_head] tensors and a causal flag,
  so swapping it changes nothing else in the model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithms.registry import get_method


@dataclass
class ModelConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 256
    max_len: int = 4096
    positional: str = "rope"           # "rope" | "sinusoidal" | "none"
    attention: str = "sdpa_flash_cpu"
    attention_params: Dict[str, Any] = field(default_factory=dict)
    causal: bool = True
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rope_cache(L: int, d_head: int, device=None, dtype=torch.float32, base: float = 10000.0):
    if d_head % 2:
        raise ValueError("RoPE needs an even head dimension")
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    pos = torch.arange(L, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv)                                   # [L, d/2]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) by angle pos * theta_i. x: [B, H, L, d]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


def sinusoidal_table(L: int, d: int) -> torch.Tensor:
    pos = torch.arange(L, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(L, d)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : d // 2])
    return pe


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.set_backend(cfg.attention, cfg.attention_params)

    def set_backend(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.backend = get_method(name)
        self.backend_name = name
        self.backend_params = dict(params or {})

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # each [B,H,L,dh]
        if self.cfg.positional == "rope":
            cos, sin = rope_cache(L, self.dh, x.device, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = self.backend(q, k, v, causal=self.cfg.causal, key_padding_mask=key_padding_mask, **self.backend_params)
        return self.proj(o.transpose(1, 2).reshape(B, L, self.cfg.d_model))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask))
        return x + self.drop(self.mlp(self.ln2(x)))


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.positional == "sinusoidal":
            self.register_buffer("pe", sinusoidal_table(cfg.max_len, cfg.d_model), persistent=False)
        elif cfg.positional not in ("rope", "none"):
            raise ValueError(f"unknown positional mode {cfg.positional!r}")
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_attention(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Swap the attention backend of every layer without touching any weights."""
        for b in self.blocks:
            b.attn.set_backend(name, params)
        self.cfg.attention = name
        self.cfg.attention_params = dict(params or {})

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = tokens.shape
        if L > self.cfg.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len {self.cfg.max_len}")
        x = self.embed(tokens)
        if self.cfg.positional == "sinusoidal":
            x = x + self.pe[:L]
        for b in self.blocks:
            x = b(x, key_padding_mask)
        return self.head(self.ln_f(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over positions whose target is not -100."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
