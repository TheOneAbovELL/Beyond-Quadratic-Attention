"""Single registry of attention implementations used by benchmarks, models, CLI and UI.

Every method is called as ``fn(q, k, v, causal=..., key_padding_mask=..., **params)``
with q/k/v shaped [B, H, L, d]. ``params`` are method-specific knobs
(tile sizes, window, block layout, ...); their defaults live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .block_sparse import block_sparse_attention
from .linear import linear_attention
from .local_window import local_window_attention
from .naive import naive_attention
from .sdpa import sdpa_flash_cpu, sdpa_math
from .tiled_exact import tiled_attention, tiled_attention_recompute


@dataclass(frozen=True)
class AttentionMethod:
    name: str
    fn: Callable
    family: str                      # "exact" | "restricted-softmax" | "different-kernel"
    computes: str                    # precise statement of the mathematical function
    implementation: str              # how it executes on this machine
    default_params: Dict[str, Any] = field(default_factory=dict)
    supports_dense_mask: bool = False
    supports_key_padding: bool = True
    supports_causal: bool = True
    supports_backward: bool = True
    dtypes: Tuple[torch.dtype, ...] = (torch.float64, torch.float32, torch.bfloat16)

    @property
    def exact(self) -> bool:
        return self.family == "exact"

    def __call__(self, q, k, v, **kwargs):
        params = {**self.default_params, **kwargs}
        return self.fn(q, k, v, **params)


_SOFTMAX = "softmax(Q K^T / sqrt(d) + M) V"

REGISTRY: Dict[str, AttentionMethod] = {
    m.name: m
    for m in [
        AttentionMethod(
            "naive", naive_attention, "exact", _SOFTMAX,
            "Reference: explicit matmul/exp/sum/div in PyTorch ops; materializes three B*H*L^2 tensors.",
            supports_dense_mask=True,
        ),
        AttentionMethod(
            "sdpa_math", sdpa_math, "exact", _SOFTMAX,
            "torch SDPA forced to SDPBackend.MATH (aten::_scaled_dot_product_attention_math); materializes scores.",
            supports_dense_mask=True,
        ),
        AttentionMethod(
            "sdpa_flash_cpu", sdpa_flash_cpu, "exact", _SOFTMAX,
            "torch SDPA forced to SDPBackend.FLASH_ATTENTION -> aten::_scaled_dot_product_flash_attention_for_cpu "
            "(PyTorch's fused tiled C++ CPU kernel; NOT the official FlashAttention CUDA library).",
            supports_dense_mask=True,
        ),
        AttentionMethod(
            "tiled", tiled_attention, "exact", _SOFTMAX,
            "Python-loop online-softmax tiling; autograd through the loop (saves every P tile for backward).",
            default_params={"q_tile": 128, "kv_tile": 128},
            supports_dense_mask=True,
        ),
        AttentionMethod(
            "tiled_recompute", tiled_attention_recompute, "exact", _SOFTMAX,
            "Python-loop online-softmax tiling; custom backward recomputes P tiles from (Q, K, lse).",
            default_params={"q_tile": 128, "kv_tile": 128},
            supports_dense_mask=True,
        ),
        AttentionMethod(
            "local_window", local_window_attention, "restricted-softmax",
            _SOFTMAX + " with M removing all pairs |i - j| > window (causal: j < i - window or j > i)",
            "Vectorized chunked band computation via unfold; O(L * (chunk + window)) scores.",
            default_params={"window": 64},
        ),
        AttentionMethod(
            "block_sparse", block_sparse_attention, "restricted-softmax",
            _SOFTMAX + " with M removing all pairs outside a local+global+random block layout",
            "Gathers active key blocks per query block; O(L * K_max * block) scores.",
            default_params={"block_size": 32, "local_blocks": 1, "global_blocks": 1, "random_blocks": 1, "seed": 0},
        ),
        AttentionMethod(
            "linear_elu", linear_attention, "different-kernel",
            "phi(Q) [phi(K)^T V] / (phi(Q) phi(K)^T 1),  phi = elu + 1   (NOT softmax attention)",
            "Associativity trick; causal uses chunked prefix-summed d x dv states.",
            default_params={"chunk": 64},
        ),
    ]
}

EXACT_METHODS = [n for n, m in REGISTRY.items() if m.exact]
APPROX_METHODS = [n for n, m in REGISTRY.items() if not m.exact]


def get_method(name: str) -> AttentionMethod:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown attention method {name!r}; available: {sorted(REGISTRY)}") from None


def run_attention(name: str, q, k, v, *, causal: bool = False, key_padding_mask: Optional[torch.Tensor] = None, **params):
    return get_method(name)(q, k, v, causal=causal, key_padding_mask=key_padding_mask, **params)
