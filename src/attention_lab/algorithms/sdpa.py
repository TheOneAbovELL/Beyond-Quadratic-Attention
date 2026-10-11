"""Implementation C (optimized framework kernels) — adapters for PyTorch SDPA.

``torch.nn.functional.scaled_dot_product_attention`` dispatches to one of
several ATen kernels. In this project's CPU environment (torch 2.10 CPU
build) we verified with the profiler that:

* ``SDPBackend.FLASH_ATTENTION`` on CPU runs
  ``aten::_scaled_dot_product_flash_attention_for_cpu`` (+ its ``_backward``):
  a fused, tiled, online-softmax C++ kernel written by the PyTorch team for
  CPUs. It is **not** the official FlashAttention CUDA library by Dao et al.;
  it applies the same algorithmic ideas (tiling, online softmax, storing
  logsumexp, recomputation in backward) with vectorized C++ and OpenMP.
* ``SDPBackend.MATH`` runs ``aten::_scaled_dot_product_attention_math``: a
  composite of ordinary ATen ops that materializes the full score matrix,
  i.e. an optimized-library version of Implementation A.

Both kernels return zeros for fully-masked rows, matching the project policy
(see ``base.py``). We force the backend explicitly so a benchmark row
always states which kernel ran.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .base import MaskSpec, Tensor, default_scale, validate_qkv

_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
}


def _sdpa_mask(spec: MaskSpec) -> tuple[Optional[Tensor], bool]:
    """Translate a MaskSpec into SDPA's (attn_mask, is_causal) arguments.

    Pure causal masking uses ``is_causal=True`` (no mask tensor at all).
    Otherwise the combined mask is built at its broadcast shape: a key
    padding mask alone stays [B,1,1,Lk]; anything involving causality or a
    dense user mask becomes [.., Lq, Lk] (O(L^2) bool/float storage —
    a documented cost of this adapter, not of the kernel).
    """
    if spec.is_empty:
        return None, False
    if spec.causal and not spec.has_dense_mask and spec.kpm is None:
        return None, True
    allowed, additive = spec.tile(0, spec.shp.Lq, 0, spec.shp.Lk)
    if additive is None:
        return allowed, False
    if allowed is None:
        return additive, False
    return additive.masked_fill(~allowed, float("-inf")), False


def sdpa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    backend: str = "flash",
) -> Tensor:
    shp = validate_qkv(q, k, v)
    sc = default_scale(shp.d, scale)
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}")
    spec = MaskSpec(shp, mask=mask, key_padding_mask=key_padding_mask, causal=causal, device=q.device)
    attn_mask, is_causal = _sdpa_mask(spec)
    if attn_mask is not None and attn_mask.is_floating_point():
        attn_mask = attn_mask.to(q.dtype)
    with sdpa_kernel(_BACKENDS[backend]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=sc)


def sdpa_flash_cpu(q, k, v, **kw) -> Tensor:
    return sdpa_attention(q, k, v, backend="flash", **kw)


def sdpa_math(q, k, v, **kw) -> Tensor:
    return sdpa_attention(q, k, v, backend="math", **kw)
