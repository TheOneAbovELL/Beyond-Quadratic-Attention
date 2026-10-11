"""Implementation A — naive exact scaled dot-product attention (the reference).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

This is the correctness baseline for every other algorithm. It is written
for clarity, not speed, and materializes the full score tensor.

Shapes (B batch, H heads, Lq queries, Lk keys, d head dim, dv value dim):

    q            [B, H, Lq, d]
    k            [B, H, Lk, d]
    s = q k^T    [B, H, Lq, Lk]   <- the O(B*H*Lq*Lk) tensor
    m = max_j s  [B, H, Lq, 1]
    e = exp(s-m) [B, H, Lq, Lk]   <- a second O(B*H*Lq*Lk) tensor
    l = sum_j e  [B, H, Lq, 1]
    p = e / l    [B, H, Lq, Lk]   <- a third (autograd keeps p for backward)
    out = p v    [B, H, Lq, dv]

With L = Lq = Lk the score, exponent and probability tensors each hold
B*H*L^2 elements. That quadratic term — not the O(B*H*L*d) inputs — dominates
memory once L >> d. See docs/complexity_analysis.md.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .base import MaskSpec, Tensor, apply_tile_mask, default_scale, safe_row_max, validate_qkv


def naive_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Reference attention. See module docstring for semantics.

    If ``return_lse`` is True, also returns the row log-sum-exp of the masked,
    scaled logits, shape [B, H, Lq] (``-inf`` for fully-masked rows). This is
    the statistic the tiled implementations store for their backward pass.
    """
    shp = validate_qkv(q, k, v)
    sc = default_scale(shp.d, scale)
    spec = MaskSpec(shp, mask=mask, key_padding_mask=key_padding_mask, causal=causal, device=q.device)

    s = torch.matmul(q, k.transpose(-2, -1)) * sc            # [B,H,Lq,Lk]
    allowed, additive = spec.tile(0, shp.Lq, 0, shp.Lk)
    s = apply_tile_mask(s, allowed, additive)

    m = safe_row_max(s)                                       # [B,H,Lq,1]
    e = torch.exp(s - m)                                      # masked -> exp(-inf) = 0
    l = e.sum(dim=-1, keepdim=True)                           # [B,H,Lq,1]
    l_safe = torch.where(l == 0, torch.ones_like(l), l)       # fully-masked rows -> p = 0
    p = e / l_safe
    out = torch.matmul(p, v)                                  # [B,H,Lq,dv]

    if return_lse:
        lse = (m + torch.log(l)).squeeze(-1)                  # log(0) = -inf for empty rows
        return out, lse
    return out


def attention_probabilities(
    q: Tensor,
    k: Tensor,
    *,
    mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """The full [B, H, Lq, Lk] probability matrix (for analysis and teaching only)."""
    v_dummy = torch.zeros(q.shape[0], q.shape[1], k.shape[2], 1, dtype=q.dtype, device=q.device)
    shp = validate_qkv(q, k, v_dummy)
    sc = default_scale(shp.d, scale)
    spec = MaskSpec(shp, mask=mask, key_padding_mask=key_padding_mask, causal=causal, device=q.device)
    s = torch.matmul(q, k.transpose(-2, -1)) * sc
    allowed, additive = spec.tile(0, shp.Lq, 0, shp.Lk)
    s = apply_tile_mask(s, allowed, additive)
    m = safe_row_max(s)
    e = torch.exp(s - m)
    l = e.sum(dim=-1, keepdim=True)
    return e / torch.where(l == 0, torch.ones_like(l), l)
