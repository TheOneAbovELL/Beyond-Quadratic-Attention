"""Approximate method 3 — kernelized linear attention (Katharopoulos et al., 2020).

What changes (read this before comparing outputs)
-------------------------------------------------
Softmax attention weights key j for query i by  exp(q_i . k_j / sqrt(d)).
Linear attention *replaces* that similarity with a factorizable kernel

    sim(q, k) = phi(q) . phi(k),     phi(x) = elu(x) + 1  (elementwise, > 0)

    out_i = sum_j [phi(q_i) . phi(k_j)] v_j  /  sum_j phi(q_i) . phi(k_j)

This is a *different attention function*, not an approximation of softmax
with any error bound: elu(x)+1 is not a random-feature estimator of exp.
Comparing its output to softmax attention therefore measures "how different
is this model family", not "approximation error". Following the paper,
phi is applied to q and k directly (no 1/sqrt(d) scaling).

Why it can be linear in L
-------------------------
By associativity, sum_j (phi(q_i)^T phi(k_j)) v_j^T = phi(q_i)^T (sum_j phi(k_j) v_j^T).
The bracket is a d x dv matrix independent of i, so the non-causal cost is
O(L d dv) instead of O(L^2 d). Causal attention needs prefix sums
S_i = sum_{j<=i} phi(k_j) v_j^T; materializing every S_i costs O(L d dv)
*memory*, so we use a chunked form: quadratic inside chunks of C tokens,
recurrent (prefix-summed d x dv states) across chunks:

    time  O(L C d + L d dv),   extra memory O(L C + (L/C) d dv) per head.

Numerical stabilization: denominators are strictly positive because
phi > 0, but can be tiny; ``eps`` is added to the denominator. Padded or
masked keys get phi(k) = 0 so they contribute nothing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Tensor, validate_qkv


def elu_feature_map(x: Tensor) -> Tensor:
    return F.elu(x) + 1.0


def linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
    chunk: int = 64,
    eps: float = 1e-6,
    stats: Optional[dict] = None,
) -> Tensor:
    shp = validate_qkv(q, k, v)
    if mask is not None:
        raise NotImplementedError("linear_attention supports causal and key_padding_mask only")
    if scale is not None:
        raise ValueError("linear_attention does not use a softmax temperature; scale must be None")
    if causal and shp.Lq != shp.Lk:
        raise ValueError("causal linear_attention requires Lq == Lk")
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    B, H, Lq, Lk, d, dv = shp.B, shp.H, shp.Lq, shp.Lk, shp.d, shp.dv
    fq = elu_feature_map(q)
    fk = elu_feature_map(k)
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool or tuple(key_padding_mask.shape) != (B, Lk):
            raise ValueError("key_padding_mask must be bool [B, Lk]")
        fk = fk * key_padding_mask[:, None, :, None].to(fk.dtype)

    if not causal:
        kv = torch.matmul(fk.transpose(-2, -1), v)                       # [B,H,d,dv]
        z = fk.sum(dim=2)                                                # [B,H,d]
        num = torch.matmul(fq, kv)                                       # [B,H,Lq,dv]
        den = torch.matmul(fq, z.unsqueeze(-1))                          # [B,H,Lq,1]
        if stats is not None:
            stats["state_entries"] = B * H * d * dv
        return num / (den + eps)

    L = Lq
    C = chunk
    nC = -(-L // C)
    Lp = nC * C
    fq_c = F.pad(fq, (0, 0, 0, Lp - L)).view(B, H, nC, C, d)
    fk_c = F.pad(fk, (0, 0, 0, Lp - L)).view(B, H, nC, C, d)            # padded keys: phi = 0
    v_c = F.pad(v, (0, 0, 0, Lp - L)).view(B, H, nC, C, dv)

    # Intra-chunk (quadratic within C, causal).
    a = torch.matmul(fq_c, fk_c.transpose(-2, -1))                       # [B,H,nC,C,C]
    tri = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    a = a * tri.to(a.dtype)
    num = torch.matmul(a, v_c)                                           # [B,H,nC,C,dv]
    den = a.sum(dim=-1, keepdim=True)                                    # [B,H,nC,C,1]

    # Inter-chunk: exclusive prefix sums of per-chunk states.
    kv_chunk = torch.matmul(fk_c.transpose(-2, -1), v_c)                 # [B,H,nC,d,dv]
    z_chunk = fk_c.sum(dim=3)                                            # [B,H,nC,d]
    kv_prefix = torch.cumsum(kv_chunk, dim=2)
    z_prefix = torch.cumsum(z_chunk, dim=2)
    kv_excl = torch.cat([torch.zeros_like(kv_prefix[:, :, :1]), kv_prefix[:, :, :-1]], dim=2)
    z_excl = torch.cat([torch.zeros_like(z_prefix[:, :, :1]), z_prefix[:, :, :-1]], dim=2)
    num = num + torch.matmul(fq_c, kv_excl)
    den = den + torch.matmul(fq_c, z_excl.unsqueeze(-1))
    out = (num / (den + eps)).reshape(B, H, Lp, dv)[:, :, :L]
    if stats is not None:
        stats["score_entries"] = B * H * nC * C * C
        stats["state_entries"] = B * H * nC * d * dv
    return out


def linear_attention_quadratic_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """O(L^2) evaluation of the *same* linear-attention function (testing only)."""
    fq, fk = elu_feature_map(q), elu_feature_map(k)
    a = torch.matmul(fq, fk.transpose(-2, -1))                           # [B,H,Lq,Lk]
    if causal:
        a = a * torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device).tril().to(a.dtype)
    if key_padding_mask is not None:
        a = a * key_padding_mask[:, None, None, :].to(a.dtype)
    return torch.matmul(a, v) / (a.sum(-1, keepdim=True) + eps)
