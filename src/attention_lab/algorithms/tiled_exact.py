"""Implementation B — tiled exact attention with online softmax.

Computes exactly softmax(Q K^T * scale + M) V (up to floating-point rounding)
without ever materializing the full [Lq, Lk] score matrix.

For one query tile Q_i (rows [qs, qe)) we stream over key/value tiles
K_j, V_j and keep three running statistics per query row r:

    m_r   running maximum logit seen so far
    l_r   running softmax denominator, expressed relative to m_r
    acc_r running un-normalized output  sum_j exp(s_rj - m_r) v_j

Processing a new tile with logits S (shape [bq, bk]):

    m_new  = max(m, rowmax(S))
    alpha  = exp(m - m_new)              # rescales everything accumulated so far
    P      = exp(S - m_new)              # [bq, bk], only this tile
    l      = alpha * l + rowsum(P)
    acc    = alpha * acc + P @ V_j
    m      = m_new

and at the end  out = acc / l,  lse = m + log(l).
The derivation (and why this equals the true softmax) is in
docs/online_softmax.md. Mapping: variables there are named identically.

Two entry points:

* ``tiled_attention`` — PyTorch autograd differentiates through the Python
  tile loop. Correct gradients, but autograd *saves every tile's P* for the
  backward pass, so training memory is still O(Lq*Lk). Inference (no_grad)
  memory is O(bq*bk) per step.

* ``tiled_attention_recompute`` — a custom ``torch.autograd.Function``. The
  forward saves only Q, K, V, O and lse (O(L*d) extra). The backward
  *recomputes* each P tile from Q, K and lse, following the structure of
  FlashAttention-2's backward pass. Training memory becomes O(L*d + bq*bk)
  at the cost of recomputing Q K^T once more.

Neither is the official FlashAttention kernel: they are pure-PyTorch,
CPU-executed illustrations of the same algorithmic ideas, executed op-by-op
through the PyTorch dispatcher with no control over caches or registers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import MaskSpec, Tensor, apply_tile_mask, default_scale, validate_qkv

DEFAULT_TILE = 128


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate running statistics in float32 for 16-bit inputs (as FlashAttention does)."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _check_tiles(q_tile: int, kv_tile: int) -> None:
    for name, t in (("q_tile", q_tile), ("kv_tile", kv_tile)):
        if not isinstance(t, int) or t <= 0:
            raise ValueError(f"{name} must be a positive int, got {t!r}")


def online_softmax_update(m: Tensor, l: Tensor, acc: Tensor, s: Tensor, v_j: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """One online-softmax step (docs/online_softmax.md, eq. U1-U5). Out-of-place.

    m, l : [..., bq, 1]  running max and denominator (relative to m)
    acc  : [..., bq, dv] running un-normalized output
    s    : [..., bq, bk] masked, scaled logits of the new tile (-inf = removed)
    v_j  : [..., bk, dv] values of the new tile
    """
    m_new = torch.maximum(m, s.detach().amax(dim=-1, keepdim=True))           # U1
    # Rows that have seen only -inf so far: use shift 0 so exp() yields 0, not NaN.
    m_shift = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
    alpha = torch.exp(m - m_shift)                                             # U2 (exp(-inf) = 0 at start)
    p = torch.exp(s - m_shift)                                                 # U3
    l = alpha * l + p.sum(dim=-1, keepdim=True)                                # U4
    acc = alpha * acc + torch.matmul(p, v_j)                                   # U5
    return m_new, l, acc


def tiled_forward_kernel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    spec: MaskSpec,
    scale: float,
    q_tile: int,
    kv_tile: int,
    stats: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:
    """Online-softmax forward. Returns (out [B,H,Lq,dv] in q.dtype, lse [B,H,Lq] in acc dtype).

    Written out-of-place so autograd can differentiate through it.
    """
    B, H, Lq, d = q.shape
    Lk, dv = k.shape[2], v.shape[3]
    acc_dt = _acc_dtype(q.dtype)
    outs, lses = [], []
    n_entries = 0
    n_tiles = 0
    for qs in range(0, Lq, q_tile):
        qe = min(qs + q_tile, Lq)
        q_i = q[:, :, qs:qe].to(acc_dt) * scale                          # [B,H,bq,d]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), dtype=acc_dt, device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), dtype=acc_dt, device=q.device)
        acc = torch.zeros((B, H, qe - qs, dv), dtype=acc_dt, device=q.device)
        for ks in range(0, Lk, kv_tile):
            if spec.tile_fully_masked_by_causality(qs, qe, ks):
                break  # every later key tile is also strictly above the diagonal
            ke = min(ks + kv_tile, Lk)
            k_j = k[:, :, ks:ke].to(acc_dt)                              # [B,H,bk,d]
            v_j = v[:, :, ks:ke].to(acc_dt)                              # [B,H,bk,dv]
            s = torch.matmul(q_i, k_j.transpose(-2, -1))                 # [B,H,bq,bk]
            allowed, additive = spec.tile(qs, qe, ks, ke)
            s = apply_tile_mask(s, allowed, additive)
            n_entries += (qe - qs) * (ke - ks)
            n_tiles += 1

            m, l, acc = online_softmax_update(m, l, acc, s, v_j)
        l_safe = torch.where(l == 0, torch.ones_like(l), l)              # fully-masked rows -> 0
        outs.append((acc / l_safe).to(q.dtype))
        lses.append((m + torch.log(l)).squeeze(-1))                      # -inf for empty rows
    if stats is not None:
        stats["score_entries"] = stats.get("score_entries", 0) + n_entries * B * H
        stats["tiles"] = stats.get("tiles", 0) + n_tiles
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


def tiled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    q_tile: int = DEFAULT_TILE,
    kv_tile: int = DEFAULT_TILE,
    return_lse: bool = False,
    stats: Optional[dict] = None,
):
    """Tiled online-softmax attention; backward via autograd through the tile loop."""
    shp = validate_qkv(q, k, v)
    _check_tiles(q_tile, kv_tile)
    sc = default_scale(shp.d, scale)
    spec = MaskSpec(shp, mask=mask, key_padding_mask=key_padding_mask, causal=causal, device=q.device)
    out, lse = tiled_forward_kernel(q, k, v, spec, sc, q_tile, kv_tile, stats)
    return (out, lse) if return_lse else out


class _TiledRecomputeFn(torch.autograd.Function):
    """Exact attention whose backward recomputes P tile-by-tile from (Q, K, lse)."""

    @staticmethod
    def forward(ctx, q, k, v, spec, scale, q_tile, kv_tile, additive_needs_grad):
        out, lse = tiled_forward_kernel(q, k, v, spec, scale, q_tile, kv_tile)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.spec, ctx.scale, ctx.q_tile, ctx.kv_tile = spec, scale, q_tile, kv_tile
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_out, d_lse):
        q, k, v, out, lse = ctx.saved_tensors
        spec, scale, bq, bk = ctx.spec, ctx.scale, ctx.q_tile, ctx.kv_tile
        acc_dt = _acc_dtype(q.dtype)
        B, H, Lq, d = q.shape
        Lk = k.shape[2]
        d_out = d_out.to(acc_dt)
        # D_r = sum_c dO_rc * O_rc  (= sum_j P_rj dP_rj, the softmax-Jacobian correction term)
        D = (d_out * out.to(acc_dt)).sum(dim=-1, keepdim=True)            # [B,H,Lq,1]
        lse_safe = torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse).unsqueeze(-1)
        dq = torch.zeros(q.shape, dtype=acc_dt, device=q.device)
        dk = torch.zeros(k.shape, dtype=acc_dt, device=q.device)
        dv = torch.zeros(v.shape, dtype=acc_dt, device=q.device)
        for ks in range(0, Lk, bk):
            ke = min(ks + bk, Lk)
            k_j = k[:, :, ks:ke].to(acc_dt)
            v_j = v[:, :, ks:ke].to(acc_dt)
            dk_j = torch.zeros_like(k_j)
            dv_j = torch.zeros_like(v_j)
            for qs in range(0, Lq, bq):
                qe = min(qs + bq, Lq)
                if spec.tile_fully_masked_by_causality(qs, qe, ks):
                    continue
                q_i = q[:, :, qs:qe].to(acc_dt)
                s = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale     # recompute logits
                allowed, additive = spec.tile(qs, qe, ks, ke)
                s = apply_tile_mask(s, allowed, additive)
                p = torch.exp(s - lse_safe[:, :, qs:qe])                 # exact softmax probs of this tile
                do_i = d_out[:, :, qs:qe]
                dv_j += torch.matmul(p.transpose(-2, -1), do_i)          # dV = P^T dO
                dp = torch.matmul(do_i, v_j.transpose(-2, -1))           # dP = dO V^T
                ds = p * (dp - D[:, :, qs:qe])                           # dS = P o (dP - D)
                dq[:, :, qs:qe] += torch.matmul(ds, k_j) * scale         # dQ = dS K * scale
                dk_j += torch.matmul(ds.transpose(-2, -1), q_i) * scale  # dK = dS^T Q * scale
            dk[:, :, ks:ke] = dk_j
            dv[:, :, ks:ke] = dv_j
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def tiled_attention_recompute(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    q_tile: int = DEFAULT_TILE,
    kv_tile: int = DEFAULT_TILE,
    return_lse: bool = False,
):
    """Tiled attention with a memory-efficient recomputation backward.

    Gradients flow to q, k, v only (masks are treated as constants).
    """
    shp = validate_qkv(q, k, v)
    _check_tiles(q_tile, kv_tile)
    sc = default_scale(shp.d, scale)
    spec = MaskSpec(shp, mask=mask, key_padding_mask=key_padding_mask, causal=causal, device=q.device)
    if spec.additive_mask is not None and spec.additive_mask.requires_grad:
        raise NotImplementedError("tiled_attention_recompute does not differentiate w.r.t. the mask")
    out, lse = _TiledRecomputeFn.apply(q, k, v, spec, sc, q_tile, kv_tile, False)
    return (out, lse) if return_lse else out
