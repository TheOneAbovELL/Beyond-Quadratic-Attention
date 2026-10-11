"""Approximate method 2 — block-sparse attention.

What is restricted
------------------
Exact softmax attention restricted to a *block layout*: the sequence is cut
into blocks of ``block_size`` tokens and a boolean layout matrix
``layout[qb, kb]`` says whether query block qb may attend key block kb.
Inside allowed blocks the softmax is exact; pairs in disallowed blocks are
removed. The result equals naive attention with the layout expanded to an
element mask (verified by tests).

Layout used here (in the spirit of BigBird's window + global + random blocks):

* local:   |qb - kb| <= local_blocks          (causal: 0 <= qb - kb <= local_blocks)
* global:  key blocks kb < global_blocks are visible to every query block,
           and (non-causal only) query blocks qb < global_blocks see every key
* random:  each query block additionally sees ``random_blocks`` key blocks
           drawn with a fixed seed from those it could legally see
* causal:  layout restricted to kb <= qb and an element-level j <= i mask
           inside diagonal blocks.

Execution
---------
For each query block we *gather* only its active key blocks (padded to the
row maximum K_max), so the work is

    scores  [B, H, nQB, b, K_max * b]    instead of   [B, H, L, L].

Global query rows (which see every block) would force K_max = nB, so they
are computed separately against all keys. Sequence lengths that are not a
multiple of the block size are padded, and padded key positions are masked
out — this is where partial-tile masking matters (tested).

Theoretical work is proportional to the number of active blocks, but the
gather copies K/V blocks and pads rows; whether that beats a dense fused
kernel on a CPU is an empirical question answered by the benchmarks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Tensor, default_scale, safe_row_max, validate_qkv


def make_layout(
    n_blocks: int,
    *,
    local_blocks: int = 1,
    global_blocks: int = 0,
    random_blocks: int = 0,
    causal: bool = False,
    seed: int = 0,
) -> Tensor:
    """Build a [n_blocks, n_blocks] boolean block layout (True = computed)."""
    for name, val in (("local_blocks", local_blocks), ("global_blocks", global_blocks), ("random_blocks", random_blocks)):
        if not isinstance(val, int) or val < 0:
            raise ValueError(f"{name} must be a non-negative int, got {val!r}")
    qb = torch.arange(n_blocks)[:, None]
    kb = torch.arange(n_blocks)[None, :]
    if causal:
        layout = (qb - kb >= 0) & (qb - kb <= local_blocks)
    else:
        layout = (qb - kb).abs() <= local_blocks
    if global_blocks > 0:
        layout |= kb < global_blocks
        if not causal:
            layout |= qb < global_blocks
    if random_blocks > 0:
        gen = torch.Generator().manual_seed(seed)
        for r in range(n_blocks):
            legal = torch.arange(r + 1) if causal else torch.arange(n_blocks)
            candidates = legal[~layout[r, legal]]
            if len(candidates) == 0:
                continue
            pick = candidates[torch.randperm(len(candidates), generator=gen)[:random_blocks]]
            layout[r, pick] = True
    if causal:
        layout &= kb <= qb
    return layout


def layout_to_element_mask(layout: Tensor, block_size: int, L: int, causal: bool) -> Tensor:
    """Expand a block layout to a dense [L, L] boolean mask (reference/testing only)."""
    m = layout.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:L, :L]
    if causal:
        m = m & torch.ones(L, L, dtype=torch.bool).tril()
    return m


def block_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    block_size: int = 32,
    local_blocks: int = 1,
    global_blocks: int = 0,
    random_blocks: int = 0,
    seed: int = 0,
    layout: Optional[Tensor] = None,
    causal: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
    stats: Optional[dict] = None,
) -> Tensor:
    shp = validate_qkv(q, k, v)
    if mask is not None:
        raise NotImplementedError("block_sparse_attention supports causal and key_padding_mask only")
    if shp.Lq != shp.Lk:
        raise ValueError("block_sparse_attention requires Lq == Lk (self-attention)")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    sc = default_scale(shp.d, scale)
    B, H, L, d, dv = shp.B, shp.H, shp.Lq, shp.d, shp.dv
    b = block_size
    nB = -(-L // b)
    Lp = nB * b
    if layout is None:
        layout = make_layout(nB, local_blocks=local_blocks, global_blocks=global_blocks,
                             random_blocks=random_blocks, causal=causal, seed=seed)
    if tuple(layout.shape) != (nB, nB) or layout.dtype != torch.bool:
        raise ValueError(f"layout must be bool [{nB}, {nB}]")
    if causal:
        layout = layout & torch.ones(nB, nB, dtype=torch.bool).tril()
    layout = layout.to(q.device)

    qp = F.pad(q, (0, 0, 0, Lp - L)) * sc
    kp = F.pad(k, (0, 0, 0, Lp - L))
    vp = F.pad(v, (0, 0, 0, Lp - L))
    key_valid = torch.arange(Lp, device=q.device) < L                       # padded keys are invalid
    key_valid = key_valid[None].expand(B, Lp)
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool or tuple(key_padding_mask.shape) != (B, L):
            raise ValueError("key_padding_mask must be bool [B, L]")
        key_valid = key_valid & F.pad(key_padding_mask, (0, Lp - L), value=False)

    counts = layout.sum(dim=1)
    dense_rows = counts == nB if nB > 1 else torch.zeros(nB, dtype=torch.bool, device=q.device)
    if (~dense_rows).any():
        k_max = int(counts[~dense_rows].max().item())
    else:
        k_max = 0
    out = torch.zeros(B, H, nB, b, dv, dtype=q.dtype, device=q.device)
    n_scores = 0

    sparse_rows = torch.nonzero(~dense_rows).flatten()
    if len(sparse_rows) > 0:
        # Active key-block indices per row, padded to k_max (padding slots flagged invalid).
        order = torch.argsort((~layout[sparse_rows]).to(torch.int8), dim=1, stable=True)
        idx = order[:, :k_max]                                              # [nR, k_max]
        slot_valid = torch.gather(layout[sparse_rows], 1, idx)              # [nR, k_max]
        kb_ = kp.view(B, H, nB, b, d)[:, :, idx]                            # [B,H,nR,k_max,b,d]
        vb_ = vp.view(B, H, nB, b, dv)[:, :, idx]
        nR = len(sparse_rows)
        kg = kb_.reshape(B, H, nR, k_max * b, d)
        vg = vb_.reshape(B, H, nR, k_max * b, dv)
        qr = qp.view(B, H, nB, b, d)[:, :, sparse_rows]                     # [B,H,nR,b,d]
        s = torch.matmul(qr, kg.transpose(-2, -1))                          # [B,H,nR,b,k_max*b]
        n_scores += B * H * nR * b * k_max * b

        key_pos = (idx[:, :, None] * b + torch.arange(b, device=q.device)).reshape(nR, k_max * b)
        allowed = slot_valid.repeat_interleave(b, dim=1)[None]              # [1,nR,k_max*b]
        allowed = allowed & torch.gather(key_valid, 1, key_pos.reshape(1, -1).expand(B, -1)).view(B, nR, k_max * b)
        allowed = allowed[:, None, :, None, :]                              # [B,1,nR,1,k_max*b]
        if causal:
            q_pos = sparse_rows[:, None] * b + torch.arange(b, device=q.device)  # [nR,b]
            allowed = allowed & (key_pos[:, None, :] <= q_pos[:, :, None])[None, None]
        s = s.masked_fill(~allowed, float("-inf"))
        m = safe_row_max(s)
        e = torch.exp(s - m)
        l = e.sum(dim=-1, keepdim=True)
        p = e / torch.where(l == 0, torch.ones_like(l), l)
        out[:, :, sparse_rows] = torch.matmul(p, vg)

    dense_idx = torch.nonzero(dense_rows).flatten()
    if len(dense_idx) > 0:
        qr = qp.view(B, H, nB, b, d)[:, :, dense_idx].reshape(B, H, len(dense_idx) * b, d)
        s = torch.matmul(qr, kp.transpose(-2, -1))                          # [B,H,nD*b,Lp]
        n_scores += B * H * len(dense_idx) * b * Lp
        allowed = key_valid[:, None, None, :]
        if causal:
            q_pos = (dense_idx[:, None] * b + torch.arange(b, device=q.device)).reshape(-1)
            allowed = allowed & (torch.arange(Lp, device=q.device)[None, :] <= q_pos[:, None])[None, None]
        s = s.masked_fill(~allowed, float("-inf"))
        m = safe_row_max(s)
        e = torch.exp(s - m)
        l = e.sum(dim=-1, keepdim=True)
        p = e / torch.where(l == 0, torch.ones_like(l), l)
        out[:, :, dense_idx] = torch.matmul(p, vp).view(B, H, len(dense_idx), b, dv)

    if stats is not None:
        stats["score_entries"] = stats.get("score_entries", 0) + n_scores
        stats["active_blocks"] = int(layout.sum().item())
        stats["k_max"] = k_max
        stats["n_blocks"] = nB
    return out.reshape(B, H, Lp, dv)[:, :, :L]
