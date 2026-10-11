"""Approximate method 1 — local (sliding-window) attention.

What is restricted
------------------
This computes *exact softmax attention restricted to a band*: query i may
only attend key j with  i - window_left <= j <= i + window_right.
Equivalently, it is naive attention with a banded boolean mask. Inside the
band the result is exact; the approximation is entirely the *removal* of
all pairs outside the band (information farther than the window is
invisible to a single layer). Tests verify equality with the
naive-with-band-mask reference.

With ``causal=True`` the band is [i - window_left, i] (window_right forced 0).

Algorithm (vectorized, no Python loop over positions)
-----------------------------------------------------
Split queries into chunks of C rows. Chunk c (rows [cC, cC+C)) can only see
keys in [cC - wl, cC + C + wr), a span of S = C + wl + wr keys. We pad K/V by
wl in front and wr (+ chunk padding) behind and take overlapping windows
with ``unfold`` (a strided view):

    q chunks     [B, H, nC, C, d]
    key windows  [B, H, nC, S, d]
    scores       [B, H, nC, C, S]      <- O(B*H*L*(C + wl + wr)), not O(L^2)
    out          [B, H, nC, C, dv] -> [B, H, L, dv]

Scores computed per row = C + wl + wr, of which only wl + wr + 1 are inside
the band; the rest are masked. Choosing C ~ wl + wr balances wasted work
against the number of (vectorized) chunks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Tensor, default_scale, safe_row_max, validate_qkv


def band_mask(Lq: int, Lk: int, window_left: int, window_right: int, device=None) -> Tensor:
    """Dense [Lq, Lk] boolean band mask (reference/testing only)."""
    i = torch.arange(Lq, device=device)[:, None]
    j = torch.arange(Lk, device=device)[None, :]
    return (j >= i - window_left) & (j <= i + window_right)


def band_entry_count(L: int, window_left: int, window_right: int) -> int:
    """Number of (i, j) pairs inside the band for an L x L self-attention, in O(L)."""
    i = torch.arange(L)
    lo = torch.clamp(i - window_left, min=0)
    hi = torch.clamp(i + window_right, max=L - 1)
    return int((hi - lo + 1).sum().item())


def _resolve_window(window, window_left, window_right, causal):
    if window is not None:
        if window_left is not None or window_right is not None:
            raise ValueError("pass either window or window_left/window_right, not both")
        window_left, window_right = window, (0 if causal else window)
    if window_left is None or window_right is None:
        raise ValueError("local_window_attention needs window (or window_left and window_right)")
    if causal:
        window_right = 0
    for name, w in (("window_left", window_left), ("window_right", window_right)):
        if not isinstance(w, int) or w < 0:
            raise ValueError(f"{name} must be a non-negative int, got {w!r}")
    return window_left, window_right


def local_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    window: Optional[int] = None,
    window_left: Optional[int] = None,
    window_right: Optional[int] = None,
    causal: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
    chunk: Optional[int] = None,
    stats: Optional[dict] = None,
) -> Tensor:
    shp = validate_qkv(q, k, v)
    if mask is not None:
        raise NotImplementedError("local_window_attention supports causal and key_padding_mask only")
    if shp.Lq != shp.Lk:
        raise ValueError("local_window_attention requires Lq == Lk (self-attention)")
    wl, wr = _resolve_window(window, window_left, window_right, causal)
    sc = default_scale(shp.d, scale)
    B, H, L, d, dv = shp.B, shp.H, shp.Lq, shp.d, shp.dv
    C = chunk if chunk is not None else max(16, min(256, wl + wr + 1))
    if not isinstance(C, int) or C <= 0:
        raise ValueError(f"chunk must be a positive int, got {C!r}")
    nC = -(-L // C)
    Lp = nC * C
    S = C + wl + wr

    qp = F.pad(q, (0, 0, 0, Lp - L)).view(B, H, nC, C, d) * sc
    kp = F.pad(k, (0, 0, wl, Lp - L + wr))                          # [B,H,Lp+wl+wr,d]
    vp = F.pad(v, (0, 0, wl, Lp - L + wr))
    kw = kp.unfold(2, S, C).transpose(-2, -1)                       # [B,H,nC,S,d]
    vw = vp.unfold(2, S, C).transpose(-2, -1)                       # [B,H,nC,S,dv]

    s = torch.matmul(qp, kw.transpose(-2, -1))                      # [B,H,nC,C,S]

    a = torch.arange(C, device=q.device)[:, None]                   # row within chunk
    b = torch.arange(S, device=q.device)[None, :]                   # key slot within window
    rel = b - a                                                     # = (j - i) + wl
    allowed = (rel >= 0) & (rel <= wl + wr)                         # [C,S] band
    c = torch.arange(nC, device=q.device)[:, None, None]
    key_pos = c * C - wl + b[None]                                  # [nC,1,S] absolute key index
    q_pos = c * C + a[None]                                         # [nC,C,1]
    allowed = allowed[None] & (key_pos >= 0) & (key_pos < L) & (q_pos < L)   # [nC,C,S]
    allowed = allowed[None, None]                                   # [1,1,nC,C,S]
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool or tuple(key_padding_mask.shape) != (B, L):
            raise ValueError("key_padding_mask must be bool [B, L]")
        kpm = F.pad(key_padding_mask, (wl, Lp - L + wr), value=False)
        kpm_w = kpm.unfold(1, S, C)                                 # [B,nC,S]
        allowed = allowed & kpm_w[:, None, :, None, :]
    s = s.masked_fill(~allowed, float("-inf"))

    m = safe_row_max(s)
    e = torch.exp(s - m)
    l = e.sum(dim=-1, keepdim=True)
    p = e / torch.where(l == 0, torch.ones_like(l), l)
    out = torch.matmul(p, vw).reshape(B, H, Lp, dv)[:, :, :L]
    if stats is not None:
        stats["score_entries"] = stats.get("score_entries", 0) + B * H * nC * C * S
        stats["band_entries"] = stats.get("band_entries", 0) + B * H * band_entry_count(L, wl, wr)
    return out
