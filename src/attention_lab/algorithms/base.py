"""Shared input validation and mask semantics for every attention algorithm.

Tensor layout (used everywhere in this project)
-----------------------------------------------
    q : [B, H, Lq, d]     queries
    k : [B, H, Lk, d]     keys
    v : [B, H, Lk, dv]    values
    out : [B, H, Lq, dv]

Mask semantics (identical for every algorithm that supports a given mask kind)
-------------------------------------------------------------------------------
* ``mask`` (bool)  — broadcastable to [B, H, Lq, Lk]. ``True`` means the
  (query, key) pair **takes part** in attention, ``False`` means it is
  removed. This matches ``torch.nn.functional.scaled_dot_product_attention``.
* ``mask`` (float) — broadcastable to [B, H, Lq, Lk]. Added to the scaled
  logits before the softmax. ``-inf`` removes a pair; finite values bias it.
  ``+inf`` and NaN are rejected.
* ``key_padding_mask`` (bool) — shape [B, Lk]. ``True`` marks a *real* key
  token, ``False`` marks padding that must never be attended to.
* ``causal=True`` — query i may attend key j iff ``j <= i`` (top-left
  alignment, the same convention as ``scaled_dot_product_attention`` with
  ``is_causal=True``). For Lq != Lk this is *not* the KV-cache
  (bottom-right) convention; see docs/attention_math.md.

Fully-masked rows
-----------------
If every key of a query row is removed, softmax is undefined (0/0). The
project-wide policy is: **such rows produce a zero output vector and zero
gradients**. This matches PyTorch's CPU SDPA kernel (verified in this
environment) and avoids NaN propagation. Tests check this policy for every
exact implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class Shapes:
    B: int
    H: int
    Lq: int
    Lk: int
    d: int
    dv: int


def validate_qkv(q: Tensor, k: Tensor, v: Tensor) -> Shapes:
    """Validate q/k/v and return their dimensions."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.dim() != 4:
            raise ValueError(f"{name} must have shape [B, H, L, d]; got {tuple(t.shape)}")
        if not t.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {t.dtype}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q/k/v dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.device == k.device == v.device):
        raise ValueError("q/k/v must be on the same device")
    B, H, Lq, d = q.shape
    Bk, Hk, Lk, dk = k.shape
    Bv, Hv, Lv, dv = v.shape
    if (Bk, Hk) != (B, H) or (Bv, Hv) != (B, H):
        raise ValueError(f"batch/head dims differ: q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)}")
    if dk != d:
        raise ValueError(f"q and k head dims differ: {d} vs {dk}")
    if Lv != Lk:
        raise ValueError(f"k and v sequence lengths differ: {Lk} vs {Lv}")
    if min(B, H, Lq, Lk, d, dv) <= 0:
        raise ValueError("all dimensions must be positive")
    return Shapes(B, H, Lq, Lk, d, dv)


def default_scale(d: int, scale: Optional[float]) -> float:
    if scale is None:
        return d ** -0.5
    if not (isinstance(scale, (int, float)) and scale > 0):
        raise ValueError(f"scale must be a positive number, got {scale!r}")
    return float(scale)


def _check_broadcastable(mask: Tensor, shp: Shapes) -> None:
    target = (shp.B, shp.H, shp.Lq, shp.Lk)
    if mask.dim() > 4:
        raise ValueError(f"mask has {mask.dim()} dims; expected <= 4")
    m_shape = (1,) * (4 - mask.dim()) + tuple(mask.shape)
    for ms, ts in zip(m_shape, target):
        if ms not in (1, ts):
            raise ValueError(f"mask shape {tuple(mask.shape)} not broadcastable to {target}")


class MaskSpec:
    """Lazily evaluates the combined mask for any (query-range, key-range) tile.

    Algorithms never need the dense [Lq, Lk] mask: they ask for the tile they
    are about to compute. A user-supplied dense ``mask`` is sliced through a
    broadcast *view*, so no Lq x Lk tensor is created by MaskSpec itself.
    """

    def __init__(
        self,
        shp: Shapes,
        *,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        causal: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.shp = shp
        self.causal = bool(causal)
        self.device = device
        self.bool_mask: Optional[Tensor] = None
        self.additive_mask: Optional[Tensor] = None
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("mask must be a tensor")
            _check_broadcastable(mask, shp)
            m = mask.reshape((1,) * (4 - mask.dim()) + tuple(mask.shape))
            if m.dtype == torch.bool:
                self.bool_mask = m
            elif m.is_floating_point():
                if torch.isnan(m).any() or torch.isposinf(m).any():
                    raise ValueError("additive mask must not contain NaN or +inf")
                self.additive_mask = m
            else:
                raise TypeError(f"mask must be bool or floating point, got {m.dtype}")
        self.kpm: Optional[Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be a bool tensor (True = real token)")
            if tuple(key_padding_mask.shape) != (shp.B, shp.Lk):
                raise ValueError(
                    f"key_padding_mask must have shape [B, Lk]=({shp.B}, {shp.Lk}); got {tuple(key_padding_mask.shape)}"
                )
            self.kpm = key_padding_mask[:, None, None, :]  # [B,1,1,Lk]

    @property
    def has_dense_mask(self) -> bool:
        return self.bool_mask is not None or self.additive_mask is not None

    @property
    def is_empty(self) -> bool:
        return not (self.causal or self.has_dense_mask or self.kpm is not None)

    def tile(self, qs: int, qe: int, ks: int, ke: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return ``(allowed, additive)`` for queries [qs, qe) x keys [ks, ke).

        ``allowed`` is a bool tensor broadcastable to [B, H, qe-qs, ke-ks]
        (True = keep) or None; ``additive`` is a float tensor or None.
        """
        allowed: Optional[Tensor] = None
        if self.causal:
            qi = torch.arange(qs, qe, device=self.device)[:, None]
            kj = torch.arange(ks, ke, device=self.device)[None, :]
            allowed = (kj <= qi)[None, None]
        if self.bool_mask is not None:
            bm = self._slice(self.bool_mask, qs, qe, ks, ke)
            allowed = bm if allowed is None else (allowed & bm)
        if self.kpm is not None:
            km = self.kpm[..., ks:ke]
            allowed = km if allowed is None else (allowed & km)
        additive = None
        if self.additive_mask is not None:
            additive = self._slice(self.additive_mask, qs, qe, ks, ke)
        return allowed, additive

    @staticmethod
    def _slice(m: Tensor, qs: int, qe: int, ks: int, ke: int) -> Tensor:
        q_sl = slice(None) if m.shape[2] == 1 else slice(qs, qe)
        k_sl = slice(None) if m.shape[3] == 1 else slice(ks, ke)
        return m[:, :, q_sl, k_sl]

    def tile_fully_masked_by_causality(self, qs: int, qe: int, ks: int) -> bool:
        """True if causality alone removes every pair in the tile (tile can be skipped)."""
        return self.causal and ks > qe - 1

    def dense_bool(self) -> Optional[Tensor]:
        """Combined *boolean* keep-mask (no additive part) at full [.., Lq, Lk] size.

        Only used by reference/dense code paths (naive, SDPA adapter, tests).
        """
        allowed, _ = self.tile(0, self.shp.Lq, 0, self.shp.Lk)
        return allowed


def apply_tile_mask(s: Tensor, allowed: Optional[Tensor], additive: Optional[Tensor]) -> Tensor:
    """Apply additive bias then boolean removal (as -inf) to a score tile. Out-of-place."""
    if additive is not None:
        s = s + additive.to(s.dtype)
    if allowed is not None:
        s = s.masked_fill(~allowed, float("-inf"))
    return s


def safe_row_max(s: Tensor) -> Tensor:
    """Row max of logits with -inf rows mapped to 0 (so exp(s - m) stays 0, not NaN).

    The max is *detached*: softmax is mathematically invariant to the shift,
    so gradients through the shift would cancel exactly in exact arithmetic
    and only add rounding noise in floating point.
    """
    m = s.detach().amax(dim=-1, keepdim=True)
    return torch.where(torch.isneginf(m), torch.zeros_like(m), m)
