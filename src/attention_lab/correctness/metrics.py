"""Error metrics and the dtype-aware tolerance policy.

Relative-error policy (explicit, because attention outputs are often near 0)
---------------------------------------------------------------------------
* ``rel_fro``  = ||x - r||_F / ||r||_F                (global, robust to zeros)
* ``max_rel``  = max |x - r| / |r|  over entries with |r| > ``tau``, where
  ``tau = REL_FLOOR * max|r|``. Entries below the floor are excluded and the
  excluded fraction is reported (``rel_excluded_frac``). Without a floor,
  a reference entry of 1e-12 would make any rounding difference look like a
  huge relative error.
* ``max_abs_normalized`` = max |x - r| / max |r|  — the quantity tolerances
  are checked against.

Tolerances
----------
For float64/float32 we use fixed thresholds on ``max_abs_normalized``
derived from unit roundoff u (f64: 1.1e-16, f32: 6.0e-8): a few hundred u
covers dot products of length d <= 256 and reductions over L <= 8192 with
typical (not worst-case) error growth. For large logits the threshold is
raised to ``LOGIT_FACTOR * u * max|S|``: a logit s is only representable to
~u*|s| absolutely, and exp turns that into a relative error of the same
size, for *any* implementation in that dtype (investigation: the fp32 naive
formula itself shows 2.0e-5 error at input scale 10, max|S| ~ 460). For 16-bit dtypes a fixed threshold
is a poor model, so the policy is *relative to a baseline*: an
implementation passes if its error vs the float64 reference is at most
``LOWP_FACTOR`` times the error of a **mixed-precision baseline** (naive
attention computed in float32 on the same rounded inputs, results rounded to
the 16-bit dtype — i.e. essentially output-rounding error), plus a small
absolute slack.

We deliberately do NOT use "naive attention computed entirely in bf16" as
the baseline: measured in this project, its error grows with logit
magnitude (0.15 normalized forward error at input scale 10 vs 2e-3 for
fp32-accumulating kernels) because bf16 logits lose absolute precision. A
baseline that bad would make the tolerance meaningless. See
docs/numerical_guide.md.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

REL_FLOOR = 1e-3
LOWP_FACTOR = 2.0

# (forward, gradient) thresholds on max_abs_normalized.
FIXED_TOLERANCES = {
    torch.float64: (1e-12, 1e-10),
    torch.float32: (2e-5, 2e-4),
}
UNIT_ROUNDOFF = {torch.float64: 2.0**-53, torch.float32: 2.0**-24, torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
# Logit-rounding term: a logit s carries absolute error ~u*|s|, which becomes a relative
# probability error of the same size. Measured: fp32 errors of every implementation
# (including the naive formula itself) are 0.1-1.5 x u*max|S| for input scales >= 1.
LOGIT_FACTOR = {"forward": 4.0, "grad": 16.0}

# Absolute slack for the relative-to-baseline policy (16-bit dtypes).
LOWP_SLACK = {
    torch.bfloat16: 4e-3,
    torch.float16: 5e-4,
}


def compare(x: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Compare ``x`` to ``ref`` (both cast to float64)."""
    xd = x.detach().to(torch.float64)
    rd = ref.detach().to(torch.float64)
    if xd.shape != rd.shape:
        raise ValueError(f"shape mismatch {tuple(xd.shape)} vs {tuple(rd.shape)}")
    nan = int(torch.isnan(xd).sum())
    inf = int(torch.isinf(xd).sum())
    finite = torch.isfinite(xd)
    diff = (xd - rd).abs()
    diff = torch.where(finite, diff, torch.full_like(diff, float("inf")))
    ref_max = float(rd.abs().max()) if rd.numel() else 0.0
    ref_norm = float(torch.linalg.vector_norm(rd))
    tau = REL_FLOOR * ref_max
    keep = rd.abs() > tau
    max_rel = float((diff[keep] / rd.abs()[keep]).max()) if keep.any() else 0.0
    xf, rf = xd.flatten(), rd.flatten()
    denom = float(torch.linalg.vector_norm(xf) * torch.linalg.vector_norm(rf))
    cos = float(torch.dot(xf, rf) / denom) if denom > 0 else float("nan")
    max_abs = float(diff.max()) if diff.numel() else 0.0
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "max_abs_normalized": max_abs / ref_max if ref_max > 0 else max_abs,
        "rel_fro": float(torch.linalg.vector_norm(torch.where(finite, xd - rd, torch.zeros_like(xd)))) / ref_norm
        if ref_norm > 0 else 0.0,
        "max_rel": max_rel,
        "rel_excluded_frac": float(1.0 - keep.float().mean()) if keep.numel() else 0.0,
        "cosine": cos,
        "nan_count": nan,
        "inf_count": inf,
        "bitwise_equal": bool(x.dtype == ref.dtype and torch.equal(x.detach(), ref.detach())),
    }


def tolerance_for(dtype: torch.dtype, kind: str, baseline_error: Optional[float] = None,
                  max_logit: Optional[float] = None) -> float:
    """Threshold on ``max_abs_normalized`` for dtype and kind in {"forward", "grad"}.

    fp64/fp32: max(fixed threshold, LOGIT_FACTOR * u * max|S|) — the second term is
    the unavoidable logit-rounding floor and only matters for large logits.
    16-bit: relative to the mixed-precision baseline (see module docstring).
    """
    if kind not in ("forward", "grad"):
        raise ValueError(kind)
    if dtype in FIXED_TOLERANCES:
        fwd, grad = FIXED_TOLERANCES[dtype]
        base = fwd if kind == "forward" else grad
        if max_logit is not None:
            base = max(base, LOGIT_FACTOR[kind] * UNIT_ROUNDOFF[dtype] * max_logit)
        return base
    if dtype in LOWP_SLACK:
        if baseline_error is None:
            raise ValueError(f"{dtype} uses the relative-to-baseline policy; pass baseline_error")
        slack = LOWP_SLACK[dtype] * (1.0 if kind == "forward" else 4.0)
        return LOWP_FACTOR * baseline_error + slack
    raise ValueError(f"no tolerance policy for {dtype}")
