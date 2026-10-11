"""Numerical correctness laboratory: compare exact implementations to a float64 reference.

Protocol for one case
---------------------
1. Draw q, k, v in float64 from N(0, 1) with a fixed seed; multiply q and k
   by ``magnitude`` (logit scale grows as magnitude^2).
2. Round them to the test dtype. The reference is the naive implementation
   in float64 applied to *those rounded inputs*, so input quantization is
   not counted as algorithm error.
3. Run the method in the test dtype; compute forward metrics and gradient
   metrics for a fixed random upstream gradient.
4. For 16-bit dtypes also run naive in that dtype ("baseline") and apply the
   relative-to-baseline tolerance policy (metrics.py).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch

from ..algorithms.naive import naive_attention
from ..algorithms.registry import get_method
from .metrics import compare, tolerance_for

DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class CorrectnessCase:
    B: int = 2
    H: int = 2
    L: int = 64
    d: int = 32
    dtype: str = "float32"
    causal: bool = False
    magnitude: float = 1.0
    key_padding: bool = False          # random padding, plus one fully padded batch row if B > 1
    fully_masked_row: bool = False     # make batch 0 have no valid keys at all
    params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def make_inputs(case: CorrectnessCase):
    g = torch.Generator().manual_seed(case.seed)
    shape = (case.B, case.H, case.L, case.d)
    q = torch.randn(shape, generator=g, dtype=torch.float64) * case.magnitude
    k = torch.randn(shape, generator=g, dtype=torch.float64) * case.magnitude
    v = torch.randn(shape, generator=g, dtype=torch.float64)
    dout = torch.randn(shape, generator=g, dtype=torch.float64)
    kpm = None
    if case.key_padding or case.fully_masked_row:
        kpm = torch.rand(case.B, case.L, generator=g) > 0.25
        kpm[:, 0] = True
        if case.fully_masked_row:
            kpm[0] = False
    return q, k, v, dout, kpm


def _fwd_bwd(fn, q, k, v, dout, **kw):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fn(q, k, v, **kw)
    grads = torch.autograd.grad(out, (q, k, v), dout.to(out.dtype))
    return out.detach(), [g.detach() for g in grads]


def evaluate_case(method_name: str, case: CorrectnessCase) -> Dict[str, Any]:
    method = get_method(method_name)
    dt = DTYPES[case.dtype]
    rec: Dict[str, Any] = {"method": method_name, **asdict(case)}
    if dt not in method.dtypes:
        rec.update(status="skipped", reason=f"{case.dtype} not supported by {method_name}")
        return rec
    q64, k64, v64, dout64, kpm = make_inputs(case)
    q, k, v = (t.to(dt) for t in (q64, k64, v64))
    common = dict(causal=case.causal, key_padding_mask=kpm)
    t0 = time.perf_counter()
    try:
        ref_out, ref_grads = _fwd_bwd(naive_attention, q.double(), k.double(), v.double(), dout64, **common)
        out, grads = _fwd_bwd(method, q, k, v, dout64, **common, **case.params)
    except Exception as exc:  # recorded, never hidden
        rec.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        return rec
    fwd = compare(out, ref_out)
    grad_metrics = [compare(g, r) for g, r in zip(grads, ref_grads)]
    worst_grad = max(grad_metrics, key=lambda m: m["max_abs_normalized"])
    rec.update({f"fwd_{k_}": v_ for k_, v_ in fwd.items()})
    rec.update({f"grad_{k_}": v_ for k_, v_ in worst_grad.items()})

    baseline_fwd = baseline_grad = None
    if dt in (torch.bfloat16, torch.float16):
        # Mixed-precision baseline: float32 compute on the rounded inputs, results rounded to dt.
        # Its error is essentially the unavoidable output-rounding error of the 16-bit format.
        b_out, b_grads = _fwd_bwd(naive_attention, q.float(), k.float(), v.float(), dout64, **common)
        baseline_fwd = compare(b_out.to(dt), ref_out)["max_abs_normalized"]
        baseline_grad = max(compare(g.to(dt), r)["max_abs_normalized"] for g, r in zip(b_grads, ref_grads))
        rec["baseline_fwd_max_abs_normalized"] = baseline_fwd
        rec["baseline_grad_max_abs_normalized"] = baseline_grad
    # Largest finite logit magnitude (feeds the logit-rounding term of the fp32/fp64 tolerance).
    logits = torch.matmul(q.double(), k.double().transpose(-2, -1)) * case.d ** -0.5
    max_logit = float(logits.abs().max())
    rec["max_abs_logit"] = max_logit
    tol_f = tolerance_for(dt, "forward", baseline_fwd, max_logit)
    tol_g = tolerance_for(dt, "grad", baseline_grad, max_logit)
    rec["tol_forward"] = tol_f
    rec["tol_grad"] = tol_g
    finite = fwd["nan_count"] == 0 and fwd["inf_count"] == 0 and worst_grad["nan_count"] == 0
    rec["pass_forward"] = bool(finite and fwd["max_abs_normalized"] <= tol_f)
    rec["pass_grad"] = bool(finite and worst_grad["max_abs_normalized"] <= tol_g)
    rec["status"] = "ok"
    rec["seconds"] = time.perf_counter() - t0
    return rec


def default_cases(quick: bool = True) -> List[CorrectnessCase]:
    """The case matrix. ``quick`` keeps it to a few seconds; full adds longer L and more dtypes."""
    cases: List[CorrectnessCase] = []
    Ls = [1, 7, 64, 129] if quick else [1, 7, 64, 129, 512, 1024]
    ds = [16, 64] if quick else [8, 16, 64, 128]
    dtypes = ["float64", "float32", "bfloat16"]
    tile_params = {"q_tile": 32, "kv_tile": 48}   # deliberately not dividing L -> partial tiles
    for L, d, dtype, causal in itertools.product(Ls, ds, dtypes, [False, True]):
        cases.append(CorrectnessCase(L=L, d=d, dtype=dtype, causal=causal, params=dict(tile_params)))
    for mag in ([0.1, 3.0, 10.0] if quick else [0.01, 0.1, 3.0, 10.0, 30.0]):
        for dtype in ["float32", "bfloat16"]:
            cases.append(CorrectnessCase(L=96, d=32, dtype=dtype, magnitude=mag, params=dict(tile_params)))
    for B, H in [(1, 1), (3, 5)]:
        cases.append(CorrectnessCase(B=B, H=H, L=50, d=16, params=dict(tile_params)))
    cases.append(CorrectnessCase(L=70, d=16, key_padding=True, causal=True, params=dict(tile_params)))
    cases.append(CorrectnessCase(L=70, d=16, fully_masked_row=True, params=dict(tile_params)))
    cases.append(CorrectnessCase(L=70, d=16, dtype="float64", fully_masked_row=True, causal=True, params=dict(tile_params)))
    return cases


def run_suite(methods: Iterable[str], cases: Optional[List[CorrectnessCase]] = None, quick: bool = True,
              progress=None) -> List[Dict[str, Any]]:
    cases = cases if cases is not None else default_cases(quick)
    records = []
    for case in cases:
        for m in methods:
            params = case.params if m.startswith("tiled") else {}
            c = CorrectnessCase(**{**asdict(case), "params": params})
            rec = evaluate_case(m, c)
            records.append(rec)
            if progress:
                progress(rec)
    return records


def unstable_softmax_demo(magnitudes=(1.0, 5.0, 10.0, 20.0), d: int = 64, L: int = 32, dtype=torch.float32):
    """Show why max-subtraction matters: softmax via exp(s)/sum(exp(s)) overflows for large logits."""
    rows = []
    for mag in magnitudes:
        g = torch.Generator().manual_seed(0)
        q = (torch.randn(1, 1, L, d, generator=g) * mag).to(dtype)
        k = (torch.randn(1, 1, L, d, generator=g) * mag).to(dtype)
        s = q @ k.transpose(-2, -1) * d ** -0.5
        e = torch.exp(s)
        unstable = e / e.sum(-1, keepdim=True)
        stable = torch.softmax(s.double(), -1)
        rows.append({
            "magnitude": mag,
            "max_logit": float(s.abs().max()),
            "unstable_nan_frac": float(torch.isnan(unstable).float().mean()),
            "unstable_max_err": float((unstable.double() - stable).abs().nan_to_num(float("inf")).max()),
            "stable_max_err": float((torch.softmax(s, -1).double() - stable).abs().max()),
        })
    return rows
