"""Run one benchmark configuration, in-process or in an isolated worker process.

Measurement order inside one configuration (chosen so instruments do not
contaminate each other):

  1. memory guard (theoretical estimate vs available RAM) — may skip the run
  2. create inputs; record process RSS
  3. for each mode (fwd, then fwd_bwd): time it; then read the OS peak
     working set (isolated mode only — it is a process-lifetime high-water mark)
  4. profiler-based tensor peak for each mode (separate, untimed calls)
  5. output error vs float64 naive reference (last: the reference itself is
     the most memory-hungry computation)

"fwd" runs under ``torch.inference_mode()`` (no autograd graph). "fwd_bwd"
runs forward with requires_grad inputs, then ``out.backward(dout)``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from ..algorithms.naive import naive_attention
from ..algorithms.registry import get_method
from ..complexity.theory import ELEM_BYTES, Shape, estimated_peak_bytes, theoretical_cost
from ..correctness.metrics import compare
from .env import environment_info
from .load import CONTENDED_THRESHOLD_CORES, LoadMonitor
from .memory import ProcessPeak, tensor_peak_bytes
from .timing import time_callable

DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchConfig:
    method: str
    B: int = 1
    H: int = 4
    L: int = 512
    d: int = 64
    dtype: str = "float32"
    causal: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    threads: int = 4
    warmup: int = 2
    repeats: int = 7
    modes: Tuple[str, ...] = ("fwd", "fwd_bwd")
    seed: int = 0
    measure_tensor_memory: bool = True
    measure_error: bool = True
    max_seconds_per_mode: float = 30.0
    mem_fraction: float = 0.5          # of currently available RAM
    max_mem_gb: float = 3.0            # absolute cap regardless of availability

    @property
    def label(self) -> str:
        if not self.params:
            return self.method
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.method}[{inner}]"

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modes"] = list(self.modes)
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "BenchConfig":
        d = dict(d)
        d["modes"] = tuple(d.get("modes", ("fwd", "fwd_bwd")))
        return cls(**d)


def memory_budget_bytes(cfg: BenchConfig) -> int:
    avail = psutil.virtual_memory().available
    return int(min(avail * cfg.mem_fraction, cfg.max_mem_gb * 1e9))


def guard(cfg: BenchConfig) -> Tuple[List[str], bool, Dict[str, Any]]:
    """Return (modes allowed to run, error measurement allowed, info). Per-mode decision."""
    shape = Shape(cfg.B, cfg.H, cfg.L, cfg.d, cfg.dtype, cfg.causal)
    params = {**get_method(cfg.method).default_params, **cfg.params}
    budget = memory_budget_bytes(cfg)
    info: Dict[str, Any] = {"guard_budget_bytes": budget}
    allowed = []
    for mode in cfg.modes:
        est = estimated_peak_bytes(cfg.method, shape, params, mode)
        info[f"guard_estimate_{mode}_bytes"] = est
        if est <= budget:
            allowed.append(mode)
    ref_est = 0
    if cfg.measure_error:
        ref_est = estimated_peak_bytes("naive", Shape(cfg.B, cfg.H, cfg.L, cfg.d, "float64", cfg.causal), {}, "fwd")
    info["guard_reference_estimate_bytes"] = ref_est
    fwd_est = info.get("guard_estimate_fwd_bytes", 0)
    return allowed, bool(allowed) and (fwd_est + ref_est) <= budget, info


def _make_inputs(cfg: BenchConfig):
    g = torch.Generator().manual_seed(cfg.seed)
    shape = (cfg.B, cfg.H, cfg.L, cfg.d)
    q, k, v, dout = (torch.randn(shape, generator=g) for _ in range(4))
    dt = DTYPES[cfg.dtype]
    return q.to(dt), k.to(dt), v.to(dt), dout.to(dt)


def run_one(cfg: BenchConfig, isolated: bool = False) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "label": cfg.label,
        **cfg.to_json(),
        "isolated_process": isolated,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": "cpu",
    }
    method = get_method(cfg.method)
    rec["method_family"] = method.family
    rec["implementation"] = method.implementation
    shape = Shape(cfg.B, cfg.H, cfg.L, cfg.d, cfg.dtype, cfg.causal)
    try:
        theory = theoretical_cost(cfg.method, shape, {**method.default_params, **cfg.params})
        rec.update({f"theory_{k}": v for k, v in theory.items()})
    except Exception as exc:
        rec["theory_error"] = str(exc)

    torch.set_num_threads(cfg.threads)
    rec["env"] = environment_info()
    modes, err_ok, ginfo = guard(cfg)
    rec.update(ginfo)
    skipped_modes = [m for m in cfg.modes if m not in modes]
    rec["modes_run"] = modes
    rec["modes_skipped_memory"] = skipped_modes
    if not modes:
        est = max(ginfo[f"guard_estimate_{m}_bytes"] for m in cfg.modes)
        rec["status"] = "skipped_memory"
        rec["reason"] = (f"estimated peak {est/1e9:.2f} GB exceeds budget "
                         f"{ginfo['guard_budget_bytes']/1e9:.2f} GB")
        return rec
    if DTYPES[cfg.dtype] not in method.dtypes:
        rec["status"] = "skipped_unsupported"
        rec["reason"] = f"{cfg.dtype} unsupported"
        return rec

    torch.manual_seed(cfg.seed)
    q, k, v, dout = _make_inputs(cfg)
    kw = {"causal": cfg.causal, **cfg.params}
    pp = ProcessPeak()
    rec["proc_rss_before_bytes"] = pp.rss()
    rec["input_bytes"] = 4 * q.numel() * ELEM_BYTES[cfg.dtype]

    def fwd():
        with torch.inference_mode():
            return method(q, k, v, **kw)

    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    def fwd_bwd():
        out = method(qg, kg, vg, **kw)
        out.backward(dout)
        qg.grad = kg.grad = vg.grad = None
        return None

    fns = {"fwd": fwd, "fwd_bwd": fwd_bwd}
    try:
        for mode in modes:
            with LoadMonitor() as mon:
                t = time_callable(fns[mode], warmup=cfg.warmup, repeats=cfg.repeats,
                                  max_seconds=cfg.max_seconds_per_mode)
            for key, val in t.items():
                rec[f"{mode}_{key}"] = val
            for key, val in mon.summary().items():
                rec[f"{mode}_{key}"] = val
            if isolated:
                peak = pp.peak()
                rec[f"proc_peak_after_{mode}_bytes"] = peak
                rec[f"proc_peak_delta_after_{mode}_bytes"] = (peak - rec["proc_rss_before_bytes"]) if peak else None
        if cfg.measure_tensor_memory:
            for mode in modes:
                m = tensor_peak_bytes(fns[mode])
                rec[f"{mode}_tensor_peak_bytes"] = m["peak_bytes"]
                rec[f"{mode}_tensor_net_bytes"] = m["net_bytes"]
        if cfg.measure_error:
            if err_ok:
                with torch.inference_mode():
                    out = method(q, k, v, **kw).double()
                    ref = naive_attention(q.double(), k.double(), v.double(), causal=cfg.causal)
                cmp = compare(out, ref)
                prefix = "error_vs_reference" if method.exact else "deviation_from_full_softmax"
                rec[f"{prefix}_max_abs_normalized"] = cmp["max_abs_normalized"]
                rec[f"{prefix}_rel_fro"] = cmp["rel_fro"]
                rec[f"{prefix}_cosine"] = cmp["cosine"]
                rec["output_nan_count"] = cmp["nan_count"]
                del out, ref
            else:
                rec["error_skipped_reason"] = "float64 reference would exceed memory budget"
        loads = [rec.get(f"{m}_other_load_cores_mean") for m in modes]
        loads = [x for x in loads if x is not None]
        rec["other_load_cores_mean"] = max(loads) if loads else None
        rec["contended"] = bool(loads) and max(loads) > CONTENDED_THRESHOLD_CORES
        rec["status"] = "ok" if not skipped_modes else "partial"
        if skipped_modes:
            rec["reason"] = f"modes {skipped_modes} skipped by memory guard"
    except Exception as exc:  # recorded honestly, never swallowed silently
        rec["status"] = "error"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=5)
    return rec


def run_isolated(cfg: BenchConfig, timeout_s: float = 900.0) -> Dict[str, Any]:
    """Run one configuration in a fresh Python process (clean OS peak-memory counter)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = str(cfg.threads)
    env["MKL_NUM_THREADS"] = str(cfg.threads)
    env["ATTENTION_LAB_CPU_NAME"] = environment_info()["cpu"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "attention_lab.benchmarking.worker"],
            input=json.dumps(cfg.to_json()), capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"label": cfg.label, **cfg.to_json(), "status": "timeout", "reason": f"exceeded {timeout_s}s"}
    wall = time.perf_counter() - t0
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"label": cfg.label, **cfg.to_json(), "status": "error",
                "reason": f"worker exit {proc.returncode}", "stderr_tail": proc.stderr[-2000:]}
    rec = json.loads(lines[-1])
    rec["worker_wall_s"] = wall
    return rec
