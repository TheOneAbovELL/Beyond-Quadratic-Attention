"""Sweep scheduler: expand a YAML spec into configs, run small cases first, stop growing when unsafe.

Policies
--------
* Configs are sorted by sequence length (then batch/head/dim), so every
  method is measured at small sizes before anything large runs.
* Memory guard: each config is checked against min(mem_fraction * available
  RAM, max_mem_gb) using the theoretical estimate (runner.guard). A skipped
  config is *recorded* with status ``skipped_memory``.
* Growth stop: once a method/setting is skipped for memory, or its median
  time for any mode exceeds ``stop_if_slower_than_s``, all larger L for that
  same series are recorded as ``skipped_after_budget`` instead of being run.
* Background load: if the spec has ``wait_for_quiet: {max_other_cores,
  timeout_s}``, each config first waits (bounded) for other processes to go
  quiet. Every timing record also carries the measured background load
  (``load.py``) and a ``contended`` flag.
* Results are appended to ``results.jsonl`` one line per config as soon as
  each finishes, so an interrupted sweep keeps everything it measured.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml

from .env import environment_info
from .load import snapshot
from .runner import BenchConfig, run_isolated, run_one

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand(spec: Dict[str, Any]) -> List[BenchConfig]:
    grid = spec.get("grid", {})
    keys = ["B", "H", "L", "d", "dtype", "causal", "threads"]
    values = [grid.get(k, [None]) for k in keys]
    common = {k: spec[k] for k in ("warmup", "repeats", "seed", "max_seconds_per_mode", "mem_fraction",
                                   "max_mem_gb", "measure_error", "measure_tensor_memory") if k in spec}
    if "modes" in spec:
        common["modes"] = tuple(spec["modes"])
    configs = []
    for m in spec["methods"]:
        m = {"name": m} if isinstance(m, str) else m
        for combo in itertools.product(*values):
            kw = {k: v for k, v in zip(keys, combo) if v is not None}
            if m.get("causal_only") and not kw.get("causal"):
                continue
            if "max_L" in m and kw.get("L", 0) > m["max_L"]:
                continue
            configs.append(BenchConfig(method=m["name"], params=dict(m.get("params", {})), **kw, **common))
    configs.sort(key=lambda c: (c.L, c.B * c.H, c.d, c.label))
    return configs


def wait_for_quiet(max_other_cores: float, timeout_s: float) -> dict:
    """Block until system-wide busy logical CPUs <= max_other_cores, or timeout. Returns what happened."""
    t0 = time.perf_counter()
    last = snapshot(1.0)
    while last > max_other_cores and time.perf_counter() - t0 < timeout_s:
        last = snapshot(2.0)
    return {"quiet_wait_s": round(time.perf_counter() - t0, 1), "load_before_cores": round(last, 2),
            "quiet_achieved": last <= max_other_cores}


def _series_key(c: BenchConfig) -> tuple:
    return (c.label, c.B, c.H, c.d, c.dtype, c.causal, c.threads)


def run_sweep(
    spec: Dict[str, Any],
    *,
    out_dir: Optional[str] = None,
    isolate: Optional[bool] = None,
    progress: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
) -> str:
    name = spec.get("name", "sweep")
    isolate = spec.get("isolate", False) if isolate is None else isolate
    stop_s = spec.get("stop_if_slower_than_s", 5.0)
    out_dir = out_dir or os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%H%M%S") + f"_{name}")
    os.makedirs(out_dir, exist_ok=True)
    configs = expand(spec)
    meta = {"spec": spec, "env": environment_info(), "n_configs": len(configs), "isolate": isolate,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    dead: Dict[tuple, str] = {}
    path = os.path.join(out_dir, "results.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for i, cfg in enumerate(configs):
            key = _series_key(cfg)
            if key in dead:
                rec = {"label": cfg.label, **cfg.to_json(), "status": "skipped_after_budget", "reason": dead[key]}
            else:
                qw = spec.get("wait_for_quiet")
                wait_info = wait_for_quiet(qw["max_other_cores"], qw["timeout_s"]) if qw else {}
                rec = run_isolated(cfg) if isolate else run_one(cfg)
                rec.update(wait_info)
                if rec.get("status") == "skipped_memory":
                    dead[key] = f"memory guard tripped at L={cfg.L}"
                elif rec.get("status") in ("error", "timeout"):
                    dead[key] = f"{rec['status']} at L={cfg.L}"
                else:
                    slow = [m for m in cfg.modes if rec.get(f"{m}_median_s", 0) > stop_s]
                    if slow:
                        dead[key] = f"median time > {stop_s}s at L={cfg.L} ({','.join(slow)})"
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            if progress:
                progress(rec, i + 1, len(configs))
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return out_dir


def load_results(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "results.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_runs(results_dir: str = RESULTS_DIR) -> List[str]:
    """Benchmark sweep directories only (those whose meta.json records a sweep spec)."""
    if not os.path.isdir(results_dir):
        return []
    runs = []
    for d in os.listdir(results_dir):
        path = os.path.join(results_dir, d)
        meta = os.path.join(path, "meta.json")
        if os.path.isfile(os.path.join(path, "results.jsonl")) and os.path.isfile(meta):
            try:
                with open(meta, encoding="utf-8") as f:
                    if "spec" in json.load(f):
                        runs.append(path)
            except (OSError, ValueError):
                continue
    return sorted(runs)
