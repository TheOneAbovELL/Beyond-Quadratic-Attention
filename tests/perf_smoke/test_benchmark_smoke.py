"""Tiny benchmark runs: every method executes, records are complete, guards work. No timing thresholds."""
import json
import os

import pytest

from attention_lab.algorithms import REGISTRY
from attention_lab.benchmarking import BenchConfig, expand, load_results, run_one, run_sweep

pytestmark = pytest.mark.perf_smoke

REQUIRED = [
    "label", "method", "device", "threads", "B", "H", "L", "d", "dtype", "params", "seed", "warmup", "repeats",
    "fwd_median_s", "fwd_iqr_s", "fwd_bwd_median_s", "fwd_tensor_peak_bytes", "fwd_bwd_tensor_peak_bytes",
    "theory_score_entries", "theory_fwd_alloc_bytes", "env", "status",
]


@pytest.mark.parametrize("method", sorted(REGISTRY))
def test_every_method_benchmarks_on_a_tiny_config(method):
    rec = run_one(BenchConfig(method=method, B=1, H=2, L=64, d=16, warmup=0, repeats=2, threads=2))
    assert rec["status"] == "ok", rec.get("reason")
    for key in REQUIRED:
        assert key in rec, key
    assert rec["env"]["torch"] and rec["env"]["cpu"]
    assert rec["fwd_tensor_peak_bytes"] < 50 * 2**20
    err_key = "error_vs_reference_max_abs_normalized" if REGISTRY[method].exact else "deviation_from_full_softmax_max_abs_normalized"
    assert err_key in rec
    if REGISTRY[method].exact:
        assert rec[err_key] < 1e-4


def test_memory_guard_skips_without_allocating():
    rec = run_one(BenchConfig(method="naive", B=8, H=16, L=65536, d=64, max_mem_gb=1.0))
    assert rec["status"] == "skipped_memory"
    assert "exceeds budget" in rec["reason"]


def test_memory_guard_can_skip_only_backward():
    # Forward-only fits, forward+backward does not, under a tiny budget.
    rec = run_one(BenchConfig(method="naive", B=1, H=1, L=512, d=16, repeats=1, warmup=0, max_mem_gb=0.0045,
                              measure_error=False))
    assert rec["status"] in ("partial", "skipped_memory")
    if rec["status"] == "partial":
        assert rec["modes_run"] == ["fwd"] and "fwd_median_s" in rec and "fwd_bwd_median_s" not in rec


def test_scheduler_orders_small_first_and_writes_jsonl(tmp_path):
    spec = {"name": "t", "methods": ["naive", {"name": "tiled", "params": {"q_tile": 16, "kv_tile": 16}}],
            "grid": {"B": [1], "H": [1], "L": [64, 32], "d": [8], "threads": [1]},
            "warmup": 0, "repeats": 1, "isolate": False}
    cfgs = expand(spec)
    assert [c.L for c in cfgs] == sorted(c.L for c in cfgs)
    out = run_sweep(spec, out_dir=str(tmp_path / "run"))
    recs = load_results(out)
    assert len(recs) == 4 and all(r["status"] == "ok" for r in recs)
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["env"]["python"] and meta["spec"]["name"] == "t"
