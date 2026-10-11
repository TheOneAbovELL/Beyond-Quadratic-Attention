"""Theoretical counts match instrumented counts; measurement utilities are trustworthy."""
import math

import numpy as np
import pytest
import torch

from attention_lab.algorithms import block_sparse_attention, local_window_attention, tiled_attention
from attention_lab.benchmarking import summarize, tensor_peak_bytes
from attention_lab.complexity import Shape, theoretical_cost
from attention_lab.reporting import fit_exponent
from tests.conftest import rand_qkv


@pytest.mark.parametrize("L,bq,bk", [(64, 16, 16), (70, 16, 24), (100, 32, 7), (5, 8, 8)])
@pytest.mark.parametrize("causal", [False, True])
def test_tiled_score_entries_theory_equals_instrumented(L, bq, bk, causal):
    q, k, v = rand_qkv(B=2, H=3, L=L, d=4)
    stats = {}
    tiled_attention(q, k, v, causal=causal, q_tile=bq, kv_tile=bk, stats=stats)
    th = theoretical_cost("tiled", Shape(2, 3, L, 4, "float64", causal), {"q_tile": bq, "kv_tile": bk})
    assert stats["score_entries"] == th["score_entries"]


def test_naive_entries_are_quadratic():
    a = theoretical_cost("naive", Shape(1, 1, 100, 8))["score_entries"]
    b = theoretical_cost("naive", Shape(1, 1, 200, 8))["score_entries"]
    assert b == 4 * a


@pytest.mark.parametrize("L,w,causal", [(100, 8, False), (100, 8, True), (257, 32, True), (64, 0, False)])
def test_local_window_entries_theory_equals_instrumented_and_scale_linearly(L, w, causal):
    q, k, v = rand_qkv(B=1, H=2, L=L, d=4)
    stats = {}
    local_window_attention(q, k, v, window=w, causal=causal, stats=stats)
    th = theoretical_cost("local_window", Shape(1, 2, L, 4, "float64", causal), {"window": w})
    assert stats["score_entries"] == th["score_entries"]
    assert stats["band_entries"] == th["band_pairs"]
    th2 = theoretical_cost("local_window", Shape(1, 2, 4 * L, 4, "float64", causal), {"window": w})
    assert th2["score_entries"] <= 4 * th["score_entries"] * 1.1          # linear in L (padding slack)


@pytest.mark.parametrize("L,b,cfg,causal", [(128, 16, (1, 1, 1), False), (130, 16, (1, 1, 1), True), (256, 32, (2, 0, 0), False)])
def test_block_sparse_entries_theory_equals_instrumented(L, b, cfg, causal):
    q, k, v = rand_qkv(B=1, H=1, L=L, d=4)
    stats = {}
    params = {"block_size": b, "local_blocks": cfg[0], "global_blocks": cfg[1], "random_blocks": cfg[2], "seed": 0}
    block_sparse_attention(q, k, v, causal=causal, stats=stats, **params)
    th = theoretical_cost("block_sparse", Shape(1, 1, L, 4, "float64", causal), params)
    assert stats["score_entries"] == th["score_entries"]
    assert stats["active_blocks"] == th["active_blocks"]


def test_linear_noncausal_has_no_quadratic_term():
    a = theoretical_cost("linear_elu", Shape(1, 1, 1000, 16))
    b = theoretical_cost("linear_elu", Shape(1, 1, 2000, 16))
    assert a["score_entries"] == 0
    assert math.isclose(b["matmul_flops"] / a["matmul_flops"], 2.0)


def test_tensor_peak_bytes_exact_on_known_allocation_pattern():
    MB = 1 << 20

    def f():
        a = torch.empty(10 * MB // 4)
        b = torch.empty(5 * MB // 4)
        del a
        c = torch.empty(3 * MB // 4)
        del b, c

    m = tensor_peak_bytes(f)
    assert m["peak_bytes"] == 15 * MB
    assert m["net_bytes"] == 0


def test_tensor_peak_sees_materialized_scores_and_tiling_reduces_it():
    q, k, v = rand_qkv(B=1, H=2, L=512, d=32, dtype=torch.float32)
    from attention_lab.algorithms import naive_attention

    naive = tensor_peak_bytes(lambda: naive_attention(q, k, v))["peak_bytes"]
    tiled = tensor_peak_bytes(lambda: tiled_attention(q, k, v, q_tile=64, kv_tile=64))["peak_bytes"]
    scores = 2 * 512 * 512 * 4
    assert naive >= 2 * scores              # at least s and e (or p) alive together
    assert tiled < scores / 4


def test_summarize_statistics():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median_s"] == 3.0 and s["min_s"] == 1.0 and s["max_s"] == 100.0
    assert s["iqr_s"] == 2.0 and s["mad_s"] == 1.0 and s["n"] == 5


def test_fit_exponent_recovers_power_laws():
    L = np.array([256, 512, 1024, 2048, 4096], dtype=float)
    assert abs(fit_exponent(L, 3e-9 * L**2) - 2.0) < 1e-9
    assert abs(fit_exponent(L, 5e-6 * L) - 1.0) < 1e-9
    assert fit_exponent(L[:2], L[:2]) is None
