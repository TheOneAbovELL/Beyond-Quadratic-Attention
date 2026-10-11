"""Forward and gradient agreement of every exact implementation with the float64 reference."""
import pytest
import torch

from attention_lab.algorithms import EXACT_METHODS, get_method, naive_attention, tiled_attention, tiled_attention_recompute
from attention_lab.correctness.lab import CorrectnessCase, evaluate_case
from attention_lab.correctness.metrics import compare, tolerance_for
from tests.conftest import rand_qkv

TILES = {"q_tile": 16, "kv_tile": 24}
OPTIMIZED = [m for m in EXACT_METHODS if m != "naive"]


def _params(m):
    return TILES if m.startswith("tiled") else {}


@pytest.mark.parametrize("method", OPTIMIZED)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("L,d", [(1, 8), (7, 16), (50, 32), (129, 64)])
def test_forward_and_grad_within_dtype_tolerance(method, dtype, causal, L, d):
    rec = evaluate_case(method, CorrectnessCase(L=L, d=d, dtype=dtype, causal=causal, params=_params(method)))
    assert rec["status"] == "ok", rec.get("reason")
    assert rec["pass_forward"], (rec["fwd_max_abs_normalized"], rec["tol_forward"])
    assert rec["pass_grad"], (rec["grad_max_abs_normalized"], rec["tol_grad"])
    assert rec["fwd_nan_count"] == 0 and rec["grad_nan_count"] == 0


@pytest.mark.parametrize("method", OPTIMIZED)
@pytest.mark.parametrize("B,H", [(1, 1), (3, 5)])
def test_batch_and_head_dims(method, B, H):
    rec = evaluate_case(method, CorrectnessCase(B=B, H=H, L=33, d=16, params=_params(method)))
    assert rec["pass_forward"] and rec["pass_grad"]


@pytest.mark.parametrize("method", OPTIMIZED)
@pytest.mark.parametrize("magnitude", [0.01, 3.0, 10.0, 30.0])
def test_input_magnitude_sensitivity_float32(method, magnitude):
    rec = evaluate_case(method, CorrectnessCase(L=64, d=32, magnitude=magnitude, params=_params(method)))
    assert rec["fwd_nan_count"] == 0 and rec["fwd_inf_count"] == 0
    assert rec["pass_forward"] and rec["pass_grad"], (rec["fwd_max_abs_normalized"], rec["grad_max_abs_normalized"])


@pytest.mark.parametrize("method", OPTIMIZED)
@pytest.mark.parametrize("magnitude", [1.0, 10.0])
def test_bfloat16_fp32_accumulating_methods_pass_mixed_precision_policy(method, magnitude):
    rec = evaluate_case(method, CorrectnessCase(L=96, d=32, dtype="bfloat16", magnitude=magnitude, params=_params(method)))
    assert rec["pass_forward"] and rec["pass_grad"], rec


def test_finding_naive_in_pure_bfloat16_degrades_with_logit_magnitude():
    """Documented finding (docs/numerical_guide.md): bf16 logits lose absolute precision."""
    small = evaluate_case("naive", CorrectnessCase(L=96, d=32, dtype="bfloat16", magnitude=1.0))
    large = evaluate_case("naive", CorrectnessCase(L=96, d=32, dtype="bfloat16", magnitude=10.0))
    tiled = evaluate_case("tiled", CorrectnessCase(L=96, d=32, dtype="bfloat16", magnitude=10.0, params=TILES))
    assert large["fwd_max_abs_normalized"] > 5 * small["fwd_max_abs_normalized"]
    assert not large["pass_forward"]
    assert tiled["fwd_max_abs_normalized"] < large["fwd_max_abs_normalized"] / 10


def test_finding_fp32_error_floor_scales_with_logit_magnitude_for_every_implementation():
    """Investigation behind the logit term of the fp32 tolerance (docs/numerical_guide.md §4).

    The naive formula evaluated in fp32 has the same error as the fused kernel at large
    logits, so the error is the dtype's logit-rounding floor, not a kernel defect.
    """
    small = {m: evaluate_case(m, CorrectnessCase(L=64, d=32, magnitude=1.0, params=_params(m))) for m in ("naive", "sdpa_flash_cpu")}
    large = {m: evaluate_case(m, CorrectnessCase(L=64, d=32, magnitude=10.0, params=_params(m))) for m in ("naive", "sdpa_flash_cpu")}
    for m in large:
        assert large[m]["fwd_max_abs_normalized"] > 10 * small[m]["fwd_max_abs_normalized"]
        u_times_logit = 2.0**-24 * large[m]["max_abs_logit"]
        assert large[m]["fwd_max_abs_normalized"] < 2 * u_times_logit
    ratio = large["sdpa_flash_cpu"]["fwd_max_abs_normalized"] / large["naive"]["fwd_max_abs_normalized"]
    assert 0.25 < ratio < 4


@pytest.mark.parametrize("method", OPTIMIZED)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_fully_masked_rows_and_padding(method, dtype):
    for kw in ({"fully_masked_row": True}, {"key_padding": True, "causal": True}):
        rec = evaluate_case(method, CorrectnessCase(L=40, d=8, dtype=dtype, params=_params(method), **kw))
        assert rec["pass_forward"] and rec["pass_grad"], (kw, rec)


@pytest.mark.parametrize("fn", [tiled_attention, tiled_attention_recompute])
def test_tiled_supports_dense_bool_and_additive_masks(fn):
    q, k, v = rand_qkv(B=2, H=3, L=21, d=8, requires_grad=True)
    g = torch.Generator().manual_seed(5)
    bmask = torch.rand(2, 1, 21, 21, generator=g) > 0.4
    amask = torch.randn(1, 3, 21, 21, generator=g, dtype=torch.float64)
    dout = torch.randn(2, 3, 21, 8, dtype=torch.float64)
    for mask in (bmask, amask):
        a = fn(q, k, v, mask=mask, causal=True, q_tile=5, kv_tile=8)
        r = naive_attention(q, k, v, mask=mask, causal=True)
        torch.testing.assert_close(a, r, rtol=1e-11, atol=1e-11)
        ga = torch.autograd.grad(a, (q, k, v), dout)
        gr = torch.autograd.grad(r, (q, k, v), dout)
        for x, y in zip(ga, gr):
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)


def test_gradcheck_tiled_recompute_custom_backward():
    q, k, v = rand_qkv(B=1, H=2, L=7, d=3, requires_grad=True)
    kpm = torch.tensor([[1, 1, 1, 1, 0, 1, 1]], dtype=torch.bool)
    f = lambda a, b, c: tiled_attention_recompute(a, b, c, causal=True, key_padding_mask=kpm, q_tile=3, kv_tile=2)
    assert torch.autograd.gradcheck(f, (q, k, v))


def test_tiled_lse_matches_reference_lse():
    q, k, v = rand_qkv(L=30, d=8)
    _, lse_ref = naive_attention(q, k, v, causal=True, return_lse=True)
    _, lse = tiled_attention(q, k, v, causal=True, q_tile=7, kv_tile=9, return_lse=True)
    torch.testing.assert_close(lse, lse_ref)


def test_sequence_length_sensitivity_float32_error_stays_small():
    errs = []
    for L in (64, 256, 1024):
        q, k, v = rand_qkv(B=1, H=2, L=L, d=32, dtype=torch.float32)
        ref = naive_attention(q.double(), k.double(), v.double())
        errs.append(compare(tiled_attention(q, k, v, q_tile=64, kv_tile=64), ref)["max_abs_normalized"])
    assert max(errs) < tolerance_for(torch.float32, "forward")


def test_numerical_vs_bitwise_equality_are_distinct_notions():
    """Different reduction orders give tiny differences: numerically equal, not bitwise equal."""
    q, k, v = rand_qkv(B=1, H=1, L=200, d=32, dtype=torch.float32)
    a = naive_attention(q, k, v)
    b = tiled_attention(q, k, v, q_tile=64, kv_tile=48)
    m = compare(b, a)
    assert m["max_abs_normalized"] < 1e-5
    assert not m["bitwise_equal"]           # if this ever becomes equal, the claim in the docs needs revisiting
    assert compare(a, a)["bitwise_equal"]
