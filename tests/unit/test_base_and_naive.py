"""Shapes, scaling, masking semantics and input validation of the reference implementation."""
import math

import pytest
import torch

from attention_lab.algorithms import attention_probabilities, naive_attention
from attention_lab.algorithms.base import MaskSpec, validate_qkv
from tests.conftest import rand_qkv


def textbook(q, k, v, scale=None, bias=None):
    """softmax(QK^T * scale + bias) V via torch.softmax — an independent formulation."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = q @ k.transpose(-2, -1) * scale
    if bias is not None:
        s = s + bias
    return torch.softmax(s, dim=-1) @ v


def test_output_shape_with_cross_attention_and_different_value_dim():
    q, k, v = rand_qkv(B=2, H=3, L=5, d=4, Lk=7, dv=6)
    out = naive_attention(q, k, v)
    assert out.shape == (2, 3, 5, 6)


def test_matches_textbook_formula_and_default_scale():
    q, k, v = rand_qkv(L=9, d=16)
    torch.testing.assert_close(naive_attention(q, k, v), textbook(q, k, v), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(naive_attention(q, k, v, scale=0.3), textbook(q, k, v, scale=0.3), rtol=1e-12, atol=1e-12)


def test_scale_is_one_over_sqrt_d():
    q, k, v = rand_qkv(L=6, d=25)
    a = naive_attention(q, k, v)
    b = naive_attention(q, k, v, scale=1 / math.sqrt(25))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rows_of_probabilities_sum_to_one():
    q, k, _ = rand_qkv(L=11, d=8)
    p = attention_probabilities(q, k, causal=True)
    torch.testing.assert_close(p.sum(-1), torch.ones(p.shape[:-1], dtype=p.dtype))
    assert (p >= 0).all()


def test_bool_mask_true_means_attend():
    q, k, v = rand_qkv(B=1, H=1, L=4, d=4)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:, 2] = True                       # every query may only see key 2
    out = naive_attention(q, k, v, mask=mask)
    torch.testing.assert_close(out, v[:, :, 2:3].expand_as(out))


def test_additive_mask_is_added_to_scaled_logits():
    q, k, v = rand_qkv(L=5, d=4)
    bias = torch.randn(5, 5, dtype=torch.float64)
    torch.testing.assert_close(naive_attention(q, k, v, mask=bias), textbook(q, k, v, bias=bias))


def test_causal_is_top_left_aligned_like_sdpa():
    q, k, v = rand_qkv(B=1, H=1, L=3, d=4, Lk=5)
    tri = torch.ones(3, 5, dtype=torch.bool).tril()
    torch.testing.assert_close(naive_attention(q, k, v, causal=True), naive_attention(q, k, v, mask=tri))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(naive_attention(q, k, v, causal=True), ref)


def test_key_padding_mask_excludes_padded_keys():
    q, k, v = rand_qkv(B=2, H=1, L=6, d=4)
    kpm = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    out = naive_attention(q, k, v, key_padding_mask=kpm)
    torch.testing.assert_close(out[0], naive_attention(q[:1], k[:1, :, :3], v[:1, :, :3])[0])
    # changing padded values must not change the output
    v2 = v.clone()
    v2[0, :, 3:] = 1e6
    torch.testing.assert_close(naive_attention(q, k, v2, key_padding_mask=kpm), out)


def test_fully_masked_rows_give_zero_output_and_zero_grad():
    q, k, v = rand_qkv(B=2, H=1, L=5, d=4, requires_grad=True)
    kpm = torch.ones(2, 5, dtype=torch.bool)
    kpm[0] = False
    out = naive_attention(q, k, v, key_padding_mask=kpm)
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all()
    out.sum().backward()
    for t in (q, k, v):
        assert torch.isfinite(t.grad).all()
        assert (t.grad[0] == 0).all()


def test_masks_combine_by_intersection():
    q, k, v = rand_qkv(B=1, H=1, L=6, d=4)
    m = torch.rand(6, 6, generator=torch.Generator().manual_seed(1)) > 0.3
    kpm = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.bool)
    combined = m & torch.ones(6, 6, dtype=torch.bool).tril() & kpm[0][None, :]
    a = naive_attention(q, k, v, mask=m, causal=True, key_padding_mask=kpm)
    b = naive_attention(q, k, v, mask=combined)
    torch.testing.assert_close(a, b)


def test_lse_matches_logsumexp():
    q, k, v = rand_qkv(L=7, d=4)
    _, lse = naive_attention(q, k, v, return_lse=True)
    s = q @ k.transpose(-2, -1) * 0.5
    torch.testing.assert_close(lse, torch.logsumexp(s, -1))


def test_gradients_match_textbook():
    q, k, v = rand_qkv(L=6, d=4, requires_grad=True)
    g = torch.randn(2, 2, 6, 4, dtype=torch.float64)
    a = torch.autograd.grad(naive_attention(q, k, v), (q, k, v), g)
    b = torch.autograd.grad(textbook(q, k, v), (q, k, v), g)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_gradcheck_naive():
    q, k, v = rand_qkv(B=1, H=1, L=4, d=3, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: naive_attention(a, b, c, causal=True), (q, k, v))


@pytest.mark.parametrize(
    "shapes, err",
    [
        (((2, 2, 4, 8), (2, 2, 4, 7), (2, 2, 4, 8)), ValueError),   # head dim mismatch
        (((2, 2, 4, 8), (2, 2, 5, 8), (2, 2, 4, 8)), ValueError),   # k/v length mismatch
        (((2, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)), ValueError),   # batch mismatch
        (((2, 4, 8), (2, 4, 8), (2, 4, 8)), ValueError),            # wrong rank
    ],
)
def test_validation_rejects_bad_shapes(shapes, err):
    q, k, v = (torch.zeros(s) for s in shapes)
    with pytest.raises(err):
        validate_qkv(q, k, v)


def test_validation_rejects_bad_dtypes_and_masks():
    q = torch.zeros(1, 1, 2, 2)
    with pytest.raises(TypeError):
        validate_qkv(q.long(), q.long(), q.long())
    with pytest.raises(TypeError):
        validate_qkv(q, q.double(), q)
    with pytest.raises(ValueError):
        naive_attention(q, q, q, mask=torch.zeros(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        naive_attention(q, q, q, mask=torch.tensor([[0.0, float("inf")], [0.0, 0.0]]))
    with pytest.raises(TypeError):
        naive_attention(q, q, q, key_padding_mask=torch.ones(1, 2))
    with pytest.raises(ValueError):
        naive_attention(q, q, q, scale=-1.0)


def test_maskspec_tiles_agree_with_dense_mask():
    from attention_lab.algorithms.base import Shapes

    shp = Shapes(2, 1, 10, 10, 4, 4)
    m = torch.rand(10, 10, generator=torch.Generator().manual_seed(0)) > 0.5
    kpm = torch.rand(2, 10, generator=torch.Generator().manual_seed(1)) > 0.2
    spec = MaskSpec(shp, mask=m, key_padding_mask=kpm, causal=True)
    dense, _ = spec.tile(0, 10, 0, 10)
    dense = dense.expand(2, 1, 10, 10)
    for qs, qe, ks, ke in [(0, 3, 0, 4), (3, 10, 4, 7), (7, 10, 7, 10)]:
        t, _ = spec.tile(qs, qe, ks, ke)
        assert torch.equal(t.expand(2, 1, qe - qs, ke - ks), dense[:, :, qs:qe, ks:ke])
    assert spec.tile_fully_masked_by_causality(0, 3, 3)
    assert not spec.tile_fully_masked_by_causality(0, 3, 2)
