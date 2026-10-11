"""Approximate methods equal their *own* exact definitions (restricted softmax / linear kernel)."""
import pytest
import torch

from attention_lab.algorithms import (
    band_mask, block_sparse_attention, layout_to_element_mask, linear_attention,
    linear_attention_quadratic_reference, local_window_attention, make_layout, naive_attention,
)
from tests.conftest import rand_qkv


def _grads(out, ins, seed=0):
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed), dtype=out.dtype)
    return torch.autograd.grad(out, ins, g)


@pytest.mark.parametrize("L", [1, 5, 37, 64])
@pytest.mark.parametrize("window", [0, 1, 4, 16, 100])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("chunk", [None, 3, 16])
def test_local_window_equals_band_masked_softmax(L, window, causal, chunk):
    q, k, v = rand_qkv(B=2, H=2, L=L, d=8, requires_grad=True)
    kpm = torch.rand(2, L, generator=torch.Generator().manual_seed(L)) > 0.2
    out = local_window_attention(q, k, v, window=window, causal=causal, key_padding_mask=kpm, chunk=chunk)
    ref = naive_attention(q, k, v, mask=band_mask(L, L, window, 0 if causal else window), causal=causal,
                          key_padding_mask=kpm)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    for a, b in zip(_grads(out, (q, k, v)), _grads(ref, (q, k, v))):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


def test_local_window_asymmetric_and_full_window_equals_full_attention():
    q, k, v = rand_qkv(L=30, d=8)
    out = local_window_attention(q, k, v, window_left=3, window_right=7)
    torch.testing.assert_close(out, naive_attention(q, k, v, mask=band_mask(30, 30, 3, 7)))
    torch.testing.assert_close(local_window_attention(q, k, v, window=29), naive_attention(q, k, v))
    torch.testing.assert_close(local_window_attention(q, k, v, window=29, causal=True), naive_attention(q, k, v, causal=True))


@pytest.mark.parametrize("L,b", [(64, 16), (77, 16), (5, 8), (130, 32)])
@pytest.mark.parametrize("cfg", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 2), (2, 2, 1)])
@pytest.mark.parametrize("causal", [False, True])
def test_block_sparse_equals_layout_masked_softmax(L, b, cfg, causal):
    local, glob, rand = cfg
    q, k, v = rand_qkv(B=2, H=2, L=L, d=8, requires_grad=True)
    kpm = torch.rand(2, L, generator=torch.Generator().manual_seed(L)) > 0.15
    out = block_sparse_attention(q, k, v, block_size=b, local_blocks=local, global_blocks=glob, random_blocks=rand,
                                 causal=causal, key_padding_mask=kpm, seed=3)
    nB = -(-L // b)
    layout = make_layout(nB, local_blocks=local, global_blocks=glob, random_blocks=rand, causal=causal, seed=3)
    ref = naive_attention(q, k, v, mask=layout_to_element_mask(layout, b, L, causal), causal=causal,
                          key_padding_mask=kpm)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    for a, c in zip(_grads(out, (q, k, v)), _grads(ref, (q, k, v))):
        torch.testing.assert_close(a, c, rtol=1e-11, atol=1e-11)


def test_block_sparse_dense_layout_equals_full_attention():
    q, k, v = rand_qkv(L=48, d=8)
    out = block_sparse_attention(q, k, v, block_size=16, local_blocks=5)
    torch.testing.assert_close(out, naive_attention(q, k, v))


def test_block_sparse_layout_properties():
    lay = make_layout(8, local_blocks=1, global_blocks=1, random_blocks=1, causal=True, seed=0)
    assert not torch.triu(lay, diagonal=1).any()            # causal
    assert lay[:, 0].all()                                  # global column
    assert torch.diagonal(lay).all()                        # each block sees itself
    lay2 = make_layout(8, local_blocks=1, global_blocks=1, random_blocks=1, causal=True, seed=0)
    assert torch.equal(lay, lay2)                           # seeded => reproducible


@pytest.mark.parametrize("L", [1, 9, 64, 100])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 16, 128])
def test_linear_chunked_equals_quadratic_definition(L, causal, chunk):
    q, k, v = rand_qkv(B=2, H=2, L=L, d=8, requires_grad=True)
    kpm = torch.rand(2, L, generator=torch.Generator().manual_seed(L)) > 0.2
    kpm[:, 0] = True
    out = linear_attention(q, k, v, causal=causal, key_padding_mask=kpm, chunk=chunk)
    ref = linear_attention_quadratic_reference(q, k, v, causal=causal, key_padding_mask=kpm)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)
    for a, b in zip(_grads(out, (q, k, v)), _grads(ref, (q, k, v))):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_linear_attention_is_not_softmax_attention():
    q, k, v = rand_qkv(L=32, d=8)
    diff = (linear_attention(q, k, v) - naive_attention(q, k, v)).abs().max()
    assert diff > 1e-2


def test_linear_attention_outputs_are_convex_combinations_of_values():
    q, k, v = rand_qkv(L=20, d=4)
    out = linear_attention(q, k, v, causal=True)
    assert (out <= v.amax(dim=2, keepdim=True) + 1e-9).all() and (out >= v.amin(dim=2, keepdim=True) - 1e-9).all()


@pytest.mark.parametrize("fn", [local_window_attention, block_sparse_attention, linear_attention])
def test_approx_methods_reject_unsupported_dense_masks(fn):
    q, k, v = rand_qkv(L=8, d=4)
    kw = {"window": 2} if fn is local_window_attention else {}
    with pytest.raises(NotImplementedError):
        fn(q, k, v, mask=torch.ones(8, 8, dtype=torch.bool), **kw)
