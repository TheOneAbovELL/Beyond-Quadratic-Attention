"""The online-softmax update in isolation, plus tiling edge cases."""
import pytest
import torch

from attention_lab.algorithms import naive_attention, tiled_attention, tiled_attention_recompute
from attention_lab.algorithms.tiled_exact import online_softmax_update
from tests.conftest import rand_qkv


def _stream(s, v, splits):
    """Run online_softmax_update over column chunks of s; return out = acc / l and lse."""
    m = torch.full(s.shape[:-1] + (1,), float("-inf"), dtype=s.dtype)
    l = torch.zeros_like(m)
    acc = torch.zeros(s.shape[:-1] + (v.shape[-1],), dtype=s.dtype)
    start = 0
    for size in splits:
        m, l, acc = online_softmax_update(m, l, acc, s[..., start:start + size], v[..., start:start + size, :])
        start += size
    return acc / l, (m + torch.log(l)).squeeze(-1)


@pytest.mark.parametrize("splits", [[10], [1] * 10, [3, 3, 4], [7, 1, 2], [2, 8]])
def test_streaming_equals_softmax_for_any_split(splits):
    g = torch.Generator().manual_seed(0)
    s = torch.randn(3, 5, 10, generator=g, dtype=torch.float64) * 4
    v = torch.randn(3, 10, 6, generator=g, dtype=torch.float64)
    out, lse = _stream(s, v, splits)
    torch.testing.assert_close(out, torch.softmax(s, -1) @ v)
    torch.testing.assert_close(lse, torch.logsumexp(s, -1))


def test_update_handles_increasing_maxima_without_overflow():
    # Later tiles have much larger logits: earlier contributions must be rescaled by alpha ~ exp(-1000) = 0.
    s = torch.tensor([[[0.0, 1.0, 1000.0, 1001.0]]], dtype=torch.float32)
    v = torch.eye(4, dtype=torch.float32)[None]
    out, _ = _stream(s, v, [2, 2])
    ref = torch.softmax(s.double(), -1) @ v.double()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), ref, rtol=1e-6, atol=1e-6)


def test_update_handles_all_masked_first_tile():
    s = torch.tensor([[[float("-inf"), float("-inf"), 0.5, 1.5]]], dtype=torch.float64)
    v = torch.randn(1, 4, 3, dtype=torch.float64)
    out, _ = _stream(s, v, [2, 2])
    torch.testing.assert_close(out, torch.softmax(s, -1) @ v)


@pytest.mark.parametrize("fn", [tiled_attention, tiled_attention_recompute])
@pytest.mark.parametrize("L, q_tile, kv_tile", [(1, 4, 4), (5, 8, 8), (17, 4, 5), (32, 32, 32), (33, 32, 32), (40, 1, 7)])
def test_partial_tiles_and_tile_larger_than_sequence(fn, L, q_tile, kv_tile):
    q, k, v = rand_qkv(L=L, d=8)
    for causal in (False, True):
        out = fn(q, k, v, causal=causal, q_tile=q_tile, kv_tile=kv_tile)
        torch.testing.assert_close(out, naive_attention(q, k, v, causal=causal), rtol=1e-12, atol=1e-12)


def test_causal_skips_tiles_above_diagonal():
    q, k, v = rand_qkv(B=1, H=1, L=64, d=4)
    full, causal = {}, {}
    tiled_attention(q, k, v, q_tile=16, kv_tile=16, stats=full)
    tiled_attention(q, k, v, q_tile=16, kv_tile=16, causal=True, stats=causal)
    assert full["tiles"] == 16
    assert causal["tiles"] == 10          # 4 + 3 + 2 + 1 lower-triangular tile blocks


@pytest.mark.parametrize("bad", [0, -3, 2.5, None])
def test_invalid_tile_sizes_rejected(bad):
    q, k, v = rand_qkv(L=4, d=4)
    with pytest.raises(ValueError):
        tiled_attention(q, k, v, q_tile=bad)
