# Attention mathematics — and where each equation lives in the code

Notation: batch `B`, heads `H`, query length `Lq`, key length `Lk` (`L` when equal),
head dimension `d`, value dimension `dv`. Tensors are laid out `[B, H, L, d]`
everywhere (`algorithms/base.py`).

## 1. Scaled dot-product attention

For one head, with `Q ∈ R^{Lq×d}`, `K ∈ R^{Lk×d}`, `V ∈ R^{Lk×dv}` (Vaswani et al., 2017):

```
S = Q Kᵀ · τ + M            τ = 1/√d (default),  M = mask bias      S ∈ R^{Lq×Lk}
P = softmax_row(S)          P_ij = exp(S_ij) / Σ_k exp(S_ik)        P ∈ R^{Lq×Lk}
O = P V                                                              O ∈ R^{Lq×dv}
```

Code: `naive.py::naive_attention` — `s = q @ kᵀ * sc` → `apply_tile_mask` →
`m = safe_row_max(s)` → `e = exp(s - m)` → `l = e.sum` → `p = e / l` → `out = p @ v`.

**Why τ = 1/√d.** If the components of `q` and `k` are independent with mean 0 and
variance 1, then `q·k = Σ_{i≤d} q_i k_i` has mean 0 and variance `d` (Vaswani et al.,
2017, footnote 4). Without scaling, logits grow like √d, the softmax saturates
towards one-hot, and its gradients become extremely small. Multiplying by `1/√d` restores unit variance.
Test: `tests/unit/test_base_and_naive.py::test_scale_is_one_over_sqrt_d`.

**Each output row is a convex combination of value rows**: `P_ij ≥ 0`,
`Σ_j P_ij = 1`, so `O_i ∈ conv{V_j}`. This bounds outputs by the values and is
why output errors are naturally normalized by `max|V|`-like quantities
(`correctness/metrics.py`).

## 2. The softmax and its stable evaluation

`softmax(x)_j = e^{x_j} / Σ_k e^{x_k}` is invariant to adding a constant `c` to all
`x_j`, because `e^{x_j - c} / Σ_k e^{x_k - c}` multiplies the numerator and the
denominator by the same `e^{-c}`. Choosing `c = max_k x_k` makes every exponent `≤ 0`,
so nothing overflows. The largest term is `e^0 = 1`, so the denominator is `≥ 1`.
Code: `base.py::safe_row_max`. The max is *detached*. The output does not depend on
`c`, so the gradient contributions through `c` cancel exactly in exact arithmetic,
and detaching only removes rounding noise.

In float32, `exp(x)` overflows for `x > 88.7`. Without the shift, logits of that size
produce `inf/inf = NaN`. This is demonstrated numerically by
`correctness/lab.py::unstable_softmax_demo` (results in the correctness report).

## 3. Masks

Three mask kinds are combined by **intersection** (a pair survives only if every mask keeps it):

| Mask | Shape | Semantics | Code |
|---|---|---|---|
| boolean `mask` | broadcast to `[B,H,Lq,Lk]` | `True` = takes part (SDPA convention) | `MaskSpec.tile` |
| float `mask` | broadcast to `[B,H,Lq,Lk]` | added to `S`; `-inf` removes the pair | `apply_tile_mask` |
| `key_padding_mask` | `[B, Lk]` | `True` = real token, `False` = padding | `MaskSpec.kpm` |
| `causal=True` | — | keep iff `j ≤ i` | `MaskSpec.tile` |

A removed pair gets `S_ij = -inf`, so `exp(S_ij - m) = 0` and the pair contributes
nothing to either the numerator or the denominator.

**Causal alignment.** `j ≤ i` is the *top-left* convention, identical to
`scaled_dot_product_attention(is_causal=True)` (verified in
`test_causal_is_top_left_aligned_like_sdpa`). For `Lq < Lk` (decoding with a KV
cache), the usual intent is *bottom-right* alignment, `j ≤ i + (Lk − Lq)`. That is
**not** what `causal=True` does here. Pass an explicit boolean mask for that case.

**Fully-masked rows.** If every key of row `i` is removed, the softmax is `0/0`.
The project policy is to output **0 for that row with zero gradients**. The code
computes the max with `-inf → 0` and divides by `l` with `0 → 1`. This matches
PyTorch's CPU SDPA kernels, which we verified return zeros. Tests:
`test_fully_masked_rows_give_zero_output_and_zero_grad` and
`test_fully_masked_rows_and_padding` (for every exact method).

## 4. Gradients

With upstream gradient `dO`:

```
dV = Pᵀ dO
dP = dO Vᵀ
dS_ij = P_ij (dP_ij − D_i),     D_i = Σ_k P_ik dP_ik = Σ_c dO_ic O_ic
dQ = τ · dS K
dK = τ · dSᵀ Q
```

The middle line is the softmax Jacobian:
`∂P_ij/∂S_ik = P_ij(δ_jk − P_ik)`. Contracting it with `dP` gives
`P_ij(dP_ij − Σ_k P_ik dP_ik)`. The identity `Σ_k P_ik dP_ik = dO_i · O_i`
(because `dP_ik = dO_i · V_k` and `O_i = Σ_k P_ik V_k`) means that `D` can be
computed from `O` and `dO` alone. No `L×L` matrix is needed for it. This identity
is what lets the recomputation backward (`tiled_exact.py::_TiledRecomputeFn.backward`)
avoid storing `P`. Validated by `torch.autograd.gradcheck` in
`test_gradcheck_tiled_recompute_custom_backward`.

## 5. Why the score tensor is O(B·H·L²)

`S` has one entry per (batch, head, query, key): `B·H·Lq·Lk` numbers. For
self-attention that is `B·H·L²`. The naive implementation holds `s`, `e` and `p` at the
same time at its peak, and autograd keeps `e` and `p` alive for the backward pass.
So the *attention-score* memory is a small multiple of `B·H·L²·bytes`.

This is distinct from the other memory a model needs:

* **Inputs/outputs** `q, k, v, out`: `4·B·H·L·d` elements — linear in L.
* **Training activations of the rest of the layer** (QKV projection inputs, MLP
  activations, LayerNorm statistics): `O(B·L·d_model)` per layer.
* **Gradients**: one per parameter, plus activation gradients during backward.
* **Optimizer state**: AdamW keeps two extra tensors per parameter (`2×` the
  parameter count).

Once `L ≫ d`, the `L²` term dominates the activation memory of an attention layer.
For example, at `B=1, H=4, L=4096`, one fp32 score tensor is `4·4096²·4 B = 256 MiB`,
while `q, k, v, out` together are `4·4·4096·64·4 B = 16 MiB`. See
`complexity_analysis.md` for the per-method accounting and `experiments/` for measured
values.

## 6. What each implementation computes

| Implementation | Function computed | Exact? |
|---|---|---|
| `naive`, `sdpa_math`, `sdpa_flash_cpu`, `tiled`, `tiled_recompute` | softmax(QKᵀτ + M)V | yes (up to rounding) |
| `local_window` | the same, with `M` also removing pairs where `j < i−w` or `j > i+w` | exact *restricted* softmax |
| `block_sparse` | the same, with `M` removing pairs outside a block layout | exact *restricted* softmax |
| `linear_elu` | φ(Q)[φ(K)ᵀV] / φ(Q)φ(K)ᵀ1 with φ = elu+1 | **a different function** |

The two restricted methods are verified to equal `naive` with the corresponding mask
(`tests/numerical/test_approx_methods.py`). Their "error" relative to full attention
is therefore entirely *information loss* from the removed pairs. Linear attention
uses a different similarity kernel (§ in `complexity_analysis.md`). Its difference from
softmax attention is a difference of model family, not an approximation error with
a bound.
