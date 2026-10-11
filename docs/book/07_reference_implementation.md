# Chapter 7 — Implementation A: the reference ("naive") attention and the mask system

> **What you will learn:** why a project like this needs a slow, obvious reference implementation; how the
> reference is written, line by line; how masks are defined once and shared by every algorithm; the
> fully-masked-row problem and how a policy was chosen; input validation; and how the reference itself is
> tested.

## 7.1 Why start with a slow implementation?

The first principle of the project was **correctness before performance**: an "optimization" that
silently computes the wrong function is not an optimization. Every other algorithm is judged by comparing it
with something we trust, so that thing must be as simple and transparent as possible. The reference
(`naive_attention`) is written to be *read*, not to be fast. It literally builds the full `L × L` tables.

## 7.2 The code, step by step

> **In the code:** `src/attention_lab/algorithms/naive.py`, function `naive_attention`.

```python
shp = validate_qkv(q, k, v)                     # check shapes, dtypes, devices
sc = default_scale(shp.d, scale)                # 1/sqrt(d) unless told otherwise
spec = MaskSpec(shp, mask=mask, key_padding_mask=key_padding_mask, causal=causal, device=q.device)

s = torch.matmul(q, k.transpose(-2, -1)) * sc   # [B,H,Lq,Lk]   the quadratic table
allowed, additive = spec.tile(0, shp.Lq, 0, shp.Lk)
s = apply_tile_mask(s, allowed, additive)       # forbidden pairs -> -inf

m = safe_row_max(s)                             # [B,H,Lq,1]    row maximum (stable softmax)
e = torch.exp(s - m)                            # [B,H,Lq,Lk]   second quadratic table
l = e.sum(dim=-1, keepdim=True)                 # [B,H,Lq,1]    row sums
l_safe = torch.where(l == 0, torch.ones_like(l), l)
p = e / l_safe                                  # [B,H,Lq,Lk]   third quadratic table
out = torch.matmul(p, v)                        # [B,H,Lq,dv]
```

Every line maps to one step of Chapter 4's formula. Notice the three `L × L` tensors `s`, `e` and `p`. At the
peak of the call they all exist at once, which is why the theory model predicts naive forward memory as
"3 score tables + output". The measured memory matched to within 0.1 MiB (Chapter 14).

`naive_attention(..., return_lse=True)` also returns the **log-sum-exp** of each row, `m + log(l)`. The
tiled implementations store exactly this per-row statistic for their backward pass (Chapter 8), and tests
compare the two.

## 7.3 The stable softmax and the detached maximum

`safe_row_max` (in `base.py`) does two things:

```python
m = s.detach().amax(dim=-1, keepdim=True)
return torch.where(torch.isneginf(m), torch.zeros_like(m), m)
```

1. **Detach.** The softmax result does not depend on which constant is subtracted (Chapter 2), so in exact
   arithmetic the gradient flowing "through" `m` cancels out. Detaching tells autograd not to compute that
   cancelling term, which avoids a little extra rounding noise.
2. **Replace `−∞` with 0.** If every score in a row is `−∞` (a fully-masked row), the maximum is `−∞`, and
   `s − m` would be `−∞ − (−∞) = NaN`. Using 0 instead gives `exp(−∞ − 0) = 0`, so the whole row of `e` is
   zero, `l` is zero, and the `l_safe` line divides by 1 instead of 0. The output row is zero.

## 7.4 The fully-masked row: choosing a policy

Mathematically, a query with no allowed keys has an undefined attention output (`0/0`). Implementations
disagree about what to return. Some produce NaN, which then spreads through the network like a poison.

> **What actually happened:** during the audit, two tiny experiments were run. A plain
> `torch.softmax` over a row of `−∞` returned `[nan, nan, nan, nan]`. PyTorch's
> `scaled_dot_product_attention` on CPU, with a mask that removed a whole row, returned a **zero vector** for
> that row, and the gradients contained no NaN. Both of PyTorch's CPU backends (the fused flash kernel and the
> math kernel) were checked with boolean and with additive masks, and all returned zeros. The project
> therefore adopted the same policy, **fully-masked rows produce zero output and zero gradient**, wrote it
> into the docstring of `base.py`, and tested it for every exact method (`test_fully_masked_rows_and_padding`,
> `test_fully_masked_rows_give_zero_output_and_zero_grad`).

## 7.5 Masks are defined in exactly one place: `MaskSpec`

> **In the code:** `src/attention_lab/algorithms/base.py`, class `MaskSpec`.

Masks are easy to get subtly wrong. Is `True` "keep" or "remove"? Is causal "`j ≤ i`" or "`j < i`"? What happens
when the query and key lengths differ? To avoid each algorithm answering these questions differently, all
mask logic lives in one class, and the semantics are written at the top of `base.py`:

* **boolean `mask`**: `True` means the pair **takes part** (the same convention as PyTorch SDPA);
* **float `mask`**: added to the scaled scores, where `−∞` removes the pair; `+∞` and NaN are rejected;
* **`key_padding_mask`** `[B, Lk]`: `True` = real token;
* **`causal=True`**: query `i` may attend key `j` iff `j ≤ i` (top-left aligned, the same as SDPA's
  `is_causal`).

The key design idea: algorithms never ask for "the mask". They ask for **the mask of the tile they are about
to compute**:

```python
allowed, additive = spec.tile(qs, qe, ks, ke)   # queries [qs, qe) × keys [ks, ke)
```

For a causal mask, the tile is built on the fly from two small `arange` vectors (`kj <= qi`). For a
user-supplied dense mask, the tile is a **slice of a broadcast view**, so no `L × L` copy is made. This is
what lets the tiled algorithm support arbitrary masks while still never materializing an `L × L` table.

`spec.tile_fully_masked_by_causality(qs, qe, ks)` tells a tiled loop when a whole tile lies above the
diagonal, so it can skip it entirely (Chapter 8).

**Causal alignment when lengths differ.** "Top-left" means query 0 lines up with key 0. When generating text
with a cache of previous keys (`Lq < Lk`), the usual intent is "bottom-right" (the last query lines up with
the last key). The project deliberately implements only top-left, matching SDPA, and documents that you need
an explicit mask for the other convention. A test (`test_causal_is_top_left_aligned_like_sdpa`) pins this
behaviour against PyTorch.

## 7.6 Input validation

`validate_qkv` rejects, with a clear message:

* tensors that are not 4-D `[B, H, L, d]`;
* non-floating-point dtypes, or q/k/v with different dtypes or devices;
* mismatched batch or head dimensions; q and k with different head dimensions; k and v with different
  lengths;
* zero-sized dimensions.

`default_scale` rejects non-positive scales. `MaskSpec` rejects masks that cannot broadcast to
`[B, H, Lq, Lk]`, padding masks with the wrong shape or dtype, and additive masks containing `+∞` or NaN. Each
rejection has a test in `tests/unit/test_base_and_naive.py`.

## 7.7 How the reference itself is tested

A reference must be checked against something *independent*. The tests compare it with a separate
formulation, `softmax(QKᵀ·scale + bias) @ V` using `torch.softmax` (`textbook()` in the test file), and with
PyTorch's own SDPA:

| Test | What it proves |
|---|---|
| `test_matches_textbook_formula_and_default_scale` | same output as an independent formula, to 1e-12 |
| `test_scale_is_one_over_sqrt_d` | default scale is exactly 1/√d |
| `test_rows_of_probabilities_sum_to_one` | probabilities are non-negative and sum to 1 |
| `test_bool_mask_true_means_attend` | with only key 2 allowed, every output equals v₂ |
| `test_additive_mask_is_added_to_scaled_logits` | float masks are biases |
| `test_causal_is_top_left_aligned_like_sdpa` | causal = `tril` mask = PyTorch's `is_causal` |
| `test_key_padding_mask_excludes_padded_keys` | changing padded values to 10⁶ changes nothing |
| `test_masks_combine_by_intersection` | mask + causal + padding = their logical AND |
| `test_lse_matches_logsumexp` | the returned LSE equals `torch.logsumexp` |
| `test_gradients_match_textbook`, `test_gradcheck_naive` | gradients are correct (including finite-difference `gradcheck`) |

`torch.autograd.gradcheck` deserves a word. It perturbs each input by a tiny amount in float64, measures how
the output changes (a numerical derivative), and compares that with the gradient computed by autograd. It is
the gold standard for "is my backward pass right?".

## Key points to remember

* The reference is deliberately simple and builds three `L × L` tables.
* The stable softmax subtracts a *detached* row max, with `−∞` replaced by 0 for empty rows.
* Fully-masked rows output zero, a policy chosen by checking what PyTorch's CPU kernels do.
* `MaskSpec` defines mask semantics once and serves masks *per tile*, without building `L × L` masks.
* Causal is top-left aligned, like SDPA.
* The reference is tested against an independent formula, PyTorch SDPA, and finite-difference gradients.
