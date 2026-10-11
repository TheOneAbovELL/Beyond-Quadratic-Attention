# Chapter 8 — Implementation B: tiled exact attention and the online softmax

> **What you will learn:** the central trick of FlashAttention, called *online softmax*: how to compute an
> exact softmax-weighted average while seeing the scores one tile at a time; a worked numerical example;
> the proof that it is exact; how the project implements it; why letting autograd differentiate through it
> does *not* save training memory; and how a custom "recompute" backward pass fixes that.

## 8.1 The problem

Naive attention builds the whole row of scores `s₁ … s_L` for a query before applying softmax, because the
softmax denominator `Σ exp(sⱼ)` needs *all* of them. The question is whether the scores can be processed in
small **tiles** (chunks of keys) so that the full row never exists, *and still get the exact answer*.

The difficulty is that the stable softmax subtracts the row **maximum**, and we only know the true maximum
after seeing every tile.

## 8.2 The idea: keep three running numbers and fix them up when the maximum changes

For one query row, while streaming over tiles, keep:

* `m`: the largest score seen so far;
* `l`: the sum of `exp(sⱼ − m)` over keys seen so far (the denominator, *relative to the current `m`*);
* `acc`: the sum of `exp(sⱼ − m) · vⱼ` over keys seen so far (the numerator, also relative to `m`).

When a new tile arrives with scores `S` and values `V_tile`:

```
U1   m_new = max(m, max(S))              new running maximum
U2   α     = exp(m − m_new)              how much to shrink the old sums (≤ 1)
U3   P     = exp(S − m_new)              this tile's weights, relative to m_new
U4   l     = α · l   + sum(P)            rescale the old denominator, add the new terms
U5   acc   = α · acc + P · V_tile        rescale the old numerator, add the new terms
     m     = m_new
```

At the end: **`output = acc / l`**, and the log-sum-exp is `lse = m + log(l)`.

The only "trick" is U2. Old sums were computed relative to the old maximum. Multiplying by
`exp(m − m_new)` re-bases them to the new maximum, because `exp(s − m) · exp(m − m_new) = exp(s − m_new)`.

## 8.3 A worked example

One query, four keys, scores `s = [1, 3, 2, 5]`, scalar values `v = [10, 20, 30, 40]`, tiles of 2 keys.

**Start:** `m = −∞, l = 0, acc = 0`.

**Tile 0** (scores `[1, 3]`, values `[10, 20]`):

```
m_new = max(−∞, 3) = 3
α     = exp(−∞ − 3) = 0
P     = [exp(1−3), exp(3−3)] = [0.135335, 1]
l     = 0·0 + 1.135335 = 1.135335
acc   = 0·0 + 0.135335·10 + 1·20 = 21.353353
```

**Tile 1** (scores `[2, 5]`, values `[30, 40]`): the maximum rises from 3 to 5, so the old sums shrink:

```
m_new = max(3, 5) = 5
α     = exp(3 − 5) = 0.135335
P     = [exp(2−5), exp(5−5)] = [0.049787, 1]
l     = 0.135335 · 1.135335 + 1.049787 = 1.203438
acc   = 0.135335 · 21.353353 + 0.049787·30 + 1·40 = 44.383474
```

**Result:** `acc / l = 36.880566`, and `lse = 5 + log(1.203438) = 5.185182`.

**Check with the direct formula:** weights `exp([1,3,2,5]) / Σ` and the weighted average of `[10,20,30,40]`
give `36.880566`, with `lse = log Σ exp = 5.185182`. They are identical (to 15 digits in this calculation).

## 8.4 Why it is exactly right (the proof)

**Invariant:** after processing tiles 1..t, `l = Σ_{seen j} exp(sⱼ − m)` and
`acc = Σ_{seen j} exp(sⱼ − m) vⱼ`, where `m` is the max over seen keys.

* It holds at the start (empty sums are 0; `m = −∞`).
* If it holds before a tile, then `α · l = Σ_old exp(sⱼ − m)·exp(m − m_new) = Σ_old exp(sⱼ − m_new)`, and U4
  adds exactly the new terms `Σ_tile exp(sⱼ − m_new)`. So it holds after. The same argument works for `acc`.
* At the end, `acc / l = Σ exp(sⱼ − m) vⱼ / Σ exp(sⱼ − m)`; the common `exp(−m)` cancels, leaving
  `Σ exp(sⱼ) vⱼ / Σ exp(sⱼ)`, **the softmax-weighted average**. ∎

Tile order and tile sizes do not matter *mathematically*. A unit test streams the same row with five
different splits (`[10]`, `[1]×10`, `[3,3,4]`, `[7,1,2]`, `[2,8]`) and gets the same answer every time
(`test_streaming_equals_softmax_for_any_split`).

**Numerical safety.** Every exponent is `≤ 0` (scores never exceed the running max), so nothing overflows.
Even if a later tile is 1,000 larger, `α = exp(−1000) = 0` simply discards the negligible old mass
(`test_update_handles_increasing_maxima_without_overflow`). If all scores seen so far are masked (`−∞`), the
code shifts by 0 instead of `−∞` so that no NaN appears (`test_update_handles_all_masked_first_tile`).

## 8.5 The implementation

> **In the code:** `src/attention_lab/algorithms/tiled_exact.py`.

The update rule is a standalone function, so that the documentation, the unit tests and the kernel all use
the same code:

```python
def online_softmax_update(m, l, acc, s, v_j):
    m_new = torch.maximum(m, s.detach().amax(dim=-1, keepdim=True))           # U1
    m_shift = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
    alpha = torch.exp(m - m_shift)                                             # U2
    p = torch.exp(s - m_shift)                                                 # U3
    l = alpha * l + p.sum(dim=-1, keepdim=True)                                # U4
    acc = alpha * acc + torch.matmul(p, v_j)                                   # U5
    return m_new, l, acc
```

The forward kernel, `tiled_forward_kernel`, loops over **query tiles** (outer loop, `q_tile` rows) and **key
tiles** (inner loop, `kv_tile` keys). For each tile pair it computes a small `q_tile × kv_tile` score block,
asks `MaskSpec` for that tile's mask, and applies the update. It works on whole `[B, H, …]` blocks at once, so
all batches and heads advance together. Three details:

* **Ragged final tiles.** If `L` is not a multiple of the tile size, the last tile is simply shorter (Python
  slicing handles it). Tests use awkward sizes such as `L = 77` with tiles of 16 and 24.
* **Causal tile skipping.** If a key tile starts after the last query of the current block, every pair is
  above the diagonal, and every later tile is too, so the inner loop `break`s. For `L = 64` with 16 × 16
  tiles, 10 of 16 tiles are computed (`test_causal_skips_tiles_above_diagonal`).
* **fp32 accumulators for 16-bit inputs.** If the inputs are `bfloat16`, the running `m`, `l`, `acc` and the
  score tiles are computed in fp32 (`_acc_dtype`). FlashAttention does the same. Chapter 11 shows this
  single choice is the difference between 0.3 % and 38 % error.

> **What actually happened:** the update function was originally written inline inside the kernel. Halfway
> through the first benchmark sweep it was moved into the standalone `online_softmax_update`, so that tests
> could exercise it alone. Because the sweep's worker processes import the code fresh, later configurations
> of that sweep ran the refactored version. The operations were identical, plus one Python function call per
> tile, which is negligible. This was noted, and that sweep's timings were later discarded anyway, for a
> different reason (Chapter 13).

## 8.6 The backward pass: where tiling alone fails

Training needs gradients. The simplest option, `tiled_attention`, lets PyTorch's **autograd** differentiate
straight through the Python loop. That gives correct gradients (tests confirm it to 1e-15 in fp64). But
autograd must **keep every tile's intermediate tensors** (each `P`, each `exp` output) alive until the
backward pass. Added up over all tiles, that is again `O(L²)` memory. The forward pass was memory-efficient;
training is not.

The measurement in Chapter 14 shows it plainly. The forward-plus-backward memory of the autograd-tiled
version grows with exponent **1.72** in L (76 MiB at L = 2048), compared with **0.75** for the recompute
version (9.9 MiB).

## 8.7 The fix: recompute instead of store

`tiled_attention_recompute` is a custom `torch.autograd.Function`, a piece of code with a hand-written
`backward`:

* **Forward:** run the same tiled kernel, but save only `Q, K, V`, the output `O` and the per-row
  `lse = m + log(l)`. That is `O(L·d)` memory, nothing quadratic.
* **Backward:** loop over key tiles (outer) and query tiles (inner). **Recompute** each score tile
  `S = Q_i K_jᵀ · scale`, and recover the exact probabilities as `P = exp(S − lse)`. Because `lse` is the
  exact log-normalizer of each row, this gives exactly the softmax weights, with no need to know the
  maximum again.

The gradient formulas, derived in `docs/attention_math.md` §4, are:

```
D   = rowsum(dO ∘ O)            computed once, O(L·d)
dV += Pᵀ dO
dP  = dO Vᵀ
dS  = P ∘ (dP − D)
dQ += dS K · scale
dK += dSᵀ Q · scale
```

The trick is in `D`. The softmax Jacobian needs `Σ_k P_ik dP_ik` for each row, which looks as if it needs the
whole row of `P`. But `dP_ik = dO_i · V_k` and `O_i = Σ_k P_ik V_k`, so the sum equals `dO_i · O_i`, which can
be computed from `O` and `dO` alone. The same structure appears in FlashAttention-2's backward pass
(Dao, 2023).

**The cost of the trade:** one extra `Q Kᵀ` per tile, about **25 % more matmul FLOPs** in the backward pass
(10·L²·d instead of 8·L²·d). **The benefit:** saved memory falls from `O(L²)` to `O(L·d)`.

**Validation:** `torch.autograd.gradcheck` on a causal, padded, ragged-tile case
(`test_gradcheck_tiled_recompute_custom_backward`), plus comparisons with naive gradients across 49–57 cases
per dtype in the correctness laboratory. All pass.

> **What actually happened:** a measured surprise. The recompute version does *more* arithmetic, yet it
> was **faster** than the autograd version at every length from 512 tokens upward (980 ms vs 1,847 ms at
> L = 2048 on one thread), and faster than naive (1,411 ms). Allocating and keeping hundreds of MiB of tiles
> costs more than recomputing them. This is a CPU echo of FlashAttention's main argument: on modern hardware,
> moving memory is often more expensive than arithmetic (Chapter 14).

## 8.8 What this implementation is *not*

It is not FlashAttention. FlashAttention (Dao et al., 2022) is a **single fused GPU kernel** that keeps tiles
in on-chip SRAM and never writes scores to main memory. Our Python version runs each line as a separate PyTorch
operation, each of which reads and writes its operands through the cache hierarchy. It reproduces the
**algorithm and its memory behaviour**, not the **hardware efficiency**. Chapter 9 introduces the fused CPU
kernel that PyTorch provides, which does exploit the hardware, and Chapter 14 compares them.

## Key points to remember

* Online softmax keeps `m`, `l` and `acc` per row and rescales by `α = exp(m_old − m_new)` when the max grows.
* It is exactly equal to softmax in exact arithmetic, for any tile order or size.
* Worked example: scores [1,3,2,5] and values [10,20,30,40] give 36.880566 by both methods.
* Tiles are processed in `[B,H]` blocks; ragged tiles are fine; causal tiles above the diagonal are skipped.
* 16-bit inputs use fp32 accumulators.
* Autograd through the loop makes training memory `O(L²)` again (measured exponent 1.72).
* The recompute backward stores only O, lse (`O(L·d)`), recomputes `P = exp(S − lse)`, and uses `D = rowsum(dO∘O)`.
* Recompute does 25 % more backward FLOPs, yet measured faster than storing on this CPU.
