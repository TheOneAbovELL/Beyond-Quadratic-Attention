# Online softmax and tiled exact attention

This derives the update rule in `algorithms/tiled_exact.py::online_softmax_update`
and proves that tiled attention computes *exactly* softmax(S)V (in exact
arithmetic). The variable names match the code.

## 1. Problem

For one query row we have logits `s_1, …, s_N` (already scaled and masked) and
value rows `v_1, …, v_N`. We want

```
o = Σ_j e^{s_j} v_j / Σ_j e^{s_j}
```

without ever holding all `N` logits at once. The keys are processed in tiles
`T_1, T_2, …` of `kv_tile` keys each. A whole *block* of `q_tile` query rows is
processed at once, but each row is independent.

## 2. Invariant

After processing tiles `T_1 … T_t`, let `U_t = T_1 ∪ … ∪ T_t` and define

```
m_t   = max_{j ∈ U_t} s_j                         (running maximum)
l_t   = Σ_{j ∈ U_t} e^{s_j − m_t}                 (running denominator, relative to m_t)
acc_t = Σ_{j ∈ U_t} e^{s_j − m_t} v_j             (running un-normalized output)
```

It holds trivially for `t = 0` with `m_0 = −∞`, `l_0 = 0` and `acc_0 = 0` (empty sums).

## 3. Update (equations U1–U5 in the code)

For the next tile `T` with logits `S` (a `[q_tile, kv_tile]` block in the code):

```
U1   m_new = max(m_t, max_{j∈T} s_j)
U2   α     = e^{m_t − m_new}
U3   P_j   = e^{s_j − m_new}            for j ∈ T
U4   l_new = α · l_t + Σ_{j∈T} P_j
U5   acc_new = α · acc_t + Σ_{j∈T} P_j v_j        (a matmul: P @ V_tile)
```

**Proof that the invariant is preserved.** Look at the old terms:
`α · l_t = Σ_{j∈U_t} e^{s_j − m_t} · e^{m_t − m_new} = Σ_{j∈U_t} e^{s_j − m_new}`.
U4 adds the new terms `Σ_{j∈T} e^{s_j − m_new}`. The result is
`Σ_{j∈U_{t+1}} e^{s_j − m_new}`, which is the definition of `l_{t+1}`. The same
argument with an extra factor `v_j` proves U5. ∎

## 4. Finalization

After the last tile, `U = all keys` and `m = max_j s_j`:

```
o   = acc / l = Σ_j e^{s_j − m} v_j / Σ_j e^{s_j − m} = Σ_j e^{s_j} v_j / Σ_j e^{s_j}
lse = m + log l = log Σ_j e^{s_j}
```

The common factor `e^{−m}` cancels, so `o` is the softmax output exactly. The order
of the tiles and the tile sizes are irrelevant mathematically. They only change the
floating-point rounding. Test: `tests/unit/test_online_softmax.py::test_streaming_equals_softmax_for_any_split`
streams the same row with five different splits.

## 5. Numerical stability

* Every exponent is `≤ 0`, because `s_j ≤ m_new` and `m_t ≤ m_new`. So `exp` never
  overflows. That holds even when a later tile has logits 1000 larger than an earlier
  one: `α = e^{−1000} = 0` cleanly discards the old, negligible mass. Test:
  `test_update_handles_increasing_maxima_without_overflow`.
* After the first tile that contains an unmasked key, `l ≥ 1` (the maximal element
  contributes `e^0`). The final division is therefore well-conditioned.
* **Masked entries** (`s = −∞`) give `P = 0`. If *every* key seen so far is masked,
  then `m = −∞` and `s − m` would be `−∞ − (−∞) = NaN`. The code shifts by
  `m_shift = 0` whenever `m_new = −∞`. Then `α = e^{−∞} = 0` and `P = e^{−∞} = 0`,
  which keeps the state at `(−∞, 0, 0)`. Test: `test_update_handles_all_masked_first_tile`.
* A row that is fully masked ends with `l = 0`. Its output is defined to be 0
  (project policy, see `attention_math.md` §3), and `lse = −∞`.
* For 16-bit inputs, `m`, `l` and `acc` are kept in **float32**
  (`_acc_dtype`), as in FlashAttention. The correctness lab shows this choice
  matters more than the algorithm: computing the naive formula entirely in bfloat16
  loses up to ~15 % (normalized) at large logit magnitudes, while the tiled kernel
  with fp32 statistics stays at bf16 output-rounding level (`numerical_guide.md`).

## 6. Why the result is not bitwise identical to the naive one

Floating-point addition is not associative. Tiling changes:

1. the **summation order** of `l` and of `acc` (per tile, then combined);
2. the **number of roundings**: each rescale `α·l`, `α·acc` is an extra rounded
   multiply. A key seen in tile 1 has its contribution rescaled once per later
   tile whose max increases.

Both effects are `O(u)` per operation (`u` = unit roundoff). Measured float32
differences are ~1e-6 relative (correctness report). They are *numerically
equivalent* within tolerance, mathematically equivalent, and **not bitwise equal**
(`test_numerical_vs_bitwise_equality_are_distinct_notions`).

## 7. Causal tiles

With `causal=True`, a tile whose first key index exceeds the last query index of
the block (`ks > qe − 1`) contains only masked pairs. The forward loop `break`s
there, because every later tile is further right. The backward `continue`s. For
`L = 64` and 16×16 tiles, 10 of the 16 tiles are computed
(`test_causal_skips_tiles_above_diagonal`), which approaches ½ for many tiles. The
diagonal tiles still apply the element-level `j ≤ i` mask.

## 8. Backward pass: store vs recompute

`tiled_attention` lets autograd differentiate through the Python loop. Autograd
must keep every tile's `P` (and `exp` outputs) until backward. That is the same
`O(L²)` storage the tiling avoided in the forward pass. **Tiling alone does not make
training memory sub-quadratic.**

`tiled_attention_recompute` is a custom `autograd.Function` in the style of the
FlashAttention backward (Dao et al., 2022; Dao, 2023):

* the forward saves only `Q, K, V, O` and the per-row `lse` (`O(L·d + L)`);
* the backward loops over KV tiles (outer) and query tiles (inner), and *recomputes*
  each `P` tile exactly as `P = exp(S − lse)`. Because `lse` is the exact
  log-normalizer, this yields exactly the softmax probabilities;
* it computes `D = rowsum(dO ∘ O)` once, then per tile:
  `dV += Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ∘ (dP − D)`, `dQ += τ dS K`, `dK += τ dSᵀ Q`.

Cost: one extra `Q Kᵀ` per tile (`≈ 2·L²·d` more FLOPs per head), in exchange for
`O(L²)` → `O(L·d)` saved memory. FlashAttention calls this trade "selective gradient
checkpointing" (recompute instead of store). It pays off on GPUs because reading a
stored `P` from HBM is slower than recomputing it from on-chip `Q, K` tiles. On this
CPU, with a Python loop, the trade is measured rather than assumed (see the results).

## 9. Relation to the literature (verified citations, see `references.md`)

* Milakov & Gimelshein (2018) compute the softmax *normalizer* in a single pass with
  a running max and a rescaled running sum — U1, U2, U4 without the value accumulator.
* Rabe & Staats (2021) apply chunking to attention to obtain sub-quadratic memory,
  with checkpointing for the backward pass.
* FlashAttention (Dao et al., 2022) adds the value accumulator (U5) and fuses the
  whole loop into one GPU kernel that keeps tiles in on-chip SRAM. It stores
  `(m, l)` for backward and recomputes `S, P`. Its IO analysis gives `Θ(N²d²/M)` HBM
  accesses vs `Θ(Nd + N²)` for standard attention (`M` = SRAM size).
* FlashAttention-2 (Dao, 2023) keeps `acc` un-normalized until the end (as we do),
  stores only `lse = m + log l`, and parallelizes over the sequence dimension too.

Our implementation follows the algorithm. It does **not** reproduce the
hardware-level part (on-chip SRAM residency, kernel fusion, warp partitioning).
In Python on a CPU, every line is a separate PyTorch op that reads and writes its
operands through the cache hierarchy (`systems_guide.md`).
