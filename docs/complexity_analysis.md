# Complexity analysis

All formulas here are **theory**. The code version is `complexity/theory.py`. Benchmarks
store its outputs under `theory_*` fields, next to measured values but never
mixed with them. The instrumented work counters (`stats=` arguments) are checked
against these formulas by `tests/scientific/test_complexity_and_measurement.py`.

Per head (multiply by `B·H`), self-attention with `L = Lq = Lk`, head dim `d`
(`dv = d`), `b` bytes per element. A multiply-add counts as 2 FLOPs.

## Six separate questions

| Quantity | What it measures | Where |
|---|---|---|
| Asymptotic time | arithmetic, as a function of L | this document |
| Asymptotic auxiliary memory | intermediates of one forward call, beyond inputs and outputs | this document |
| Total training memory | everything held between forward and backward, plus parameters, gradients and optimizer state | §7 |
| Practical wall-clock | measured median time on this CPU | benchmark reports |
| Memory traffic | bytes moved between cache levels and DRAM; hardware-dependent | `systems_guide.md` |
| Numerical accuracy | rounding error relative to a float64 reference | `numerical_guide.md` |

An algorithm can win on one row and lose on another. Most of the interesting
results in this lab are exactly such disagreements.

## 1. Dense exact attention (naive, sdpa_math)

* Scores `S = QKᵀ`: `L²` dot products of length `d` → `2L²d` FLOPs.
* `P V`: another `2L²d` FLOPs. Total **`4L²d` matmul FLOPs**, plus `O(L²)`
  elementwise operations (scale, mask, exp, sum, divide).
* Auxiliary memory: the naive code holds `s`, `e = exp(s − m)` and `p` together
  → about **`3L²b` bytes** at its peak (plus an `L²`-byte bool mask when causal).
* Time `O(L²d)`, auxiliary memory `O(L²)`.

The quadratic factor is fundamental to *exact* attention: every output depends on
every (query, key) pair. Any exact algorithm must evaluate all `L²` logits
(or `L(L+1)/2` with a causal mask). Algorithms can change **where those logits
live**, not how many there are.

## 2. Tiled exact attention (tiled, tiled_recompute; also sdpa_flash_cpu)

* Logits evaluated: still `L²` (non-causal). With causal tile skipping, the
  count is `Σ_i |{tiles intersecting j ≤ i}|·tile area`, which approaches `L²/2`
  as tiles shrink. For example, `L=64`, 16×16 tiles: 10 of 16 tiles.
* Matmul FLOPs: `4L²d`, the same as dense. Extra elementwise work: the `α`
  rescaling of `l` and `acc` once per tile, i.e. `O(L·d·L/bk)` extra multiplies.
* Auxiliary memory per step: one `bq×bk` score tile, its `P`, and temporaries →
  **`O(bq·bk)`**, plus the running `m, l` (`O(bq)`) and `acc` (`O(bq·d)`).
  Across the whole call: output blocks `O(L·d)` and `lse` `O(L)`.
* Time `O(L²d)`, auxiliary memory `O(bq·bk + L·d)`.

**Avoiding the `L×L` matrix does not remove the `L²` work.** It changes the
*storage* from `O(L²)` to `O(tile²)`. On a GPU, that is what turns a
memory-bandwidth-bound computation into a compute-bound one: FlashAttention's
analysis gives `Θ(N²d²/M)` HBM accesses vs `Θ(Nd + N²)` for standard attention
(Dao et al., 2022, Thm. 2; `M` = SRAM size). The asymptotic arithmetic is unchanged.

**Python-loop overhead** is not in the asymptotic formulas but dominates small
tiles here. The loop issues `≈ 10` PyTorch ops per tile, each costing
microseconds of dispatch, and there are `(L/bq)(L/bk)` tiles. The number of tile
iterations is recorded as `theory_python_tile_iterations`.

## 3. Training (backward) for the exact methods

| | saved between fwd and bwd | backward matmul FLOPs |
|---|---|---|
| naive / sdpa_math | `P` and `e` (`O(L²)`) | `dV = PᵀdO`, `dP = dO Vᵀ`, `dQ`, `dK`: `4 × 2L²d = 8L²d` |
| tiled (autograd through loop) | every tile's `P`/`e` → **`O(L²)`** in total | same `8L²d` |
| tiled_recompute / FlashAttention-style | `Q, K, V, O, lse` → **`O(L·d)`** | `8L²d` + recomputing `S` (`2L²d`) = `10L²d` |

So the recomputation backward does **25 % more matmul work** in the backward pass
to reduce saved memory from quadratic to linear. Whether that is "faster" depends
entirely on whether the machine is limited by memory traffic or by arithmetic.

## 4. Local-window attention (local_window)

With half-window `w` (non-causal: `|i − j| ≤ w`; causal: `0 ≤ i − j ≤ w`):

* Useful pairs: `≈ L(2w + 1)` non-causal, `≈ L(w + 1)` causal (fewer at the edges;
  exact count in `band_entry_count`).
* The implementation evaluates query chunks of `C` rows against a span of
  `C + w_l + w_r` keys, so **logits evaluated `= ⌈L/C⌉·C·(C + w_l + w_r)`**.
  Wasted (masked) fraction `≈ C/(C + w_l + w_r)`. The default `C ≈ w_l + w_r + 1`
  wastes about half.
* Time `O(L(w + C)d)`, auxiliary memory `O(L(w + C))` — **linear in `L`** for fixed `w`.
* What is lost: one layer cannot see beyond `w` tokens. Stacking `n` layers gives a
  receptive field of `n·w`, but information must be relayed through intermediate tokens.

## 5. Block-sparse attention (block_sparse)

With block size `b`, `n_B = ⌈L/b⌉` blocks, and a layout with `k_r` active key
blocks for query-block row `r`:

* Logits evaluated by our gather implementation: `Σ_rows b·K_max·b` for sparse rows,
  where `K_max = max_r k_r` (rows are padded to the widest row), plus `b·n_B·b` for
  each *global* row (rows that attend to everything).
* For a local+global+random layout, `k_r ≈ (2·local + 1) + global + random`, a
  constant. Work is therefore `O(L·k·b·d)` — **linear in L** — plus
  `O(global·b·L·d)` for the global rows, which is also linear in L.
* **Distribution matters**: because rows are padded to `K_max`, one row with many
  active blocks raises the cost of every row. Our implementation splits off "dense"
  rows for exactly this reason. Random blocks add reach, but not to a *chosen*
  token.
* Theoretical density = active blocks / `n_B²`, reported as `theory_layout_density`.
  It falls like `1/n_B`.

## 6. Linear attention (linear_elu)

Replace `exp(q·k/√d)` by `φ(q)·φ(k)` with `φ = elu + 1`
(Katharopoulos et al., 2020). Associativity gives

```
out_i = φ(q_i)ᵀ [Σ_j φ(k_j) v_jᵀ] / φ(q_i)ᵀ [Σ_j φ(k_j)]
```

* Non-causal: computing `Σ_j φ(k_j)v_jᵀ` (a `d×d` matrix) costs `2Ld²`, and applying
  it to all queries costs `2Ld²`. Time **`O(Ld²)`**, auxiliary memory `O(d²)` per head.
* Causal: needs prefix states `S_i = Σ_{j≤i} φ(k_j)v_jᵀ`. Storing all of them costs
  `O(Ld²)` memory. Our chunked form uses chunks of `C`: quadratic inside a chunk
  (`O(LCd)`) and prefix-summed `d×d` states across chunks (`O((L/C)d²)` memory,
  `O(Ld²)` time). Time **`O(LCd + Ld²)`**, auxiliary memory `O(LC + (L/C)d²)`.
* **When does this beat softmax attention?** When `L ≫ d`. The crossover in
  arithmetic is roughly `4L²d` vs `4Ld²`, i.e. `L > d`. For `d = 64` this is
  satisfied at every length benchmarked here. Constants (elementwise feature map,
  extra reductions) and kernel efficiency decide the measured crossover.
* The scaling changes **only because the kernel is factorizable**. Softmax's
  `exp(q·k)` is not factorizable into finite-dimensional features. Random-feature
  methods such as Performer/FAVOR+ (Choromanski et al., 2020) approximate it
  *in expectation*. `elu+1` does not approximate it at all.

## 7. Total training memory of a model (beyond attention scores)

For one Transformer layer with width `D = H·d`, MLP width `4D`, batch `B`:

| Item | Size |
|---|---|
| Parameters | `≈ 12D²` per layer |
| Gradients | same as parameters |
| AdamW state | `2×` parameters |
| Saved activations (non-attention) | `O(B·L·D)` per layer: inputs to the QKV/output projections, MLP hidden (`4D`), LayerNorm statistics |
| Saved attention activations | `O(B·H·L²)` for materializing / autograd-tiled methods; `O(B·H·L)` for recompute methods |

At the tiny model sizes trained here (`D = 64`, 2 layers, ~100k parameters), the
parameter and optimizer terms are negligible. At `L = 64` the attention term
`B·H·L² = 64·4·4096 ≈ 1M` floats per layer is comparable to the `B·L·D` terms.
This is why the training experiments are not memory-bound.

## 8. Inference vs training

* **Inference, no KV cache**: forward only. Tiled methods need `O(tile²)`
  auxiliary memory, while the naive method needs `O(L²)`.
* **Autoregressive decoding with a KV cache**: each new token attends to `L`
  cached keys, which is `O(L·d)` per token and `O(L²·d)` over a whole generation.
  The `L×L` matrix is never formed, even naively, because `Lq = 1`. This lab
  benchmarks full-sequence calls only.
* **Training**: §3 applies. The backward pass is where store-vs-recompute matters.
