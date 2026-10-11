# Chapter 10 — The approximate methods: local window, block sparse and linear attention

> **What you will learn:** what each approximate method computes *exactly* (restricted softmax versus a
> different function); how each is implemented efficiently in PyTorch (the `unfold` trick, gathering active
> blocks, the associativity trick and chunked prefix sums); how each is verified against its *own* definition;
> and where each one's cost comes from.

## 10.1 Two kinds of "approximate"

* **Restricted softmax** (local window, block sparse): exactly the softmax attention of Chapter 4, but with
  many pairs forbidden, as if a fixed mask were applied. Inside the allowed pairs everything is exact. What is
  lost is *information*: the forbidden pairs.
* **Different function** (linear attention): replaces `exp(q·k/√d)` by another similarity. It is not softmax
  attention with an error; it is a different model.

The project verifies each method against its **own exact definition**. For restricted methods that is naive
attention with the corresponding mask. For linear attention it is a slow, direct evaluation of the same formula.
Their difference from full softmax attention is then measured separately and called **deviation**, never
"error".

## 10.2 Local-window (sliding-window) attention

**What it computes:** query `i` may attend to key `j` only if `i − w_left ≤ j ≤ i + w_right`. With
`window = w` non-causal, both sides are `w`; causal means `[i − w, i]`. This idea is used, for example, in
Longformer (Beltagy et al., 2020).

**The naive way would waste the savings:** build the full `L × L` table and mask most of it. That costs as much
as full attention.

**The project's way: chunks and `unfold`.**

> **In the code:** `src/attention_lab/algorithms/local_window.py`.

1. Split the queries into chunks of `C` rows. Chunk `c` covers queries `[cC, cC + C)`.
2. Those queries can only see keys in `[cC − w_l, cC + C + w_r)`, a **span** of `S = C + w_l + w_r` keys.
3. Pad K and V with `w_l` zeros in front and `w_r` (plus padding to a multiple of `C`) behind, then call
   `kp.unfold(2, S, C)`. That produces, *as a strided view*, all overlapping windows of `S` keys starting every
   `C` positions: a `[B, H, nC, S, d]` tensor.
4. Compute scores chunk by chunk, all at once: `[B, H, nC, C, d] @ [B, H, nC, d, S] → [B, H, nC, C, S]`.
5. Build the band mask for a chunk: query row `a` and window slot `b` are allowed iff `0 ≤ b − a ≤ w_l + w_r`.
   Also remove positions that fall in the padding (`key_pos < 0` or `≥ L`), and apply the padding mask after
   unfolding it in the same way.
6. Stable softmax, multiply by the unfolded values, reshape back to `[B, H, L, dv]`.

Fully vectorized, with no Python loop over positions. Scores evaluated: `⌈L/C⌉ · C · (C + w_l + w_r)`, which is
linear in `L`. About `C/(C + w_l + w_r)` of them are masked waste; the default `C ≈ w_l + w_r + 1` wastes
about half.

**Verified by:** `test_local_window_equals_band_masked_softmax`, a grid of lengths (1, 5, 37, 64), windows
(0, 1, 4, 16, 100), causal/non-causal and chunk sizes, comparing output *and gradients* with naive attention
under the band mask, to 1e-12. A window as large as the sequence equals full attention.

## 10.3 Block-sparse attention

**What it computes:** cut the sequence into blocks of `b` tokens. A boolean **layout** `layout[qb, kb]` says
whether query block `qb` may attend key block `kb`. The project's layout, in the spirit of BigBird (Zaheer et
al., 2020), combines:

* **local**: `|qb − kb| ≤ local_blocks` (causal: `0 ≤ qb − kb ≤ local_blocks`);
* **global**: the first `global_blocks` key blocks are visible to everyone, and (non-causal) the first query
  blocks see everything;
* **random**: each query block also sees `random_blocks` extra key blocks, chosen with a fixed seed so the
  layout is reproducible.

> **In the code:** `src/attention_lab/algorithms/block_sparse.py` — `make_layout`, `block_sparse_attention`.

**Execution by gathering.** For each query block, collect the indices of its active key blocks, pad every row
to the widest row (`K_max`), and **gather** those key and value blocks into a dense
`[B, H, n_rows, K_max·b, d]` tensor. Then compute exact softmax attention over just those keys:
`[B, H, n_rows, b, K_max·b]` scores instead of `[B, H, L, L]`. Padded slots, padded positions (when `L` is not a
multiple of `b`), padding-mask positions and (causal) pairs above the diagonal inside diagonal blocks are all
masked.

**Global rows are handled separately.** A query block that sees *every* block would force `K_max = n_blocks` for
all rows and make the whole computation dense. So rows with full coverage are computed against all keys
directly, and the rest use the gather path.

**Verified by:** `test_block_sparse_equals_layout_masked_softmax` (lengths including ragged 77 and 130, five
layout configurations, causal/non-causal, gradients) against naive attention with the layout expanded to an
element mask. Also tested: the layout is causal when asked, always includes the diagonal and the global column,
and is reproducible from its seed.

> **What actually happened:** at short lengths a "sparse" layout can cover everything. With `b = 8` at
> `L = 32` there are only 4 blocks, and local ±1 + global block 0 + 1 random block already includes all of them.
> The layout keeps 528 of 528 causal pairs, which is full attention. This was discovered when block_sparse scored
> exactly like exact attention on the induction task. The book and the results flag those cells as "not evidence
> about sparsity" (Chapter 18). Likewise, the deviation measurements show 0 at L = 256 with b = 64.

## 10.4 Linear attention (elu + 1)

**What it computes** (Katharopoulos et al., 2020): pick a feature map `φ(x) = elu(x) + 1` (applied to each entry;
always positive), and define

```
out_i = Σ_j [φ(q_i)·φ(k_j)] v_j  /  Σ_j φ(q_i)·φ(k_j)
```

It looks like attention with weights `φ(q)·φ(k)` instead of `exp(q·k/√d)`. **It is not an approximation of
softmax**: `elu + 1` is not designed to estimate `exp` in any sense. (Random-feature methods such as Performer do
estimate it, but they are not implemented here.) Following the paper, `φ` is applied to `q` and `k` directly,
with no `1/√d`, and passing a `scale` raises an error.

**Why it can be linear in L: associativity.**
`Σ_j (φ(q_i)ᵀ φ(k_j)) v_jᵀ = φ(q_i)ᵀ (Σ_j φ(k_j) v_jᵀ)`. The bracket is a `d × d` matrix that does not depend
on `i`. Compute it once (`O(L·d²)`) and reuse it for every query (`O(L·d²)`). No `L × L` table at all.

**Causal is harder.** Query `i` needs the prefix sum `S_i = Σ_{j≤i} φ(k_j) v_jᵀ`, a different `d × d` matrix for
every position. Storing all of them would cost `O(L·d²)` memory. The project uses a **chunked** form:

* inside each chunk of `C` tokens, compute the small `C × C` causal attention directly (quadratic, but only in `C`);
* across chunks, keep one `d × d` summary per chunk, take **exclusive prefix sums** (each chunk sees only earlier
  chunks), and apply them to the chunk's queries.

The exclusive prefix is built as "zeros, then the cumulative sum shifted by one", rather than
"cumulative sum minus own chunk", to avoid cancellation error. Padded or masked keys get `φ(k) = 0`, so they
contribute nothing. The denominator gets `+ 1e-6` for safety (it is positive anyway, because `φ > 0`).

> **In the code:** `src/attention_lab/algorithms/linear.py` — `linear_attention` and
> `linear_attention_quadratic_reference`.

**Verified by:** `test_linear_chunked_equals_quadratic_definition` (lengths 1–100, chunks 1–128, causal and padded,
gradients) against the slow direct formula. Two more tests show it is *not* softmax attention, and that its outputs
are convex combinations of values.

## 10.5 What each method costs, in theory

From `complexity/theory.py`, for `B = 1, H = 4, L = 4096, d = 64` (theory, not measurement):

| Method (default params) | Logits evaluated | Matmul GFLOP | Forward allocations |
|---|---|---|---|
| exact methods | 67.1 million | 17.2 | naive 772 MiB; flash 4.1 MiB; tiled 9.2 MiB |
| local_window (w = 64) | 4.2 million | 1.09 | 68.6 MiB |
| block_sparse (b = 32, 1 local, 1 global, 1 random) | 3.1 million | 0.80 | 79.8 MiB |
| linear_elu (non-causal) | 0 | 0.27 | 12.1 MiB |

Note that the approximate methods' *theoretical* allocations are already much larger than the fused exact kernel's,
because they materialize their restricted score tables or gathered blocks. Chapter 15 confirms this by
measurement.

## Key points to remember

* Local and block-sparse attention are **exact softmax on a restricted set of pairs**. Linear attention is a **different function**.
* Each is verified against its *own* definition (masked naive, or the direct formula), including gradients.
* Local window: chunks + `unfold` + band mask, `O(L·(C + 2w))` scores, about half of them masked waste.
* Block sparse: layout (local + global + random), gather active blocks, pad to `K_max`, global rows done densely.
* At short lengths a sparse layout may be dense (L = 32, b = 8 covers everything).
* Linear: associativity gives `O(L·d²)`; causal uses chunked exclusive prefix sums of `d × d` states.
