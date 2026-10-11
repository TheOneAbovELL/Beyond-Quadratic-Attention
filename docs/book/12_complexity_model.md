# Chapter 12 — The complexity model: theory you can test

> **What you will learn:** how the project turns big-O statements into concrete numbers for each method and
> configuration; why those numbers are always labeled "theory"; how the theory was *checked* against counters
> built into the algorithms; and how the memory guard uses it.

## 12.1 Purpose

The brief insisted that **theory and measurement must stay distinguishable**. The project therefore has one
module that produces only theoretical estimates, and every estimate it produces is stored under a name
beginning with `theory_`. A measured value never carries that prefix, and a theoretical value never appears
without it.

> **In the code:** `src/attention_lab/complexity/theory.py`, function `theoretical_cost(method, shape, params)`.

## 12.2 What it computes

For a `Shape(B, H, L, d, dtype, causal)` and a method's parameters, it returns (among others):

* `score_entries`: how many query–key logits the implementation **actually evaluates**, including
  masked-but-computed entries (local window wastes about half; padded block slots also count);
* `matmul_flops`: `2 × 2 × entries × d` for softmax methods (`QKᵀ` plus `PV`);
* `elementwise_ops`: rough counts of exp, max, divide (reported separately, because they cost very differently
  from matmul FLOPs on a CPU);
* `aux_fwd_bytes`: peak intermediate storage of one inference forward, excluding inputs and outputs;
* `fwd_alloc_bytes` = `aux_fwd_bytes` + the output tensor, comparable with the measured tensor peak;
* `saved_for_bwd_bytes`: what autograd keeps between forward and backward;
* `asymptotic_time`, `asymptotic_aux_memory`: the big-O strings.

Method-specific details:

* **tiled**: counts tiles *exactly*, loop by loop, including ragged final tiles and causal tile skipping.
  Autograd-tiled saves all tiles; recompute saves only `lse`.
* **local_window**: `⌈L/C⌉ · C · (C + w_l + w_r)` scores, and `band_pairs` (the useful ones) counted exactly.
* **block_sparse**: builds the actual layout with `make_layout`, then counts `rows × b × K_max × b` plus dense
  global rows, the layout density and the number of active blocks.
* **linear_elu**: zero scores for non-causal; for causal, `C × C` per chunk plus the `d × d` states.

## 12.3 Checking theory against the code

A model is only useful if it is right, so the algorithms can report what they actually did. They accept an
optional `stats={}` dictionary and fill in counters as they run (`score_entries`, `tiles`, `band_entries`,
`active_blocks`, `k_max`).

Scientific tests (`tests/scientific/test_complexity_and_measurement.py`) then assert **exact equality**
between the theory and the counters:

| Test | Checks |
|---|---|
| `test_tiled_score_entries_theory_equals_instrumented` | 8 cases incl. ragged tiles (L = 70 with 16 × 24 tiles; L = 100 with 32 × 7) and causal |
| `test_local_window_entries_theory_equals_instrumented_and_scale_linearly` | entries and band pairs; 4× longer input ≤ 4.4× entries |
| `test_block_sparse_entries_theory_equals_instrumented` | entries and active blocks, incl. ragged L = 130 causal |
| `test_naive_entries_are_quadratic` | doubling L quadruples entries |
| `test_linear_noncausal_has_no_quadratic_term` | 0 score entries; doubling L doubles FLOPs |

> **What actually happened:** the first version of the tiled formula counted `tiles × q_tile × kv_tile`, which
> overcounts when the last tile is ragged. It was rewritten to loop over the real tile boundaries so that it
> equals the counter exactly. Separately, local_window's `band_entries` statistic was first computed by building
> a dense `L × L` band mask, which is `O(L²)` work inside a supposedly `O(L)` method. It was replaced by an
> `O(L)` formula (`band_entry_count`) before any benchmark used it.

## 12.4 Checking theory against measured memory

Chapter 14 compares `theory_fwd_alloc_bytes` with the measured tensor peak. For naive attention the model
predicted 0.875, 3.25, 12.5, 49.0 and 194 MiB at L = 128, 256, 512, 1024 and 2048. The measurements were 0.88,
3.26, 12.52, 49.05 and 194.09 MiB. For methods whose internals are a black box (the fused kernel's per-thread
scratch), the model can only give a lower bound, and it says so.

## 12.5 Use in the memory guard

Before running any configuration, the benchmark runner asks the model for an `estimated_peak_bytes` for each
mode (forward, forward+backward) and refuses modes that would exceed `min(50 % of currently available RAM,
max_mem_gb)`. The estimate deliberately overcounts (backward is assumed to need another set of `L²`
temporaries). Refused configurations are **recorded** with the reason, never silently dropped (Chapter 13).

## Key points to remember

* `theory_*` fields are estimates, always labeled; measured fields never carry that prefix.
* The model counts the scores each implementation *actually* computes, including masked waste and padding.
* Algorithms expose `stats=` counters; tests assert theory == counters exactly.
* The naive memory model matched measurement to within 0.1 MiB.
* The memory guard uses the model to refuse (and record) configurations that could exhaust RAM.
