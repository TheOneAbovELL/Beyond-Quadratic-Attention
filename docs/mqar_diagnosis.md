# MQAR plateau diagnosis

**Status: diagnosed to the level needed by this project. The exact attention implementations are not
the cause. Learning content-based retrieval is the bottleneck for this model, task and CPU budget.**

Logs: `experiments/diagnostics/mqar_v2/*.jsonl` (first line = configuration). Instruments:
`src/attention_lab/diagnostics/recall.py`. Runner: `scripts/diagnose_recall.py`. Summaries:
`python scripts/summarize_diag.py <log>`. Each diagnostic was run alone, single-threaded, on a CPU that
other workloads kept saturated.

## 1. Symptom

Tiny Transformers (2 layers, d_model 64, RoPE) trained on MQAR settled exactly at a **binding-free
baseline** for every backend that can see the key-value pairs:

* **Generator v1** (values drawn with replacement): the plateau was 25–26 %. Predicting the most frequent
  value in the context scores 25.7 % on the same evaluation set. That is a *shortcut in the task
  generator*, which was fixed in v2 (distinct values; `shortcut_baseline` = 1/n_pairs; regression test
  `test_mqar_v2_values_are_distinct_so_bag_of_values_shortcut_is_one_over_pairs`).
* **Generator v2**: loss sits at log(n_pairs) and accuracy at 1/n_pairs. Examples:
  4 pairs → 1.386 / 25 %, unchanged after 3000 steps; 8 pairs → 2.08 / 12.5 % at 2500 steps. This is
  *uniform guessing among the values present in the context*. It held for lr 3e-3 and 1e-2, with and
  without warmup, cosine decay and gradient clipping (the plain-loop run also stayed at 1.39 through
  step 1750).
* A model trained on "plain induction" (second half = exact repeat of the first) reached 100 %. That
  appeared to show the architecture could learn the needed circuit. §3 shows that it did not.

## 2. Pipeline checks (implementation bugs ruled out)

`tests/unit/test_mqar_pipeline.py` (10 tests, all pass) checks each link:

| Suspected failure | Test | Result |
|---|---|---|
| wrong token classes per region | `test_generator_token_classes_by_region` | pass |
| duplicate keys/values or unbound query keys | `test_generator_keys_and_values_distinct_and_each_query_key_bound_once` | pass |
| answer derivable without binding / leaked | `test_answer_is_not_derivable_without_binding` | pass: each answer occurs exactly once, right after its key |
| biased query/target marginals | `test_query_marginals_are_uniform_over_pairs_and_values` | pass |
| loss at wrong positions or shifted targets | `test_loss_is_computed_only_at_query_positions_and_aligned_unshifted` | pass: only the query positions are scored, aligned p → targets[p] |
| plateau value | `test_uniform_over_context_values_gives_log_n_pairs_loss` | uniform over the 4 context values gives exactly log 4 = the observed plateau |
| key or pairs invisible to the query (mask bug), future leakage | `test_query_position_can_see_its_key_token_and_all_pairs` | pass |
| dead parameters / broken gradient flow | `test_every_parameter_receives_a_nonzero_gradient_on_mqar` (RoPE and sinusoidal) | pass |
| metric bug | `test_metric_oracle_and_shortcut` | oracle = 1.0, shortcut = 0.25 |

The attention backends used for training are verified numerically equivalent to the reference in
forward and backward (`docs/results.md` §1; `test_exact_backends_preserve_model_outputs`). Training
with any exact backend therefore computes the same function to ~1e-6, so **the plateau cannot come
from the exact-attention implementations**. The diagnostic instruments are validated as well:
`tests/unit/test_diagnostics.py` checks that the replayed attention maps reproduce each layer's output
to 1e-5, and that the circuit probes read the intended positions.

## 3. Diagnostic experiments (one at a time)

### D1 — fixed-dataset overfit (`overfit_pairs4_L32_lr0.003_ntrain64.jsonl`)

64 fixed MQAR sequences (4 pairs, L = 32), full-batch AdamW, lr 3e-3, 2000 steps.

| step | train loss | train acc | held-out acc | held-out loss | grad norm | prev-token attn (L0) | attn on correct value (L1) | attn on other 3 values (L1) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.92 | 0.14 | 0.08 | 4.03 | 2.68 | 0.27 | 0.06 | 0.17 |
| 100 | 0.007 | **1.00** | 0.20 | 5.37 | 0.018 | 0.37 | 0.14 | 0.41 |
| 2000 | 8.5e-5 | **1.00** | **0.20** | 7.84 | 2.5e-4 | 0.36 | 0.13 | 0.39 |

**Interpretation.** The model fits the training set perfectly, so the forward pass, backward pass,
loss and optimizer can drive MQAR loss to zero. It does so by **memorization**: held-out accuracy is
below the 25 % shortcut, and held-out loss rises. The binding circuit never forms. Layer-0
previous-token attention stays near its initial value. Layer-1 weight on the correct value (0.13)
equals the weight on each of the other three values (0.39 / 3).

### D2 — is "plain induction succeeds" evidence of content-based retrieval? (`induction_trainfixedgap_lr0.003.jsonl`)

Trained on [segment][same segment] (fixed offset 12), evaluated on fixed-offset and random-gap
(gap 0–8 noise tokens) repeats:

| step | train loss | acc, fixed offset | acc, random gap |
|---|---|---|---|
| 100 | 0.244 | **1.00** | 0.18 |
| 2000 | 1.5e-4 | **1.00** | **0.18** |

**Interpretation.** The 100 % was obtained with a **positional shortcut**: with a fixed offset, a RoPE
head can copy from "12 positions back" without matching any content. When the offset varies, accuracy
collapses to 18 %. The earlier sanity check therefore did not show that the model learns content-based
induction, and the conclusion drawn from it is withdrawn.

### D3 — content-based induction (`induction_trainrandomgap_lr0.003.jsonl`)

Same model, loop and budget, trained on repeats with a random 0–8 token gap (offset 12–20), so the copy
source must be found by content:

| step | train loss | held-out acc | acc fixed offset | acc random gap | grad norm L0 attn |
|---|---|---|---|---|---|
| 100 | 3.25 | 0.09 | 0.08 | 0.09 | 0.30 |
| 200 | 2.64 | 0.24 | 0.19 | 0.22 | **0.75** |
| 500 | 0.006 | **0.999** | 1.00 | 0.999 | 0.52 |
| 1000 | 6e-4 | **1.00** | 1.00 | 1.00 | 0.003 |

**Interpretation.** The architecture and training loop *do* learn content-based retrieval, and
quickly (about 500 steps). The rise in layer-0 attention gradient at step 200, just before accuracy
jumps, is consistent with a previous-token / induction circuit forming. The circuit itself was not
probed here, because the probes are defined for the MQAR layout.

## 4. Conclusion

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Bug in the exact-attention implementations | **rejected** | numerical equivalence in fwd and bwd (`results.md` §1); identical model outputs across exact backends |
| Bug in generator, targets, loss positions, masks, metric or gradient flow | **rejected** (v2) | §2, 10 targeted tests |
| v1 generator shortcut | **confirmed and fixed** | shortcut baseline 25.7 % = the observed v1 plateau; v2 regression test |
| Model cannot fit MQAR data at all | **rejected** | D1: 100 % train accuracy, loss 8.5e-5 |
| Architecture cannot learn content-based retrieval | **rejected** | D3: 100 % on random-offset induction |
| "Plain induction works" proves content-based retrieval | **rejected** | D2: positional shortcut, 18 % when the offset varies |
| Learning-rate / loop configuration is the cause | **not supported** | plateau identical with lr 3e-3 and 1e-2, and with the plain loop (const lr, no clip/warmup); not exhaustively swept |
| Insufficient training budget | **open** | no run beyond 3000 steps; Zoology-style MQAR training uses far more examples than this CPU budget allows |
| MQAR-specific optimization difficulty (sparser supervision: 4 vs 11 targets per sequence; the "uniform over context values" solution, loss log n_pairs, as an intermediate attractor) | **plausible, untested** | plateau value equals that solution exactly; D1 shows binding is not found even with zero training loss available |

**Decision for the project.** The plateau says nothing about the correctness or efficiency of the
attention implementations, so resolving it is not required for this project's claims. The trained
quality comparison therefore uses **variable-offset induction** (`tasks/synthetic.py::induction`),
which the exact model demonstrably learns and which requires retrieving information 12–20 tokens
back, beyond the local window used in the comparison. MQAR remains available (`--which M`) and is
reported only with its baseline-relative status.

**Next experiments, if MQAR binding is ever needed:** (1) n_queries > n_pairs (denser supervision) at
fixed steps; (2) a curriculum from 1 → 2 → 4 pairs; (3) a longer budget with the circuit probes, to see
whether `prevtok_l0` rises before accuracy does. Run them one at a time with `scripts/diagnose_recall.py`.
