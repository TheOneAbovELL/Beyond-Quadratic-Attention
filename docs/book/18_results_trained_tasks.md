# Chapter 18 — Results IV: what models trained with each attention method can do

> **What you will learn:** the results of training the same small Transformer with exact, local-window,
> block-sparse and linear attention on the local-copy and variable-offset induction tasks; the "swap"
> experiments; and the analyses that turned surprising numbers into understood ones (relay through layers,
> a layout that was secretly dense, and distractors that are not explained by ambiguity).

Run: `20261011-205830_tasks` (`python lab.py tasks --which AB --steps 1500 --threads 1 --batch-size 32`). Single
seed. Backends: exact (`sdpa_flash_cpu`), `local_window` (w = 8), `block_sparse` (block 8, ±1 local, global block 0,
1 random), `linear_elu` (chunk 16). **Swap** = the model trained with exact attention, evaluated with another
backend substituted, no retraining.

## 18.1 Task A: local copy (lag 3), 500 steps, trained at L = 64

| backend | trained: L = 64 | trained: L = 256 | swapped into exact model: L = 64 | swapped: L = 256 |
|---|---|---|---|---|
| exact | 1.000 | 0.869 | 1.000 | 0.869 |
| local_window | 1.000 | **1.000** | 1.000 | **1.000** |
| block_sparse | 1.000 | 0.978 | 1.000 | 0.989 |
| linear_elu | 0.560 | 0.293 | 0.240 | 0.151 |

An earlier 200-step run (`20261011-200406_tasks_quick`) gave the same pattern.

1. **A task needing only local information is solved perfectly by restricted attention.**
2. **Restriction improved length generalization.** The exact model fell to 86.9 % at 4× its training length.
   *The same weights* with a window of 8 swapped in scored 100 %. So the exact model's errors at long lengths come
   from attending to far positions it never saw in training, not from any missing information. Removing those pairs
   fixes it.
3. **linear_elu learned the 3-token copy only partly** (56 %), consistent with the probes: its kernel cannot easily
   select one position sharply.

![Figure 18.1 — Task A at L = 256 (4× the training length): models trained with each backend.](../../experiments/reports/20261011-205830_tasks/taskA_local_copy/taskA_local_copy_trained_with_L256.png)

![Figure 18.2 — Task A at L = 256: the exact-trained model with each backend swapped in at inference.](../../experiments/reports/20261011-205830_tasks/taskA_local_copy/taskA_local_copy_swapped_into_sdpa_flash_cpu_model_L256.png)

## 18.2 Task B/C/D: variable-offset induction, 1500 steps, trained at L = 32

| backend | B: in distribution (offset 12–20) | C: distractor gap | D: offset 28–36 | D: offset 36–52 |
|---|---|---|---|---|
| exact | **1.000** | 0.183 | 0.031 | 0.125 |
| local_window (w = 8) | 0.491 | 0.155 | 0.029 | 0.029 |
| block_sparse (b = 8) | 1.000 † | 0.167 † | 0.029 | 0.084 |
| linear_elu | 0.966 | **0.421** | **0.174** | **0.245** |
| swap → local_window | 0.030 | 0.030 | 0.029 | 0.031 |
| swap → block_sparse | 1.000 † | 0.183 † | 0.028 | 0.082 |
| swap → linear_elu | 0.149 | 0.147 | 0.089 | 0.073 |

![Figure 18.3 — Task B (in distribution): accuracy of models trained with each backend.](../../experiments/reports/20261011-205830_tasks/taskBCD_induction/taskBCD_induction_trained_with_B_in_distribution.png)

### Finding 1: exact attention learns content-based retrieval

100 % in distribution, and in the diagnostic run it got there by step 500 (Chapter 19, D3).

### Finding 2: stacking layers lets a restricted model reach beyond its window

One layer with window 8 cannot see a source 12 or more tokens back. Yet the local-window model scored 49 %. The
accuracy-by-distance breakdown explains it:

| local_window, task B | sources 8–16 back | sources 16–32 back |
|---|---|---|
| accuracy | **0.99** | 0.06 |

Two layers of window 8 can **relay** information 16 tokens: layer 1 copies it 8 positions forward, layer 2 another 8.
Offsets 12–16 are reachable, offsets 17–20 are not, and the measured split matches. The swap confirms it: the
exact-trained model with a window swapped in scores 3 %, because it never learned to relay (it had no need to).

### Finding 3: a "sparse" layout that was secretly dense

block_sparse scored exactly like exact attention at L = 32. Checking the layout (`make_layout(4, local=1, global=1,
random=1, causal=True)`) showed it keeps **528 of 528 causal pairs**: at L = 32 with block 8 there are only 4 blocks,
and local ±1 + global 0 + 1 random covers all of them. So those cells (marked †) are **full attention under another
name**, not evidence about sparsity. At L = 64 the layout keeps 69.2 % of pairs, so the D columns do test a real
restriction.

### Finding 4: trained linear attention can retrieve when there are few candidates

linear_elu reached 96.6 %, far better than the untrained probes suggested. Training can learn query and key
projections that make the elu + 1 kernel sharp enough to pick one of about 30 positions. The swap result (15 %) shows
it learned a different solution from the exact model. In this single-seed run linear_elu also degraded least on C and
D. That is a hint, not an established result.

### Finding 5: nobody generalizes to unseen offsets

At offsets 28–52 every model collapses to 3–25 %. RoPE gives the model *relative* positions, but a circuit trained only
on offsets 12–20 does not transfer to 28+.

### Finding 6: distractor gaps break every model, and ambiguity does not explain it

Filling the gap with segment-vocabulary tokens dropped exact attention from 100 % to 18 %. One possible
explanation is that the task becomes ambiguous (the current token might also occur in the gap, so "its earlier
occurrence" is not unique). That was **measured**, not assumed:

| eval set | targets whose current token occurs more than once earlier |
|---|---|
| B (normal gap) | 30.1 % |
| C (distractor gap) | 36.7 % |

The exact model handles B's 30 % of ambiguous targets *perfectly*, so a rise to 36.7 % cannot explain a drop to 18 %.
The failure is **out-of-distribution brittleness**. An untested hypothesis: the models locate the segment by its
boundary with noise tokens, and that cue disappears when the gap uses segment tokens.

## 18.3 Summary of the quality picture

| question | evidence |
|---|---|
| Does restriction lose long-range information? | Yes, deterministically at the pattern's edge (probes; local_window 0.06 beyond its 2-layer reach) |
| Can depth compensate? | Partly: two layers relay 2w tokens (0.99 within 16) |
| Can restriction help? | Yes: length generalization on local tasks (100 % vs 87 %); immunity to out-of-pattern decoys |
| Is linear attention useless for retrieval? | No: trained, it reached 96.6 % with about 30 candidates; untrained, it is weak at L ≥ 64 |
| Are these general truths about language models? | **No.** They are single-seed, tiny models on synthetic tasks: evidence about mechanisms, not about LLM quality |

## Key points to remember

* Task A: restricted attention solves local tasks and length-generalizes better (window swap: 86.9 % → 100 %).
* Task B: exact 100 %; local_window 49 % via 2-layer relay (99 % within 16 back, 6 % beyond).
* block_sparse at L = 32 was dense (528/528 pairs): those cells are not sparsity evidence.
* Trained linear attention: 96.6 % in distribution, a different solution from exact.
* All models fail beyond their training offsets and on distractor gaps. The distractor drop is brittleness, not ambiguity (30.1 % vs 36.7 %).
