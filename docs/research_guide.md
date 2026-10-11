# Research guide: experiments, hypotheses, controls, interpretation

Each experiment maps a research question to an executable command. Results are in
`docs/results.md` (written from the stored measurements) and
`experiments/reports/`.

| ID | Question | Command |
|---|---|---|
| C1 | Are the exact implementations numerically equivalent? | `python lab.py correctness [--full]` |
| E1 | How do exact methods scale in time and memory with L? | `python lab.py bench --spec experiments/configs/scaling_exact.yaml` |
| E2 | When do approximate methods save resources? | `... scaling_approx.yaml` |
| E3 | How does tile size trade overhead against working set? | `... tile_sizes.yaml` |
| E4 | How do implementations scale with threads? | `... threads.yaml` |
| E5 | How do d, B·H and dtype affect runtime? | `... shapes_dtypes.yaml` |
| Q1 | Which methods can route information from a given distance? (Tasks A/B/C, attention-level) | `python lab.py probes` |
| Q2 | What quality do models trained with each backend reach? (Tasks A–D) | `python lab.py tasks` |

## C1 — Numerical equivalence
* **Hypothesis**: all exact methods agree with fp64 naive to within rounding-level tolerances in fp64/fp32; with bf16 inputs, methods that keep fp32 statistics stay at output-rounding error.
* **Baseline**: naive attention in fp64 on the same rounded inputs.
* **Controlled**: inputs (seeded), masks, upstream gradient.
* **Varied**: L (1–1024), d (8–128), B·H, dtype, causal, padding, fully-masked rows, input magnitude (0.01–30), tile sizes (not dividing L).
* **Interpretation rule**: a failure is investigated (source of error, dependence on dtype, L and magnitude) before any tolerance change.
* **Confounders**: thread-count-dependent reduction order in MKL (tiny); tolerance choice.

## E1 — Exact scaling
* **Hypothesis**: all exact methods have time ∝ L² (fitted exponent ≈ 2 at large L). Tensor peak ∝ L² for naive/sdpa_math, ∝ L for flash-CPU and the tiled variants in forward; tiled-autograd ∝ L² in fwd+bwd, tiled-recompute ∝ L. The Python tiled code is slower than the fused kernel at every L.
* **Baselines**: naive (reference), sdpa_flash_cpu (best available exact kernel).
* **Controlled**: B=1, H=4, d=64, fp32, 4 threads, seed 0, isolated process per config.
* **Measurements**: fwd and fwd+bwd median/IQR, tensor peak, process peak Δ, error vs fp64, background load.
* **Interpretation**: compare exponents over L ≥ 512; compare absolute times only between uncontended runs with non-overlapping IQRs.
* **Confounders**: background load (measured), memory pressure (guard), cache-size transitions (the 16 MB L3 is reached by one [4, L, L] fp32 tensor at L=1024), Python overhead at small L.

## E2 — Approximate scaling
* **Hypothesis**: local_window, block_sparse and linear_elu have time and memory ∝ L, and overtake sdpa_flash_cpu beyond a crossover L. Below it they may be slower.
* **Controlled**: as E1. Window 64, block 64 (1 local + 1 global + 1 random block), linear chunk 64.
* **Extra measurement**: deviation from full softmax output. It is not an error for these methods; it is the size of the functional change on random inputs.
* **Interpretation**: report the crossover length per mode (fwd, fwd+bwd), not a single "speedup".

## E3 — Tile size
* **Hypothesis**: time decreases steeply with tile size (fewer Python iterations), then flattens; tensor peak grows ∝ tile² and approaches naive at tile = L.
* **Controlled**: L = 2048, B=1, H=4, d=64, fp32, non-causal.

## E4 — Threads
* **Hypothesis**: sdpa_flash_cpu and naive speed up with threads up to the 6 physical cores; the Python tiled code speeds up much less.
* **Confounders**: background load is especially damaging here (SMT siblings shared with other processes).

## E5 — Shapes and dtype
* **Hypothesis**: time ∝ d and ∝ B·H for exact methods; bf16 is not faster than fp32 on this AVX2 CPU.

## Q1 — Retrieval probes (no training)
* **Hypothesis**: exact softmax retrieves at every distance (accuracy limited only by dilution as L grows); local_window succeeds iff distance ≤ window; block_sparse succeeds iff the target block is in the layout (local, global block 0, or a random block); linear_elu degrades with L, because its kernel is not sharp.
* **Controlled**: identical q/k/v per trial across methods; β = 10, d = 64; 32–128 trials.
* **Interpretation**: a probe tests *routing capacity of one attention operation*, not what a trained model would learn.

## Q2 — Trained synthetic tasks
* **Task A (local copy, lag 3)**: all softmax-based methods should succeed (window 8 ≥ lag 3).
* **Task B (variable-offset induction)**: segment of 12, gap 0–8, then the segment again. Each target
  is the successor of the current token's earlier occurrence, 12–20 tokens back, found by content.
  Hypotheses: exact attention learns it (shown in `mqar_diagnosis.md` D3); local_window (w=8) cannot
  (the source is out of reach); block_sparse (block 8, local ±1 block + global block 0 + 1 random block)
  succeeds only where its layout covers the source; linear_elu is limited by its non-sharp kernel (probes, Q1).
* **Task C (interference)**: the same models evaluated with the gap filled by segment-vocabulary
  distractors instead of noise.
* **Task D (offset generalization)**: trained with offsets 12–20 at L=32; evaluated at L=64 with offsets
  28–36 and 36–52, never seen in training.
* **MQAR** (optional, `--which M`): not learned by any backend within the CPU budgets tried. The
  diagnosis is in `mqar_diagnosis.md`. It is not used as evidence about backends.
* **Controls**: identical initialization seed, training stream, budget, optimizer and evaluation sets. A
  "swap" evaluation runs the exact-trained model with each approximate backend at inference, which
  isolates functional change from learning dynamics.
* **Interpretation**: synthetic tasks probe specific capabilities and are not evidence about general
  language modelling. Single-seed results are indicative; differences of a few points are not
  meaningful without multiple seeds.
