# Results on this machine

Every number below was measured in this project on 2026-10-11. The raw records are in
`experiments/results/<run>/results.jsonl` and the generated tables and figures are in
`experiments/reports/<run>/`. The run directory is named for each section. Theory is
labeled as theory.

## 0. Conditions (read first)

* **Hardware**: AMD Ryzen 5 7530U (6 cores / 12 threads, AVX2, L2 512 KB/core, L3 16 MB),
  16 GB RAM, Windows 11, Python 3.13.3, torch 2.10.0+cpu (MKL, oneDNN, OpenMP). No GPU.
* **The machine was shared.** Throughout the session, unrelated workloads (training
  scripts, a model-serving backend and test runs from other projects) kept between 5 and
  all 12 logical CPUs busy. Free RAM was mostly 0.6–1.7 GB. We did not stop them; they
  are not ours. The consequences:
  * **Memory and accuracy measurements are unaffected.** Allocation sizes and model
    outputs are deterministic.
  * **Wall-clock timings are contaminated.** The first exact-scaling sweep recorded
    7–11 *other* busy logical CPUs during its L ≥ 2048 timings, and identical forward
    code measured 13 ms vs 35 ms in neighbouring configs. Its timing exponents (e.g. 3.0
    for the tiled forward) are physically implausible for O(L²) work and are **not used
    as evidence**. Timing claims below come from the later single-thread runs, which
    record background load and process CPU time per configuration.
  * The memory guard (budget = min(50 % of available RAM, 2.5 GB)) skipped the
    configurations that would have needed more than ~0.6–0.8 GB at the time they were
    scheduled. Those points are missing, not estimated.

## 1. Numerical correctness (C1)

Run: `20261011-194506_correctness` (`python lab.py correctness --full`): 795
comparisons, with L ∈ {1, 7, 64, 129, 512, 1024}, d ∈ {8, 16, 64, 128}, causal / non-causal,
B·H variants, padding, fully-masked rows, input magnitudes 0.01–30, and tiles that do not
divide L. The reference is naive attention in fp64 on the same rounded inputs. The metric
is max|err| / max|ref|.

| method | dtype | cases | pass fwd | pass grad | worst fwd err | worst grad err | median fwd err |
|---|---|---|---|---|---|---|---|
| sdpa_math | fp64 | 49 | 100 % | 100 % | 3.4e-15 | 4.1e-15 | 3.7e-16 |
| sdpa_flash_cpu | fp64 | 49 | 100 % | 100 % | 3.7e-15 | 3.9e-15 | 3.0e-16 |
| tiled | fp64 | 49 | 100 % | 100 % | 3.0e-15 | 4.3e-15 | 4.6e-16 |
| tiled_recompute | fp64 | 49 | 100 % | 100 % | 3.0e-15 | 4.7e-15 | 4.6e-16 |
| naive | fp32 | 57 | 100 % | 100 % | 6.5e-05 | 1.1e-04 | 2.7e-07 |
| sdpa_math | fp32 | 57 | 100 % | 100 % | 1.6e-05 | 6.3e-05 | 2.7e-07 |
| sdpa_flash_cpu | fp32 | 57 | 100 % | 100 % | 6.5e-05 | 7.4e-04 | 3.0e-07 |
| tiled | fp32 | 57 | 100 % | 100 % | 3.8e-05 | 6.1e-05 | 2.5e-07 |
| tiled_recompute | fp32 | 57 | 100 % | 100 % | 3.8e-05 | 7.4e-04 | 2.5e-07 |
| sdpa_math | bf16 | 53 | 100 % | 100 % | 3.1e-03 | 4.9e-03 | 2.1e-03 |
| sdpa_flash_cpu | bf16 | 53 | 100 % | 100 % | 4.0e-03 | 9.6e-03 | 2.3e-03 |
| tiled | bf16 | 53 | 100 % | 100 % | 3.1e-03 | 1.8e-02 | 2.1e-03 |
| tiled_recompute | bf16 | 53 | 100 % | 100 % | 3.1e-03 | 9.1e-03 | 2.1e-03 |
| **naive (all ops in bf16)** | bf16 | 53 | **73.6 %** | 94.3 % | **0.377** | **2.24** | 5.7e-03 |

Findings:

1. **All four optimized exact implementations are numerically equivalent to the reference**,
   in forward and gradients, for every shape, mask, dtype and magnitude tested. In fp64
   they agree to ~1e-15. The `tiled_recompute` custom backward also passes
   `torch.autograd.gradcheck`.
2. **Accumulator precision matters more than the algorithm.** Naive attention computed
   entirely in bf16 fails 14 of 53 cases. Its error grows with logit magnitude (0.059 at
   input scale 3, 0.147 at 10, 0.377 at 30) and with L and d (0.95–1.6 % at L ≥ 512, magnitude 1, vs
   ~0.3 % for the fp32-accumulating kernels). Logits in the hundreds have bf16 spacing ≈ 1,
   so `exp(s − m)` is off by up to `e^0.5`. The tiled code keeps `S, m, l, acc` in fp32,
   as FlashAttention does, and stays at bf16 output-rounding error at every magnitude.
3. **The fp32 error floor scales with logit magnitude for *every* implementation**, about
   (0.1–1.5)·u·max|S|. At input scale 10, the naive formula in fp32 and the fused kernel
   have the *identical* error, 2.04e-5. This was found when `sdpa_flash_cpu` first failed
   a fixed 2e-5 threshold; the investigation is in `numerical_guide.md` §4.
4. **Max-subtraction is not optional.** Without it, fp32 softmax produced NaN in 17.9 % of
   outputs at max logit 335 and in 40.3 % at max logit 1340. The stable version's error
   stayed below 2e-7.
5. Exact methods' fp32 output error stays flat at ~6e-7 to 2e-6 from L = 128 to 2048
   (scaling run below), with no growth in L.

## 2. Memory scaling of exact methods (E1, memory part)

Run: `20261011-191952_scaling_exact` (isolated process per config; B=1, H=4, d=64,
fp32). Figure: `experiments/reports/20261011-191952_scaling_exact/memory_compare_noncausal.png`.

**Measured tensor peak per call (MiB), non-causal** (PyTorch CPU allocator, excluding
pre-existing inputs):

| L | naive fwd | sdpa_math fwd | flash_cpu fwd | tiled fwd | naive fwd+bwd | flash_cpu fwd+bwd | tiled (autograd) fwd+bwd | tiled_recompute fwd+bwd |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.88 | 0.69 | 0.22 | 1.13 | 1.75 | 0.75 | 1.13 | 2.01 |
| 512 | 12.5 | 9.5 | 1.07 | 1.78 | 25.0 | 3.51 | 7.03 | 3.78 |
| 1024 | 49.0 | 37.0 | 3.27 | 2.54 | 98.0 | 9.02 | 22.0 | 5.81 |
| 2048 | 194.1 | 146.0 | 4.29 | 4.57 | 388.0 | 14.0 | 76.4 | 9.88 |
| 4096 | skipped (guard) | skipped (guard) | 6.32 | 8.63 | skipped | 24.1 | skipped (guard) | 18.0 |

**Fitted log-log exponents of tensor peak vs L (L ≥ 512):**

| method | fwd | fwd+bwd |
|---|---|---|
| naive | 1.98 | 1.98 |
| sdpa_math | 1.97 | (fwd+bwd skipped at L=2048) |
| sdpa_flash_cpu | 0.81 | 0.90 |
| tiled (autograd through loop) | 0.77 | **1.72** |
| tiled_recompute | 0.84 | **0.75** |

Findings:

1. **Materializing methods are quadratic in memory, exactly as modeled.** The theory
   predicted the naive forward allocation (3 score-sized tensors plus the output) as
   0.875 / 12.5 / 49.0 / 194 MiB. The measurements are 0.88 / 12.5 / 49.0 / 194.1 MiB.
2. **Tiling makes forward memory near-linear**, and so does the fused CPU flash kernel.
   At L = 2048: 4.3–4.6 MiB vs 194 MiB, a 42–45× reduction.
3. **Tiling alone does *not* make training memory sub-quadratic.** With autograd
   through the tile loop, fwd+bwd memory grows with exponent 1.72 (76 MiB at L=2048),
   because every tile's P is saved. The custom recompute backward brings it back to
   linear (0.75; 9.9 MiB at L=2048, 39× less than naive), as the store-vs-recompute
   analysis predicts.
4. Process-level peak working set (OS, isolated workers) shows the same ordering. At
   L = 2048, the fwd+bwd process peak Δ was 446 MiB for naive vs 57 MiB for
   sdpa_flash_cpu, including ~35–40 MiB of fixed process and allocator overhead seen
   even at L = 128.

## 2b. Runtime of exact methods (E1b)

Run: `20261011-210946_scaling_exact_timing` (`python lab.py bench --spec experiments/configs/scaling_exact_timing.yaml`).
B=1, H=4, d=64, fp32, non-causal, **1 thread**, warm-up 2, 7 timed repeats, medians.
Background load from other processes during the timed regions: 4.5–10.6 logical CPUs, so every config is
flagged `contended`. Process CPU time per call agrees with wall time to within a few percent at L ≥ 512
(e.g. flash fwd at L=4096: 516 ms wall vs 509 ms CPU), so the single benchmark thread was rarely
descheduled. Absolute values on an idle machine would still differ somewhat.

**Median time per call (ms)**

| L | naive fwd | sdpa_math fwd | **sdpa_flash_cpu fwd** | tiled fwd | tiled_recompute fwd | naive fwd+bwd | sdpa_math fwd+bwd | **sdpa_flash_cpu fwd+bwd** | tiled (autograd) fwd+bwd | tiled_recompute fwd+bwd |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 4.1 | 5.3 | 2.6 | 5.1 | 5.5 | 16.0 | 10.5 | 10.7 | 19.6 | 20.1 |
| 512 | 28.1 | 23.6 | 8.4 | 21.8 | 23.3 | 82.4 | 54.7 | 36.4 | 75.1 | 68.7 |
| 1024 | 105.5 | 86.5 | 31.5 | 73.4 | 87.7 | 304.4 | 210.0 | 135.9 | 353.1 | 268.7 |
| 2048 | 392.4 | 355.6 | 146.7 | 313.0 | 345.0 | 1411.2 | 803.7 | 503.9 | 1847.0 | 979.9 |
| 4096 | 1540.4 | 1512.9 | 515.6 | 1445.4 | 1600.3 | (guard) | (guard) | 2017.0 | 14489.5 | 4637.2 |

**Fitted exponents of time vs L (L ≥ 512; wall / CPU time)**

| method | fwd | fwd+bwd |
|---|---|---|
| naive | 1.92 / 1.97 | 2.05 / 2.02 |
| sdpa_math | 2.00 / 1.97 | 1.94 / 1.91 |
| sdpa_flash_cpu | 2.01 / 2.10 | 1.93 / 1.91 |
| tiled (autograd) | 2.02 / 2.00 | **2.52 / 2.52** |
| tiled_recompute | 2.03 / 2.02 | 2.01 / 2.02 |

Findings:

1. **Every exact method is quadratic in time** (exponents 1.9–2.1). That includes the ones whose *memory*
   is linear (§2). Tiling changes where scores live, not how many there are.
2. **The fused CPU kernel is the fastest exact method at every L: about 3× faster than naive in forward**
   (515.6 vs 1540.4 ms at L=4096) and **2.8× in fwd+bwd** at L=2048. It does the same arithmetic, so the
   gain is fusion and memory traffic (`systems_guide.md`).
3. **Single-threaded, the Python tiled forward is not slower than naive.** It is 6–30 % faster at L ≥ 512
   (313 vs 392 ms at L=2048). The per-tile dispatch overhead is offset by avoiding three `L²` tensors that
   overflow the 16 MB L3. This depends on the thread count. With 4 threads, tiled was slower than naive in
   both saved 4-thread runs: `20261011-191916_smoke` (L=256, H=2, d=32: 6.34 vs 0.69 ms) and the contended
   `20261011-191952_scaling_exact` (L=512: 13.1 vs 5.2 ms). Naive's large ops parallelize; tiled's small
   ops do not. The claim is therefore conditional, not general (thread scaling: §2c).
4. **The recompute backward is faster than storing, despite 25 % more matmul FLOPs.** At L=2048,
   tiled_recompute fwd+bwd takes 980 ms against 1847 ms for autograd-tiled and 1411 ms for naive. At L=4096
   it takes 4.6 s against 14.5 s for autograd-tiled. The autograd variant's exponent of 2.52 (super-quadratic)
   indicates growing memory-system cost from keeping every P tile alive. This is a measured CPU result
   consistent with FlashAttention's store-vs-recompute argument. It is not a GPU result.
5. The fused kernel's fwd+bwd is still about 2× faster than tiled_recompute (504 vs 980 ms at L=2048): same
   algorithm, compiled and fused versus dispatched op by op.

## 2c. Approximate methods: time, memory and deviation (E2)

Run: `20261011-212017_scaling_approx` (`python lab.py bench --spec experiments/configs/scaling_approx.yaml`).
Isolated worker per config; B=1, H=4, d=64, fp32, **1 thread**; local_window w=64; block_sparse b=64
(±1 local block, global block 0, 1 random block); linear_elu chunk 64. 55 of 56 configs measured. The fused
exact kernel at L=16384 non-causal was skipped by the time budget (its fwd+bwd at L=8192 took 8.4 s).
Background load during timing: mostly 6–11 other logical CPUs, with lulls of 2.7–3.9. Individual points
(e.g. block_sparse at L=2048 vs 4096) are therefore noisier than the trends.

**Median forward time (ms), non-causal / causal**

| L | sdpa_flash_cpu (exact) | local_window | block_sparse | linear_elu |
|---|---|---|---|---|
| 256 | 2.7 / 2.6 | 9.7 / 6.9 | 8.2 / 10.7 | 2.0 / 5.3 |
| 1024 | 32.7 / 27.5 | 37.4 / 21.8 | 57.0 / 44.4 | 8.4 / 19.7 |
| 2048 | 131.2 / 82.5 | 86.4 / 50.7 | 143.9 / 97.1 | 16.6 / 48.6 |
| 4096 | 439.4 / 269.4 | 92.2 / 71.8 | 147.4 / 109.3 | 25.4 / 69.1 |
| 8192 | 2446.1 / 1033.5 | 392.2 / 229.4 | 366.1 / 277.1 | 60.0 / 176.1 |
| 16384 | (time budget) / 3959.4 | 779.7 / 191.1 * | 962.4 / 742.6 | 133.3 / 361.2 |

\* measured during a load lull (2.7 other cores vs ~10 for its neighbours); treat as an outlier.

**Median fwd+bwd time (ms), non-causal:** at L=2048, flash 484, local_window 247, block_sparse 412,
linear 60. At L=8192: 8376, 925, 1044, 223.

**Measured tensor peak per forward call (MiB), non-causal**

| L | sdpa_flash_cpu | local_window | block_sparse | linear_elu | *naive (from §2)* |
|---|---|---|---|---|---|
| 1024 | **1.6** | 19.0 | 29.4 | 4.1 | 49.0 |
| 4096 | **4.6** | 74.5 | 122.6 | 16.2 | (guard) |
| 16384 | — | 296.3 | 496.1 | 64.6 | — |

**Fitted exponents (L ≥ 512):** time: local_window 0.82–1.02, block_sparse 0.89–0.97, linear_elu 0.90–0.97,
flash 1.73–1.99. Tensor memory: all approximate methods 0.99–1.03, flash 0.90–0.96.

**Deviation from full softmax output (relative Frobenius, random inputs, non-causal):** local_window 1.15 →
3.85 and block_sparse 0 → 2.32 from L=256 to 2048. linear_elu is flat at 0.75–0.79. At L=256, block_sparse
with b=64 covers all 4 blocks (deviation 0: it *is* full attention there). On random Gaussian inputs,
attention is diffuse, so dropping most keys changes the output by more than its own norm. This measures how
much the function changes, not task quality (see §3–4 for quality).

Findings:

1. **The approximate methods are linear in L in both time and memory, as theory predicts, and they overtake
   the best exact kernel only beyond a crossover.** Forward, non-causal: linear_elu is faster at every
   measured L. local_window wins from L ≈ 2048. block_sparse is roughly even at 2048 and clearly faster from
   4096. Below about 1024, local_window and block_sparse are 1.1–3.6× slower than the fused exact kernel,
   despite evaluating fewer scores.
2. **At long lengths the gap is large:** at L=8192 non-causal forward, the fused exact kernel takes 2.4 s,
   local_window 0.39 s (6.2×) and linear_elu 0.06 s (41×). Causal attention narrows it, because the fused
   kernel skips masked tiles (forward 1.2× faster than its non-causal self at L=1024, rising to 2.4× at L=8192).
3. **"Approximate attention saves memory" is false relative to a fused exact kernel.** Every approximate
   method here allocates *more* tensor memory than sdpa_flash_cpu at every L (16–27× more for local_window
   and block_sparse at L=4096), because they materialize O(L·window) score tensors and gathered blocks while
   the fused kernel streams tiles. They save memory only relative to *materializing* exact implementations:
   at L=1024, local_window uses 19 MiB against naive's 49 MiB, and the gap grows with L (§2: naive 194 MiB at
   L=2048).
4. Linear growth at long L is real, but the constant factors (padding, gather and unfold copies, masked waste
   of about C/(C+2w)) decide who wins at the lengths people actually use. The measured crossover, not the
   asymptotic class, is the useful quantity, and it is specific to this CPU and these implementations.

## 2d. Tile size in the Python tiled kernels (E3)

Run: `20261011-215718_tile_sizes` (L=2048, B=1, H=4, d=64, fp32, non-causal, 1 thread). **Heavily contended**
(about 11 other logical CPUs busy for every config), so only comparisons *within* this run are meaningful.
naive measured 784 ms forward here, against 392 ms in §2b.

| implementation | tile | Python tile iterations | fwd (ms) | fwd tensor peak (MiB) | fwd+bwd (ms) | fwd+bwd tensor peak (MiB) |
|---|---|---|---|---|---|---|
| tiled (autograd) | 32 | 4096 | 1941 | 4.1 | **41418** | 79.0 |
| tiled (autograd) | 64 | 1024 | 704 | 4.3 | 4693 | 77.1 |
| tiled (autograd) | 128 | 256 | **528** | 4.6 | 4177 | 76.4 |
| tiled (autograd) | 256 | 64 | 639 | 5.6 | 2190 | 75.8 |
| tiled (autograd) | 512 | 16 | 899 | 14.6 | 1391 | 79.1 |
| tiled_recompute | 32 | 4096 | 1478 | 4.1 | 4523 | 8.3 |
| tiled_recompute | 128 | 256 | 466 | 4.6 | 1362 | 9.9 |
| tiled_recompute | 512 | 16 | 442 | 14.6 | 1273 | 33.1 |
| naive | — | — | 784 | 194.1 | 2258 | 388.0 |
| sdpa_flash_cpu | — | — | 152 | 2.6 | 773 | 11.0 |

Findings:

1. **Forward time is U-shaped in tile size, with a minimum at 128.** Small tiles pay Python and dispatcher
   overhead per tile (4096 iterations at tile 32 → 3.7× slower than tile 128). Larger tiles get slower again:
   at tile 512 a [4, 512, 512] fp32 score tile is 4 MiB, far above the 512 KB per-core L2, against 256 KiB
   at tile 128. The cache explanation fits the data but was not measured with hardware counters.
2. **Forward memory is flat until the tile term dominates** (4.1 → 4.6 MiB from tile 32 to 128, then
   14.6 MiB at 512 ≈ L·d terms + tile²), as `complexity/theory.py` models.
3. **With autograd through the loop, tile size cannot fix training memory.** fwd+bwd stays at 76–79 MiB
   for every tile size (the saved P tiles always total O(L²)). Small tiles are catastrophic for backward
   time: 41 s at tile 32, because autograd records and replays about 10 nodes per tile. The recompute
   backward keeps 8–10 MiB at tiles ≤ 128 and is 1.1–9.2× faster than autograd at equal tile size (9.2× at tile 32, 3.1× at 128, 1.1× at 512).
4. Even at its best tile, the Python implementation is 3.1–3.5× slower than the fused kernel in forward
   (466–528 vs 152 ms). The algorithm is the same; the execution is not.

## 2e. Thread scaling (E4)

Run: `20261011-221221_threads` (isolated worker per config; L=2048, B=1, H=4, d=64, fp32, non-causal;
threads 1, 2, 4, 6). Background load was about 9–11 other logical CPUs for naive, flash and tiled, and
about 3–9 for linear and local. Multi-threaded runs suffer most from contention, so read these as
indicative.

**Forward median (ms) and speedup over 1 thread**

| method | 1 | 2 | 4 | 6 | speedup at 2 / 4 / 6 |
|---|---|---|---|---|---|
| sdpa_flash_cpu | 124.3 | 85.9 | 42.7 | 35.1 | 1.45 / 2.91 / **3.54** |
| naive | 402.9 | 319.2 | 234.3 | 230.0 | 1.26 / 1.72 / 1.75 |
| linear_elu | 7.8 | 4.5 | 4.0 | 3.5 | 1.74 / 1.95 / 2.23 |
| local_window | 35.6 | 29.1 | 23.4 | 42.0 | 1.22 / 1.52 / 0.85 |
| tiled (Python) | 368.9 | 564.3 | 839.1 | 1772.1 | **0.65 / 0.44 / 0.21** |

Process CPU time per tiled forward call rose from 342 ms (1 thread) to 3692 ms (6 threads), while its
CPU/wall ratio stayed near 1.1–1.7. The extra threads mostly spun or waited rather than doing useful
parallel work.

Findings:

1. **The fused kernel scales best:** 3.5× with 6 threads on 6 physical cores, despite the background
   load. Its parallel work units are large (whole query blocks per head).
2. **Materializing naive attention scales poorly beyond 4 threads** (1.75×). Its elementwise passes over
   L² tensors are memory-bandwidth-bound, and more cores do not add bandwidth.
3. **The Python tiled implementation gets slower with more threads, 4.8× slower at 6 threads than at 1.**
   Each tile operation is too small to split profitably. Every op pays OpenMP fork/join synchronization,
   and under background load a descheduled thread stalls the whole op. This is the strongest
   demonstration in the project that *how* an algorithm is executed can matter more than *what* it
   computes. It also explains why §2b's single-thread result (tiled ≈ naive) and the 4-thread result
   (tiled ≪ naive) differ.
4. local_window's drop at 6 threads coincides with a load increase (8.8 other cores vs 2.6 at 1 thread),
   so it is not interpretable on its own.

## 2f. Shapes and number formats (E5)

Run: `20261011-222558_shapes_dtypes` (L=1024, H=4, 1 thread; B ∈ {1,4}; d ∈ {16,32,64,128}; fp32 and bf16; 46 measured,
2 fwd+bwd modes skipped by the guard; background load 7–11 other logical CPUs, so cells are noisy).

| B=1, forward ms | naive fp32 | naive bf16 | flash fp32 | flash bf16 | tiled fp32 | tiled bf16 |
|---|---|---|---|---|---|---|
| d=16 | 102 | 1009 | 15.0 | 74.8 | 60.1 | 44.4 |
| d=64 | 128 | 1587 | 77.1 | 119.0 | 165.9 | 80.2 |
| d=128 | 132 | 3152 | 69.8 | 168.6 | 185.7 | 198.3 |

Findings: **bf16 is never faster than fp32 on this AVX2 CPU**. It is 10–30× slower for naive attention (B=4, d=128:
12.5 s vs 0.41 s) and 1.5–5.5× slower for the fused kernel. The tiled code is roughly unchanged because it accumulates
in fp32 anyway. The hypothesis is confirmed. B=4 is 1.6–5.5× B=1. The d-dependence is not resolvable at this noise
level, so no claim is made.

## 3. Quality probes: what a single attention operation can retrieve (Q1)

Run: `20261011-195716_probes` (`python lab.py probes --full`, 128 trials per point, β=10,
d=64). Report: `experiments/reports/20261011-195716_probes/probes.md`.

| distance to target (L=1024) | 1–63 | 64 | 65 | 128 | 256 | 512 | 1023 |
|---|---|---|---|---|---|---|---|
| exact softmax (sdpa_flash_cpu) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| local_window (w=64) | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| block_sparse (b=64, 1 local + global block 0 + 1 random) | 1.00 | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 | **1.00** |
| linear_elu | ≤0.008 | 0 | 0 | 0 | 0.008 | 0.008 | 0.008 |

Findings:

1. **Restricted methods fail deterministically and exactly where their pattern ends.**
   local_window drops from 100 % to 0 % between distance 64 and 65. block_sparse
   succeeds inside its own and the neighbouring block, fails at 128–512, and succeeds
   again at distance 1023, because the target at position 0 lies in the global block.
   Its reach is a property of the layout, not of distance.
2. **Exact softmax retrieved at 100 % up to L = 4096** (target at position 0), so
   dilution was not a factor at β = 10 in this range.
3. **Interference cuts both ways.** With 16 decoy keys at cosine 0.9 to the query (L=512),
   exact softmax fell to 86.7 % (97.7 % with 8 decoys). local_window and block_sparse
   stayed at 100 %, because the decoys fell outside their pattern. A restriction can
   protect against distractors it cannot see.
4. **linear_elu's kernel is not sharp enough to retrieve one item among many.** It is
   at chance (≈1/L) at β = 10. The fairness check that sweeps input scale shows it is not
   zero by construction: accuracy rises with β (L=16: 6 % → 73 % from β=1 to 100;
   L=64: 2 % → 19 %), whereas softmax reaches 100 % from β = 3 at both lengths. φ = elu+1
   sharpens only polynomially with scale, while exp sharpens exponentially.

## 4. Trained synthetic tasks (Q2)

Run: `20261011-205830_tasks` (`python lab.py tasks --which AB --steps 1500 --threads 1 --batch-size 32`).
Model: 2-layer pre-LN Transformer, d_model 64, 4 heads, RoPE, about 100k parameters, AdamW lr 3e-3, batch 32,
1 thread. Every backend uses the identical seed, data stream and evaluation sets (512 sequences). Backends:
exact (sdpa_flash_cpu), local_window (w=8), block_sparse (block 8; ±1 local block, global block 0, 1 random
block), linear_elu (chunk 16). "Swap" = the model trained with exact attention, with the other backend
substituted at inference, no retraining. Single seed.

### Task A — local copy (lag 3), 500 steps, trained at L = 64

| backend | trained: L=64 | trained: L=256 | swapped into exact model: L=64 | swapped: L=256 |
|---|---|---|---|---|
| exact | 1.000 | 0.869 | 1.000 | 0.869 |
| local_window | 1.000 | **1.000** | 1.000 | **1.000** |
| block_sparse | 1.000 | 0.978 | 1.000 | 0.989 |
| linear_elu | 0.560 | 0.293 | 0.240 | 0.151 |

(An earlier 200-step run, `20261011-200406_tasks_quick`, gave the same pattern: 1.000/0.867, 1.000/1.000,
1.000/0.978, 0.499/0.268.)

1. A task that needs only local information is solved perfectly by the restricted methods.
2. **Restriction improved length generalization.** The exact model drops to 86.9 % at 4× the training
   length. The *same weights* with a window of 8 swapped in score 100 %. Its long-length errors therefore
   come from attending to far positions never seen in training, not from missing information.
3. linear_elu does not learn even a 3-token copy well in this budget (56 %). That is consistent with its
   kernel being unable to select one position sharply (probes, §3).

### Task B/C/D — variable-offset induction, 1500 steps, trained at L = 32

Each target is the successor of the current token's earlier occurrence, 12–20 tokens back, so it must be
found by content (`docs/mqar_diagnosis.md` D2/D3). Columns C and D are evaluated on the models trained on B.

| backend | B: in distribution (offset 12–20) | C: gap filled with distractors | D: offset 28–36 (L=64) | D: offset 36–52 (L=64) |
|---|---|---|---|---|
| exact | **1.000** | 0.183 | 0.031 | 0.125 |
| local_window (w=8) | 0.491 | 0.155 | 0.029 | 0.029 |
| block_sparse (b=8) | 1.000 † | 0.167 † | 0.029 | 0.084 |
| linear_elu | 0.966 | **0.421** | **0.174** | **0.245** |
| *swap into exact model:* local_window | 0.030 | 0.030 | 0.029 | 0.031 |
| *swap:* block_sparse | 1.000 † | 0.183 † | 0.028 | 0.082 |
| *swap:* linear_elu | 0.149 | 0.147 | 0.089 | 0.073 |

† At L = 32 with block 8 the layout keeps **all** 528 causal pairs (density 1.000, checked with
`make_layout`), so block_sparse computes full causal attention there. Those cells are *not* evidence about
sparsity. At L = 64 the layout keeps 69.2 % of causal pairs.

**local_window accuracy by distance (B):** 0.99 for sources 8–16 tokens back, 0.06 for 16–32.

Findings:

1. **Exact attention learns content-based retrieval** (100 % in distribution, by step 500 in the
   diagnostic run).
2. **A restricted model can partly route around its restriction by stacking layers.** One layer with
   window 8 cannot see a source 12 or more tokens back, but two layers can relay information 16 tokens.
   The trained local-window model solves 99 % of targets 8–16 back and 6 % of those further back (49 %
   overall). The exact-trained model with the window swapped in gets 3 %: it never learned the relay,
   because it did not need to.
3. **Trained linear attention retrieves well when there are few candidates.** At L = 32 it reaches 96.6 %,
   unlike the untrained probes at L ≥ 128 (§3). Learned projections can sharpen the elu+1 kernel enough
   for about 30 keys. Swapping linear attention into the exact model gives 15 %, so the two learn
   different solutions.
4. **No backend generalizes to unseen offsets** (D: 3–25 %). Every model fails out of distribution. In this
   single-seed run linear_elu degrades least (17–25 %), which is a hint, not a supported conclusion.
5. **Distractor gaps hurt every model badly (C), and ambiguity does not explain it.** In the C evaluation
   set, the share of targets whose current token occurs more than once earlier rises only from 30.1 %
   (B) to 36.7 % (C). The exact model handles the B ambiguity perfectly, so the drop to 18 % is
   out-of-distribution brittleness. *Hypothesis (untested):* the models locate the segment by its
   boundary with noise tokens, and that cue disappears when the gap uses segment tokens.

### MQAR

No backend learned MQAR key-value binding within the CPU budgets tried. Every model sits at the
binding-free baseline. The investigation (`docs/mqar_diagnosis.md`) rules out implementation defects,
including the attention backends, and is not used as evidence about the backends.
