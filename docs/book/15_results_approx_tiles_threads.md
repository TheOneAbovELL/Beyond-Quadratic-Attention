# Chapter 15 — Results II: approximate methods, tile sizes, threads, shapes and number formats

> **What you will learn:** when the approximate methods actually beat the best exact kernel, and when they do
> not; the surprising memory result; how tile size shapes the Python tiled kernel's speed and memory; how each
> method responds to more threads; and how head dimension, batch and bf16 affect time.

## 15.1 Approximate methods vs the fused exact kernel (E2)

Run: `20261011-212017_scaling_approx`: isolated workers, 1 thread, B = 1, H = 4, d = 64, fp32. local_window
w = 64; block_sparse b = 64 (±1 local, global 0, 1 random); linear_elu chunk 64. 55 of 56 configurations measured.
The fused kernel at L = 16,384 non-causal was skipped by the time budget, because its fwd+bwd at 8,192 took 8.4 s.
Background load was mostly 6–11 other logical CPUs, so single points are noisy and trends are reliable.

**Median forward time (ms), non-causal / causal:**

| L | flash (exact) | local_window | block_sparse | linear_elu |
|---|---|---|---|---|
| 256 | 2.7 / 2.6 | 9.7 / 6.9 | 8.2 / 10.7 | 2.0 / 5.3 |
| 1,024 | 32.7 / 27.5 | 37.4 / 21.8 | 57.0 / 44.4 | 8.4 / 19.7 |
| 2,048 | 131.2 / 82.5 | 86.4 / 50.7 | 143.9 / 97.1 | 16.6 / 48.6 |
| 4,096 | 439.4 / 269.4 | 92.2 / 71.8 | 147.4 / 109.3 | 25.4 / 69.1 |
| 8,192 | 2,446.1 / 1,033.5 | 392.2 / 229.4 | 366.1 / 277.1 | 60.0 / 176.1 |
| 16,384 | (budget) / 3,959.4 | 779.7 / 191.1* | 962.4 / 742.6 | 133.3 / 361.2 |

\* measured during a lull in background load (2.7 other cores vs about 10 for neighbouring points); treat as an outlier.

**Fitted time exponents:** local_window 0.82–1.02, block_sparse 0.89–0.97, linear_elu 0.90–0.97 (linear in L); flash
1.73–1.99 (quadratic).

![Figure 15.1 — Measured runtime vs L for the approximate methods and the fused exact kernel (run 20261011-212017_scaling_approx).](../../experiments/reports/20261011-212017_scaling_approx/runtime_vs_L.png)

**Findings.**

1. **Linear in theory, linear in measurement**: every approximate method's time grows about ∝ L.
2. **But they only win beyond a crossover.** In non-causal forward, linear_elu is faster at every measured length,
   local_window from about 2,048 tokens, and block_sparse is roughly even at 2,048 and clearly faster from 4,096. Below
   about 1,024 tokens, local_window and block_sparse are **1.1–3.6× slower** than the fused exact kernel, even though they
   evaluate far fewer scores: padding, `unfold` copies, gathering and masked waste cost more than they save at short
   lengths.
3. **At long lengths the gap is large**: at L = 8,192 non-causal, flash takes 2.4 s, local_window 0.39 s (6.2×) and
   linear_elu 0.06 s (41×).
4. **Causal attention narrows the gap**, because the fused kernel skips tiles above the diagonal: its causal forward is
   1.2× faster than its non-causal one at L = 1,024, rising to 2.4× at 8,192.

## 15.2 The memory surprise

**Measured tensor peak per forward call (MiB), non-causal:**

| L | flash (exact) | local_window | block_sparse | linear_elu | naive (Chapter 14) |
|---|---|---|---|---|---|
| 1,024 | **1.6** | 19.0 | 29.4 | 4.1 | 49.0 |
| 4,096 | **4.6** | 74.5 | 122.6 | 16.2 | (guard) |
| 16,384 | — | 296.3 | 496.1 | 64.6 | — |

![Figure 15.2 — Measured peak tensor memory, approximate methods vs the fused exact kernel.](../../experiments/reports/20261011-212017_scaling_approx/memory_compare_noncausal.png)

**"Approximate attention saves memory" is false relative to a fused exact kernel.** All three approximate methods
allocate *more* than `sdpa_flash_cpu` at every length: 16× (local) and 27× (block sparse) more at 4,096 tokens. They
materialize O(L·window) score tables or gathered key blocks, while the fused kernel streams small tiles and allocates
little beyond its output. The approximate methods save memory only relative to **materializing** exact methods: at
L = 1,024, local_window needs 19 MiB against naive's 49 MiB, and the gap widens with L. The theory model predicted this
(Chapter 10 §10.5). The measurement confirms it.

This is Principle B of the project in its purest form: **memory efficiency, computational efficiency and asymptotic
class are three different things.**

**Deviation from full softmax** (relative Frobenius norm, random inputs, non-causal): local_window grows from 1.15 to
3.85 between L = 256 and 2,048, block_sparse from 0 (its layout is dense at L = 256) to 2.32, and linear_elu stays about
0.75–0.79. On random inputs attention is diffuse, so dropping most keys changes outputs by more than their own size.
This number describes how much the *function* changes. Quality is measured by the probes and tasks.

## 15.3 Tile size in the Python tiled kernels (E3)

Run: `20261011-215718_tile_sizes` (L = 2,048, 1 thread). **Heavily contended** (about 11 other logical CPUs for every
config), so compare only within this run. (naive measured 784 ms here, against 392 ms in Chapter 14.)

| implementation | tile | Python tile iterations | fwd (ms) | fwd tensor peak (MiB) | fwd+bwd (ms) | fwd+bwd peak (MiB) |
|---|---|---|---|---|---|---|
| tiled (autograd) | 32 | 4,096 | 1,941 | 4.1 | **41,418** | 79.0 |
| tiled (autograd) | 64 | 1,024 | 704 | 4.3 | 4,693 | 77.1 |
| tiled (autograd) | 128 | 256 | **528** | 4.6 | 4,177 | 76.4 |
| tiled (autograd) | 256 | 64 | 639 | 5.6 | 2,190 | 75.8 |
| tiled (autograd) | 512 | 16 | 899 | 14.6 | 1,391 | 79.1 |
| tiled_recompute | 32 | 4,096 | 1,478 | 4.1 | 4,523 | 8.3 |
| tiled_recompute | 128 | 256 | 466 | 4.6 | 1,362 | 9.9 |
| tiled_recompute | 512 | 16 | 442 | 14.6 | 1,273 | 33.1 |
| naive | — | — | 784 | 194.1 | 2,258 | 388.0 |
| sdpa_flash_cpu | — | — | 152 | 2.6 | 773 | 11.0 |

1. **Forward time is U-shaped in tile size, with the minimum at 128.** Tiny tiles drown in Python and dispatcher
   overhead (4,096 iterations at tile 32, 3.7× slower than tile 128). Big tiles slow down again: a 512 × 512 tile for 4
   heads is 4 MiB, against the 512 KB per-core L2 cache, while tile 128 is 256 KiB. The cache explanation fits but was
   not measured with hardware counters.
2. **Forward memory stays flat until the tile² term dominates** (4.1 → 4.6 → 14.6 MiB).
3. **Tile size cannot fix autograd's training memory**: 76–79 MiB at every tile size. Small tiles are disastrous for
   autograd's backward time (41 s at tile 32), because it records and replays about 10 graph nodes per tile.
4. **The recompute backward stays at 8–10 MiB** for tiles ≤ 128 and is 1.1–9.2× faster than autograd at equal tile
   size.
5. Even at its best tile, Python tiling is **3.1–3.5× slower than the fused kernel**.

## 15.4 Threads (E4)

Run: `20261011-221221_threads` (L = 2,048, isolated workers, threads 1/2/4/6). Background load about 9–11 other logical
CPUs for naive/flash/tiled, about 3–9 for linear/local. Indicative.

| method | 1 thread (ms) | 2 | 4 | 6 | speed-up at 2 / 4 / 6 |
|---|---|---|---|---|---|
| sdpa_flash_cpu | 124.3 | 85.9 | 42.7 | 35.1 | 1.45 / 2.91 / **3.54** |
| naive | 402.9 | 319.2 | 234.3 | 230.0 | 1.26 / 1.72 / 1.75 |
| linear_elu | 7.8 | 4.5 | 4.0 | 3.5 | 1.74 / 1.95 / 2.23 |
| local_window | 35.6 | 29.1 | 23.4 | 42.0 | 1.22 / 1.52 / 0.85 |
| **tiled (Python)** | 368.9 | 564.3 | 839.1 | 1,772.1 | **0.65 / 0.44 / 0.21** |

![Figure 15.3 — Forward time vs number of threads at L = 2,048.](../../experiments/reports/20261011-221221_threads/threads_fwd.png)

1. **The fused kernel scales best** (3.5× on 6 cores): its parallel units are large blocks of work.
2. **Naive scales poorly beyond 4 threads** (1.75×): its passes over L² tensors are limited by memory bandwidth, and more
   cores do not add bandwidth.
3. **The Python tiled code gets slower with every added thread**, 4.8× slower at 6 threads than at 1. Its CPU time per
   call rose from 342 ms to 3,692 ms: the extra threads burned CPU synchronizing on operations too small to split, and
   under background load a single descheduled thread stalls every operation. This explains why the 4-thread runs showed
   tiled far slower than naive, while the 1-thread runs showed it slightly faster.
4. local_window's drop at 6 threads coincides with a large load increase, so it cannot be interpreted on its own.

## 15.5 Head dimension, batch and number format (E5)

Run: `20261011-222558_shapes_dtypes` (L = 1,024, H = 4, 1 thread; B ∈ {1, 4}; d ∈ {16, 32, 64, 128}; fp32 and bf16).
46 configurations measured; naive fwd+bwd at B = 4, d = 16 and 32 was skipped by the memory guard. Background load was
7–11 other logical CPUs throughout, so individual cells are noisy.

**Median forward time (ms), B = 1:**

| d | naive fp32 | naive bf16 | flash fp32 | flash bf16 | tiled fp32 | tiled bf16 |
|---|---|---|---|---|---|---|
| 16 | 102 | 1,009 | 15.0 | 74.8 | 60.1 | 44.4 |
| 32 | 74 | 1,230 | 18.2 | 64.1 | 52.4 | 75.0 |
| 64 | 128 | 1,587 | 77.1 | 119.0 | 165.9 | 80.2 |
| 128 | 132 | 3,152 | 69.8 | 168.6 | 185.7 | 198.3 |

**Median forward time (ms), B = 4:** naive fp32 332–561 vs **bf16 4,234–12,529**; flash fp32 57–284 vs bf16 312–514;
tiled fp32 266–866 vs bf16 324–922.

1. **bf16 is never faster than fp32 on this CPU, and for naive attention it is catastrophically slower**: 10–30× (for
   example 1,587 vs 128 ms at B = 1, d = 64, and 12.5 s vs 0.41 s at B = 4, d = 128). This AVX2 processor has no native bf16
   arithmetic, so every bf16 elementwise operation on the large L² tensors is converted to and from fp32.
2. **The fused kernel is 1.5–5.5× slower in bf16**, which is less severe because it converts small tiles internally.
3. **Our tiled code is roughly unaffected** (ratios 0.5–1.4×, within the noise), because it casts each tile to fp32
   accumulators anyway (Chapter 8).
4. B = 4 is 1.6–5.5× slower than B = 1 (about 4× expected for 4× the work), but the trend with `d` is buried in
   load noise. These data do not support a quantitative claim about time ∝ d.

On a GPU with tensor cores bf16 is usually *faster* than fp32, which is one more reason CPU conclusions must not be
transferred to GPUs.

## Key points to remember

* Approximate methods are linear in L but win only beyond about 2,048 tokens (local, block sparse); linear_elu wins everywhere here.
* At L = 8,192: local 6.2× and linear 41× faster than the fused exact kernel.
* They use 16–27× **more** memory than the fused exact kernel at 4,096. They save memory only against materializing methods.
* Tile size: U-shaped time (best 128), and autograd's training memory is about 77 MiB regardless; recompute is about 9 MiB.
* Threads: the fused kernel reaches 3.5× at 6 threads; Python tiled becomes 4.8× slower.
* bf16 on this AVX2 CPU: never faster; naive bf16 is 10–30× slower.
