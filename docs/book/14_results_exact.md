# Chapter 14 — Results I: the exact methods (correctness, memory, time)

> **What you will learn:** what the experiments showed about the five exact implementations: that they agree
> numerically; how their memory scales (measured); how their time scales (measured on one thread); and how
> to read these results without over-reading them.

*Conditions for everything in this chapter:* B = 1, H = 4, d = 64, fp32, non-causal unless stated. Runs:
`20261011-191952_scaling_exact` (memory, isolated workers, 4 threads, timing contaminated and not used) and
`20261011-210946_scaling_exact_timing` (timing, 1 thread, in-process; background load 4.5–10.6 other logical CPUs).
Figures are in `experiments/reports/<run>/`.

## 14.1 They compute the same function

From Chapter 11: all exact methods agree with the fp64 reference to about 10⁻¹⁵ in fp64 and pass every fp32
check (forward and gradients) across 57 cases each. In the scaling run, the fp32 output error against fp64 stayed
between 6 × 10⁻⁷ and 2 × 10⁻⁶ from L = 128 to 2048. It does not grow with length.

## 14.2 Memory: measured peak tensor allocation per call

| L | naive fwd | sdpa_math fwd | flash fwd | tiled fwd | naive fwd+bwd | flash fwd+bwd | tiled (autograd) fwd+bwd | tiled_recompute fwd+bwd |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.88 | 0.69 | 0.22 | 1.13 | 1.75 | 0.75 | 1.13 | 2.01 |
| 512 | 12.5 | 9.5 | 1.07 | 1.78 | 25.0 | 3.51 | 7.03 | 3.78 |
| 1024 | 49.0 | 37.0 | 3.27 | 2.54 | 98.0 | 9.02 | 22.0 | 5.81 |
| 2048 | 194.1 | 146.0 | 4.29 | 4.57 | 388.0 | 14.0 | 76.4 | 9.88 |
| 4096 | skipped | skipped | 6.32 | 8.63 | skipped | 24.1 | skipped | 18.0 |

(MiB. "skipped" = refused by the memory guard at the time.)

**Fitted exponents** of memory vs L (L ≥ 512): naive 1.98 / 1.98 (fwd / fwd+bwd); sdpa_math 1.97; flash 0.81 /
0.90; **tiled (autograd) 0.77 / 1.72**; **tiled_recompute 0.84 / 0.75**.

What this shows:

1. **Materializing methods are quadratic in memory, exactly as predicted.** The theory said 0.875 / 12.5 / 49.0 /
   194 MiB for naive; the measurements were 0.88 / 12.5 / 49.0 / 194.1.
2. **Tiling and fusion make forward memory almost linear:** 4.3–4.6 MiB at L = 2048, against 194 MiB for naive, a
   42–45× reduction.
3. **Autograd through the tile loop brings quadratic memory back for training** (exponent 1.72, 76 MiB at 2048).
4. **The recompute backward keeps training memory linear** (0.75; 9.9 MiB at 2048, 39× less than naive).
5. At the process level (OS peak working set, isolated workers), naive needed 446 MiB more than its starting size at
   L = 2048 for fwd+bwd, against 57 MiB for the flash kernel. About 35–40 MiB of that is fixed overhead, visible even
   at L = 128.

Figure 14.1 (below) shows all methods on one log-log plot against dashed L and L² guides.
Naive and math track the L² line; flash and recompute track the L line; autograd-tiled bends toward L² in
fwd+bwd.

![Figure 14.1 — Measured peak tensor memory per call vs L, all exact methods on shared log-log axes (run 20261011-191952_scaling_exact). Dashed grey lines are theory slopes ∝ L and ∝ L².](../../experiments/reports/20261011-191952_scaling_exact/memory_compare_noncausal.png)

## 14.3 Time: measured on one thread

Median milliseconds per call:

| L | naive fwd | sdpa_math fwd | flash fwd | tiled fwd | recompute fwd | naive fwd+bwd | math fwd+bwd | flash fwd+bwd | tiled fwd+bwd | recompute fwd+bwd |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 4.1 | 5.3 | 2.6 | 5.1 | 5.5 | 16.0 | 10.5 | 10.7 | 19.6 | 20.1 |
| 512 | 28.1 | 23.6 | 8.4 | 21.8 | 23.3 | 82.4 | 54.7 | 36.4 | 75.1 | 68.7 |
| 1024 | 105.5 | 86.5 | 31.5 | 73.4 | 87.7 | 304.4 | 210.0 | 135.9 | 353.1 | 268.7 |
| 2048 | 392.4 | 355.6 | 146.7 | 313.0 | 345.0 | 1411.2 | 803.7 | 503.9 | 1847.0 | 979.9 |
| 4096 | 1540.4 | 1512.9 | 515.6 | 1445.4 | 1600.3 | (guard) | (guard) | 2017.0 | 14489.5 | 4637.2 |

**Fitted exponents of time vs L** (wall clock / CPU time): every exact method's forward is 1.92–2.03 (CPU
1.97–2.10). Forward+backward is 1.91–2.05 for all of them except **autograd-tiled at 2.52**.

What this shows:

1. **Every exact method is quadratic in time, including the memory-efficient ones.** Tiling moves the scores into
   small tiles; it does not reduce how many scores there are.
2. **The fused CPU kernel is the fastest exact method at every length**: about 3× faster than naive in forward
   (516 vs 1,540 ms at L = 4096) and 2.8× in fwd+bwd at 2048. Same arithmetic, so the gain is fusion (no `L²` tensor
   ever written to RAM) and no Python dispatch per step.
3. **On one thread, our Python tiled forward was *not* slower than naive.** It was 6–30 % faster at L ≥ 512 (313 vs
   392 ms at 2048). Its per-tile dispatch overhead is offset by avoiding three `L²` tables that overflow the 16 MB L3
   cache. *This depends on threads:* in the saved 4-thread runs, tiled was much slower than naive (6.3 vs 0.7 ms at
   L = 256 in the smoke run; 13.1 vs 5.2 ms at 512 in the contended run), and Chapter 15's thread experiment explains
   why.
4. **Storing costs more than recomputing.** tiled_recompute does 25 % more backward arithmetic than autograd-tiled,
   yet fwd+bwd took 980 ms vs 1,847 ms at 2048, and 4.6 s vs 14.5 s at 4096. Autograd-tiled's super-quadratic exponent
   (2.52) indicates growing memory-system cost from keeping every tile alive.
5. **Fusion still wins over the best Python version:** flash fwd+bwd 504 ms vs recompute 980 ms at 2048. The algorithm
   is the same; the execution is not.

![Figure 14.2 — Measured median time per call vs L on one thread (run 20261011-210946_scaling_exact_timing), with IQR bars. Every exact method runs parallel to the L² guide.](../../experiments/reports/20261011-210946_scaling_exact_timing/runtime_vs_L.png)

## 14.4 How *not* to read these results

* They are **CPU results**. A GPU has different cache sizes, bandwidth, launch costs and bf16 support, and the ratios
  would differ.
* The absolute times include background load (flagged `contended`). Ratios measured in the same sweep, close together
  in time, are more trustworthy than absolute numbers.
* "Tiled ≈ naive on one thread" is a statement about *this* Python implementation, *this* tile size (128) and *one*
  thread, not about tiling in general.

## Key points to remember

* All exact methods agree numerically (≈ 1e-15 fp64, ≈ 1e-6 fp32).
* Memory: naive/math ∝ L² (194 MiB at 2048); flash/tiled ∝ L (≈ 4.5 MiB); training: autograd-tiled ∝ L^1.72, recompute ∝ L^0.75.
* Time: everything exact ∝ L² (1.9–2.1).
* Fused kernel ≈ 3× faster than naive; recompute faster than storing (980 vs 1,847 ms).
* Single-thread Python tiled ≈ naive or better; multi-thread it is much worse (Chapter 15).
