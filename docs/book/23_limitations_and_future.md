# Chapter 23 — Limitations, open questions and what to do next

> **What you will learn:** exactly what this project cannot tell you, which questions remain open, and the most
> valuable next experiments, in priority order.

## 23.1 What the results do not establish

**Hardware scope.** Every measurement is from one CPU (Ryzen 5 7530U, AVX2, 16 MB L3) running PyTorch 2.10. GPUs have
different memory hierarchies, launch overheads and bf16 hardware, so crossover lengths, dtype speeds and the fused vs
unfused ratios do **not** transfer. What does transfer: exactness, the storage-scaling laws, the store-vs-recompute
logic and the numerical findings about accumulator precision.

**Measurement conditions.** The machine was shared and often fully loaded. Timings carry measured background load; the
same configuration varied up to 2× between sweeps an hour apart. Ratios measured within one sweep are the reliable
quantities. Run-to-run variance was not estimated by repeating whole sweeps.

**Memory instruments.** The tensor peak sees only PyTorch-allocator memory, not MKL/oneDNN/OpenMP scratch. The
process peak is coarse and cumulative. Memory *traffic* was reasoned about but never measured (that needs hardware
performance counters).

**Implementations.** The Python tiled kernels illustrate the algorithm, not peak performance. The block-sparse
implementation uses gather/pad, and a dedicated kernel would have different constants. Only one linear-attention
feature map was tested. Approximate methods support causal and padding masks but not arbitrary dense masks. There is
no KV-cache decoding benchmark and no attention dropout.

**Quality experiments.** Synthetic tasks with tiny models, single seeds and short CPU budgets: evidence about
*mechanisms*, not about language-model quality. Small accuracy differences are within seed noise. At L = 32, the
block-sparse layout was dense.

## 23.2 Open questions

1. **MQAR binding.** Why does the model learn content-based induction in 500 steps but not MQAR binding in 3,000?
   Candidates: sparser supervision; the "uniform over context values" solution acting as an attractor; budget.
2. **Distractor brittleness.** Do the induction models really locate the segment by its noise boundary? An attention
   probe on the trained checkpoints could test this directly.
3. **Linear attention's better out-of-distribution behaviour** in Task C/D: real, or seed noise? It needs several seeds.
4. **Where exactly the CPU time goes** in naive vs fused attention: needs hardware counters (cache misses, bandwidth).

## 23.3 The most valuable next experiments

1. **Multiple seeds for the trained tasks** (3–5 seeds), to put error bars on Chapter 18.
2. **A clean timing re-run on an idle machine**, with whole-sweep repetition to measure run-to-run variance.
3. **The MQAR follow-ups** listed in Chapter 19: denser supervision, a curriculum, and longer training with circuit probes.
4. **A GPU run** with the device-aware additions of Chapter 13 §13.8, if hardware becomes available, to compare
   crossover lengths across hardware.
5. **A compiled tiled kernel** (`torch.compile`, or a small C++ extension), to separate "algorithm" from "Python
   dispatch" in the tiled-vs-fused gap.

## Key points to remember

* CPU-only, shared machine, single-seed tiny models: read every result within those limits.
* Within-sweep ratios and memory measurements are the most trustworthy results.
* Open: MQAR binding, distractor brittleness, linear attention's OOD hint, hardware-counter profiles.
* Next: seeds, a clean re-run, MQAR follow-ups, GPU, a compiled tiled kernel.
