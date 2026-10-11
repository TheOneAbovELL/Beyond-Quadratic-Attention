# Systems guide: why equal complexity does not mean equal speed

## The machine (measured, `python lab.py env` + Win32_CacheMemory)

* AMD Ryzen 5 7530U (Zen 3): 6 cores / 12 hardware threads, AVX2 (PyTorch reports
  `cpu capability AVX2`; no AVX-512 and no native bf16 dot-product instructions).
* Caches: L1 384 KB total (32 KB data per core), L2 3 MB (512 KB per core),
  **L3 16 MB shared**.
* 16 GB RAM, typically only 1–2 GB *available* during this work, because other
  applications were running.
* PyTorch 2.10 CPU build: MKL and oneDNN for matmuls, OpenMP intra-op threads
  (default 6).

## 1. Where the time goes on a CPU

An attention call is a sequence of PyTorch operators. Each one:

1. **is dispatched** from Python through the PyTorch dispatcher, which costs a few
   microseconds of fixed overhead (argument parsing, dtype/device checks, output
   allocation, autograd bookkeeping if gradients are on);
2. **allocates its output**, a fresh tensor from the CPU allocator;
3. **streams its inputs and output through the cache hierarchy.** A [1·4·1024·1024]
   fp32 score tensor is 16 MiB, which already fills L3. Every elementwise pass over
   it (`* scale`, `masked_fill`, `- m`, `exp`, `/ l`) reads and writes 16 MiB from
   DRAM.

Matmuls (`QKᵀ`, `PV`) are compute-bound and well optimized by MKL/oneDNN. The
elementwise and reduction passes do almost no arithmetic per byte, so they are
**memory-bandwidth-bound**. In naive attention, those passes over `L²` tensors cost
a large share of the time at moderate L, even though they are "only" `O(L²)`
versus the matmuls' `O(L²d)`.

## 2. Fusion: the single most important optimization

`sdpa_flash_cpu` (ATen's `_scaled_dot_product_flash_attention_for_cpu`) computes
the same `4L²d` FLOPs as the naive code, but:

* it processes a block of queries against a block of keys **inside one C++ kernel**:
  scale, mask, max, exp, sum and the `P·V` accumulation all happen on a tile that
  stays in L1/L2;
* it never writes `S` or `P` to DRAM, and it allocates only the output and `lse`;
* it parallelizes over (batch, head, query-block) with OpenMP;
* it uses vectorized `exp` and register-level accumulation.

That is the CPU analogue of FlashAttention's design. **The speedup over naive is
a memory-traffic and overhead win, not an arithmetic win.** The measured tensor
peaks show the storage side directly: naive allocates `O(L²)` bytes per call,
flash-CPU allocates `O(L·d)`.

## 3. Why the Python tiled implementation is slow

`tiled.py` implements the *same algorithm* as the fused kernel, yet it is the
slowest exact method here. Reasons:

* **Dispatch overhead × number of tiles.** About 10 ops per tile ×
  `(L/128)²` tiles: at `L = 4096`, that is 1024 tiles → ~10k operator dispatches
  per head-batch, compared with ~6 for naive.
* **No fusion.** Each line `exp(s − m)`, `α·l`, `P @ V`… is a separate kernel
  that materializes its result. The *tile* stays small enough for L2, but every
  intermediate is still written and read once.
* **Small matmuls.** A `128×64 @ 64×128` matmul is far below the size at which
  MKL reaches peak throughput, and thread-parallel efficiency is poor at that size.
* **Autograd graph.** With gradients enabled, every tile adds nodes and saved
  tensors to the graph (and the backward replays all of them).

The asymptotic auxiliary memory is better, and the measured tensor peaks confirm
it. The wall-clock is worse. This is the central lesson: **an algorithm's memory
complexity is realized only if the implementation also controls *where* the data
lives and *how many times* it moves.**

## 4. Recompute vs store on this CPU

`tiled_recompute` saves `O(L·d)` instead of `O(L²)` for backward and pays one extra
`QKᵀ` per tile. On a GPU, recomputing from SRAM-resident tiles is cheaper than
re-reading `P` from HBM. On this CPU, with a Python loop, the extra loop
iterations and ops cost time, but the reduced allocation volume also reduces
DRAM traffic. The measurement decides which effect wins; see the results document.

## 5. Approximate methods: fewer FLOPs is not automatically faster

* `local_window` builds padded, overlapping key windows with `unfold`. When the
  matmul consumes them, the overlapping view is materialized:
  `(C + 2w)/C` times the key bytes. About half of the logits computed are masked.
  It is fully vectorized (no Python loop), so its overhead is constant per call.
* `block_sparse` *gathers* the active key/value blocks for each query block (a
  copy of `L·K_max·b·d` elements), pads rows to `K_max`, and computes global rows
  separately. Gather and scatter are memory-bound, irregular operations.
* `linear_elu` replaces the `L×L` matmuls by `d×d` state updates. Its elementwise
  feature map and `cumsum` over chunk states are cheap but not free.

Whether these beat a fused `O(L²)` kernel depends on L. Below some crossover
length, the fused kernel's efficiency outweighs the approximate method's smaller
FLOP count. The benchmark sweep (`scaling_approx.yaml`) locates that crossover on
this machine.

## 6. Threads

Large matmuls and fused kernels scale with intra-op threads up to the number of
physical cores. SMT threads share a core's execution units, which helps only with
latency-bound work. Python-loop code with small ops scales poorly: the
interpreter is serial, and each small op has too little work to split
(`threads.yaml` measures this).

**Shared machine caveat.** During this project, unrelated workloads on the same machine
kept between 5 and all 12 logical CPUs busy for long periods. Each timing record therefore carries
`other_load_cores` (`benchmarking/load.py`). A 4-thread benchmark competing with
6 busy cores on a 6-core CPU can easily run 2–3× slower, and that slowdown is not
uniform across methods.

## 7. CPU vs GPU: what transfers and what does not

| Aspect | Transfers to GPU reasoning? |
|---|---|
| Exactness and numerics of online softmax | Yes — the same math |
| `O(L²)` vs `O(L·d)` storage | Yes |
| Fused vs unfused gap | Qualitatively yes; the magnitude differs (HBM vs DRAM bandwidth, SRAM size, kernel-launch cost ~µs per kernel on GPU) |
| Python tiled being slow | Partially: on a GPU, per-op launch overhead is even more punishing |
| bf16 speed | **No.** This AVX2 CPU has no native bf16 math, while GPUs (tensor cores) are much faster in bf16 |
| Crossover lengths for sparse/linear methods | **No.** They depend on the hardware's compute-to-bandwidth ratio and kernel quality |

GPU experiments can be added by extending `benchmarking/runner.py`: synchronize
the device before and after each timed call, record `torch.cuda.max_memory_allocated`
after `reset_peak_memory_stats`, and register CUDA-only backends in the registry
with their own `dtypes`. No algorithm code depends on the device.
