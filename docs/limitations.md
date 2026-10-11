# Limitations

## Hardware and environment
* **CPU only.** There is no CUDA device. Nothing here measures GPU behaviour. Crossover
  lengths, dtype speed and fused-vs-unfused ratios are specific to this AMD Ryzen 5
  7530U (Zen 3, AVX2, 16 MB L3) and PyTorch 2.10 CPU build.
* **Shared, memory-constrained desktop.** While these experiments ran, other
  workloads (training jobs and servers from unrelated projects) kept up to all 12
  logical CPUs busy, and available RAM was often below 1 GB. Consequences:
  * timings carry measured background load (`other_load_cores`, `contended`) and must
    be read with it. Contended timings are biased upward by unequal amounts per
    method;
  * the memory guard skipped the largest configurations (e.g. naive/sdpa_math at
    L=4096, the tiled autograd backward at L=4096). Those points are *missing*, not
    estimated;
  * the absolute numbers would differ on an idle machine. Ratios measured at the same
    moment, and memory measurements, are more trustworthy than absolute times. Example: naive
    forward at L=2048 (1 thread, identical config) measured 392 ms in one sweep and 784 ms in another
    an hour later;
  * the definitive timing sweeps were therefore run **single-threaded** (fewer OpenMP stalls under
    contention, and CPU time per call stays close to wall time). Multi-threaded behaviour comes from the
    thread sweep and the contended first sweep, and is less certain.
* The OneDrive folder that holds this project may be synced by the user's OneDrive
  client. The lab itself makes no network calls.

## Implementations
* `tiled` / `tiled_recompute` are **pedagogical** Python-loop implementations of the
  FlashAttention *algorithm*. They do not reproduce its hardware-level optimizations
  (on-chip residency, fusion, warp scheduling). Measured, they are about 2–3× slower than the fused CPU
  kernel. Single-threaded they match or beat naive; multi-threaded they lose to it (`results.md` §2b).
* `sdpa_flash_cpu` is PyTorch's CPU flash kernel, treated as a black box. Its
  internal block sizes and scratch memory are not visible to our instruments (the
  tensor-peak instrument sees only allocations through PyTorch's allocator).
* `block_sparse` uses gather/pad execution. A real block-sparse kernel (e.g. Triton
  or custom C++) would have very different constants. Global rows are computed densely.
* `linear_elu` is one feature map. Random-feature approximations of softmax
  (Performer) and gated or decayed linear recurrences are not implemented, so no claim
  is made about "linear attention" in general.
* Masks: the approximate methods support `causal` and `key_padding_mask`, but not
  arbitrary dense masks. `causal=True` is top-left aligned (not the KV-cache
  convention). Dropout inside attention is not implemented.
* No KV-cache / incremental-decoding benchmark.

## Measurement
* Tensor peak excludes library-internal scratch (MKL/oneDNN/OpenMP). Process peak is
  a cumulative high-water mark and page-granular. Neither is "true peak memory" on
  its own.
* Windows `process_time` has ~15.6 ms resolution. CPU-time figures are loop
  averages and are unreliable for very short loops.
* Each timing configuration is one process run, so run-to-run (as opposed to
  repeat-to-repeat) variance is not estimated except where an experiment was
  repeated.
* Error-vs-reference is measured on random Gaussian inputs. Adversarial or
  structured inputs can behave differently (the magnitude sweep covers one axis of this).

## Quality experiments
* MQAR key-value binding was not learned by any backend within the CPU budgets tried. The cause is
  diagnosed down to "MQAR-specific learning dynamics or insufficient budget", not implementation
  (`mqar_diagnosis.md`). MQAR results are not used to compare backends.
* At L=32 with block 8, the block-sparse layout covers every causal pair, so block_sparse cells for the
  L=32 tasks are full attention, not evidence about sparsity (flagged in `results.md`).
* Synthetic tasks test specific capabilities (local copying, associative recall,
  interference, length extrapolation). They are **not** evidence about language
  modelling quality.
* Trained comparisons use one seed, small models (2 layers, d_model 64) and short
  CPU budgets. Small accuracy differences are within seed noise. The approximate
  methods' hyperparameters (window 8, block 8) were chosen to make the restriction
  bite at L = 64, not tuned for performance.
* Probes test a single attention operation with hand-constructed inputs. A trained
  multi-layer model can route information around a restriction (e.g. relaying
  through intermediate tokens), which probes deliberately do not capture.
