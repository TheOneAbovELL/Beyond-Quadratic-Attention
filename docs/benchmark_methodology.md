# Benchmark methodology

Code: `benchmarking/` (runner, timing, memory, load, scheduler). Specs:
`experiments/configs/*.yaml`. Raw results: `experiments/results/<run>/results.jsonl`
(one JSON object per configuration, including failures). Reports:
`experiments/reports/<run>/`.

## 1. What one configuration measures

A configuration is (method, params, B, H, L, d, dtype, causal, threads, seed).
`runner.run_one` executes, in this order:

1. **Memory guard** (per mode). The theoretical peak estimate (`complexity/theory.py`)
   is compared with `min(mem_fraction × available RAM, max_mem_gb)`. Modes that
   do not fit are recorded as skipped (`status = partial` or `skipped_memory`),
   never silently dropped.
2. **Inputs**: `q, k, v, dout ~ N(0, 1)` from a seeded generator, cast to the dtype.
3. **Timing per mode** (`timing.time_callable`):
   * `fwd`: forward under `torch.inference_mode()` (no autograd graph);
   * `fwd_bwd`: forward with `requires_grad` inputs + `out.backward(dout)`.
   * The first call is timed separately (`first_call_s`: one-off allocation and
     oneDNN primitive creation). Then come `warmup` untimed calls and `repeats` timed
     calls using `time.perf_counter_ns` (monotonic).
   * The Python GC is disabled inside the timed loop.
   * Reported: median (headline), q1, q3, IQR, MAD, min, max, mean, CV, and every raw
     sample. The minimum is never reported on its own.
   * A `LoadMonitor` thread samples how many logical CPUs *other* processes kept busy.
4. **Process peak** (isolated mode only): OS peak working set after each mode, minus
   RSS before the first call.
5. **Tensor peak** per mode: a separate, untimed call under the profiler (see §3).
6. **Error**: output vs naive float64 (exact methods → `error_vs_reference_*`;
   approximate methods → `deviation_from_full_softmax_*`, a different quantity).
   This runs last because the fp64 reference is the most memory-hungry step.

Each record also stores the full config, the environment (CPU, cores, RAM, OS,
Python/torch/numpy versions, CPU capability, MKL/oneDNN, thread env vars), the
theoretical estimates, the guard's numbers, and a timestamp.

## 2. Controls

| Confounder | Control |
|---|---|
| warm-up effects (allocator growth, lazy init, page faults) | first call reported separately + warm-up calls |
| OS scheduling noise | median and IQR over repeats; CV reported |
| different conditions between methods | same seed, shapes, dtype, mask, thread count; configs at the same L run back-to-back (scheduler sorts by L) |
| thread count | `torch.set_num_threads(threads)` in-process; `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set for isolated workers *before* torch is imported |
| leftover memory from previous configs | `isolate: true` runs each config in a fresh process |
| other processes on the machine | measured (`other_load_cores`, `contended` flag); optional `wait_for_quiet` before each config |
| RAM exhaustion | per-mode memory guard; small-L-first scheduling; series stop after a memory skip |
| runaway time | `max_seconds_per_mode` caps repeats; series stop when median > `stop_if_slower_than_s` |

## 3. Memory: three different quantities

| Field | Scope | Caveats |
|---|---|---|
| `{mode}_tensor_peak_bytes` | peak bytes allocated through PyTorch's CPU allocator *during* one call, from profiler `[memory]` events replayed in time order | excludes pre-existing inputs; excludes MKL/oneDNN/OpenMP internal scratch; includes the output tensor. Validated byte-exact on known allocation patterns (`test_tensor_peak_bytes_exact_on_known_allocation_pattern`) |
| `proc_peak_delta_after_{mode}_bytes` | OS peak working set minus RSS before the first call (isolated runs) | cumulative high-water mark, so the `fwd_bwd` value is the max over everything so far; page-granular; includes allocator caching and library scratch; Windows `peak_wset` |
| `theory_*_bytes` | analytical estimate | never presented as a measurement |

A theoretical allocation estimate is **never** reported as measured peak memory.
When the tensor peak and the theory agree in scaling but differ by a constant,
the difference is usually temporaries the theory does not model, such as a
`masked_fill` copy or the bool mask.

## 4. Variability and what counts as a difference

* Rule of thumb used in reports: two medians differ meaningfully only if their
  IQRs do not overlap **and** both runs are uncontended.
* Scaling exponents are least-squares slopes of `log t` vs `log L` over `L ≥ 512`
  (`fit_exponent`). Below that, fixed overheads dominate and bend the curve.
* No test asserts a timing threshold. Correctness assertions and performance
  measurements are kept separate (`tests/perf_smoke` checks only that runs complete
  and records are well-formed).

## 5. Scheduler

`scheduler.run_sweep` expands the YAML grid, sorts configs by L (small first),
writes `meta.json` (spec + environment), and appends each result to
`results.jsonl` immediately, so a crash keeps everything measured so far. A
series (same method/params/shape except L) stops growing after a memory skip,
an error, a timeout, or a median above `stop_if_slower_than_s`. The remaining
configs are written as `skipped_after_budget` with the reason.

## 6. Adding GPU measurements later

For CUDA: call `torch.cuda.synchronize()` before starting and after stopping each
timer (or use CUDA events), call `torch.cuda.reset_peak_memory_stats()` before
and read `torch.cuda.max_memory_allocated()` after each call, warm up kernels and
autotuning before timing, and record the GPU model, driver and CUDA versions in
`environment_info`. CPU and GPU results must never share a plot axis without a
device label.
