# Chapter 13 — The benchmarking system: measuring time and memory honestly

> **What you will learn:** how one benchmark configuration is run, step by step; how timing is done and
> summarized; the three memory instruments and what each one really measures (and how the main one was
> validated); the memory guard; isolated worker processes; the sweep scheduler; and the story of the shared
> machine, which led to the background-load monitor, CPU-time measurements and single-threaded runs.

## 13.1 One configuration, step by step

> **In the code:** `src/attention_lab/benchmarking/runner.py`, `run_one(BenchConfig)`.

A configuration is: method + parameters + `B, H, L, d` + dtype + causal + threads + warm-up/repeats + seed. The
runner executes, **in this order** (chosen so that the instruments do not contaminate each other):

1. **Theory** and **environment** are recorded (CPU, cores, RAM, OS, Python/torch/numpy versions, thread
   settings).
2. **Memory guard** (per mode): modes whose estimate exceeds the budget are skipped and recorded.
3. **Inputs** `q, k, v, dout ~ N(0, 1)` from a seeded generator.
4. **Timing** of each mode, wrapped in the background-load monitor:
   * `fwd`: forward under `torch.inference_mode()`;
   * `fwd_bwd`: forward with gradient-requiring inputs, then `out.backward(dout)`.
5. In isolated mode, the OS **peak working set** is read after each mode.
6. **Tensor peak** of each mode, from a separate untimed call under the profiler.
7. **Error** against the fp64 naive reference. This runs last, because the reference is the most memory-hungry step.

The result is one JSON record with every number and every setting.

## 13.2 Timing

> **In the code:** `src/attention_lab/benchmarking/timing.py`, `time_callable`.

* The first call is timed separately (`first_call_s`), because it includes one-off costs such as allocator
  growth and oneDNN setup.
* `warmup` untimed calls follow, then `repeats` timed calls with `time.perf_counter_ns()`, a monotonic,
  high-resolution clock.
* Python's garbage collector is paused during the timed loop.
* The summary reports the **median** (the headline number), q1, q3, IQR, MAD, min, max, mean and coefficient of
  variation, plus every raw sample. The minimum is never reported alone, because it describes the luckiest run
  rather than typical behaviour.
* A time budget (`max_seconds_per_mode`) stops repetition early for slow configurations, and the actual number
  of repeats done is recorded.
* **Process CPU time** per call is recorded too (explained in §13.7).

No test ever asserts a timing threshold. A test that fails because the machine is busy would teach nothing.
Correctness and performance are kept separate.

## 13.3 Three memory instruments

| Instrument | What it measures | What it misses |
|---|---|---|
| **Tensor peak** (`memory.tensor_peak_bytes`) | peak bytes allocated through PyTorch's CPU allocator *during one call*, reconstructed from profiler `[memory]` events | memory existing before the call (inputs); scratch memory allocated inside MKL/oneDNN/OpenMP outside PyTorch's allocator; Python objects |
| **Process peak Δ** (`memory.ProcessPeak`) | Windows' peak working set of the whole process minus its size before the first call | it is a *lifetime high-water mark*, so it needs a fresh process per configuration and the fwd+bwd value is really max(fwd, fwd+bwd); it includes allocator caches; page granularity |
| **Theory** (`complexity/theory.py`) | analytical estimate | everything not modeled; never reported as measured |

**How the tensor peak works.** The profiler with `profile_memory=True` records an event every time PyTorch's CPU
allocator hands out memory (`+bytes`) or takes it back (`−bytes`), with a timestamp. Sorting these events by time
and keeping a running sum gives the memory in use at each moment. Its maximum is the peak.

**How it was validated.** Before any result relied on it, it was tested on a known pattern: allocate 10 MiB,
allocate 5 MiB, free the 10 MiB, allocate 3 MiB, free everything. The true peak is 15 MiB (10 + 5), and the
instrument reported **exactly 15.0 MiB**, with a net of 0. Computing one 16 MiB score matrix from 1 MiB of inputs
reported 17.0 MiB. Both are now tests
(`test_tensor_peak_bytes_exact_on_known_allocation_pattern`, `test_tensor_peak_sees_materialized_scores_and_tiling_reduces_it`).

## 13.4 The memory guard

Before each mode, the runner compares the theoretical estimate with
`budget = min(mem_fraction × available RAM, max_mem_gb)` (defaults 50 % and a few GB). If forward fits but
forward+backward does not, it runs forward only and records `status = "partial"` with
`modes_skipped_memory`. If nothing fits, it records `status = "skipped_memory"` with the estimate and budget in the
reason.

> **What actually happened:** the first guard decided for the whole configuration using the
> forward+backward estimate. That would have skipped useful forward measurements at large L, so it was
> changed to decide **per mode**. Then the scheduler was briefly made to stop growing a series after a
> "partial" result. That was wrong, because forward-only may still fit at the next length, so it was
> reverted: a series stops only after a full `skipped_memory`. The guard did its job repeatedly. For example, at
> L = 4096 the naive and math methods (about 0.8 GB for forward alone) were skipped when only 0.6–0.8 GB was budgeted,
> and the free RAM varied from run to run because of other programs.

## 13.5 Isolated workers

The OS peak working set can only be reset by starting a new process, so the runner can execute a configuration in a
fresh Python process (`run_isolated` → `python -m attention_lab.benchmarking.worker`, config in on stdin, JSON record
out on stdout). The parent sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` **before** torch is imported in the child,
which controls threads more faithfully than `set_num_threads` alone. Isolation also protects the main process if a
configuration runs out of memory.

The cost is that starting Python and importing torch took about 5 s per worker on this machine.

> **What actually happened:** each worker initially also spent 1–2 s asking PowerShell for the CPU name. The parent
> now passes the name through an environment variable (`ATTENTION_LAB_CPU_NAME`).

## 13.6 The scheduler

> **In the code:** `src/attention_lab/benchmarking/scheduler.py`; specs in `experiments/configs/*.yaml`.

A YAML spec lists methods (with params), a grid (`B, H, L, d, dtype, causal, threads`), repeats, budgets and a
**hypothesis** sentence. The scheduler:

* expands the grid and **sorts by L**, so every method is measured small before anything large runs;
* runs configurations one at a time (in-process or isolated);
* **appends each result to `results.jsonl` immediately**, so an interrupted sweep keeps everything measured so far;
* stops growing a series (same method/shape, larger L) after a memory skip, an error, a timeout, or a median time above
  `stop_if_slower_than_s`, recording later configurations as `skipped_after_budget` with the reason;
* optionally waits, for a bounded time, for the machine to go quiet before each configuration (`wait_for_quiet`).

## 13.7 The shared-machine story

> **What actually happened.** The first full exact-scaling sweep ran with 4 threads in isolated workers. Its numbers
> looked wrong. At L = 512, `tiled` and `tiled_recompute`, whose forward passes run *identical* code, measured
> 13.1 ms and 34.7 ms. Causal tiled (half the work) was slower than non-causal tiled. A quick check showed
> **all 12 logical CPUs at 100 %**, while the benchmark used only 4 threads.
>
> Listing the Python processes showed why: **workloads from other projects** were running at the same time (three
> training scripts, another Transformer experiment, a chess-engine server with its engine processes, and a test run).
> They were not ours, so they were not touched. But they made fine-grained timing unreliable: a 4-thread
> OpenMP computation competing with up to 11 busy logical CPUs is slowed by different amounts for different methods.

The response, in steps:

1. **Measure the load.** `benchmarking/load.py` adds a `LoadMonitor`: a background thread samples system CPU use and
   this process's CPU use every 0.2 s during each timed region, and records `other_load_cores` (how many logical CPUs
   *other* processes kept busy). Records with more than 1 other busy core are flagged `contended`.
   *Bug found while testing it:* short timed regions finished before the first sample, giving no data. Fixed by
   priming the counters on entry and always taking a final sample on exit.
2. **Record CPU time.** Process CPU time per call excludes time spent waiting for a core.
   *Bug found:* on Windows, `time.process_time` ticks in about 15.6 ms steps, so per-call samples of fast calls read
   0.0. Fixed by measuring CPU time over the whole timed loop and dividing by the number of calls.
3. **Wait for quiet, within limits.** `wait_for_quiet` waits up to a timeout for the load to drop. The load stayed
   high for hours, so the timeout was reduced from 60 s to 10 s per configuration; the load is recorded either way.
4. **Run the definitive timing sweeps single-threaded.** With one thread, no OpenMP barrier can stall on a
   descheduled sibling thread, so contention degrades every method more evenly. The check that this worked: CPU time per
   call matched wall time within a few percent (for example 509 ms CPU vs 516 ms wall), and the fitted exponents became
   physically sensible (1.92–2.03, instead of up to 3.0 in the contaminated run).
5. **Never mix contended and clean claims.** The first sweep's *memory* data is used (memory is not affected by CPU
   contention). Its *timing* data is reported as contaminated and not used as evidence.

How noisy is the machine? The same naive forward at L = 2048 on one thread measured **392 ms in one sweep and 784 ms
in another** an hour later. The book therefore compares methods only *within* a sweep (measured close together in
time), and states background load next to every timing table.

## 13.8 What a GPU benchmark would need

None of this is GPU-specific, and adding a GPU would mean: synchronizing the device before and after each timed call
(GPU work is asynchronous), measuring peak memory with `torch.cuda.max_memory_allocated` after resetting it, warming
up kernel autotuning, and recording the GPU model and driver. CPU and GPU numbers must never share a plot without
labels. The algorithms themselves need no changes.

## Key points to remember

* Order of measurement: guard → inputs → timing (+ load) → process peak → tensor peak → error.
* Median + IQR, warm-up, monotonic clock, GC paused, raw samples kept, and no timing assertions in tests.
* Three memory instruments with different scopes. The tensor peak was validated byte-exactly (15.0 MiB on a 15 MiB pattern).
* The guard decides per mode and records every skip.
* Isolated workers give clean process-peak numbers and set threads before torch loads (about 5 s overhead each).
* The machine was shared. The load was measured, CPU time recorded, definitive sweeps run on 1 thread, and contaminated timings discarded.
