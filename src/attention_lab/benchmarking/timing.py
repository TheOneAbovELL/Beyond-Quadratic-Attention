"""Wall-clock timing with warm-up, repetition and robust statistics.

* Timer: ``time.perf_counter_ns`` (monotonic, highest available resolution).
* The first call is timed separately as ``first_call_s`` (includes one-off
  costs: allocator growth, oneDNN primitive creation, lazy init).
* ``warmup`` untimed calls follow, then ``repeats`` timed calls.
* We report median, IQR, MAD, min, max and the coefficient of variation;
  the median is the headline number (robust to OS scheduling noise), the
  minimum is reported but never used alone.
* An optional time budget stops repetition early (recorded as ``repeats_done``).
* Process CPU time per call (``time.process_time`` over the whole timed loop
  divided by the number of calls; summed over all threads; Windows tick
  resolution ~15.6 ms, so only meaningful when the loop is >> 16 ms) is
  recorded as a secondary measure: it excludes time spent waiting for a
  core under background load, but includes OpenMP spin-waiting, so for
  multi-threaded runs it can exceed wall time even on an idle machine.
"""
from __future__ import annotations

import gc
import statistics
import time
from typing import Callable, Dict, List, Optional


def summarize(samples_s: List[float]) -> Dict[str, float]:
    s = sorted(samples_s)
    n = len(s)
    med = statistics.median(s)
    if n >= 4:
        q = statistics.quantiles(s, n=4, method="inclusive")
        q1, q3 = q[0], q[2]
    else:
        q1, q3 = s[0], s[-1]
    mad = statistics.median([abs(x - med) for x in s])
    mean = statistics.fmean(s)
    std = statistics.pstdev(s) if n > 1 else 0.0
    return {
        "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1, "mad_s": mad,
        "min_s": s[0], "max_s": s[-1], "mean_s": mean, "cv": std / mean if mean > 0 else 0.0, "n": n,
    }


def time_callable(
    fn: Callable[[], object],
    *,
    warmup: int = 2,
    repeats: int = 10,
    max_seconds: Optional[float] = None,
    min_repeats: int = 3,
) -> Dict[str, object]:
    gc.collect()
    t0 = time.perf_counter_ns()
    fn()
    first = (time.perf_counter_ns() - t0) / 1e9
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()  # avoid collector pauses inside timed regions
    try:
        budget_start = time.perf_counter()
        cpu_start = time.process_time()
        for i in range(repeats):
            t0 = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - t0) / 1e9)
            if max_seconds is not None and i + 1 >= min_repeats and time.perf_counter() - budget_start > max_seconds:
                break
        cpu_total = time.process_time() - cpu_start
    finally:
        if gc_was_enabled:
            gc.enable()
    out: Dict[str, object] = summarize(samples)
    out["cpu_time_per_call_s"] = cpu_total / len(samples)
    out["cpu_to_wall_ratio"] = cpu_total / sum(samples) if sum(samples) > 0 else None
    out.update(first_call_s=first, warmup=warmup, repeats_requested=repeats, repeats_done=len(samples),
               samples_s=samples)
    return out
