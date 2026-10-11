"""Memory instruments. Each measures something different; none is "the" memory.

1. ``tensor_peak_bytes`` (framework allocations, MEASURED)
   Runs a callable under ``torch.profiler`` with ``profile_memory=True`` and
   replays the CPU allocator's [memory] events (+bytes on alloc, -bytes on
   free) in timestamp order. The peak of the running sum is the peak of
   *tensor bytes allocated through PyTorch's CPU allocator during the call*,
   relative to the start. It excludes memory that existed before the call
   (inputs), scratch buffers allocated internally by MKL/oneDNN/OpenMP
   outside the c10 allocator, Python object overhead and allocator caching.
   Validated in tests against known allocation patterns (exact to the byte).

2. ``ProcessPeak`` (OS working set, MEASURED, coarse)
   On Windows, psutil exposes the process *peak working set* (``peak_wset``),
   a monotone high-water mark maintained by the OS. It can only be reset by
   starting a new process, so it is meaningful only in an isolated worker
   process that runs a single configuration. Includes everything the process
   touched: interpreter, libraries, allocator caches, page-granular.

3. Theoretical estimates live in ``complexity.theory`` and are never stored
   under a "measured" field.
"""
from __future__ import annotations

import sys
import warnings
from typing import Callable, Dict, Optional

import psutil
import torch
from torch.profiler import ProfilerActivity, profile


def tensor_peak_bytes(fn: Callable[[], object]) -> Dict[str, int]:
    """Peak and net bytes allocated by PyTorch's CPU allocator while running ``fn``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
            result = fn()
            del result
    events = [
        e for e in prof.profiler.kineto_results.events()
        if e.name() == "[memory]" and e.device_type() == torch.autograd.DeviceType.CPU
    ]
    events.sort(key=lambda e: e.start_ns())
    cur = peak = 0
    for e in events:
        cur += e.nbytes()
        peak = max(peak, cur)
    return {"peak_bytes": int(peak), "net_bytes": int(cur), "n_events": len(events)}


class ProcessPeak:
    """Process-level resident memory: current RSS and (Windows) monotone peak working set."""

    def __init__(self) -> None:
        self.proc = psutil.Process()

    def rss(self) -> int:
        return int(self.proc.memory_info().rss)

    def peak(self) -> Optional[int]:
        mi = self.proc.memory_info()
        if hasattr(mi, "peak_wset"):
            return int(mi.peak_wset)
        if sys.platform != "win32":
            try:
                import resource

                return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # KiB on Linux
            except Exception:
                return None
        return None
