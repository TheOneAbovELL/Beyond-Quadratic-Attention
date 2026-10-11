"""Background-load monitor: how much CPU did *other* processes use while we were timing?

Why this exists: this lab runs on a shared desktop. During development we
observed unrelated workloads occupying 5-12 logical CPUs while benchmarks ran, which
made identical code differ by 2-3x in wall time. Timing records therefore
carry the measured background load, and reports flag contended runs instead
of silently mixing them with clean ones.

Method: a daemon thread samples every ``interval`` seconds
  system busy logical CPUs = psutil.cpu_percent() / 100 * n_logical
  our busy logical CPUs    = this_process.cpu_percent() / 100
  other = max(0, system - ours)
Both counters are deltas since the previous sample, so each sample covers
one interval. Counters are primed on entry and a final sample is taken on
exit, so even a timed region shorter than ``interval`` gets one sample.
"""
from __future__ import annotations

import statistics
import threading
from typing import Dict, List, Optional

import psutil

CONTENDED_THRESHOLD_CORES = 1.0


class LoadMonitor:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()
        self._n = psutil.cpu_count(logical=True) or 1

    def _sample(self) -> None:
        system = psutil.cpu_percent(interval=None) / 100.0 * self._n
        ours = self._proc.cpu_percent(interval=None) / 100.0
        self.samples.append(max(0.0, system - ours))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()
        self._sample()  # final partial interval, so short timed regions still get one sample

    def __enter__(self) -> "LoadMonitor":
        psutil.cpu_percent(interval=None)        # prime both counters at the start of the region
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.samples:
            return {"other_load_cores_mean": None, "other_load_cores_max": None, "load_samples": 0}
        return {
            "other_load_cores_mean": statistics.fmean(self.samples),
            "other_load_cores_max": max(self.samples),
            "load_samples": len(self.samples),
        }


def snapshot(seconds: float = 1.0) -> float:
    """Busy logical CPUs system-wide over ``seconds`` (includes the caller)."""
    n = psutil.cpu_count(logical=True) or 1
    return psutil.cpu_percent(interval=seconds) / 100.0 * n
