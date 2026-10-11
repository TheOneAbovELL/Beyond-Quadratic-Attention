"""Environment capture: recorded with every experiment so results are attributable."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict

import psutil
import torch


def _cpu_name() -> str:
    if os.environ.get("ATTENTION_LAB_CPU_NAME"):          # set by the parent for isolated workers
        return os.environ["ATTENTION_LAB_CPU_NAME"]
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


_CACHED: Dict[str, Any] | None = None


def environment_info(refresh: bool = False) -> Dict[str, Any]:
    global _CACHED
    if _CACHED is not None and not refresh:
        info = dict(_CACHED)
    else:
        info = {
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": __import__("numpy").__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkl": torch.backends.mkl.is_available(),
            "mkldnn": torch.backends.mkldnn.is_available(),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        }
        _CACHED = dict(info)
    info["torch_num_threads"] = torch.get_num_threads()
    info["available_ram_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
    return info
