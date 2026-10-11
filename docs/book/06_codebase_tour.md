# Chapter 6 — A guided tour of the codebase, and how the project started

> **What you will learn:** what was on the machine on day one and how it was checked; how the project is
> organized and why; the single calling convention that ties everything together; the registry of
> algorithms; and where every kind of output is stored.

## 6.1 Day one: the audit

The project started in an **empty folder** (not even a git repository) inside OneDrive on the Desktop. The
first job was to verify the environment rather than trust the brief:

| What was checked | How | Finding |
|---|---|---|
| Folder contents | `ls -la`, `find` | empty |
| CPU | PowerShell `Get-CimInstance Win32_Processor` | **Ryzen 5 7530U** (the brief said 5300U), 6 cores / 12 threads |
| RAM | `Win32_ComputerSystem`, `psutil` | 16.4 GB total, **only 0.86–1.27 GB free** at the time |
| GPU | `Win32_VideoController` | AMD Radeon integrated graphics, no CUDA |
| Disk | `Get-PSDrive C` | 30 GB free (not the 500 GB the brief described) |
| Python | `python --version`, `py -0p` | 3.13.3 |
| Packages | import each | torch 2.10.0+cpu, numpy 2.5.2, scipy, pandas 3.0.0, matplotlib 3.10.8, streamlit 1.58, pytest 8.3.4, psutil 7.2.2, pyyaml |
| PyTorch internals | `torch.__config__.parallel_info()`, `torch.backends.cpu.get_cpu_capability()` | MKL and oneDNN present, OpenMP, 6 intra-op threads by default, capability **AVX2** |
| Attention kernels | profiler on `scaled_dot_product_attention` | CPU SDPA runs `aten::_scaled_dot_product_flash_attention_for_cpu` (Chapter 9) |
| Fully-masked rows | tiny experiment | SDPA returns zeros, a naive softmax returns NaN (Chapter 7) |
| Profiler memory events | tiny experiment | `[memory]` events carry `nbytes()` and timestamps, so a peak can be reconstructed (Chapter 13) |
| Python start-up cost | timed `import torch` | 5.5 s cold, so running every benchmark in a fresh process would be expensive (Chapter 13) |

Three decisions came straight out of the audit:

1. **Everything must work on CPU**, and must never be described as a GPU result.
2. **Every benchmark needs a memory guard**, because free RAM was below 1.3 GB.
3. **The project would not be installed into the system Python.** A tiny launcher, `lab.py`, adds `src/` to
   the import path, so `python lab.py ...` works from a fresh checkout. (`pip install -e .` also works, but
   is optional.)

> **What actually happened:** the free disk space (30 GB) and the CPU model (7530U) both differed from the
> brief. They are recorded in the documentation as found, not as described. Later, the free RAM and the CPU
> load turned out to change minute by minute because of other programs. That discovery reshaped the
> benchmarking methodology (Chapters 13 and 22).

## 6.2 The layout

```
Beyond Quadratic Attention/
├── lab.py                      launcher: python lab.py <command>
├── pyproject.toml              package metadata + pytest settings
├── README.md                   quick start and command list
├── src/attention_lab/          the library (≈ 4,450 lines)
│   ├── algorithms/             the 8 attention implementations + shared mask logic + registry
│   ├── correctness/            error metrics, tolerance policy, correctness laboratory
│   ├── complexity/             theoretical cost model
│   ├── benchmarking/           environment capture, timing, memory, load monitor, runner, scheduler
│   ├── tasks/                  retrieval probes + synthetic token tasks
│   ├── models/                 tiny Transformer with pluggable attention
│   ├── training/               training loop + multi-backend task experiments
│   ├── diagnostics/            instrumented training for the MQAR investigation
│   ├── reporting/              results → tables, fitted exponents, markdown reports
│   ├── visualization/          matplotlib figures from measured data
│   ├── cli/                    command-line interface
│   └── ui/                     Streamlit web interface
├── tests/                      unit / numerical / integration / scientific / perf_smoke
├── scripts/                    diagnose_recall.py, summarize_diag.py
├── experiments/
│   ├── configs/                YAML specifications of every benchmark sweep
│   ├── results/<run>/          raw results: results.jsonl + meta.json  (never edited by hand)
│   ├── reports/<run>/          generated summary.md, CSV tables, PNG figures
│   ├── diagnostics/mqar_v2/    logs of the MQAR diagnostic experiments
│   └── checkpoints/            saved model weights (git-ignored)
└── docs/                       guides, results, this book
```

The Python source is about 5,600 lines in total, including tests and scripts. The largest files are the CLI
(360 lines), the benchmark runner (241), the tiled attention (230) and the reporting module (227).

**Why this separation?** The brief demanded that algorithms, benchmarks, model integration and UI be kept
apart. The practical reason is trust: an algorithm file contains *only* computation (no timing, no file
writing), so it can be tested in isolation, and a benchmark cannot accidentally change what it measures.

## 6.3 One calling convention for everything

Every attention implementation, exact or approximate, is called the same way:

```python
out = method(q, k, v, causal=False, key_padding_mask=None, **params)
# q: [B, H, L, d]   k: [B, H, L, d]   v: [B, H, L, dv]   →   out: [B, H, L, dv]
```

`params` holds method-specific knobs: `q_tile` and `kv_tile` for the tiled methods, `window` for local
attention, `block_size` and friends for block-sparse attention, `chunk` for linear attention. Because the
signature is shared, the benchmark runner, the correctness laboratory, the probes, the Transformer model and
the UI are each written **once** and switch algorithms by name.

## 6.4 The registry

> **In the code:** `src/attention_lab/algorithms/registry.py`

The registry maps each name to an `AttentionMethod` record:

```python
AttentionMethod(
    name="tiled",
    fn=tiled_attention,
    family="exact",                       # or "restricted-softmax" or "different-kernel"
    computes="softmax(Q K^T / sqrt(d) + M) V",
    implementation="Python-loop online-softmax tiling; autograd through the loop ...",
    default_params={"q_tile": 128, "kv_tile": 128},
    supports_dense_mask=True, ...
)
```

The `family` and `computes` fields make the scientific status of each method explicit. Reports and the UI
use `family` to decide whether an output difference should be called an **error** (exact methods) or a
**deviation** (approximate methods). This matters because, for example, linear attention's difference from
softmax is not a mistake: it is a different function (Chapter 10).

The eight registered names: `naive`, `sdpa_math`, `sdpa_flash_cpu`, `tiled`, `tiled_recompute`,
`local_window`, `block_sparse`, `linear_elu`.

## 6.5 Where outputs go

* **Raw results** (`experiments/results/<timestamp>_<name>/`): `results.jsonl` holds one JSON object per
  configuration, *including* skipped and failed ones with a reason. `meta.json` holds the full spec and the
  environment. These files are the evidence and are never edited by hand.
* **Reports** (`experiments/reports/<same name>/`): generated from the raw results by
  `python lab.py report <run>`: `summary.md`, `summary.csv`, `exponents.csv` and PNG figures.
* **Documentation** (`docs/`): `results.md` collects the findings and names the run behind every number.

Keeping raw data, generated reports and interpretation separate means any number in this book can be traced
back: book → `docs/results.md` → run name → `results.jsonl` line.

## Key points to remember

* The audit found a different CPU (7530U), little free RAM and no GPU, and the design followed from that.
* Algorithms contain only computation. Benchmarks, models, probes and UI call them by name.
* Shared signature: `fn(q, k, v, causal=..., key_padding_mask=..., **params)` on `[B, H, L, d]` tensors.
* The registry labels each method's family, so errors and deviations are never confused.
* Raw results → reports → docs: every number is traceable.
