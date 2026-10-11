# Beyond Quadratic Attention

A local, CPU-first laboratory for studying how attention algorithms trade off
**arithmetic, memory, memory traffic, numerical accuracy and wall-clock speed** as
sequence length grows.

> Core question: *How do attention algorithms trade off computational complexity,
> memory traffic, numerical accuracy, and practical execution speed as sequence
> length increases?*

Everything runs locally. There are no cloud services, no telemetry and no network
calls. Every number in `docs/results.md` and `experiments/reports/` comes from
executions stored in `experiments/results/`.

## What is implemented

| Name | Family | What it is |
|---|---|---|
| `naive` | exact | reference softmax(QKᵀ/√d + M)V, materializes `B·H·L²` scores |
| `sdpa_math` | exact | PyTorch SDPA forced to its math backend |
| `sdpa_flash_cpu` | exact | PyTorch SDPA forced to `aten::_scaled_dot_product_flash_attention_for_cpu` — PyTorch's fused tiled CPU kernel (**not** the official FlashAttention CUDA library) |
| `tiled` | exact | online-softmax tiling in Python; autograd through the loop |
| `tiled_recompute` | exact | same forward; custom backward that recomputes P tiles from (Q, K, logsumexp) |
| `local_window` | restricted softmax | sliding-window band, vectorized |
| `block_sparse` | restricted softmax | local + global + random block layout, gather execution |
| `linear_elu` | different kernel | Katharopoulos et al. linear attention, φ = elu + 1 (not softmax) |

There is also:

- a numerical-correctness laboratory;
- a benchmark runner with three separately scoped memory instruments and a background-load monitor;
- an analytical complexity model whose outputs are always labeled `theory_*`;
- attention-level retrieval probes;
- token-level synthetic tasks (local copy, variable-offset induction with interference and offset-generalization variants, and MQAR);
- a tiny Transformer whose attention backend is switchable;
- a CLI;
- a Streamlit UI.

## Quick start

**One-click start:** double-click `start_lab.bat` (or run `python start.py`). It checks the
required packages and shows a menu; pressing Enter opens the research interface at
`http://localhost:8501` (the next free port if that one is busy). The menu also offers a
health check, the test suite, the correctness lab, the book and the results summary. For
scripting, run `python start.py <action>` with one of
`ui | check | test | correctness | book | results | shell | env`.

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1 (CPU build is fine), numpy, pandas,
matplotlib, psutil, pyyaml, pytest; streamlit for the UI. No installation step is
needed, because `lab.py` adds `src/` to the path. (`pip install -e .` also works if
you prefer.)

```bash
python lab.py env
```

```bash
python lab.py test
```

(420 tests; about 1–5 minutes depending on machine load.)

```bash
python lab.py bench --smoke
```

`test` runs the essential tests (~3–5 min on this machine). `bench --smoke` runs
every method on tiny shapes in under a minute and writes results and a report.

## All commands

| Command | Purpose |
|---|---|
| `python lab.py correctness [--full]` | exact methods vs float64 reference: forward, gradients, dtypes, masks, magnitudes |
| `python lab.py bench --spec experiments/configs/<spec>.yaml` | a benchmark sweep (`scaling_exact`, `scaling_approx`, `tile_sizes`, `threads`, `shapes_dtypes`) |
| `python lab.py report experiments/results/<run>` | rebuild tables and figures for a run |
| `python lab.py complexity --L 1024 4096` | theoretical cost table (labeled as theory) |
| `python lab.py probes [--full]` | attention-level retrieval probes (Tasks A/B/C without training) |
| `python lab.py tasks [--quick] [--which AB(M)]` | train tiny Transformers per backend: A local copy, B/C/D variable-offset induction, M MQAR (optional) |
| `python scripts/diagnose_recall.py ...` | one instrumented training diagnostic (gradient norms, update sizes, attention-circuit probes) |
| `python lab.py reproduce [--dry-run] [--only E2]` | run every experiment behind `docs/results.md`, sequentially |
| `python lab.py ui [--headless]` | Streamlit research interface on localhost |
| `python -m pytest tests` | full test suite |

## Repository layout

See `docs/architecture.md`. In short, `src/attention_lab/` holds algorithms,
correctness, complexity, benchmarking, tasks, models, training, reporting,
visualization, cli and ui. `tests/` holds unit, numerical, integration, scientific
and perf_smoke tests. Experiment specs are in `experiments/configs/`, raw results
in `experiments/results/`, and generated reports in `experiments/reports/`.

## Documentation

**Start with the book:** `docs/book/Beyond_Quadratic_Attention_Book.pdf` (or the `.html` / `.md` version next to it).
It explains the whole project from zero, including every experiment and every bug that was found and fixed.
Rebuild it with `python scripts/build_book.py`.

| Document | Contents |
|---|---|
| `docs/results.md` | **findings from this machine, with the evidence** |
| `docs/attention_math.md` | attention, softmax, masks, gradients, connected to the code |
| `docs/online_softmax.md` | derivation and proof of the online-softmax update; store vs recompute backward |
| `docs/complexity_analysis.md` | time, auxiliary memory and training memory per method |
| `docs/systems_guide.md` | caches, fusion, dispatch overhead, CPU vs GPU |
| `docs/numerical_guide.md` | error sources, comparison protocol, tolerance policy and how it was derived |
| `docs/benchmark_methodology.md` | timing, memory instruments, controls, scheduler |
| `docs/research_guide.md` | hypotheses, controls and interpretation rules for every experiment |
| `docs/interview_guide.md` | hard questions with answers grounded in this project |
| `docs/mqar_diagnosis.md` | why MQAR training plateaued: tests, diagnostics, verdicts |
| `docs/limitations.md` | what this lab cannot tell you |
| `docs/references.md` | verified citations |

## Ground rules this project follows

* Theoretical numbers are never presented as measurements. Failed, skipped and
  contended runs are recorded, not hidden.
* No timing assertions in tests. Correctness and performance are separate.
* CPU results are not claims about GPUs.
