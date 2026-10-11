# Chapter 20 — Operating the lab: install, run, explore, reproduce, extend

> **What you will learn:** every command, what it produces and where; how to use the web interface; how to read a
> report; how to reproduce all results; and how to add a new attention algorithm or a new experiment.

## 20.1 Requirements and setup

* Python ≥ 3.10 (developed on 3.13.3).
* Packages: `torch` (the CPU build is fine), `numpy`, `pandas`, `matplotlib`, `psutil`, `pyyaml`, `pytest`, and
  `streamlit` for the UI. On this machine all were already installed.
* **No installation step is required.** Open a terminal in the project folder and run `python lab.py ...`. The
  launcher adds `src/` to Python's search path. Optionally, `pip install -e .` installs the package in development
  mode and adds an `attention-lab` command.

## 20.2 The commands

All commands are `python lab.py <command> [options]`:

| Command | What it does | Where output goes |
|---|---|---|
| `env` | prints the environment audit (CPU, cores, RAM, versions, thread settings) | terminal |
| `test [--all]` | runs the test suite | terminal |
| `correctness [--full] [--methods ...]` | the numerical-correctness laboratory | `experiments/results/<time>_correctness/` (results.jsonl, summary.md, meta.json) |
| `bench --smoke` | every method on tiny shapes, about 20 s | `experiments/results/<time>_smoke/` + report |
| `bench --spec experiments/configs/<name>.yaml [--isolate yes/no] [--threads ...]` | a benchmark sweep | results + `experiments/reports/<run>/` |
| `report experiments/results/<run>` | rebuilds the report of a run | `experiments/reports/<run>/` |
| `complexity --L 1024 4096 [--causal] [--d 64]` | prints the theoretical cost table, labeled THEORETICAL | terminal |
| `probes [--full]` | attention-level retrieval probes | results + report (`probes.md`, figures) |
| `tasks [--which AB(M)] [--steps N] [--threads 1] [--batch-size 32] [--quick]` | trains the tiny Transformer with each backend | results + report (`tasks.md`, figures) + checkpoints |
| `reproduce [--dry-run] [--only E2 Q1] [--keep-going]` | runs every experiment behind the results, one after another | all of the above |
| `ui [--port 8501] [--headless]` | starts the local web interface | browser at `http://localhost:8501` |

Diagnostics for training (Chapter 19): `python scripts/diagnose_recall.py overfit|pairs|induction ...` and
`python scripts/summarize_diag.py <log>`.

**Sweep specs** (`experiments/configs/`): `smoke`, `scaling_exact` (memory, isolated), `scaling_exact_timing`
(1 thread), `scaling_approx`, `tile_sizes`, `threads`, `shapes_dtypes`. Each contains a `hypothesis:` line, the method
list, the grid and the budgets. To change an experiment, copy a spec and edit it. No code changes are needed.

## 20.3 The web interface

`python lab.py ui` opens a Streamlit app bound to **localhost only**, with usage statistics disabled. Tabs:

1. **Environment**: the live audit.
2. **Algorithms**: the registry: family, what each computes, how it executes.
3. **Correctness**: compare any exact method with the fp64 reference for chosen shapes, dtype, magnitude and tiles.
4. **Benchmark**: run one guarded configuration and see its timing, memory and theory numbers.
5. **Results & plots**: pick a stored sweep and see its hypothesis, status counts, the full table, interactive
   runtime and memory charts, fitted exponents and the report figures.
6. **Complexity**: the theoretical table and chart for chosen lengths.
7. **Quality probes**: run a probe interactively, or browse stored probe runs.
8. **Task training**: browse trained-task results.
9. **History & export**: every run, plus a button that builds and downloads a report.

Everything displayed is read from `experiments/results/` or computed live. Nothing is hard-coded.

> **What actually happened:** when the UI was first tested in a browser, the Results tab crashed with
> `FileNotFoundError: ...probes/meta.json`. The function listing "benchmark runs" included every folder containing a
> `results.jsonl`, including probe and correctness runs, which had no `meta.json` and are not sweeps. Fixes: list only
> folders whose `meta.json` contains a sweep spec, and make the probe and correctness commands write a `meta.json` too.
> A second lesson: Streamlit reloads `app.py` on change but keeps imported modules cached, so the server had to be
> restarted to pick up the fix. A third: on first launch Streamlit can ask for an email on the console and open a
> browser, so a `--headless` flag was added for automated launches.

## 20.4 Reading a report

`experiments/reports/<run>/summary.md` contains: the hypothesis from the spec; the environment; status counts; a
**measurement-scopes** section (what each column means); the background-load summary; fitted exponents; the full
results table (`summary.csv` has the same data); figures; and a list of every skipped, partial or failed configuration
with its reason.

Columns: `*_ms_median`/`*_ms_iqr` (wall time), `*_cpu_ms_per_call` (process CPU time), `*_tensor_peak_MiB` (PyTorch
allocator peak), `proc_peak_delta_*_MiB` (OS working set, isolated runs), `theory_*` (estimates), `err_vs_ref`
(exact methods) or `deviation_from_softmax` (approximate methods), `other_load_cores` and `contended`.

## 20.5 Reproducing everything

```
python lab.py reproduce --dry-run     # list the steps
python lab.py reproduce               # run them all (hours on a busy laptop)
python lab.py reproduce --only E2     # one step
```

Steps, in order: tests; correctness (C1); exact scaling memory (E1); exact timing (E1b); approximate scaling (E2);
tile sizes (E3); threads (E4); shapes and dtypes (E5); probes (Q1); trained tasks (Q2). They run sequentially,
because running benchmarks in parallel would contaminate each other's timings. A reproduction gives new timestamped
folders; compare them with the runs named in `docs/results.md`. Expect identical correctness, memory and accuracy
results (they are deterministic given the seeds), and different absolute timings (they depend on machine load).

## 20.6 Extending the lab

**A new attention algorithm:** write a function `fn(q, k, v, *, causal=False, key_padding_mask=None, mask=None,
**params)` in `src/attention_lab/algorithms/`, use `validate_qkv` and `MaskSpec` from `base.py`, and add one
`AttentionMethod(...)` entry to `registry.py` with its family and a precise `computes` string. Add its cost formulas to
`complexity/theory.py`. Then add tests: exact methods join `EXACT_METHODS` automatically and are picked up by the
numerical tests; restricted methods need an equality test against naive with the corresponding mask. Benchmarks,
probes, the model and the UI will offer it by name with no further changes.

**A new experiment:** copy a YAML spec, edit the grid and hypothesis, and run `python lab.py bench --spec ...`.

**GPU support:** see Chapter 13 §13.8. Only the benchmarking instruments need device-aware additions.

## Key points to remember

* `python lab.py env | test | correctness | bench | report | complexity | probes | tasks | reproduce | ui`.
* Experiments are YAML specs with a stated hypothesis; outputs are timestamped result and report folders.
* The UI reads only stored or live-computed results, runs on localhost, and has no telemetry.
* `reproduce` runs everything sequentially; deterministic results reproduce exactly, timings do not.
* Adding an algorithm = one function + one registry entry + theory formulas + tests.
