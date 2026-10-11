# Chapter 26 — File-by-file index

Every file in the project, what it is for, and where the book discusses it.

## Top level

| File | Purpose | Chapter |
|---|---|---|
| `lab.py` | launcher: adds `src/` to the path and runs the CLI | 6, 20 |
| `pyproject.toml` | package metadata; pytest settings (paths, markers, warning filters, `-p no:asyncio`) | 6, 21 |
| `README.md` | quick start, command list, documentation map | 20 |
| `.gitignore` | excludes caches, result/report folders and checkpoints from version control | 6 |
| `.claude/launch.json` | launch configuration used to start the UI for in-app testing | 20 |

## `src/attention_lab/algorithms/`

| File | Purpose | Chapter |
|---|---|---|
| `base.py` | `validate_qkv`, `default_scale`, `MaskSpec`, `apply_tile_mask`, `safe_row_max`; mask semantics and the fully-masked-row policy | 7 |
| `naive.py` | reference attention; `attention_probabilities` | 7 |
| `tiled_exact.py` | `online_softmax_update`, `tiled_forward_kernel`, `tiled_attention`, `_TiledRecomputeFn`, `tiled_attention_recompute` | 8 |
| `sdpa.py` | adapters forcing SDPA's flash-CPU or math backend; mask translation | 9 |
| `local_window.py` | sliding-window attention (chunks + unfold); `band_mask`, `band_entry_count` | 10 |
| `block_sparse.py` | `make_layout`, `layout_to_element_mask`, `block_sparse_attention` (gather) | 10 |
| `linear.py` | `linear_attention` (associative and chunked causal), `linear_attention_quadratic_reference` | 10 |
| `registry.py` | `AttentionMethod`, `REGISTRY`, `EXACT_METHODS`, `get_method`, `run_attention` | 6 |

## Other library modules

| File | Purpose | Chapter |
|---|---|---|
| `correctness/metrics.py` | `compare`, the tolerance policy (`tolerance_for`, unit roundoffs, logit factor) | 11 |
| `correctness/lab.py` | `CorrectnessCase`, `evaluate_case`, `default_cases`, `run_suite`, `unstable_softmax_demo` | 11 |
| `complexity/theory.py` | `theoretical_cost`, `estimated_peak_bytes` | 12 |
| `benchmarking/env.py` | environment capture | 13 |
| `benchmarking/timing.py` | `time_callable`, `summarize`, CPU time per call | 13 |
| `benchmarking/memory.py` | `tensor_peak_bytes` (profiler replay), `ProcessPeak` | 13 |
| `benchmarking/load.py` | `LoadMonitor`, `snapshot` (background load) | 13 |
| `benchmarking/runner.py` | `BenchConfig`, `guard`, `run_one`, `run_isolated` | 13 |
| `benchmarking/worker.py` | isolated-process entry point | 13 |
| `benchmarking/scheduler.py` | `expand`, `run_sweep`, `wait_for_quiet`, `load_results`, `list_runs` | 13 |
| `tasks/probes.py` | `make_probe`, `run_probe`, `probe_suite` | 16 |
| `tasks/synthetic.py` | `local_copy`, `mqar`, `induction`, `fixed_eval_set`, `accuracy_by_distance`, `shortcut_baseline`, `TASK_VERSION` | 17, 19 |
| `models/transformer.py` | `ModelConfig`, `TinyTransformer`, RoPE, sinusoidal encodings, `lm_loss` | 17 |
| `training/train.py` | `TrainConfig`, `train`, `evaluate`, `run_task_experiment` (swap evals, checkpoints) | 17 |
| `diagnostics/recall.py` | instrumented training: gradient norms, update ratios, prediction categories, circuit probes | 19 |
| `reporting/report.py` | `results_frame`, `fit_exponent`, `exponent_table`, `summary_table`, `build_report` | 13, 20 |
| `reporting/tasks_report.py` | probe and task reports | 16, 18 |
| `visualization/plots.py` | runtime, memory (small multiples and shared-axis comparison), metric and bar plots | 14, 15 |
| `cli/main.py` | every command, including `reproduce` | 20 |
| `ui/app.py` | Streamlit interface (9 tabs) | 20 |

## `scripts/`

| File | Purpose | Chapter |
|---|---|---|
| `diagnose_recall.py` | one MQAR/induction diagnostic per invocation | 19 |
| `summarize_diag.py` | print selected columns of a diagnostic log | 19 |
| `build_book.py` | assemble this book into one Markdown, HTML and PDF file | front matter |

## `experiments/`

| Path | Contents |
|---|---|
| `configs/*.yaml` | `smoke`, `scaling_exact`, `scaling_exact_timing`, `scaling_approx`, `tile_sizes`, `threads`, `shapes_dtypes` |
| `results/20261011-191916_smoke` | smoke benchmark |
| `results/20261011-191952_scaling_exact` | E1 memory (timing contaminated) |
| `results/20261011-194506_correctness` | C1 full correctness (795 comparisons) |
| `results/20261011-195411_probes` | first probe run (before the sharpness sweep) |
| `results/20261011-195716_probes` | Q1 probes (used in the book) |
| `results/20261011-200406_tasks_quick` | quick tasks (MQAR v1, with a NOTE about the shortcut) |
| `results/20261011-205830_tasks` | Q2 trained tasks (A and B/C/D) |
| `results/20261011-210946_scaling_exact_timing` | E1b timing, 1 thread |
| `results/20261011-212017_scaling_approx` | E2 approximate methods |
| `results/20261011-215718_tile_sizes` | E3 tile sizes |
| `results/20261011-221221_threads` | E4 threads |
| `results/…_shapes_dtypes` | E5 shapes and dtypes |
| `diagnostics/mqar_v2/` | exploration log snapshot, D1–D3 logs, exploration scripts |
| `reports/<run>/` | generated summary.md, CSVs and figures for each run |
| `checkpoints/` | trained model weights (local only) |

## `docs/`

`results.md` (all findings with run names), `attention_math.md`, `online_softmax.md`, `complexity_analysis.md`,
`systems_guide.md`, `numerical_guide.md`, `benchmark_methodology.md`, `research_guide.md`, `interview_guide.md`,
`mqar_diagnosis.md`, `limitations.md`, `references.md`, `architecture.md`, and `book/` (this book).
