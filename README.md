# Beyond Quadratic Attention

<div align="center">

### A Local, CPU-First Laboratory for Understanding Attention Beyond the Quadratic Baseline

**Studying the trade-offs between arithmetic, memory, memory traffic, numerical accuracy, and practical execution speed as sequence length grows.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-CPU--First-555555?style=flat-square&logo=processor&logoColor=white)](#quick-start)
[![Research](https://img.shields.io/badge/Focus-Attention%20Research-7057FF?style=flat-square)](#research-objectives)
[![Execution](https://img.shields.io/badge/Execution-Local--First-2EA44F?style=flat-square)](#reproducibility-and-research-principles)
[![Interface](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#interfaces)
[![Tests](https://img.shields.io/badge/Tests-420%20reported-2EA44F?style=flat-square&logo=pytest&logoColor=white)](#quick-start)

**Local execution · Reproducible experiments · Numerical correctness · Systems-level analysis**

[Quick Start](#quick-start) · [Attention Methods](#attention-methods) · [Experiments](#experimental-capabilities) · [Documentation](#documentation) · [Commands](#command-reference)

</div>

---

## Overview

**Beyond Quadratic Attention** is a local, CPU-first research laboratory for investigating how different attention algorithms behave as sequence length increases.

The project explores more than asymptotic computational complexity. It studies the practical trade-offs between arithmetic operations, intermediate tensor storage, memory consumption, memory traffic, numerical accuracy, backward-pass computation, and wall-clock execution time.

Rather than treating attention optimization as a single speed contest, the laboratory provides a controlled environment in which multiple implementations can be examined using numerical tests, benchmark sweeps, analytical cost models, retrieval probes, and small Transformer training experiments.

> **Core research question**
>
> How do attention algorithms trade off computational complexity, memory traffic, numerical accuracy, and practical execution speed as sequence length increases?

Everything is designed to run locally. There are no required cloud services, telemetry, or network calls during normal experimentation. Reported findings in `docs/results.md` and `experiments/reports/` are intended to be traceable to execution artifacts stored in `experiments/results/`.

The project distinguishes theoretical estimates from empirical measurements, correctness from performance, and exact softmax attention from restricted or fundamentally different attention formulations.

## Research Objectives

The laboratory investigates five interconnected dimensions of attention performance.

| Dimension | Research question |
|---|---|
| **Computational complexity** | How does the amount of computation scale with sequence length and tensor dimensions? |
| **Memory consumption** | Which intermediate tensors are materialized, and how do forward and backward passes affect memory requirements? |
| **Memory traffic and execution systems** | How do tiling, kernel fusion, recomputation, allocation, cache behavior, and dispatch overhead influence execution? |
| **Numerical accuracy** | How closely do exact implementations agree with a high-precision reference across outputs, gradients, masks, dtypes, and numerical magnitudes? |
| **Task-level behavior** | How do attention mechanisms perform on controlled retrieval and synthetic sequence-learning tasks? |

These dimensions must be evaluated independently before drawing conclusions.

An algorithm with a better theoretical complexity does not necessarily execute faster on every CPU, tensor shape, or sequence length. Similarly, lower measured memory usage does not establish numerical equivalence, and faster execution does not automatically imply better task performance.

The purpose of this project is to make those distinctions measurable and understandable.

## Attention Methods

The laboratory implements eight attention backends covering exact softmax attention, restricted attention patterns, and a linear-attention alternative.

| Backend | Family | Implementation |
|---|---|---|
| `naive` | Exact softmax | Reference implementation of `softmax(QKᵀ / √d + M)V`. Materializes the attention-score tensor, with storage proportional to `B · H · L²`. |
| `sdpa_math` | Exact softmax | PyTorch scaled dot-product attention (SDPA), forced to use its math backend. |
| `sdpa_flash_cpu` | Exact softmax | PyTorch SDPA forced to use `aten::_scaled_dot_product_flash_attention_for_cpu`, PyTorch's fused tiled CPU kernel. This is **not** the official FlashAttention CUDA library. |
| `tiled` | Exact softmax | Python implementation of online-softmax tiling, with autograd operating through the loop. |
| `tiled_recompute` | Exact softmax | Uses the same tiled forward strategy with a custom backward pass that recomputes probability tiles from `Q`, `K`, and log-sum-exp information. |
| `local_window` | Restricted softmax | Vectorized sliding-window attention, restricting the attention pattern to a local band. |
| `block_sparse` | Restricted softmax | Local, global, and random block layout, executed using gather operations. |
| `linear_elu` | Different attention formulation | Linear attention associated with Katharopoulos et al., using `φ(x) = elu(x) + 1`. It is **not softmax attention**. |

### Understanding the comparison groups

**Exact softmax implementations**

`naive`, `sdpa_math`, `sdpa_flash_cpu`, `tiled`, and `tiled_recompute` target the same underlying dense softmax-attention operation, subject to implementation details, numerical precision, masks, and supported configurations.

These methods make it possible to investigate how execution strategies affect the trade-off between materialization, memory use, computation, and runtime.

**Restricted softmax implementations**

`local_window` and `block_sparse` constrain which positions can interact. They can reduce the work associated with unrestricted attention, but they do not implement the same unrestricted attention pattern as dense softmax attention.

**Alternative attention formulation**

`linear_elu` uses a different attention mechanism. It is useful for studying alternative computational behavior, but its outputs should not be interpreted as numerically equivalent to those of standard softmax attention.

**Important implementation distinction:** `sdpa_flash_cpu` refers to a PyTorch CPU kernel selected through SDPA. It should not be confused with the official FlashAttention CUDA implementation or used to make unsupported GPU-performance claims.

## Experimental Capabilities

Beyond the attention implementations, the repository includes the following research components.

### 1. Numerical-Correctness Laboratory

A dedicated correctness framework compares exact methods against a float64 reference.

It investigates:

- Forward-output agreement.
- Gradient agreement.
- Supported data types and dtype-related numerical behavior.
- Attention masks and their effects.
- Different numerical magnitudes and potential stability problems.

The correctness workflow is separate from benchmarking. A method must be evaluated for numerical behavior independently of its runtime.

Run:

```bash
python lab.py correctness
```

For the expanded checks:

```bash
python lab.py correctness --full
```

### 2. Benchmark Runner

The benchmark runner executes configured experiments and generates result artifacts and reports.

Its instrumentation includes:

- Three separately scoped memory instruments.
- A background-load monitor.
- Configurable benchmark sweeps.
- Raw execution results.
- Generated tables and figures.

Available experiment-spec families include:

| Experiment specification | Purpose |
|---|---|
| `scaling_exact` | Scaling comparisons for exact attention methods. |
| `scaling_approx` | Scaling comparisons for restricted and alternative attention methods. |
| `tile_sizes` | Investigating the effects of tile-size choices. |
| `threads` | Investigating CPU thread-count effects. |
| `shapes_dtypes` | Comparing tensor shapes and data types. |

The benchmark methodology documentation describes the timing procedure, memory instruments, controls, and scheduling.

### 3. Analytical Complexity Model

The project includes an analytical model for examining the theoretical cost of supported attention methods.

Its outputs are explicitly labeled with the `theory_*` prefix.

These values represent theoretical estimates, not measured execution times or observed memory consumption. Keeping this distinction explicit prevents complexity calculations from being mistaken for experimental evidence.

Example:

```bash
python lab.py complexity --L 1024 4096
```

### 4. Attention-Level Retrieval Probes

Retrieval probes investigate attention behavior without requiring the training of a Transformer.

They provide a controlled way to study whether an attention mechanism can retrieve relevant information under specified task conditions.

Run the standard configuration:

```bash
python lab.py probes
```

Run the expanded configuration:

```bash
python lab.py probes --full
```

The probe suite includes Tasks A, B, and C in its documented configuration.

### 5. Token-Level Synthetic Tasks

The laboratory includes synthetic tasks for investigating attention behavior at the token level.

| Task | Description |
|---|---|
| **A — Local copy** | Tests local copying behavior. |
| **B/C/D — Variable-offset induction** | Investigates variable-offset induction, including interference and offset-generalization variants. |
| **M — MQAR** | Evaluates behavior on the multi-query associative-retrieval task. |

A small Transformer can be trained with a switchable attention backend so different methods can be evaluated under controlled task configurations.

Example:

```bash
python lab.py tasks --quick
```

To select task families:

```bash
python lab.py tasks --which ABM
```

Here, `A` selects local copy, `B`, `C`, and `D` represent variable-offset induction variants, and `M` selects MQAR. MQAR is optional in the task workflow.

### 6. Instrumented Training Diagnostics

The repository includes a dedicated recall-training diagnostic for investigating training behavior.

The diagnostic can examine:

- Gradient norms.
- Parameter update sizes.
- Attention-circuit probes.
- Training-related behavior associated with recall performance.

Example invocation:

```bash
python scripts/diagnose_recall.py ...
```

Replace `...` with the arguments required by the specific diagnostic run.

The repository also contains `docs/mqar_diagnosis.md`, which documents the investigation into the MQAR training plateau, including tests, diagnostics, and conclusions.

### 7. Switchable Tiny Transformer

A small Transformer implementation allows the attention backend to be switched between supported methods.

This connects implementation-level comparisons with controlled training and synthetic-task experiments.

The aim is not to establish broad model capability from a toy experiment, but to investigate how attention choices interact with the task and training setup.

### 8. Command-Line Interface

The CLI exposes the major workflows:

- Environment checks.
- Test execution.
- Numerical correctness.
- Benchmark sweeps.
- Complexity calculations.
- Retrieval probes.
- Synthetic tasks.
- Diagnostic experiments.
- Report generation.
- Experiment reproduction.
- UI launch.

### 9. Streamlit Research Interface

A Streamlit UI provides an interactive local interface for exploring the laboratory.

It can be launched using the project launcher or directly through `lab.py`.

The default address is `http://localhost:8501`, with the next available port used if that port is occupied.

## Quick Start

The project is designed to be straightforward to run on a local machine. A dedicated GPU is not required for the CPU-first workflows.

### Prerequisites

| Requirement | Minimum version or note |
|---|---|
| Python | 3.10 or newer |
| PyTorch | 2.1 or newer; a CPU build is sufficient |
| NumPy | Required |
| pandas | Required |
| Matplotlib | Required |
| psutil | Required |
| PyYAML | Required |
| pytest | Required for tests |
| Streamlit | Required for the UI |

### Option 1 — One-Click Launcher

On Windows, double-click:

```text
start_lab.bat
```

Alternatively, run:

```bash
python start.py
```

The launcher checks the required packages and displays a menu.

Press Enter to open the research interface. The launcher also provides options for:

- Environment health checks.
- Test execution.
- The numerical-correctness laboratory.
- The project book.
- The results summary.

For scripted usage, supported actions include:

```bash
python start.py ui
python start.py check
python start.py test
python start.py correctness
python start.py book
python start.py results
python start.py shell
python start.py env
```

### Option 2 — Run the CLI Directly

No installation step is required for the commands below because `lab.py` adds `src/` to the Python path.

First, inspect the environment:

```bash
python lab.py env
```

Run the essential test command:

```bash
python lab.py test
```

Run a smoke benchmark:

```bash
python lab.py bench --smoke
```

The project documentation reports **420 tests**. The essential test command is described as taking approximately 3–5 minutes, while the broader test-suite runtime is estimated at roughly 1–5 minutes depending on machine load. Actual timings vary with the environment and should not be treated as performance guarantees.

The smoke benchmark exercises every method on small shapes, writes results and a report, and is intended as a quick execution-path check rather than a basis for performance conclusions.

### Optional Editable Installation

If you prefer to install the project into your active Python environment:

```bash
pip install -e .
```

### Launch the Research Interface Directly

```bash
python lab.py ui
```

For headless operation:

```bash
python lab.py ui --headless
```

Open the local URL printed by the application.

## Command Reference

Run commands from the repository root.

| Command | Purpose |
|---|---|
| `python lab.py env` | Inspect the active environment and required packages. |
| `python lab.py test` | Run the essential test command. |
| `python lab.py correctness` | Run standard numerical-correctness checks. |
| `python lab.py correctness --full` | Run expanded correctness checks. |
| `python lab.py bench --smoke` | Run a small benchmark across all methods. |
| `python lab.py bench --spec experiments/configs/<spec>.yaml` | Execute a configured benchmark sweep. |
| `python lab.py report experiments/results/<run>` | Rebuild tables and figures for an existing run. |
| `python lab.py complexity --L 1024 4096` | Generate theoretical cost estimates for the specified sequence lengths. |
| `python lab.py probes` | Run attention-level retrieval probes without training. |
| `python lab.py probes --full` | Run expanded retrieval probes. |
| `python lab.py tasks --quick` | Run a quick synthetic-task configuration. |
| `python lab.py tasks --which ABM` | Select synthetic task families. |
| `python scripts/diagnose_recall.py ...` | Run an instrumented recall-training diagnostic. |
| `python lab.py reproduce --dry-run` | Preview the experiment reproduction plan without executing it. |
| `python lab.py reproduce --only E2` | Run only experiment E2. |
| `python lab.py reproduce` | Run the experiments underlying `docs/results.md`, sequentially. |
| `python lab.py ui` | Launch the Streamlit research interface. |
| `python lab.py ui --headless` | Launch the UI in headless mode. |
| `python -m pytest tests` | Run the full pytest suite. |

Benchmark sweeps can be selected from the experiment specifications in `experiments/configs/`:

```bash
python lab.py bench --spec experiments/configs/<spec>.yaml
```

Available sweep families are `scaling_exact`, `scaling_approx`, `tile_sizes`, `threads`, and `shapes_dtypes`.

For exact options and arguments supported by the current checkout, consult the relevant command help and experiment configuration files.

## Benchmarking and Reproducibility

A central design principle of this laboratory is that every performance conclusion should be traceable to its experimental context.

### Evidence and artifact organization

| Location | Purpose |
|---|---|
| `experiments/configs/` | Experiment specifications and benchmark configurations. |
| `experiments/results/` | Raw execution results. |
| `experiments/reports/` | Generated tables, figures, and experiment reports. |
| `docs/results.md` | Findings from the current machine, accompanied by evidence. |

The findings in `docs/results.md` and generated reports in `experiments/reports/` are intended to come from executions recorded in `experiments/results/`.

### Reproducing the experiments

Preview the planned reproduction sequence:

```bash
python lab.py reproduce --dry-run
```

Reproduce a single experiment:

```bash
python lab.py reproduce --only E2
```

Run the complete configured sequence:

```bash
python lab.py reproduce
```

The full reproduction workflow runs sequentially and may take substantially longer than the smoke benchmark. Review the experiment configurations and available system resources before starting a long run.

### Research principles

The project follows these rules when producing and interpreting results.

1. **Theory is never presented as measurement.** Analytical outputs use the `theory_*` naming convention.
2. **Failed runs remain visible.** Failed, skipped, and contended runs are recorded rather than silently removed.
3. **Correctness and performance are separate.** Correctness tests do not enforce wall-clock timing thresholds.
4. **CPU results are not GPU claims.** Findings are limited to the hardware and execution paths actually evaluated.
5. **Experiment context matters.** Shape, dtype, thread count, background load, and backend availability can affect results.
6. **Instrumentation has limits.** Memory measurements depend on the scope and behavior of each instrument.
7. **Comparisons require equivalent assumptions.** Restricted and alternative attention formulations must not be treated as mathematically interchangeable with dense softmax attention.

For additional detail, see [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md), [`docs/research_guide.md`](docs/research_guide.md), and [`docs/limitations.md`](docs/limitations.md).

## Repository Structure

The full architecture is documented in [`docs/architecture.md`](docs/architecture.md).

```text
.
├── src/
│   └── attention_lab/          # Algorithms, correctness, complexity,
│                                # benchmarking, tasks, models, training,
│                                # reporting, visualization, CLI and UI
│
├── tests/                      # Unit, numerical, integration, scientific,
│                                # and performance-smoke tests
│
├── experiments/
│   ├── configs/                # Experiment and benchmark specifications
│   ├── results/                # Raw execution results
│   └── reports/                # Generated reports, tables and figures
│
├── docs/
│   ├── book/                   # Long-form project book
│   ├── results.md              # Machine-specific findings and evidence
│   ├── architecture.md         # Repository architecture
│   ├── attention_math.md       # Attention mathematics
│   ├── online_softmax.md       # Online-softmax derivation
│   ├── complexity_analysis.md  # Complexity and memory analysis
│   ├── systems_guide.md        # Systems-level interpretation
│   ├── numerical_guide.md      # Numerical correctness methodology
│   ├── benchmark_methodology.md# Benchmarking methodology
│   ├── research_guide.md       # Hypotheses and experiment controls
│   ├── interview_guide.md      # Technical questions and explanations
│   ├── mqar_diagnosis.md       # MQAR diagnostic investigation
│   ├── limitations.md          # Scope and limitations
│   └── references.md           # Verified references
│
├── scripts/                    # Book-building and diagnostic utilities
├── lab.py                      # Main command-line entry point
├── start.py                    # Interactive launcher
├── start_lab.bat               # Windows one-click launcher
└── README.md                   # Project overview and usage guide
```

This is a conceptual map of the repository rather than an exhaustive listing of every file.

## Documentation

The project includes detailed documentation connecting the mathematical foundations, implementations, experiments, and observed results.

### Start Here: The Project Book

**[`docs/book/Beyond_Quadratic_Attention_Book.pdf`](docs/book/Beyond_Quadratic_Attention_Book.pdf)**

HTML and Markdown versions are also provided alongside the PDF.

The book explains the project from the ground up, including the implementations, experiments, and bugs discovered and fixed during development.

Rebuild the book using:

```bash
python scripts/build_book.py
```

### Technical Documentation

| Document | Contents |
|---|---|
| [`docs/results.md`](docs/results.md) | **Findings from this machine, with the evidence.** |
| [`docs/architecture.md`](docs/architecture.md) | Repository architecture and module organization. |
| [`docs/attention_math.md`](docs/attention_math.md) | Attention, softmax, masks, gradients, and their connection to the implementation. |
| [`docs/online_softmax.md`](docs/online_softmax.md) | Derivation and proof of the online-softmax update; storing versus recomputing values during backward computation. |
| [`docs/complexity_analysis.md`](docs/complexity_analysis.md) | Time complexity, auxiliary memory, and training-memory analysis for each method. |
| [`docs/systems_guide.md`](docs/systems_guide.md) | Caches, fusion, dispatch overhead, and CPU-versus-GPU behavior. |
| [`docs/numerical_guide.md`](docs/numerical_guide.md) | Numerical error sources, comparison protocol, tolerance policy, and how tolerances were derived. |
| [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md) | Timing, memory instruments, controls, and scheduler behavior. |
| [`docs/research_guide.md`](docs/research_guide.md) | Research hypotheses, experimental controls, and interpretation rules. |
| [`docs/interview_guide.md`](docs/interview_guide.md) | Difficult technical questions with answers grounded in the project. |
| [`docs/mqar_diagnosis.md`](docs/mqar_diagnosis.md) | Why MQAR training plateaued, including tests, diagnostics, and conclusions. |
| [`docs/limitations.md`](docs/limitations.md) | What this laboratory can and cannot establish. |
| [`docs/references.md`](docs/references.md) | Verified citations supporting the project. |

## Interpretation and Limitations

Beyond Quadratic Attention is an experimental research environment, not a claim that one implementation is universally superior.

Its findings must be interpreted within the configurations, hardware, and execution paths actually evaluated.

### Important distinctions

**1. Exact attention versus restricted attention**

Dense softmax attention, local-window attention, and block-sparse attention do not necessarily expose a query to the same set of key positions. Their runtime and task-level behavior must be interpreted in light of their different attention patterns.

**2. Softmax attention versus linear attention**

The `linear_elu` implementation uses a different mathematical formulation. It should not be treated as a numerically equivalent replacement for softmax attention.

**3. Theoretical complexity versus practical speed**

Asymptotic analysis describes how computational requirements scale. It does not, by itself, predict actual wall-clock execution time on a specific CPU.

**4. Memory estimates versus measured memory**

Different memory instruments observe different aspects of memory use. Their scopes, sampling behavior, and limitations must be considered before comparing readings.

**5. CPU results versus GPU behavior**

CPU benchmarks cannot establish how an implementation will perform on a GPU. Different hardware architectures, kernels, memory systems, and execution strategies can lead to different results.

**6. Synthetic tasks versus general model capability**

Performance on local copy, variable-offset induction, or MQAR provides evidence about the specified task setup. It does not establish general language-model capability or broad real-world performance.

For a deeper discussion of the project's scope, consult [`docs/limitations.md`](docs/limitations.md), [`docs/research_guide.md`](docs/research_guide.md), and [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md).

## Reproducibility and Research Principles

The laboratory follows a set of explicit rules designed to keep experimental findings interpretable and auditable.

- Theoretical numbers are never presented as measured results.
- Failed, skipped, and contended benchmark runs are recorded rather than hidden.
- Tests do not enforce timing assertions.
- Numerical correctness and execution performance are evaluated separately.
- CPU results are not generalized into GPU-performance claims.
- Results should be interpreted alongside their experiment specifications and raw execution artifacts.
- Different attention formulations are compared with their mathematical and task-level differences made explicit.

These principles are central to the project: the objective is not simply to find the fastest number, but to understand **why an attention method behaves as it does, what it costs, and what the evidence actually supports**.

---

## References

The project's verified references are maintained in [`docs/references.md`](docs/references.md).

The implementation and documentation include the online-softmax approach, the PyTorch SDPA execution paths, and the linear-attention formulation associated with Katharopoulos et al. Refer to the project's bibliography for the corresponding verified citations.

---

<div align="center">

### Beyond Quadratic Attention

**Measure the computation. Understand the memory. Verify the numerics. Question the benchmark.**

A local laboratory for exploring attention algorithms through mathematical analysis, numerical experiments, and practical CPU execution.

</div>
