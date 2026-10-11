# Chapter 21 — The test suite: what is checked, and why it is trustworthy

> **What you will learn:** how the tests are organized; what each test file proves; the rules the tests
> follow; and the small configuration problems that had to be solved to make the suite run cleanly.

## 21.1 Organization

| Folder | Purpose | Files |
|---|---|---|
| `tests/unit/` | single components in isolation | `test_base_and_naive.py`, `test_online_softmax.py`, `test_tasks.py`, `test_mqar_pipeline.py`, `test_diagnostics.py` |
| `tests/numerical/` | agreement with the reference (forward and gradients) | `test_exact_vs_reference.py`, `test_approx_methods.py` |
| `tests/integration/` | the Transformer with pluggable attention | `test_transformer.py` |
| `tests/scientific/` | theory vs instrumented counts; validity of measurement tools | `test_complexity_and_measurement.py` |
| `tests/perf_smoke/` | every method benchmarks end to end; guards and scheduler work (no timing thresholds) | `test_benchmark_smoke.py` |

`tests/conftest.py` limits each test to at most 4 threads and provides `rand_qkv`, which builds seeded random q/k/v.

## 21.2 What each file proves

* **`test_base_and_naive.py`**: the reference matches an independent formula and PyTorch SDPA; scale, masks (boolean,
  additive, causal, padding, combinations), fully-masked rows, LSE, gradients (`gradcheck`), and every input-validation
  rejection (Chapter 7).
* **`test_online_softmax.py`**: the update rule equals softmax for any split; it survives huge later maxima and
  all-masked first tiles; ragged tiles and tiles larger than the sequence work; causal tile skipping counts correctly;
  invalid tile sizes are rejected (Chapter 8).
* **`test_exact_vs_reference.py`**: every optimized exact method × fp64/fp32 × causal × 4 shapes, plus batch/head
  shapes, input magnitudes 0.01–30, bf16 policy, fully-masked rows, dense masks for the tiled methods, `gradcheck` of the
  custom backward, LSE agreement, flat error vs L, numerical-vs-bitwise equality, and the two "finding" tests that pin
  the investigations of Chapter 11.
* **`test_approx_methods.py`**: local window, block sparse and linear each equal their own definitions (outputs and
  gradients) over grids of lengths and parameters; limit cases equal full attention; layouts are causal and
  reproducible; linear attention is *not* softmax; unsupported dense masks raise errors (Chapter 10).
* **`test_transformer.py`**: shapes; exact backends give identical outputs and gradients; approximate backends change
  outputs; no future leakage for any backend; RoPE relativity; a training step reduces the loss (Chapter 17).
* **`test_tasks.py`**: generator invariants for local copy, MQAR (v2 distinctness, decoys, source positions) and induction
  (successor of the earlier occurrence, offsets 12–20, distractor vocabulary); fixed eval sets are deterministic; probe
  construction gives target logit β; probes behave as expected (Chapters 16, 17, 19).
* **`test_mqar_pipeline.py`**: the 10 failure-mode tests of the MQAR diagnosis (Chapter 19).
* **`test_diagnostics.py`**: the diagnostic instruments reproduce the model's real attention and read the intended
  positions (Chapter 19).
* **`test_complexity_and_measurement.py`**: theory equals instrumented counts exactly; the tensor-peak instrument is
  byte-exact on a known pattern and sees the score matrix; statistics and exponent fitting are correct (Chapters 12–13).
* **`test_benchmark_smoke.py`**: every method runs and produces a complete record; the memory guard skips without
  allocating, and can skip backward only; the scheduler orders small-first and writes JSONL + meta (Chapter 13).

## 21.3 Rules the tests follow

1. **No timing thresholds.** Correctness is asserted; speed is measured and reported elsewhere.
2. **Independent references.** The naive reference is checked against `torch.softmax` and SDPA; approximate methods
   against masked naive or the direct formula; gradients by finite differences.
3. **Findings are tests too.** When an investigation established a fact (bf16 naive degrades; the fp32 floor scales
   with logits; v2 removes the MQAR shortcut; the induction generator's offset varies, so it cannot be solved
   positionally), a test pins it, so a future change that breaks the fact is noticed. (The positional-shortcut finding
   itself, D2 in Chapter 19, is evidence in a diagnostic log, not a unit test; it needs training.)
4. **Tolerances come from the numerical policy**, never adjusted to make a test pass.

## 21.4 Problems solved to make the suite run

> **What actually happened:**
> * Tests import a helper as `from tests.conftest import rand_qkv`. That failed until the project root was added
>   to pytest's path (`pythonpath = ["src", "."]` in `pyproject.toml`) and `tests/__init__.py` was created.
> * A globally installed `pytest-asyncio` plugin printed a deprecation warning on every run. It is not used by this
>   project, so it is disabled locally (`-p no:asyncio` in `addopts`).
> * Passing `-q` on the command line *and* in `addopts` made pytest so quiet that it hid the pass/fail summary line.
> * The profiler printed "Profiler clears events at the end of each cycle" warnings; they are filtered in the config.

## Key points to remember

* Five test categories: unit, numerical, integration, scientific, perf_smoke.
* Every claim in the results has a test behind it, including the investigations' findings.
* No timing assertions; tolerances come from the documented policy.
* Run with `python lab.py test` (fast set) or `python -m pytest tests` (all).
