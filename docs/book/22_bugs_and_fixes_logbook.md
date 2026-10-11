# Chapter 22 — The logbook: every bug, mistake, surprise and fix, in order

> **What you will learn:** the true history of the project, as a numbered log. Each entry gives the
> **symptom** (what was seen), the **cause** (what was really going on), **how it was found**, the **fix**, and
> **where** the fix lives. Read this chapter twice: it contains most of the practical lessons of the project.

Legend: 🐞 bug in our code · ⚠️ methodological mistake · 🔍 surprise that needed investigation · 🧭 decision.

---

### Phase 0–1: audit and reference implementation

**1. 🔍 The hardware was not what the brief said.**
*Symptom:* the brief said Ryzen 5 5300U and 500 GB disk. *Found by:* PowerShell `Win32_Processor`, `Get-PSDrive`.
*Reality:* Ryzen 5 **7530U**, 30 GB free disk, and only about 0.9–1.3 GB of free RAM at the time. *Fix:* documented as
found; a memory guard became mandatory. *Where:* `docs/systems_guide.md`, `benchmarking/runner.py::guard`.

**2. 🔍 Importing torch takes 5.5 seconds.**
*Found by:* timing `import torch`. *Consequence:* running every benchmark in a fresh process costs about 5 s each.
*Decision:* process isolation is **optional** (`isolate:` in specs); it is used where the OS peak memory is needed.

**3. 🔍 What does attention return for a row with no allowed keys?**
*Symptom:* a plain softmax over `[−∞, −∞, …]` gives NaN. *Found by:* a 10-line experiment comparing
`torch.softmax` with PyTorch SDPA (both CPU backends, bool and float masks). *Reality:* SDPA returns zeros with no NaN
gradients. *Fix/decision:* project-wide policy "fully-masked rows → zero output, zero gradient", implemented via
`safe_row_max` (−∞ → 0) and `l_safe` (0 → 1). *Where:* `algorithms/base.py`, `naive.py`; tests in
`test_base_and_naive.py`, `test_exact_vs_reference.py`.

**4. 🔍 Which kernel does PyTorch SDPA actually run on a CPU?**
*Question:* the docs describe optimized kernels in CUDA terms. *Found by:* the profiler's operator names.
*Reality:* `aten::_scaled_dot_product_flash_attention_for_cpu`, PyTorch's own fused CPU kernel. *Decision:* force the
backend explicitly (`sdpa_flash_cpu`, `sdpa_math`) and **never call it FlashAttention**. *Where:* `algorithms/sdpa.py`;
the docs discrepancy is recorded in `docs/references.md`.

**5. 🧭 Citations checked, not remembered.** A research subagent verified every paper's title, authors, arXiv ID and
the specific claims. Corrections applied: Milakov & Gimelshein compute a single-pass *normalizer*, not a single-pass
softmax; FlashAttention's IO bounds use Θ, not Ω; FlashAttention-2 has a single author; the Zoology paper studies
gated-convolution models (not linear attention) in its main results; induction heads are not called "associative
recall" in their paper. *Where:* `docs/references.md`.

### Phase 2–3: tiled attention and numerical validation

**6. ⚠️ The first bf16 tolerance accepted almost anything.**
*Symptom:* everything passed, but the baseline, naive attention computed in bf16, had errors of 0.147 (magnitude 10).
*Cause:* bf16 logits in the hundreds are rounded to the nearest ~1, which multiplies softmax weights by up to 1.65.
*Found by:* breaking errors down by input magnitude. *Fix:* use a **mixed-precision baseline** (fp32 compute, rounded
output). Naive-in-bf16 now *fails* 14/53 cases, and that is reported as a finding. *Where:* `correctness/lab.py`,
`metrics.py`; `test_finding_naive_in_pure_bfloat16_degrades_with_logit_magnitude`.

**7. 🔍 sdpa_flash_cpu "failed" fp32 at magnitude 10 (2.04e-5 vs 2e-5).**
*Temptation:* raise the threshold. *Instead:* the hypothesis "the error scales with logit size for every implementation"
was tested. *Evidence:* the naive formula in fp32 had the *identical* error, 2.04e-5; errors were 0.1–1.5 × u·max|S|
for all methods. *Fix:* tolerance = max(2e-5, 4·u·max|S|). *Where:* `metrics.py::tolerance_for`;
`test_finding_fp32_error_floor_scales_with_logit_magnitude_for_every_implementation`.

**8. 🐞 Mark the lse output as non-differentiable.**
The custom autograd function returns `(out, lse)`. Without `ctx.mark_non_differentiable(lse)`, the backward would
receive a gradient for `lse` and had an expensive check for it. *Fix:* mark it non-differentiable and drop the check.
*Where:* `tiled_exact.py::_TiledRecomputeFn`.

**9. 🐞 An O(L²) statistic hidden inside an O(L) method.**
local_window's `band_entries` counter built a dense `L × L` band mask. *Fix:* `band_entry_count`, an O(L) formula.
*Where:* `local_window.py`.

**10. 🐞 Theory overcounted ragged tiles.**
The tiled formula `tiles × q_tile × kv_tile` overcounts when the last tile is shorter. *Found by:* the scientific test
comparing theory with instrumented counts. *Fix:* loop over real tile boundaries. *Where:* `complexity/theory.py`.

**11. 🧭 Refactor during a sweep.** The online-softmax update was extracted into `online_softmax_update` while the first
sweep was running; later workers imported the new version. Same operations, plus one function call per tile; noted, and
that sweep's timings were discarded anyway (entry 15).

### Phase 4: benchmarking

**12. 🐞 Leftover no-op line.** A meaningless line (`rec.pop(...) if False else None`) was left in the runner's
`finally` block during writing. Removed.

**13. ⚠️ The memory guard decided for the whole configuration.** Using the forward+backward estimate would have skipped
valid forward measurements. *Fix:* decide **per mode**, with status `partial`. Then a second mistake: the scheduler
briefly stopped a series after a `partial`; reverted, because forward may still fit at the next length.
*Where:* `runner.py::guard`, `scheduler.py`.

**14. 🐞 Workers asked PowerShell for the CPU name every time** (1–2 s each). *Fix:* the parent passes it through the
environment variable `ATTENTION_LAB_CPU_NAME`. *Where:* `env.py`, `runner.py::run_isolated`.

**15. 🔍 The timings made no physical sense, and the machine was shared.**
*Symptom:* identical forward code measured 13.1 vs 34.7 ms; causal slower than non-causal; tiled forward exponent 3.0.
*Found by:* `psutil.cpu_percent` showed all 12 logical CPUs busy; listing processes showed other projects' training
jobs, servers and tests. *Fix:* a series of changes (entries 16–19); that sweep's timing data was declared contaminated.
Its memory data remained valid. *Where:* `docs/results.md` §0.

**16. 🐞 The load monitor recorded nothing for short timings.** The first sample came after 0.2 s, but fast regions had
already finished. *Fix:* prime the counters on entry and always take a final sample on exit. *Where:* `benchmarking/load.py`.

**17. 🐞 CPU time per call read 0.0.** Windows' `process_time` ticks in about 15.6 ms steps. *Fix:* measure over the whole
timed loop and divide by the number of calls. *Where:* `timing.py`.

**18. 🧭 Definitive timing on one thread.** Multi-threaded OpenMP code under contention stalls unevenly. Single-threaded
runs gave CPU ≈ wall time and sensible exponents (1.92–2.03). The quiet-wait timeout was cut from 60 s to 20 s, then to
10 s, because the load never went away. *Where:* `experiments/configs/*.yaml`.

**19. ⚠️ "Other users".** Early comments called the background load "other users' jobs". It came from the owner's other
projects on the same account. Wording corrected everywhere.

**20. ⚠️ Plots with independent y-axes.** The first memory figure gave each method its own axis, which hid the
comparison. *Fix:* added a shared-axis comparison figure with L and L² guides. *Where:* `visualization/plots.py::plot_memory_compare`.

### Phase 5–6: probes, training and tasks

**21. 🔍 Was the probe unfair to linear attention?** It scored at chance everywhere. *Check:* a sharpness sweep over β.
*Result:* it improves with β (L = 16: 6 % → 73 %) but stays far below softmax. The claim was rephrased accordingly.
*Where:* `tasks/probes.py` (sharpness experiments).

**22. 🐞 Training output invisible.** Redirected stdout is block-buffered. *Fix:* `python -u`; rely on per-record
JSONL flushes.

**23. 🔍 Training was starved.** 4 threads on a saturated CPU got about 0.25 cores. *Fix:* 1 thread by default for
`tasks`.

**24. ⚠️ A wrong expectation about step time.** 500 ms per step looked like a bug. *Found by:* profiling. Matrix
multiplies dominated; about 2.5 GFLOP per step is real work. *Fix:* expectation corrected; `--batch-size 32` option.

**25. 🐞 `float(loss)` warning** on a tensor requiring grad → `float(loss.detach())`.

**26. 🔍 MQAR v1 plateau at 25–26 % for every backend.** *Cause:* a data-generator shortcut (values drawn with
replacement, so "predict the most frequent value" scores 25.7 %). *Fix:* v2 with distinct values; `shortcut_baseline`;
`TASK_VERSION`; note in the v1 folder; regression test. *Where:* `tasks/synthetic.py`, Chapter 19.

**27. ⚠️ Concurrent runs garbled a shared log.** Three explorations appended to one file. *Fix:* one diagnostic at a
time, one log each. *Where:* `experiments/diagnostics/mqar_v2/`.

**28. 🔍 MQAR v2 plateau at log(n).** Diagnosed with 10 pipeline tests, validated instruments and three diagnostics.
Not an implementation bug. *Where:* Chapter 19, `docs/mqar_diagnosis.md`.

**29. ⚠️ A false "success": plain induction was solved positionally.** Fixed offset → 100 %; random offset → 18 %. The
earlier conclusion was withdrawn. *Where:* Chapter 19, D2.

**30. 🧭 Use variable-offset induction for the trained comparison**, since the exact model demonstrably learns it (D3).
*Where:* `tasks/synthetic.py::induction`, `cli/main.py::cmd_tasks`.

**31. 🔍 block_sparse = exact at L = 32.** The layout covered 528/528 pairs. Reported as "not evidence about sparsity".

**32. 🔍 The distractor collapse is not ambiguity.** Measured: ambiguous targets 30.1 % (normal) vs 36.7 %
(distractors), yet accuracy fell from 100 % to 18 %. Reported as brittleness, with a labeled hypothesis.

### Phase 7–9: interface, reports and final checks

**33. 🐞 UI crash: `FileNotFoundError ... meta.json`.** The run list included non-sweep folders. *Fix:* list only folders
whose meta has a spec; probes and correctness now write meta. Streamlit had to be restarted to reload modules.
`--headless` added. *Where:* `scheduler.py::list_runs`, `cli/main.py`.

**34. 🐞 Relative spec paths broke outside the project folder.** *Fix:* resolve against the project root, with a clear
error listing the available specs.

**35. ⚠️ Untraceable evidence in a draft.** A draft of the results cited 4-thread numbers from an ad-hoc check that was
never saved. *Fix:* replaced with numbers from saved runs (smoke run and first scaling run). Rule adopted: **every number
must come from a file**.

**36. ⚠️ Arithmetic slips in drafts, caught by re-checking.** "1.0–1.6 %" became 0.95–1.6 %; a causal speed-up of
"1.8–4×" became 1.2–2.4×; the tile ratios "1.3–9×" became 1.1–9.2×; "about 3×" became 3.1–3.5×; the library size was
4,451 lines, not about 4,000. Each was found by recomputing from the data before publishing.

**37. 🐞 Test-suite plumbing.** `tests.conftest` import path; the `pytest-asyncio` warning; the double `-q` that hid
the summary; the profiler warnings (Chapter 21).

**38. ⚠️ "Tiled code is slow" was too simple.** On one thread it matched or beat naive; with more threads it got slower
(4.8× slower at 6 threads than at 1). The limitation text was corrected to the measured, conditional statement.

## The ten most important lessons

1. Verify the environment; do not trust a description of it.
2. Decide edge-case policies (empty rows) by checking a trusted implementation, then test them everywhere.
3. A tolerance is a model of error. Derive it, and change it only when the model is shown to be wrong.
4. When a check fails, form a hypothesis about the cause and test it before changing anything.
5. Measure the conditions of your measurements (background load), and say when they are bad.
6. Validate every instrument on a case with a known answer.
7. Identical results from different mechanisms point at the task or the data, not the mechanisms.
8. Rule out shortcuts before claiming a model learned something.
9. Every number must be traceable to a saved file. Recompute before you publish.
10. Know when an investigation has answered the question your project needs, and stop there deliberately.
