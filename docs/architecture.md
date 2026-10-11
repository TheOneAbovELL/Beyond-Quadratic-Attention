# Architecture

```
lab.py                       launcher (adds src/ to sys.path; same as `python -m attention_lab`)
src/attention_lab/
  algorithms/                computation only — no I/O, no timing
    base.py                  shape validation, MaskSpec (one definition of mask semantics), stable-softmax helpers
    naive.py                 Implementation A: reference
    tiled_exact.py           Implementation B: online softmax; autograd and recompute-backward variants
    sdpa.py                  Implementation C: adapters forcing PyTorch SDPA's flash-CPU or math kernel
    local_window.py          approximate 1: banded softmax (vectorized chunks)
    block_sparse.py          approximate 2: block layout (local + global + random), gather execution
    linear.py                approximate 3: elu+1 linear attention (non-causal + chunked causal)
    registry.py              name -> AttentionMethod(fn, family, what it computes, defaults, capabilities)
  correctness/               metrics (error definitions, tolerance policy) + correctness laboratory
  complexity/theory.py       analytical work/memory model (outputs always labeled theory_*)
  benchmarking/              env capture, timing, memory instruments, load monitor, runner, worker, scheduler
  tasks/                     probes.py (attention-level, no training), synthetic.py (token tasks A-D)
  models/transformer.py      tiny pre-LN Transformer; attention backend = registry name
  training/train.py          training loop + multi-method task experiments
  reporting/                 results -> tables, exponents, markdown; probe and task reports
  visualization/plots.py     matplotlib figures from measured data (theory dashed + labeled)
  cli/main.py                every experiment as a subcommand
  ui/app.py                  Streamlit interface over the same backend functions
tests/                       unit / numerical / integration / scientific / perf_smoke
experiments/configs/         YAML sweep specs (each states its hypothesis)
experiments/results/         raw JSONL + meta.json per run (machine-readable, never edited by hand)
experiments/reports/         generated markdown, CSV and PNG
docs/                        theory, methodology, findings
```

## Design decisions

1. **One calling convention.** Every method has the signature
   `fn(q, k, v, *, causal, key_padding_mask, [mask], **params)` on `[B, H, L, d]`
   tensors. The model, benchmarks, probes and UI are therefore written once and
   switch implementations by name.
2. **Mask semantics live in one class** (`MaskSpec`). Algorithms ask it for the mask of
   the tile they are about to compute, so the tiled code never builds an `L×L` mask.
   A dense user mask is sliced through a broadcast view.
3. **Exactness is declared, then tested.** The registry records each method's
   `family` (`exact`, `restricted-softmax`, `different-kernel`) and the precise
   function it computes. Tests check exact methods against `naive`, and restricted
   methods against `naive` with the corresponding mask.
4. **Theory and measurement are separated by name.** `theory_*` fields come from
   `complexity/theory.py`. Measured fields have no prefix and state their scope (see
   `benchmark_methodology.md`).
5. **Failures are data.** Skipped, partial, errored and timed-out configurations are
   written to the same JSONL with a reason.
6. **Device-agnostic algorithms.** No algorithm references CPU specifics. Device
   concerns (synchronization, memory counters) live in `benchmarking/`, so GPU
   support means adding instrumentation, not rewriting algorithms.
7. **Local only.** No network calls, no telemetry. Streamlit is launched bound to
   localhost with usage statistics disabled.
