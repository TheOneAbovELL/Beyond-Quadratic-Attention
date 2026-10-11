# Chapter 19 — The MQAR investigation: a detective story about a model that would not learn

> **What you will learn:** how an innocent-looking training experiment stalled at a suspicious accuracy; how a
> flaw in our own data generator was found by computing a "shortcut baseline"; how, after fixing it, the
> models stalled again; how a systematic diagnosis ruled out bugs one by one; how an earlier "success" turned
> out to be a positional shortcut; and why the investigation was deliberately stopped where it was. This
> chapter is also the best illustration in the book of the scientific method applied to machine learning.

## 19.1 The task

**MQAR** (multi-query associative recall) tests whether a model can bind keys to values and retrieve them, a
known weakness of many efficient attention alternatives (Arora et al., 2023). A sequence starts with pairs
`k₁ v₁ k₂ v₂ … k₈ v₈`, followed by noise tokens among which some keys reappear. At each reappeared key the model
must output that key's value. With 16 possible values, random guessing scores 6.25 %.

## 19.2 Act 1: the suspicious plateau (generator v1)

The first quick training run (600 steps) gave, for every backend that could see the pairs:

| backend | accuracy (8 pairs) | accuracy (12 pairs) |
|---|---|---|
| exact | 0.258 | 0.222 |
| block_sparse | 0.253 | 0.197 |
| linear_elu | 0.256 | 0.214 |
| local_window (cannot see the pairs) | 0.068 | 0.064 |

Everyone except local_window landed at almost the *same* value, about 25–26 %. A separate learning-curve run of the
exact model was stuck at 24–26 % from step 250 to step 1,500. Identical plateaus across very different mechanisms
suggest that the plateau is a property of the **task**, not of the models.

**Hypothesis:** the models had found a way to score 25 % *without* binding keys to values. In generator v1, values were
drawn **with replacement**, so a sequence often contained the same value twice. A model that ignores keys and
**predicts the most frequent value in the context** would then be right fairly often.

**Test:** compute that strategy's accuracy on the exact evaluation set: **25.7 % for 8 pairs, 21.4 % for 12 pairs.**
That matches the observed plateaus (25–26 % and 20–22 %) almost exactly.

**Fix (generator v2):** draw values **without replacement**, so each value appears once and the best binding-free
strategy is a uniform guess among the `n` values present: exactly `1/n_pairs`. A `shortcut_baseline` metric was added
to every MQAR evaluation, a `TASK_VERSION` constant was introduced so old and new results are never mixed, a note was
placed in the v1 results folder, and a regression test pins the fix.

## 19.3 Act 2: stuck again, at a different number

With generator v2, the models stalled at exactly the new baseline:

* 4 pairs: loss **1.386 = log 4**, accuracy 25 %, unchanged after 3,000 steps;
* 8 pairs: loss 2.08, accuracy 12.5 %, unchanged at 2,500 steps.

A loss of `log(n)` means the model spreads its probability evenly over the `n` values present: it knows *which
values are in the context* but not *which one belongs to the key*. Changing the learning rate (3e-3 → 1e-2) and the
training loop (constant learning rate, no warm-up, no clipping) changed nothing.

An apparently strong clue existed: a simple **induction** task (the second half of the sequence repeats the first,
predict the next token) had reached 100 % with the same model. That seemed to show the model *could* learn retrieval,
which would point at something MQAR-specific.

## 19.4 Act 3: a systematic diagnosis

At this point the instruction was: stop launching runs, preserve everything, diagnose rigorously, one experiment at
a time, and distinguish bugs from optimization problems, insufficient training and genuine difficulty.

**Lesson recorded on the way:** three exploration runs had been writing to the same log file at once, which garbled
the lines. From then on, each diagnostic ran alone with its own log.

### Step 1: targeted tests for every link of the pipeline

`tests/unit/test_mqar_pipeline.py`, 10 tests, **all passing**:

| suspected failure | result |
|---|---|
| wrong token types in regions | correct |
| duplicate keys or values; unbound query keys | none |
| the answer is leaked or derivable without binding | each answer occurs exactly once, right after its key |
| biased query/target marginals | uniform |
| loss at wrong positions or shifted targets | loss only at the 4 query positions, aligned correctly |
| plateau value | "uniform over the 4 context values" gives *exactly* log 4 |
| query cannot see its key or the pairs (mask bug); future leakage | visible; no leakage |
| dead parameters / broken gradient flow | every parameter gets a gradient |
| metric bug | oracle 100 %, shortcut 25 % |

Since the exact attention backends compute the same function to about 1e-6 (Chapter 11), **the attention
implementation cannot be the cause**.

### Step 2: build trustworthy instruments

`src/attention_lab/diagnostics/recall.py` records, during training: per-module gradient norms, per-module relative
update sizes, prediction categories (correct / another context value / absent value / key / other), and **circuit
probes**:

* `prevtok_l0`: how much layer-0 attention at each value position goes to the position just before it (its key).
  An induction circuit needs this "previous-token head";
* `target_l1`: how much layer-1 attention from the query goes to the *correct* value position, compared with the other
  values.

Before use, the instruments were themselves tested: the attention maps they replay reproduce each layer's real output
to 1e-5, and the probes read exactly the intended positions on hand-made maps (`tests/unit/test_diagnostics.py`).

### Step 3 (D1): can the model fit MQAR data at all?

64 fixed sequences, trained repeatedly for 2,000 steps:

| step | train acc | held-out acc | held-out loss | prev-token attn (L0) | attn on correct value (L1) | on each other value |
|---|---|---|---|---|---|---|
| 1 | 0.14 | 0.08 | 4.03 | 0.27 | 0.06 | ~0.06 |
| 100 | **1.00** | 0.20 | 5.37 | 0.37 | 0.14 | ~0.14 |
| 2000 | **1.00** | **0.20** | 7.84 | 0.36 | 0.13 | ~0.13 |

The model **memorized** the 64 sequences perfectly (so forward, backward, loss and optimizer all work), but held-out
accuracy stayed below the 25 % shortcut and held-out loss rose: classic memorization. The binding circuit never formed.
Layer-1 attention weighted the correct value no more than any other.

### Step 4 (D2): was the induction "success" real?

The induction task that reached 100 % had a hidden property: the second half always started exactly 12 positions
after the first, so **the answer was always at the same relative offset**. With RoPE (Chapter 4), a single head can
learn "copy from 12 positions back" without matching any content. To test this, the same task was trained with a fixed
offset and evaluated with a **random gap** (offset 12–20):

| step | accuracy, fixed offset | accuracy, random offset |
|---|---|---|
| 100 | 1.00 | 0.18 |
| 2000 | 1.00 | **0.18** |

**It was a positional shortcut.** The earlier conclusion "the model can learn content-based retrieval" was
unsupported, and was withdrawn.

### Step 5 (D3): can the model learn *content-based* induction?

Trained directly on random offsets, where content matching is required:

| step | held-out acc | layer-0 attention gradient norm |
|---|---|---|
| 100 | 0.09 | 0.30 |
| 200 | 0.24 | **0.75** |
| 500 | **0.999** | 0.52 |
| 1000 | **1.00** | 0.003 |

Yes, and quickly. The spike in layer-0 gradient just before accuracy jumps is consistent with a previous-token circuit
forming. So the architecture and the training loop **can** learn content-based retrieval.

## 19.5 The verdict

| hypothesis | verdict |
|---|---|
| bug in the exact attention implementations | **rejected** (numerical equivalence; identical model outputs) |
| bug in generator, targets, loss, masks, metric, gradients | **rejected** for v2 (10 targeted tests) |
| v1 generator shortcut | **confirmed and fixed** |
| the model cannot fit MQAR data | **rejected** (D1: 100 % train) |
| the architecture cannot learn content-based retrieval | **rejected** (D3: 100 %) |
| "plain induction works" proves content-based retrieval | **rejected** (D2: positional shortcut) |
| learning rate / loop settings are the cause | **not supported** (same plateau with several settings; not exhaustively swept) |
| insufficient training budget | **open** |
| MQAR-specific optimization difficulty (sparser supervision; the "uniform over context values" solution as an attractor) | **plausible, untested** |

## 19.6 Why the investigation stopped here

The project's question is about the **efficiency and correctness of attention algorithms**. The diagnosis had
established that the MQAR plateau says nothing about the attention implementations. Resolving MQAR's learning
dynamics would be a separate research project. So the decision was:

* **preserve** everything (logs in `experiments/diagnostics/mqar_v2/`, the full write-up in `docs/mqar_diagnosis.md`);
* **use the task the exact model demonstrably learns** (variable-offset induction, D3) for the trained quality
  comparison (Chapter 18);
* keep MQAR available (`python lab.py tasks --which M`) and list concrete next experiments: more queries per sequence
  (denser supervision), a curriculum 1 → 2 → 4 pairs, and a longer budget with the circuit probes, run one at a time.

## 19.7 Lessons

1. **Identical plateaus across different models point at the task.** Compute the best "cheating" baseline.
2. **A loss equal to `log(n)` is a fingerprint**: uniform over `n` options.
3. **Test each link of a pipeline separately** before blaming optimization.
4. **Validate the instruments** before trusting what they show.
5. **"It learned X" requires ruling out shortcuts.** Position, frequency and other cues can mimic the intended
   skill. Test out of distribution.
6. **Know when to stop.** Answer the question your project needs answered, document the rest, and move on.

## Key points to remember

* v1 plateau 25–26 % = "predict the most frequent context value" (25.7 %), a generator flaw, fixed in v2.
* v2 plateau at loss log(n) = uniform over context values; no binding learned.
* 10 pipeline tests pass; the model memorizes (D1) but the circuit never forms.
* The induction "success" was positional (fixed 100 %, random-gap 18 %); content-based induction is learnable (D3: 100 % by step 1000).
* The cause is MQAR-specific learning dynamics or budget, not the attention code. Stopped and documented deliberately.
