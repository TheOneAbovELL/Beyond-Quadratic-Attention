# Chapter 17 — The tiny Transformer and how it was trained

> **What you will learn:** the exact model used for the quality experiments; how its attention is swapped by
> name; how the training loop works; how fairness between backends is enforced; the synthetic tasks and their
> generators; and the practical lessons about training on a busy CPU.

## 17.1 The model

> **In the code:** `src/attention_lab/models/transformer.py`.

`TinyTransformer(ModelConfig(...))` is a decoder-only, pre-LayerNorm Transformer:

```
tokens → Embedding(64 → 64) → [Block] × 2 → LayerNorm → Linear(64 → 64) → logits
Block: x = x + Attention(LayerNorm(x));  x = x + MLP(LayerNorm(x))
MLP:   Linear(64 → 256) → GELU → Linear(256 → 64)
```

Defaults: `vocab_size 64`, `d_model 64`, `n_heads 4` (head dimension 16), `n_layers 2`, `d_mlp 256`, positional
encoding **RoPE**, causal attention. About 100,000 parameters. Weights are initialized from a normal distribution
with standard deviation 0.02.

**Pluggable attention.** The `Attention` module computes Q, K and V with one linear layer, reshapes them to
`[B, H, L, d]`, applies RoPE, then calls `self.backend(q, k, v, causal=..., **params)`, where `backend` is a
registry entry chosen by name. `model.set_attention("local_window", {"window": 8})` swaps the algorithm in every
layer **without touching any weights**. That makes "swap" experiments possible: train with exact attention, then
evaluate the same weights with another algorithm.

**Tested properties** (`tests/integration/test_transformer.py`): output shapes for every positional mode;
**exact backends give identical logits** (to 1e-10 in fp64) and gradients (relative difference < 1e-4 in fp32);
an approximate backend changes the logits by a measurable amount; **no backend lets future tokens affect past
logits**; RoPE scores depend only on relative position; a training step reduces the loss.

## 17.2 The training loop

> **In the code:** `src/attention_lab/training/train.py`, `train(TrainConfig)`.

Each step draws a fresh batch from the task generator, computes the cross-entropy loss **only at positions with
a target** (others are labeled −100 and ignored), backpropagates, clips the gradient norm to 1.0, and takes an
AdamW step (lr 3e-3, weight decay 0.01, betas 0.9/0.98) with 100 warm-up steps and cosine decay. Every
`eval_every` steps it evaluates on a fixed held-out set.

**Fairness controls** in `run_task_experiment`: every backend gets the same initialization seed, the same stream
of training batches (same generator seed), the same step budget and optimizer, and the **same fixed evaluation
sets** (`fixed_eval_set`, seeded separately from training). Results go to JSONL with the full history. Checkpoints
are saved locally to `experiments/checkpoints/` (excluded from version control).

## 17.3 The synthetic tasks

> **In the code:** `src/attention_lab/tasks/synthetic.py`.

Vocabulary layout: 0 = padding; 1–8 = NOISE tokens; 9–24 = KEY tokens; 25–40 = VALUE tokens.

* **Task A, local copy (`local_copy`)**: random tokens; the target at position `i` is the token at `i − lag`
  (lag 3). It only needs information 3 positions back.
* **Task B, variable-offset induction (`induction`)**: a random segment of 12 tokens, a gap of 0–8 noise tokens,
  then the same segment again. Inside the repeat, each target is the **next** segment token. To produce it, the
  model must find where the current token occurred before and copy its successor. Because the gap varies, the
  offset (12–20) varies, so **the source must be found by content, not by position**.
  * **Task C (interference):** the same, with the gap filled by segment-vocabulary tokens (distractors).
  * **Task D (offset generalization):** evaluated at L = 64 with offsets 28–36 or 36–52, never seen in training.
* **MQAR (`mqar`)**: multi-query associative recall in the style of Zoology (Arora et al., 2023). Key–value pairs
  `k₁ v₁ k₂ v₂ …`, then noise in which some keys reappear; the target at a reappeared key is its bound value. Its
  story is Chapter 19.

Each generator also returns `distance` (how far back the answer's source is) and `source` (its position), so
accuracy can be broken down by distance. A shortcut-baseline function (`shortcut_baseline`) computes the best
possible accuracy *without* binding keys to values (Chapter 19).

## 17.4 Lessons from training on a busy CPU

> **What actually happened (1): starved threads.** The first quick training run used 4 threads. After 7 minutes
> it had not finished its first 200-step model. The process had used only 102 CPU-seconds in about 420 seconds of
> wall time, a quarter of one core. With all 12 logical CPUs busy, a 4-thread OpenMP job running tiny matrices
> stalls at every synchronization point. Fix: **train with 1 thread** (now the CLI default for `tasks`), which is
> faster *and* more robust for models this small.
>
> **What actually happened (2): "why is one step 500 ms?"** Even single-threaded, one step took about 500 ms,
> which looked impossibly slow for a 100k-parameter model. Before changing anything, the step was profiled. Matrix
> multiplications dominated, at about 4 ms each. The arithmetic explains it: a batch of 64 sequences × 64 tokens is
> 4,096 tokens, and training costs about 6 × parameters × tokens ≈ 2.5 GFLOP per step, on one contended core. The
> expectation of 10–20 ms had simply been wrong. Fix: batch size 32 (a `--batch-size` option was added).
>
> **What actually happened (3): invisible progress.** Output redirected to a file showed nothing for minutes, because
> Python buffers stdout when it is not a terminal. Fix: run with `python -u` (unbuffered), and rely on the JSONL
> results, which are flushed after every record.
>
> **What actually happened (4): a misleading warning.** `float(loss)` on a tensor that requires gradients triggered
> a PyTorch warning. Fix: `float(loss.detach())`.

## Key points to remember

* 2-layer, d_model 64, 4 heads, RoPE, about 100k parameters; attention is swapped by name with weights untouched.
* Exact backends give identical model outputs; no backend leaks future tokens.
* Fairness: same seed, data stream, budget, optimizer and fixed evaluation sets for every backend.
* Tasks: A local copy; B/C/D variable-offset induction (content-based); MQAR.
* On a busy CPU, tiny models train faster on 1 thread; profile before assuming code is slow.
