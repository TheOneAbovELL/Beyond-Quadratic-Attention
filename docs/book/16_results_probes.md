# Chapter 16 — Results III: what a single attention operation can retrieve (probes)

> **What you will learn:** how to test what an attention method *can* do, without any training, using
> hand-constructed inputs; the four probe experiments (distance, length, decoys, sharpness); the results for
> each method; and a fairness check that changed how the linear-attention result is stated.

Run: `20261011-195716_probes` (`python lab.py probes --full`, 128 trials per point). Code:
`src/attention_lab/tasks/probes.py`. Report: `experiments/reports/20261011-195716_probes/probes.md`.

## 16.1 The idea of a probe

Before training any model, we can ask a sharper question: **if the right answer is stored at one specific
position, can this attention operation route it to the query?** That tests the operation itself, free of
any learning effects.

**Construction** (one trial = one batch element, one head):

* `L` keys: random directions, scaled so that a query equal to a key direction gets a score
  `q·k/√d = β` (default **β = 10**), while unrelated keys get scores of typical size `β/√d`.
* The **query** sits at the last position. It is set equal to the key at the **target** position
  `t = L − 1 − distance` (the "stored association").
* **Values** are random vectors. The method's output at the last position is decoded as "whichever value
  vector it is most similar to" (cosine similarity). The trial succeeds if that is the target's value.
* **Decoys** (optional): some other keys are made similar to the query (cosine 0.9), which gives them scores
  near `0.9β`. These are misleading near-matches.

Every method receives *identical* q, k, v per trial, with causal attention.

## 16.2 Distance sweep (L = 1024)

| distance to target | 1–63 | 64 | 65 | 128 | 256 | 512 | 1023 |
|---|---|---|---|---|---|---|---|
| exact softmax | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| local_window (w = 64) | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| block_sparse (b = 64, ±1 local, global 0, 1 random) | 1.00 | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 | **1.00** |
| linear_elu | ≤ 0.008 | 0 | 0 | 0 | 0.008 | 0.008 | 0.008 |

* **local_window** drops from 100 % to 0 % between distance 64 and 65: exactly the window edge.
* **block_sparse** succeeds within its own and the neighbouring block, fails at 128–512, and **succeeds again at
  1023**, because the target at position 0 is inside global block 0. What a block-sparse method can reach is a
  property of its layout, not of distance.
* **Exact softmax** succeeds everywhere.
* **linear_elu** is at chance level (≈ 1/L).

![Figure 16.1 — Retrieval accuracy vs distance at L = 1024 (run 20261011-195716_probes). The dashed line marks the local window w = 64.](../../experiments/reports/20261011-195716_probes/probe_distance.png)

## 16.3 Length sweep

With the target 8 tokens back, exact, local and block-sparse attention scored 100 % for L = 128 … 4096. With the
target at position 0 (distance L − 1), exact softmax stayed at **100 % up to L = 4096**: the target's weight was
never diluted away at β = 10. local_window scored 0 % (out of reach), block_sparse 100 % (global block), and
linear_elu was at chance.

## 16.4 Decoys: interference cuts both ways (L = 512, target 8 back)

| decoys (cos 0.9) | 0 | 4 | 8 | 16 |
|---|---|---|---|---|
| exact softmax | 1.00 | 1.00 | 0.977 | **0.867** |
| local_window | 1.00 | 1.00 | 1.00 | 1.00 |
| block_sparse | 1.00 | 1.00 | 1.00 | 1.00 |
| linear_elu | 0 | 0 | 0 | 0 |

Exact attention sees every decoy, and with 16 near-matches it starts confusing them with the target. The restricted
methods are **immune here, because the decoys fall outside their pattern**: they cannot see them. A restriction can
protect against distractors it cannot see, and it can equally hide a needed target it cannot see. Both effects follow
from the same mechanism.

![Figure 16.2 — Accuracy vs number of decoy keys (cosine 0.9) at L = 512.](../../experiments/reports/20261011-195716_probes/probe_decoys.png)

## 16.5 The fairness check on linear attention

Reporting "linear attention retrieves at chance" would have been misleading if the probe was simply unfair to it,
for example if β = 10 suited softmax but not elu + 1. So a **sharpness sweep** was added: distance 4, L = 16 and 64,
β = 1 … 100.

| β | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|
| exact softmax, L = 16 | 0.914 | 1 | 1 | 1 | 1 |
| linear_elu, L = 16 | 0.062 | 0.125 | 0.242 | 0.469 | **0.727** |
| exact softmax, L = 64 | 0.305 | 1 | 1 | 1 | 1 |
| linear_elu, L = 64 | 0.023 | 0.031 | 0.031 | 0.086 | **0.188** |

So linear attention is **not zero by construction**. Its accuracy rises with input scale, but slowly, and it falls
quickly with L. The reason: `exp(q·k)` sharpens **exponentially** as scores grow, so one key can take nearly all the
weight. `φ(q)·φ(k)` with `φ = elu + 1` grows only about **polynomially**, so the target's weight stays comparable to
the sum of many others. That is the precise statement the project makes, rather than "linear attention cannot
retrieve". Chapter 18 adds that a *trained* linear model can do well when there are few candidates.

![Figure 16.3 — Sharpness check: accuracy vs target-logit scale β at L = 16.](../../experiments/reports/20261011-195716_probes/probe_sharpness_L16.png)

## 16.6 What probes do and do not show

Probes test **one attention operation with hand-made inputs**. A trained multi-layer model can learn inputs that suit
its attention, and can route information around restrictions by relaying it through intermediate tokens. Both effects
appear in Chapter 18. Probes show the *mechanism*; trained tasks show the *consequences*.

## Key points to remember

* Probe = known target position + identical inputs for every method + nearest-value decoding.
* local_window: 100 % → 0 % exactly at the window edge. block_sparse: reach is set by the layout (global block 0 rescues distance 1023).
* Exact softmax: 100 % to L = 4096; degrades with many decoys (86.7 % at 16).
* Restricted methods were immune to decoys outside their pattern.
* Fairness check: linear_elu improves with β (L = 16: 6 % → 73 %) but stays far below softmax, because elu + 1 is polynomially, not exponentially, sharp.
