# Chapter 4 — Transformers and attention, from scratch

> **What you will learn:** what tokens and embeddings are; what a Transformer is made of; exactly how
> attention computes its output, worked through by hand with real numbers; why the scores are divided by
> √d; multiple heads; masks (causal, padding, custom); and how a model knows *where* each token is
> (positional encodings, including RoPE, which this project uses).

## 4.1 From text to vectors: tokens and embeddings

A language model cannot read letters directly. Text is first cut into **tokens** (words or word pieces),
and each token is replaced by an integer ID from a fixed **vocabulary**. In this project the synthetic tasks
use a vocabulary of 64 IDs (Chapter 17), for example "ID 9 = a key symbol" and "ID 25 = a value symbol".

Each ID is then turned into a vector by an **embedding** table: a learned matrix with one row per vocabulary
entry. Token ID 9 becomes row 9, a vector of `d_model` numbers (64 in our models). A sequence of `L` tokens
becomes an `L × d_model` matrix.

## 4.2 The Transformer in one picture

A (decoder-only) Transformer is a stack of identical **blocks**. Each block has two parts, each wrapped in a
"residual connection" (the input is added back to the output) and preceded by a **LayerNorm**, which
rescales each token vector to a standard size:

```
x  →  x + Attention(LayerNorm(x))      ← tokens exchange information
   →  x + MLP(LayerNorm(x))            ← each token is processed on its own
```

After the last block, a final LayerNorm and a linear "head" turn each token vector into scores over the
vocabulary: the model's prediction. The **MLP** (multi-layer perceptron) is two matrix multiplies with a
nonlinearity (GELU) between them. It processes each token independently. **Attention is the only place where
tokens interact.** That is why attention gets all the attention, and why its cost dominates long texts.

> **In the code:** `src/attention_lab/models/transformer.py` — `TinyTransformer`, `Block`, `Attention`.
> The `Attention` module computes Q, K and V, then calls whichever attention algorithm you choose by
> name. This "plug-in" design (Chapter 6) lets the same model run with any of the eight implementations.

## 4.3 Queries, keys and values

For each token, attention computes three vectors by multiplying its representation with three learned
matrices:

* **Query (q)**: "what am I looking for?"
* **Key (k)**: "what do I contain?"
* **Value (v)**: "what will I hand over if you pick me?"

For a sequence, these form three matrices `Q`, `K`, `V`, each of shape `L × d` (one row per token). Then:

```
S = Q Kᵀ · (1/√d)          scores: how relevant is token j to token i?        (L × L)
P = softmax(S, per row)    attention weights: each row sums to 1                (L × L)
O = P V                    output: each token's weighted average of values      (L × d)
```

In one line, as in the original Transformer paper (Vaswani et al., 2017):
**Attention(Q, K, V) = softmax(QKᵀ/√d) V.**

## 4.4 A worked example by hand

Three tokens, `d = 2`. (The numbers are invented for illustration, then checked with the project's own
`naive_attention`.)

```
Q = [[1, 0],      K = [[2, 0],      V = [[1, 0],
     [0, 2],           [0, 1],           [0, 1],
     [1, 1]]           [1, 1]]           [5, 5]]
```

**Step 1: scores `S = Q Kᵀ`.** Entry `[i][j]` is query `i` dotted with key `j`:

```
token 0: q=[1,0]: with k0=[2,0] → 2;  with k1=[0,1] → 0;  with k2=[1,1] → 1
token 1: q=[0,2]: → 0, 2, 2
token 2: q=[1,1]: → 2, 1, 2

S = [[2, 0, 1],
     [0, 2, 2],
     [2, 1, 2]]
```

**Step 2: scale by `1/√d = 1/√2 ≈ 0.7071`:**

```
[[1.4142, 0.0000, 0.7071],
 [0.0000, 1.4142, 1.4142],
 [1.4142, 0.7071, 1.4142]]
```

**Step 3: softmax each row.** Row 0: `exp(1.4142) ≈ 4.113`, `exp(0) = 1`, `exp(0.7071) ≈ 2.028`; total
≈ 7.141, so the weights are `[0.576, 0.140, 0.284]`. Doing all rows:

```
P = [[0.5760, 0.1400, 0.2840],
     [0.1084, 0.4458, 0.4458],
     [0.4011, 0.1978, 0.4011]]
```

Every row sums to 1. Token 0 pays most attention to token 0. Token 1 splits equally between tokens 1 and 2.

**Step 4: output `O = P V`.** Each output row is a weighted average of the rows of `V`:

```
token 0: 0.576·[1,0] + 0.140·[0,1] + 0.284·[5,5] = [1.996, 1.560]
token 1: = [2.337, 2.675]
token 2: = [2.407, 2.203]
```

The project's `naive_attention` returns `[1.99595, 1.56001]` for token 0, matching to rounding.

## 4.5 Why divide by √d?

If the entries of `q` and `k` are random with mean 0 and variance 1, their dot product (a sum of `d`
products) has variance `d`. With `d = 64` the typical score would be about ±8, and the softmax would almost
always put nearly all weight on one token. Its gradients would then be tiny, and learning would stall.
Dividing by `√d` brings the variance back to 1 (Vaswani et al., 2017, footnote 4). The project's default is
`scale = d^(−1/2)`, and a test checks it (`test_scale_is_one_over_sqrt_d`).

## 4.6 Multiple heads

Instead of one attention with vectors of size `d_model`, the model splits the vectors into `H` **heads** of
size `d = d_model / H` and runs `H` independent attentions in parallel. Each head can specialize: one may
look at the previous token, another for matching content. The outputs are concatenated and mixed by a final
linear layer. All of this is why our tensors are 4-D `[B, H, L, d]`: batch × heads × length × head-dimension.

## 4.7 Masks: forbidding some pairs

Often some query–key pairs must not interact. A mask removes them by setting their score to `−∞` before the
softmax, because `exp(−∞) = 0` gives them zero weight.

* **Causal mask.** A model that predicts the next token must not peek at future tokens: token `i` may
  attend only to `j ≤ i`. In the example above the causal weights become
  `[[1, 0, 0], [0.196, 0.804, 0], [0.401, 0.198, 0.401]]`: token 0 can only see itself, so its output is
  exactly `v0 = [1, 0]`.
* **Padding mask.** When texts of different lengths are batched together, short ones are padded with
  filler tokens that must be ignored. Our `key_padding_mask` has shape `[B, L]`, with `True` meaning
  "real token".
* **Custom masks.** Any boolean table (`True` = allowed) or an additive table of numbers (added to the
  scores, `−∞` removes) can be supplied.

The project defines the semantics of all masks **in one place** (`algorithms/base.py`, class `MaskSpec`),
matching PyTorch's own convention (Chapter 7).

**Edge case: a fully-masked row.** If every key is forbidden for some query, the softmax is `0/0`, which is
undefined. The project's policy is **output zero and zero gradient**, chosen after checking that this is
what PyTorch's own CPU kernels do. The story is in Chapter 7.

## 4.8 Where am I? Positional information

Attention by itself ignores order: shuffle the tokens and each output is the same weighted average,
just shuffled. Models therefore add **positional information**:

* **Sinusoidal absolute encodings** add a fixed pattern of sines and cosines, different for each
  position, to the embeddings (Vaswani et al., 2017).
* **Rotary position embedding (RoPE)** rotates each query and key vector by an angle proportional to its
  position (Su et al., 2021). The dot product of a rotated query at position `i` and a rotated key at
  position `j` then depends only on the **difference** `i − j`. The model therefore naturally sees
  *relative* positions. Our models use RoPE by default, and a test checks the relative-position property
  (`test_rope_scores_depend_only_on_relative_position`).

RoPE is applied **before** the attention algorithm is called, so every attention implementation receives
the same Q and K. Keep RoPE's relative-position nature in mind: in Chapter 19 it is exactly what allowed a
model to solve a supposedly content-based task with a purely positional trick.

## 4.9 Exact versus approximate attention

Everything above is **exact attention**: every allowed pair is scored, and the softmax is computed exactly
(up to rounding). The project also studies methods that change the computation:

* **Restricted attention** (local window, block sparse) still uses the exact softmax but forbids many pairs,
  as if a fixed mask were applied.
* **Linear attention** replaces `exp(q·k)` with a different similarity that can be computed faster. It is
  not softmax attention at all, just a related function.

These are different research questions. For exact methods we ask "do they compute the same thing, and how
fast?" For approximate methods we ask "how fast, and what do we lose?" (Chapter 10).

## Key points to remember

* Attention(Q, K, V) = softmax(QKᵀ/√d)V; each output is a weighted average of value vectors.
* `QKᵀ` is an `L × L` table, the source of the quadratic cost.
* Divide by √d so that scores have unit variance.
* Masks set forbidden scores to `−∞`. Causal: `j ≤ i`. Fully-masked rows output 0 by project policy.
* Our tensors are `[B, H, L, d]`. Heads are independent attentions.
* RoPE makes attention depend on relative position. That helps the model, and can also create shortcuts.
