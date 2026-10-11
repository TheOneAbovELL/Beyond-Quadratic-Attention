# Chapter 2 — The mathematics you need, from zero

> **What you will learn:** numbers in lists (vectors) and tables (matrices); the dot product; matrix
> multiplication; exponentials and logarithms; the softmax function; what a gradient is and why
> training needs it; and "big-O" notation for describing how cost grows. Everything else in the
> book builds on these ideas.

## 2.1 Vectors: lists of numbers

A **vector** is an ordered list of numbers, for example `[2, -1, 0.5]`. The number of entries is its
**dimension**; this one has dimension 3. In AI, a vector is how we represent *a thing*: a word, an
image patch, a position. Each entry captures some aspect of it, though usually no single entry has an
obvious human meaning.

In this project, each token is represented by vectors of dimension **d** (we mostly used `d = 64`).

## 2.2 The dot product: how similar are two vectors?

The **dot product** of two vectors of the same dimension multiplies matching entries and adds the results:

```
a = [1, 2, 3]
b = [4, 0, -1]
a · b = 1×4 + 2×0 + 3×(−1) = 4 + 0 − 3 = 1
```

Why it matters: when two vectors "point in the same direction" their dot product is large and
positive. When they point in opposite directions it is large and negative. When they are unrelated it
is near zero. **Attention uses the dot product to measure how relevant one token is to another.**

Computing one dot product of dimension `d` costs `d` multiplications and `d − 1` additions. In this
book one "multiply-add" counts as **2 floating-point operations (FLOPs)**, so a dot product costs about
`2d` FLOPs.

## 2.3 Matrices and matrix multiplication

A **matrix** is a rectangular table of numbers with *rows* and *columns*. A matrix with `L` rows and `d`
columns is called "`L × d`". You can think of it as `L` vectors stacked on top of each other, one per row.

**Matrix multiplication** `C = A × B` (requires `A` to be `m × k` and `B` to be `k × n`) produces an `m × n`
matrix where **each entry `C[i][j]` is the dot product of row `i` of `A` with column `j` of `B`**:

```
A (2×3) = [[1, 2, 3],        B (3×2) = [[1, 0],
           [0, 1, 0]]                   [0, 1],
                                        [1, 1]]

C = A × B (2×2):
C[0][0] = 1·1 + 2·0 + 3·1 = 4      C[0][1] = 1·0 + 2·1 + 3·1 = 5
C[1][0] = 0·1 + 1·0 + 0·1 = 0      C[1][1] = 0·0 + 1·1 + 0·1 = 1
```

Cost: `m × n` dot products of length `k`, so about **`2·m·n·k` FLOPs**. This formula is used throughout the
book.

The **transpose** of a matrix, written `Bᵀ`, swaps rows and columns: an `L × d` matrix becomes `d × L`. If
`Q` and `K` are both `L × d`, then `Q × Kᵀ` is `L × L`, and entry `[i][j]` is the dot product of row `i` of
`Q` with row `j` of `K`. **That single matrix product computes the similarity of every token with every
other token, and its `L × L` shape is where the "quadratic" in this project's title comes from.**

### Tensors

A **tensor** is the generalization to any number of dimensions: a vector is a 1-D tensor, a matrix is
2-D, and so on. Attention in this project uses 4-D tensors shaped `[B, H, L, d]`:

* `B`: **batch**, how many independent texts are processed together;
* `H`: **heads**, how many independent attention computations run side by side (Chapter 4);
* `L`: **length**, how many tokens per text;
* `d`: **head dimension**, the length of each token's vector.

## 2.4 Exponentials and logarithms

The **exponential** `exp(x) = eˣ` (with `e ≈ 2.71828`) turns any number into a positive number:
`exp(0) = 1`, `exp(1) ≈ 2.718`, `exp(−1) ≈ 0.368`, `exp(10) ≈ 22,026`, `exp(−10) ≈ 0.0000454`.
Two properties matter here:

1. It is **always positive**, which makes it useful for building weights.
2. It **grows explosively**. A difference of 10 in the input is a factor of about 22,000 in the output.
   That is exactly what lets attention focus sharply on one item. It is also what causes overflow:
   `exp(89)` is already too large for a 32-bit float (Chapter 3).

A key identity: **`exp(a − b) = exp(a) / exp(b)`**. Online softmax (Chapter 8) is built on it.

The **logarithm** `log(x)` is the inverse: `log(exp(x)) = x`. The quantity `log(Σ exp(xᵢ))` is called
**log-sum-exp** (LSE). It appears in Chapter 8 as the single number per row that the memory-efficient
backward pass stores.

## 2.5 Softmax: turning scores into weights

Given a list of scores `x₁, …, xₙ`, the **softmax** turns them into weights that are positive and sum
to 1:

```
softmax(x)ᵢ = exp(xᵢ) / (exp(x₁) + exp(x₂) + … + exp(xₙ))
```

Worked example, scores `[2, 1, 0]`:

```
exp(2) ≈ 7.389,  exp(1) ≈ 2.718,  exp(0) = 1        total ≈ 11.107
weights ≈ [7.389/11.107, 2.718/11.107, 1/11.107] ≈ [0.665, 0.245, 0.090]
```

The largest score gets most of the weight, but not all of it: softmax is a "soft" version of picking
the maximum. Scale the scores up and it becomes harder: `[20, 10, 0]` gives about `[0.99995, 0.00005, 0]`.

**The shift trick.** Adding the same number `c` to every score does not change the softmax, because
`exp(xᵢ + c) = exp(xᵢ)·exp(c)` and the `exp(c)` cancels between top and bottom. Programs therefore
subtract the maximum score before exponentiating, so the largest exponent is `exp(0) = 1` and nothing
overflows. This is called the **stable softmax**, and every implementation in the project uses it
(Chapters 7 and 8).

## 2.6 Weighted averages

If weights `w₁ … wₙ` are non-negative and sum to 1, then `w₁·v₁ + … + wₙ·vₙ` is a **weighted average**
(also called a *convex combination*) of the vectors `vᵢ`. Attention's output for each token is
exactly this: a weighted average of "value" vectors, with softmax weights. A consequence used in
Chapter 11 is that **an attention output can never be larger than the largest value it averages**.

## 2.7 Probability and a few statistics

* A **probability distribution** over `n` outcomes is `n` non-negative numbers summing to 1, which
  is exactly what softmax produces.
* **Uniform guessing** among `n` options is right with probability `1/n`. For example, chance level for
  16 possible answers is 6.25 %. That number appears in Chapter 19.
* The **median** of repeated measurements is the middle value after sorting. It is robust to a few
  extreme values, which is why our timing reports use it (Chapter 13).
* The **inter-quartile range (IQR)** is the spread between the 25th and 75th percentiles, a robust
  measure of variability.
* **Cross-entropy loss** measures how much probability a model gave to the correct answer:
  `loss = −log(p_correct)`. If a model spreads probability evenly over `n` answers, `p_correct = 1/n` and
  `loss = log(n)`. So a training loss stuck at `log(4) ≈ 1.386` means "uniform over 4 options", a fact
  that was crucial in the MQAR investigation (Chapter 19).

## 2.8 Derivatives and gradients: how learning works

A **derivative** says how much a function's output changes when its input changes a little. If
`f(x) = x²`, its derivative is `2x`: at `x = 3`, nudging `x` by 0.001 changes `f` by about 0.006.

For a function of many inputs, the list of derivatives with respect to each input is the
**gradient**. A neural network is a big function with millions of adjustable numbers (**parameters**).
**Training** repeatedly does three things:

1. compute the loss (how wrong the network is);
2. compute the gradient of the loss with respect to every parameter. This step is called the
   **backward pass**, or *backpropagation*, while computing the output is the **forward pass**;
3. nudge every parameter a little in the direction that reduces the loss (an **optimizer**
   such as AdamW decides exactly how).

Two things to remember for this project:

* The backward pass usually needs values computed during the forward pass. Keeping those values in
  memory is what makes **training** use much more memory than **inference** (just computing outputs).
  This is central to Chapter 8.
* The **chain rule** combines derivatives of nested functions. PyTorch's **autograd** applies it
  automatically: it records every operation in the forward pass, then replays them backward. Writing
  a **custom backward** means doing this by hand, which `tiled_recompute` does.

## 2.9 Big-O notation: how cost grows

"Big-O" describes how a cost grows as the input gets larger, ignoring constant factors:

| Notation | Name | Doubling the input multiplies the cost by |
|---|---|---|
| `O(L)` | linear | 2 |
| `O(L log L)` | "n log n" | a bit more than 2 |
| `O(L²)` | quadratic | 4 |
| `O(L³)` | cubic | 8 |

Standard attention costs `O(L²·d)` time: the `L × L` table of dot products, each costing `O(d)`.

**Measuring the exponent.** If time `t` grows like `L^p`, then `log t = p·log L + constant`, so a plot of
`log t` against `log L` is a straight line with slope `p`. The project fits that slope to measured data
(`reporting/report.py::fit_exponent`). A fitted exponent near 2 means "quadratic in the measured range",
and near 1 means "linear". This is how theory and measurement are connected in Chapters 14 and 15.

**Big-O hides constants, and constants decide real speed.** An `O(L)` method that does 100 operations per
token loses to an `O(L²)` method doing 0.01 operations per pair until `L > 10,000`. Chapter 15 measures such
a "crossover" on this machine.

## Key points to remember

* Dot product = similarity. `Q × Kᵀ` = all pairwise similarities = an `L × L` table.
* Matrix multiply cost: `2·m·n·k` FLOPs.
* Softmax turns scores into positive weights summing to 1, and it is unchanged by subtracting the max.
* Attention output = weighted average of value vectors.
* Loss stuck at `log(n)` = uniform guessing among `n` options.
* Training needs the backward pass, which needs stored forward values, which needs memory.
* Fitted log-log slope ≈ growth exponent: 2 = quadratic, 1 = linear.
