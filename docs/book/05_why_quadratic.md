# Chapter 5 — Why attention is "quadratic", and what that does and does not mean

> **What you will learn:** exactly which parts of attention grow with L², by counting; how big the score
> tables get; the difference between time, auxiliary memory, training memory and memory traffic; what each
> algorithm family changes; and why "quadratic" is only the start of the story.

## 5.1 Counting the work

For one head with `L` tokens and head dimension `d`:

| Step | What it computes | Size of result | Arithmetic |
|---|---|---|---|
| `S = Q Kᵀ` | `L × L` dot products of length `d` | `L²` numbers | `2·L²·d` FLOPs |
| scale, mask, max, exp, sum, divide | elementwise on `S` | `L²` numbers each | about `4·L²` operations |
| `O = P V` | `L` weighted averages over `L` values of length `d` | `L·d` numbers | `2·L²·d` FLOPs |

Total ≈ **`4·L²·d` multiply-add FLOPs per head**, plus `O(L²)` cheaper elementwise operations. With a batch
of `B` texts and `H` heads, multiply by `B·H`. In big-O: **time `O(B·H·L²·d)`**.

The inputs and outputs (`Q`, `K`, `V`, `O`) hold only `4·B·H·L·d` numbers, which is *linear* in `L`. The
quadratic part is the **intermediate** score and probability tables.

## 5.2 How big the tables get

The project's complexity model (`python lab.py complexity`, labeled THEORETICAL) gives, for `B = 1`,
`H = 4`, `d = 64`, fp32:

| L | logits evaluated (all exact methods) | matmul GFLOP per forward | naive: forward allocations | naive: saved for backward |
|---|---|---|---|---|
| 1,024 | 4.2 million | 1.07 | 49 MiB | 48 MiB |
| 4,096 | 67 million | 17.2 | 772 MiB | 768 MiB |
| 16,384 | 1.07 billion | 275 | 12,304 MiB (12 GiB) | 12,288 MiB |

Quadrupling `L` multiplies all of these by 16. At 16,384 tokens, *one forward call of one attention layer*
written naively would need 12 GiB of temporary memory, which is most of this laptop's RAM. That is the
practical meaning of "quadratic".

Compare the inputs: `q, k, v, out` at `L = 16,384` are `4 × 4 × 16384 × 64 × 4 bytes = 64 MiB`. **The
temporary tables are 190 times larger than the actual inputs and outputs.**

## 5.3 Four different kinds of "memory cost"

Confusing these is the most common mistake in discussions of efficient attention. The project keeps them
separate:

1. **Auxiliary memory of one forward call:** temporary tensors that exist only during the call. Naive:
   `O(L²)`. Tiled and fused: `O(tile² + L·d)`.
2. **Memory saved for the backward pass:** what autograd keeps alive between forward and backward when
   training. Naive: the `L²` probability tables. Autograd-tiled: *also* `L²` in total, because it keeps
   every tile. Recompute: only `O(L)` statistics.
3. **Total training memory of a model:** parameters, gradients, optimizer state (AdamW keeps 2 extra
   numbers per parameter), activations of every layer, plus the attention terms above.
4. **Memory traffic:** how many bytes travel between RAM and the caches. This is not storage at all; it is
   *movement*. A method can allocate little and still move a lot, or the other way round. Traffic is what
   FlashAttention's analysis is about (Chapter 8) and what largely decides speed (Chapter 3).

Chapter 13 explains which of these the project measures (1 and 2 directly, and process-level memory),
which it estimates (all, in theory), and which it cannot measure directly (4, which would need hardware
counters).

## 5.4 What each family of algorithms changes

| Family | Time | Auxiliary memory | Computes exact attention? |
|---|---|---|---|
| Naive / materializing | `O(L²d)` | `O(L²)` | yes |
| Tiled / fused (FlashAttention idea) | `O(L²d)`, **unchanged** | `O(tile² + Ld)` | yes |
| Local window (width `w`) | `O(L·w·d)` | `O(L·w)` | no: only nearby pairs |
| Block sparse (`k` blocks per row) | `O(L·k·b·d)` | `O(L·k·b)` | no: only selected blocks |
| Linear attention | `O(L·d²)` | `O(d²)` (non-causal) | no: a different function |

Two things stand out:

* **Tiling attacks memory, not arithmetic.** Every exact algorithm must look at every allowed pair, because
  every output depends on every pair. Tiling only decides *where* the scores live: in small, cache-resident
  tiles instead of a giant table. Chapter 14 confirms this experimentally: every exact method's time grows
  like `L^1.9–2.1`, while the tiled methods' memory grows like `L^0.8`.
* **Only methods that change the function can change the time complexity.** Local, block-sparse and linear
  attention are faster asymptotically *because* they compute something different. Whether that difference
  matters is a quality question (Chapters 16 and 18).

## 5.5 Why "quadratic" is only the start of the story

Big-O hides constant factors, and real performance depends on:

* **Memory traffic** (Chapter 3): the number of passes over large tables, not just their count.
* **Dispatch overhead**: Python-level loops pay microseconds per operation.
* **Parallel efficiency**: how well the work splits across threads.
* **Hardware**: cache sizes, vector width, supported number formats.

These factors explain every "surprising" result in this book:

* the C++ fused kernel is 3× faster than the Python tiled code, despite the same complexity (Chapter 14);
* local-window attention, which is `O(L)`, is *slower* than exact attention below about 2,048 tokens
  (Chapter 15);
* adding threads makes the Python tiled code *slower* (Chapter 15).

## Key points to remember

* Exact attention: `4·L²·d` FLOPs per head; quadratic time **always**.
* Naive auxiliary memory: about `3·L²` numbers. At L = 16,384 and 4 heads that is 12 GiB.
* Separate auxiliary memory, saved-for-backward memory, total training memory and memory traffic.
* Tiling reduces memory, not arithmetic. Only changing the function reduces arithmetic.
* Measured speed depends on constants, traffic, dispatch, threads and hardware, not only on big-O.
