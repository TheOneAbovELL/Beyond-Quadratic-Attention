# Chapter 1 — The whole project in one chapter

> **What you will learn:** what the project is for, the one question it asks, what was built,
> what was measured, the most important findings, and the shape of the rest of the book.

## 1.1 The problem in plain words

Modern AI language systems such as chatbots are built from a design called the **Transformer**. Its
key ingredient is an operation called **attention**: every word (more precisely, every *token*) in a
piece of text looks at every other token and decides how much to "pay attention" to it.

"Every token looks at every other token" is the problem. If a text has 1,000 tokens, there are
1,000 × 1,000 = one million token pairs. With 10,000 tokens there are 100 million pairs, a hundred times
more work for only ten times more text. Mathematicians call this growth **quadratic**: double the length
and the work quadruples. That is why long documents are expensive for AI systems, and why researchers have
invented many cleverer ways to compute attention.

This project is a **laboratory** for studying those ways. Not a single program that does one thing, but a
collection of tools to implement, verify, measure and compare different attention algorithms on an
ordinary laptop.

## 1.2 The one question

The project brief stated the central question like this:

> *How do attention algorithms trade off computational complexity, memory traffic, numerical accuracy,
> and practical execution speed as sequence length increases?*

Each phrase in that sentence turned into a part of the project:

| Phrase | What we built to study it |
|---|---|
| "computational complexity" | an analytical model that counts operations and memory for each algorithm (Chapter 12) |
| "memory traffic" | memory instruments that measure how many bytes each algorithm actually allocates (Chapter 13) |
| "numerical accuracy" | a correctness laboratory comparing every method with a high-precision reference (Chapter 11) |
| "practical execution speed" | a benchmarking system with careful timing and background-load monitoring (Chapter 13) |
| "as sequence length increases" | sweeps over lengths from 64 to 16,384 tokens (Chapters 14–15) |

A second question came from the fact that some algorithms do not compute exactly the same thing as
standard attention. They *approximate* or *restrict* it. For those, we also asked: *what do you lose?* That
led to the quality experiments (Chapters 16 and 18).

## 1.3 What was built

Everything lives in one folder and runs locally. Nothing is sent to the internet.

**Eight attention implementations** (Chapters 7–10):

| Name | Family | One-line description |
|---|---|---|
| `naive` | exact | the textbook formula, written plainly; the "ground truth" everything else is compared with |
| `sdpa_math` | exact | PyTorch's built-in attention, forced to use its plain "math" path |
| `sdpa_flash_cpu` | exact | PyTorch's built-in attention, forced to use its fused, tiled C++ kernel for CPUs |
| `tiled` | exact | our own implementation of the *online softmax* / tiling idea behind FlashAttention, written in Python |
| `tiled_recompute` | exact | the same, plus a custom backward pass that recomputes instead of storing; this is the trick that saves training memory |
| `local_window` | restricted | each token may only look at nearby tokens |
| `block_sparse` | restricted | tokens are grouped into blocks, and each block may look only at selected blocks |
| `linear_elu` | different function | "linear attention", which replaces the exponential in softmax with a simpler formula that can be reorganized to avoid the quadratic cost |

**Tools around them:**

* a **numerical-correctness laboratory**: 795 comparisons of every exact method against the reference, in
  three number formats, with an explicit and justified tolerance policy;
* a **complexity model**: formulas for operations and memory per method, checked against counters built
  into the algorithms;
* a **benchmarking system**: repeated timing with statistics, three different memory instruments, a memory
  guard that refuses to run configurations that would exhaust RAM, isolated worker processes, and a monitor
  that records how busy *other* programs kept the CPU;
* **controlled tasks**: hand-built "retrieval probes" and small Transformer models trained on synthetic tasks,
  to measure what the approximate methods lose;
* a **command-line interface** (`python lab.py ...`), a **local web interface** (Streamlit), automatic
  **reports** with tables and figures, and a **test suite** of more than 400 tests.

## 1.4 The machine

The brief described an AMD Ryzen 5 5300U. The actual machine, checked on day one, has an **AMD Ryzen 5
7530U**: 6 cores, 12 hardware threads, AVX2 vector instructions, a 16 MB L3 cache, 16 GB of RAM, Windows 11,
Python 3.13.3 and PyTorch 2.10 (CPU build). There is no NVIDIA graphics card, so **every result in this
book is a CPU result**. GPUs behave differently, and Chapter 13 explains how.

A fact that shaped the whole project: **the computer was shared.** Throughout the work, other programs
from other projects kept between 5 and all 12 logical CPUs busy, and free memory was often under 1 GB. The
project had to learn to measure honestly under those conditions (Chapters 13 and 22).

## 1.5 The most important findings

Each of these is explained in detail later, with its evidence.

1. **All exact methods compute the same function.** In 64-bit floating point they agree with the reference
   to about 10⁻¹⁵. In 32-bit, every method passed every forward and gradient check (Chapter 14).
2. **How you store intermediate numbers matters more than the algorithm.** Computing attention entirely in
   the 16-bit `bfloat16` format gave errors up to 38 %, while the kernels that keep intermediate values in
   32-bit stayed at 0.3 % (Chapter 11).
3. **Tiling removes the quadratic *memory*, but not the quadratic *time*.** At 2,048 tokens the reference
   allocates 194 MiB per call, while the tiled and fused methods allocate under 5 MiB. Yet every exact method's
   time still grows like length squared (fitted exponents 1.9–2.1) (Chapter 14).
4. **Tiling alone does not fix training memory.** Letting PyTorch's automatic differentiation run through
   the tiled loop brings the quadratic memory back (exponent 1.72). A custom "recompute" backward pass keeps it
   linear, and was also **faster**: 0.98 s against 1.85 s for the autograd version at 2,048 tokens (Chapter 14).
5. **The same algorithm can be 3× faster or slower depending on how it is executed.** PyTorch's fused C++
   kernel and our Python tiled code implement the same mathematics, but the fused one is about 3× faster
   (Chapter 14).
6. **Approximate methods are faster only beyond a crossover length**, about 2,048 tokens for local-window
   and block-sparse attention on this machine. Below it they are slower than the fused exact kernel
   (Chapter 15).
7. **Surprise: the approximate methods used *more* memory than the fused exact kernel**, 16–27× more at 4,096
   tokens, because they store their restricted score tables while the fused kernel stores almost nothing
   (Chapter 15).
8. **Restriction has predictable costs and sometimes benefits.** Local attention cannot retrieve anything
   beyond its window (accuracy drops from 100 % to 0 % exactly at the window edge). Yet on a task needing only
   local information, it generalized to 4× longer texts *better* than full attention (100 % vs 87 %)
   (Chapters 16 and 18).
9. **A model that "learned" a task may have learned a shortcut.** Twice in this project a model appeared to
   solve a retrieval task but had found a shortcut: once in the data generator, once through fixed positions.
   Finding those shortcuts is told in Chapter 19.

## 1.6 How the work proceeded

Roughly in this order (the true, messy history is in Chapter 22):

1. **Audit:** inspect the empty folder, the hardware and the installed software.
2. **Reference implementation** and its tests.
3. **Tiled exact attention** (online softmax), then the recompute backward.
4. **Numerical validation** across shapes, masks, magnitudes and number formats.
5. **Approximate methods.**
6. **Benchmarking** system and **complexity** model.
7. **Experiments:** correctness, memory scaling, timing (repeated after the background-load problem was
   discovered), approximate-method scaling, tile sizes, threads, shapes and dtypes.
8. **Quality:** probes, then trained models, then the long MQAR investigation.
9. **Interface, reports and documentation**, followed by a final audit.

## Key points to remember

* The project is a *laboratory* for comparing attention algorithms on speed, memory, accuracy and quality.
* Eight implementations: five exact, two restricted, one different-function.
* Everything was measured on a shared, CPU-only laptop, and the project measures and reports that load.
* The headline: memory efficiency, speed and accuracy are *different* things. An algorithm can win one and lose
  another, and the winner depends on length, threads and hardware.
