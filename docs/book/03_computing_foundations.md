# Chapter 3 — How computers store and compute numbers

> **What you will learn:** bytes and memory sizes; the memory hierarchy (registers, caches, RAM) and why
> moving data often costs more than computing with it; floating-point numbers and rounding; the three
> number formats used in the project (fp64, fp32, bf16); CPU cores, threads and parallelism; and what
> Python, PyTorch, tensors and autograd are.

## 3.1 Bytes and sizes

A **byte** is 8 bits. Memory sizes in this book use binary units: 1 KiB = 1,024 bytes, 1 MiB =
1,048,576 bytes (about one million), 1 GiB ≈ one billion bytes. "MB" and "GB" are used loosely for the same
ideas.

A 32-bit floating-point number (fp32) occupies **4 bytes**. So an attention score table for one head with
`L = 1024` tokens holds `1024 × 1024 = 1,048,576` numbers, which is **4 MiB**. With 4 heads that is 16 MiB.
At `L = 4096` it is 64 MiB per head and 256 MiB for 4 heads. Keep these sizes in mind; they explain most of
Chapter 14.

## 3.2 The memory hierarchy: why moving data is expensive

A CPU does arithmetic on numbers held in tiny storage slots called **registers**. Data must be brought
there from main memory (**RAM**), which is large but slow, through a series of **caches**, which are
progressively smaller and faster:

| Level | Size on this machine | Rough speed |
|---|---|---|
| Registers | a few KB per core | instant |
| L1 cache | 32 KB data per core (384 KB total) | ~1 ns |
| L2 cache | 512 KB per core (3 MB total) | a few ns |
| L3 cache | **16 MB shared** | ~10 ns |
| RAM | 16 GB | ~100 ns and limited bandwidth |

(The sizes were measured on this machine with Windows' `Win32_CacheMemory`; the speeds are typical
orders of magnitude, not measurements.)

The crucial consequence: **an operation that does little arithmetic per byte is limited by how fast data
can be moved, not by how fast the CPU computes.** This is called being **memory-bandwidth-bound**. Adding
up a large table, or applying `exp` to every entry, is memory-bound. Multiplying two large matrices is
**compute-bound**, because each number fetched is reused many times.

Naive attention creates `L × L` tables and makes several passes over them (scale, mask, subtract max, exp,
sum, divide). Once a table exceeds the 16 MB L3 cache (at `L = 1024` with 4 heads), every pass goes to RAM.
Much of the speed of an optimized attention kernel comes from **never writing that table to RAM at all**
(Chapter 9).

## 3.3 Floating-point numbers

Computers store real numbers in **floating-point** format, a binary version of scientific notation:
`value = sign × mantissa × 2^exponent`. The **mantissa** (significant digits) has a fixed number of bits,
so most real numbers can only be stored **approximately**: they are **rounded** to the nearest
representable value.

The size of the rounding is described by the **unit roundoff `u`**: the relative error of a single
rounding is at most about `u`.

| Format | Bits | Mantissa bits | Unit roundoff u | Largest value | Used here for |
|---|---|---|---|---|---|
| fp64 ("double") | 64 | 52 | 1.1 × 10⁻¹⁶ | ~10³⁰⁸ | the reference ("ground truth") |
| fp32 ("float") | 32 | 23 | 6.0 × 10⁻⁸ | ~3.4 × 10³⁸ | default for all benchmarks |
| bf16 ("bfloat16") | 16 | 7 | 3.9 × 10⁻³ | ~3.4 × 10³⁸ | low-precision experiments |
| fp16 ("half") | 16 | 10 | 4.9 × 10⁻⁴ | 65,504 | tolerance defined, not benchmarked |

Three consequences matter in this project:

1. **Overflow.** `exp(x)` exceeds fp32's maximum when `x > 88.7`, and becomes `inf` (infinity).
   Then `inf / inf = NaN` ("not a number"). The stable softmax (Chapter 2) prevents this.
2. **Absolute precision shrinks as numbers grow.** bf16 has only 8 significant bits, so near 256 the
   gap between neighbouring representable numbers is about 1. A logit of 300.4 might be stored as 300.
   For softmax that is a disaster: an error of 0.5 in a logit multiplies its weight by `e^0.5 ≈ 1.65`.
   This is the cause of the biggest numerical finding in the project (Chapter 11).
3. **Addition is not associative.** `(a + b) + c` can differ from `a + (b + c)` in the last bits. Two
   correct programs that sum in different orders therefore give slightly different answers. That is why
   we compare with **tolerances**, not exact equality (Chapter 11).

Special values: `inf` and `-inf` (overflow, or deliberately used for masking: `exp(-inf) = 0`), and `NaN`,
which poisons every computation it touches.

## 3.4 CPU cores, threads and parallelism

A **core** is an independent processor. This machine has **6 physical cores**. Each can run **2
hardware threads** (*simultaneous multithreading*, SMT), giving **12 logical CPUs**, but two threads on
one core share its arithmetic units, so they rarely give 2× the speed.

PyTorch splits large operations across **intra-op threads** using a library called **OpenMP**: a matrix
multiply might be divided into 6 pieces computed at the same time. Coordination costs time (starting the
pieces, waiting for the slowest one, a *barrier*). For large operations the overhead is negligible. For
tiny operations it can exceed the useful work. If another program is using a core, the piece running there
is delayed and **every other thread waits at the barrier**. Chapter 15 shows this happening: the
Python tiled code became 4.8× *slower* with 6 threads.

`torch.set_num_threads(n)` sets the thread count in our code. The environment variables `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` set it before PyTorch starts.

**Vector instructions (SIMD).** A single CPU instruction can operate on several numbers at once. This CPU
supports **AVX2** (256-bit registers: 8 fp32 numbers per instruction) but not AVX-512, and it has **no native
bf16 arithmetic**. That is why bf16 is not faster than fp32 here (Chapter 15).

## 3.5 Python, C++ and the cost of "dispatch"

**Python** is the language this project is written in. It is easy to read but slow at doing many small
things, because every line is interpreted. **C++** is compiled to machine code and is fast.

**PyTorch** is a Python library whose heavy work is done in C++ (and specialized libraries: **MKL** and
**oneDNN** for matrix math on Intel/AMD CPUs). When Python calls a PyTorch operation such as `a @ b`,
PyTorch must:

1. parse the arguments and check types, shapes and devices;
2. choose the right C++ implementation (the **dispatcher**);
3. allocate memory for the result;
4. record the operation for autograd, if gradients are needed;
5. run the C++ code.

Steps 1–4 cost a few **microseconds**, regardless of how big the operation is. For one huge matrix multiply
that is negligible. For 10,000 tiny ones it is 10–50 milliseconds of pure overhead. This **dispatch
overhead** explains why our Python tiled attention, which issues roughly 10 small operations per tile, is
slower than the C++ fused kernel that implements the same algorithm (Chapters 8, 14 and 15).

## 3.6 Tensors, autograd and memory in PyTorch

A PyTorch **tensor** is an n-dimensional array with a data type (`torch.float32`, …) and a device (here
always the CPU). Some operations return **views**, which reuse the same memory with a different shape
(for example `transpose`, `expand`, `unfold`). Others allocate new memory. The project relies on views to
avoid building large masks (Chapter 7).

**Autograd**: if a tensor has `requires_grad=True`, PyTorch records every operation applied to it in a
graph and keeps the intermediate tensors the backward pass will need. `loss.backward()` walks the graph
backward and fills each parameter's `.grad`. The kept intermediates are why training memory is larger
than inference memory, and why "letting autograd differentiate through a tiled loop" keeps all the tiles
alive (Chapter 8).

`torch.inference_mode()` and `torch.no_grad()` switch recording off. Our forward-only benchmarks use
inference mode, so they measure the computation without autograd bookkeeping.

**The CPU allocator.** PyTorch requests memory for each new tensor from its CPU allocator. The
profiler can record every allocation and free, with sizes and timestamps. The project replays those
events to measure the peak memory of a call (Chapter 13).

## Key points to remember

* One fp32 number = 4 bytes. One `L × L` fp32 table at `L = 1024` = 4 MiB; with 4 heads = 16 MiB = the whole L3 cache.
* Moving data is often the real cost, so avoiding large intermediate tables is the main lever for speed.
* Floating point rounds. `u` is fp64 1e-16, fp32 6e-8, bf16 4e-3. `exp` overflows fp32 above 88.7.
* bf16's coarse spacing at large magnitudes ruins softmax logits unless they are kept in fp32.
* Python dispatch costs microseconds per operation, so many tiny ops are slow.
* More threads help large operations and hurt tiny ones, especially on a busy machine.
* Autograd keeps forward intermediates alive for backward: that is training memory.
