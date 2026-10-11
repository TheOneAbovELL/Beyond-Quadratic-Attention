# Chapter 24 — Glossary

**Accumulator (dtype).** The number format used for running sums inside an algorithm. Using fp32 accumulators with
bf16 inputs was the key to accuracy (Chapter 11).

**Activation.** An intermediate value computed in a network's forward pass; training keeps many of them for backward.

**AdamW.** The optimizer used for training: adaptive per-parameter step sizes plus weight decay.

**Additive mask.** A table of numbers added to the attention scores; `−∞` removes a pair.

**Approximate attention.** Any method that does not compute exact softmax attention: restricted (local, block
sparse) or a different function (linear).

**Autograd.** PyTorch's automatic differentiation: it records operations in the forward pass and computes gradients
backward.

**AVX2.** The vector instruction set of this CPU (8 fp32 numbers per instruction). No native bf16 arithmetic.

**Backward pass.** Computing gradients of the loss with respect to all parameters and inputs.

**Batch (B).** Number of independent sequences processed together.

**bf16 (bfloat16).** 16-bit floating point with fp32's range but only 8 significant bits (u ≈ 3.9 × 10⁻³).

**Bitwise equality.** Identical bit patterns, which is stricter than numerical equivalence.

**Block-sparse attention.** Exact softmax restricted to a layout of query-block × key-block pairs.

**Cache (L1/L2/L3).** Small, fast memory between the CPU and RAM. L3 here is 16 MB.

**Causal mask.** Query `i` may attend only keys `j ≤ i` (top-left aligned).

**Chunk.** A group of consecutive positions processed together (local window, linear attention).

**Contended.** A timing record during which other processes used more than one logical CPU.

**Crossover length.** The length beyond which one method becomes faster than another.

**Cross-entropy loss.** `−log(probability of the correct answer)`. Uniform guessing among n options gives log n.

**Deviation.** The difference between an approximate method's output and full softmax attention. It is not called error.

**Dispatch overhead.** The fixed cost (microseconds) of issuing one PyTorch operation from Python.

**dtype.** Data type, such as fp64, fp32, bf16.

**Embedding.** A learned table mapping token IDs to vectors.

**Exact attention.** Computes softmax(QKᵀ/√d + M)V up to floating-point rounding.

**Exponent (fitted).** The slope of log(cost) vs log(L): about 2 means quadratic, about 1 means linear.

**FlashAttention.** The fused, IO-aware GPU attention algorithm and library by Dao et al. (2022). Not used here; its
algorithm is reproduced in Python, and PyTorch's CPU flash kernel applies the same ideas.

**FLOP.** Floating-point operation; one multiply-add counts as 2.

**fp32 / fp64.** 32-bit and 64-bit floating point (u ≈ 6 × 10⁻⁸ and 1.1 × 10⁻¹⁶).

**Fully-masked row.** A query with no allowed keys; the project outputs 0 by policy.

**Fused kernel.** One compiled function doing several steps on data held in cache, without writing intermediates
to RAM.

**Gradient.** The vector of derivatives of the loss with respect to parameters.

**gradcheck.** PyTorch's finite-difference check of a backward pass.

**Head (H).** One of several independent attention computations in parallel; head dimension `d`.

**Induction (task / head).** Find the earlier occurrence of the current token and copy what followed it.
Variable-offset induction requires content matching.

**Inference mode.** Running a model without recording gradients.

**IQR.** Inter-quartile range: the spread between the 25th and 75th percentiles.

**Isolated worker.** A fresh Python process running one benchmark configuration.

**Key / Query / Value.** The three projections of each token: what it offers to be matched, what it looks for, and
what it hands over.

**Layout.** The block-level allowed/forbidden table of block-sparse attention.

**Linear attention.** Attention with weights φ(q)·φ(k) (here φ = elu + 1), computable in O(L·d²).

**Load monitor.** The project's background sampler of other processes' CPU use.

**Local-window attention.** Exact softmax restricted to keys within w positions of the query.

**Logit / score.** The scaled dot product q·k/√d before softmax.

**Log-sum-exp (lse).** log Σ exp(sⱼ) per row. Stored by memory-efficient backward passes.

**MaskSpec.** The project class defining mask semantics and serving masks per tile.

**Materialize.** To actually create a tensor in memory (for example, the full L × L score matrix).

**Memory-bandwidth-bound.** Limited by data movement rather than arithmetic.

**Memory guard.** Refuses benchmark modes whose estimated memory exceeds the budget, and records the refusal.

**MQAR.** Multi-query associative recall: retrieve values bound to keys.

**Numerical equivalence.** Equal within a justified tolerance.

**Online softmax.** Computing a softmax-weighted sum incrementally by rescaling running sums when the max changes.

**OpenMP.** The threading library PyTorch uses on CPUs.

**Padding mask (key_padding_mask).** Marks real (True) vs padding (False) keys.

**Probe.** A hand-constructed input testing whether one attention operation can retrieve a known target.

**Profiler.** PyTorch's tool recording operators and memory events.

**Recompute (backward).** Recomputing score tiles in the backward pass instead of storing them.

**Registry.** The table of all attention methods with their metadata.

**RoPE.** Rotary position embedding; it makes attention depend on relative positions.

**SDPA.** PyTorch's `scaled_dot_product_attention`, with several backends.

**Shortcut baseline.** The best accuracy achievable without the intended skill (for example, without key–value binding).

**SMT.** Simultaneous multithreading: 2 hardware threads per core.

**Softmax.** exp(xᵢ)/Σ exp(xⱼ): positive weights summing to 1.

**Swap experiment.** Evaluating a trained model with a different attention backend substituted, with the weights unchanged.

**Tensor peak.** Peak bytes allocated through PyTorch's CPU allocator during one call (profiler replay).

**Tile.** A small block of queries × keys processed at once.

**Tolerance.** The maximum accepted difference, derived from the dtype and the operation.

**Unit roundoff (u).** The maximum relative error of one rounding in a format.

**Working set (peak).** The OS's high-water mark of a process's resident memory.
