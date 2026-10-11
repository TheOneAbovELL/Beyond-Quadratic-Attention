# Technical interview guide

Questions about attention algorithms, numerics, complexity and benchmarking. Each
has a model answer grounded in this project, with pointers to code or data.

## Attention and complexity

1. **Why is standard attention quadratic in L, and in what exactly — time, memory, or both?**
   Every query scores every key: `L²` logits, each a length-`d` dot product, so
   `O(L²d)` time. A naive implementation also *stores* those logits, so `O(L²)` memory.
   Tiling removes the memory term but not the time term (`complexity_analysis.md` §1–2).

2. **FlashAttention is "exact" and "fast". Does it reduce FLOPs?**
   No. It performs the same `4L²d` forward matmul FLOPs, and *more* in backward,
   because it recomputes `S`. Its win is IO: `Θ(N²d²/M)` vs `Θ(Nd + N²)` HBM accesses
   (Dao et al., 2022). That turns a bandwidth-bound computation into a compute-bound one.

3. **If tiled attention has lower memory complexity, why is your tiled implementation the slowest exact method?**
   It is a Python loop of unfused PyTorch ops. Per-op dispatch overhead × `(L/b)²`
   tiles, small inefficient matmuls, and every intermediate still written and read
   once. The fused ATen kernel implements the same algorithm in C++ with tiles held in
   cache. Asymptotic memory complexity ≠ memory traffic ≠ wall-clock.

4. **Does tiling alone make training memory sub-quadratic?**
   No. Autograd saves every tile's `P`, which is `O(L²)` in total. You need a custom
   backward that recomputes `P` from `(Q, K, lse)` (`tiled_attention_recompute`).

5. **Derive the backward of softmax attention without storing P.**
   `dS = P ∘ (dP − D)` with `D_i = Σ_k P_ik dP_ik = dO_i·O_i`. Recompute
   `P = exp(S − lse)` per tile (`attention_math.md` §4).

6. **When is linear attention actually linear, and what does it give up?**
   When the similarity factorizes, `sim(q,k) = φ(q)·φ(k)`. Then `Σ_j φ(k_j)v_jᵀ` can be
   shared across queries: `O(Ld²)`. It gives up softmax's exponential sharpness:
   `elu+1` cannot concentrate weight on one key among many. That shows up directly in
   the retrieval probes as L grows.

7. **A block-sparse layout has 5 % density. Is it 20× faster?**
   Not necessarily. Gathers, padding to the widest row, global rows, and the loss of a
   fused dense kernel all add cost. Check the measured crossover.

## Online softmax and numerics

8. **Prove online softmax is exact.**
   Invariant: `l_t = Σ_{seen} e^{s_j − m_t}`. Rescaling by `α = e^{m_t − m_new}`
   re-bases the old terms. The final `acc/l` has a common factor `e^{−m}` that cancels
   (`online_softmax.md` §3–4).

9. **Why subtract the max? What breaks without it?**
   fp32 `exp` overflows above ~88.7, giving `inf/inf = NaN`. The project's
   `unstable_softmax_demo` measures exactly this.

10. **What happens to a fully-masked row? What should happen?**
    Mathematically `0/0`. The naive formula yields NaN. Choose a policy (here: output 0,
    zero gradient, matching PyTorch CPU SDPA) and test it for every implementation.

11. **Two exact implementations differ by 1e-6 in fp32. Bug?**
    Not by itself. Reduction order and extra rescaling roundings explain `O(√n·u)`
    differences. Check that the difference scales like rounding (dtype, L, magnitude),
    not like an algorithmic error (it would not shrink in fp64). Here fp64 differences
    are ~1e-15.

12. **Why keep logits and statistics in fp32 when inputs are bf16?**
    A logit error `δ` becomes a relative probability error `≈ δ`. bf16 has 8 mantissa
    bits, so logits in the hundreds carry errors around 0.5. Measured here: naive
    attention computed in bf16 has 15 % normalized error at input scale 10, while
    fp32-accumulating kernels stay at 0.2 %.

13. **How do you choose tolerances?**
    From the dtype's unit roundoff and the length of the reductions involved, and for
    16-bit dtypes relative to a mixed-precision baseline. Never by loosening until the
    tests pass (see the investigation in `numerical_guide.md` §4).

## Systems and benchmarking

14. **Why report the median and IQR rather than the minimum?**
    The minimum estimates best-case latency under ideal conditions and hides
    variance. The median is robust to OS noise, and the IQR shows dispersion.

15. **Your two runs of the same code differ 2.5×. What do you check?**
    Background load (we found unrelated workloads occupying up to all 12 logical CPUs), thread
    settings, thermal/power state, memory pressure and paging, and first-call effects.
    Record the load, rerun when quiet, and never compare contended with clean runs.

16. **How do you measure memory on a CPU, and what does each number mean?**
    Profiler-replayed allocator events give framework tensor bytes during the call.
    The OS peak working set gives process-level resident memory (monotone; needs
    process isolation). Theory gives an estimate. Three different quantities; never
    label an estimate as a measurement.

17. **Why might bf16 be slower than fp32 on a CPU?**
    Without native bf16 arithmetic (AVX2 only here), kernels convert to fp32 and back,
    which adds work and traffic. GPUs with tensor cores behave the opposite way.

18. **What can a CPU benchmark of attention tell you about GPUs?**
    Algorithmic facts transfer (exactness, storage scaling, recompute trade). Crossover
    lengths, dtype speed, and fused-vs-unfused ratios do not.

## Experimental design

19. **How would you show that a sparse method's quality loss is due to the sparsity pattern rather than training noise?**
    Equality tests against the masked reference; attention-level probes with a known
    target position, independent of training; trained comparisons with identical seeds,
    data stream and evaluation sets; and swap-at-inference experiments that hold the
    weights fixed.

20. **What is the most common invalid conclusion from attention benchmarks?**
    "Fewer theoretical FLOPs ⇒ faster" or "linear complexity ⇒ better", drawn from
    measurements taken at lengths where constant factors dominate, on hardware unlike
    the deployment target, or under uncontrolled background load.
