# Chapter 25 — Self-test: know this project by heart

Answer each question aloud, then check the answer and the chapter. Repeat until you get them all right without
looking.

## Part A — The project

1. **What is the project's central question?** How attention algorithms trade off complexity, memory traffic,
   numerical accuracy and practical speed as sequence length grows. *(Ch. 1)*
2. **Name the eight implementations and their families.** naive, sdpa_math, sdpa_flash_cpu, tiled, tiled_recompute
   (exact); local_window, block_sparse (restricted softmax); linear_elu (different function). *(Ch. 1, 6)*
3. **What machine was used, and what was special about it?** Ryzen 5 7530U, 6C/12T, AVX2, 16 MB L3, 16 GB RAM, no GPU.
   Shared with other projects' workloads (5–12 logical CPUs busy) and often under 1 GB of free RAM. *(Ch. 1, 6, 13)*
4. **Why is there a `lab.py` launcher?** So the lab runs without installing into the system Python; it adds `src/` to
   the path. *(Ch. 6)*

## Part B — The mathematics

5. **Write the attention formula.** softmax(QKᵀ/√d + M)V. *(Ch. 4)*
6. **Why divide by √d?** The dot product of random unit-variance vectors has variance d; scaling restores unit variance
   and avoids saturated softmax with vanishing gradients. *(Ch. 4)*
7. **Why is attention quadratic, and in what?** QKᵀ is L × L. Time is always O(L²d) for exact attention; naive memory
   is O(L²), but tiling makes memory O(L). *(Ch. 5)*
8. **How many matmul FLOPs does exact attention need per head?** About 4·L²·d. *(Ch. 5)*
9. **What does a training loss of log 4 mean?** Uniform guessing among 4 options. *(Ch. 2, 19)*

## Part C — Algorithms

10. **State the online-softmax update.** m_new = max(m, max S); α = exp(m − m_new); P = exp(S − m_new);
    l = α·l + ΣP; acc = α·acc + P·V; finally out = acc/l, lse = m + log l. *(Ch. 8)*
11. **Online softmax on scores [1,3,2,5] and values [10,20,30,40], tiles of 2?** 36.880566, the same as the direct
    softmax. *(Ch. 8)*
12. **Why doesn't the tiled method with autograd save training memory?** Autograd keeps every tile's P alive, which
    adds up to O(L²). *(Ch. 8)*
13. **What does the recompute backward store, and how does it get P back?** Q, K, V, O and lse; P = exp(S − lse),
    with S recomputed per tile. *(Ch. 8)*
14. **What is D in the backward pass, and why does it matter?** D = rowsum(dO ∘ O) = Σ P·dP. It replaces a full row of
    P in the softmax Jacobian. *(Ch. 8)*
15. **What does the policy say about fully-masked rows, and where did it come from?** Zero output and zero gradient,
    matching what PyTorch's CPU SDPA was observed to do. *(Ch. 7)*
16. **Is `sdpa_flash_cpu` FlashAttention?** No: it is PyTorch's own fused CPU kernel
    `_scaled_dot_product_flash_attention_for_cpu`, identified with the profiler. *(Ch. 9)*
17. **How does local_window avoid the L² table?** Query chunks plus overlapping key windows via `unfold`, then a band
    mask: O(L·(C + 2w)) scores. *(Ch. 10)*
18. **Why does block_sparse compute global rows separately?** A row that sees every block would force padding every
    row to full width. *(Ch. 10)*
19. **Why is linear attention O(L) non-causal?** Associativity: φ(Q)(φ(K)ᵀV) shares one d × d matrix across queries.
    *(Ch. 10)*
20. **Is linear attention an approximation of softmax?** No. elu + 1 defines a different kernel. *(Ch. 10)*

## Part D — Measurement

21. **What is the reference in the correctness lab?** Naive attention in fp64 on the *same rounded* inputs. *(Ch. 11)*
22. **What are the tolerances?** fp64 1e-12; fp32 2e-5 or 4·u·max|S|; bf16 2× a mixed-precision baseline + slack.
    *(Ch. 11)*
23. **Tell the bf16-baseline story.** Naive-in-bf16 had up to 15 % error at magnitude 10 (38 % at 30), so "2× it"
    accepted anything. The baseline was switched to mixed precision; naive-bf16 now fails 14/53, reported as a
    finding. *(Ch. 11, 22)*
24. **Tell the fp32 "failure" story.** Flash at magnitude 10 had 2.04e-5 > 2e-5. The naive formula in fp32 had the
    same error; errors scale with u·max|S| for every method. A logit term was added to the tolerance. *(Ch. 11)*
25. **Name the three memory instruments.** Tensor peak (profiler allocator replay, validated at exactly 15.0 MiB),
    process peak Δ (OS working set, isolated workers), theory (labeled). *(Ch. 13)*
26. **How was background load handled?** A load monitor recorded other processes' CPU use; CPU time was recorded;
    definitive sweeps ran on 1 thread; contaminated timings were discarded. *(Ch. 13)*
27. **Why median and not minimum?** The median is robust and typical; the minimum is the luckiest run. *(Ch. 13)*

## Part E — Results

28. **Naive vs flash forward memory at L = 2048?** 194.1 MiB vs 4.3 MiB. *(Ch. 14)*
29. **Training-memory exponents of tiled-autograd vs tiled-recompute?** 1.72 vs 0.75. *(Ch. 14)*
30. **Time exponents of the exact methods?** All about 1.9–2.1 (autograd-tiled fwd+bwd 2.52). *(Ch. 14)*
31. **Flash vs naive forward on one thread at L = 4096?** 516 vs 1,540 ms, about 3×. *(Ch. 14)*
32. **Recompute vs autograd-tiled fwd+bwd at 2048?** 980 vs 1,847 ms: recompute is faster despite 25 % more FLOPs.
    *(Ch. 14)*
33. **Where do approximate methods overtake the fused kernel (non-causal forward)?** linear_elu at every L;
    local_window from about 2,048; block_sparse clearly from 4,096. *(Ch. 15)*
34. **Do approximate methods use less memory than the fused exact kernel?** No: 16–27× *more* at L = 4096. They save
    memory only relative to materializing methods. *(Ch. 15)*
35. **What happens to the Python tiled code with more threads?** It gets slower: 4.8× slower at 6 threads than at 1.
    *(Ch. 15)*
36. **What did the distance probe show for local_window w = 64?** 100 % up to distance 64, 0 % from 65. *(Ch. 16)*
37. **Why did block_sparse succeed at distance 1023?** The target was in the global block 0. *(Ch. 16)*
38. **Which methods resisted decoys, and why?** local_window and block_sparse: the decoys were outside their pattern.
    *(Ch. 16)*
39. **What did the β-sweep show about linear attention?** It improves with scale (L = 16: 6 % → 73 %) but stays far
    below softmax, because the sharpness is polynomial, not exponential. *(Ch. 16)*
40. **Task A: what happened when a window was swapped into the exact model at L = 256?** 86.9 % → 100 %. *(Ch. 18)*
41. **How did local_window score 49 % on induction with offsets beyond its window?** Two layers relay 16 tokens:
    99 % within 8–16 back, 6 % beyond. *(Ch. 18)*
42. **Why is block_sparse at L = 32 not sparsity evidence?** Its layout covered 528/528 causal pairs. *(Ch. 18)*
43. **Is the distractor-gap collapse caused by ambiguity?** No: ambiguous targets went 30.1 % → 36.7 %, while accuracy
    went 100 % → 18 %. *(Ch. 18)*

## Part F — The MQAR story

44. **What caused the v1 plateau?** Values drawn with replacement; "predict the most frequent value" scored 25.7 %.
    *(Ch. 19)*
45. **What did D1 (overfit) show?** 100 % train, 20 % held-out: memorization, no circuit. *(Ch. 19)*
46. **What did D2 show?** Plain induction was solved positionally: 100 % at a fixed offset, 18 % at random offsets.
    *(Ch. 19)*
47. **What did D3 show?** Content-based induction is learnable: 100 % by step 1000. *(Ch. 19)*
48. **Why was the MQAR investigation stopped?** It had established that the attention code was not at fault, which is
    the project's question; the remaining causes are a separate research topic. *(Ch. 19)*

## Part G — Operating

49. **How do you reproduce everything?** `python lab.py reproduce` (or `--dry-run` to list the steps). *(Ch. 20)*
50. **How do you add a new attention method?** Write a function with the shared signature, add a registry entry, add
    theory formulas, add tests. *(Ch. 20)*
