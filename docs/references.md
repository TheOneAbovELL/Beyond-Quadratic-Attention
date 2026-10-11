# References

Titles, authors and arXiv IDs were checked against the arXiv abstract pages during
this project. The claims attributed below were checked against the paper text.
Venues marked *(unverified)* come from memory and were not confirmed on arXiv.

| Short name | Citation | What this project uses it for |
|---|---|---|
| Vaswani 2017 | Vaswani et al., "Attention Is All You Need", arXiv:1706.03762 (NeurIPS 2017, *unverified*) | scaled dot-product attention; `1/√d_k` motivated by the variance of `q·k` growing with `d_k` (footnote 4), and by softmax saturation giving small gradients |
| Milakov 2018 | Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867 | single-pass computation of the softmax *normalizer* with a running max and a rescaled running sum |
| Rabe 2021 | Rabe & Staats, "Self-attention Does Not Need O(n²) Memory", arXiv:2112.05682 | chunked attention: O(log n) memory in theory, O(√n) in their practical implementation; checkpointing for the backward pass |
| FlashAttention | Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135 (NeurIPS 2022, *unverified*) | tiling + online softmax in one fused GPU kernel; stores (m, ℓ) and recomputes S, P in backward; Thm. 2: Θ(N²d²/M) HBM accesses vs Θ(Nd + N²) for standard attention |
| FlashAttention-2 | Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv:2307.08691 (ICLR 2024, *unverified*) | un-normalized accumulator rescaled once at the end; store only logsumexp; parallelize over sequence length |
| Katharopoulos 2020 | Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", arXiv:2006.16236, ICML 2020 | linear attention with φ(x) = elu(x) + 1, O(N) causal recurrence |
| Child 2019 | Child et al., "Generating Long Sequences with Sparse Transformers", arXiv:1904.10509 | strided/fixed factorized sparse patterns, O(n√n) |
| Longformer | Beltagy et al., "Longformer: The Long-Document Transformer", arXiv:2004.05150 | sliding-window + global attention |
| BigBird | Zaheer et al., "Big Bird: Transformers for Longer Sequences", arXiv:2007.14062, NeurIPS 2020 | block-sparse window + global + random pattern (the inspiration for `block_sparse`) |
| Performer | Choromanski et al., "Rethinking Attention with Performers", arXiv:2009.14794, ICLR 2021 | FAVOR+ random features approximating the softmax kernel (mentioned as the alternative to elu+1, not implemented) |
| RoFormer | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864 | RoPE, used in the tiny Transformer |
| Zoology | Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models", arXiv:2312.04927 | the MQAR task; recall gap between attention and *gated-convolution* models |
| Based | Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff", arXiv:2402.18668 | argues fixed-size recurrent state limits recall in linear-attention models |
| Induction heads | Olsson et al., "In-context Learning and Induction Heads", arXiv:2209.11895 | [A][B]…[A]→[B] pattern completion. That paper does not use the term "associative recall"; the connection is our framing |
| LRA | Tay et al., "Long Range Arena: A Benchmark for Efficient Transformers", arXiv:2011.04006, ICLR 2021 | context for efficient-attention benchmarking |
| Survey | Tay et al., "Efficient Transformers: A Survey", arXiv:2009.06732 | taxonomy context |
| PyTorch SDPA | `torch.nn.functional.scaled_dot_product_attention` docs, https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | bool `attn_mask` True = takes part; `is_causal` is top-left aligned; `attn_mask` and `is_causal` cannot both be set |

## Note on the PyTorch CPU flash kernel

The SDPA documentation lists FlashAttention-2, memory-efficient attention and a C++
math implementation, and says that optimized kernels are used with the CUDA backend.
That statement is out of date for CPU. In this environment (torch 2.10.0+cpu), the
profiler shows `aten::_scaled_dot_product_flash_attention_for_cpu` and its
`_backward` running for `SDPBackend.FLASH_ATTENTION`. The op is declared in PyTorch's
`native_functions.yaml`, returns (output, logsumexp), and is selected by the CPU
backend selector. It is PyTorch's own fused CPU kernel and is **not** the official
FlashAttention CUDA library.
