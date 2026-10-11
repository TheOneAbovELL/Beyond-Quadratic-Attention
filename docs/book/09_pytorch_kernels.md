# Chapter 9 — Implementation C: PyTorch's built-in kernels and how we found out which one runs

> **What you will learn:** what PyTorch's `scaled_dot_product_attention` (SDPA) is; why "which kernel ran?"
> is a real question; how the profiler answered it; the two adapters the project registers; how masks are
> translated; and why the CPU "flash" kernel must not be called FlashAttention.

## 9.1 SDPA: one function, several implementations

PyTorch offers `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False,
scale=None)`. Behind that single function, PyTorch chooses among several **backends** (implementations):
FlashAttention-style kernels, "memory-efficient" kernels and cuDNN on CUDA GPUs, and a "math" fallback. You
can force a backend with the context manager `torch.nn.attention.sdpa_kernel(SDPBackend.X)`.

The PyTorch documentation (checked during the project) describes the optimized kernels in terms of the CUDA
backend and says that other backends use the plain PyTorch implementation. On a CPU-only machine it was
therefore unclear what would actually run.

## 9.2 Asking the profiler

The PyTorch **profiler** records every operator executed inside a `with profile(...)` block. During the audit,
SDPA was run with each backend and mask type, and the operator names were collected:

| Forced backend | Operators observed |
|---|---|
| default (none forced) | `aten::_scaled_dot_product_flash_attention_for_cpu` (+ `_backward`) |
| `SDPBackend.FLASH_ATTENTION` | `aten::_scaled_dot_product_flash_attention_for_cpu` (+ `_backward`) |
| `SDPBackend.MATH` | `aten::_scaled_dot_product_attention_math` (backward = ordinary ops) |

This was true with `is_causal=True`, with a boolean mask and with an additive float mask. So **this CPU build of
PyTorch has a dedicated fused "flash" kernel for CPUs and uses it by default.** A research subagent then checked
PyTorch's source: the operator is declared in `native_functions.yaml`, returns `(output, logsumexp)`, and is
selected by the CPU backend selector. It came in through a pull request that initially supported forward fp32
only. The docs sentence about "CUDA only" is out of date for CPU (recorded in `docs/references.md`).

The same experiment confirmed the fully-masked-row behaviour: both CPU backends return zeros (Chapter 7).

## 9.3 The two adapters

> **In the code:** `src/attention_lab/algorithms/sdpa.py`.

```python
_BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH}

def sdpa_attention(q, k, v, *, mask=None, key_padding_mask=None, causal=False, scale=None, backend="flash"):
    ...
    attn_mask, is_causal = _sdpa_mask(spec)
    with sdpa_kernel(_BACKENDS[backend]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=sc)
```

Two registry entries use it: `sdpa_flash_cpu` (forced flash) and `sdpa_math` (forced math). Forcing the backend
means every benchmark row states which kernel ran, instead of relying on an automatic choice that might change
between PyTorch versions.

**Mask translation (`_sdpa_mask`).** SDPA does not accept `attn_mask` and `is_causal` together, and it has no
separate padding mask. The adapter therefore:

* passes `is_causal=True` with no mask tensor when causality is the only mask (the cheapest case);
* otherwise builds the combined mask at its *broadcast* shape: a padding mask alone stays `[B, 1, 1, L]`;
  anything combined with causality becomes `[…, L, L]` (an `O(L²)` boolean, a documented cost of the adapter,
  not of the kernel);
* merges boolean and additive masks into one additive mask (`additive.masked_fill(~allowed, −∞)`).

## 9.4 What the flash CPU kernel does, and what it is not

The ATen CPU kernel processes blocks of queries against blocks of keys inside one C++ function. Scaling, masking,
the running max, `exp`, the running sums and the `P·V` accumulation all happen on small blocks held in cache.
It uses vectorized `exp`, parallelizes over (batch, head, query block) with OpenMP, and stores the logsumexp for
the backward pass. That is the same algorithm as Chapter 8, but **fused**: no intermediate ever goes to RAM, and
there is no per-operation Python dispatch.

It is **not** the official FlashAttention library by Dao et al., which is a CUDA implementation for NVIDIA GPUs.
The project never calls it FlashAttention. Its registry entry reads "PyTorch's fused tiled C++ CPU kernel; NOT
the official FlashAttention CUDA library".

`sdpa_math` is PyTorch's own materializing implementation, essentially an optimized version of our naive
reference.

## 9.5 How they compare (preview of Chapter 14)

Both adapters agree with the reference to about 1e-15 in fp64. On one thread at L = 4096 (forward), the fused
kernel took **516 ms**, against about 1,500 ms for naive, the math backend, and the Python tiled code. All four
compute the same mathematics, so the factor of about 3 comes entirely from *how* the computation is executed.

## Key points to remember

* SDPA has several backends. On this CPU build, the default is a fused flash-style CPU kernel.
* The profiler (operator names) is how we *verified* which kernel ran, instead of assuming.
* `sdpa_flash_cpu` and `sdpa_math` force their backends, so results always name the kernel.
* The adapter merges masks into SDPA's format, and combined causal+padding masks cost an `L × L` boolean.
* The CPU flash kernel is PyTorch's own and must not be called "FlashAttention".
