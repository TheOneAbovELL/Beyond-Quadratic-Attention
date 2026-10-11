"""Theoretical work and memory model for each implementation.

Everything here is an *analytical estimate*, never a measurement. Benchmark
records store these numbers in fields prefixed ``theory_`` so they can never
be confused with measured values.

Conventions
-----------
* One multiply-add = 2 FLOPs. Exponentials, max and division are counted
  as "elementwise ops" separately (they are not matmul FLOPs and their cost
  on a CPU is very different).
* ``score_entries`` = number of (query, key) logits actually evaluated by
  the implementation (including masked-but-computed entries), per forward.
* Memory is in bytes for the given element size. ``aux_fwd_bytes`` is the
  peak *intermediate* storage of one forward call beyond inputs/outputs,
  for inference (no autograd graph). ``saved_for_bwd_bytes`` is what the
  autograd graph keeps alive between forward and backward when training.
* Causal attention is modelled as computing half of the entries when the
  implementation skips masked tiles (tiled variants, SDPA flash), and all
  of them when it does not (naive, sdpa_math).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

ELEM_BYTES = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}


@dataclass
class Shape:
    B: int
    H: int
    L: int
    d: int
    dtype: str = "float32"
    causal: bool = False

    @property
    def bytes(self) -> int:
        return ELEM_BYTES[self.dtype]

    @property
    def bh(self) -> int:
        return self.B * self.H


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def theoretical_cost(method: str, shape: Shape, params: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Return analytical estimates for one forward call (and training-memory notes)."""
    p = dict(params or {})
    B, H, L, d, eb = shape.B, shape.H, shape.L, shape.d, shape.bytes
    bh = shape.bh
    io_bytes = 4 * bh * L * d * eb                       # q, k, v, out
    full_pairs = L * L
    causal_pairs = L * (L + 1) // 2
    useful_pairs = causal_pairs if shape.causal else full_pairs
    out_bytes = bh * L * d * eb
    res: Dict[str, float] = {"io_bytes": io_bytes, "out_bytes": out_bytes, "useful_pairs": bh * useful_pairs}

    if method in ("naive", "sdpa_math"):
        entries = bh * full_pairs
        res.update(
            score_entries=entries,
            matmul_flops=2 * 2 * entries * d,           # QK^T and PV
            elementwise_ops=4 * entries,                # scale/mask, exp, sum, div
            # naive: s, e, p alive together at the peak (3 L^2 tensors, + bool mask if causal)
            aux_fwd_bytes=3 * entries * eb + (full_pairs if shape.causal else 0),
            saved_for_bwd_bytes=3 * entries * eb,
            asymptotic_time="O(B H L^2 d)",
            asymptotic_aux_memory="O(B H L^2)",
        )
    elif method == "sdpa_flash_cpu":
        # Fused kernel: per-thread score blocks only; block sizes are internal to ATen.
        entries = bh * (causal_pairs if shape.causal else full_pairs)
        res.update(
            score_entries=entries,
            matmul_flops=2 * 2 * entries * d,
            elementwise_ops=4 * entries,
            aux_fwd_bytes=bh * L * 4 * 2,               # logsumexp (+ small per-thread buffers, unknown)
            saved_for_bwd_bytes=bh * L * 4,             # logsumexp; q,k,v,out already counted as io
            asymptotic_time="O(B H L^2 d)",
            asymptotic_aux_memory="O(B H L)  (+ O(threads * block^2) scratch)",
        )
    elif method in ("tiled", "tiled_recompute"):
        bq = p.get("q_tile", 128)
        bk = p.get("kv_tile", 128)
        tiles = 0
        tile_entries = 0
        for qs in range(0, L, bq):
            qe = min(qs + bq, L)
            for ks in range(0, L, bk):
                if shape.causal and ks > qe - 1:
                    break
                tiles += 1
                tile_entries += (qe - qs) * (min(ks + bk, L) - ks)
        entries = bh * tile_entries
        tile_bytes = bh * min(bq, L) * min(bk, L) * eb
        aux = 4 * tile_bytes + bh * L * d * eb + bh * L * eb * 3   # s, p, temporaries; out blocks; m, l, lse
        saved = (3 * entries * eb) if method == "tiled" else bh * L * eb   # autograd keeps every tile vs lse only
        res.update(
            score_entries=entries,
            matmul_flops=2 * 2 * entries * d,
            elementwise_ops=6 * entries,               # + rescaling by alpha
            aux_fwd_bytes=aux,
            saved_for_bwd_bytes=saved,
            backward_matmul_flops=(2 * 4 * entries * d) if method == "tiled" else (2 * 5 * entries * d),
            python_tile_iterations=tiles,
            asymptotic_time="O(B H L^2 d)",
            asymptotic_aux_memory="O(B H bq bk) per step (inference)"
            + ("; training O(B H L^2) via autograd" if method == "tiled" else "; training O(B H L d)"),
        )
    elif method == "local_window":
        w = p.get("window", 64)
        wl, wr = (w, 0) if shape.causal else (w, w)
        C = p.get("chunk") or max(16, min(256, wl + wr + 1))
        nC = _ceil_div(L, C)
        span = C + wl + wr
        entries = bh * nC * C * span
        res.update(
            score_entries=entries,
            band_pairs=bh * sum(min(L - 1, i + wr) - max(0, i - wl) + 1 for i in range(L)),
            matmul_flops=2 * 2 * entries * d,
            elementwise_ops=4 * entries,
            aux_fwd_bytes=3 * entries * eb + 2 * bh * nC * span * d * eb,
            saved_for_bwd_bytes=3 * entries * eb,
            asymptotic_time="O(B H L (w + C) d)",
            asymptotic_aux_memory="O(B H L (w + C))",
        )
    elif method == "block_sparse":
        from ..algorithms.block_sparse import make_layout

        b = p.get("block_size", 32)
        nB = _ceil_div(L, b)
        layout = make_layout(nB, local_blocks=p.get("local_blocks", 1), global_blocks=p.get("global_blocks", 0),
                             random_blocks=p.get("random_blocks", 0), causal=shape.causal, seed=p.get("seed", 0))
        counts = layout.sum(1)
        dense = counts == nB if nB > 1 else counts < 0
        k_max = int(counts[~dense].max()) if (~dense).any() else 0
        entries = bh * (int((~dense).sum()) * b * k_max * b + int(dense.sum()) * b * nB * b)
        res.update(
            score_entries=entries,
            active_blocks=int(layout.sum()),
            layout_density=float(layout.float().mean()),
            matmul_flops=2 * 2 * entries * d,
            elementwise_ops=4 * entries,
            aux_fwd_bytes=3 * entries * eb + 2 * bh * nB * k_max * b * d * eb,
            saved_for_bwd_bytes=3 * entries * eb,
            asymptotic_time="O(B H L k b d)  with k active blocks per row",
            asymptotic_aux_memory="O(B H L k b)",
        )
    elif method == "linear_elu":
        C = p.get("chunk", 64)
        nC = _ceil_div(L, C)
        if shape.causal:
            entries = bh * nC * C * C
            state = bh * nC * d * d * 3                # chunk states, prefix, exclusive prefix
            flops = 2 * (2 * entries * d + 2 * bh * nC * C * d * d + bh * nC * C * d * d)
            aux = (2 * entries + state) * eb
        else:
            entries = 0
            state = bh * d * d
            flops = 2 * 2 * bh * L * d * d
            aux = (state + 2 * bh * L * d) * eb
        res.update(
            score_entries=entries,
            state_entries=state,
            matmul_flops=flops,
            elementwise_ops=2 * bh * L * d,
            aux_fwd_bytes=aux,
            saved_for_bwd_bytes=aux,
            asymptotic_time="O(B H L C d + B H L d^2)" if shape.causal else "O(B H L d^2)",
            asymptotic_aux_memory="O(B H (L C + (L/C) d^2))" if shape.causal else "O(B H (L d + d^2))",
        )
    else:
        raise KeyError(f"no complexity model for {method!r}")
    # Comparable to the measured forward tensor peak, which includes the output allocation.
    res["fwd_alloc_bytes"] = res["aux_fwd_bytes"] + out_bytes
    return res


def estimated_peak_bytes(method: str, shape: Shape, params: Dict[str, Any] | None, mode: str) -> int:
    """Conservative estimate of peak process memory growth, used only by the memory guard."""
    c = theoretical_cost(method, shape, params)
    base = c["io_bytes"] * (3 if mode == "fwd_bwd" else 1)            # + grads of q,k,v
    if mode == "fwd_bwd":
        extra = c["saved_for_bwd_bytes"] + 2 * c["aux_fwd_bytes"]      # backward needs ~ another set of L^2 temps
    else:
        extra = c["aux_fwd_bytes"]
    return int(base + extra)
