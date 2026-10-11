"""Attention algorithms. All take q/k/v shaped [B, H, L, d]; see base.py for mask semantics."""
from .block_sparse import block_sparse_attention, layout_to_element_mask, make_layout
from .linear import linear_attention, linear_attention_quadratic_reference
from .local_window import band_mask, local_window_attention
from .naive import attention_probabilities, naive_attention
from .registry import APPROX_METHODS, EXACT_METHODS, REGISTRY, AttentionMethod, get_method, run_attention
from .sdpa import sdpa_attention, sdpa_flash_cpu, sdpa_math
from .tiled_exact import tiled_attention, tiled_attention_recompute

__all__ = [
    "APPROX_METHODS", "EXACT_METHODS", "REGISTRY", "AttentionMethod", "attention_probabilities",
    "band_mask", "block_sparse_attention", "get_method", "layout_to_element_mask", "linear_attention",
    "linear_attention_quadratic_reference", "local_window_attention", "make_layout", "naive_attention",
    "run_attention", "sdpa_attention", "sdpa_flash_cpu", "sdpa_math", "tiled_attention",
    "tiled_attention_recompute",
]
