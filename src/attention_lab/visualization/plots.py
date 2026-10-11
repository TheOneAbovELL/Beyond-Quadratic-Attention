"""Static matplotlib figures built only from measured benchmark records.

Conventions
-----------
* Each method keeps one color and one marker in every figure (color follows
  the entity, never its rank); markers are a secondary encoding so identity
  is never carried by color alone.
* Measured values: solid lines with markers. Theoretical curves/slopes:
  dashed gray (or dashed method color), always labeled "theory".
* Log-log axes for scaling plots: a power law t ~ L^p appears as a straight
  line of slope p, which makes O(L) vs O(L^2) visually testable.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_STYLE: Dict[str, Dict[str, str]] = {
    "naive":           {"color": "#2a78d6", "marker": "o"},
    "sdpa_math":       {"color": "#eb6834", "marker": "s"},
    "sdpa_flash_cpu":  {"color": "#1baf7a", "marker": "D"},
    "tiled":           {"color": "#eda100", "marker": "^"},
    "tiled_recompute": {"color": "#e87ba4", "marker": "v"},
    "local_window":    {"color": "#008300", "marker": "P"},
    "block_sparse":    {"color": "#4a3aa7", "marker": "X"},
    "linear_elu":      {"color": "#e34948", "marker": "*"},
}
THEORY_COLOR = "#8a8984"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def style_for(method: str) -> Dict[str, str]:
    return METHOD_STYLE.get(method, {"color": TEXT_2, "marker": "o"})


def _setup_axes(ax, title: str, xlabel: str, ylabel: str, log: bool = True) -> None:
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    if log:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _slope_guides(ax, xs: List[float], y0: float, slopes=(1, 2)) -> None:
    """Dashed reference lines through (min x, y0) with the given log-log slopes (labeled as theory)."""
    if not xs or not y0 or y0 <= 0:
        return
    x0, x1 = min(xs), max(xs)
    for p in slopes:
        ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** p], ls="--", lw=1, color=THEORY_COLOR, zorder=1)
        ax.annotate(f"theory ∝ L^{p}", (x1, y0 * (x1 / x0) ** p), fontsize=7, color=THEORY_COLOR,
                    xytext=(3, 0), textcoords="offset points", va="center")


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(["ok", "partial"])]


def plot_runtime_vs_L(df: pd.DataFrame, out_path: str, title_suffix: str = "", metric: str = "median_s") -> Optional[str]:
    """metric: "median_s" (wall clock, with IQR bars) or "cpu_time_per_call_s" (no dispersion available)."""
    d = _ok(df)
    if d.empty:
        return None
    facets = [(c, m) for c in sorted(d["causal"].unique()) for m in ("fwd", "fwd_bwd")
              if f"{m}_{metric}" in d and d[f"{m}_{metric}"].notna().any()]
    fig, axes = plt.subplots(1, len(facets), figsize=(5.2 * len(facets), 4.2), squeeze=False)
    for ax, (causal, mode) in zip(axes[0], facets):
        col = f"{mode}_{metric}"
        sub = d[(d["causal"] == causal) & d[col].notna()]
        for label, g in sub.groupby("label"):
            g = g.sort_values("L")
            st = style_for(g["method"].iloc[0])
            ms = g[col] * 1e3
            if metric == "median_s":
                lo = (g[f"{mode}_median_s"] - g[f"{mode}_q1_s"]) * 1e3
                hi = (g[f"{mode}_q3_s"] - g[f"{mode}_median_s"]) * 1e3
                ax.errorbar(g["L"], ms, yerr=[lo, hi], color=st["color"], marker=st["marker"], ms=6, lw=2,
                            capsize=2, label=label, zorder=3)
            else:
                ax.plot(g["L"], ms, color=st["color"], marker=st["marker"], ms=6, lw=2, label=label, zorder=3)
        if not sub.empty:
            first_L = sub["L"].min()
            y0 = sub[sub["L"] == first_L][col].min() * 1e3
            _slope_guides(ax, sorted(sub["L"].unique()), y0)
        ylabel = "median time per call (ms)  [IQR bars]" if metric == "median_s" else "CPU time per call (ms)"
        _setup_axes(ax, f"{'causal' if causal else 'non-causal'} · {mode.replace('_', '+')}", "sequence length L", ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8, frameon=False)
    fig.suptitle(f"Measured {'runtime' if metric == 'median_s' else 'CPU time'} vs sequence length{title_suffix}", fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_memory_compare(df: pd.DataFrame, out_path: str, causal: bool = False) -> Optional[str]:
    """All methods on one shared log-log axis per mode: measured tensor peak vs L, with L and L^2 guides."""
    d = _ok(df)
    d = d[d["causal"] == causal]
    modes = [m for m in ("fwd", "fwd_bwd") if f"{m}_tensor_peak_bytes" in d and d[f"{m}_tensor_peak_bytes"].notna().any()]
    if not modes:
        return None
    fig, axes = plt.subplots(1, len(modes), figsize=(5.4 * len(modes), 4.2), squeeze=False, sharey=True)
    for ax, mode in zip(axes[0], modes):
        col = f"{mode}_tensor_peak_bytes"
        sub = d[d[col].notna()]
        for label, g in sub.groupby("label"):
            g = g.sort_values("L")
            st = style_for(g["method"].iloc[0])
            ax.plot(g["L"], g[col] / 2**20, color=st["color"], marker=st["marker"], lw=2, ms=6, label=label, zorder=3)
        Ls = sorted(sub["L"].unique())
        y0 = sub[sub["L"] == Ls[0]][col].median() / 2**20
        _slope_guides(ax, Ls, y0)
        _setup_axes(ax, f"{mode.replace('_', '+')} · {'causal' if causal else 'non-causal'}", "sequence length L",
                    "measured tensor peak (MiB)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8, frameon=False)
    fig.suptitle("Measured peak tensor memory per call vs L (PyTorch CPU allocator)", fontsize=11, color=TEXT,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_memory_vs_L(df: pd.DataFrame, out_path: str, mode: str = "fwd", causal: bool = False) -> Optional[str]:
    """Small multiples: one panel per method, measured tensor peak vs theoretical estimate."""
    d = _ok(df)
    d = d[(d["causal"] == causal) & d.get(f"{mode}_tensor_peak_bytes", pd.Series(dtype=float)).notna()]
    if d.empty:
        return None
    labels = sorted(d["label"].unique())
    n = len(labels)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        g = d[d["label"] == label].sort_values("L")
        st = style_for(g["method"].iloc[0])
        ax.plot(g["L"], g[f"{mode}_tensor_peak_bytes"] / 2**20, color=st["color"], marker=st["marker"], lw=2,
                label="measured: tensor peak", zorder=3)
        proc_col = f"proc_peak_delta_after_{mode}_bytes"
        if proc_col in g and g[proc_col].notna().any():
            ax.plot(g["L"], g[proc_col] / 2**20, color=st["color"], marker=st["marker"], lw=1, alpha=0.45,
                    label="measured: process peak Δ", zorder=2)
        theory_col = "theory_fwd_alloc_bytes" if mode == "fwd" else None
        if theory_col and theory_col in g:
            ax.plot(g["L"], g[theory_col] / 2**20, ls="--", lw=1.2, color=THEORY_COLOR, label="theory: fwd allocations")
        _setup_axes(ax, label, "L", "MiB")
        ax.legend(fontsize=7, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Memory vs L ({mode.replace('_', '+')}, {'causal' if causal else 'non-causal'}) — measured vs theory",
                 fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_metric_by_label(df: pd.DataFrame, out_path: str, x: str, y: str, ylabel: str, title: str,
                         logx: bool = False, logy: bool = False, labels: Optional[Iterable[str]] = None) -> Optional[str]:
    d = _ok(df)
    d = d[d[y].notna()] if y in d else d.iloc[0:0]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, g in d.groupby("label"):
        if labels is not None and label not in labels:
            continue
        g = g.groupby(x, as_index=False)[y].median().sort_values(x)
        st = style_for(label.split("[")[0])
        ax.plot(g[x], g[y], color=st["color"], marker=st["marker"], lw=2, ms=6, label=label)
    _setup_axes(ax, title, x, ylabel, log=False)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_bars(categories: List[str], values: List[float], out_path: str, title: str, ylabel: str,
              colors: Optional[List[str]] = None, annotate_fmt: str = "{:.1f}") -> str:
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(categories)), 4))
    colors = colors or [TEXT_2] * len(categories)
    bars = ax.bar(range(len(categories)), values, color=colors, width=0.7, edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
    for b, v in zip(bars, values):
        ax.annotate(annotate_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", fontsize=7, color=TEXT_2, xytext=(0, 2), textcoords="offset points")
    _setup_axes(ax, title, "", ylabel, log=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
