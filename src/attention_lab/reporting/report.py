"""Turn a results directory into tables, fitted scaling exponents, figures and a markdown summary.

Every number in a generated report comes from ``results.jsonl``. Fields
prefixed ``theory_`` are labeled as theory wherever they appear.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..benchmarking.scheduler import PROJECT_ROOT, load_results
from ..visualization.plots import ensure_dir, plot_memory_compare, plot_memory_vs_L, plot_metric_by_label, plot_runtime_vs_L

REPORTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "reports")
MiB = 2**20


def results_frame(run_dir: str) -> pd.DataFrame:
    recs = load_results(run_dir)
    for r in recs:
        r.pop("env", None)
        r.pop("traceback", None)
        for k in list(r):
            if k.endswith("_samples_s"):
                r.pop(k)
        if isinstance(r.get("params"), dict):
            r["params"] = json.dumps(r["params"], sort_keys=True)
    return pd.DataFrame(recs)


def fit_exponent(L: np.ndarray, t: np.ndarray, min_L: int = 512) -> Optional[float]:
    """Least-squares slope of log(t) vs log(L) over L >= min_L (needs >= 3 points)."""
    mask = (L >= min_L) & np.isfinite(t) & (t > 0)
    if mask.sum() < 3:
        return None
    slope, _ = np.polyfit(np.log(L[mask]), np.log(t[mask]), 1)
    return float(slope)


def exponent_table(df: pd.DataFrame, min_L: int = 512) -> pd.DataFrame:
    rows = []
    ok = df[df["status"].isin(["ok", "partial"])]
    for (label, causal), g in ok.groupby(["label", "causal"]):
        row: Dict[str, Any] = {"label": label, "causal": causal}
        for mode in ("fwd", "fwd_bwd"):
            col = f"{mode}_median_s"
            if col in g:
                gg = g[g[col].notna()]
                row[f"{mode}_time_exponent"] = fit_exponent(gg["L"].to_numpy(float), gg[col].to_numpy(float), min_L)
            ccol = f"{mode}_cpu_time_per_call_s"
            if ccol in g:
                gg = g[g[ccol].notna()]
                row[f"{mode}_cputime_exponent"] = fit_exponent(gg["L"].to_numpy(float), gg[ccol].to_numpy(float), min_L)
            mcol = f"{mode}_tensor_peak_bytes"
            if mcol in g:
                gg = g[g[mcol].notna()]
                row[f"{mode}_mem_exponent"] = fit_exponent(gg["L"].to_numpy(float), gg[mcol].to_numpy(float), min_L)
        row["L_range_fit"] = f">= {min_L}"
        row["max_L_measured"] = int(g["L"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "label": "method", "L": "L", "causal": "causal", "threads": "threads", "dtype": "dtype", "status": "status",
    }
    out = df[list(cols)].rename(columns=cols).copy()

    def ms(c):
        return df[c] * 1e3 if c in df else np.nan

    def mib(c):
        return df[c] / MiB if c in df else np.nan

    out["fwd_ms_median"] = ms("fwd_median_s")
    out["fwd_ms_iqr"] = ms("fwd_iqr_s")
    out["fwd_bwd_ms_median"] = ms("fwd_bwd_median_s")
    out["fwd_bwd_ms_iqr"] = ms("fwd_bwd_iqr_s")
    out["fwd_cpu_ms_per_call"] = ms("fwd_cpu_time_per_call_s")
    out["fwd_bwd_cpu_ms_per_call"] = ms("fwd_bwd_cpu_time_per_call_s")
    out["fwd_tensor_peak_MiB"] = mib("fwd_tensor_peak_bytes")
    out["fwd_bwd_tensor_peak_MiB"] = mib("fwd_bwd_tensor_peak_bytes")
    out["proc_peak_delta_fwd_MiB"] = mib("proc_peak_delta_after_fwd_bytes")
    out["proc_peak_delta_fwd_bwd_MiB"] = mib("proc_peak_delta_after_fwd_bwd_bytes")
    out["theory_fwd_alloc_MiB"] = mib("theory_fwd_alloc_bytes")
    out["theory_score_entries"] = df.get("theory_score_entries")
    err = df.get("error_vs_reference_max_abs_normalized")
    dev = df.get("deviation_from_full_softmax_max_abs_normalized")
    out["err_vs_ref (exact)"] = err
    out["deviation_from_softmax (approx)"] = dev
    out["other_load_cores"] = df.get("other_load_cores_mean")
    out["contended"] = df.get("contended")
    out["note"] = df.get("reason")
    return out.sort_values(["causal", "method", "L"])


def _fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return ""
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1000 or abs(v) < 0.01:
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_report(run_dir: str, out_dir: Optional[str] = None) -> str:
    name = os.path.basename(os.path.normpath(run_dir))
    out_dir = ensure_dir(out_dir or os.path.join(REPORTS_DIR, name))
    df = results_frame(run_dir)
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    spec = meta.get("spec", {})
    env = meta.get("env", {})
    figs: List[str] = []
    multi_L = df["L"].nunique() > 1
    if multi_L:
        p = plot_runtime_vs_L(df, os.path.join(out_dir, "runtime_vs_L.png"))
        if p:
            figs.append(p)
        if "fwd_cpu_time_per_call_s" in df and df["fwd_cpu_time_per_call_s"].notna().any():
            p = plot_runtime_vs_L(df, os.path.join(out_dir, "cputime_vs_L.png"), metric="cpu_time_per_call_s",
                                  title_suffix=" — process CPU time per call")
            if p:
                figs.append(p)
        for causal in sorted(df["causal"].unique()):
            p = plot_memory_compare(df, os.path.join(out_dir, f"memory_compare_{'causal' if causal else 'noncausal'}.png"),
                                    causal=bool(causal))
            if p:
                figs.append(p)
            for mode in ("fwd", "fwd_bwd"):
                p = plot_memory_vs_L(df, os.path.join(out_dir, f"memory_{mode}_{'causal' if causal else 'noncausal'}.png"),
                                     mode=mode, causal=bool(causal))
                if p:
                    figs.append(p)
        if "error_vs_reference_max_abs_normalized" in df:
            p = plot_metric_by_label(df[df["causal"] == False], os.path.join(out_dir, "error_vs_L.png"), "L",  # noqa: E712
                                     "error_vs_reference_max_abs_normalized", "max |err| / max |ref|",
                                     "Exact methods: float32 error vs float64 naive (non-causal)", logx=True, logy=True)
            if p:
                figs.append(p)
    if df["threads"].nunique() > 1:
        p = plot_metric_by_label(df, os.path.join(out_dir, "threads_fwd.png"), "threads", "fwd_median_s",
                                 "median fwd time (s)", "Forward time vs CPU threads", logy=True)
        if p:
            figs.append(p)

    summ = summary_table(df)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    expo = exponent_table(df) if multi_L else pd.DataFrame()
    if not expo.empty:
        expo.to_csv(os.path.join(out_dir, "exponents.csv"), index=False)

    status_counts = df["status"].value_counts().to_dict()
    lines = [
        f"# Benchmark report: {name}",
        "",
        f"*Generated from `{os.path.relpath(run_dir, PROJECT_ROOT)}/results.jsonl` — all numbers are measurements "
        "unless the column name says `theory`.*",
        "",
        f"**Hypothesis (from spec):** {spec.get('hypothesis', '(none recorded)')}",
        "",
        "## Environment",
        "",
        f"- CPU: {env.get('cpu')} ({env.get('physical_cores')} cores / {env.get('logical_cores')} threads), "
        f"capability {env.get('torch_cpu_capability')}",
        f"- RAM: {env.get('total_ram_gb')} GB total, {env.get('available_ram_gb')} GB available at start",
        f"- OS: {env.get('os')}; Python {env.get('python')}; torch {env.get('torch')}; CUDA available: {env.get('cuda_available')}",
        f"- Isolated worker per config: {meta.get('isolate')}; warm-up {spec.get('warmup')}, repeats {spec.get('repeats')}",
        "",
        f"Run status counts: {status_counts}",
        "",
        "## Measurement scopes",
        "",
        "- `*_ms_median` / `*_ms_iqr`: wall-clock per call, median and inter-quartile range over timed repeats.",
        "- `*_cpu_ms_per_call`: process CPU time (all threads) over the timed loop / number of calls. Excludes time "
        "spent waiting for a core under background load; includes OpenMP spin-waiting; ~15.6 ms Windows tick "
        "resolution over the whole loop.",
        "- `*_tensor_peak_MiB`: peak bytes allocated through PyTorch's CPU allocator *during* one call "
        "(profiler memory events); excludes pre-existing inputs and library-internal scratch.",
        "- `proc_peak_delta_*_MiB`: OS peak working set minus RSS before the first call (isolated runs only). "
        "It is a cumulative high-water mark: the `fwd_bwd` value is max(fwd, fwd+bwd) and includes allocator caching.",
        "- `theory_*`: analytical estimates from `complexity/theory.py`, never measurements.",
        "- `other_load_cores`: mean number of logical CPUs kept busy by *other* processes during the timed "
        "region (worst mode); `contended` = above 1.0 core. Timings from contended runs are not comparable "
        "with clean ones; memory measurements are unaffected.",
        "",
    ]
    if "contended" in df:
        n_cont = int(df["contended"].fillna(False).astype(bool).sum())
        n_ok = int(df["status"].isin(["ok", "partial"]).sum())
        lines += [f"**Background load:** {n_cont} of {n_ok} measured configs were contended "
                  f"(median other-load {df['other_load_cores_mean'].median():.1f} cores, "
                  f"max {df['other_load_cores_mean'].max():.1f}).", ""]
    if not expo.empty:
        lines += ["## Fitted scaling exponents (slope of log-log fit, L >= 512)", "",
                  "An exponent near 2 means quadratic growth in L over the measured range; near 1 means linear. "
                  "Fits over few points are indicative only.", "", to_markdown(expo), ""]
    lines += ["## Results", "", to_markdown(summ), ""]
    if figs:
        lines += ["## Figures", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in figs] + [""]
    failed = df[~df["status"].isin(["ok"])]
    if not failed.empty:
        lines += ["## Skipped / failed / partial runs", ""]
        for _, r in failed.iterrows():
            lines.append(f"- `{r['label']}` L={r['L']} causal={r['causal']}: **{r['status']}** — {r.get('reason', '')}")
        lines.append("")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
