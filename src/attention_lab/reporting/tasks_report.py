"""Reports for the quality experiments (attention-level probes and trained synthetic tasks)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd

from ..visualization.plots import METHOD_STYLE, THEORY_COLOR, ensure_dir, plot_bars, style_for
from .report import REPORTS_DIR, to_markdown

import matplotlib.pyplot as plt  # noqa: E402  (backend already set by visualization.plots)


def _line_plot(df: pd.DataFrame, x: str, y: str, out: str, title: str, xlabel: str, logx: bool = False,
               vlines: List[float] | None = None) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, g in df.groupby("label", sort=False):
        g = g.sort_values(x)
        st = style_for(g["method"].iloc[0])
        ax.plot(g[x], g[y], color=st["color"], marker=st["marker"], lw=2, ms=6, label=label)
    for v in vlines or []:
        ax.axvline(v, ls="--", lw=1, color=THEORY_COLOR)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("accuracy", fontsize=9)
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def probe_report(records: List[Dict[str, Any]], out_dir: str) -> str:
    ensure_dir(out_dir)
    df = pd.DataFrame(records)
    df["params"] = df["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    df.to_csv(os.path.join(out_dir, "probes.csv"), index=False)
    figs = []
    d = df[df.experiment == "distance_sweep"]
    if not d.empty:
        figs.append(_line_plot(d, "distance", "accuracy", os.path.join(out_dir, "probe_distance.png"),
                               f"Retrieval accuracy vs distance (L={int(d.L.iloc[0])}, beta={d.beta.iloc[0]})",
                               "distance from query to target key", logx=True, vlines=[64]))
    for exp, title in (("length_sweep_near", "target 8 tokens back"), ("length_sweep_far", "target at position 0")):
        d = df[df.experiment == exp]
        if not d.empty:
            figs.append(_line_plot(d, "L", "accuracy", os.path.join(out_dir, f"probe_{exp}.png"),
                                   f"Retrieval accuracy vs sequence length ({title})", "sequence length L", logx=True))
    d = df[df.experiment == "decoys"]
    if not d.empty:
        figs.append(_line_plot(d, "n_decoys", "accuracy", os.path.join(out_dir, "probe_decoys.png"),
                               f"Interference: accuracy vs number of decoy keys (cos={d.decoy_cos.iloc[0]}, L={int(d.L.iloc[0])})",
                               "number of decoy keys"))
    for L in sorted(df.L[df.experiment.str.startswith("sharpness")].unique()):
        d = df[df.experiment == f"sharpness_L{L}"]
        figs.append(_line_plot(d, "beta", "accuracy", os.path.join(out_dir, f"probe_sharpness_L{L}.png"),
                               f"Accuracy vs target-logit scale beta (L={L}, distance 4; chance = {1/L:.3f})",
                               "beta (target logit q.k/sqrt(d))", logx=True))
    lines = ["# Attention-level retrieval probes", "",
             "No training: hand-constructed q/k/v (see `tasks/probes.py`). Accuracy = fraction of trials where the "
             "output at the last position is closest (cosine) to the target's value vector.", ""]
    for exp in df.experiment.unique():
        sub = df[df.experiment == exp]
        xcol = {"distance_sweep": "distance", "decoys": "n_decoys"}.get(exp, "beta" if exp.startswith("sharpness") else "L")
        piv = sub.pivot_table(index=xcol, columns="label", values="accuracy").reset_index()
        lines += [f"## {exp}", "", to_markdown(piv), ""]
    lines += ["## Figures", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in figs]
    path = os.path.join(out_dir, "probes.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def training_report(results_path: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    recs = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "eval_task_params", "params")} for r in recs])
    df.to_csv(os.path.join(out_dir, "tasks.csv"), index=False)
    lines = [f"# Synthetic-task training results ({os.path.basename(os.path.dirname(results_path))})", "",
             "`trained_with`: a model trained from scratch with that backend. `swapped_into_*`: the reference model "
             "(trained with the first backend) evaluated with another backend substituted at inference, no retraining.",
             ""]
    ok = df[df.status == "ok"]
    figs = []
    for (exp, mode), g in ok.groupby(["experiment", "mode"], sort=False):
        cols = ["label", "eval_tag", "L", "accuracy", "loss"] + [c for c in g.columns if c.startswith("acc_dist_")]
        if "train_seconds" in g and g["train_seconds"].notna().any():
            cols += ["train_seconds"]
        lines += [f"## {exp} — {mode}", "", to_markdown(g[cols].dropna(axis=1, how="all")), ""]
        tags = list(dict.fromkeys(g.eval_tag))
        for tag in tags:
            gg = g[g.eval_tag == tag]
            p = plot_bars(list(gg.label), list(gg.accuracy), os.path.join(out_dir, f"{exp}_{mode}_{tag}.png".replace("/", "_")),
                          f"{exp} [{mode}] eval={tag}", "accuracy",
                          colors=[style_for(m)["color"] for m in gg.method], annotate_fmt="{:.2f}")
            figs.append(p)
    err = df[df.status != "ok"]
    if not err.empty:
        lines += ["## Failed runs", ""] + [f"- {r.label}: {r.reason}" for r in err.itertuples()]
    lines += ["", "## Figures", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in figs]
    path = os.path.join(out_dir, "tasks.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
