"""One diagnostic experiment per invocation (MQAR v2 plateau investigation).

    python scripts/diagnose_recall.py overfit
    python scripts/diagnose_recall.py pairs --n 2
    python scripts/diagnose_recall.py induction --train-gap fixed|random

Logs: experiments/diagnostics/mqar_v2/<experiment>.jsonl (first line = config).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from attention_lab.diagnostics import DiagConfig, run_diagnostic  # noqa: E402
from attention_lab.tasks.synthetic import Batch, fixed_eval_set, induction, mqar  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "diagnostics", "mqar_v2")


def induction_batch(n: int, gen: torch.Generator, seg: int, max_gap: int, L: int) -> Batch:
    """Variable-offset induction (tasks.synthetic.induction). max_gap = 0 gives the fixed-offset variant."""
    return induction(n, L, seg=seg, min_gap=0, max_gap=max_gap, generator=gen)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["overfit", "pairs", "induction"])
    p.add_argument("--n", type=int, default=4, help="pairs (MQAR)")
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--train-gap", choices=["fixed", "random"], default="fixed")
    p.add_argument("--n-train", type=int, default=64, help="fixed dataset size (overfit)")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.experiment in ("overfit", "pairs"):
        n = a.n
        tp = {"n_pairs": n, "n_queries": n}
        heldout = fixed_eval_set("mqar", 256, a.L, 0, **tp)
        sample = lambda bs, g: mqar(bs, a.L, generator=g, **tp)  # noqa: E731
        fixed = None
        name = f"{a.experiment}_pairs{n}_L{a.L}_lr{a.lr:g}"
        if a.experiment == "overfit":
            fixed = mqar(a.n_train, a.L, generator=torch.Generator().manual_seed(123), **tp)
            name += f"_ntrain{a.n_train}"
        cfg = DiagConfig(name=name + a.tag, steps=a.steps, lr=a.lr, n_pairs=n,
                         notes=f"MQAR v2; fixed_train={a.n_train if fixed is not None else None}")
        run_diagnostic(cfg, sample, heldout, os.path.join(OUT, cfg.name + ".jsonl"), fixed_train=fixed)
    else:
        seg, L = 12, 32
        train_gap = 0 if a.train_gap == "fixed" else 8
        sample = lambda bs, g: induction_batch(bs, g, seg, train_gap, L)  # noqa: E731
        heldout = induction_batch(256, torch.Generator().manual_seed(10_000), seg, train_gap, L)
        extra = {"eval_fixedgap": induction_batch(256, torch.Generator().manual_seed(10_001), seg, 0, L),
                 "eval_randomgap": induction_batch(256, torch.Generator().manual_seed(10_002), seg, 8, L)}
        cfg = DiagConfig(name=f"induction_train{a.train_gap}gap_lr{a.lr:g}{a.tag}", steps=a.steps, lr=a.lr,
                         notes="segment 12, vocab 9..40; gap noise 1..8")
        run_diagnostic(cfg, sample, heldout, os.path.join(OUT, cfg.name + ".jsonl"), extra_evals=extra)
    print("done", cfg.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
