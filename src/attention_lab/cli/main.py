"""Command-line interface. Every experiment in the lab can be run from here without the UI.

    python lab.py env                      # environment audit
    python lab.py test                     # essential test suite
    python lab.py correctness [--full]     # numerical correctness laboratory
    python lab.py bench --smoke            # minimal benchmark (< 1 min)
    python lab.py bench --spec experiments/configs/scaling_exact.yaml
    python lab.py report <results-dir>     # tables + figures for a benchmark run
    python lab.py complexity --L 4096      # theoretical cost table (labeled theory)
    python lab.py probes [--full]          # attention-level retrieval probes
    python lab.py tasks [--quick]          # train tiny Transformers on synthetic tasks
    python lab.py ui                       # Streamlit research interface
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

from ..benchmarking.scheduler import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

CONFIG_DIR = os.path.join(PROJECT_ROOT, "experiments", "configs")


def _stamp(name: str) -> str:
    return os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%H%M%S") + f"_{name}")


def _write_meta(out_dir: str, kind: str, **extra) -> None:
    from ..benchmarking.env import environment_info

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"kind": kind, "env": environment_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra},
                  f, indent=2, default=str)


def cmd_env(args) -> int:
    from ..benchmarking.env import environment_info

    info = environment_info()
    width = max(len(k) for k in info)
    for k, v in info.items():
        print(f"{k:<{width}}  {v}")
    return 0


def cmd_test(args) -> int:
    cmd = [sys.executable, "-m", "pytest", os.path.join(PROJECT_ROOT, "tests")]
    if not args.all:
        cmd += ["-m", "not slow"]
    cmd += args.pytest_args
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def cmd_correctness(args) -> int:
    import pandas as pd

    from ..algorithms.registry import EXACT_METHODS
    from ..correctness.lab import run_suite, unstable_softmax_demo
    from ..reporting.report import to_markdown

    methods = args.methods or EXACT_METHODS
    out_dir = _stamp("correctness")
    _write_meta(out_dir, "correctness", methods=list(methods), full=bool(args.full))
    n = [0]

    def prog(rec):
        n[0] += 1
        if n[0] % 50 == 0:
            print(f"  {n[0]} comparisons ...", flush=True)

    t0 = time.perf_counter()
    recs = run_suite(methods, quick=not args.full, progress=prog)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")
    df = pd.DataFrame(recs)
    ok = df[df.status == "ok"]
    summ = ok.groupby(["method", "dtype"]).agg(
        cases=("pass_forward", "size"), pass_forward=("pass_forward", "mean"), pass_grad=("pass_grad", "mean"),
        worst_fwd_err=("fwd_max_abs_normalized", "max"), worst_grad_err=("grad_max_abs_normalized", "max"),
        median_fwd_err=("fwd_max_abs_normalized", "median"),
    ).reset_index()
    fails = ok[~(ok.pass_forward & ok.pass_grad)]
    demo = pd.DataFrame(unstable_softmax_demo())
    lines = ["# Numerical correctness report", "",
             f"{len(df)} comparisons ({'full' if args.full else 'quick'} matrix) in {time.perf_counter() - t0:.1f}s. "
             "Errors are max|x - ref| / max|ref| against naive attention in float64 on the same rounded inputs.", "",
             to_markdown(summ), "", "## Failures", ""]
    if fails.empty:
        lines.append("None.")
    else:
        cols = ["method", "dtype", "L", "d", "causal", "magnitude", "fwd_max_abs_normalized", "tol_forward",
                "grad_max_abs_normalized", "tol_grad"]
        lines.append(to_markdown(fails[cols]))
    lines += ["", "## Why max-subtraction matters (unstable softmax demo, float32)", "", to_markdown(demo)]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(summ.to_string(index=False))
    if not fails.empty:
        print("\nFailures:")
        print(fails[["method", "dtype", "L", "d", "causal", "magnitude", "fwd_max_abs_normalized", "tol_forward",
                     "grad_max_abs_normalized", "tol_grad"]].to_string(index=False))
    print(f"\nwrote {out_dir}")
    errors = df[df.status == "error"]
    if not errors.empty:
        print(errors[["method", "reason"]].to_string(index=False))
    return 0


def cmd_bench(args) -> int:
    from ..benchmarking.scheduler import load_spec, run_sweep
    from ..reporting.report import build_report

    spec_path = os.path.join(CONFIG_DIR, "smoke.yaml") if args.smoke else args.spec
    if not spec_path:
        print("pass --smoke or --spec PATH", file=sys.stderr)
        return 2
    if not os.path.isfile(spec_path) and os.path.isfile(os.path.join(PROJECT_ROOT, spec_path)):
        spec_path = os.path.join(PROJECT_ROOT, spec_path)
    if not os.path.isfile(spec_path):
        print(f"spec file not found: {spec_path} (available: {sorted(os.listdir(CONFIG_DIR))})", file=sys.stderr)
        return 2
    spec = load_spec(spec_path)
    if args.threads:
        spec.setdefault("grid", {})["threads"] = args.threads

    def prog(r, i, n):
        fwd = r.get("fwd_median_s")
        fb = r.get("fwd_bwd_median_s")
        t = (f"fwd {fwd*1e3:9.2f} ms" if fwd else " " * 16) + (f"  fwd+bwd {fb*1e3:9.2f} ms" if fb else "")
        print(f"[{i:3d}/{n}] {r['label'][:44]:44s} L={r['L']:6d} causal={str(r['causal']):5s} {r['status']:20s} {t} "
              f"{r.get('reason', '') if r['status'] != 'ok' else ''}", flush=True)

    isolate = None if args.isolate is None else (args.isolate == "yes")
    run_dir = run_sweep(spec, isolate=isolate, progress=prog)
    print(f"results: {run_dir}")
    if not args.no_report:
        print(f"report:  {build_report(run_dir)}")
    return 0


def cmd_report(args) -> int:
    from ..reporting.report import build_report

    print(build_report(args.run_dir))
    return 0


def cmd_complexity(args) -> int:
    import pandas as pd

    from ..algorithms.registry import REGISTRY
    from ..complexity.theory import Shape, theoretical_cost

    rows = []
    for L in args.L:
        for name, m in REGISTRY.items():
            c = theoretical_cost(name, Shape(args.B, args.H, L, args.d, args.dtype, args.causal), m.default_params)
            rows.append({"method": name, "L": L, "theory_score_entries": c["score_entries"],
                         "theory_matmul_GFLOP": c["matmul_flops"] / 1e9,
                         "theory_fwd_alloc_MiB": c["fwd_alloc_bytes"] / 2**20,
                         "theory_saved_for_bwd_MiB": c["saved_for_bwd_bytes"] / 2**20,
                         "time": c["asymptotic_time"], "aux_memory": c["asymptotic_aux_memory"]})
    pd.set_option("display.width", 220)
    print("THEORETICAL ESTIMATES (not measurements)")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_probes(args) -> int:
    from ..reporting.report import REPORTS_DIR
    from ..reporting.tasks_report import probe_report
    from ..tasks.probes import probe_suite

    import torch

    torch.set_num_threads(args.threads)
    out_dir = _stamp("probes")
    _write_meta(out_dir, "probes", full=bool(args.full), threads=args.threads)
    f = open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8")

    def prog(r):
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"{r['experiment']:18s} {r['label'][:60]:60s} L={r['L']:5d} dist={r['distance']:5d} "
              f"decoys={r['n_decoys']:2d} acc={r['accuracy']:.3f}", flush=True)

    recs = probe_suite(quick=not args.full, progress=prog)
    f.close()
    print(probe_report(recs, os.path.join(REPORTS_DIR, os.path.basename(out_dir))))
    return 0


def cmd_tasks(args) -> int:
    from ..models.transformer import ModelConfig
    from ..reporting.report import REPORTS_DIR
    from ..reporting.tasks_report import training_report
    from ..tasks.synthetic import VOCAB_SIZE
    from ..training.train import TrainConfig, run_task_experiment

    import torch

    torch.set_num_threads(args.threads)
    steps = args.steps or (600 if args.quick else 1500)
    out_dir = _stamp("tasks" + ("_quick" if args.quick else ""))
    methods = [
        {"name": "sdpa_flash_cpu"},
        {"name": "local_window", "params": {"window": 8}},
        {"name": "block_sparse", "params": {"block_size": 8, "local_blocks": 1, "global_blocks": 1, "random_blocks": 1}},
        {"name": "linear_elu", "params": {"chunk": 16}},
    ]
    model = ModelConfig(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, n_layers=2, d_mlp=256)
    experiments = []
    if "A" in args.which:
        experiments.append(("taskA_local_copy", TrainConfig(task="local_copy", task_params={"lag": 3}, L_train=64,
                                                             steps=max(200, steps // 3), model=model, threads=args.threads,
                                                             batch_size=args.batch_size),
                            [{"L": 64, "tag": "L64"}, {"L": 256, "tag": "L256"}]))
    if "B" in args.which:
        # Variable-offset induction: copy source 12-20 tokens back (beyond window 8), found by content.
        experiments.append(("taskBCD_induction", TrainConfig(task="induction",
                                                              task_params={"seg": 12, "min_gap": 0, "max_gap": 8},
                                                              L_train=32, steps=steps, model=model, threads=args.threads,
                                                              batch_size=args.batch_size),
                            [{"L": 32, "tag": "B_in_distribution"},
                             {"L": 32, "task_params": {"gap_tokens": "segment"}, "tag": "C_distractor_gap"},
                             {"L": 64, "task_params": {"min_gap": 16, "max_gap": 24}, "tag": "D_offset_28_36"},
                             {"L": 64, "task_params": {"min_gap": 24, "max_gap": 40}, "tag": "D_offset_36_52"}]))
    if "M" in args.which:
        # MQAR (generator v2). Status: not learned within CPU budgets tried; see docs/mqar_diagnosis.md.
        experiments.append(("taskM_mqar", TrainConfig(task="mqar", task_params={"n_pairs": 4, "n_queries": 4},
                                                      L_train=32, steps=steps, model=model, threads=args.threads,
                                                      batch_size=args.batch_size),
                            [{"L": 32, "tag": "M_L32"}]))
    for name, base, evals in experiments:
        path = run_task_experiment(name, base, methods, evals, os.path.join(out_dir, name), progress=print)
        print(training_report(path, os.path.join(REPORTS_DIR, os.path.basename(out_dir), name)))
    return 0


REPRODUCE_STEPS = [
    ("tests", ["test"]),
    ("numerical correctness (C1)", ["correctness", "--full"]),
    ("exact scaling, memory (E1)", ["bench", "--spec", "experiments/configs/scaling_exact.yaml"]),
    ("exact scaling, timing (E1b)", ["bench", "--spec", "experiments/configs/scaling_exact_timing.yaml"]),
    ("approximate scaling (E2)", ["bench", "--spec", "experiments/configs/scaling_approx.yaml"]),
    ("tile sizes (E3)", ["bench", "--spec", "experiments/configs/tile_sizes.yaml"]),
    ("threads (E4)", ["bench", "--spec", "experiments/configs/threads.yaml"]),
    ("shapes and dtypes (E5)", ["bench", "--spec", "experiments/configs/shapes_dtypes.yaml"]),
    ("retrieval probes (Q1)", ["probes", "--full"]),
    ("trained tasks (Q2)", ["tasks", "--which", "AB", "--steps", "1500", "--threads", "1", "--batch-size", "32"]),
]


def cmd_reproduce(args) -> int:
    """Run every experiment behind docs/results.md, one after another (never in parallel)."""
    steps = REPRODUCE_STEPS if not args.only else [s for s in REPRODUCE_STEPS if any(o in s[0] for o in args.only)]
    for title, argv in steps:
        line = "python lab.py " + " ".join(argv)
        print(f"== {title}: {line}", flush=True)
        if args.dry_run:
            continue
        code = main(argv)
        if code != 0:
            print(f"step failed with exit code {code}: {title}", file=sys.stderr)
            if not args.keep_going:
                return code
    return 0


def cmd_ui(args) -> int:
    app = os.path.join(PROJECT_ROOT, "src", "attention_lab", "ui", "app.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.address", "localhost",
           "--browser.gatherUsageStats", "false", "--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="attention-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="print environment audit").set_defaults(fn=cmd_env)

    s = sub.add_parser("test", help="run the essential test suite")
    s.add_argument("--all", action="store_true", help="include slow tests")
    s.add_argument("pytest_args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("correctness", help="numerical correctness laboratory")
    s.add_argument("--full", action="store_true")
    s.add_argument("--methods", nargs="*")
    s.set_defaults(fn=cmd_correctness)

    s = sub.add_parser("bench", help="run a benchmark sweep")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--spec")
    s.add_argument("--isolate", choices=["yes", "no"], default=None, help="override spec's isolate setting")
    s.add_argument("--threads", type=int, nargs="*")
    s.add_argument("--no-report", action="store_true")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("report", help="build a report for a results directory")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("complexity", help="print theoretical cost table")
    s.add_argument("--L", type=int, nargs="*", default=[1024, 4096])
    s.add_argument("--B", type=int, default=1)
    s.add_argument("--H", type=int, default=4)
    s.add_argument("--d", type=int, default=64)
    s.add_argument("--dtype", default="float32")
    s.add_argument("--causal", action="store_true")
    s.set_defaults(fn=cmd_complexity)

    s = sub.add_parser("probes", help="attention-level retrieval probes")
    s.add_argument("--full", action="store_true")
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(fn=cmd_probes)

    s = sub.add_parser("tasks", help="train tiny Transformers on synthetic tasks")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--steps", type=int)
    s.add_argument("--which", default="AB", help="A = local copy; B = variable-offset induction with C (distractor) "
                   "and D (longer offset) evaluations; M = MQAR (optional, see docs/mqar_diagnosis.md)")
    s.add_argument("--threads", type=int, default=1, help="1 is fastest for these tiny models on a busy CPU")
    s.add_argument("--batch-size", type=int, default=32)
    s.set_defaults(fn=cmd_tasks)

    s = sub.add_parser("reproduce", help="run all experiments behind docs/results.md sequentially (hours on a busy CPU)")
    s.add_argument("--dry-run", action="store_true", help="only print the commands")
    s.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    s.add_argument("--only", nargs="*", help="substring filter on step titles, e.g. E2 Q1")
    s.set_defaults(fn=cmd_reproduce)

    s = sub.add_parser("ui", help="launch the Streamlit research interface (localhost only)")
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--headless", action="store_true", help="do not open a browser or prompt")
    s.set_defaults(fn=cmd_ui)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
