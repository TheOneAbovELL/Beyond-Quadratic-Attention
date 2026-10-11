"""Streamlit research interface. Displays ONLY results produced by the backend.

Launch:  python lab.py ui      (binds to localhost; usage statistics disabled)

Nothing here is hard-coded: every table/plot is read from experiments/results
or computed on demand by calling the same functions the CLI uses.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from attention_lab.algorithms import REGISTRY  # noqa: E402
from attention_lab.benchmarking import BenchConfig, environment_info, list_runs, run_one  # noqa: E402
from attention_lab.benchmarking.scheduler import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from attention_lab.complexity import Shape, theoretical_cost  # noqa: E402
from attention_lab.correctness.lab import CorrectnessCase, evaluate_case  # noqa: E402
from attention_lab.reporting.report import REPORTS_DIR, build_report, exponent_table, results_frame, summary_table  # noqa: E402
from attention_lab.tasks.probes import run_probe  # noqa: E402

st.set_page_config(page_title="Beyond Quadratic Attention", layout="wide")
st.title("Beyond Quadratic Attention — research lab")
st.caption("CPU-only measurements from this machine. Columns prefixed `theory_` are analytical estimates, not measurements.")

tabs = st.tabs(["Environment", "Algorithms", "Correctness", "Benchmark", "Results & plots", "Complexity",
                "Quality probes", "Task training", "History & export"])

# ---------------------------------------------------------------- Environment
with tabs[0]:
    info = environment_info(refresh=True)
    st.table(pd.DataFrame({"value": [str(v) for v in info.values()]}, index=list(info.keys())))
    st.info("No CUDA device is used. CPU results do not reproduce GPU performance characteristics.")

# ---------------------------------------------------------------- Algorithms
with tabs[1]:
    rows = [{"name": n, "family": m.family, "computes": m.computes, "implementation": m.implementation,
             "default params": json.dumps(m.default_params), "dense mask": m.supports_dense_mask}
            for n, m in REGISTRY.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _method_params(prefix: str, name: str) -> dict:
    defaults = REGISTRY[name].default_params
    params = {}
    cols = st.columns(max(1, len(defaults)))
    for col, (k, v) in zip(cols, defaults.items()):
        params[k] = col.number_input(f"{k}", value=int(v), min_value=0, step=1, key=f"{prefix}_{name}_{k}")
    return params


# ---------------------------------------------------------------- Correctness
with tabs[2]:
    st.subheader("Compare one exact method against the float64 naive reference")
    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("method", [n for n, m in REGISTRY.items() if m.exact], index=3, key="corr_m")
    dtype = c2.selectbox("dtype", ["float64", "float32", "bfloat16"], index=1)
    L = c3.number_input("L", 1, 2048, 128)
    d = c4.number_input("head dim d", 1, 256, 64)
    c5, c6, c7, c8 = st.columns(4)
    causal = c5.checkbox("causal", True)
    mag = c6.number_input("input magnitude", 0.001, 100.0, 1.0)
    B = c7.number_input("B", 1, 8, 2)
    H = c8.number_input("H", 1, 16, 2)
    params = _method_params("corr", method) if REGISTRY[method].default_params else {}
    if st.button("Run comparison"):
        rec = evaluate_case(method, CorrectnessCase(B=int(B), H=int(H), L=int(L), d=int(d), dtype=dtype, causal=causal,
                                                    magnitude=float(mag), params=params))
        if rec["status"] != "ok":
            st.error(rec.get("reason"))
        else:
            st.write({k: rec[k] for k in rec if k.startswith(("fwd_", "grad_", "tol_", "pass_", "baseline_"))})

# ---------------------------------------------------------------- Benchmark
with tabs[3]:
    st.subheader("Run a single guarded benchmark configuration (in-process)")
    st.caption("Memory guard: skipped if the theoretical estimate exceeds min(50% of available RAM, cap).")
    c1, c2, c3, c4, c5 = st.columns(5)
    bm = c1.selectbox("method", list(REGISTRY), index=2, key="bench_m")
    bL = c2.number_input("L", 16, 16384, 1024, key="bL")
    bd = c3.number_input("d", 8, 256, 64, key="bd")
    bH = c4.number_input("H", 1, 16, 4, key="bH")
    bthreads = c5.number_input("threads", 1, os.cpu_count() or 1, min(4, os.cpu_count() or 1))
    c6, c7, c8, c9 = st.columns(4)
    bcausal = c6.checkbox("causal", False, key="bc")
    bdtype = c7.selectbox("dtype", ["float32", "bfloat16", "float64"], key="bdt")
    brep = c8.number_input("repeats", 1, 50, 5)
    bcap = c9.number_input("memory cap (GB)", 0.1, 8.0, 1.5)
    bparams = _method_params("bench", bm) if REGISTRY[bm].default_params else {}
    if st.button("Run benchmark"):
        with st.spinner("measuring..."):
            rec = run_one(BenchConfig(method=bm, L=int(bL), d=int(bd), H=int(bH), threads=int(bthreads), causal=bcausal,
                                      dtype=bdtype, repeats=int(brep), params=bparams, max_mem_gb=float(bcap)))
        rec.pop("env", None)
        st.write(f"status: **{rec['status']}** {rec.get('reason', '')}")
        keep = [k for k in rec if k.endswith(("median_s", "iqr_s", "tensor_peak_bytes", "_normalized")) or k.startswith("theory_")]
        st.dataframe(pd.DataFrame({"value": [rec[k] for k in keep]}, index=keep), use_container_width=True)

# ---------------------------------------------------------------- Results & plots
runs = list_runs()
with tabs[4]:
    if not runs:
        st.warning("No benchmark runs found. Run `python lab.py bench --smoke` first.")
    else:
        run = st.selectbox("benchmark run", runs[::-1], format_func=os.path.basename)
        df = results_frame(run)
        meta = json.load(open(os.path.join(run, "meta.json"), encoding="utf-8"))
        st.write("**Hypothesis:**", meta.get("spec", {}).get("hypothesis", "(none)"))
        st.write("status counts:", df["status"].value_counts().to_dict())
        summ = summary_table(df)
        st.dataframe(summ, use_container_width=True, hide_index=True)
        ok = df[df.status.isin(["ok", "partial"])]
        if ok["L"].nunique() > 1:
            st.subheader("Measured runtime vs L (median, ms)")
            for causal in sorted(ok.causal.unique()):
                for mode in ("fwd", "fwd_bwd"):
                    col = f"{mode}_median_s"
                    if col in ok:
                        piv = ok[ok.causal == causal].pivot_table(index="L", columns="label", values=col) * 1e3
                        st.caption(f"{'causal' if causal else 'non-causal'} · {mode}")
                        st.line_chart(piv)
            st.subheader("Measured tensor peak (MiB) vs L — forward")
            if "fwd_tensor_peak_bytes" in ok:
                st.line_chart(ok[ok.causal == False].pivot_table(index="L", columns="label",  # noqa: E712
                                                                 values="fwd_tensor_peak_bytes") / 2**20)
            st.subheader("Fitted log-log exponents (L >= 512)")
            st.dataframe(exponent_table(ok), use_container_width=True, hide_index=True)
        rep_dir = os.path.join(REPORTS_DIR, os.path.basename(run))
        if os.path.isdir(rep_dir):
            for f in sorted(os.listdir(rep_dir)):
                if f.endswith(".png"):
                    st.image(os.path.join(rep_dir, f), caption=f)

# ---------------------------------------------------------------- Complexity
with tabs[5]:
    st.subheader("Theoretical cost model (estimates, NOT measurements)")
    c1, c2, c3 = st.columns(3)
    cL = c1.text_input("L values", "512,1024,2048,4096,8192")
    cd = c2.number_input("d", 8, 256, 64, key="cd")
    ccausal = c3.checkbox("causal", False, key="cc")
    rows = []
    for L_ in [int(x) for x in cL.split(",") if x.strip()]:
        for n, m in REGISTRY.items():
            c = theoretical_cost(n, Shape(1, 4, L_, int(cd), "float32", ccausal), m.default_params)
            rows.append({"method": n, "L": L_, "theory_score_entries": c["score_entries"],
                         "theory_GFLOP": c["matmul_flops"] / 1e9, "theory_fwd_alloc_MiB": c["fwd_alloc_bytes"] / 2**20,
                         "time": c["asymptotic_time"], "aux memory": c["asymptotic_aux_memory"]})
    tdf = pd.DataFrame(rows)
    st.dataframe(tdf, use_container_width=True, hide_index=True)
    st.line_chart(tdf.pivot_table(index="L", columns="method", values="theory_fwd_alloc_MiB"))

# ---------------------------------------------------------------- Quality probes
with tabs[6]:
    st.subheader("Attention-level retrieval probe (no training)")
    c1, c2, c3, c4 = st.columns(4)
    pm = c1.selectbox("method", list(REGISTRY), index=2, key="pm")
    pL = c2.number_input("L", 16, 4096, 512, key="pL")
    pdist = c3.number_input("distance", 1, 4095, 100, key="pdist")
    pdec = c4.number_input("decoys", 0, 64, 0, key="pdec")
    pparams = _method_params("probe", pm) if REGISTRY[pm].default_params else {}
    if st.button("Run probe"):
        if pdist >= pL:
            st.error("distance must be < L")
        else:
            st.write(run_probe(pm, L=int(pL), distance=int(pdist), n_decoys=int(pdec), params=pparams, n_trials=32))
    probe_runs = [r for r in sorted(os.listdir(RESULTS_DIR)) if "probes" in r] if os.path.isdir(RESULTS_DIR) else []
    if probe_runs:
        pr = st.selectbox("stored probe runs", probe_runs[::-1])
        recs = [json.loads(l) for l in open(os.path.join(RESULTS_DIR, pr, "results.jsonl"), encoding="utf-8")]
        pdf = pd.DataFrame(recs)
        for exp, x in (("distance_sweep", "distance"), ("length_sweep_far", "L"), ("decoys", "n_decoys")):
            sub = pdf[pdf.experiment == exp]
            if not sub.empty:
                st.caption(exp)
                st.line_chart(sub.pivot_table(index=x, columns="label", values="accuracy"))

# ---------------------------------------------------------------- Task training
with tabs[7]:
    task_runs = [r for r in sorted(os.listdir(RESULTS_DIR)) if r.split("_", 1)[-1].startswith("tasks")] if os.path.isdir(RESULTS_DIR) else []
    if not task_runs:
        st.info("No task-training runs yet: `python lab.py tasks --quick`.")
    for tr in task_runs[::-1]:
        st.markdown(f"**{tr}**")
        for sub in sorted(os.listdir(os.path.join(RESULTS_DIR, tr))):
            p = os.path.join(RESULTS_DIR, tr, sub, "results.jsonl")
            if os.path.isfile(p):
                recs = [json.loads(l) for l in open(p, encoding="utf-8")]
                tdf = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "params", "eval_task_params")}
                                    for r in recs])
                st.caption(sub)
                cols = [c for c in ["mode", "label", "eval_tag", "L", "accuracy", "loss", "train_seconds"] if c in tdf]
                st.dataframe(tdf[cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- History & export
with tabs[8]:
    all_runs = sorted(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else []
    st.dataframe(pd.DataFrame({"run": all_runs[::-1]}), use_container_width=True, hide_index=True)
    if runs:
        er = st.selectbox("export benchmark report for", runs[::-1], format_func=os.path.basename, key="export")
        if st.button("Build report (markdown + PNG + CSV)"):
            path = build_report(er)
            st.success(f"wrote {path}")
            st.download_button("download summary.md", open(path, encoding="utf-8").read(), file_name="summary.md")
            st.download_button("download summary.csv", open(os.path.join(os.path.dirname(path), "summary.csv"),
                                                             encoding="utf-8").read(), file_name="summary.csv")
