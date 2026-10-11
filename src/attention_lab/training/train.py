"""Train/evaluate tiny Transformers on synthetic tasks with interchangeable attention backends.

Fairness controls for a comparison across methods:
* identical model config except ``attention``/``attention_params``;
* identical parameter-initialization seed and training-batch stream (same
  generator seed), identical step budget, optimizer and schedule;
* identical fixed evaluation sets (``fixed_eval_set``), generated once per
  (task, L, params) and never seen in training.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from ..models.transformer import ModelConfig, TinyTransformer, lm_loss
from ..tasks.synthetic import TASK_VERSION, VOCAB_SIZE, accuracy_by_distance, fixed_eval_set, make_batch, shortcut_baseline

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                              "experiments", "checkpoints")


@dataclass
class TrainConfig:
    task: str = "mqar"
    task_params: Dict[str, Any] = field(default_factory=lambda: {"n_pairs": 8, "n_queries": 8})
    L_train: int = 64
    steps: int = 1500
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 0
    threads: int = 4
    eval_every: int = 250
    eval_size: int = 512
    model: ModelConfig = field(default_factory=lambda: ModelConfig(vocab_size=VOCAB_SIZE))


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: TinyTransformer, task: str, L: int, n: int, seed: int, chunk: int = 128, **task_params) -> Dict[str, Any]:
    model.eval()
    batch = fixed_eval_set(task, n, L, seed, **task_params)
    preds, losses = [], []
    for s in range(0, n, chunk):
        logits = model(batch.tokens[s:s + chunk])
        losses.append(float(lm_loss(logits, batch.targets[s:s + chunk])))
        preds.append(logits.argmax(-1))
    model.train()
    out = accuracy_by_distance(torch.cat(preds), batch)
    out["loss"] = sum(losses) / len(losses)
    out["L"] = L
    out["task_version"] = TASK_VERSION
    if task == "mqar":
        out["shortcut_baseline_acc"] = shortcut_baseline(batch, task_params.get("n_pairs", 8))
    return out


def train(cfg: TrainConfig, progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    model = TinyTransformer(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    gen = torch.Generator().manual_seed(cfg.seed)
    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    nonfinite = 0
    for step in range(cfg.steps):
        for gparam in opt.param_groups:
            gparam["lr"] = _lr_at(step, cfg)
        batch = make_batch(cfg.task, cfg.batch_size, cfg.L_train, gen, **cfg.task_params)
        loss = lm_loss(model(batch.tokens), batch.targets)
        if not torch.isfinite(loss):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            ev = evaluate(model, cfg.task, cfg.L_train, min(cfg.eval_size, 256), cfg.seed, **cfg.task_params)
            rec = {"step": step + 1, "train_loss": float(loss.detach()), "eval_loss": ev["loss"], "eval_acc": ev["accuracy"],
                   "elapsed_s": time.perf_counter() - t0}
            history.append(rec)
            if progress:
                progress(rec)
    return {
        "model": model,
        "history": history,
        "train_seconds": time.perf_counter() - t0,
        "nonfinite_steps": nonfinite,
        "n_parameters": model.num_parameters(),
    }


def config_to_dict(cfg: TrainConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d["model"] = cfg.model.to_dict()
    return d


def run_task_experiment(
    name: str,
    base: TrainConfig,
    methods: List[Dict[str, Any]],
    eval_settings: List[Dict[str, Any]],
    out_dir: str,
    progress: Optional[Callable[[str], None]] = None,
    swap_eval: bool = True,
) -> str:
    """Train one model per method; evaluate each on every eval setting. Append JSONL records.

    ``eval_settings`` items: {"L": int, "task_params": {...}, "tag": str}.
    If ``swap_eval``, the model trained with the first (reference) method is
    also evaluated with every other backend swapped in at inference time
    (no retraining) — this isolates "drop-in replacement" degradation.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.jsonl")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "task_version": TASK_VERSION, "base_config": config_to_dict(base), "methods": methods,
                   "eval_settings": eval_settings}, f, indent=2)
    ref_model = None
    with open(path, "a", encoding="utf-8") as f:
        for m in methods:
            cfg = TrainConfig(**{**asdict(base), "model": ModelConfig(**{**base.model.to_dict(),
                                                                          "attention": m["name"],
                                                                          "attention_params": m.get("params", {})})})
            label = m["name"] + ("" if not m.get("params") else json.dumps(m["params"], sort_keys=True))
            if progress:
                progress(f"training {label} ...")
            try:
                res = train(cfg, progress=(lambda r: progress(f"  {label} step {r['step']} loss {r['train_loss']:.3f} "
                                                              f"eval_acc {r['eval_acc']:.3f}")) if progress else None)
            except Exception as exc:
                f.write(json.dumps({"experiment": name, "label": label, "status": "error",
                                    "reason": f"{type(exc).__name__}: {exc}"}) + "\n")
                f.flush()
                continue
            model = res["model"]
            ckpt_dir = os.path.join(CHECKPOINT_DIR, os.path.basename(os.path.dirname(out_dir)), name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model_config": cfg.model.to_dict()},
                       os.path.join(ckpt_dir, f"{m['name']}.pt"))
            if ref_model is None:
                ref_model = model
            for es in eval_settings:
                tp = {**base.task_params, **es.get("task_params", {})}
                ev = evaluate(model, base.task, es["L"], base.eval_size, base.seed, **tp)
                rec = {"experiment": name, "mode": "trained_with", "label": label, "method": m["name"],
                       "params": m.get("params", {}), "eval_tag": es.get("tag", ""), "eval_task_params": tp,
                       "train_L": base.L_train, "train_steps": base.steps, "seed": base.seed,
                       "train_seconds": res["train_seconds"], "n_parameters": res["n_parameters"],
                       "nonfinite_steps": res["nonfinite_steps"], "final_train_loss": res["history"][-1]["train_loss"],
                       "history": res["history"], "status": "ok", **ev}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if progress:
                    progress(f"  eval {es.get('tag', '')} L={es['L']}: acc {ev['accuracy']:.3f}")
        if swap_eval and ref_model is not None:
            ref_label = methods[0]["name"]
            for m in methods:
                ref_model.set_attention(m["name"], m.get("params", {}))
                for es in eval_settings:
                    tp = {**base.task_params, **es.get("task_params", {})}
                    ev = evaluate(ref_model, base.task, es["L"], base.eval_size, base.seed, **tp)
                    rec = {"experiment": name, "mode": f"swapped_into_{ref_label}_model",
                           "label": m["name"] + ("" if not m.get("params") else json.dumps(m["params"], sort_keys=True)),
                           "method": m["name"], "params": m.get("params", {}), "eval_tag": es.get("tag", ""),
                           "eval_task_params": tp, "train_L": base.L_train, "seed": base.seed, "status": "ok", **ev}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            ref_model.set_attention(methods[0]["name"], methods[0].get("params", {}))
    return path
