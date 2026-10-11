"""Diagnostic harness for associative-recall training (MQAR plateau investigation).

Records, at every evaluation step:
  * training loss, held-out loss/accuracy, train-set accuracy (for fixed-dataset runs);
  * global and per-module gradient norms (before the optimizer step);
  * per-module relative update size ||delta theta|| / ||theta|| over the last step;
  * prediction categories at target positions: correct value / other value present in the
    context / value absent from the context / key token / anything else; mean confidence;
  * circuit probes (MQAR layout only), computed by replaying the model's own modules:
      prevtok_l{n}: mean over value positions of the max-over-heads attention weight on the
                    immediately preceding position (the "previous-token head" an induction
                    circuit needs in layer 1);
      target_l{n}:  mean over query positions of the max-over-heads attention weight on the
                    position holding the correct value (layer 2 retrieval);
      othervals_l{n}: same, on the other pair-value positions (summed);
      key_l{n}:     same, on the position of the matching key token.

All of this is observation only: the training step itself is a plain AdamW step identical
to ``training.train`` except for the explicitly listed loop options.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..models.transformer import ModelConfig, TinyTransformer, apply_rope, lm_loss, rope_cache
from ..tasks.synthetic import KEYS, VALUES, Batch

MODULE_GROUPS = ("embed", "blocks.0.attn", "blocks.0.mlp", "blocks.1.attn", "blocks.1.mlp", "head")


def _group(name: str) -> str:
    if name.startswith("embed"):
        return "embed"
    for g in MODULE_GROUPS[1:5]:
        if name.startswith(g):
            return g
    if name.startswith("blocks.0"):
        return "blocks.0.norm"
    if name.startswith("blocks.1"):
        return "blocks.1.norm"
    return "head"  # ln_f + head


def grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    sq: Dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            sq[_group(n)] = sq.get(_group(n), 0.0) + float(p.grad.pow(2).sum())
    out = {f"gn_{k}": math.sqrt(v) for k, v in sq.items()}
    out["gn_total"] = math.sqrt(sum(sq.values()))
    return out


def snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def update_ratios(model: torch.nn.Module, before: Dict[str, torch.Tensor]) -> Dict[str, float]:
    d2: Dict[str, float] = {}
    p2: Dict[str, float] = {}
    for n, p in model.named_parameters():
        g = _group(n)
        d2[g] = d2.get(g, 0.0) + float((p.detach() - before[n]).pow(2).sum())
        p2[g] = p2.get(g, 0.0) + float(before[n].pow(2).sum())
    return {f"upd_{g}": math.sqrt(d2[g]) / max(math.sqrt(p2[g]), 1e-12) for g in d2}


@torch.no_grad()
def attention_maps(model: TinyTransformer, tokens: torch.Tensor) -> List[torch.Tensor]:
    """Replay the forward pass and return per-layer softmax attention [B, H, L, L] (exact, causal)."""
    cfg = model.cfg
    x = model.embed(tokens)
    B, L = tokens.shape
    if cfg.positional == "sinusoidal":
        x = x + model.pe[:L]
    maps = []
    for blk in model.blocks:
        a = blk.attn
        h = blk.ln1(x)
        q, k, v = a.qkv(h).view(B, L, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
        if cfg.positional == "rope":
            cos, sin = rope_cache(L, a.dh, x.device, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = q @ k.transpose(-2, -1) * a.dh ** -0.5
        s = s.masked_fill(~torch.ones(L, L, dtype=torch.bool).tril(), float("-inf"))
        maps.append(torch.softmax(s, -1))
        x = blk(x)
    return maps


@torch.no_grad()
def circuit_probes(model: TinyTransformer, batch: Batch, n_pairs: int) -> Dict[str, float]:
    maps = attention_maps(model, batch.tokens)
    out: Dict[str, float] = {}
    vpos = torch.arange(1, 2 * n_pairs, 2)
    qmask = batch.targets != -100
    bi, qi = qmask.nonzero(as_tuple=True)
    src = batch.source[bi, qi]                       # key position; value is at src + 1
    for li, A in enumerate(maps):
        Am = A.max(dim=1).values                     # max over heads: [B, L, L]
        out[f"prevtok_l{li}"] = float(Am[:, vpos, vpos - 1].mean())
        tgt = Am[bi, qi, src + 1]
        keyw = Am[bi, qi, src]
        allv = Am[bi, qi][:, vpos].sum(-1)
        out[f"target_l{li}"] = float(tgt.mean())
        out[f"othervals_l{li}"] = float((allv - tgt).mean())
        out[f"key_l{li}"] = float(keyw.mean())
    return out


@torch.no_grad()
def prediction_categories(logits: torch.Tensor, batch: Batch, n_pairs: Optional[int]) -> Dict[str, float]:
    m = batch.targets != -100
    probs = torch.softmax(logits, -1)
    pred = logits.argmax(-1)
    out = {"conf_mean": float(probs.max(-1).values[m].mean()), "p_target_mean": float(
        probs.gather(-1, batch.targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)[m].mean())}
    if n_pairs is None:
        return out
    cats = {"correct": 0, "other_ctx_value": 0, "absent_value": 0, "key": 0, "other": 0}
    bi, qi = m.nonzero(as_tuple=True)
    for b, q in zip(bi.tolist(), qi.tolist()):
        p = int(pred[b, q])
        ctx_vals = set(batch.tokens[b, 1:2 * n_pairs:2].tolist())
        if p == int(batch.targets[b, q]):
            cats["correct"] += 1
        elif p in ctx_vals:
            cats["other_ctx_value"] += 1
        elif VALUES[0] <= p < VALUES[1]:
            cats["absent_value"] += 1
        elif KEYS[0] <= p < KEYS[1]:
            cats["key"] += 1
        else:
            cats["other"] += 1
    n = max(1, len(bi))
    out.update({f"pred_{k}": v / n for k, v in cats.items()})
    return out


@dataclass
class DiagConfig:
    name: str
    steps: int = 2000
    lr: float = 3e-3
    batch_size: int = 32
    eval_every: int = 100
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    grad_clip: Optional[float] = None
    warmup_steps: int = 0
    cosine: bool = False
    seed: int = 0
    n_pairs: Optional[int] = None          # set for MQAR-layout tasks (enables circuit probes)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(vocab_size=64, d_model=64, n_heads=4, n_layers=2,
                                                                    d_mlp=256))
    notes: str = ""


def run_diagnostic(cfg: DiagConfig, sample: Callable[[int, torch.Generator], Batch], heldout: Batch, log_path: str,
                   fixed_train: Optional[Batch] = None, extra_evals: Optional[Dict[str, Batch]] = None) -> Dict[str, Any]:
    """Train with per-step instrumentation. If ``fixed_train`` is given, every step uses that whole set."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    model = TinyTransformer(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    gen = torch.Generator().manual_seed(cfg.seed)
    t0 = time.perf_counter()
    rows = []
    with open(log_path, "w", encoding="utf-8") as f:
        meta = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items() if k != "model"}
        meta["model"] = cfg.model.to_dict()
        f.write(json.dumps({"meta": meta}) + "\n")
        for step in range(1, cfg.steps + 1):
            lr = cfg.lr
            if cfg.warmup_steps and step <= cfg.warmup_steps:
                lr = cfg.lr * step / cfg.warmup_steps
            elif cfg.cosine:
                prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
                lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            for g in opt.param_groups:
                g["lr"] = lr
            batch = fixed_train if fixed_train is not None else sample(cfg.batch_size, gen)
            loss = lm_loss(model(batch.tokens), batch.targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            record = step % cfg.eval_every == 0 or step == 1
            if record:
                gn = grad_norms(model)
                before = snapshot(model)
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if not record:
                continue
            row: Dict[str, Any] = {"step": step, "lr": lr, "train_loss": float(loss.detach()), **gn,
                                   **update_ratios(model, before), "elapsed_s": time.perf_counter() - t0}
            model.eval()
            with torch.no_grad():
                for tag, b in [("heldout", heldout)] + ([("train", fixed_train)] if fixed_train is not None else []) \
                        + list((extra_evals or {}).items()):
                    lg = model(b.tokens)
                    m = b.targets != -100
                    row[f"{tag}_loss"] = float(lm_loss(lg, b.targets))
                    row[f"{tag}_acc"] = float(((lg.argmax(-1) == b.targets) & m).sum() / m.sum())
                    if tag == "heldout":
                        row.update({f"heldout_{k}": v for k, v in prediction_categories(lg, b, cfg.n_pairs).items()})
                if cfg.n_pairs is not None and heldout.source is not None:
                    row.update(circuit_probes(model, heldout, cfg.n_pairs))
            model.train()
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
    return {"model": model, "rows": rows}
