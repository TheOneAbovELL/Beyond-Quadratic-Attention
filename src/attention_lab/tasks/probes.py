"""Attention-level retrieval probes (no training): does a method route information from one key?

Construction (one trial = one batch element, one head)
-----------------------------------------------------
* L keys: random unit directions with norm r, where r^2 = beta * sqrt(d), so a
  query equal to a key direction (norm r) gives the *target* logit
  q.k/sqrt(d) = beta, while an unrelated key gives logits ~ N(0, beta^2 / d).
* The query sits at the last position (L-1). Its own key is replaced by an
  unrelated random key, and the query vector is copied from the key at the
  target position t = L-1-distance (the "stored association").
* Values: i.i.d. N(0, I) vectors. The method's output at the last position is
  decoded by nearest value (max cosine over all L values); a trial succeeds if
  that is the target's value.
* Decoys (Task C, interference): ``n_decoys`` other keys are set to have cosine
  ``decoy_cos`` with the query (misleading near-matches, logit ~ beta*decoy_cos).

Every method receives *identical* q/k/v per trial. Causal attention is used,
so the query can only see positions <= L-1 (all of them).

Interpretation: exact softmax retrieval is limited only by "dilution" (sum
of exp(other logits) vs exp(beta)); restricted methods fail deterministically
when the target lies outside their pattern; linear attention's kernel lacks
the exponential sharpness of softmax.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from ..algorithms.registry import get_method


def make_probe(L: int, distance: int, n_trials: int, d: int = 64, beta: float = 10.0, n_decoys: int = 0,
               decoy_cos: float = 0.9, seed: int = 0):
    if not 0 < distance <= L - 1:
        raise ValueError(f"distance must be in [1, L-1]; got {distance} for L={L}")
    g = torch.Generator().manual_seed(seed)
    r = math.sqrt(beta * math.sqrt(d))
    k = torch.randn(n_trials, 1, L, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True) * r
    v = torch.randn(n_trials, 1, L, d, generator=g)
    q = torch.randn(n_trials, 1, L, d, generator=g)
    q = q / q.norm(dim=-1, keepdim=True) * r
    target = L - 1 - distance
    q[:, :, L - 1] = k[:, :, target]
    if n_decoys:
        qhat = q[:, 0, L - 1] / r                                               # [T, d]
        candidates = [i for i in range(L - 1) if i != target]
        perm = torch.randperm(len(candidates), generator=g)[:n_decoys]
        for idx in perm.tolist():
            pos = candidates[idx]
            noise = torch.randn(n_trials, d, generator=g)
            noise = noise - (noise * qhat).sum(-1, keepdim=True) * qhat          # orthogonal to q
            noise = noise / noise.norm(dim=-1, keepdim=True)
            k[:, 0, pos] = r * (decoy_cos * qhat + math.sqrt(1 - decoy_cos**2) * noise)
    return q, k, v, target


@torch.inference_mode()
def run_probe(method: str, L: int, distance: int, n_trials: int = 64, d: int = 64, beta: float = 10.0,
              n_decoys: int = 0, decoy_cos: float = 0.9, params: Optional[Dict[str, Any]] = None,
              seed: int = 0, batch: int = 16) -> Dict[str, Any]:
    q, k, v, target = make_probe(L, distance, n_trials, d, beta, n_decoys, decoy_cos, seed)
    fn = get_method(method)
    params = dict(params or {})
    correct = 0
    cos_target = []
    for s in range(0, n_trials, batch):
        sl = slice(s, s + batch)
        out = fn(q[sl], k[sl], v[sl], causal=True, **params)[:, 0, L - 1]       # [b, d]
        vv = v[sl, 0]                                                           # [b, L, d]
        cos = torch.nn.functional.cosine_similarity(out[:, None, :], vv, dim=-1)  # [b, L]
        correct += int((cos.argmax(-1) == target).sum())
        cos_target.append(cos[:, target])
    ct = torch.cat(cos_target)
    return {
        "method": method, "params": params, "L": L, "distance": distance, "n_trials": n_trials, "d": d,
        "beta": beta, "n_decoys": n_decoys, "decoy_cos": decoy_cos, "seed": seed,
        "accuracy": correct / n_trials, "mean_cos_to_target_value": float(ct.mean()),
    }


DEFAULT_PROBE_METHODS: List[Dict[str, Any]] = [
    {"name": "sdpa_flash_cpu"},                                  # exact softmax (fast exact kernel)
    {"name": "local_window", "params": {"window": 64}},
    {"name": "block_sparse", "params": {"block_size": 64, "local_blocks": 1, "global_blocks": 1, "random_blocks": 1}},
    {"name": "linear_elu"},
]


def probe_suite(methods=None, quick: bool = True, progress=None) -> List[Dict[str, Any]]:
    """Task A/B (distance sweep), length sweep (dilution) and Task C (decoys)."""
    methods = methods or DEFAULT_PROBE_METHODS
    n_trials = 32 if quick else 128
    recs = []

    def go(m, experiment, **kw):
        rec = run_probe(m["name"], params=m.get("params"), n_trials=n_trials, **kw)
        rec["label"] = m["name"] + ("" if not m.get("params") else str(m["params"]))
        rec["experiment"] = experiment
        recs.append(rec)
        if progress:
            progress(rec)

    L_fixed = 1024
    distances = [1, 4, 16, 63, 64, 65, 128, 256, 512, 1023]
    lengths = [128, 256, 512, 1024, 2048] if quick else [128, 256, 512, 1024, 2048, 4096]
    for m in methods:
        for dist in distances:
            go(m, "distance_sweep", L=L_fixed, distance=dist)
        for L in lengths:
            go(m, "length_sweep_near", L=L, distance=8)
            go(m, "length_sweep_far", L=L, distance=L - 1)
        for nd in [0, 1, 2, 4, 8, 16]:
            go(m, "decoys", L=512, distance=8, n_decoys=nd)
        # Fairness check: does any input scale make the method's kernel sharp enough to retrieve?
        for L in (16, 64):
            for beta in (1.0, 3.0, 10.0, 30.0, 100.0):
                go(m, f"sharpness_L{L}", L=L, distance=4, beta=beta)
    return recs
