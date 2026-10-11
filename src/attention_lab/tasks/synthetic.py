"""Token-level synthetic tasks for training small models.

Vocabulary layout (shared by all tasks, size 64):
    0                 PAD (unused by generators)
    1 .. 8            NOISE tokens (filler / distractors)
    9 .. 24           KEY tokens   (16)
    25 .. 40          VALUE tokens (16)
    41 .. 63          unused

Targets use -100 for "no prediction here" (ignored by the loss and metrics).

Task A — local copy (``local_copy``)
    x_i uniform over VALUE tokens; target y_i = x_{i - lag} for i >= lag.
    Solvable by any mechanism that can see ``lag`` tokens back.

Generator version 2 (``TASK_VERSION``). Version 1 drew MQAR values *with
replacement*, which created a shortcut: predicting the most frequent value in
the context scored 25.7 % (8 pairs) — exactly the plateau every non-local
model reached in the first training runs, while none had learned key-value
binding. Version 2 draws values without replacement, so the best
context-only ("bag of values") guess is 1/n_pairs. ``shortcut_baseline``
computes that baseline for any eval set.

Task B — multi-query associative recall (``mqar``), in the style of Zoology
    (Arora et al., 2023): the sequence starts with ``n_pairs`` key-value pairs
    "k v k v ...", followed by filler NOISE tokens in which ``n_queries`` of
    the keys reappear at random positions. The target at a query position is
    the value bound to that key; everything else is -100. The *distance*
    (query position - key position) of every query is recorded.

Task B' — variable-offset induction (``induction``): a random segment of
    ``seg`` tokens (from the KEY+VALUE range), a gap of ``min_gap..max_gap``
    filler tokens, then the segment again. Inside the repeat, the target at each
    position is the next segment token, i.e. "find where the current token
    occurred before, and copy its successor". The offset varies per sequence,
    so the copy source must be found by content. (With a fixed offset, a RoPE
    head can solve it positionally; see docs/mqar_diagnosis.md D2.)
    ``gap_tokens="segment"`` fills the gap with random segment-vocabulary tokens,
    which act as distractors.

Task C — interference: MQAR with more pairs than queries (unqueried pairs
    are distractor associations) and a configurable fraction of filler
    positions replaced by *decoy keys* that are never bound in this sequence
    (``decoy_frac``). Accuracy is measured as interference increases.

Task D — length generalization: train MQAR at ``L_train``; evaluate the same
    generator at longer L (same number of pairs, larger gaps => larger
    distances than ever seen in training).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

TASK_VERSION = 2
VOCAB_SIZE = 64
NOISE = (1, 9)
KEYS = (9, 25)
VALUES = (25, 41)


@dataclass
class Batch:
    tokens: torch.Tensor      # [B, L] int64
    targets: torch.Tensor     # [B, L] int64, -100 = ignore
    distance: torch.Tensor    # [B, L] int64, -1 where no target (or not defined)
    source: torch.Tensor | None = None   # [B, L] position of the token holding the answer's key, -1 if none


def local_copy(batch_size: int, L: int, lag: int = 3, generator: torch.Generator | None = None) -> Batch:
    if not 0 < lag < L:
        raise ValueError("need 0 < lag < L")
    x = torch.randint(VALUES[0], VALUES[1], (batch_size, L), generator=generator)
    y = torch.full_like(x, -100)
    y[:, lag:] = x[:, :-lag]
    dist = torch.full_like(x, -1)
    dist[:, lag:] = lag
    src = torch.full_like(x, -1)
    src[:, lag:] = torch.arange(L - lag)
    return Batch(x, y, dist, src)


def mqar(batch_size: int, L: int, n_pairs: int = 8, n_queries: int = 8, decoy_frac: float = 0.0,
         generator: torch.Generator | None = None) -> Batch:
    n_keys = KEYS[1] - KEYS[0]
    if n_pairs > n_keys or n_pairs > VALUES[1] - VALUES[0]:
        raise ValueError(f"n_pairs must be <= {min(n_keys, VALUES[1] - VALUES[0])}")
    if n_queries > n_pairs:
        raise ValueError("n_queries must be <= n_pairs")
    if L < 2 * n_pairs + n_queries:
        raise ValueError(f"L={L} too short for {n_pairs} pairs + {n_queries} queries")
    g = generator
    tokens = torch.randint(NOISE[0], NOISE[1], (batch_size, L), generator=g)
    targets = torch.full((batch_size, L), -100, dtype=torch.long)
    dist = torch.full((batch_size, L), -1, dtype=torch.long)
    src = torch.full((batch_size, L), -1, dtype=torch.long)
    region = L - 2 * n_pairs
    for b in range(batch_size):
        keys = torch.randperm(n_keys, generator=g)[:n_pairs] + KEYS[0]
        vals = torch.randperm(VALUES[1] - VALUES[0], generator=g)[:n_pairs] + VALUES[0]   # distinct (v2)
        tokens[b, 0:2 * n_pairs:2] = keys
        tokens[b, 1:2 * n_pairs:2] = vals
        if decoy_frac > 0:
            unbound = torch.tensor([t for t in range(KEYS[0], KEYS[1]) if t not in set(keys.tolist())])
            if len(unbound):
                n_dec = int(round(decoy_frac * region))
                pos = torch.randperm(region, generator=g)[:n_dec] + 2 * n_pairs
                tokens[b, pos] = unbound[torch.randint(len(unbound), (n_dec,), generator=g)]
        which = torch.randperm(n_pairs, generator=g)[:n_queries]
        qpos = torch.randperm(region, generator=g)[:n_queries] + 2 * n_pairs
        tokens[b, qpos] = keys[which]
        targets[b, qpos] = vals[which]
        dist[b, qpos] = qpos - 2 * which
        src[b, qpos] = 2 * which
    return Batch(tokens, targets, dist, src)


SEG_VOCAB = (KEYS[0], VALUES[1])       # 32 symbols used by induction segments


def induction(batch_size: int, L: int, seg: int = 12, min_gap: int = 0, max_gap: int = 8,
              gap_tokens: str = "noise", generator: torch.Generator | None = None) -> Batch:
    if seg < 2 or min_gap < 0 or max_gap < min_gap:
        raise ValueError("need seg >= 2 and 0 <= min_gap <= max_gap")
    if 2 * seg + max_gap > L:
        raise ValueError(f"L={L} too short for seg={seg}, max_gap={max_gap}")
    if gap_tokens not in ("noise", "segment"):
        raise ValueError("gap_tokens must be 'noise' or 'segment'")
    g = generator
    tok = torch.randint(NOISE[0], NOISE[1], (batch_size, L), generator=g)
    tgt = torch.full((batch_size, L), -100, dtype=torch.long)
    dist = torch.full((batch_size, L), -1, dtype=torch.long)
    src = torch.full((batch_size, L), -1, dtype=torch.long)
    for i in range(batch_size):
        x = torch.randint(SEG_VOCAB[0], SEG_VOCAB[1], (seg,), generator=g)
        gap = int(torch.randint(min_gap, max_gap + 1, (1,), generator=g))
        s2 = seg + gap
        tok[i, :seg] = x
        if gap_tokens == "segment" and gap:
            tok[i, seg:s2] = torch.randint(SEG_VOCAB[0], SEG_VOCAB[1], (gap,), generator=g)
        tok[i, s2:s2 + seg] = x
        tgt[i, s2:s2 + seg - 1] = x[1:]
        src[i, s2:s2 + seg - 1] = torch.arange(seg - 1)       # earlier occurrence of the current token
        dist[i, s2:s2 + seg - 1] = s2                          # query position - source position
    return Batch(tok, tgt, dist, src)


TASKS = {"local_copy": local_copy, "mqar": mqar, "induction": induction}


def make_batch(task: str, batch_size: int, L: int, generator: torch.Generator, **params) -> Batch:
    try:
        fn = TASKS[task]
    except KeyError:
        raise KeyError(f"unknown task {task!r}; available {sorted(TASKS)}") from None
    return fn(batch_size, L, generator=generator, **params)


def fixed_eval_set(task: str, n: int, L: int, seed: int, **params) -> Batch:
    """Deterministic evaluation set: identical for every method compared."""
    return make_batch(task, n, L, torch.Generator().manual_seed(10_000 + seed), **params)


def accuracy_by_distance(pred: torch.Tensor, batch: Batch, edges: Tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256, 512, 1 << 30)
                         ) -> Dict[str, float]:
    """Overall accuracy over target positions plus accuracy per distance bucket [lo, hi)."""
    m = batch.targets != -100
    hit = (pred == batch.targets) & m
    out = {"accuracy": float(hit.sum() / m.sum().clamp(min=1)), "n_targets": int(m.sum())}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = m & (batch.distance >= lo) & (batch.distance < hi)
        if sel.any():
            key = f"acc_dist_{lo}_{hi if hi < (1 << 30) else 'inf'}"
            out[key] = float((hit & sel).sum() / sel.sum())
    if batch.source is not None:
        # Accuracy by position of the bound key (diagnoses "only early pairs recalled" shortcuts).
        for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16), (16, 1 << 30)):
            sel = m & (batch.source >= lo) & (batch.source < hi)
            if sel.any():
                out[f"acc_src_{lo}_{hi if hi < (1 << 30) else 'inf'}"] = float((hit & sel).sum() / sel.sum())
    return out


def shortcut_baseline(batch: Batch, n_pairs: int) -> float:
    """Accuracy of the best binding-free guess: predict the most frequent value in the pair region."""
    hits = tot = 0
    for i in range(batch.tokens.shape[0]):
        vals = batch.tokens[i, 1:2 * n_pairs:2]
        mode = torch.mode(vals).values
        m = batch.targets[i] != -100
        hits += int((batch.targets[i][m] == mode).sum())
        tot += int(m.sum())
    return hits / max(tot, 1)
