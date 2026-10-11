"""Targeted tests for each suspected MQAR failure mode (diagnosis of the v2 plateau).

Each test isolates one link of the pipeline: generator -> targets -> loss positions ->
causal visibility -> model gradients -> evaluation metric.
"""
import math

import pytest
import torch

from attention_lab.models import ModelConfig, TinyTransformer, lm_loss
from attention_lab.tasks.synthetic import KEYS, NOISE, VALUES, accuracy_by_distance, fixed_eval_set, mqar, shortcut_baseline


@pytest.fixture
def batch():
    return mqar(64, 32, n_pairs=4, n_queries=4, generator=torch.Generator().manual_seed(7))


# ---------------------------------------------------------------- generator
def test_generator_token_classes_by_region(batch):
    t = batch.tokens
    pr = t[:, :8]
    assert ((pr[:, 0::2] >= KEYS[0]) & (pr[:, 0::2] < KEYS[1])).all(), "even pair slots must be keys"
    assert ((pr[:, 1::2] >= VALUES[0]) & (pr[:, 1::2] < VALUES[1])).all(), "odd pair slots must be values"
    region, tgt = t[:, 8:], batch.targets[:, 8:]
    is_query = tgt != -100
    assert ((region[~is_query] >= NOISE[0]) & (region[~is_query] < NOISE[1])).all(), "non-query region is noise only"
    assert ((region[is_query] >= KEYS[0]) & (region[is_query] < KEYS[1])).all(), "query positions hold keys"


def test_generator_keys_and_values_distinct_and_each_query_key_bound_once(batch):
    for i in range(batch.tokens.shape[0]):
        keys = batch.tokens[i, 0:8:2].tolist()
        vals = batch.tokens[i, 1:8:2].tolist()
        assert len(set(keys)) == 4 and len(set(vals)) == 4
        for p in (batch.targets[i] != -100).nonzero().flatten().tolist():
            k = int(batch.tokens[i, p])
            assert keys.count(k) == 1
            assert int(batch.targets[i, p]) == vals[keys.index(k)]


def test_answer_is_not_derivable_without_binding(batch):
    """Each target value appears exactly once in the context, as the token right after its key."""
    for i in range(batch.tokens.shape[0]):
        for p in (batch.targets[i] != -100).nonzero().flatten().tolist():
            v = int(batch.targets[i, p])
            occ = (batch.tokens[i, :p + 1] == v).nonzero().flatten().tolist()
            assert len(occ) == 1 and int(batch.tokens[i, occ[0] - 1]) == int(batch.tokens[i, p])


def test_query_marginals_are_uniform_over_pairs_and_values():
    b = mqar(4000, 32, n_pairs=4, n_queries=1, generator=torch.Generator().manual_seed(0))
    which = (b.source[b.targets != -100] // 2)
    counts = torch.bincount(which, minlength=4).float() / len(which)
    assert (counts - 0.25).abs().max() < 0.03
    vals = b.targets[b.targets != -100] - VALUES[0]
    vcounts = torch.bincount(vals, minlength=16).float() / len(vals)
    assert (vcounts - 1 / 16).abs().max() < 0.02


# ---------------------------------------------------------------- loss positions / alignment
def test_loss_is_computed_only_at_query_positions_and_aligned_unshifted(batch):
    torch.manual_seed(0)
    logits = torch.randn(*batch.tokens.shape, 64)
    perfect = logits.clone()
    m = batch.targets != -100
    perfect[m] = -1e4
    perfect[m, batch.targets[m]] = 1e4
    assert lm_loss(perfect, batch.targets) < 1e-6           # loss ignores non-query positions entirely
    shifted = torch.full_like(batch.targets, -100)
    shifted[:, :-1] = batch.targets[:, 1:]
    assert lm_loss(perfect, shifted) > 1.0                  # alignment is position p -> targets[p], not p+1
    assert int(m.sum()) == 64 * 4


def test_uniform_over_context_values_gives_log_n_pairs_loss(batch):
    """The observed plateau (loss ~1.386 for 4 pairs) equals log(4): uniform over the 4 context values."""
    logits = torch.full((*batch.tokens.shape, 64), -1e4)
    for i in range(batch.tokens.shape[0]):
        logits[i, :, batch.tokens[i, 1:8:2]] = 0.0
    assert math.isclose(float(lm_loss(logits, batch.targets)), math.log(4), rel_tol=1e-4)


# ---------------------------------------------------------------- visibility / causality
def test_query_position_can_see_its_key_token_and_all_pairs(batch):
    """Causal mask includes the diagonal: changing the token AT the query position changes its logits."""
    torch.manual_seed(0)
    model = TinyTransformer(ModelConfig(vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_mlp=64)).eval()
    i = 0
    p = int((batch.targets[i] != -100).nonzero()[0])
    t1 = batch.tokens[i:i + 1].clone()
    t2 = t1.clone()
    t2[0, p] = KEYS[0] + (int(t1[0, p]) - KEYS[0] + 1) % 16
    with torch.no_grad():
        assert (model(t1)[0, p] - model(t2)[0, p]).abs().max() > 1e-6
        t3 = t1.clone()
        t3[0, 1] = VALUES[0] + (int(t1[0, 1]) - VALUES[0] + 1) % 16   # change a pair value (position 1 < p)
        assert (model(t1)[0, p] - model(t3)[0, p]).abs().max() > 1e-6
        t4 = t1.clone()
        if p + 1 < t1.shape[1]:
            t4[0, p + 1:] = NOISE[0]
            assert torch.equal(model(t1)[0, :p + 1], model(t4)[0, :p + 1])


# ---------------------------------------------------------------- gradients
@pytest.mark.parametrize("positional", ["rope", "sinusoidal"])
def test_every_parameter_receives_a_nonzero_gradient_on_mqar(batch, positional):
    torch.manual_seed(0)
    model = TinyTransformer(ModelConfig(vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_mlp=64, positional=positional))
    lm_loss(model(batch.tokens), batch.targets).backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    # Embedding rows for unused vocabulary (41..63) legitimately get zero gradient; check the matrix as a whole.
    assert dead == [], dead


# ---------------------------------------------------------------- metric
def test_metric_oracle_and_shortcut():
    ev = fixed_eval_set("mqar", 128, 32, 0, n_pairs=4, n_queries=4)
    assert accuracy_by_distance(ev.targets.clamp(min=0), ev)["accuracy"] == 1.0
    assert abs(shortcut_baseline(ev, 4) - 0.25) < 1e-9
