"""Minimal Transformer: forward, training step, evaluation and backend switching."""
import pytest
import torch

from attention_lab.algorithms import EXACT_METHODS
from attention_lab.models import ModelConfig, TinyTransformer, apply_rope, lm_loss, rope_cache
from attention_lab.tasks import make_batch
from attention_lab.training import TrainConfig, evaluate, train


def _model(attention="sdpa_flash_cpu", positional="rope", **kw):
    torch.manual_seed(0)
    return TinyTransformer(ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_mlp=64,
                                       attention=attention, positional=positional, **kw))


@pytest.mark.parametrize("positional", ["rope", "sinusoidal", "none"])
def test_forward_shapes(positional):
    m = _model(positional=positional)
    tok = torch.randint(0, 64, (3, 20))
    assert m(tok).shape == (3, 20, 64)


def test_exact_backends_preserve_model_outputs():
    m = _model().double()
    tok = torch.randint(0, 64, (2, 37), generator=torch.Generator().manual_seed(1))
    ref = m(tok)
    for name in EXACT_METHODS:
        params = {"q_tile": 8, "kv_tile": 16} if name.startswith("tiled") else {}
        m.set_attention(name, params)
        torch.testing.assert_close(m(tok), ref, rtol=1e-10, atol=1e-10)


def test_exact_backends_give_same_gradients_float32():
    m = _model()
    tok = torch.randint(0, 64, (2, 33), generator=torch.Generator().manual_seed(2))
    tgt = torch.randint(0, 64, (2, 33), generator=torch.Generator().manual_seed(3))
    grads = {}
    for name in ("naive", "sdpa_flash_cpu", "tiled_recompute"):
        m.set_attention(name, {"q_tile": 8, "kv_tile": 8} if name.startswith("tiled") else {})
        m.zero_grad()
        lm_loss(m(tok), tgt).backward()
        grads[name] = torch.cat([p.grad.flatten() for p in m.parameters()])
    for name in ("sdpa_flash_cpu", "tiled_recompute"):
        rel = (grads[name] - grads["naive"]).norm() / grads["naive"].norm()
        assert rel < 1e-4, (name, float(rel))


def test_approximate_backend_changes_outputs_by_a_measurable_amount():
    m = _model()
    tok = torch.randint(0, 64, (2, 64), generator=torch.Generator().manual_seed(4))
    ref = m(tok)
    m.set_attention("local_window", {"window": 4})
    diff = (m(tok) - ref).abs().max()
    assert diff > 1e-6                     # not equivalent ...
    assert torch.isfinite(diff)            # ... but well-defined; quantified in experiments, not assumed


def test_causality_future_tokens_do_not_affect_past_logits():
    for name, params in [("sdpa_flash_cpu", {}), ("tiled", {"q_tile": 4, "kv_tile": 4}),
                         ("local_window", {"window": 3}), ("linear_elu", {"chunk": 4}),
                         ("block_sparse", {"block_size": 4, "local_blocks": 1, "global_blocks": 1})]:
        m = _model(attention=name, attention_params=params)
        tok = torch.randint(0, 64, (1, 24), generator=torch.Generator().manual_seed(5))
        tok2 = tok.clone()
        tok2[0, 15:] = (tok2[0, 15:] + 7) % 64
        torch.testing.assert_close(m(tok)[:, :15], m(tok2)[:, :15], msg=name)


def test_rope_scores_depend_only_on_relative_position():
    d = 8
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 1, 1, d, generator=g).expand(1, 1, 20, d)
    k = torch.randn(1, 1, 1, d, generator=g).expand(1, 1, 20, d)
    cos, sin = rope_cache(20, d)
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (qr @ kr.transpose(-2, -1))[0, 0]
    torch.testing.assert_close(s[5, 2], s[15, 12])        # same offset 3
    torch.testing.assert_close(s[7, 7], s[0, 0])


def test_training_step_reduces_loss_and_evaluation_runs():
    cfg = TrainConfig(task="local_copy", task_params={"lag": 2}, L_train=32, steps=60, batch_size=32, lr=3e-3,
                      warmup_steps=10, eval_every=60, eval_size=64, threads=2,
                      model=ModelConfig(vocab_size=64, d_model=32, n_heads=2, n_layers=1, d_mlp=64))
    res = train(cfg)
    first = lm_loss(TinyTransformer(cfg.model)(make_batch("local_copy", 32, 32, torch.Generator().manual_seed(0), lag=2).tokens),
                    make_batch("local_copy", 32, 32, torch.Generator().manual_seed(0), lag=2).targets)
    assert res["history"][-1]["train_loss"] < float(first.detach())
    ev = evaluate(res["model"], "local_copy", 32, 64, seed=0, lag=2)
    assert 0.0 <= ev["accuracy"] <= 1.0 and ev["n_targets"] == 64 * 30
