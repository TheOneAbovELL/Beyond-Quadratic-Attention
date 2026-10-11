"""The diagnostic instruments must measure what the model actually computes."""
import torch

from attention_lab.diagnostics import attention_maps, circuit_probes, prediction_categories
from attention_lab.models import ModelConfig, TinyTransformer
from attention_lab.tasks.synthetic import mqar


def test_replayed_attention_maps_reproduce_each_layers_attention_output():
    torch.manual_seed(0)
    model = TinyTransformer(ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_mlp=64)).eval()
    tok = mqar(3, 32, n_pairs=4, n_queries=4, generator=torch.Generator().manual_seed(1)).tokens
    maps = attention_maps(model, tok)
    x = model.embed(tok)
    B, L = tok.shape
    with torch.no_grad():
        for blk, A in zip(model.blocks, maps):
            a = blk.attn
            h = blk.ln1(x)
            v = a.qkv(h).view(B, L, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)[2]
            replay = a.proj((A @ v).transpose(1, 2).reshape(B, L, -1))
            torch.testing.assert_close(replay, a(h), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(A.sum(-1), torch.ones(A.shape[:-1]))
            x = blk(x)


def test_circuit_probes_read_the_intended_positions():
    """With hand-set attention maps, each probe must return the weight placed at its position."""
    b = mqar(2, 16, n_pairs=4, n_queries=2, generator=torch.Generator().manual_seed(0))
    import attention_lab.diagnostics.recall as R

    L = 16
    A = torch.zeros(2, 1, L, L)
    for i in range(L):
        A[:, :, i, max(i - 1, 0)] = 1.0                          # pure previous-token attention
    A2 = torch.zeros(2, 1, L, L)
    for bi in range(2):
        for q in (b.targets[bi] != -100).nonzero().flatten().tolist():
            A2[bi, 0, q, int(b.source[bi, q]) + 1] = 1.0          # pure correct-value retrieval
    orig = R.attention_maps
    R.attention_maps = lambda model, tokens: [A, A2]
    try:
        out = circuit_probes(None, b, 4)
    finally:
        R.attention_maps = orig
    assert out["prevtok_l0"] == 1.0 and out["target_l1"] == 1.0
    assert out["othervals_l1"] == 0.0 and out["key_l1"] == 0.0


def test_prediction_categories():
    b = mqar(4, 16, n_pairs=4, n_queries=2, generator=torch.Generator().manual_seed(0))
    logits = torch.full((4, 16, 64), -10.0)
    logits.scatter_(-1, b.targets.clamp(min=0).unsqueeze(-1), 10.0)
    assert prediction_categories(logits, b, 4)["pred_correct"] == 1.0
