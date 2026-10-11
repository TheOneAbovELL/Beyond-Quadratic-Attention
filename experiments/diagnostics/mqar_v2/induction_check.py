"""Sanity check: can the 2-layer model learn plain induction (second half repeats the first)?"""
import sys, time, torch
sys.path.insert(0, r"C:\Users\omjee\OneDrive\Desktop\PROJECTS\Beyond Quadratic Attention\src")
from attention_lab.models import ModelConfig, TinyTransformer, lm_loss
torch.set_num_threads(1); torch.manual_seed(0)
lr = float(sys.argv[1]); V, half = 32, 16
m = TinyTransformer(ModelConfig(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_mlp=256))
opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
g = torch.Generator().manual_seed(0); t = time.time()
for step in range(1, 2001):
    x = torch.randint(0, V, (32, half), generator=g); tok = torch.cat([x, x], 1)
    tgt = torch.full_like(tok, -100); tgt[:, half:-1] = tok[:, half + 1:]          # predict next token in the repeat
    loss = lm_loss(m(tok), tgt); opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        with torch.no_grad():
            acc = ((m(tok).argmax(-1) == tgt) & (tgt != -100)).sum() / (tgt != -100).sum()
        print(f"induction lr={lr} step {step} loss {loss.item():.3f} acc {acc:.3f} t={time.time()-t:.0f}s", flush=True)
