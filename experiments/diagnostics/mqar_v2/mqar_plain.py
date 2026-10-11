"""Isolate loop vs task: MQAR v2 trained with the plain loop that learned induction (const lr, no clip/warmup)."""
import sys, time, torch
sys.path.insert(0, r"C:\Users\omjee\OneDrive\Desktop\PROJECTS\Beyond Quadratic Attention\src")
from attention_lab.models import ModelConfig, TinyTransformer, lm_loss
from attention_lab.tasks import make_batch, fixed_eval_set
torch.set_num_threads(1); torch.manual_seed(0)
lr, pairs, L, steps = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m = TinyTransformer(ModelConfig(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_mlp=256))
opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
g = torch.Generator().manual_seed(0); ev = fixed_eval_set("mqar", 256, L, 0, n_pairs=pairs, n_queries=pairs); t = time.time()
for step in range(1, steps + 1):
    b = make_batch("mqar", 32, L, g, n_pairs=pairs, n_queries=pairs)
    loss = lm_loss(m(b.tokens), b.targets); opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        with torch.no_grad():
            p = m(ev.tokens).argmax(-1); mk = ev.targets != -100
            acc = ((p == ev.targets) & mk).sum() / mk.sum()
        print(f"mqar-plain lr={lr} pairs={pairs} L={L} step {step} loss {loss.item():.3f} acc {acc:.3f} t={time.time()-t:.0f}s", flush=True)
