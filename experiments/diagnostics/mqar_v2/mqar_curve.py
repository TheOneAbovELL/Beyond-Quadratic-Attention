"""Exploration: how many steps does the exact backend need to learn MQAR? Prints per-key-position accuracy."""
import sys, time, torch
sys.path.insert(0, r"C:\Users\omjee\OneDrive\Desktop\PROJECTS\Beyond Quadratic Attention\src")
from attention_lab.models import ModelConfig
from attention_lab.tasks.synthetic import VOCAB_SIZE
from attention_lab.training import TrainConfig, train, evaluate
lr, steps, d_model, n_pairs, L = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
tp = {"n_pairs": n_pairs, "n_queries": n_pairs}
cfg = TrainConfig(task="mqar", task_params=tp, L_train=L, steps=steps, batch_size=32, lr=lr, warmup_steps=100,
                  eval_every=250, eval_size=256, threads=1,
                  model=ModelConfig(vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=4, n_layers=2, d_mlp=4 * d_model))
t = time.time()
tag = f"lr={lr} d={d_model} pairs={n_pairs} L={L}"
def prog(r):
    print(f"{tag} step {r['step']} loss {r['train_loss']:.3f} acc {r['eval_acc']:.3f} t={time.time()-t:.0f}s", flush=True)
    if r["step"] % 1000 == 0 or r["step"] == steps:
        pass
res = train(cfg, progress=prog)
ev = evaluate(res["model"], "mqar", L, 512, 0, **tp)
print(tag, "FINAL", {k: round(v, 3) for k, v in ev.items() if k.startswith("acc")}, flush=True)
