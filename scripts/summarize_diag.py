"""Print selected columns of a diagnostic log at selected steps.

    python scripts/summarize_diag.py experiments/diagnostics/mqar_v2/<name>.jsonl [col ...]
"""
import json
import sys

DEFAULT = ["train_loss", "train_acc", "heldout_loss", "heldout_acc", "heldout_pred_correct",
           "heldout_pred_other_ctx_value", "gn_total", "gn_blocks.0.attn", "gn_blocks.1.attn",
           "upd_blocks.0.attn", "upd_blocks.1.attn", "prevtok_l0", "target_l1", "othervals_l1", "key_l1"]


def main():
    path = sys.argv[1]
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    meta, rows = rows[0]["meta"], rows[1:]
    cols = sys.argv[2:] or [c for c in DEFAULT if c in rows[0]]
    print(meta["name"], "|", meta.get("notes", ""))
    pick = [r for r in rows if r["step"] in (1, 100, 200, 500) or r["step"] % 250 == 0]
    print("step".rjust(5), *[c.replace("blocks.", "b").replace("heldout_", "ho_")[:14].rjust(14) for c in cols])
    for r in pick:
        print(str(r["step"]).rjust(5), *[(f"{r[c]:.4g}" if isinstance(r.get(c), (int, float)) else "-").rjust(14)
                                        for c in cols])


if __name__ == "__main__":
    main()
