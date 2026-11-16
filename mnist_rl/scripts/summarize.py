"""Final-epoch summary table (markdown) for all runs -> runs/summary.md"""
import json, os
import numpy as np

ROWS = [
    ("Pretrained base", ["p_ddpo"], "first"),
    ("DDPO (PPO-clip), r = p(7)", ["p_ddpo", "p_ddpo_s1", "p_ddpo_s2"], "last"),
    ("DDPO, 3 passes/batch (75 ep)", ["p_ddpo_reuse3"], "last"),
    ("DDPO + KL β=3e-4", ["p_kl_0.0003"], "last"),
    ("DDPO + KL β=1e-3", ["p_kl_0.001"], "last"),
    ("DDPO + KL β=0.1", ["p_kl_0.1"], "last"),
    ("DDPO 3 passes + KL β=3e-4 (75 ep)", ["p_reuse3_kl"], "last"),
    ("RWR", ["p_rwr"], "last"),
    ("DRaFT-1", ["p_draft", "p_draft_s1", "p_draft_s2"], "last"),
    ("DDPO, r = log p(7)  (100 ep)", ["e1_ddpo"], "last"),
    ("DRaFT-1, r = log p(7)  (100 ep)", ["e1_draft"], "last"),
    ("Oracle: DDPO on gold p(7)", ["p_oracle"], "last"),
    ("Oracle: DRaFT on gold p(7) †", ["p_draft_oracle"], "last"),
    ("Random-reward control (100 ep)", ["e0_random"], "last"),
]
KEYS = [("proxy_p", "proxy p(7) ↑", 3), ("gold_acc", "gold acc ↑", 3), ("fd_target", "FD→real 7s ↓", 2),
        ("diversity", "diversity ↑", 2), ("kl", "path-KL (nats)", 0)]


def last(run, which):
    f = f"runs/{run}/metrics.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    return rows[0] if which == "first" else rows[-1]


out = ["| Method | " + " | ".join(k[1] for k in KEYS) + " | n |", "|---|" + "---|" * (len(KEYS) + 1)]
for name, runs, which in ROWS:
    ms = [m for m in (last(r, which) for r in runs) if m is not None]
    if not ms:
        continue
    cells = []
    for k, _, dp in KEYS:
        v = np.array([m[k] for m in ms])
        cells.append(f"{v.mean():.{dp}f}" + (f" ± {v.std():.{dp}f}" if len(v) > 1 else ""))
    out.append(f"| {name} | " + " | ".join(cells) + f" | {len(ms)} |")
bon = json.load(open("runs/base/best_of_n.json"))
for r in bon:
    if r["N"] in (16, 64):
        out.append(f"| Best-of-{r['N']} (inference only) | {r['proxy_p']:.3f} | {r['gold_acc']:.3f} | – | – | {r['kl']:.1f} (exact, marginal) | – |")
open("runs/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
