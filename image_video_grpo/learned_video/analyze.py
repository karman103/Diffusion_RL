"""Paired analysis of all evaluated policies against the pretrained model (same noise per clip)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import metrics
from .grpo import RUNS, PROT
from .viz import save_grid

KEYS = ["success", "direction_ok", "legible", "eval_conf", "train_reward"]


def boot(x, n=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, x.shape[0], (n, x.shape[0]), generator=g)
    m = x[idx].mean(1)
    return float(x.mean()), float(m.quantile(0.025)), float(m.quantile(0.975))


def main():
    ev = {p.stem: torch.load(p, weights_only=False) for p in sorted((RUNS / "eval").glob("*.pt"))}
    base = ev["pretrained"]["metrics"]
    grpo = [k for k in ev if k.startswith("grpo_")]
    table = {}
    for name, e in ev.items():
        m = e["metrics"]
        row = {k: float(m[k].float().mean()) for k in KEYS}
        row["median_angle_err"] = float(m["angle_err"].median())
        row["class_entropy"] = metrics.class_entropy(m["digit"])
        row["pixel_diversity"] = metrics.pixel_diversity(e["clips"], e["direction"])
        if name != "pretrained":
            row["paired_vs_pretrained"] = {k: boot(m[k].float() - base[k].float()) for k in KEYS}
        table[name] = row
    if grpo:
        pooled = {k: torch.stack([ev[g]["metrics"][k].float() for g in grpo]).mean(0) - base[k].float() for k in KEYS}
        table["grpo_pooled"] = {"seeds": grpo, "paired_vs_pretrained": {k: boot(v) for k, v in pooled.items()},
                                "per_seed_success_delta": [table[g]["paired_vs_pretrained"]["success"][0] for g in grpo]}
    # robustness: does the gain survive outside the class GRPO drifts toward? (post hoc, labelled as such)
    for name, e in ev.items():
        m = e["metrics"]
        table[name]["digit_counts"] = torch.bincount(m["digit"], minlength=10).tolist()
        keep = (m["digit"] != 8) & (base["digit"] != 8)
        table[name]["post_hoc_success_excluding_8"] = float(m["success"][keep].mean())
        if name != "pretrained":
            table[name]["post_hoc_paired_success_excluding_8"] = boot(m["success"][keep] - base["success"][keep])
    (RUNS / "analysis.json").write_text(json.dumps(table, indent=2))

    # figure 1: success / legibility with paired CIs
    names = [n for n in ["cfg2", "best4", *grpo, "grpo_pooled"] if n in table]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, k, lab in zip(axs, ["success", "direction_ok", "legible"],
                          ["Success (legible AND right direction)", "Direction correct (phase corr.)", "Legible (evaluator CNN)"]):
        vals = [table[n]["paired_vs_pretrained"][k] for n in names]
        ax.errorbar(range(len(names)), [v[0] for v in vals],
                    yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="o", capsize=3)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_title(lab, fontsize=9); ax.set_ylabel("change vs pretrained (95% CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(RUNS / "paired_effects.png", dpi=130); plt.close(fig)

    # figure 2: training curves
    if grpo:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for g in grpo:
            rows = [json.loads(l) for l in open(RUNS / g / "train.jsonl")]
            u = [r["update"] for r in rows]
            for ax, k in zip(axs, ["reward_mean", "kl_ref", "clip_frac"]):
                v = torch.tensor([r[k] for r in rows])
                if k == "reward_mean":
                    v = torch.nn.functional.conv1d(v[None, None], torch.ones(1, 1, 10) / 10)[0, 0]
                    u2 = u[9:]
                else:
                    u2 = u
                ax.plot(u2, v.tolist(), label=g); ax.set_title(k + (" (10-update avg)" if k == "reward_mean" else ""), fontsize=9)
        axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(RUNS / "training_curves.png", dpi=130); plt.close(fig)

    # figure 3: same noise, same direction, pretrained vs cfg vs grpo
    show = ["pretrained", "cfg2"] + grpo[:1]
    idx = list(range(8))
    clips = torch.cat([ev[n]["clips"][idx] for n in show])
    dirs = torch.cat([ev[n]["direction"][idx] for n in show])
    save_grid(clips, dirs, RUNS / "comparison_grid.png", rows_label=[f"{n}" for n in show for _ in idx])
    print(json.dumps({n: {k: round(v, 3) for k, v in r.items() if isinstance(v, float)} for n, r in table.items()}, indent=1))


if __name__ == "__main__":
    main()
