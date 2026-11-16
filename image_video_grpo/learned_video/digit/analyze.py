"""Paired analysis for the digit-conditioned study, plus per-digit collapse check."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ..analyze import boot
from ..metrics import pixel_diversity
from ..viz import save_grid
from .common import RUNS

KEYS = ["success", "correct_digit", "direction_ok", "legible", "eval_conf", "train_reward"]


def within_prompt_diversity(clips, cond):
    key = cond[:, 0] * 10 + cond[:, 1]
    return pixel_diversity(clips, key)


def main():
    ev = {p.stem: torch.load(p, weights_only=False) for p in sorted((RUNS / "eval").glob("*.pt"))}
    base = ev["pretrained"]["metrics"]
    grpo = [k for k in ev if k.startswith("grpo_")]
    T = {}
    for n, e in ev.items():
        m = e["metrics"]
        g = e["cond"][:, 1]
        row = {k: float(m[k].float().mean()) for k in KEYS}
        row["per_digit_success"] = [float(m["success"][g == d].mean()) for d in range(10)]
        row["min_digit_success"] = min(row["per_digit_success"])
        row["evaluator_digit_counts"] = torch.bincount(m["digit"], minlength=10).tolist()
        row["within_prompt_pixel_diversity"] = within_prompt_diversity(e["clips"], e["cond"])
        if n != "pretrained":
            row["paired_vs_pretrained"] = {k: boot(m[k].float() - base[k].float()) for k in KEYS}
        T[n] = row
    if grpo:
        pooled = {k: torch.stack([ev[g]["metrics"][k].float() for g in grpo]).mean(0) - base[k].float() for k in KEYS}
        T["grpo_pooled"] = {"seeds": grpo, "paired_vs_pretrained": {k: boot(v) for k, v in pooled.items()},
                            "per_seed_success_delta": [T[g]["paired_vs_pretrained"]["success"][0] for g in grpo]}
    (RUNS / "analysis.json").write_text(json.dumps(T, indent=2))

    names = [n for n in ["cfg2", "best4", *grpo, "grpo_pooled"] if n in T]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, k, lab in zip(axs, ["success", "correct_digit", "direction_ok"],
                          ["Success (requested digit AND direction)", "Requested digit, legible (evaluator)", "Direction correct (phase corr.)"]):
        v = [T[n]["paired_vs_pretrained"][k] for n in names]
        ax.errorbar(range(len(names)), [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], fmt="o", capsize=3)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_title(lab, fontsize=9); ax.set_ylabel("change vs pretrained (95% CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(RUNS / "paired_effects.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    for n in [k for k in ["pretrained", "best4", *grpo] if k in T]:
        ax.plot(range(10), T[n]["per_digit_success"], marker="o", label=n)
    ax.set_xticks(range(10)); ax.set_xlabel("requested digit"); ax.set_ylabel("success"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RUNS / "per_digit_success.png", dpi=130); plt.close(fig)

    if grpo:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for g in grpo:
            rows = [json.loads(l) for l in open(RUNS / g / "train.jsonl")]
            for ax, k in zip(axs, ["reward_mean", "kl_ref", "clip_frac"]):
                v = torch.tensor([r[k] for r in rows])
                if k == "reward_mean":
                    v = torch.nn.functional.conv1d(v[None, None], torch.ones(1, 1, 10) / 10)[0, 0]
                ax.plot(v.tolist(), label=g); ax.set_title(k, fontsize=9)
        axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(RUNS / "training_curves.png", dpi=130); plt.close(fig)
        idx = torch.arange(0, 80, 9)[:8]
        show = ["pretrained", grpo[0]]
        save_grid(torch.cat([ev[n]["clips"][idx] for n in show]), torch.cat([ev[n]["cond"][idx, 0] for n in show]),
                  RUNS / "comparison_grid.png",
                  rows_label=[f"{n} d{int(ev[n]['cond'][i,1])}" for n in show for i in idx])
    for n, r in T.items():
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in KEYS + ["min_digit_success", "within_prompt_pixel_diversity"]},
              r.get("paired_vs_pretrained", {}).get("success"))


if __name__ == "__main__":
    main()
