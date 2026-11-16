"""Paired analysis: full-UNet GRPO vs pretrained on 256 identical-noise samples, independent gold classifier."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from rldiff.rewards import frechet_distance
from ..learned_video.analyze import boot
from .grpo import PARENT, RUNS, classifiers, DEV, FROG

KEYS = ["gold_accuracy", "gold_probability", "reward"]


def main():
    ev = {p.stem: torch.load(p, weights_only=False) for p in sorted((RUNS / "eval").glob("*.pt"))}
    base = ev["pretrained"]["metrics"]; grpo = [k for k in ev if k.startswith("grpo_")]
    _, gold = classifiers()
    real = datasets.CIFAR10(str(PARENT / "data"), train=False, download=False,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1)]))
    idx = [i for i, l in enumerate(real.targets) if l == FROG][:1000]
    with torch.no_grad():
        rf = torch.cat([gold.features(torch.stack([real[j][0] for j in idx[q:q + 100]]).to(DEV)).cpu() for q in range(0, 1000, 100)])
    T = {}
    for n, e in ev.items():
        m = e["metrics"]
        row = {k: float(m[k].mean()) for k in KEYS}
        row["frechet_vs_real_frogs"] = float(frechet_distance(m["gold_features"], rf))
        row["pixel_diversity"] = float(torch.pdist(e["images"].float().flatten(1)).mean())
        if n != "pretrained":
            row["paired_vs_pretrained"] = {k: boot(m[k] - base[k]) for k in KEYS}
        T[n] = row
    if grpo:
        pooled = {k: torch.stack([ev[g]["metrics"][k] for g in grpo]).mean(0) - base[k] for k in KEYS}
        T["grpo_pooled"] = {"paired_vs_pretrained": {k: boot(v) for k, v in pooled.items()}}
    (RUNS / "analysis.json").write_text(json.dumps(T, indent=2))
    names = [n for n in ["best4", *grpo, "grpo_pooled"] if n in T]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, k, lab in zip(axs, ["gold_accuracy", "gold_probability"], ["Independent classifier: frog accuracy", "Independent classifier: frog probability"]):
        v = [T[n]["paired_vs_pretrained"][k] for n in names]
        ax.errorbar(range(len(names)), [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], fmt="o", capsize=3)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_title(lab, fontsize=9); ax.set_ylabel("change vs pretrained (95% CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(RUNS / "paired_effects.png", dpi=130); plt.close(fig)
    show = ["pretrained"] + grpo
    fig, axs = plt.subplots(len(show), 12, figsize=(12, len(show)))
    for r, n in enumerate(show):
        for c in range(12):
            axs[r, c].imshow(((ev[n]["images"][c].float() + 1) / 2).permute(1, 2, 0).clamp(0, 1)); axs[r, c].axis("off")
        axs[r, 0].set_title(n, fontsize=7, loc="left")
    fig.tight_layout(pad=0.2); fig.savefig(RUNS / "sample_grid.png", dpi=120); plt.close(fig)
    for n, r in T.items():
        print(n, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, r.get("paired_vs_pretrained", {}).get("gold_accuracy"))


if __name__ == "__main__":
    main()
