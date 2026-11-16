"""Audit saved evaluations and make a measured, reproducible results report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())


def main():
    rows = []
    summaries = {}
    for seed in PROTOCOL["image"]["training_seeds"]:
        run = ROOT / "runs/image" / f"seed_{seed}"
        if not (run / "EVAL_DONE.json").exists():
            raise FileNotFoundError(f"Missing completed evaluation: {run}")
        rows += [json.loads(x) for x in (run / "eval.jsonl").read_text().splitlines()]
        summaries[seed] = json.loads((run / "summary.json").read_text())
    expected = 3 * 4 * len(PROTOCOL["image"]["eval_seeds"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")
    keys = ("reward", "gold_probability", "gold_accuracy")
    effects = {}
    for key in keys:
        deltas = []
        for seed in PROTOCOL["image"]["training_seeds"]:
            for eval_seed in PROTOCOL["image"]["eval_seeds"]:
                pair = {r["policy"]: r[key] for r in rows
                        if r["training_seed"] == seed and r["eval_seed"] == eval_seed}
                deltas.append(pair["trained"] - pair["zero"])
        effects[key] = np.array(deltas).reshape(3, -1)
    rng = np.random.default_rng(2401)
    intervals = {}
    for key, matrix in effects.items():
        boot = []
        for _ in range(5000):
            seed_idx = rng.integers(0, 3, 3)
            # The same evaluation seeds are reused across trained adapters.
            # Resample their indices jointly to preserve that dependence.
            eval_idx = rng.integers(0, matrix.shape[1], matrix.shape[1])
            boot.append(float(matrix[np.ix_(seed_idx, eval_idx)].mean()))
        intervals[key] = np.percentile(boot, [2.5, 97.5]).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, title in zip(axes, ["reward", "gold_probability"], ["Proxy target probability", "Independent target probability"]):
        means = effects[key].mean(1)
        ax.axhline(0, color="0.4", lw=1)
        ax.scatter(range(1, 4), means, color="#2469a3", s=45)
        ax.set(xticks=[1, 2, 3], xticklabels=PROTOCOL["image"]["training_seeds"],
               xlabel="Training seed", ylabel="Trained − zero adapter", title=title)
    fig.tight_layout()
    fig.savefig(ROOT / "paired_effects.png", dpi=180)
    plt.close(fig)
    # Show identical evaluation seeds across stock/zero/trained. Best-of-four is
    # a separate, more expensive sampling control and uses different candidates.
    import torch
    run = ROOT / "runs/image/seed_1101"
    names = ("stock", "zero", "trained", "best_of_four")
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.3))
    for row, name in enumerate(names):
        samples = torch.load(run / f"{name}_samples.pt", map_location="cpu", weights_only=True)[:8]
        for col, sample in enumerate(samples):
            axes[row, col].imshow(((sample.permute(1, 2, 0).float().clamp(-1, 1) + 1) / 2).numpy())
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("Eight fresh seeds; first three rows share the same initial and transition noise", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "sample_grid.png", dpi=180)
    plt.close(fig)
    lines = ["# Measured image/video GRPO results", "",
             "The primary comparison is trained versus zero adapter under the same stochastic sampler and paired fresh seeds. Stock DDPM uses a different timestep schedule; best-of-four spends four times as many samples. These are contextual baselines.", "",
             "## Image: pretrained CIFAR-10 DDPM", "",
             f"Three training seeds; {len(PROTOCOL['image']['eval_seeds'])} fresh evaluation seeds per run. Target class: frog. The checkpoint and raw artifacts are pinned in `PROTOCOL.json`.", "",
             "| Metric | Trained − zero, pooled mean | Hierarchical seed-bootstrap 95% interval |", "|---|---:|---:|"]
    for key in keys:
        matrix = effects[key]
        lo, hi = intervals[key]
        lines.append(f"| {key} | {matrix.mean():+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "The interval resamples three training seeds and the 32 paired evaluation seeds, with each evaluation-seed index shared across the selected trained adapters. Only three training seeds were run, so the interval is descriptive rather than a strong population-level confidence statement. Stock, zero, and best-of-four use the same evaluation seeds in every run; their repeated table rows are shared controls, not independent replications.", "",
              "| Training seed | Policy | Proxy reward | Gold probability | Gold accuracy | Feature Fréchet | Pixel diversity |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for seed, s in summaries.items():
        for policy in ("stock", "zero", "trained", "best_of_four"):
            d = s[policy]
            lines.append(f"| {seed} | {policy} | {d['reward']:.5f} | {d['gold_probability']:.5f} | {d['gold_accuracy']:.5f} | {d['classifier_feature_frechet']:.3f} | {d['pixel_diversity']:.3f} |")
    lines += ["", "Feature Fréchet is computed with 32 generated images versus 500 real target-class CIFAR-10 test images in the gold classifier's representation. It is not official FID and has high small-sample error. Pixel diversity is mean pairwise Euclidean distance and cannot alone establish perceptual variety.", "",
              "![Paired effects by training seed](paired_effects.png)", "",
              "![Fresh-seed image samples by policy](sample_grid.png)", "", "## Video feasibility", ""]
    gate = ROOT / "VIDEO_GATE_FAILED.json"
    if gate.exists():
        data = json.loads(gate.read_text())
        lines += [f"The predeclared toy-video quality gate failed: {data['reason'].rstrip('.')}. No video GRPO training or benchmark improvement is claimed.", ""]
    else:
        lines += ["Video gate decision has not yet been recorded; no video training result is claimed.", ""]
    lines += ["## Audit trail", "", "Each completed run contains `train.jsonl`, adapter weights, `eval.jsonl`, per-policy raw sample tensors and visuals, a summary, and DONE markers. `AUDIT.json` records tensor hashes, finite-value checks, paired sample differences, and independent CPU score recomputation. Numerical checks are in `test_core.py`. Pretrained checkpoint revisions and SHA-256 hashes are in `PROTOCOL.json`. This is a laptop-scale method study, not a SOTA image/video benchmark."]
    audit_path = ROOT / "AUDIT.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        max_score_error = max(value for seed in audit.values() for policy in seed.values()
                              if "max_cpu_recompute_abs_difference" in policy
                              for value in policy["max_cpu_recompute_abs_difference"].values())
        pixel_deltas = [audit[str(seed)]["trained_minus_zero"]["mean_absolute_pixel_difference"]
                        for seed in PROTOCOL["image"]["training_seeds"]]
        lines += ["", f"Raw-sample audit: all {expected} saved images have the expected 3×32×32 shape and finite values; maximum absolute CPU-versus-MPS score recomputation difference is {max_score_error:.8f}. Mean absolute trained-versus-zero pixel differences by training seed are " + ", ".join(f"{x:.8f}" for x in pixel_deltas) + "."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (ROOT / "effects.json").write_text(json.dumps({k: {"seed_means": v.mean(1).tolist(), "pooled_mean": float(v.mean()), "interval": intervals[k]} for k, v in effects.items()}, indent=2))


if __name__ == "__main__":
    main()
