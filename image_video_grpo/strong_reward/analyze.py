"""Apply the raw-sample audit to the stronger-reward experiment."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import image_video_grpo.ddpm_followup.analyze as base
from image_video_grpo.strong_reward.experiment import CFG, ROOT, classifier_pair

base.CFG = CFG
base.ROOT = ROOT
base.classifier_pair = classifier_pair


if __name__ == "__main__":
    base.main()
    classifiers = json.loads((ROOT / "classifier_results.json").read_text())
    lines = ["", "## Reward-model and update diagnostics", "",
             f"The proxy and disjoint evaluator reached {classifiers['proxy']['test_accuracy']:.2%} and {classifiers['gold']['test_accuracy']:.2%} CIFAR-10 test accuracy. Both were trained for the predeclared 16 epochs on separate 25,000-example subsets; no evaluator weights entered GRPO.", "",
             "| Seed | Early 50 on-policy reward | Late 50 on-policy reward | Final adapter delta norm | Max action log-ratio | Max clip fraction |",
             "|---:|---:|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for seed in CFG["training_seeds"]:
        run = ROOT / "runs" / f"seed_{seed}"
        records = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        if len(records) != CFG["updates"]:
            raise AssertionError(f"Unexpected training length: {seed}")
        rewards = np.array([np.mean(row["reward"]) for row in records])
        norm = np.array([row["adapter_delta_norm"] for row in records])
        ratio = max(d["max_abs_log_ratio"] for row in records for d in row["diagnostics"])
        clipping = max(row["clip_fraction"] for row in records)
        lines.append(f"| {seed} | {rewards[:50].mean():.4f} | {rewards[-50:].mean():.4f} | {norm[-1]:.4f} | {ratio:.4f} | {clipping:.3f} |")
        smoothed = np.convolve(rewards, np.ones(25)/25, mode="valid")
        axes[0].plot(np.arange(25, len(rewards)+1), smoothed, label=str(seed))
        axes[1].plot(np.arange(1, len(norm)+1), norm, label=str(seed))
    axes[0].set(xlabel="Update", ylabel="25-group moving mean", title="On-policy proxy reward (different noise each group)")
    axes[1].set(xlabel="Update", ylabel="Parameter L2 distance", title="Adapter movement from zero")
    axes[0].legend(title="Seed")
    fig.tight_layout()
    fig.savefig(ROOT / "training_diagnostics.png", dpi=180)
    plt.close(fig)
    lines += ["", "The on-policy curve is a noisy training diagnostic, not a paired evaluation. The paired fresh-seed effects and independent classifier above determine the outcome.", "",
              "![Training diagnostics](training_diagnostics.png)", ""]
    path = ROOT / "RESULTS.md"
    report = path.read_text().replace("# Exploratory stock-DDPM GRPO follow-up",
                                       "# Stronger-reward stock-DDPM GRPO follow-up")
    report = report.replace("This study was motivated by the earlier null image result.",
                            "This exploratory study was motivated by two earlier null image studies and fixed new classifier splits and fresh seeds before training.")
    path.write_text(report + "\n".join(lines))
