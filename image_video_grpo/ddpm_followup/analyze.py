"""Audit raw paired samples and summarize completed exploratory DDPM runs."""
from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from image_video_grpo.ddpm_followup.experiment import CFG, ROOT, classifier_pair
from image_video_grpo.rewards import image_scores


def main() -> None:
    torch.set_num_threads(2)
    proxy, gold = classifier_pair("image", "cpu")
    keys = ("reward", "gold_probability", "gold_accuracy")
    effects, summaries, audits = {key: [] for key in keys}, {}, {}
    for seed in CFG["training_seeds"]:
        run = ROOT / "runs" / f"seed_{seed}"
        if not (run / "DONE.json").exists() or not (run / "EVAL_DONE.json").exists():
            raise RuntimeError(f"Incomplete training/evaluation seed {seed}")
        rows = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
        if len(rows) != 3 * len(CFG["eval_seeds"]):
            raise AssertionError(f"Wrong row count for {seed}")
        by_policy, audit = {}, {}
        for name in ("zero", "trained", "best_of_four"):
            path = run / f"{name}_samples.pt"
            samples = torch.load(path, map_location="cpu", weights_only=True)
            if tuple(samples.shape) != (len(CFG["eval_seeds"]), 3, 32, 32):
                raise AssertionError(f"Unexpected sample shape: {path}")
            if not torch.isfinite(samples).all():
                raise AssertionError(f"Nonfinite sample: {path}")
            subset = [row for row in rows if row["policy"] == name]
            if [row["eval_seed"] for row in subset] != CFG["eval_seeds"]:
                raise AssertionError(f"Unpaired evaluation seeds for {seed}/{name}")
            recalculated = image_scores(samples, proxy, gold, CFG["target_class"])
            errors = {key: max(abs(float(recalculated[key][i]) - row[key])
                               for i, row in enumerate(subset)) for key in keys}
            if errors["gold_accuracy"] != 0 or max(errors.values()) > 0.005:
                raise AssertionError(f"Score recomputation failed for {seed}/{name}: {errors}")
            audit[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "shape": list(samples.shape), "finite": True,
                           "range": [float(samples.min()), float(samples.max())],
                           "max_cpu_recompute_abs_difference": errors}
            by_policy[name] = {"samples": samples, "rows": subset}
        pixel = (by_policy["trained"]["samples"] - by_policy["zero"]["samples"]).abs()
        audit["paired_pixel_difference"] = {"mean": float(pixel.mean()), "max": float(pixel.max())}
        audits[str(seed)] = audit
        summaries[str(seed)] = json.loads((run / "summary.json").read_text())
        for key in keys:
            delta = [t[key] - z[key] for t, z in zip(by_policy["trained"]["rows"],
                                                      by_policy["zero"]["rows"])]
            effects[key].append(delta)
    (ROOT / "AUDIT.json").write_text(json.dumps(audits, indent=2) + "\n")
    rng = np.random.default_rng(31114)
    result = {}
    for key, matrix in effects.items():
        x = np.array(matrix)
        bootstrap = []
        for _ in range(5000):
            seed_index = rng.integers(0, x.shape[0], x.shape[0])
            # Shared fresh seeds must be resampled jointly across trained runs.
            eval_index = rng.integers(0, x.shape[1], x.shape[1])
            bootstrap.append(float(x[np.ix_(seed_index, eval_index)].mean()))
        result[key] = {"pooled_mean": float(x.mean()), "seed_means": x.mean(1).tolist(),
                       "descriptive_95pct_interval": np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    (ROOT / "EFFECTS.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, title in zip(axes, ("reward", "gold_probability"),
                               ("Proxy frog probability", "Independent frog probability")):
        ax.axhline(0, color="0.4", lw=1)
        ax.scatter(range(1, 4), result[key]["seed_means"], color="#2469a3", s=50)
        ax.set(xticks=[1, 2, 3], xticklabels=CFG["training_seeds"],
               xlabel="Training seed", ylabel="Trained − zero", title=title)
    fig.tight_layout()
    fig.savefig(ROOT / "paired_effects.png", dpi=180)
    plt.close(fig)
    first = ROOT / "runs" / f"seed_{CFG['training_seeds'][0]}"
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for row, name in enumerate(("zero", "trained", "best_of_four")):
        samples = torch.load(first / f"{name}_samples.pt", map_location="cpu", weights_only=True)[:8]
        for col, sample in enumerate(samples):
            axes[row, col].imshow(((sample.permute(1, 2, 0).float().clamp(-1, 1) + 1) / 2).numpy())
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("Fresh seeds; zero and trained use identical sampling noise", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "sample_grid.png", dpi=180)
    plt.close(fig)
    lines = ["# Exploratory stock-DDPM GRPO follow-up", "",
             "This study was motivated by the earlier null image result. It uses the released checkpoint's 50-step DDPM sampler for both zero and trained adapters, with fresh paired evaluation seeds. Best-of-four samples from the same zero-adapter policy at four times the sampling cost.", "",
             "| Metric | Trained − zero mean | Descriptive hierarchical 95% bootstrap interval |",
             "|---|---:|---:|"]
    for key in keys:
        entry = result[key]
        lo, hi = entry["descriptive_95pct_interval"]
        lines.append(f"| {key} | {entry['pooled_mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "The interval resamples three training seeds and the same 32 paired evaluation-seed indices jointly across runs. With only three training seeds, it is descriptive. Repeated zero controls are shared samples, not independent replications.", "",
              "| Seed | Policy | Proxy reward | Independent probability | Independent accuracy | Feature Fréchet | Pixel diversity |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for seed, summary in summaries.items():
        for name in ("zero", "trained", "best_of_four"):
            d = summary[name]
            lines.append(f"| {seed} | {name} | {d['reward']:.5f} | {d['gold_probability']:.5f} | {d['gold_accuracy']:.5f} | {d['classifier_feature_frechet']:.3f} | {d['pixel_diversity']:.3f} |")
    lines += ["", "Feature Fréchet uses the independent classifier's representation on 32 samples versus 500 real CIFAR-10 frog images. It is a small-sample diagnostic, not official FID. Pixel diversity is mean pairwise Euclidean distance.", "",
              "![Paired effect by seed](paired_effects.png)", "",
              "![Fresh-seed samples](sample_grid.png)", "",
              "Every raw saved sample is finite, has shape 3×32×32, and was rescored on CPU; hashes and maximum discrepancies are in `AUDIT.json`. `EFFECTS.json` has the raw paired-effect summaries. Conditional DDPM transition log densities are exact for this policy; the clipped GRPO objective is a surrogate, and no final-image likelihood is claimed."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
