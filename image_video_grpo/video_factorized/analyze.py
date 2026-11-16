"""Audit saved factorized video tensors, recompute scores, and plot paired effects."""
from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from image_video_grpo.experiment import classifier_pair
from image_video_grpo.video_rescue.metrics import reward_from_metrics, video_metrics
from image_video_grpo.video_factorized.model import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(delta: np.ndarray, seed: int = 20261115):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, size=len(delta), replace=True).mean()
                      for _ in range(5000)])
    return [float(np.quantile(means, q)) for q in (0.025, 0.975)]


def main():
    proxy, gold = classifier_pair("video", "cpu")
    studies = {
        "initial": json.loads((ROOT / "PROTOCOL.json").read_text())["grpo_if_gate_passes"],
        "higher_step_size": json.loads((ROOT / "FOLLOWUP_PROTOCOL.json").read_text()),
    }
    results = {}
    audit = {"protocol_sha256": {name: sha(ROOT / filename) for name, filename in
                                 (("initial", "PROTOCOL.json"),
                                  ("higher_step_size", "FOLLOWUP_PROTOCOL.json"))},
             "studies": {}}
    for study_name, cfg in studies.items():
        if any(not (ROOT / "runs" / f"seed_{seed}" / "EVAL_DONE.json").exists()
               for seed in cfg["training_seeds"]):
            raise RuntimeError(f"Incomplete evaluation for {study_name}")
        study_audit, study_results = {}, {}
        for seed in cfg["training_seeds"]:
            run = ROOT / "runs" / f"seed_{seed}"
            expected = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
            if len(expected) != len(cfg["evaluation_seeds"]) * 3:
                raise AssertionError("Evaluation row count mismatch")
            details = {}
            scores = {}
            for policy in ("zero", "trained", "best_of_four"):
                path = run / f"{policy}_videos.pt"
                video = torch.load(path, map_location="cpu", weights_only=True)
                velocity = torch.load(run / f"{policy}_velocities.pt",
                                      map_location="cpu", weights_only=True)
                assert video.shape == (24, 4, 28, 28)
                assert velocity.shape == (24, 2)
                assert torch.isfinite(video).all()
                rows = video_metrics(video, proxy, gold, velocity)
                reward = reward_from_metrics(video_metrics(video, proxy, proxy, velocity))
                saved = [row for row in expected if row["policy"] == policy]
                assert [row["eval_seed"] for row in saved] == cfg["evaluation_seeds"]
                max_error = max(abs(float(saved[i][key]) - float(value))
                                for i, row in enumerate(rows) for key, value in row.items()
                                if isinstance(value, (int, float)) and value is not None
                                and key not in ("finite", "gold_mode_digit"))
                max_error = max(max_error, max(abs(float(saved[i]["reward"]) - float(reward[i]))
                                               for i in range(len(saved))))
                if max_error > 0.001:
                    raise AssertionError(f"CPU rescore mismatch {max_error}")
                details[policy] = {"video_sha256": sha(path), "shape": list(video.shape),
                                   "finite": True, "max_saved_vs_rescored_error": max_error,
                                   "mean_absolute_pixel_change_from_zero": None}
                scores[policy] = saved
            zero = torch.load(run / "zero_videos.pt", map_location="cpu", weights_only=True)
            trained = torch.load(run / "trained_videos.pt", map_location="cpu", weights_only=True)
            details["trained"]["mean_absolute_pixel_change_from_zero"] = float(
                (trained - zero).abs().mean())
            study_audit[str(seed)] = details
            metrics = ("reward", "gold_confidence", "gold_class_consistency",
                       "centroid_motion_px", "centroid_acceleration_px", "ink_mass",
                       "proxy_confidence", "proxy_class_consistency")
            effects = {}
            for metric in metrics:
                z = np.array([row[metric] for row in scores["zero"]], dtype=float)
                t = np.array([row[metric] for row in scores["trained"]], dtype=float)
                b = np.array([row[metric] for row in scores["best_of_four"]], dtype=float)
                d = t - z
                effects[metric] = {"zero_mean": float(z.mean()), "trained_mean": float(t.mean()),
                                   "trained_minus_zero": float(d.mean()),
                                   "paired_bootstrap_95pct": bootstrap(d, seed + len(metric)),
                                   "best_of_four_mean": float(b.mean())}
            study_results[str(seed)] = effects
        audit["studies"][study_name] = study_audit
        results[study_name] = study_results
    (ROOT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    (ROOT / "ANALYSIS.json").write_text(json.dumps(results, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), layout="constrained")
    for ax, metric, title in zip(axes, ("reward", "gold_confidence", "gold_class_consistency"),
                                 ("Proxy video reward", "Independent digit confidence",
                                  "Independent class consistency")):
        labels, vals, lows, highs = [], [], [], []
        for study, seeds in results.items():
            for seed, metrics in seeds.items():
                row = metrics[metric]
                labels.append(f"{study[:4]} {seed}")
                vals.append(row["trained_minus_zero"])
                lows.append(row["paired_bootstrap_95pct"][0])
                highs.append(row["paired_bootstrap_95pct"][1])
        x = np.arange(len(vals))
        ax.errorbar(x, vals, yerr=[np.array(vals)-lows, np.array(highs)-vals],
                    fmt="o", capsize=3, color="#2b6cb0")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Trained − zero")
    fig.savefig(ROOT / "paired_effects.png", dpi=180)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
