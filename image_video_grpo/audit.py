"""Recheck completed image evaluations from saved raw samples and row files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .experiment import ROOT, PROTOCOL, classifier_pair
from .rewards import image_scores


def main() -> None:
    torch.set_num_threads(2)
    proxy, gold = classifier_pair("image", "cpu")
    expected_seeds = PROTOCOL["image"]["eval_seeds"]
    checks = {}
    for seed in PROTOCOL["image"]["training_seeds"]:
        run = ROOT / "runs/image" / f"seed_{seed}"
        if not (run / "DONE.json").exists() or not (run / "EVAL_DONE.json").exists():
            raise RuntimeError(f"Incomplete run: {seed}")
        rows = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
        if len(rows) != 4 * len(expected_seeds):
            raise AssertionError(f"Unexpected evaluation row count for {seed}")
        samples_by_policy = {}
        result = {}
        for policy in ("stock", "zero", "trained", "best_of_four"):
            path = run / f"{policy}_samples.pt"
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if tuple(tensor.shape) != (len(expected_seeds), 3, 32, 32):
                raise AssertionError(f"Unexpected shape: {path}: {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"Nonfinite samples: {path}")
            matched = [row for row in rows if row["policy"] == policy]
            if [row["eval_seed"] for row in matched] != expected_seeds:
                raise AssertionError(f"Unpaired or reordered evaluation seeds: {seed}/{policy}")
            recalculated = image_scores(tensor, proxy, gold, PROTOCOL["image"]["target_class"])
            differences = {
                key: max(abs(float(value[i]) - row[key]) for i, row in enumerate(matched))
                for key, value in recalculated.items() if key != "gold_features"
            }
            if differences["gold_accuracy"] != 0 or max(differences.values()) > 0.005:
                raise AssertionError(f"Saved scores disagree with raw samples: {seed}/{policy}: {differences}")
            samples_by_policy[policy] = tensor
            result[policy] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "finite": True,
                "shape": list(tensor.shape),
                "range": [float(tensor.min()), float(tensor.max())],
                "max_cpu_recompute_abs_difference": differences,
            }
        paired = samples_by_policy["trained"] - samples_by_policy["zero"]
        result["trained_minus_zero"] = {
            "mean_absolute_pixel_difference": float(paired.abs().mean()),
            "max_absolute_pixel_difference": float(paired.abs().max()),
        }
        checks[str(seed)] = result
    (ROOT / "AUDIT.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({seed: checks[seed]["trained_minus_zero"] for seed in checks}, indent=2))


if __name__ == "__main__":
    main()
