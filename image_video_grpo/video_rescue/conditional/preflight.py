"""Prospectively fixed content/motion gate for conditional toy-video DDPM."""
from __future__ import annotations

import hashlib
import json

import torch
from torchvision.utils import save_image

from image_video_grpo.experiment import classifier_pair
from image_video_grpo.video_rescue.metrics import video_metrics
from image_video_grpo.video_rescue.conditional.model import CFG, ROOT, load_pretrained, velocity_for_seed
from image_video_grpo.video_rescue.conditional.sampler import ConditionalVideoPolicy


def main():
    if (ROOT / "BASELINE_DONE.json").exists() or (ROOT / "baseline_raw.pt").exists():
        raise FileExistsError("Conditional-video baseline already exists")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS required")
    model = load_pretrained(device)
    samples, velocities = [], []
    try:
        for seed in CFG["baseline_seeds"]:
            velocity = velocity_for_seed(seed, device)
            policy = ConditionalVideoPolicy(model, velocity, device)
            video, _ = policy.rollout(seed)
            samples.append(video.cpu())
            velocities.append(velocity.cpu().flatten().tolist())
            print(json.dumps({"seed": seed, "velocity": velocities[-1]}), flush=True)
        videos = torch.cat(samples)
        proxy, gold = classifier_pair("video", "cpu")
        rows = video_metrics(videos, proxy, gold, torch.tensor(velocities))
        passed = [bool(row["finite"] and 0.03 <= row["ink_mass"] <= 0.40
                       and row["gold_confidence"] >= 0.7
                       and row["gold_class_consistency"] >= 0.75
                       and row["centroid_motion_px"] >= 0.7
                       and row["velocity_direction_cosine"] >= 0.5) for row in rows]
        torch.save(videos, ROOT / "baseline_raw.pt")
        save_image((videos.clamp(-1, 1).reshape(-1, 1, 28, 28) + 1) / 2,
                   ROOT / "baseline_grid.png", nrow=4, padding=2)
        record = {"seeds": CFG["baseline_seeds"], "velocities": velocities,
                  "rows": rows, "passed": passed, "passed_count": sum(passed),
                  "automatic_gate_passed": sum(passed) >= 6,
                  "raw_sha256": hashlib.sha256((ROOT / "baseline_raw.pt").read_bytes()).hexdigest(),
                  "visual_review_required": True}
        (ROOT / "BASELINE_DONE.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"passed": passed, "passed_count": sum(passed)}, indent=2), flush=True)
    except BaseException as exc:
        (ROOT / "BASELINE_FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
