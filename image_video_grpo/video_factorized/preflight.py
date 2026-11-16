"""Check whether factorized diffusion-plus-motion generates valid toy videos."""
from __future__ import annotations

import hashlib
import json

import torch
from torchvision.utils import save_image

from image_video_grpo.experiment import classifier_pair
from image_video_grpo.video_rescue.metrics import video_metrics
from image_video_grpo.video_factorized.model import CFG, ROOT, AdaptedImage, FactorizedPolicy


def main():
    if (ROOT / "BASELINE_DONE.json").exists() or (ROOT / "baseline_videos.pt").exists():
        raise FileExistsError("Factorized baseline already exists")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS required")
    model = AdaptedImage().to(device).eval()
    policy = FactorizedPolicy(model, device)
    videos, canonical, velocities = [], [], []
    for seed in CFG["baseline_seeds"]:
        clip, image, velocity, _ = policy.rollout(seed)
        videos.append(clip.cpu())
        canonical.append(image.cpu())
        velocities.append(velocity.cpu())
    videos = torch.cat(videos)
    canonical = torch.cat(canonical)
    velocities = torch.cat(velocities)
    proxy, gold = classifier_pair("video", "cpu")
    rows = video_metrics(videos, proxy, gold, velocities)
    passed = [bool(row["finite"] and 0.03 <= row["ink_mass"] <= 0.35
                   and row["gold_confidence"] >= 0.7
                   and row["gold_class_consistency"] >= 0.75
                   and row["centroid_motion_px"] >= 0.7
                   and row["velocity_direction_cosine"] >= 0.5) for row in rows]
    torch.save(videos, ROOT / "baseline_videos.pt")
    torch.save(canonical, ROOT / "baseline_canonical.pt")
    save_image((videos.clamp(-1, 1).reshape(-1, 1, 28, 28) + 1) / 2,
               ROOT / "baseline_grid.png", nrow=4, padding=2)
    record = {"seeds": CFG["baseline_seeds"], "rows": rows,
              "velocities": velocities.tolist(), "passed": passed,
              "passed_count": sum(passed), "automatic_gate_passed": sum(passed) >= 6,
              "video_sha256": hashlib.sha256((ROOT / "baseline_videos.pt").read_bytes()).hexdigest(),
              "canonical_sha256": hashlib.sha256((ROOT / "baseline_canonical.pt").read_bytes()).hexdigest(),
              "visual_review_required": True}
    (ROOT / "BASELINE_DONE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": passed, "passed_count": sum(passed)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
