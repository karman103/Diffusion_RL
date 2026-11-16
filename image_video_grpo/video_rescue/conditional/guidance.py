"""Check whether velocity signal is recoverable by conditioning-scale diagnostics."""
from __future__ import annotations

import json

import torch
from torchvision.utils import save_image

from image_video_grpo.experiment import classifier_pair
from image_video_grpo.video_rescue.metrics import video_metrics
from image_video_grpo.video_rescue.conditional.model import ROOT, load_pretrained, velocity_for_seed
from image_video_grpo.video_rescue.conditional.sampler import ConditionalVideoPolicy

CFG = json.loads((ROOT / "GUIDANCE_PROTOCOL.json").read_text())


def main():
    if (ROOT / "GUIDANCE_DONE.json").exists():
        raise FileExistsError("Guidance diagnostic already exists")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS required")
    model = load_pretrained(device)
    proxy, gold = classifier_pair("video", "cpu")
    result = {}
    for scale in CFG["scales"]:
        samples, velocities = [], []
        for seed in CFG["development_seeds"]:
            velocity = velocity_for_seed(seed, device)
            policy = ConditionalVideoPolicy(model, velocity * scale, device)
            video, _ = policy.rollout(seed)
            samples.append(video.cpu())
            velocities.append(velocity.cpu().flatten().tolist())
        videos = torch.cat(samples)
        rows = video_metrics(videos, proxy, gold, torch.tensor(velocities))
        passed = [bool(row["finite"] and 0.03 <= row["ink_mass"] <= 0.40
                       and row["gold_confidence"] >= 0.7
                       and row["gold_class_consistency"] >= 0.75
                       and row["centroid_motion_px"] >= 0.7
                       and row["velocity_direction_cosine"] >= 0.5) for row in rows]
        key = str(scale)
        torch.save(videos, ROOT / f"guidance_{key}_raw.pt")
        save_image((videos.clamp(-1, 1).reshape(-1, 1, 28, 28) + 1) / 2,
                   ROOT / f"guidance_{key}_grid.png", nrow=4, padding=2)
        result[key] = {"passed": passed, "passed_count": sum(passed), "rows": rows,
                       "velocities": velocities}
        print(json.dumps({"scale": scale, "passed_count": sum(passed)}), flush=True)
    (ROOT / "GUIDANCE_DONE.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
