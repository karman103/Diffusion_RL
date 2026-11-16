"""Gate locally pretrained video model before GRPO."""
from __future__ import annotations

import hashlib
import json

import torch
from torchvision.utils import save_image

from image_video_grpo.experiment import classifier_pair
from image_video_grpo.video_rescue.metrics import video_metrics
from image_video_grpo.video_rescue.model import ROOT, VideoDenoiser
from image_video_grpo.video_rescue.sampler import VideoPolicy

CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


def main():
    if (ROOT / "BASELINE_DONE.json").exists() or (ROOT / "baseline_raw.pt").exists():
        raise FileExistsError("Video baseline already exists")
    if not (ROOT / "TEMPORAL_DONE.json").exists():
        raise FileNotFoundError("Temporal pretraining completion marker missing")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS required")
    model = VideoDenoiser()
    model.temporal.load_state_dict(torch.load(ROOT / "temporal.pt", map_location="cpu", weights_only=True))
    model.to(device).eval().requires_grad_(False)
    policy = VideoPolicy(model, device, CFG["sampling_steps"])
    samples = []
    try:
        for seed in CFG["feasibility_seeds"]:
            video, _ = policy.rollout(seed)
            samples.append(video.cpu())
            print(json.dumps({"seed": seed, "range": [float(video.min()), float(video.max())]}), flush=True)
        videos = torch.cat(samples)
        proxy, gold = classifier_pair("video", "cpu")
        rows = video_metrics(videos, proxy, gold)
        passed = [bool(row["finite"] and 0.03 <= row["ink_mass"] <= 0.35
                       and row["gold_confidence"] >= 0.7
                       and row["gold_class_consistency"] >= 0.75
                       and row["centroid_motion_px"] >= 0.5) for row in rows]
        torch.save(videos, ROOT / "baseline_raw.pt")
        save_image((videos.clamp(-1, 1).reshape(-1, 1, 28, 28) + 1) / 2,
                   ROOT / "baseline_grid.png", nrow=4, padding=2)
        record = {"seeds": CFG["feasibility_seeds"], "rows": rows, "passed": passed,
                  "passed_count": sum(passed), "automatic_gate_passed": sum(passed) >= 6,
                  "raw_sha256": hashlib.sha256((ROOT / "baseline_raw.pt").read_bytes()).hexdigest(),
                  "visual_review_required": True}
        (ROOT / "BASELINE_DONE.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    except BaseException as exc:
        (ROOT / "BASELINE_FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
