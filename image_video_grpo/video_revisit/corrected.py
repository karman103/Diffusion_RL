"""Diagnostic corrected Euler sampler; intentionally distinct from upstream sampler."""
from __future__ import annotations

import argparse
import hashlib
import json

import torch
from einops import rearrange

from image_video_grpo.experiment import _save_visual, classifier_pair
from image_video_grpo.models import load_video, video_ids
from image_video_grpo.rewards import video_scores
from image_video_grpo.vendor.utils import ScheduleLinear
from image_video_grpo.video_revisit.preflight import ROOT, PARENT

CFG = json.loads((ROOT / "CORRECTED_PROTOCOL.json").read_text())


@torch.no_grad()
def sample(model, seed: int, device: str):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((1, 20, 1, 64, 64), generator=generator).to(device)
    ids = video_ids(1, 20, device)
    sigmas = ScheduleLinear(1000).sample_sigmas(100)
    for previous, sigma in zip(sigmas[1:], sigmas[:-1]):
        tokens = rearrange(x, "b t c (h ph) (w pw) -> b (t h w) (ph pw c)", ph=4, pw=4)
        prediction = model(tokens, ids, torch.full((1,), float(sigma), device=device))
        velocity = rearrange(prediction, "b (t h w) (ph pw c) -> b t c (h ph) (w pw)",
                             t=20, h=16, w=16, ph=4, pw=4, c=1)
        x = x + velocity * (float(previous) - float(sigma))
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite video at sigma {sigma}")
    return x.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.seed not in CFG["paired_seeds"]:
        raise ValueError("Seed not predeclared")
    dest = ROOT / "corrected" / f"seed_{args.seed}"
    if dest.exists():
        raise FileExistsError(f"Corrected preflight already exists: {dest}")
    digest = hashlib.sha256((PARENT / "checkpoints/moving_mnist_epoch_79.pth").read_bytes()).hexdigest()
    if digest != CFG["checkpoint_sha256"]:
        raise RuntimeError("Video checkpoint hash mismatch")
    dest.mkdir(parents=True)
    try:
        model = load_video(args.device)
        video = sample(model, args.seed, args.device)
        torch.save(video, dest / "raw.pt")
        _save_visual("video", video, dest / "sample.gif")
        proxy, gold = classifier_pair("video", "cpu")
        score = video_scores(video, proxy, gold)
        row = {"seed": args.seed, "shape": list(video.shape),
               "min": float(video.min()), "max": float(video.max()),
               "mean": float(video.mean()), "std": float(video.std()),
               "frame_delta": float((video[:, 1:] - video[:, :-1]).abs().mean()),
               **{key: float(value[0]) for key, value in score.items()}}
        (dest / "RESULT.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    except BaseException as exc:
        (dest / "FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2))
        raise


if __name__ == "__main__":
    main()
