"""Audit both video samplers, paired noise retention, and checkpoint metadata."""
from __future__ import annotations

import hashlib
import json

import torch

from image_video_grpo.video_revisit.preflight import ROOT, PARENT, CFG


def inspect(kind: str, seed: int):
    folder = ROOT / ("corrected" if kind == "corrected" else "") / f"seed_{seed}"
    path = folder / "raw.pt"
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if list(raw.shape) != [1, 20, 1, 64, 64] or not torch.isfinite(raw).all():
        raise ValueError(f"Invalid saved video: {path}")
    row = json.loads((folder / "RESULT.json").read_text())
    initial = torch.rand((1, 20, 1, 64, 64), generator=torch.Generator().manual_seed(seed))
    a, b = raw.flatten().double(), initial.flatten().double()
    corr = float(torch.corrcoef(torch.stack((a, b)))[0, 1])
    passed = bool(row["gold_digit_confidence"] >= 0.5 and
                  0.01 < row["ink_mass"] < 0.35 and row["frame_delta"] > 0.005)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed, "sampler": kind, "finite": True,
            "correlation_to_initial_noise": corr, "passed_automatic_gate": passed,
            "summary": row}


def main():
    checkpoint = PARENT / "checkpoints/moving_mnist_epoch_79.pth"
    with torch.serialization.safe_globals([]):
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    audit = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             "checkpoint_filename": checkpoint.name,
             "checkpoint_internal_epoch": int(data["epoch"]),
             "checkpoint_internal_loss": float(data["loss"]),
             "native": [], "corrected": []}
    if audit["checkpoint_sha256"] != CFG["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for kind in ("native", "corrected"):
        for seed in CFG["seeds"]:
            folder = ROOT / ("corrected" if kind == "corrected" else "") / f"seed_{seed}"
            if kind == "corrected" and not folder.exists():
                continue
            audit[kind].append(inspect(kind, seed))
    if len(audit["native"]) != len(CFG["seeds"]):
        raise ValueError("Native predeclared baseline is incomplete")
    if len(audit["corrected"]) != 2 or any(row["passed_automatic_gate"] for row in audit["corrected"]):
        raise ValueError("Corrected early-stop audit does not match the saved decision")
    audit["corrected_early_stop_reason"] = "First two of four fixed seeds failed; at most two could pass, below the three-of-four gate"
    (ROOT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({kind: {"passed_count": sum(row["passed_automatic_gate"] for row in audit[kind]),
                             "noise_correlation": [row["correlation_to_initial_noise"] for row in audit[kind]]}
                      for kind in ("native", "corrected")}, indent=2))


if __name__ == "__main__":
    main()
