"""Bind the audited DDPM policy code to the stronger independent reward pair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import image_video_grpo.ddpm_followup.experiment as base
from image_video_grpo.rewards import CifarClassifier

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


def classifier_pair(kind: str, device: str):
    if kind != "image":
        raise ValueError("This study covers images only")
    pair = []
    for name in ("proxy", "gold"):
        model = CifarClassifier(48)
        model.load_state_dict(torch.load(ROOT / f"{name}.pt", map_location="cpu", weights_only=True))
        pair.append(model.to(device).eval().requires_grad_(False))
    return pair


def gate() -> None:
    checks = json.loads((ROOT / "classifier_results.json").read_text())
    for name in ("proxy", "gold"):
        if checks[name]["test_accuracy"] < CFG["classifier_test_gate"]:
            raise RuntimeError(f"{name} classifier missed its predeclared accuracy gate")
    digest = hashlib.sha256((ROOT.parent / "checkpoints/cifar10_ddpm.safetensors").read_bytes()).hexdigest()
    if digest != CFG["checkpoint_sha256"]:
        raise RuntimeError("Pretrained checkpoint hash mismatch")


# This task-local process reuses the already tested transition, rollout,
# clipped objective, and evaluation implementation. The previous completed
# experiment's on-disk artifacts and protocol are never changed.
base.ROOT = ROOT
base.CFG = CFG
base.classifier_pair = classifier_pair
base.gate = gate


if __name__ == "__main__":
    base.main()
