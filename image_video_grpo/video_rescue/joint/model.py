"""Four-frame joint DDPM initialized from the earlier pretrained image DDPM."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from rldiff.models import UNet

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


def build_from_image() -> UNet:
    path = WORKSPACE / "mnist_rl/runs/base/ckpt.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CFG["parent_image_checkpoint_sha256"]:
        raise RuntimeError("Parent image checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)["ema"]
    model = UNet(ch=32)
    model.inp = nn.Conv2d(4, 32, 3, padding=1)
    model.out[-1] = nn.Conv2d(32, 4, 3, padding=1)
    upgraded = dict(state)
    upgraded["inp.weight"] = state["inp.weight"].repeat(1, 4, 1, 1) / 4
    upgraded["out.2.weight"] = state["out.2.weight"].repeat(4, 1, 1, 1)
    upgraded["out.2.bias"] = state["out.2.bias"].repeat(4)
    model.load_state_dict(upgraded, strict=True)
    return model


def load_pretrained(device: str) -> UNet:
    if not (ROOT / "JOINT_DONE.json").exists():
        raise FileNotFoundError("Joint-video pretraining completion marker missing")
    model = build_from_image()
    state = torch.load(ROOT / "joint_ema.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
