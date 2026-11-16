"""Velocity-conditioned four-frame UNet initialized from joint video checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from rldiff.models import UNet

ROOT = Path(__file__).resolve().parent
JOINT = ROOT.parent / "joint"
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(ch=32)
        self.net.inp = nn.Conv2d(6, 32, 3, padding=1)
        self.net.out[-1] = nn.Conv2d(32, 4, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor):
        batch = x.shape[0]
        cond = (velocity / 2)[:, :, None, None].expand(batch, 2, 28, 28)
        return self.net(torch.cat((x, cond), dim=1), t)


def build_from_joint() -> ConditionalUNet:
    path = JOINT / "joint_ema.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CFG["parent_joint_checkpoint_sha256"]:
        raise RuntimeError("Parent joint video checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ConditionalUNet()
    upgraded = {f"net.{key}": val for key, val in state.items()}
    old = upgraded["net.inp.weight"]
    upgraded["net.inp.weight"] = torch.cat((old, torch.zeros_like(old[:, :2])), dim=1)
    model.load_state_dict(upgraded, strict=True)
    return model


def load_pretrained(device: str) -> ConditionalUNet:
    if not (ROOT / "CONDITIONAL_DONE.json").exists():
        raise FileNotFoundError("Conditional video checkpoint incomplete")
    model = build_from_joint()
    model.load_state_dict(torch.load(ROOT / "conditional_ema.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


def velocity_for_seed(seed: int, device: str):
    generator = torch.Generator().manual_seed(seed)
    velocity = (torch.randint(0, 2, (1, 2), generator=generator).float() * 2 - 1) * 2
    return velocity.to(device)
