"""Factorized toy-video generator: pretrained digit DDPM plus known motion renderer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rldiff.diffusion import Diffusion
from rldiff.models import UNet
from image_video_grpo.core import normal_logprob

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


class AppearanceResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, base):
        return self.net(torch.cat((x, base), dim=1))


class AdaptedImage(nn.Module):
    def __init__(self):
        super().__init__()
        path = WORKSPACE / "mnist_rl/runs/base/ckpt.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != CFG["image_checkpoint_sha256"]:
            raise RuntimeError("Image checkpoint hash mismatch")
        self.base = UNet(ch=32)
        self.base.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["ema"])
        self.base.eval().requires_grad_(False)
        self.adapter = AppearanceResidual()

    def forward(self, x, t):
        with torch.no_grad():
            eps = self.base(x, t)
        return eps + self.adapter(x, eps)


def velocity_for_seed(seed: int, device: str, n: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    v = (torch.randint(0, 2, (1, 2), generator=gen).float() * 2 - 1) * 2
    return v.to(device).expand(n, -1)


def render_video(canonical: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Translate the *same* generated digit, retaining differentiability for diagnostics."""
    batch = canonical.shape[0]
    digit = ((canonical.clamp(-1, 1) + 1) / 2)
    frames = []
    for frame in range(4):
        theta = torch.zeros(batch, 2, 3, device=canonical.device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = -2 * velocity[:, 0] * (frame - 1.5) / 28
        theta[:, 1, 2] = -2 * velocity[:, 1] * (frame - 1.5) / 28
        grid = F.affine_grid(theta, digit.shape, align_corners=False)
        frames.append(F.grid_sample(digit, grid, align_corners=False, padding_mode="zeros")[:, 0])
    return torch.stack(frames, dim=1) * 2 - 1


class FactorizedPolicy:
    def __init__(self, model: AdaptedImage, device: str):
        self.model, self.device = model, device
        self.diffusion = Diffusion(device=device, clip_x0=True, t_min=20)
        self.times, self.prevs = self.diffusion.timesteps(20)

    def transition(self, x, i, adapt: bool = True):
        if adapt:
            return self.diffusion.step_mean_std(self.model, x, self.times[i], self.prevs[i], eta=1.0)[:2]
        return self.diffusion.step_mean_std(self.model.base, x, self.times[i], self.prevs[i], eta=1.0)[:2]

    def rollout(self, seed: int, n: int = 1, record: bool = False,
                velocity_override: torch.Tensor | None = None):
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn((n, 1, 28, 28), generator=gen).to(self.device)
        trajectory = []
        with torch.no_grad():
            for i in range(20):
                mean, std = self.transition(x, i)
                if self.prevs[i] < 0:
                    x = mean
                else:
                    action = mean + std * torch.randn(x.shape, generator=gen).to(self.device)
                    if record:
                        trajectory.append((x.cpu(), action.cpu(), normal_logprob(action, mean, std).cpu(), i))
                    x = action
                if not torch.isfinite(x).all():
                    raise FloatingPointError(f"Nonfinite appearance at step {i}")
        velocity = (velocity_for_seed(seed, self.device, n) if velocity_override is None
                    else velocity_override.to(self.device).expand(n, -1))
        video = render_video(x, velocity)
        return video.detach(), x.detach(), velocity, trajectory
