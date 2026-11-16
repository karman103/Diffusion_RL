"""Four-frame toy-video denoiser with frozen pretrained image prior."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rldiff.models import UNet

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


class TemporalResidual(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, width, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(width, width, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(width, 1, (3, 3, 3), padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack((x, eps), dim=1)).squeeze(1)


class VideoDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = UNet(ch=32)
        ckpt = torch.load(WORKSPACE / "mnist_rl/runs/base/ckpt.pt", map_location="cpu", weights_only=True)
        self.image.load_state_dict(ckpt["ema"], strict=True)
        self.image.eval().requires_grad_(False)
        self.temporal = TemporalResidual()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, frames = x.shape[:2]
        with torch.no_grad():
            base = self.image(x.reshape(batch * frames, 1, 28, 28),
                              t.repeat_interleave(frames)).reshape_as(x)
        return base + self.temporal(x, base)


def moving_mnist_batch(images: torch.Tensor, batch: int, generator: torch.Generator,
                       speed: int = 1, return_velocity: bool = False):
    """Affine-shift one MNIST digit across four frames, preserving a black background."""
    device = images.device
    idx = torch.randint(len(images), (batch,), generator=generator)
    digit = images[idx].to(device)
    vx = (torch.randint(0, 2, (batch,), generator=generator).to(device).float() * 2 - 1) * speed
    vy = (torch.randint(0, 2, (batch,), generator=generator).to(device).float() * 2 - 1) * speed
    frames = []
    for frame in range(4):
        offset = frame - 1.5
        theta = torch.zeros(batch, 2, 3, device=device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = -2 * vx * offset / 28
        theta[:, 1, 2] = -2 * vy * offset / 28
        grid = F.affine_grid(theta, digit.shape, align_corners=False)
        frames.append(F.grid_sample(digit, grid, align_corners=False, padding_mode="zeros")[:, 0])
    video = torch.stack(frames, dim=1) * 2 - 1
    return (video, torch.stack((vx, vy), dim=1)) if return_velocity else video
