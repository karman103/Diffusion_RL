"""Shared, numerically explicit Gaussian-policy GRPO machinery.

The density returned here is a *conditional transition* density for a modified
sampler. It is not the likelihood of a completed image or video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def normal_logprob(action: torch.Tensor, mean: torch.Tensor, std: float | torch.Tensor) -> torch.Tensor:
    """Exact joint log density, one scalar per batch member."""
    if torch.any(torch.as_tensor(std) <= 0):
        raise ValueError("Gaussian transition std must be positive")
    z = ((action.float() - mean.float()) / std).square()
    lp = -0.5 * (z + 2 * torch.log(torch.as_tensor(std, device=z.device)) + math.log(2 * math.pi))
    return lp.flatten(1).sum(1)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two sibling rewards")
    centered = rewards - rewards.mean()
    scale = rewards.std(unbiased=False)
    return torch.where(scale > eps, centered / scale.clamp_min(eps), torch.zeros_like(centered))


def clipped_objective(new_lp: torch.Tensor, old_lp: torch.Tensor, advantages: torch.Tensor,
                      clip_eps: float = 0.2) -> tuple[torch.Tensor, dict]:
    if new_lp.shape != old_lp.shape or new_lp.shape != advantages.shape:
        raise ValueError("Log probabilities and advantages must have matching shapes")
    delta = new_lp - old_lp
    ratio = torch.exp(delta.clamp(-20, 20))
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    return loss, {"clip_fraction": ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item(),
                  "max_abs_log_ratio": delta.abs().max().item()}


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    old_logprob: torch.Tensor
    step: int


class ImageResidual(nn.Module):
    """Zero-initialized residual on predicted DDPM noise."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return prediction + self.net(torch.cat((state, prediction), dim=1))


class VideoResidual(nn.Module):
    """Zero-initialized temporal-spatial residual on predicted flow velocity."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(2, 8, 3, padding=1), nn.SiLU(),
                                 nn.Conv3d(8, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        # Native video layout is B,T,C,H,W; Conv3d expects B,C,T,H,W.
        x = torch.cat((state, prediction), dim=2).permute(0, 2, 1, 3, 4)
        return prediction + self.net(x).permute(0, 2, 1, 3, 4)
