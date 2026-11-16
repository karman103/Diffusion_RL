"""Stochastic 20-step DDIM policy conditioned on requested digit velocity."""
from __future__ import annotations

import torch

from rldiff.diffusion import Diffusion


class ConditionalVideoPolicy:
    def __init__(self, model, velocity: torch.Tensor, device: str, steps: int = 20):
        self.model, self.velocity, self.device = model, velocity, device
        self.diffusion = Diffusion(device=device, clip_x0=True, t_min=20)
        self.times, self.prevs = self.diffusion.timesteps(steps)

    def transition(self, x: torch.Tensor, i: int):
        t, prev = self.times[i], self.prevs[i]
        at, ap = self.diffusion._ab(t), self.diffusion._ab(prev)
        tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        velocity = self.velocity.expand(len(x), -1)
        eps = self.model(x, tt, velocity)
        x0 = ((x - (1 - at).sqrt() * eps) / at.sqrt()).clamp(-1, 1)
        eps = (x - at.sqrt() * x0) / (1 - at).sqrt()
        if prev < 0:
            return x0, None
        std = ((1 - ap) / (1 - at) * (1 - at / ap)).clamp_min(0).sqrt()
        mean = ap.sqrt() * x0 + (1 - ap - std.square()).clamp_min(0).sqrt() * eps
        return mean, std

    def rollout(self, seed: int, n: int = 1, record: bool = False):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn((n, 4, 28, 28), generator=generator).to(self.device)
        transitions = []
        with torch.no_grad():
            for i in range(len(self.times)):
                mean, std = self.transition(x, i)
                if std is None:
                    x = mean
                else:
                    action = mean + std * torch.randn(x.shape, generator=generator).to(self.device)
                    if record:
                        transitions.append((x.cpu(), action.cpu(), i))
                    x = action
                if not torch.isfinite(x).all():
                    raise FloatingPointError(f"Nonfinite conditional video at step {i}")
        return x.detach(), transitions
