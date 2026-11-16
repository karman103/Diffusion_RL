"""Stochastic DDIM transitions on the locally pretrained four-frame denoiser."""
from __future__ import annotations

import torch

from rldiff.diffusion import Diffusion


class VideoPolicy:
    def __init__(self, model, device: str, steps: int = 20):
        self.model = model
        self.device = device
        self.diffusion = Diffusion(device=device, clip_x0=True, t_min=20)
        self.times, self.prevs = self.diffusion.timesteps(steps)

    def transition(self, x: torch.Tensor, i: int):
        mean, std, _ = self.diffusion.step_mean_std(
            self.model, x, self.times[i], self.prevs[i], eta=1.0)
        return mean, None if self.prevs[i] < 0 else std

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
                    raise FloatingPointError(f"Nonfinite video sample at step {i}")
        return x.detach(), transitions
