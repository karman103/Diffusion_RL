"""Cosine-schedule DDPM training loss and a stochastic DDIM(eta=1) policy with exact
per-step Gaussian transition log-densities (not a likelihood of the finished clip)."""
from __future__ import annotations

import math

import torch

from ..core import normal_logprob

T = 1000


def alphas_bar(device="cpu") -> torch.Tensor:
    s = 0.008
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0])[1:].clamp(1e-5, 0.9999)
    return ab.float().to(device)


def train_loss(model, x0, direction, ab, gen=None):
    b = x0.shape[0]
    t = torch.randint(0, T, (b,), device=x0.device)
    eps = torch.randn_like(x0)
    a = ab[t].view(b, 1, 1, 1, 1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return (model(xt, t, direction) - eps).square().mean()


def schedule(steps: int):
    ts = torch.linspace(T - 1, 0, steps).round().long()
    return list(zip(ts.tolist(), ts[1:].tolist() + [-1]))


def step_mean_std(model, x, t, t_prev, direction, ab, cfg: float = 0.0):
    b = x.shape[0]
    tt = torch.full((b,), t, device=x.device, dtype=torch.long)
    eps = model(x, tt, direction)
    if cfg:
        from .data import NDIR
        if direction.ndim == 2:
            null = torch.tensor([NDIR, 10], device=direction.device).expand_as(direction)
        else:
            null = torch.full_like(direction, NDIR)
        eps_u = model(x, tt, null)
        eps = eps + cfg * (eps - eps_u)
    a = ab[t]
    a_prev = ab[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
    x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
    eps = (x - a.sqrt() * x0) / (1 - a).sqrt()          # consistent with clamped x0
    sigma = ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).clamp_min(0).sqrt()
    mean = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps
    return mean, sigma, x0


@torch.no_grad()
def sample(model, direction, noise_seed: int | list[int], ab, steps=40, cfg=0.0, record=False, device="cpu"):
    """Returns final clips and (optionally) the per-step (t, t_prev, state, action, logprob)."""
    from .data import FRAMES, SIZE
    seeds = noise_seed if isinstance(noise_seed, list) else [noise_seed + i for i in range(direction.shape[0])]
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    x = torch.stack([torch.randn(FRAMES, 1, SIZE, SIZE, generator=g) for g in gens]).to(device)
    traj = []
    for t, tp in schedule(steps):
        mean, sigma, x0 = step_mean_std(model, x, t, tp, direction, ab, cfg)
        if tp < 0:
            x = x0
            break
        z = torch.stack([torch.randn(FRAMES, 1, SIZE, SIZE, generator=g) for g in gens]).to(device)
        nxt = mean + sigma * z
        if record:
            traj.append((t, tp, x, nxt, normal_logprob(nxt, mean, sigma)))
        x = nxt
    return x, traj
