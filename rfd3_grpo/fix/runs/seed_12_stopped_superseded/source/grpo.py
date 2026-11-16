"""Small GRPO and noise-specialized LoRA primitives, independent of Foundry."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def group_advantages(rewards, epsilon=1e-6):
    """Normalize within each condition, never across different conditions."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Expected [conditions, siblings>=2]")
    centered = rewards-rewards.mean(-1,keepdim=True)
    std = rewards.std(-1,correction=0,keepdim=True)
    return torch.where(std > epsilon,centered/std.clamp_min(epsilon),torch.zeros_like(rewards))


class NoiseLoRA(nn.Module):
    """Zero-initialized low-rank deltas to to_r_update[-1], per noise band.

    This adapter is applied ONLY after the last recycling pass. Earlier recycling
    uses original weights; consequently cached last-pass features remain exact.
    """
    def __init__(self, channels=128, rank=2, boundaries=(2.,8.), seed=0):
        super().__init__()
        if not 1 <= rank <= 3:
            raise ValueError("The output head has only three coordinates; use rank1..3")
        self.boundaries = tuple(boundaries)
        n = len(self.boundaries)+1
        gen = torch.Generator().manual_seed(seed)
        self.a = nn.Parameter(torch.randn(n,rank,channels,generator=gen)*.02)
        self.b = nn.Parameter(torch.zeros(n,3,rank))
        self.rank = rank

    def band(self, sigma):
        return sum(float(sigma) >= edge for edge in self.boundaries)

    def forward(self, features, sigma):
        band = self.band(sigma)
        return F.linear(F.linear(features,self.a[band]),self.b[band])

    def delta_weight(self, sigma):
        band = self.band(sigma)
        return self.b[band] @ self.a[band]


def gaussian_logprob(action,mean,std):
    return (-.5*((action-mean)/std).square()-math.log(std)-.5*math.log(2*math.pi)).sum()


def log_ratio(action, old_mean, new_mean, std):
    """Exact joint Gaussian ratio without subtracting large log-density constants."""
    delta = new_mean-old_mean
    return (((action-old_mean)*delta-.5*delta.square())/(std*std)).sum()


def ve_sde_coefficients(sigma, next_sigma, noise_level):
    """Reverse-VE Euler coefficients for score=(D-x)/sigma².

    At noise_level=0 this is the EDM probability-flow Euler step. Positive
    levels produce an explicit Gaussian action; this does not prove finite-step
    or learned-score marginal preservation for RFD3.
    """
    sigma,next_sigma,noise_level=map(float,(sigma,next_sigma,noise_level))
    if sigma<=0 or not 0<=next_sigma<sigma or not 0<=noise_level<=1:
        raise ValueError("Expected descending positive sigma and noise level in [0,1]")
    dt=next_sigma-sigma
    drift_scale=1+noise_level**2
    std=math.sqrt(2*noise_level**2*sigma*(-dt))
    return drift_scale,dt,std


def transition_loss(record,adapter,advantage,clip=.2,kl_beta=.01):
    delta = record["coefficient"] * adapter(record["features"],record["sigma_hat"])
    new_mean = record["ref_mean"]+delta
    lr = log_ratio(record["action"],record["old_mean"],new_mean,record["std"])
    if not torch.isfinite(lr) or lr.abs()>30:
        raise FloatingPointError("Unstable joint likelihood ratio; reduce learning rate/reuse")
    ratio = lr.exp()
    pg = -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    kl = delta.square().sum()/(2*record["std"]**2)
    return pg+kl_beta*kl,dict(logratio=lr.detach(),kl=kl.detach(),
                            clipped=((ratio-1).abs()>clip).float().detach())
