"""Pinned pretrained model loading and model-specific stochastic transitions."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import numpy as np
from diffusers import UNet2DModel
from einops import rearrange, repeat
from safetensors.torch import load_file

from .vendor.model import ToyVideoDiT

ROOT = Path(__file__).resolve().parent


def load_image(device: str) -> UNet2DModel:
    config = json.loads((ROOT / "checkpoints/cifar_config.json").read_text())
    model = UNet2DModel.from_config(config)
    raw = load_file(str(ROOT / "checkpoints/cifar10_ddpm.safetensors"))
    # This 2022 checkpoint predates diffusers' AttentionProcessor key names.
    # Shapes are checked by load_state_dict; no weights are dropped.
    renames = {".query.": ".to_q.", ".key.": ".to_k.",
               ".value.": ".to_v.", ".proj_attn.": ".to_out.0."}
    state = {}
    for key, value in raw.items():
        for old, new in renames.items():
            key = key.replace(old, new)
        state[key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Image state mismatch: {missing}, {unexpected}")
    return model.to(device).eval().requires_grad_(False)


def load_video(device: str) -> ToyVideoDiT:
    model = ToyVideoDiT(4, 4, depth=16, hidden_multiplier=32)
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]):
        raw = torch.load(ROOT / "checkpoints/moving_mnist_epoch_79.pth", map_location="cpu", weights_only=True)
    state = raw["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


class ImagePolicy:
    """Stochastic DDIM policy; stock-reference DDPM uses checkpoint variance."""

    def __init__(self, backbone: UNet2DModel, adapter, steps: int):
        self.backbone, self.adapter = backbone, adapter
        self.steps = steps
        self.sampler = "ddim"
        cfg = json.loads((ROOT / "checkpoints/cifar_scheduler.json").read_text())
        if cfg["beta_schedule"] != "linear" or cfg["variance_type"] != "fixed_large":
            raise ValueError("Unexpected image scheduler; formulas must be re-audited")
        beta = torch.linspace(cfg["beta_start"], cfg["beta_end"], cfg["num_train_timesteps"], device=next(backbone.parameters()).device)
        self.alpha_bar = torch.cumprod(1 - beta, 0)
        self.times = torch.linspace(999, 0, steps).round().long().tolist()

    @property
    def shape(self):
        return (3, 32, 32)

    def transition(self, x: torch.Tensor, i: int, adapt: bool = True):
        t = self.times[i]
        prev = self.times[i + 1] if i + 1 < len(self.times) else -1
        with torch.no_grad():
            eps0 = self.backbone(x, torch.full((len(x),), t, device=x.device, dtype=torch.long)).sample
        eps = self.adapter(x, eps0) if adapt else eps0
        at = self.alpha_bar[t]
        ap = self.alpha_bar[prev] if prev >= 0 else torch.ones_like(at)
        bt, bp = 1 - at, 1 - ap
        a = at / ap
        b = 1 - a
        x0 = ((x - bt.sqrt() * eps) / at.sqrt()).clamp(-1, 1)
        if self.sampler == "ddim":
            std = ((bp / bt) * (1 - at / ap)).clamp_min(0).sqrt() if prev >= 0 else None
            mean = ap.sqrt() * x0 + (bp - std.square()).clamp_min(0).sqrt() * eps if prev >= 0 else x0
        elif self.sampler == "ddpm":
            mean = ap.sqrt() * b / bt * x0 + a.sqrt() * bp / bt * x
            std = b.sqrt() if prev >= 0 else None
        else:
            raise ValueError(self.sampler)
        return mean, std


def video_ids(batch: int, frames: int, device: str) -> torch.Tensor:
    ids = torch.zeros(frames, 16, 16, 4, device=device)
    ids[..., 0] = torch.arange(16, device=device)[None, :, None]
    ids[..., 1] = torch.arange(16, device=device)[None, None, :]
    ids[..., 2] = torch.arange(frames, device=device)[:, None, None]
    return repeat(ids, "t h w c -> b (t h w) c", b=batch)


class VideoPolicy:
    """Modified noisy Euler policy around upstream toy-flow sampler.

    Upstream uses uniform initial noise and x += delta_sigma * velocity / 10.
    The trained target is x_data - noise, so the experimental sampler uses the
    usual Euler scale 1 rather than the upstream /10. Every nonterminal step
    adds Gaussian noise for tractable action densities.
    This does not assert marginal equivalence with the stock deterministic flow.
    """

    def __init__(self, backbone: ToyVideoDiT, adapter, steps: int, frames: int = 4, noise: float = 0.025):
        self.backbone, self.adapter = backbone, adapter
        self.steps, self.frames, self.noise = steps, frames, noise
        self.velocity_scale = 1.0
        self.sigmas = torch.linspace(1.0, 0.001, steps + 1).tolist()

    @property
    def shape(self):
        return (self.frames, 1, 64, 64)

    def transition(self, x: torch.Tensor, i: int, adapt: bool = True):
        batch, frames = x.shape[:2]
        tokens = rearrange(x, "b t c (h ph) (w pw) -> b (t h w) (ph pw c)", ph=4, pw=4)
        ids = video_ids(batch, frames, str(x.device))
        sigma = self.sigmas[i]
        delta = self.sigmas[i] - self.sigmas[i + 1]
        with torch.no_grad():
            pred = self.backbone(tokens, ids, torch.full((batch,), sigma, device=x.device))
            vel0 = rearrange(pred, "b (t h w) (ph pw c) -> b t c (h ph) (w pw)", t=frames, h=16, w=16, ph=4, pw=4, c=1)
        vel = self.adapter(x, vel0) if adapt else vel0
        mean = x + (delta * self.velocity_scale) * vel
        std = self.noise * math.sqrt(delta) if i < self.steps - 1 else None
        return mean, std
