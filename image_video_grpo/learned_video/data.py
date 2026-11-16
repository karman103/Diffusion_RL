"""Fresh Moving-MNIST clips with a known, requested straight-line direction."""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import torch

MNIST = Path(__file__).resolve().parents[2] / "data/MNIST/raw"
FRAMES, SIZE, DIGIT, SPEED, NDIR = 8, 32, 20, 1.5, 8
DIRS = torch.tensor([[math.cos(2 * math.pi * k / NDIR), math.sin(2 * math.pi * k / NDIR)] for k in range(NDIR)])


def _idx(name: str) -> np.ndarray:
    path = MNIST / name
    raw = path.read_bytes() if path.exists() else gzip.open(str(path) + ".gz").read()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def mnist(split: str):
    pre = "train" if split == "train" else "t10k"
    x = torch.from_numpy(_idx(f"{pre}-images-idx3-ubyte").copy()).float() / 255.0
    y = torch.from_numpy(_idx(f"{pre}-labels-idx1-ubyte").copy()).long()
    return x, y


def make_clips(digits: torch.Tensor, direction: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """digits: B,28,28 in [0,1] (resized to 20x20); direction: B int. Returns B,T,1,32,32 in [-1,1].

    Sub-pixel positions are rendered by bilinear shifting so speed is exactly SPEED.
    """
    b = digits.shape[0]
    digits = torch.nn.functional.interpolate(digits[:, None], size=(DIGIT, DIGIT), mode="bilinear",
                                             antialias=True, align_corners=False)[:, 0]
    v = DIRS[direction] * SPEED                              # B,2 (dx,dy)
    travel = v * (FRAMES - 1)
    room = SIZE - DIGIT
    lo = (-travel).clamp_min(0)
    hi = room - travel.clamp_min(0)
    start = lo + torch.rand(b, 2, generator=gen) * (hi - lo)
    t = torch.arange(FRAMES).float()
    pos = start[:, None, :] + t[None, :, None] * v[:, None, :]   # B,T,2 top-left (x,y)
    canvas = torch.zeros(b * FRAMES, 1, SIZE, SIZE)
    canvas[:, :, :DIGIT, :DIGIT] = digits.repeat_interleave(FRAMES, 0)[:, None]
    # translate via affine grid (pixel shift -> normalised coords)
    shift = pos.reshape(-1, 2) * 2 / SIZE
    theta = torch.zeros(b * FRAMES, 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = -shift
    grid = torch.nn.functional.affine_grid(theta, canvas.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(canvas, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (out.reshape(b, FRAMES, 1, SIZE, SIZE) * 2 - 1).clamp(-1, 1)
