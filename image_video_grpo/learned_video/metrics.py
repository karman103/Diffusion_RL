"""Training reward (centroid motion + proxy legibility) and independent evaluation measures
(phase-correlation motion + evaluator legibility). Clips: B,T,1,S,S in [-1,1]."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import DIGIT, DIRS, FRAMES, SIZE, SPEED

TRAVEL = SPEED * (FRAMES - 1)
INK_RANGE = (0.02, 0.15)   # fraction of canvas ink per frame; real clips measured in tests at ~0.05


def _ink(clips):
    return ((clips.clamp(-1, 1) + 1) / 2)


def centroids(clips):
    w = (_ink(clips)[:, :, 0] - 0.1).clamp_min(0)
    m = w.sum((-1, -2)).clamp_min(1e-6)
    ar = torch.arange(SIZE, dtype=clips.dtype, device=clips.device)
    return torch.stack(((w * ar).sum((-1, -2)) / m, (w * ar[:, None]).sum((-1, -2)) / m), -1)   # B,T,(x,y)


def crops(clips, cen):
    """20x20 crop centred on each frame's centroid (bilinear, so sub-pixel), for the classifiers."""
    b, t = clips.shape[:2]
    x = clips.reshape(b * t, 1, SIZE, SIZE)
    c = cen.reshape(b * t, 2)
    lin = (torch.arange(DIGIT, dtype=x.dtype, device=x.device) - (DIGIT - 1) / 2)
    gx = (c[:, 0, None] + lin[None]) * 2 / (SIZE - 1) - 1
    gy = (c[:, 1, None] + lin[None]) * 2 / (SIZE - 1) - 1
    grid = torch.stack((gx[:, None, :].expand(-1, DIGIT, -1), gy[:, :, None].expand(-1, -1, DIGIT)), -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="border").clamp(-1, 1)


def phase_shift(a, b):
    """Sub-pixel displacement (dx,dy) that moves frame a onto frame b, by phase correlation."""
    A, B = torch.fft.fft2(a - a.mean((-1, -2), keepdim=True)), torch.fft.fft2(b - b.mean((-1, -2), keepdim=True))
    r = B * A.conj()
    r = torch.fft.ifft2(r / r.abs().clamp_min(1e-8)).real
    n, s = r.shape[0], r.shape[-1]
    idx = r.reshape(n, -1).argmax(1)
    py, px = idx // s, idx % s
    out = []
    for k in range(n):
        y, x = int(py[k]), int(px[k])
        def sub(m, p, l):  # parabolic refinement
            c, lo, hi = m[p], m[(p - 1) % l], m[(p + 1) % l]
            den = lo - 2 * c + hi
            return float(0.5 * (lo - hi) / den) if den.abs() > 1e-9 else 0.0
        dx = x + sub(r[k][y], x, s); dy = y + sub(r[k][:, x], y, s)
        dx = dx - s if dx > s / 2 else dx
        dy = dy - s if dy > s / 2 else dy
        out.append((dx, dy))
    return torch.tensor(out)


def _angle_err(d, direction):
    u = DIRS[direction].to(d)
    cos = (d * u).sum(-1) / d.norm(dim=-1).clamp_min(1e-6)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1))), cos


@torch.no_grad()
def reward(clips, direction, proxy):
    clips = clips.float().cpu(); direction = direction.cpu()
    cen = centroids(clips)
    p = proxy(crops(clips, cen)).softmax(1).reshape(clips.shape[0], FRAMES, 10)
    lab = p.argmax(-1)
    mode = lab.mode(1).values
    consist = (lab == mode[:, None]).float().mean(1)
    legible = p.max(-1).values.mean(1) * consist
    d = cen[:, -1] - cen[:, 0]
    _, cos = _angle_err(d, direction)
    speed = (1 - (d.norm(dim=-1) - TRAVEL).abs() / TRAVEL).clamp_min(0)
    ink = _ink(clips).mean((2, 3, 4))
    guard = ((ink >= INK_RANGE[0]) & (ink <= INK_RANGE[1])).all(1).float()
    return guard * (0.4 * legible + 0.6 * cos.clamp_min(0) * speed)


@torch.no_grad()
def evaluate(clips, direction, evaluator):
    """Independent measures: no centroid motion, no proxy classifier."""
    clips = clips.float().cpu(); direction = direction.cpu()
    b = clips.shape[0]
    cen = centroids(clips)
    p = evaluator(crops(clips, cen)).softmax(1).reshape(b, FRAMES, 10)
    lab = p.argmax(-1)
    mode = lab.mode(1).values
    consist = (lab == mode[:, None]).float().mean(1)
    conf = p.max(-1).values.mean(1)
    legible = (conf >= 0.7) & (consist >= 0.75)
    img = clips[:, :, 0]
    d = phase_shift(img[:, 0], img[:, -1])
    ang, _ = _angle_err(d, direction)
    ok_dir = ang < 22.5
    return {"legible": legible.float(), "eval_conf": conf, "consistency": consist, "digit": mode,
            "xcorr_disp": d, "angle_err": ang, "speed_err": (d.norm(dim=-1) - TRAVEL).abs(),
            "direction_ok": ok_dir.float(), "success": (legible & ok_dir).float()}


def class_entropy(digits):
    p = torch.bincount(digits, minlength=10).float(); p = p / p.sum()
    return float(-(p[p > 0] * p[p > 0].log()).sum() / math.log(10))


def pixel_diversity(clips, direction):
    """Mean pairwise RMS distance between clips that share a requested direction."""
    vals = []
    for k in direction.unique():
        c = clips[direction == k].flatten(1).float().cpu()
        if c.shape[0] > 1:
            dist = torch.cdist(c, c) / math.sqrt(c.shape[1])
            n = c.shape[0]
            vals.append(dist.sum() / (n * (n - 1)))
    return float(torch.stack(vals).mean())
