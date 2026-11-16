"""Digit+direction conditioned variant: reward, evaluation, paths."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .. import metrics as M
from ..data import FRAMES, NDIR

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("LVD_RUNS", HERE / "runs"))
PROT = json.loads((HERE / "PROTOCOL.json").read_text())
INK = (0.02, 0.30)


def _probs(clips, net):
    cen = M.centroids(clips)
    return net(M.crops(clips, cen)).softmax(1).reshape(clips.shape[0], FRAMES, 10)


@torch.no_grad()
def reward(clips, cond, proxy):
    clips = clips.float().cpu(); cond = cond.cpu()
    d, g = cond[:, 0], cond[:, 1]
    p = _probs(clips, proxy)
    req = p[torch.arange(len(g)), :, g].mean(1)
    cen = M.centroids(clips)
    disp = cen[:, -1] - cen[:, 0]
    _, cos = M._angle_err(disp, d)
    speed = (1 - (disp.norm(dim=-1) - M.TRAVEL).abs() / M.TRAVEL).clamp_min(0)
    ink = ((clips.clamp(-1, 1) + 1) / 2).mean((2, 3, 4))
    guard = ((ink >= INK[0]) & (ink <= INK[1])).all(1).float()
    return guard * (0.4 * req + 0.6 * cos.clamp_min(0) * speed)


@torch.no_grad()
def evaluate(clips, cond, evaluator):
    clips = clips.float().cpu(); cond = cond.cpu()
    d, g = cond[:, 0], cond[:, 1]
    e = M.evaluate(clips, d, evaluator)            # legible/direction measures, evaluator + phase corr
    correct = (e["digit"] == g) & (e["legible"] > 0)
    e["correct_digit"] = correct.float()
    e["success"] = (correct & (e["direction_ok"] > 0)).float()
    e["requested_digit"] = g
    return e


def eval_conditions(n):
    i = torch.arange(n)
    return torch.stack((i % NDIR, (i // NDIR) % 10), 1)
