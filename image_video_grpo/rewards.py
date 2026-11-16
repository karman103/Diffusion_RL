"""Predeclared rewards and independent evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn


class CifarClassifier(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(width * 4, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def image_scores(images: torch.Tensor, proxy: nn.Module, gold: nn.Module, target: int = 6):
    x = images.clamp(-1, 1)
    p = proxy(x).softmax(1)
    g = gold(x).softmax(1)
    return {"reward": p[:, target], "gold_probability": g[:, target],
            "gold_accuracy": (g.argmax(1) == target).float(),
            "gold_features": gold.features(x)}


def _digit_crops(frame: np.ndarray) -> list[np.ndarray]:
    mask = frame > 0.30
    labels, count = ndimage.label(mask)
    components = []
    for idx in range(1, count + 1):
        area = int((labels == idx).sum())
        if area < 12:
            continue
        ys, xs = np.nonzero(labels == idx)
        crop = frame[max(0, ys.min()-2):min(64, ys.max()+3), max(0, xs.min()-2):min(64, xs.max()+3)]
        components.append((area, crop))
    components.sort(key=lambda z: -z[0])
    return [c for _, c in components[:2]]


@torch.no_grad()
def video_scores(videos: torch.Tensor, proxy: nn.Module, gold: nn.Module) -> dict:
    """Black-box digit proxy plus motion; independent gold CNN is evaluator.

    Component detection is deliberately simple and its failures are retained in
    the report. This reward does not establish human-perceived video quality.
    """
    vids = videos.detach().float().clamp(0, 1).cpu().numpy()[:, :, 0]
    all_proxy, all_gold, all_motion, all_static, all_mass = [], [], [], [], []
    dev = next(proxy.parameters()).device
    for vid in vids:
        crops = [_digit_crops(frame) for frame in vid]
        chosen = [crop for frame_crops in crops for crop in frame_crops]
        if chosen:
            tensors = [F.interpolate(torch.from_numpy(c)[None, None].float(), size=(28, 28), mode="bilinear", align_corners=False)[0] for c in chosen]
            x = torch.stack(tensors).to(dev) * 2 - 1
            p = proxy(x).softmax(1).max(1).values.mean().item()
            g = gold(x).softmax(1).max(1).values.mean().item()
        else:
            p = g = 0.0
        # The reward requests visible movement, but penalizes arbitrary flicker.
        d1 = np.abs(np.diff(vid, axis=0)).mean()
        d2 = np.abs(np.diff(vid, n=2, axis=0)).mean() if len(vid) > 2 else 0.0
        motion = float(np.exp(-((d1 - 0.06) / 0.06) ** 2) * np.exp(-d2 / 0.12))
        mass = float(vid.mean())
        all_proxy.append(float(p))
        all_gold.append(float(g))
        all_motion.append(motion)
        all_static.append(float(d1 < 0.005))
        all_mass.append(mass)
    p = torch.tensor(all_proxy)
    m = torch.tensor(all_motion)
    mass = torch.tensor(all_mass)
    # Guard against all-black/all-white clips and classifier-only reward hacking.
    mass_guard = ((mass > 0.01) & (mass < 0.35)).float()
    return {"reward": (0.5 * p + 0.5 * m) * mass_guard,
            "proxy_digit_confidence": p, "gold_digit_confidence": torch.tensor(all_gold),
            "motion_score": m, "static": torch.tensor(all_static), "ink_mass": mass}
