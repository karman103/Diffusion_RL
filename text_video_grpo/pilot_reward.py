"""Declared red-object motion proxy; never a biological or semantic-quality claim."""

from __future__ import annotations

import numpy as np


def red_motion_score(video: np.ndarray) -> dict[str, float]:
    """Return bounded rightward-motion reward from an RGB video in [0, 1]."""
    x = np.asarray(video, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 3 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError(f"Expected finite [frames,height,width,3] video; got {x.shape}")
    if x.min() < -0.001 or x.max() > 1.001:
        raise ValueError("Video must be in [0,1]")
    red = np.maximum(0, x[..., 0] - 0.5 * (x[..., 1] + x[..., 2]) - 0.08)
    mass = red.mean(axis=(1, 2))
    xs = np.linspace(0, 1, x.shape[2], dtype=np.float32)[None, None, :]
    center = (red * xs).sum(axis=(1, 2)) / (red.sum(axis=(1, 2)) + 1e-6)
    visibility = float(np.minimum(mass.mean() / 0.08, 1.0))
    displacement = float(center[-1] - center[0])
    smoothness = float(np.mean(np.abs(np.diff(center, n=2)))) if len(center) > 2 else 0.0
    reward = visibility * (displacement - 0.25 * smoothness)
    return {"reward": float(reward), "red_visibility": visibility,
            "red_displacement": displacement, "center_jerk": smoothness,
            "frame_difference": float(np.abs(np.diff(x, axis=0)).mean())}


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float32)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("At least two finite group rewards required")
    std = float(r.std())
    if std < 1e-7:
        return np.zeros_like(r)
    return (r - r.mean()) / std
