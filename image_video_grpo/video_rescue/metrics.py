"""Frame validity and temporal diagnostics for four-frame MNIST clips."""
from __future__ import annotations

import torch


@torch.no_grad()
def video_metrics(videos: torch.Tensor, proxy, gold,
                  velocities: torch.Tensor | None = None) -> list[dict]:
    batch, frames = videos.shape[:2]
    x = videos.clamp(-1, 1).reshape(batch * frames, 1, 28, 28)
    device = next(gold.parameters()).device
    pp = proxy(x.to(device)).softmax(1).cpu().reshape(batch, frames, 10)
    gp = gold(x.to(device)).softmax(1).cpu().reshape(batch, frames, 10)
    pix = ((videos.detach().cpu().clamp(-1, 1) + 1) / 2)
    yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
    rows = []
    for i in range(batch):
        weights = pix[i].clamp_min(0)
        mass = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        cx = (weights * xx).sum(dim=(-2, -1)) / mass
        cy = (weights * yy).sum(dim=(-2, -1)) / mass
        centroids = torch.stack((cx, cy), dim=-1)
        motion = float((centroids[1:] - centroids[:-1]).norm(dim=-1).mean())
        smoothness = float((centroids[2:] - 2 * centroids[1:-1] + centroids[:-2]).norm(dim=-1).mean())
        displacement = centroids[-1] - centroids[0]
        direction_cosine = None
        if velocities is not None:
            expected = velocities[i].detach().cpu().float()
            direction_cosine = float((displacement @ expected) /
                                     (displacement.norm() * expected.norm()).clamp_min(1e-6))
        pl = pp[i].argmax(1)
        gl = gp[i].argmax(1)
        proxy_mode = torch.bincount(pl, minlength=10).argmax()
        gold_mode = torch.bincount(gl, minlength=10).argmax()
        rows.append({"proxy_confidence": float(pp[i].max(1).values.mean()),
                     "gold_confidence": float(gp[i].max(1).values.mean()),
                     "proxy_class_consistency": float((pl == proxy_mode).float().mean()),
                     "gold_class_consistency": float((gl == gold_mode).float().mean()),
                     "gold_mode_digit": int(gold_mode),
                     "ink_mass": float(pix[i].mean()),
                     "centroid_motion_px": motion,
                     "centroid_dx_px": float(displacement[0]),
                     "centroid_dy_px": float(displacement[1]),
                     "velocity_direction_cosine": direction_cosine,
                     "centroid_acceleration_px": smoothness,
                     "frame_delta": float((pix[i, 1:] - pix[i, :-1]).abs().mean()),
                     "finite": bool(torch.isfinite(videos[i]).all())})
    return rows


def reward_from_metrics(rows: list[dict]) -> torch.Tensor:
    """Proxy-only reward; hard validity guard prevents blank/noisy solutions."""
    scores = []
    for row in rows:
        guard = 0.03 <= row["ink_mass"] <= 0.35 and 0.5 <= row["centroid_motion_px"] <= 5.0
        motion_shape = max(0.0, 1.0 - row["centroid_acceleration_px"] / 5.0)
        score = (0.6 * row["proxy_confidence"] * row["proxy_class_consistency"]
                 + 0.4 * motion_shape) if guard else 0.0
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)
