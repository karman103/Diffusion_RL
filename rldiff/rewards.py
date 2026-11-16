"""Reward functions + evaluation metrics."""
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .models import CNN, MLPClassifier


def load_clfs(device):
    gold = CNN(32); gold.load_state_dict(torch.load("runs/clf/gold.pt", map_location="cpu"))
    feat = CNN(24); feat.load_state_dict(torch.load("runs/clf/feat.pt", map_location="cpu"))
    proxy = MLPClassifier(); proxy.load_state_dict(torch.load("runs/clf/proxy.pt", map_location="cpu"))
    return [m.to(device).eval().requires_grad_(False) for m in (gold, feat, proxy)]


def jpeg_size_kb(x, quality=95):
    """Black-box, non-differentiable reward (as in DDPO): JPEG file size of the image."""
    imgs = ((x.detach().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).squeeze(1).cpu().numpy()
    out = []
    for im in imgs:
        buf = io.BytesIO()
        Image.fromarray(im, mode="L").save(buf, format="JPEG", quality=quality)
        out.append(buf.tell() / 1000.0)
    return torch.tensor(out, device=x.device)


class Reward:
    """name in {proxy:<c>, gold:<c>, jpeg_compress, jpeg_incompress}"""

    def __init__(self, name, gold, proxy):
        self.name = name
        self.gold, self.proxy = gold, proxy
        self.differentiable = name.startswith(("proxy", "gold"))  # covers proxyp/goldp

    def __call__(self, x):
        if self.name.startswith("proxyp:"):  # bounded probability reward
            c = int(self.name.split(":")[1])
            return F.softmax(self.proxy(x), 1)[:, c]
        if self.name.startswith("goldp:"):
            c = int(self.name.split(":")[1])
            return F.softmax(self.gold(x), 1)[:, c]
        if self.name.startswith("proxy:"):
            c = int(self.name.split(":")[1])
            return F.log_softmax(self.proxy(x), 1)[:, c]
        if self.name.startswith("gold:"):
            c = int(self.name.split(":")[1])
            return F.log_softmax(self.gold(x), 1)[:, c]
        if self.name == "random":  # null-reward control: pure noise advantages
            return torch.randn(x.shape[0], device=x.device)
        if self.name == "jpeg_compress":
            return -jpeg_size_kb(x)
        if self.name == "jpeg_incompress":
            return jpeg_size_kb(x)
        raise ValueError(self.name)


@torch.no_grad()
def frechet_distance(f1, f2):
    from scipy import linalg
    f1, f2 = f1.float().cpu().numpy().astype(np.float64), f2.float().cpu().numpy().astype(np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    eps = 1e-6 * np.eye(s1.shape[0])
    s1, s2 = s1 + eps, s2 + eps
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = np.nan_to_num(np.real(covmean))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def diversity(feats):
    """Mean pairwise L2 distance in feature space (higher = more diverse)."""
    d = torch.cdist(feats, feats)
    n = feats.shape[0]
    return float(d.sum() / (n * (n - 1)))
