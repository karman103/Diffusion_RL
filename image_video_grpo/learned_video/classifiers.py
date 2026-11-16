"""Two digit classifiers on 20x20 crops, trained on disjoint halves of MNIST train.
proxy (MLP) is used in the training reward; evaluator (CNN) is used only for evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .data import DIGIT, mnist

import os
OUT = Path(os.environ.get("LV_RUNS", Path(__file__).resolve().parent / "runs"))


class Proxy(nn.Module):
    """Attempt 1 (MLP, 0.921) and attempt 2 (global-avg-pool CNN, 0.800) failed the 0.95 gate.
    Attempt 3: narrower CNN than the evaluator (16/24 channels, no hidden layer, avg pooling)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
                                 nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                                 nn.Flatten(), nn.Linear(24 * 5 * 5, 10))

    def forward(self, x):
        return self.net(x)


class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(64 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def prep(x28: torch.Tensor) -> torch.Tensor:
    """28x28 [0,1] -> 20x20 in [-1,1], same resize as the video renderer."""
    x = F.interpolate(x28[:, None], size=(DIGIT, DIGIT), mode="bilinear", antialias=True, align_corners=False)
    return x * 2 - 1


def jitter(x: torch.Tensor, gen) -> torch.Tensor:
    """Random +-2 px shift so classifiers tolerate imperfect centroid crops."""
    pad = F.pad(x, (2, 2, 2, 2), value=-1)
    i, j = torch.randint(0, 5, (2,), generator=gen).tolist()
    return pad[..., i:i + DIGIT, j:j + DIGIT]


def train(model, x, y, epochs, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], generator=g)
        for k in range(0, x.shape[0], 128):
            b = perm[k:k + 128]
            loss = F.cross_entropy(model(jitter(x[b], g)), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def load(device="cpu"):
    p, e = Proxy(), Evaluator()
    p.load_state_dict(torch.load(OUT / "proxy.pt", weights_only=True))
    e.load_state_dict(torch.load(OUT / "evaluator.pt", weights_only=True))
    return [m.to(device).eval().requires_grad_(False) for m in (p, e)]


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    x, y = mnist("train"); xt, yt = mnist("test")
    x, xt = prep(x), prep(xt)
    proxy = train(Proxy(), x[:30000], y[:30000], 5, 11)
    ev = train(Evaluator(), x[30000:], y[30000:], 4, 12)
    with torch.no_grad():
        res = {"proxy_test_acc": float((proxy(xt).argmax(1) == yt).float().mean()),
               "evaluator_test_acc": float((ev(xt).argmax(1) == yt).float().mean()),
               "agreement_on_test": float((proxy(xt).argmax(1) == ev(xt).argmax(1)).float().mean())}
    res["gate_pass"] = res["proxy_test_acc"] >= 0.95 and res["evaluator_test_acc"] >= 0.95
    torch.save(proxy.state_dict(), OUT / "proxy.pt"); torch.save(ev.state_dict(), OUT / "evaluator.pt")
    (OUT / "classifiers.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
