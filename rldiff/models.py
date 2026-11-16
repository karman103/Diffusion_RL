"""Small networks: a time-conditioned UNet for 28x28 diffusion, and CNN classifiers
used as reward models (proxy), gold evaluators, and feature extractors."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.o = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        h = (v @ a.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.o(h)


class UNet(nn.Module):
    """28 -> 14 -> 7 UNet (~1.5M params at ch=32), eps-prediction."""

    def __init__(self, ch=32, mult=(1, 2, 2), dropout=0.0):
        super().__init__()
        tdim = ch * 4
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        chs = [ch * m for m in mult]
        self.down = nn.ModuleList()
        skips = [ch]
        c = ch
        for i, co in enumerate(chs):
            self.down.append(ResBlock(c, co, tdim, dropout)); c = co; skips.append(c)
            self.down.append(ResBlock(c, co, tdim, dropout)); skips.append(c)
            if i < len(chs) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c)
        self.mid1 = ResBlock(c, c, tdim, dropout)
        self.mid_attn = Attn(c)
        self.mid2 = ResBlock(c, c, tdim, dropout)
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(chs))):
            for _ in range(3):
                self.up.append(ResBlock(c + skips.pop(), co, tdim, dropout)); c = co
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], 1), temb)
            else:
                h = m(h)
        return self.out(h)


class CNN(nn.Module):
    """Classifier on [-1,1] images. `width` controls capacity; `.features` gives penultimate embedding."""

    def __init__(self, width=32, nclass=10, dropout=0.3):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * w, nclass))

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


class MLPClassifier(nn.Module):
    """Deliberately weak/different-inductive-bias reward model (used as the 'proxy')."""

    def __init__(self, hidden=256, nclass=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, nclass))

    def forward(self, x):
        return self.net(x)
