"""Small factorised (2+1)D UNet epsilon-predictor, conditioned on timestep and direction."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .data import NDIR


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat((a.sin(), a.cos()), 1)


class SConv(nn.Module):
    """3x3 spatial conv applied per frame (Conv2d on B*T; Conv3d is slow on MPS)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.c(x.transpose(1, 2).reshape(b * t, c, h, w))
        return y.reshape(b, t, -1, h, w).transpose(1, 2)


class TConv(nn.Module):
    """k=3 temporal conv applied per pixel: Conv2d with a (3,1) kernel on B,C,T,H*W (no permute)."""

    def __init__(self, c):
        super().__init__()
        self.c = nn.Conv2d(c, c, (3, 1), padding=(1, 0))

    def forward(self, x):
        b, c, t, h, w = x.shape
        return self.c(x.reshape(b, c, t, h * w)).reshape(b, -1, t, h, w)


class Block(nn.Module):
    """Spatial 3x3 conv per frame, then temporal k=3 conv per pixel; FiLM from embedding."""

    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.s1, self.t1 = SConv(cin, cout), TConv(cout)
        self.film = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.s2, self.t2 = SConv(cout, cout), TConv(cout)
        self.skip = Pointwise(cin, cout) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.t1(self.s1(F.silu(self.n1(x))))
        sc, sh = self.film(e)[:, :, None, None, None].chunk(2, 1)
        h = F.silu(self.n2(h) * (1 + sc) + sh)
        h = self.t2(self.s2(h))
        return h + self.skip(x)


class Pointwise(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.l = nn.Linear(cin, cout)

    def forward(self, x):
        return self.l(x.movedim(1, -1)).movedim(-1, 1)


class TemporalAttention(nn.Module):
    """Full space-time attention at the coarsest resolution (8x8x8 = 512 tokens); optional."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = Pointwise(c, 3 * c)
        self.out = Pointwise(c, c)
        self.heads = 4

    def forward(self, x):
        b, c, t, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, t * h * w).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(o.transpose(-1, -2).reshape(b, c, t, h, w))


class VideoUNet(nn.Module):
    def __init__(self, ch=(32, 64, 128), emb=128, attention=False, digits=False):
        super().__init__()
        self.emb_dim = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.dir_emb = nn.Embedding(NDIR + 1, emb)        # index NDIR = unconditional (CFG)
        self.digit_emb = None
        if digits:                                        # index 10 = unconditional; zero-init for warm start
            self.digit_emb = nn.Embedding(11, emb)
            nn.init.zeros_(self.digit_emb.weight)
        self.inp = SConv(1, ch[0])
        self.d0, self.d1, self.d2 = Block(ch[0], ch[0], emb), Block(ch[0], ch[1], emb), Block(ch[1], ch[2], emb)
        self.mid1, self.mid2 = Block(ch[2], ch[2], emb), Block(ch[2], ch[2], emb)
        self.attn = TemporalAttention(ch[2]) if attention else nn.Identity()
        self.u2, self.u1, self.u0 = Block(2 * ch[2], ch[1], emb), Block(2 * ch[1], ch[0], emb), Block(2 * ch[0], ch[0], emb)
        self.outn = nn.GroupNorm(8, ch[0])
        self.out = SConv(ch[0], 1)

    @staticmethod
    def _down(x):
        b, c, t, h, w = x.shape
        return x.reshape(b, c, t, h // 2, 2, w // 2, 2).mean((4, 6))

    @staticmethod
    def _up(x):
        return x.repeat_interleave(2, 3).repeat_interleave(2, 4)

    def forward(self, x, t, direction):
        """x: B,T,1,H,W; t: B long; direction: B long (NDIR = null), or B,2 [direction, digit] (digit 10 = null)."""
        if direction.ndim == 2:
            direction, digit = direction[:, 0], direction[:, 1]
        else:
            digit = None
        e = self.t_mlp(timestep_embedding(t, self.emb_dim)) + self.dir_emb(direction)
        if self.digit_emb is not None and digit is not None:
            e = e + self.digit_emb(digit)
        h = self.inp(x.permute(0, 2, 1, 3, 4))
        h0 = self.d0(h, e)                    # 32
        h1 = self.d1(self._down(h0), e)       # 16
        h2 = self.d2(self._down(h1), e)       # 8
        m = self.mid2(self.attn(self.mid1(h2, e)), e)
        u = self.u2(torch.cat((m, h2), 1), e)
        u = self.u1(torch.cat((self._up(u), h1), 1), e)
        u = self.u0(torch.cat((self._up(u), h0), 1), e)
        return self.out(F.silu(self.outn(u))).permute(0, 2, 1, 3, 4)
