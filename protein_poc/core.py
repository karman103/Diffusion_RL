"""Genie DDIM policy with an exact, cached final translation-head update.

Upstream weights stay frozen. A 3 x 128 weight delta and 3 biases adapt the final
translation projection. This is equivalent to updating those 387 pretrained
parameters. The deterministic terminal decoder always uses the frozen head.
"""
from pathlib import Path
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/genie"
sys.path.insert(0, str(VENDOR))
from genie.config import Config
from genie.model.model import Denoiser
from genie.diffusion.schedule import get_betas
from genie.utils.affine_utils import T, rot_vec_mul


def frames(x):
    """Vectorized upstream Frenet frames, for unpadded fixed-length chains."""
    t = x[:, 1:] - x[:, :-1]
    t = t / (t.square().sum(-1, keepdim=True) + 1e-10).sqrt()
    b = torch.cross(t[:, :-1], t[:, 1:], dim=-1)
    b = b / (b.square().sum(-1, keepdim=True) + 1e-10).sqrt()
    n = torch.cross(b, t[:, 1:], dim=-1)
    r = torch.stack((t[:, 1:], b, n), dim=-1)
    return torch.cat((r[:, :1], r, r[:, -1:]), dim=1)


class TranslationHead(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, channels))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, features, rotations):
        return rot_vec_mul(rotations, F.linear(features, self.weight, self.bias))


def joint_logprob(x, mean, sigma):
    """Joint density over every coordinate (sum, not a per-coordinate mean)."""
    return (-0.5 * ((x - mean) / sigma).square()
            - math.log(sigma) - 0.5 * math.log(2 * math.pi)).sum((-1, -2))


class ProteinPolicy:
    def __init__(self, device="cpu", steps=100, length=64, active_min=50, active_max=600):
        if not 3 <= length <= 128:
            raise ValueError("This checkpoint supports lengths 3..128; use 50..128 for experiments")
        if not 2 <= steps <= 1000:
            raise ValueError("steps must be in [2, 1000]")
        self.device, self.steps, self.length = device, steps, length
        self.active_min, self.active_max = active_min, active_max
        self.checkpoint = VENDOR / "weights/scope_l_128/epoch=49999.ckpt"
        config = Config(str(self.checkpoint.parent / "configuration"))
        self.model = Denoiser(**config.model, n_timestep=1000)
        state = torch.load(self.checkpoint, weights_only=True, map_location="cpu")["state_dict"]
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
        self.model.to(device).eval().requires_grad_(False)
        self.head = TranslationHead(config.model["c_s"]).to(device)
        self.ab = torch.cumprod(1 - get_betas(1000, "cosine"), dim=0).tolist()
        self.ts = torch.linspace(999, 0, steps).round().long().tolist()
        self.prev = self.ts[1:] + [-1]
        # With a single structure block, these are unaffected by the final head.
        assert self.model.structure_net.n_structure_block == 1
        layer = self.model.structure_net.net[-1]
        layer.register_forward_pre_hook(self._capture_rotations)
        layer.bb_update.linear.register_forward_pre_hook(self._capture_features)

    def _capture_rotations(self, module, args):
        self.rotations = args[0][2].rots.detach()

    def _capture_features(self, module, args):
        self.features = args[0].detach()

    def coefficients(self, t, prev):
        at = self.ab[t]
        ap = self.ab[prev] if prev >= 0 else 1.0
        sigma = math.sqrt(max(0, (1 - ap) / (1 - at) * (1 - at / ap)))
        eps_coeff = math.sqrt(max(0, 1 - ap - sigma ** 2)) - math.sqrt(ap / at) * math.sqrt(1 - at)
        return math.sqrt(ap / at), eps_coeff, sigma

    @torch.no_grad()
    def frozen_stats(self, x, t, prev):
        mask = torch.ones(x.shape[:2], device=x.device)
        tt = torch.full((len(x),), t, dtype=torch.long, device=x.device)
        output = self.model(T(frames(x), x), tt, mask).trans
        a, b, sigma = self.coefficients(t, prev)
        # Genie epsilon prediction = input coordinates - predicted coordinates.
        mean = a * x + b * (x - output)
        return mean, self.features, self.rotations, -b, sigma

    def adapted_mean(self, stats, enabled=True):
        mean, features, rotations, coefficient, _ = stats
        return mean + coefficient * self.head(features, rotations) if enabled else mean

    @torch.no_grad()
    def rollout(self, n, seed, adapted=True, record=False):
        # CPU RNG makes noise explicit and avoids changing training RNG state.
        gen = torch.Generator().manual_seed(seed)
        def noise(shape):
            return torch.randn(shape, generator=gen).to(self.device)
        x = noise((n, self.length, 3))
        records, kl = [], torch.zeros(n, device=self.device)
        for t, prev in zip(self.ts, self.prev):
            stats = self.frozen_stats(x, t, prev)
            refmean, features, rotations, coefficient, sigma = stats
            active = prev >= 0 and self.active_min <= t <= self.active_max
            mean = self.adapted_mean(stats, adapted and active)
            if prev < 0:
                x = mean  # shared deterministic readout: excluded from likelihood
            else:
                xn = mean + sigma * noise(x.shape)
                kl += (mean - refmean).square().sum((-1, -2)) / (2 * sigma ** 2)
                if record and active:
                    records.append(dict(refmean=refmean, features=features, rotations=rotations,
                                        coefficient=coefficient, sigma=sigma, action=xn,
                                        old_logprob=joint_logprob(xn, mean, sigma)))
                x = xn
        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite generated coordinates")
        return x, records, kl


def geometry(x):
    """C-alpha geometry diagnostics; none establish foldability or function."""
    centered = x - x.mean(1, keepdim=True)
    rg = centered.square().sum(-1).mean(-1).sqrt()
    bonds = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
    distances = torch.cdist(x, x)
    ids = torch.arange(x.shape[1], device=x.device)
    mask = (ids[:, None] - ids[None, :]) > 2
    nonlocal_d = distances[:, mask]
    bond_error = (bonds - 3.8).abs().mean(-1)
    bond_ok = ((bonds >= 3.4) & (bonds <= 4.2)).float().mean(-1)
    clash_fraction = (nonlocal_d < 3.0).float().mean(-1)
    geometry_pass = (bond_ok >= .95) & (nonlocal_d.min(-1).values >= 3.0)
    return dict(rg=rg, bond_mae=bond_error, bond_ok=bond_ok, clash_fraction=clash_fraction,
                min_nonlocal_distance=nonlocal_d.min(-1).values,
                geometry_pass=geometry_pass.float())


def reward(x, kind="compact"):
    m = geometry(x)
    # Radius is a transparent, deliberately incomplete proxy objective.
    compact = -m["rg"] / 10
    if kind == "compact":
        return compact
    if kind == "guarded":
        return compact - 2 * m["bond_mae"] - 10 * m["clash_fraction"]
    raise ValueError(kind)


def distance_diversity(x):
    """Mean pairwise RMS difference of within-chain distance maps, in Angstroms."""
    ids = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
    d = torch.cdist(x, x)[:, ids[0], ids[1]]
    if len(x) < 2:
        return float("nan")
    return (torch.pdist(d) / math.sqrt(d.shape[-1])).mean().item()


def save_pdb(path, x):
    """C-alpha-only UNK chain; no designed amino-acid sequence is implied."""
    coords = x.detach().cpu().tolist()
    with open(path, "w") as f:
        f.write("REMARK   1 CA-ONLY GENIE SAMPLE; RESIDUE IDENTITIES NOT DESIGNED\n")
        for i, (a, b, c) in enumerate(coords, 1):
            f.write(f"ATOM  {i:5d}  CA  UNK A{i:4d}    {a:8.3f}{b:8.3f}{c:8.3f}  1.00  0.00           C\n")
        f.write("TER\nEND\n")
