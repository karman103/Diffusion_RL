"""GRPO on the stochastic DDIM(eta=1) transitions of the locally pretrained video denoiser.

Per-step log-ratios are averaged over pixels, as in the public Flow-GRPO implementation,
and the KL to the frozen pretrained model is the closed-form Gaussian KL (same sigma),
also averaged over pixels.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from ..core import group_advantages
from . import classifiers, metrics
from .data import NDIR
from .diffusion import alphas_bar, sample, step_mean_std
from .model import VideoUNet

HERE = Path(__file__).resolve().parent
import os
RUNS = Path(os.environ.get("LV_RUNS", HERE / "runs"))
PROT = json.loads((HERE / "PROTOCOL.json").read_text())
G = dict(PROT["grpo"], **json.loads(os.environ.get("LV_GRPO_OVERRIDE", "{}")))
STEPS = int(os.environ.get("LV_STEPS", 40))
DEV = os.environ.get("LV_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")


def load_pretrained():
    net = VideoUNet()
    net.load_state_dict(torch.load(RUNS / "pretrained_ema.pt", map_location="cpu", weights_only=True))
    return net.to(DEV).eval()


def mean_lp(action, mean, sigma):
    """Gaussian log density averaged over pixels (constant terms included)."""
    z = ((action - mean) / sigma).square()
    return (-0.5 * z - torch.log(sigma) - 0.9189385332046727).flatten(1).mean(1)


def train(seed: int):
    out = RUNS / f"grpo_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise SystemExit("already trained")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    ab = alphas_bar(DEV)
    policy, ref = load_pretrained(), load_pretrained().requires_grad_(False)
    proxy, _ = classifiers.load()
    opt = torch.optim.AdamW(policy.parameters(), G["lr"], weight_decay=0.0)
    log = open(out / "train.jsonl", "w")
    for upd in range(G["updates"]):
        t0 = time.time()
        dirs = torch.randperm(NDIR, generator=gen)[:G["directions_per_update"]]
        direction = dirs.repeat_interleave(G["group"]).to(DEV)
        base = int(torch.randint(0, 2**30, (1,), generator=gen))
        policy.eval()
        clips, traj = sample(policy, direction, base, ab, steps=STEPS, record=True, device=DEV)
        r = metrics.reward(clips, direction, proxy)
        if not torch.isfinite(r).all():
            raise RuntimeError("non-finite reward")
        adv = torch.cat([group_advantages(g) for g in r.split(G["group"])]).to(DEV)
        pick = torch.randperm(len(traj), generator=gen)[:G["train_timesteps_per_update"]].tolist()
        stats = {"clip": [], "max_log_ratio": [], "kl": [], "grad": []}
        policy.train()
        snap = copy.deepcopy(policy).requires_grad_(False).eval()
        for k in pick:
            t, tp, s, a, _ = traj[k]
            with torch.no_grad():
                m_old, sigma, _ = step_mean_std(snap, s, t, tp, direction, ab)
                m_ref, _, _ = step_mean_std(ref, s, t, tp, direction, ab)
                lp_old = mean_lp(a, m_old, sigma)
            m_new, _, _ = step_mean_std(policy, s, t, tp, direction, ab)
            lp_new = mean_lp(a, m_new, sigma)
            delta = lp_new - lp_old
            ratio = torch.exp(delta.clamp(-20, 20))
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - G["clip"], 1 + G["clip"]) * adv).mean()
            kl = ((m_new - m_ref).square() / (2 * sigma ** 2)).flatten(1).mean(1).mean()
            loss = pg + G["kl_beta"] * kl
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            stats["clip"].append(float(((ratio < 1 - G["clip"]) | (ratio > 1 + G["clip"])).float().mean()))
            stats["max_log_ratio"].append(float(delta.detach().abs().max()))
            stats["kl"].append(float(kl.detach())); stats["grad"].append(float(gn))
        del snap
        row = {"update": upd + 1, "reward_mean": float(r.mean()), "reward_std": float(r.std()),
               "clip_frac": sum(stats["clip"]) / len(pick), "max_abs_log_ratio": max(stats["max_log_ratio"]),
               "kl_ref": sum(stats["kl"]) / len(pick), "grad_norm": sum(stats["grad"]) / len(pick),
               "seconds": time.time() - t0}
        log.write(json.dumps(row) + "\n"); log.flush()
    torch.save(policy.state_dict(), out / "policy.pt")
    (out / "done.json").write_text(json.dumps({"updates": G["updates"]}))


@torch.no_grad()
def generate(net, direction, seeds, ab, cfg=0.0, bs=64):
    outs = []
    for i in range(0, len(seeds), bs):
        x, _ = sample(net, direction[i:i + bs].to(DEV), seeds[i:i + bs], ab, steps=STEPS, cfg=cfg, device=DEV)
        outs.append(x.cpu())
    return torch.cat(outs)


def evaluate(name: str):
    """name: pretrained | cfg2 | best4 | grpo_<seed>. Saves raw clips and per-clip measures."""
    E = PROT["evaluation"]
    path = RUNS / "eval" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("already evaluated")
    ab = alphas_bar(DEV)
    n = E["n"]
    direction = torch.arange(n) % NDIR
    seeds = [E["seed"] + i for i in range(n)]
    proxy, ev = classifiers.load()
    net = load_pretrained()
    if name.startswith("grpo_"):
        net.load_state_dict(torch.load(RUNS / name / "policy.pt", map_location=DEV, weights_only=True))
    if name == "best4":
        cands = [generate(net, direction, [s + 100000 * k for s in seeds], ab) if k else
                 generate(net, direction, seeds, ab) for k in range(4)]
        rs = torch.stack([metrics.reward(c, direction, proxy) for c in cands])     # 4,n
        best = rs.argmax(0)
        clips = torch.stack([cands[best[i]][i] for i in range(n)])
    else:
        clips = generate(net, direction, seeds, ab, cfg=2.0 if name == "cfg2" else 0.0)
    m = metrics.evaluate(clips, direction, ev)
    m["train_reward"] = metrics.reward(clips, direction, proxy)
    torch.save({"clips": clips.half(), "direction": direction, "metrics": m}, path)
    print(name, {k: float(v.float().mean()) for k, v in m.items() if k not in ("xcorr_disp", "digit")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--name")
    a = ap.parse_args()
    train(a.seed) if a.cmd == "train" else evaluate(a.name)
