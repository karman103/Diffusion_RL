"""GRPO for the (direction, digit) conditioned model. Same algorithm and hyperparameters as ../grpo.py;
only the conditioning, reward and evaluation layout differ."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from ...core import group_advantages
from .. import classifiers
from ..data import NDIR
from ..diffusion import alphas_bar, sample, step_mean_std
from ..grpo import mean_lp
from ..model import VideoUNet
from .common import PROT, RUNS, eval_conditions, evaluate, reward

G = dict(PROT["grpo"], **json.loads(os.environ.get("LVD_GRPO_OVERRIDE", "{}")))
STEPS = int(os.environ.get("LV_STEPS", 40))
DEV = os.environ.get("LV_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")


def load_pretrained():
    net = VideoUNet(digits=True)
    net.load_state_dict(torch.load(RUNS / "pretrained_ema.pt", map_location="cpu", weights_only=True))
    return net.to(DEV).eval()


def train(seed):
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
        prompts = torch.stack((torch.randint(0, NDIR, (G["prompts_per_update"],), generator=gen),
                               torch.randint(0, 10, (G["prompts_per_update"],), generator=gen)), 1)
        cond = prompts.repeat_interleave(G["group"], 0).to(DEV)
        base = int(torch.randint(0, 2**30, (1,), generator=gen))
        clips, traj = sample(policy, cond, base, ab, steps=STEPS, record=True, device=DEV)
        r = reward(clips, cond, proxy)
        if not torch.isfinite(r).all():
            raise RuntimeError("non-finite reward")
        adv = torch.cat([group_advantages(x) for x in r.split(G["group"])]).to(DEV)
        pick = torch.randperm(len(traj), generator=gen)[:G["train_timesteps_per_update"]].tolist()
        snap = copy.deepcopy(policy).requires_grad_(False).eval()
        st = {"clip": [], "lr": [], "kl": [], "grad": []}
        for k in pick:
            t, tp, s, a, _ = traj[k]
            with torch.no_grad():
                m_old, sigma, _ = step_mean_std(snap, s, t, tp, cond, ab)
                m_ref, _, _ = step_mean_std(ref, s, t, tp, cond, ab)
                lp_old = mean_lp(a, m_old, sigma)
            m_new, _, _ = step_mean_std(policy, s, t, tp, cond, ab)
            delta = mean_lp(a, m_new, sigma) - lp_old
            ratio = torch.exp(delta.clamp(-20, 20))
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - G["clip"], 1 + G["clip"]) * adv).mean()
            kl = ((m_new - m_ref).square() / (2 * sigma ** 2)).flatten(1).mean(1).mean()
            loss = pg + G["kl_beta"] * kl
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            st["clip"].append(float(((ratio < 1 - G["clip"]) | (ratio > 1 + G["clip"])).float().mean()))
            st["lr"].append(float(delta.detach().abs().max())); st["kl"].append(float(kl.detach())); st["grad"].append(float(gn))
        del snap
        log.write(json.dumps({"update": upd + 1, "reward_mean": float(r.mean()), "reward_std": float(r.std()),
                              "clip_frac": sum(st["clip"]) / len(pick), "max_abs_log_ratio": max(st["lr"]),
                              "kl_ref": sum(st["kl"]) / len(pick), "grad_norm": sum(st["grad"]) / len(pick),
                              "seconds": time.time() - t0}) + "\n"); log.flush()
    torch.save(policy.state_dict(), out / "policy.pt")
    (out / "done.json").write_text(json.dumps({"updates": G["updates"]}))


@torch.no_grad()
def generate(net, cond, seeds, ab, cfg=0.0, bs=64):
    outs = []
    for i in range(0, len(seeds), bs):
        x, _ = sample(net, cond[i:i + bs].to(DEV), seeds[i:i + bs], ab, steps=STEPS, cfg=cfg, device=DEV)
        outs.append(x.cpu())
    return torch.cat(outs)


def run_eval(name):
    E = PROT["evaluation"]
    path = RUNS / "eval" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("already evaluated")
    ab = alphas_bar(DEV)
    cond = eval_conditions(E["n"])
    seeds = [E["seed"] + i for i in range(E["n"])]
    proxy, ev = classifiers.load()
    net = load_pretrained()
    if name.startswith("grpo_"):
        net.load_state_dict(torch.load(RUNS / name / "policy.pt", map_location=DEV, weights_only=True))
    if name == "best4":
        cands = [generate(net, cond, [s + 100000 * k for s in seeds], ab) for k in range(4)]
        best = torch.stack([reward(c, cond, proxy) for c in cands]).argmax(0)
        clips = torch.stack([cands[best[i]][i] for i in range(E["n"])])
    else:
        clips = generate(net, cond, seeds, ab, cfg=2.0 if name == "cfg2" else 0.0)
    m = evaluate(clips, cond, ev)
    m["train_reward"] = reward(clips, cond, proxy)
    torch.save({"clips": clips.half(), "cond": cond, "metrics": m}, path)
    print(name, {k: round(float(m[k].float().mean()), 4) for k in ["success", "correct_digit", "direction_ok", "legible", "train_reward"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--seed", type=int); ap.add_argument("--name"); a = ap.parse_args()
    train(a.seed) if a.cmd == "train" else run_eval(a.name)
