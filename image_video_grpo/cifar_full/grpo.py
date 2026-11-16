"""Full-UNet GRPO on the stock CIFAR-10 DDPM sampler (frog reward)."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from diffusers import DDPMScheduler

from ..core import group_advantages
from ..models import load_image
from ..rewards import CifarClassifier

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
P = json.loads((HERE / "PROTOCOL.json").read_text())
G = P["grpo"]
RUNS = HERE / "runs"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
FROG = 6


def classifiers():
    out = []
    for n in ("proxy", "gold"):
        m = CifarClassifier(48)
        m.load_state_dict(torch.load(PARENT / f"strong_reward/{n}.pt", map_location="cpu", weights_only=True))
        out.append(m.to(DEV).eval().requires_grad_(False))
    return out


class Sched:
    def __init__(self):
        cfg = json.loads((PARENT / "checkpoints/cifar_scheduler.json").read_text())
        s = DDPMScheduler.from_config(cfg); s.set_timesteps(50)
        self.times = s.timesteps.tolist()
        beta = torch.linspace(cfg["beta_start"], cfg["beta_end"], cfg["num_train_timesteps"])
        self.ab = torch.cumprod(1 - beta, 0).to(DEV)

    def mean_std(self, net, x, i):
        t = self.times[i]; prev = self.times[i + 1] if i + 1 < len(self.times) else -1
        eps = net(x, torch.full((len(x),), t, device=x.device, dtype=torch.long)).sample
        at = self.ab[t]; ap = self.ab[prev] if prev >= 0 else torch.ones_like(at)
        bt, bp = 1 - at, 1 - ap; a = at / ap; b = 1 - a
        x0 = ((x - bt.sqrt() * eps) / at.sqrt()).clamp(-1, 1)
        mean = ap.sqrt() * b / bt * x0 + a.sqrt() * bp / bt * x
        return mean, (b.sqrt() if prev >= 0 else None)


@torch.no_grad()
def sample(net, sch, seeds, record=False):
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    x = torch.stack([torch.randn(3, 32, 32, generator=g) for g in gens]).to(DEV)
    traj = []
    for i in range(len(sch.times)):
        mean, std = sch.mean_std(net, x, i)
        if std is None:
            x = mean; break
        nxt = mean + std * torch.stack([torch.randn(3, 32, 32, generator=g) for g in gens]).to(DEV)
        if record:
            traj.append((i, x, nxt, std))
        x = nxt
    return x.clamp(-1, 1), traj


@torch.no_grad()
def reward(x, proxy):
    return 0.5 * (proxy(x).softmax(1)[:, FROG] + proxy(x.flip(-1)).softmax(1)[:, FROG])


def mean_lp(a, m, s):
    return (-0.5 * ((a - m) / s).square() - torch.log(s) - 0.9189385332046727).flatten(1).mean(1)


def train(seed):
    out = RUNS / f"grpo_{seed}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise SystemExit("done")
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    sch = Sched(); proxy, _ = classifiers()
    policy = load_image(DEV).requires_grad_(True); ref = load_image(DEV).requires_grad_(False).eval()
    opt = torch.optim.AdamW(policy.parameters(), G["lr"], weight_decay=0.0)
    log = open(out / "train.jsonl", "w")
    for upd in range(G["updates"]):
        t0 = time.time()
        base = int(torch.randint(0, 2**30, (1,), generator=gen))
        policy.eval()
        x, traj = sample(policy, sch, [base + k for k in range(G["group"])], record=True)
        r = reward(x, proxy)
        adv = group_advantages(r.cpu()).to(DEV)
        snap = copy.deepcopy(policy).requires_grad_(False).eval()
        pick = torch.randperm(len(traj), generator=gen)[:G["train_timesteps_per_update"]].tolist()
        st = {"clip": [], "kl": [], "lr": []}
        policy.train()
        for k in pick:
            i, s, a, std = traj[k]
            with torch.no_grad():
                m_old, _ = sch.mean_std(snap, s, i); m_ref, _ = sch.mean_std(ref, s, i)
                lp_old = mean_lp(a, m_old, std)
            m_new, _ = sch.mean_std(policy, s, i)
            delta = mean_lp(a, m_new, std) - lp_old
            ratio = torch.exp(delta.clamp(-20, 20))
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - G["clip"], 1 + G["clip"]) * adv).mean()
            kl = ((m_new - m_ref).square() / (2 * std ** 2)).flatten(1).mean(1).mean()
            loss = pg + G["kl_beta"] * kl
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), G["grad_clip"]); opt.step()
            st["clip"].append(float(((ratio < 1 - G["clip"]) | (ratio > 1 + G["clip"])).float().mean()))
            st["kl"].append(float(kl.detach())); st["lr"].append(float(delta.detach().abs().max()))
        del snap
        log.write(json.dumps({"update": upd + 1, "reward_mean": float(r.mean()), "clip_frac": sum(st["clip"]) / len(pick),
                              "kl_ref": sum(st["kl"]) / len(pick), "max_abs_log_ratio": max(st["lr"]),
                              "seconds": time.time() - t0}) + "\n"); log.flush()
    torch.save(policy.state_dict(), out / "policy.pt")
    (out / "done.json").write_text(json.dumps({"updates": G["updates"]}))


def evaluate(name):
    E = P["evaluation"]; path = RUNS / "eval" / f"{name}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("done")
    sch = Sched(); proxy, gold = classifiers(); net = load_image(DEV).eval()
    if name.startswith("grpo_"):
        net.load_state_dict(torch.load(RUNS / name / "policy.pt", map_location=DEV, weights_only=True))
    seeds = [E["seed"] + i for i in range(E["n"])]
    def gen_all(sd):
        return torch.cat([sample(net, sch, sd[i:i + 64])[0] for i in range(0, len(sd), 64)])
    if name == "best4":
        c = [gen_all([s + 1000000 * k for s in seeds]) for k in range(4)]
        best = torch.stack([reward(x, proxy) for x in c]).argmax(0)
        x = torch.stack([c[best[i]][i] for i in range(len(seeds))])
    else:
        x = gen_all(seeds)
    with torch.no_grad():
        g = gold(x).softmax(1)
        m = {"reward": reward(x, proxy).cpu(), "gold_probability": g[:, FROG].cpu(),
             "gold_accuracy": (g.argmax(1) == FROG).float().cpu(), "gold_features": gold.features(x).cpu()}
    torch.save({"images": x.cpu().half(), "metrics": m}, path)
    print(name, {k: round(float(v.mean()), 4) for k, v in m.items() if k != "gold_features"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--seed", type=int); ap.add_argument("--name"); a = ap.parse_args()
    train(a.seed) if a.cmd == "train" else evaluate(a.name)
