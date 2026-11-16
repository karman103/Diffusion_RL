"""RL fine-tuning of a pretrained diffusion model.

Methods
  ddpo     : DDPO-IS  (PPO-clipped importance-sampled policy gradient over the denoising MDP)
  ddpo_sf  : DDPO-SF  (REINFORCE / score-function, one on-policy step per batch)
  rwr      : reward-weighted regression (filter-and-finetune style, Lee et al. 2023)
  draft    : DRaFT-K  (backprop the reward gradient through the last K sampling steps)

Any method can add a KL(policy || pretrained) penalty (DPOK-style, closed-form per-step
Gaussian KL on the visited states) with --kl beta.
"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
from rldiff.models import UNet
from rldiff.diffusion import Diffusion
from rldiff.rewards import Reward, load_clfs, frechet_distance, diversity, jpeg_size_kb

p = argparse.ArgumentParser()
p.add_argument("--method", default="ddpo", choices=["ddpo", "ddpo_sf", "rwr", "draft"])
p.add_argument("--reward", default="proxy:7")
p.add_argument("--target", type=int, default=7, help="class used for gold eval / FD reference")
p.add_argument("--kl", type=float, default=0.0)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--mb", type=int, default=32)
p.add_argument("--inner", type=int, default=1)
p.add_argument("--n_steps", type=int, default=20)
p.add_argument("--t_frac", type=float, default=0.5, help="fraction of timesteps trained per trajectory")
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--clip", type=float, default=1e-4 * 0 + 0.2)
p.add_argument("--adv_clip", type=float, default=5.0)
p.add_argument("--rwr_temp", type=float, default=0.5)
p.add_argument("--draft_k", type=int, default=1)
p.add_argument("--eval_every", type=int, default=10)
p.add_argument("--n_eval", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--clip_x0", type=int, default=1)
p.add_argument("--t_min", type=int, default=20)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(a.seed)

diff = Diffusion(device=dev, clip_x0=a.clip_x0, t_min=a.t_min)
ref = UNet().to(dev)
ref.load_state_dict(torch.load("runs/base/ckpt.pt", map_location="cpu")["ema"])
ref.eval().requires_grad_(False)
policy = copy.deepcopy(ref).requires_grad_(True).train()
opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)
gold, feat, proxy = load_clfs(dev)
R = Reward(a.reward, gold, proxy)

# fixed evaluation noise + reference features of real target-class digits
g = torch.Generator(device="cpu").manual_seed(1234)
eval_noise = torch.randn(a.n_eval, 1, 28, 28, generator=g).to(dev)
te = torchvision.datasets.MNIST("data", train=False)
real = (te.data[te.targets == a.target][:1000].float() / 127.5 - 1).unsqueeze(1).to(dev)
real_all = (te.data[:2000].float() / 127.5 - 1).unsqueeze(1).to(dev)
with torch.no_grad():
    real_feat, real_all_feat = feat.features(real), feat.features(real_all)


def step_kl(x_t, t, tp, mean):
    """Per-sample KL(N(mean, s^2) || N(mean_ref, s^2)) summed over pixels (nats)."""
    with torch.no_grad():
        mref, s, _ = diff.step_mean_std(ref, x_t, t, tp)
    return ((mean - mref) ** 2).sum(dim=(1, 2, 3)) / (2 * s ** 2)


@torch.no_grad()
def evaluate(epoch):
    policy.eval()
    ts, prevs = diff.timesteps(a.n_steps)
    x = eval_noise
    kl = torch.zeros(a.n_eval, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(99)
    for t, tp in zip(ts, prevs):
        mean, s, x0 = diff.step_mean_std(policy, x, t, tp)
        if tp < 0:
            x = x0
        else:
            mref, _, _ = diff.step_mean_std(ref, x, t, tp)
            kl += ((mean - mref) ** 2).sum(dim=(1, 2, 3)) / (2 * s ** 2)
            x = mean + s * torch.randn(x.shape, generator=gen).to(dev)
    r = R(x)
    pg = F.softmax(gold(x), 1)
    pp = F.softmax(proxy(x), 1)
    f = feat.features(x)
    m = dict(
        epoch=epoch,
        reward=r.mean().item(), reward_std=r.std().item(),
        proxy_p=pp[:, a.target].mean().item(),
        gold_p=pg[:, a.target].mean().item(),
        gold_acc=(pg.argmax(1) == a.target).float().mean().item(),
        gold_conf=pg.max(1).values.mean().item(),
        class_entropy=float(-(pg.mean(0) * pg.mean(0).clamp_min(1e-12).log()).sum()),
        jpeg_kb=jpeg_size_kb(x).mean().item(),
        fd_target=frechet_distance(f, real_feat),
        fd_all=frechet_distance(f, real_all_feat),
        diversity=diversity(f),
        kl=kl.mean().item(),
        mean_ink=((x + 1) / 2).mean().item(),
    )
    save_image((x[:64] + 1) / 2, f"{a.out}/samples_{epoch:04d}.png", nrow=8, padding=1)
    policy.train()
    return m


log = open(f"{a.out}/metrics.jsonl", "w")
t0 = time.time()
for epoch in range(a.epochs + 1):
    if epoch % a.eval_every == 0 or epoch == a.epochs:
        m = evaluate(epoch); m["time"] = time.time() - t0
        log.write(json.dumps(m) + "\n"); log.flush()
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
    if epoch == a.epochs:
        break

    if a.method == "draft":
        # ---- direct reward backprop through the last K steps ----
        for i in range(0, a.bs, a.mb):
            x = diff.sample_with_grad(policy, a.mb, a.n_steps, grad_last_k=a.draft_k)
            loss = -R(x).mean() / (a.bs // a.mb)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        continue

    # ---- collect on-policy trajectories ----
    policy.eval()
    x0, xs, lps, ts, prevs = diff.sample(policy, a.bs, a.n_steps, return_traj=True)
    policy.train()
    with torch.no_grad():
        r = R(x0)
    adv = ((r - r.mean()) / (r.std() + 1e-6)).clamp(-a.adv_clip, a.adv_clip)

    if a.method == "rwr":
        w = torch.softmax(r / (a.rwr_temp * (r.std() + 1e-6)), 0) * a.bs  # mean weight 1
        for _ in range(a.inner):
            for i in range(0, a.bs, a.mb):
                loss = diff.loss(policy, x0[i:i + a.mb].detach(), weights=w[i:i + a.mb])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        continue

    # ---- DDPO (IS w/ PPO clip, or SF) over the denoising MDP ----
    n_trans = lps.shape[1]
    clip = a.clip if a.method == "ddpo" else float("inf")
    inner = a.inner if a.method == "ddpo" else 1
    for _ in range(inner):
        perm = torch.randperm(a.bs, device=dev)
        for i in range(0, a.bs, a.mb):
            idx = perm[i:i + a.mb]
            steps = torch.randperm(n_trans)[: max(1, int(round(a.t_frac * n_trans)))].tolist()
            opt.zero_grad(set_to_none=True)
            for j in steps:
                xt, xn = xs[idx, j], xs[idx, j + 1]
                mean, s, _ = diff.step_mean_std(policy, xt, ts[j], prevs[j])
                lp = diff.gaussian_logprob(xn, mean, s)
                ratio = torch.exp(lp - lps[idx, j])
                A = adv[idx]
                pg = torch.max(-A * ratio, -A * ratio.clamp(1 - clip, 1 + clip)).mean()
                loss = pg
                if a.kl > 0:
                    loss = loss + a.kl * step_kl(xt, ts[j], prevs[j], mean).mean()
                (loss / len(steps)).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

torch.save(policy.state_dict(), f"{a.out}/policy.pt")
print("done", time.time() - t0)
