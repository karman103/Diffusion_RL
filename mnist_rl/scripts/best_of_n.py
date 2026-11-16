"""Inference-time baseline: best-of-N sampling from the *pretrained* model w.r.t. the proxy reward.
KL(best-of-N || base) = log N - (N-1)/N (Stiennon et al. 2020 / Gao et al. 2022)."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F
from rldiff.models import UNet
from rldiff.diffusion import Diffusion
from rldiff.rewards import load_clfs

dev = "mps"
torch.manual_seed(0)
diff = Diffusion(device=dev, clip_x0=1, t_min=20)
m = UNet().to(dev); m.load_state_dict(torch.load("runs/base/ckpt.pt", map_location="cpu")["ema"]); m.eval()
gold, feat, proxy = load_clfs(dev)
M, target = 4096, 7
xs = torch.cat([diff.sample(m, 256, n_steps=20) for _ in range(M // 256)])
with torch.no_grad():
    rp = torch.cat([F.log_softmax(proxy(xs[i:i+512]), 1)[:, target] for i in range(0, M, 512)]).cpu()
    pg = torch.cat([F.softmax(gold(xs[i:i+512]), 1) for i in range(0, M, 512)]).cpu()
torch.save({"x": xs.cpu(), "proxy": rp, "gold": pg}, "runs/base/base_samples.pt")
res = []
g = torch.Generator().manual_seed(0)
for N in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    trials = 2000
    idx = torch.randint(0, M, (trials, N), generator=g)
    best = idx.gather(1, rp[idx].argmax(1, keepdim=True)).squeeze(1)
    res.append(dict(N=N, kl=math.log(N) - (N - 1) / N, proxy_reward=rp[best].mean().item(),
                    gold_p=pg[best, target].mean().item(), gold_acc=(pg[best].argmax(1) == target).float().mean().item()))
    print(res[-1])
json.dump(res, open("runs/base/best_of_n.json", "w"), indent=1)
print("base frac of 7s (gold):", (pg.argmax(1) == target).float().mean().item())
