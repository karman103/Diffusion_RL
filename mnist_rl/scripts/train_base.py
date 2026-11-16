"""Pretrain an unconditional DDPM on MNIST (the 'reference policy' for RL)."""
import argparse, copy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torchvision
from torchvision.utils import save_image
from rldiff.models import UNet
from rldiff.diffusion import Diffusion

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=15000)
p.add_argument("--bs", type=int, default=96)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--out", default="runs/base")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

ds = torchvision.datasets.MNIST("data", train=True, download=False)
X = (ds.data.float() / 127.5 - 1).unsqueeze(1).to(dev)  # whole dataset on device
model = UNet().to(dev)
ema = copy.deepcopy(model).eval().requires_grad_(False)
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")
diff = Diffusion(device=dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 1000))
t0 = time.time()
for step in range(1, a.steps + 1):
    idx = torch.randint(0, X.shape[0], (a.bs,), device=dev)
    loss = diff.loss(model, X[idx])
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    with torch.no_grad():
        d = min(0.9995, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - d)
    if step % 200 == 0:
        print(f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    if step % 2500 == 0 or step == a.steps:
        x = diff.sample(ema, 64, n_steps=20)
        save_image((x + 1) / 2, f"{a.out}/samples_{step}.png", nrow=8)
        torch.save({"ema": ema.state_dict(), "step": step}, f"{a.out}/ckpt.pt")
print("done")
