"""Supervised epsilon-prediction pretraining on fresh Moving-MNIST clips (EMA kept for sampling)."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from .data import NDIR, make_clips, mnist
from .diffusion import alphas_bar, train_loss
from .model import VideoUNet

OUT = Path(__file__).resolve().parent / "runs"
P = json.loads((Path(__file__).resolve().parent / "PROTOCOL.json").read_text())["pretrain"]


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "pretrain_done.json").exists():
        raise SystemExit("pretraining already completed")
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x, _ = mnist("train")
    net = VideoUNet().to(dev)
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), P["lr"], weight_decay=0.0)
    ab = alphas_bar(dev)
    gen = torch.Generator().manual_seed(1)
    start, log = 0, open(OUT / "pretrain.jsonl", "a")
    ck = OUT / "pretrain_ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        start = s["step"]; gen.set_state(s["gen"])
    t0, run = time.time(), 0.0
    for step in range(start, P["steps"]):
        idx = torch.randint(0, x.shape[0], (P["batch"],), generator=gen)
        d = torch.randint(0, NDIR, (P["batch"],), generator=gen)
        clips = make_clips(x[idx], d, gen).to(dev)
        cond = torch.where(torch.rand(P["batch"], generator=gen) < 0.1, torch.full_like(d, NDIR), d).to(dev)
        for g in opt.param_groups:
            g["lr"] = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        loss = train_loss(net, clips, cond, ab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - P["ema"])
        run = 0.98 * run + 0.02 * loss.item() if step > start else loss.item()
        if (step + 1) % 50 == 0:
            log.write(json.dumps({"step": step + 1, "loss_ema": run, "elapsed_s": time.time() - t0}) + "\n"); log.flush()
        if (step + 1) % 1000 == 0 or step + 1 == P["steps"]:
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "gen": gen.get_state()}, ck)
    torch.save(ema.state_dict(), OUT / "pretrained_ema.pt")
    (OUT / "pretrain_done.json").write_text(json.dumps({"steps": P["steps"], "final_loss_ema": run}))


if __name__ == "__main__":
    main()
