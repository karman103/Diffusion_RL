"""Warm-start the direction model into a (direction, digit) conditioned model."""
from __future__ import annotations

import copy
import json
import time

import torch
from torch import nn

from ..data import NDIR, make_clips, mnist
from ..diffusion import alphas_bar, train_loss
from ..model import VideoUNet
from .common import HERE, PROT, RUNS

P = PROT["pretrain"]


def main():
    RUNS.mkdir(exist_ok=True)
    if (RUNS / "pretrain_done.json").exists():
        raise SystemExit("done")
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x, y = mnist("train")
    net = VideoUNet(digits=True)
    nn.init.normal_(net.digit_emb.weight, std=1.0)   # from scratch: same init as the direction embedding
    net = net.to(dev)
    ema = copy.deepcopy(net).requires_grad_(False)
    emb = [net.digit_emb.weight]
    rest = [p for n, p in net.named_parameters() if not n.startswith("digit_emb")]
    opt = torch.optim.AdamW([{"params": rest, "lr": P["lr"], "base": P["lr"]},
                             {"params": emb, "lr": P["digit_emb_lr"], "base": P["digit_emb_lr"]}], weight_decay=0.0)
    ab = alphas_bar(dev)
    gen = torch.Generator().manual_seed(2)
    ck = RUNS / "pretrain_ckpt.pt"
    start = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        start = st["step"]; gen.set_state(st["gen"].cpu())
    log = open(RUNS / "pretrain.jsonl", "a" if start else "w")
    t0, run = time.time(), None
    for step in range(start, P["steps"]):
        idx = torch.randint(0, x.shape[0], (P["batch"],), generator=gen)
        d = torch.randint(0, NDIR, (P["batch"],), generator=gen)
        clips = make_clips(x[idx], d, gen).to(dev)
        dc = torch.where(torch.rand(P["batch"], generator=gen) < 0.1, torch.full_like(d, NDIR), d)
        gc = torch.where(torch.rand(P["batch"], generator=gen) < 0.1, torch.full_like(d, 10), y[idx])
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (step + 1) / P["warmup"])
        loss = train_loss(net, clips, torch.stack((dc, gc), 1).to(dev), ab)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - P["ema"])
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if (step + 1) % 50 == 0:
            log.write(json.dumps({"step": step + 1, "loss_ema": run, "elapsed_s": time.time() - t0}) + "\n"); log.flush()
        if (step + 1) % 500 == 0:
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "gen": gen.get_state()}, ck)
    torch.save(ema.state_dict(), RUNS / "pretrained_ema.pt")
    (RUNS / "pretrain_done.json").write_text(json.dumps({"steps": P["steps"], "final_loss_ema": run}))


if __name__ == "__main__":
    main()
