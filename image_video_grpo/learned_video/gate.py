"""Predeclared content/motion gate on the pretrained EMA denoiser. GRPO runs only if it passes.
`--preview` samples the latest intermediate EMA checkpoint and is recorded as a preview, not the gate."""
from __future__ import annotations

import argparse
import json

import torch

from . import classifiers, metrics
from .data import NDIR
from .grpo import DEV, RUNS, STEPS, PROT, generate
from .diffusion import alphas_bar
from .model import VideoUNet
from .viz import save_grid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); a = ap.parse_args()
    g = PROT["gate"]
    net = VideoUNet()
    if a.preview:
        ck = torch.load(RUNS / "pretrain_ckpt.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["ema"]); tag = f"preview_step{ck['step']}"
    else:
        if (RUNS / "gate.json").exists():
            raise SystemExit("gate already run")
        net.load_state_dict(torch.load(RUNS / "pretrained_ema.pt", map_location="cpu", weights_only=True)); tag = "gate"
    net = net.to(DEV).eval()
    direction = torch.arange(g["n"]) % NDIR
    clips = generate(net, direction, [g["seed"] + i for i in range(g["n"])], alphas_bar(DEV))
    _, ev = classifiers.load()
    m = metrics.evaluate(clips, direction, ev)
    disp = m["xcorr_disp"].norm(dim=-1)
    res = {"tag": tag, "sampler_steps": STEPS,
           "legible_fraction": float(m["legible"].mean()),
           "median_displacement_px": float(disp.median()),
           "direction_ok_fraction": float(m["direction_ok"].mean()),
           "success_fraction": float(m["success"].mean()),
           "median_angle_err_deg": float(m["angle_err"].median()),
           "digit_class_entropy": metrics.class_entropy(m["digit"])}
    res["pass"] = (res["legible_fraction"] >= g["require_legible_fraction"]
                   and res["median_displacement_px"] >= g["require_median_displacement_px"])
    (RUNS / f"{tag}.json").write_text(json.dumps(res, indent=2))
    save_grid(clips[:16], direction[:16], RUNS / f"{tag}_grid.png")
    print(res)


if __name__ == "__main__":
    main()
