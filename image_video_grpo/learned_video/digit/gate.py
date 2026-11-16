"""Predeclared gate for the digit-conditioned model (one clip per (direction, digit))."""
from __future__ import annotations

import json

import torch

from .. import classifiers
from ..diffusion import alphas_bar
from ..viz import save_grid
from .common import PROT, RUNS, eval_conditions, evaluate
from .grpo import DEV, generate, load_pretrained


def main():
    g = PROT["gate"]
    if (RUNS / "gate.json").exists():
        raise SystemExit("gate already run")
    cond = eval_conditions(g["n"])
    clips = generate(load_pretrained(), cond, [g["seed"] + i for i in range(g["n"])], alphas_bar(DEV))
    _, ev = classifiers.load()
    m = evaluate(clips, cond, ev)
    res = {"correct_digit_fraction": float(m["correct_digit"].mean()),
           "direction_ok_fraction": float(m["direction_ok"].mean()),
           "success_fraction": float(m["success"].mean()),
           "median_displacement_px": float(m["xcorr_disp"].norm(dim=-1).median())}
    res["pass"] = (res["correct_digit_fraction"] >= g["require_correct_digit_fraction"]
                   and res["median_displacement_px"] >= g["require_median_displacement_px"])
    (RUNS / "gate.json").write_text(json.dumps(res, indent=2))
    sel = torch.arange(0, 80, 9)[:10]  # spread over digits and directions
    save_grid(clips[sel], cond[sel, 0], RUNS / "gate_grid.png", rows_label=[f"digit {int(cond[i,1])}" for i in sel])
    print(res)


if __name__ == "__main__":
    main()
