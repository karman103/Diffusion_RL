"""Compact pretrained-vs-GRPO figure for the blog.

One test clip per requested digit, picked by a fixed rule (index 8*d + d%8, so
each row also asks for a different direction), not by outcome. Same starting
noise for both models. A green/red bar marks the independent evaluator's verdict.

    python -m image_video_grpo.learned_video.digit.blog_figure
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .common import RUNS

ARROWS = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]
FRAMES = [0, 2, 4, 7]


def main(policy: str = "grpo_3102"):
    a = torch.load(RUNS / "eval/pretrained.pt", weights_only=False)
    b = torch.load(RUNS / f"eval/{policy}.pt", weights_only=False)
    rows = [8 * d + d % 8 for d in range(10)]
    fig, axs = plt.subplots(10, 2 * len(FRAMES) + 1, figsize=(7.2, 9.4),
                            gridspec_kw={"width_ratios": [1] * len(FRAMES) + [0.25] + [1] * len(FRAMES)})
    for r, i in enumerate(rows):
        direction, digit = (int(v) for v in a["cond"][i])
        for side, ev in enumerate((a, b)):
            ok = bool(ev["metrics"]["success"][i])
            for k, f in enumerate(FRAMES):
                ax = axs[r, k + side * (len(FRAMES) + 1)]
                ax.imshow(ev["clips"][i, f, 0].float().numpy(), cmap="gray", vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if k == 0:
                    ax.add_patch(plt.Rectangle((-0.5, 29.5), 32, 2.5, color="#2a9d55" if ok else "#c8402f", clip_on=False))
        axs[r, 0].set_ylabel(f"{digit} {ARROWS[direction]}", rotation=0, ha="right", va="center", fontsize=11)
        axs[r, len(FRAMES)].axis("off")
    axs[0, 1].set_title("Pretrained", fontsize=11, loc="left")
    axs[0, len(FRAMES) + 2].set_title("After GRPO", fontsize=11, loc="left")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.01, wspace=0.05, hspace=0.12)
    out = RUNS / "blog_comparison.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
