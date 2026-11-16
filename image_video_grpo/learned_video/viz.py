"""Clip grids: one row per clip, 8 frames left to right, requested direction as an arrow."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import DIRS


def save_grid(clips, direction, path, title=None, rows_label=None):
    clips = ((clips.float().clamp(-1, 1) + 1) / 2).cpu()
    n, t = clips.shape[:2]
    fig, ax = plt.subplots(n, t + 1, figsize=(t + 1, n), squeeze=False)
    for i in range(n):
        u = DIRS[int(direction[i])]
        ax[i, 0].arrow(0.5 - 0.3 * u[0], 0.5 + 0.3 * u[1], 0.6 * u[0], -0.6 * u[1], width=0.03, color="tab:red",
                       length_includes_head=True)
        ax[i, 0].set_xlim(0, 1); ax[i, 0].set_ylim(0, 1)
        if rows_label:
            ax[i, 0].set_title(rows_label[i], fontsize=6)
        for j in range(t):
            ax[i, j + 1].imshow(clips[i, j, 0], cmap="gray", vmin=0, vmax=1)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=110); plt.close(fig)
