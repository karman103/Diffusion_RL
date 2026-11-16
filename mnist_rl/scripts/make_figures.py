"""Generate all blog figures from runs/*/metrics.jsonl."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140, "savefig.bbox": "tight"})
PAL = ["#2a6fdb", "#e4572e", "#29a36a", "#9b5de5", "#f2a541", "#555555", "#00a6a6", "#c2185b"]


def load(run):
    f = f"runs/{run}/metrics.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else None


def agg(prefix_runs):
    """mean/std over seeds for list of run names (that exist)."""
    ds = [d for d in (load(r) for r in prefix_runs) if d is not None]
    if not ds:
        return None
    n = min(len(d["epoch"]) for d in ds)
    out = {}
    for k in ds[0]:
        arr = np.stack([d[k][:n] for d in ds])
        out[k], out[k + "_sd"] = arr.mean(0), arr.std(0)
    out["n"] = len(ds)
    return out


def seeds(name):
    return [name] + [f"{name}_s{s}" for s in (1, 2)]


def curves(groups, keys, fname, titles):
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.0))
    for (label, runs), c in zip(groups, PAL):
        d = agg(runs)
        if d is None:
            continue
        for ax, k in zip(axes, keys):
            ax.plot(d["epoch"], d[k], color=c, label=f"{label}" + (f" (n={d['n']})" if d["n"] > 1 else ""), lw=1.8)
            if d["n"] > 1:
                ax.fill_between(d["epoch"], d[k] - d[k + "_sd"], d[k] + d[k + "_sd"], color=c, alpha=0.15)
    for ax, t, k in zip(axes, titles, keys):
        ax.set_title(t); ax.set_xlabel("RL epoch")
        if k == "kl" and ax.get_ylim()[1] > 1000:
            ax.set_yscale("symlog", linthresh=10); ax.set_title(t + " (log)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"figures/{fname}"); plt.close(fig)


def goodhart(groups, fname, bon="runs/base/best_of_n.json"):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    if os.path.exists(bon):
        b = json.load(open(bon))
        kl = np.sqrt([r["kl"] for r in b])
        axes[0].plot(kl, [r["proxy_p"] for r in b], "k--", marker="o", ms=3, label="best-of-N (base)")
        axes[1].plot(kl, [r["gold_acc"] for r in b], "k--", marker="o", ms=3, label="best-of-N (base)")
    for (label, runs), c in zip(groups, PAL):
        d = agg(runs)
        if d is None:
            continue
        x = np.sqrt(np.maximum(d["kl"], 0))
        axes[0].plot(x, d["reward"], color=c, marker=".", label=label)
        axes[1].plot(x, d["gold_acc"], color=c, marker=".", label=label)
    axes[0].set_title("Proxy reward (what we optimise)"); axes[1].set_title("Gold accuracy (what we want)")
    for ax in axes:
        ax.set_xlabel(r"$\sqrt{\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})}$  (nats$^{1/2}$, path-space)")
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"figures/{fname}"); plt.close(fig)


def grid_strip(rows, fname, epochs):
    """rows: list of (label, run). Shows top-left 4x4 crop of sample grids at given epochs."""
    fig, axes = plt.subplots(len(rows), len(epochs), figsize=(1.9 * len(epochs), 1.95 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (label, run) in enumerate(rows):
        for j, e in enumerate(epochs):
            ax = axes[i, j]; ax.axis("off")
            f = f"runs/{run}/samples_{e:04d}.png"
            if not os.path.exists(f):
                fs = sorted(glob.glob(f"runs/{run}/samples_*.png"))
                if not fs: continue
                f = fs[-1] if j == len(epochs) - 1 else None
                if f is None: continue
            im = np.array(Image.open(f))
            ax.imshow(im[: 4 * 29 + 1, : 4 * 29 + 1], cmap="gray")
            if i == 0: ax.set_title(f"epoch {e}", fontsize=9)
            if j == 0: ax.text(-8, 58, label, fontsize=8.5, ha="right", va="center", rotation=0)
    fig.tight_layout(); fig.savefig(f"figures/{fname}"); plt.close(fig)


def proxy_vs_gold(groups, fname, bon="runs/base/best_of_n.json"):
    """The cleanest Goodhart picture: x = proxy reward (what we optimise), y = gold accuracy."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls=":")
    if os.path.exists(bon):
        b = json.load(open(bon))
        ax.plot([r["proxy_p"] for r in b], [r["gold_acc"] for r in b], "k--", marker="o", ms=3, label="best-of-N (base)")
    for (label, runs), c in zip(groups, PAL):
        d = agg(runs)
        if d is None:
            continue
        ax.plot(d["proxy_p"], d["gold_acc"], color=c, marker=".", label=label)
        ax.annotate("", xy=(d["proxy_p"][-1], d["gold_acc"][-1]), xytext=(d["proxy_p"][-2], d["gold_acc"][-2]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    ax.set_xlabel("proxy reward  p_proxy(7 | x)"); ax.set_ylabel("gold accuracy")
    ax.set_title("Optimising the proxy: who also improves the truth?", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"figures/{fname}"); plt.close(fig)


def sample_eff(groups, fname, bs=64):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for (label, runs, inner), c in zip(groups, PAL):
        d = agg(runs)
        if d is None:
            continue
        ax.plot(d["epoch"] * bs, d["gold_acc"], color=c, label=label)
    ax.set_xlabel("trajectories sampled"); ax.set_ylabel("gold accuracy")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"figures/{fname}"); plt.close(fig)


if __name__ == "__main__":
    METHODS = [("DDPO (PPO)", seeds("p_ddpo")),
               ("RWR", ["p_rwr"]), ("DRaFT-1", ["p_draft"]), ("DDPO + KL β=3e-4", ["p_kl_0.0003"]),
               ("DDPO on gold (oracle)", ["p_oracle"])]
    curves(METHODS, ["reward", "gold_acc", "fd_target", "diversity"], "fig_methods.png",
           ["Proxy reward  p(7)", "Gold accuracy (is it a 7?)", "FD to real 7s (lower better)", "Feature diversity"])
    goodhart(METHODS[:-1], "fig_goodhart.png")
    KLG = [("β=0", seeds("p_ddpo")), ("β=3e-4", ["p_kl_0.0003"]), ("β=1e-3", ["p_kl_0.001"]), ("β=0.1", ["p_kl_0.1"])]
    curves(KLG, ["reward", "gold_acc", "fd_target", "kl"], "fig_klsweep.png",
           ["Proxy reward p(7)", "Gold accuracy", "FD to real 7s", "KL to base (nats)"])
    goodhart(KLG, "fig_klsweep_goodhart.png")
    SHAPE = [("DDPO, r = log p(7)", ["e1_ddpo"]), ("DDPO, r = p(7)", ["e1p_ddpo"]),
             ("DRaFT, r = log p(7)", ["e1_draft"]), ("DRaFT, r = p(7)", ["p_draft"]), ("random reward (control)", ["e0_random"])]
    curves(SHAPE, ["gold_acc", "fd_target", "mean_ink", "kl"], "fig_shaping.png",
           ["Gold accuracy", "FD to real 7s", "Mean pixel intensity", "KL to base (nats)"])
    curves([("JPEG incompressible", ["e2_incompress"]), ("JPEG compressible", ["e2_compress"])],
           ["reward", "jpeg_kb", "mean_ink", "kl"], "fig_jpeg.png",
           ["Reward", "JPEG size (kB)", "Mean pixel intensity", "KL to base"])
    grid_strip([("log p(7)", "e1_ddpo"), ("p(7)", "e1p_ddpo")], "fig_grid_shaping.png", [0, 30, 60, 100])
    grid_strip([("DDPO", "p_ddpo"), ("DDPO+KL", "p_kl_0.1"), ("DRaFT", "p_draft"), ("RWR", "p_rwr")], "fig_grid_methods.png", [0, 50, 100, 150])
    grid_strip([("incompressible", "e2_incompress"), ("compressible", "e2_compress")], "fig_grid_jpeg.png", [0, 20, 40, 100])
    proxy_vs_gold([("DDPO", seeds("p_ddpo")), ("DDPO + KL β=3e-4", ["p_kl_0.0003"]), ("RWR", ["p_rwr"]),
                   ("DRaFT-1", ["p_draft"]), ("DDPO, 3× reuse", ["p_ddpo_reuse3"])], "fig_proxy_vs_gold.png")
    sample_eff([("DDPO, 1 pass / batch", seeds("p_ddpo"), 1), ("DDPO, 3 passes / batch", ["p_ddpo_reuse3"], 3),
                ("DDPO, 3 passes + KL", ["p_reuse3_kl"], 3)], "fig_sample_eff.png")
    print("figures written")
