"""Regenerate tables and publication figures strictly from recorded run artifacts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A running process may be in the middle of its final line.
            continue
    return rows


def paired_ci(first, last, key="rg"):
    delta = last[key] - first[key]
    idx = np.random.default_rng(776).integers(len(delta), size=(5000,len(delta)))
    return float(delta.mean()), np.quantile(delta[idx].mean(1), [.025,.975]).tolist()


def main():
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                        "axes.spines.right":False, "figure.dpi":140, "savefig.dpi":200})
    records = []
    history = {r["name"]:r["returncode"] for r in read_rows(ROOT/"queue_history.jsonl")}
    for folder in sorted((ROOT / "runs").glob("*")):
        rows = [r for r in read_rows(folder / "metrics.jsonl")
                if (folder / f"eval_{r['epoch']:04d}.npz").exists()]
        if not rows:
            continue
        config = json.loads((folder/"config.json").read_text())
        done = json.loads((folder/"DONE.json").read_text()) if (folder/"DONE.json").exists() else {}
        records.append((folder,config,rows,done))
    if not records:
        return
    lines = ["# Measured protein diffusion results", "",
             f"Generated {datetime.now(timezone.utc).isoformat()} from on-disk metrics. Unfinished runs are marked partial; queue failures are identified when logged.", "",
             "This report contains Genie SCOPe length-128 checkpoint experiments only. The subsequent RFdiffusion3 GRPO experiments are reported separately in ../rfd3_grpo/RESULTS.md.", "",
             "| Run | State | Epoch | n eval | Rg (Å) ↓ | Geometry pass | Bond MAE (Å) ↓ | Diversity (Å) ↑ | Path KL (nats) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for folder,config,rows,done in records:
        m = rows[-1]
        state = "complete" if done.get("complete") else "partial"
        if state != "complete" and folder.name in history:
            code = history[folder.name]
            if code == "deadline": state = "deadline stopped"
            elif code != 0: state = "failed"
        lines.append(f"| {folder.name} | {state} | {m['epoch']} | {m['n']} | {m['rg']:.3f} | {m['geometry_pass']:.1%} | {m['bond_mae']:.4f} | {m['diversity']:.3f} | {m['path_kl']:.3f} |")
        bon = folder / "best_of_n.json"
        if bon.exists():
            b = json.loads(bon.read_text())
            lines.append(f"| {folder.name}: best-of-{b['best_of']} | complete | — | {b['selected']} | {b['rg']:.3f} | {b['geometry_pass']:.1%} | {b['bond_mae']:.4f} | {b['diversity']:.3f} | not estimated |")
    lines += ["", "Geometry pass: ≥95% of adjacent Cα distances within 3.4–4.2 Å and no nonlocal Cα pair below 3 Å (sequence separation >2). These are declared heuristics, not foldability labels.",
              "A displayed 100% pass rate is an observed fraction in a small sample, not a population success-rate guarantee. New-run metric logs also include a Wilson interval for this fraction.",
              "Diversity is mean pairwise RMS difference of within-chain Cα distance maps. It is invariant to rigid motion, but also cannot distinguish mirror images.",
              "Path KL is measured under the adapted policy against the same respaced frozen sampler. It is not a KL to the original 1,000-step Genie sampler. Best-of-N KL is deliberately not compared to this path quantity.",
              "", "## Paired evaluation changes", "", "Common random numbers pair the initial and latest checkpoint. Intervals below bootstrap evaluation samples, not independent training seeds; intermediate evaluations are exploratory.", ""]
    train_records = [r for r in records if r[1]["mode"] == "train" and len(r[2]) > 1]
    for folder,config,rows,done in train_records:
        first = np.load(folder / f"eval_{rows[0]['epoch']:04d}.npz")
        last = np.load(folder / f"eval_{rows[-1]['epoch']:04d}.npz")
        delta, ci = paired_ci(first,last)
        lines.append(f"- **{folder.name}**: ΔRg = {delta:+.3f} Å, paired sample bootstrap 95% interval [{ci[0]:+.3f}, {ci[1]:+.3f}] Å; {rows[-1]['training_trajectories']} training trajectories, {rows[-1]['seconds']/60:.1f} elapsed minutes through evaluation.")
    lines += ["", "## Independent training seeds", ""]
    for prefix in ["compact_s", "guarded_s", "random_s"]:
        selected = [r for r in records if r[0].name.startswith(prefix) and r[3].get("complete")]
        if len(selected) >= 2:
            values = np.array([r[2][-1]["rg"] for r in selected])
            lines.append(f"- {prefix.rstrip('_s')}: Rg {values.mean():.3f} ± {values.std(ddof=1):.3f} Å (mean ± sample SD across {len(values)} independently trained heads; shared evaluation noise).")
        else:
            lines.append(f"- {prefix.rstrip('_s')}: fewer than two completed training seeds; no seed-level uncertainty reported.")
    fresh_base = next((r for r in records if r[0].name == "test_base"),None)
    lines += ["", "## Fresh sampling seeds", ""]
    if fresh_base:
        base_npz = np.load(fresh_base[0]/"eval_0000.npz")
        for folder,config,rows,done in records:
            if folder.name.startswith("test_") and folder.name != "test_base":
                delta,ci = paired_ci(base_npz,np.load(folder/"eval_0000.npz"))
                lines.append(f"- {folder.name}: ΔRg against fresh-seed base {delta:+.3f} Å, paired sample bootstrap 95% interval [{ci[0]:+.3f}, {ci[1]:+.3f}] Å.")
    else:
        lines.append("Fresh-seed evaluation has not completed. Validation curves alone do not establish a final generalization result.")
    lines += ["", "## Compute and comparison limits", "",
              "All runs use float32. Each trajectory costs the configured number of denoiser evaluations; this cost plus training and evaluation trajectories is recorded in config/metrics. Best-of-4 uses four inference candidates per returned trace. Training is not free: an amortization claim needs the recorded training cost and the intended deployment volume.",
              "The narrow head adaptation and small evaluation sets limit conclusions. No sequence design, folding-model validation, novelty search, all-atom checks or experimental validation has been performed. A no-change result is a valid outcome.",
              "", "## Figures", "", "![Learning curves](figures/learning_curves.png)", "", "![Backbone traces](figures/backbones.png)"]
    text = "\n".join(lines)+"\n"
    tmp = ROOT / "RESULTS.md.tmp"; tmp.write_text(text); tmp.replace(ROOT/"RESULTS.md")

    fig, axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    keys = [("rg","Radius of gyration (Å)"),("geometry_pass","Geometry-pass fraction"),
            ("diversity","Distance-map diversity (Å)"),("path_kl","Path KL (nats)")]
    for folder,config,rows,done in records:
        if config["mode"] != "train":
            continue
        for ax,(key,label) in zip(axes.flat,keys):
            xx = [m["training_trajectories"] for m in rows]
            line, = ax.plot(xx,[m[key] for m in rows],marker="o",ms=3,label=folder.name)
            if key == "rg":
                ci = [m["sample_bootstrap_ci95"]["rg"] for m in rows]
                ax.fill_between(xx,[c[0] for c in ci],[c[1] for c in ci],alpha=.08,color=line.get_color())
    for ax,(key,label) in zip(axes.flat,keys):
        ax.set(xlabel="Training trajectories",ylabel=label)
        ax.grid(alpha=.18)
        if key == "geometry_pass": ax.set_ylim(-.05,1.05)
        if key in {"rg","diversity"}:
            # Do not visually inflate a thousandth-Angstrom pilot change.
            lo,hi = ax.get_ylim()
            if hi-lo < 1:
                middle = (lo+hi)/2
                ax.set_ylim(middle-.5,middle+.5)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Genie translation-head adaptation · recorded evaluations only\nRg shading: evaluation-sample bootstrap 95% intervals",fontweight="bold",fontsize=12)
    fig.savefig(figdir/"learning_curves.png"); fig.savefig(figdir/"learning_curves.svg"); plt.close(fig)

    # Fixed first four examples, not hand-picked successes.
    chosen = next((r for r in records if r[0].name == "compact_s0" and len(r[2])>1),
                  next((r for r in records if r[0].name=="pilot"),records[0]))
    folder,config,rows,done = chosen
    first = np.load(folder / f"eval_{rows[0]['epoch']:04d}.npz")["coordinates"]
    latest = np.load(folder / f"eval_{rows[-1]['epoch']:04d}.npz")["coordinates"]
    fig = plt.figure(figsize=(12,6),layout="constrained")
    for row,coords in enumerate([first,latest]):
        for col,x in enumerate(coords[:4]):
            ax = fig.add_subplot(2,4,row*4+col+1,projection="3d")
            x = x-x.mean(0)
            for i in range(len(x)-1):
                ax.plot(*x[i:i+2].T,color=plt.cm.viridis(i/(len(x)-1)),lw=2)
            limit = max(18,float(np.abs(np.concatenate([first[col]-first[col].mean(0),latest[col]-latest[col].mean(0)])).max())*1.05)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit))
            ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(20,35)
            ax.set_title(f"{'Base' if row==0 else 'Latest'} · sample {col+1}",fontsize=10)
    fig.suptitle(f"{folder.name} · shared random noise · Cα traces, sequence unspecified\nBase vs epoch {rows[-1]['epoch']}; same viewing scale within each column",fontsize=12,fontweight="bold")
    fig.savefig(figdir/"backbones.png"); plt.close(fig)
    print(f"Updated {ROOT/'RESULTS.md'} from {len(records)} runs")


if __name__ == "__main__":
    main()
