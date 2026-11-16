"""Plot paired fresh-seed effects from audited benchmark coordinate artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfd3_flow_benchmark.benchmark import ROOT, aligned_rmsds, ca_metrics, native_ca


def main():
    plan = json.loads((ROOT / "PROTOCOL.json").read_text())
    pdbs = [plan["train_pdb"], *plan["heldout_pdbs"]]
    metrics = ("reward", "gap_rmsd_to_native")
    labels = ("Guarded Cα score", "Gap-to-native Cα RMSD (Å)")
    colors = ("#176f8a", "#de6c37", "#693b9a")
    rows = {pdb: {} for pdb in pdbs}
    observed = 0

    for pdb in pdbs:
        native = native_ca(ROOT / "pdbs" / f"{pdb}.pdb")
        for seed in plan["evaluation_seeds"]:
            paths = {
                policy: ROOT / "evaluations" / pdb / f"{policy}_{seed}.npz"
                for policy in ("stock", "ve_zero", "ve_trained")
            }
            rows[pdb][seed] = {}
            for policy, path in paths.items():
                if not path.exists():
                    continue
                observed += 1
                with np.load(path) as data:
                    ca = data["ca"]
                    all_coordinates = data["all_coordinates"]
                    metadata = json.loads(str(data["metadata"]))
                score = ca_metrics(ca)
                fixed_rmsd, gap_rmsd = aligned_rmsds(ca, native)
                if not np.isfinite(ca).all() or ca.shape != (40, 3):
                    raise AssertionError(f"Invalid Cα coordinates: {path}")
                if (all_coordinates.ndim != 3 or all_coordinates.shape[0] != 1
                        or all_coordinates.shape[-1] != 3
                        or not np.isfinite(all_coordinates).all()):
                    raise AssertionError(f"Invalid raw atom coordinates: {path}")
                if fixed_rmsd > 1e-3:
                    raise AssertionError(f"Fixed motif moved: {path}")
                if abs(score["reward"] - metadata["reward"]) > 1e-4:
                    raise AssertionError(f"Reward audit failed: {path}")
                if abs(gap_rmsd - metadata["gap_rmsd_to_native"]) > 1e-4:
                    raise AssertionError(f"Native-gap audit failed: {path}")
                rows[pdb][seed][policy] = metadata

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout="constrained")
    for col, (metric, ylabel) in enumerate(zip(metrics, labels)):
        ax = axes[0, col]
        for i, pdb in enumerate(pdbs):
            shared = [seed for seed in plan["evaluation_seeds"]
                      if all(policy in rows[pdb][seed] for policy in ("ve_zero", "ve_trained"))]
            for seed in shared:
                values = rows[pdb][seed]
                ax.plot([i - .12, i + .12],
                        [values["ve_zero"][metric], values["ve_trained"][metric]],
                        color=colors[i], alpha=.35, linewidth=1)
                ax.scatter([i - .12, i + .12],
                           [values["ve_zero"][metric], values["ve_trained"][metric]],
                           color=colors[i], s=15, alpha=.7)
            if shared:
                ax.scatter([i - .12, i + .12],
                           [np.mean([rows[pdb][s][p][metric] for s in shared])
                            for p in ("ve_zero", "ve_trained")],
                           color=colors[i], s=85, marker="_", linewidth=2, zorder=4)
        ax.set_xticks(range(len(pdbs)),
                      [f"{p}\n{sum(all(q in rows[p][s] for q in ('ve_zero','ve_trained')) for s in plan['evaluation_seeds'])} pairs"
                       for p in pdbs])
        ax.set_ylabel(ylabel)
        ax.set_title("Same-seed VE sampler: individual pairs and means")
        ax.grid(axis="y", alpha=.25)

        delta_ax = axes[1, col]
        for i, pdb in enumerate(pdbs):
            deltas = [rows[pdb][s]["ve_trained"][metric] - rows[pdb][s]["ve_zero"][metric]
                      for s in plan["evaluation_seeds"]
                      if all(q in rows[pdb][s] for q in ("ve_zero", "ve_trained"))]
            if not deltas:
                continue
            delta_ax.scatter(np.full(len(deltas), i) + np.linspace(-.07, .07, len(deltas)),
                             deltas, color=colors[i], s=32, alpha=.7)
            delta_ax.scatter(i, np.mean(deltas), color=colors[i], s=120,
                             marker="_", linewidth=2, zorder=4)
        delta_ax.axhline(0, color="black", linewidth=.8)
        delta_ax.set_xticks(range(len(pdbs)), pdbs)
        delta_ax.set_ylabel(f"Trained − zero: {ylabel}")
        delta_ax.set_title("Paired effect (dots = available seeds; bar = mean)")
        delta_ax.grid(axis="y", alpha=.25)

    fig.suptitle(f"RFdiffusion3 Flow-GRPO-style PDB benchmark ({observed}/36 samples)", fontsize=14)
    target = ROOT / "paired_effects.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
