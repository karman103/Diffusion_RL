"""Audit fresh-noise evaluation coordinates and summarize paired comparisons."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from .heldout_eval import TASK
from .metrics import metrics, reward

os.environ.setdefault("MPLCONFIGDIR", str(TASK / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(TASK / ".cache"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=TASK / "heldout_3000000_n16")
    args = parser.parse_args()
    plan = json.loads((args.folder / "plan.json").read_text())
    evaluation_length = plan.get("evaluation_length", 32)
    torch.set_num_threads(2)
    data = {}
    audit_count = 0
    policies = [*plan["policies"]]
    if "late_zero" not in policies and (args.folder / "late_zero_plan.json").exists():
        policies.insert(policies.index("late_s0"), "late_zero")
    for policy in policies:
        data[policy] = {}
        for seed in range(plan["first_seed"], plan["first_seed"] + plan["count"]):
            path = args.folder / f"{policy}_{seed}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as saved:
                ca = torch.from_numpy(saved["ca"])
                actual = metrics(ca)
                actual["guarded_reward"] = reward(ca, "guarded")
                for key, value in actual.items():
                    np.testing.assert_allclose(value.numpy(), saved[key], rtol=1e-5, atol=1e-6)
                data[policy][seed] = {key: float(saved[key].item()) for key in
                    ("guarded_reward", "rg", "bond_mae", "bond_ok", "clash_fraction",
                     "geometry_pass", "path_kl")}
            audit_count += 1
    lines = [f"# Fresh-noise RFdiffusion3 evaluation · {evaluation_length} residues", "",
             f"Updated {datetime.now(timezone.utc).isoformat()}. Fresh noise seeds "
             f"{plan['first_seed']}–{plan['first_seed'] + plan['count'] - 1}; "
             "no adapter was updated using these samples."]
    if evaluation_length != plan.get("training_length", 32):
        lines += ["", f"This is an untrained length-transfer check: adapters were fitted at 32 residues and evaluated at {evaluation_length} residues."]
    lines += ["", "| Policy | n | Guarded reward | Rg (Å) | Bond MAE (Å) | Geometry pass | Path KL | Paired Δreward vs matched zero [95% bootstrap] |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in policies:
        values = data[policy]
        if not values:
            continue
        n = len(values)
        mean = lambda key: np.mean([x[key] for x in values.values()])
        reference = "late_zero" if policy == "late_s0" else "zero"
        paired = sorted(set(values) & set(data.get(reference, {})))
        if policy in ("zero", "late_zero"):
            comparison = "reference"
        elif policy == "late_s0" and not paired:
            comparison = "different action window; matched zero pending"
        elif paired:
            delta = np.array([values[s]["guarded_reward"] - data[reference][s]["guarded_reward"]
                              for s in paired])
            indices = np.random.default_rng(187).integers(len(delta), size=(5000, len(delta)))
            ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            comparison = (f"{delta.mean():+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] (n={len(paired)})"
                          if len(paired) >= 4 else f"{delta.mean():+.5f} (n={len(paired)}; too few for interval)")
        else:
            comparison = "pending reference"
        lines.append(f"| {policy} | {n} | {mean('guarded_reward'):.5f} | {mean('rg'):.4f} | "
                     f"{mean('bond_mae'):.4f} | {mean('geometry_pass'):.1%} | "
                     f"{mean('path_kl'):.4f} | {comparison} |")
    common_seeds = sorted(set.intersection(*(set(data[name]) for name in
        ("zero", "main_s0", "main_s1", "main_s2"))))
    if common_seeds:
        delta = np.array([np.mean([data[name][seed]["guarded_reward"] -
                                   data["zero"][seed]["guarded_reward"]
                                   for name in ("main_s0", "main_s1", "main_s2")])
                          for seed in common_seeds])
        indices = np.random.default_rng(188).integers(len(delta), size=(5000, len(delta)))
        ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        lines += ["", f"Three reward-directed adapters, averaged per shared noise seed: "
                  f"Δguarded reward {delta.mean():+.5f}, bootstrap 95% interval "
                  f"[{ci[0]:+.5f}, {ci[1]:+.5f}] over {len(common_seeds)} fresh noise seeds."]
    lines += ["", "[Paired difference figure](paired_deltas.png) shows individual noise-seed differences as points, means as horizontal bars and bootstrap intervals as vertical bars when at least four pairs are present.",
              "", "Late-window adaptation must be compared to late_zero, which uses the same restricted action-noise window. Other adapted policies use the full-window zero reference.",
              "", f"Coordinate audit passed for {audit_count} saved evaluations. "
              f"All comparisons within this report use the same unconditional {evaluation_length}-residue input and share noise seeds. "
              "Bootstrap intervals describe only variation across these noise seeds for fixed trained adapters; "
              "they do not measure variation across protein targets, training seeds, or reward definitions. "
              "Cα geometry is a heuristic, not foldability or sequence validation."]
    (args.folder / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    labels = []
    for policy in ("stock", "main_s0", "main_s1", "main_s2", "random_s0", "single_s0", "late_s0"):
        reference = "late_zero" if policy == "late_s0" else "zero"
        paired = sorted(set(data.get(policy, {})) & set(data.get(reference, {})))
        if not paired:
            continue
        delta = np.array([data[policy][s]["guarded_reward"] -
                          data[reference][s]["guarded_reward"] for s in paired])
        x = len(labels)
        labels.append(policy.replace("_", " "))
        ax.scatter(np.full(len(delta), x) + np.linspace(-.13, .13, len(delta)), delta,
                   s=16, alpha=.55, color="tab:blue")
        ax.plot([x-.22, x+.22], [delta.mean(), delta.mean()], color="black", lw=2)
        if len(delta) >= 4:
            indices = np.random.default_rng(187).integers(len(delta), size=(5000, len(delta)))
            ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            ax.plot([x, x], ci, color="black", lw=2)
            ax.plot([x-.06, x+.06], [ci[0], ci[0]], color="black", lw=2)
            ax.plot([x-.06, x+.06], [ci[1], ci[1]], color="black", lw=2)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set(ylabel="Paired guarded reward change (higher is better)",
           title=f"RFdiffusion3 · fresh noise seeds · fixed {evaluation_length}-residue condition")
    ax.grid(axis="y", alpha=.15)
    fig.savefig(args.folder / "paired_deltas.png", dpi=180)
    plt.close(fig)
    print(f"Audited {audit_count} heldout samples; report {args.folder / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
