"""Focused, audited figure for the complete 1CRN fresh-seed block."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfd3_flow_benchmark.benchmark import ROOT, ca_metrics


def main():
    plan=json.loads((ROOT/"PROTOCOL.json").read_text())
    seeds=plan["evaluation_seeds"]
    observations={}
    for seed in seeds:
        observations[seed]={}
        for policy in ("ve_zero","ve_trained"):
            path=ROOT/"evaluations"/"1CRN"/f"{policy}_{seed}.npz"
            with np.load(path) as saved:
                ca=saved["ca"]
                raw=saved["all_coordinates"]
                row=json.loads(str(saved["metadata"]))
            if not np.isfinite(raw).all() or abs(ca_metrics(ca)["reward"]-row["reward"])>1e-4:
                raise AssertionError(f"Coordinate audit failed: {path}")
            observations[seed][policy]=row["reward"]
    zero=np.array([observations[s]["ve_zero"] for s in seeds])
    trained=np.array([observations[s]["ve_trained"] for s in seeds])
    delta=trained-zero
    x=np.arange(len(seeds))

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.5,4.1),layout="constrained")
    fig.suptitle("The adapter trained, but fresh-seed scores barely moved",fontsize=14,fontweight="semibold")
    ax1.plot(x,zero,"o-",color="#256d85",linewidth=1.5,markersize=6,label="VE, zero adapter")
    ax1.plot(x,trained,"s--",color="#df7139",linewidth=1.2,markersize=5,
             markerfacecolor="white",markeredgewidth=1.7,label="VE, trained adapter")
    ax1.set_xticks(x,[str(i+1) for i in x])
    ax1.set_xlabel("Fresh evaluation seed")
    ax1.set_ylabel("Guarded Cα score (higher is better)")
    ax1.set_title("Output scores",fontsize=11)
    ax1.grid(axis="y",alpha=.2)
    ax1.legend(loc="lower left",frameon=False,fontsize=9)
    ax1.text(.98,.98,f"Zero-policy seed SD = {zero.std(ddof=1):.4f}",
             transform=ax1.transAxes,ha="right",va="top",fontsize=9)

    micro=delta*1e6
    ax2.axhline(0,color="#444",linewidth=.9)
    ax2.vlines(x,0,micro,color="#637f94",linewidth=2)
    ax2.scatter(x,micro,s=45,color="#256d85",zorder=3)
    ax2.axhline(micro.mean(),color="#df7139",linewidth=1.6,linestyle="--",
                label=f"Mean = {micro.mean():+.2f} × 10⁻⁶")
    ax2.set_xticks(x,[str(i+1) for i in x])
    ax2.set_xlabel("Same evaluation seed")
    ax2.set_ylabel("Trained − zero score (× 10⁻⁶)")
    ax2.set_title("Paired effect, magnified",fontsize=11)
    ax2.set_ylim(-9,9)
    ax2.grid(axis="y",alpha=.2)
    ax2.legend(loc="upper right",frameon=False,fontsize=9)

    target=ROOT/"blog_figure.png"
    fig.savefig(target,dpi=190)
    plt.close(fig)
    print(target)


if __name__=="__main__":
    main()
