"""Build a measured RFD3 report and audit metrics from saved coordinates."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from .metrics import metrics,reward

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/".cache"))


def rows(path):
    result=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    return result


def main():
    torch.set_num_threads(2)
    all_runs=[];audit_count=0
    for folder in sorted((ROOT/"runs").glob("*")):
        if not (folder/"config.json").exists():continue
        cfg=json.loads((folder/"config.json").read_text())
        values=[]
        objective="compact" if cfg["objective"]=="random" else cfg["objective"]
        for m in rows(folder/"metrics.jsonl"):
            tag="stock" if m["stock"] else f"eval_{m['epoch']:04d}"
            path=folder/f"{tag}.npz"
            if not path.exists():continue
            with np.load(path,allow_pickle=False) as arrays:
                ca=torch.from_numpy(arrays["ca"])
                actual=metrics(ca);actual["reward"]=reward(ca,objective)
                for key,value in actual.items():
                    np.testing.assert_allclose(value.numpy(),arrays[key],rtol=1e-5,atol=1e-6)
                    np.testing.assert_allclose(value.mean().item(),m[key],rtol=1e-5,atol=1e-6)
            values.append(m);audit_count+=1
        checks=json.loads((folder/"checks.json").read_text()) if (folder/"checks.json").exists() else {}
        done=json.loads((folder/"DONE.json").read_text()) if (folder/"DONE.json").exists() else {}
        all_runs.append((folder,cfg,values,rows(folder/"training.jsonl"),checks,done))
    lines=["# RFdiffusion3 GRPO: measured results","",
           f"Updated {datetime.now(timezone.utc).isoformat()}. Values are recomputed from saved RFD3 coordinates.","",
           "| Run | Sampler / checkpoint | n | Reward | Rg (Å) | Bond MAE (Å) | Geometry pass | Path KL |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    plot_runs=[r for r in all_runs if r[1]["steps"]>=32 and any(not m["stock"] for m in r[2])]
    if not plot_runs:
        plot_runs=[r for r in all_runs if any(not m["stock"] for m in r[2])]
    for folder,cfg,values,training,checks,done in all_runs:
        for m in values:
            kind="stock" if m["stock"] else ("modified / zero adapter" if m["epoch"]==0 else f"modified / epoch {m['epoch']}")
            lines.append(f"| {folder.name} | {kind} | {m['n']} | {m['reward']:.4f} | {m['rg']:.4f} | {m['bond_mae']:.4f} | {m['geometry_pass']:.1%} | {m['path_kl']:.4f} |")
    if not all_runs:lines += ["","No completed pretrained RFD3 measurements yet."]
    lines += ["","Geometry pass is an observed small-sample fraction under a Cα-distance/clash heuristic. It does not establish foldability or all-atom validity. Stock and modified samplers must be compared before attributing changes to optimization. Path KL refers only to the modified stochastic policy and its reference.","","## Actual update evidence",""]
    for folder,cfg,values,training,checks,done in all_runs:
        state="complete" if done.get("complete") else ("failed" if (folder/"FAILED.json").exists() else "unfinished")
        lines += [f"### {folder.name}","",f"State: {state}; {cfg['base_parameters']:,} frozen network parameters, {cfg['trainable_parameters']:,} trainable adapter parameters. {cfg['length']} residues, {cfg['steps']-1} denoising calls per trajectory, {cfg['recycles']} recycling passes per call.",""]
        for key,value in checks.items():lines.append(f"- {key}: {value:.7g}")
        if training:
            last=training[-1]
            lines += [f"- Completed recorded updates: {last['epoch']}; training siblings: {last['epoch']*cfg['group_size']}.",
                      f"- Adapter weight-delta norm: {last['adapter_delta_norm']:.7g}; frozen-network tensors receiving gradients: {last['frozen_base_gradient_count']}.",
                      f"- Final group rewards: {last['group_rewards']}; relative advantages: {last['group_advantages']}.",
                      f"- Final reuse clip fraction: {last['clip_fraction']:.3%}; max absolute log ratio: {last['max_abs_logratio']:.4g}."]
        before=next((m for m in values if not m["stock"] and m["epoch"]==0),None)
        after=[m for m in values if not m["stock"] and m["epoch"]>0]
        if before and after:
            last=after[-1]
            with np.load(folder/"eval_0000.npz") as b,np.load(folder/f"eval_{last['epoch']:04d}.npz") as e:
                delta=e["rg"]-b["rg"]
                indices=np.random.default_rng(19).integers(len(delta),size=(3000,len(delta)))
                ci=np.quantile(delta[indices].mean(1),[.025,.975])
                lines.append(f"- Paired ΔRg: {delta.mean():+.5f} Å; evaluation-sample bootstrap 95% interval [{ci[0]:+.5f}, {ci[1]:+.5f}] Å. This is not uncertainty across training seeds.")
        lines.append("")
    lines += [f"Coordinate audit: passed for {audit_count} saved evaluations.","",
              "The tiny smoke run tests implementation and an actual pretrained-weight update. It is not a benchmark or evidence of useful protein design. See METHOD.md for the action-density derivation and the exact diffusion-head adaptation."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    if not plot_runs:
        print("Report ready; no pretrained measurements yet")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for folder,cfg,values,training,checks,done in plot_runs:
        v=[m for m in values if not m["stock"]]
        if not v:continue
        for ax,key,label in zip(axes,["reward","rg","path_kl"],["Reward","Radius of gyration (Å)","Path KL (nats)"]):
            ax.plot([m["training_trajectories"] for m in v],[m[key] for m in v],marker="o",label=folder.name)
            ax.set(xlabel="Training sibling trajectories",ylabel=label);ax.grid(alpha=.15)
            if key in {"rg","reward"}:
                lo,hi=ax.get_ylim()
                span=1 if key=="rg" else .5
                if hi-lo<span:ax.set_ylim((lo+hi)/2-span/2,(lo+hi)/2+span/2)
                ax.ticklabel_format(useOffset=False,axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle("RFdiffusion3 · modified Gaussian Cα policy · recorded evaluations only")
    fig.savefig(ROOT/"results.png",dpi=180);plt.close(fig)
    chosen=next((r for r in plot_runs if r[0].name=="main_s0"),plot_runs[0])
    folder,cfg,values,training,checks,done=chosen
    baseline=next((m for m in values if not m["stock"] and m["epoch"]==0),None)
    learned=[m for m in values if not m["stock"] and m["epoch"]>0]
    stock=next((m for m in values if m["stock"]),None)
    if baseline and (learned or stock):
        left=folder/("eval_0000.npz" if learned else "stock.npz")
        right=folder/(f"eval_{learned[-1]['epoch']:04d}.npz" if learned else "eval_0000.npz")
        with np.load(left) as f:left_ca=f["ca"]
        with np.load(right) as f:right_ca=f["ca"]
        columns=min(4,len(left_ca),len(right_ca))
        fig=plt.figure(figsize=(3.5*columns,6),layout="constrained")
        labels=("Zero adapter",f"Trained epoch {learned[-1]['epoch']}") if learned else ("Stock sampler","Modified sampler / zero adapter")
        for column in range(columns):
            pair=[left_ca[column]-left_ca[column].mean(0),right_ca[column]-right_ca[column].mean(0)]
            limit=max(10,float(np.abs(np.concatenate(pair)).max())*1.1)
            for row,x in enumerate(pair):
                ax=fig.add_subplot(2,columns,row*columns+column+1,projection="3d")
                for i in range(len(x)-1):ax.plot(*x[i:i+2].T,color=plt.cm.viridis(i/max(1,len(x)-1)),lw=2)
                ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit))
                ax.set_axis_off();ax.set_box_aspect((1,1,1));ax.view_init(20,35)
                ax.set_title(f"{labels[row]} · sample {column+1}",fontsize=9)
        fig.suptitle(f"RFD3 {folder.name} · {cfg['length']} residues · fixed first sample indices\nSame scale within each column; sequence unspecified",fontsize=11)
        fig.savefig(ROOT/"backbones.png",dpi=180);plt.close(fig)
    print(f"Reported {len(all_runs)} runs; coordinate audit passed for {audit_count} evaluations")


if __name__=="__main__":main()
