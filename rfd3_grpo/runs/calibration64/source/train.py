"""Actual pretrained RFD3 GRPO: python -m rfd3_grpo.train --out ..."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .backend import RFD3Backend,ROOT
from .grpo import group_advantages,transition_loss
from .metrics import metrics,reward,save_ca_pdb


def json_write(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--length",type=int,default=32)
    p.add_argument("--recycles",type=int,default=2)
    p.add_argument("--device",choices=["mps","cpu"],default="mps")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--group-size",type=int,default=4)
    p.add_argument("--eval-n",type=int,default=4)
    p.add_argument("--rank",type=int,default=2)
    p.add_argument("--noise-experts",choices=["single","binned"],default="binned")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--reuse",type=int,default=4)
    p.add_argument("--clip",type=float,default=.2)
    p.add_argument("--kl-beta",type=float,default=.01)
    p.add_argument("--extra-noise",type=float,default=.1)
    p.add_argument("--active-min",type=float,default=.5)
    p.add_argument("--active-max",type=float,default=32.)
    p.add_argument("--objective",choices=["compact","guarded","random"],default="guarded")
    p.add_argument("--check-stock-parity",action="store_true")
    p.add_argument("--evaluate-stock",action="store_true")
    p.add_argument("--deadline",default="2026-11-13T10:26:00Z")
    a=p.parse_args()
    if min(a.group_size,a.eval_n)<2 or a.extra_noise<=0:
        raise ValueError("At least two siblings/evaluation samples and positive action noise required")
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run {a.out}")
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    started=time.time()
    backend=RFD3Backend(a.checkpoint,a.steps,a.length,a.recycles,a.device,a.rank,a.noise_experts)
    for module in [backend.net,backend.dm]:
        assert not any(p.requires_grad for p in module.parameters())
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(started_utc=datetime.now(timezone.utc).isoformat(),
                  foundry_commit=subprocess.check_output(["git","-C",str(ROOT/"third_party/foundry-mps"),"rev-parse","HEAD"],text=True).strip(),
                  torch=torch.__version__,atomworks=importlib.metadata.version("atomworks"),
                  base_parameters=backend.model_parameters,
                  trainable_parameters=sum(p.numel() for p in backend.adapter.parameters()),
                  output_channels=backend.channels,schedule=backend.schedule,
                  adaptation=f"{a.noise_experts} rank-{a.rank} LoRA on to_r_update[-1], LAST recycle, free CA atoms only",
                  sampler="stock churn plus explicit post-Euler CA Gaussian actions in the active window",
                  conditioning=backend.specification,
                  atoms=len(backend.ca),ca_atoms=int(backend.ca.sum()),
                  source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    # Hash in chunks, never materialize the multi-GB checkpoint again.
    checkpoint=a.checkpoint or ROOT/"rfd3_grpo/checkpoints/rfd3_latest.ckpt"
    with checkpoint.open("rb") as f:
        config["checkpoint_sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
    json_write(a.out/"config.json",config)
    np.savez_compressed(a.out/"atom_metadata.npz",is_ca=backend.ca.cpu().numpy(),
        fixed=backend.fixed.cpu().numpy(),atom_to_token=backend.f["atom_to_token_map"].cpu().numpy(),
        atom_name=np.array(backend.example["atom_array"].atom_name,dtype="U8"))
    head_error=backend.head_equivalence_check()
    checks=dict(last_recycle_head_equivalence_max_error=head_error)
    print("HEAD CHECK "+json.dumps(checks),flush=True)
    if a.check_stock_parity:
        checks["stock_sampler_parity_max_error"]=backend.stock_parity_check()
    json_write(a.out/"checks.json",checks)

    def sample(seed,adapted=True,record=True,stock=False):
        return backend.rollout(seed,adapted,record,a.extra_noise,a.active_min,a.active_max,stock)

    log=(a.out/"metrics.jsonl").open("w",buffering=1)
    training=(a.out/"training.jsonl").open("w",buffering=1)
    objective="compact" if a.objective=="random" else a.objective
    def evaluate(epoch,stock=False):
        outputs=[sample(2_000_000+i,record=False,stock=stock) for i in range(a.eval_n)]
        ca=torch.cat([o["ca"].cpu() for o in outputs])
        all_coordinates=torch.cat([o["coordinates"].cpu() for o in outputs])
        m=metrics(ca);m["reward"]=reward(ca,objective)
        values={k:v.numpy() for k,v in m.items()}
        row={k:float(v.mean()) for k,v in values.items()}
        row.update(epoch=epoch,n=len(ca),stock=stock,path_kl=float(np.mean([o["path_kl"] for o in outputs])),
                   seconds=time.time()-started,training_trajectories=epoch*a.group_size)
        tag="stock" if stock else f"eval_{epoch:04d}"
        target=a.out/f"{tag}.npz"
        with target.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f,coordinates=all_coordinates.numpy(),ca=ca.numpy(),**values)
        target.with_suffix(".tmp").replace(target)
        for i,x in enumerate(ca):
            save_ca_pdb(a.out/f"{tag}_{i:02d}.pdb",x)
        log.write(json.dumps(row,allow_nan=False)+"\n")
        print("EVAL "+json.dumps(row,allow_nan=False),flush=True)
        return row

    if a.evaluate_stock:
        evaluate(0,stock=True)
    evaluate(0)
    opt=torch.optim.Adam(backend.adapter.parameters(),lr=a.lr)
    last_epoch=0
    deadline=datetime.fromisoformat(a.deadline.replace("Z","+00:00"))
    for epoch in range(1,a.epochs+1):
        if datetime.now(timezone.utc)>=deadline:
            break
        # Each group shares exactly the same condition/featurization; independent
        # initial and churn/action draws produce its sibling designs.
        siblings=[sample(100_000+a.seed*10000+epoch*100+i) for i in range(a.group_size)]
        with torch.no_grad():
            if a.objective=="random":
                rng=torch.Generator().manual_seed(700_000+a.seed*1000+epoch)
                rewards=torch.randn(1,a.group_size,generator=rng).to(a.device)
            else:
                rewards=torch.stack([reward(o["ca"],a.objective).squeeze() for o in siblings])[None]
            advantages=group_advantages(rewards)[0]
        transitions=sum(len(o["records"]) for o in siblings)
        if not transitions:
            raise RuntimeError("No stochastic policy actions: widen active sigma interval or increase steps")
        clips=[];kls=[];lrs=[];losses=[]
        if advantages.abs().max()>0:
            for reuse in range(a.reuse):
                opt.zero_grad(set_to_none=True)
                for adv,output in zip(advantages,siblings):
                    for record in output["records"]:
                        loss,diag=transition_loss(record,backend.adapter,adv,a.clip,a.kl_beta)
                        (loss/transitions).backward()
                        clips.append(float(diag["clipped"]));kls.append(float(diag["kl"]))
                        lrs.append(abs(float(diag["logratio"])));losses.append(float(loss.detach()))
                norm=torch.nn.utils.clip_grad_norm_(backend.adapter.parameters(),1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite adapter gradients")
                opt.step()
        else:
            norm=torch.tensor(0.)
        assert all(p.grad is None for p in backend.net.parameters()),"Base checkpoint received a gradient"
        with torch.no_grad():
            band_norms=[float((backend.adapter.b[i]@backend.adapter.a[i]).norm())
                        for i in range(len(backend.adapter.b))]
        row=dict(epoch=epoch,group_rewards=rewards[0].cpu().tolist(),
                 group_advantages=advantages.cpu().tolist(),transitions=transitions,
                 clip_fraction=float(np.mean(clips)) if clips else 0.,
                 max_abs_logratio=max(lrs,default=0.),mean_conditional_kl=float(np.mean(kls)) if kls else 0.,
                 grad_norm=float(norm),adapter_delta_norm=float(np.linalg.norm(band_norms)),
                 adapter_band_delta_norm=band_norms,
                 policy_sigma_levels=sorted({tr["sigma_hat"] for tr in siblings[0]["records"]}),
                 frozen_base_gradient_count=sum(p.grad is not None for p in backend.net.parameters()),
                 seconds=time.time()-started)
        training.write(json.dumps(row,allow_nan=False)+"\n")
        print("TRAIN "+json.dumps(row,allow_nan=False),flush=True)
        torch.save({k:v.detach().cpu() for k,v in backend.adapter.state_dict().items()},a.out/"adapter.pt")
        last_epoch=epoch
    if last_epoch:
        evaluate(last_epoch)
    json_write(a.out/"DONE.json",dict(complete=last_epoch==a.epochs,epochs_completed=last_epoch,
                                     seconds=time.time()-started))


if __name__=="__main__":
    main()
