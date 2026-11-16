"""Predeclared three-PDB benchmark for the RFD3 VE-SDE/GRPO adapter."""
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
POLICIES=("stock","ve_zero","ve_trained")


def native_ca(pdb):
    coords={}
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM  ") and line[21]=="A" and line[12:16].strip()=="CA":
            coords[int(line[22:26])]=[float(line[i:i+8]) for i in (30,38,46)]
    return np.asarray([coords[i] for i in range(1,41)],dtype=np.float64)


def aligned_rmsds(ca,native):
    fixed=np.r_[0:14,22:40]
    gap=np.r_[14:22]
    source_center=ca[fixed].mean(0)
    native_center=native[fixed].mean(0)
    u,_,vt=np.linalg.svd((ca[fixed]-source_center).T@(native[fixed]-native_center))
    sign=np.linalg.det(u@vt)
    rotation=u@np.diag([1.,1.,np.sign(sign)])@vt
    aligned=(ca-source_center)@rotation+native_center
    rmsd=lambda indices: float(np.sqrt(np.mean(np.sum((aligned[indices]-native[indices])**2,axis=1))))
    return rmsd(fixed),rmsd(gap)


def ca_metrics(ca):
    # The score is identical to rfd3_grpo.metrics.reward(..., 'guarded').
    center=ca-ca.mean(0)
    rg=float(np.sqrt(np.mean(np.sum(center**2,axis=1))))
    bonds=np.linalg.norm(np.diff(ca,axis=0),axis=1)
    bond_mae=float(np.mean(np.abs(bonds-3.8)))
    bond_ok=float(np.mean((bonds>=3.4)&(bonds<=4.2)))
    pair=np.linalg.norm(ca[:,None]-ca[None,:],axis=2)
    nonlocal_dist=pair[np.subtract.outer(np.arange(40),np.arange(40))>2]
    clash=float(np.mean(nonlocal_dist<3.))
    loop_bonds=bonds[13:22]
    loop_bond_mae=float(np.mean(np.abs(loop_bonds-3.8)))
    return dict(reward=-rg/10-2*bond_mae-10*clash,rg=rg,bond_mae=bond_mae,
                loop_bond_mae=loop_bond_mae,clash_fraction=clash,
                geometry_pass=bool(bond_ok>=.95 and np.min(nonlocal_dist)>=3.))


def run():
    import torch
    from rfd3_grpo.backend import RFD3Backend
    plan=json.loads((ROOT/"PROTOCOL.json").read_text())
    marker=ROOT/"train_1crn"/"DONE.json"
    if not marker.exists() or not json.loads(marker.read_text()).get("complete"):
        raise RuntimeError("Training is incomplete; benchmark cannot start")
    adapter_path=ROOT/"train_1crn"/"adapter.pt"
    deadline=datetime.fromisoformat(plan.get("evaluation_deadline_utc","2026-11-14T08:00:00Z").replace("Z","+00:00"))
    for pdb_id in [plan["train_pdb"],*plan["heldout_pdbs"]]:
        spec=json.loads((ROOT/f"{pdb_id}.json").read_text())
        backend=RFD3Backend(steps=plan["eval_schedule_points"],length=40,specification=spec)
        state=torch.load(adapter_path,map_location="cpu",weights_only=True)
        backend.adapter.load_state_dict(state)
        native=native_ca(ROOT/"pdbs"/f"{pdb_id}.pdb")
        folder=ROOT/"evaluations"/pdb_id
        folder.mkdir(parents=True,exist_ok=True)
        for seed in plan["evaluation_seeds"]:
            for policy in POLICIES:
                target=folder/f"{policy}_{seed}.npz"
                if target.exists():
                    continue
                if datetime.now(timezone.utc)>=deadline:
                    raise TimeoutError("Benchmark deadline reached before starting next sample")
                stock=policy=="stock"
                out=backend.rollout(seed,adapted=policy=="ve_trained",record=False,
                    extra_noise=plan["noise_level"],active_min=plan["active_sigma"][0],
                    active_max=plan["active_sigma"][1],stock=stock,
                    sampler_kind="stock_churn" if stock else "ve_sde")
                ca=out["ca"][0].cpu().numpy().astype(np.float64)
                fixed_rmsd,gap_rmsd=aligned_rmsds(ca,native)
                if fixed_rmsd>1e-3:
                    raise AssertionError(f"Fixed motif moved: {fixed_rmsd} Å")
                row=dict(pdb=pdb_id,policy=policy,seed=seed,
                         fixed_rmsd=fixed_rmsd,gap_rmsd_to_native=gap_rmsd,
                         path_kl=out["path_kl"],**ca_metrics(ca))
                with target.with_suffix(".tmp").open("wb") as f:
                    np.savez_compressed(f,ca=ca,all_coordinates=out["coordinates"].cpu().numpy(),
                                        metadata=json.dumps(row,allow_nan=False))
                target.with_suffix(".tmp").replace(target)
                print(json.dumps(row,allow_nan=False),flush=True)
        (folder/"DONE.json").write_text(json.dumps({"complete":True,"samples":len(plan["evaluation_seeds"])*len(POLICIES)})+"\n")
        del backend
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    report()


def report():
    plan=json.loads((ROOT/"PROTOCOL.json").read_text())
    expected=len(plan["evaluation_seeds"])*len(POLICIES)*3
    observed=sum(1 for p in (ROOT/"evaluations").glob("*/*.npz"))
    complete=observed==expected
    lines=["# PDB-conditioned RFD3 Flow-GRPO-style benchmark","",
        f"**{'Complete' if complete else 'Incomplete'} predeclared benchmark: {observed}/{expected} samples saved.** "
        + ("All predeclared samples finished." if complete else "The evaluation stopped at its predeclared 2026-11-14 08:00 UTC start-sample deadline. Missing samples were not rerun; results below are descriptive and unevenly sampled."),"",
        "Predeclared condition: fixed chain-A residues 1–14 and 23–40; eight residues generated between them. The trained adapter saw only 1CRN. 1PGA and 1UBQ are held-out structures. Higher guarded score and lower gap-to-native Cα RMSD are better. Native gap coordinates were **not** used by the training reward. Gap RMSD is descriptive because the generated gap sequence is unconstrained.","",
        "| PDB | Policy | n | Guarded score | Geometry passes | Loop bond MAE (Å) | Gap-to-native RMSD (Å) | Path KL |","|---|---|---:|---:|---:|---:|---:|---:|"]
    data={}
    for pdb_id in [plan["train_pdb"],*plan["heldout_pdbs"]]:
        data[pdb_id]={}
        for policy in POLICIES:
            rows=[]
            for seed in plan["evaluation_seeds"]:
                target=ROOT/"evaluations"/pdb_id/f"{policy}_{seed}.npz"
                if target.exists():
                    with np.load(target) as saved:
                        ca=saved["ca"]
                        all_coordinates=saved["all_coordinates"]
                        row=json.loads(str(saved["metadata"]))
                    if (ca.shape!=(40,3) or not np.isfinite(ca).all() or
                        all_coordinates.ndim!=3 or all_coordinates.shape[0]!=1 or
                        all_coordinates.shape[-1]!=3 or not np.isfinite(all_coordinates).all()):
                        raise AssertionError(f"Raw coordinate audit failed: {target}")
                    actual=ca_metrics(ca)
                    for key in ("reward","rg","bond_mae","loop_bond_mae","clash_fraction"):
                        if abs(row[key]-actual[key])>1e-4:
                            raise AssertionError(f"Coordinate audit failed: {target}, {key}")
                    native=native_ca(ROOT/"pdbs"/f"{pdb_id}.pdb")
                    fixed_rmsd,gap_rmsd=aligned_rmsds(ca,native)
                    if fixed_rmsd>1e-3 or abs(gap_rmsd-row["gap_rmsd_to_native"])>1e-4:
                        raise AssertionError(f"Alignment audit failed: {target}")
                    rows.append(row)
            data[pdb_id][policy]={r["seed"]:r for r in rows}
            if rows:
                avg=lambda key:np.mean([r[key] for r in rows])
                lines.append(f"| {pdb_id} | {policy} | {len(rows)} | {avg('reward'):.5f} | {sum(r['geometry_pass'] for r in rows)}/{len(rows)} | {avg('loop_bond_mae'):.4f} | {avg('gap_rmsd_to_native'):.3f} | {avg('path_kl'):.3f} |")
            else:
                lines.append(f"| {pdb_id} | {policy} | 0 | — | — | — | — | — |")
    lines += ["","## Paired trained-minus-zero differences","",
              "Both policies use the same VE-SDE sampler and evaluation seeds. For n≥2, intervals bootstrap the available seeds for a fixed adapter and one PDB; with at most four seeds they are exploratory, not a population claim.",""]
    for pdb_id,policies in data.items():
        shared=sorted(set(policies["ve_trained"])&set(policies["ve_zero"]))
        if not shared: continue
        rng=np.random.default_rng(20261114)
        chunks=[]
        for key in ("reward","gap_rmsd_to_native"):
            delta=np.asarray([policies["ve_trained"][s][key]-policies["ve_zero"][s][key] for s in shared])
            ci=np.quantile(rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1),[.025,.975]) if len(delta)>=2 else [float('nan')]*2
            interval=f" [{ci[0]:+.8f}, {ci[1]:+.8f}]" if len(delta)>=2 else " (one seed; no interval)"
            chunks.append(f"Δ{key}={delta.mean():+.8f}"+interval)
        lines.append(f"- **{pdb_id}** (n={len(shared)}): "+"; ".join(chunks)+".")
    for pdb_id,policies in data.items():
        if not set(policies["ve_trained"])&set(policies["ve_zero"]):
            lines.append(f"- **{pdb_id}**: no trained-versus-zero pair was saved.")
    lines += ["","The stock policy uses RFD3's original churn and is not a density-matched zero-adapter reference. VE-SDE uses an approximate EDM score and a finite Euler schedule; no exact stock-marginal equivalence is claimed. Fixed-motif RMSD was checked below 0.001 Å for every saved sample. These metrics do not establish sequence designability, all-atom physical validity or function.",""]
    (ROOT/"RESULTS.md").write_text("\n".join(lines))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",action="store_true")
    args=parser.parse_args()
    report() if args.report else run()
