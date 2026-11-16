"""Run an auditable Genie head-adaptation experiment; python -m protein_poc.experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .core import ProteinPolicy, geometry, reward, distance_diversity, joint_logprob, save_pdb, ROOT


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def summary(x, kl, objective):
    values = geometry(x)
    values["reward"] = reward(x, objective)
    values["path_kl"] = kl
    raw = {k: v.detach().cpu().numpy() for k, v in values.items()}
    result = {k: float(v.mean()) for k, v in raw.items()}
    result["diversity"] = distance_diversity(x)
    result["n"] = len(x)
    # Unlike a degenerate all-success bootstrap, Wilson remains informative
    # when the finite evaluation set has zero observed failures.
    phat, n, z = result["geometry_pass"], len(x), 1.959963984540054
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n))/denom
    half = z*np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/denom
    result["geometry_pass_wilson_ci95"] = [max(0,center-half),min(1,center+half)]
    # Bootstrap uncertainty over generated samples, NOT training-seed uncertainty.
    rng = np.random.default_rng(9876)
    idx = rng.integers(len(x), size=(2000, len(x)))
    result["sample_bootstrap_ci95"] = {
        k: np.quantile(v[idx].mean(1), [.025, .975]).tolist() for k, v in raw.items()
    }
    return result, raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--mode", choices=["train", "baseline", "evaluate"], default="train")
    p.add_argument("--head", type=Path, help="Load a saved head for fresh-seed evaluation")
    p.add_argument("--eval-seed", type=int, default=1_000_000)
    p.add_argument("--objective", choices=["compact", "guarded", "random"], default="compact")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--eval-n", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--active-min", type=int, default=50)
    p.add_argument("--active-max", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kl-beta", type=float, default=.01)
    p.add_argument("--reuse", type=int, default=4)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--bon-n", type=int, default=4)
    p.add_argument("--bon-groups", type=int, default=16)
    p.add_argument("--deadline", default=None, help="UTC ISO deadline, checked between epochs")
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    if min(a.batch, a.eval_n) < 2 or a.bon_n < 1 or a.bon_groups < 2 or a.eval_every < 1:
        raise ValueError("Need batch/eval >=2, BoN>=1, groups>=2, and eval interval>=1")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if a.device == "auto" else a.device
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    policy = ProteinPolicy(device, a.steps, a.length, a.active_min, a.active_max)
    if a.head:
        if a.mode != "evaluate":
            raise ValueError("Loading a head is supported only in evaluation mode")
        policy.head.load_state_dict(torch.load(a.head, map_location=device, weights_only=True))
    start = time.time()
    objective = a.objective if a.objective != "random" else "compact"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(actual_device=device, torch=torch.__version__, python=platform.python_version(),
                  base_parameters=sum(t.numel() for t in policy.model.parameters()),
                  trainable_parameters=sum(t.numel() for t in policy.head.parameters()),
                  checkpoint_sha256=hashlib.sha256(policy.checkpoint.read_bytes()).hexdigest(),
                  upstream_commit=subprocess.check_output(["git", "-C", str(ROOT / "third_party/genie"), "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")},
                  started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                  sampler="respaced DDIM eta=1; fixed terminal decoder; time-gated final translation head")
    if a.head:
        config["loaded_head_sha256"] = hashlib.sha256(a.head.read_bytes()).hexdigest()
    write_json(a.out / "config.json", config)
    logs = (a.out / "metrics.jsonl").open("w", buffering=1)
    training = (a.out / "training.jsonl").open("w", buffering=1)

    def evaluate(epoch):
        chunks, kls = [], []
        # Disjoint per-chunk seeds; common random numbers across checkpoints/runs.
        for k in range(0, a.eval_n, 8):
            x, _, kl = policy.rollout(min(8, a.eval_n-k), a.eval_seed + k, adapted=True)
            chunks.append(x.cpu()); kls.append(kl.cpu())
        x, kl = torch.cat(chunks), torch.cat(kls)
        result, raw = summary(x, kl, objective)
        result.update(epoch=epoch, seconds=time.time()-start, training_trajectories=epoch*a.batch)
        target = a.out / f"eval_{epoch:04d}.npz"
        with target.with_suffix(".npz.tmp").open("wb") as f:
            np.savez_compressed(f, coordinates=x.numpy(), **raw)
        target.with_suffix(".npz.tmp").replace(target)
        for i in range(min(4, len(x))):
            save_pdb(a.out / f"sample_{epoch:04d}_{i:02d}.pdb", x[i])
        torch.save({k: v.detach().cpu() for k, v in policy.head.state_dict().items()}, a.out / "head.pt")
        logs.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
        return result

    evaluate(0)
    if a.mode == "evaluate":
        write_json(a.out / "DONE.json", {"complete": True, "seconds": time.time()-start})
        return
    if a.mode == "baseline":
        # Independent candidate groups; select by proxy only.
        pool = []
        total = a.bon_n * a.bon_groups
        for k in range(0, total, 8):
            x, _, _ = policy.rollout(min(8, total-k), 2_000_000 + k, adapted=False)
            pool.append(x.cpu())
        pool = torch.cat(pool)
        vals = reward(pool, objective).reshape(a.bon_groups, a.bon_n)
        ids = vals.argmax(1) + torch.arange(a.bon_groups)*a.bon_n
        selected = pool[ids]
        metrics = geometry(selected)
        raw = {k: v.numpy() for k, v in metrics.items()}
        np.savez_compressed(a.out / "best_of_n.npz", coordinates=selected.numpy(), pool=pool.numpy(),
                            selected_indices=ids.numpy(), reward=reward(selected, objective).numpy(), **raw)
        result = {k: float(v.mean()) for k, v in raw.items()}
        result.update(reward=reward(selected, objective).mean().item(), diversity=distance_diversity(selected),
                      candidates=total, selected=a.bon_groups, best_of=a.bon_n, seconds=time.time()-start)
        write_json(a.out / "best_of_n.json", result)
        write_json(a.out / "DONE.json", {"complete": True, "seconds": time.time()-start})
        return

    opt = torch.optim.Adam(policy.head.parameters(), lr=a.lr)
    last_eval, last_epoch = 0, 0
    for epoch in range(1, a.epochs+1):
        if a.deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(a.deadline.replace("Z", "+00:00")):
            break
        x, trajectory, _ = policy.rollout(a.batch, 10_000 + 10_000*a.seed + epoch, record=True)
        with torch.no_grad():
            if a.objective == "random":
                r = torch.randn(a.batch, generator=torch.Generator().manual_seed(3_000_000 + a.seed*10000 + epoch)).to(device)
            else:
                r = reward(x, a.objective)
            # Leave-one-out baseline avoids the finite-batch self-baseline shrinkage.
            adv = (r - (r.sum()-r)/(len(r)-1)) / r.std().clamp_min(1e-6)
            adv = adv.clamp(-5, 5)
        if not trajectory:
            raise ValueError("No stochastic steps within the active time window")
        cf, maxlog, losses, kls = [], [], [], []
        for reuse in range(a.reuse):
            opt.zero_grad(set_to_none=True)
            for tr in trajectory:
                mean = tr["refmean"] + tr["coefficient"] * policy.head(tr["features"], tr["rotations"])
                logratio = joint_logprob(tr["action"], mean, tr["sigma"]) - tr["old_logprob"]
                if not torch.isfinite(logratio).all() or logratio.abs().max() > 40:
                    raise FloatingPointError("PPO ratios unstable: reduce LR/reuse")
                ratio = logratio.exp()
                pg = -torch.minimum(ratio*adv, ratio.clamp(1-a.clip, 1+a.clip)*adv).mean()
                kl = (mean - tr["refmean"]).square().sum((-1,-2)).mean() / (2*tr["sigma"]**2)
                loss = (pg + a.kl_beta*kl) / len(trajectory)
                loss.backward()
                cf.append(((ratio-1).abs() > a.clip).float().mean().item())
                maxlog.append(logratio.abs().max().item()); losses.append(pg.item()); kls.append(kl.item())
            gradnorm = torch.nn.utils.clip_grad_norm_(policy.head.parameters(), 1.0)
            if not torch.isfinite(gradnorm):
                raise FloatingPointError("Non-finite gradient")
            opt.step()
        row = dict(epoch=epoch, reward=float(r.mean()), clip_fraction=float(np.mean(cf)),
                   max_abs_logratio=max(maxlog), mean_step_kl=float(np.mean(kls)),
                   policy_loss=float(np.mean(losses)), grad_norm=float(gradnorm), seconds=time.time()-start)
        training.write(json.dumps(row, allow_nan=False)+"\n")
        print("train " + json.dumps(row), flush=True)
        last_epoch = epoch
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            evaluate(epoch); last_eval = epoch
    if last_eval != last_epoch:
        evaluate(last_epoch)
    write_json(a.out / "DONE.json", {"complete": last_epoch == a.epochs, "epochs_completed": last_epoch,
                                      "seconds": time.time()-start})


if __name__ == "__main__":
    main()
