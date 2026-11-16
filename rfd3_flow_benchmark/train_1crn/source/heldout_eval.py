"""Fresh-noise, paired evaluation of completed real RFD3 adapter runs.

This process acquires the same exclusive lock as run_queue, so it cannot overlap
the sequential training queue. It never trains or edits the saved adapters.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .backend import RFD3Backend, ROOT
from .grpo import NoiseLoRA
from .metrics import metrics, reward


TASK = ROOT / "rfd3_grpo"
RUNS = ("main_s0", "main_s1", "main_s2", "random_s0", "single_s0", "late_s0")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=TASK / "heldout_3000000_n16")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--first-seed", type=int, default=3_000_000)
    parser.add_argument("--deadline", default="2026-11-13T10:26:00Z")
    args = parser.parse_args()
    if args.count < 2 or args.first_seed < 2_000_100:
        raise ValueError("Use at least two new evaluation seeds beyond the training evaluation range")
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("The experiment deadline has passed")
    lock = (TASK / "queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("The RFD3 training queue is active; do not overlap GPU runs") from exc
    if not (TASK / "QUEUE_FINISHED.json").exists():
        raise RuntimeError("Queue lock is free but queue completion marker is missing")
    configs = {}
    adapter_hashes = {}
    for name in RUNS:
        folder = TASK / "runs" / name
        if not json.loads((folder / "DONE.json").read_text()).get("complete"):
            raise RuntimeError(f"Run {name} is not complete")
        configs[name] = json.loads((folder / "config.json").read_text())
        adapter_hashes[name] = hashlib.sha256((folder / "adapter.pt").read_bytes()).hexdigest()
    common = configs[RUNS[0]]
    for name, cfg in configs.items():
        for key in ("checkpoint_sha256", "foundry_commit", "length", "steps", "recycles", "conditioning"):
            if cfg[key] != common[key]:
                raise ValueError(f"{name} has different {key}; paired comparison would be invalid")
    plan = dict(first_seed=args.first_seed, count=args.count, policies=["stock", "zero", *RUNS],
                adapter_sha256=adapter_hashes, checkpoint_sha256=common["checkpoint_sha256"],
                foundry_commit=common["foundry_commit"], length=common["length"],
                steps=common["steps"], recycles=common["recycles"],
                declared_before_sampling_utc=datetime.now(timezone.utc).isoformat(),
                deadline=args.deadline)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        for key in plan:
            if key != "declared_before_sampling_utc" and previous[key] != plan[key]:
                raise ValueError(f"Existing heldout plan differs at {key}")
    else:
        if any(args.out.iterdir()):
            raise FileExistsError("Heldout output exists without a matching plan")
        write_json(plan_path, plan)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("WANDB_MODE", "disabled")
    torch.set_num_threads(4)
    backend = RFD3Backend(steps=common["steps"], length=common["length"],
                          recycles=common["recycles"], device=common["device"],
                          rank=common["rank"], noise_experts=common["noise_experts"],
                          adapter_seed=common["adapter_initialization_seed"])
    adapter_modules = {}
    for name, cfg in configs.items():
        boundaries = () if cfg["noise_experts"] == "single" else (2., 8.)
        module = NoiseLoRA(backend.channels, rank=cfg["rank"],
                           boundaries=boundaries, seed=cfg["adapter_initialization_seed"])
        module.load_state_dict(torch.load(TASK / "runs" / name / "adapter.pt",
                                          map_location="cpu", weights_only=True))
        adapter_modules[name] = module.to(common["device"]).eval()
    if any(p.requires_grad for p in backend.net.parameters()):
        raise AssertionError("Pretrained RFD3 network must remain frozen")
    policies = plan["policies"]
    for i in range(args.count):
        seed = args.first_seed + i
        for policy in policies:
            target = args.out / f"{policy}_{seed}.npz"
            if target.exists():
                continue
            if datetime.now(timezone.utc) >= deadline:
                write_json(args.out / "STOPPED_AT_DEADLINE.json", dict(seed=seed, policy=policy))
                return
            cfg = common if policy in ("stock", "zero") else configs[policy]
            if policy not in ("stock", "zero"):
                backend.adapter = adapter_modules[policy]
            result = backend.rollout(seed, adapted=policy not in ("stock", "zero"),
                                     record=False, stock=policy == "stock",
                                     extra_noise=cfg["extra_noise"],
                                     active_min=cfg["active_min"],
                                     active_max=cfg["active_max"])
            ca = result["ca"].cpu()
            values = metrics(ca)
            values["guarded_reward"] = reward(ca, "guarded")
            values["path_kl"] = torch.tensor([result["path_kl"]])
            arrays = {key: value.numpy() for key, value in values.items()}
            arrays["ca"] = ca.numpy()
            arrays["coordinates"] = result["coordinates"].cpu().numpy()
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f, **arrays)
            temporary.replace(target)
            print(json.dumps(dict(policy=policy, seed=seed,
                                  guarded_reward=float(values["guarded_reward"].item()),
                                  geometry_pass=float(values["geometry_pass"].item()),
                                  path_kl=float(result["path_kl"]))), flush=True)
    write_json(args.out / "DONE.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                            count=args.count, policies=policies))


if __name__ == "__main__":
    main()
