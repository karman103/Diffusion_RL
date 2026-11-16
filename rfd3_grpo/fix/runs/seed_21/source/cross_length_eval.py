"""Predeclared transfer check of length-32 GRPO adapters at another length."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .backend import RFD3Backend
from .grpo import NoiseLoRA
from .heldout_eval import RUNS, TASK, write_json
from .metrics import metrics, reward


POLICIES = ("stock", "zero", "late_zero", *RUNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, choices=(24, 40), required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--first-seed", type=int, default=4_000_000)
    parser.add_argument("--deadline", default="2026-11-13T10:26:00Z")
    args = parser.parse_args()
    if args.count < 2 or args.first_seed < 4_000_000:
        raise ValueError("Use at least two new noise seeds for cross-length evaluation")
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("The experiment deadline has passed")
    lock = (TASK / "queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another RFD3 GPU experiment is active") from exc
    if not (TASK / "heldout_3000000_n16/LATE_ZERO_DONE.json").exists():
        raise RuntimeError("Finish the declared length-32 heldout evaluation first")
    configs = {}
    hashes = {}
    for name in RUNS:
        folder = TASK / "runs" / name
        if not json.loads((folder / "DONE.json").read_text()).get("complete"):
            raise RuntimeError(f"Training run {name} is incomplete")
        configs[name] = json.loads((folder / "config.json").read_text())
        hashes[name] = hashlib.sha256((folder / "adapter.pt").read_bytes()).hexdigest()
    common = configs[RUNS[0]]
    if any(cfg["length"] != 32 or cfg["checkpoint_sha256"] != common["checkpoint_sha256"]
           or cfg["steps"] != common["steps"] or cfg["recycles"] != common["recycles"]
           for cfg in configs.values()):
        raise ValueError("Saved adapters do not share the declared training condition")
    folder = TASK / "cross_length" / f"length{args.length}"
    folder.mkdir(parents=True, exist_ok=True)
    plan = dict(training_length=32, evaluation_length=args.length,
                count=args.count, first_seed=args.first_seed, policies=POLICIES,
                checkpoint_sha256=common["checkpoint_sha256"],
                adapter_sha256=hashes, deadline=args.deadline)
    plan_path = folder / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing cross-length plan differs")
    else:
        if any(folder.iterdir()):
            raise FileExistsError("Cross-length output exists without a plan")
        write_json(plan_path, plan)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("WANDB_MODE", "disabled")
    torch.set_num_threads(4)
    backend = RFD3Backend(steps=common["steps"], length=args.length,
                          recycles=common["recycles"], device=common["device"],
                          rank=common["rank"], noise_experts=common["noise_experts"],
                          adapter_seed=common["adapter_initialization_seed"])
    modules = {}
    for name, cfg in configs.items():
        boundaries = () if cfg["noise_experts"] == "single" else (2., 8.)
        module = NoiseLoRA(backend.channels, rank=cfg["rank"],
                           boundaries=boundaries, seed=cfg["adapter_initialization_seed"])
        module.load_state_dict(torch.load(TASK / "runs" / name / "adapter.pt",
                                          map_location="cpu", weights_only=True))
        modules[name] = module.to(common["device"]).eval()
    if any(p.requires_grad for p in backend.net.parameters()):
        raise AssertionError("Pretrained RFD3 network must remain frozen")
    for index in range(args.count):
        seed = args.first_seed + index
        for policy in POLICIES:
            target = folder / f"{policy}_{seed}.npz"
            if target.exists():
                continue
            if datetime.now(timezone.utc) >= deadline:
                write_json(folder / "STOPPED_AT_DEADLINE.json", dict(seed=seed, policy=policy))
                return
            cfg = configs["late_s0"] if policy == "late_zero" else (
                common if policy in ("stock", "zero") else configs[policy])
            if policy in RUNS:
                backend.adapter = modules[policy]
            result = backend.rollout(seed, adapted=policy in RUNS,
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
            print(json.dumps(dict(length=args.length, policy=policy, seed=seed,
                                  guarded_reward=float(values["guarded_reward"].item()),
                                  geometry_pass=float(values["geometry_pass"].item()),
                                  path_kl=float(result["path_kl"]))), flush=True)
    write_json(folder / "DONE.json",
               dict(completed_utc=datetime.now(timezone.utc).isoformat(), count=args.count))


if __name__ == "__main__":
    main()
