"""Matched zero-adapter reference for the heldout late-window policy."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from .backend import RFD3Backend
from .heldout_eval import TASK, write_json
from .metrics import metrics, reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=TASK / "heldout_3000000_n16")
    parser.add_argument("--deadline", default="2026-11-13T10:26:00Z")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("The experiment deadline has passed")
    lock = (TASK / "queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another RFD3 GPU evaluation is active") from exc
    if not (args.folder / "DONE.json").exists():
        raise RuntimeError("Primary heldout evaluation must finish first")
    plan = json.loads((args.folder / "plan.json").read_text())
    cfg = json.loads((TASK / "runs/late_s0/config.json").read_text())
    late_plan = dict(first_seed=plan["first_seed"], count=plan["count"],
                     active_min=cfg["active_min"], active_max=cfg["active_max"],
                     extra_noise=cfg["extra_noise"], checkpoint_sha256=cfg["checkpoint_sha256"],
                     deadline=args.deadline)
    path = args.folder / "late_zero_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != late_plan:
            raise ValueError("Existing matched baseline plan differs")
    else:
        write_json(path, late_plan)
    torch.set_num_threads(4)
    backend = RFD3Backend(steps=cfg["steps"], length=cfg["length"],
                          recycles=cfg["recycles"], device=cfg["device"],
                          rank=cfg["rank"], noise_experts=cfg["noise_experts"],
                          adapter_seed=cfg["adapter_initialization_seed"])
    for seed in range(plan["first_seed"], plan["first_seed"] + plan["count"]):
        target = args.folder / f"late_zero_{seed}.npz"
        if target.exists():
            continue
        if datetime.now(timezone.utc) >= deadline:
            write_json(args.folder / "LATE_ZERO_STOPPED_AT_DEADLINE.json", dict(seed=seed))
            return
        result = backend.rollout(seed, adapted=False, record=False, stock=False,
                                 extra_noise=cfg["extra_noise"],
                                 active_min=cfg["active_min"], active_max=cfg["active_max"])
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
        print(json.dumps(dict(policy="late_zero", seed=seed,
                              guarded_reward=float(values["guarded_reward"].item()),
                              geometry_pass=float(values["geometry_pass"].item()))), flush=True)
    write_json(args.folder / "LATE_ZERO_DONE.json",
               dict(completed_utc=datetime.now(timezone.utc).isoformat(), count=plan["count"]))


if __name__ == "__main__":
    main()
