"""Paired fresh-seed evaluation: zero adapter vs trained adapter, identical noise."""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from ..backend import RFD3Backend
from ..metrics import metrics, reward

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--count", type=int, default=16)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    cfg = json.loads((run / "config.json").read_text())
    out = run / "fresh_eval.json"
    if out.exists():
        raise SystemExit("done")
    torch.set_num_threads(4)
    be = RFD3Backend(None, cfg["steps"], cfg["length"], cfg["recycles"], "mps", cfg["rank"], cfg["noise_experts"], cfg["seed"], None)
    trained = torch.load(run / "adapter.pt", map_location="cpu")
    zero = {k: torch.zeros_like(v) if k.startswith("b") else v for k, v in trained.items()}
    rows = {"zero": [], "trained": []}
    ca_all = {"zero": [], "trained": []}
    for name, state in (("zero", None), ("trained", trained)):
        if state is None:
            be.adapter.load_state_dict({k: v.to(be.device) for k, v in zero.items()})
        else:
            be.adapter.load_state_dict({k: v.to(be.device) for k, v in state.items()})
        for i in range(a.count):
            o = be.rollout(5_000_000 + i, True, False, cfg["extra_noise"], cfg["active_min"], cfg["active_max"], False, cfg["sampler_kind"])
            ca = o["ca"].cpu()
            m = metrics(ca); m["reward"] = reward(ca, "guarded")
            rows[name].append({k: float(v.squeeze()) for k, v in m.items()})
            ca_all[name].append(ca.numpy())
    np.savez_compressed(run / "fresh_eval_ca.npz", zero=np.concatenate(ca_all["zero"]), trained=np.concatenate(ca_all["trained"]))
    out.write_text(json.dumps(rows, indent=2))
    d = np.array([t["reward"] - z["reward"] for t, z in zip(rows["trained"], rows["zero"])])
    print(a.run, "mean delta", d.mean(), "zero", np.mean([r["reward"] for r in rows["zero"]]), "trained", np.mean([r["reward"] for r in rows["trained"]]))


if __name__ == "__main__":
    main()
