"""Real-checkpoint MPS policy identity and finite rollout gate."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from image_video_grpo.core import normal_logprob
from image_video_grpo.experiment import build, rollout

ROOT = Path(__file__).resolve().parent


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; do not substitute CPU training")
    _, adapter, policy = build("image", "mps")
    out, trajectory = rollout(policy, seed=2401001, n=4, record=True)
    errors = []
    for transition in (trajectory[0], trajectory[len(trajectory) // 2], trajectory[-1]):
        state, action = transition.state.to("mps"), transition.action.to("mps")
        with torch.no_grad():
            mean, std = policy.transition(state, transition.step)
            new_lp = normal_logprob(action, mean, std).cpu()
        errors += (new_lp - transition.old_logprob).abs().tolist()
    result = {"finite": bool(torch.isfinite(out).all()),
              "sample_shape": list(out.shape),
              "max_identity_log_ratio": max(errors),
              "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
              "frozen_backbone_parameters": sum(p.numel() for p in policy.backbone.parameters())}
    print(json.dumps(result), flush=True)
    if not result["finite"] or result["max_identity_log_ratio"] > 0.05:
        (ROOT / "PREFLIGHT_FAILED.json").write_text(json.dumps(result, indent=2))
        raise RuntimeError("Preflight finite/log-ratio identity gate failed")
    (ROOT / "PREFLIGHT.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
