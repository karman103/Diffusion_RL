"""Joint four-frame DDPM pretraining with EMA and two resumable checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import time

import torch
from torchvision.datasets import MNIST

from rldiff.diffusion import Diffusion
from image_video_grpo.video_rescue.model import moving_mnist_batch
from image_video_grpo.video_rescue.joint.model import CFG, ROOT, WORKSPACE, build_from_image


def main():
    if (ROOT / "JOINT_DONE.json").exists():
        raise FileExistsError("Joint-video pretraining already completed")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS required for this laptop run")
    torch.manual_seed(CFG["training_seed"])
    rng = torch.Generator().manual_seed(CFG["training_seed"] + 1)
    images = MNIST(str(WORKSPACE / "data"), train=True, download=False).data.float().to(device).unsqueeze(1) / 255
    model = build_from_image().to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0)
    diffusion = Diffusion(device=device)
    start = 1
    resume = ROOT / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["optimizer"])
        rng.set_state(state["data_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.mps.set_rng_state(state["mps_rng"])
        start = state["step"] + 1
    first = time.monotonic()
    try:
        with (ROOT / "train.jsonl").open("a") as log:
            for step in range(start, CFG["training_steps"] + 1):
                x0 = moving_mnist_batch(images, CFG["batch_size"], rng)
                t = torch.randint(0, diffusion.T, (len(x0),), device=device)
                noise = torch.randn_like(x0)
                pred = model(diffusion.q_sample(x0, t, noise), t)
                loss = (pred - noise).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint-video loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Nonfinite joint-video gradient")
                opt.step()
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.lerp_(p, 1 - CFG["ema_decay"])
                if step == start or step % 50 == 0:
                    row = {"step": step, "loss": float(loss.detach()),
                           "grad_norm": float(grad), "elapsed_s_this_process": round(time.monotonic()-first, 2)}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if step in CFG["save_at"]:
                    torch.save({"step": step,
                                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                                "optimizer": opt.state_dict(), "data_rng": rng.get_state(),
                                "torch_rng": torch.get_rng_state(),
                                "mps_rng": torch.mps.get_rng_state()}, resume)
        torch.save({k: v.detach().cpu() for k, v in ema.state_dict().items()}, ROOT / "joint_ema.pt")
        (ROOT / "JOINT_DONE.json").write_text(json.dumps({
            "steps": CFG["training_steps"], "sha256": hashlib.sha256((ROOT / "joint_ema.pt").read_bytes()).hexdigest(),
            "elapsed_s_this_process": round(time.monotonic()-first, 2)}, indent=2) + "\n")
    except BaseException as exc:
        (ROOT / "JOINT_FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
