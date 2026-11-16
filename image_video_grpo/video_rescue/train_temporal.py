"""Supervised temporal pretraining on synthetic moving-MNIST clips."""
from __future__ import annotations

import hashlib
import json
import time

import torch
from torchvision.datasets import MNIST

from rldiff.diffusion import Diffusion
from image_video_grpo.video_rescue.model import ROOT, WORKSPACE, VideoDenoiser, moving_mnist_batch

CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


def main():
    dest = ROOT / "temporal.pt"
    if dest.exists() or (ROOT / "TEMPORAL_DONE.json").exists():
        raise FileExistsError("Temporal pretraining already exists")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("This predeclared laptop run requires accessible MPS")
    torch.manual_seed(CFG["temporal_pretraining_seed"])
    rng = torch.Generator().manual_seed(CFG["temporal_pretraining_seed"] + 1)
    dataset = MNIST(str(WORKSPACE / "data"), train=True, download=False)
    images = dataset.data.float().to(device).unsqueeze(1) / 255
    model = VideoDenoiser().to(device)
    model.temporal.train()
    opt = torch.optim.AdamW(model.temporal.parameters(), lr=CFG["temporal_learning_rate"], weight_decay=0)
    diffusion = Diffusion(device=device)
    first = time.monotonic()
    try:
        with (ROOT / "temporal_train.jsonl").open("w") as log:
            for update in range(1, CFG["temporal_pretraining_steps"] + 1):
                x0 = moving_mnist_batch(images, CFG["temporal_batch"], rng)
                t = torch.randint(0, diffusion.T, (len(x0),), device=device)
                noise = torch.randn_like(x0)
                xt = diffusion.q_sample(x0, t, noise)
                prediction = model(xt, t)
                loss = (prediction - noise).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite temporal pretraining loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.temporal.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Nonfinite temporal pretraining gradient")
                opt.step()
                if update == 1 or update % 50 == 0:
                    row = {"update": update, "loss": float(loss.detach()),
                           "grad_norm": float(grad), "elapsed_s": round(time.monotonic() - first, 2)}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
        torch.save(model.temporal.cpu().state_dict(), dest)
        (ROOT / "TEMPORAL_DONE.json").write_text(json.dumps({
            "updates": CFG["temporal_pretraining_steps"],
            "checkpoint_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "elapsed_s": round(time.monotonic() - first, 2)}, indent=2) + "\n")
    except BaseException as exc:
        (ROOT / "TEMPORAL_FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
