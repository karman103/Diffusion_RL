"""One bounded ZeroScope UNet LoRA backward pass; no GRPO claim."""

import json
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from diffusers import UNet3DConditionModel
from peft import LoraConfig, get_peft_model


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "checkpoints" / "zeroscope_v2_576w" / "unet"
OUTPUT = ROOT / "zeroscope_preflight" / "adapter_backward_f8_160_m95_slice.json"


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.manual_seed(20261115)
    torch.mps.set_per_process_memory_fraction(0.95)
    started = time.monotonic()
    result = {
        "scope": "single synthetic-noise UNet forward/backward, no text encoder or video decoding",
        "model": "cerspense/zeroscope_v2_576w",
        "revision": "6963642a64dbefa93663d1ecebb4ceda2d9ecb28",
        "weight_dtype": "float32",
        "adapter_dtype": "float32",
        "latent_shape": [1, 4, 8, 20, 32],
        "cross_attention_shape": [1, 77, 1024],
        "mps_memory_fraction": 0.95,
    }
    try:
        model = UNet3DConditionModel.from_pretrained(MODEL, torch_dtype=torch.float32, local_files_only=True)
        model.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_v"]))
        model.eval()
        model.base_model.model.set_attention_slice("auto")
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()
        model.to("mps")
        trainable = [p for p in model.parameters() if p.requires_grad]
        result["trainable_parameters"] = sum(p.numel() for p in trainable)
        result["frozen_parameters"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        x = torch.randn(*result["latent_shape"], dtype=torch.float32, device="mps", requires_grad=True)
        text = torch.randn(*result["cross_attention_shape"], dtype=torch.float32, device="mps")
        target = torch.randn_like(x)
        timestep = torch.tensor(500, device="mps")
        prediction = checkpoint(
            lambda state: model(state, timestep, encoder_hidden_states=text).sample,
            x,
            use_reentrant=True,
        )
        loss = (prediction.float() - target.float()).square().mean()
        loss.backward()
        result["output_finite"] = bool(torch.isfinite(prediction).all().item())
        result["loss_finite"] = bool(torch.isfinite(loss).item())
        result["grad_finite"] = all(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in trainable)
        result["frozen_grad_count"] = sum(p.grad is not None for p in model.parameters() if not p.requires_grad)
        result["grad_norm"] = float(torch.sqrt(sum((p.grad.float().square().sum() for p in trainable))).item())
        result["status"] = "passed" if all((result["output_finite"], result["loss_finite"], result["grad_finite"], result["frozen_grad_count"] == 0, result["grad_norm"] > 0)) else "failed_numerics"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
