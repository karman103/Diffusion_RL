"""Small MPS capability probe; does not load or train a video checkpoint.

Checks whether a frozen INT8 torchao linear layer can pass gradients to a
separate low-rank adapter on this machine. This is only a kernel/autograd gate.
"""

import json
import platform

import torch
from torch import nn
from torchao.quantization import Int8WeightOnlyConfig, quantize_


def main() -> None:
    result = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "scope": "1024x1024 frozen INT8 linear plus rank-4 LoRA; not CogVideoX",
    }
    if not result["mps_available"]:
        print(json.dumps(result, indent=2))
        raise SystemExit(2)

    torch.manual_seed(17)
    device = torch.device("mps")
    base = nn.Linear(1024, 1024, bias=False)
    base.requires_grad_(False)
    quantize_(base, Int8WeightOnlyConfig(version=2))
    base.to(device)
    # Keep the small optimizer state in fp32; fp16 AdamW underflowed on MPS.
    down = nn.Linear(1024, 4, bias=False, device=device, dtype=torch.float32)
    up = nn.Linear(4, 1024, bias=False, device=device, dtype=torch.float32)
    nn.init.zeros_(up.weight)
    opt = torch.optim.AdamW([*down.parameters(), *up.parameters()], lr=1e-3)
    x = torch.randn(2, 1024, device=device, dtype=torch.float16)

    y = base(x).float() + up(down(x.float()))
    loss = y.square().mean()
    loss.backward()
    result["forward_finite"] = bool(torch.isfinite(y).all().item())
    result["loss_finite"] = bool(torch.isfinite(loss).item())
    result["lora_grad_finite"] = bool(torch.isfinite(up.weight.grad).all().item())
    result["lora_grad_norm"] = float(up.weight.grad.norm().item())
    result["base_grad_is_none"] = base.weight.grad is None
    before = up.weight.detach().clone()
    opt.step()
    result["adapter_delta_norm"] = float((up.weight - before).norm().item())
    result["adapter_weights_finite"] = bool(torch.isfinite(up.weight).all().item())
    result["mps_peak_bytes"] = torch.mps.driver_allocated_memory()
    print(json.dumps(result, indent=2))
    assert all(result[k] for k in ("forward_finite", "loss_finite", "lora_grad_finite", "base_grad_is_none", "adapter_weights_finite"))
    assert result["lora_grad_norm"] > 0 and result["adapter_delta_norm"] > 0


if __name__ == "__main__":
    main()
