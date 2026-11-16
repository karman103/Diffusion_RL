"""Inspect the first ZeroScope denoising step without decoding or training."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import TextToVideoSDPipeline


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "checkpoints" / "zeroscope_v2_576w"
PROMPT = "A single red ball rolls from left to right across a plain floor."


def stats(x):
    finite = torch.isfinite(x)
    y = x[finite].float()
    return {"dtype": str(x.dtype), "shape": list(x.shape),
            "finite_fraction": float(finite.float().mean()),
            "min_finite": float(y.min()) if y.numel() else None,
            "max_finite": float(y.max()) if y.numel() else None,
            "mean_abs_finite": float(y.abs().mean()) if y.numel() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), required=True)
    args = parser.parse_args()
    output = ROOT / "zeroscope_preflight" / f"first_step_{args.dtype}.json"
    if output.exists():
        raise FileExistsError(output)
    dtype = getattr(torch, args.dtype)
    torch.mps.set_per_process_memory_fraction(0.90)
    report = {"dtype": args.dtype, "status": "started"}
    try:
        pipe = TextToVideoSDPipeline.from_pretrained(MODEL, torch_dtype=dtype, local_files_only=True)
        pipe.enable_attention_slicing()
        pipe.to("mps")
        positive, negative = pipe.encode_prompt(PROMPT, "mps", 1, True, None)
        pipe.scheduler.set_timesteps(16, device="mps")
        t = pipe.scheduler.timesteps[0]
        latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, 8, 192, 320,
                                        positive.dtype, "mps", torch.Generator(device="cpu").manual_seed(20261115))
        report["initial"] = stats(latents)
        with torch.inference_mode():
            uncond = pipe.unet(latents, t, encoder_hidden_states=negative, return_dict=False)[0]
            report["unconditional_prediction"] = stats(uncond)
            cond = pipe.unet(latents, t, encoder_hidden_states=positive, return_dict=False)[0]
            report["conditional_prediction"] = stats(cond)
            guided = uncond + 7.5 * (cond - uncond)
            report["guided_prediction"] = stats(guided)
            prior = pipe.scheduler.step(guided, t, latents).prev_sample
            report["next_latents"] = stats(prior)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
