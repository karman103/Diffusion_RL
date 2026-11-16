"""Bounded, stock-sampler ZeroScope inference gate on Apple MPS."""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import TextToVideoSDPipeline

from frame_export import save_frames


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "checkpoints" / "zeroscope_v2_576w"
OUTPUT_ROOT = ROOT / "zeroscope_preflight"
PROMPT = "A single red ball rolls from left to right across a plain floor."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="original_70")
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--vae-fp32", action="store_true")
    parser.add_argument("--unet-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run outside the tool sandbox")
    if not 0 < args.memory_fraction <= 0.95:
        raise ValueError("Memory fraction must be in (0, 0.95]")
    output = OUTPUT_ROOT / args.run_name
    if (output / "report.json").exists():
        raise FileExistsError(f"Run already recorded: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    started = time.monotonic()
    report = {
        "model": "cerspense/zeroscope_v2_576w",
        "revision": "6963642a64dbefa93663d1ecebb4ceda2d9ecb28",
        "sampler": "stock DDIMScheduler, eta=0",
        "prompt": PROMPT,
        "seed": 20261115,
        "width": args.width,
        "height": args.height,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": 7.5,
        "unet_dtype": args.unet_dtype,
        "vae_decode_dtype": "float32" if args.vae_fp32 else "float16",
        "mps_memory_fraction": args.memory_fraction,
    }
    try:
        pipeline = TextToVideoSDPipeline.from_pretrained(
            MODEL,
            torch_dtype=getattr(torch, args.unet_dtype),
            local_files_only=True,
        )
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
        pipeline.to("mps")
        report["last_finite_step"] = -1

        def check_step(index, timestep, state):
            if not bool(torch.isfinite(state).all().item()):
                raise FloatingPointError(f"Non-finite latents after denoising step {index}, timestep {int(timestep)}")
            report["last_finite_step"] = int(index)

        with torch.inference_mode():
            latents = pipeline(
                PROMPT,
                num_frames=report["num_frames"],
                height=report["height"],
                width=report["width"],
                num_inference_steps=report["num_inference_steps"],
                guidance_scale=report["guidance_scale"],
                generator=torch.Generator(device="cpu").manual_seed(report["seed"]),
                output_type="latent",
                callback=check_step,
                callback_steps=1,
            ).frames
        report["latents_finite"] = bool(torch.isfinite(latents).all().item())
        report["latent_min"] = float(latents.min().item())
        report["latent_max"] = float(latents.max().item())
        torch.save(latents.detach().cpu(), output / "latents.pt")
        if not report["latents_finite"]:
            raise FloatingPointError("Denoising produced non-finite latents")
        if args.vae_fp32:
            torch.mps.empty_cache()
            pipeline.vae.to(dtype=torch.float32)
            latents = latents.float()
        with torch.inference_mode():
            decoded = pipeline.decode_latents(latents)
            frames = pipeline.video_processor.postprocess_video(video=decoded, output_type="np")[0]
        report.update(save_frames(frames, output, report["num_frames"]))
        report["status"] = "finite_generated; visual review required"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
