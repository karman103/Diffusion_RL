"""One zero-adapter sample from the modified stochastic ZeroScope policy."""

import json
import time

import torch
from diffusers import TextToVideoSDPipeline

from text_video_grpo.frame_export import save_frames
from text_video_grpo.pilot_reward import red_motion_score
from text_video_grpo.video_grpo_pilot import CFG, MODEL, OutputResidual, ROOT, rollout


OUTPUT = ROOT / "zeroscope_preflight" / "stochastic8_residual"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    torch.mps.set_per_process_memory_fraction(CFG["mps_memory_fraction"])
    started = time.monotonic()
    report = {"status": "started", "protocol": CFG, "seed": 20261115,
              "sampler": "modified stochastic DDIM eta=0.5, zero output residual"}
    try:
        pipe = TextToVideoSDPipeline.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                     local_files_only=True)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.to("mps")
        pipe.unet.eval().requires_grad_(False)
        adapter = OutputResidual().to("mps").eval()
        embeds = pipe.encode_prompt(CFG["prompt"], "mps", 1, True, None)
        pipe.text_encoder.to("cpu")
        pipe.vae.to("cpu", dtype=torch.float32)
        torch.mps.empty_cache()
        pipe.scheduler.set_timesteps(CFG["steps"], device="mps")
        frames, _ = rollout(pipe, adapter, embeds, report["seed"], record=False)
        report.update(save_frames(frames, OUTPUT, CFG["frames"]))
        report.update(red_motion_score(frames))
        report["status"] = "finite_generated; visual review required"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
