"""One-update, real-checkpoint stochastic-DDIM GRPO feasibility pilot.

This uses a modified low-cost ZeroScope sampler. It is deliberately separate
from stock 16-frame/40-step inference, and is not a full video benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from diffusers import TextToVideoSDPipeline

from text_video_grpo.ddim_policy import action_log_probability, transition_parameters
from text_video_grpo.frame_export import save_frames
from text_video_grpo.pilot_reward import group_advantages, red_motion_score


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "PILOT_PROTOCOL.json").read_text())
RUN = ROOT / "runs" / "red_ball_residual_grpo_v1"
MODEL = ROOT / "checkpoints" / "zeroscope_v2_576w"


class OutputResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 4, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, state, base_epsilon):
        return self.conv2(torch.nn.functional.silu(self.conv1(
            torch.cat((state.float(), base_epsilon.float()), dim=1))))


def noise(shape, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def single_eps(pipe, state, timestep, embeds):
    model_input = pipe.scheduler.scale_model_input(state.float(), timestep)
    with torch.no_grad():
        return pipe.unet(model_input, timestep, encoder_hidden_states=embeds,
                         return_dict=False)[0].float()


def eps_prediction(pipe, adapter, state, timestep, embeds, *, adapt=True):
    positive, negative = embeds
    unconditional = single_eps(pipe, state, timestep, negative)
    conditional = single_eps(pipe, state, timestep, positive)
    if adapt:
        conditional = conditional + adapter(state, conditional)
    return unconditional + CFG["guidance_scale"] * (conditional - unconditional)


def decode(pipe, latents) -> np.ndarray:
    with torch.no_grad():
        # Keep the VAE off MPS during policy evaluation and optimization.
        raw = pipe.decode_latents(latents.float().cpu())
        frames = pipe.video_processor.postprocess_video(video=raw, output_type="np")[0]
    x = np.stack([np.asarray(frame) for frame in frames]).astype(np.float32)
    if x.shape != (CFG["frames"], CFG["height"], CFG["width"], 3):
        raise ValueError(f"Unexpected decoded shape {x.shape}")
    if not np.isfinite(x).all():
        raise FloatingPointError("Non-finite decoded frames")
    return x


def rollout(pipe, adapter, embeds, seed: int, *, record: bool, adapt: bool = True):
    shape = (1, pipe.unet.config.in_channels, CFG["frames"],
             CFG["height"] // pipe.vae_scale_factor,
             CFG["width"] // pipe.vae_scale_factor)
    state = noise(shape, seed, "mps") * pipe.scheduler.init_noise_sigma
    selected = len(pipe.scheduler.timesteps) // 2
    transition = None
    with torch.no_grad():
        for i, t in enumerate(pipe.scheduler.timesteps):
            epsilon = eps_prediction(pipe, adapter, state, t, embeds, adapt=adapt)
            mean, std = transition_parameters(pipe.scheduler, state, epsilon, int(t), eta=0.5)
            action = mean + std * noise(shape, seed * 1009 + i + 1, "mps")
            if record and i == selected:
                old_logp = action_log_probability(action, mean, std)
                transition = {
                    "state": state.cpu(), "action": action.cpu(),
                    "old_logp": old_logp.cpu(), "reference_mean": mean.cpu(),
                    "timestep": int(t), "std": float(std.cpu()),
                }
            state = action
    frames = decode(pipe, state)
    return frames, transition


def write_video(frames: np.ndarray, name: str) -> dict:
    return save_frames(frames, RUN / name, CFG["frames"])


def train_step(pipe, adapter, embeds, transitions, rewards):
    advantages = group_advantages(np.asarray(rewards, dtype=np.float32))
    if not np.any(advantages):
        raise RuntimeError("All group rewards equal; no valid GRPO update")
    params = list(adapter.parameters())
    if not params or any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("Expected trainable float32 residual parameters")
    optimizer = torch.optim.AdamW(params, lr=CFG["learning_rate"], weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    diagnostics = []
    before = torch.cat([p.detach().flatten().cpu() for p in params])
    for entry, advantage in zip(transitions, advantages):
        state = entry["state"].to("mps")
        action = entry["action"].to("mps")
        timestep = entry["timestep"]
        epsilon = eps_prediction(pipe, adapter, state, timestep, embeds)
        mean, std = transition_parameters(pipe.scheduler, state, epsilon, timestep, eta=0.5)
        logp = action_log_probability(action, mean, std)
        old_logp = entry["old_logp"].to("mps")
        log_ratio = logp - old_logp
        if float(log_ratio.detach().abs().max()) > 0.02:
            raise RuntimeError(f"Initial policy mismatch: log ratio {float(log_ratio.detach())}")
        ratio = log_ratio.exp()
        clipped = ratio.clamp(1 - CFG["clip_ratio"], 1 + CFG["clip_ratio"])
        a = torch.tensor(float(advantage), device="mps")
        objective = torch.minimum(ratio * a, clipped * a).mean()
        ref_mean = entry["reference_mean"].to("mps")
        reference_kl = ((mean - ref_mean).square() / (2 * std.square())).flatten(1).sum(1).mean()
        loss = -objective / len(transitions)
        loss.backward()
        diagnostics.append({"old_logp": float(old_logp), "new_logp": float(logp.detach()),
                            "initial_ratio": float(ratio.detach()),
                            "reference_kl_before_update": float(reference_kl.detach()),
                            "advantage": float(advantage), "loss": float(loss.detach())})
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError(f"Invalid residual gradient norm {grad_norm}")
    optimizer.step()
    after = torch.cat([p.detach().flatten().cpu() for p in params])
    delta = float((after - before).norm())
    if not np.isfinite(delta) or delta <= 0:
        raise FloatingPointError("No finite adapter update")
    return {"advantages": advantages.tolist(), "grad_norm": grad_norm,
            "adapter_delta_norm": delta, "transitions": diagnostics}


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if not (ROOT / "zeroscope_preflight" / "pilot8_fp32" / "frames_raw.npy").exists():
        raise RuntimeError("Finite eight-frame float32 stock visual gate has not produced raw frames")
    if not (ROOT / "zeroscope_preflight" / "stochastic8_residual" / "report.json").exists():
        raise RuntimeError("Stochastic zero-adapter visual gate has not run")
    gate = json.loads((ROOT / "zeroscope_preflight" / "stochastic8_residual" / "report.json").read_text())
    if gate.get("status") != "passed_visual_gate":
        raise RuntimeError("Stochastic zero-adapter visual gate did not pass")
    if RUN.exists():
        raise FileExistsError(f"Pilot already started: {RUN}")
    RUN.mkdir(parents=True)
    torch.mps.set_per_process_memory_fraction(CFG["mps_memory_fraction"])
    started = time.monotonic()
    report = {"protocol": CFG, "scope": "one real-checkpoint frozen-denoiser residual GRPO update",
              "status": "started", "train": [], "evaluation": []}
    try:
        pipe = TextToVideoSDPipeline.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                     local_files_only=True)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.to("mps")
        pipe.vae.to(dtype=torch.float32)
        pipe.unet.requires_grad_(False)
        pipe.unet.eval()
        adapter = OutputResidual().to("mps")
        embeds = pipe.encode_prompt(CFG["prompt"], "mps", 1, True, None)
        pipe.text_encoder.to("cpu")
        pipe.vae.to("cpu")
        torch.mps.empty_cache()
        pipe.scheduler.set_timesteps(CFG["steps"], device="mps")
        transitions, rewards = [], []
        for index, seed in enumerate(CFG["train_seeds"]):
            frames, trans = rollout(pipe, adapter, embeds, seed, record=True)
            score = red_motion_score(frames)
            audit = write_video(frames, f"train_{index}_seed_{seed}")
            report["train"].append({"seed": seed, **score, **audit})
            transitions.append(trans)
            rewards.append(score["reward"])
            torch.mps.empty_cache()
            (RUN / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"event": "train_rollout", "index": index, "seed": seed,
                              "reward": score["reward"]}), flush=True)
        report["update"] = train_step(pipe, adapter, embeds, transitions, rewards)
        report["frozen_backbone_grad_count"] = sum(
            p.grad is not None for p in pipe.unet.parameters())
        if report["frozen_backbone_grad_count"]:
            raise RuntimeError("Frozen U-Net received gradients")
        torch.save(adapter.cpu().state_dict(), RUN / "adapter.pt")
        adapter.to("mps")
        report["status"] = "update_complete; paired evaluation pending"
        (RUN / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        for seed in CFG["eval_seeds"]:
            trained_frames, _ = rollout(pipe, adapter, embeds, seed, record=False)
            trained = red_motion_score(trained_frames)
            write_video(trained_frames, f"trained_eval_seed_{seed}")
            torch.mps.empty_cache()
            report["evaluation"].append({"seed": seed, "policy": "trained", **trained})
            print(json.dumps({"event": "trained_eval", "seed": seed, **trained}), flush=True)
        report["status"] = "trained evaluation complete; zero-adapter paired control pending"
        (RUN / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        for seed in CFG["eval_seeds"]:
            zero_frames, _ = rollout(pipe, adapter, embeds, seed, record=False, adapt=False)
            zero = red_motion_score(zero_frames)
            write_video(zero_frames, f"zero_eval_seed_{seed}")
            torch.mps.empty_cache()
            report["evaluation"].append({"seed": seed, "policy": "zero", **zero})
            print(json.dumps({"event": "zero_eval", "seed": seed, **zero}), flush=True)
        report["status"] = "complete"
        (RUN / "DONE.json").write_text(json.dumps({"status": "complete"}) + "\n")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        (RUN / "FAILED.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        (RUN / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "final", "status": report["status"],
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
