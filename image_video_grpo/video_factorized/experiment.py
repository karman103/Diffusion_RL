"""GRPO-style adapter training and paired evaluation for factorized toy videos."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from image_video_grpo.core import clipped_objective, group_advantages, normal_logprob
from image_video_grpo.experiment import classifier_pair
from image_video_grpo.video_rescue.metrics import reward_from_metrics, video_metrics
from image_video_grpo.video_factorized.model import (CFG, ROOT, AdaptedImage,
                                                       FactorizedPolicy, velocity_for_seed)
from rldiff.models import MLPClassifier

RL = CFG["grpo_if_gate_passes"]


def gate():
    baseline = json.loads((ROOT / "BASELINE_DONE.json").read_text())
    visual = json.loads((ROOT / "VISUAL_REVIEW.json").read_text())
    if not baseline["automatic_gate_passed"] or not visual["grpo_allowed"]:
        raise RuntimeError("Factorized video baseline did not pass both gates")
    if baseline["passed_count"] < 6:
        raise RuntimeError("Baseline pass count below predeclared threshold")


def proxy_only(device: str):
    proxy = MLPClassifier()
    proxy.load_state_dict(torch.load(ROOT.parents[1] / "mnist_rl/runs/clf/proxy.pt",
                                     map_location="cpu", weights_only=True))
    return proxy.to(device).eval().requires_grad_(False)


def build(device: str):
    model = AdaptedImage().to(device).eval()
    model.base.requires_grad_(False)
    return model, FactorizedPolicy(model, device)


def selected_steps(seed: int, update: int):
    rng = np.random.default_rng(seed * 100000 + update)
    return [int(rng.integers(lo, hi)) for lo, hi in ((0, 6), (6, 12), (12, 19))]


def train(seed: int, device: str):
    if seed not in RL["training_seeds"]:
        raise ValueError("Undeclared training seed")
    gate()
    run = ROOT / "runs" / f"seed_{seed}"
    if run.exists():
        raise FileExistsError(f"Training run exists: {run}")
    run.mkdir(parents=True)
    try:
        torch.manual_seed(seed)
        model, policy = build(device)
        proxy = proxy_only("cpu")
        opt = torch.optim.AdamW(model.adapter.parameters(), lr=RL["learning_rate"], weight_decay=0)
        initial = torch.cat([p.detach().flatten().cpu() for p in model.adapter.parameters()])
        saturation = 0
        with (run / "train.jsonl").open("w") as log:
            for update in range(RL["updates"]):
                rollout_seed = seed * 100000 + update
                videos, _, velocities, traj = policy.rollout(rollout_seed, n=RL["group_size"], record=True)
                if len(traj) != 19:
                    raise AssertionError("Expected 19 stochastic actions")
                rows = video_metrics(videos.cpu(), proxy, proxy, velocities.cpu())
                reward = reward_from_metrics(rows).to(device)
                advantage = group_advantages(reward)
                if not torch.isfinite(advantage).all():
                    raise FloatingPointError("Nonfinite group advantages")
                chosen = selected_steps(seed, update)
                diagnostics = []
                for epoch in range(RL["inner_epochs"]):
                    opt.zero_grad(set_to_none=True)
                    for step in chosen:
                        state, action, old_lp, _ = traj[step]
                        state, action = state.to(device), action.to(device)
                        new_mean, std = policy.transition(state, step)
                        with torch.no_grad():
                            ref_mean, _ = policy.transition(state, step, adapt=False)
                        new_lp = normal_logprob(action, new_mean, std)
                        loss, info = clipped_objective(new_lp, old_lp.to(device), advantage,
                                                       RL["clip_epsilon"])
                        kl = ((new_mean - ref_mean).square() / (2 * std.square())).flatten(1).sum(1).mean()
                        total = (loss + RL["reference_kl_weight"] * kl) / len(chosen)
                        if not torch.isfinite(total):
                            raise FloatingPointError("Nonfinite GRPO surrogate")
                        total.backward()
                        diagnostics.append({"epoch": epoch+1, "step": step, **info,
                                            "reference_kl": float(kl.detach())})
                    grad = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), RL["grad_norm_clip"])
                    if not torch.isfinite(grad):
                        raise FloatingPointError("Nonfinite adapter gradient")
                    opt.step()
                    if any(not torch.isfinite(p).all() for p in model.adapter.parameters()):
                        raise FloatingPointError("Nonfinite adapter weights")
                clip_fraction = float(np.mean([d["clip_fraction"] for d in diagnostics]))
                saturation = saturation + 1 if clip_fraction > 0.5 else 0
                if saturation >= 5:
                    raise FloatingPointError("Action-ratio clipping saturation")
                params = torch.cat([p.detach().flatten().cpu() for p in model.adapter.parameters()])
                row = {"update": update+1, "selected_steps": chosen,
                       "rewards": reward.detach().cpu().tolist(),
                       "advantages": advantage.detach().cpu().tolist(),
                       "mean_reward": float(reward.mean()), "clip_fraction": clip_fraction,
                       "grad_norm": float(grad),
                       "adapter_delta_norm": float((params-initial).norm()),
                       "max_abs_log_ratio": max(d["max_abs_log_ratio"] for d in diagnostics),
                       "max_reference_kl": max(d["reference_kl"] for d in diagnostics)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                if (update+1) % 10 == 0 or update == 0:
                    print(json.dumps(row), flush=True)
        torch.save(model.adapter.cpu().state_dict(), run / "adapter.pt")
        (run / "DONE.json").write_text(json.dumps({"seed": seed, "updates": RL["updates"],
                                                  "adapter_delta_norm": row["adapter_delta_norm"]}, indent=2)+"\n")
    except BaseException as exc:
        (run / "FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2)+"\n")
        raise


def save_grid(videos: torch.Tensor, path: Path):
    save_image((videos[:8].clamp(-1, 1).reshape(-1, 1, 28, 28) + 1) / 2,
               path, nrow=4, padding=2)


def evaluate(seed: int, device: str):
    if seed not in RL["training_seeds"]:
        raise ValueError("Undeclared training seed")
    gate()
    run = ROOT / "runs" / f"seed_{seed}"
    if not (run / "DONE.json").exists():
        raise FileNotFoundError("Training completion marker missing")
    if (run / "EVAL_DONE.json").exists() or (run / "eval.jsonl").exists() or any(run.glob("*_videos.pt")):
        raise FileExistsError("Evaluation exists or is incomplete; refusing duplicate")
    model, policy = build(device)
    trained = torch.load(run / "adapter.pt", map_location="cpu", weights_only=True)
    zero = {key: value.detach().clone() for key, value in model.adapter.state_dict().items()}
    proxy, gold = classifier_pair("video", "cpu")
    all_rows, summaries = [], {}
    for name in ("zero", "trained", "best_of_four"):
        model.adapter.load_state_dict(trained if name == "trained" else zero)
        if seed != RL["training_seeds"][0] and name != "trained":
            first = ROOT / "runs" / f"seed_{RL['training_seeds'][0]}"
            if not (first / "EVAL_DONE.json").exists():
                raise RuntimeError("Shared control evaluation incomplete")
            videos = torch.load(first / f"{name}_videos.pt", map_location="cpu", weights_only=True)
            velocities = torch.load(first / f"{name}_velocities.pt", map_location="cpu", weights_only=True)
            rows = [json.loads(line) for line in (first / "eval.jsonl").read_text().splitlines()
                    if json.loads(line)["policy"] == name]
            rows = [{**row, "training_seed": seed} for row in rows]
        else:
            samples, conditions = [], []
            for eval_seed in RL["evaluation_seeds"]:
                velocity = velocity_for_seed(eval_seed, device)
                if name == "best_of_four":
                    candidates = [policy.rollout(eval_seed*10+k, velocity_override=velocity)[0].cpu()
                                  for k in range(4)]
                    batch = torch.cat(candidates)
                    proxy_rows = video_metrics(batch, proxy, proxy, velocity.cpu().expand(4, -1))
                    index = int(reward_from_metrics(proxy_rows).argmax())
                    video = batch[index:index+1]
                else:
                    video = policy.rollout(eval_seed)[0].cpu()
                samples.append(video)
                conditions.append(velocity.cpu())
            videos = torch.cat(samples)
            velocities = torch.cat(conditions)
            metrics = video_metrics(videos, proxy, gold, velocities)
            rewards = reward_from_metrics(video_metrics(videos, proxy, proxy, velocities))
            rows = [{"training_seed": seed, "eval_seed": eval_seed, "policy": name,
                     "reward": float(rewards[i]), **metric}
                    for i, (eval_seed, metric) in enumerate(zip(RL["evaluation_seeds"], metrics))]
        torch.save(videos, run / f"{name}_videos.pt")
        torch.save(velocities, run / f"{name}_velocities.pt")
        save_grid(videos, run / f"{name}_grid.png")
        all_rows.extend(rows)
        summaries[name] = {key: float(np.mean([row[key] for row in rows]))
                           for key in ("reward", "gold_confidence", "gold_class_consistency",
                                       "centroid_motion_px", "centroid_acceleration_px", "ink_mass")}
        summaries[name]["pixel_diversity"] = float(torch.pdist(videos.flatten(1).float()).mean())
        print(json.dumps({"seed": seed, "policy": name, **summaries[name]}), flush=True)
    (run / "eval.jsonl").write_text("".join(json.dumps(row)+"\n" for row in all_rows))
    (run / "summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    (run / "EVAL_DONE.json").write_text(json.dumps({"rows": len(all_rows)}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS required")
    (train if args.command == "train" else evaluate)(args.seed, device)


if __name__ == "__main__":
    main()
