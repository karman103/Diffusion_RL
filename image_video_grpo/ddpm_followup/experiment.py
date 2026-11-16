"""Run one predeclared stock-DDPM image GRPO seed, or its paired evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler

from image_video_grpo.core import ImageResidual, clipped_objective, group_advantages, normal_logprob
from image_video_grpo.experiment import ROOT as PARENT, _save_visual, classifier_pair, rollout
from image_video_grpo.models import ImagePolicy, load_image
from image_video_grpo.rewards import image_scores

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


def build(device: str):
    backbone = load_image(device)
    adapter = ImageResidual().to(device)
    policy = ImagePolicy(backbone, adapter, steps=CFG["steps"])
    scheduler = DDPMScheduler.from_config(json.loads((PARENT / "checkpoints/cifar_scheduler.json").read_text()))
    scheduler.set_timesteps(CFG["steps"])
    policy.times = scheduler.timesteps.tolist()
    policy.sampler = "ddpm"
    return backbone, adapter, policy


def gate() -> None:
    check = json.loads((PARENT / "checkpoints/classifier_results.json").read_text())
    if check["proxy"]["test_accuracy"] < 0.40 or check["gold"]["test_accuracy"] < 0.60:
        raise RuntimeError("Predeclared classifier quality gate failed")
    digest = hashlib.sha256((PARENT / "checkpoints/cifar10_ddpm.safetensors").read_bytes()).hexdigest()
    if digest != CFG["checkpoint_sha256"]:
        raise RuntimeError("Pretrained checkpoint hash mismatch")


def select_steps(seed: int, update: int) -> list[int]:
    rng = np.random.default_rng(seed * 100000 + update)
    return [int(rng.integers(lo, hi)) for lo, hi in ((0, 16), (16, 32), (32, 49))]


def train(seed: int, device: str) -> None:
    run = ROOT / "runs" / f"seed_{seed}"
    if run.exists():
        raise FileExistsError(f"Training run already exists: {run}")
    gate()
    run.mkdir(parents=True)
    try:
        torch.manual_seed(seed)
        backbone, adapter, policy = build(device)
        proxy, gold = classifier_pair("image", device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=CFG["learning_rate"], weight_decay=0)
        initial = torch.cat([p.detach().flatten().cpu() for p in adapter.parameters()])
        saturation_streak = 0
        for update in range(CFG["updates"]):
            output, traj = rollout(policy, seed * 100000 + update, n=CFG["group_size"], record=True)
            if len(traj) != 49:
                raise AssertionError(f"Expected 49 stochastic DDPM actions, got {len(traj)}")
            reward = image_scores(output, proxy, gold, CFG["target_class"])["reward"].to(device)
            advantage = group_advantages(reward)
            if not torch.isfinite(reward).all() or not torch.isfinite(advantage).all():
                raise FloatingPointError("Nonfinite reward or advantage")
            chosen = select_steps(seed, update)
            diagnostics = []
            for epoch in range(CFG["inner_epochs"]):
                opt.zero_grad(set_to_none=True)
                for step in chosen:
                    tr = traj[step]
                    state, action = tr.state.to(device), tr.action.to(device)
                    new_mean, std = policy.transition(state, step)
                    ref_mean, _ = policy.transition(state, step, adapt=False)
                    new_lp = normal_logprob(action, new_mean, std)
                    old_lp = tr.old_logprob.to(device)
                    loss, info = clipped_objective(new_lp, old_lp, advantage, CFG["clip_epsilon"])
                    kl = ((new_mean - ref_mean).square() / (2 * std.square())).flatten(1).sum(1).mean()
                    total = (loss + CFG["kl_weight"] * kl) / len(chosen)
                    if not torch.isfinite(total):
                        raise FloatingPointError("Nonfinite objective")
                    total.backward()
                    diagnostics.append({"epoch": epoch + 1, "step": step, **info,
                                        "kl": float(kl.detach()), "loss": float(total.detach())})
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), CFG["grad_norm_clip"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite adapter gradient")
                opt.step()
                if any(not torch.isfinite(p).all() for p in adapter.parameters()):
                    raise FloatingPointError("Nonfinite adapter parameter")
            clip_fraction = float(np.mean([d["clip_fraction"] for d in diagnostics]))
            saturation_streak = saturation_streak + 1 if clip_fraction > 0.5 else 0
            if saturation_streak >= 5:
                raise FloatingPointError("Action-ratio saturation exceeded 50% for five updates")
            params = torch.cat([p.detach().flatten().cpu() for p in adapter.parameters()])
            row = {"update": update + 1, "steps": chosen, "reward": reward.detach().cpu().tolist(),
                   "advantage": advantage.detach().cpu().tolist(), "diagnostics": diagnostics,
                   "clip_fraction": clip_fraction, "grad_norm": float(grad_norm),
                   "adapter_delta_norm": float((params - initial).norm())}
            with (run / "train.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "diagnostics"}), flush=True)
        torch.save(adapter.cpu().state_dict(), run / "adapter.pt")
        (run / "DONE.json").write_text(json.dumps({"seed": seed, "updates": CFG["updates"],
                                                  "adapter_delta_norm": row["adapter_delta_norm"]}, indent=2))
    except BaseException as exc:
        (run / "FAILED.json").write_text(json.dumps({"seed": seed, "error": repr(exc)}, indent=2))
        raise


def evaluate(seed: int, device: str) -> None:
    run = ROOT / "runs" / f"seed_{seed}"
    if not (run / "DONE.json").exists():
        raise FileNotFoundError("Training completion marker missing")
    if ((run / "eval.jsonl").exists() or (run / "EVAL_DONE.json").exists()
            or any(run.glob("*_samples.pt"))):
        raise FileExistsError("Evaluation exists or is incomplete; refusing duplicate")
    backbone, adapter, policy = build(device)
    proxy, gold = classifier_pair("image", device)
    trained = torch.load(run / "adapter.pt", map_location="cpu", weights_only=True)
    zero = ImageResidual().state_dict()
    rows, visuals = [], {}
    for name in ("zero", "trained", "best_of_four"):
        adapter.load_state_dict(trained if name == "trained" else zero)
        # These controls use the same frozen checkpoint, adapter, sampler, and
        # evaluation seeds for every training run. Reuse their *saved raw*
        # samples instead of spending GPU time on identical inference.
        if seed != CFG["training_seeds"][0] and name != "trained":
            first = ROOT / "runs" / f"seed_{CFG['training_seeds'][0]}"
            if not (first / "EVAL_DONE.json").exists():
                raise RuntimeError("Shared control evaluation is incomplete")
            visuals[name] = torch.load(first / f"{name}_samples.pt", map_location="cpu", weights_only=True)
            source_rows = [json.loads(line) for line in (first / "eval.jsonl").read_text().splitlines()
                           if json.loads(line)["policy"] == name]
            rows.extend([{**row, "training_seed": seed} for row in source_rows])
            torch.save(visuals[name], run / f"{name}_samples.pt")
            _save_visual("image", visuals[name][:8], run / f"{name}.png")
            continue
        samples = []
        for eval_seed in CFG["eval_seeds"]:
            if name == "best_of_four":
                candidates = torch.cat([rollout(policy, eval_seed * 10 + k, n=1)[0] for k in range(4)])
                rewards = image_scores(candidates, proxy, gold, CFG["target_class"])["reward"]
                image = candidates[int(rewards.argmax())][None]
            else:
                image, _ = rollout(policy, eval_seed, n=1)
            score = image_scores(image, proxy, gold, CFG["target_class"])
            row = {"training_seed": seed, "eval_seed": eval_seed, "policy": name,
                   **{key: float(value[0]) for key, value in score.items() if key != "gold_features"}}
            rows.append(row)
            samples.append(image.cpu())
            print(json.dumps(row), flush=True)
        visuals[name] = torch.cat(samples)
        torch.save(visuals[name], run / f"{name}_samples.pt")
        _save_visual("image", visuals[name][:8], run / f"{name}.png")
    (run / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {}
    from torchvision import datasets, transforms
    from rldiff.rewards import frechet_distance
    real = datasets.CIFAR10(str(PARENT / "data"), train=False, download=False,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2*x-1)]))
    indices = [i for i, label in enumerate(real.targets) if label == CFG["target_class"]][:500]
    with torch.no_grad():
        real_features = torch.cat([gold.features(torch.stack([real[j][0] for j in indices[q:q+64]]).to(device)).cpu()
                                   for q in range(0, len(indices), 64)])
        for name, samples in visuals.items():
            subset = [row for row in rows if row["policy"] == name]
            metrics = {key: float(np.mean([row[key] for row in subset]))
                       for key in ("reward", "gold_probability", "gold_accuracy")}
            metrics["pixel_diversity"] = float(torch.pdist(samples.flatten(1).float()).mean())
            features = torch.cat([gold.features(samples[q:q+16].to(device)).cpu()
                                  for q in range(0, len(samples), 16)])
            metrics["classifier_feature_frechet"] = frechet_distance(features, real_features)
            summary[name] = metrics
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    (run / "EVAL_DONE.json").write_text(json.dumps({"rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.seed not in CFG["training_seeds"]:
        raise ValueError("Seed was not predeclared")
    if args.command == "train":
        train(args.seed, args.device)
    else:
        evaluate(args.seed, args.device)


if __name__ == "__main__":
    main()
