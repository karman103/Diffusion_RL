"""One-seed GRPO train/evaluate entrypoint. Never silently resumes a run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_video_grpo.core import (ImageResidual, Transition, VideoResidual,
                                   clipped_objective, group_advantages, normal_logprob)
from image_video_grpo.models import ImagePolicy, VideoPolicy, load_image, load_video
from image_video_grpo.rewards import CifarClassifier, image_scores, video_scores
from rldiff.models import CNN, MLPClassifier

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())


def classifier_pair(kind: str, device: str):
    if kind == "image":
        proxy, gold = CifarClassifier(24), CifarClassifier(48)
        proxy.load_state_dict(torch.load(ROOT / "checkpoints/cifar_proxy.pt", map_location="cpu", weights_only=True))
        gold.load_state_dict(torch.load(ROOT / "checkpoints/cifar_gold.pt", map_location="cpu", weights_only=True))
    else:
        proxy, gold = MLPClassifier(), CNN(32)
        proxy.load_state_dict(torch.load(ROOT.parent / "mnist_rl/runs/clf/proxy.pt", map_location="cpu", weights_only=True))
        gold.load_state_dict(torch.load(ROOT.parent / "mnist_rl/runs/clf/gold.pt", map_location="cpu", weights_only=True))
    return [m.to(device).eval().requires_grad_(False) for m in (proxy, gold)]


def build(kind: str, device: str):
    if kind == "image":
        backbone = load_image(device)
        adapter = ImageResidual().to(device)
        policy = ImagePolicy(backbone, adapter, steps=PROTOCOL["image"]["steps"])
    else:
        backbone = load_video(device)
        adapter = VideoResidual().to(device)
        p = PROTOCOL["video"]
        policy = VideoPolicy(backbone, adapter, steps=p["steps"], frames=p["frames"], noise=p["noise"])
    return backbone, adapter, policy


def _random(shape, seed: int, uniform: bool, device: str) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cpu = torch.rand(shape, generator=gen) if uniform else torch.randn(shape, generator=gen)
    return cpu.to(device)


@torch.no_grad()
def rollout(policy, seed: int, n: int = 4, record: bool = False):
    device = str(next(policy.backbone.parameters()).device)
    video = isinstance(policy, VideoPolicy)
    x = _random((n, *policy.shape), seed, uniform=video, device=device)
    traj: list[Transition] = []
    for step in range(policy.steps):
        mean, std = policy.transition(x, step)
        if std is None:
            x = mean
        else:
            noise = _random(tuple(x.shape), seed * 1009 + step + 1, uniform=False, device=device)
            action = mean + std * noise
            if record:
                traj.append(Transition(x.cpu(), action.cpu(), normal_logprob(action, mean, std).cpu(), step))
            x = action
    if not torch.isfinite(x).all():
        raise FloatingPointError("Generated sample contains non-finite values")
    return x.detach(), traj


def scores(kind: str, output: torch.Tensor, proxy, gold):
    if kind == "image":
        return image_scores(output, proxy, gold, PROTOCOL["image"]["target_class"])
    return video_scores(output, proxy, gold)


def train(kind: str, seed: int, device: str):
    if kind == "video" and (ROOT / "VIDEO_GATE_FAILED.json").exists():
        raise RuntimeError("Video feasibility gate failed; do not train")
    cfg = PROTOCOL[kind]
    run = ROOT / "runs" / kind / f"seed_{seed}"
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}; refusing duplicate training")
    run.mkdir(parents=True)
    torch.manual_seed(seed)
    backbone, adapter, policy = build(kind, device)
    proxy, gold = classifier_pair(kind, device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    records = []
    baseline_params = torch.cat([p.detach().flatten().cpu() for p in adapter.parameters()])
    for update in range(cfg["updates"]):
        output, traj = rollout(policy, seed * 100000 + update, n=4, record=True)
        reward = scores(kind, output, proxy, gold)["reward"].to(device)
        adv = group_advantages(reward)
        chosen = list(range(0, len(traj), max(1, len(traj) // 3)))[:3]
        diag = []
        for inner in range(cfg["inner_epochs"]):
            opt.zero_grad(set_to_none=True)
            for j in chosen:
                trans = traj[j]
                state, action = trans.state.to(device), trans.action.to(device)
                new_mean, std = policy.transition(state, trans.step)
                ref_mean, _ = policy.transition(state, trans.step, adapt=False)
                new_lp = normal_logprob(action, new_mean, std)
                old_lp = trans.old_logprob.to(device)
                loss, info = clipped_objective(new_lp, old_lp, adv)
                exact_kl = ((new_mean - ref_mean).square() / (2 * std * std)).flatten(1).sum(1).mean()
                total = (loss + cfg["kl_weight"] * exact_kl) / len(chosen)
                total.backward()
                diag.append({**info, "inner_epoch": inner + 1,
                             "kl": float(exact_kl.detach()), "loss": float(total.detach())})
            grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
            opt.step()
        params = torch.cat([p.detach().flatten().cpu() for p in adapter.parameters()])
        rec = {"update": update + 1, "rewards": reward.detach().cpu().tolist(),
               "advantages": adv.detach().cpu().tolist(), "diagnostics": diag,
               "grad_norm": grad_norm, "adapter_delta_norm": float((params - baseline_params).norm())}
        records.append(rec)
        print(json.dumps(rec), flush=True)
        (run / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    torch.save(adapter.cpu().state_dict(), run / "adapter.pt")
    (run / "DONE.json").write_text(json.dumps({"kind": kind, "seed": seed, "updates": len(records),
                                              "adapter_delta_norm": records[-1]["adapter_delta_norm"]}, indent=2))
    return run


def _save_visual(kind: str, samples: torch.Tensor, path: Path):
    x = samples.detach().float().cpu()
    if kind == "image":
        x = (x.clamp(-1, 1) + 1) / 2
        from torchvision.utils import save_image
        save_image(x, path, nrow=4, padding=2)
    else:
        x = x.clamp(0, 1)
        frames = []
        for t in range(x.shape[1]):
            row = torch.cat([x[i, t, 0] for i in range(min(4, len(x)))], dim=1)
            frames.append(Image.fromarray((row.numpy() * 255).astype(np.uint8), mode="L"))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0)


def evaluate(kind: str, seed: int, device: str):
    if kind == "video" and (ROOT / "VIDEO_GATE_FAILED.json").exists():
        raise RuntimeError("Video feasibility gate failed; no trained policy to evaluate")
    cfg = PROTOCOL[kind]
    run = ROOT / "runs" / kind / f"seed_{seed}"
    if not (run / "DONE.json").exists():
        raise FileNotFoundError("Training completion marker missing")
    if (run / "eval.jsonl").exists():
        raise FileExistsError("Evaluation already exists; refusing duplicate")
    backbone, adapter, policy = build(kind, device)
    proxy, gold = classifier_pair(kind, device)
    trained_state = torch.load(run / "adapter.pt", map_location="cpu", weights_only=True)
    rows, visuals = [], {}
    for policy_name in ("stock", "zero", "trained", "best_of_four"):
        if policy_name == "trained":
            adapter.load_state_dict(trained_state)
        else:
            adapter.load_state_dict(type(adapter)().state_dict())
        chunks = []
        if policy_name == "stock" and kind == "image":
            from diffusers import DDPMScheduler
            config = json.loads((ROOT / "checkpoints/cifar_scheduler.json").read_text())
            official = DDPMScheduler.from_config(config)
            official.set_timesteps(cfg["steps"])
            policy.times = official.timesteps.tolist()
            policy.sampler = "ddpm"
        elif kind == "image":
            policy.times = torch.linspace(999, 0, cfg["steps"]).round().long().tolist()
            policy.sampler = "ddim"
        if policy_name == "stock" and kind == "video":
            policy.noise = 0.0
            policy.velocity_scale = 0.1
        elif kind == "video":
            policy.noise = cfg["noise"]
            policy.velocity_scale = 1.0
        for sample_seed in cfg["eval_seeds"]:
            if policy_name == "best_of_four":
                candidates = []
                for k in range(4):
                    x, _ = rollout(policy, sample_seed * 10 + k, n=1)
                    candidates.append(x)
                candidate = torch.cat(candidates, 0)
                rs = scores(kind, candidate, proxy, gold)["reward"]
                x = candidate[int(rs.argmax())][None]
            else:
                x, _ = rollout(policy, sample_seed, n=1)
            sc = scores(kind, x, proxy, gold)
            row = {"training_seed": seed, "eval_seed": sample_seed, "policy": policy_name,
                   **{k: float(v[0]) for k, v in sc.items() if k != "gold_features"}}
            rows.append(row)
            chunks.append(x.cpu())
            print(json.dumps(row), flush=True)
        visuals[policy_name] = torch.cat(chunks, 0)
        torch.save(visuals[policy_name], run / f"{policy_name}_samples.pt")
    (run / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name, samples in visuals.items():
        _save_visual(kind, samples[:8], run / f"{name}.{'png' if kind == 'image' else 'gif'}")
    summaries = {}
    for name, samples in visuals.items():
        subset = [r for r in rows if r["policy"] == name]
        summary = {k: float(np.mean([r[k] for r in subset])) for k in subset[0]
                   if k not in ("training_seed", "eval_seed", "policy")}
        flat = samples.flatten(1).float()
        summary["pixel_diversity"] = float(torch.pdist(flat).mean())
        if kind == "image":
            from torchvision import datasets, transforms
            from rldiff.rewards import frechet_distance
            real = datasets.CIFAR10(str(ROOT / "data"), train=False, download=False,
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)]))
            target = cfg["target_class"]
            idx = [i for i, y in enumerate(real.targets) if y == target][:500]
            with torch.no_grad():
                real_feat = torch.cat([gold.features(torch.stack([real[j][0] for j in idx[q:q+64]]).to(device)).cpu()
                                       for q in range(0, len(idx), 64)])
                sample_feat = torch.cat([gold.features(samples[q:q+16].to(device)).cpu()
                                         for q in range(0, len(samples), 16)])
            summary["classifier_feature_frechet"] = frechet_distance(sample_feat, real_feat)
        summaries[name] = summary
    (run / "summary.json").write_text(json.dumps(summaries, indent=2))
    (run / "EVAL_DONE.json").write_text(json.dumps({"rows": len(rows)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "eval"])
    ap.add_argument("--kind", choices=["image", "video"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.seed not in PROTOCOL[args.kind]["training_seeds"]:
        raise ValueError("Seed not predeclared in PROTOCOL.json")
    if args.command == "train":
        train(args.kind, args.seed, args.device)
    else:
        evaluate(args.kind, args.seed, args.device)


if __name__ == "__main__":
    main()
