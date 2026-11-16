"""Audit raw ZeroScope pilot clips and make a descriptive paired result."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from text_video_grpo.pilot_reward import red_motion_score


RUN = Path(__file__).resolve().parent / "runs" / "red_ball_residual_grpo_v1"


def audit_clip(folder: Path, expected_shape):
    frames = np.load(folder / "frames_raw.npy", allow_pickle=False)
    if list(frames.shape) != expected_shape or not np.isfinite(frames).all():
        raise ValueError(f"Invalid raw video {folder}: {frames.shape}")
    if frames.min() < -0.001 or frames.max() > 1.001:
        raise ValueError(f"Out-of-range raw video {folder}")
    pixels = np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)
    return frames, hashlib.sha256(pixels.tobytes()).hexdigest()


def main():
    if not (RUN / "DONE.json").exists():
        raise RuntimeError("GRPO pilot completion marker missing")
    report = json.loads((RUN / "report.json").read_text())
    if report["status"] != "complete":
        raise RuntimeError(f"Pilot status is {report['status']}")
    cfg = report["protocol"]
    shape = [cfg["frames"], cfg["height"], cfg["width"], 3]
    audit = []
    for i, row in enumerate(report["train"]):
        path = RUN / f"train_{i}_seed_{row['seed']}"
        frames, digest = audit_clip(path, shape)
        if digest != row["sha256"]:
            raise ValueError(f"Hash mismatch: {path}")
        score = red_motion_score(frames)
        if abs(score["reward"] - row["reward"]) > 1e-7:
            raise ValueError(f"Reward re-score mismatch: {path}")
        audit.append((path.name, digest, score))
    eval_by = {}
    for row in report["evaluation"]:
        name = f"{row['policy']}_eval_seed_{row['seed']}"
        path = RUN / name
        frames, digest = audit_clip(path, shape)
        score = red_motion_score(frames)
        if abs(score["reward"] - row["reward"]) > 1e-7:
            raise ValueError(f"Reward re-score mismatch: {path}")
        audit.append((name, digest, score))
        eval_by.setdefault(row["seed"], {})[row["policy"]] = score
    seeds = cfg["eval_seeds"]
    if sorted(eval_by) != sorted(seeds) or any(set(eval_by[s]) != {"trained", "zero"} for s in seeds):
        raise ValueError("Incomplete paired evaluation")
    metrics = ("reward", "red_visibility", "red_displacement", "center_jerk", "frame_difference")
    deltas = {metric: [eval_by[s]["trained"][metric] - eval_by[s]["zero"][metric]
                       for s in seeds] for metric in metrics}
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), constrained_layout=True)
    for ax, metric in zip(axes, ("reward", "red_displacement", "frame_difference")):
        d = np.asarray(deltas[metric])
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.scatter(range(len(seeds)), d, color="#1f77b4", zorder=2)
        ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
        ax.set_xlabel("paired seed")
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("trained − zero")
    fig.savefig(RUN / "paired_effects.png", dpi=180)
    plt.close(fig)
    lines = [
        "# ZeroScope GRPO pilot: audited results", "",
        f"Status: {report['status']}; {len(report['train'])} training trajectories, "
        f"{len(seeds)} paired fresh evaluation seeds, one optimizer update.", "",
        "This is an exploratory, one-prompt, eight-frame pilot. The output residual is trained; "
        "the pretrained ZeroScope U-Net is frozen. Scores concern red-object centroid motion, "
        "not text adherence or video quality. The model visibly renders multiple red spheres "
        "instead of the requested single ball on a plain floor.", "",
        "| Fresh seed | Zero reward | Trained reward | Trained − zero |", "|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        z, t = eval_by[seed]["zero"]["reward"], eval_by[seed]["trained"]["reward"]
        lines.append(f"| {seed} | {z:+.6f} | {t:+.6f} | {t-z:+.6f} |")
    lines += ["", "| Metric | Mean trained − zero | Per-seed differences |", "|---|---:|---|",]
    for metric in metrics:
        d = deltas[metric]
        lines.append(f"| {metric} | {np.mean(d):+.6f} | " + ", ".join(f"{v:+.6f}" for v in d) + " |")
    lines += ["", "With two fresh seeds, these are descriptive paired differences, not a confidence interval or a generalization claim.", "",
              f"Adapter delta norm: {report['update']['adapter_delta_norm']:.6g}; "
              f"gradient norm before clipping: {report['update']['grad_norm']:.6g}; "
              f"frozen U-Net gradients: {report['frozen_backbone_grad_count']}.", "",
              "All saved raw videos were finite, shape-checked, independently re-scored, and hashed. "
              "The table below records the raw-frame pixel hashes.", "",
              "| Clip | SHA-256 of uint8 pixels |", "|---|---|",]
    for name, digest, _ in audit:
        lines.append(f"| {name} | `{digest}` |")
    lines += ["", "![Paired trained-minus-zero effects](paired_effects.png)", ""]
    (RUN / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"audit_count": len(audit), "paired_deltas": deltas,
                      "mean_reward_delta": float(np.mean(deltas["reward"]))}, indent=2))


if __name__ == "__main__":
    main()
