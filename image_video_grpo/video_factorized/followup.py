"""Run a prospectively saved higher-step-size follow-up through the same policy code."""
import argparse
import json

from image_video_grpo.video_factorized import experiment
from image_video_grpo.video_factorized.model import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "FOLLOWUP_PROTOCOL.json").read_text())
    if args.seed not in cfg["training_seeds"]:
        raise ValueError("Undeclared follow-up seed")
    experiment.RL = cfg
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    (experiment.train if args.command == "train" else experiment.evaluate)(args.seed, "mps")


if __name__ == "__main__":
    main()
