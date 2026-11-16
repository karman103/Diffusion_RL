"""Train separate proxy/gold classifiers on disjoint CIFAR-10 training subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from image_video_grpo.rewards import CifarClassifier

ROOT = Path(__file__).resolve().parent


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2401)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])
    train = datasets.CIFAR10(str(ROOT / "data"), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(str(ROOT / "data"), train=False, download=True, transform=transform)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(2401)).tolist()
    results = {}
    for name, indices, width, epochs in [("proxy", perm[:5000], 24, 8), ("gold", perm[5000:], 48, 12)]:
        model = CifarClassifier(width).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(Subset(train, indices), batch_size=128, shuffle=True, num_workers=0)
        for epoch in range(epochs):
            model.train()
            for x, y in loader:
                x, y = x.to(args.device), y.to(args.device)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            print(json.dumps({"classifier": name, "epoch": epoch + 1, "loss": float(loss.detach())}), flush=True)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in DataLoader(test, batch_size=256):
                pred = model(x.to(args.device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        accuracy = correct / total
        torch.save(model.cpu().state_dict(), ROOT / "checkpoints" / f"cifar_{name}.pt")
        results[name] = {"test_accuracy": accuracy, "train_examples": len(indices), "epochs": epochs}
        print(json.dumps({"classifier": name, "test_accuracy": accuracy}), flush=True)
    (ROOT / "checkpoints/classifier_results.json").write_text(json.dumps(results, indent=2))
    if results["proxy"]["test_accuracy"] < 0.40 or results["gold"]["test_accuracy"] < 0.60:
        raise RuntimeError("Classifier quality gate failed; do not run image RL")


if __name__ == "__main__":
    run()
