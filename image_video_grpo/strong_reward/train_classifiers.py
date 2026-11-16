"""Train two disjoint, stronger CIFAR classifiers before GRPO."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from image_video_grpo.rewards import CifarClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device)).argmax(1).cpu()
            correct += int((prediction == y).sum())
            total += len(y)
    return correct / total


def main():
    if (ROOT / "classifier_results.json").exists() or (ROOT / "classifiers_running.json").exists():
        raise FileExistsError("Classifiers already trained or in progress; refusing duplicate")
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Lambda(lambda x: 2*x-1)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2*x-1)])
    training = datasets.CIFAR10(str(DATA), train=True, download=False, transform=train_transform)
    testing = datasets.CIFAR10(str(DATA), train=False, download=False, transform=test_transform)
    permutation = torch.randperm(len(training), generator=torch.Generator().manual_seed(3401)).tolist()
    split = {"proxy": permutation[:25000], "gold": permutation[25000:]}
    test_loader = DataLoader(testing, batch_size=256, num_workers=0)
    (ROOT / "classifiers_running.json").write_text(json.dumps({"device": device, "split_seed": 3401}))
    try:
        results = {}
        for name, indices in split.items():
            torch.manual_seed(3401 if name == "proxy" else 3402)
            model = CifarClassifier(48).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)
            loader = DataLoader(Subset(training, indices), batch_size=128, shuffle=True, num_workers=0)
            for epoch in range(16):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                print(json.dumps({"classifier": name, "epoch": epoch+1,
                                  "last_batch_loss": float(loss.detach())}), flush=True)
            final = accuracy(model, test_loader, device)
            torch.save(model.cpu().state_dict(), ROOT / f"{name}.pt")
            results[name] = {"test_accuracy": final, "train_examples": len(indices), "epochs": 16}
            print(json.dumps({"classifier": name, "final_test_accuracy": final}), flush=True)
        (ROOT / "classifier_results.json").write_text(json.dumps(results, indent=2))
    except BaseException as exc:
        (ROOT / "CLASSIFIERS_FAILED.json").write_text(json.dumps({"error": repr(exc)}))
        raise
    finally:
        (ROOT / "classifiers_running.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
