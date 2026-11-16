"""Train the reward/eval models.

gold  : strong CNN on all 60k labels + augmentation  -> plays the role of the *true* objective
proxy : small MLP on only 1,000 labels               -> plays the role of a *learned reward model*
feat  : independent CNN (different seed/width)       -> feature extractor for FD / diversity metrics
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F, torchvision
import torchvision.transforms.v2 as T
from rldiff.models import CNN, MLPClassifier

dev = "cpu"
torch.set_num_threads(4)
tr = torchvision.datasets.MNIST("data", train=True)
te = torchvision.datasets.MNIST("data", train=False)
Xtr = (tr.data.float() / 127.5 - 1).unsqueeze(1); Ytr = tr.targets
Xte = (te.data.float() / 127.5 - 1).unsqueeze(1); Yte = te.targets
aug = T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))


def train(model, X, Y, epochs, bs=128, lr=1e-3, augment=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x = X[idx]
            if augment:
                x = aug((x + 1) / 2) * 2 - 1
            loss = F.cross_entropy(model(x), Y[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = sum((model(Xte[i:i + 1000]).argmax(1) == Yte[i:i + 1000]).sum().item() for i in range(0, 10000, 1000)) / 10000
        print(f"  epoch {ep} test acc {acc:.4f}", flush=True)
    return model.eval(), acc


os.makedirs("runs/clf", exist_ok=True)
torch.manual_seed(1)
print("gold");  gold, ga = train(CNN(32), Xtr, Ytr, 4, augment=True, seed=1)
torch.save(gold.state_dict(), "runs/clf/gold.pt")
torch.manual_seed(2)
print("feat");  feat, fa = train(CNN(24), Xtr, Ytr, 3, augment=True, seed=2)
torch.save(feat.state_dict(), "runs/clf/feat.pt")
torch.manual_seed(3)
sub = torch.cat([torch.nonzero(Ytr == c)[:100, 0] for c in range(10)])
print("proxy"); proxy, pa = train(MLPClassifier(), Xtr[sub], Ytr[sub], 60, bs=64, seed=3)
torch.save(proxy.state_dict(), "runs/clf/proxy.pt")
with open("runs/clf/acc.txt", "w") as f:
    f.write(f"gold {ga:.4f}\nfeat {fa:.4f}\nproxy {pa:.4f}\n")
print("done")
