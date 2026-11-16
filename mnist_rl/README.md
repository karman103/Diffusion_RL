# When the reward model lies: RL fine-tuning on an MNIST DDPM

A small, fully reproducible study of **reinforcement-learning fine-tuning for diffusion models**.
It treats the denoising chain as an MDP and compares the main families of reward fine-tuning:
DDPO, reward-weighted regression, direct reward backprop (DRaFT) and best-of-N. It measures
**reward hacking** directly by optimising a weak *proxy* reward model while scoring with an
independent *gold* model.

Everything runs on a laptop (Apple M3, MPS backend). The base model trains in about an hour, and
each RL run takes 5–15 minutes.

Write-up: [published page](../docs/reward-hacking/index.html) · [markdown version](POST.md) · figures in [`figures/`](figures)

## Setup

| Component | What it is |
|---|---|
| Base model | 1.5M-param UNet, unconditional DDPM on MNIST (cosine schedule, eps-pred, EMA), 10k steps |
| Sampler / MDP | 20-step DDIM with η = 1. Each step is a Gaussian policy; the stochastic chain ends at t = 20 and the final step outputs x̂₀ |
| Proxy reward | MLP classifier trained on **1,000** labels (89.9% test acc.). r = p(7 \| x) |
| Gold evaluator | CNN on all 60k labels + augmentation (99.2%). Never seen by the optimiser |
| Feature net | Independent CNN (99.0%) used for Fréchet distance (FD) and diversity |
| Black-box reward | JPEG file size (DDPO's (in)compressibility task), which has no gradient |

## Layout

The shared library [`rldiff/`](../rldiff) lives at the repository root, because the later studies import it too.

```
../rldiff/diffusion.py        schedule, DDIM-as-MDP sampler with exact per-step log-probs, DRaFT sampler
../rldiff/models.py           UNet, CNN/MLP classifiers
../rldiff/rewards.py          rewards (proxy / gold / JPEG / random control), FD + diversity metrics
scripts/train_base.py         pretrain the DDPM
scripts/train_classifiers.py  gold / feature / proxy models
scripts/rl_finetune.py        DDPO (PPO-clip), DDPO-SF, RWR, DRaFT-K, optional DPOK-style KL
scripts/best_of_n.py          inference-time baseline and its analytic KL
scripts/queue.sh              sequential experiment runner (runs/queue.txt)
scripts/make_figures.py       all plots
runs/                         per-run metrics.jsonl, sample grids, summary.md (weights are not committed)
```

## Reproduce

Run from this folder. MNIST downloads into `data/` on first use.

```bash
uv venv ../.venv -p 3.12 && source ../.venv/bin/activate
uv pip install torch torchvision numpy scipy matplotlib pillow tqdm
python scripts/train_base.py --steps 10000
python scripts/train_classifiers.py
python scripts/best_of_n.py
python scripts/rl_finetune.py --method ddpo --reward proxyp:7 --epochs 150 --out runs/p_ddpo
python scripts/make_figures.py
```

Runs are deterministic for a given seed on MPS. Two runs with identical settings matched
bit-for-bit for 100 epochs.

The later image/video study loads this study's classifiers from `runs/clf/` and base model from
`runs/base/`, so train those first if you want to reproduce it.
