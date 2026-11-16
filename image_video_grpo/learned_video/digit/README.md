# Digit-conditioned video GRPO

The fix for the class collapse found in the [first learned-video study](../RESULTS.md). The model
is conditioned on a **(direction, digit)** prompt, and the GRPO reward pays for the *requested*
digit, so collapsing onto one easy class lowers the reward.

Write-up: [RL for diffusion models](../../../docs/index.html), Part 2, video section.

## Result (320 identical-noise test clips)

| Method | Sampling cost | Success | Δ vs pretrained (95% CI) | Worst digit |
|---|---|---|---|---|
| Pretrained | 1× | 49.1% | n/a | 34% |
| GRPO seed 3101 | 1× | 75.9% | +26.9 (+21.6, +32.2) | 53% |
| GRPO seed 3102 | 1× | 84.1% | +35.0 (+29.7, +40.6) | 69% |
| GRPO seed 3103 | 1× | 84.4% | +35.3 (+29.7, +40.9) | 72% |
| **GRPO pooled** | 1× | **81.5%** | **+32.4 (+27.7, +37.3)** | n/a |
| Best-of-4 | 4× | 82.8% | +33.8 (+28.1, +39.4) | 56% |
| CFG w = 2 | 2× | 85.6% | +36.6 (+30.9, +42.2) | 75% |

Success means the independent evaluator CNN (trained on a disjoint MNIST half) reads a legible digit,
it is the requested digit, and phase correlation says it moves within 22.5° of the requested direction.
Full numbers are in [`runs/analysis.json`](runs/analysis.json), and figures are in [`runs/`](runs).

## Protocol

[`PROTOCOL.json`](PROTOCOL.json) was written before training. Amendments 1–3 record every change
and the reason for it:

1. A warm-started model ignored the digit (5% correct), archived in `runs_attempt1_gate_failed/`.
2. A higher digit-embedding learning rate still failed (6%), archived in `runs_attempt2_warmstart_failed/`.
   Retrained from scratch.
3. At 8,000 steps the gate scored 36% against the 40% bar, archived in `runs_attempt3_gate_8000/`.
   Training continued to 12,000 steps; the gate was unchanged and then passed at 47.5%.

GRPO hyperparameters are identical to the first study. Final weights are evaluated, with no early stopping.

## Reproduce

From the repository root, with the `image_video_grpo/.venv` environment. MNIST is expected under `data/MNIST/raw`.

```bash
PY=image_video_grpo/.venv/bin/python
$PY -m image_video_grpo.learned_video.digit.pretrain        # ~3 h on an M3
bash image_video_grpo/learned_video/digit/run_queue.sh      # gate → controls → 3 GRPO seeds → analysis (~3.5 h)
$PY -m image_video_grpo.learned_video.digit.blog_figure     # compact before/after figure
$PY -m pytest image_video_grpo/learned_video/digit/test_digit.py
```

The gate and GRPO use the proxy and evaluator CNNs trained by `python -m image_video_grpo.learned_video.classifiers`.
Weights (`*.pt`) are not committed.
