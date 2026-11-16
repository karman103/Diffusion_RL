# Stronger-reward image GRPO follow-up

This is an exploratory third image study prompted by two completed null studies. The [protocol](PROTOCOL.json) fixes the data split, training budget, seeds, controls, and stop rules before the new classifiers or GRPO training run.

The image generator and stock-matched DDPM policy are unchanged from [`ddpm_followup`](../ddpm_followup/README.md). Two new CIFAR-10 classifiers use disjoint 25,000-example training subsets. The proxy's frog probability is the reward; the other classifier measures transfer. Both must pass 70% held-out test accuracy before GRPO starts. The independent classifier's features also support a small-sample distribution diagnostic. This avoids calling a weak 46%-accurate proxy a strong reward model.

Each of three predeclared GRPO seeds runs 250 groups of four with normalized relative advantages, a clipped likelihood-ratio surrogate, and a frozen-reference KL. Evaluation uses 32 fresh paired noise seeds. Best-of-four is a four-times-cost sampling control. Stronger proxy reward alone does not count as success; the independent metric and quality checks determine the claim.

From the workspace root, sequentially and only when no unrelated compute job is active:

```sh
image_video_grpo/.venv/bin/python -m image_video_grpo.strong_reward.train_classifiers
image_video_grpo/.venv/bin/python -m image_video_grpo.strong_reward.experiment train --seed 1501
image_video_grpo/.venv/bin/python -m image_video_grpo.strong_reward.experiment eval --seed 1501
```

Repeat the GRPO commands for 1502 and 1503, then run `image_video_grpo/.venv/bin/python -m image_video_grpo.strong_reward.analyze`. The program refuses duplicate training and incomplete evaluations. The zero and best-of-four controls are computed once and their saved raw samples are reused for later adapter seeds; they are shared controls.

The run is complete. See [status](STATUS.md), [results](RESULTS.md), [audited paired effects](EFFECTS.json), and the [employer-facing account](../FOLLOWUP_BLOG_POST.md). The proxy improved; the independent outcome did not establish improvement.
