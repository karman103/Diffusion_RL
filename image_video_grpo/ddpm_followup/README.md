# Stock-DDPM GRPO follow-up

This exploratory follow-up tests a specific diagnosis of the [completed null study](../RESULTS.md): the earlier 50-step stochastic DDIM zero policy scored far below the released DDPM sampler, and its 12 updates at learning rate 1e-7 barely changed the adapter. The original study and its artifacts remain unchanged.

Here, the pinned pretrained CIFAR-10 checkpoint is sampled with its released `DDPMScheduler` configuration at 50 leading-schedule steps. The zero residual adapter matches that stock sampler. Each of 49 nonterminal actions is Gaussian with the scheduler's `fixed_large` variance; the final transition is deterministic. `test_policy.py` compares the policy transition to the official scheduler on four timesteps. Conditional action log densities are exact for this policy, but they are not final-image likelihoods. The clipped group-relative objective is an optimization surrogate.

The protocol in `PROTOCOL.json` was written before this follow-up's training: three training seeds, 60 updates each, four samples per group, two inner epochs, three stratified denoising steps per update, a frozen reference, and fresh paired evaluation seeds. The reward is a frozen 5,000-example CIFAR-10 frog classifier; an independently trained disjoint-data classifier is used to check transfer. Best-of-four sampling is a four-times-cost control. A result is useful only if independent target recognition improves without an obvious diversity or feature-quality collapse.

From the workspace root, after installing the existing `image_video_grpo/.venv` and verifying the pinned checkpoint and classifiers:

```sh
image_video_grpo/.venv/bin/python -m unittest image_video_grpo.test_core image_video_grpo.ddpm_followup.test_policy
image_video_grpo/.venv/bin/python -m image_video_grpo.ddpm_followup.experiment train --seed 1301
image_video_grpo/.venv/bin/python -m image_video_grpo.ddpm_followup.experiment eval --seed 1301
```

Repeat sequentially for seeds 1302 and 1303, checking that no unrelated heavy job is active. The commands refuse an existing run or evaluation. After all three complete:

The zero and best-of-four controls are identical across training seeds by construction. Evaluations for seeds 1302 and 1303 reuse the first seed's saved raw control samples and scores; only their trained adapters generate new samples. The analysis treats these as shared controls, not three independent baseline draws.

```sh
image_video_grpo/.venv/bin/python -m image_video_grpo.ddpm_followup.analyze
```

The prior toy-video feasibility gate remains failed. This follow-up does not silently turn a poor video checkpoint into a positive GRPO result.
