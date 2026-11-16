# Factorized toy-video GRPO study

This is a transparent feasibility fallback after two learned Moving-MNIST video diffusion models and the external checkpoint failed their predeclared content gates. A pretrained 28×28 MNIST image diffusion model samples one canonical digit. A deterministic affine renderer translates that *same* image across four frames at a seeded velocity. The result is a recognizable moving-digit clip, but **the temporal dynamics are procedural, not learned by a video diffusion model**.

The GRPO policy is the canonical digit sampler: 20 DDIM reverse steps, of which 19 are stochastic Gaussian transitions and the terminal step is deterministic. Each four-sample group shares a velocity. The reward is computed on completed clips from a proxy digit classifier, frame-label consistency, motion smoothness, and a hard ink/motion guard. Three randomly selected transitions per trajectory receive group-normalized advantages, a clipped importance-ratio objective and reference penalty. The adapter has a zero-initialized residual on the pretrained image model's noise prediction; the backbone and separate evaluation classifier stay frozen. The evaluator is a different CNN trained on all 60,000 MNIST training images; its data overlap the 1,000-example proxy subset, though its weights and role are separate. The conditional Gaussian action density is exact for this *modified* sampler. The clipped objective is a surrogate, and neither density is a final-video likelihood. This is diffusion-policy GRPO in the sense of [DDPO](https://arxiv.org/abs/2305.13301), not Flow-GRPO or a pretrained learned-video diffusion result. [CHECKPOINTS.json](CHECKPOINTS.json) records model hashes and provenance.

`PROTOCOL.json` fixes the first two training seeds and fresh evaluation seeds. `FOLLOWUP_PROTOCOL.json` fixes a distinct exploratory two-seed higher-step-size run after the first pair was null. The first eight-seed feasibility gate is in `BASELINE_DONE.json`; six passed, and `baseline_grid.png` permits visual inspection. `VISUAL_REVIEW.json` records the qualitative decision. The first study is preserved even when the follow-up fails. `analyze.py` verifies tensor shapes, finiteness, pairing, hashes, and CPU-rescored metrics before writing `AUDIT.json`, `ANALYSIS.json` and `paired_effects.png`. `RESULTS.md` interprets those audited values.

Use the repository's `image_video_grpo/.venv` environment from the workspace root. MPS is required for sampling and training; check that there is no active GPU job and that run directories do not already exist before starting a command. Commands:

```sh
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.test_policy
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.experiment train --seed 11801
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.experiment eval --seed 11801
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.experiment train --seed 11802
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.experiment eval --seed 11802
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.followup train --seed 12101
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.followup eval --seed 12101
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.followup train --seed 12102
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.followup eval --seed 12102
image_video_grpo/.venv/bin/python -m image_video_grpo.video_factorized.analyze
```

The `runs/seed_*` directories include adapters, per-update logs, raw video tensors, raw paired evaluation rows and filmstrip grids. Best-of-four selects one of four independent zero-adapter samples with the proxy reward and costs four times as much sampling as the single-sample comparison. It is a useful practical control, not a compute-matched policy comparison. The independent digit classifier never selects training samples or updates the adapter. These are toy MNIST clips and classifier/geometry metrics, not evidence of general video quality.
