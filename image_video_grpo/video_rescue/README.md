# Laptop toy-video rescue

The external Moving-MNIST checkpoint did not produce digit-bearing clips and its internal epoch metadata contradicted its filename. This separate experiment builds a new toy-video baseline from the existing 10,000-step MNIST image DDPM checkpoint in `runs/base/ckpt.pt`. The image network is frozen. An 8,241-parameter 3D convolutional temporal residual is trained by noise-prediction MSE on four-frame sequences made by translating a single MNIST training digit. The result is **locally pretrained video diffusion**, not an externally released video checkpoint and not a cinematic-video claim.

The [protocol](PROTOCOL.json) fixes supervised training, sample seeds, and a digit/motion gate before any GRPO. If the locally pretrained baseline does not pass, the video RL stage remains stopped. If it passes, the stochastic 20-step DDIM sampler has conditional Gaussian action densities suitable for a selected-step GRPO surrogate. The released image prior and the temporal pretraining checkpoint remain frozen reference points for a paired comparison.

Run sequentially on the laptop, after checking active jobs:

```sh
image_video_grpo/.venv/bin/python -m image_video_grpo.video_rescue.train_temporal
image_video_grpo/.venv/bin/python -m image_video_grpo.video_rescue.preflight
```

The scripts refuse duplicate completed runs. `baseline_grid.png` and `BASELINE_DONE.json` are the feasibility evidence. The new work is isolated from the earlier image and protein artifacts.
