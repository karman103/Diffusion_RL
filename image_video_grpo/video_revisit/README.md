# Native toy-video checkpoint recheck

The first video gate used four frames and eight stochastic Euler steps and found no digits. The [upstream training code](https://github.com/kuterd/toy_video_dit/blob/d5c2ae0dee8ccd90fe50d8cffbcb0ef4b4126f8a/train.py) trains on **20 frames**. The [upstream sampling code](https://github.com/kuterd/toy_video_dit/blob/d5c2ae0dee8ccd90fe50d8cffbcb0ef4b4126f8a/sampling.py) uses **100 deterministic steps**, uniform initial noise, and velocity scale 0.1. The model card says 1000 steps for its example GIF; the checked source code says 100. This mismatch merits a source-aligned baseline test before another video RL attempt.

[`PROTOCOL.json`](PROTOCOL.json) fixes four new seeds and a digit, ink, and motion gate before sampling. `preflight.py` uses the pinned checkpoint and source sampler semantics and saves raw 20-frame videos, GIFs, and independent digit scores. `analyze.py` applies the automatic gate. Every GIF must also be inspected. **No video GRPO is claimed by this recheck.** If the released baseline still fails, changing the sampler or optimizing a motion-only reward would not make a sound result.

Run one seed at a time, only after other GPU work is idle:

```sh
image_video_grpo/.venv/bin/python -m image_video_grpo.video_revisit.preflight --seed 630001
```

Repeat for 630002–630004, then run `image_video_grpo/.venv/bin/python -m image_video_grpo.video_revisit.analyze`.

## Completed diagnosis

All four native 20-frame/100-step clips failed the declared digit-content gate and were visually noise-like. Their saved frames remained 0.9983 correlated with the initial uniform noise. The released sampler's descending noise schedule uses `(sigma - sigma_prev) * (data - noise) / 10`, which moves opposite the target denoising direction and attenuates its magnitude. The [source training code](https://github.com/kuterd/toy_video_dit/blob/d5c2ae0dee8ccd90fe50d8cffbcb0ef4b4126f8a/train.py) defines the target as `data - noise`; the [source sampler](https://github.com/kuterd/toy_video_dit/blob/d5c2ae0dee8ccd90fe50d8cffbcb0ef4b4126f8a/sampling.py) defines the update. The checkpoint file is named `moving_mnist_epoch_79.pth`, but its own `epoch` field is **4**, at the pinned SHA-256. That discrepancy prevents us from verifying the model card's 79-epoch claim from this file.

I declared a separate [corrected-Euler diagnostic](CORRECTED_PROTOCOL.json) that flips the update sign and removes the `/10` factor. The first two paired seeds still looked like noise and failed the content gate. Since the requirement was at least three of four clips, it was mathematically impossible to pass after two failures, so remaining diagnostic seeds were not run. [AUDIT.json](AUDIT.json) records raw hashes, noise correlations, checkpoint metadata, and all measured rows. The corrected sampler is a modification, not the released sampler. No video GRPO was run.
