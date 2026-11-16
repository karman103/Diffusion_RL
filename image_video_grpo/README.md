# GRPO on compact image diffusion and toy video flow models

This is a separate, laptop-scale research study. The frozen pretrained backbones
are Google's [CIFAR-10 DDPM](https://huggingface.co/google/ddpm-cifar10-32) and
the [Moving-MNIST toy video flow model](https://huggingface.co/kuterd/moving_mnist_diffusion).
Exact revisions and predeclared seeds are in `PROTOCOL.json`. The video source is
from [upstream commit d5c2ae0](https://github.com/kuterd/toy_video_dit/tree/d5c2ae0dee8ccd90fe50d8cffbcb0ef4b4126f8a),
which has no declared repository license; its local source copy is for reproducing
this experiment, and should not be redistributed in a published code package
without permission. The checkpoint model card lists MIT.

The CIFAR checkpoint is sampled with a 50-step stochastic DDIM Gaussian policy,
with a zero-initialized residual adapter on predicted noise. A 50-step stock
DDPM-scheduler sample is reported only as a separate reference. The video checkpoint is a
flow-matching model trained with uniform-noise interpolation. Its upstream
sampler is deterministic and divides the Euler velocity by ten despite training
on the unscaled data-minus-noise target. This study uses the ordinary Euler
scale, adds Gaussian noise to each nonterminal action, and adapts predicted
velocity. **This is a modified stochastic
sampler, not an equivalent SDE for the released video model.** Each modified
transition has an exact conditional Gaussian density. Neither sampler yields an
exact final-image/video likelihood from the recorded action densities.

The completed image experiment used groups of four, normalized within-group
terminal rewards, clipped conditional action ratios, and a reference KL. The
denoiser was frozen; only a 2,627-parameter residual adapter trained. Three
selected denoising timesteps contributed action losses per update. The video
arm had the same planned GRPO structure, but stopped at its feasibility gate.
Train/eval seeds, reward and stopping conditions are fixed in `PROTOCOL.json`.

The video feasibility gate failed before training: all four fixed four-frame
samples had zero detected digit confidence, and two earned positive reward from
motion alone. `VIDEO_GATE_FAILED.json` records the stop decision and
`video_feasibility.json` holds the measurements. There is **no video GRPO
training result**.

From the workspace root, install the pinned packages in a separate environment:

```bash
python3 -m venv --system-site-packages image_video_grpo/.venv
image_video_grpo/.venv/bin/python -m pip install -r image_video_grpo/requirements.txt
image_video_grpo/.venv/bin/python -m unittest image_video_grpo.test_core
```

The local `checkpoints/` directory contains the pinned downloads and hashes in
`PROTOCOL.json`; the CIFAR-10 archive is under `data/`. For a clean checkout,
download the files from the exact model revisions in the protocol and verify
their SHA-256 values before use. The completed disjoint proxy/gold controls
reached 46.2%/79.88% held-out test accuracy, exceeding their predeclared
40%/60% gates. From the workspace root, with no competing GPU job, run:

```bash
IVG_PY=image_video_grpo/.venv/bin/python
$IVG_PY -m image_video_grpo.train_cifar_classifiers
$IVG_PY -m image_video_grpo.preflight
for seed in 1101 1102 1103; do
  $IVG_PY -m image_video_grpo.experiment train --kind image --seed "$seed"
  $IVG_PY -m image_video_grpo.experiment eval --kind image --seed "$seed"
done
$IVG_PY -m image_video_grpo.audit
$IVG_PY -m image_video_grpo.analyze
```

The guarded `image_video_grpo.run_queue` implements the sequential preflight,
training, evaluation, and analysis steps and refuses duplicates. The commands
above are for a clean reproduction, not for re-running this completed local
study. Video seeds 2101 and 2102 remain
predeclared but must not be run after the failed gate. Never duplicate a run
directory: the script refuses to overwrite one. GPU runs should be sequential
and should not overlap unrelated jobs.

The reward classifier is deliberately weaker than the independent evaluator;
this makes reward hacking observable. CIFAR distribution quality will be a
small-sample Fréchet distance in independent classifier features, **not**
official FID. The video motion/digit heuristics do not establish perceived
quality or cinematic-video transfer. All claims in the final blog must come
from completed fresh-seed evaluations; a memory or checkpoint failure must be
reported as such.

Related work: [DDPO](https://arxiv.org/abs/2305.13301) treats denoising as a
policy; [Flow-GRPO](https://arxiv.org/abs/2505.05470) constructs a stochastic
flow policy. Sana's published 9 GB figure is an A100 **inference** requirement,
whereas CogVideoX-2B's published LoRA training figure is 47 GB; low precision
inference does not imply that GRPO training fits this 16 GB M3 laptop.

The [measured results](RESULTS.md), [paired-effects plot](paired_effects.png),
[sample grid](sample_grid.png), [raw-sample audit](AUDIT.json), and
[research blog](BLOG_POST.md) are complete. Across 96 paired image evaluations,
trained minus zero-adapter reward was +0.00000055 on average, with no change in
independent frog-class accuracy. This is a measured null result, not an image
quality improvement. The four-seed toy-video feasibility failure is reported
without a video GRPO result.
