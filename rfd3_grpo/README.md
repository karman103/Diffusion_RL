# GRPO on RFdiffusion3

This is the user's requested RFD3-specific continuation. The older Genie queue has been stopped. See [STATUS.md](STATUS.md) for real execution progress and [METHOD.md](METHOD.md) for the diffusion-component fine-tuning design.

Read the [research account](BLOG.md) for a narrative explanation of the policy definition, cached denoising-head adaptation and measured outcomes. The [published-method audit](LITERATURE_AUDIT.md) explains which parts come from DDPO, Flow-GRPO, DPOK, EDM and LoRA, and which parts are our RFD3-specific engineering choices.

The real-checkpoint smoke in `runs/smoke_v3` completed one GRPO update and both numerical parity checks. Its eight-point schedule has poor geometry; it tests the implementation. The 64-point calibration passed the declared Cα geometry checks in two samples per sampler. Six longer GRPO comparisons completed, followed by [16 fresh-noise samples per policy](heldout_3000000_n16/RESULTS.md) and frozen-adapter transfer checks at [24](cross_length/length24/RESULTS.md) and [40](cross_length/length40/RESULTS.md) residues. The overall result is no consistent improvement on the declared geometry proxy. See the [measured training results](RESULTS.md) and [blog-style research account](BLOG.md).

The implementation uses a pinned Apple-MPS Foundry fork at commit `7a9c246fbf140a743d14a35248f5039e7723a23a` and the official RFD3 checkpoint from the IPD registry. It installs in `.venv-rfd3`, separately from the earlier experiments.

```bash
.venv-rfd3/bin/python -m unittest rfd3_grpo.test_grpo -v
.venv-rfd3/bin/python -m rfd3_grpo.train --out rfd3_grpo/runs/my_run --steps 20 --length 32 --epochs 5 --group-size 4 --evaluate-stock --check-stock-parity --deadline YOUR_FUTURE_UTC_ISO_TIME
```

Use `--objective random` for a null-reward control. `--active-min` and `--active-max` choose the denoising noise window. Defaults adapt three rank-two final-projection experts, restricted to the final recycle and free Cα atoms. Model weights remain float32 on MPS.

The training sampler adds explicit post-update Gaussian exploration and is compared against its own zero-adapter baseline. It is not presented as an exact-density wrapper around the unmodified RFD3 sampler or as a marginal-preserving SDE conversion.

Each run saves source/checkpoint hashes, configuration, actual-model parity checks, per-sample coordinates and geometry, group rewards/advantages, gradient and clipping diagnostics, the trained adapter and a completion marker. Numerical unit tests can run without a checkpoint; they do not establish pretrained-model performance.

After the six-run queue completes, a fresh-noise evaluation can compare saved adapters on 16 new shared noise seeds. It acquires the queue's exclusive lock and refuses to run while training is active. It does not update adapter weights.

```bash
.venv-rfd3/bin/python -u -m rfd3_grpo.heldout_eval --count 16 --first-seed 3000000 --deadline YOUR_FUTURE_UTC_ISO_TIME
.venv-rfd3/bin/python -u -m rfd3_grpo.heldout_late_zero --deadline YOUR_FUTURE_UTC_ISO_TIME
.venv/bin/python -m rfd3_grpo.heldout_report
```

The late-window policy needs its own zero-adapter reference because it adds exploration noise in a narrower window. The report recomputes the same guarded score from saved coordinates for every policy and gives paired seed differences against the matching reference. Those samples still share one unconditional 32-residue input; they do not test generalization across protein targets.

The additional length-transfer protocol was declared in [PROTOCOL.md](cross_length/PROTOCOL.md) before sampling. It evaluates the saved length-32 adapters at 24 and 40 residues without retraining:

```bash
.venv-rfd3/bin/python -u -m rfd3_grpo.cross_length_eval --length 24 --count 8 --first-seed 4000000 --deadline YOUR_FUTURE_UTC_ISO_TIME
.venv-rfd3/bin/python -u -m rfd3_grpo.cross_length_eval --length 40 --count 8 --first-seed 4000000 --deadline YOUR_FUTURE_UTC_ISO_TIME
.venv/bin/python -m rfd3_grpo.heldout_report --folder rfd3_grpo/cross_length/length24
.venv/bin/python -m rfd3_grpo.heldout_report --folder rfd3_grpo/cross_length/length40
```
