# RL on a real pretrained protein diffusion model

This POC runs **Genie**, the published SCOPe length-128 backbone checkpoint, and adapts 387 translation-head parameters using a DDPO-style PPO objective. It is separate from the existing MNIST project. The subsequent user-requested [RFdiffusion3 GRPO implementation](../rfd3_grpo/README.md) has its own real-checkpoint experiments and results.

- [Blog draft](BLOG.md)
- [Measured results and figures](RESULTS.md)
- [Current status and experiment deadline](STATUS.md)
- [RFdiffusion3 source audit and integration plan](RFD3_INTEGRATION.md)
- [Planned experiment configurations](queue.json)

## Setup

The current workspace already contains the required environment and checkpoint. From the project root:

```bash
.venv/bin/python -m unittest protein_poc.test_core -v
.venv/bin/python -m protein_poc.experiment --out protein_poc/runs/my_run --epochs 24 --batch 8 --eval-n 32 --lr 0.001
.venv/bin/python -m protein_poc.report
.venv/bin/python -m protein_poc.audit_artifacts
```

For a fresh checkout, create a Python 3.12 environment and install `protein_poc/requirements.txt`. Clone `https://github.com/aqlaboratory/genie.git` into `third_party/genie` and check out commit `42e4c996d6e0e1f8c4987b2db84f133f66f8ad4c`. Preserve its Apache-2.0 license. Only its model/config/schedule utilities are imported, so the training framework, logging services and upstream dependency pins are not needed here.

Checkpoint: `third_party/genie/weights/scope_l_128/epoch=49999.ckpt`.
SHA-256: `5e6943a27a9fcf34457d75debf2eba37b9639673229bf34ad89ef9ced41cb068`.

`--device auto` selects MPS when available and otherwise CPU. A restricted sandbox can hide MPS even on an Apple GPU machine. All executed experiments use float32; precision reduction has not been validated. The source checkpoint loads with PyTorch's restricted `weights_only=True` loader.

## Experiment choices

- Length 64, 50 respaced stochastic DDIM steps, η=1, original Genie cosine schedule.
- Base model: 4,087,574 parameters; adapted head: 387.
- Adaptation active only at original timesteps 50..600. Terminal deterministic decoder is frozen.
- Exact coordinate-summed log probabilities; PPO clipping 0.2, four reuse passes, KL coefficient 0.01.
- Compactness reward versus a geometry-penalized variant, random-reward control, independent training seeds and best-of-four.
- Cα geometry and diversity are diagnostics. No amino-acid sequence or foldability claim is produced. PDB files use `UNK` residues.

The numerical tests verify parity with direct weight modification and the log-probability/gradient math. The vectorized Frenet-frame code fixes the upstream default cross-product-axis ambiguity at batch size three. Upstream source files are not modified.

## Outputs

Each run has an immutable `config.json`, append-only metric and training logs, evaluation NPZs with all coordinates and per-sample metrics, Cα PDB examples, `head.pt`, and `DONE.json` when the process exits successfully. Existing run directories are never overwritten. A missing DONE file means interrupted, failed or still running, not success.

`report.py` regenerates `RESULTS.md` and PNG/SVG plots. `run_queue.py` runs the JSON manifest sequentially with an exclusive file lock, logs return codes and enforces a deadline. For a new study supply a future UTC deadline explicitly:

```bash
.venv/bin/python -m protein_poc.run_queue --deadline YOUR_FUTURE_UTC_ISO_TIME
```

The saved queue for this task is bounded by 2026-11-13 05:26 America/Toronto. Do not start a duplicate queue while its process is alive.
