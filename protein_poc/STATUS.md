# Protein diffusion RL POC — continuation state

Started 2026-11-12 21:26 America/Toronto. Eight-hour experiment deadline: **2026-11-13 05:26 America/Toronto (10:26Z)**. Never start new experiments after that deadline. Finalize with whatever actually completed, and disable heartbeat `protein-diffusion-research-poc` when done.

## User intent and deliverables
The user adopted an attached request for an exceptional, employer-facing RL-for-protein-diffusion POC with a blog and a 10-minute loop over eight hours. Downloads/local model runs are authorized. No paid compute or publication was requested.

The original root `rldiff/`, `runs/`, `blog/POST.md`, `figures/` are a completed MNIST study. They are preserved. Root README now points to the separate protein extension. Never call the MNIST results protein results.

New deliverables:
- `protein_poc/BLOG.md`: substantial research blog draft, explicitly unfinished pending results.
- `protein_poc/RESULTS.md`: generated measured-results table, bootstrap intervals, figures.
- `protein_poc/RFD3_INTEGRATION.md`: source-audited RFdiffusion3 integration plan. **RFD3 not run.**
- `protein_poc/core.py`, `experiment.py`, `report.py`, `audit_artifacts.py`, `test_core.py`, `run_queue.py`, `queue.json`.
- `protein_poc/README.md`, pinned requirements, source snapshots, NPZ/PDB/head artifacts.

## Current processes — inspect before any new GPU job
- Pilot: exec session 9657, PID **10250**, `python -u -m protein_poc.experiment --out protein_poc/runs/pilot --epochs 10 --batch 8 --eval-n 16 --eval-every 5 --lr 0.001 --reuse 4 --device mps`.
- Queue: detached PID **11157**, recorded in `queue_process.json`. It waits for pilot PID 10250, then runs `queue.json` sequentially. Exclusive lock in `queue.lock`; logs `queue.log`, individual `NAME.log`, `queue_history.jsonl`.
- A task-scoped `caffeinate -i -w 11157` prevents idle sleep only for the lifetime of this queue.
- Heartbeat **protein-diffusion-research-poc** active every 10 minutes, in this chat, bounded by the deadline above. It should not produce unchanged status spam.
- There are unrelated ML jobs in other projects on the same GPU. **Do not stop them.**

Queue loads the manifest after the pilot wait, once. Changes to `queue.json` after that point do not alter an already-running queue. Existing run directories are skipped, never overwritten. A missing `DONE.json` does not imply success. `QUEUE_FINISHED.json` means the runner exited, not that every experiment completed. Inspect history and individual DONE states.

## Actual implementation
Published Genie SCOPe length-128 checkpoint, 4,087,574 model parameters. 64-residue Cα generation, float32, 50-step respaced stochastic DDIM eta=1 using the original schedule. This sampler differs from upstream Genie's 1,000-step sampler; all primary comparisons share it.
387 trainable parameters update only the final local-frame translation projection, enabled at original timesteps 50..600. Cached features exactly reproduce an actual update to the corresponding pretrained weight rows. Terminal decoder fixed; coordinate-summed exact joint log likelihood; PPO clip .2, reuse 4, KL beta .01. No full-network finetuning or LoRA claim.
Main rewards: negative radius of gyration, and compactness plus explicit geometry penalties. Evaluators: bond-distance error, Cα clashes, geometry-pass heuristic, distance-map diversity, path KL. No sequence design/foldability/function/novelty claims. PDB residues are UNK.

## Evidence so far
- Initial four-sample GPU smoke succeeded in 22.6 seconds; all finite, plausible bond spacing, no declared clashes.
- Pilot baseline n=16: Rg **11.21254 Å**, bond MAE **0.01843 Å**, geometry pass **16/16**, diversity **6.40324 Å**.
- Pilot epoch 5 (40 training trajectories): Rg **11.20487 Å**, geometry pass **16/16**, diversity **6.40575 Å**, path KL **2.48981 nats**.
- Paired ΔRg **-0.00767 Å**, evaluation-sample bootstrap CI roughly [-.013,-.002]. Tiny exploratory effect, not a meaningful performance win or proof beyond random drift.
- PPO clipping is active (first update clip fraction .0324; max |log ratio| .6957).
- Latest exact state is in run logs; this prose may lag them.

## Planned queue (21 runs)
Baseline n=64 + independent best-of-4 (32 groups), main compact/random/guarded seed0, exploratory earlier-window compact/random ablation, compact+guarded seeds1/2, 100-step sampler sensitivity baseline, fresh-seed base + held-out evaluations of completed learned heads. Main training 24 epochs x 8 trajectories per run, n=32 validation, eval every12. Earlier-window ablation changes active window to200..999 and LR1e-4. It was added after seeing the tiny pilot change and is exploratory, not pre-registered.
Fresh test noise starts at seed4,000,000; validation at1,000,000. Fresh head evaluation is skipped if the source training is incomplete. A deadline stops a stalled subprocess too. If time is tight, report missing experiments instead of claiming completion.

## Verification completed
Seven numerical tests passed; actual cached-head vs direct weight change, Gaussian log density and KL, finite-difference gradient, terminal decoder, geometry metrics, frames. Upstream `torch.cross` omitted dim and selected batch axis for batch size3; the vectorized local code explicitly uses coordinate axis and was compared against singleton upstream calls. Upstream files unchanged.
End-to-end CPU smoke passed training, checkpoint reload, fresh-seed evaluation and NPZ/PDB artifacts. Details `verification.log`.
Run `.venv/bin/python -m protein_poc.audit_artifacts` to recompute geometry/reward/diversity directly from saved coordinates.
Run `.venv/bin/python -m protein_poc.report` to refresh result tables and plots; visually inspect final plots with view_image.

## Environment/provenance
Python3.12.11 / Torch2.15.0 / NumPy2.5.4 / SciPy1.18.1 / Matplotlib3.11.2. MPS available **outside sandbox**; use normal escalation review for GPU or network. Sandbox reports MPS false. 16GB shared machine, recommended GPU working memory11.45GB, disk around17GB free at start. Jobs sequential.
Genie source `third_party/genie` at `42e4c996d6e0e1f8c4987b2db84f133f66f8ad4c`, 47MB checkpoint SHA256 `5e6943a27a9fcf34457d75debf2eba37b9639673229bf34ad89ef9ced41cb068`.
Foundry source audit pinned `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`; selected source files saved under references.
RFD3 sampler adds noise BEFORE nonlinear denoiser; its transition is not the simple Gaussian used here. Source audit documents why naive DDPO port would be wrong. RFD3 installation/weights were not attempted; Genie was deliberately chosen for local compute.

## Remaining work for follow-ups
1. Confirm pilot finishes and queue advances without duplication. Investigate failures using logs; preserve failed run directories and use new names for reruns.
2. Monitor meaningful experimental changes. Don't optimize against fresh test outcomes. If changing protocol, label it exploratory and keep history.
3. When sufficient runs finish (or deadline): audit artifacts, regenerate report, inspect figures, finalize BLOG.md with measured results, seed variability, compute accounting and limitations. Remove stale 'running' language only when accurate. Include negative results honestly.
4. Keep README and this file in sync. Link final deliverables to user. Disable the heartbeat on completion or deadline. Do not publish externally or contact employers.

## User redirection (21:48 Nov12)
User now explicitly requests **GRPO on RFdiffusion3**. Our waiting Genie queue PID11157 was stopped before it started experiments. Do not restart it. The ten-epoch Genie pilot has finished; results are supplementary. Continue from `rfd3_grpo/STATUS.md`. The existing heartbeat has been updated to the new priority.
