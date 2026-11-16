# Final status: PDB-conditioned Flow-GRPO-style RFD3 benchmark

Updated 2026-11-14 16:50 UTC. User explicitly requested a new Flow-GRPO benchmark on PDBs after the earlier unconditional RFD3 POC. The earlier study remains in `../rfd3_grpo/` and must not be rerun or overwritten.

- Pinned real RFD3 checkpoint and Foundry MPS fork are reused. New sampler `ve_sde` implements a VE reverse-SDE family using RFD3's EDM denoiser as an approximate score. Exact conditional Gaussian action ratios are available. No exact stock or finite-step marginal-equivalence claim.
- Public PDB files 1CRN, 1PGA, 1UBQ downloaded from RCSB and hashed in `PROTOCOL.json`. Fixed chain A1-14 and A23-40; eight de novo residues between. Training only on 1CRN, held-out PDBs 1PGA/1UBQ, four fresh evaluation seeds each. No native-gap coordinates in training reward.
- 16-step real-checkpoint smoke on 1CRN completed: stock and VE zero-adapter both finite, Cα geometry 1/1 each. VE zero score -0.98683, stock -0.97649; different samplers, not a treatment effect.
- New numerical test and code compile passed. Native-alignment and guarded-score CPU checks passed.
- Single predeclared training process PID 39873 started 02:52 UTC: `rfd3_grpo.train --sampler-kind ve_sde --spec-json rfd3_flow_benchmark/1CRN.json --steps 32 --epochs 3 --group-size 4 --eval-n 4`, with stock reference and 06:00 UTC training deadline. Log `train_1crn.log`, run marker and source snapshot under `train_1crn/`. Inspect this process and markers before starting any GPU job; do not duplicate.
- The single benchmark ran sequentially, saved 17/36 samples, and stopped at the predeclared 08:00 UTC start-sample deadline. No benchmark job remains. Results are in `RESULTS.md` and `paired_effects.png`; missing samples were not rerun.
- The task-scoped `caffeinate -i -t 11400` PID 40260 was stopped after the benchmark ended. No paid compute or external publishing.

## 03:04 UTC update

- Stock 1CRN reference first evaluation batch complete: n=4, guarded score -0.9792398, Cα geometry 4/4, bond MAE 0.02754 Å. This is a different sampler and is not the GRPO effect baseline.
- Training PID39873 remains active; VE zero-adapter batch and group updates continue. No other RFD3 GPU job started.
- The task-specific foreground caffeinate timer PID40260 was verified active. A resumed heartbeat now monitors this new benchmark every 10 minutes and should remain quiet while unchanged. Its final action is to disable itself after the report.

## 03:25 UTC update

- The first real-checkpoint Flow-GRPO-style update completed on four 1CRN trajectories. Group rewards were −0.97422, −0.97002, −0.97410, and −0.98520, yielding nonzero within-group advantages. The update used 32 transitions, had 0 clipped ratios, maximum absolute log ratio 0.00854, mean conditional KL 2.25e−6, and adapter delta norm 0.00125; frozen backbone gradient count was 0. This verifies a small parameter update occurred, not benchmark improvement.
- Training and task-specific caffeinate are still active. No benchmark run has started.

## 03:55 UTC update

- Training finished successfully: `train_1crn/DONE.json` reports three epochs and 12 training trajectories; `adapter.pt` exists. Final in-training 1CRN guarded score was −0.9761157 versus VE zero −0.9761162 on the training evaluation batch, an essentially negligible difference; path KL was 0.000430. This is not the fresh-seed benchmark result.
- After verifying the training process exited and no benchmark process existed, started the single planned benchmark at ~03:54 UTC (initial PID 43100, log `benchmark.log`). It is running sequentially and saves/resumes each PDB-policy-seed sample. Caffeinate PID 40260 remains active.

## 04:35 UTC update

- All twelve predeclared 1CRN evaluation samples finished and its DONE marker exists. On four paired fresh seeds, trained-minus-zero mean guarded score was −0.000000054; mean native-gap RMSD difference −0.0000223 Å; mean loop bond MAE difference −0.00000134 Å. All twelve geometry checks passed. These are effectively null changes on the RL condition, not evidence of improvement.
- The same single benchmark process has moved to 1PGA. No additional GPU job was started.

## Final audit

- Training DONE marker confirms three real-checkpoint GRPO-style updates on 12 1CRN trajectories. The 786-parameter adapter changed; frozen backbone gradient count stayed zero. The in-training score change was negligible.
- Deadline stopped the benchmark at 17/36 saved samples: 1CRN 12/12, 1PGA 5/12, 1UBQ 0/12. Four fresh paired 1CRN seeds gave trained-minus-zero mean guarded score −0.00000005 and native-gap RMSD −0.000022 Å. One 1PGA pair had +0.00000122 score and −0.000025 Å RMSD. No held-out generalization conclusion is possible. `BLOG_DRAFT.md` gives a literature-grounded narrative without claiming improvement.
- Recomputed rewards, geometric metrics, fixed-motif alignments and native-gap RMSD from all 17 saved finite raw atom/Cα coordinate artifacts. Every fixed-motif RMSD was <0.001 Å. All eight numerical/unit tests and source compilation passed. Plot was rendered and visually checked.
- No task training, benchmark or keep-awake process remains. The deadline is past; no new GPU samples should start under this protocol. Disable the task heartbeat after verifying its state.

## Blog handoff

- `BLOG_POST.md` is the employer-facing article. It uses the complete four-seed 1CRN result as the main finding, links the literature and protocol, and states the incomplete held-out coverage. `blog_figure.png` is an audited plot of the paired 1CRN scores and their magnified differences. `BLOG_DRAFT.md` and `BLOG_EDIT_NOTES.md` preserve the writing revision record. No new experiments were run for the blog.
