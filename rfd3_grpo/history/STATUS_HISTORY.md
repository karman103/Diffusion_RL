# RFdiffusion3 GRPO — current priority

User explicitly redirected work to **GRPO on RFD3**, asking for interesting diffusion-component fine-tuning choices. Do not substitute Genie results for this request.
Original eight-hour deadline remains 2026-11-13T10:26:00Z (05:26 America/Toronto). Heartbeat `protein-diffusion-research-poc` was updated to this priority.

## Current work
- Stopped only our waiting Genie queue PID11157 before it scheduled further experiments. Do not resume it.
- Original Genie pilot PID10250 may still finish its last evaluation. Preserve its artifacts; it is supplementary work.
- Downloading the official README-recommended Apple-MPS community fork: `git clone --depth 1 --filter=blob:none https://github.com/fnachon/foundry.git third_party/foundry-mps`, exec session48578.
- No RFD3 inference or training result yet.
- MPS requires escalation outside sandbox. Laptop has16GB shared memory, around13GB disk free at the start of this turn. Other projects have active GPU jobs; never stop them.

## Intended method
Validate actual RFD3 checkpoint inference first. Inspect final denoising atom-coordinate head (`to_r_update`) for a small exact trainable update. Implement group-relative advantages within each identical input condition, clipped likelihood ratios, frozen reference regularization, time-window ablations. The stock sampler adds churn noise before nonlinear denoising: its output transition is not a simple Gaussian. Any GRPO Gaussian policy must be an explicitly modified stochastic sampler and validated against its own baseline. Do not claim marginal equivalence without deriving and testing it.

## Deliverables
Real pretrained checkpoint smoke, gradient/update evidence, numerical checks, measured metrics, reproduction instructions and a clear method explanation. If local hardware or installation blocks a real run, record the exact failure and distinguish any untrained numerical tests from pretrained results. Continue useful implementation work within the deadline.

## Implementation progress (22:04 Nov12)
- Apple-MPS fork cloned at commit7a9c246fbf140a743d14a35248f5039e7723a23a. No upstream source edits.
- Isolated `.venv-rfd3` installed successfully: PyTorch2.15.0, AtomWorks3.0.1, rc-foundry1.0.0. RFD3 engine imports successfully and MPS is available outside sandbox.
- Official 2,690,316,669-byte checkpoint downloading with eight range requests using `download_checkpoint.py`, exec session38734. Progress in `download_ranges.log`; sparse file size alone does NOT mean complete. `verified_checkpoint` refuses partial downloads based on `.download.json`.
- Automatic real-model smoke is waiting for download, exec session75050, log `rfd3_grpo/smoke.log`. It invokes `run_when_ready.py` then execs `train.py --out rfd3_grpo/runs/smoke --steps8 --length16 --epochs1 --group-size2 --eval-n2 --lr1e-4 --check-stock-parity --evaluate-stock`. Do not duplicate this GPU job.
- `backend.py`: real engine loading, frozen EMA inference network, actual final-head feature hook, stock sampler parity check, last-recycle head-injection equivalence check, explicit stochastic CA policy.
- `grpo.py`: three noise-bin rank2 LoRA experts on128->3 head (786 parameters), group-relative advantages, exact joint Gaussian log-ratios with stable mean-difference formula, clipping and conditional KL.
- `train.py`: stock/modified-before/modified-after evaluations, actual GRPO updates, hashes, gradient/parameter diagnostics, adapter checkpoint and raw all-atom/CA coordinates.
- Seven numerical tests pass in `verification.log`. These are NOT pretrained-model results.
- `METHOD.md` explains the implemented denoising adaptations and which suggested deeper LoRA extensions remain unimplemented. `README.md` has reproduction commands.
- No real RFD3 forward/update has completed yet as of this entry.

## Checkpoint ready and real-model validation (22:08 Nov12)
- Official checkpoint download completed (81/81 range blocks, exact2,690,316,669 bytes). `download_ranges.log` records completion.
- First smoke failed during GPU allocator initialization because a process-local high watermark0.8 was lower than PyTorch's default low watermark1.4. No model forward/update occurred. Failure retained in `smoke.log`.
- Corrected both process-local watermarks: high0.8 / low0.6. Retry is running as exec session11233, log `smoke_v2.log`, output `runs/smoke_v2`. It is loading the real checkpoint. Do not duplicate it.
- All training/backend source edits were completed before starting this retry. Avoid changing loaded training source during a run; preserve hash provenance.
- Prepared `rfd3_grpo/queue.json` with six follow-on RFD3 experiments (main/random/single-expert/late-window/main seeds1/2), but have NOT launched them yet. Validate smoke first.
- `report.py` recomputes metrics from raw coordinates and produces RESULTS.md and a PNG. Run it with the old `.venv/bin/python` (Matplotlib is available there); training uses `.venv-rfd3`.

## REAL RFD3 GRPO UPDATE COMPLETED (22:13 Nov12)
`runs/smoke_v3/DONE.json` confirms completion. Log `smoke_v3.log`.
- Official checkpoint SHA2569b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1.
-168,038,994 frozen network parameters;786 trainable rank2 noise-expert LoRA parameters.
- Cached last-recycle head injection equivalence error6.198883e-6 Å.
- Stock-sampler wrapper parity error3.814697e-5 Å under common random draws.
- Group rewards[-26.9146004,-38.2368813], advantages[+1,-1], one stochastic step per sibling at sigma4.110205.
- One GRPO update epoch / four reuse passes completed. Adapter weight-delta norm0.0002321208. Frozen network gradient tensor count0. Maximum |log ratio|0.1389475.
- n=2 modified-sampler evaluation: reward-30.604185 before versus-30.604057 after; negligible change. Stock and modified geometry both BAD (bond MAE~14.46 Å). This eight-point schedule is an implementation smoke only; NEVER market it as useful protein design.
- First attempt `smoke` failed allocator configuration; second `smoke_v2` loaded the actual checkpoint but failed low-memory P_LL interface. Both failures preserved and corrected in v3.

## Current running job
- Longer schedule calibration: exec session55705, log `calibration64.log`, output `runs/calibration64`, length32 /64 schedule points /2 evaluation samples /epochs0 /stock and modified baselines. No competing RFD3 GPU job should be started.
- Choose the longer training schedule after looking at this calibration. If64-point stock geometry is poor, try200 points before committing to GRPO runs. Avoid spending the budget optimizing an unusable baseline. Report all exploratory choices.
- `queue.json` prepared: main_s0, random_s0, main_s1, main_s2, single_s0, late_s0;64 points, length32,6 epochs x4 siblings, evaln4. Queue NOT LAUNCHED yet.
- `run_queue.py` verifies a successful real smoke and adapter update before starting, with an exclusive lock and hard subprocess deadline. Use `--smoke rfd3_grpo/runs/smoke_v3`, since default currently names the failed v2. Stop on any failure and review it.
- At deadline10:26Z finalize measured report/method/blog and disable heartbeat. Do not resume Genie queue.

##64-point calibration PASSED; MAIN QUEUE STARTED (22:22 Nov12)
- `runs/calibration64/DONE.json` complete, stock and modified baseline n=2 each,32 residues,64 schedule points (63 denoiser calls),2 recycles.
- Stock: Rg9.67403 Å, bond MAE0.03057 Å, adjacent-distance pass1.0, clash fraction0, geometry pass2/2.
- Modified zero adapter: Rg9.61573 Å, bond MAE0.03363 Å, adjacent-distance pass1.0, clash fraction0, geometry pass2/2.
- These are small calibration samples. They justify proceeding, not claims of broad designability or functional improvement. Schedule selection was exploratory.
- Root started detached RFD3 queue PID15104. Source `queue_process.json`, log `queue.log`; first `main_s0` started03:22:45Z. DO NOT DUPLICATE IT. A scoped caffeinate process follows its lifetime.
- Queue order: main_s0, random_s0, main_s1, main_s2, single_s0, late_s0. Each length32,64 schedule points,6 epochs x4 sibling trajectories, evaln4, LR3e-4, reuse4, KL0.01. Stop starting at10:26Z; subprocess timeouts enforce the same bound.
- Queue now defaults to successful smoke_v3, and gates on the completed calibration's geometry before starting. It stops on any failure for review.
- Source updates after calibration: snapshot experimental Python sources in each run, full sampler configuration, separate adapter-initialization seed, PID/FAILED markers. Future runs include independent LoRA initialization plus trajectory noise for seeds1/2; input-conditioning seed remains0. Historical smoke_v3 and calibration experimental source copies were preserved under their `source/` directories with hash checks.
- `report.py` regenerated and audited all5 saved evaluations. Backbone PNG visually inspected; the64-point stock vs modified examples have plausible Cα geometry. Benchmark plot excludes the deliberately under-resolved8-point smoke, but its numeric rows and update evidence remain in RESULTS.md.
- `METHOD.md` and README now describe actual execution. Root README points to RFD3; old Genie-only statements were updated to link this follow-up.
- Remaining: monitor main queue; inspect failures without overwriting artifacts; when complete/deadline regenerate and visually inspect report, finalize a blog-style results discussion and explanation of which adaptation choices actually helped. Keep time-binned-vs-single parameter-count confound explicit. Distinguish Cα geometry from foldability and all-atom validation. Disable heartbeat on completion/deadline.
