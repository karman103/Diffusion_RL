# RFdiffusion3 GRPO — current state

Updated 2026-11-13 03:38 UTC. User priority: run GRPO directly on real RFD3 and explain diffusion-component fine-tuning choices. Deadline: **2026-11-13 10:26 UTC** (05:26 America/Toronto). Stop starting experiments at that time, finalize measured results and limitations, and disable heartbeat. No paid compute, publishing, or external contact.

## Inspect these jobs before any new run
- Sequential queue PID **15104**; current `main_s0` training PID **15109**. Both verified alive at03:37 UTC. `queue_process.json`, `queue.log`, `main_s0.log`, and run `RUNNING.json` identify them. Do not duplicate or stop unrelated jobs.
- Main stock baseline n=4 completed: Rg9.15996 Å, bond MAE0.02894 Å, geometry4/4, reward−0.973867. Modified zero-adapter baseline n=4 also completed: Rg9.13232 Å, bond MAE0.03141 Å, geometry4/4, reward−0.976044. Training epochs not yet recorded. No new GPU experiment launched this heartbeat.
- Queue order: main_s0, random_s0, main_s1, main_s2, single_s0, late_s0. Each: length32,64 schedule points,6 epochs×4 siblings,evaln4,LR3e−4,reuse4,KL0.01. Driver stops on failure and enforces deadline. Its manifest is already loaded; editing queue.json does not change the active queue.
- Preserve loaded training sources while jobs run. Each new run snapshots task Python sources and hashes. Docs/report edits are safe.
- Other projects have active GPU jobs. Local laptop16GB unified memory, limited disk. Run RFD3 jobs sequentially. MPS/process checks require sandbox escalation. No unresolved permission blocks.

## Completed evidence
- Real checkpoint2,690,316,669 bytes; SHA256 `9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`. MPS fork pinned at `7a9c246fbf140a743d14a35248f5039e7723a23a`. Training env `.venv-rfd3`.
- `smoke_v3` COMPLETE: actual pretrained model GRPO update,786 adapter parameters,168,038,994 frozen network parameters, zero base gradients, adapter delta0.0002321. Stock parity3.81e−5 Å and cached head parity6.20e−6 Å. Eight-point schedule geometry failed; negligible reward change. Implementation smoke only.
- `calibration64` COMPLETE:32 residues,64-point schedule,n2 stock and modified each. Both geometry2/2; stock bond MAE0.03057 Å, modified0.03363 Å. Small exploratory calibration justifies proceeding, not biological claims.
- First smoke failures preserved: allocator watermarks, then low-memory P_LL interface. Both corrected before successful smoke. Seven meaningful numerical tests passed.

## Implemented method and interpretation
- Rank2 LoRA on actual final atom-coordinate head, three noise experts (786 params), last recycle only, free Cα atoms, sigma0.5..32. Earlier recycles and backbone network frozen; cached final features enable exact head replay.
- Explicit MODIFIED sampler: original pre-denoiser churn retained as auxiliary randomness, independent Gaussian actions added after Euler on free Cα. Exact conditional Gaussian likelihood for this augmented policy; no stock-density or marginal-equivalence claim. Stepwise clipped GRPO objective remains a surrogate, not exact whole-trajectory importance sampling.
- Group-relative advantages, clipped ratios, conditional Gaussian KL to zero adapter. CA geometry/compactness rewards are heuristics, not foldability/function validation. UNK PDBs contain no designed sequence.
- Single expert ablation has262 parameters vs786; capacity confound must remain explicit. Deeper attention LoRA/shared-prefix branching are proposed, not implemented.

## Deliverables / next steps
- `METHOD.md`: implemented math and adaptation explanation. `BLOG.md`: working narrative with pending main results explicitly labeled. `RESULTS.md`: recomputed raw-coordinate audit. `README.md`: reproduction.
- CPU report command: `.venv/bin/python -m rfd3_grpo.report`. Last audit:4 run directories,7 evaluations. Do not confuse partial main stock baseline with completed training.
- Monitor logs, failed markers and queue history without overwriting artifacts. Analyze main vs random controls, seeds and ablations as they finish; finalize blog with measured effect sizes and small-sample limits. Consider fresh-noise evaluation only after existing queue, with sufficient deadline/compute budget.
- At completion/deadline, regenerate and visually inspect figures, finalize honest report, disable heartbeat `protein-diffusion-research-poc`.
- Do not resume Genie queue. Preserve prior Genie and MNIST work. `protein_poc/STATUS.md` is historical only. Detailed earlier session history is retained in `STATUS_HISTORY.md`.

## Update 04:27 UTC Nov 13
- main_s0 completed six real-checkpoint GRPO updates, 24 training siblings, 786 learned adapter parameters, zero frozen-network gradients. Fixed n4 evaluation guarded reward changed from -0.976044 (zero adapter) to -0.977251 (epoch6); geometry pass stayed 4/4. The small observed deterioration does not demonstrate a general effect.
- Queue PID15104 continues sequentially. Current random-control child PID17381 was verified at04:25 UTC. Its training advantages are random; its evaluation log uses compactness. The report now recomputes the common guarded score for all saved coordinates to permit comparison.
- RESULTS.md regenerated: 5 run directories, 10 saved evaluation audits. BLOG.md now includes the completed main result. No new GPU experiment started.

## Sleep prevention (04:36 UTC Nov 13)
- Existing caffeinate PID15105 prevents idle sleep while queue PID15104 runs.
- User explicitly asked to keep computer awake. Added independent deadline-bounded caffeinate PID19210 with -i -t 20990, verified active. It prevents idle system sleep until approximately 10:26 UTC even if queue ends earlier. Display sleep may still occur. Stop this exact PID when final report/heartbeat is done early; do not touch unrelated caffeinate processes.

## Update 04:39 UTC Nov 13
- random_s0 completed six actual updates with random training advantages, 24 training sibling trajectories. Common guarded evaluation reward on same n4 seeds changed -0.97604 to approximately -0.97599 (paired delta +0.00006), with path KL1.056. Main reward-directed seed0 delta was -0.00121, so it did not beat the drift control on this tiny evaluation.
- Queue PID15104 automatically advanced to main_s1 child PID18851, verified alive. Sleep prevention PID19210 also alive. Do not duplicate jobs.
- RESULTS.md was regenerated by queue and BLOG.md now discusses this control. Remaining seeds and ablations still pending; no population effect claim.

## Update 04:59 UTC Nov 13
- main_s1 completed six real-checkpoint GRPO updates; fixed n4 guarded reward -0.97604 to -0.97917 (paired delta -0.00312), geometry pass 4/4, path KL0.863. Seed0 delta was -0.00121; neither improved this narrow score.
- Queue PID15104 advanced to main_s2 child PID20072, verified alive; deadline sleep prevention PID19210 active. No duplicate jobs.
- RESULTS.md audited 6 run directories/14 evaluations. BLOG.md includes interim two-seed finding. Third seed/ablations pending.

## Update 05:19 UTC Nov 13
- main_s2 completed six real-checkpoint GRPO updates; guarded reward delta -0.000758 on fixed n4 evaluation, geometry4/4, path KL1.714.
- All three reward-directed seed deltas are negative: -0.001207, -0.003123, -0.000758; mean -0.001696. Random-advantage seed0 delta +0.000063. This is a narrow observed failure to improve, not a population or protein-design claim. All four runs share evaluation noise seeds.
- Queue PID15104 automatically advanced to single_s0 child PID21135, verified alive. Deadline sleep prevention PID19210 active. No duplicate runs.
- RESULTS.md regenerated by CPU report with cross-seed summary; 8 run directories/19 saved evaluations. BLOG.md updated. Single-expert and late-window ablations pending.

## Update 05:39 UTC Nov 13
- single_s0 completed six GRPO updates with one rank2 adapter (262 parameters vs786 for three experts). Fixed n4 guarded reward delta +0.00010, path KL0.130, geometry4/4. Three-expert seed0 delta -0.00121, path KL0.743. Cannot attribute difference to noise specialization because parameter count and policy movement differ.
- Queue PID15104 advanced to late_s0 child PID22169, verified alive. Deadline sleep prevention PID19210 active. No duplicate jobs. Queue will finish after this run if successful.
- BLOG.md now includes single-expert result; RESULTS.md audited 8 run directories/20 saved evaluations. Late-window ablation pending.

## Fresh-noise evaluation prepared (05:52 UTC Nov 13)
- Added heldout_eval.py and heldout_report.py, syntax checked. Planned 16 new shared noise seeds 3000000..3000015, stock/zero/main three seeds/random/single/late. No GPU eval launched yet. Evaluator acquires queue.lock nonblocking and requires QUEUE_FINISHED plus six successful run markers, so it cannot duplicate or overlap the active training queue.
- It saves raw all-atom/CA coordinates and metrics per sample, supports resume without overwriting, enforces 10:26 UTC deadline, and audits/recomputes guarded score. Bootstrap intervals will describe variation across new noise seeds on the same unconditional 32-residue input, not variation across proteins or training seeds.
- Start only after queue finishes. README documents commands. If remaining time is short, reduce count before first run (and record the plan); never change plan after output exists.

## Queue complete; fresh-noise evaluation running (05:59 UTC Nov 13)
- All six queued real-RFD3 runs completed successfully by05:57 UTC; QUEUE_FINISHED.json and queue_history.jsonl confirm. No training queue process remains.
- Late-window ablation (sigma<=4, 20 stochastic group transitions vs56) changed own modified-sampler guarded reward -0.973708 to -0.969130, paired delta +0.004578 on fixed n4 samples. Path KL0.00927; geometry4/4. Its zero-adapter baseline differs from full-window sampler; do not compare final scores without matching baselines. Small-sample interval includes zero.
- Started heldout_eval.py PID23251 at05:59 UTC, verified alive. Plan saved before samples: 16 fresh shared seeds 3000000..3000015, eight policies (stock, zero, three main seeds, random, single, late). It holds queue.lock exclusively, has no training updates and stops starting samples at10:26 UTC. This is the only current RFD3 GPU job. Deadline caffeinate PID19210 remains active.
- BLOG.md updated with all queued results. Finalize after heldout evaluation, audit raw coordinates, inspect figures, then disable heartbeat and stop only our caffeinate PID if before deadline.

## Heldout evaluation audit correction (06:10 UTC Nov 13)
- Heldout evaluator PID23251 active; approximately two fresh noise seeds underway, no failure. Partial heldout_report.py audit passed.
- The late-window policy cannot be paired against full-window zero because their exploration windows differ. Report now leaves that comparison pending and a separate heldout_late_zero.py script is prepared, syntax check pending. Start it only after the primary heldout evaluator finishes; it acquires the same exclusive queue.lock and uses the same 16 seeds with the matching sigma<=4 zero-adapter sampler. No GPU job for this correction launched yet.

## Primary heldout complete; matched late baseline running (07:19 UTC Nov 13)
- Primary heldout evaluator completed all 128 samples (16 fresh seeds x 8 policies) at07:09 UTC. Raw-coordinate audit passed 128/128 via heldout_report.py; no adapter updates. Full-window zero reward mean -0.954866, sample SD0.074225.
- Paired guarded reward deltas vs full-window zero: main_s0 -0.00179, main_s1 +0.00047, main_s2 -0.00058; per-seed average across three main adapters -0.00063, bootstrap 95% interval [-0.00210,+0.00096] across 16 fresh noise seeds. Random-control +0.00070, single-expert -0.00140. Geometry heuristic 100% for all policies so far. These are descriptive fixed-condition results, not across-target validation.
- Primary heldout process PID23251 exited. Matched late-window zero evaluator PID27301 started at07:19 UTC, verified alive and holds exclusive queue.lock; no other RFD3 GPU job. Deadline caffeinate PID19210 active.
- BLOG.md updated with fresh-noise findings; late-window heldout conclusion pending matched baseline. Heldout report and paired-difference plot generated and coordinate-audited; partial plot was visually inspected.

## Fresh-noise conclusion and predeclared transfer check (07:33 UTC Nov 13)
- Matched late-window zero reference completed 16 fresh seeds; heldout_report.py audited all 144 saved trajectories. Late adapter paired guarded reward delta +0.00031 with seed bootstrap 95% interval [-0.00003,+0.00095], much smaller than the original four-sample +0.00458. All geometry heuristic passes. Full-window three-seed mean delta -0.00063 with interval crossing zero. No consistent proxy gain.
- Final heldout paired-difference PNG, main results PNG and backbone PNG visually inspected; chart scale was corrected to show small differences. BLOG.md updated from draft to completed local POC, README updated. Final report still needs cross-length results and closing audit.
- To assess transfer beyond the one trained length, PROTOCOL.md predeclared unconditional lengths24 and40, eight new shared seeds 4000000..4000007, nine policies including matched zero references; no retraining. cross_length_eval.py and revised heldout_report.py syntax checked. Length24 evaluator PID27754 started07:32 UTC and verified alive, holding exclusive queue.lock. No other RFD3 GPU job. Length40 planned sequentially after length24 if time/memory permit. Deadline caffeinate PID19210 active.

## Length-transfer progress (07:49 UTC Nov 13)
- Predeclared length24 evaluation completed all72 samples (8 fresh seeds x9 policies), raw-coordinate audit passed. Three main adapters averaged paired guarded reward delta -0.00098 vs full-window zero; seed bootstrap interval [-0.00304,+0.00041]. Late adapter vs matched late-window zero essentially0. All C-alpha geometry heuristics passed. These are fixed-condition length-transfer checks, not diverse-target validation.
- Length24 PID27754 exited. Sequential length40 evaluator PID28262 started07:49 UTC and verified alive; holds exclusive queue.lock. Deadline caffeinate PID19210 active. No overlapping RFD3 GPU jobs. Length40 report pending.
