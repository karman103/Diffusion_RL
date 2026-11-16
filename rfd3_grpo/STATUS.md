# RFdiffusion3 GRPO — completed local POC

Updated 2026-11-13 08:12 UTC. The user's RFD3 request is complete. Read [FINAL_REPORT.md](FINAL_REPORT.md), [BLOG.md](BLOG.md), and [METHOD.md](METHOD.md) for measured results and fine-tuning choices. Earlier running-state notes are preserved verbatim in [STATUS_PRE_FINAL.md](STATUS_PRE_FINAL.md) and [STATUS_HISTORY.md](STATUS_HISTORY.md). Do not resume the historical Genie queue.

## Final state

- Official RFD3 checkpoint SHA256 `9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`; pinned Apple-MPS Foundry fork `7a9c246fbf140a743d14a35248f5039e7723a23a`.
- Six sequential real-checkpoint training runs completed successfully: three group-relative reward seeds, random-advantage control, one-expert control, and late-noise-window ablation. Seven policy numerical tests passed. Stock-wrapper and cached-head parity checked on the actual pretrained model.
- No RFD3 training or evaluation job is active. No experiments should be started merely because the heartbeat resumes. The queue has `QUEUE_FINISHED.json`; raw-coordinate audit passed 23 saved training evaluation batches.
- Fresh-noise evaluation: 16 seeds at 32 residues (144 traces), eight at 24 (72 traces), eight at 40 (72 traces), all audited. The three-expert mean paired reward changes were −0.00063, −0.00098, and −0.00034, respectively; all seed-bootstrap intervals crossed zero. All 288 traces passed the declared Cα geometry heuristic. No consistent proxy gain or biological-design claim.
- Explicit modified stochastic sampler, exact conditional Gaussian action likelihood, clipped stepwise GRPO surrogate, and 786-parameter noise-banded rank-two LoRA on the final atom-coordinate projection. Stock RFD3 marginal likelihood was not claimed. Base 168,038,994 parameters stayed frozen.
- Reproduction and limitations are in `README.md`; final auditable figures and per-policy tables are under `heldout_3000000_n16/` and `cross_length/`.

## Sleep and automation

The user asked to keep the computer awake during the original eight-hour window. The independent `caffeinate -i -t 20990` process (PID 19210 when started) should remain active until its own timeout around 2026-11-13 10:26 UTC (05:26 Toronto); do not stop unrelated processes. The research heartbeat can be disabled now because deliverables are finished.
