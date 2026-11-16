# Factorized toy-video study status

Completed. The frozen pretrained MNIST image DDPM plus deterministic motion renderer passed its prospective content gate (6/8) and produced finite, recognizable four-frame clips. Four GRPO adapter trainings and all four paired evaluations completed without failure markers. The initial two-seed study showed small negative effects; the higher-step-size exploratory replication split across seeds. Best-of-four selection improved independent recognition at 4× sampling cost. There is no reproducible GRPO gain and no successful learned-video diffusion checkpoint. See [RESULTS.md](RESULTS.md) and [VIDEO_RESCUE_BLOG_POST.md](VIDEO_RESCUE_BLOG_POST.md).

All 288 evaluation rows and twelve raw clip tensors are saved under `runs/`. `AUDIT.json` verifies hashes, shapes, finite values, pairing and exact CPU rescoring. `ANALYSIS.json` has per-seed paired effects and conditional bootstrap intervals. Numerical tests passed. No paid compute, publishing or external contact was used.

The task-scoped `caffeinate` process was stopped after the final audit; no study process remains active. Unrelated jobs were left untouched.
