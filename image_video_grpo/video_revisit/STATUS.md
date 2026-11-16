# Toy-video recheck status

Completed 2026-11-15 UTC. Four of four fixed native-sampler clips failed the predeclared digit-content gate and were visually noise-like. They were 0.9983 correlated with their initial uniform noise. The separately declared corrected-Euler diagnostic failed on its first two paired seeds (correlations about 0.9494); with two failures, the required three-of-four pass count became impossible, so the remaining two diagnostic seeds were not started.

The pinned `moving_mnist_epoch_79.pth` file has SHA-256 `ec9f5e39099e9cbcf282cfef05f74f8565cf68c07a1aad9857b3cc0f444e8af7`, but its internal `epoch` metadata is 4. The file name and model-card claim are therefore inconsistent with the checkpoint metadata. The released sampler's update sign and `/10` scale are inconsistent with the training target under its descending schedule; correcting those alone did not pass the content gate. These are diagnoses, not proof of the upstream author's intent.

`GATE.json` records the native decision. `AUDIT.json` contains hashes, raw-video checks, every measured clip, checkpoint metadata, and initial-noise correlations. Each GIF was visually inspected. **No video GRPO run was started or is claimed.**
