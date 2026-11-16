# Stock-DDPM follow-up status

Completed three predeclared real-checkpoint GRPO seeds (1301–1303), each with 60 groups of four. All runs and evaluations have DONE markers. The 50-step zero-adapter policy matches the released DDPM scheduler's transition in numerical tests; no clipping or nonfinite updates occurred. Fresh paired evaluations and CPU rescoring are complete in [`RESULTS.md`](RESULTS.md) and [`AUDIT.json`](AUDIT.json).

The measured outcome remains a quality null: mean trained-minus-zero proxy reward +0.01224 (descriptive 95% interval −0.00897 to +0.03655), independent frog probability +0.00186 (−0.04067 to +0.03872), independent frog accuracy −0.03125. Best-of-four substantially increased both proxy and independent recognition at four times the sampling cost, but is not GRPO. The prior DDIM study and failed toy-video gate remain untouched.

This result motivated a separate [`strong_reward`](../strong_reward/README.md) exploratory study. Do not combine its fresh seeds with this study as if both were predeclared together.
