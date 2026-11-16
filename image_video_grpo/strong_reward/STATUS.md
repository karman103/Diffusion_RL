# Stronger-reward image GRPO status

Completed and audited 2026-11-15 UTC. All three predeclared seeds (1501–1503) finished 250 updates and fresh paired evaluation on 32 seeds. No run had nonfinite samples, gradients, or parameters; maximum clip fraction was zero. The two disjoint 25,000-example classifiers passed the held-out gate at 81.66% (proxy) and 81.46% (independent evaluator).

The trained-minus-zero proxy frog probability was **+0.05589** with a descriptive hierarchical 95% interval **[+0.00039, +0.12912]**. Independent frog probability was **+0.00893 [−0.05603, +0.07215]**, and independent frog accuracy was **+0.02083 [−0.07292, +0.12500]**. The third training seed had a negative independent effect. The evidence supports a proxy reward gain, not a reliable generalizable image-quality improvement.

The independent classifier feature-Fréchet diagnostic worsened in all three trained runs versus the shared zero control (18.263 to 18.490, 18.482, 20.776). This is a noisy 32-image diagnostic, not official FID. Pixel diversity was 35.445 for zero and 35.685, 34.056, 34.234 for trained. Best-of-four selection from the zero policy achieved 0.957 proxy reward and 1.000 independent frog accuracy on these 32 seeds, at four times the sampling cost; it is a different compute budget.

`AUDIT.json` records SHA-256 hashes and CPU rescoring of every saved raw image. All saved samples had shape 3×32×32 and finite values; the largest CPU-versus-MPS score difference was under 0.000005. Ten numerical tests across the core and DDPM policy passed after completion. `RESULTS.md`, `EFFECTS.json`, and the plots contain the full results. The earlier image experiments and protein artifacts were preserved.
