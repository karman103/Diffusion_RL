# Factorized toy-video GRPO: audited results

**Outcome.** A local diffusion-driven four-frame Moving-MNIST generator now produces recognizable translated digits. Its prespecified eight-seed feasibility gate passed **6/8** clips, and all 24 zero-adapter clips in each evaluation set were finite and moving in the requested direction. Four GRPO adapters were actually trained and evaluated. **No reproducible GRPO improvement was found.** The two initial runs slightly reduced proxy reward and independent digit confidence. A larger-step-size follow-up split across seeds; one was worse and one better on the independent classifier, with no clear proxy gain. Best-of-four proxy selection improved independent confidence at four times sampling cost.

The generator factorizes video as a sampled canonical MNIST digit plus deterministic affine motion. This solves the *noise-like clips* prerequisite for a constrained toy generator, **not** the failed learned video-diffusion checkpoint or learned temporal generation. The policy-gradient update acts on the image diffusion transitions and receives a video-level reward. Motion is not trainable. See [protocol](PROTOCOL.json), [follow-up protocol](FOLLOWUP_PROTOCOL.json), [raw audit](AUDIT.json), [per-sample analysis](ANALYSIS.json), [filmstrip](baseline_grid.png), and [paired animation](paired_motion.gif).

## Fixed-seed comparisons

All values are means over 24 fresh paired clips per adapter. Δ is trained minus the zero adapter using the same image sampler and procedural renderer. The independent classifier was never used to train or select an adapter.

The separate evaluator is a CNN trained on all 60,000 MNIST training examples; the reward proxy is an MLP trained on 1,000 of those examples. Thus their weights and roles are separate, but their **training sets overlap**. Their original MNIST test accuracies were 99.23% and 89.89%, respectively; translated generated clips are a different input distribution. [CHECKPOINTS.json](CHECKPOINTS.json) records both hashes.

| Study / training seed | Proxy reward zero → trained (Δ) | Independent digit confidence zero → trained (Δ) | Independent class consistency zero → trained | Pixel diversity zero → trained |
|:--|--:|--:|--:|--:|
| Initial, 11801 | 0.6009 → 0.5982 (−0.0028) | 0.8515 → 0.8410 (−0.0105) | 0.9792 → 0.9688 | 39.24 → 39.33 |
| Initial, 11802 | 0.6009 → 0.5976 (−0.0034) | 0.8515 → 0.8338 (−0.0176) | 0.9792 → 0.9688 | 39.24 → 39.23 |
| Higher step size, 12101 | 0.5762 → 0.5734 (−0.0029) | 0.8494 → 0.8141 (−0.0353) | 0.9583 → 0.9479 | 40.98 → 37.23 |
| Higher step size, 12102 | 0.5762 → 0.5776 (+0.0014) | 0.8494 → 0.9017 (+0.0523) | 0.9583 → 1.0000 | 40.98 → 40.96 |

The two exploratory follow-up outcomes diverged; averaging them would hide that instability. Paired clip bootstrap intervals in [ANALYSIS.json](ANALYSIS.json) describe evaluation-seed variation **conditional on each trained adapter**. Two training seeds per setting are too few for a reliable population-level claim, and their control clips are shared rather than independent replicates. The [paired-effects plot](paired_effects.png) shows the per-seed outcomes and conditional intervals.

On the initial evaluation set, best-of-four increased proxy reward from 0.6009 to 0.6783 and independent confidence from 0.8515 to 0.9211. On the follow-up set, the corresponding means were 0.5762 → 0.6705 and 0.8494 → 0.9532. This uses four zero-adapter rollouts and one proxy-scored selection per clip. It is a practical quality/cost control, not a compute-matched GRPO win.

## Validity and failure checks

- The first gate's eight filmstrip rows were visually inspected: recognizable digits translate across four frames; two had weak independent recognition. This gate was fixed before GRPO. The external Moving-MNIST checkpoint and two locally trained full-video variants failed separate content gates; those failures remain documented in `video_revisit/` and `video_rescue/`.
- The four training runs completed 120, 120, 400 and 400 group-of-four updates. Every group had nonzero advantage. The final adapter parameter-change norms were 0.00475, 0.00601, 0.203 and 0.165 respectively. Maximum clipping fractions per update were 0.042, 0, 0.167 and 0.167. No numerical or saturation stop rule fired.
- Every saved evaluation tensor has shape `[24,4,28,28]`, all values are finite, and CPU rescoring exactly reproduces saved metrics under the same code. The saved sample SHA-256 values are in [AUDIT.json](AUDIT.json). All evaluation clips had centroid motion above the 0.7-pixel static threshold, correct direction cosine above 0.5, and ink mass in the 0.03–0.35 guard range. These motion checks mostly verify the prescribed renderer; they do not demonstrate learned motion.
- Independent confidence below 0.7 occurred in 4/24 zero clips on the initial set and 6/24 on the follow-up set. The first two trained adapters had 4/24 and 5/24 low-confidence clips. The stronger-update adapters had 5/24 and 2/24. This, the filmstrips, classifier confidence, diversity, and pixel differences together support a mixed rather than improved quality conclusion.
- Three factorized-policy tests and seven shared numerical tests pass. They cover finite samples, 19 stochastic actions, zero-adapter/reference equality, zero-advantage gradients, frozen-backbone gradients, and Gaussian action densities. The logged Gaussian density is exact for each implemented *conditional* image transition. It is neither the likelihood of a final video nor an exact-gradient claim for the clipped, selected-step surrogate.

The most useful result is the boundary: a simple factorization makes this laptop-sized toy video task generate valid clips and enables an actual video-scored diffusion GRPO run. The observed GRPO update did **not** reliably improve the measured objective, and none of this validates GRPO on a learned video-diffusion model or modern video generation.
