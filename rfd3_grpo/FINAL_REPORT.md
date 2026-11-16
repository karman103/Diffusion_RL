# GRPO on pretrained RFdiffusion3 — local research POC

**Result.** GRPO updated a real RFdiffusion3 denoising component with an exact conditional action likelihood and frozen base checkpoint. Across three training seeds and fresh-noise checks at 24, 32 and 40 residues, it did **not** consistently improve the declared Cα geometry reward. This is a reproducible method result, not a protein-design success claim.

The [published-method audit](LITERATURE_AUDIT.md) shows how this POC adapts DDPO's stepwise policy gradient, Flow-GRPO's group-relative clipped objective, and DPOK's reference KL to RFD3. It also records the key departure from Flow-GRPO: our explicit stochastic sampler is **not** a proven marginal-preserving ODE-to-SDE conversion.

## What was actually trained

The official 168,038,994-parameter RFD3 checkpoint remained frozen. A rank-two LoRA correction to the final atom-coordinate projection was applied on free Cα atoms during only the last recycling pass. Three experts selected by noise level used 786 trainable parameters; a shared-expert comparison used 262. The correction passed through RFD3's EDM output scale and the Euler step coefficient. Earlier frozen recycles made the final-pass decoder features cacheable, and a real-model injection check matched cached replay within 6.20×10⁻⁶ Å. A stock sampler wrapper matched native RFD3 under shared random draws within 3.81×10⁻⁵ Å.

Stock RFD3 adds churn noise **before** its nonlinear denoiser. We therefore declared a **modified sampler**: retain stock churn as auxiliary randomness and add independent Gaussian actions after Euler only on free Cα coordinates in a chosen noise window. The likelihood ratio is exact for those conditional Gaussian actions; it is not a Gaussian density for the stock sampler's marginal transition. The clipped, step-averaged GRPO objective is a policy surrogate, not an exact whole-trajectory output likelihood. Four siblings under the same input supplied group-relative advantages, with clipping and a conditional KL penalty to the zero adapter. No frozen-network tensor received a gradient.

## Measured outcomes

The guarded score is −Rg/10 − 2×adjacent Cα bond MAE − 10×nonlocal Cα clash fraction; higher is better. Values below are paired changes against the zero adapter with the **same** exploration window and shared noise seeds. The three-expert column averages the three trained adapter outcomes per noise seed. Intervals bootstrap evaluation noise seeds for fixed adapters; they do not reflect variation across biological targets.

| Unconditional length | Fresh seeds | Three-expert mean Δscore [95% interval] | Late-window Δscore [95% interval] |
|---|---:|---:|---:|
| 24 (transfer) | 8 | −0.00098 [−0.00304, +0.00041] | ~0.00000 [−0.00002, +0.00001] |
| 32 (trained length) | 16 | −0.00063 [−0.00210, +0.00096] | +0.00031 [−0.00003, +0.00095] |
| 40 (transfer) | 8 | −0.00034 [−0.00129, +0.00037] | +0.00039 [−0.00002, +0.00116] |

On the original four evaluation seeds, the three reward-directed runs changed score by −0.00121, −0.00312 and −0.00076; a random-advantage control changed it by +0.00006. Restricting adaptation to σ≤4 initially appeared favorable (+0.00458), but its matched 16-seed estimate was only +0.00031. The shared 262-parameter expert had a small +0.00010 four-seed change and −0.00140 on 16 fresh seeds. Expert count and trainable capacity differ, so that ablation cannot isolate the value of noise specialization.

All **288** fresh-noise Cα traces across the three lengths passed the declared adjacent-spacing and clash heuristic. That heuristic does not establish folding, sequence designability, all-atom validity or function. The study uses one unconditional input at each length and a compactness-biased proxy reward. Results cannot support a broad biological claim.

## Artifacts and reproduction

- [Research narrative](BLOG.md) and [mathematical method](METHOD.md)
- [Literature-to-implementation audit](LITERATURE_AUDIT.md)
- [Training measurements](RESULTS.md) and audited fresh-noise results for [24](cross_length/length24/RESULTS.md), [32](heldout_3000000_n16/RESULTS.md), and [40](cross_length/length40/RESULTS.md) residues
- [Paired-difference figure](heldout_3000000_n16/paired_deltas.png) and [Cα backbone examples](backbones.png)
- [Reproduction commands](README.md); per-run source snapshots, adapter weights, configurations, raw coordinates and audit logs under runs/ and cross_length/

The checkpoint SHA256 is `9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`; the Apple-MPS Foundry fork is pinned at `7a9c246fbf140a743d14a35248f5039e7723a23a`. Seven numerical policy tests passed, six full RFD3 training runs completed, and the final raw-coordinate audits passed for 23 saved training evaluation batches plus 288 fresh-noise trajectories. The deliberately under-resolved smoke run is preserved and labeled separately.
